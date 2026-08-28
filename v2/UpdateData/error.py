"""Resume an interrupted reserved-axis expansion without shifting data.

This is an operational recovery tool.  It accepts only matrices that match
either the current axis capacity or the requested target capacity.  Current
matrices are expanded row-by-row from the end; target matrices are skipped.
The axis files are replaced only after every matrix validates at target size.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__:
    from .axis.reset_axis import (
        DATE_RESERVE,
        TICK_RESERVE,
        MatrixSpec,
        _matrix_middle,
        _resize_matrix,
        load_axes,
    )
    from .config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.axis.reset_axis import (
        DATE_RESERVE,
        TICK_RESERVE,
        MatrixSpec,
        _matrix_middle,
        _resize_matrix,
        load_axes,
    )
    from v2.UpdateData.config import ROOT


DTYPES = {
    1: np.dtype(np.bool_),
    2: np.dtype(np.float16),
    4: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


@dataclass(frozen=True)
class RecoverySpec:
    matrix: MatrixSpec
    state: str


def _candidate(path, middle, old_shape, new_shape):
    size = path.stat().st_size
    matches = []
    for state, shape in (("old", old_shape), ("new", new_shape)):
        elements = shape[0] * middle * shape[1]
        for itemsize, dtype in DTYPES.items():
            if size == elements * itemsize:
                matches.append(RecoverySpec(MatrixSpec(path, dtype, middle), state))
    if len(matches) != 1:
        labels = [(item.state, str(item.matrix.dtype)) for item in matches]
        raise ValueError(f"unrecognized or ambiguous matrix {path}: size={size}, matches={labels}")
    return matches[0]


def scan(stock_root, old_shape, new_shape):
    specs = []
    for path in Path(stock_root).rglob("*.bin"):
        parts = {part.lower() for part in path.relative_to(stock_root).parts}
        if "l2" in parts:
            continue
        specs.append(_candidate(path, _matrix_middle(path), old_shape, new_shape))
    if not specs:
        raise ValueError(f"no axis-backed matrices found under {stock_root}")
    return specs


def _atomic_save(path, values):
    temporary = path.with_name(path.name + ".recovering")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def recover(root, apply=False):
    root = Path(root)
    dates_path, ticks_path, dates, ticks = load_axes(root)
    old_shape = (len(dates), len(ticks))
    date_valid = int(np.count_nonzero(~np.isnat(dates)))
    tick_valid = int(np.count_nonzero(ticks != ""))
    new_shape = (
        max(old_shape[0], date_valid + DATE_RESERVE),
        max(old_shape[1], tick_valid + TICK_RESERVE),
    )
    if new_shape == old_shape:
        raise ValueError(f"axes already have full reserve: {old_shape}")

    specs = scan(root / "stock", old_shape, new_shape)
    old_count = sum(item.state == "old" for item in specs)
    new_count = len(specs) - old_count
    print(
        f"axes={old_shape}, target={new_shape}, matrices={len(specs)}, "
        f"old={old_count}, already_new={new_count}",
        flush=True,
    )
    if not apply:
        print("dry run only; pass --apply to migrate", flush=True)
        return

    lock_path = root / ".axis_recovery.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"recovery lock already exists: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started={datetime.now().isoformat()}\n".encode())
        os.close(descriptor)
        descriptor = None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = root / "axis" / f"recovery_backup_{stamp}"
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(dates_path, backup / dates_path.name)
        shutil.copy2(ticks_path, backup / ticks_path.name)
        print(f"axis backup: {backup}", flush=True)

        for index, item in enumerate(specs, 1):
            if item.state == "old":
                _resize_matrix(
                    item.matrix,
                    old_shape[0], old_shape[1],
                    new_shape[0], new_shape[1],
                )
            if index % 25 == 0 or index == len(specs):
                print(f"matrices checked/migrated: {index}/{len(specs)}", flush=True)

        verified = scan(root / "stock", new_shape, new_shape)
        if len(verified) != len(specs) or any(item.state != "old" for item in verified):
            raise RuntimeError("target-shape verification failed")

        new_dates = np.full(new_shape[0], np.datetime64("NaT"), dtype=dates.dtype)
        new_dates[:old_shape[0]] = dates
        new_ticks = np.full(new_shape[1], "", dtype=ticks.dtype)
        new_ticks[:old_shape[1]] = ticks
        _atomic_save(dates_path, new_dates)
        _atomic_save(ticks_path, new_ticks)

        _, _, check_dates, check_ticks = load_axes(root)
        if (len(check_dates), len(check_ticks)) != new_shape:
            raise RuntimeError("axis commit verification failed")
        print(f"recovery complete: axes={new_shape}, matrices={len(specs)}", flush=True)
        print(f"resume update_history from {dates[date_valid - 1]}", flush=True)
    finally:
        if 'descriptor' in locals() and descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    recover(args.root, apply=args.apply)


if __name__ == "__main__":
    main()
