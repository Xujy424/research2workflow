"""Daily cross-sectional long/short constituent masks for Barra factors."""

from pathlib import Path
import sys
import os
import tempfile

import numpy as np

if __package__:
    from .update_barra import BARRA_NAMES
    from ..utils import date_index
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.stock.update_barra import BARRA_NAMES
    from v2.UpdateData.utils import date_index


def _group_mask(values, ticks):
    """Top/bottom floor(N / 10) finite stocks get +1/-1; others get 0.

    Empty stock-axis slots are excluded. Ties follow stock-axis order,
    so both tails have the same size and never overlap.
    """
    mask = np.zeros(len(ticks), dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(values) & (ticks != ""))
    count = len(valid) // 10
    if count:
        ordered = valid[np.argsort(values[valid], kind="stable")]
        mask[ordered[:count]] = -1
        mask[ordered[-count:]] = 1
    return mask


def _check_size(file, expected):
    size = os.fstat(file.fileno()).st_size
    if size != expected:
        raise ValueError(f'{file.name}: size {size} does not match axes; expected {expected}')


def _write_bytes(file, data):
    if file.write(data) != len(data):
        raise OSError(f'{file.name}: incomplete mask write')


def _write_rows(path, shape, start, masks):
    """Write just the affected rows; publish new files only after initialization.

    Like the other UpdateData writers, this assumes one updater per output.
    Existing malformed files are rejected, never resized or overwritten.
    """
    expected = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    try:
        file = path.open('r+b')
    except FileNotFoundError:
        # Bounded sequential writes avoid dirtying a whole network memmap.
        # A crash leaves a .tmp file, rather than an incomplete final .bin.
        temporary = None
        print(f'[Barra groups] initialize {path.name}: {expected / 2**20:.1f} MiB', flush=True)
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                             suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                block = np.full(min(expected//4, 1 << 20), np.nan, dtype=np.float32).tobytes()
                remaining = expected
                while remaining:
                    chunk = block[:min(remaining, len(block))]
                    _write_bytes(file, chunk)
                    remaining -= len(chunk)
                file.seek(start * shape[1] * 4)
                _write_bytes(file, masks.tobytes())
                file.flush()
                os.fsync(file.fileno())
                _check_size(file, expected)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return
    with file:
        _check_size(file, expected)
        file.seek(start * shape[1] * 4)
        _write_bytes(file, masks.tobytes())
        file.flush()
        os.fsync(file.fileno())


def _update_rows(start, stop, dates, ticks, root):
    """Bounded contiguous I/O shared by daily and historical updates."""
    shape = (len(dates), len(ticks))
    folder = root / 'index' / 'mask'
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in BARRA_NAMES:
        source = root / "barra" / f"{name}.bin"
        with source.open('rb') as file:
            _check_size(file, shape[0] * shape[1] * 8)
            file.seek(start * shape[1] * 8)
            values = np.fromfile(file, dtype=np.float64, count=(stop-start)*shape[1])
        if values.size != (stop-start)*shape[1]:
            raise ValueError(f'{source}: incomplete input read')
        masks = np.array([_group_mask(row, ticks) for row in values.reshape(-1, shape[1])])
        _write_rows(folder / f'{name}_mask.bin', shape, start, masks)
        result[name] = masks
    return result


def update_barra_group(date, dates, ticks, conn, root):
    """Write float32 masks from float64 factors, using the UpdateData interface."""
    dt = date_index(date, dates)
    return {name: rows[0] for name, rows in
            _update_rows(dt, dt+1, dates, ticks, Path(root)).items()}


def update_history(root, start_date=None, end_date=None):
    """Build masks from existing Barra data without database queries.

    root is the data directory containing axis/ and stock/. Omitted
    bounds include all populated dates; reserved NaT rows are skipped.
    """
    root = Path(root)
    dates = np.load(root / "axis" / "dates.npy", allow_pickle=False)
    ticks = np.load(root / "axis" / "stock_ticks.npy", allow_pickle=False)
    start = np.datetime64(start_date, "D") if start_date is not None else None
    end = np.datetime64(end_date, "D") if end_date is not None else None
    if (start is not None and np.isnat(start)) or (end is not None and np.isnat(end)):
        raise ValueError("date bounds must not be NaT")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be later than end_date")
    selected = ~np.isnat(dates)
    if start is not None:
        selected &= dates >= start
    if end is not None:
        selected &= dates <= end
    indices = np.flatnonzero(selected)
    # Never bridge missing/reserved rows; bound each batch to 64 dates.
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1):
        for offset in range(0, len(run), 64):
            batch = run[offset:offset+64]
            _update_rows(int(batch[0]), int(batch[-1])+1, dates, ticks, root / 'stock')
            print(f'[OK] Barra groups: {dates[batch[0]]} .. {dates[batch[-1]]}', flush=True)


if __name__ == "__main__":
    from v2.UpdateData.config import ROOT

    update_history(root=ROOT)
