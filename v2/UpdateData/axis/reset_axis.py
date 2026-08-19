from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__:
    from ..config import ROOT
    from ..utils import ensure_memmap
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.config import ROOT
    from v2.UpdateData.utils import ensure_memmap


DATE_RESERVE = 500
TICK_RESERVE = 1000
FORECAST_FOLDERS = {
    "con_forecast", "con_forecast_eq", "con_forecast_wgt",
}


@dataclass(frozen=True)
class AxisResize:
    date_valid: int
    tick_valid: int
    old_date_len: int
    old_tick_len: int
    new_date_len: int
    new_tick_len: int

    @property
    def changed(self) -> bool:
        return (
            self.old_date_len != self.new_date_len
            or self.old_tick_len != self.new_tick_len
        )


@dataclass(frozen=True)
class MatrixSpec:
    path: Path
    dtype: np.dtype
    middle: int


def axis_paths(root=ROOT):
    axis_dir = Path(root) / "axis"
    return axis_dir / "dates.npy", axis_dir / "stock_ticks.npy"


def init_axis(root=ROOT, date_reserve=DATE_RESERVE, tick_reserve=5000):
    """Create empty reserved axes when the index files do not exist."""
    dates_path, ticks_path = axis_paths(root)
    dates_path.parent.mkdir(parents=True, exist_ok=True)

    if not dates_path.exists():
        np.save(
            dates_path,
            np.full(date_reserve, np.datetime64("NaT"), dtype="datetime64[D]"),
            allow_pickle=False,
        )

    if not ticks_path.exists():
        stock_ticks = np.full(tick_reserve, "", dtype="<U6")
        np.save(ticks_path, stock_ticks, allow_pickle=False)

    return dates_path, ticks_path


def load_axes(root=ROOT):
    dates_path, ticks_path = init_axis(root)
    dates = np.load(dates_path, allow_pickle=False)
    ticks = np.load(ticks_path, allow_pickle=False)
    if np.any(np.isnat(dates[:-1]) & ~np.isnat(dates[1:])):
        raise ValueError("dates.npy must keep all NaT reserve rows at the end")
    if np.any((ticks[:-1] == "") & (ticks[1:] != "")):
        raise ValueError(
            "stock_ticks.npy must keep all empty reserve columns at the end"
        )
    return dates_path, ticks_path, dates, ticks


def _matrix_middle(path):
    parent = path.parent.name.lower()
    if parent == "m_essentials":
        return 241
    if parent in FORECAST_FOLDERS:
        return 4
    return 1


def discover_matrix_specs(root, date_len, tick_len):
    """Identify all axis-backed matrices, explicitly excluding Level-2."""
    specs = []
    base = int(date_len) * int(tick_len)
    if base <= 0:
        return specs

    root = Path(root)
    for path in root.rglob("*.bin"):
        relative_parts = {
            part.lower() for part in path.relative_to(root).parts
        }
        if "l2" in relative_parts:
            continue
        middle = _matrix_middle(path)
        elements = base * middle
        size = path.stat().st_size
        if size % elements:
            raise ValueError(
                f"cannot infer matrix dtype/shape for {path}: "
                f"size={size}, axes=({date_len}, {middle}, {tick_len})"
            )
        itemsize = size // elements
        dtype_by_size = {
            1: np.dtype(np.bool_),
            2: np.dtype(np.float16),
            4: np.dtype(np.float32),
            8: np.dtype(np.float64),
        }
        if itemsize not in dtype_by_size:
            raise ValueError(
                f"unsupported itemsize {itemsize} for matrix {path}"
            )
        specs.append(MatrixSpec(path, dtype_by_size[itemsize], middle))
    return specs


def _resize_matrix(spec, old_dates, old_ticks, new_dates, new_ticks):
    if new_dates < old_dates or new_ticks < old_ticks:
        raise ValueError("axis shrinking is not supported")
    if new_dates == old_dates and new_ticks == old_ticks:
        return

    old_width = spec.middle * old_ticks
    new_width = spec.middle * new_ticks
    new_elements = new_dates * new_width
    fill_value = False if spec.dtype == np.dtype(np.bool_) else np.nan

    with spec.path.open("r+b") as file:
        file.truncate(new_elements * spec.dtype.itemsize)
    flat = np.memmap(
        spec.path, dtype=spec.dtype, mode="r+", shape=(new_elements,)
    )
    for row_index in range(old_dates - 1, -1, -1):
        source_start = row_index * old_width
        target_start = row_index * new_width
        old_row = flat[
            source_start:source_start + old_width
        ].copy().reshape(spec.middle, old_ticks)
        target = flat[
            target_start:target_start + new_width
        ].reshape(spec.middle, new_ticks)
        target[:] = fill_value
        target[:, :old_ticks] = old_row
    if new_dates > old_dates:
        flat[old_dates * new_width:] = fill_value
    flat.flush()
    del flat


def insert_matrix_date_rows(root, index, n_valid, date_len, tick_len):
    """Insert one empty date row into every matrix without changing axes."""
    specs = discover_matrix_specs(root, date_len, tick_len)
    for spec in specs:
        shape = (
            (date_len, tick_len)
            if spec.middle == 1
            else (date_len, spec.middle, tick_len)
        )
        array = np.memmap(
            spec.path, dtype=spec.dtype, mode="r+", shape=shape
        )
        for row_index in range(n_valid - 1, index - 1, -1):
            array[row_index + 1] = array[row_index]
        array[index] = (
            False if spec.dtype == np.dtype(np.bool_) else np.nan
        )
        array.flush()
        del array


def ensure_axis_capacity(
    root=ROOT,
    *,
    min_date_free=0,
    min_tick_free=0,
):
    """Grow axes and every matrix together; Level-2 files are excluded."""
    dates_path, ticks_path, dates, ticks = load_axes(root)
    date_valid = int(np.count_nonzero(~np.isnat(dates)))
    tick_valid = int(np.count_nonzero(ticks != ""))
    old_date_len, old_tick_len = len(dates), len(ticks)

    new_date_len = old_date_len
    if old_date_len - date_valid < min_date_free:
        new_date_len = date_valid + max(DATE_RESERVE, min_date_free)
    new_tick_len = old_tick_len
    if old_tick_len - tick_valid < min_tick_free:
        new_tick_len = tick_valid + max(TICK_RESERVE, min_tick_free)

    result = AxisResize(
        date_valid, tick_valid,
        old_date_len, old_tick_len,
        new_date_len, new_tick_len,
    )
    if not result.changed:
        return result

    specs = discover_matrix_specs(root, old_date_len, old_tick_len)
    for spec in specs:
        _resize_matrix(
            spec,
            old_date_len,
            old_tick_len,
            new_date_len,
            new_tick_len,
        )

    new_dates = np.full(
        new_date_len, np.datetime64("NaT"), dtype=dates.dtype
    )
    new_dates[:date_valid] = dates[:date_valid]
    np.save(dates_path, new_dates, allow_pickle=False)

    new_ticks = np.full(new_tick_len, "", dtype="<U6")
    new_ticks[:tick_valid] = ticks[:tick_valid]
    np.save(ticks_path, new_ticks, allow_pickle=False)
    return result


def reset_axis(root=ROOT):
    """Ensure a full annual reserve on both axes and resize matrices."""
    return ensure_axis_capacity(
        root,
        min_date_free=DATE_RESERVE,
        min_tick_free=TICK_RESERVE,
    )


def reset_field_axis(root, *args, **kwargs):
    """Compatibility validator; reset_axis already resizes every matrix."""
    dates_path, ticks_path = init_axis(root)
    dates = np.load(dates_path, allow_pickle=False)
    ticks = np.load(ticks_path, allow_pickle=False)
    discover_matrix_specs(root, len(dates), len(ticks))
    return len(dates), len(ticks)


def init_empty_field(
    dates,
    ticks,
    fileshare,
    name,
    typ,
    dim=None,
    root=ROOT,
):
    shape = (
        (len(dates), len(ticks))
        if dim is None
        else (len(dates), int(dim), len(ticks))
    )
    return ensure_memmap(
        Path(root) / fileshare / f"{name}.bin",
        shape,
        dtype=typ,
    )
