from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


_ROW_CHUNK = 64

if __package__:
    from ..config import ROOT
    from ..utils import asof
    from .reset_axis import (
        DATE_RESERVE,
        TICK_RESERVE,
        discover_matrix_specs,
        load_axes,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.config import ROOT
    from v2.UpdateData.utils import asof
    from v2.UpdateData.axis.reset_axis import (
        DATE_RESERVE,
        TICK_RESERVE,
        discover_matrix_specs,
        load_axes,
    )


@dataclass(frozen=True)
class AxisDeleteResult:
    cutoff: np.datetime64
    old_date_len: int
    old_tick_len: int
    old_date_valid: int
    old_tick_valid: int
    new_date_len: int
    new_tick_len: int
    new_date_valid: int
    new_tick_valid: int
    dropped_dates: int
    dropped_ticks: int
    matrices: int


def _matrix_shape(spec, date_len, tick_len):
    if spec.middle == 1:
        return date_len, tick_len
    return date_len, spec.middle, tick_len


def _active_tick_mask(
    close_spec,
    old_date_len,
    old_tick_len,
    valid_tick_count,
    keep_rows,
):
    """active=True保留的是在cutoff之后仍有数据的股票."""
    close = np.memmap(
        close_spec.path,
        dtype=close_spec.dtype,
        mode="r",
        shape=(old_date_len, old_tick_len),
    )
    rows = slice(keep_rows[0], keep_rows[-1] + 1)
    active = ~np.all(np.isnan(close[rows, :valid_tick_count]), axis=0)
    del close
    return active


def _rewrite_with_temp(
    spec,
    old_date_len,
    old_tick_len,
    new_date_len,
    new_tick_len,
    keep_rows,
    keep_cols,
):
    """Fallback when the target cannot safely fit inside the source file."""
    old = np.memmap(
        spec.path,
        dtype=spec.dtype,
        mode="r",
        shape=_matrix_shape(spec, old_date_len, old_tick_len),
    )
    scratch_path = spec.path.with_suffix(spec.path.suffix + ".tmp")
    new_shape = _matrix_shape(spec, new_date_len, new_tick_len)
    with scratch_path.open("wb") as file:
        file.truncate(int(np.prod(new_shape)) * spec.dtype.itemsize)
    new = np.memmap(
        scratch_path, dtype=spec.dtype, mode="r+", shape=new_shape
    )
    fill_value = False if spec.dtype == np.dtype(np.bool_) else np.nan
    for start in range(0, len(keep_rows), _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, len(keep_rows))
        rows = slice(keep_rows[start], keep_rows[stop - 1] + 1)
        new[start:stop, ..., :len(keep_cols)] = np.take(
            old[rows], keep_cols, axis=-1
        )
        new[start:stop, ..., len(keep_cols):] = fill_value
    new[len(keep_rows):] = fill_value
    new.flush()
    del new
    del old
    scratch_path.replace(spec.path)


def _rewrite_matrix(
    spec,
    old_date_len,
    old_tick_len,
    new_date_len,
    new_tick_len,
    keep_rows,
    keep_cols,
):
    """Compact in place using one target date row as extra memory."""
    old_width = spec.middle * old_tick_len
    new_width = spec.middle * new_tick_len
    old_elements = old_date_len * old_width
    new_elements = new_date_len * new_width
    if new_width > old_width or new_elements > old_elements:
        _rewrite_with_temp(
            spec,
            old_date_len,
            old_tick_len,
            new_date_len,
            new_tick_len,
            keep_rows,
            keep_cols,
        )
        return

    fill_value = False if spec.dtype == np.dtype(np.bool_) else np.nan
    flat = np.memmap(
        spec.path,
        dtype=spec.dtype,
        mode="r+",
        shape=(old_elements,),
    )
    old = flat.reshape(old_date_len, spec.middle, old_tick_len)
    for start in range(0, len(keep_rows), _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, len(keep_rows))
        rows = slice(keep_rows[start], keep_rows[stop - 1] + 1)
        retained = np.take(old[rows], keep_cols, axis=2)
        target = flat[start * new_width:stop * new_width].reshape(
            stop - start,
            spec.middle,
            new_tick_len,
        )
        target[:, :, :len(keep_cols)] = retained
        target[:, :, len(keep_cols):] = fill_value
    del target
    del retained
    flat.flush()
    del old
    del flat

    with spec.path.open("r+b") as file:
        file.truncate(new_elements * spec.dtype.itemsize)
    new = np.memmap(
        spec.path,
        dtype=spec.dtype,
        mode="r+",
        shape=_matrix_shape(spec, new_date_len, new_tick_len),
    )
    new[len(keep_rows):] = fill_value
    new.flush()
    del new


def delete_before(
    cutoff,
    root=ROOT,
    *,
    date_reserve=DATE_RESERVE,
    tick_reserve=TICK_RESERVE,
    dry_run=False,
):
    """
    Drop data before cutoff and remove stock columns with no retained values.

    Dates are kept when date >= cutoff. Stocks are kept when daily close has
    at least one finite value in that range. Axis files are saved last.
    """
    cutoff_value = np.datetime64(asof(cutoff).date(), "D")
    dates_path, ticks_path, dates, ticks = load_axes(root)
    old_date_valid = int(np.count_nonzero(~np.isnat(dates)))
    old_tick_valid = int(np.count_nonzero(ticks != ""))
    old_date_len, old_tick_len = len(dates), len(ticks)
    valid_dates = dates[:old_date_valid].astype("datetime64[D]")
    valid_ticks = ticks[:old_tick_valid]

    keep_row_mask = valid_dates >= cutoff_value
    if not np.any(keep_row_mask):
        raise ValueError(f"cutoff removes every valid date: {cutoff_value}")
    keep_rows = np.flatnonzero(keep_row_mask)

    specs = discover_matrix_specs(root, old_date_len, old_tick_len)
    close_path = Path(root) / "stock" / "d_essentials" / "close.bin"
    close_spec = next(
        (spec for spec in specs if spec.path == close_path),
        None,
    )
    if close_spec is None:
        raise FileNotFoundError(
            f"daily close matrix is required to identify stocks: {close_path}"
        )
    if close_spec.middle != 1 or not np.issubdtype(
        close_spec.dtype, np.floating
    ):
        raise ValueError(f"invalid daily close matrix: {close_path}")
    
    active = _active_tick_mask(
        close_spec,
        old_date_len,
        old_tick_len,
        old_tick_valid,
        keep_rows,
    )
    keep_cols = np.flatnonzero(active)
    if len(keep_cols) == 0:
        raise ValueError("no stock has a finite close after the cutoff")

    new_date_valid = len(keep_rows)
    new_tick_valid = len(keep_cols)
    new_date_len = new_date_valid + int(date_reserve)
    new_tick_len = new_tick_valid + int(tick_reserve)
    result = AxisDeleteResult(
        cutoff=cutoff_value,
        old_date_len=old_date_len,
        old_tick_len=old_tick_len,
        old_date_valid=old_date_valid,
        old_tick_valid=old_tick_valid,
        new_date_len=new_date_len,
        new_tick_len=new_tick_len,
        new_date_valid=new_date_valid,
        new_tick_valid=new_tick_valid,
        dropped_dates=old_date_valid - new_date_valid,
        dropped_ticks=old_tick_valid - new_tick_valid,
        matrices=len(specs),
    )
    if dry_run:
        return result

    for spec in specs:
        _rewrite_matrix(
            spec,
            old_date_len,
            old_tick_len,
            new_date_len,
            new_tick_len,
            keep_rows,
            keep_cols,
        )

    new_dates = np.full(
        new_date_len, np.datetime64("NaT"), dtype=dates.dtype
    )
    new_dates[:new_date_valid] = valid_dates[keep_rows]
    new_ticks = np.full(new_tick_len, "", dtype=ticks.dtype)
    new_ticks[:new_tick_valid] = valid_ticks[keep_cols]

    np.save(dates_path, new_dates, allow_pickle=False)
    np.save(ticks_path, new_ticks, allow_pickle=False)
    return result


__all__ = ["AxisDeleteResult", "delete_before"]
