from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def asof(date) -> pd.Timestamp:
    """Normalize a date-like value to a timezone-naive midnight Timestamp."""
    value = pd.Timestamp(date)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


def date_string(date, compact: bool = False) -> str:
    """Return an ISO basic or extended date string."""
    return asof(date).strftime("%Y%m%d" if compact else "%Y-%m-%d")


def date_index(date, dates, *, require_present: bool = True) -> int:
    """Locate a date in a reserved datetime64 axis."""
    target = np.datetime64(asof(date).date(), "D")
    axis = np.asarray(dates, dtype="datetime64[D]")
    valid = axis[~np.isnat(axis)]
    index = int(np.searchsorted(valid, target))
    if require_present and (index >= len(valid) or valid[index] != target):
        raise ValueError(f"date is not present in dates axis: {target}")
    return index


def valid_stock_ticks(ticks) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized non-empty stock codes and their axis positions."""
    values: list[str] = []
    positions: list[int] = []
    for position, tick in enumerate(np.asarray(ticks)):
        if tick is None or pd.isna(tick):
            continue
        value = str(tick).strip()
        if not value:
            continue
        values.append(value.zfill(6))
        positions.append(position)
    return np.asarray(values, dtype="<U6"), np.asarray(positions, dtype=np.int64)


def ensure_memmap(
    path,
    shape,
    dtype=np.float32,
    fill_value=None,
    mode: str = "r+",
) -> np.memmap:
    """Create or validate a raw matrix file and return its memmap."""
    path = Path(path)
    shape = tuple(int(value) for value in shape)
    dtype = np.dtype(dtype)
    if any(value <= 0 for value in shape):
        raise ValueError(f"invalid memmap shape for {path}: {shape}")
    if fill_value is None:
        fill_value = False if dtype == np.dtype(np.bool_) else np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if not path.exists():
        with path.open("wb") as file:
            file.truncate(expected_size)
        array = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
        array[:] = fill_value
        array.flush()

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size {actual_size} does not match "
            f"dtype={dtype}, shape={shape}, expected={expected_size}"
        )
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
