"""Common contract and matrix storage for alpha factors."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import numpy as np


@dataclass(frozen=True)
class AlphaMeta:
    name: str
    description: str
    frequency: str = "daily"
    direction: int = 1

class AlphaContext:
    """Minimal data environment shared by every alpha factor."""

    def __init__(self, data):
        self.data = data

    def close(self):
        self.data.close()
        if getattr(self, "_owns_conn", False):
            connection = getattr(self, "conn", None)
            if connection is not None:
                connection.close()

    def align(self, frame, value="value"):
        """Align a tick/value table to the valid local instrument axis."""
        axis = self.data.axis
        out = np.full(axis.tick_count, np.nan, dtype=np.float32)
        for tick, item in frame.select("tick", value).iter_rows():
            position = axis._tick_positions.get(str(tick).strip().zfill(6))
            if position is not None and item is not None and np.isfinite(item):
                out[position] = item
        return out

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AlphaBase(ABC):
    """Base class for one date-by-instrument float32 alpha matrix."""

    meta: ClassVar[AlphaMeta]
    dependencies: ClassVar[tuple[str, ...]] = ()

    def __init__(self, context):
        self.context = context

    @abstractmethod
    def calculate(self, asof) -> np.ndarray:
        """Return one cross-section aligned to valid instrument ticks."""

    def _cal_align(self, asof) -> np.ndarray:
        values = np.asarray(self.calculate(asof), dtype=np.float32)
        axis = self.context.data.axis
        if values.ndim != 1:
            raise ValueError(f"{self.meta.name} must return a 1-D array")
        if len(values) == axis.tick_count:
            full = np.full(len(axis.full_ticks), np.nan, dtype=np.float32)
            full[:axis.tick_count] = values
            return full
        if len(values) == len(axis.full_ticks):
            return values
        raise ValueError(
            f"{self.meta.name} length {len(values)} does not match "
            f"valid/full tick axes {axis.tick_count}/{len(axis.full_ticks)}"
        )

    def output_path(self, folder="factor_pool") -> Path:
        data = self.context.data
        return data.root / data.asset / folder / f"{self.meta.name}.bin"

    def update(self, asof, folder="factor_pool") -> np.ndarray:
        """Calculate and write the cross-section to its axis date row."""
        axis = self.context.data.axis
        row = axis.date_position(asof)
        values = self._cal_align(asof)
        path = self.output_path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (len(axis.full_dates), len(axis.full_ticks))
        expected_size = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        if not path.exists():
            matrix = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
            matrix[:] = np.nan
        else:
            if path.stat().st_size != expected_size:
                raise ValueError(f"{path} size does not match axes {shape}")
            matrix = np.memmap(path, dtype=np.float32, mode="r+", shape=shape)
        matrix[row] = values
        matrix.flush()
        del matrix
        return values

    def __call__(self, asof) -> np.ndarray:
        return self._cal_align(asof)


__all__ = ["AlphaBase", "AlphaContext", "AlphaMeta"]
