"""Common contract and matrix storage for alpha factors."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import numpy as np

from ..ResearchFlow.matrix_math import (
    cross_sectional_zscore,
    industry_size_neutralize,
    winsorize,
)


@dataclass(frozen=True)
class AlphaMeta:
    name: str
    description: str
    frequency: str = "daily"
    direction: int = 1


@dataclass(frozen=True)
class AlphaPreprocessConfig:
    """Shared processing applied to every factor before it is stored."""

    enabled: bool = True
    winsor_method: str = "mad"
    quantile_p: float = 0.01
    n_sigma: float = 3.0
    neutralize: bool = True
    standardize: bool = True
    tradable_field: str = "basic/tradable"
    industry_field: str = "industry/industry"
    market_cap_field: str = "d_essentials/total_mv"

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
    preprocess_config: ClassVar[AlphaPreprocessConfig] = AlphaPreprocessConfig()

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

    def filter_tradable(self, asof, values: np.ndarray) -> np.ndarray:
        """Mask stocks that cannot be traded on the signal/execution date."""
        axis = self.context.data.axis
        row = axis.date_position(asof)
        n = axis.tick_count
        tradable = np.asarray(
            self.context.data.read(self.preprocess_config.tradable_field, row),
            dtype=bool,
        )[:n]
        result = np.full_like(values, np.nan, dtype=float)
        result[:n] = np.where(tradable, values[:n], np.nan)
        return result

    def winsorize(self, values: np.ndarray) -> np.ndarray:
        """Apply cross-sectional robust winsorization."""
        config = self.preprocess_config
        return winsorize(
            values[None, :],
            method=config.winsor_method,
            p=config.quantile_p,
            n_sigma=config.n_sigma,
        )[0]

    def neutralize(self, asof, values: np.ndarray) -> np.ndarray:
        """Remove same-day industry and prior-day log-size exposures."""
        axis = self.context.data.axis
        row = axis.date_position(asof)
        n = axis.tick_count
        industry = np.asarray(
            self.context.data.read(self.preprocess_config.industry_field, row),
            dtype=float,
        )[:n]
        market_cap = np.asarray(
            self.context.data.read(
                self.preprocess_config.market_cap_field, max(0, row - 1)
            ),
            dtype=float,
        )[:n]
        residual = industry_size_neutralize(
            values[None, :n],
            industry[None, :],
            market_cap[None, :],
            mask=np.isfinite(values[:n])[None, :],
            standardize=False,
        )[0]
        result = np.full_like(values, np.nan, dtype=float)
        result[:n] = residual
        return result

    def standardize(self, values: np.ndarray) -> np.ndarray:
        """Convert the valid cross-section to zero-mean, unit-variance scores."""
        return cross_sectional_zscore(values[None, :])[0]

    def preprocess(self, asof, values: np.ndarray) -> np.ndarray:
        """Run tradable -> winsorize -> neutralize -> standardize."""
        config = self.preprocess_config
        if not config.enabled:
            return values
        result = self.filter_tradable(asof, values)
        result = self.winsorize(result)
        if config.neutralize:
            result = self.neutralize(asof, result)
        if config.standardize:
            result = self.standardize(result)
        return result.astype(np.float32, copy=False)

    def output_path(self, folder="factor_pool") -> Path:
        data = self.context.data
        return data.root / data.asset / folder / f"{self.meta.name}.bin"

    def update(self, asof, folder="factor_pool") -> np.ndarray:
        """Calculate and write the cross-section to its axis date row."""
        axis = self.context.data.axis
        row = axis.date_position(asof)
        values = self.preprocess(asof, self._cal_align(asof))
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


__all__ = [
    "AlphaBase",
    "AlphaContext",
    "AlphaMeta",
    "AlphaPreprocessConfig",
]
