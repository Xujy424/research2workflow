"""Daily-price and liquidity factors from the 2026 factor calendar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
from ..operators import safe_ratio_return
from ...GetData import DataPool
from ...UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class CalendarDailyConfig:
    lookback_days: int = 20
    min_valid_days: int = 12

    def __post_init__(self):
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be within lookback_days")


class CalendarDailyContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=CalendarDailyConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def window(self, asof, fields, *, previous=False):
        end = self.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1 - int(previous)
        if start < 0:
            return None
        return {
            field: np.asarray(self.data.read(field, end, start), dtype=np.float64)
            for field in fields
        }


def _mean(values, min_count):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.divide(
        np.where(valid, values, 0.0).sum(axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count >= min_count,
    )


def _sum(values, min_count):
    valid = np.isfinite(values)
    result = np.where(valid, values, 0.0).sum(axis=0)
    result[valid.sum(axis=0) < min_count] = np.nan
    return result


class _CalendarDailyFactor(AlphaBase):
    def empty(self):
        return np.full(self.context.data.axis.tick_count, np.nan, dtype=np.float32)

    def finish(self, values):
        return np.asarray(values, dtype=np.float32)
