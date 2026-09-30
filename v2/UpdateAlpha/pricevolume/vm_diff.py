"""VM_Diff factor from Kaiyuan microstructure series (20)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..operators import selected_mean_diff
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.UpdateAlpha.operators import selected_mean_diff
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class VMDiffConfig:
    lookback_days: int = 20
    selected_days: int = 4
    min_valid_days: int = 16
    min_minutes: int = 120

    def __post_init__(self):
        if not 1 <= self.selected_days * 2 <= self.lookback_days:
            raise ValueError("selected_days must not exceed half the lookback")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if self.min_minutes < 2:
            raise ValueError("min_minutes must be at least 2")


class VMDiffContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=VMDiffConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def read_window(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        return {
            "minute_close": np.asarray(
                self.data.read("m_essentials/close", end, start_date=start),
                dtype=np.float64,
            ),
            "close": np.asarray(
                self.data.read(
                    "d_essentials/close_ratio_adj", end, start_date=start
                ),
                dtype=np.float64,
            ),
        }


class VMDiffFactor(AlphaBase):
    meta = AlphaMeta(
        "vm_diff",
        "20D high-price minus low-price minute-return volatility",
        direction=-1,
    )
    dependencies = (
        "m_essentials/close",
        "d_essentials/close_ratio_adj",
    )

    @staticmethod
    def _minute_return_volatility(minute_close, min_minutes):
        close = np.asarray(minute_close, dtype=np.float64)
        returns = np.divide(
            close[:, 1:],
            close[:, :-1],
            out=np.full_like(close[:, 1:], np.nan),
            where=(
                np.isfinite(close[:, 1:])
                & np.isfinite(close[:, :-1])
                & (close[:, 1:] > 0)
                & (close[:, :-1] > 0)
            ),
        ) - 1.0
        valid = np.isfinite(returns)
        count = valid.sum(axis=1)
        mean = np.divide(
            np.where(valid, returns, 0.0).sum(axis=1),
            count,
            out=np.full(count.shape, np.nan, dtype=np.float64),
            where=count > 0,
        )
        centered = np.where(valid, returns - mean[:, None, :], 0.0)
        return np.sqrt(np.divide(
            np.sum(centered * centered, axis=1),
            count,
            out=np.full(count.shape, np.nan, dtype=np.float64),
            where=count >= min_minutes,
        ))

    def calculate(self, asof):
        raw = self.context.read_window(asof)
        tick_count = self.context.data.axis.tick_count
        if raw is None:
            return np.full(tick_count, np.nan, dtype=np.float32)
        cfg = self.context.config
        volatility = self._minute_return_volatility(
            raw["minute_close"], cfg.min_minutes
        )
        return selected_mean_diff(
            volatility,
            raw["close"],
            cfg.selected_days,
            cfg.min_valid_days,
        ).astype(np.float32)


__all__ = ["VMDiffConfig", "VMDiffContext", "VMDiffFactor"]
