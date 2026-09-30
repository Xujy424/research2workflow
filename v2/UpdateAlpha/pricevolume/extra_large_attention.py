"""Extra-large-order attention factor from Kaiyuan microstructure series (20)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..operators import selected_mean_diff, time_series_zscore
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.UpdateAlpha.operators import selected_mean_diff, time_series_zscore
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class ExtraLargeAttentionConfig:
    lookback_days: int = 20
    selected_days: int = 4
    min_valid_days: int = 16
    moneyflow_folder: str = "d_moneyflow"

    def __post_init__(self):
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if not 1 <= self.selected_days * 2 <= self.lookback_days:
            raise ValueError("selected_days must not exceed half the lookback")
        if not 2 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [2, lookback_days]")


class ExtraLargeAttentionContext(AlphaContext):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        config=ExtraLargeAttentionConfig(),
        universe="self",
    ):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def read_window(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        folder = self.config.moneyflow_folder
        return {
            name: np.asarray(
                self.data.read(f"{folder}/{field}", end, start_date=start),
                dtype=np.float64,
            )
            for name, field in {
                "extra_large_buy": "buy_elg_amount",
                "extra_large_sell": "sell_elg_amount",
                "small_buy": "buy_sm_amount",
                "small_sell": "sell_sm_amount",
            }.items()
        }


class ExtraLargeAttentionFactor(AlphaBase):
    meta = AlphaMeta(
        "extra_large_attention",
        "20D extra-large-order attention cut by small-order strength",
        direction=1,
    )
    dependencies = (
        "d_moneyflow/buy_elg_amount",
        "d_moneyflow/sell_elg_amount",
        "d_moneyflow/buy_sm_amount",
        "d_moneyflow/sell_sm_amount",
    )

    @staticmethod
    def _daily_net_flow(buy, sell):
        available = np.isfinite(buy) | np.isfinite(sell)
        return np.where(
            available,
            np.where(np.isfinite(buy), buy, 0.0)
            - np.where(np.isfinite(sell), sell, 0.0),
            np.nan,
        )

    def calculate(self, asof):
        raw = self.context.read_window(asof)
        tick_count = self.context.data.axis.tick_count
        if raw is None:
            return np.full(tick_count, np.nan, dtype=np.float32)
        cfg = self.context.config
        extra_large = self._daily_net_flow(
            raw["extra_large_buy"], raw["extra_large_sell"]
        )
        small = self._daily_net_flow(raw["small_buy"], raw["small_sell"])
        extra_large = time_series_zscore(extra_large, cfg.min_valid_days)
        small = time_series_zscore(small, cfg.min_valid_days)
        return selected_mean_diff(
            extra_large,
            small,
            cfg.selected_days,
            cfg.min_valid_days,
        ).astype(np.float32)


__all__ = [
    "ExtraLargeAttentionConfig",
    "ExtraLargeAttentionContext",
    "ExtraLargeAttentionFactor",
]
