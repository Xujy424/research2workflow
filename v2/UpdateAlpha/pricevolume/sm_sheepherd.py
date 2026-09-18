"""Small-money sheep-herd factor from Kaiyuan microstructure series (14)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class SMSHConfig:
    lookback_days: int = 20
    min_valid_days: int = 20
    return_field: str = "d_essentials/pct"
    moneyflow_folder: str = "d_moneyflow"

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")


class SMSHContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=SMSHConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)


def _daily_net_inflow(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Return buy minus sell, treating only one missing side as zero."""
    buy = np.asarray(buy, dtype=np.float64)
    sell = np.asarray(sell, dtype=np.float64)
    available = np.isfinite(buy) | np.isfinite(sell)
    return np.where(
        available,
        np.where(np.isfinite(buy), buy, 0.0)
        - np.where(np.isfinite(sell), sell, 0.0),
        np.nan,
    )


def _rank_columns(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ranked = pd.DataFrame(np.where(valid, values, np.nan)).rank(
        axis=0,
        method="average",
    )
    return ranked.to_numpy(dtype=np.float64)


def _spearman_by_column(
    x: np.ndarray,
    y: np.ndarray,
    min_valid: int,
) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    x_rank = _rank_columns(x, valid)
    y_rank = _rank_columns(y, valid)

    x_mean = np.divide(
        np.where(valid, x_rank, 0.0).sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    y_mean = np.divide(
        np.where(valid, y_rank, 0.0).sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    x_centered = np.where(valid, x_rank - x_mean, 0.0)
    y_centered = np.where(valid, y_rank - y_mean, 0.0)
    numerator = np.sum(x_centered * y_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=0)
        * np.sum(y_centered * y_centered, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=(count >= min_valid) & (denominator > 0),
    ).astype(np.float32)


class SMSHFactor(AlphaBase):
    """20D RankCorr(R_{t-1}, S_t) without looking ahead to S_{t+1}."""

    meta = AlphaMeta(
        "smsh",
        "20D RankCorr of lagged return and current small-order net inflow",
        direction=-1,
    )
    dependencies = (
        "d_essentials/pct",
        "d_moneyflow/buy_sm_amount",
        "d_moneyflow/sell_sm_amount",
    )

    def calculate(self, asof):
        cfg = self.context.config
        axis = self.context.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)

        data = self.context.data
        folder = cfg.moneyflow_folder
        returns = np.asarray(
            data.read(cfg.return_field, end, start_date=start),
            dtype=np.float64,
        ) / 100.0
        buy = np.asarray(
            data.read(f"{folder}/buy_sm_amount", end, start_date=start),
            dtype=np.float64,
        )
        sell = np.asarray(
            data.read(f"{folder}/sell_sm_amount", end, start_date=start),
            dtype=np.float64,
        )
        small_net = _daily_net_inflow(buy, sell)

        return _spearman_by_column(
            returns[:-1],
            small_net[1:],
            cfg.min_valid_days,
        )


__all__ = ["SMSHConfig", "SMSHContext", "SMSHFactor"]
