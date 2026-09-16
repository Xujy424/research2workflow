"""Large- and small-order money-flow strength factors.

The definitions follow Kaiyuan Securities' market microstructure series (12):

    S = sum(buy - sell) / sum(abs(buy - sell))

over the latest 20 trading days.  The improved factor is the cross-sectional
OLS residual from ``S = intercept + beta * Ret20 + residual``.
"""

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
class MoneyflowStrengthConfig:
    lookback_days: int = 20
    min_valid_days: int = 20
    min_cross_section_observations: int = 30
    return_field: str = "d_essentials/pct"
    moneyflow_folder: str = "d_moneyflow"

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if self.min_cross_section_observations < 2:
            raise ValueError("min_cross_section_observations must be at least 2")


def _daily_net_inflow(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Return buy minus sell, treating only an absent leg as zero."""
    buy = np.asarray(buy, dtype=np.float64)
    sell = np.asarray(sell, dtype=np.float64)
    available = np.isfinite(buy) | np.isfinite(sell)
    return np.where(
        available,
        np.where(np.isfinite(buy), buy, 0.0)
        - np.where(np.isfinite(sell), sell, 0.0),
        np.nan,
    )


def _moneyflow_strength(
    buy: np.ndarray,
    sell: np.ndarray,
    min_valid_days: int,
) -> np.ndarray:
    """Compute report S3: sum(net inflow) / sum(abs(net inflow))."""
    net = _daily_net_inflow(buy, sell)
    valid = np.isfinite(net)
    numerator = np.where(valid, net, 0.0).sum(axis=0)
    denominator = np.where(valid, np.abs(net), 0.0).sum(axis=0)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(net.shape[1], np.nan, dtype=np.float64),
        where=(valid.sum(axis=0) >= min_valid_days) & (denominator > 0),
    )
    return result.astype(np.float32)


def _compound_return(
    daily_pct: np.ndarray,
    min_valid_days: int,
) -> np.ndarray:
    """Compound percentage-point daily returns into Ret20."""
    daily_return = np.asarray(daily_pct, dtype=np.float64) / 100.0
    valid = np.isfinite(daily_return)
    result = np.prod(1.0 + np.where(valid, daily_return, 0.0), axis=0) - 1.0
    result[valid.sum(axis=0) < min_valid_days] = np.nan
    return result


def _cross_section_residual(
    y: np.ndarray,
    x: np.ndarray,
    min_observations: int,
) -> np.ndarray:
    """OLS residual of y on an intercept and x for one cross-section."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(x)
    result = np.full_like(y, np.nan)
    if valid.sum() < min_observations:
        return result.astype(np.float32)
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    coefficient, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    result[valid] = y[valid] - design @ coefficient
    return result.astype(np.float32)


class MoneyflowStrengthContext(AlphaContext):
    """Load and cache one common 20-day window for all four factors."""

    def __init__(self, root=DEFAULT_ROOT, config=MoneyflowStrengthConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)
        self._cache_key = None
        self._cache_value = None

    def history(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        key = (start, end)
        if key != self._cache_key:
            folder = self.config.moneyflow_folder
            read = lambda field: np.asarray(
                self.data.read(f"{folder}/{field}", end, start_date=start),
                dtype=np.float64,
            )
            self._cache_value = {
                "large_buy": read("buy_lg_amount"),
                "large_sell": read("sell_lg_amount"),
                "small_buy": read("buy_sm_amount"),
                "small_sell": read("sell_sm_amount"),
                "daily_pct": np.asarray(
                    self.data.read(
                        self.config.return_field, end, start_date=start
                    ),
                    dtype=np.float64,
                ),
            }
            self._cache_key = key
        return self._cache_value

class MoneyflowStrengthFactor(AlphaBase):
    """Common calculation layer for raw and Ret20-residualized factors."""

    size: str
    residualized = False

    def values(self, asof):
        history = self.context.history(asof)
        if history is None:
            empty = np.full(
                self.context.data.axis.tick_count, np.nan, np.float32
            )
            return empty, empty.copy()

        config = self.context.config
        strength = _moneyflow_strength(
            history[f"{self.size}_buy"],
            history[f"{self.size}_sell"],
            config.min_valid_days,
        )
        ret20 = _compound_return(
            history["daily_pct"], config.min_valid_days
        )
        residual = _cross_section_residual(
            strength, ret20, config.min_cross_section_observations
        )
        return strength, residual

    def calculate(self, asof):
        strength, residual = self.values(asof)
        return residual if self.residualized else strength


class LargeMoneyflowStrengthFactor(MoneyflowStrengthFactor):
    meta = AlphaMeta(
        "large_moneyflow_strength",
        "20D large-order net inflow normalized by absolute daily net inflow",
        direction=1,
    )
    dependencies = (
        "d_moneyflow/buy_lg_amount",
        "d_moneyflow/sell_lg_amount",
    )
    size = "large"


class SmallMoneyflowStrengthFactor(MoneyflowStrengthFactor):
    meta = AlphaMeta(
        "small_moneyflow_strength",
        "20D small-order net inflow normalized by absolute daily net inflow",
        direction=-1,
    )
    dependencies = (
        "d_moneyflow/buy_sm_amount",
        "d_moneyflow/sell_sm_amount",
    )
    size = "small"


class LargeResidualMoneyflowStrengthFactor(MoneyflowStrengthFactor):
    meta = AlphaMeta(
        "large_residual_moneyflow_strength",
        "Large-order money-flow strength residualized against Ret20",
        direction=1,
    )
    dependencies = LargeMoneyflowStrengthFactor.dependencies + (
        "d_essentials/pct",
    )
    size = "large"
    residualized = True


class SmallResidualMoneyflowStrengthFactor(MoneyflowStrengthFactor):
    meta = AlphaMeta(
        "small_residual_moneyflow_strength",
        "Small-order money-flow strength residualized against Ret20",
        direction=-1,
    )
    dependencies = SmallMoneyflowStrengthFactor.dependencies + (
        "d_essentials/pct",
    )
    size = "small"
    residualized = True


MONEYFLOW_STRENGTH_FACTORS = (
    LargeMoneyflowStrengthFactor,
    SmallMoneyflowStrengthFactor,
    LargeResidualMoneyflowStrengthFactor,
    SmallResidualMoneyflowStrengthFactor,
)


__all__ = [
    "MoneyflowStrengthConfig",
    "MoneyflowStrengthContext",
    "MoneyflowStrengthFactor",
    "LargeMoneyflowStrengthFactor",
    "SmallMoneyflowStrengthFactor",
    "LargeResidualMoneyflowStrengthFactor",
    "SmallResidualMoneyflowStrengthFactor",
    "MONEYFLOW_STRENGTH_FACTORS",
]
