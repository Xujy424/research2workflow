"""MOD-adjusted broad large-money-flow factor from Kaiyuan series (16)."""

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
class ModLargeFlowConfig:
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


class ModLargeFlowContext(AlphaContext):
    """Cache the common money-flow window used by MOD/CNIR factors."""

    def __init__(self, root=DEFAULT_ROOT, config=ModLargeFlowConfig(), universe="self"):
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

            def read(field):
                return np.asarray(
                    self.data.read(f"{folder}/{field}", end, start_date=start),
                    dtype=np.float64,
                )

            self._cache_value = {
                "daily_pct": np.asarray(
                    self.data.read(self.config.return_field, end, start_date=start),
                    dtype=np.float64,
                ),
                "buy_elg": read("buy_elg_amount"),
                "sell_elg": read("sell_elg_amount"),
                "buy_lg": read("buy_lg_amount"),
                "sell_lg": read("sell_lg_amount"),
                "buy_mid": read("buy_md_amount"),
                "sell_mid": read("sell_md_amount"),
            }
            self._cache_key = key
        return self._cache_value


def _sum_available(*arrays: np.ndarray) -> np.ndarray:
    values = np.stack(arrays, axis=0).astype(np.float64, copy=False)
    available = np.isfinite(values).any(axis=0)
    total = np.nansum(values, axis=0)
    total[~available] = np.nan
    return total


def _cross_section_residual(y: np.ndarray, x: np.ndarray, min_observations: int) -> np.ndarray:
    valid = np.isfinite(y) & np.isfinite(x)
    residual = np.full_like(y, np.nan, dtype=np.float64)
    if valid.sum() < min_observations:
        return residual
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    coefficient, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    residual[valid] = y[valid] - design @ coefficient
    return residual


def _mod_adjusted_flow(
    buy: np.ndarray,
    sell: np.ndarray,
    daily_pct: np.ndarray,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return daily MOD NIR and total principal-money amount."""
    buy = np.asarray(buy, dtype=np.float64)
    sell = np.asarray(sell, dtype=np.float64)
    daily_return = np.asarray(daily_pct, dtype=np.float64) / 100.0

    total = buy + sell
    daily_mod_nir = np.full_like(total, np.nan, dtype=np.float64)
    principal_amount = np.where(
        np.isfinite(total) & (total > 0),
        total,
        np.nan,
    )
    for row in range(total.shape[0]):
        valid = (
            np.isfinite(buy[row])
            & np.isfinite(sell[row])
            & (buy[row] > 0)
            & (sell[row] > 0)
            & np.isfinite(daily_return[row])
        )
        imbalance = np.full(total.shape[1], np.nan, dtype=np.float64)
        imbalance[valid] = np.log(buy[row, valid] / sell[row, valid])
        epsilon = _cross_section_residual(
            imbalance,
            daily_return[row],
            min_observations,
        )
        exp_epsilon = np.exp(np.clip(epsilon, -50.0, 50.0))
        daily_mod_nir[row] = np.divide(
            exp_epsilon - 1.0,
            exp_epsilon + 1.0,
            out=np.full(total.shape[1], np.nan, dtype=np.float64),
            where=np.isfinite(exp_epsilon),
        )
    return daily_mod_nir, principal_amount


def _net_inflow_ratio(
    daily_mod_nir: np.ndarray,
    principal_amount: np.ndarray,
    min_valid_days: int,
) -> np.ndarray:
    valid = (
        np.isfinite(daily_mod_nir)
        & np.isfinite(principal_amount)
        & (principal_amount > 0)
    )
    numerator = np.where(valid, daily_mod_nir * principal_amount, 0.0).sum(axis=0)
    denominator = np.where(valid, principal_amount, 0.0).sum(axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(daily_mod_nir.shape[1], np.nan, dtype=np.float64),
        where=(valid.sum(axis=0) >= min_valid_days) & (denominator > 0),
    ).astype(np.float32)


class CNIRFactor(AlphaBase):
    """20D CNIR: MOD-adjusted NIR of extra-large, large and medium flows."""

    meta = AlphaMeta(
        "cnir",
        "20D MOD-adjusted broad principal-money net inflow ratio",
        direction=1,
    )
    dependencies = (
        "d_essentials/pct",
        "d_moneyflow/buy_elg_amount",
        "d_moneyflow/sell_elg_amount",
        "d_moneyflow/buy_lg_amount",
        "d_moneyflow/sell_lg_amount",
        "d_moneyflow/buy_mid_amount",
        "d_moneyflow/sell_mid_amount",
    )

    def calculate(self, asof):
        history = self.context.history(asof)
        if history is None:
            return np.full(
                self.context.data.axis.tick_count,
                np.nan,
                dtype=np.float32,
            )
        cfg = self.context.config
        buy = _sum_available(
            history["buy_elg"],
            history["buy_lg"],
            history["buy_mid"],
        )
        sell = _sum_available(
            history["sell_elg"],
            history["sell_lg"],
            history["sell_mid"],
        )
        daily_mod_nir, principal_amount = _mod_adjusted_flow(
            buy,
            sell,
            history["daily_pct"],
            cfg.min_cross_section_observations,
        )
        return _net_inflow_ratio(
            daily_mod_nir,
            principal_amount,
            cfg.min_valid_days,
        )


__all__ = ["ModLargeFlowConfig", "ModLargeFlowContext", "CNIRFactor"]
