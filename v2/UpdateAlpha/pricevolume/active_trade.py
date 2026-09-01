"""Active buy/sell factors from Kaiyuan Securities' ACT report.

The report forms two factors from the previous 20 trading days:

* ``act_positive``: mean daily ACT of large and medium orders on the
  highest-return 10% of days;
* ``act_negative``: mean daily ACT of small orders on the lowest-return
  10% of days.

Daily ACT is ``(active buy amount - active sell amount) / total amount``.
The negative factor is intentionally stored in its raw report direction;
its :class:`AlphaMeta` direction is -1 because lower values predict higher
future returns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


@dataclass(frozen=True)
class ACTConfig:
    lookback_days: int = 20
    cut_ratio: float = 0.10
    min_valid_days: int = 10
    return_field: str = "d_essentials/pct"
    moneyflow_folder: str = "d_moneyflow"

    def __post_init__(self):
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if not 0 < self.cut_ratio <= 1:
            raise ValueError("cut_ratio must be in (0, 1]")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be within lookback_days")


def _sum_available(*arrays: np.ndarray) -> np.ndarray:
    """Sum legs, treating a missing leg as zero but all-missing as NaN."""
    values = np.stack(arrays, axis=0).astype(np.float64, copy=False)
    available = np.isfinite(values).any(axis=0)
    total = np.nansum(values, axis=0)
    total[~available] = np.nan
    return total


def _active_ratio(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Compute daily ACT while preserving days with no money-flow data."""
    available = np.isfinite(buy) | np.isfinite(sell)
    buy = np.where(np.isfinite(buy), buy, 0.0)
    sell = np.where(np.isfinite(sell), sell, 0.0)
    denominator = buy + sell
    return np.divide(
        buy - sell,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=available & (denominator > 0),
    )


def _cut_mean(
    act: np.ndarray,
    returns: np.ndarray,
    *,
    ratio: float,
    min_valid_days: int,
    highest: bool,
) -> np.ndarray:
    """Mean ACT on each stock's highest/lowest-return fraction of days."""
    valid = np.isfinite(act) & np.isfinite(returns)
    valid_count = valid.sum(axis=0)
    selected_count = np.maximum(1, np.ceil(valid_count * ratio).astype(np.int64))

    key = np.where(valid, -returns if highest else returns, np.inf)
    order = np.argsort(key, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(np.arange(len(act))[:, None], order.shape),
        axis=0,
    )
    selected = valid & (rank < selected_count[None, :])
    numerator = np.sum(np.where(selected, act, 0.0), axis=0)
    denominator = selected.sum(axis=0)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(act.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    result[valid_count < min_valid_days] = np.nan
    return result.astype(np.float32)


class ACTContext(AlphaContext):
    """Shared, read-only retrieval of returns and order-size money flows."""

    def __init__(self, root=ROOT, config=ACTConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))
        self._history_key = None
        self._history_value = None

    def _field(self, name: str) -> str:
        return f"{self.config.moneyflow_folder}/{name}"

    def history(self, asof):
        """Return one aligned lookback window, cached across both factors."""
        row = self.data.axis.date_position(asof)
        start = max(0, row - self.config.lookback_days + 1)
        key = (start, row)
        if key == self._history_key:
            return self._history_value

        read = lambda name: np.asarray(
            self.data.read(self._field(name), row, start_date=start),
            dtype=np.float64,
        )
        history = {
            "returns": np.asarray(
                self.data.read(
                    self.config.return_field, row, start_date=start,
                ),
                dtype=np.float64,
            ),
            "lg_buy": read("lg_buy_amount"),
            "lg_sell": read("lg_sell_amount"),
            "mid_buy": read("mid_buy_amount"),
            "mid_sell": read("mid_sell_amount"),
            "sm_buy": read("sm_buy_amount"),
            "sm_sell": read("sm_sell_amount"),
        }
        self._history_key = key
        self._history_value = history
        return history


class ACTPositiveFactor(AlphaBase):
    meta = AlphaMeta(
        "act_positive",
        "large-plus-medium active buy/sell ACT on high-return days",
        direction=1,
    )
    dependencies = (
        "d_essentials/pct",
        "d_moneyflow/lg_buy_amount",
        "d_moneyflow/lg_sell_amount",
        "d_moneyflow/mid_buy_amount",
        "d_moneyflow/mid_sell_amount",
    )

    def calculate(self, asof):
        h = self.context.history(asof)
        buy = _sum_available(h["lg_buy"], h["mid_buy"])
        sell = _sum_available(h["lg_sell"], h["mid_sell"])
        act = _active_ratio(buy, sell)
        cfg = self.context.config
        return _cut_mean(
            act,
            h["returns"],
            ratio=cfg.cut_ratio,
            min_valid_days=cfg.min_valid_days,
            highest=True,
        )


class ACTNegativeFactor(AlphaBase):
    meta = AlphaMeta(
        "act_negative",
        "small-order active buy/sell ACT on low-return days",
        direction=-1,
    )
    dependencies = (
        "d_essentials/pct",
        "d_moneyflow/sm_buy_amount",
        "d_moneyflow/sm_sell_amount",
    )

    def calculate(self, asof):
        h = self.context.history(asof)
        act = _active_ratio(h["sm_buy"], h["sm_sell"])
        cfg = self.context.config
        return _cut_mean(
            act,
            h["returns"],
            ratio=cfg.cut_ratio,
            min_valid_days=cfg.min_valid_days,
            highest=False,
        )



