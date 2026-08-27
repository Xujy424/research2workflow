"""Execution-capacity overlay, deliberately separate from alpha discovery."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import CapacityConfig
from .metrics import performance


@dataclass(frozen=True)
class CapacityResult:
    summary: pd.DataFrame
    equity: dict[float, pd.Series]
    fill_ratio: dict[float, pd.Series]


class CapacitySimulator:
    """Simulate target weights with lots, participation, costs and impact."""

    def __init__(self, config=CapacityConfig()):
        self.config = config

    def run(self, target_weight: pd.DataFrame, execution_price: pd.DataFrame,
            traded_amount: pd.DataFrame) -> CapacityResult:
        for frame in (execution_price, traded_amount):
            if not target_weight.index.equals(frame.index) or not target_weight.columns.equals(frame.columns):
                raise ValueError("capacity inputs must have identical axes")
        price, amount = execution_price.to_numpy(float), traded_amount.to_numpy(float)
        target = target_weight.to_numpy(float)
        summaries, curves, fills = [], {}, {}
        for initial in self.config.capital:
            cash, shares = float(initial), np.zeros(target.shape[1])
            equity_curve, fill_curve = np.zeros(len(target)), np.ones(len(target))
            for t in range(len(target)):
                valid = np.isfinite(price[t]) & (price[t] > 0)
                mark = np.where(valid, price[t], 0.0)
                equity = cash + np.sum(shares * mark)
                desired = np.zeros_like(shares)
                desired[valid] = np.trunc(
                    equity * target[t, valid] / price[t, valid]
                    / self.config.lot_size) * self.config.lot_size
                requested = desired - shares
                requested_value = np.abs(requested * mark)
                capacity = np.nan_to_num(amount[t], nan=0.0).clip(min=0) * self.config.max_participation
                ratio = np.divide(capacity, requested_value,
                                  out=np.ones_like(capacity), where=requested_value > 0).clip(0, 1)
                filled = requested * ratio
                traded_value = filled * mark
                participation = np.divide(np.abs(traded_value), np.nan_to_num(amount[t]),
                                          out=np.zeros_like(mark), where=np.nan_to_num(amount[t]) > 0)
                commission = np.abs(traded_value).sum() * self.config.commission_bps / 1e4
                impact = np.sum(np.abs(traded_value) * self.config.impact_coefficient
                                * np.sqrt(participation.clip(min=0)))
                shares += filled
                cash -= traded_value.sum() + commission + impact
                equity_curve[t] = cash + np.sum(shares * mark)
                active = requested_value > 0
                fill_curve[t] = np.mean(ratio[active]) if active.any() else 1.0
            equity_series = pd.Series(equity_curve, target_weight.index, name="equity")
            returns = equity_series.pct_change().fillna(0.0)
            stat = performance(returns)
            stat["capital"] = initial
            stat["average_fill_ratio"] = fill_curve.mean()
            summaries.append(stat)
            curves[initial] = equity_series
            fills[initial] = pd.Series(fill_curve, target_weight.index, name="fill_ratio")
        return CapacityResult(pd.DataFrame(summaries).set_index("capital"), curves, fills)
