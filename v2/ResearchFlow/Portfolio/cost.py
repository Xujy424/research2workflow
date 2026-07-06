"""Transaction-cost and capacity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostEstimate:
    linear: np.ndarray
    impact: np.ndarray
    total: np.ndarray


class TransactionCostModel:
    def __init__(self, *, linear_rate: float = 0.001, impact_coef: float = 0.001) -> None:
        self.linear_rate = linear_rate
        self.impact_coef = impact_coef

    def estimate(self, trade_weight: np.ndarray, *, adv_weight: np.ndarray | None = None) -> CostEstimate:
        trade = np.abs(np.nan_to_num(trade_weight, nan=0.0))
        linear = self.linear_rate * trade.sum(axis=1)
        if adv_weight is None:
            impact = np.zeros(trade.shape[0], dtype=float)
        else:
            participation = np.divide(trade, adv_weight, out=np.zeros_like(trade), where=adv_weight > 1e-12)
            impact = self.impact_coef * np.nansum(trade * np.sqrt(np.clip(participation, 0.0, None)), axis=1)
        return CostEstimate(linear=linear, impact=impact, total=linear + impact)

