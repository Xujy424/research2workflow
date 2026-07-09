"""Transaction, impact, and holding-cost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostEstimate:
    linear: np.ndarray
    impact: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.linear + self.impact


class TransactionCostModel:
    """Linear spread/fees plus square-root market impact in weight space."""

    def __init__(
        self,
        *,
        linear_rate: float = 0.001,
        impact_coef: float = 0.001,
        adv_floor: float = 1e-5,
        sell_tax_rate: float = 0.0,
    ) -> None:
        self.linear_rate = float(linear_rate)
        self.impact_coef = float(impact_coef)
        self.adv_floor = float(adv_floor)
        self.sell_tax_rate = float(sell_tax_rate)

    def calc_cost(self, n_assets: int, *, adv_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        linear = np.full(n_assets, self.linear_rate, dtype=float)
        if adv_weight is None:
            return linear, np.zeros(n_assets, dtype=float)
        adv = np.clip(np.nan_to_num(np.asarray(adv_weight, dtype=float), nan=self.adv_floor), self.adv_floor, None)
        return linear, self.impact_coef / np.sqrt(adv)

    def cost_expressions(self, cp: Any, trades, n_assets: int, *, adv_weight: np.ndarray | None = None):
        linear, impact = self.calc_cost(n_assets, adv_weight=adv_weight)
        linear_cost = linear @ cp.abs(trades)
        if self.sell_tax_rate > 0:
            linear_cost += self.sell_tax_rate * cp.sum(cp.pos(-trades))
        impact_cost = impact @ cp.power(cp.abs(trades), 1.5)
        return linear_cost, impact_cost

    def estimate(self, trade_weight: np.ndarray, *, adv_weight: np.ndarray | None = None) -> CostEstimate:
        trade = np.nan_to_num(np.asarray(trade_weight, dtype=float), nan=0.0)
        if trade.ndim != 1:
            raise ValueError("trade_weight must be a 1D daily cross-section")
        linear, impact = self.calc_cost(trade.size, adv_weight=adv_weight)
        abs_trade = np.abs(trade)
        linear_cost = linear @ abs_trade + self.sell_tax_rate * np.maximum(-trade, 0.0).sum()
        impact_cost = impact @ np.power(abs_trade, 1.5)
        return CostEstimate(np.asarray(linear_cost), np.asarray(impact_cost))


@dataclass(frozen=True)
class HoldingCostEstimate:
    borrow: float
    carry: float

    @property
    def total(self) -> float:
        return self.borrow + self.carry


class HoldingCostModel:
    """Borrow, futures basis, and other position-level carry costs."""

    def estimate(
        self,
        weights: np.ndarray,
        holding_days: int,
        *,
        annual_borrow_rate: np.ndarray | None = None,
        annual_carry_rate: np.ndarray | None = None,
        day_count: float = 365.0,
    ) -> HoldingCostEstimate:
        w = np.asarray(weights, dtype=float)
        borrow = 0.0
        carry = 0.0
        if annual_borrow_rate is not None:
            short = np.maximum(-w, 0.0)
            borrow = float(np.nansum(short * np.nan_to_num(annual_borrow_rate, nan=0.0) * holding_days / day_count))
        if annual_carry_rate is not None:
            carry = float(np.nansum(np.abs(w) * np.nan_to_num(annual_carry_rate, nan=0.0) * holding_days / day_count))
        return HoldingCostEstimate(borrow=borrow, carry=carry)


