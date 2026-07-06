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
    def __init__(
        self,
        *,
        linear_rate: float = 0.001,
        impact_coef: float = 0.001,
        adv_floor: float = 1e-5,
        sell_tax_rate: float = 0.0,
    ) -> None:
        self.linear_rate = linear_rate
        self.impact_coef = impact_coef
        self.adv_floor = adv_floor
        self.sell_tax_rate = sell_tax_rate

    def component_arrays(self, n_assets: int, *, adv_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        linear = np.full(n_assets, float(self.linear_rate), dtype=float)
        if adv_weight is None:
            adv = np.ones(n_assets, dtype=float)
        else:
            adv = np.clip(np.nan_to_num(np.asarray(adv_weight, dtype=float), nan=self.adv_floor), self.adv_floor, None)
        impact = self.impact_coef / np.sqrt(adv)
        return linear, impact

    def estimate(self, trade_weight: np.ndarray, *, adv_weight: np.ndarray | None = None) -> CostEstimate:
        trade = np.abs(np.nan_to_num(trade_weight, nan=0.0))
        linear = self.linear_rate * trade.sum(axis=1) + self.sell_tax_rate * np.maximum(-np.nan_to_num(trade_weight, nan=0.0), 0.0).sum(axis=1)
        if adv_weight is None:
            impact = np.zeros(trade.shape[0], dtype=float)
        else:
            adv = np.clip(np.nan_to_num(adv_weight, nan=self.adv_floor), self.adv_floor, None)
            participation = np.divide(trade, adv, out=np.zeros_like(trade), where=adv > 1e-12)
            impact = self.impact_coef * np.nansum(trade * np.sqrt(np.clip(participation, 0.0, None)), axis=1)
        return CostEstimate(linear=linear, impact=impact, total=linear + impact)


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

