"""Convex transaction-cost model suitable for portfolio optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


# 中文说明：定义 `CostEstimate`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class CostEstimate:
    linear: float
    impact: float
    borrow: float = 0.0
    carry: float = 0.0

    # 中文说明：`total`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def total(self) -> float:
        return self.linear + self.impact + self.borrow + self.carry


# 中文说明：定义 `TransactionCostModel`，封装本模块对应的数据、配置与行为。
class TransactionCostModel:
    """Linear spread/fees plus a square-root market-impact approximation.

    In weight space, impact is proportional to ``|trade| ** 1.5``. ADV inputs
    are expressed as a fraction of portfolio NAV to keep the objective scale
    independent of account currency.
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        linear_rate: float | pd.Series = 0.0008,
        impact_coefficient: float = 0.10,
        adv_floor: float = 1e-5,
        sell_tax_rate: float = 0.0,
    ) -> None:
        self.linear_rate = linear_rate
        self.impact_coefficient = impact_coefficient
        self.adv_floor = adv_floor
        self.sell_tax_rate = sell_tax_rate

    # 中文说明：`arrays`：执行该名称对应的业务计算，并返回调用方所需结果。
    def arrays(
        self,
        assets: pd.Index,
        adv_fraction: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.linear_rate, pd.Series):
            linear = self.linear_rate.reindex(assets).fillna(self.linear_rate.median()).to_numpy()
        else:
            linear = np.full(len(assets), float(self.linear_rate))
        if adv_fraction is None:
            adv = np.ones(len(assets))
        else:
            adv = adv_fraction.reindex(assets).fillna(self.adv_floor).clip(lower=self.adv_floor).to_numpy(float)
        impact_scale = self.impact_coefficient / np.sqrt(adv)
        return linear, impact_scale

    # 中文说明：`expression`：执行该名称对应的业务计算，并返回调用方所需结果。
    def expression(
        self,
        trades: cp.Expression,
        assets: pd.Index,
        adv_fraction: pd.Series | None,
    ) -> cp.Expression:
        linear, impact = self.component_expressions(trades, assets, adv_fraction)
        return linear + impact

    # 中文说明：`component_expressions`：执行该名称对应的业务计算，并返回调用方所需结果。
    def component_expressions(
        self,
        trades: cp.Expression,
        assets: pd.Index,
        adv_fraction: pd.Series | None,
    ) -> tuple[cp.Expression, cp.Expression]:
        linear, impact_scale = self.arrays(assets, adv_fraction)
        return (
            linear @ cp.abs(trades) + self.sell_tax_rate * cp.sum(cp.pos(-trades)),
            impact_scale @ cp.power(cp.abs(trades), 1.5),
        )

    # 中文说明：`estimate`：估计模型量或交易成本。
    def estimate(
        self,
        trades: pd.Series,
        adv_fraction: pd.Series | None = None,
    ) -> CostEstimate:
        linear, impact_scale = self.arrays(trades.index, adv_fraction)
        values = np.abs(trades.to_numpy(float))
        return CostEstimate(
            linear=float(
                linear @ values
                + self.sell_tax_rate
                * np.maximum(-trades.to_numpy(float), 0.0).sum()
            ),
            impact=float(impact_scale @ np.power(values, 1.5)),
        )


# 中文说明：定义 `HoldingCostModel`，封装本模块对应的数据、配置与行为。
class HoldingCostModel:
    """Borrow, futures basis, and other position-level carry costs."""

    # 中文说明：`estimate`：估计模型量或交易成本。
    def estimate(
        self,
        weights: pd.Series,
        holding_days: int,
        annual_borrow_rate: pd.Series | None = None,
        annual_carry_rate: pd.Series | None = None,
        day_count: float = 365.0,
    ) -> CostEstimate:
        borrow = 0.0
        carry = 0.0
        if annual_borrow_rate is not None:
            short = weights.clip(upper=0.0).abs()
            borrow = float(
                (
                    short
                    * annual_borrow_rate.reindex(weights.index).fillna(0.0)
                    * holding_days
                    / day_count
                ).sum()
            )
        if annual_carry_rate is not None:
            carry = float(
                (
                    weights.abs()
                    * annual_carry_rate.reindex(weights.index).fillna(0.0)
                    * holding_days
                    / day_count
                ).sum()
            )
        return CostEstimate(0.0, 0.0, borrow=borrow, carry=carry)
