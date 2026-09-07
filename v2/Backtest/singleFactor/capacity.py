"""Execution-capacity overlay, deliberately separate from alpha discovery."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import CapacityConfig
from .metrics import performance


@dataclass(frozen=True)
class CapacityResult:
    summary: pd.DataFrame
    equity: dict[float, pd.Series]
    fill_ratio: dict[float, pd.Series]
    returns: dict[float, pd.Series] = field(default_factory=dict)
    turnover: dict[float, pd.Series] = field(default_factory=dict)
    commission_cost_ratio: dict[float, pd.Series] = field(default_factory=dict)
    impact_cost_ratio: dict[float, pd.Series] = field(default_factory=dict)


class CapacitySimulator:
    """Simulate target weights with lots, participation, costs and impact."""

    def __init__(self, config=CapacityConfig()):
        self.config = config

    def _costs(self, shares, price, amount):
        traded_value = np.abs(shares * price)
        gross_value = traded_value.sum()
        participation = np.divide(
            traded_value,
            amount,
            out=np.zeros_like(traded_value),
            where=amount > 0,
        )
        commission = gross_value * self.config.commission_bps / 1e4
        impact = np.sum(
            traded_value
            * (self.config.impact_coefficient / 100.0)
            * np.sqrt(np.clip(participation, 0.0, None))
        )
        return float(gross_value), float(commission), float(impact)

    def _cash_limited_buys(self, candidate, price, amount, cash):
        """Scale buys so gross consideration plus all costs fits in cash."""
        if cash <= 0 or not np.any(candidate > 0):
            return np.zeros_like(candidate)

        def required(scale):
            gross, commission, impact = self._costs(
                candidate * scale, price, amount
            )
            return gross + commission + impact

        if required(1.0) <= cash:
            return candidate

        low, high = 0.0, 1.0
        for _ in range(50):
            middle = 0.5 * (low + high)
            if required(middle) <= cash:
                low = middle
            else:
                high = middle
        return candidate * low

    def run(
        self, 
        target_weight: pd.DataFrame, 
        execution_price: pd.DataFrame,
        traded_amount: pd.DataFrame
    ) -> CapacityResult:
        for frame in (execution_price, traded_amount):
            if not target_weight.index.equals(frame.index) or not target_weight.columns.equals(frame.columns):
                raise ValueError("capacity inputs must have identical axes")
            
        price, amount = execution_price.to_numpy(float), traded_amount.to_numpy(float)
        target = target_weight.to_numpy(float)

        summaries, curves, fills = [], {}, {}
        return_curves, turnovers = {}, {}
        commission_curves, impact_curves = {}, {}
        for initial in self.config.capital:
            cash, shares = float(initial), np.zeros(target.shape[1])
            equity_curve, fill_curve = np.zeros(len(target)), np.ones(len(target))
            turnover_curve = np.zeros(len(target))
            commission_curve = np.zeros(len(target))
            impact_curve = np.zeros(len(target))
            last_price = np.zeros(target.shape[1], dtype=float)
            for t in range(len(target)):
                valid_price = np.isfinite(price[t]) & (price[t] > 0)
                last_price[valid_price] = price[t, valid_price]
                mark = last_price
                equity = cash + np.sum(shares * mark)

                # An unavailable price means no trade, not liquidation at zero.
                desired = shares.copy()
                valid_target = valid_price & np.isfinite(target[t])
                desired[valid_target] = np.trunc(
                    equity * target[t, valid_target] / price[t, valid_target]
                    / self.config.lot_size) * self.config.lot_size
                requested = desired - shares
                amount_t = np.nan_to_num(
                    amount[t], nan=0.0, posinf=0.0, neginf=0.0
                ).clip(min=0.0)
                capacity_value = (
                    amount_t * self.config.max_participation
                )

                # Sell first so proceeds are available for buys. This also
                # handles opening/increasing short positions.
                requested_sell = np.maximum(-requested, 0.0)
                sell_value = requested_sell * np.where(
                    valid_price, price[t], 0.0
                )
                sell_ratio = np.divide(
                    capacity_value,
                    sell_value,
                    out=np.ones_like(sell_value),
                    where=sell_value > 0,
                ).clip(0.0, 1.0)
                sold = requested_sell * sell_ratio
                sell_gross, sell_commission, sell_impact = self._costs(
                    sold, np.where(valid_price, price[t], 0.0), amount_t
                )
                shares -= sold
                cash += sell_gross - sell_commission - sell_impact

                requested_buy = np.maximum(requested, 0.0)
                buy_value = requested_buy * np.where(
                    valid_price, price[t], 0.0
                )
                buy_ratio = np.divide(
                    capacity_value,
                    buy_value,
                    out=np.ones_like(buy_value),
                    where=buy_value > 0,
                ).clip(0.0, 1.0)
                candidate_buy = requested_buy * buy_ratio
                bought = self._cash_limited_buys(
                    candidate_buy,
                    np.where(valid_price, price[t], 0.0),
                    amount_t,
                    cash,
                )
                buy_gross, buy_commission, buy_impact = self._costs(
                    bought, np.where(valid_price, price[t], 0.0), amount_t
                )
                shares += bought
                cash -= buy_gross + buy_commission + buy_impact

                total_commission = sell_commission + buy_commission
                total_impact = sell_impact + buy_impact
                equity_curve[t] = cash + np.sum(shares * mark)
                base_equity = max(abs(equity_curve[t]), 1e-12)
                turnover_curve[t] = (
                    0.5 * (sell_gross + buy_gross) / base_equity
                )
                commission_curve[t] = total_commission / base_equity
                impact_curve[t] = total_impact / base_equity

                requested_abs = np.abs(requested)
                actual_abs = sold + bought
                active = requested_abs > 0
                if active.any():
                    actual_ratio = np.divide(
                        actual_abs[active],
                        requested_abs[active],
                        out=np.zeros(active.sum(), dtype=float),
                        where=requested_abs[active] > 0,
                    )
                    fill_curve[t] = np.mean(actual_ratio)

            equity_series = pd.Series(equity_curve, target_weight.index, name="equity")
            previous_equity = np.r_[initial, equity_curve[:-1]]
            daily_return = np.divide(
                equity_curve,
                previous_equity,
                out=np.ones_like(equity_curve),
                where=previous_equity != 0,
            ) - 1.0
            return_series = pd.Series(
                daily_return, target_weight.index, name="return"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            stat = performance(return_series)
            stat["capital"] = initial
            stat["average_fill_ratio"] = fill_curve.mean()
            stat["average_turnover"] = turnover_curve.mean()
            stat["average_commission_cost"] = commission_curve.mean()
            stat["average_impact_cost"] = impact_curve.mean()
            stat["online_decision"] = (
                "Pass"
                if (
                    stat["average_fill_ratio"] > self.config.min_fill_ratio
                    and stat.get("sharpe", np.nan) > self.config.min_sharpe
                )
                else "Fail"
            )
            summaries.append(stat)
            curves[initial] = equity_series
            fills[initial] = pd.Series(fill_curve, target_weight.index, name="fill_ratio")
            return_curves[initial] = return_series
            turnovers[initial] = pd.Series(
                turnover_curve, target_weight.index, name="turnover"
            )
            commission_curves[initial] = pd.Series(
                commission_curve,
                target_weight.index,
                name="commission_cost_ratio",
            )
            impact_curves[initial] = pd.Series(
                impact_curve,
                target_weight.index,
                name="impact_cost_ratio",
            )
        return CapacityResult(
            summary=pd.DataFrame(summaries).set_index("capital"),
            equity=curves,
            fill_ratio=fills,
            returns=return_curves,
            turnover=turnovers,
            commission_cost_ratio=commission_curves,
            impact_cost_ratio=impact_curves,
        )
