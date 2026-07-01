"""Tradability adjustment, lot rounding, order generation, and fill simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


# 中文说明：定义 `OrderBook`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class OrderBook:
    orders: pd.DataFrame
    residual_cash: float
    diagnostics: Mapping[str, float]


# 中文说明：定义 `PositionPostProcessor`，封装本模块对应的数据、配置与行为。
class PositionPostProcessor:
    # 中文说明：`apply_tradability`：应用当前规则并更新结果。
    def apply_tradability(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        can_buy: pd.Series,
        can_sell: pd.Series,
    ) -> pd.Series:
        assets = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(assets).fillna(0.0)
        current = current_weights.reindex(assets).fillna(0.0)
        buy_locked = (target > current) & ~can_buy.reindex(assets).fillna(False)
        sell_locked = (target < current) & ~can_sell.reindex(assets).fillna(False)
        adjusted = target.copy()
        adjusted.loc[buy_locked | sell_locked] = current.loc[buy_locked | sell_locked]
        free = ~(buy_locked | sell_locked)
        residual = target.sum() - adjusted.sum()
        capacity = adjusted.loc[free].clip(lower=0.0)
        if abs(residual) > 1e-12 and free.any():
            allocator = capacity if capacity.sum() > 0 else pd.Series(1.0, index=capacity.index)
            adjusted.loc[free] += residual * allocator / allocator.sum()
        return adjusted.rename("tradability_adjusted_weight")

    # 中文说明：`round_lots`：执行该名称对应的业务计算，并返回调用方所需结果。
    def round_lots(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        portfolio_value: float,
        lot_sizes: int | pd.Series = 100,
    ) -> tuple[pd.Series, float]:
        prices = prices.reindex(target_weights.index)
        if prices.isna().any() or (prices <= 0).any():
            raise ValueError("prices must be positive for every target asset")
        lots = (
            lot_sizes.reindex(target_weights.index).fillna(100).astype(int)
            if isinstance(lot_sizes, pd.Series)
            else pd.Series(int(lot_sizes), index=target_weights.index)
        )
        ideal_shares = target_weights * portfolio_value / prices
        rounded = np.sign(ideal_shares) * (
            np.floor(np.abs(ideal_shares) / lots) * lots
        )
        used_cash = float((rounded * prices).sum())
        residual_cash = portfolio_value - used_cash
        return rounded.astype(int).rename("target_shares"), residual_cash


# 中文说明：定义 `OrderGenerator`，封装本模块对应的数据、配置与行为。
class OrderGenerator:
    # 中文说明：`generate`：生成下游所需对象。
    def generate(
        self,
        target_shares: pd.Series,
        current_shares: pd.Series,
        prices: pd.Series,
        adv_shares: pd.Series | None = None,
    ) -> OrderBook:
        assets = target_shares.index.union(current_shares.index)
        target = target_shares.reindex(assets).fillna(0).astype(int)
        current = current_shares.reindex(assets).fillna(0).astype(int)
        quantity = target - current
        price = prices.reindex(assets)
        orders = pd.DataFrame(
            {
                "side": np.where(quantity > 0, "BUY", np.where(quantity < 0, "SELL", "NONE")),
                "quantity": quantity.abs(),
                "signed_quantity": quantity,
                "reference_price": price,
                "notional": quantity.abs() * price,
            }
        )
        if adv_shares is not None:
            orders["adv_participation"] = orders["quantity"] / adv_shares.reindex(
                assets
            ).clip(lower=1.0)
        orders = orders.loc[orders["quantity"] > 0].sort_values("notional", ascending=False)
        return OrderBook(
            orders=orders,
            residual_cash=0.0,
            diagnostics={
                "order_count": float(len(orders)),
                "gross_notional": float(orders["notional"].sum()),
            },
        )


# 中文说明：定义 `ParticipationScheduler`，封装本模块对应的数据、配置与行为。
class ParticipationScheduler:
    # 中文说明：`schedule`：生成执行计划。
    def schedule(
        self,
        orders: pd.DataFrame,
        volume_curve: pd.Series,
        max_participation: float = 0.10,
    ) -> pd.DataFrame:
        curve = volume_curve.clip(lower=0.0)
        curve = curve / curve.sum()
        rows: list[dict[str, object]] = []
        for asset, order in orders.iterrows():
            remaining = int(order["quantity"])
            for bucket, fraction in curve.items():
                planned = min(
                    remaining,
                    int(np.ceil(float(order["quantity"]) * float(fraction))),
                )
                rows.append(
                    {
                        "asset": asset,
                        "bucket": bucket,
                        "side": order["side"],
                        "planned_quantity": planned,
                        "participation_cap": max_participation,
                    }
                )
                remaining -= planned
            if remaining > 0:
                rows[-1]["planned_quantity"] = int(rows[-1]["planned_quantity"]) + remaining
        return pd.DataFrame(rows)


# 中文说明：定义 `FillSimulator`，封装本模块对应的数据、配置与行为。
class FillSimulator:
    # 中文说明：`simulate`：模拟执行过程。
    def simulate(
        self,
        schedule: pd.DataFrame,
        market_volume: pd.DataFrame,
        mid_prices: pd.DataFrame,
        spread_bps: float = 5.0,
        impact_coefficient: float = 0.10,
        seed: int = 7,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        fills: list[dict[str, object]] = []
        for row in schedule.itertuples(index=False):
            available = float(market_volume.loc[row.bucket, row.asset]) * float(
                row.participation_cap
            )
            quantity = min(float(row.planned_quantity), available)
            participation = quantity / max(float(market_volume.loc[row.bucket, row.asset]), 1.0)
            direction = 1.0 if row.side == "BUY" else -1.0
            noise = rng.normal(scale=spread_bps / 20_000.0)
            slippage = direction * (
                spread_bps / 20_000.0
                + impact_coefficient * np.sqrt(max(participation, 0.0))
                + noise
            )
            mid = float(mid_prices.loc[row.bucket, row.asset])
            fills.append(
                {
                    "asset": row.asset,
                    "bucket": row.bucket,
                    "side": row.side,
                    "fill_quantity": quantity,
                    "fill_price": mid * (1.0 + slippage),
                    "slippage": slippage,
                }
            )
        return pd.DataFrame(fills)
