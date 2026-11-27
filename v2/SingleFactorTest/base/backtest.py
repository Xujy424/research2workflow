"""按目标权重生成订单并模拟日频成交、成本和组合收益。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    """回测结果，所有表均按日期和股票对齐。"""
    daily: pd.DataFrame
    target: pd.DataFrame
    shares: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: pd.DataFrame


def calc_cost(change, price, amount, commission, stamp_tax, impact_coefficient):
    value = change * price
    traded = np.abs(value)
    participation = np.divide(
        traded, amount, out=np.zeros_like(traded), where=amount > 0
    )
    return (
        traded.sum() * commission,
        -np.minimum(value, 0.).sum() * stamp_tax,
        np.sum(traded * impact_coefficient / 100 * np.sqrt(participation)),
    )


def simulate(
        base, frame, commission=.003, stamp_tax=.0005,
        execution_price="vwap", capital=1e8, max_participation=.1,
        impact_coefficient=.001, lot_size=100, borrow_rate=0.,
        rebalance_days=1):
    """T 日信号于 T+1 日执行；按执行价估值，收益使用 base.pct。"""
    dates, stocks = frame.index, frame.columns
    target = frame.fillna(0).to_numpy(float)
    rebalance = (np.arange(len(dates)) > 0) & ((np.arange(len(dates)) - 1) % rebalance_days == 0)
    for t in range(1, len(dates)):
        if not rebalance[t]:
            target[t] = target[t - 1]

    names = (execution_price, "st", "amount", "tradable_t")
    data = {
        name: base.fields[name].reindex(index=dates, columns=stocks).to_numpy()
        for name in names
    }
    prices = data[execution_price].astype(float)
    n_stocks = target.shape[1]
    holdings = np.zeros_like(target)
    trades = np.zeros_like(target)
    equity = float(capital)
    cash = equity
    shares = np.zeros(n_stocks)
    rows, diagnostics = [], []

    for t, date in enumerate(dates):
        trade_price = np.where(
            np.isfinite(prices[t]) & (prices[t] > 0), prices[t], 0.
        )
        amount = np.nan_to_num(data["amount"][t].astype(float), nan=0., posinf=0., neginf=0.).clip(0.)

        desired = np.divide(
            target[t] * equity, trade_price,
            out=np.zeros(n_stocks), where=trade_price > 0,
        )
        desired = np.trunc(desired / lot_size) * lot_size
        desired[data["st"][t] == 1] = 0.
        
        requested = desired - shares
        max_shares = np.divide(
            amount * max_participation, trade_price,
            out=np.zeros(n_stocks), where=trade_price > 0,
        )
        order = np.trunc(
            np.sign(requested) * np.minimum(np.abs(requested), max_shares)
            / lot_size
        ) * lot_size
        tradable = (trade_price > 0) & (data["tradable_t"][t] == 1)
        order[~tradable] = 0.

        sell, buy = np.minimum(order, 0.), np.maximum(order, 0.)
        sell_commission, sell_tax, sell_impact = calc_cost(
            sell, trade_price, amount, commission, stamp_tax, impact_coefficient
        )
        cash += -sell @ trade_price 
        sell_cost = sell_commission + sell_tax + sell_impact

        def buy_requirement(candidate):
            fee = calc_cost(
                candidate, trade_price, amount,
                commission, stamp_tax, impact_coefficient,
            )
            return candidate @ trade_price + sum(fee)

        if buy_requirement(buy) > cash-sell_cost:
            low, high = 0., 1.
            for _ in range(50):
                scale = (low + high) / 2
                candidate = np.floor(buy * scale / lot_size) * lot_size
                if buy_requirement(candidate) <= cash-sell_cost:
                    low = scale
                else:
                    high = scale
            buy = np.floor(buy * low / lot_size) * lot_size

        order = sell + buy
        cash -= buy @ trade_price
        buy_commission, buy_tax, buy_impact = calc_cost(
            buy, trade_price, amount, commission, stamp_tax, impact_coefficient
        )
        buy_cost = buy_commission + buy_tax + buy_impact

        shares += order

        position_value = shares * base.fields['close']*base.fields['adjustment']

        borrow_cost = np.abs(np.minimum(position_value, 0.)).sum() * borrow_rate / 252
        total_cost = buy_cost + sell_cost + borrow_cost 

        current_gross_equity = cash + position_value  # 扣费前资产价值
        gross_pnl = current_gross_equity - equity     # 扣费前pnl
        net_pnl = gross_pnl - total_cost      # 扣费后pnl
        current_equity = current_gross_equity - total_cost    # 扣费后资产价值
        if current_equity <= 0:
            raise ValueError(f"{date} 组合净资产非正，回测终止")

        holdings[t], trades[t] = shares, order
        requested_value = np.abs(requested * trade_price).sum()
        traded_value = np.abs(order * trade_price).sum()
        rows.append({
            "nav": current_equity / capital,
            "equity": position_value,
            "cash": cash,
            "net_return": current_equity / equity,
            "gross_return": gross_pnl / equity,
            "pnl": net_pnl,
            "fill_ratio": np.clip(traded_value / requested_value, 0, 1) if requested_value else np.nan,
            "turnover": traded_value / equity / 2,
            "commission": (sell_commission+buy_commission) / equity,
            "stamp_tax": (sell_tax+buy_tax)/ equity,
            "impact": (sell_impact+buy_impact) / equity,
            "borrow": borrow_cost / equity,
            "requested_value": requested_value,
            "traded_value": traded_value,
            "rebalance": bool(rebalance[t]),
        })
        diagnostics.append({
            "target_gap": np.abs(order - requested).sum(),
            "gross_exposure": np.abs(order).sum(),
            "net_exposure": order.sum(),
        })
        equity = current_equity

    return PortfolioResult(
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(target, index=dates, columns=stocks),
        pd.DataFrame(holdings, index=dates, columns=stocks),
        pd.DataFrame(trades, index=dates, columns=stocks),
        pd.DataFrame(diagnostics, index=dates),
    )
