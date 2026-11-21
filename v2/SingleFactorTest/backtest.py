"""T日目标权重 -> T+1日实际成交、现金、持仓及净值；不做二次权重投影。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    """成交日维度的组合结果。"""
    daily: pd.DataFrame
    target: pd.DataFrame
    actual: pd.DataFrame
    shares: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: pd.DataFrame


def calc_cost(change, mark, amount, commission, stamp_tax, impact_coefficient):
    """计算成交佣金、卖出印花税和成交参与率平方根冲击成本（货币金额）。"""
    value = change * mark
    gross = np.abs(value)
    participation = np.divide(gross, amount, out=np.zeros_like(gross), where=amount > 0)
    return (gross.sum() * commission,
            -np.minimum(value, 0.).sum() * stamp_tax,
            np.sum(gross * impact_coefficient * np.sqrt(participation)))


def calc_trade(desired, shares, equity, mark, valid, prices, amount,
               hastrade, suspended, limit_up, limit_down, st,
               max_participation, lot_size, rebalance):
    """按理想权重生成可成交股数，不修改目标权重；旧股不可交易则继续持有。"""
    desired_shares = np.divide(desired * equity, mark, out=np.zeros_like(shares), where=mark > 0)
    desired_shares = np.trunc(desired_shares / lot_size) * lot_size
    if not rebalance:
        desired_shares = np.where(st == 1, 0., shares)
    requested = desired_shares - shares
    tradable = (valid & (hastrade == 1) & (suspended == 0)
                & (limit_up == 0) & (limit_down == 0) & np.isfinite(st))
    max_shares = np.divide(amount * max_participation, prices,
                           out=np.zeros_like(prices), where=valid)
    delta = np.sign(requested) * np.minimum(np.abs(requested), max_shares)
    delta = np.trunc(delta / lot_size) * lot_size
    delta[~tradable] = 0.
    # 原持仓完整退出且参与率允许时，可清理零碎股。
    liquidation = (desired == 0) & tradable & (np.abs(shares) <= max_shares)
    delta[liquidation] = -shares[liquidation]
    return requested, delta


def simulate(
        base, frame, commission=.0003, stamp_tax=.0005,
        execution_price="vwap", capital=1e8, max_participation=.1,
        impact_coefficient=.001, lot_size=100, borrow_rate=0.,
        rebalance_days=1):
    """生成目标持仓并执行T+1日成交仿真，参数全部显式传入。

    T日收盘生成目标，T+1日open/vwap执行、收盘估值；不进行目标权重投影。
    使用实际成交股数、停牌/ST/涨跌停限制、整手、成交额参与率和成本。
    无法成交时保留旧持仓；返回PortfolioResult。
    注意：VWAP/全日成交额和当日交易状态为事后成交近似，不是无前视逐笔模拟；
    做空假定有券源，未建模保证金及股指期货对冲。
    """
    if execution_price not in ("open", "vwap"):
        raise ValueError("execution_price须为open/vwap")
    if not isinstance(rebalance_days, (int, np.integer)) or rebalance_days < 1:
        raise ValueError("rebalance_days须为正整数交易日")
    if capital <= 0 or not 0 < max_participation <= 1 or lot_size < 1 or int(lot_size) != lot_size:
        raise ValueError("资金/参与率/整手参数无效")
    if min(commission, stamp_tax, impact_coefficient, borrow_rate) < 0:
        raise ValueError("费用参数不可为负")
    required = (execution_price, "close", "amount", "suspended", "st",
                "limit_up", "limit_down", "hastrade", "adjustment")
    missing = [k for k in required if k not in base.fields]
    if missing:
        raise KeyError(f"回测缺少字段: {missing}")

    index, columns = frame.index, frame.columns
    target = frame.shift(1).fillna(0).to_numpy(float)
    rebalance = (np.arange(len(index)) > 0) & ((np.arange(len(index)) - 1) % rebalance_days == 0)
    for t in range(1, len(index)):
        if not rebalance[t]:
            target[t] = target[t - 1]
    arrays = {k: v.to_numpy() for k, v in base.fields.items() if isinstance(v, pd.DataFrame)}
    prices, closes = arrays[execution_price], arrays["close"]
    actual, holdings, trades = [np.zeros_like(target) for _ in range(3)]
    shares, last_mark = np.zeros(len(columns)), np.zeros(len(columns))
    cash, previous_equity = float(capital), float(capital)
    rows, diagnostics = [], []
    for t in range(len(index)):
        factor = arrays["adjustment"][t].astype(float)
        if t:
            old_factor = arrays["adjustment"][t - 1].astype(float)
            held = shares != 0
            if np.any(held & (~np.isfinite(factor) | ~np.isfinite(old_factor) | (factor <= 0) | (old_factor <= 0))):
                raise ValueError(f"{index[t]}持仓复权因子缺失，无法安全估值")
            ratio = np.divide(factor, old_factor, out=np.ones_like(factor), where=(factor > 0) & (old_factor > 0))
            shares *= ratio
            last_mark /= ratio
        p = prices[t].astype(float)
        valid = np.isfinite(p) & (p > 0)
        execution_mark = np.where(valid, p, last_mark)
        equity = cash + shares @ execution_mark
        if equity <= 0:
            raise ValueError(f"{index[t]}组合净资产非正，回测终止")
        previous = shares * execution_mark / equity
        # 不优化目标权重：调仓日直接执行T-1信号目标；非调仓日保持实际股数。
        desired = target[t].copy() if rebalance[t] else previous.copy()
        is_st = arrays["st"][t] == 1
        # ST强制退出；不可交易时留仓并在之后重试。
        desired[is_st] = 0.
        amount = np.nan_to_num(arrays["amount"][t].astype(float), nan=0., posinf=0., neginf=0.).clip(0)
        requested, delta = calc_trade(
            desired, shares, equity, execution_mark, valid, p, amount,
            arrays["hastrade"][t], arrays["suspended"][t],
            arrays["limit_up"][t], arrays["limit_down"][t],
            arrays["st"][t], max_participation, lot_size,
            rebalance=bool(rebalance[t]))

        # 先卖后买，现金足以覆盖买入和全部交易成本；缩量仍遵守整手。
        sell = np.minimum(delta, 0)
        sell_costs = calc_cost(sell, execution_mark, amount, commission, stamp_tax, impact_coefficient)
        cash_after_sell = cash - sell @ execution_mark - sum(sell_costs)
        buy = np.maximum(delta, 0)
        full_buy_cost = buy @ execution_mark + sum(calc_cost(buy, execution_mark, amount, commission, stamp_tax, impact_coefficient))
        if full_buy_cost > cash_after_sell:
            low, high = 0., 1.
            for _ in range(40):
                scale = (low + high) / 2
                candidate = np.floor(buy * scale / lot_size) * lot_size
                if candidate @ execution_mark + sum(calc_cost(candidate, execution_mark, amount, commission, stamp_tax, impact_coefficient)) <= cash_after_sell:
                    low = scale
                else:
                    high = scale
            buy = np.floor(buy * low / lot_size) * lot_size
        delta = sell + buy
        comm, tax, impact = calc_cost(delta, execution_mark, amount, commission, stamp_tax, impact_coefficient)
        cash -= delta @ execution_mark + comm + tax + impact
        shares += delta
        closing = closes[t].astype(float)
        close_valid = np.isfinite(closing) & (closing > 0)
        last_mark = np.where(close_valid, closing, execution_mark)
        borrow = np.abs(np.minimum(shares * last_mark, 0)).sum() * borrow_rate / 252
        cash -= borrow
        end_equity = cash + shares @ last_mark
        actual[t], holdings[t], trades[t] = shares * last_mark / end_equity, shares, delta
        req_value = np.abs(requested * execution_mark).sum()
        traded_value = np.abs(delta * execution_mark).sum()
        fill = traded_value / req_value if req_value > 0 else np.nan
        cost_total = comm + tax + impact + borrow
        rows.append({"nav": end_equity / capital, "equity": end_equity, "cash": cash,
                     "net_return": end_equity / previous_equity - 1,
                     "gross_return": (end_equity + cost_total) / previous_equity - 1,
                     "pnl": end_equity - previous_equity, "fill_ratio": fill,
                     "turnover": traded_value / previous_equity / 2,
                     "commission": comm / previous_equity, "stamp_tax": tax / previous_equity,
                     "impact": impact / previous_equity, "borrow": borrow / previous_equity,
                     "requested_value": req_value, "traded_value": traded_value,
                     "rebalance": bool(rebalance[t])})
        diagnostics.append({
            "target_gap": np.abs(actual[t] - target[t]).sum(),
            "gross_exposure": np.abs(actual[t]).sum(),
            "net_exposure": actual[t].sum(),
            "st_remaining_value": np.abs((shares * last_mark)[is_st]).sum(),
            "stale_mark_count": int(((shares != 0) & ~close_valid).sum())
        })
        previous_equity = end_equity
    result = PortfolioResult(pd.DataFrame(rows, index), pd.DataFrame(target, index, columns),
                             pd.DataFrame(actual, index, columns), pd.DataFrame(holdings, index, columns),
                             pd.DataFrame(trades, index, columns), pd.DataFrame(diagnostics, index))
    return result

