"""作者：xujiayi | 创建日期：2026-11-11
日频自融资影子成交：次日价格、交易禁限、整手、参与率和成本统一核算。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    """组合结果；权重与成交均为实际交易日，成本为当日盘前NAV比例。"""
    daily: pd.DataFrame
    target: pd.DataFrame
    actual: pd.DataFrame
    shares: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: pd.DataFrame


def _project(target, previous, benchmark, industry, turnover, deviation, cap, ind_dev):
    """求最近可行目标；Params: 各向量/约束；Return: 权重、优化是否成功。

    不强制满仓，允许因上限/换手约束保留现金；绝不通过事后重归一化突破上限。
    """
    lo, hi = np.full(len(target), -cap), np.full(len(target), cap)
    if deviation is not None:
        lo, hi = np.maximum(lo, benchmark - deviation), np.minimum(hi, benchmark + deviation)
    # 单边策略不得因约束生成相反方向仓位。
    if (target >= 0).all():
        lo = np.maximum(lo, 0)
    if (target <= 0).all():
        hi = np.minimum(hi, 0)
    clipped = np.clip(target, lo, hi)
    if deviation is None and ind_dev is None:
        change = clipped - previous
        scale = min(1., turnover * 2 / max(np.abs(change).sum(), 1e-15))
        return previous + scale * change, True
    constraints = [{"type": "ineq", "fun": lambda w: turnover * 2 - np.abs(w - previous).sum()}]
    if ind_dev is not None:
        for code in pd.unique(industry[pd.notna(industry)]):
            mask = industry == code
            constraints.append({"type": "ineq", "fun": lambda w, mask=mask:
                                ind_dev - abs((w[mask] - benchmark[mask]).sum())})
    result = minimize(lambda w: .5 * np.square(w - target).sum(), clipped,
                      jac=lambda w: w - target, method="SLSQP", bounds=list(zip(lo, hi)),
                      constraints=constraints, options={"maxiter": 150, "ftol": 1e-9})
    if not result.success:
        raise ValueError(f"组合约束不可行或求解未收敛: {result.message}；请放宽偏离/换手约束")
    return result.x, True


def simulate(base, turnover_limit=1., stock_deviation=None, stock_cap=1.,
             industry_deviation=None, commission=.0003, stamp_tax=.0005,
             execution_price="vwap", capital=1e8, max_participation=.1,
             impact_coefficient=.001, lot_size=100, borrow_rate=0., rebalance_days=1):
    """以t-1目标在t成交，t收盘估值；停牌/ST退出优先于自主换手约束。

    Params: base: 已生成目标的FactorBase；其余参数同FactorBase.cal_portfolio。
    Return: PortfolioResult，包含逐股目标/实际/股数/成交、现金、各项成本和约束偏离。

    VWAP/整日成交额为事后成交近似；默认涨跌停双向禁止交易，ST仅允许减仓。
    卖空为可借券的研究假设；borrow_rate为借券成本，未建模真实券源/保证金。
    """
    if execution_price not in ("open", "vwap"):
        raise ValueError("execution_price须为open/vwap")
    if not isinstance(rebalance_days, (int, np.integer)) or rebalance_days < 1:
        raise ValueError("rebalance_days须为正整数交易日")
    if capital <= 0 or not 0 < max_participation <= 1 or lot_size < 1 or int(lot_size) != lot_size:
        raise ValueError("资金/参与率/整手参数无效")
    if min(turnover_limit, stock_cap, commission, stamp_tax, impact_coefficient, borrow_rate) < 0:
        raise ValueError("上限及费用不可为负")
    if any(x is not None and x < 0 for x in (stock_deviation, industry_deviation)):
        raise ValueError("偏离约束不可为负")
    base.require(execution_price, "close", "amount", "suspended", "st", "limit_up", "limit_down", "hastrade", "adjustment")
    if industry_deviation is not None:
        base.require("industry")
    frame = base.target_holding
    index, columns = frame.index, frame.columns
    target = frame.shift(1).fillna(0).to_numpy(float)
    rebalance = (np.arange(len(index)) > 0) & ((np.arange(len(index)) - 1) % rebalance_days == 0)
    for t in range(1, len(index)):
        if not rebalance[t]:
            target[t] = target[t - 1]
    arrays = {k: v.to_numpy() for k, v in base.fields.items()}
    prices, closes = arrays[execution_price], arrays["close"]
    benchmarks = (base.benchmark_weight.shift(1).fillna(0).to_numpy() if
                  stock_deviation is not None or industry_deviation is not None else np.zeros_like(target))
    actual, holdings, trades = [np.zeros_like(target) for _ in range(3)]
    projected = np.zeros_like(target)
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
        industry = arrays["industry"][t] if "industry" in arrays else np.full(len(columns), np.nan)
        if rebalance[t]:
            desired, _ = _project(target[t], previous, benchmarks[t], industry,
                                  turnover_limit, stock_deviation, stock_cap, industry_deviation)
        else:
            desired = previous.copy()
        is_st = arrays["st"][t] == 1
        # ST强制退出：若当日无法交易，保留持仓并逐日继续请求退出。
        desired[is_st] = 0
        projected[t] = desired
        desired_shares = np.divide(desired * equity, execution_mark,
                                   out=np.zeros_like(shares), where=execution_mark > 0)
        desired_shares = np.trunc(desired_shares / lot_size) * lot_size
        if not rebalance[t]:
            desired_shares = np.where(is_st, 0., shares)
        requested = desired_shares - shares
        tradable = (valid & (arrays["hastrade"][t] == 1) & (arrays["suspended"][t] == 0)
                    & (arrays["limit_up"][t] == 0) & (arrays["limit_down"][t] == 0)
                    & np.isfinite(arrays["st"][t]))
        amount = np.nan_to_num(arrays["amount"][t].astype(float), nan=0., posinf=0., neginf=0.).clip(0)
        max_shares = np.divide(amount * max_participation, p, out=np.zeros_like(p), where=valid)
        delta = np.sign(requested) * np.minimum(np.abs(requested), max_shares)
        delta = np.trunc(delta / lot_size) * lot_size
        delta[~tradable] = 0
        # 零碎股仅在完整退出原有仓位且参与率允许时卖出/回补。
        liquidation = (desired == 0) & tradable & (np.abs(shares) <= max_shares)
        delta[liquidation] = -shares[liquidation]

        def costs(change):
            value = change * execution_mark
            gross = np.abs(value)
            participation = np.divide(gross, amount, out=np.zeros_like(gross), where=amount > 0)
            return (gross.sum() * commission, -value.clip(max=0).sum() * stamp_tax,
                    np.sum(gross * impact_coefficient * np.sqrt(participation)))

        # 先卖后买，现金足以覆盖买入和全部交易成本；缩量仍遵守整手。
        sell = np.minimum(delta, 0)
        sell_costs = costs(sell)
        cash_after_sell = cash - sell @ execution_mark - sum(sell_costs)
        buy = np.maximum(delta, 0)
        full_buy_cost = buy @ execution_mark + sum(costs(buy))
        if full_buy_cost > cash_after_sell:
            low, high = 0., 1.
            for _ in range(40):
                scale = (low + high) / 2
                candidate = np.floor(buy * scale / lot_size) * lot_size
                if candidate @ execution_mark + sum(costs(candidate)) <= cash_after_sell:
                    low = scale
                else:
                    high = scale
            buy = np.floor(buy * low / lot_size) * lot_size
        delta = sell + buy
        comm, tax, impact = costs(delta)
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
        ind_error = np.nan
        if industry_deviation is not None:
            ind_error = max((abs((actual[t] - benchmarks[t])[industry == c].sum())
                             for c in pd.unique(industry[pd.notna(industry)])), default=np.nan)
        diagnostics.append({"target_gap": np.abs(actual[t] - target[t]).sum(),
                            "stock_cap_excess": max(0., np.max(np.abs(actual[t])) - stock_cap),
                            "stock_deviation": np.max(np.abs(actual[t] - benchmarks[t])),
                            "industry_deviation": ind_error,
                            "gross_exposure": np.abs(actual[t]).sum(), "net_exposure": actual[t].sum(),
                            "st_remaining_value": np.abs((shares * last_mark)[is_st]).sum(),
                            "stale_mark_count": int(((shares != 0) & ~close_valid).sum())})
        previous_equity = end_equity
    result = PortfolioResult(pd.DataFrame(rows, index), pd.DataFrame(target, index, columns),
                             pd.DataFrame(actual, index, columns), pd.DataFrame(holdings, index, columns),
                             pd.DataFrame(trades, index, columns), pd.DataFrame(diagnostics, index))
    result.diagnostics.attrs["projected_target"] = pd.DataFrame(projected, index, columns)
    return result
