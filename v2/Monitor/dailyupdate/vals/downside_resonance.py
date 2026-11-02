"""市场下行共振; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _limit_down_series
from ..utils import _p
from ..utils import _pct
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns, _p


def _compute(returns, hit_floor=None, has_trade=None, period=1, history=252,
                      limit_eligible=None):
    """下行共振：负收益、极端下跌和跌停扩散三个分量等权平均。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = _pct(returns)
    valid = np.isfinite(r)
    if has_trade is not None:
        valid &= np.isfinite(has_trade) & (np.asarray(has_trade) > 0)
    counts = valid.sum(axis=1)
    neg = np.divide(((r < 0) & valid).sum(1), counts,
                    out=np.full(len(r), np.nan), where=counts > 0)
    extreme = np.divide(((r < -.05) & valid).sum(1), counts,
                        out=np.full(len(r), np.nan), where=counts > 0)
    limit = (np.full(len(r), np.nan) if hit_floor is None else
             _limit_down_series(hit_floor, has_trade, limit_eligible))
    # All three components are required by the research definition.
    vals = (neg + extreme + limit) / 3
    return float(np.nanmean(vals[-period:])),_p(vals,history)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + 260)
    returns = context.matrix(RISK_FIELDS['returns'], start, end)
    trade = context.matrix(RISK_FIELDS['has_trade'], start, end)
    rows = []
    floor = context.matrix(RISK_FIELDS['hit_floor'], start, end)
    price_floor = context.matrix(RISK_FIELDS['price_floor'], start, end)
    eligible = np.isfinite(price_floor) & (price_floor > 0)
    for domain in config.domains:
        weights = context.stock_weights(domain, start, end)
        member = np.isfinite(weights) & (weights > 0)
        active = member & np.isfinite(trade) & (trade > 0)
        values = np.where(active & np.isfinite(returns), returns, np.nan)
        domain_trade = np.where(member, trade, 0)
        benchmark = lagged_weighted_returns(values/100, weights)
        for period in config.periods:
            value, rank = _compute(values, floor, domain_trade, period, config.history, eligible)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
