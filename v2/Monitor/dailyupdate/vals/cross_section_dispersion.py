"""横截面收益离散度; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _p
from ..utils import _pct
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns, _p


def _compute(returns, period=1, history=252):
    """横截面收益离散度：每日股票收益标准差，再按周期取均值并计算历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r=_pct(returns); daily=np.nanstd(r,axis=1,ddof=1)
    return float(np.nanmean(daily[-period:])), _p(daily,history)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + 260)
    returns = context.matrix(RISK_FIELDS['returns'], start, end)
    trade = context.matrix(RISK_FIELDS['has_trade'], start, end)
    rows = []
    for domain in config.domains:
        weights = context.stock_weights(domain, start, end)
        member = np.isfinite(weights) & (weights > 0)
        active = member & np.isfinite(trade) & (trade > 0)
        values = np.where(active & np.isfinite(returns), returns, np.nan)
        domain_trade = np.where(member, trade, 0)
        benchmark = lagged_weighted_returns(values/100, weights)
        for period in config.periods:
            value, rank = _compute(values, period, config.history)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
