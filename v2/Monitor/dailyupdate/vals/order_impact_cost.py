"""订单冲击成本; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..vals.effective_spread import _compute as effective_spread
from ..vals.estimate_kyle_lambda import _compute as estimate_kyle_lambda
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes, order_notional, kyle_lambda=None):
    """给定委托名义金额的成本估计：半价差 + Kyle lambda × 委托名义金额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    if kyle_lambda is None:
        kyle_lambda = estimate_kyle_lambda(quotes)
    return effective_spread(quotes) / 20000 + np.asarray(kyle_lambda) * np.asarray(order_notional)


def calculate(context, date, config=None, *, scope='wide', order_fraction=0.10):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {'order_fraction': order_fraction}
    for row, idx in enumerate(range(start, end+1)):
        key = ('order_impact_cost', tuple(sorted(parameters.items())), idx)
        values = context.factor_cache.get(key)
        if values is None:
            quotes = context.aligned_snapshot(context.dates[idx])
            amount = context.row('m_essentials/amount', idx)
            values = _compute(quotes, order_fraction * amount)
            values = np.full(len(context.ticks), np.nan) if values is None else values
            context.cache_factor(key, values)
        daily[row] = values
    # 按域取权重；barra域按高低组分别加权后相减。
    rows = []
    for domain in context.eligible_domains(config.domains, scope):
        high, low = context.domain_pair_weights(start, end, domain)
        series = _weighted_rows(daily, high)
        if low is not None:
            series = series - _weighted_rows(daily, low)
        for period in config.periods:
            rows.append(dict(domain=domain, period=period,
                             percentile=_history_percentile(series, config.history, period)))
    return metric_result(rows)
