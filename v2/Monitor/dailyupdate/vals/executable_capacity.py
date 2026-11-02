"""可成交量; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..vals.five_level_depth_notional import _compute as five_level_depth_notional
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes, adtv, participation=0.10):
    """可成交金额代理值：min(10% ADTV, 五档盘口名义深度)。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return np.minimum(participation * np.asarray(adtv), five_level_depth_notional(quotes))


def calculate(context, date, config=None, *, scope='wide', participation=0.10):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {'participation': participation}
    for row, idx in enumerate(range(start, end+1)):
        key = ('executable_capacity', tuple(sorted(parameters.items())), idx)
        values = context.factor_cache.get(key)
        if values is None:
            quotes = context.aligned_snapshot(context.dates[idx])
            amount = context.row('m_essentials/amount', idx)
            values = _compute(quotes, amount, participation=participation)
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
