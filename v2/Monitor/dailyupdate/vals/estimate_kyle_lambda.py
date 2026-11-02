"""KyleLambda; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes):
    """以最优档OFI名义额回归分钟中间价收益，逐股估计 Kyle lambda。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow_notional = quotes["ofi"][1:] * mid[1:]
    valid = np.isfinite(ret) & np.isfinite(flow_notional)
    count = valid.sum(axis=0)
    x = np.where(valid, flow_notional, 0.0)
    y = np.where(valid, ret, 0.0)
    mean_x = safe_divide(x.sum(axis=0), count)
    mean_y = safe_divide(y.sum(axis=0), count)
    x_centered = np.where(valid, flow_notional - mean_x, 0.0)
    y_centered = np.where(valid, ret - mean_y, 0.0)
    variance = safe_divide((x_centered ** 2).sum(axis=0), count)
    covariance = safe_divide((x_centered * y_centered).sum(axis=0), count)
    result = safe_divide(covariance, variance)
    result[(count < 30) | (variance <= 0)] = np.nan
    return result


def calculate(context, date, config=None, *, scope='all'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {}
    for row, idx in enumerate(range(start, end+1)):
        key = ('estimate_kyle_lambda', tuple(sorted(parameters.items())), idx)
        values = context.factor_cache.get(key)
        if values is None:
            quotes = context.aligned_snapshot(context.dates[idx])
            values = _compute(quotes, **parameters)
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
