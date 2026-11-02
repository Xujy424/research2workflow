"""30分钟跳跃频率; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes):
    """滚动30分钟波动率三倍标准差阈值以上跳跃次数/有效观测次数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    result = np.full(ret.shape[1], np.nan)
    if len(ret) <= 30:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)[:-1]
    valid_count = np.isfinite(samples).sum(axis=1)
    sigma = np.nanstd(samples, axis=1)
    eligible = np.isfinite(ret[30:]) & (valid_count[:len(ret) - 30] >= 25) & (sigma[:len(ret) - 30] > 0)
    events = eligible & (np.abs(ret[30:]) > 3 * sigma[:len(ret) - 30])
    denominator = eligible.sum(axis=0)
    numerator = events.sum(axis=0)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def calculate(context, date, config=None, *, scope='wide'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {}
    for row, idx in enumerate(range(start, end+1)):
        key = ('jump_frequency_30m', tuple(sorted(parameters.items())), idx)
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
