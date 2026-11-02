"""ACD预期跳跃强度; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from functools import reduce
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes):
    """由30分钟跳跃事件间隔估计 ACD(1,1) 的预期跳跃强度 1/psi。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    n_minutes, n_stocks = ret.shape
    result = np.full(n_stocks, np.nan)
    if n_minutes <= 30:
        return result

    # rolling[k] = ret[k:k+30]；第 i 根收益对应此前样本 rolling[i-30]。
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)
    past_samples = samples[:-1]
    valid_count = np.isfinite(past_samples).sum(axis=1)
    sigma = np.nanstd(past_samples, axis=1)
    eligible_tail = np.isfinite(ret[30:]) & (valid_count[:n_minutes - 30] >= 25) & (sigma[:n_minutes - 30] > 0)
    jump_tail = eligible_tail & (np.abs(ret[30:]) > 3 * sigma[:n_minutes - 30])
    jump = np.zeros(ret.shape, dtype=bool)
    jump[30:] = jump_tail

    # 把逐分钟事件矩阵压成逐股事件位置；仅对实际发生事件的股票构造等待时间。
    alpha, beta = 0.10, 0.80
    for j in np.flatnonzero(jump.sum(axis=0) >= 2):
        events = np.flatnonzero(jump[:, j])
        waits = np.diff(events).astype(float)
        if waits.size == 0:
            continue
        omega = (1 - alpha - beta) * np.mean(waits)
        initial_psi = max(float(np.mean(waits)), 1.0)
        psi = reduce(
            lambda previous_psi, wait: omega + alpha * wait + beta * previous_psi,
            waits,
            initial_psi,
        )
        result[j] = 1 / psi
    return result


def calculate(context, date, config=None, *, scope='wide'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {}
    for row, idx in enumerate(range(start, end+1)):
        key = ('acd_jump_intensity', tuple(sorted(parameters.items())), idx)
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
