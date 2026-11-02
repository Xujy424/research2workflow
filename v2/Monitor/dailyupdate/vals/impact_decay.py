"""冲击保留率_1m / 冲击保留率_5m / 冲击保留率_15m / 冲击保留率_30m; Context entry for fetching and calculating this indicator."""
import numpy as np
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes, horizon):
    """逐个最优档 OFI 冲击计算 horizon 分钟后剩余响应比例，再取日内中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow = quotes["ofi"][1:]
    if len(ret) <= horizon:
        return np.full(mid.shape[1], np.nan)
    future_windows = np.lib.stride_tricks.sliding_window_view(ret, horizon, axis=0)
    future_returns = np.moveaxis(future_windows, -1, 1)[1:]
    immediate_return = ret[:-horizon]
    signed_flow = np.sign(flow[:-horizon])
    valid = (np.isfinite(future_returns).all(axis=1)
             & np.isfinite(immediate_return)
             & np.isfinite(signed_flow) & (signed_flow != 0))
    immediate = signed_flow * immediate_return
    remaining = signed_flow * future_returns.sum(axis=1)
    ratios = safe_divide(remaining, immediate + remaining)
    ratios[(np.abs(immediate) <= 1e-12) | ~valid] = np.nan
    valid_columns = np.isfinite(ratios).any(axis=0)
    result = np.full(mid.shape[1], np.nan)
    if valid_columns.any():
        result[valid_columns] = np.nanmedian(ratios[:, valid_columns], axis=0)
    return result


def calculate(context, date, config=None, *, scope='all', horizon=1):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {'horizon': horizon}
    for row, idx in enumerate(range(start, end+1)):
        key = ('impact_decay', tuple(sorted(parameters.items())), idx)
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
