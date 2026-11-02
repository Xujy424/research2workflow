"""新高新低差占比历史分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _series_percentile
from ..utils import as_float
from ..config import MonitorConfig
from ..utils import metric_result


def _compute(daily_returns, membership, lookback=252, period=1, history=252):
    """先计算period日平滑值，再计算其history日历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r, masks = as_float(daily_returns), np.asarray(membership, dtype=bool)
    values = []
    for end in range(lookback, len(r)):
        segment = r[end-lookback:end+1]
        if len(segment) != lookback+1:
            values.append(np.nan)
            continue
        path = np.vstack([np.ones(segment.shape[1]), np.cumprod(1+segment, axis=0)])
        valid = np.isfinite(segment).all(0) & masks[end]
        high = path[-1] > np.max(path[:-1], axis=0)
        low = path[-1] < np.min(path[:-1], axis=0)
        values.append(float((high[valid].sum()-low[valid].sum())/valid.sum()) if valid.any() else np.nan)
    return _series_percentile(values, history, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    returns = context.matrix('d_essentials/pct', start, end)/100
    for universe in config.universes:
        weights = context.stock_weights(universe, start, end)
        members = np.isfinite(weights) & (weights > 0)
        for period in config.periods:
            value = _compute(returns, members, config.high_lookback, period, config.history) if period < end-start+1 else np.nan
            rows.append(dict(domain=universe, period=period, percentile=value))
    return metric_result(rows)
