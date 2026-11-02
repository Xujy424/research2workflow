"""点位252历史分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
from ..utils import _series_percentile
from ..config import MonitorConfig, INDEX_MAP, INDEX_CODES
from ..utils import metric_result
import numpy as np
import pandas as pd


def _compute(index_close, history=252, period=1):
    """指数点位先做 period 日平滑，再计算 history 日历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _series_percentile(index_close, history, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    for universe in config.universes:
        # 使用官方指数点位；未配置指数代码时保留缺失值。
        code = INDEX_CODES.get(INDEX_MAP.get(universe, universe))
        close = (context.wsd(code, 'close', context.dates[start], context.dates[end])
                 if code is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        close = close.reindex(context.dates[start:end+1]).ffill().to_numpy(float)
        for period in config.periods:
            value = _compute(close, config.level_history, period) if period < end-start+1 else np.nan
            rows.append(dict(domain=universe, period=period, percentile=value))
    return metric_result(rows)
