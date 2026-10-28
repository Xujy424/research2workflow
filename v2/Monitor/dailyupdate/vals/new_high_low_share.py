"""独立指标：new_high_low_share；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _series_percentile
from ..utils import as_float


def new_high_low_share(daily_returns, membership, lookback=252, period=1, history=252):
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
