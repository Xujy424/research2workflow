"""独立指标：amount_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _series_percentile
from ..utils import as_float


def amount_percentile(amount_history, membership_history, history=252, period=1):
    """股票池当日成交金额先求和，再比较同period平滑口径历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    a = np.where(membership_history, as_float(amount_history), np.nan)
    values = np.nansum(a, axis=1)
    return _series_percentile(values, history, period)
