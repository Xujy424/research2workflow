"""独立指标：amount_share_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _series_percentile
from ..utils import as_float, safe_divide


def amount_share_percentile(amount_history, membership_history, history=252, period=1):
    """组内成交金额/全A成交金额；先按日计算份额，再对同period历史序列取分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    a = as_float(amount_history)
    part = np.nansum(np.where(membership_history, a, np.nan), axis=1)
    total = np.nansum(a, axis=1)
    return _series_percentile(safe_divide(part, total), history, period)
