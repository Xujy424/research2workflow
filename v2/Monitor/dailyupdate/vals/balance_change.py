"""独立指标：balance_change；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..utils import safe_divide


def balance_change(balance, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    balance = np.asarray(balance, float)
    if balance.ndim > 1:
        balance = np.nansum(balance, axis=1)
    return _percentile(np.r_[np.nan, safe_divide(balance[period:], balance[:-period]) - 1], history, 1)
