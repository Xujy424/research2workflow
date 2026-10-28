"""独立指标：turnover；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..utils import safe_divide


def turnover(amount, float_mv, history=252, period=1):
    """组成交额/前一日自由流通市值的历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    amount = np.nansum(np.asarray(amount, float), axis=1)
    float_mv = np.nansum(np.asarray(float_mv, float), axis=1)
    return _percentile(np.r_[np.nan, safe_divide(amount[1:], float_mv[:-1])], history, period)
