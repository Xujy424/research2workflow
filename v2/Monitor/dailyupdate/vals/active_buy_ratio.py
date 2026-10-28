"""独立指标：active_buy_ratio；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..utils import safe_divide


def active_buy_ratio(buy, sell, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    buy, sell = np.asarray(buy, float), np.asarray(sell, float)
    return _percentile(safe_divide(np.nansum(buy, axis=1), np.nansum(buy + sell, axis=1)), history, period)
