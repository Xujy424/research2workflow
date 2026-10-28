"""独立指标：amount_change；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _percentile


def amount_change(amount, history=252, period=1, baseline=20):
    """近 h 日均成交额相对紧邻前 20 日基准均额的历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    amount = np.nansum(np.asarray(amount, float), axis=1)
    values = []
    for i in range(len(amount)):
        if i < baseline + period - 1:
            values.append(np.nan)
            continue
        current = np.nanmean(amount[i-period+1:i+1])
        base = np.nanmean(amount[i-period-baseline+1:i-period+1])
        values.append(current / base - 1 if base > 0 else np.nan)
    return _percentile(values, history)
