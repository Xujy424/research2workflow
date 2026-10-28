"""独立指标：effective_spread；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np


def effective_spread(quotes):
    """日内相对买卖价差中位数，单位 bp。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    x = quotes["spread"]
    return np.nanmedian(x, axis=0)
