"""独立指标：amihud_illiquidity；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide


def amihud_illiquidity(daily_return, daily_amount):
    """Amihud 非流动性：绝对日收益/日成交额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return safe_divide(np.abs(daily_return), daily_amount)
