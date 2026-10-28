"""独立指标：five_level_depth_notional；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _sum_preserve_nan


def five_level_depth_notional(quotes):
    """日内五档买卖盘名义深度中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    bid = _sum_preserve_nan(quotes["bid_px"] * quotes["bid_qty"], 0)
    ask = _sum_preserve_nan(quotes["ask_px"] * quotes["ask_qty"], 0)
    return np.nanmedian(bid + ask, axis=0)
