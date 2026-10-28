"""独立指标：growth_value_spread_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import mask_spread_percentile


def growth_value_spread_percentile(mask, returns, weights, period=5, history=252):
    """直接使用成长/价值mask：成长组(1)减价值组(-1)，避免临时重建未经公告日处理的盈利增长。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)
