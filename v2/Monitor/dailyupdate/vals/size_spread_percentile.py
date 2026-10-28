"""独立指标：size_spread_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import mask_spread_percentile


def size_spread_percentile(mask, returns, weights, period=5, history=252):
    """直接使用大小盘mask，高市值组(1)收益减低市值组(-1)收益的历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)
