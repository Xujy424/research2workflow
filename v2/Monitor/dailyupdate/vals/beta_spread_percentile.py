"""独立指标：beta_spread_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import mask_spread_percentile


def beta_spread_percentile(mask, returns, weights, period=5, history=252):
    """使用已存储的Beta高低分组mask；Beta暴露本身直接从本地barra/beta读取而不在此估计。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)
