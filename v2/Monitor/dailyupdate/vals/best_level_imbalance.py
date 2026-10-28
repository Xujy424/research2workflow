"""独立指标：best_level_imbalance；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _nanmean_rows


def best_level_imbalance(quotes):
    """日内最优买卖队列失衡均值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return _nanmean_rows(quotes["best_imbalance"])
