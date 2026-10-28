"""独立指标：average_correlation；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _correlation_series
from ..utils import _summarize


def average_correlation(returns, window=60, period=1, history=252):
    """Rolling mean pair correlation via normalized column sums."""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _summarize(_correlation_series(returns, window), period, history)
