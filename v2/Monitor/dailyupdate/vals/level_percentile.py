"""独立指标：level_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _series_percentile


def level_percentile(index_close, history=252, period=1):
    """指数点位先做 period 日平滑，再计算 history 日历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _series_percentile(index_close, history, period)
