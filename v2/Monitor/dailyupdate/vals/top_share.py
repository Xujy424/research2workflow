"""独立指标：top_share；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _percentile


def top_share(amount, history=252, period=1, fraction=0.1):
    """每日成交额最高 fraction 股票的成交额占比历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = []
    for row in np.asarray(amount, float):
        valid = row[np.isfinite(row) & (row >= 0)]
        n = max(1, int(np.ceil(len(valid) * fraction))) if len(valid) else 0
        values.append(np.sum(np.sort(valid)[-n:]) / np.sum(valid) if n and np.sum(valid) > 0 else np.nan)
    return _percentile(values, history, period)
