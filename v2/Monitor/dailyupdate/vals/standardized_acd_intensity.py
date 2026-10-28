"""独立指标：standardized_acd_intensity；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide


def standardized_acd_intensity(current, prior_history, min_observations=10):
    """当前 ACD 强度相对之前20个交易日的横截面逐股 z-score。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    x = np.asarray(prior_history, float)[-20:]
    current = np.asarray(current, float)
    valid = np.isfinite(x)
    count = valid.sum(axis=0)
    mean = safe_divide(np.where(valid, x, 0.0).sum(axis=0), count)
    centered = np.where(valid, x - mean, 0.0)
    std = np.sqrt(safe_divide((centered ** 2).sum(axis=0), count))
    zscore = safe_divide(current - mean, std)
    zscore[(count < min_observations) | (std <= 0) | ~np.isfinite(current)] = np.nan
    return zscore
