"""独立指标：jump_volume_divergence；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide


def jump_volume_divergence(acd_z, volume_history):
    """ACD跳跃强度z分数减去近5日成交量z分数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    volume = np.asarray(volume_history, float)[-5:]
    valid = np.isfinite(volume)
    count = valid.sum(axis=0)
    mean = safe_divide(np.where(valid, volume, 0.0).sum(axis=0), count)
    centered = np.where(valid, volume - mean, 0.0)
    std = np.sqrt(safe_divide((centered ** 2).sum(axis=0), count))
    volume_z = safe_divide(volume[-1] - mean, std)
    result = np.asarray(acd_z, float) - volume_z
    result[(count < 3) | (std <= 0) | ~np.isfinite(acd_z)] = np.nan
    return result
