"""独立指标：jump_frequency_30m；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np


def jump_frequency_30m(quotes):
    """滚动30分钟波动率三倍标准差阈值以上跳跃次数/有效观测次数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    result = np.full(ret.shape[1], np.nan)
    if len(ret) <= 30:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)[:-1]
    valid_count = np.isfinite(samples).sum(axis=1)
    sigma = np.nanstd(samples, axis=1)
    eligible = np.isfinite(ret[30:]) & (valid_count[:len(ret) - 30] >= 25) & (sigma[:len(ret) - 30] > 0)
    events = eligible & (np.abs(ret[30:]) > 3 * sigma[:len(ret) - 30])
    denominator = eligible.sum(axis=0)
    numerator = events.sum(axis=0)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result
