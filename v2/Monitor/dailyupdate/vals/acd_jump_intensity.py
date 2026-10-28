"""独立指标：acd_jump_intensity；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from functools import reduce


def acd_jump_intensity(quotes):
    """由30分钟跳跃事件间隔估计 ACD(1,1) 的预期跳跃强度 1/psi。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    n_minutes, n_stocks = ret.shape
    result = np.full(n_stocks, np.nan)
    if n_minutes <= 30:
        return result

    # rolling[k] = ret[k:k+30]；第 i 根收益对应此前样本 rolling[i-30]。
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)
    past_samples = samples[:-1]
    valid_count = np.isfinite(past_samples).sum(axis=1)
    sigma = np.nanstd(past_samples, axis=1)
    eligible_tail = np.isfinite(ret[30:]) & (valid_count[:n_minutes - 30] >= 25) & (sigma[:n_minutes - 30] > 0)
    jump_tail = eligible_tail & (np.abs(ret[30:]) > 3 * sigma[:n_minutes - 30])
    jump = np.zeros(ret.shape, dtype=bool)
    jump[30:] = jump_tail

    # 把逐分钟事件矩阵压成逐股事件位置；仅对实际发生事件的股票构造等待时间。
    alpha, beta = 0.10, 0.80
    for j in np.flatnonzero(jump.sum(axis=0) >= 2):
        events = np.flatnonzero(jump[:, j])
        waits = np.diff(events).astype(float)
        if waits.size == 0:
            continue
        omega = (1 - alpha - beta) * np.mean(waits)
        initial_psi = max(float(np.mean(waits)), 1.0)
        psi = reduce(
            lambda previous_psi, wait: omega + alpha * wait + beta * previous_psi,
            waits,
            initial_psi,
        )
        result[j] = 1 / psi
    return result
