"""独立指标：estimate_kyle_lambda；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide


def estimate_kyle_lambda(quotes):
    """以最优档OFI名义额回归分钟中间价收益，逐股估计 Kyle lambda。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow_notional = quotes["ofi"][1:] * mid[1:]
    valid = np.isfinite(ret) & np.isfinite(flow_notional)
    count = valid.sum(axis=0)
    x = np.where(valid, flow_notional, 0.0)
    y = np.where(valid, ret, 0.0)
    mean_x = safe_divide(x.sum(axis=0), count)
    mean_y = safe_divide(y.sum(axis=0), count)
    x_centered = np.where(valid, flow_notional - mean_x, 0.0)
    y_centered = np.where(valid, ret - mean_y, 0.0)
    variance = safe_divide((x_centered ** 2).sum(axis=0), count)
    covariance = safe_divide((x_centered * y_centered).sum(axis=0), count)
    result = safe_divide(covariance, variance)
    result[(count < 30) | (variance <= 0)] = np.nan
    return result
