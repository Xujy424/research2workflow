"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
import numpy as np
from ..utils import safe_divide


def calculate(quotes, horizon):
    """逐时点计算 horizon 窗口累计对数收益/首分钟对数收益，再取日内中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    if len(mid) <= horizon:
        return np.full(mid.shape[1], np.nan)
    start, first, end = mid[:-horizon], mid[0:len(mid) - horizon], mid[horizon:]
    # For horizon == 1 the first-step return is the full window return; the ratio is undefined.
    if horizon == 1:
        return np.full(mid.shape[1], np.nan)
    ratios = safe_divide(np.log(safe_divide(end, start)), np.log(safe_divide(first, start)))
    result = np.full(mid.shape[1], np.nan)
    valid = np.isfinite(ratios).any(axis=0)
    if valid.any():
        result[valid] = np.nanmedian(ratios[:, valid], axis=0)
    return result
