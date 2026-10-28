"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
import numpy as np
from ..utils import safe_divide


def calculate(quotes, horizon):
    """逐个最优档 OFI 冲击计算 horizon 分钟后剩余响应比例，再取日内中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow = quotes["ofi"][1:]
    if len(ret) <= horizon:
        return np.full(mid.shape[1], np.nan)
    future_windows = np.lib.stride_tricks.sliding_window_view(ret, horizon, axis=0)
    future_returns = np.moveaxis(future_windows, -1, 1)[1:]
    immediate_return = ret[:-horizon]
    signed_flow = np.sign(flow[:-horizon])
    valid = (np.isfinite(future_returns).all(axis=1)
             & np.isfinite(immediate_return)
             & np.isfinite(signed_flow) & (signed_flow != 0))
    immediate = signed_flow * immediate_return
    remaining = signed_flow * future_returns.sum(axis=1)
    ratios = safe_divide(remaining, immediate + remaining)
    ratios[(np.abs(immediate) <= 1e-12) | ~valid] = np.nan
    valid_columns = np.isfinite(ratios).any(axis=0)
    result = np.full(mid.shape[1], np.nan)
    if valid_columns.any():
        result[valid_columns] = np.nanmedian(ratios[:, valid_columns], axis=0)
    return result
