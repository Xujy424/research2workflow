"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
import numpy as np
from ..utils import _percentile, safe_divide


def calculate(amount, category, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    amount, category = np.asarray(amount, float), np.asarray(category, bool)
    values = safe_divide(np.nansum(np.where(category, amount, np.nan), axis=1), np.nansum(amount, axis=1))
    values[~category.any(axis=1)] = np.nan
    return _percentile(values, history, period)
