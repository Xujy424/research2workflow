"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
import numpy as np
from ..utils import as_float, compound_return, _series_percentile
from ..utils import weighted_mean


def calculate(values, returns, weights, period=5, history=252):
    """市值加权计算估值最低 10% 与最高 10% 的收益差历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    spreads = []
    for t in range(period, len(values)):
        v = as_float(values[t - period])
        w = as_float(weights[t - period])
        r = compound_return(returns[t - period + 1:t + 1], axis=0)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0) & np.isfinite(r)
        if valid.sum() < 10:
            spreads.append(np.nan)
            continue
        cut = max(1, int(valid.sum() * 0.1))
        order = np.argsort(v[valid])
        idx = np.flatnonzero(valid)
        low, high = idx[order[:cut]], idx[order[-cut:]]
        spreads.append(weighted_mean(r[low], w[low]) - weighted_mean(r[high], w[high]))
    return _series_percentile(spreads, history, 1)
