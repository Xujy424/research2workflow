"""市场监控通用数值运算；不读取文件、不访问数据库。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def as_float(x):
    return np.asarray(x, dtype=float)


def safe_divide(a, b):
    a, b = np.broadcast_arrays(as_float(a), as_float(b))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b != 0))


def weighted_mean(values, weights):
    x, w = as_float(values), as_float(weights)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def compound_return(returns, axis=0):
    """NaN不作为零收益；无完整有效观察的证券返回NaN。"""
    r = as_float(returns)
    ans = np.prod(1 + np.where(np.isfinite(r), r, 0), axis=axis) - 1
    return np.where(np.isfinite(r).all(axis=axis), ans, np.nan)


def historical_percentile(values):
    """末值在有效历史样本中的中位秩，包含当期，结果0~1。"""
    v = as_float(values).ravel()
    if not len(v) or not np.isfinite(v[-1]):
        return np.nan
    v = v[np.isfinite(v)]
    return float(((v < v[-1]).sum() + 0.5 * (v == v[-1]).sum()) / len(v))


def rolling_percentile(series, history=252):
    x = as_float(series)
    return np.array([historical_percentile(x[max(0, i-history+1):i+1]) for i in range(len(x))])


def window_mean(series, window=1):
    x = as_float(series)
    return float(np.nanmean(x[-window:])) if len(x) >= window else np.nan


def rank_rotation(previous, current, method='spearman'):
    """相邻两个行业排序：Spearman越低、1-rho越大，轮动越快。"""
    x, y = as_float(previous), as_float(current)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    a, b = pd.Series(x[valid]).rank(), pd.Series(y[valid]).rank()
    rho = a.corr(b, method='pearson')
    return float(1 - rho) if method == 'spearman' else float(np.mean(np.abs(a - b)) / len(a))
