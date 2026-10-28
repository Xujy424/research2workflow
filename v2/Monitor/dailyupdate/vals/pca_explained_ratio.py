"""独立指标：pca_explained_ratio；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _pca_series
from ..utils import _summarize


def pca_explained_ratio(returns, window=60, period=1, history=252):
    """Leading correlation eigenvalue share via the smaller Gram matrix."""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _summarize(_pca_series(returns, window), period, history)
