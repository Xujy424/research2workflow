"""独立指标：relative_return；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import as_float, compound_return, weighted_mean


def relative_return(stock_returns, start_weights, benchmark_returns, period=1):
    """按窗口起始时点成分权重计算区间收益，再减同区间基准收益。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = compound_return(as_float(stock_returns)[-period:])
    portfolio = weighted_mean(r, start_weights)
    benchmark = float(compound_return(as_float(benchmark_returns)[-period:]))
    return portfolio - benchmark
