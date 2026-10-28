"""独立指标：industry_relative_return；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..vals.relative_return import relative_return


def industry_relative_return(stock_returns, industry_weight, benchmark_returns, period=5):
    """以期初行业成分权重求行业区间收益，减去同期市场基准区间收益。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return relative_return(stock_returns, industry_weight, benchmark_returns, period)
