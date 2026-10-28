"""独立指标：order_impact_cost；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..vals.effective_spread import effective_spread
from ..vals.estimate_kyle_lambda import estimate_kyle_lambda


def order_impact_cost(quotes, order_notional, kyle_lambda=None):
    """给定委托名义金额的成本估计：半价差 + Kyle lambda × 委托名义金额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    if kyle_lambda is None:
        kyle_lambda = estimate_kyle_lambda(quotes)
    return effective_spread(quotes) / 20000 + np.asarray(kyle_lambda) * np.asarray(order_notional)
