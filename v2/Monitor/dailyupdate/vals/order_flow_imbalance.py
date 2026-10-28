"""独立指标：order_flow_imbalance；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _sum_preserve_nan
from ..utils import safe_divide


def order_flow_imbalance(quotes):
    """最优档 OFI 净流量/绝对流量总和。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    flow = quotes["ofi"]
    return safe_divide(_sum_preserve_nan(flow, 0), _sum_preserve_nan(np.abs(flow), 0))
