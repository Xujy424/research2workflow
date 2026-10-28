"""独立指标：earnings_bond_spread；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import as_float, safe_divide, weighted_mean


def earnings_bond_spread(pe_history, weight_history, bond_yield, period=1):
    """股债收益差：股票池盈利收益率(加权1/PE)减同期10年国债收益率，最近period日取均值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    earnings = np.array([weighted_mean(safe_divide(1, p), w) for p, w in zip(pe_history, weight_history)])
    return float(np.nanmean((earnings - as_float(bond_yield))[-period:]))
