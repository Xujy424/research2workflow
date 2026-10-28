"""独立指标：cross_section_dispersion；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _p
from ..utils import _pct


def cross_section_dispersion(returns, period=1, history=252):
    """横截面收益离散度：每日股票收益标准差，再按周期取均值并计算历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r=_pct(returns); daily=np.nanstd(r,axis=1,ddof=1)
    return float(np.nanmean(daily[-period:])), _p(daily,history)
