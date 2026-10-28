"""独立指标：limit_down_ratio；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _limit_down_series
from ..utils import _p


def limit_down_ratio(hit_floor, has_trade=None, period=1, history=252, limit_eligible=None):
    """跌停扩散：跌停股票数 / 有效交易且适用涨跌幅限制股票数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    vals = _limit_down_series(hit_floor, has_trade, limit_eligible)
    return float(np.nanmean(vals[-period:])),_p(vals,history)
