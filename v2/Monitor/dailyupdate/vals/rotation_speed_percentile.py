"""独立指标：rotation_speed_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _series_percentile
from ..utils import as_float, compound_return, rank_rotation


def rotation_speed_percentile(industry_returns, rank_period=20, history=252, period=1):
    """以相邻日行业过去rank_period日收益排名Spearman相关性定义轮动速度=1-rho。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = as_float(industry_returns)
    speeds = []
    for t in range(rank_period, len(r)):
        previous = compound_return(r[t-rank_period:t], axis=0)
        current = compound_return(r[t-rank_period+1:t+1], axis=0)
        speeds.append(rank_rotation(previous, current))
    return _series_percentile(speeds, history, period)
