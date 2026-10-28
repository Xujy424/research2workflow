"""独立指标：liquidity_pressure_percentile；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import _p
from ..utils import spread


def liquidity_pressure_percentile(spread, depth_decline, amihud, history=252):
    """先等权合成三个已标准化压力分位，再计算综合历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    frame = pd.concat([pd.Series(spread), pd.Series(depth_decline), pd.Series(amihud)], axis=1)
    score = frame.mean(axis=1, skipna=False)
    return (float(score.iloc[-1]), _p(score.to_numpy(), history)) if len(score) else (np.nan, np.nan)
