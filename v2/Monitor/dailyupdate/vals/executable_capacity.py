"""独立指标：executable_capacity；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..vals.five_level_depth_notional import five_level_depth_notional


def executable_capacity(quotes, adtv, participation=0.10):
    """可成交金额代理值：min(10% ADTV, 五档盘口名义深度)。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return np.minimum(participation * np.asarray(adtv), five_level_depth_notional(quotes))
