"""流动性压力历史分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import _p
from ..utils import spread
from ..config import MonitorConfig
from ..utils import metric_result


def _compute(spread, depth_decline, amihud, history=252):
    """先等权合成三个已标准化压力分位，再计算综合历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    frame = pd.concat([pd.Series(spread), pd.Series(depth_decline), pd.Series(amihud)], axis=1)
    score = frame.mean(axis=1, skipna=False)
    return (float(score.iloc[-1]), _p(score.to_numpy(), history)) if len(score) else (np.nan, np.nan)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    context.window(date, 0)
    components = config.module_options.get('risk', {}).get('micro_result') or {}
    required = ('相对价差分位', '深度下降分位', 'Amihud分位')
    value, rank = np.nan, np.nan
    if all(name in components for name in required):
        value, rank = _compute(*(components[name] for name in required), history=config.history)
    return metric_result([dict(domain='全市场', period=1, value=value, percentile=rank)])
