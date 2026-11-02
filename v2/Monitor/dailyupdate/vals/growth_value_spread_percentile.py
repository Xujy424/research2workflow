"""成长价值收益差分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
from ..utils import mask_spread_percentile
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np


def _compute(mask, returns, weights, period=5, history=252):
    """直接使用成长/价值mask：成长组(1)减价值组(-1)，避免临时重建未经公告日处理的盈利增长。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return mask_spread_percentile(mask, returns, weights, period, history)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    # 期初分组和期初流通市值权重，防止用期末分组回看历史收益。
    mask = context.matrix(context.style_masks['growth_value'], start, end)
    returns = context.matrix('d_essentials/pct', start, end)/100
    weights = context.stock_weights(None, start, end)
    for period in config.periods:
        value = _compute(mask, returns, weights, period, config.history) if period < len(returns) else np.nan
        rows.append(dict(domain='全市场', period=period, percentile=value))
    return metric_result(rows)
