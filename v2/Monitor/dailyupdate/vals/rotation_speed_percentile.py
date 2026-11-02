"""行业轮动速度分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
from ..utils import _series_percentile
from ..utils import as_float, compound_return, rank_rotation
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..utils import weighted_mean


def _compute(industry_returns, rank_period=20, history=252, period=1):
    """以相邻日行业过去rank_period日收益排名Spearman相关性定义轮动速度=1-rho。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = as_float(industry_returns)
    speeds = []
    for t in range(rank_period, len(r)):
        previous = compound_return(r[t-rank_period:t], axis=0)
        current = compound_return(r[t-rank_period+1:t+1], axis=0)
        speeds.append(rank_rotation(previous, current))
    return _series_percentile(speeds, history, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    returns = context.matrix('d_essentials/pct', start, end)/100
    industry = context.matrix('industry/industry', start, end)
    weights = context.stock_weights(None, start, end)
    industries = tuple(context.industry_map) if config.industries is None else config.industries
    daily = np.full((len(returns), len(industries)), np.nan)
    for j, name in enumerate(industries):
        code = context.industry_map.get(name, name)
        for t in range(1, len(returns)):
            w = np.where(industry[t-1] == code, weights[t-1], np.nan)
            daily[t,j] = weighted_mean(returns[t], w)
    for period in config.periods:
        rank = _compute(daily, config.rotation_window, config.history, period) if period < len(daily) else np.nan
        rows.append(dict(domain='全市场', period=period, percentile=rank))
    return metric_result(rows)
