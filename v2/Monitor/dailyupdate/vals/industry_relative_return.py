"""行业相对收益; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
from ..vals.relative_return import _compute as relative_return
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..utils import lagged_weighted_returns


def _compute(stock_returns, industry_weight, benchmark_returns, period=5):
    """以期初行业成分权重求行业区间收益，减去同期市场基准区间收益。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return relative_return(stock_returns, industry_weight, benchmark_returns, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    returns = context.matrix('d_essentials/pct', start, end)/100
    benchmark = lagged_weighted_returns(returns, context.stock_weights(config.benchmark, start, end))
    industries = tuple(context.industry_map) if config.industries is None else config.industries
    for industry in industries:
        for period in config.periods:
            value = np.nan
            if period < len(returns):
                weights = context.stock_weights(industry, end-period)
                value = _compute(returns, weights, benchmark, period)
            rows.append(dict(domain=industry, period=period, value=value))
    return metric_result(rows)
