"""最差滚动5日收益; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import _p
from ..utils import _pct
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns, _p


def _compute(index_returns, lookback=252, loss_window=5, history=252):
    """最差滚动5日收益：过去回看窗口内5日复合收益的最低值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r=_pct(index_returns).reshape(-1)
    five=pd.Series(r).rolling(loss_window).apply(lambda x: np.prod(1+x)-1).to_numpy()
    vals=pd.Series(five).rolling(lookback).min().to_numpy()
    return float(vals[-1]) if len(vals) else np.nan,_p(vals,history)


def calculate(context, date, config=None, *, loss_window=5):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + 260)
    returns = context.matrix(RISK_FIELDS['returns'], start, end)
    trade = context.matrix(RISK_FIELDS['has_trade'], start, end)
    rows = []
    for domain in config.domains:
        weights = context.stock_weights(domain, start, end)
        member = np.isfinite(weights) & (weights > 0)
        active = member & np.isfinite(trade) & (trade > 0)
        values = np.where(active & np.isfinite(returns), returns, np.nan)
        domain_trade = np.where(member, trade, 0)
        benchmark = lagged_weighted_returns(values/100, weights)
        for period in config.periods:
            value, rank = _compute(benchmark*100, config.history, loss_window, config.history)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
