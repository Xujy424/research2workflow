"""实现波动率RV5 / 实现波动率RV20 / 实现波动率RV60; Context entry for fetching and calculating this indicator."""
import numpy as np
import pandas as pd
from ..utils import _pct, _p
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns, _p


def _compute(index_returns, window=20, period=1, history=252):
    """实现波动率：窗口内日收益标准差年化，返回当前值及历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    vals=pd.Series(_pct(index_returns).reshape(-1)).rolling(window).std().to_numpy()*np.sqrt(252)
    return float(np.nanmean(vals[-period:])),_p(vals,history)


def calculate(context, date, config=None, *, window=20):
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
        for period in (1,):
            value, rank = _compute(benchmark*100, window=window, period=period, history=config.history)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
