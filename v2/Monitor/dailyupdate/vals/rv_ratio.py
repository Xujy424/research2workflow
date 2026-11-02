"""波动率期限结构RV5/RV20 / 波动率期限结构RV20/RV60; Context entry for fetching and calculating this indicator."""
import pandas as pd
from ..utils import historical_percentile
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns


def _compute(decimal_returns, short_window=5, long_window=20, history=252):
    # 两个窗口共用同一条小数收益；样本标准差比值中的年化因子相消。
    r = pd.Series(decimal_returns, dtype=float)
    left, right = r.rolling(short_window).std(), r.rolling(long_window).std()
    ratio = left / right.where(right > 0)
    return float(ratio.iloc[-1]), historical_percentile(ratio.iloc[-history:].to_numpy())


def calculate(context, date, config=None, *, short_window=5, long_window=20):
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
            value, rank = _compute(benchmark, short_window=short_window, long_window=long_window, history=config.history)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
