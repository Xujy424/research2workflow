"""首主成分解释率; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
from ..utils import _pca_series
from ..utils import _summarize
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..config import RISK_FIELDS
from ..utils import lagged_weighted_returns, _summarize, _pca_series


def calculate(context, date, config=None, *, window=60):
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
        # 序列只计算一次，各period仅复用历史序列做汇总。
        series = _pca_series(values, window)
        for period in config.periods:
            value, rank = _summarize(series, period, config.history)
            rows.append(dict(domain=domain, period=period, value=value, percentile=rank))
    return metric_result(rows)
