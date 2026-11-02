"""成交额占比历史分位; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _series_percentile
from ..utils import as_float, safe_divide
from ..config import MonitorConfig
from ..utils import metric_result


def _compute(amount_history, membership_history, history=252, period=1):
    """组内成交金额/全A成交金额；先按日计算份额，再对同period历史序列取分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    a = as_float(amount_history)
    part = np.nansum(np.where(membership_history, a, np.nan), axis=1)
    total = np.nansum(a, axis=1)
    return _series_percentile(safe_divide(part, total), history, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    amount = context.matrix('d_essentials/amount', start, end)
    for universe in config.universes:
        weights = context.stock_weights(universe, start, end)
        members = np.isfinite(weights) & (weights > 0)
        for period in config.periods:
            value = _compute(amount, members, config.history, period) if period < end-start+1 else np.nan
            rows.append(dict(domain=universe, period=period, percentile=value))
    return metric_result(rows)
