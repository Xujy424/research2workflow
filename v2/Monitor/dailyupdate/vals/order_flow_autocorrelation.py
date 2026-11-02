"""订单流自相关-lag1 / 订单流自相关-lag5; Context entry for fetching and calculating this indicator."""
from ..utils import _lag_correlation_columns
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes, lag):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return _lag_correlation_columns(quotes["ofi"], lag)


def calculate(context, date, config=None, *, scope='wide', lag=1):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {'lag': lag}
    for row, idx in enumerate(range(start, end+1)):
        key = ('order_flow_autocorrelation', tuple(sorted(parameters.items())), idx)
        values = context.factor_cache.get(key)
        if values is None:
            quotes = context.aligned_snapshot(context.dates[idx])
            values = _compute(quotes, **parameters)
            values = np.full(len(context.ticks), np.nan) if values is None else values
            context.cache_factor(key, values)
        daily[row] = values
    # 按域取权重；barra域按高低组分别加权后相减。
    rows = []
    for domain in context.eligible_domains(config.domains, scope):
        high, low = context.domain_pair_weights(start, end, domain)
        series = _weighted_rows(daily, high)
        if low is not None:
            series = series - _weighted_rows(daily, low)
        for period in config.periods:
            rows.append(dict(domain=domain, period=period,
                             percentile=_history_percentile(series, config.history, period)))
    return metric_result(rows)
