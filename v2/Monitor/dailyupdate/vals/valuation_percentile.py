"""PE_TTM历史分位 / PB_LF历史分位; Context entry for fetching and calculating this indicator."""
from ..utils import _aggregated_percentile, harmonic_valuation
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np


def _compute(multiple_history, weight_history, history=252, period=1):
    # 每日按市值权重聚合倒数估值，再对period日均值求history日历史分位。
    return _aggregated_percentile(multiple_history, weight_history, harmonic_valuation, history, period)


def calculate(context, date, config=None, *, field='fundamental/valuation/pe_ttm'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    valuation = context.matrix(field, start, end)
    for universe in config.universes:
        weights = context.stock_weights(universe, start, end)
        members = np.isfinite(weights) & (weights > 0)
        for period in config.periods:
            value = _compute(valuation, weights, config.history, period) if period < end-start+1 else np.nan
            rows.append(dict(domain=universe, period=period, percentile=value))
    return metric_result(rows)
