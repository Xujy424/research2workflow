"""PE收益差分位 / PB收益差分位; Context entry for fetching and calculating this indicator."""
import numpy as np
from ..utils import as_float, compound_return, _series_percentile
from ..utils import weighted_mean
from ..config import MonitorConfig
from ..utils import metric_result


def _compute(values, returns, weights, period=5, history=252):
    """市值加权计算估值最低 10% 与最高 10% 的收益差历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    spreads = []
    for t in range(period, len(values)):
        v = as_float(values[t - period])
        w = as_float(weights[t - period])
        r = compound_return(returns[t - period + 1:t + 1], axis=0)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0) & np.isfinite(r)
        if valid.sum() < 10:
            spreads.append(np.nan)
            continue
        cut = max(1, int(valid.sum() * 0.1))
        order = np.argsort(v[valid])
        idx = np.flatnonzero(valid)
        low, high = idx[order[:cut]], idx[order[-cut:]]
        spreads.append(weighted_mean(r[low], w[low]) - weighted_mean(r[high], w[high]))
    return _series_percentile(spreads, history, 1)


def calculate(context, date, config=None, *, field='fundamental/valuation/pe_ttm'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    valuation = context.matrix(field, start, end)
    returns = context.matrix('d_essentials/pct', start, end)/100
    weights = context.stock_weights(None, start, end)
    for period in config.periods:
        rank = _compute(valuation, returns, weights, period, config.history) if period < len(returns) else np.nan
        rows.append(dict(domain='全市场', period=period, percentile=rank))
    return metric_result(rows)
