"""冲击路径_10m / 冲击路径_30m / 冲击路径_60m / 冲击路径_120m; Context entry for fetching and calculating this indicator."""
import numpy as np
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(quotes, horizon):
    """逐时点计算 horizon 窗口累计对数收益/首分钟对数收益，再取日内中位数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    mid = quotes["mid"]
    if len(mid) <= horizon:
        return np.full(mid.shape[1], np.nan)
    # 起点P_t、首分钟P_(t+1)、终点P_(t+h)必须错开；旧索引将first误写为P_t。
    start, first, end = mid[:-horizon], mid[1:len(mid) - horizon + 1], mid[horizon:]
    # For horizon == 1 the first-step return is the full window return; the ratio is undefined.
    if horizon == 1:
        return np.full(mid.shape[1], np.nan)
    ratios = safe_divide(np.log(safe_divide(end, start)), np.log(safe_divide(first, start)))
    result = np.full(mid.shape[1], np.nan)
    valid = np.isfinite(ratios).any(axis=0)
    if valid.any():
        result[valid] = np.nanmedian(ratios[:, valid], axis=0)
    return result


def calculate(context, date, config=None, *, scope='all', horizon=10):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {'horizon': horizon}
    for row, idx in enumerate(range(start, end+1)):
        key = ('impact_path', tuple(sorted(parameters.items())), idx)
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
