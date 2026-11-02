"""跳跃-成交量偏离; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile
from .acd_jump_intensity import _compute as compute_acd
from .standardized_acd_intensity import _compute as compute_acd_z


def _compute(acd_z, volume_history):
    """ACD跳跃强度z分数减去近5日成交量z分数。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    volume = np.asarray(volume_history, float)[-5:]
    valid = np.isfinite(volume)
    count = valid.sum(axis=0)
    mean = safe_divide(np.where(valid, volume, 0.0).sum(axis=0), count)
    centered = np.where(valid, volume - mean, 0.0)
    std = np.sqrt(safe_divide((centered ** 2).sum(axis=0), count))
    volume_z = safe_divide(volume[-1] - mean, std)
    result = np.asarray(acd_z, float) - volume_z
    result[(count < 3) | (std <= 0) | ~np.isfinite(acd_z)] = np.nan
    return result


def calculate(context, date, config=None, *, scope='index'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    acd_history = []
    amount_history = []
    for row, idx in enumerate(range(start, end+1)):
        # ACD的单日原始强度可复用，但标准化只使用当前日之前的历史。
        key = ('acd_jump_intensity', (), idx)
        acd = context.factor_cache.get(key)
        if acd is None:
            acd = compute_acd(context.aligned_snapshot(context.dates[idx]))
            acd = np.full(len(context.ticks), np.nan) if acd is None else acd
            context.cache_factor(key, acd)
        acd_history.append(acd)
        amount_history.append(context.row('m_essentials/amount', idx))
        if len(acd_history) > 20:
            acd_z = compute_acd_z(acd_history[-1], acd_history[:-1])
            daily[row] = _compute(acd_z, amount_history[-5:])
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
