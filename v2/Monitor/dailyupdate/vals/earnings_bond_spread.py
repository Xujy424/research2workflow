"""盈利-10年国债收益率; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import as_float, safe_divide, weighted_mean
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import weighted_mean


def _compute(pe_history, weight_history, bond_yield, period=1):
    """股债收益差：股票池盈利收益率(加权1/PE)减同期10年国债收益率，最近period日取均值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    earnings = np.array([weighted_mean(safe_divide(1, p), w) for p, w in zip(pe_history, weight_history)])
    return float(np.nanmean((earnings - as_float(bond_yield))[-period:]))


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + config.high_lookback + max(config.periods) + config.rotation_window + 260)
    rows = []
    valuation = context.matrix('fundamental/valuation/pe_ttm', start, end)
    code = context.edb_codes.get('bond_10y')
    bond = (context.edb(code, context.dates[start], context.dates[end])
            if code is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    bond = bond.reindex(context.dates[start:end+1]).ffill().to_numpy(float)
    if np.isfinite(bond).any() and np.nanmedian(np.abs(bond)) > 1:
        bond = bond/100
    for universe in config.universes:
        weights = context.stock_weights(universe, start, end)
        members = np.isfinite(weights) & (weights > 0)
        for period in config.periods:
            value = _compute(valuation, weights, bond, period) if period < end-start+1 else np.nan
            rows.append(dict(domain=universe, period=period, value=value))
    return metric_result(rows)
