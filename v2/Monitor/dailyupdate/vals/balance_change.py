"""融资余额变化率; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result


def _compute(balance, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    balance = np.asarray(balance, float)
    if balance.ndim > 1:
        balance = np.nansum(balance, axis=1)
    return _percentile(np.r_[np.nan, safe_divide(balance[period:], balance[:-period]) - 1], history, 1)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + max(config.periods) + 30)
    rows = []
    code = context.edb_codes.get('margin_balance', 'M0075990')
    balance = context.edb(code, context.dates[start], context.dates[end])
    balance = balance.reindex(context.dates[start:end+1]).ffill().to_numpy(float)
    for domain in context.eligible_domains(config.domains, 'market'):
        high, low = context.domain_pair_weights(start, end, domain)
        members = [np.isfinite(high)] if low is None else [np.isfinite(high), np.isfinite(low)]
        for period in config.periods:
            values = []
            for member in members:
                values.append(_compute(balance, config.history, period))
            rank = values[0] if len(values)==1 else values[0]-values[1]
            rows.append(dict(domain=domain, period=period, percentile=rank))
    return metric_result(rows)
