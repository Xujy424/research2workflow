"""top10%成交额占比; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import MONEYFLOW_FIELDS
from ..utils import _sum_matrices


def _compute(amount, history=252, period=1, fraction=0.1):
    """每日成交额最高 fraction 股票的成交额占比历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    values = []
    for row in np.asarray(amount, float):
        valid = row[np.isfinite(row) & (row >= 0)]
        n = max(1, int(np.ceil(len(valid) * fraction))) if len(valid) else 0
        values.append(np.sum(np.sort(valid)[-n:]) / np.sum(valid) if n and np.sum(valid) > 0 else np.nan)
    return _percentile(values, history, period)


def calculate(context, date, config=None, *, fraction=0.1):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + max(config.periods) + 30)
    rows = []
    # 原口径的总成交额由四类主动买卖金额相加得到。
    amount = _sum_matrices([
        context.matrix(MONEYFLOW_FIELDS[f'{size}_buy'], start, end)
        + context.matrix(MONEYFLOW_FIELDS[f'{size}_sell'], start, end)
        for size in ('large', 'medium', 'small', 'extra')
    ])
    for domain in context.eligible_domains(config.domains, 'market'):
        high, low = context.domain_pair_weights(start, end, domain)
        members = [np.isfinite(high)] if low is None else [np.isfinite(high), np.isfinite(low)]
        for period in config.periods:
            values = []
            for member in members:
                values.append(_compute(np.where(member, amount, np.nan), config.history, period, fraction=fraction))
            rank = values[0] if len(values)==1 else values[0]-values[1]
            rows.append(dict(domain=domain, period=period, percentile=rank))
    return metric_result(rows)
