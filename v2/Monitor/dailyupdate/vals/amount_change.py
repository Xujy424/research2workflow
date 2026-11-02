"""成交额变化; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import MONEYFLOW_FIELDS
from ..utils import _sum_matrices


def _compute(amount, history=252, period=1, baseline=20):
    """近 h 日均成交额相对紧邻前 20 日基准均额的历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    amount = np.nansum(np.asarray(amount, float), axis=1)
    values = []
    for i in range(len(amount)):
        if i < baseline + period - 1:
            values.append(np.nan)
            continue
        current = np.nanmean(amount[i-period+1:i+1])
        base = np.nanmean(amount[i-period-baseline+1:i-period+1])
        values.append(current / base - 1 if base > 0 else np.nan)
    return _percentile(values, history)


def calculate(context, date, config=None, *, baseline=20):
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
    for domain in context.eligible_domains(config.domains, 'market_industry'):
        high, low = context.domain_pair_weights(start, end, domain)
        members = [np.isfinite(high)] if low is None else [np.isfinite(high), np.isfinite(low)]
        for period in config.periods:
            values = []
            for member in members:
                values.append(_compute(np.where(member, amount, np.nan), config.history, period, baseline=baseline))
            rank = values[0] if len(values)==1 else values[0]-values[1]
            rows.append(dict(domain=domain, period=period, percentile=rank))
    return metric_result(rows)
