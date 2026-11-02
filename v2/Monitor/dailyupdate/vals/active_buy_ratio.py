"""主动买入占比; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
from ..utils import _percentile
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import MONEYFLOW_FIELDS
from ..utils import _sum_matrices


def _compute(buy, sell, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    buy, sell = np.asarray(buy, float), np.asarray(sell, float)
    return _percentile(safe_divide(np.nansum(buy, axis=1), np.nansum(buy + sell, axis=1)), history, period)


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + max(config.periods) + 30)
    rows = []
    # 原口径的总成交额由四类主动买卖金额相加得到。
    buy = _sum_matrices([context.matrix(MONEYFLOW_FIELDS[f'{size}_buy'], start, end)
                         for size in ('large', 'medium', 'small', 'extra')])
    sell = _sum_matrices([context.matrix(MONEYFLOW_FIELDS[f'{size}_sell'], start, end)
                          for size in ('large', 'medium', 'small', 'extra')])
    for domain in context.eligible_domains(config.domains, 'all'):
        high, low = context.domain_pair_weights(start, end, domain)
        members = [np.isfinite(high)] if low is None else [np.isfinite(high), np.isfinite(low)]
        for period in config.periods:
            values = []
            for member in members:
                values.append(_compute(np.where(member, buy, np.nan), np.where(member, sell, np.nan), config.history, period))
            rank = values[0] if len(values)==1 else values[0]-values[1]
            rows.append(dict(domain=domain, period=period, percentile=rank))
    return metric_result(rows)
