"""大单成交占比 / 中单成交占比 / 小单成交占比; Context entry for fetching and calculating this indicator."""
import numpy as np
from ..utils import _percentile, safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..config import MONEYFLOW_FIELDS
from ..utils import _sum_matrices


def _compute(amount, category, history=252, period=1):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    amount, category = np.asarray(amount, float), np.asarray(category, bool)
    values = safe_divide(np.nansum(np.where(category, amount, np.nan), axis=1), np.nansum(amount, axis=1))
    values[~category.any(axis=1)] = np.nan
    return _percentile(values, history, period)


def calculate(context, date, config=None, *, size='large'):
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
    category = np.isfinite(context.matrix(MONEYFLOW_FIELDS[f'{size}_buy'], start, end))
    for domain in context.eligible_domains(config.domains, 'index'):
        high, low = context.domain_pair_weights(start, end, domain)
        members = [np.isfinite(high)] if low is None else [np.isfinite(high), np.isfinite(low)]
        for period in config.periods:
            values = []
            for member in members:
                values.append(_compute(np.where(member, amount, np.nan), category, config.history, period))
            rank = values[0] if len(values)==1 else values[0]-values[1]
            rows.append(dict(domain=domain, period=period, percentile=rank))
    return metric_result(rows)
