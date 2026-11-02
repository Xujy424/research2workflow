"""DR007-政策利率利差 / 中短期票据AA-AAA信用利差 / 中短期票据AA-国开债信用利差 / 国债期限利差(10Y-1Y) / 10Y实际利率代理(Wind一致预期); Context entry for fetching and calculating this indicator."""
import pandas as pd
from ..config import MonitorConfig
from ..utils import aligned_spread, macro_result


def calculate(context, date, config=None, *, left, right, scale=100):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    context.window(date, 0)
    end = pd.Timestamp(date)
    start = end - pd.DateOffset(years=3)
    # 两侧序列分别取数，再按原口径计算利差及其变化分位。
    lhs = context.trading_series(left, start, end)
    rhs = context.trading_series(right, start, end)
    values = aligned_spread(lhs, rhs)
    return macro_result(values, config.history, scale=scale, change_scale=100)
