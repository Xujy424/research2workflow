"""DR007 / Shibor3M / 10Y国债收益率; Context entry for fetching and calculating this indicator."""
import pandas as pd
from ..config import MonitorConfig
from ..utils import macro_result


def calculate(context, date, config=None, *, series):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    context.window(date, 0)
    end = pd.Timestamp(date)
    start = end - pd.DateOffset(years=3)
    # 利率水平保持百分数口径，变化按bp展示。
    values = context.trading_series(series, start, end)
    return macro_result(values, config.history, scale=1, change_scale=100)
