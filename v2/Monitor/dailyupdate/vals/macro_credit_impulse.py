"""信用脉冲; Context entry for fetching and calculating this indicator."""
import pandas as pd
from ..config import MonitorConfig
from ..utils import credit_impulse, macro_result


def calculate(context, date, config=None, *, window=12, lag=3):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    context.window(date, 0)
    end = pd.Timestamp(date)
    start = end - pd.DateOffset(years=3)
    # 取新增社融和GDP，以12个月社融/GDP的3个月变化形成信用脉冲。
    tsf = context.trading_series('新增社融', start, end)
    gdp = context.trading_series('名义GDP', start, end)
    values = credit_impulse(tsf, gdp, window=window, lag=lag)
    dates = context.dates[(context.dates >= start) & (context.dates <= end)]
    values = values.reindex(dates).ffill()
    return macro_result(values, config.history, scale=100, change_scale=100)
