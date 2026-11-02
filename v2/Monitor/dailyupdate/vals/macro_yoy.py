"""M2同比 / 社融存量同比; Context entry for fetching and calculating this indicator."""
import pandas as pd
from ..config import MonitorConfig
from ..utils import yoy_series, macro_result


def calculate(context, date, config=None, *, yoy_name, balance_name):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    context.window(date, 0)
    end = pd.Timestamp(date)
    start = end - pd.DateOffset(years=3)
    # 保留原版同比/余额回退逻辑；不在调度器中拼接宏观指标。
    values = context.trading_series(yoy_name, start, end)
    if not len(values):
        balance = context.trading_series(balance_name, start, end)
        values = yoy_series(balance)
    return macro_result(values, config.history)
