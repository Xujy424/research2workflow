"""实现波动率期限结构，两个窗口均为参数。"""
import pandas as pd
from ..utils import historical_percentile


def calculate(decimal_returns, short_window=5, long_window=20, history=252):
    # 两个窗口共用同一条小数收益；样本标准差比值中的年化因子相消。
    r = pd.Series(decimal_returns, dtype=float)
    left, right = r.rolling(short_window).std(), r.rolling(long_window).std()
    ratio = left / right.where(right > 0)
    return float(ratio.iloc[-1]), historical_percentile(ratio.iloc[-history:].to_numpy())
