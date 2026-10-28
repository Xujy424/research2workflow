"""方差风险溢价代理 IV30²-RV30²。"""
import numpy as np


def calculate(atm, close):
    # IV与RV统一使用年化小数；RV窗口沿用30个交易日，非严格的30日历日匹配。
    close = close.where(np.isfinite(close) & (close > 0)).sort_index()
    rv = close.pct_change(fill_method=None).rolling(30).std(ddof=1) * np.sqrt(252)
    return atm**2 - rv**2
