"""独立指标：worst_rolling_loss；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import _p
from ..utils import _pct


def worst_rolling_loss(index_returns, lookback=252, loss_window=5, history=252):
    """最差滚动5日收益：过去回看窗口内5日复合收益的最低值。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r=_pct(index_returns).reshape(-1)
    five=pd.Series(r).rolling(loss_window).apply(lambda x: np.prod(1+x)-1).to_numpy()
    vals=pd.Series(five).rolling(lookback).min().to_numpy()
    return float(vals[-1]) if len(vals) else np.nan,_p(vals,history)
