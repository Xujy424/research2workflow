"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
import numpy as np
import pandas as pd
from ..utils import _pct, _p


def calculate(index_returns, window=20, period=1, history=252):
    """实现波动率：窗口内日收益标准差年化，返回当前值及历史分位。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    vals=pd.Series(_pct(index_returns).reshape(-1)).rolling(window).std().to_numpy()*np.sqrt(252)
    return float(np.nanmean(vals[-period:])),_p(vals,history)
