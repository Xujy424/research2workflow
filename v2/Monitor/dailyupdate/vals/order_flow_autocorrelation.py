"""参数化指标计算；窗口参数由 config 注册表统一指定。"""
from ..utils import _lag_correlation_columns


def calculate(quotes, lag):
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    if quotes is None:
        return None
    return _lag_correlation_columns(quotes["ofi"], lag)
