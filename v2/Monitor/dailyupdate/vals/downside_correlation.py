"""独立指标：downside_correlation；支持直接注入数组或时间序列。"""
from __future__ import annotations
from ..utils import _correlation_series
from ..utils import _summarize


def downside_correlation(returns, window=120, period=1, history=252, market_returns=None):
    """Rolling correlation on the lowest 20% of benchmark-return days.

    market_returns, when supplied, must be aligned daily decimal returns.
    """
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    return _summarize(_correlation_series(returns, window, downside=True,
                                        market_returns=market_returns), period, history)
