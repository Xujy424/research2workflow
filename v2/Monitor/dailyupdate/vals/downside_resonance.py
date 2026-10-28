"""独立指标：downside_resonance；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
from ..utils import _limit_down_series
from ..utils import _p
from ..utils import _pct


def downside_resonance(returns, hit_floor=None, has_trade=None, period=1, history=252,
                      limit_eligible=None):
    """下行共振：负收益、极端下跌和跌停扩散三个分量等权平均。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    r = _pct(returns)
    valid = np.isfinite(r)
    if has_trade is not None:
        valid &= np.isfinite(has_trade) & (np.asarray(has_trade) > 0)
    counts = valid.sum(axis=1)
    neg = np.divide(((r < 0) & valid).sum(1), counts,
                    out=np.full(len(r), np.nan), where=counts > 0)
    extreme = np.divide(((r < -.05) & valid).sum(1), counts,
                        out=np.full(len(r), np.nan), where=counts > 0)
    limit = (np.full(len(r), np.nan) if hit_floor is None else
             _limit_down_series(hit_floor, has_trade, limit_eligible))
    # All three components are required by the research definition.
    vals = (neg + extreme + limit) / 3
    return float(np.nanmean(vals[-period:])),_p(vals,history)
