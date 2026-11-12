"""作者：xujiayi | 创建日期：2026-11-11
整体表现（时序）：择时信号分类与收益检验，重叠标签使用HAC标准误。
"""
import numpy as np
import pandas as pd
from . import maths as m


def _samples(base, signal, benchmark, threshold, side, horizon):
    """对齐择时样本；Params: 信号/基准/阈值/方向/期限；Return: 触发、收益、基准。"""
    if side not in ("long", "short"):
        raise ValueError("择时side须为long/short")
    if signal is None:
        raise ValueError("择时检验须显式提供日期索引的一维signal，不自动聚合截面因子")
    signal = signal.reindex(base.dates).loc[base.report_start:]
    benchmark = base.cal_benchmark() if benchmark is None else benchmark.reindex(base.dates)
    r = m.forward_returns(benchmark.to_frame("r"), horizon)["r"].reindex(signal.index)
    b = r.copy()
    valid = signal.notna() & r.notna()
    return signal[valid] > threshold, r[valid] * (1 if side == "long" else -1), b[valid]


def classification_table(base, signal, benchmark=None, threshold=0., side="long"):
    """择时分类；Params: 基类/一维信号/基准日收益/触发阈值/方向；Return: 期限×指标。

    正类为方向调整后的未来收益>0，基准胜率取全部有效日期。
    """
    rows = {}
    for h in (1, 3, 5, 10, 20):
        event, r, _ = _samples(base, signal, benchmark, threshold, side, h)
        win = r > 0
        tp = (event & win).sum()
        precision = tp / event.sum() if event.sum() else np.nan
        recall = tp / win.sum() if win.sum() else np.nan
        rate = win.mean()
        rows[h] = {"precision": precision, "benchmark_win_rate": rate,
                   "lift": precision / rate if rate > 0 else np.nan, "recall": recall,
                   "F1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan,
                   "trigger_count": int(event.sum())}
    result = pd.DataFrame(rows).T
    base.results["timing_classification"] = result
    return result


def returns_table(base, signal, benchmark=None, threshold=0., side="long"):
    """择时条件收益与HAC均值检验。

    Params: 同classification_table；Return: 期限×指标。
    超额=触发条件收益-全部样本平均收益；边际窗口依需求为(2,2)/(3,3)/(4,5)/(6,10)/(11,20)。
    """
    rows = {}
    market = base.cal_benchmark() if benchmark is None else benchmark.reindex(base.dates)
    for h, start, length in ((1, 2, 1), (3, 3, 1), (5, 4, 2), (10, 6, 5), (20, 11, 10)):
        event, r, _ = _samples(base, signal, market, threshold, side, h)
        selected = r.where(event)
        excess = selected - r.mean()
        t, p = m.hac_mean_test(excess, h - 1)
        marginal = m.forward_returns(market.to_frame("r"), length, start)["r"]
        marginal = marginal.reindex(event.index) * (1 if side == "long" else -1)
        rows[h] = {"mean_excess_return": excess.mean(), "win_rate": (selected.dropna() > 0).mean(),
                   "t_value": t, "p_value": p, "mean_marginal_return": marginal.where(event).mean(),
                   "mean_return": selected.mean(), "sample_count": selected.count()}
    result = pd.DataFrame(rows).T
    base.results["timing_returns"] = result
    return result
