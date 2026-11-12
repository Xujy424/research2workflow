"""作者：xujiayi | 创建日期：2026-11-11
IC检验：条件IC、分布、累积曲线和独立日边际收益衰减。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from . import maths as m


def _ensure(base, horizon=1):
    """准备标签；Params: base；Return: None。"""
    base.cal_labels((horizon,))


def table(base, horizon=1, kind="IC"):
    """自然年及Overall的IC表。

    Params: base/horizon: 1/3/5/10/20；kind: IC/rankIC，控制pos/neg条件统计的主序列。
    Return: DataFrame；IR=日均/日标准差，未乘sqrt252；正负IC按未来收益正负分截面子样本。
    """
    _ensure(base, horizon)
    if kind not in ("IC", "rankIC"):
        raise ValueError("kind须为IC/rankIC")
    ic, ric = base.ics[horizon], base.rankics[horizon]
    rows = {}
    for y, mask in m.periods(base.dates).items():
        x, r = ic.loc[mask], ric.loc[mask]
        primary = x if kind == "IC" else r
        prefix = "" if kind == "IC" else "rank_"
        pos = base.sign_ics[horizon][f"{prefix}pos_ic"].loc[mask]
        neg = base.sign_ics[horizon][f"{prefix}neg_ic"].loc[mask]
        ratio = lambda s: s.mean() / s.std() if s.std() > 0 else np.nan
        rows[y] = {"avg_ic": x.mean(), "ic_ir": ratio(x), "avg_pos_ic": pos.mean(),
                   "avg_neg_ic": neg.mean(), "pos_icir": ratio(pos), "neg_icir": ratio(neg),
                   "avg_rankic": r.mean(), "rank_icir": ratio(r), "count": primary.count()}
    result = pd.DataFrame(rows).T
    result.attrs["conditional_series"] = kind
    return result


def plot_distribution(base, horizon=1):
    """最近5年IC/rankIC分布和统计量。

    Params: base/horizon；Return: Figure，统计量另存results['ic_distribution_stats']。
    """
    _ensure(base, horizon)
    start = base.dates[-1] - pd.DateOffset(years=5)
    data = pd.DataFrame({"IC": base.ics[horizon], "rankIC": base.rankics[horizon]}).loc[start:]
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = {}
    for name in data:
        x = data[name].dropna()
        summary[name] = {"mean": x.mean(), "std": x.std(), "skew": x.skew(), "kurtosis": x.kurt(), "count": len(x)}
        if x.nunique() > 1:
            grid = np.linspace(-1, 1, 400)
            ax.plot(grid, stats.gaussian_kde(x)(grid), label=name)
        elif len(x):
            ax.axvline(x.iloc[0], label=name)
    base.results["ic_distribution_stats"] = pd.DataFrame(summary).T
    ax.set_title(base.results["ic_distribution_stats"].round(3).to_string())
    ax.legend()
    fig.tight_layout()
    return fig


def cumulative(base, years=5, horizon=1):
    """累积IC（求和非复利）；Params: base/years:3或5/horizon；Return: DataFrame。"""
    _ensure(base, horizon)
    if years not in (1, 3, 5):
        raise ValueError("years须为1/3/5")
    start = base.dates[-1] - pd.DateOffset(years=years)
    ic, ric = base.ics[horizon].loc[start:], base.rankics[horizon].loc[start:]
    data = base.sign_ics[horizon].loc[start:].copy()
    data["ic"], data["rank_ic"] = ic, ric
    result = data[["ic", "pos_ic", "neg_ic", "rank_ic", "rank_pos_ic", "rank_neg_ic"]].cumsum()
    base.results["cumulative_ic"] = result
    return result


def decay_table(base, max_lag=20):
    """独立未来日收益的IC衰减与自相关。

    Params: base/max_lag: 1至20；Return: DataFrame，index为k，
    IC_k=corr(alpha_t,r_t+k)，另给IC_k与IC_1的时序相关及因子rank截面自相关。
    """
    _ensure(base)
    if not 1 <= max_lag <= 20:
        raise ValueError("max_lag须在1-20")
    alpha = base.alpha.loc[base.report_start:]
    first = m.cross_ic(alpha, base.pct.shift(-1).reindex(alpha.index))
    first_rank = m.cross_ic(alpha, base.pct.shift(-1).reindex(alpha.index), True)
    rows = {}
    for k in range(1, max_lag + 1):
        label = base.pct.shift(-k).reindex(alpha.index)
        ic, ric = m.cross_ic(alpha, label), m.cross_ic(alpha, label, True)
        rows[k] = {"IC": ic.mean(), "rankIC": ric.mean(),
                   "IC_autocorrelation": first.corr(ic), "rankIC_autocorrelation": first_rank.corr(ric),
                   "rank_autocorrelation": m.cross_ic(alpha, base.alpha.shift(k).reindex(alpha.index), True).mean()}
    result = pd.DataFrame(rows).T
    base.results["ic_decay"] = result
    return result


def plot_decay(base, max_lag=20):
    """边际衰减图；Params: base/max_lag；Return: Figure。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    decay_table(base, max_lag).plot(ax=ax, marker=".")
    ax.set_xlabel("Future trading day k")
    fig.tight_layout()
    return fig
