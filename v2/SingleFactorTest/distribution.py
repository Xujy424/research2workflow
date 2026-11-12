"""作者：xujiayi | 创建日期：2026-11-11
因子值分布、分组收益及拥挤度代理指标。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import maths as m


def value_table(base, processed=True):
    """因子分布统计；Params: base/processed是否处理后；Return: 年份×统计量。

    max/min为样本极值，*_avg为每日截面统计量的时序均值，mean/std为合并样本统计。
    """
    if not hasattr(base, "alpha"):
        base.process_alpha()
    x = base.alpha if processed else base.raw_alpha
    rows = {}
    for year, mask in m.periods(x.index).items():
        v = x.loc[mask]
        pooled = v.to_numpy().ravel()
        pooled = pooled[np.isfinite(pooled)]
        rows[year] = {"max": pooled.max() if len(pooled) else np.nan,
                      "min": pooled.min() if len(pooled) else np.nan,
                      "max_avg": v.max(axis=1).mean(), "min_avg": v.min(axis=1).mean(),
                      "skew_avg": v.skew(axis=1).mean(), "kurt_avg": v.kurt(axis=1).mean(),
                      "mean": pooled.mean() if len(pooled) else np.nan,
                      "std": pooled.std() if len(pooled) else np.nan}
    return pd.DataFrame(rows).T


def plot_values(base, processed=True, bins=60):
    """各年和Overall直方图；Params: base/processed/bins；Return: Figure。"""
    if not hasattr(base, "alpha"):
        base.process_alpha()
    x = base.alpha if processed else base.raw_alpha
    periods = m.periods(x.index)
    fig, axes = plt.subplots(int(np.ceil(len(periods) / 3)), 3, figsize=(13, 9), squeeze=False)
    for ax, (year, mask) in zip(axes.flat, periods.items()):
        values = x.loc[mask].to_numpy().ravel()
        values = values[np.isfinite(values)]
        if len(values):
            ax.hist(values, bins=bins, density=True)
        ax.set_title(year)
    for ax in list(axes.flat)[len(periods):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def group_table(base, domain="universe"):
    """分组诊断；Params: base/domain；Return: G10→G1、Average指标表。

    BP=组持仓的账面市值比，不是基点收益；IC_IR为组内信号预测能力。
    拥挤度=(换手/平均组换手+波动/平均组波动+|beta|/|平均组beta|)/3，仅为代理指标。
    """
    returns = base.cal_groups(domain).loc[base.report_start:]
    benchmark = base.cal_benchmark(domain=domain).reindex(returns.index)
    base.cal_labels((1,))
    rows = {}
    mean_weight = m.normalize(sum(base.group_weights.values()))
    for name in returns.columns:
        w = mean_weight if name == "Average" else base.group_weights[int(name[1:])]
        turnover = w.diff().abs().sum(axis=1).div(2).shift(1)
        row = m.performance(returns[name], turnover)
        variance = benchmark.var()
        row["beta"] = returns[name].cov(benchmark) / variance if variance > 0 else np.nan
        row["BP"] = m.weighted_returns(w, base.fields["bp_lf"]).loc[base.report_start:].mean() if "bp_lf" in base.fields else np.nan
        mask = w > 0
        ic = m.cross_ic(base.alpha.where(mask), base.labels[1]).loc[base.report_start:]
        row["IC_IR"] = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        rows[name] = row
    result = pd.DataFrame(rows).T
    ratios = result[["turnover", "annual_vol", "beta"]].abs().div(
        result.loc["Average", ["turnover", "annual_vol", "beta"]].abs().replace(0, np.nan), axis=1)
    result["crowding"] = ratios.mean(axis=1, skipna=False)
    base.results["group_table"] = result
    return result.reindex([f"G{g}" for g in range(base.groups, 0, -1)] + ["Average"])


def plot_groups(base, domain="universe"):
    """分组净值；Params: base/domain；Return: Figure，不绘制平均组。"""
    r = base.cal_groups(domain).drop(columns="Average").loc[base.report_start:]
    fig, ax = plt.subplots(figsize=(12, 6))
    m.nav(r).plot(ax=ax)
    fig.tight_layout()
    return fig
