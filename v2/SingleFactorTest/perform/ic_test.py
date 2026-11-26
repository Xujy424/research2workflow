"""作者：xujiayi | 创建日期：2026-11-11
IC检验：条件IC、分布、累积曲线和独立日边际收益衰减。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __package__:
    from ..base import maths as m
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base import maths as m


def table(base, horizon=1):
    """自然年及Overall的IC表。
    Params: base/horizon: 1/3/5/10/20；
    Return: DataFrame；IR=日均/日标准差，未乘sqrt252；正负IC按未来收益正负分截面子样本。
    """
    base.cal_labels((horizon,))
    base.cal_ics((horizon,))
    ic = pd.Series(base.ics[horizon], index=base.dates)
    ric = pd.Series(base.rankics[horizon], index=base.dates)
    conditional = {
        key: pd.Series(value.to_numpy(), index=base.dates)
        for key, value in base.sign_ics[horizon].items()
    }
    rows = {}
    for y, mask in m.periods(base.dates).items():
        x, r = ic.loc[mask], ric.loc[mask]
        pos, neg = conditional["pos_ic"].loc[mask], conditional["neg_ic"].loc[mask]
        rank_pos, rank_neg = conditional["rank_pos_ic"].loc[mask], conditional["rank_neg_ic"].loc[mask]
        ratio = lambda s: s.mean() / s.std() if s.std() > 0 else np.nan
        rows[y] = {
            "avg_ic": x.mean(), "ic_ir": ratio(x), 
            "avg_pos_ic": pos.mean(), "avg_neg_ic": neg.mean(), 
            "pos_icir": ratio(pos), "neg_icir": ratio(neg),
            "avg_rankic": r.mean(), "rank_icir": ratio(r),
            "avg_rank_pos_ic": rank_pos.mean(), "avg_rank_neg_ic": rank_neg.mean(),
            "rank_pos_icir": ratio(rank_pos), "rank_neg_icir": ratio(rank_neg),
            "ic_count": x.count(), "rankic_count": r.count()}
    result = pd.DataFrame(rows).T
    return result


def plot_distribution(base, horizon=1):
    """最近5年IC/rankIC分布和统计量。
    Params: base/horizon；Return: Figure，统计量另存results['ic_distribution_stats']。
    """
    base.cal_labels((horizon,))
    base.cal_ics((horizon,))
    start = base.dates[-1] - pd.DateOffset(years=5)
    data = pd.DataFrame({
        "IC": pd.Series(base.ics[horizon], index=base.dates),
        "rankIC": pd.Series(base.rankics[horizon], index=base.dates),
    }).loc[start:]
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = {}
    for name in data:
        x = data[name].dropna()
        stats_row = {"mean": x.mean(), "std": x.std(), "skew": x.skew(), "kurtosis": x.kurt(), "count": len(x)}
        summary[name] = stats_row
        label = (f"{name} (mean={stats_row['mean']:.3f}, std={stats_row['std']:.3f}, "
                 f"skew={stats_row['skew']:.3f}, kurtosis={stats_row['kurtosis']:.3f})")
        if len(x):
            ax.hist(x, bins=40, density=True, alpha=0.2, label=f"{name} histogram")
            ax.axvline(stats_row["mean"], linestyle="--", linewidth=1.5, label=f"{name} mean={stats_row['mean']:.3f}")
            if x.nunique() > 1:
                grid = np.linspace(x.min(), x.max(), 400)
                ax.plot(grid, stats.gaussian_kde(x)(grid), label=label)
    ax.set_title(f"IC / RankIC distribution (last 5 years, horizon={horizon})")
    ax.set_xlabel("IC value")
    ax.set_ylabel("Density")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ic_cumulative(base, years=3, horizon=1):
    """绘制累计IC（求和非复利），返回Figure。"""
    base.cal_labels((horizon,))
    base.cal_ics((horizon,))
    if years not in (1, 3, 5):
        raise ValueError("years须为1/3/5")
    start = base.dates[-1] - pd.DateOffset(years=years)
    ic = pd.Series(base.ics[horizon], index=base.dates).loc[start:]
    ric = pd.Series(base.rankics[horizon], index=base.dates).loc[start:]
    data = base.sign_ics[horizon].set_axis(base.dates).loc[start:]
    data["ic"], data["rank_ic"] = ic, ric
    result = data[["ic", "pos_ic", "neg_ic", "rank_ic", "rank_pos_ic", "rank_neg_ic"]].cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    result.plot(ax=ax)
    ax.set(title=f"Cumulative IC / RankIC ({years}y, horizon={horizon})",
           xlabel="Date", ylabel="Cumulative IC")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    return fig


def plot_decay(base, max_lag=20):
    """独立未来日收益的IC衰减与自相关。
    Params: base/max_lag: 1至20；Return: DataFrame，index为k，
    IC_k=corr(alpha_t,r_t+k)，另给IC_k与IC_1的时序相关及因子rank截面自相关。
    """
    if not 1 <= max_lag <= 20:
        raise ValueError("max_lag须在1-20")
    alpha = base.alpha.loc[base.report_start:]
    first = pd.Series(m.cross_ic(alpha, base.pct.shift(-2).reindex(alpha.index)), index=alpha.index)
    first_rank = pd.Series(m.cross_ic(alpha, base.pct.shift(-2).reindex(alpha.index), True), index=alpha.index)
    rows = {}
    for k in range(1, max_lag + 1):
        label = base.pct.shift(-2-k).reindex(alpha.index)
        ic = pd.Series(m.cross_ic(alpha, label), index=alpha.index)
        ric = pd.Series(m.cross_ic(alpha, label, True), index=alpha.index)
        factor_rank_ic = pd.Series(
            m.cross_ic(alpha, base.alpha.shift(k).reindex(alpha.index), True), index=alpha.index
        )
        rows[k] = {"IC": ic.mean(), "rankIC": ric.mean(),
                   "IC_autocorrelation": first.corr(ic), "rankIC_autocorrelation": first_rank.corr(ric),
                   "rank_autocorrelation": factor_rank_ic.mean()}
    result = pd.DataFrame(rows).T

    fig, ax = plt.subplots(figsize=(11, 5))
    right = ax.twinx()
    result.drop(columns="rank_autocorrelation").plot(ax=ax, marker=".")
    result["rank_autocorrelation"].plot(
        ax=right, marker=".", linestyle="--", color="black", label="rank_autocorrelation"
    )
    ax.set_ylabel("IC / IC autocorrelation")
    right.set_ylabel("Factor rank autocorrelation")
    ic_curve = result["IC"].dropna()
    if len(ic_curve) > 1 and ic_curve.iloc[0] != 0:
        target = ic_curve.iloc[0] / 2
        crossed = ic_curve[ic_curve.index > ic_curve.index[0]]
        crossed = crossed[(crossed - target) * ic_curve.iloc[0] <= 0]
        if len(crossed):
            half_life = crossed.index[0]
            ax.axhline(target, color="gray", linestyle=":", alpha=0.6)
            ax.axvline(half_life, color="black", linestyle="--", alpha=0.7,
                       label=f"IC half-life = {half_life}")
    left_values = result.drop(columns="rank_autocorrelation").to_numpy(dtype=float)
    right_values = result["rank_autocorrelation"].to_numpy(dtype=float)
    left_limit = np.nanmax(np.abs(left_values))
    right_limit = np.nanmax(np.abs(right_values))
    ax.set_ylim(-left_limit * 1.1, left_limit * 1.1)
    right.set_ylim(-right_limit * 1.1, right_limit * 1.1)
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.7)
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    ax.set_xlabel("Future trading day k")
    fig.tight_layout()
    return result, fig


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base.create_base import create_base

    base = create_base()
    horizon = 1
    print("IC summary:")
    print(table(base, horizon).to_string(float_format=lambda value: f"{value:.4f}"))
    figures = [
        plot_distribution(base, horizon),
        plot_ic_cumulative(base, years=3, horizon=horizon),
    ]
    decay, decay_fig = plot_decay(base)
    print("IC decay:")
    print(decay.to_string(float_format=lambda value: f"{value:.4f}"))
    plt.show()


