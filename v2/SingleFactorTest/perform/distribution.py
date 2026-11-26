"""作者：xujiayi | 创建日期：2026-11-11
因子值分布、分组收益及拥挤度代理指标。
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
        rows[year] = {
            "max": pooled.max() if len(pooled) else np.nan,
            "min": pooled.min() if len(pooled) else np.nan,
            "max_avg": v.max(axis=1).mean(), 
            "min_avg": v.min(axis=1).mean(),
            "skew_avg": v.skew(axis=1).mean(), 
            "kurt_avg": v.kurt(axis=1).mean(),
            "mean": pooled.mean() if len(pooled) else np.nan,
            "std": pooled.std() if len(pooled) else np.nan
        }
    return pd.DataFrame(rows).T


def plot_values(base, processed=True, bins=100, period=None):
    """画因子值分布；period=None时画Overall直方图及逐年PDF，否则画指定年份。"""
    if not hasattr(base, "alpha"):
        base.process_alpha()
    x = base.alpha if processed else base.raw_alpha
    periods = m.periods(x.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    period_key = str(period) if period is not None else None
    if period_key is not None and period_key not in periods:
        raise ValueError(f"period={period} 不在数据年份中，可选年份：{list(periods)}")

    if period_key is None:
        hist_values, selected = x.to_numpy().ravel(), periods.items()
        title = "Overall histogram and yearly PDFs"
    else:
        hist_values = x.loc[periods[period_key]].to_numpy().ravel()
        selected = [(period_key, periods[period_key])]
        title = period_key

    hist_values = hist_values[np.isfinite(hist_values)]
    if hist_values.size:
        ax.hist(hist_values, bins=bins, density=True, alpha=0.3,
                label=f"{period_key or 'Overall'} histogram")
    for year, mask in selected:
        values = x.loc[mask].to_numpy().ravel()
        values = values[np.isfinite(values)]
        if values.size > 1 and np.ptp(values) > 0:
            grid = np.linspace(values.min(), values.max(), 400)
            ax.plot(grid, stats.gaussian_kde(values)(grid), label=f"{year} PDF")

    ax.set_title(title)
    ax.set_xlabel("Factor value")
    ax.set_ylabel("Density")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def group_table(base, horizon=1, domain="universe", name=None, num_groups=10):
    """返回各组及平均组的年化绩效、BP、IC_IR、beta和拥挤度。"""
    returns = base.cal_groups(
        num_groups=num_groups, domain=domain, name=name, horizon=horizon
    )[horizon].loc[base.report_start:]
    base.cal_labels((horizon,))
    benchmark = base.cal_benchmark()["ret"][domain][name or "equal"].reindex(returns.index)
    codes = base.group_codes.loc[returns.index]
    rows = {}

    for group in [f"G{i}" for i in range(num_groups, 0, -1)]+["Average"]:
        ret = returns[group]
        group_id = int(group[1:]) if group != "Average" else None
        membership = codes.eq(group_id) if group_id is not None else codes.notna()
        members = membership.astype(float)
        changes = members.diff().abs().sum(axis=1)
        group_size = members.sum(axis=1).add(members.shift(1).sum(axis=1))
        turnover = changes.div(group_size.replace(0, np.nan))
        row = m.performance(ret, turnover)
        var = benchmark.var()
        row["beta"] = ret.cov(benchmark) / var if var > 0 else np.nan
        if "pb_lf" in base.fields:
            pb = base.fields["pb_lf"].reindex(returns.index)
            valid = membership & pb.gt(0)
            weights = valid.astype(float).div(valid.sum(axis=1).replace(0, np.nan), axis=0)
            bp = m.weighted_returns(weights, 1 / pb.where(valid))
            row["BP"] = bp.mean()
        else:
            row["BP"] = np.nan
        mask = membership.loc[returns.index]
        ic = m.cross_ic(
            base.alpha.loc[returns.index].where(mask),
            base.labels[horizon].loc[returns.index],
        )
        ic = np.asarray(ic, dtype=float)
        ic_mean = np.nanmean(ic) if np.isfinite(ic).any() else np.nan
        ic_std = np.nanstd(ic, ddof=1) if np.isfinite(ic).sum() > 1 else np.nan
        row["IC_IR"] = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
        rows[group] = row

    result = pd.DataFrame.from_dict(rows, orient="index")
    cols = ["turnover", "annual_vol", "beta"]
    result["crowding"] = result[cols].abs().div(
        result.loc["Average", cols].abs().replace(0, np.nan)
    ).mean(axis=1)
    return result


def plot_groups(base, horizon=1, domain="universe", name=None, num_groups=10):
    """绘制各组相对平均组的累计收益。"""
    r = base.cal_groups(
        num_groups=num_groups, domain=domain, name=name, horizon=horizon
    )[horizon].loc[base.report_start:]
    r = r.drop(columns="Average").sub(r["Average"], axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    nav = pd.DataFrame(m.nav(r), index=r.index, columns=r.columns).sub(1)
    nav.plot(ax=ax)
    ax.set(title="Cumulative group return relative to Average", xlabel="Date", ylabel="Cumulative excess return")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    return fig



if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base.create_base import create_base

    base = create_base()
    table1 = value_table(base, processed=True)
    fig = plot_values(base, processed=True, period='2025')
    print(table1)
    table2 = group_table(base)
    print(table2)
    fig = plot_groups(base)
    plt.show()

    
