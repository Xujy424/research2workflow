"""跨行业或指数域的截面策略诊断。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __package__:
    from ..base import maths as m
    from .overall_cross_section import basic_info
else:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base import maths as m
    from v2.SingleFactorTest.perform.overall_cross_section import basic_info


def domain_table(base, domain="industry", construction="longshort", num_groups=10,
                 group_numbers=(10, 1), horizon=1):
    """按行业或指数汇总绩效、域内RankIC及策略持仓暴露。"""
    if domain not in ("industry", "index"):
        raise ValueError("domain须为'industry'或'index'")
    masks = base.domain_masks[domain]
    names = list(masks) if domain == "index" else sorted(
        {str(int(code)) for code in masks.stack().dropna().unique()}
    )
    base.cal_labels((horizon,))
    labels = base.labels[horizon]
    original_alpha = base.alpha
    overall_alpha_mean = original_alpha.mean(axis=1)
    rows, curves = {}, {}

    try:
        for name in names:
            mask = masks[name].eq(1) if domain == "index" else masks.eq(int(name))
            base.alpha = original_alpha.where(mask)
            info = basic_info(base, construction, num_groups, group_numbers, domain, name)
            curves[name] = info["excess_return"]
            ret = info["excess_return"]
            stats = m.performance(ret, info["turnover"])
            domain_alpha_mean = base.alpha.mean(axis=1)
            stats["alpha_mean_diff"] = domain_alpha_mean.sub(overall_alpha_mean).mean()

            rankic = pd.Series(m.cross_ic(base.alpha, labels, True), index=base.dates)
            stats["rankIC"] = rankic.mean()
            stats["rankIC_IR"] = rankic.mean() / rankic.std() if rankic.std() > 0 else np.nan

            benchmark = base.benchmark["ret"][domain][name].shift(-2).reindex(ret.index)
            var = benchmark.var()
            stats["benchmark_beta"] = ret.cov(benchmark) / var if var > 0 else np.nan

            codes = base.group_codes.reindex(ret.index)
            l = codes.eq(group_numbers[0]).astype(float)
            s = codes.eq(group_numbers[-1]).astype(float)
            l = l.div(l.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            s = s.div(s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            holds = l - s if construction == "longshort" else l if construction == "long" else -s
            domain_mask = mask.reindex(ret.index).fillna(False)
            domain_holds = holds.where(domain_mask, 0)
            exposure = domain_holds.sum(axis=1)
            stats["domain_holding_weight"] = exposure.mean()
            stock_ret = base.pct.shift(-2).reindex(ret.index)
            holding_ret = pd.Series(m.weighted_returns(domain_holds, stock_ret), index=ret.index)
            stats["domain_holding_return"] = m.performance(holding_ret)["annual_return"]
            rows[name] = stats
    finally:
        base.alpha = original_alpha

    base.section_excess_returns = pd.DataFrame(curves)
    return pd.DataFrame.from_dict(rows, orient="index")


def _barra_statistics(base, construction, num_groups, group_numbers):
    base.cal_groups(num_groups, "universe", "equal", horizon=1)
    codes = base.group_codes
    l = codes.eq(group_numbers[0]).astype(float)
    s = codes.eq(group_numbers[-1]).astype(float)
    l = l.div(l.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    s = s.div(s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    holds = l - s if construction == "longshort" else l if construction == "long" else -s
    barra = base.fields["barra"]
    styles = list(barra)
    returns = basic_info(base, construction, num_groups, group_numbers, "universe", "equal")["excess_return"]

    correlations, exposures = {}, {}
    for style in styles:
        values = barra[style]
        correlations[style] = np.nanmean(m.cross_ic(base.alpha, values))
        exposures[style] = (holds * values).sum(axis=1).mean()

    barra_returns = getattr(base, "barra_returns", None)
    if barra_returns is None:
        barra_returns = base.cal_barra_ret()
    barra_returns = barra_returns.reindex(returns.index)
    pair = pd.concat([returns.rename("excess_return"), barra_returns.loc[:, styles]], axis=1).dropna()
    regression_betas = pd.Series(np.nan, index=styles)
    if len(pair) > len(styles) + 1:
        design = np.column_stack([np.ones(len(pair)), pair[styles].to_numpy()])
        coefficients, _, rank, _ = np.linalg.lstsq(design, pair["excess_return"].to_numpy(), rcond=None)
        if rank == design.shape[1]:
            regression_betas[:] = coefficients[1:]

    signal_table = pd.DataFrame({
        "alpha_barra_corr": pd.Series(correlations),
        "holding_exposure": pd.Series(exposures),
    })
    return_table = pd.DataFrame({
        "excess_return_beta": regression_betas,
        "barra_return": pair[styles].add(1).prod(min_count=1).sub(1),
        "beta_return_contribution": pair[styles].mul(regression_betas).sum(min_count=1),
    })
    base.barra_return_table = return_table
    return signal_table, return_table


def plot_barra_signal_exposure(base, construction="longshort", num_groups=10, group_numbers=(10, 1)):
    """绘制整体Signal beta和持仓Barra风格敞口。"""
    result = _barra_statistics(base, construction, num_groups, group_numbers)[0]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(result))
    width = 0.36
    ax.bar(x - width / 2, result["alpha_barra_corr"], width,
           label="Signal beta", color="tab:blue")
    ax.bar(x + width / 2, result["holding_exposure"], width,
           label="Holding exposure", color="tab:orange")
    ax.set_xticks(x, result.index, rotation=45, ha="right")
    limit = np.nanmax(np.abs(result.to_numpy()))
    ax.set_ylim((-limit * 1.15, limit * 1.15) if np.isfinite(limit) and limit else (-1, 1))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Overall signal beta and holding exposure", xlabel="Barra style", ylabel="Value")
    ax.grid(axis="y", color="gray", alpha=0.2, linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    base.barra_section_table = result
    return fig


def plot_barra_return_beta(base, construction="longshort", num_groups=10, group_numbers=(10, 1)):
    """绘制超额收益回归 beta 和 Barra 风格累计收益。"""
    result = _barra_statistics(base, construction, num_groups, group_numbers)[1]
    fig, ax = plt.subplots(figsize=(13, 6))
    right = ax.twinx()
    contribution_axis = ax.twinx()
    contribution_axis.spines["right"].set_position(("outward", 65))
    x = np.arange(len(result))
    width = 0.25
    ax.bar(x - width, result["excess_return_beta"], width,
           label="Excess return beta", color="tab:red", alpha=.75)
    right.bar(x, result["barra_return"], width,
              label="Cumulative Barra return", color="tab:blue", alpha=.7)
    contribution_axis.bar(x + width, result["beta_return_contribution"], width,
                          label="Sum(beta x daily style return)", color="tab:green", alpha=.75)
    ax.set_xticks(x, result.index, rotation=45, ha="right")
    for axis, values in ((ax, result["excess_return_beta"]),
                         (right, result["barra_return"]),
                         (contribution_axis, result["beta_return_contribution"])):
        limit = np.nanmax(np.abs(np.asarray(values)))
        axis.set_ylim((-limit * 1.15, limit * 1.15) if np.isfinite(limit) and limit else (-1, 1))
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Excess return beta and Barra return contribution", xlabel="Barra style",
           ylabel="Excess return beta")
    right.set_ylabel("Cumulative Barra return", color="tab:blue")
    right.tick_params(axis="y", colors="tab:blue")
    contribution_axis.set_ylabel("Sum(beta x daily style return)", color="tab:green")
    contribution_axis.tick_params(axis="y", colors="tab:green")
    ax.grid(axis="y", color="gray", alpha=.2, linewidth=.7)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    handles3, labels3 = contribution_axis.get_legend_handles_labels()
    ax.legend(handles + handles2 + handles3, labels + labels2 + labels3, loc="best")
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
    domain = "index"
    table = domain_table(base, domain=domain)
    print(f"{domain} domain summary:")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))

    fig = plot_barra_signal_exposure(base)
    print("Signal beta and holding exposure:")
    print(base.barra_section_table.to_string(float_format=lambda value: f"{value:.4f}"))
    fig = plot_barra_return_beta(base)
    print("Regression beta, cumulative Barra return, and contribution:")
    print(base.barra_return_table.to_string(float_format=lambda value: f"{value:.6f}"))
    plt.show()
