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


def plot_barra_exposure(base, construction="longshort", num_groups=10, group_numbers=(10, 1)):
    """用全体股票绘制Alpha-Barra相关、持仓风格敞口及超额收益风格Beta。"""
    base.cal_groups(num_groups, "universe", "equal", horizon=1)
    codes = base.group_codes
    l = codes.eq(group_numbers[0]).astype(float)
    s = codes.eq(group_numbers[-1]).astype(float)
    l = l.div(l.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    s = s.div(s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    holds = l - s if construction == "longshort" else l if construction == "long" else -s
    styles = base.fields["barra"]
    returns = basic_info(base, construction, num_groups, group_numbers, "universe", "equal")["excess_return"]

    correlations, exposures = {}, {}
    for style, values in styles.items():
        correlations[style] = pd.Series(
            m.cross_ic(base.alpha, values), index=base.dates
        ).mean()
        exposures[style] = (holds * values).sum(axis=1).mean()

    barra_returns = getattr(base, "barra_returns", None)
    if barra_returns is None:
        barra_returns = base.cal_barra_ret()
    barra_returns = barra_returns.reindex(returns.index)
    regression_betas = {}
    for style in styles:
        pair = pd.concat([returns, barra_returns[style]], axis=1).dropna()
        if len(pair) < 2 or pair.iloc[:, 1].var() == 0:
            regression_betas[style] = np.nan
        else:
            regression_betas[style] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()

    result = pd.DataFrame({
        "alpha_barra_corr": pd.Series(correlations),
        "holding_exposure": pd.Series(exposures),
        "excess_return_beta": pd.Series(regression_betas),
    })
    fig, ax = plt.subplots(figsize=(13, 6))
    right = ax.twinx()
    x = np.arange(len(result))
    width = 0.24
    ax.bar(x - width, result["alpha_barra_corr"], width, label="Signal beta")
    ax.bar(x, result["holding_exposure"], width, label="Holding exposure")
    right.bar(x + width, result["excess_return_beta"], width,
              color="tab:red", alpha=0.7, label="Return beta")
    ax.set_xticks(x, result.index, rotation=45, ha="right")
    left_max = np.nanmax(np.abs(result[["alpha_barra_corr", "holding_exposure"]].to_numpy()))
    right_max = np.nanmax(np.abs(result["excess_return_beta"].to_numpy()))
    ax.set_ylim(-left_max * 1.15, left_max * 1.15)
    right.set_ylim(-right_max * 1.15, right_max * 1.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Overall Barra style exposures", xlabel="Barra style",
           ylabel="Signal beta / holding exposure")
    right.set_ylabel("Return beta")
    ax.grid(axis="y", color="gray", alpha=0.2, linewidth=0.7)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    base.barra_section_table = result
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

    fig = plot_barra_exposure(base)
    print("Overall Barra style table:")
    print(base.barra_section_table.to_string(float_format=lambda value: f"{value:.4f}"))
    plt.show()
