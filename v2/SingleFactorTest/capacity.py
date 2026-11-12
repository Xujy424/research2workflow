"""作者：xujiayi | 创建日期：2026-11-11
容量测试：固定信号和约束，对不同资金规模运行同一成交模型。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import maths as m


def table(base, capitals=(1e7, 5e7, 1e8, 5e8), min_fill=.8,
          max_tracking_gap=.2, **portfolio_params):
    """容量表与全部逐规模明细。

    Params: base/capitals: 元；min_fill/max_tracking_gap: 可配置准入阈值；
        portfolio_params: 同cal_portfolio，不得重复指定capital。
    Return: 资金规模×指标表；base.capacity_results保留每档成交结果。
    成交率采用累计成交金额/累计请求金额，另报告5%分位日成交率；不裁剪收益。
    """
    if "capital" in portfolio_params:
        raise ValueError("通过capitals指定规模")
    capitals = tuple(capitals)
    if not capitals or len(set(capitals)) != len(capitals):
        raise ValueError("资金规模不可空或重复")
    rows, results = {}, {}
    previous = getattr(base, "portfolio", None)
    try:
        for capital in capitals:
            p = base.cal_portfolio(capital=capital, **portfolio_params)
            results[capital] = p
            daily = p.daily.loc[base.report_start:]
            d = p.diagnostics.reindex(daily.index)
            row = m.performance(daily.net_return, daily.turnover)
            requested = daily.requested_value.sum()
            row["fill_ratio"] = daily.traded_value.sum() / requested if requested > 0 else np.nan
            row["fill_ratio_p05"] = daily.fill_ratio.quantile(.05)
            row["tracking_gap"] = d.target_gap.mean()
            row["gross_exposure"] = d.gross_exposure.mean()
            row["net_exposure"] = d.net_exposure.mean()
            row["cash_ratio"] = (daily.cash / daily.equity).mean()
            row["st_remaining_max"] = d.st_remaining_value.max()
            row["stock_cap_excess_max"] = d.stock_cap_excess.max()
            for cost in ("commission", "stamp_tax", "impact", "borrow"):
                row[f"annual_{cost}"] = daily[cost].mean() * 252
            row["capacity_pass"] = bool(row.fill_ratio >= min_fill and row.tracking_gap <= max_tracking_gap)
            rows[capital] = row
    finally:
        if previous is not None:
            base.portfolio = previous
        else:
            base.__dict__.pop("portfolio", None)
    base.capacity_results = results
    result = pd.DataFrame(rows).T
    result.index.name = "capital"
    base.results["capacity_table"] = result
    return result


def plot(base, **capacity_params):
    """规模净值/成交率/冲击成本图；Params: base/同table参数；Return: Figure。"""
    if capacity_params or not hasattr(base, "capacity_results"):
        table(base, **capacity_params)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for capital, p in base.capacity_results.items():
        data = p.daily.loc[base.report_start:]
        data.nav.plot(ax=axes[0], label=f"{capital:,.0f}")
        data.fill_ratio.rolling(20, min_periods=1).mean().plot(ax=axes[1], label=f"{capital:,.0f}")
        data.impact.cumsum().plot(ax=axes[2], label=f"{capital:,.0f}")
    for ax, name in zip(axes, ("Net NAV", "20D fill ratio", "Cumulative impact")):
        ax.set_ylabel(name)
        ax.legend()
    fig.tight_layout()
    return fig
