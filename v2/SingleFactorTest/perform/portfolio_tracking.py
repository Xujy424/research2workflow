"""投资组合净值、PnL、回撤和绩效统计。"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from ..base import maths as m
else:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base import maths as m


def _daily(base, portfolio_result, params):
    key = tuple(sorted((name, repr(value)) for name, value in params.items()))
    if portfolio_result is not None:
        result = portfolio_result
    elif getattr(base, "_portfolio_tracking_key", None) == key:
        result = base._portfolio_tracking_result
    elif not params and hasattr(base, "portfolio"):
        result = base.portfolio
        base._portfolio_tracking_key, base._portfolio_tracking_result = key, result
    else:
        result = base.cal_portfolio(**params)
        base._portfolio_tracking_key, base._portfolio_tracking_result = key, result
    daily = result.daily.loc[base.report_start:]
    domain = params.get("domain", "universe")
    name = params.get("name") or ("market_cap" if domain == "universe" and
                                   params.get("strategy", "enhanced") == "enhanced" else "equal")
    if not hasattr(base, "benchmark") or name not in base.benchmark["nav"].get(domain, {}):
        base.cal_benchmark()
    benchmark_nav = base.benchmark["nav"][domain][name].reindex(daily.index)
    benchmark_ret = base.benchmark["ret"][domain][name].reindex(daily.index)
    capital = result.daily.equity.iloc[0] - result.daily.pnl.iloc[0]
    return result, daily, benchmark_nav, benchmark_ret, capital


def plot_nav(base, portfolio_result=None, **portfolio_params):
    """绘制策略组合扣费前后净值；报告区间首日归一为1。"""
    _, daily, benchmark_nav, _, _ = _daily(base, portfolio_result, portfolio_params)
    fig, ax = plt.subplots(figsize=(12, 5))
    gross_nav = pd.Series(m.nav(daily.gross_return.replace([np.inf, -np.inf], np.nan).fillna(0)), index=daily.index)
    net_nav = daily.nav
    gross_nav = gross_nav / gross_nav.iloc[0]
    net_nav = net_nav / net_nav.iloc[0]
    gross_nav.plot(ax=ax, label="Strategy portfolio (before costs)")
    net_nav.plot(ax=ax, label="Strategy portfolio (after costs)")
    benchmark = benchmark_nav.reindex(daily.index)
    benchmark = benchmark / benchmark.iloc[0]
    benchmark.plot(ax=ax, label="Benchmark")
    ax.set(title="Strategy portfolio and benchmark NAV", xlabel="Date", ylabel="NAV")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pnl(base, portfolio_result=None, **portfolio_params):
    """绘制策略组合累计PnL和调仓换手率。"""
    _, daily, _, _, capital = _daily(base, portfolio_result, portfolio_params)
    fig, ax = plt.subplots(figsize=(12, 5))
    previous_equity = daily.equity.shift(1).fillna(capital)
    gross_pnl = (daily.gross_return * previous_equity).cumsum()
    net_pnl = daily.pnl.cumsum()
    gross_pnl.plot(ax=ax, label="Cumulative strategy PnL (before costs)")
    net_pnl.plot(ax=ax, label="Cumulative strategy PnL (after costs)")
    ax.set(title="Cumulative strategy portfolio PnL", xlabel="Date", ylabel="PnL")
    turnover_ax = ax.twinx()
    turnover = daily.loc[daily.rebalance.fillna(False), "turnover"]
    turnover_ax.plot(turnover.index, turnover, color="gray", alpha=.45, linewidth=.9,
                     label="Rebalance turnover")
    turnover_ax.set_ylabel("Rebalance turnover")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = turnover_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    return fig


def plot_ics(base, portfolio_result=None, horizon=1, **portfolio_params):
    """绘制调仓日RankIC柱图和累计RankIC。"""
    result, daily, _, _, _ = _daily(base, portfolio_result, portfolio_params)
    base.cal_ics((horizon,))
    rankic = pd.Series(base.rankics[horizon], index=base.dates).shift(1).reindex(daily.index)
    rankic = rankic.where(daily.rebalance.fillna(False)).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = np.where(rankic >= 0, "tab:red", "tab:green")
    ax.bar(rankic.index, rankic, width=3, color=colors, alpha=.65, label="Rebalance RankIC")
    if len(rankic):
        ax.axhline(rankic.mean(), color="tab:orange", linestyle="--", linewidth=1.2,
                   label=f"Mean RankIC ({rankic.mean():.4f})")
    ax.axhline(0, color="gray", linewidth=.8)
    right = ax.twinx()
    cumulative = rankic.cumsum()
    right.plot(cumulative.index, cumulative, color="tab:blue", alpha=.6, label="Cumulative RankIC")
    right.fill_between(cumulative.index, 0, cumulative.to_numpy(), color="tab:blue", alpha=.1)
    ax.set(title=f"RankIC at rebalance dates and cumulative RankIC (horizon={horizon})",
           xlabel="Date", ylabel="Rebalance RankIC")
    right.set_ylabel("Cumulative RankIC")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    ax.grid(True, alpha=.2)
    fig.tight_layout()
    return fig


def plot_maxdrawdown(base, portfolio_result=None, **portfolio_params):
    """绘制策略组合NAV对应的回撤。"""
    _, daily, benchmark_nav, _, _ = _daily(base, portfolio_result, portfolio_params)
    nav = daily.nav / daily.nav.iloc[0]
    benchmark = benchmark_nav.reindex(daily.index)
    benchmark = benchmark / benchmark.iloc[0]
    drawdown = pd.DataFrame({
        "Strategy portfolio": nav / nav.cummax() - 1,
        "Benchmark": benchmark / benchmark.cummax() - 1,
    })
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown.plot(ax=ax)
    for column in drawdown:
        ax.fill_between(drawdown.index, drawdown[column].to_numpy(), 0,
                        alpha=.1, color=ax.lines[list(drawdown.columns).index(column)].get_color())
    ax.set(title="Portfolio Drawdown", xlabel="Date", ylabel="Drawdown")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
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
    result = base.cal_portfolio()
    result.daily['nav'].plot()
    base.benchmark['nav']['universe']['market_cap'].plot()
    # plot_nav(base, result)
    # plot_pnl(base, result)
    # plot_ics(base, result)
    # plot_maxdrawdown(base, result)
    plt.show()
