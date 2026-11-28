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
    domain = params.get("domain", getattr(base, "domain", "universe"))
    strategy = params.get("strategy", getattr(base, "strategy", "enhanced"))
    name = params.get("name", getattr(base, "name", None))
    if name is None:
        name = "market_cap" if domain == "universe" and strategy == "enhanced" else "equal"
    
    if portfolio_result is not None:
        result = portfolio_result
    elif not params and hasattr(base, "portfolio"):
        result = base.portfolio
    else:
        result = base.cal_portfolio(**params)
    daily = result.daily.loc[base.report_start:]
    if not hasattr(base, "benchmark") or name not in base.benchmark["nav"].get(domain, {}):
        base.cal_benchmark()
    benchmark_nav = base.benchmark["nav"][domain][name].reindex(daily.index)
    benchmark_ret = base.benchmark["ret"][domain][name].reindex(daily.index)
    init = daily['equity'].iloc[0] + daily['cash'].iloc[0]
    return result, daily, benchmark_nav, benchmark_ret, init

def _cal_nav(daily, init):
    cash, equity, trade_fee = daily['cash'], daily['equity'], daily['trade_fee']
    net_nav = (cash+equity)/init
    gross_nav = (cash+equity+trade_fee)/init
    return gross_nav, net_nav


def plot_nav(base, portfolio_result=None, **portfolio_params):
    """绘制策略组合扣费前后净值；报告区间首日归一为1。"""
    result, daily, benchmark_nav, _, init = _daily(base, portfolio_result, portfolio_params)
    gross_nav, net_nav = _cal_nav(daily, init)

    fig, ax = plt.subplots(figsize=(12, 5))
    gross_nav.plot(ax=ax, label="portfolio_gross")
    net_nav.plot(ax=ax, label="portfolio_net")
    benchmark = benchmark_nav.reindex(daily.index)
    benchmark = benchmark / benchmark.iloc[0]
    benchmark.plot(ax=ax, label="Benchmark")
    ax.set(title="Portfolio and benchmark NAV", xlabel="Date", ylabel="NAV")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pnl(base, portfolio_result=None, **portfolio_params):
    """绘制策略组合累计PnL和调仓换手率。"""
    _, daily, _, _, init = _daily(base, portfolio_result, portfolio_params)
    gross_nav, net_nav = _cal_nav(daily, init)
    gross_pnl = gross_nav.diff().fillna(0) * init
    net_pnl = net_nav.diff().fillna(0) * init

    fig, ax = plt.subplots(figsize=(12, 5))
    gross_pnl.cumsum().plot(ax=ax, label="active gross")
    net_pnl.cumsum().plot(ax=ax, label="active net")
    ax.set(title="Active PnL before and after costs", xlabel="Date", ylabel="PnL")
    turnover_ax = ax.twinx()
    turnover = daily.turnover
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
    _, daily, benchmark_nav, _, init = _daily(base, portfolio_result, portfolio_params)
    gross, net = _cal_nav(daily, init)
    benchmark = benchmark_nav.reindex(daily.index)
    benchmark = benchmark / benchmark.iloc[0]
    drawdown = pd.DataFrame({
        "portfolio_net": net / net.cummax() - 1,
        "portfolio_gross": gross / gross.cummax() - 1,
        "benchmark": benchmark / benchmark.cummax() - 1,
    })
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown.plot(ax=ax)
    for column in drawdown:
        ax.fill_between(drawdown.index, drawdown[column].to_numpy(), 0,
                        alpha=.1, color=ax.lines[list(drawdown.columns).index(column)].get_color())
    ax.set(title="Underwater drawdown", xlabel="Date", ylabel="Drawdown")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def performance_table(base, capitals=(1e7, 5e7, 1e8, 5e8), **portfolio_params):
    """不同初始资金下的组合绩效与成本；返回行=资金档位、列=指标。"""
    rows = {}
    for capital in capitals:
        params = dict(portfolio_params, capital=capital)
        result = base.cal_portfolio(**params)
        daily = result.daily.loc[base.report_start:]
        if daily.empty:
            continue

        init = daily['equity'].iloc[0] + daily['cash'].iloc[0]
        gross, net = _cal_nav(daily, init)
        net_return = (net/net.shift(1)-1).fillna(0)
        row = m.performance(net_return, daily.turnover)
        row.update({
            "fill_ratio": float(daily.fill_ratio.mean()),
            "total_cost": float(daily.trade_fee.sum()/init),
            "commission": float(daily.commission.sum()/init),
            "stamp_tax": float(daily.stamp_tax.sum()/init),
            "impact": float(daily.impact.sum()/init),
            "borrow": float(daily.borrow.sum()/init),
        })
        rows[capital] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "initial_capital"
    return table


def plot_performance_table(base, capitals=(1e7, 5e7, 1e8, 5e8), **portfolio_params):
    """绘制不同资金档位的绩效统计表。"""
    table = performance_table(base, capitals=capitals, **portfolio_params)
    fig, ax = plt.subplots(figsize=(max(10, len(table.columns) * 1.1), max(2.5, len(table) * .55)))
    ax.axis("off")
    display = table.copy()
    display.index = [f"{x:,.0f}" for x in display.index]
    for col in display.select_dtypes(include="number"):
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    artist = ax.table(cellText=display.values, rowLabels=display.index,
                      colLabels=display.columns, loc="center", cellLoc="center")
    artist.auto_set_font_size(False)
    artist.set_fontsize(8)
    artist.scale(1, 1.35)
    ax.set_title("Portfolio performance by initial capital")
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
    plot_nav(base, result)
    plot_pnl(base, result)
    plot_ics(base, result)
    plot_maxdrawdown(base, result)
    print(performance_table(base))
    plot_performance_table(base)
    plt.show()
