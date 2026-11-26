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
    result = portfolio_result if portfolio_result is not None else base.cal_portfolio(**params)
    daily = result.daily.loc[base.report_start:]
    if not hasattr(base, "benchmark"):
        base.cal_benchmark()
    domain = params.get("domain", "universe")
    name = params.get("name") or "equal"
    if domain == "universe" and params.get("strategy", "enhanced") == "enhanced":
        weight = getattr(base, "benchmark_weight", None)
        if weight is None:
            mv = base.fields["market_cap"]
            valid = base.pct.notna() & mv.gt(0) & np.isfinite(mv)
            weight = mv.where(valid, 0)
            weight = weight.div(weight.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0)
        prior = weight.shift(1).fillna(0)
        missing = ((prior != 0) & base.pct.isna()).any(axis=1)
        benchmark = (prior * base.pct.fillna(0)).sum(axis=1).where(~missing)
    else:
        benchmark = base.benchmark["ret"][domain][name].reindex(daily.index)
    benchmark = benchmark.reindex(daily.index)
    capital = result.daily.equity.iloc[0] - result.daily.pnl.iloc[0]
    return daily, benchmark, result.daily, capital


def _curves(daily, benchmark, full_daily, capital):
    previous_equity = full_daily.equity.shift(1).fillna(capital).reindex(daily.index)
    costs = daily[["commission", "stamp_tax", "impact", "borrow"]].sum(axis=1) * previous_equity
    start_equity = previous_equity.iloc[0]
    pnl = pd.DataFrame({
        "Portfolio (before costs)": daily.pnl + costs,
        "Benchmark": benchmark * benchmark.cumprod().shift(1).fillna(1) * start_equity,
        "Portfolio (after costs)": daily.pnl,
    }).cumsum()
    nav = pnl.div(start_equity).add(1)
    return nav, pnl


def performance_table(base, portfolio_result=None, capitals=None, **portfolio_params):
    """统计绩效和容量指标；行按初始资金区分，列为各项指标。"""
    if portfolio_result is not None:
        results = portfolio_result if isinstance(portfolio_result, (list, tuple)) else [portfolio_result]
    elif capitals is not None:
        results = [base.cal_portfolio(capital=capital, **portfolio_params) for capital in capitals]
    else:
        results = [base.cal_portfolio(**portfolio_params)]

    rows = {}
    for result in results:
        daily = result.daily.loc[base.report_start:]
        stats = m.performance(daily.net_return, daily.turnover)
        for col in ("commission", "stamp_tax", "impact", "borrow"):
            if col in daily:
                stats[f"cumulative_{col}"] = daily[col].sum()
                stats[f"annual_{col}"] = daily[col].mean() * 252
        cost_cols = [c for c in ("commission", "stamp_tax", "impact", "borrow") if c in daily]
        if cost_cols:
            stats["cumulative_total_cost"] = daily[cost_cols].sum(axis=1).sum()
        if "fill_ratio" in daily:
            stats["avg_fill_ratio"] = daily.fill_ratio.mean()
            stats["fill_ratio_p05"] = daily.fill_ratio.quantile(.05)
        diagnostics = result.diagnostics.reindex(daily.index)
        if "target_gap" in diagnostics:
            stats["tracking_gap"] = diagnostics.target_gap.mean()
        for col in ("gross_exposure", "net_exposure"):
            if col in diagnostics:
                stats[col] = diagnostics[col].mean()
        if {"st_remaining_value"}.issubset(diagnostics.columns):
            stats["st_remaining_max"] = diagnostics.st_remaining_value.max()
        if "stock_cap_excess" in diagnostics:
            stats["stock_cap_excess_max"] = diagnostics.stock_cap_excess.max()
        if {"cash", "equity"}.issubset(daily.columns):
            stats["cash_ratio"] = (daily.cash / daily.equity).mean()
        capital = daily.equity.iloc[0] - daily.pnl.iloc[0] if len(daily) else "Portfolio"
        rows[f"Initial capital {capital:,.0f}" if isinstance(capital, (int, float)) else capital] = stats
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_nav(base, portfolio_result=None, **portfolio_params):
    """绘制扣费前组合、benchmark和扣费后组合净值。"""
    daily, benchmark, full_daily, capital = _daily(base, portfolio_result, portfolio_params)
    fig, ax = plt.subplots(figsize=(12, 5))
    nav, _ = _curves(daily, benchmark, full_daily, capital)
    nav.plot(ax=ax)
    ax.set(title="Portfolio NAV", xlabel="Date", ylabel="NAV")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pnl(base, portfolio_result=None, **portfolio_params):
    """绘制扣费前组合、benchmark和扣费后组合累计PnL。"""
    daily, benchmark, full_daily, capital = _daily(base, portfolio_result, portfolio_params)
    fig, ax = plt.subplots(figsize=(12, 5))
    _, pnl = _curves(daily, benchmark, full_daily, capital)
    pnl.plot(ax=ax)
    ax.set(title="Cumulative PnL", xlabel="Date", ylabel="PnL")
    ax.grid(True, color="gray", alpha=0.2, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drawdown(base, portfolio_result=None, **portfolio_params):
    """绘制扣费前组合、benchmark和扣费后组合回撤。"""
    daily, benchmark, full_daily, capital = _daily(base, portfolio_result, portfolio_params)
    nav, _ = _curves(daily, benchmark, full_daily, capital)
    drawdown = nav.div(nav.cummax()).sub(1)
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown.plot(ax=ax)
    ax.fill_between(drawdown.index, drawdown.min(axis=1), 0, alpha=0.12)
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
    print("Portfolio performance:")
    print(performance_table(base, result).to_string(float_format=lambda value: f"{value:.4f}"))
    plot_nav(base, result)
    plot_pnl(base, result)
    plot_drawdown(base, result)
    plt.show()
