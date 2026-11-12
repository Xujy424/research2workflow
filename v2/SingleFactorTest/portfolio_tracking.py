"""作者：xujiayi | 创建日期：2026-11-11
组合跟踪：四类图独立返回Figure，默认复用最近一次实际成交结果。
"""
import matplotlib.pyplot as plt


def _result(base, params):
    """获取组合；Params: base/参数；Return: PortfolioResult。"""
    if params or not hasattr(base, "portfolio"):
        return base.cal_portfolio(**params)
    return base.portfolio


def plot_performance(base, **portfolio_params):
    """净值/回撤图；Params: base/组合参数；Return: Figure。"""
    p = _result(base, portfolio_params)
    daily = p.daily.loc[base.report_start:]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    daily.nav.plot(ax=axes[0], label="Net NAV")
    drawdown = daily.nav / p.daily.nav.cummax().clip(lower=1).reindex(daily.index) - 1
    drawdown.plot(ax=axes[1], label="Drawdown")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_holdings(base, **portfolio_params):
    """目标/实际总敞口与净敞口；Params: base/组合参数；Return: Figure。"""
    p = _result(base, portfolio_params)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, w in (("Target", p.target), ("Actual", p.actual)):
        w.abs().sum(axis=1).loc[base.report_start:].plot(ax=axes[0], label=name)
        w.sum(axis=1).loc[base.report_start:].plot(ax=axes[1], label=name)
    axes[0].set_ylabel("Gross exposure")
    axes[1].set_ylabel("Net exposure")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_execution(base, **portfolio_params):
    """换手/成交率/目标跟踪偏离；Params: base/组合参数；Return: Figure。"""
    p = _result(base, portfolio_params)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    p.daily[["turnover", "fill_ratio"]].loc[base.report_start:].plot(ax=axes[0])
    p.diagnostics[["target_gap", "stock_cap_excess", "industry_deviation"]].loc[base.report_start:].plot(ax=axes[1])
    fig.tight_layout()
    return fig


def plot_costs(base, **portfolio_params):
    """各项累计成本和PnL；Params: base/组合参数；Return: Figure。"""
    p = _result(base, portfolio_params)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    p.daily[["commission", "stamp_tax", "impact", "borrow"]].loc[base.report_start:].cumsum().plot(ax=axes[0])
    p.daily.pnl.loc[base.report_start:].cumsum().plot(ax=axes[1], label="Cumulative PnL")
    axes[1].legend()
    fig.tight_layout()
    return fig
