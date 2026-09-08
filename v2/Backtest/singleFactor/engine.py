"""Orchestration and self-financing daily simulation."""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .builders import (benchmark_active_weights, event_equal_weight_components,
                       event_weights,
                       quantile_weights, short_only_weights)
from .config import (BacktestConfig, EventPortfolioMode, Method,
                     RebalanceFrequency)
from .data import FactorData
from .metrics import cross_sectional_ic, performance


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    diagnostics: dict[str, object]

    def report(self, print_summary=True, plot=True, show=True,
               figsize=(14, 18)):
        """Print the performance table and optionally show diagnostics."""
        return report_backtest_result(
            self,
            print_summary=print_summary,
            plot=plot,
            show=show,
            figsize=figsize,
        )


class SingleFactorBacktester:
    def __init__(self, config=BacktestConfig()):
        self.config = config

    def _targets(self, data):
        p, method = self.config.portfolio, self.config.portfolio.method
        signal, tradable = data.signal.to_numpy(), data.tradable.to_numpy()

        # 基准股票池
        benchmark = (None if data.benchmark_weight is None else
                     data.benchmark_weight.to_numpy(float))
        universe = tradable if benchmark is None else (
            tradable & np.isfinite(benchmark) & (benchmark > 0)
        )

        if method == Method.QUANTILE_LONG_SHORT:
            return quantile_weights(signal, universe, p), {}
        
        if method == Method.QUANTILE_LONG_ONLY:
            return quantile_weights(signal, universe, p, True), {}
        
        if method == Method.QUANTILE_SHORT_ONLY:
            return short_only_weights(signal, universe, p), {}

        # 事件主动权重
        if method == Method.EVENT:
            ind = None if data.industry is None else data.industry.to_numpy()
            # 事件股票等权纯多头组合
            if self.config.event.portfolio_mode == EventPortfolioMode.TRIGGERED_EQUAL_WEIGHT:
                components = event_equal_weight_components(
                    signal, tradable, self.config.event, benchmark
                )
                return components["active"], components
            # 事件多空Alpha
            event_active = event_weights(
                signal, universe, ind, self.config.event
            )
            # 事件多空Alpha指增
            if self.config.event.portfolio_mode == EventPortfolioMode.BENCHMARK_ENHANCED:
                if benchmark is None:
                    raise ValueError("benchmark_enhanced requires benchmark_weight")
                if not self.config.event.cross_sectionalize:
                    raise ValueError("benchmark_enhanced requires cross_sectionalize=True")
                benchmark = np.where(np.isfinite(benchmark) & (benchmark > 0), benchmark, 0.0)
                benchmark = benchmark / np.where(
                    benchmark.sum(1, keepdims=True) > 0,
                    benchmark.sum(1, keepdims=True), 1.0
                )
                active = event_active * p.active_gross
                components = {
                    "active": active,
                    "benchmark": benchmark,
                    "portfolio": benchmark + active,
                    "event_active": event_active,
                }
                return active, components
            return event_active, {}

        # 指增主动权重
        if data.benchmark_weight is None:
            raise ValueError("benchmark_hedged requires benchmark_weight")
        industry = None if data.industry is None else data.industry.to_numpy()
        components = benchmark_active_weights(
            signal, tradable, industry,
            data.benchmark_weight.to_numpy(float), p)
        
        return components["active"], components

    def run(self, raw: FactorData) -> BacktestResult:
        data, execution = raw.aligned(), self.config.execution
        target, components = self._targets(data)

        rebalance = _rebalance_mask(
            data.signal.index, execution.rebalance_frequency,
            execution.rebalance_days
        )

        held = np.zeros_like(target)
        for t in range(len(target)):
            held[t] = target[t] if rebalance[t] else held[t-1]
        total_held = held
        benchmark_held = np.zeros_like(held)
        if "portfolio" in components:
            benchmark_held = components["benchmark"].copy()
            total_held = benchmark_held + held

        n = len(held) - execution.signal_lag
        future_return = np.nan_to_num(data.returns.to_numpy()[execution.signal_lag:])

        active_gross_return = np.full(len(held), np.nan)
        active_gross_return[:n] = np.sum(held[:n] * future_return, axis=1)
        portfolio_gross_return = benchmark_return = None
        if "portfolio" in components:
            portfolio_gross_return = np.full(len(held), np.nan)
            benchmark_return = np.full(len(held), np.nan)
            portfolio_gross_return[:n] = np.sum(total_held[:n] * future_return, axis=1)
            benchmark_return[:n] = np.sum(benchmark_held[:n] * future_return, axis=1)

        pnl = active_gross_return.copy()
        previous = np.vstack((np.zeros((1, held.shape[1])), held[:-1])) # 上一期主动权重
        turnover = .5 * np.abs(held-previous).sum(1)  # 每日换手率
        pnl[:n] -= turnover[:n] * execution.cost_bps / 1e4
        # 扣除空头成本
        short_gross = np.abs(np.minimum(total_held, 0)).sum(1)
        pnl[:n] -= short_gross[:n] * execution.short_cost_bps_annual / 1e4 / execution.annual_days

        return_columns = {"net": pnl, "active_gross": active_gross_return}
        if portfolio_gross_return is not None:
            return_columns.update({
                "portfolio_gross": portfolio_gross_return,
                "portfolio_net": benchmark_return + pnl,
                "benchmark": benchmark_return,
            })
        returns = pd.DataFrame(return_columns, index=data.signal.index)

        index, columns = data.signal.index, data.signal.columns
        weights = {
            "active": pd.DataFrame(held, index, columns),
            "portfolio": pd.DataFrame(total_held, index, columns),
            "benchmark": pd.DataFrame(benchmark_held, index, columns)
        }

        summary = performance(returns["net"], execution.annual_days).to_frame().T
        future = data.returns.shift(-execution.signal_lag)
        diagnostics = {
            "turnover": pd.Series(turnover, data.signal.index),
            "gross_exposure": pd.Series(
                np.abs(held).sum(1), data.signal.index
            ),
            "net_exposure": pd.Series(held.sum(1), data.signal.index),
            "rank_ic": cross_sectional_ic(data.signal, future),
            "rebalance": pd.Series(
                rebalance, data.signal.index, name="rebalance"
            ),
        }
        if "benchmark" in weights:
            diagnostics["benchmark_weight"] = weights["benchmark"]
            diagnostics["portfolio_gross"] = weights["portfolio"].abs().sum(axis=1)
        return BacktestResult(returns, weights, summary, diagnostics)


def _nav(series):
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + clean).cumprod()


def _drawdown(series):
    nav = _nav(series)
    return nav / nav.cummax() - 1.0


def _align_twin_zero(reference_axis, target_axis):
    """Align a twin y-axis zero with the reference without sharing scale."""
    reference_low, reference_high = reference_axis.get_ylim()
    reference_span = reference_high - reference_low
    if reference_span <= 0 or not (reference_low < 0 < reference_high):
        return

    zero_fraction = -reference_low / reference_span
    target_low, target_high = target_axis.get_ylim()
    negative = max(-target_low, 0.0)
    positive = max(target_high, 0.0)
    if negative == 0.0 and positive == 0.0:
        return

    span = max(
        negative / zero_fraction,
        positive / (1.0 - zero_fraction),
    )
    target_axis.set_ylim(
        -zero_fraction * span,
        (1.0 - zero_fraction) * span,
    )


def report_backtest_result(
    result: BacktestResult,
    print_summary=True,
    plot=True,
    show=True,
    figsize=(14, 18),
):
    """Report portfolio performance, costs, IC and implementation diagnostics."""
    summary = result.summary.copy()
    if print_summary:
        print(summary.to_string(index=False))
    if not plot:
        return summary, None

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required when plot=True") from exc

    figure, axes = plt.subplots(
        4, 1, figsize=figsize, constrained_layout=True
    )
    returns = result.returns

    nav_columns = (
        ["portfolio_net", "portfolio_gross", "benchmark"]
        if "portfolio_net" in returns
        else ["net", "active_gross"]
    )
    for column in nav_columns:
        axes[0].plot(
            returns.index, _nav(returns[column]), label=column
        )
    axes[0].set_title("Portfolio and benchmark NAV")
    axes[0].set_ylabel("NAV")

    axes[1].plot(
        returns.index, _nav(returns["active_gross"]),
        label="active gross",
    )
    axes[1].plot(
        returns.index, _nav(returns["net"]), label="active net"
    )
    axes[1].set_title("Active PnL before and after costs")
    axes[1].set_ylabel("NAV")
    turnover_axis = axes[1].twinx()
    turnover = result.diagnostics.get("turnover")
    if isinstance(turnover, pd.Series):
        rebalance = result.diagnostics.get("rebalance")
        if isinstance(rebalance, pd.Series):
            turnover_mask = rebalance.fillna(False).astype(bool)
        else:
            turnover_mask = pd.Series(True, index=turnover.index)
        selected_turnover = turnover[
            turnover_mask & turnover.notna() & turnover.ne(0.0)
        ]
        turnover_axis.plot(
            selected_turnover.index, selected_turnover,
            color="tab:grey", alpha=0.45, linewidth=0.9,
            label="rebalance turnover",
        )
    turnover_axis.set_ylabel("Rebalance turnover")

    rank_ic = result.diagnostics.get("rank_ic")
    ic_axis = axes[2].twinx()
    if isinstance(rank_ic, pd.Series):
        rebalance = result.diagnostics.get("rebalance")
        if isinstance(rebalance, pd.Series):
            selected_ic = rank_ic[rebalance.fillna(False).astype(bool)]
        else:
            selected_ic = rank_ic
        selected_ic = selected_ic.dropna()
        colors = np.where(selected_ic >= 0.0, "tab:red", "tab:green")
        axes[2].bar(
            selected_ic.index, selected_ic,
            width=3.0, color=colors, alpha=0.65,
            label="rebalance RankIC",
        )
        if len(selected_ic):
            mean_ic = selected_ic.mean()
            axes[2].axhline(
                mean_ic,
                color="tab:orange",
                linestyle="--",
                linewidth=1.2,
                label=f"mean RankIC ({mean_ic:.4f})",
            )
        cumulative_ic = selected_ic.cumsum()
        if len(cumulative_ic):
            ic_axis.plot(
                cumulative_ic.index, cumulative_ic,
                color="tab:blue", alpha=0.55, linewidth=1.2,
                label="cumulative RankIC",
            )
            ic_axis.fill_between(
                cumulative_ic.index, 0.0, cumulative_ic.to_numpy(),
                color="tab:blue", alpha=0.10,
            )
    axes[2].axhline(0.0, color="grey", linewidth=0.8)
    axes[2].set_title("RankIC at rebalance dates and cumulative RankIC")
    axes[2].set_ylabel("Rebalance RankIC")
    ic_axis.set_ylabel("Cumulative RankIC")
    _align_twin_zero(axes[2], ic_axis)

    drawdown_columns = (
        ["portfolio_net", "portfolio_gross", "benchmark"]
        if "portfolio_net" in returns
        else ["net", "active_gross"]
    )
    for column in drawdown_columns:
        drawdown = _drawdown(returns[column])
        line, = axes[3].plot(
            returns.index, drawdown, label=column
        )
        axes[3].fill_between(
            returns.index, drawdown.to_numpy(), 0.0,
            color=line.get_color(), alpha=0.10,
        )
    axes[3].axhline(0.0, color="grey", linewidth=0.8)
    axes[3].set_title("Underwater drawdown")
    axes[3].set_ylabel("Drawdown")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=30)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize="small")

    pnl_handles, pnl_labels = axes[1].get_legend_handles_labels()
    turnover_handles, turnover_labels = (
        turnover_axis.get_legend_handles_labels()
    )
    axes[1].legend(
        pnl_handles + turnover_handles,
        pnl_labels + turnover_labels,
        fontsize="small",
    )
    ic_handles, ic_labels = axes[2].get_legend_handles_labels()
    cumulative_handles, cumulative_labels = ic_axis.get_legend_handles_labels()
    axes[2].legend(
        ic_handles + cumulative_handles,
        ic_labels + cumulative_labels,
        fontsize="small",
    )

    if show:
        plt.show()
    return summary, figure


def _rebalance_mask(index, frequency, every_n_days=1):
    """标记调仓日."""
    dates = pd.DatetimeIndex(index)
    frequency = RebalanceFrequency(frequency)
    if frequency == RebalanceFrequency.DAILY:
        return np.ones(len(dates), dtype=bool)
    if frequency == RebalanceFrequency.EVERY_N_DAYS:
        mask = np.zeros(len(dates), dtype=bool)
        mask[::every_n_days] = True
        return mask
    period = dates.to_period(
        "W-FRI" if frequency == RebalanceFrequency.WEEKLY else "M")
    return np.r_[True, np.asarray(period[1:] != period[:-1])]


def compare_rebalance_frequencies(raw, base_config=BacktestConfig(), frequencies=("daily", "weekly", "monthly")):
    """依次跑不同调仓频率的结果."""
    results = {}
    summaries = []
    for value in frequencies:
        frequency = RebalanceFrequency(value)
        execution = replace(
            base_config.execution, rebalance_frequency=frequency)
        config = replace(base_config, execution=execution)
        result = SingleFactorBacktester(config).run(raw)
        results[frequency.value] = result
        row = result.summary.copy()
        row.index = pd.Index([frequency.value], name="rebalance_frequency")
        row["average_turnover"] = result.diagnostics["turnover"].mean()
        summaries.append(row)
    return pd.concat(summaries), results
