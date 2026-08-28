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


class SingleFactorBacktester:
    def __init__(self, config=BacktestConfig()):
        self.config = config

    def _targets(self, data):
        p, method = self.config.portfolio, self.config.portfolio.method
        signal, tradable = data.signal.to_numpy(), data.tradable.to_numpy()
        if method == Method.QUANTILE_LONG_SHORT:
            return quantile_weights(signal, tradable, p), {}
        if method == Method.QUANTILE_LONG_ONLY:
            return quantile_weights(signal, tradable, p, True), {}
        if method == Method.QUANTILE_SHORT_ONLY:
            return short_only_weights(signal, tradable, p), {}
        if method == Method.EVENT:
            ind = None if data.industry is None else data.industry.to_numpy()
            if self.config.event.portfolio_mode == EventPortfolioMode.TRIGGERED_EQUAL_WEIGHT:
                components = event_equal_weight_components(
                    signal, tradable, self.config.event
                )
                return components["active"], components
            return event_weights(signal, tradable, ind, self.config.event), {}
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
            execution.rebalance_days)
        held = np.zeros_like(target)
        for t in range(len(target)):
            held[t] = target[t] if rebalance[t] else held[t-1]
        total_held = held
        benchmark_held = np.zeros_like(held)
        if "total_portfolio" in components:
            total_held = np.zeros_like(components["total_portfolio"])
            for t in range(len(total_held)):
                total_held[t] = (components["total_portfolio"][t]
                                 if rebalance[t] else total_held[t-1])
            for t in range(len(benchmark_held)):
                benchmark_held[t] = (components["benchmark"][t]
                                     if rebalance[t] else benchmark_held[t-1])
        n = len(held) - execution.signal_lag
        future_return = np.nan_to_num(
            data.returns.to_numpy()[execution.signal_lag:])
        active_gross_return = np.full(len(held), np.nan)
        active_gross_return[:n] = np.sum(held[:n] * future_return, axis=1)
        portfolio_gross_return = benchmark_return = None
        if "total_portfolio" in components:
            portfolio_gross_return = np.full(len(held), np.nan)
            benchmark_return = np.full(len(held), np.nan)
            portfolio_gross_return[:n] = np.sum(
                total_held[:n] * future_return, axis=1)
            benchmark_return[:n] = np.sum(
                benchmark_held[:n] * future_return, axis=1)
        pnl = active_gross_return.copy()
        previous = np.vstack((np.zeros((1, held.shape[1])), held[:-1]))
        turnover = .5 * np.abs(held-previous).sum(1)
        pnl[:n] -= turnover[:n] * execution.cost_bps / 1e4
        short_gross = np.abs(np.minimum(total_held, 0)).sum(1)
        pnl[:n] -= short_gross[:n] * execution.short_cost_bps_annual / 1e4 / execution.annual_days
        return_columns = {"net": pnl, "active_gross": active_gross_return}
        if portfolio_gross_return is not None:
            return_columns.update({
                "portfolio_gross": portfolio_gross_return,
                "benchmark": benchmark_return,
                "portfolio_minus_benchmark": (
                    portfolio_gross_return - benchmark_return),
            })
        returns = pd.DataFrame(return_columns, index=data.signal.index)
        weights = {
            "active": pd.DataFrame(held, data.signal.index, data.signal.columns),
            "portfolio": pd.DataFrame(held, data.signal.index, data.signal.columns),
        }
        for name, values in components.items():
            component_held = np.zeros_like(values)
            for t in range(len(values)):
                component_held[t] = values[t] if rebalance[t] else component_held[t-1]
            weights[name] = pd.DataFrame(
                component_held, data.signal.index, data.signal.columns)
        if "total_portfolio" in weights:
            weights["portfolio"] = weights["total_portfolio"]
        summary = performance(returns["net"], execution.annual_days).to_frame().T
        future = data.returns.shift(-execution.signal_lag)
        diagnostics = {"turnover": pd.Series(turnover, data.signal.index),
                       "gross_exposure": pd.Series(np.abs(held).sum(1), data.signal.index),
                       "net_exposure": pd.Series(held.sum(1), data.signal.index),
                       "rank_ic": cross_sectional_ic(data.signal, future)}
        if "benchmark" in weights:
            diagnostics["benchmark_weight"] = weights["benchmark"]
            diagnostics["total_portfolio_gross"] = weights[
                "total_portfolio"].abs().sum(axis=1)
        return BacktestResult(returns, weights, summary, diagnostics)


def _rebalance_mask(index, frequency, every_n_days=1):
    """Select the first available trading day in each week or month."""
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


def compare_rebalance_frequencies(raw, base_config=BacktestConfig(),
                                  frequencies=("daily", "weekly", "monthly")):
    """Run the same factor under several calendars and compare metrics."""
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
