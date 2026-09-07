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
                    "gross_exposure": pd.Series(np.abs(held).sum(1), data.signal.index),
                    "net_exposure": pd.Series(held.sum(1), data.signal.index),
                    "rank_ic": cross_sectional_ic(data.signal, future)
        }
        if "benchmark" in weights:
            diagnostics["benchmark_weight"] = weights["benchmark"]
            diagnostics["portfolio_gross"] = weights["portfolio"].abs().sum(axis=1)
        return BacktestResult(returns, weights, summary, diagnostics)


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
