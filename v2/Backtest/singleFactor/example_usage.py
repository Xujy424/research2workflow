"""单因子回测项目的完整调用示例。"""

from . import (BacktestConfig, CapacityConfig, CapacitySimulator, EventConfig,
               ExecutionConfig, FactorData, PairDefinition, PortfolioConfig,
               SingleFactorBacktester, compare_rebalance_frequencies,
               explicit_pair_weights, factor_data_from_arrays,
               run_event_study)
from .config import (ActiveSide, EventTrigger, Method, SignalInput,
                     Weighting)


def run_cov(cov, stock_return, tradable, industry, index_weight):
    data = FactorData(cov, stock_return, tradable, industry, index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED, quantiles=10, top_groups=2,
            weighting=Weighting.SIGNAL, industry_align=True,
            active_side=ActiveSide.LONG, active_gross=1.0),
        execution=ExecutionConfig(
            signal_lag=2, rebalance_days=5, cost_bps=5),
    )
    return SingleFactorBacktester(config).run(data)


def run_prebuilt_long_factor(weight, stock_return, tradable, industry,
                             index_weight):
    """因子层已经选组并归一化；回测层不再重复排序。"""
    data = FactorData(weight, stock_return, tradable, industry, index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED,
            signal_input=SignalInput.PREBUILT_WEIGHT,
            active_side=ActiveSide.LONG,
            weighting=Weighting.SIGNAL,
            industry_align=True,
            active_gross=1.0),
        execution=ExecutionConfig(signal_lag=2, cost_bps=5),
    )
    return SingleFactorBacktester(config).run(data)


def run_continuous_factor_vs_index(factor, stock_return, tradable, industry,
                                   index_weight):
    """多空腿分别归一化，转成统一风险预算的指数主动权重。"""
    data = FactorData(factor, stock_return, tradable, industry, index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED,
            active_side=ActiveSide.LONG_SHORT,
            signal_input=SignalInput.SCORE,
            quantiles=10, top_groups=1, bottom_groups=1,
            weighting=Weighting.EQUAL,
            industry_align=True,
            active_gross=1.0),
        execution=ExecutionConfig(signal_lag=1, cost_bps=5),
    )
    return SingleFactorBacktester(config).run(data)


def compare_cov_rebalance(cov, stock_return, tradable, industry, index_weight):
    """一次得到日、周、月调仓的绩效汇总及各自完整结果。"""
    data = FactorData(cov, stock_return, tradable, industry, index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED, quantiles=10, top_groups=2,
            weighting=Weighting.SIGNAL, industry_align=True),
        execution=ExecutionConfig(signal_lag=2, cost_bps=5),
    )
    return compare_rebalance_frequencies(
        data, config, frequencies=("daily", "weekly", "monthly"))


def compare_cov_matrix_rebalance(cov, stock_return, dates, ticks, tradable,
                                 industry, index_weight):
    """现有 numpy/memmap 数据存储的推荐调用方式。"""
    data = factor_data_from_arrays(
        signal=cov, returns=stock_return, dates=dates, ticks=ticks,
        tradable=tradable, industry=industry,
        benchmark_weight=index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED, quantiles=10, top_groups=2,
            weighting=Weighting.SIGNAL, industry_align=True),
        execution=ExecutionConfig(signal_lag=2, cost_bps=5),
    )
    return compare_rebalance_frequencies(data, config)


def run_short_only(factor, stock_return, tradable):
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.QUANTILE_SHORT_ONLY, quantiles=10,
            bottom_groups=2, weighting=Weighting.EQUAL),
        execution=ExecutionConfig(
            signal_lag=1, cost_bps=5, short_cost_bps_annual=300),
    )
    return SingleFactorBacktester(config).run(
        FactorData(factor, stock_return, tradable))


def run_event(event_signal, stock_return, tradable, industry):
    data = FactorData(event_signal, stock_return, tradable, industry)
    event = EventConfig(
        trigger=EventTrigger.NONZERO, holding_days=5, cooldown_days=5,
        cross_sectionalize=True, pair_within_industry=True)
    portfolio = SingleFactorBacktester(BacktestConfig(
        portfolio=PortfolioConfig(method=Method.EVENT),
        execution=ExecutionConfig(signal_lag=1, cost_bps=5),
        event=event)).run(data)
    study = run_event_study(
        data, event, horizons=(1, 3, 5, 10, 20), adjustment="industry")
    return portfolio, study


def run_capacity(target_weight, next_open, next_amount):
    simulator = CapacitySimulator(CapacityConfig(
        capital=(1e7, 5e7, 1e8, 5e8), max_participation=.1,
        commission_bps=10, impact_coefficient=.001, lot_size=100))
    return simulator.run(target_weight, next_open, next_amount)


def build_pair_book(pair_signal):
    pair = PairDefinition(
        left="600000", right="601398", hedge_ratio=1.0,
        rationale="同业可比公司，价差关系已在样本外验证")
    return explicit_pair_weights(pair_signal, (pair,), holding_days=5)
