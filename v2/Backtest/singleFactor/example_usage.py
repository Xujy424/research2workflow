"""单因子回测项目的完整调用示例。"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from . import (BacktestConfig, CapacityConfig, CapacitySimulator,
                   EventConfig, ExecutionConfig, FactorData, PairDefinition,
                   PortfolioConfig, SingleFactorBacktester,
                   compare_rebalance_frequencies, explicit_pair_weights,
                   factor_data_from_arrays, run_event_study)
    from .config import (ActiveSide, EventPortfolioMode, EventTrigger, Method,
                         RebalanceFrequency, SignalInput, Weighting)
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.Backtest.singleFactor import (
        BacktestConfig, CapacityConfig, CapacitySimulator, EventConfig,
        ExecutionConfig, FactorData, PairDefinition, PortfolioConfig,
        SingleFactorBacktester, compare_rebalance_frequencies,
        explicit_pair_weights, factor_data_from_arrays, run_event_study,
    )
    from v2.Backtest.singleFactor.config import (
        ActiveSide, EventPortfolioMode, EventTrigger, Method,
        RebalanceFrequency, SignalInput, Weighting,
    )
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


def run_cov(cov, stock_return, tradable, industry, index_weight):
    data = FactorData(cov, stock_return, tradable, industry, index_weight)
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.BENCHMARK_HEDGED, 
            quantiles=10, 
            top_groups=2,
            weighting=Weighting.SIGNAL, 
            industry_align=True,
            active_side=ActiveSide.LONG, 
            active_gross=1.0
        ),
        execution=ExecutionConfig(
            signal_lag=2, 
            rebalance_days=5, 
            cost_bps=5
        ),
    )
    return SingleFactorBacktester(config).run(data)


def run_prebuilt_long_factor(weight, stock_return, tradable, industry, index_weight):
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


def run_continuous_factor_vs_index(factor, stock_return, tradable, industry, index_weight):
    """多空腿分别行业中性化后归一化，转成指增主动权重。"""
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


def run_event(event_signal, stock_return, tradable, industry=None,
              index_weight=None,
              portfolio_mode=EventPortfolioMode.ACTIVE,
              run_study=True,
              study_adjustment=None,
              active_gross=0.20):
    """Run event alpha, an event-only book, or an event index overlay.

    portfolio_mode selects standalone neutralized alpha, a long-only event
    portfolio, or a benchmark plus neutralized event-alpha overlay.
    """
    portfolio_mode = EventPortfolioMode(portfolio_mode)
    if (portfolio_mode == EventPortfolioMode.BENCHMARK_ENHANCED and index_weight is None):
        raise ValueError("benchmark_enhanced requires index_weight")

    data = FactorData(
        signal=event_signal,
        returns=stock_return,
        tradable=tradable,
        industry=industry,
        benchmark_weight=index_weight,
    )
    neutralized = portfolio_mode in {
        EventPortfolioMode.ACTIVE,
        EventPortfolioMode.BENCHMARK_ENHANCED,
    }
    event = EventConfig(
        trigger=EventTrigger.NONZERO,
        holding_days=5,
        cooldown_days=5,
        cross_sectionalize=neutralized,
        neutralize_within_industry=neutralized and industry is not None,
        portfolio_mode=portfolio_mode,
    )
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method=Method.EVENT,
            active_gross=active_gross,
        ),
        execution=ExecutionConfig(signal_lag=1, cost_bps=5),
        event=event,
    )
    portfolio = SingleFactorBacktester(config).run(data)

    study = None
    if run_study:
        adjustment = study_adjustment
        if adjustment is None:
            adjustment = "industry" if industry is not None else "market"
        study = run_event_study(
            data,
            event,
            horizons=(1, 3, 5, 10, 20),
            adjustment=adjustment,
        )
    return portfolio, study


def run_capacity(target_weight, execution_price, traded_amount,
                 exposure_group=None, show_report=True):
    """Run the capacity simulation and optionally print/plot its report."""
    simulator = CapacitySimulator(
        CapacityConfig(
            capital=(1e7, 5e7, 1e8, 5e8), 
            max_participation=.1,
            commission_bps=10, 
            impact_coefficient=.001, 
            lot_size=100
        )
    )
    result = simulator.run(
        target_weight,
        execution_price,
        traded_amount,
        exposure_group=exposure_group,
    )
    if show_report:
        result.report()
    return result


def build_pair_book(pair_signal):
    pair = PairDefinition(
        left="600000", right="601398", hedge_ratio=1.0,
        rationale="同业可比公司，价差关系已在样本外验证")
    return explicit_pair_weights(pair_signal, (pair,), holding_days=5)


def load_backtest_inputs(root, name, start_date, end_date, benchmark="zzfull",
                         execution_lag=1):
    """Load factor inputs and execution-day capacity inputs on one date axis."""
    if not isinstance(execution_lag, (int, np.integer)) or execution_lag < 0:
        raise ValueError("execution_lag must be a non-negative integer")

    with DataPool(root, asset="stock") as data:
        dates = pd.DatetimeIndex(data.axis.trade_dates)
        selected = np.flatnonzero(
            (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        )
        if selected.size == 0:
            raise ValueError("no trade dates found in the requested range")

        start, end = int(selected[0]), int(selected[-1])
        execution_start = start + execution_lag
        execution_end = end + execution_lag
        if execution_end >= len(dates):
            raise ValueError(
                "requested range does not have enough future trade dates for "
                f"execution_lag={execution_lag}"
            )

        next_vwap = np.asarray(
            data.read("d_essentials/open", execution_end, execution_start),
            dtype=float,
        )
        next_amount = np.asarray(
            data.read("d_essentials/amount", execution_end, execution_start),
            dtype=float,
        )

        args = {
            "factor": data.read(f"factor_pool/{name}", end, start),
            "stock_return": data.read("d_essentials/pct", end, start) / 100.0,
            "tradable": data.read("basic/tradable", end, start),
            "industry": data.read("industry/industry", end, start),
            "index_weight": data.read(f"index/weight/{benchmark}_weight", end, start),
            "next_vwap": next_vwap,
            "next_amount": next_amount,
        }
        index = dates[start:end + 1]
        columns = pd.Index(data.axis.ticks, name="tick")

    return {
        name: pd.DataFrame(values, index=index, columns=columns)
        for name, values in args.items()
    }


if __name__ == "__main__":
    # Run with: python -m v2.Backtest.singleFactor.example_usage
    ROOT_PATH = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT
    START_DATE = "2023-01-01"
    END_DATE = "2026-06-30"
    BENCHMARK = "zzfull"
    SIGNAL_LAG = 2

    # 输入与参数设置
    inputs = load_backtest_inputs(
        ROOT_PATH,
        name="sue0",
        start_date=START_DATE,
        end_date=END_DATE,
        benchmark=BENCHMARK,
        execution_lag=SIGNAL_LAG,
    )
    inputs["factor"][~inputs["tradable"]] = np.nan
    data = FactorData(
        inputs["factor"],
        inputs["stock_return"],
        inputs["tradable"],
        inputs["industry"],
        inputs["index_weight"],
    )
    p_config = PortfolioConfig(
        method=Method.BENCHMARK_HEDGED, 
        quantiles=10, 
        top_groups=2,
        weighting=Weighting.SIGNAL, 
        industry_align=True,
        active_side=ActiveSide.LONG, 
        active_gross=1.0
    )
    exe_config = ExecutionConfig(
        signal_lag=SIGNAL_LAG, 
        rebalance_frequency=RebalanceFrequency.WEEKLY,
        cost_bps=10
    )
    config = BacktestConfig(
        portfolio=p_config,
        execution=exe_config,
    )

    # 运行单因子计算与分组收益
    backtest_result = SingleFactorBacktester(config).run(data)

    print("SUE0 backtest summary")
    print(
        f"Average turnover: "
        f"{backtest_result.diagnostics['turnover'].mean():.6f}"
    )
    print(
        f"Mean RankIC: "
        f"{backtest_result.diagnostics['rank_ic'].mean():.6f}"
    )
    backtest_result.report(show=False)

    # 运行回测模拟器
    target = backtest_result.weights["portfolio"]

    simulator = CapacitySimulator(
            CapacityConfig(
                capital=(1e7, 5e7, 1e8, 5e8), 
                max_participation=.1,
                commission_bps=10, 
                impact_coefficient=.001, 
                lot_size=100
            )
        )
    result = simulator.run(
        target_weight=target,
        execution_price=inputs["next_vwap"],
        traded_amount=inputs["next_amount"],
        exposure_group=inputs["industry"],
    )
    result.report(show=False)

    import matplotlib.pyplot as plt
    plt.show()
