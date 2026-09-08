"""Production-oriented single-factor backtesting toolkit."""

from .capacity import (CapacityResult, CapacitySimulator,
                       report_capacity_result)
from .adapters import factor_data_from_arrays
from .config import (BacktestConfig, CapacityConfig, EventConfig,
                     ExecutionConfig, PairDefinition, PortfolioConfig,
                     RebalanceFrequency, ActiveSide, EventPortfolioMode,
                     SignalInput)
from .data import FactorData
from .engine import (BacktestResult, SingleFactorBacktester,
                     compare_rebalance_frequencies,
                     report_backtest_result)
from .event_study import EventStudyResult, run_event_study
from .pairs import explicit_pair_weights

__all__ = ["BacktestConfig", "CapacityConfig", "EventConfig", "ExecutionConfig",
           "PortfolioConfig", "FactorData", "BacktestResult",
           "SingleFactorBacktester", "CapacityResult", "CapacitySimulator",
           "report_capacity_result",
           "report_backtest_result",
           "PairDefinition", "EventStudyResult", "run_event_study",
           "explicit_pair_weights", "RebalanceFrequency",
           "compare_rebalance_frequencies", "factor_data_from_arrays",
           "ActiveSide", "EventPortfolioMode", "SignalInput"]
