"""Event-driven stock backtesting and paper-trading subsystem."""

from .account import ChinaEquityAccount, FeeSchedule
from .data import (
    DailyL2Bundle,
    DailyL2FilePattern,
    L2ColumnMap,
    L2DataQualityValidator,
    L2QualityReport,
    L2TableGateway,
    tonglian_l2_gateway,
)
from .engine import HistoricalReplayEngine, PaperTradingEngine, ReplayResult
from .l2_preprocess import (
    CanonicalL2Gateway,
    CanonicalL2Preprocessor,
    L2PreprocessReport,
    PreprocessedL2Bundle,
)
from .events import (
    Exchange,
    L2OrderEvent,
    L2TradeEvent,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    SimOrder,
    SimTrade,
)
from .risk import PreTradeRiskConfig, PreTradeRiskEngine
from .strategy import (
    DailyTargetWeightStrategy,
    TargetWeightExecutionStrategy,
    TradingStrategy,
)
from .integration import PortfolioTradingBridge, TradingSimulationConfig
from .persistence import (
    AccountReconciler,
    AtomicStateStore,
    ReconciliationReport,
    TradingJournal,
)

__all__ = [
    "AccountReconciler",
    "AtomicStateStore",
    "ChinaEquityAccount",
    "CanonicalL2Gateway",
    "CanonicalL2Preprocessor",
    "DailyL2Bundle",
    "DailyL2FilePattern",
    "DailyTargetWeightStrategy",
    "Exchange",
    "FeeSchedule",
    "HistoricalReplayEngine",
    "L2ColumnMap",
    "L2DataQualityValidator",
    "L2QualityReport",
    "L2OrderEvent",
    "L2PreprocessReport",
    "L2TableGateway",
    "L2TradeEvent",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "PaperTradingEngine",
    "PortfolioTradingBridge",
    "PreTradeRiskConfig",
    "PreTradeRiskEngine",
    "ReconciliationReport",
    "PreprocessedL2Bundle",
    "ReplayResult",
    "Side",
    "SimOrder",
    "SimTrade",
    "TargetWeightExecutionStrategy",
    "TradingSimulationConfig",
    "TradingJournal",
    "TradingStrategy",
    "tonglian_l2_gateway",
]
