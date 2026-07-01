"""Bridge portfolio optimiser output into historical replay or paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Sequence

import pandas as pd

from quant_shared.contracts import OptimizationResult
from .account import ChinaEquityAccount, FeeSchedule
from .data import DailyL2Bundle, L2TableGateway
from .engine import HistoricalReplayEngine, PaperTradingEngine, ReplayResult
from .events import OrderType
from .risk import PreTradeRiskConfig, PreTradeRiskEngine
from .strategy import TargetWeightExecutionStrategy


# 中文说明：定义 `TradingSimulationConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class TradingSimulationConfig:
    initial_cash: float = 100_000_000.0
    start_time: time = time(9, 31)
    order_type: OrderType = OrderType.LIMIT
    price_offset_ticks: int = 0
    price_tick: float = 0.01
    lot_size: int = 100
    paper_speed: float = 100.0
    order_latency: timedelta = timedelta(0)


# 中文说明：定义 `PortfolioTradingBridge`，封装本模块对应的数据、配置与行为。
class PortfolioTradingBridge:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        simulation: TradingSimulationConfig | None = None,
        risk: PreTradeRiskConfig | None = None,
        fees: FeeSchedule | None = None,
        gateway: L2TableGateway | None = None,
    ) -> None:
        self.simulation = simulation or TradingSimulationConfig()
        self.risk_config = risk or PreTradeRiskConfig()
        self.fees = fees or FeeSchedule()
        self.gateway = gateway or L2TableGateway()

    # 中文说明：`backtest`：执行该名称对应的业务计算，并返回调用方所需结果。
    def backtest(
        self,
        optimization: OptimizationResult,
        bundles: Sequence[DailyL2Bundle],
        exchange_map: pd.Series | None = None,
    ) -> ReplayResult:
        engine = self._engine(paper=False)
        engine.add_strategy(self._strategy(optimization.weights, exchange_map))
        return engine.run_bundles(bundles)

    # 中文说明：`paper_trade`：执行该名称对应的业务计算，并返回调用方所需结果。
    def paper_trade(
        self,
        optimization: OptimizationResult,
        bundle: DailyL2Bundle,
        exchange_map: pd.Series | None = None,
    ) -> ReplayResult:
        engine = self._engine(paper=True)
        engine.add_strategy(self._strategy(optimization.weights, exchange_map))
        return engine.run_bundle(bundle)

    # 中文说明：`_engine`：内部辅助步骤，不作为稳定公共接口。
    def _engine(
        self, paper: bool
    ) -> HistoricalReplayEngine:
        account = ChinaEquityAccount(
            self.simulation.initial_cash, self.fees
        )
        risk = PreTradeRiskEngine(self.risk_config)
        if paper:
            return PaperTradingEngine(
                account,
                risk,
                self.gateway,
                speed=self.simulation.paper_speed,
                order_latency=self.simulation.order_latency,
            )
        return HistoricalReplayEngine(
            account,
            risk,
            self.gateway,
            order_latency=self.simulation.order_latency,
        )

    # 中文说明：`_strategy`：内部辅助步骤，不作为稳定公共接口。
    def _strategy(
        self,
        weights: pd.Series,
        exchange_map: pd.Series | None,
    ) -> TargetWeightExecutionStrategy:
        if (weights < -1e-12).any():
            raise ValueError(
                "negative target weights require a securities-lending or "
                "futures-hedging execution adapter; ChinaEquityAccount is cash-only"
            )
        return TargetWeightExecutionStrategy(
            weights,
            exchange_map=exchange_map,
            start_time=self.simulation.start_time,
            order_type=self.simulation.order_type,
            price_offset_ticks=self.simulation.price_offset_ticks,
            price_tick=self.simulation.price_tick,
            lot_size=self.simulation.lot_size,
        )
