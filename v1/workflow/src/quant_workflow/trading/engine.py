"""Historical replay and accelerated paper-trading engines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .account import ChinaEquityAccount
from .data import DailyL2Bundle, L2TableGateway
from .events import L2OrderEvent, L2TradeEvent, MarketEvent, SimOrder, SimTrade
from .oms import SimulationOms
from .risk import PreTradeRiskEngine
from .strategy import TradingStrategy
from .persistence import AtomicStateStore, TradingJournal


# 中文说明：定义 `ReplayResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ReplayResult:
    orders: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    statistics: dict[str, float]


# 中文说明：定义 `HistoricalReplayEngine`，封装本模块对应的数据、配置与行为。
class HistoricalReplayEngine:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        account: ChinaEquityAccount,
        risk_engine: PreTradeRiskEngine | None = None,
        gateway: L2TableGateway | None = None,
        journal: TradingJournal | None = None,
        state_store: AtomicStateStore | None = None,
        order_latency: timedelta | None = None,
    ) -> None:
        self.account = account
        self.risk_engine = risk_engine or PreTradeRiskEngine()
        self.gateway = gateway or L2TableGateway()
        self.journal = journal
        self.state_store = state_store
        self.current_time: datetime | None = None
        self.strategy: TradingStrategy | None = None
        self.order_updates: list[dict[str, object]] = []
        self.trade_updates: list[dict[str, object]] = []
        self.equity_updates: list[pd.Series] = []
        self.oms = SimulationOms(
            account,
            self.risk_engine,
            order_callback=self._on_order,
            trade_callback=self._on_trade,
            order_latency=order_latency,
        )

    # 中文说明：`add_strategy`：执行该名称对应的业务计算，并返回调用方所需结果。
    def add_strategy(self, strategy: TradingStrategy) -> None:
        self.strategy = strategy
        strategy.bind(self)

    # 中文说明：`run_bundle`：执行主流程并返回结构化结果。
    def run_bundle(self, bundle: DailyL2Bundle) -> ReplayResult:
        return self.run_events(
            self.gateway.stream(bundle), trading_date=bundle.trading_date
        )

    # 中文说明：`run_bundles`：执行主流程并返回结构化结果。
    def run_bundles(self, bundles: Sequence[DailyL2Bundle]) -> ReplayResult:
        for bundle in sorted(bundles, key=lambda item: item.trading_date):
            self._run_session(self.gateway.stream(bundle), bundle.trading_date)
        return self.result()

    # 中文说明：`run_events`：执行主流程并返回结构化结果。
    def run_events(
        self,
        events: Iterable[MarketEvent],
        trading_date: object,
    ) -> ReplayResult:
        self._run_session(events, trading_date)
        return self.result()

    # 中文说明：`_run_session`：内部辅助步骤，不作为稳定公共接口。
    def _run_session(
        self,
        events: Iterable[MarketEvent],
        trading_date: object,
    ) -> None:
        date_value = pd.Timestamp(trading_date).date()
        self.account.start_session(date_value)
        self.risk_engine.reset_session()
        if self.strategy is not None:
            self.strategy.on_session_start()
        for event in events:
            self.current_time = event.timestamp
            # Orders sent earlier become visible to the exchange before the next
            # market event when their configured latency has elapsed.
            self.oms.advance_time(event.timestamp)
            if self.journal is not None:
                self.journal.append("market", event, event.timestamp)
            self.oms.on_market_event(event)
            if self.strategy is not None:
                self.strategy.on_market_event(event)
        if self.current_time is not None:
            self.oms.cancel_all(self.current_time)
            self.equity_updates.append(self.account.snapshot(self.current_time))
        if self.strategy is not None:
            self.strategy.on_session_end()
        if self.state_store is not None:
            self.state_store.save_account(self.account)

    # 中文说明：`result`：执行该名称对应的业务计算，并返回调用方所需结果。
    def result(self) -> ReplayResult:
        orders = pd.DataFrame(self.order_updates)
        trades = pd.DataFrame(self.trade_updates)
        equity = pd.DataFrame(self.equity_updates)
        positions = pd.DataFrame(
            [
                {
                    "symbol": position.symbol,
                    "quantity": position.total_quantity,
                    "sellable": position.sellable_quantity,
                    "average_cost": position.average_cost,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                }
                for position in self.account.positions.values()
            ]
        )
        return ReplayResult(
            orders=orders,
            trades=trades,
            equity_curve=equity,
            positions=positions,
            statistics=self._statistics(equity, trades),
        )

    # 中文说明：`_on_order`：内部辅助步骤，不作为稳定公共接口。
    def _on_order(self, order: SimOrder) -> None:
        self.order_updates.append(
            {
                "timestamp": self.current_time or order.created_at,
                "order_id": order.order_id,
                "strategy_id": order.request.strategy_id,
                "symbol": order.request.symbol,
                "exchange": order.request.exchange.value,
                "side": order.request.side.value,
                "price": order.request.price,
                "quantity": order.request.quantity,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "queue_ahead": order.queue_ahead,
                "sent_at": order.created_at,
                "arrived_at": order.arrived_at,
                "status": order.status.value,
                "reject_reason": order.reject_reason,
            }
        )
        if self.journal is not None:
            self.journal.append(
                "order", order, self.current_time or order.created_at
            )
        if self.strategy is not None:
            self.strategy.on_order(order)

    # 中文说明：`_on_trade`：内部辅助步骤，不作为稳定公共接口。
    def _on_trade(self, trade: SimTrade) -> None:
        self.trade_updates.append(
            {
                "timestamp": trade.timestamp,
                "trade_id": trade.trade_id,
                "order_id": trade.order_id,
                "symbol": trade.symbol,
                "exchange": trade.exchange.value,
                "side": trade.side.value,
                "price": trade.price,
                "quantity": trade.quantity,
                "notional": trade.price * trade.quantity,
                "commission": trade.commission,
                "tax": trade.tax,
                "slippage": trade.slippage,
            }
        )
        if self.journal is not None:
            self.journal.append("trade", trade, trade.timestamp)
        self.equity_updates.append(self.account.snapshot(trade.timestamp))
        if self.strategy is not None:
            self.strategy.on_trade(trade)

    # 中文说明：`_statistics`：内部辅助步骤，不作为稳定公共接口。
    def _statistics(
        self, equity: pd.DataFrame, trades: pd.DataFrame
    ) -> dict[str, float]:
        if equity.empty:
            return {
                "initial_cash": self.account.initial_cash,
                "ending_equity": self.account.equity,
                "total_return": self.account.equity / self.account.initial_cash - 1.0,
            }
        curve = equity["equity"].astype(float)
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        trade_notional = (
            float(trades["notional"].sum()) if not trades.empty else 0.0
        )
        return {
            "initial_cash": self.account.initial_cash,
            "ending_equity": float(curve.iloc[-1]),
            "total_return": float(curve.iloc[-1] / self.account.initial_cash - 1.0),
            "max_drawdown": float(drawdown.min()),
            "trade_count": float(len(trades)),
            "turnover": trade_notional / max(self.account.initial_cash, 1.0),
            "commission": self.account.total_commission,
            "tax": self.account.total_tax,
            **self._daily_statistics(equity),
        }

    # 中文说明：`_daily_statistics`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _daily_statistics(equity: pd.DataFrame) -> dict[str, float]:
        if equity.empty:
            return {}
        values = equity.copy()
        values["date"] = pd.to_datetime(values["timestamp"]).dt.date
        daily = values.groupby("date")["equity"].last()
        returns = daily.pct_change().dropna()
        if len(returns) < 2 or returns.std(ddof=1) == 0:
            return {"annual_return": np.nan, "annual_volatility": np.nan, "sharpe": np.nan}
        annual_return = float(returns.mean() * 252.0)
        annual_volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": annual_return / annual_volatility,
        }


# 中文说明：定义 `PaperTradingEngine`，封装本模块对应的数据、配置与行为。
class PaperTradingEngine(HistoricalReplayEngine):
    """Use the same OMS/risk stack while replaying data at wall-clock speed."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        account: ChinaEquityAccount,
        risk_engine: PreTradeRiskEngine | None = None,
        gateway: L2TableGateway | None = None,
        speed: float = 100.0,
        max_sleep: float = 0.25,
        order_latency: timedelta | None = None,
    ) -> None:
        super().__init__(
            account,
            risk_engine,
            gateway,
            order_latency=order_latency,
        )
        self.speed = speed
        self.max_sleep = max_sleep

    # 中文说明：`_run_session`：内部辅助步骤，不作为稳定公共接口。
    def _run_session(
        self,
        events: Iterable[MarketEvent],
        trading_date: object,
    ) -> None:
        previous: datetime | None = None

        # 中文说明：`paced`：执行该名称对应的业务计算，并返回调用方所需结果。
        def paced() -> Iterable[MarketEvent]:
            nonlocal previous
            for event in events:
                if previous is not None and self.speed > 0:
                    delay = (event.timestamp - previous).total_seconds() / self.speed
                    if delay > 0:
                        sleep(min(delay, self.max_sleep))
                previous = event.timestamp
                yield event

        super()._run_session(paced(), trading_date)
