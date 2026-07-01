"""Order management system shared by backtesting and paper trading."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from heapq import heappop, heappush
from typing import Callable

from .account import ChinaEquityAccount
from .events import (
    OrderRequest,
    OrderStatus,
    Side,
    SimOrder,
    SimTrade,
)
from .matching import QueueAwareMatcher
from .risk import PreTradeRiskEngine


OrderCallback = Callable[[SimOrder], None]
TradeCallback = Callable[[SimTrade], None]


# 中文说明：定义 `SimulationOms`，封装本模块对应的数据、配置与行为。
class SimulationOms:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        account: ChinaEquityAccount,
        risk_engine: PreTradeRiskEngine,
        order_callback: OrderCallback | None = None,
        trade_callback: TradeCallback | None = None,
        order_latency: timedelta | None = None,
    ) -> None:
        self.account = account
        self.risk_engine = risk_engine
        self.order_callback = order_callback or (lambda order: None)
        self.trade_callback = trade_callback or (lambda trade: None)
        self.order_count = 0
        self.trade_count = 0
        self.orders: dict[str, SimOrder] = {}
        self.trades: dict[str, SimTrade] = {}
        self.order_latency = order_latency or timedelta(0)
        self.pending_orders: list[tuple[datetime, int, str]] = []
        self.matcher = QueueAwareMatcher(self._on_fill, self._on_order)

    # 中文说明：`active_orders`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def active_orders(self) -> list[SimOrder]:
        return [order for order in self.orders.values() if order.is_active]

    # 中文说明：`send_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def send_order(self, request: OrderRequest, timestamp: datetime) -> str:
        book = self.matcher.books.get(request.symbol)
        passed, reason = self.risk_engine.check(
            request,
            self.account,
            book,
            len(self.active_orders),
            timestamp,
        )
        self.order_count += 1
        order_id = request.client_order_id or f"SIM-{self.order_count:010d}"
        order = SimOrder(order_id, request, timestamp)
        self.orders[order_id] = order
        if not passed:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            self._on_order(order)
            return order_id
        arrival = timestamp + self.order_latency
        order.arrived_at = arrival
        if arrival <= timestamp:
            self.matcher.submit(order)
        else:
            heappush(self.pending_orders, (arrival, self.order_count, order_id))
            self._on_order(order)
        return order_id

    # 中文说明：`advance_time`：执行该名称对应的业务计算，并返回调用方所需结果。
    def advance_time(self, timestamp: datetime) -> None:
        """Activate orders whose exchange-arrival time is not later than timestamp."""
        while self.pending_orders and self.pending_orders[0][0] <= timestamp:
            arrival, _, order_id = heappop(self.pending_orders)
            order = self.orders.get(order_id)
            if order is None or order.status != OrderStatus.SUBMITTING:
                continue
            order.arrived_at = arrival
            self.matcher.submit(order)

    # 中文说明：`cancel_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cancel_order(self, order_id: str, timestamp: datetime) -> bool:
        order = self.orders.get(order_id)
        if order is None or not order.is_active:
            return False
        if order.status == OrderStatus.SUBMITTING:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = timestamp
            reference = order.request.price or 0.0
            self.account.release(order.request.side, reference, order.remaining)
            self.risk_engine.record_cancel(order.request.symbol)
            self._on_order(order)
            return True
        cancelled = self.matcher.cancel(order_id, timestamp)
        if cancelled is None:
            return False
        reference = cancelled.request.price or 0.0
        self.account.release(
            cancelled.request.side, reference, cancelled.remaining
        )
        self.risk_engine.record_cancel(cancelled.request.symbol)
        return True

    # 中文说明：`cancel_all`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cancel_all(self, timestamp: datetime, strategy_id: str | None = None) -> None:
        for order in tuple(self.active_orders):
            if strategy_id is None or order.request.strategy_id == strategy_id:
                self.cancel_order(order.order_id, timestamp)

    # 中文说明：`on_market_event`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_market_event(self, event: object) -> None:
        self.matcher.on_market_event(event)
        symbol = getattr(event, "symbol", None)
        if symbol is not None:
            book = self.matcher.get_book(symbol)
            price = book.last_price or book.best_bid or book.best_ask
            if price:
                self.account.mark(symbol, float(price))

    # 中文说明：`_on_fill`：内部辅助步骤，不作为稳定公共接口。
    def _on_fill(
        self,
        order: SimOrder,
        quantity: int,
        price: float,
        timestamp: datetime,
    ) -> None:
        old_filled = order.filled_quantity
        new_filled = old_filled + quantity
        order.average_price = (
            order.average_price * old_filled + price * quantity
        ) / new_filled
        order.filled_quantity = new_filled
        order.status = (
            OrderStatus.FILLED
            if order.remaining == 0
            else OrderStatus.PARTIALLY_FILLED
        )
        reserve_price = order.request.price or price
        self.account.release(order.request.side, reserve_price, quantity)
        commission, tax = self.account.apply_fill(
            order.request.exchange,
            order.request.side,
            order.request.symbol,
            price,
            quantity,
            timestamp.date(),
        )
        self.risk_engine.record_trade(price * quantity)
        self.trade_count += 1
        trade = SimTrade(
            trade_id=f"TRADE-{self.trade_count:010d}",
            order_id=order.order_id,
            symbol=order.request.symbol,
            exchange=order.request.exchange,
            side=order.request.side,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            commission=commission,
            tax=tax,
            slippage=(
                (price - reserve_price)
                * (1.0 if order.request.side == Side.BUY else -1.0)
            ),
        )
        self.trades[trade.trade_id] = trade
        self.trade_callback(trade)
        self._on_order(order)

    # 中文说明：`_on_order`：内部辅助步骤，不作为稳定公共接口。
    def _on_order(self, order: SimOrder) -> None:
        self.orders[order.order_id] = order
        self.order_callback(deepcopy(order))
