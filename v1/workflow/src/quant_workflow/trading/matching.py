"""Queue-aware matching of simulated orders against reconstructed L2 flow."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable

from .book import LimitOrderBook
from .events import (
    L2TradeEvent,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    SimOrder,
)


FillHandler = Callable[[SimOrder, int, float, datetime], None]
OrderHandler = Callable[[SimOrder], None]


# 中文说明：定义 `QueueAwareMatcher`，封装本模块对应的数据、配置与行为。
class QueueAwareMatcher:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        fill_handler: FillHandler,
        order_handler: OrderHandler,
        queue_join_ratio: float = 1.0,
    ) -> None:
        self.fill_handler = fill_handler
        self.order_handler = order_handler
        self.queue_join_ratio = queue_join_ratio
        self.books: dict[str, LimitOrderBook] = {}
        self.active_orders: dict[str, SimOrder] = {}
        # Dict keys form an insertion-ordered set, preserving exchange arrival.
        self.symbol_orders: dict[str, dict[str, None]] = defaultdict(dict)

    # 中文说明：`get_book`：执行该名称对应的业务计算，并返回调用方所需结果。
    def get_book(self, symbol: str) -> LimitOrderBook:
        if symbol not in self.books:
            self.books[symbol] = LimitOrderBook(symbol)
        return self.books[symbol]

    # 中文说明：`submit`：执行该名称对应的业务计算，并返回调用方所需结果。
    def submit(self, order: SimOrder) -> None:
        book = self.get_book(order.request.symbol)
        self.active_orders[order.order_id] = order
        self.symbol_orders[order.request.symbol][order.order_id] = None
        order.status = OrderStatus.ACTIVE
        self._cross_immediately(order, book)
        if order.is_active:
            price = order.request.price
            if price is not None:
                own_ahead = sum(
                    existing.remaining
                    for existing in self.active_orders.values()
                    if existing.order_id != order.order_id
                    and existing.request.symbol == order.request.symbol
                    and existing.request.side == order.request.side
                    and existing.request.price == price
                    and (existing.arrived_at or existing.created_at)
                    <= (order.arrived_at or order.created_at)
                )
                order.queue_ahead = (
                    book.level_quantity(order.request.side, price)
                    * self.queue_join_ratio
                    + own_ahead
                )
        self.order_handler(order)

    # 中文说明：`cancel`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cancel(self, order_id: str, timestamp: datetime) -> SimOrder | None:
        order = self.active_orders.pop(order_id, None)
        if order is None:
            return None
        self.symbol_orders[order.request.symbol].pop(order_id, None)
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = timestamp
        self.order_handler(order)
        return order

    # 中文说明：`on_market_event`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_market_event(self, event: object) -> None:
        symbol = getattr(event, "symbol", None)
        if symbol is None:
            return
        book = self.get_book(symbol)
        if isinstance(event, L2TradeEvent):
            self._match_passive(event)
        book.apply(event)

    # 中文说明：`_cross_immediately`：内部辅助步骤，不作为稳定公共接口。
    def _cross_immediately(self, order: SimOrder, book: LimitOrderBook) -> None:
        request = order.request
        opposite = Side.SELL if request.side == Side.BUY else Side.BUY
        depth = book.depth(opposite)
        remaining = order.remaining
        for price, displayed in depth:
            if remaining <= 0:
                break
            marketable = request.order_type == OrderType.MARKET
            if request.order_type != OrderType.MARKET and request.price is not None:
                marketable = (
                    request.side == Side.BUY and request.price >= price
                ) or (
                    request.side == Side.SELL and request.price <= price
                )
            if not marketable:
                break
            fill = min(remaining, displayed)
            if fill > 0:
                self.fill_handler(
                    order,
                    fill,
                    price,
                    order.arrived_at or order.created_at,
                )
                book.consume_level(opposite, price, fill)
                remaining -= fill
        if order.remaining == 0:
            self._finish(order)
        elif request.order_type == OrderType.MARKET:
            order.status = (
                OrderStatus.PARTIALLY_FILLED
                if order.filled_quantity
                else OrderStatus.REJECTED
            )
            if not order.filled_quantity:
                order.reject_reason = "no executable depth"
            self._remove(order)

    # 中文说明：`_match_passive`：内部辅助步骤，不作为稳定公共接口。
    def _match_passive(self, trade: L2TradeEvent) -> None:
        for order_id in tuple(self.symbol_orders.get(trade.symbol, ())):
            order = self.active_orders.get(order_id)
            if order is None or not order.is_active or order.request.price is None:
                continue
            if not self._trade_reaches_order(trade, order):
                continue
            available = trade.quantity
            if order.queue_ahead > 0:
                consumed = min(order.queue_ahead, available)
                order.queue_ahead -= consumed
                available -= int(consumed)
            if available <= 0:
                continue
            fill = min(order.remaining, available)
            self.fill_handler(order, fill, order.request.price, trade.timestamp)
            if order.remaining == 0:
                self._finish(order)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                self.order_handler(order)

    # 中文说明：`_trade_reaches_order`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _trade_reaches_order(trade: L2TradeEvent, order: SimOrder) -> bool:
        price = order.request.price
        if order.request.side == Side.BUY:
            return (
                trade.price <= price
                and trade.aggressor_side in {Side.SELL, Side.UNKNOWN}
            )
        return (
            trade.price >= price
            and trade.aggressor_side in {Side.BUY, Side.UNKNOWN}
        )

    # 中文说明：`_finish`：内部辅助步骤，不作为稳定公共接口。
    def _finish(self, order: SimOrder) -> None:
        order.status = OrderStatus.FILLED
        self._remove(order)
        self.order_handler(order)

    # 中文说明：`_remove`：内部辅助步骤，不作为稳定公共接口。
    def _remove(self, order: SimOrder) -> None:
        self.active_orders.pop(order.order_id, None)
        self.symbol_orders[order.request.symbol].pop(order.order_id, None)
