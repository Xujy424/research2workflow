"""Limit-order-book reconstruction from exchange order and trade events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

from .events import L2OrderEvent, L2TradeEvent, MarketEvent, Side


# 中文说明：定义 `MarketOrderState`，封装本模块对应的数据、配置与行为。
@dataclass
class MarketOrderState:
    side: Side
    price: float
    remaining: int


# 中文说明：定义 `LimitOrderBook`，封装本模块对应的数据、配置与行为。
class LimitOrderBook:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.orders: dict[str, MarketOrderState] = {}
        self.bid_levels: dict[float, int] = defaultdict(int)
        self.ask_levels: dict[float, int] = defaultdict(int)
        self._bid_heap: list[float] = []
        self._ask_heap: list[float] = []
        self.last_price: float = 0.0
        self.cumulative_volume: int = 0

    # 中文说明：`best_bid`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def best_bid(self) -> float | None:
        while self._bid_heap and -self._bid_heap[0] not in self.bid_levels:
            heappop(self._bid_heap)
        return -self._bid_heap[0] if self._bid_heap else None

    # 中文说明：`best_ask`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def best_ask(self) -> float | None:
        while self._ask_heap and self._ask_heap[0] not in self.ask_levels:
            heappop(self._ask_heap)
        return self._ask_heap[0] if self._ask_heap else None

    # 中文说明：`level_quantity`：执行该名称对应的业务计算，并返回调用方所需结果。
    def level_quantity(self, side: Side, price: float) -> int:
        levels = self.bid_levels if side == Side.BUY else self.ask_levels
        return int(levels.get(price, 0))

    # 中文说明：`apply`：应用当前规则并更新结果。
    def apply(self, event: MarketEvent) -> None:
        if isinstance(event, L2OrderEvent):
            self._apply_order(event)
        else:
            self._apply_trade(event)

    # 中文说明：`_apply_order`：内部辅助步骤，不作为稳定公共接口。
    def _apply_order(self, event: L2OrderEvent) -> None:
        if event.action == "CANCEL":
            self.cancel(event.order_id, event.quantity)
            return
        if event.quantity <= 0 or event.side == Side.UNKNOWN:
            return
        if event.order_id in self.orders:
            self.cancel(event.order_id)
        state = MarketOrderState(event.side, event.price, event.quantity)
        self.orders[event.order_id] = state
        levels = self._levels(event.side)
        if event.price not in levels:
            if event.side == Side.BUY:
                heappush(self._bid_heap, -event.price)
            else:
                heappush(self._ask_heap, event.price)
        levels[event.price] += event.quantity

    # 中文说明：`_apply_trade`：内部辅助步骤，不作为稳定公共接口。
    def _apply_trade(self, event: L2TradeEvent) -> None:
        self.last_price = event.price
        self.cumulative_volume += event.quantity
        for side, order_id in (
            (Side.BUY, event.buy_order_id),
            (Side.SELL, event.sell_order_id),
        ):
            if order_id is None:
                continue
            state = self.orders.get(order_id)
            if state is None:
                # A simulated aggressive order may already have consumed the
                # historical order. Continue at the same price queue.
                self.consume_level(side, event.price, event.quantity)
                continue
            reduction = min(state.remaining, event.quantity)
            self._reduce(order_id, reduction)

    # 中文说明：`cancel`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cancel(self, order_id: str, quantity: int | None = None) -> None:
        state = self.orders.get(order_id)
        if state is None:
            return
        reduction = state.remaining if quantity is None or quantity <= 0 else min(
            state.remaining, quantity
        )
        self._reduce(order_id, reduction)

    # 中文说明：`_reduce`：内部辅助步骤，不作为稳定公共接口。
    def _reduce(self, order_id: str, quantity: int) -> None:
        state = self.orders[order_id]
        levels = self._levels(state.side)
        levels[state.price] = max(levels[state.price] - quantity, 0)
        if levels[state.price] == 0:
            levels.pop(state.price, None)
        state.remaining -= quantity
        if state.remaining <= 0:
            self.orders.pop(order_id, None)

    # 中文说明：`_levels`：内部辅助步骤，不作为稳定公共接口。
    def _levels(self, side: Side) -> dict[float, int]:
        return self.bid_levels if side == Side.BUY else self.ask_levels

    # 中文说明：`consume_level`：执行该名称对应的业务计算，并返回调用方所需结果。
    def consume_level(self, side: Side, price: float, quantity: int) -> int:
        """Consume visible FIFO quantity at one price and return actual quantity."""
        remaining = max(int(quantity), 0)
        consumed = 0
        for order_id, state in tuple(self.orders.items()):
            if remaining <= 0:
                break
            if state.side != side or state.price != price:
                continue
            reduction = min(state.remaining, remaining)
            self._reduce(order_id, reduction)
            remaining -= reduction
            consumed += reduction
        return consumed

    # 中文说明：`depth`：执行该名称对应的业务计算，并返回调用方所需结果。
    def depth(
        self, side: Side, levels: int | None = None
    ) -> list[tuple[float, int]]:
        source = self.bid_levels if side == Side.BUY else self.ask_levels
        reverse = side == Side.BUY
        values = sorted(source.items(), reverse=reverse)
        return values if levels is None else values[:levels]
