"""Strategy protocol and target-weight execution strategy."""

from __future__ import annotations

from abc import ABC
from datetime import time
from typing import TYPE_CHECKING

import pandas as pd

from .events import (
    Exchange,
    L2OrderEvent,
    L2TradeEvent,
    MarketEvent,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    SimOrder,
    SimTrade,
)

if TYPE_CHECKING:
    from .engine import HistoricalReplayEngine


# 中文说明：定义 `TradingStrategy`，封装本模块对应的数据、配置与行为。
class TradingStrategy(ABC):
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, strategy_id: str = "strategy") -> None:
        self.strategy_id = strategy_id
        self.engine: HistoricalReplayEngine | None = None

    # 中文说明：`bind`：执行该名称对应的业务计算，并返回调用方所需结果。
    def bind(self, engine: "HistoricalReplayEngine") -> None:
        self.engine = engine

    # 中文说明：`on_session_start`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_session_start(self) -> None:
        pass

    # 中文说明：`on_market_event`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_market_event(self, event: MarketEvent) -> None:
        pass

    # 中文说明：`on_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_order(self, order: SimOrder) -> None:
        pass

    # 中文说明：`on_trade`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_trade(self, trade: SimTrade) -> None:
        pass

    # 中文说明：`on_session_end`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_session_end(self) -> None:
        pass

    # 中文说明：`send_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def send_order(self, request: OrderRequest) -> str:
        if self.engine is None or self.engine.current_time is None:
            raise RuntimeError("strategy is not bound to an active engine")
        return self.engine.oms.send_order(request, self.engine.current_time)

    # 中文说明：`cancel_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cancel_order(self, order_id: str) -> bool:
        if self.engine is None or self.engine.current_time is None:
            return False
        return self.engine.oms.cancel_order(order_id, self.engine.current_time)


# 中文说明：定义 `TargetWeightExecutionStrategy`，封装本模块对应的数据、配置与行为。
class TargetWeightExecutionStrategy(TradingStrategy):
    """Turn optimiser target weights into exchange-aware stock orders."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        target_weights: pd.Series,
        exchange_map: pd.Series | None = None,
        start_time: time = time(9, 31),
        order_type: OrderType = OrderType.LIMIT,
        price_offset_ticks: int = 0,
        price_tick: float = 0.01,
        lot_size: int = 100,
        strategy_id: str = "target_weight",
    ) -> None:
        super().__init__(strategy_id)
        self.target_weights = target_weights.astype(float)
        self.exchange_map = exchange_map
        self.start_time = start_time
        self.order_type = order_type
        self.price_offset_ticks = price_offset_ticks
        self.price_tick = price_tick
        self.lot_size = lot_size
        self.submitted_symbols: set[str] = set()
        self.order_ids: set[str] = set()

    # 中文说明：`on_session_start`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_session_start(self) -> None:
        self.submitted_symbols.clear()
        self.order_ids.clear()

    # 中文说明：`on_market_event`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_market_event(self, event: MarketEvent) -> None:
        if event.timestamp.time() < self.start_time:
            return
        symbol = event.symbol
        if symbol in self.submitted_symbols or symbol not in self.target_weights:
            return
        assert self.engine is not None
        book = self.engine.oms.matcher.get_book(symbol)
        reference = book.last_price or book.best_ask or book.best_bid
        if reference is None or reference <= 0:
            return
        account = self.engine.account
        target_quantity = int(
            self.target_weights[symbol] * account.equity / reference / self.lot_size
        ) * self.lot_size
        current = account.get_position(symbol).total_quantity
        difference = target_quantity - current
        if difference == 0:
            self.submitted_symbols.add(symbol)
            return
        side = Side.BUY if difference > 0 else Side.SELL
        executable = abs(difference)
        if side == Side.SELL:
            executable = min(
                executable, account.get_position(symbol).sellable_quantity
            )
        executable = executable // self.lot_size * self.lot_size
        if executable <= 0:
            self.submitted_symbols.add(symbol)
            return
        if side == Side.BUY:
            base_price = book.best_ask or reference
            price = base_price + self.price_offset_ticks * self.price_tick
        else:
            base_price = book.best_bid or reference
            price = base_price - self.price_offset_ticks * self.price_tick
        exchange = self._exchange(symbol, event)
        order_id = self.send_order(
            OrderRequest(
                symbol=symbol,
                exchange=exchange,
                side=side,
                quantity=executable,
                order_type=self.order_type,
                price=round(price, 3),
                strategy_id=self.strategy_id,
            )
        )
        self.order_ids.add(order_id)
        self.submitted_symbols.add(symbol)

    # 中文说明：`on_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_order(self, order: SimOrder) -> None:
        if order.request.strategy_id != self.strategy_id:
            return
        if order.status in {OrderStatus.REJECTED, OrderStatus.CANCELLED}:
            self.order_ids.discard(order.order_id)

    # 中文说明：`_exchange`：内部辅助步骤，不作为稳定公共接口。
    def _exchange(self, symbol: str, event: MarketEvent) -> Exchange:
        if self.exchange_map is not None and symbol in self.exchange_map.index:
            value = self.exchange_map[symbol]
            return value if isinstance(value, Exchange) else Exchange(str(value))
        if isinstance(event, (L2OrderEvent, L2TradeEvent)):
            return event.exchange
        return Exchange.SSE if symbol.startswith("6") else Exchange.SZSE


# 中文说明：定义 `DailyTargetWeightStrategy`，封装本模块对应的数据、配置与行为。
class DailyTargetWeightStrategy(TargetWeightExecutionStrategy):
    """Use a date-by-symbol target-weight matrix in multi-day replay."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, daily_target_weights: pd.DataFrame, **kwargs: object) -> None:
        first = daily_target_weights.iloc[0] if len(daily_target_weights) else pd.Series(dtype=float)
        super().__init__(first, **kwargs)
        self.daily_target_weights = daily_target_weights.copy()

    # 中文说明：`on_session_start`：执行该名称对应的业务计算，并返回调用方所需结果。
    def on_session_start(self) -> None:
        super().on_session_start()
        assert self.engine is not None
        trading_date = self.engine.account.current_date
        if trading_date is None:
            return
        timestamp = pd.Timestamp(trading_date)
        if timestamp in self.daily_target_weights.index:
            self.target_weights = self.daily_target_weights.loc[timestamp].dropna()
