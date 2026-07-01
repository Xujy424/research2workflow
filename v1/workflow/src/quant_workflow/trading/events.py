"""Canonical immutable events used by replay and paper-trading engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Mapping


# 中文说明：定义 `Exchange`，封装本模块对应的数据、配置与行为。
class Exchange(str, Enum):
    SSE = "SSE"
    SZSE = "SZSE"


# 中文说明：定义 `Side`，封装本模块对应的数据、配置与行为。
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


# 中文说明：定义 `OrderType`，封装本模块对应的数据、配置与行为。
class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    BEST = "BEST"


# 中文说明：定义 `OrderStatus`，封装本模块对应的数据、配置与行为。
class OrderStatus(str, Enum):
    SUBMITTING = "SUBMITTING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# 中文说明：定义 `L2OrderEvent`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class L2OrderEvent:
    exchange: Exchange
    symbol: str
    timestamp: datetime
    sequence: int
    order_id: str
    side: Side
    price: float
    quantity: int
    action: str = "ADD"
    order_type: OrderType = OrderType.LIMIT
    channel: int | None = None
    raw: Mapping[str, object] = field(default_factory=dict, compare=False)


# 中文说明：定义 `L2TradeEvent`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class L2TradeEvent:
    exchange: Exchange
    symbol: str
    timestamp: datetime
    sequence: int
    trade_id: str
    price: float
    quantity: int
    aggressor_side: Side = Side.UNKNOWN
    buy_order_id: str | None = None
    sell_order_id: str | None = None
    channel: int | None = None
    raw: Mapping[str, object] = field(default_factory=dict, compare=False)


MarketEvent = L2OrderEvent | L2TradeEvent


# 中文说明：定义 `OrderRequest`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    exchange: Exchange
    side: Side
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    price: float | None = None
    strategy_id: str = "portfolio"
    client_order_id: str | None = None
    participation_limit: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


# 中文说明：定义 `SimOrder`，封装本模块对应的数据、配置与行为。
@dataclass
class SimOrder:
    order_id: str
    request: OrderRequest
    created_at: datetime
    arrived_at: datetime | None = None
    status: OrderStatus = OrderStatus.SUBMITTING
    filled_quantity: int = 0
    average_price: float = 0.0
    queue_ahead: float = 0.0
    reject_reason: str = ""
    cancelled_at: datetime | None = None

    # 中文说明：`remaining`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def remaining(self) -> int:
        return max(self.request.quantity - self.filled_quantity, 0)

    # 中文说明：`is_active`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def is_active(self) -> bool:
        return self.status in {
            OrderStatus.SUBMITTING,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        }


# 中文说明：定义 `SimTrade`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class SimTrade:
    trade_id: str
    order_id: str
    symbol: str
    exchange: Exchange
    side: Side
    timestamp: datetime
    price: float
    quantity: int
    commission: float
    tax: float
    slippage: float = 0.0


# 中文说明：定义 `PositionLot`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class PositionLot:
    trade_date: date
    quantity: int
    price: float
