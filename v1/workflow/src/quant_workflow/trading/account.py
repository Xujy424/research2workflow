"""China A-share cash account with T+1 settlement and fee accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .events import Exchange, PositionLot, Side, SimTrade


# 中文说明：定义 `FeeSchedule`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class FeeSchedule:
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    sell_stamp_tax_rate: float = 0.0005
    sse_transfer_rate: float = 0.00001

    # 中文说明：`calculate`：计算目标指标。
    def calculate(
        self,
        exchange: Exchange,
        side: Side,
        price: float,
        quantity: int,
    ) -> tuple[float, float]:
        notional = price * quantity
        commission = max(notional * self.commission_rate, self.minimum_commission)
        if exchange == Exchange.SSE:
            commission += notional * self.sse_transfer_rate
        tax = notional * self.sell_stamp_tax_rate if side == Side.SELL else 0.0
        return commission, tax


# 中文说明：定义 `EquityPosition`，封装本模块对应的数据、配置与行为。
@dataclass
class EquityPosition:
    symbol: str
    total_quantity: int = 0
    sellable_quantity: int = 0
    average_cost: float = 0.0
    last_price: float = 0.0
    today_buys: int = 0
    lots: list[PositionLot] = field(default_factory=list)

    # 中文说明：`market_value`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def market_value(self) -> float:
        return self.total_quantity * self.last_price


# 中文说明：定义 `ChinaEquityAccount`，封装本模块对应的数据、配置与行为。
class ChinaEquityAccount:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        initial_cash: float,
        fee_schedule: FeeSchedule | None = None,
        account_id: str = "SIM",
    ) -> None:
        self.account_id = account_id
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.frozen_cash = 0.0
        self.positions: dict[str, EquityPosition] = {}
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.current_date: date | None = None
        self.fee_schedule = fee_schedule or FeeSchedule()

    # 中文说明：`available_cash`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def available_cash(self) -> float:
        return self.cash - self.frozen_cash

    # 中文说明：`market_value`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def market_value(self) -> float:
        return sum(position.market_value for position in self.positions.values())

    # 中文说明：`equity`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    # 中文说明：`start_session`：执行该名称对应的业务计算，并返回调用方所需结果。
    def start_session(self, trading_date: date) -> None:
        self.current_date = trading_date
        for position in self.positions.values():
            position.sellable_quantity = position.total_quantity
            position.today_buys = 0

    # 中文说明：`get_position`：执行该名称对应的业务计算，并返回调用方所需结果。
    def get_position(self, symbol: str) -> EquityPosition:
        if symbol not in self.positions:
            self.positions[symbol] = EquityPosition(symbol)
        return self.positions[symbol]

    # 中文说明：`seed_position`：执行该名称对应的业务计算，并返回调用方所需结果。
    def seed_position(
        self,
        symbol: str,
        quantity: int,
        average_cost: float,
        last_price: float | None = None,
        sellable: bool = True,
    ) -> None:
        position = self.get_position(symbol)
        position.total_quantity = quantity
        position.sellable_quantity = quantity if sellable else 0
        position.average_cost = average_cost
        position.last_price = last_price if last_price is not None else average_cost
        if self.current_date is not None:
            position.lots = [
                PositionLot(self.current_date, quantity, average_cost)
            ]

    # 中文说明：`mark`：执行该名称对应的业务计算，并返回调用方所需结果。
    def mark(self, symbol: str, price: float) -> None:
        if price > 0:
            self.get_position(symbol).last_price = price

    # 中文说明：`reserve`：执行该名称对应的业务计算，并返回调用方所需结果。
    def reserve(self, side: Side, symbol: str, price: float, quantity: int) -> bool:
        if side == Side.BUY:
            estimated = price * quantity * 1.002 + self.fee_schedule.minimum_commission
            if estimated > self.available_cash:
                return False
            self.frozen_cash += estimated
            return True
        position = self.get_position(symbol)
        return quantity <= position.sellable_quantity

    # 中文说明：`release`：执行该名称对应的业务计算，并返回调用方所需结果。
    def release(self, side: Side, price: float, quantity: int) -> None:
        if side == Side.BUY:
            estimated = price * quantity * 1.002 + self.fee_schedule.minimum_commission
            self.frozen_cash = max(self.frozen_cash - estimated, 0.0)

    # 中文说明：`apply_fill`：应用当前规则并更新结果。
    def apply_fill(
        self,
        exchange: Exchange,
        side: Side,
        symbol: str,
        price: float,
        quantity: int,
        trade_date: date,
    ) -> tuple[float, float]:
        position = self.get_position(symbol)
        commission, tax = self.fee_schedule.calculate(
            exchange, side, price, quantity
        )
        notional = price * quantity
        if side == Side.BUY:
            old_cost = position.average_cost * position.total_quantity
            position.total_quantity += quantity
            position.today_buys += quantity
            position.average_cost = (
                old_cost + notional + commission
            ) / position.total_quantity
            position.lots.append(PositionLot(trade_date, quantity, price))
            self.cash -= notional + commission
        else:
            if quantity > position.sellable_quantity:
                raise ValueError("T+1 sellable quantity exceeded")
            pnl = (price - position.average_cost) * quantity - commission - tax
            self.realized_pnl += pnl
            position.total_quantity -= quantity
            position.sellable_quantity -= quantity
            self.cash += notional - commission - tax
            self._consume_lots(position, quantity)
            if position.total_quantity == 0:
                position.average_cost = 0.0
        position.last_price = price
        self.total_commission += commission
        self.total_tax += tax
        return commission, tax

    # 中文说明：`_consume_lots`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _consume_lots(position: EquityPosition, quantity: int) -> None:
        remaining = quantity
        new_lots: list[PositionLot] = []
        for lot in position.lots:
            if remaining <= 0:
                new_lots.append(lot)
                continue
            consumed = min(lot.quantity, remaining)
            remaining -= consumed
            if lot.quantity > consumed:
                new_lots.append(
                    PositionLot(lot.trade_date, lot.quantity - consumed, lot.price)
                )
        position.lots = new_lots

    # 中文说明：`snapshot`：执行该名称对应的业务计算，并返回调用方所需结果。
    def snapshot(self, timestamp: object) -> pd.Series:
        return pd.Series(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "market_value": self.market_value,
                "equity": self.equity,
                "realized_pnl": self.realized_pnl,
                "commission": self.total_commission,
                "tax": self.total_tax,
            }
        )
