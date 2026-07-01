"""Pre-trade and intraday controls for simulated and live-compatible engines."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .account import ChinaEquityAccount
from .book import LimitOrderBook
from .events import OrderRequest, Side


# 中文说明：定义 `PreTradeRiskConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class PreTradeRiskConfig:
    max_order_quantity: int = 1_000_000
    max_order_notional: float = 20_000_000.0
    max_symbol_weight: float = 0.05
    max_gross_exposure: float = 1.0
    max_active_orders: int = 200
    max_orders_per_second: int = 50
    max_cancels_per_symbol: int = 500
    max_daily_turnover: float = 2.0
    lot_size: int = 100
    reject_unknown_book: bool = True


# 中文说明：定义 `PreTradeRiskEngine`，封装本模块对应的数据、配置与行为。
class PreTradeRiskEngine:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: PreTradeRiskConfig | None = None) -> None:
        self.config = config or PreTradeRiskConfig()
        self.order_times: deque[datetime] = deque()
        self.cancel_counts: dict[str, int] = defaultdict(int)
        self.daily_traded_notional = 0.0
        self.kill_switch = False
        self.kill_reason = ""

    # 中文说明：`check`：检查业务约束并返回判定。
    def check(
        self,
        request: OrderRequest,
        account: ChinaEquityAccount,
        book: LimitOrderBook | None,
        active_order_count: int,
        timestamp: datetime,
    ) -> tuple[bool, str]:
        cfg = self.config
        if self.kill_switch:
            return False, f"kill switch active: {self.kill_reason}"
        if request.quantity <= 0 or request.quantity % cfg.lot_size != 0:
            return False, f"quantity must be a positive multiple of {cfg.lot_size}"
        if request.quantity > cfg.max_order_quantity:
            return False, "single-order quantity limit exceeded"
        if active_order_count >= cfg.max_active_orders:
            return False, "active-order limit exceeded"
        if book is None and cfg.reject_unknown_book:
            return False, "market book is unavailable"
        reference = request.price or (
            book.best_ask if request.side == Side.BUY else book.best_bid
        )
        if reference is None or reference <= 0:
            return False, "no valid reference price"
        notional = reference * request.quantity
        if notional > cfg.max_order_notional:
            return False, "single-order notional limit exceeded"
        self._expire_flow(timestamp)
        if len(self.order_times) >= cfg.max_orders_per_second:
            return False, "order-flow limit exceeded"
        if request.side == Side.BUY:
            projected_value = (
                account.get_position(request.symbol).market_value + notional
            )
            if projected_value / max(account.equity, 1.0) > cfg.max_symbol_weight:
                return False, "single-symbol weight limit exceeded"
            current_gross = account.market_value
            current_symbol = account.get_position(request.symbol).market_value
            projected_gross = current_gross - current_symbol + projected_value
            if projected_gross / max(account.equity, 1.0) > cfg.max_gross_exposure:
                return False, "gross-exposure limit exceeded"
            if not account.reserve(request.side, request.symbol, reference, request.quantity):
                return False, "insufficient available cash"
        else:
            position = account.get_position(request.symbol)
            if request.quantity > position.sellable_quantity:
                return False, "insufficient T+1 sellable position"
        if (
            self.daily_traded_notional + notional
        ) / max(account.equity, 1.0) > cfg.max_daily_turnover:
            if request.side == Side.BUY:
                account.release(request.side, reference, request.quantity)
            return False, "daily turnover limit exceeded"
        self.order_times.append(timestamp)
        return True, ""

    # 中文说明：`record_trade`：执行该名称对应的业务计算，并返回调用方所需结果。
    def record_trade(self, notional: float) -> None:
        self.daily_traded_notional += abs(notional)

    # 中文说明：`record_cancel`：执行该名称对应的业务计算，并返回调用方所需结果。
    def record_cancel(self, symbol: str) -> None:
        self.cancel_counts[symbol] += 1
        if self.cancel_counts[symbol] >= self.config.max_cancels_per_symbol:
            self.activate_kill_switch(f"cancel limit reached for {symbol}")

    # 中文说明：`activate_kill_switch`：执行该名称对应的业务计算，并返回调用方所需结果。
    def activate_kill_switch(self, reason: str) -> None:
        self.kill_switch = True
        self.kill_reason = reason

    # 中文说明：`reset_session`：重置会话内状态。
    def reset_session(self) -> None:
        self.order_times.clear()
        self.cancel_counts.clear()
        self.daily_traded_notional = 0.0
        self.kill_switch = False
        self.kill_reason = ""

    # 中文说明：`_expire_flow`：内部辅助步骤，不作为稳定公共接口。
    def _expire_flow(self, timestamp: datetime) -> None:
        threshold = timestamp - timedelta(seconds=1)
        while self.order_times and self.order_times[0] <= threshold:
            self.order_times.popleft()
