"""Example wiring for four daily L2 order/trade tables."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pandas as pd

from quant_workflow.trading import (
    ChinaEquityAccount,
    DailyL2Bundle,
    Exchange,
    HistoricalReplayEngine,
    L2ColumnMap,
    L2TableGateway,
    PreTradeRiskConfig,
    PreTradeRiskEngine,
    TargetWeightExecutionStrategy,
)


# 中文说明：`build_engine` 是本示例的函数入口。
def build_engine() -> HistoricalReplayEngine:
    common = L2ColumnMap(
        symbol="symbol",
        timestamp="timestamp",
        sequence="sequence",
        order_id="order_id",
        trade_id="trade_id",
        side="side",
        price="price",
        quantity="quantity",
        action="action",
        buy_order_id="buy_order_id",
        sell_order_id="sell_order_id",
    )
    gateway = L2TableGateway(
        order_columns={Exchange.SSE: common, Exchange.SZSE: common},
        trade_columns={Exchange.SSE: common, Exchange.SZSE: common},
    )
    account = ChinaEquityAccount(100_000_000.0)
    risk = PreTradeRiskEngine(
        PreTradeRiskConfig(max_symbol_weight=0.05)
    )
    return HistoricalReplayEngine(account, risk, gateway)


if __name__ == "__main__":
    data_root = Path("data/l2")
    bundle = DailyL2Bundle(
        date(2025, 1, 2),
        data_root / "20250102_SSE_orders.parquet",
        data_root / "20250102_SSE_trades.parquet",
        data_root / "20250102_SZSE_orders.parquet",
        data_root / "20250102_SZSE_trades.parquet",
    )
    engine = build_engine()
    engine.add_strategy(
        TargetWeightExecutionStrategy(
            pd.Series({"600000": 0.02, "000001": 0.02}),
            start_time=time(9, 31),
        )
    )
    result = engine.run_bundle(bundle)
    print(result.statistics)
