"""Preprocess four raw daily L2 tables into two canonical exchange streams."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant_workflow.trading import (
    CanonicalL2Preprocessor,
    DailyL2Bundle,
    Exchange,
    L2ColumnMap,
    L2TableGateway,
)


if __name__ == "__main__":
    trading_date = date(2025, 1, 2)
    raw_root = Path("data/l2/raw")
    bundle = DailyL2Bundle(
        trading_date=trading_date,
        sse_orders=raw_root / "20250102_SSE_orders.csv",
        sse_trades=raw_root / "20250102_SSE_trades.csv",
        szse_orders=raw_root / "20250102_SZSE_orders.csv",
        szse_trades=raw_root / "20250102_SZSE_trades.csv",
    )
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
    raw_gateway = L2TableGateway(
        order_columns={Exchange.SSE: common, Exchange.SZSE: common},
        trade_columns={Exchange.SSE: common, Exchange.SZSE: common},
    )
    canonical, report = CanonicalL2Preprocessor(raw_gateway).preprocess(
        bundle,
        output_root=Path("data/l2/canonical"),
    )
    print(canonical)
    print(
        f"rows={report.sse_rows + report.szse_rows:,}, "
        f"speed={report.rows_per_second:,.0f} rows/s, "
        f"size={report.output_bytes / 1024**2:.1f} MiB"
    )
