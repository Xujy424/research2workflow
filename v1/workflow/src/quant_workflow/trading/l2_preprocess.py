"""Offline normalization of four raw L2 tables into reusable event streams.

Raw vendor files are parsed once.  Orders and trades are merged separately for
SSE and SZSE and stored in a compact canonical Parquet schema.  Repeated
backtests then skip column discovery, string normalization and four-way CSV
parsing, which is normally the dominant data-loading cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from heapq import merge
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator

from .data import DailyL2Bundle, L2TableGateway
from .events import (
    Exchange,
    L2OrderEvent,
    L2TradeEvent,
    MarketEvent,
    OrderType,
    Side,
)


# 中文说明：定义 `PreprocessedL2Bundle`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class PreprocessedL2Bundle:
    """One trading day's two exchange-normalized event files."""

    trading_date: date
    sse_events: Path
    szse_events: Path
    manifest: Path


# 中文说明：定义 `L2PreprocessReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class L2PreprocessReport:
    trading_date: date
    sse_rows: int
    szse_rows: int
    elapsed_seconds: float
    output_bytes: int

    # 中文说明：`rows_per_second`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def rows_per_second(self) -> float:
        return (self.sse_rows + self.szse_rows) / max(self.elapsed_seconds, 1e-12)


# 中文说明：定义 `CanonicalL2Preprocessor`，封装本模块对应的数据、配置与行为。
class CanonicalL2Preprocessor:
    """Convert vendor-specific daily tables to canonical exchange event files."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        source_gateway: L2TableGateway,
        batch_rows: int = 250_000,
        compression: str = "zstd",
    ) -> None:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        self.source_gateway = source_gateway
        self.batch_rows = batch_rows
        self.compression = compression

    # 中文说明：`preprocess`：执行该名称对应的业务计算，并返回调用方所需结果。
    def preprocess(
        self,
        bundle: DailyL2Bundle,
        output_root: Path,
        overwrite: bool = False,
    ) -> tuple[PreprocessedL2Bundle, L2PreprocessReport]:
        """Write SSE/SZSE event streams and an auditable JSON manifest."""
        pa, pq = _pyarrow()
        day_root = output_root / bundle.trading_date.strftime("%Y%m%d")
        day_root.mkdir(parents=True, exist_ok=True)
        paths = {
            Exchange.SSE: day_root / "SSE_events.parquet",
            Exchange.SZSE: day_root / "SZSE_events.parquet",
        }
        manifest_path = day_root / "manifest.json"
        if not overwrite:
            existing = [str(path) for path in (*paths.values(), manifest_path) if path.exists()]
            if existing:
                raise FileExistsError(f"preprocessed outputs already exist: {existing}")

        started = perf_counter()
        counts: dict[Exchange, int] = {}
        for exchange, path in paths.items():
            counts[exchange] = self._write_exchange(
                self.source_gateway.stream_exchange(bundle, exchange),
                path,
                pa,
                pq,
            )
        elapsed = perf_counter() - started
        output_bytes = sum(path.stat().st_size for path in paths.values())
        manifest = {
            "schema_version": 1,
            "trading_date": bundle.trading_date.isoformat(),
            "source": {
                "sse_orders": str(bundle.sse_orders),
                "sse_trades": str(bundle.sse_trades),
                "szse_orders": str(bundle.szse_orders),
                "szse_trades": str(bundle.szse_trades),
            },
            "outputs": {exchange.value: str(path) for exchange, path in paths.items()},
            "rows": {exchange.value: counts[exchange] for exchange in Exchange},
            "sort_key": ["timestamp", "sequence", "event_kind"],
            "compression": self.compression,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = PreprocessedL2Bundle(
            bundle.trading_date,
            paths[Exchange.SSE],
            paths[Exchange.SZSE],
            manifest_path,
        )
        report = L2PreprocessReport(
            bundle.trading_date,
            counts[Exchange.SSE],
            counts[Exchange.SZSE],
            elapsed,
            output_bytes,
        )
        return result, report

    # 中文说明：`_write_exchange`：内部辅助步骤，不作为稳定公共接口。
    def _write_exchange(self, events: Iterable[MarketEvent], path: Path, pa, pq) -> int:
        schema = pa.schema(
            [
                ("timestamp", pa.timestamp("ns")),
                ("sequence", pa.int64()),
                ("event_kind", pa.int8()),
                ("symbol", pa.string()),
                ("event_id", pa.string()),
                ("side", pa.int8()),
                ("price", pa.float64()),
                ("quantity", pa.int64()),
                ("action", pa.int8()),
                ("order_type", pa.int8()),
                ("buy_order_id", pa.string()),
                ("sell_order_id", pa.string()),
                ("channel", pa.int32()),
            ]
        )
        writer = pq.ParquetWriter(
            path,
            schema,
            compression=self.compression,
            use_dictionary=["symbol", "side", "action", "order_type"],
            write_statistics=True,
        )
        rows: list[dict[str, object]] = []
        count = 0
        try:
            for event in events:
                rows.append(_event_record(event))
                if len(rows) >= self.batch_rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    count += len(rows)
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
        finally:
            writer.close()
        return count


# 中文说明：定义 `CanonicalL2Gateway`，封装本模块对应的数据、配置与行为。
class CanonicalL2Gateway:
    """Stream canonical Parquet events with bounded memory usage."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        batch_rows: int = 250_000,
        symbols: Iterable[str] | None = None,
    ) -> None:
        self.batch_rows = batch_rows
        self.symbols = set(symbols) if symbols is not None else None

    # 中文说明：`stream`：按顺序流式输出数据。
    def stream(self, bundle: PreprocessedL2Bundle) -> Iterator[MarketEvent]:
        streams = [
            self.stream_exchange(bundle.sse_events, Exchange.SSE),
            self.stream_exchange(bundle.szse_events, Exchange.SZSE),
        ]
        yield from merge(*streams, key=L2TableGateway._sort_key)

    # 中文说明：`stream_exchange`：按顺序流式输出数据。
    def stream_exchange(
        self,
        path: Path,
        exchange: Exchange,
    ) -> Iterator[MarketEvent]:
        _, pq = _pyarrow()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=self.batch_rows):
            columns = batch.to_pydict()
            for i in range(batch.num_rows):
                symbol = columns["symbol"][i]
                if self.symbols is not None and symbol not in self.symbols:
                    continue
                timestamp = columns["timestamp"][i]
                if not isinstance(timestamp, datetime):
                    timestamp = timestamp.to_pydatetime()
                side = _decode_side(columns["side"][i])
                if columns["event_kind"][i] == 0:
                    yield L2OrderEvent(
                        exchange=exchange,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=columns["sequence"][i],
                        order_id=columns["event_id"][i],
                        side=side,
                        price=columns["price"][i],
                        quantity=columns["quantity"][i],
                        action="ADD" if columns["action"][i] == 0 else "CANCEL",
                        order_type=_decode_order_type(columns["order_type"][i]),
                        channel=columns["channel"][i],
                    )
                else:
                    yield L2TradeEvent(
                        exchange=exchange,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=columns["sequence"][i],
                        trade_id=columns["event_id"][i],
                        price=columns["price"][i],
                        quantity=columns["quantity"][i],
                        aggressor_side=side,
                        buy_order_id=columns["buy_order_id"][i],
                        sell_order_id=columns["sell_order_id"][i],
                        channel=columns["channel"][i],
                    )


# 中文说明：`_event_record`：内部辅助步骤，不作为稳定公共接口。
def _event_record(event: MarketEvent) -> dict[str, object]:
    if isinstance(event, L2OrderEvent):
        return {
            "timestamp": event.timestamp,
            "sequence": event.sequence,
            "event_kind": 0,
            "symbol": event.symbol,
            "event_id": event.order_id,
            "side": _encode_side(event.side),
            "price": event.price,
            "quantity": event.quantity,
            "action": 0 if event.action == "ADD" else 1,
            "order_type": _encode_order_type(event.order_type),
            "buy_order_id": None,
            "sell_order_id": None,
            "channel": event.channel,
        }
    return {
        "timestamp": event.timestamp,
        "sequence": event.sequence,
        "event_kind": 1,
        "symbol": event.symbol,
        "event_id": event.trade_id,
        "side": _encode_side(event.aggressor_side),
        "price": event.price,
        "quantity": event.quantity,
        "action": -1,
        "order_type": -1,
        "buy_order_id": event.buy_order_id,
        "sell_order_id": event.sell_order_id,
        "channel": event.channel,
    }


# 中文说明：`_encode_side`：内部辅助步骤，不作为稳定公共接口。
def _encode_side(side: Side) -> int:
    return {Side.UNKNOWN: 0, Side.BUY: 1, Side.SELL: 2}[side]


# 中文说明：`_decode_side`：内部辅助步骤，不作为稳定公共接口。
def _decode_side(value: int) -> Side:
    return {0: Side.UNKNOWN, 1: Side.BUY, 2: Side.SELL}[value]


# 中文说明：`_encode_order_type`：内部辅助步骤，不作为稳定公共接口。
def _encode_order_type(order_type: OrderType) -> int:
    return {OrderType.LIMIT: 0, OrderType.MARKET: 1, OrderType.BEST: 2}[order_type]


# 中文说明：`_decode_order_type`：内部辅助步骤，不作为稳定公共接口。
def _decode_order_type(value: int) -> OrderType:
    return {0: OrderType.LIMIT, 1: OrderType.MARKET, 2: OrderType.BEST}[value]


# 中文说明：`_pyarrow`：内部辅助步骤，不作为稳定公共接口。
def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "L2 preprocessing requires the optional 'l2' dependency: "
            "pip install 'quant-workflow[l2]'"
        ) from exc
    return pa, pq
