"""Shanghai/Shenzhen order and trade table adapters with chronological merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from heapq import merge
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .events import Exchange, L2OrderEvent, L2TradeEvent, MarketEvent, OrderType, Side
from quant_shared.restoreSHOrder import SseOrderRestoreColumns, restore_sse_order_files


DEFAULT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "symbol": (
        "symbol", "code", "securityid", "security_id", "instrument",
        "SecurityID", "InstrumentID", "Symbol", "证券代码",
    ),
    "timestamp": (
        "timestamp", "datetime", "time", "tradetime", "ordertime",
        "TickTime", "TradeTime", "TransactTime", "OrderTime",
        "委托时间", "成交时间",
    ),
    "sequence": (
        "sequence", "seq", "applseqnum", "index", "bizindex",
        "BizIndex", "TradeIndex", "ApplSeqNum", "OrderIndex",
    ),
    "order_id": (
        "order_id", "orderid", "orderno", "orderindex", "OrderIndex",
        "OrderNo", "OrderNO", "ApplSeqNum", "BizIndex", "委托编号",
    ),
    "trade_id": (
        "trade_id", "tradeid", "tradeno", "TradeIndex", "TradeNo",
        "BizIndex", "成交编号",
    ),
    "side": (
        "side", "bsflag", "direction", "Side", "BSFlag", "OrderBSFlag",
        "BuySellFlag", "委托方向", "买卖方向",
    ),
    "price": (
        "price", "orderprice", "tradeprice", "OrderPrice", "TradePrice",
        "Price", "委托价格", "成交价格",
    ),
    "quantity": (
        "quantity", "qty", "volume", "ordervolume", "tradevolume",
        "OrderQty", "TradeQty", "Qty", "Volume", "委托数量", "成交数量",
    ),
    "action": (
        "action", "ordertype", "functioncode", "ExecType", "MDStreamID",
        "BizType", "TickType", "OrderType", "委托类型",
    ),
    "buy_order_id": (
        "buy_order_id", "buyorderno", "bidapplseqnum", "BuyOrderNo",
        "BidApplSeqNum", "BuyNo", "买方委托编号",
    ),
    "sell_order_id": (
        "sell_order_id", "sellorderno", "offerapplseqnum", "SellOrderNo",
        "OfferApplSeqNum", "SellNo", "卖方委托编号",
    ),
    "channel": (
        "channel", "channelno", "channelid", "TradeChannel", "OrderChannel",
        "ChannelNo", "ChannelID", "频道代码",
    ),
}


# 中文说明：定义 `L2ColumnMap`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class L2ColumnMap:
    symbol: str | None = None
    timestamp: str | None = None
    sequence: str | None = None
    order_id: str | None = None
    trade_id: str | None = None
    side: str | None = None
    price: str | None = None
    quantity: str | None = None
    action: str | None = None
    buy_order_id: str | None = None
    sell_order_id: str | None = None
    channel: str | None = None
    side_values: Mapping[object, Side] = field(
        default_factory=lambda: {
            "B": Side.BUY,
            "BUY": Side.BUY,
            "1": Side.BUY,
            1: Side.BUY,
            "S": Side.SELL,
            "SELL": Side.SELL,
            "2": Side.SELL,
            2: Side.SELL,
        }
    )
    add_values: tuple[object, ...] = ("ADD", "A", "1", 1)
    cancel_values: tuple[object, ...] = ("CANCEL", "D", "2", 2)
    price_scale: float = 1.0
    quantity_scale: float = 1.0

    # 中文说明：`resolve`：解析配置或数据映射。
    @classmethod
    def tonglian_order(cls, *, price_scale: float = 1.0) -> "L2ColumnMap":
        """Column map for Tonglian SSE/SZSE tick-by-tick order tables."""

        return cls(
            symbol="SecurityID",
            timestamp="TickTime",
            sequence="BizIndex",
            order_id="BizIndex",
            side="Side",
            price="Price",
            quantity="Volume",
            action="MDStreamID",
            channel="ChannelNo",
            price_scale=price_scale,
            add_values=("A", "ADD", "1", 1),
            cancel_values=("D", "DELETE", "CANCEL", "2", 2),
        )

    @classmethod
    def tonglian_sse_trade(cls, *, price_scale: float = 1.0) -> "L2ColumnMap":
        """Column map for Tonglian SSE tick-by-tick trade tables."""

        return cls(
            symbol="SecCode",
            timestamp="TransactTime",
            sequence="ApplSeqNum",
            trade_id="ApplSeqNum",
            price="LastPx",
            quantity="LastQty",
            side=None,
            buy_order_id="BidApplSeqNum",
            sell_order_id="OfferApplSeqNum",
            channel="ChannelNo",
            price_scale=price_scale,
        )
    @classmethod
    def tonglian_sse_restored_order(cls, *, price_scale: float = 1.0) -> "L2ColumnMap":
        """Column map for restored Tonglian SSE order tables."""

        return cls(
            symbol="SecCode",
            timestamp="TransactTime",
            sequence="ApplSeqNum",
            order_id="ApplSeqNum",
            side="Side",
            price="Price",
            quantity="OrderQty",
            action="OrderStatus",
            channel="ChannelNo",
            price_scale=price_scale,
            add_values=("partial_active_trade", "passive_or_untouched", "fully_active_trade"),
            cancel_values=("D", "DELETE", "CANCEL", "2", 2),
        )
    @classmethod
    def tonglian_trade(cls, *, price_scale: float = 1.0) -> "L2ColumnMap":
        """Column map for Tonglian SSE/SZSE tick-by-tick trade tables."""

        return cls(
            symbol="SecurityID",
            timestamp="TradeTime",
            sequence="TradeIndex",
            trade_id="TradeIndex",
            price="TradePrice",
            quantity="TradeQty",
            side="Side",
            buy_order_id="BuyOrderNo",
            sell_order_id="SellOrderNo",
            channel="TradeChannel",
            price_scale=price_scale,
        )
    def resolve(
        self,
        columns: Iterable[object],
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
    ) -> dict[str, str]:
        original = {str(column).lower(): str(column) for column in columns}
        resolved: dict[str, str] = {}
        for field_name in dict.fromkeys(required + optional):
            explicit = getattr(self, field_name)
            if explicit is not None:
                if explicit not in columns:
                    raise ValueError(f"configured column {explicit!r} is missing")
                resolved[field_name] = explicit
                continue
            for alias in DEFAULT_ALIASES[field_name]:
                match = original.get(alias.lower())
                if match is not None:
                    resolved[field_name] = match
                    break
        missing = [field_name for field_name in required if field_name not in resolved]
        if missing:
            raise ValueError(
                f"missing required L2 columns {missing}; available={list(columns)}"
            )
        return resolved


# 中文说明：定义 `DailyL2Bundle`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class DailyL2Bundle:
    trading_date: date
    sse_orders: Path
    sse_trades: Path
    szse_orders: Path
    szse_trades: Path


# 中文说明：定义 `DailyL2FilePattern`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class DailyL2FilePattern:
    root: Path
    sse_orders: str = "{date}_SSE_orders.csv"
    sse_trades: str = "{date}_SSE_trades.csv"
    szse_orders: str = "{date}_SZSE_orders.csv"
    szse_trades: str = "{date}_SZSE_trades.csv"
    date_format: str = "%Y%m%d"

    # 中文说明：`resolve`：解析配置或数据映射。
    def resolve(self, trading_date: date, require_exists: bool = True) -> DailyL2Bundle:
        token = trading_date.strftime(self.date_format)
        bundle = DailyL2Bundle(
            trading_date,
            self.root / self.sse_orders.format(date=token),
            self.root / self.sse_trades.format(date=token),
            self.root / self.szse_orders.format(date=token),
            self.root / self.szse_trades.format(date=token),
        )
        if require_exists:
            missing = [
                str(path)
                for path in (
                    bundle.sse_orders,
                    bundle.sse_trades,
                    bundle.szse_orders,
                    bundle.szse_trades,
                )
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(f"missing daily L2 tables: {missing}")
        return bundle


# 中文说明：定义 `L2TableGateway`，封装本模块对应的数据、配置与行为。
class L2TableGateway:
    """Read four daily tables and emit one exchange-time ordered event stream."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        order_columns: Mapping[Exchange, L2ColumnMap] | None = None,
        trade_columns: Mapping[Exchange, L2ColumnMap] | None = None,
        chunksize: int = 500_000,
        validate_ordering: bool = True,
        retain_raw: bool = False,
        use_polars: bool = False,
        restore_sse_orders: bool = False,
        sse_restore_columns: SseOrderRestoreColumns | None = None,
    ) -> None:
        self.order_columns = dict(order_columns or {})
        self.trade_columns = dict(trade_columns or {})
        self.chunksize = chunksize
        self.validate_ordering = validate_ordering
        self.retain_raw = retain_raw
        self.use_polars = use_polars
        self.restore_sse_orders = restore_sse_orders
        self.sse_restore_columns = sse_restore_columns

    # 中文说明：`stream`：按顺序流式输出数据。
    def stream(self, bundle: DailyL2Bundle) -> Iterator[MarketEvent]:
        streams = [
            self.stream_exchange(bundle, Exchange.SSE),
            self.stream_exchange(bundle, Exchange.SZSE),
        ]
        yield from merge(*streams, key=self._sort_key)

    # 中文说明：`stream_exchange`：按顺序流式输出数据。
    def stream_exchange(
        self,
        bundle: DailyL2Bundle,
        exchange: Exchange,
    ) -> Iterator[MarketEvent]:
        """Merge one exchange's order and trade tables into exchange sequence."""
        if exchange == Exchange.SSE:
            order_path, trade_path = bundle.sse_orders, bundle.sse_trades
        else:
            order_path, trade_path = bundle.szse_orders, bundle.szse_trades
        order_stream = (
            self._stream_restored_sse_orders(order_path, trade_path, bundle.trading_date)
            if exchange == Exchange.SSE and self.restore_sse_orders
            else self._stream_orders(order_path, exchange, bundle.trading_date)
        )
        yield from merge(
            order_stream,
            self._stream_trades(trade_path, exchange, bundle.trading_date),
            key=self._sort_key,
        )

    # 中文说明：`_sort_key`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _sort_key(event: MarketEvent) -> tuple[datetime, int, int]:
        priority = 0 if isinstance(event, L2OrderEvent) else 1
        return event.timestamp, event.sequence, priority

    # 中文说明：`_stream_orders`：内部辅助步骤，不作为稳定公共接口。
    def _stream_orders(
        self, path: Path, exchange: Exchange, trading_date: date
    ) -> Iterator[L2OrderEvent]:
        mapping = self.order_columns.get(exchange, L2ColumnMap())
        previous: tuple[datetime, int, int] | None = None
        for frame in self._read_chunks(path):
            for event in self._order_events_from_frame(
                frame,
                exchange,
                trading_date,
                mapping,
            ):
                previous = self._validate_key(event, previous, path)
                yield event

    def _order_events_from_frame(
        self,
        frame: pd.DataFrame,
        exchange: Exchange,
        trading_date: date,
        mapping: L2ColumnMap,
    ) -> Iterator[L2OrderEvent]:
        columns = mapping.resolve(
            frame.columns,
            ("symbol", "timestamp", "sequence", "order_id", "side", "price", "quantity"),
            ("action", "channel"),
        )
        positions = {str(column): i for i, column in enumerate(frame.columns)}
        for values in frame.itertuples(index=False, name=None):
            get = lambda field, default=None: (
                values[positions[columns[field]]] if field in columns else default
            )
            action_value = get("action", "ADD")
            action = "CANCEL" if action_value in mapping.cancel_values else "ADD"
            yield L2OrderEvent(
                exchange=exchange,
                symbol=self._normalize_symbol(get("symbol")),
                timestamp=self._parse_timestamp(get("timestamp"), trading_date),
                sequence=int(get("sequence")),
                order_id=str(get("order_id")),
                side=self._parse_side(get("side"), mapping),
                price=float(get("price")) * mapping.price_scale,
                quantity=int(round(float(get("quantity")) * mapping.quantity_scale)),
                action=action,
                order_type=OrderType.LIMIT,
                channel=self._optional_value_int(get("channel")),
                raw=(dict(zip(frame.columns.astype(str), values)) if self.retain_raw else {}),
            )
    def _stream_restored_sse_orders(
        self,
        order_path: Path,
        trade_path: Path,
        trading_date: date,
    ) -> Iterator[L2OrderEvent]:
        mapping = self.order_columns.get(Exchange.SSE, L2ColumnMap.tonglian_sse_restored_order())
        restored = restore_sse_order_files(
            order_path,
            trade_path,
            self.sse_restore_columns,
        ).to_pandas()
        previous: tuple[datetime, int, int] | None = None
        for event in self._order_events_from_frame(
            restored,
            Exchange.SSE,
            trading_date,
            mapping,
        ):
            previous = self._validate_key(event, previous, order_path)
            yield event
    # 中文说明：`_stream_trades`：内部辅助步骤，不作为稳定公共接口。
    def _stream_trades(
        self, path: Path, exchange: Exchange, trading_date: date
    ) -> Iterator[L2TradeEvent]:
        mapping = self.trade_columns.get(exchange, L2ColumnMap())
        previous: tuple[datetime, int, int] | None = None
        for frame in self._read_chunks(path):
            columns = mapping.resolve(
                frame.columns,
                ("symbol", "timestamp", "sequence", "trade_id", "price", "quantity"),
                ("side", "buy_order_id", "sell_order_id", "channel"),
            )
            positions = {str(column): i for i, column in enumerate(frame.columns)}
            for values in frame.itertuples(index=False, name=None):
                get = lambda field, default=None: (
                    values[positions[columns[field]]]
                    if field in columns
                    else default
                )
                side = (
                    self._parse_side(get("side"), mapping)
                    if "side" in columns
                    else Side.UNKNOWN
                )
                event = L2TradeEvent(
                    exchange=exchange,
                    symbol=self._normalize_symbol(get("symbol")),
                    timestamp=self._parse_timestamp(get("timestamp"), trading_date),
                    sequence=int(get("sequence")),
                    trade_id=str(get("trade_id")),
                    price=float(get("price")) * mapping.price_scale,
                    quantity=int(round(float(get("quantity")) * mapping.quantity_scale)),
                    aggressor_side=side,
                    buy_order_id=self._optional_value_str(get("buy_order_id")),
                    sell_order_id=self._optional_value_str(get("sell_order_id")),
                    channel=self._optional_value_int(get("channel")),
                    raw=(
                        dict(zip(frame.columns.astype(str), values))
                        if self.retain_raw
                        else {}
                    ),
                )
                previous = self._validate_key(event, previous, path)
                yield event

    # 中文说明：`_read_chunks`：内部辅助步骤，不作为稳定公共接口。
    def _read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        suffix = path.suffix.lower()
        if self.use_polars and suffix in {".csv", ".txt", ".parquet"}:
            yield from self._read_chunks_polars(path)
        elif suffix in {".csv", ".txt"}:
            yield from pd.read_csv(path, chunksize=self.chunksize, low_memory=False)
        elif suffix == ".parquet":
            yield pd.read_parquet(path)
        elif suffix in {".feather", ".ftr"}:
            yield pd.read_feather(path)
        elif suffix in {".pkl", ".pickle"}:
            yield pd.read_pickle(path)
        else:
            raise ValueError(f"unsupported L2 table format: {path}")

    def _read_chunks_polars(self, path: Path) -> Iterator[pd.DataFrame]:
        pl = _polars()
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            reader = pl.read_csv_batched(path, batch_size=self.chunksize)
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                for batch in batches:
                    yield batch.to_pandas()
            return
        if suffix == ".parquet":
            frame = pl.scan_parquet(path).collect(streaming=True)
            for offset in range(0, frame.height, self.chunksize):
                yield frame.slice(offset, self.chunksize).to_pandas()
            return
        raise ValueError(f"unsupported Polars L2 table format: {path}")

    # 中文说明：`_parse_side`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _parse_side(value: object, mapping: L2ColumnMap) -> Side:
        if value in mapping.side_values:
            return mapping.side_values[value]
        normalized = str(value).strip().upper()
        if normalized in mapping.side_values:
            return mapping.side_values[normalized]
        return Side.UNKNOWN

    # 中文说明：`_normalize_symbol`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _normalize_symbol(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(6) if text.isdigit() else text

    # 中文说明：`_parse_timestamp`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _parse_timestamp(value: object, trading_date: date) -> datetime:
        if isinstance(value, pd.Timestamp):
            timestamp = value.to_pydatetime()
        elif isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, time):
            timestamp = datetime.combine(trading_date, value)
        elif isinstance(value, (int, np.integer, float, np.floating)):
            digits = str(int(value)).zfill(9)
            main = digits[:6]
            fraction = digits[6:]
            timestamp = datetime.strptime(
                f"{trading_date:%Y%m%d}{main}{fraction}", "%Y%m%d%H%M%S%f"
            )
        else:
            text = str(value).strip()
            if _looks_like_time_only(text):
                parsed_time = pd.to_datetime(text, errors="coerce")
                if pd.isna(parsed_time):
                    raise ValueError(f"cannot parse L2 timestamp: {value!r}")
                timestamp = datetime.combine(trading_date, parsed_time.time())
            else:
                parsed = pd.to_datetime(text, errors="coerce")
                if pd.isna(parsed):
                    raise ValueError(f"cannot parse L2 timestamp: {value!r}")
                timestamp = parsed.to_pydatetime()
        if timestamp.date() != trading_date and timestamp.year == 1900:
            timestamp = datetime.combine(trading_date, timestamp.time())
        return timestamp

    # 中文说明：`_optional_str`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _optional_str(row: Mapping[str, object], column: str | None) -> str | None:
        if column is None or pd.isna(row.get(column)):
            return None
        return str(row[column])

    # 中文说明：`_optional_int`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _optional_int(row: Mapping[str, object], column: str | None) -> int | None:
        if column is None or pd.isna(row.get(column)):
            return None
        return int(row[column])

    # 中文说明：`_optional_value_str`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _optional_value_str(value: object) -> str | None:
        return None if value is None or pd.isna(value) else str(value)

    # 中文说明：`_optional_value_int`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _optional_value_int(value: object) -> int | None:
        return None if value is None or pd.isna(value) else int(value)

    # 中文说明：`_validate_key`：内部辅助步骤，不作为稳定公共接口。
    def _validate_key(
        self,
        event: MarketEvent,
        previous: tuple[datetime, int, int] | None,
        path: Path,
    ) -> tuple[datetime, int, int]:
        key = self._sort_key(event)
        if self.validate_ordering and previous is not None and key < previous:
            raise ValueError(
                f"L2 table is not ordered by timestamp/sequence: {path}; "
                f"previous={previous}, current={key}"
            )
        return key


# 中文说明：定义 `L2QualityReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class L2QualityReport:
    rows: int
    duplicate_sequences: int
    invalid_prices: int
    invalid_quantities: int
    unknown_sides: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None

    # 中文说明：`passed`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def passed(self) -> bool:
        return (
            self.duplicate_sequences == 0
            and self.invalid_prices == 0
            and self.invalid_quantities == 0
        )


# 中文说明：定义 `L2DataQualityValidator`，封装本模块对应的数据、配置与行为。

def tonglian_l2_gateway(
    *,
    chunksize: int = 500_000,
    validate_ordering: bool = True,
    retain_raw: bool = False,
    use_polars: bool = True,
    price_scale: float = 1.0,
    restore_sse_orders: bool = True,
) -> L2TableGateway:
    """Build an L2 gateway for Tonglian SSE/SZSE order and trade files."""

    order_map = L2ColumnMap.tonglian_order(price_scale=price_scale)
    sse_order_map = L2ColumnMap.tonglian_sse_restored_order(price_scale=price_scale)
    trade_map = L2ColumnMap.tonglian_trade(price_scale=price_scale)
    sse_trade_map = L2ColumnMap.tonglian_sse_trade(price_scale=price_scale)
    return L2TableGateway(
        order_columns={Exchange.SSE: sse_order_map, Exchange.SZSE: order_map},
        trade_columns={Exchange.SSE: sse_trade_map, Exchange.SZSE: trade_map},
        chunksize=chunksize,
        validate_ordering=validate_ordering,
        retain_raw=retain_raw,
        use_polars=use_polars,
        restore_sse_orders=restore_sse_orders,
    )
class L2DataQualityValidator:
    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(self, events: Iterable[MarketEvent]) -> L2QualityReport:
        rows = 0
        duplicates = 0
        invalid_prices = 0
        invalid_quantities = 0
        unknown_sides = 0
        first: datetime | None = None
        last: datetime | None = None
        seen: set[tuple[Exchange, int]] = set()
        for event in events:
            rows += 1
            first = first or event.timestamp
            last = event.timestamp
            key = (event.exchange, event.sequence)
            if key in seen:
                duplicates += 1
            seen.add(key)
            invalid_prices += int(event.price < 0)
            invalid_quantities += int(event.quantity <= 0)
            side = (
                event.side
                if isinstance(event, L2OrderEvent)
                else event.aggressor_side
            )
            unknown_sides += int(side == Side.UNKNOWN)
        return L2QualityReport(
            rows,
            duplicates,
            invalid_prices,
            invalid_quantities,
            unknown_sides,
            first,
            last,
        )


def _polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "Polars L2 loading requires the optional dependency: pip install polars"
        ) from exc
    return pl

def _looks_like_time_only(value: str) -> bool:
    parts = value.split(":")
    return len(parts) == 3 and all(part[:2].isdigit() for part in parts)