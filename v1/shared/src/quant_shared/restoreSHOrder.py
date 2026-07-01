"""Restore SSE tick-by-tick order table from Tonglian order and trade files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SseOrderRestoreColumns:
    channel: str = "ChannelNo"
    order_seq: str = "ApplSeqNum"
    security: str = "SecCode"
    side: str = "Side"
    order_qty: str = "OrderQty"
    order_price: str = "Price"
    order_time: str = "TransactTime"
    trade_seq: str = "ApplSeqNum"
    bid_seq: str = "BidApplSeqNum"
    offer_seq: str = "OfferApplSeqNum"
    trade_qty: str = "LastQty"
    trade_price: str = "LastPx"
    trade_time: str = "TransactTime"
    exec_type: str = "ExecType"
    trade_date: str = "FDate"
    order_type: str = "OrdType"
    seq_no: str = "SeqNo"


def restore_sse_order_table(
    order_table,
    trade_table,
    columns: SseOrderRestoreColumns | None = None,
):
    """Restore initial SSE order quantities from order and trade tables."""

    pl = _polars()
    cols = columns or SseOrderRestoreColumns()
    order_df = _ensure_polars(order_table)
    trade_df = _ensure_polars(trade_table)
    trade_df = trade_df.with_columns(
        pl.col(cols.bid_seq, cols.offer_seq).cast(pl.Int64),
        pl.col(cols.trade_time).str.strptime(pl.Time, "%H:%M:%S%.3f", strict=False),
    )
    order_df = order_df.with_columns(
        pl.col(cols.order_time).str.strptime(pl.Time, "%H:%M:%S%.3f", strict=False)
    )
    trade_df = trade_df.filter(
        ~(
            (pl.col(cols.trade_time) <= pl.time(9, 29))
            | (pl.col(cols.trade_time) >= pl.time(14, 57))
        )
    )
    deals = trade_df.filter(pl.col(cols.exec_type) != 52).sort(
        [cols.channel, cols.trade_seq, cols.security, cols.trade_time]
    )
    deals = deals.with_columns(
        pl.when(pl.col(cols.bid_seq) > pl.col(cols.offer_seq))
        .then(pl.lit("B"))
        .otherwise(pl.lit("S"))
        .alias("RestoredSide")
    )
    buy = deals.filter(pl.col("RestoredSide") == "B").select(
        cols.channel,
        cols.bid_seq,
        cols.security,
        cols.trade_qty,
        cols.trade_price,
        cols.trade_time,
        cols.trade_date,
    ).rename({cols.bid_seq: cols.order_seq}).with_columns(pl.lit("B").alias(cols.side))
    sell = deals.filter(pl.col("RestoredSide") == "S").select(
        cols.channel,
        cols.offer_seq,
        cols.security,
        cols.trade_qty,
        cols.trade_price,
        cols.trade_time,
        cols.trade_date,
    ).rename({cols.offer_seq: cols.order_seq}).with_columns(pl.lit("S").alias(cols.side))
    deal_summary = pl.concat([buy, sell]).group_by(
        [cols.channel, cols.order_seq, cols.security, cols.side]
    ).agg(
        pl.sum(cols.trade_qty).alias("DealQty"),
        pl.last(cols.trade_price).alias(cols.order_price),
        pl.last(cols.trade_time).alias(cols.order_time),
        pl.last(cols.trade_date).alias(cols.trade_date),
    )
    keys = [cols.channel, cols.order_seq, cols.security, cols.side]
    partial = order_df.join(
        deal_summary.select(*keys, "DealQty"),
        on=keys,
        how="inner",
    ).with_columns(
        (pl.col(cols.order_qty) + pl.col("DealQty")).alias(cols.order_qty),
        pl.lit("partial_active_trade").alias("OrderStatus"),
    ).drop("DealQty")
    untouched = order_df.join(
        deal_summary.select(*keys),
        on=keys,
        how="anti",
    ).with_columns(pl.lit("passive_or_untouched").alias("OrderStatus"))
    untouched = untouched.select(partial.columns)
    new = deal_summary.join(
        order_df.select(*keys),
        on=keys,
        how="anti",
    ).with_columns(
        pl.lit("fully_active_trade").alias("OrderStatus"),
        pl.lit(50, dtype=pl.Int64).alias(cols.order_type),
        pl.lit(0, dtype=pl.Int64).alias(cols.seq_no),
        pl.lit(0, dtype=pl.Int64).alias("__index_level_0__"),
    ).rename({"DealQty": cols.order_qty})
    new = new.select(partial.columns)
    return pl.concat([partial, untouched, new])


def restore_sse_order_files(
    order_path: str | Path,
    trade_path: str | Path,
    columns: SseOrderRestoreColumns | None = None,
):
    pl = _polars()
    return restore_sse_order_table(
        _read_polars(order_path, pl),
        _read_polars(trade_path, pl),
        columns,
    )


# Backward-compatible name for older research scripts.
def RestoreOrder(order_dff, deal_dff):
    return restore_sse_order_table(order_dff, deal_dff)


def _read_polars(path: str | Path, pl):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path)
    if suffix in {".feather", ".ftr"}:
        return pl.read_ipc(path)
    raise ValueError(f"unsupported SSE order restore table format: {path}")


def _ensure_polars(value):
    pl = _polars()
    if isinstance(value, pl.DataFrame):
        return value
    return pl.from_pandas(value)


def _polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("SSE order restore requires polars: pip install polars") from exc
    return pl