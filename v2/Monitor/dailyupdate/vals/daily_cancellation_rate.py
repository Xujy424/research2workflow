"""独立指标：daily_cancellation_rate；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import cancellation_rate


def daily_cancellation_rate(context, date):
    """每日撤单金额/有效申报金额；委托初始名义额按该日委托表计算。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    n = len(context.ticks)
    import polars as pl
    daily_frames = []
    order_frames = []
    cancel_frames = []
    for exchange in ("sh", "sz"):
        order_path = context.l2_path(date, exchange, "wt")
        cancel_path = context.l2_path(date, exchange, "cancel")
        if not order_path.exists():
            continue
        orders = pl.scan_parquet(order_path).select(
            "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price", "OrderQty", "TransactTime"
        ).filter(
            (pl.col("Price") > 0) & (pl.col("OrderQty") > 0)
            & (pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00")
        )
        order_totals = orders.with_columns(
            pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
            (pl.col("Price") * pl.col("OrderQty")).alias("_submitted"),
        ).group_by("_tick").agg(pl.sum("_submitted").alias("submitted"))
        order_frames.append(order_totals)
        cancel_totals = None
        if cancel_path.exists():
            cancels = pl.scan_parquet(cancel_path).select(
                "ChannelNo", "SecurityID", "BidApplSeqNum", "OfferApplSeqNum", "OrderQty", "TransactTime"
            ).filter(pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00")
            buy_cancel = cancels.select(
                "ChannelNo", "SecurityID", pl.col("BidApplSeqNum").alias("ApplSeqNum"),
                pl.lit(1).alias("Side"), "OrderQty",
            )
            sell_cancel = cancels.select(
                "ChannelNo", "SecurityID", pl.col("OfferApplSeqNum").alias("ApplSeqNum"),
                pl.lit(-1).alias("Side"), "OrderQty",
            )
            cancel_legs = pl.concat([buy_cancel, sell_cancel]).filter(
                pl.col("ApplSeqNum").is_not_null() & (pl.col("ApplSeqNum") != 0)
            )
            order_prices = orders.select("ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price")
            cancel_totals = cancel_legs.join(
                order_prices, on=["ChannelNo", "SecurityID", "Side", "ApplSeqNum"], how="inner"
            ).with_columns(
                pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
                (pl.col("Price") * pl.col("OrderQty")).alias("_cancelled"),
            ).group_by("_tick").agg(pl.sum("_cancelled").alias("cancelled"))
        if cancel_totals is None:
            cancel_totals = pl.DataFrame(schema={"_tick": pl.String, "cancelled": pl.Float64}).lazy()
        cancel_frames.append(cancel_totals)
    if not order_frames:
        return np.full(n, np.nan)
    positions = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
    order_totals = pl.concat(order_frames).group_by("_tick").agg(pl.sum("submitted")).rename(
        {"submitted": "submitted"}
    ).collect(engine="streaming")
    cancel_totals = pl.concat(cancel_frames).group_by("_tick").agg(pl.sum("cancelled")).collect(engine="streaming")
    codes = order_totals["_tick"].to_numpy()
    indices = positions.get_indexer(codes)
    matched = indices >= 0
    submitted = np.zeros(n)
    submitted[indices[matched]] = order_totals["submitted"].to_numpy().astype(float)[matched]
    cancel_indices = positions.get_indexer(cancel_totals["_tick"].to_numpy())
    cancel_matched = cancel_indices >= 0
    cancelled = np.zeros(n)
    np.add.at(cancelled, cancel_indices[cancel_matched], cancel_totals["cancelled"].to_numpy().astype(float)[cancel_matched])
    observed = np.zeros(n, dtype=bool)
    observed[np.unique(indices[matched])] = True
    result = cancellation_rate(cancelled, submitted)
    result[~observed] = np.nan
    return result
