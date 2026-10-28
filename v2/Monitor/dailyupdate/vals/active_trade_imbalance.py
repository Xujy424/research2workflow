"""独立指标：active_trade_imbalance；支持直接注入数组或时间序列。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import safe_divide


def active_trade_imbalance(context, date):
    """(主动买入成交额-主动卖出成交额)/两侧主动成交额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    n = len(context.ticks)
    positions = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
    import polars as pl
    exchange_frames = []
    for exchange in ("sh", "sz"):
        path = context.l2_path(date, exchange, "cj")
        if not path.exists():
            continue
        frame = pl.scan_parquet(path).select("SecurityID", "Side", "Price", "OrderQty", "TransactTime").filter(
            pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00"
        ).with_columns(
            pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
            pl.when(pl.col("Side") > 0).then(pl.col("Price") * pl.col("OrderQty")).otherwise(0).alias("_buy"),
            pl.when(pl.col("Side") < 0).then(pl.col("Price") * pl.col("OrderQty")).otherwise(0).alias("_sell"),
        ).group_by("_tick").agg(pl.sum("_buy"), pl.sum("_sell"))
        exchange_frames.append(frame)
    if not exchange_frames:
        return np.full(n, np.nan)
    frame = pl.concat(exchange_frames).group_by("_tick").agg(
        pl.sum("_buy"), pl.sum("_sell")
    ).collect(engine="streaming")
    indices = positions.get_indexer(frame["_tick"].to_numpy())
    matched = indices >= 0
    buy, sell = np.zeros(n), np.zeros(n)
    np.add.at(buy, indices[matched], frame["_buy"].to_numpy().astype(float)[matched])
    np.add.at(sell, indices[matched], frame["_sell"].to_numpy().astype(float)[matched])
    observed = np.zeros(n, dtype=bool)
    observed[np.unique(indices[matched])] = True
    result = safe_divide(buy - sell, buy + sell)
    result[~observed] = np.nan
    return result
