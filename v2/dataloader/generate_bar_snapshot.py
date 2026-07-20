"""Rebuild L2 order-book snapshots at arbitrary intraday bar endpoints.

The input tables are the normalized outputs of ``preprocess_l2data.py``.  A
snapshot is right-closed: an event whose ``TransactTime`` equals a bar endpoint
is included in that bar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from tqdm import tqdm
import warnings

import polars as pl


ORDER_KEY = ["ChannelNo", "SecurityID", "Side", "ApplSeqNum"]


def _as_time(value: dt.time | dt.datetime | str) -> dt.time:
    if isinstance(value, dt.datetime):
        return value.time()
    if isinstance(value, dt.time):
        return value
    return dt.time.fromisoformat(value)


def make_bar_times(
    interval: dt.timedelta | str = "1m",
    sessions: Sequence[tuple[str | dt.time, str | dt.time]] = (
        ("09:15:00", "11:30:00"),
        ("13:00:00", "15:00:00"),
    ),
) -> list[dt.time]:
    if isinstance(interval, str):
        unit = interval[-1].lower()
        scale = {"s": 1, "m": 60, "h": 3600}.get(unit)
        if scale is None:
            raise ValueError("interval must end in s, m or h")
        interval = dt.timedelta(seconds=float(interval[:-1]) * scale)
    if interval <= dt.timedelta(0):
        raise ValueError("interval must be positive")

    anchor = dt.date(2000, 4, 24)
    result: list[dt.time] = []
    for start, end in sessions:
        cursor = dt.datetime.combine(anchor, _as_time(start)) + interval
        finish = dt.datetime.combine(anchor, _as_time(end))
        while cursor < finish:
            result.append(cursor.time())
            cursor += interval
        result.append(finish.time())
    return sorted(set(result))


def prepare_events(
    orders: pl.DataFrame,
    trades: pl.DataFrame,
    cancels: pl.DataFrame,
) -> pl.DataFrame:
    """Return normalized visible add/reduce events sorted by event time."""
    reductions = pl.concat([trades, cancels], how="diagonal_relaxed")
    optional_order_columns = [
        column for column in ("OrdType", "OrderStatus", "SeqNo")
        if column in orders.columns
    ]
    order_info = orders.select(
        ORDER_KEY + ["Price", "OrderQty", "TransactTime"] + optional_order_columns
    )
    if "OrdType" not in order_info.columns:
        order_info = order_info.with_columns(pl.lit(None).alias("OrdType"))
    if "OrderStatus" not in order_info.columns:
        order_info = order_info.with_columns(pl.lit(None).alias("OrderStatus"))

    duplicate_keys = order_info.group_by(ORDER_KEY).len().filter(pl.col("len") > 1).height
    if duplicate_keys:
        raise ValueError(f"order table contains {duplicate_keys} duplicated ORDER_KEY values")

    legs: list[pl.DataFrame] = []
    for side, seq_col in ((1, "BidApplSeqNum"), (-1, "OfferApplSeqNum")):
        legs.append(
            reductions.select(
                pl.col("ApplSeqNum").alias("SortNo"),
                "ChannelNo",
                "SecurityID",
                pl.col("Side").alias("_AggressorSide"),
                pl.lit(side).cast(pl.Int8).alias("Side"),
                pl.col(seq_col).alias("ApplSeqNum"),
                pl.col("OrderQty").alias("QtyDelta"),
                pl.col("TransactTime").alias("EventTime"),
            ).filter(pl.col("ApplSeqNum").is_not_null() & (pl.col("ApplSeqNum") != 0))
        )
    reduction_legs = pl.concat(legs, how="diagonal_relaxed")
    joined_reductions = reduction_legs.join(
        order_info.select(
            ORDER_KEY + ["Price", "OrdType", "OrderStatus",
                         pl.lit(True).alias("_OrderMatched")]
        ),
        on=ORDER_KEY,
        how="left",
    )
    unusable_count = joined_reductions.filter(
        (pl.col("_OrderMatched").is_null() | pl.col("Price").is_null())
        & (pl.col("Side") != pl.col("_AggressorSide"))
    ).height
    if unusable_count:
        warnings.warn(
            f"{unusable_count} passive reduction legs cannot be mapped to an order price",
            RuntimeWarning,
            stacklevel=2,
        )

    reduce_events = joined_reductions.filter(
        pl.col("_OrderMatched").is_not_null()
        & pl.col("Price").is_not_null()
        & (pl.col("Price") > 0)
    ).select(
        "EventTime",
        "SortNo",
        pl.lit(1).cast(pl.Int8).alias("EventType"),
        "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price",
        "OrdType", "OrderStatus",
        (-pl.col("QtyDelta")).alias("QtyDelta"),
    )

    add_events = order_info.filter(
        pl.col("Price").is_not_null()
        & (pl.col("Price") > 0)
        & (pl.col("OrderQty") > 0)
    ).select(
        pl.col("TransactTime").alias("EventTime"),
        pl.col("ApplSeqNum").alias("SortNo"),
        pl.lit(0).cast(pl.Int8).alias("EventType"),
        "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price",
        "OrdType", "OrderStatus",
        pl.col("OrderQty").alias("QtyDelta"),
    )
    return pl.concat([add_events, reduce_events], how="vertical_relaxed").sort(
        ["EventTime", "SortNo", "EventType", "ChannelNo", "ApplSeqNum"]
    )


def generate_bar_snapshots(
    orders: pl.DataFrame,
    trades: pl.DataFrame,
    cancels: pl.DataFrame,
    bar_times: Iterable[dt.time | dt.datetime | str],
    topn: int = 10,
    securities: Iterable[int | str] | None = None,
    wide: bool = True,
    status_events: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if topn <= 0: raise ValueError("topn must be positive")
    bars = sorted({_as_time(t) for t in bar_times})
    if not bars: raise ValueError("bar_times cannot be empty")

    wanted = set(securities) if securities is not None else None
    if wanted is not None:
        wanted_list = list(wanted)
        orders = orders.filter(pl.col("SecurityID").is_in(wanted_list))
        trades = trades.filter(pl.col("SecurityID").is_in(wanted_list))
        cancels = cancels.filter(pl.col("SecurityID").is_in(wanted_list))
        if status_events is not None:
            status_events = status_events.filter(
                pl.col("SecurityID").is_in(wanted_list)
            )

    events = prepare_events(orders, trades, cancels)
    if status_events is not None and status_events.height:
        status_code_column = (
            "TradingPhaseCode"
            if "TradingPhaseCode" in status_events.columns
            else "TickBSFlag"
        )
        status_rows = status_events.select(
            pl.col("TransactTime").alias("EventTime"),
            pl.col("ApplSeqNum").alias("SortNo"),
            pl.lit(2).cast(pl.Int8).alias("EventType"),
            "ChannelNo", "SecurityID",
            pl.lit(None).cast(pl.Int8).alias("Side"),
            pl.lit(None).alias("ApplSeqNum"),
            pl.lit(None).alias("Price"),
            pl.lit(None).alias("OrdType"),
            pl.col(status_code_column).alias("OrderStatus"),
            pl.lit(0.0).alias("QtyDelta"),
        )
        events = pl.concat([events, status_rows], how="vertical_relaxed").sort(
            ["EventTime", "SortNo", "EventType", "ChannelNo", "ApplSeqNum"]
        )
    all_securities = sorted(
        events.get_column("SecurityID").unique().to_list(),
        key=str,
    )

    # Keep order-level events.  A Shenzhen U order has to be repriced to the
    # same-side best price that exists exactly when that order arrives, so
    # grouping deltas by the raw Price before replay is not valid.
    level_deltas = events

    levels: dict[object, dict[int, dict[object, float]]] = defaultdict(
        lambda: {1: defaultdict(float), -1: defaultdict(float)}
    )
    remaining: dict[tuple, float] = {}
    effective_prices: dict[tuple, object] = {}
    top_cache: dict[object, tuple[list[tuple], list[tuple]]] = {}
    if wide:
        output: dict[str, list] = {"BarTime": [], "SecurityID": []}
        for level in range(1, topn + 1):
            for column in ("BidPrice", "BidQty", "AskPrice", "AskQty"):
                output[f"{column}{level}"] = []
    else:
        output = {column: [] for column in (
            "BarTime", "SecurityID", "Level", "BidPrice", "BidQty", "AskPrice", "AskQty"
        )}

    delta_iter = iter(level_deltas.iter_rows(named=False))
    delta_event = next(delta_iter, None)
    negative_level_count = 0
    crossed_samples: list[tuple] = []
    for bar in tqdm(bars):
        changed_securities: set[object] = set()
        while delta_event is not None and delta_event[0] <= bar:
            (
                _, _, event_type, channel, security, side, appl_seq_num, price,
                ord_type, order_status, delta,
            ) = delta_event
            if event_type == 2:
                phase = (
                    order_status.decode(errors="ignore")
                    if isinstance(order_status, bytes)
                    else str(order_status)
                ).strip()
                if phase in {"SUSP", "CLOSE", "ENDTR"}:
                    levels.pop(security, None)
                    top_cache[security] = ([], [])
                    stale_keys = [key for key in remaining if key[1] == security]
                    for stale_key in stale_keys:
                        remaining.pop(stale_key, None)
                        effective_prices.pop(stale_key, None)
                    changed_securities.add(security)
                delta_event = next(delta_iter, None)
                continue
            key = (channel, security, side, appl_seq_num)
            normalized_ord_type = (
                ord_type.decode(errors="ignore") if isinstance(ord_type, bytes)
                else str(ord_type)
            )

            if event_type == 0:
                # Shenzhen 1/ASCII-49 is a market order and never rests at its
                # raw protection price.  U/ASCII-85 rests at the current
                # same-side best, not at the raw Price field.
                if normalized_ord_type in {"1", "49"}:
                    applied = 0.0
                else:
                    effective_price = price
                    if normalized_ord_type in {"U", "85"}:
                        visible_prices = [
                            level_price
                            for level_price, level_qty in levels[security][side].items()
                            if level_qty > 0
                        ]
                        if visible_prices:
                            effective_price = (
                                max(visible_prices) if side == 1 else min(visible_prices)
                            )
                        else:
                            effective_price = None
                    applied = max(float(delta), 0.0) if effective_price is not None else 0.0
                    if applied:
                        effective_prices[key] = effective_price
                remaining[key] = applied
            else:
                available = remaining.get(key, 0.0)
                applied = -min(max(-float(delta), 0.0), available)
                remaining[key] = available + applied

            price = effective_prices.get(key)
            if price is None or abs(applied) < 1e-12:
                delta_event = next(delta_iter, None)
                continue
            side_levels = levels[security][side]
            updated_qty = side_levels[price] + applied
            if updated_qty < -1e-9:
                negative_level_count += 1
                updated_qty = 0.0
            if abs(updated_qty) < 1e-12:
                side_levels.pop(price, None)
            else:
                side_levels[price] = updated_qty
            changed_securities.add(security)
            delta_event = next(delta_iter, None)

        for security in changed_securities:
            bids = heapq.nlargest(
                topn,
                ((price, qty) for price, qty in levels[security][1].items() if qty > 0),
                key=lambda item: item[0],
            )
            asks = heapq.nsmallest(
                topn,
                ((price, qty) for price, qty in levels[security][-1].items() if qty > 0),
                key=lambda item: item[0],
            )
            top_cache[security] = (bids, asks)
            if (
                dt.time(9, 30) <= bar < dt.time(14, 57)
                and bids and asks and bids[0][0] >= asks[0][0]
                and len(crossed_samples) < 20
            ):
                crossed_samples.append((bar, security, bids[0][0], asks[0][0]))

        for security in all_securities:
            bids, asks = top_cache.get(security, ([], []))
            if wide:
                output["BarTime"].append(bar)
                output["SecurityID"].append(security)
                for level in range(1, topn + 1):
                    bid = bids[level - 1] if level <= len(bids) else (None, None)
                    ask = asks[level - 1] if level <= len(asks) else (None, None)
                    output[f"BidPrice{level}"].append(bid[0])
                    output[f"BidQty{level}"].append(bid[1])
                    output[f"AskPrice{level}"].append(ask[0])
                    output[f"AskQty{level}"].append(ask[1])
            else:
                for level in range(1, topn + 1):
                    bid = bids[level - 1] if level <= len(bids) else (None, None)
                    ask = asks[level - 1] if level <= len(asks) else (None, None)
                    for column, value in (
                        ("BarTime", bar), ("SecurityID", security), ("Level", level),
                        ("BidPrice", bid[0]), ("BidQty", bid[1]),
                        ("AskPrice", ask[0]), ("AskQty", ask[1]),
                    ):
                        output[column].append(value)

    if negative_level_count:
        warnings.warn(
            f"{negative_level_count} price-level quantities became negative and were clamped to zero",
            RuntimeWarning,
            stacklevel=2,
        )
    if crossed_samples:
        warnings.warn(
            f"crossed books detected during continuous auction; first samples: {crossed_samples}",
            RuntimeWarning,
            stacklevel=2,
        )
    return pl.DataFrame(output).sort(["SecurityID", "BarTime"])


def generate_from_proc(
    root: str | Path,
    date: str,
    exchange: str,
    interval: str = "1m",
    topn: int = 10,
) -> pl.DataFrame:
    exchange = exchange.lower()
    if exchange not in {"sh", "sz"}:
        raise ValueError("exchange must be 'sh' or 'sz'")
    folder = Path(root) / "proc" / date.replace("-", "")
    status_path = folder / f"{exchange}status.pq"
    return generate_bar_snapshots(
        pl.read_parquet(folder / f"{exchange}wt.pq"),
        pl.read_parquet(folder / f"{exchange}cj.pq"),
        pl.read_parquet(folder / f"{exchange}cancel.pq"),
        make_bar_times(interval),
        topn=topn,
        status_events=pl.read_parquet(status_path) if status_path.exists() else None,
    )


if __name__ == "__main__":
    # IDE 手动运行配置：改为 True 后，直接修改下面参数并点击“运行”。
    # 保持为 False 时，仍然使用下方 argparse 命令行参数。
    USE_MANUAL_CONFIG = True
    MANUAL_CONFIG = {
        "root": "/data/xujiayi/xjy/l2",
        "date": "20260624",
        "exchange": "sz",       # "sh" 或 "sz"
        "interval": "1m",       # 例如 "30s"、"1m"、"5m"
        "topn": 10,
        "output": None,          # None 表示保存到默认路径
    }

    parser = argparse.ArgumentParser(description="Generate intraday L2 top-N snapshots")
    parser.add_argument("--root", required=True, help="L2 root containing proc/YYYYMMDD")
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--exchange", required=True, choices=("sh", "sz"))
    parser.add_argument("--interval", default="1m", help="e.g. 30s, 1m, 5m")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--output")
    if USE_MANUAL_CONFIG:
        args = argparse.Namespace(**MANUAL_CONFIG)
    else:
        args = parser.parse_args()

    result = generate_from_proc(args.root, args.date, args.exchange, args.interval, args.topn)
    normalized_date = args.date.replace("-", "")
    output = Path(args.output) if args.output else (
        Path(args.root) / "proc" / normalized_date / f"{args.exchange}shot_{args.interval}.pq"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="gzip")
    print(f"saved {result.height} rows to {output}")
