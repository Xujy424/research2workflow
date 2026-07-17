"""Rebuild L2 order-book snapshots at arbitrary intraday bar endpoints.

The input tables are the normalized outputs of ``preprocess_l2data.py``.  A
snapshot is right-closed: an event whose ``TransactTime`` equals a bar endpoint
is included in that bar.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from tqdm import tqdm

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
    """Return a normalized order-level event table in exchange-time order."""
    reductions = pl.concat([trades, cancels], how="diagonal_relaxed")
    order_info = orders.select(ORDER_KEY + ["Price", "OrderQty", "TransactTime"])

    legs: list[pl.DataFrame] = []
    for side, seq_col in ((1, "BidApplSeqNum"), (-1, "OfferApplSeqNum")):
        legs.append(
            reductions.select(
                "ChannelNo",
                "SecurityID",
                pl.lit(side).cast(pl.Int8).alias("Side"),
                pl.col(seq_col).alias("ApplSeqNum"),
                pl.col("OrderQty").alias("QtyDelta"),
                pl.col("TransactTime").alias("EventTime"),
            ).filter(pl.col("ApplSeqNum").is_not_null() & (pl.col("ApplSeqNum") != 0))
        )
    reduce_events = pl.concat(legs, how="diagonal_relaxed").join(
        order_info.select(ORDER_KEY + ["Price"]), on=ORDER_KEY, how="left"
    ).select(
        "EventTime",
        pl.lit(1).cast(pl.Int8).alias("EventType"),
        "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price",
        (-pl.col("QtyDelta")).alias("QtyDelta"),
    )

    add_events = order_info.select(
        pl.col("TransactTime").alias("EventTime"),
        pl.lit(0).cast(pl.Int8).alias("EventType"),
        "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price",
        pl.col("OrderQty").alias("QtyDelta"),
    )
    return pl.concat([add_events, reduce_events], how="vertical_relaxed").sort(
        ["EventTime", "EventType", "ChannelNo", "ApplSeqNum"]
    )


def generate_bar_snapshots(
    orders: pl.DataFrame,
    trades: pl.DataFrame,
    cancels: pl.DataFrame,
    bar_times: Iterable[dt.time | dt.datetime | str],
    topn: int = 10,
    securities: Iterable[int | str] | None = None,
    wide: bool = True,
) -> pl.DataFrame:
    if topn <= 0: raise ValueError("topn must be positive")
    bars = sorted({_as_time(t) for t in bar_times})
    if not bars: raise ValueError("bar_times cannot be empty")

    events = prepare_events(orders, trades, cancels)
    wanted = set(securities) if securities is not None else None
    if wanted is not None:
        events = events.filter(pl.col("SecurityID").is_in(list(wanted)))
    all_securities = sorted(
        events.get_column("SecurityID").unique().to_list(),
        key=str,
    )

    remaining: dict[tuple, float] = {}
    levels: dict[object, dict[int, dict[object, float]]] = defaultdict(
        lambda: {1: defaultdict(float), -1: defaultdict(float)}
    )
    top_cache: dict[object, tuple[list[tuple], list[tuple]]] = {}
    rows: list[dict] = []
    event_iter = iter(events.iter_rows(named=False))
    event = next(event_iter, None)
    for bar in tqdm(bars):
        changed_securities: set[object] = set()
        while event is not None and event[0] <= bar:
            _, kind, channel, security, side, seq, price, delta = event
            key = (channel, security, side, seq)
            if kind == 0:
                applied = max(float(delta), 0.0)
                remaining[key] = remaining.get(key, 0.0) + applied
            else:
                applied = -min(-float(delta), remaining.get(key, 0.0))
                remaining[key] = remaining.get(key, 0.0) + applied
            side_levels = levels[security][side]
            side_levels[price] += applied
            if abs(side_levels[price]) < 1e-12:
                side_levels.pop(price, None)
            changed_securities.add(security)
            event = next(event_iter, None)

        for security in changed_securities:
            bids = sorted(
                ((price, qty) for price, qty in levels[security][1].items() if qty > 0),
                reverse=True,
            )[:topn]
            asks = sorted(
                ((price, qty) for price, qty in levels[security][-1].items() if qty > 0)
            )[:topn]
            top_cache[security] = (bids, asks)

        for security in all_securities:
            bids, asks = top_cache.get(security, ([], []))
            if wide:
                row = {"BarTime": bar, "SecurityID": security}
                for level in range(1, topn + 1):
                    bid = bids[level - 1] if level <= len(bids) else (None, None)
                    ask = asks[level - 1] if level <= len(asks) else (None, None)
                    row[f"BidPrice{level}"] = bid[0]
                    row[f"BidQty{level}"] = bid[1]
                    row[f"AskPrice{level}"] = ask[0]
                    row[f"AskQty{level}"] = ask[1]
                rows.append(row)
            else:
                for level in range(1, topn + 1):
                    bid = bids[level - 1] if level <= len(bids) else (None, None)
                    ask = asks[level - 1] if level <= len(asks) else (None, None)
                    rows.append({"BarTime": bar, "SecurityID": security, "Level": level,
                                 "BidPrice": bid[0], "BidQty": bid[1],
                                 "AskPrice": ask[0], "AskQty": ask[1]})

    return pl.DataFrame(rows).sort(["SecurityID", "BarTime"])


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
    return generate_bar_snapshots(
        pl.read_parquet(folder / f"{exchange}wt.pq"),
        pl.read_parquet(folder / f"{exchange}cj.pq"),
        pl.read_parquet(folder / f"{exchange}cancel.pq"),
        make_bar_times(interval),
        topn=topn,
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
