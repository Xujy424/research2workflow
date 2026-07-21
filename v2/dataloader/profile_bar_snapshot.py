"""Lightweight performance diagnostics for generate_bar_snapshot.py.

Run this file directly.  By default it profiles the 300 most active
securities, which is large enough to expose bottlenecks without loading the
whole market into the generated snapshot result.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime as dt
import pstats
from pathlib import Path
from time import perf_counter

import polars as pl

import generate_bar_snapshot as snapshot


def _size_mb(frame: pl.DataFrame | None) -> float:
    return 0.0 if frame is None else frame.estimated_size("mb")


def _print_distribution(name: str, counts: pl.DataFrame) -> None:
    if counts.is_empty():
        print(f"{name}: empty")
        return
    values = counts.get_column("len")
    print(
        f"{name}: bars={counts.height:,}, "
        f"p50={values.quantile(0.50):,.0f}, "
        f"p90={values.quantile(0.90):,.0f}, "
        f"p99={values.quantile(0.99):,.0f}, "
        f"max={values.max():,}"
    )
    print("largest bars:")
    print(counts.sort("len", descending=True).head(10))


def _select_securities(
    order_path: Path,
    securities: list[int] | None,
    sample_size: int | None,
) -> list[int] | None:
    if securities:
        return securities
    if sample_size is None:
        return None
    return (
        pl.scan_parquet(order_path)
        .group_by("SecurityID")
        .len()
        .sort("len", descending=True)
        .head(sample_size)
        .select("SecurityID")
        .collect()
        .get_column("SecurityID")
        .to_list()
    )


def _read_table(path: Path, securities: list[int] | None) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    if securities is not None:
        scan = scan.filter(pl.col("SecurityID").is_in(securities))
    return scan.collect()


def _append_status_events(
    events: pl.DataFrame,
    status_events: pl.DataFrame | None,
) -> pl.DataFrame:
    if status_events is None or status_events.is_empty():
        return events
    status_column = (
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
        pl.col(status_column).alias("OrderStatus"),
        pl.lit(0.0).alias("QtyDelta"),
    )
    return pl.concat([events, status_rows], how="vertical_relaxed")


def run_profile(
    root: Path,
    date: str,
    exchange: str,
    interval: str,
    topn: int,
    sample_size: int | None,
    securities: list[int] | None,
    run_full: bool,
) -> None:
    folder = root / "proc" / date.replace("-", "")
    order_path = folder / f"{exchange}wt.pq"
    trade_path = folder / f"{exchange}cj.pq"
    cancel_path = folder / f"{exchange}cancel.pq"
    status_path = folder / f"{exchange}status.pq"

    selected = _select_securities(order_path, securities, sample_size)
    print("=" * 78)
    print("snapshot performance diagnostics")
    print(f"folder: {folder}")
    print(f"exchange: {exchange}; interval: {interval}; topn: {topn}")
    print("securities:", "all" if selected is None else len(selected))
    print("=" * 78)

    started = perf_counter()
    orders = _read_table(order_path, selected)
    trades = _read_table(trade_path, selected)
    cancels = _read_table(cancel_path, selected)
    status_events = (
        _read_table(status_path, selected) if status_path.exists() else None
    )
    loaded = perf_counter()

    bars = snapshot.make_bar_times(interval)
    events = snapshot.prepare_events(
        orders, trades, cancels, sort_events=False
    )
    events = _append_status_events(events, status_events)
    prepared = perf_counter()

    bar_actions, dynamic_events = snapshot._prepare_bar_actions(events, bars)
    actions_ready = perf_counter()

    u_orders = (
        orders.filter(
            pl.col("OrdType")
            .cast(pl.String, strict=False)
            .fill_null("")
            .is_in(["U", "85"])
        )
        if "OrdType" in orders.columns else orders.head(0)
    )
    security_count = orders.get_column("SecurityID").n_unique()
    u_security_count = u_orders.get_column("SecurityID").n_unique()
    dynamic_ratio = dynamic_events.height / max(events.height, 1)
    static_count = events.height - dynamic_events.height
    compression = static_count / max(bar_actions.height, 1)

    print("\nTIMING")
    print(f"load parquet:          {loaded - started:10.3f}s")
    print(f"prepare_events:        {prepared - loaded:10.3f}s")
    print(f"prepare_bar_actions:   {actions_ready - prepared:10.3f}s")
    print(f"pre-replay total:      {actions_ready - started:10.3f}s")

    print("\nCOUNTS")
    print(f"orders:                {orders.height:12,}")
    print(f"trades:                {trades.height:12,}")
    print(f"cancels:               {cancels.height:12,}")
    print(f"events:                {events.height:12,}")
    print(f"dynamic events:        {dynamic_events.height:12,}")
    print(f"bar actions:           {bar_actions.height:12,}")
    print(f"dynamic event ratio:   {dynamic_ratio:12.2%}")
    print(f"static compression:    {compression:12.2f}x")
    print(f"securities:            {security_count:12,}")
    print(f"U securities:          {u_security_count:12,}")
    print(
        f"U security ratio:      "
        f"{u_security_count / max(security_count, 1):12.2%}"
    )
    print(f"U orders:              {u_orders.height:12,}")

    print("\nESTIMATED DATAFRAME MEMORY")
    for name, frame in (
        ("orders", orders),
        ("trades", trades),
        ("cancels", cancels),
        ("status", status_events),
        ("events", events),
        ("bar_actions", bar_actions),
        ("dynamic_events", dynamic_events),
    ):
        print(f"{name:20s} {_size_mb(frame):10.2f} MB")

    action_counts = (
        bar_actions.group_by("EventTime").len().sort("EventTime")
    )
    bar_frame = pl.DataFrame({"BarTime": bars}).sort("BarTime")
    dynamic_counts = (
        dynamic_events.sort("EventTime")
        .join_asof(
            bar_frame,
            left_on="EventTime",
            right_on="BarTime",
            strategy="forward",
        )
        .filter(pl.col("BarTime").is_not_null())
        .group_by("BarTime")
        .len()
        .sort("BarTime")
    )
    print("\nWORK PER BAR")
    _print_distribution("bar actions", action_counts)
    _print_distribution("dynamic events", dynamic_counts)

    estimated_rows = security_count * len(bars)
    estimated_cells = estimated_rows * (2 + 4 * topn)
    print("\nOUTPUT LOWER BOUND")
    print(f"bars:                  {len(bars):12,}")
    print(f"result rows:           {estimated_rows:12,}")
    print(f"wide-table cells:      {estimated_cells:12,}")

    if not run_full:
        print("\nFull snapshot benchmark skipped. Use --run-full to enable it.")
        return

    print("\nFULL SNAPSHOT CPROFILE")
    profiler = cProfile.Profile()
    full_started = perf_counter()
    profiler.enable()
    result = snapshot.generate_bar_snapshots(
        orders,
        trades,
        cancels,
        bars,
        topn=topn,
        status_events=status_events,
    )
    profiler.disable()
    full_finished = perf_counter()

    print(f"full runtime:          {full_finished - full_started:10.3f}s")
    print(f"result shape:          {result.shape}")
    print(f"result memory:         {_size_mb(result):10.2f} MB")
    crossed = result.filter(
        pl.col("BidPrice1").is_not_null()
        & pl.col("AskPrice1").is_not_null()
        & (pl.col("BidPrice1") >= pl.col("AskPrice1"))
        & (
            ((pl.col("BarTime") >= pl.time(9, 30))
             & (pl.col("BarTime") < pl.time(11, 30)))
            | ((pl.col("BarTime") >= pl.time(13, 0))
               & (pl.col("BarTime") < pl.time(14, 57)))
        )
    )
    print(f"continuous crossed:    {crossed.height:12,}")
    print("\nTop cumulative-time functions:")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile L2 bar snapshot generation without full-market load"
    )
    parser.add_argument("--root", default="/data/xujiayi/xjy/l2")
    parser.add_argument("--date", default="20260624")
    parser.add_argument("--exchange", choices=("sh", "sz"), default="sz")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Most active securities; use 0 for the full market",
    )
    parser.add_argument(
        "--securities",
        help="Comma-separated SecurityID list; overrides --sample-size",
    )
    parser.add_argument("--run-full", action="store_true")
    args = parser.parse_args()

    security_list = (
        [int(value) for value in args.securities.split(",")]
        if args.securities else None
    )
    run_profile(
        root=Path(args.root),
        date=args.date,
        exchange=args.exchange,
        interval=args.interval,
        topn=args.topn,
        sample_size=None if args.sample_size == 0 else args.sample_size,
        securities=security_list,
        run_full=args.run_full,
    )
