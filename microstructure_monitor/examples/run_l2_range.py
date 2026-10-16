"""Run available trading-day directories in chronological order."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import pandas as pd
from .run_l2_day import load_proc_day
from ..pipeline import run_daily_monitor
from ..quality import ensure_history_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/stock/l2")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", default=["000001.SZ", "600000.SH"])
    parser.add_argument("--output", default="microstructure_monitor/output/l2_range_v5")
    parser.add_argument("--metadata-root", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse a completed chronological prefix; always recompute the final date")
    parser.add_argument("--prefetch", type=int, default=1, choices=(1, 2, 3), help="Bounded concurrent input reads; feature/history writes remain chronological")
    parser.add_argument("--read-strategy", default="auto", choices=("auto", "prefiltered"), help="Optional installed-Polars late materialization for highly selective stock reads")
    args = parser.parse_args()
    dates = sorted(p.name for p in (Path(args.root)/"proc").iterdir() if p.is_dir() and args.start.replace("-", "") <= p.name <= args.end.replace("-", ""))
    if not dates:
        raise ValueError("No trading-day directories in requested range")
    identity = "selected:" + ",".join(sorted(args.symbols))
    ensure_history_identity(args.output, args.symbols, identity)
    output = Path(args.output)
    history_path = output/"history.csv"
    completed = set(pd.read_csv(history_path, index_col=0).index.astype(str).str.replace("-", "")) if history_path.exists() else set()
    started = perf_counter()
    prefix = args.resume
    status = {"source": "v2_proc", "symbols": args.symbols, "dates": dates, "completed_dates": []}
    pool = ThreadPoolExecutor(max_workers=args.prefetch)
    pending = {}
    def progress(state, date):
        status.update(state=state, current_date=date, elapsed_seconds=round(perf_counter()-started, 2))
        temporary = output/"run_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(output/"run_status.json")
    for index, date in enumerate(dates):
        artifacts = ("features.csv", "baseline.csv", "memberships.json", "bars_feature_1min.csv")
        if prefix and index < len(dates)-1 and date in completed and all((output/date/name).exists() for name in artifacts):
            status["completed_dates"].append(date)
            progress("resuming", date)
            print(f"Reusing completed date {date}", flush=True)
            continue
        prefix = False
        progress("reading", date)
        print(f"Processing real L2 date {date}", flush=True)
        try:
            for upcoming in dates[index:index+args.prefetch]:
                if upcoming not in pending:
                    pending[upcoming] = pool.submit(load_proc_day, args.root, upcoming, args.symbols, read_strategy=args.read_strategy)
            data = pending.pop(date).result()
            if args.metadata_root:
                from ..data.daily_loader import load_v2_daily
                from dataclasses import replace
                data = replace(data, daily=load_v2_daily(args.metadata_root, date, args.symbols))
            progress("computing", date)
            result = run_daily_monitor(date, data, output_dir=args.output, universe_id=identity, data_source="v2_proc")
        except Exception as error:
            status["error"] = f"{type(error).__name__}: {error}"
            progress("failed", date)
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        status["completed_dates"].append(date)
        progress("running", date)
        print(date, result.regime, result.dashboard_path, flush=True)
    progress("complete", dates[-1])
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
