"""Run available trading-day directories in chronological order."""
import argparse
from pathlib import Path
from .run_l2_day import load_proc_day
from ..pipeline import run_daily_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/stock/l2")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", default=["000001.SZ", "600000.SH"])
    parser.add_argument("--output", default="microstructure_monitor/output/l2_range")
    args = parser.parse_args()
    dates = sorted(p.name for p in (Path(args.root)/"proc").iterdir() if p.is_dir() and args.start.replace("-", "") <= p.name <= args.end.replace("-", ""))
    if not dates:
        raise ValueError("No trading-day directories in requested range")
    for date in dates:
        result = run_daily_monitor(date, load_proc_day(args.root, date, args.symbols), output_dir=args.output)
        print(date, result.regime, result.dashboard_path, flush=True)


if __name__ == "__main__":
    main()
