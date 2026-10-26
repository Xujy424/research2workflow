"""Read-only real/synthetic timings; never writes factor matrices to data root.

Examples:
  python -m v2.UpdateAlpha.tests.benchmark_volume_dispersion --factors bayesHighVol_onset --dates 2026-06-29 2026-06-30 --mode rolling
  python -m v2.UpdateAlpha.tests.benchmark_volume_dispersion --synthetic
"""

import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from v2.UpdateAlpha import get_factor_spec
from v2.UpdateAlpha.tests.test_volume_dispersion import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/")
    parser.add_argument("--factors", nargs="+", default=NAMES)
    parser.add_argument("--dates", nargs="+", default=["2026-06-30"])
    parser.add_argument("--mode", choices=("daily", "rolling"), default="daily")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output/benchmarks/timing.json")
    args = parser.parse_args()
    fixture = None
    if args.synthetic:
        from .test_factor_workflow import NewFactorWorkflowTests
        fixture = NewFactorWorkflowTests
        fixture.setUpClass()
        args.root, args.dates = fixture.root, [str(fixture.dates[9])]
    rows = []
    try:
        for name in args.factors:
            spec = get_factor_spec(name)
            config = inspect.signature(spec.context_class).parameters["config"].default
            overrides = {"show_progress": not args.quiet}
            if "workers" in config.__dataclass_fields__:
                overrides["workers"] = args.workers
            config = replace(config, **overrides)
            with spec.create_context(root=args.root, config=config) as context:
                factor = spec.create_factor(context)
                read_seconds = []
                read = context._read

                def timed_read(*a, **kw):
                    start = perf_counter()
                    result = read(*a, **kw)
                    read_seconds.append(perf_counter() - start)
                    return result

                context._read = timed_read
                for date in args.dates:
                    read_seconds.clear()
                    print(f"START {spec.name} {date} mode={args.mode} workers={args.workers}", flush=True)
                    day = context.data.axis.date_position(date)
                    start = perf_counter()
                    values = (factor.calculate(date) if args.mode == "rolling" else
                              context.daily_features(day)[spec.name])
                    seconds = perf_counter() - start
                    row = {
                        "factor": spec.name, "date": date, "mode": args.mode,
                        "synthetic": args.synthetic, "requested_workers": args.workers,
                        "workers": getattr(getattr(context, "_executor", None), "_max_workers", 1),
                        "stocks": context.data.axis.tick_count,
                        "finite": int(np.isfinite(values).sum()),
                        "seconds": seconds, "minute_read_seconds": sum(read_seconds),
                        "source_days_read": len(read_seconds) // (3 if "m_essentials/amount" in spec.factor_class.dependencies else 2),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        if fixture is not None:
            fixture.tearDownClass()


if __name__ == "__main__":
    main()
