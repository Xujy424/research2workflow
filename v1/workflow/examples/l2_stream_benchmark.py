"""Measure canonical L2 decoding throughput without running a strategy."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from time import perf_counter

from quant_workflow.trading import CanonicalL2Gateway, PreprocessedL2Bundle


if __name__ == "__main__":
    root = Path("data/l2/canonical/20250102")
    bundle = PreprocessedL2Bundle(
        trading_date=date(2025, 1, 2),
        sse_events=root / "SSE_events.parquet",
        szse_events=root / "SZSE_events.parquet",
        manifest=root / "manifest.json",
    )
    started = perf_counter()
    rows = sum(1 for _ in CanonicalL2Gateway().stream(bundle))
    elapsed = perf_counter() - started
    print(
        f"events={rows:,}, elapsed={elapsed:.3f}s, "
        f"throughput={rows / max(elapsed, 1e-12):,.0f} events/s"
    )
