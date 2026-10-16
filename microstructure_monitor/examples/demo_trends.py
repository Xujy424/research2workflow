"""Generate a synthetic historical trend artifact; never presents it as real L2."""
import numpy as np
import pandas as pd
from pathlib import Path
from .example_usage import make_demo_data
from ..pipeline import run_daily_monitor
from ..history import save_observation


def main():
    directory = Path("microstructure_monitor/output/synthetic_trends_v5")
    sample = run_daily_monitor("2026-06-01", make_demo_data("2026-06-01"), output_dir=directory, data_source="synthetic")
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2026-06-02", periods=65)
    for i, date in enumerate(dates):
        metrics = sample.baseline.raw * (1 + rng.normal(0, .03, len(sample.baseline)))
        metrics["trade.large_ratio"] = .2 + i*.001 + rng.normal(0, .01)
        metrics["flow.imbalance"] = -.1 + rng.normal(0, .03)
        save_observation(directory, date, metrics)
    date = dates[-1] + pd.offsets.BDay(1)
    result = run_daily_monitor(date, make_demo_data(str(date.date())), output_dir=directory, data_source="synthetic")
    print(f"Synthetic trends: {result.dashboard_path}")


if __name__ == "__main__":
    main()
