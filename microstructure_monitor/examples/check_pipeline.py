"""Focused contract checks: python -m microstructure_monitor.examples.check_pipeline."""
import tempfile
import numpy as np
import pandas as pd
from .example_usage import make_demo_data
from ..pipeline import run_daily_monitor
from ..preprocess import preprocess
from ..schema import MarketData
from ..statistics import compare_with_history


def main():
    data = make_demo_data()
    unknown = MarketData(data.orders, data.trades.drop(columns="direction"), data.quotes)
    cleaned = preprocess(unknown, "2026-09-29")
    assert cleaned.trades.direction.eq("unknown").all()
    orders = data.orders.copy()
    orders.loc[0, "time"] = pd.Timestamp("2026-09-29 12:00")
    assert len(preprocess(MarketData(orders, data.trades, data.quotes), "2026-09-29").orders) == len(orders) - 1
    today = pd.Series({"metric": 5.0})
    dates = pd.bdate_range(end="2026-09-28", periods=60)
    history = pd.DataFrame({"metric": np.arange(60, dtype=float)}, index=dates)
    expected = compare_with_history(today, history, "2026-09-29")
    contaminated = pd.concat([history, pd.DataFrame({"metric": [1e9]}, index=[pd.Timestamp("2026-09-29")])])
    pd.testing.assert_frame_equal(expected, compare_with_history(today, contaminated, "2026-09-29"))
    assert expected.loc["metric", "ma20"] == 49.5
    with tempfile.TemporaryDirectory() as output:
        result = run_daily_monitor("2026-09-29", data, output_dir=output,
                                   universe_map={"one": ["000001.SZ"]})
        assert result.report_path.exists()
        assert len(result.features) == 2
        assert result.regime["overall"] == "INSUFFICIENT_HISTORY"
        assert set(result.cross_section.index) == {"all", "one"}
        assert np.isclose(result.bars["1min"].amount.sum(), data.trades.eval("price * volume").sum())
    print("Pipeline checks passed")


if __name__ == "__main__":
    main()
