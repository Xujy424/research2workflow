"""Focused tests for lifecycle accounting and persistent regime detection."""
import tempfile
import numpy as np
import pandas as pd
from .example_usage import make_demo_data
from ..schema import MarketData
from ..preprocess import preprocess
from ..features.extended import lifecycle
from ..history import save_observation, load_history, detect_migration
from ..pipeline import run_daily_monitor
from ..statistics.migration import ks_distance


def main():
    orders = pd.DataFrame([
        ["09:30:00", "X", "b", 10, 100, "buy", "submit"],
        ["09:30:00", "X", "s", 10, 40, "sell", "submit"],
        ["09:32:00", "X", "b", 0, 60, "buy", "cancel"],
    ], columns=["time", "symbol", "order_id", "price", "volume", "side", "status"])
    trades = pd.DataFrame([["09:31:00", "X", 10, 40, "b", "s", "buy"]], columns=["time", "symbol", "price", "volume", "buy_order_id", "sell_order_id", "direction"])
    quotes = pd.DataFrame(columns=["time", "symbol", "bid_price1", "ask_price1"])
    data = preprocess(MarketData(orders, trades, quotes), "2026-09-29")
    life, submissions = lifecycle(data)
    assert len(submissions) == 2
    assert np.isclose(life.loc["X", "fill_rate"], .7)
    assert np.isclose(life.loc["X", "cancel_rate"], 60/140)
    assert life.loc["X", "lifetime_seconds"] == 90
    assert ks_distance(pd.Series([1, 2]), pd.Series([3, 4])) == 1
    with tempfile.TemporaryDirectory() as directory:
        rng = np.random.default_rng(7)
        dates = pd.bdate_range(end="2026-09-29", periods=65)
        history = pd.DataFrame({"trade.large_ratio": rng.normal(.2, .01, 65), "flow.imbalance": rng.normal(0, .01, 65)}, index=dates)
        history.iloc[-5:] = [.8, .9]
        migration = detect_migration(history)
        assert migration.overall.iloc[-1] == "NEW_REGIME"
        save_observation(directory, "2026-09-28", pd.Series({"metric": 1.}))
        save_observation(directory, "2026-09-28", pd.Series({"metric": 2.}))
        assert len(load_history(directory)) == 1
        assert load_history(directory).metric.iloc[0] == 2
        report_directory = directory + "/reports"
        first = run_daily_monitor("2026-09-28", make_demo_data("2026-09-28"), output_dir=report_directory)
        second = run_daily_monitor("2026-09-29", make_demo_data(), output_dir=report_directory)
        assert second.dashboard_path.exists()
        assert not second.structural_change.empty
        assert second.baseline.loc["trade.count", "sample_count"] == 1
        assert first.features["trade.active_buy_ratio"].between(0, 1).all()
    print("Extended checks passed")


if __name__ == "__main__":
    main()
