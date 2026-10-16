"""Run with python -m microstructure_monitor.examples.example_usage."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microstructure_monitor import run_daily_monitor
from microstructure_monitor.schema import MarketData


def make_demo_data(date="2026-09-29"):
    rng = np.random.default_rng(42)
    times = pd.date_range(date + " 09:30", periods=120, freq="min").append(
        pd.date_range(date + " 13:00", periods=117, freq="min"))
    parts = []
    for symbol in ("000001.SZ", "600000.SH"):
        prices = 10 + rng.normal(0, .005, len(times)).cumsum()
        parts.append(pd.DataFrame({"symbol": symbol, "time": times, "price": prices,
                                   "volume": rng.integers(100, 80000, len(times)),
                                   "buy_order_id": np.arange(len(times)).astype(str),
                                   "sell_order_id": (np.arange(len(times)) + 1000).astype(str),
                                   "direction": rng.choice(["buy", "sell"], len(times))}))
    trades = pd.concat(parts, ignore_index=True)
    orders = trades.rename(columns={"buy_order_id": "order_id", "direction": "side"}).copy()
    orders["status"] = "submit"
    quotes = trades[["symbol", "time"]].copy()
    quotes["bid_price1"] = trades.price - .01
    quotes["ask_price1"] = trades.price + .01
    quotes["bid_volume1"] = 10000
    quotes["ask_volume1"] = 12000
    return MarketData(orders, trades, quotes)


def main():
    result = run_daily_monitor("2026-09-29", make_demo_data(), output_dir="microstructure_monitor/output/demo_v5", data_source="synthetic")
    print(result.regime)
    print(f"Report: {result.report_path}")
    return result


if __name__ == "__main__":
    main()
