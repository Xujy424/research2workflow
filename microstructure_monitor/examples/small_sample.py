"""Three synthetic days, two securities and dated index scopes; no real L2 claim."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import pandas as pd
from .example_usage import make_demo_data
from ..pipeline import run_daily_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="microstructure_monitor/output/small_sample_july_v5")
    args = parser.parse_args()
    groups = {"hs300": ["000001.SZ"], "csi1000": ["600000.SH"]}
    started = perf_counter()
    for day, date in enumerate(pd.bdate_range("2026-07-22", periods=3)):
        date = date.strftime("%Y-%m-%d")
        data = make_demo_data(date)
        trades, orders, quotes = data.trades.copy(), data.orders.copy(), data.quotes.copy()
        closing = trades.time.dt.hour >= 14
        trades.loc[closing, "volume"] = (trades.loc[closing, "volume"]*(1+day*.3)).round()
        orders["volume"] = trades.volume
        small_cap = quotes.symbol.eq("600000.SH")
        quotes.loc[small_cap, "ask_price1"] += day*.004
        # Visible depth depletion and recovery, with an unrecovered final shock.
        for symbol, indices in quotes.groupby("symbol").groups.items():
            indices = list(indices)
            quotes.loc[indices[30:33], "bid_volume1"] = [3000, 6000, 9500]
            quotes.loc[indices[-1], "bid_volume1"] = 3000
        result = run_daily_monitor(date, replace(data, orders=orders, trades=trades, quotes=quotes),
                                   output_dir=args.output, universe_map=groups, data_source="synthetic")
        print(date, result.regime["overall"], "features", result.features.shape)
    print("Elapsed seconds:", round(perf_counter()-started, 3))
    print("Web report:", Path(result.dashboard_path).resolve())
    print("HTML MB:", round(result.dashboard_path.stat().st_size/1e6, 2))


if __name__ == "__main__":
    main()
