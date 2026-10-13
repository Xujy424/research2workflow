"""Load standardized CSV or Parquet market tables."""
from pathlib import Path
import pandas as pd
from ..schema import MarketData


def load_table(pattern, date):
    path = Path(str(pattern).format(date=date))
    if path.suffix.lower() in {".pq", ".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"symbol": str, "order_id": str,
                                    "buy_order_id": str, "sell_order_id": str})


def load_market_data(date, paths):
    daily = load_table(paths["daily_pattern"], date) if paths.get("daily_pattern") else None
    return MarketData(*(load_table(paths[f"{kind}_pattern"], date)
                        for kind in ("order", "trade", "quote")), daily=daily)
