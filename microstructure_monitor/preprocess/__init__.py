"""Continuous-session cleaning and quote-based trade classification."""
import numpy as np
import pandas as pd
from ..schema import MarketData, require_columns, REQUIRED_ORDER_COLUMNS, REQUIRED_TRADE_COLUMNS, REQUIRED_QUOTE_COLUMNS


def clean(frame, required, name, date):
    require_columns(frame, required, name)
    frame = frame.copy()
    # Time-only vendor fields must be anchored to the requested trading date.
    values = frame.time.astype(str)
    time_only = values.str.match(r"^\d{2}:\d{2}:\d{2}")
    values.loc[time_only] = str(pd.Timestamp(date).date()) + " " + values.loc[time_only]
    frame["time"] = pd.to_datetime(values, errors="coerce", format="mixed")
    frame["symbol"] = frame.symbol.astype(str)
    minute = frame.time.dt.hour * 60 + frame.time.dt.minute
    session = ((minute >= 570) & (minute < 690)) | ((minute >= 780) & (minute < 897))
    frame = frame.loc[session & (frame.time.dt.normalize() == pd.Timestamp(date).normalize())]
    for col in ("price", "volume", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "price" in frame:
        cancellation = frame.get("status", pd.Series("", index=frame.index)).astype(str).str.lower().isin(["cancel", "cancelled", "canceled"])
        frame = frame.loc[((frame.price > 0) | cancellation) & (frame.volume > 0)]
        frame["amount"] = frame.price * frame.volume
    else:
        frame = frame.loc[(frame.bid_price1 > 0) & (frame.ask_price1 >= frame.bid_price1)]
    return frame.sort_values(["symbol", "time"]).reset_index(drop=True)


def normalize_side(series):
    return series.astype(str).str.lower().map({"1": "buy", "-1": "sell", "b": "buy", "s": "sell", "buy": "buy", "sell": "sell", "49": "buy", "50": "sell"}).fillna("unknown")


def preprocess(data, date):
    orders = clean(data.orders, REQUIRED_ORDER_COLUMNS, "orders", date)
    trades = clean(data.trades, REQUIRED_TRADE_COLUMNS, "trades", date)
    quotes = clean(data.quotes, REQUIRED_QUOTE_COLUMNS, "quotes", date)
    orders["side"] = normalize_side(orders.side)
    for column in ("order_id",):
        orders[column] = orders[column].astype(str)
    for column in ("buy_order_id", "sell_order_id"):
        trades[column] = trades[column].astype(str)
    if "direction" in trades:
        trades["direction"] = normalize_side(trades.direction)
    else:
        classified = []
        for symbol, part in trades.groupby("symbol", sort=False):
            quote = quotes.loc[quotes.symbol == symbol, ["time", "bid_price1", "ask_price1"]]
            joined = pd.merge_asof(part, quote, on="time", direction="backward", tolerance=pd.Timedelta("30s"))
            mid = (joined.bid_price1 + joined.ask_price1) / 2
            joined["direction"] = np.select([joined.price > mid, joined.price < mid], ["buy", "sell"], default="unknown")
            classified.append(joined.drop(columns=["bid_price1", "ask_price1"]))
        trades = pd.concat(classified, ignore_index=True) if classified else trades.assign(direction="unknown")
    # Fixed amount thresholds make size classes comparable across dates.
    trades["size"] = pd.cut(trades.amount, [0, 20000, 100000, 500000, np.inf], labels=["small", "medium", "large", "xl"])
    return MarketData(orders, trades, quotes, data.daily)
