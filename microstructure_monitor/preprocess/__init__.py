"""Continuous-session cleaning and quote-based trade classification."""
import numpy as np
import pandas as pd
from ..schema import MarketData, require_columns, REQUIRED_ORDER_COLUMNS, REQUIRED_TRADE_COLUMNS, REQUIRED_QUOTE_COLUMNS


def continuous_mask(frame):
    minute = frame.time.dt.hour * 60 + frame.time.dt.minute
    return ((minute >= 570) & (minute < 690)) | ((minute >= 780) & (minute < 897))


def clean(frame, required, name, date, continuous=True):
    require_columns(frame, required, name)
    frame = frame.copy()
    # Time-only vendor fields must be anchored to the requested trading date.
    values = frame.time.astype(str)
    time_only = values.str.match(r"^\d{2}:\d{2}:\d{2}")
    values.loc[time_only] = str(pd.Timestamp(date).date()) + " " + values.loc[time_only]
    frame["time"] = pd.to_datetime(values, errors="coerce", format="mixed")
    frame["symbol"] = frame.symbol.astype(str)
    minute = frame.time.dt.hour * 60 + frame.time.dt.minute
    session = continuous_mask(frame) if continuous else frame.time.notna()
    frame = frame.loc[session & (frame.time.dt.normalize() == pd.Timestamp(date).normalize())]
    for col in ("price", "volume", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "price" in frame:
        cancellation = frame.get("status", pd.Series("", index=frame.index)).astype(str).str.lower().isin(["cancel", "cancelled", "canceled"])
        valid_price = frame.price >= 0 if name == "orders" else frame.price > 0
        frame = frame.loc[((valid_price & np.isfinite(frame.price)) | cancellation) & np.isfinite(frame.volume) & (frame.volume > 0)]
        frame["amount"] = (frame.price * frame.volume).where(frame.price > 0)
    else:
        frame = frame.loc[np.isfinite(frame.bid_price1) & np.isfinite(frame.ask_price1) & (frame.bid_price1 > 0) & (frame.ask_price1 >= frame.bid_price1)]
        for column in ("bid_volume1", "ask_volume1"):
            values = frame.get(column, pd.Series(np.nan, index=frame.index))
            frame[column] = values.where(np.isfinite(values) & (values >= 0))
    return frame.sort_values(["symbol", "time"]).reset_index(drop=True)


def normalize_side(series):
    return series.astype(str).str.lower().str.replace(r"\.0$", "", regex=True).map({"1": "buy", "-1": "sell", "b": "buy", "s": "sell", "buy": "buy", "sell": "sell", "49": "buy", "50": "sell"}).fillna("unknown")


def preprocess(data, date):
    orders = clean(data.orders, REQUIRED_ORDER_COLUMNS, "orders", date, continuous=False)
    trades = clean(data.trades, REQUIRED_TRADE_COLUMNS, "trades", date, continuous=False)
    quotes = clean(data.quotes, REQUIRED_QUOTE_COLUMNS, "quotes", date, continuous=False)
    orders["side"] = normalize_side(orders.side)
    for column in ("order_id",):
        orders[column] = orders[column].astype(str)
    for column in ("buy_order_id", "sell_order_id"):
        trades[column] = trades[column].astype(str)
    if "status" in orders:
        cancel = orders.status.astype(str).str.lower().isin(["cancel", "cancelled", "canceled"])
        keys = ["symbol", "side", "order_id"]
        original = orders.loc[~cancel].drop_duplicates(keys).set_index(keys).price
        inferred = original.reindex(pd.MultiIndex.from_frame(orders[keys])).to_numpy()
        replace = cancel & ((orders.price <= 0) | ~np.isfinite(orders.price))
        orders.loc[replace, "price"] = inferred[replace]
        orders["amount"] = (orders.price*orders.volume).where((orders.price > 0) & np.isfinite(orders.price))
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
    orders["size"] = pd.cut(orders.amount, [0, 20000, 100000, 500000, np.inf], labels=["small", "medium", "large", "xl"])
    trades["active_side"] = trades.direction
    trades["passive_side"] = trades.direction.map({"buy": "sell", "sell": "buy"}).fillna("unknown")
    full_orders, full_trades = orders.copy(), trades.copy()
    tables = {"orders": orders, "trades": trades, "quotes": quotes}
    auctions, rows = {}, []
    continuous_tables = {}
    for name, frame in tables.items():
        mask = continuous_mask(frame)
        minutes = frame.time.dt.hour*60 + frame.time.dt.minute
        auction_mask = ((minutes >= 555) & (minutes < 570)) | ((minutes >= 897) & (minutes <= 900))
        auctions[name] = frame.loc[auction_mask].copy()
        continuous_tables[name] = frame.loc[mask].reset_index(drop=True)
        rows.append({"table": name, "input_rows": len(getattr(data, name)), "valid_day_rows": len(frame), "invalid_rows": len(getattr(data, name))-len(frame), "continuous_rows": int(mask.sum()), "auction_rows": int(auction_mask.sum())})
    return MarketData(continuous_tables["orders"], continuous_tables["trades"], continuous_tables["quotes"], data.daily,
                      full_orders, full_trades, auctions, pd.DataFrame(rows).set_index("table"))
