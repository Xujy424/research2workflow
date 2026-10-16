"""Session-aware aggregation with additive numerators and recomputed ratios."""
import numpy as np
import pandas as pd


def minute_returns(minute):
    previous = minute.groupby("symbol").close.shift()
    previous_time = minute.groupby("symbol").time.shift()
    adjacent = (minute.time - previous_time).eq(pd.Timedelta("1min"))
    return (minute.close / previous - 1).where(adjacent)


def ratios(frame):
    frame = frame.copy()
    frame["oi"] = (frame.buy_order_amount-frame.sell_order_amount)/(frame.buy_order_amount+frame.sell_order_amount).replace(0, np.nan)
    frame["ti"] = (frame.buy_amount-frame.sell_amount)/(frame.buy_amount+frame.sell_amount).replace(0, np.nan)
    return frame


def aggregate(frame, keys, quote_columns):
    rules = {c: ("last" if c == "close" else "mean" if c in quote_columns+["volatility"] else "sum")
             for c in frame.select_dtypes("number") if c not in ("oi", "ti")}
    result = frame.groupby(keys).agg(rules)
    return ratios(result)


def build_bars(data):
    trade = data.trades.copy()
    trade["time"] = trade.time.dt.floor("min")
    additive = []
    for name, mask in {
        "buy": trade.direction == "buy", "sell": trade.direction == "sell",
        "passive_buy": trade.passive_side == "buy", "passive_sell": trade.passive_side == "sell",
        "large": trade["size"].isin(["large", "xl"]), "xl": trade["size"] == "xl",
    }.items():
        for field, values in (("count", pd.Series(1, index=trade.index)), ("volume", trade.volume), ("amount", trade.amount)):
            column = name+"_"+field
            trade[column] = values.where(mask, 0)
            additive.append(column)
    rules = {"count": ("amount", "size"), "volume": ("volume", "sum"),
             "amount": ("amount", "sum"), "close": ("price", "last")}
    rules.update({c: (c, "sum") for c in additive})
    minute = trade.groupby(["symbol", "time"]).agg(**rules)
    quote = data.quotes.copy()
    quote["time"] = quote.time.dt.floor("min")
    quote["spread"] = (quote.ask_price1-quote.bid_price1)/((quote.ask_price1+quote.bid_price1)/2)
    if {"bid_volume1", "ask_volume1"}.issubset(quote.columns):
        quote["bid_depth"], quote["ask_depth"] = quote.bid_volume1, quote.ask_volume1
        quote["depth"] = quote.bid_volume1+quote.ask_volume1
        quote["depth_imbalance"] = (quote.bid_volume1-quote.ask_volume1)/quote.depth.replace(0, np.nan)
        paired = quote.bid_volume1.notna() & quote.ask_volume1.notna()
        quote["paired_bid_depth"] = quote.bid_volume1.where(paired)
        quote["paired_ask_depth"] = quote.ask_volume1.where(paired)
    qcols = [c for c in ("spread", "bid_depth", "ask_depth", "depth", "depth_imbalance", "paired_bid_depth", "paired_ask_depth") if c in quote]
    minute = minute.join(quote.groupby(["symbol", "time"])[qcols].mean(), how="outer")
    orders = data.orders.copy()
    orders["time"] = orders.time.dt.floor("min")
    status = orders.get("status", pd.Series("submit", index=orders.index)).astype(str).str.lower()
    cancelled = status.isin(["cancel", "cancelled", "canceled"])
    submitted = ~cancelled & (status != "fill")
    masks = {"cancel": cancelled, "order": submitted,
             "buy_order": submitted & (orders.side == "buy"),
             "sell_order": submitted & (orders.side == "sell"),
             "large_order": submitted & orders["size"].isin(["large", "xl"]),
             "buy_cancel": cancelled & (orders.side == "buy"),
             "sell_cancel": cancelled & (orders.side == "sell")}
    ocols = []
    for name, mask in masks.items():
        for field, values in (("count", pd.Series(1, index=orders.index)), ("volume", orders.volume), ("amount", orders.amount)):
            column = name+"_"+field
            orders[column] = values.where(mask, 0)
            ocols.append(column)
    minute = minute.join(orders.groupby(["symbol", "time"])[ocols].sum(min_count=1), how="outer").reset_index().sort_values(["symbol", "time"])
    # Missing event groups represent zero events, not absent quote observations.
    event_columns = ["count", "volume", "amount"]+additive+ocols
    minute[event_columns] = minute[event_columns].fillna(0)
    minute = ratios(minute)
    minute["volatility"] = minute_returns(minute).abs()
    five = aggregate(minute.assign(time=minute.time.dt.floor("5min")), ["symbol", "time"], qcols).reset_index()
    period = minute.assign(period=np.select([minute.time.dt.hour < 10, minute.time.dt.hour*60+minute.time.dt.minute >= 870], ["open", "close"], default="mid"))
    return {"1min": minute, "5min": five, "period": period,
            "period_summary": aggregate(period, ["symbol", "period"], qcols),
            "daily": aggregate(minute, ["symbol"], qcols)}
