"""Trade and quote aggregates at one and five minute resolution."""
import numpy as np
import pandas as pd


def build_bars(data):
    trade = data.trades.copy()
    trade["time"] = trade.time.dt.floor("min")
    for side in ("buy", "sell"):
        trade[f"{side}_amount"] = trade.amount.where(trade.direction == side, 0)
    trade["large_amount"] = trade.amount.where(trade["size"].isin(["large", "xl"]), 0)
    minute = trade.groupby(["symbol", "time"]).agg(count=("amount", "size"), volume=("volume", "sum"), amount=("amount", "sum"), buy_amount=("buy_amount", "sum"), sell_amount=("sell_amount", "sum"), large_amount=("large_amount", "sum"), close=("price", "last"))
    quote = data.quotes.copy()
    quote["time"] = quote.time.dt.floor("min")
    quote["spread"] = (quote.ask_price1 - quote.bid_price1) / ((quote.ask_price1 + quote.bid_price1) / 2)
    if {"bid_volume1", "ask_volume1"}.issubset(quote.columns):
        quote["depth"] = quote.bid_volume1 + quote.ask_volume1
        quote["depth_imbalance"] = (quote.bid_volume1 - quote.ask_volume1) / quote.depth.replace(0, np.nan)
    qcols = [c for c in ("spread", "depth", "depth_imbalance") if c in quote]
    minute = minute.join(quote.groupby(["symbol", "time"])[qcols].mean(), how="outer").reset_index()
    orders = data.orders.copy()
    orders["time"] = orders.time.dt.floor("min")
    status = orders.get("status", pd.Series("submit", index=orders.index)).astype(str).str.lower()
    cancelled = status.isin(["cancel", "cancelled", "canceled"])
    orders["cancel_count"] = cancelled.astype(int)
    orders["cancel_volume"] = orders.volume.where(cancelled, 0)
    orders["cancel_amount"] = orders.amount.where(cancelled, 0)
    orders["order_count"] = (~cancelled).astype(int)
    for side in ("buy", "sell"):
        orders[f"{side}_order_amount"] = orders.amount.where((orders.side == side) & ~cancelled, 0)
    ocols = ["cancel_count", "cancel_volume", "cancel_amount", "order_count", "buy_order_amount", "sell_order_amount"]
    minute = minute.set_index(["symbol", "time"]).join(orders.groupby(["symbol", "time"])[ocols].sum(), how="outer").reset_index().sort_values(["symbol", "time"])
    minute["oi"] = (minute.buy_order_amount - minute.sell_order_amount) / (minute.buy_order_amount + minute.sell_order_amount).replace(0, np.nan)
    minute["ti"] = (minute.buy_amount - minute.sell_amount) / (minute.buy_amount + minute.sell_amount).replace(0, np.nan)
    minute["volatility"] = minute.groupby("symbol").close.pct_change(fill_method=None).abs()
    five = minute.assign(time=minute.time.dt.floor("5min"))
    aggregation = {c: ("last" if c == "close" else "mean" if c in qcols + ["oi", "ti", "volatility"] else "sum") for c in minute.columns if c not in ("symbol", "time")}
    five = five.groupby(["symbol", "time"]).agg(aggregation).reset_index()
    period = minute.assign(period=np.select([minute.time.dt.hour < 10, minute.time.dt.hour * 60 + minute.time.dt.minute >= 870], ["open", "close"], default="mid"))
    return {"1min": minute, "5min": five, "period": period,
            "period_summary": period.groupby(["symbol", "period"]).agg(aggregation),
            "daily": minute.groupby("symbol").agg(aggregation)}
