"""Daily per-security features; rates retain explicit denominators."""
import numpy as np
import pandas as pd
from ..schema import safe_divide
from .extended import enrich_features


def compute_features(data, bars):
    trades, orders = data.trades, data.orders
    grouped = trades.groupby("symbol")
    trade = grouped.amount.agg(count="size", amount="sum", mean="mean", median="median", std="std", skew="skew", maximum="max")
    for percentile in (.9, .95, .99):
        trade[f"p{int(percentile * 100)}"] = grouped.amount.quantile(percentile)
    trade["tail_ratio"] = safe_divide(trade.p99, trade["median"])
    classified = trades.direction.isin(["buy", "sell"])
    trade["classified_ratio"] = safe_divide(trades.amount.where(classified, 0).groupby(trades.symbol).sum(), trade.amount)
    flows = pd.DataFrame(index=trade.index)
    signed = trades.amount * trades.direction.map({"buy": 1, "sell": -1}).fillna(0)
    for size in ("small", "medium", "large", "xl"):
        flows[size + "_flow"] = signed.where(trades["size"] == size, 0).groupby(trades.symbol).sum()
    flows["imbalance"] = safe_divide(signed.groupby(trades.symbol).sum(), trade.amount)
    trade["large_ratio"] = safe_divide(trades.amount.where(trades["size"].isin(["large", "xl"]), 0).groupby(trades.symbol).sum(), trade.amount)
    order = orders.groupby("symbol").amount.agg(count="size", amount="sum")
    signed_orders = orders.amount * orders.side.map({"buy": 1, "sell": -1}).fillna(0)
    order["imbalance"] = safe_divide(signed_orders.groupby(orders.symbol).sum(), order.amount)
    if "status" in orders:
        cancelled = orders.status.astype(str).str.lower().isin(["cancel", "cancelled", "canceled"])
        order["cancel_rate"] = cancelled.groupby(orders.symbol).mean()
    minute = bars["1min"].sort_values(["symbol", "time"])
    liquidity_columns = [c for c in ("spread", "depth", "depth_imbalance") if c in minute]
    liquidity = minute.groupby("symbol")[liquidity_columns].mean()
    returns = minute.groupby("symbol").close.pct_change(fill_method=None)
    liquidity["impact"] = safe_divide(returns.abs(), minute.amount).groupby(minute.symbol).mean()
    shape = pd.DataFrame(index=trade.index)
    for metric in ("amount", "large_amount", "spread"):
        for period in ("open", "mid", "close"):
            part = bars["period"].loc[bars["period"].period == period]
            shape[f"{metric}_{period}_mean"] = part.groupby("symbol")[metric].mean()
            if metric != "spread":
                shape[f"{metric}_{period}_share"] = safe_divide(part.groupby("symbol")[metric].sum(), minute.groupby("symbol")[metric].sum())
        shape[f"{metric}_close_minus_open"] = shape[f"{metric}_close_mean"] - shape[f"{metric}_open_mean"]
    return enrich_features(data, bars, {"trade": trade, "order": order, "flow": flows, "liquidity": liquidity, "intraday": shape})
