"""Shared vectorized scope aggregation with explicit economic weights."""
import numpy as np
import pandas as pd


def aggregate_features(frame, by=None):
    keys = frame[by] if by else pd.Series("all", index=frame.index, name="scope")
    numeric = frame.select_dtypes(include="number")
    grouped = numeric.groupby(keys, sort=True)
    result = grouped.mean()
    totals = grouped.sum(min_count=1)
    result["securities"] = keys.groupby(keys).size()
    additive = [c for c in numeric if c in ("count", "amount", "volume", "trade.count", "trade.amount", "order.count", "order.amount", "order.buy_amount", "order.sell_amount")
                or c.startswith("flow.") and c.endswith(("_amount", "_flow"))
                or c.startswith("lifecycle.") and c.endswith(("_volume", "_sum", "_samples", "_count"))
                or c.startswith("liquidity.") and c.endswith(("_count", "_quotes"))
                or c.startswith("discovery.") and "_samples_" in c
                or c.endswith(("_count", "_volume", "_amount")) and "." not in c]
    result[additive] = totals[additive]
    for namespace in ("trade", "order"):
        amount = namespace+".amount"
        if amount not in numeric:
            continue
        columns = [c for c in numeric if c.startswith(namespace+".") and c.endswith("_ratio")]
        if columns:
            weights = numeric[columns].notna().mul(numeric[amount], axis=0).where(numeric[columns].notna())
            denominator = weights.groupby(keys).sum(min_count=1)
            numerator = numeric[columns].mul(numeric[amount], axis=0).groupby(keys).sum(min_count=1)
            result[columns] = numerator.div(denominator.where(denominator > 0))

    def ratio(target, numerator, denominator):
        if numerator in totals and denominator in totals:
            result[target] = totals[numerator]/totals[denominator].where(totals[denominator] > 0)

    for size in ("small", "medium", "large", "xl"):
        ratio(f"flow.{size}_normalized", f"flow.{size}_flow", f"flow.{size}_classified_amount")
    for namespace, buys, sells in (
        ("flow", [f"flow.{s}_buy_amount" for s in ("small", "medium", "large", "xl")], [f"flow.{s}_sell_amount" for s in ("small", "medium", "large", "xl")]),
        ("order", ["order.buy_amount"], ["order.sell_amount"]),
    ):
        if any(c in totals for c in buys+sells):
            amounts = totals.reindex(columns=buys+sells, fill_value=0)
            buy, sell = amounts[buys].sum(axis=1, min_count=1), amounts[sells].sum(axis=1, min_count=1)
            result[namespace+".imbalance"] = (buy-sell)/(buy+sell).where(buy+sell > 0)
    for target, first, second in (("flow.oi_ti_gap", "order.imbalance", "flow.imbalance"), ("flow.small_large_gap", "flow.small_normalized", "flow.large_normalized"), ("flow.large_xl_gap", "flow.large_normalized", "flow.xl_normalized"), ("lifecycle.cancel_side_gap", "lifecycle.buy_cancel_rate", "lifecycle.sell_cancel_rate")):
        if first in result and second in result:
            result[target] = result[first]-result[second]
    if {"flow.small_normalized", "flow.xl_normalized"}.issubset(result.columns):
        result["flow.small_xl_product"] = result["flow.small_normalized"]*result["flow.xl_normalized"]
    if {"liquidity.bid_depth", "liquidity.ask_depth"}.issubset(totals.columns):
        result = result.copy()
        columns = ["liquidity.bid_depth", "liquidity.ask_depth"]
        paired_columns = ["liquidity.paired_bid_depth", "liquidity.paired_ask_depth"]
        if set(paired_columns).issubset(numeric.columns):
            columns = paired_columns
        depths = numeric[columns]
        valid = np.isfinite(depths).all(axis=1) & depths.ge(0).all(axis=1)
        paired = depths.where(valid).groupby(keys).sum(min_count=1)
        bid, ask = paired[columns[0]], paired[columns[1]]
        result["liquidity.depth_imbalance"] = (bid-ask)/(bid+ask).where(bid+ask > 0)
        result["liquidity.depth_pair_samples"] = valid.groupby(keys).sum()
    for column in numeric:
        samples = None
        if column.startswith("discovery."):
            for group in ("large", "buy", "sell"):
                prefix = f"discovery.{group}_ret_"
                if column.startswith(prefix):
                    samples = f"discovery.{group}_samples_"+column[len(prefix):]
            for group in ("continuation", "reversal"):
                prefix = "discovery."+group+"_"
                if column.startswith(prefix) and "samples" not in column:
                    samples = "discovery."+group+"_samples_"+column[len(prefix):]
        if column.endswith("_depth_recovery_seconds"):
            samples = column.replace("_recovery_seconds", "_recovered_count")
        if samples in numeric:
            weight = numeric[samples].where(numeric[column].notna() & numeric[samples].gt(0))
            denominator = weight.groupby(keys).sum(min_count=1)
            result[column] = (numeric[column]*weight).groupby(keys).sum(min_count=1)/denominator
            result[samples] = denominator.fillna(0)
    for target, numerator, denominator in (("cancel_rate", "cancelled_volume", "submitted_volume"), ("fill_rate", "fill_ratio_sum", "context_order_count"), ("lifetime_seconds", "lifetime_seconds_sum", "lifetime_samples"), ("buy_cancel_rate", "buy_cancelled_volume", "buy_submitted_volume"), ("sell_cancel_rate", "sell_cancelled_volume", "sell_submitted_volume")):
        ratio("lifecycle."+target, "lifecycle."+numerator, "lifecycle."+denominator)
    if "lifecycle.cancel_rate" in result:
        result["order.cancel_rate"] = result["lifecycle.cancel_rate"]
    if {"lifecycle.buy_cancel_rate", "lifecycle.sell_cancel_rate"}.issubset(result.columns):
        result["lifecycle.cancel_side_gap"] = result["lifecycle.buy_cancel_rate"]-result["lifecycle.sell_cancel_rate"]
    for raw, canonical in (("ti", "flow.imbalance"), ("oi", "order.imbalance"), ("amount", "trade.amount"), ("count", "trade.count")):
        if canonical in result:
            result[raw] = result[canonical]
    return result.copy()
