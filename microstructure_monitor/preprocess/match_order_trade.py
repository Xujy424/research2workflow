"""Full-day order context and continuous-session lifecycle cohorts."""
import numpy as np
import pandas as pd
from ..schema import safe_divide


def lifecycle(data, with_orders=False):
    orders = (data.order_context if data.order_context is not None else data.orders).copy()
    status = orders.get("status", pd.Series("submit", index=orders.index)).astype(str).str.lower()
    cancels = orders.loc[status.isin(["cancel", "cancelled", "canceled"])]
    submissions = orders.loc[~status.isin(["cancel", "cancelled", "canceled", "fill"])]
    keys = ["symbol", "side", "order_id"]
    if submissions.duplicated(keys).any():
        raise ValueError("Duplicate submitted order keys; order IDs must include channel")
    legs = []
    for side, column in (("buy", "buy_order_id"), ("sell", "sell_order_id")):
        context = data.trade_context if data.trade_context is not None else data.trades
        leg = context[["symbol", column, "volume", "time"]].rename(columns={column: "order_id"})
        leg["side"] = side
        legs.append(leg)
    fills = pd.concat(legs).groupby(keys).agg(filled_volume=("volume", "sum"), last_fill=("time", "max"))
    cancel = cancels.groupby(keys).agg(cancel_volume=("volume", "sum"), last_cancel=("time", "max"))
    table = submissions.set_index(keys).join(fills).join(cancel)
    observed = data.orders[keys].copy()
    for side, column in (("buy", "buy_order_id"), ("sell", "sell_order_id")):
        observed = pd.concat([observed, data.trades[["symbol", column]].rename(columns={column: "order_id"}).assign(side=side)], ignore_index=True)
    cohort = pd.MultiIndex.from_frame(observed[keys].drop_duplicates())
    table = table.loc[table.index.isin(cohort)]
    table["fill_ratio"] = safe_divide(table.filled_volume.fillna(0), table.volume)
    table["cancel_ratio"] = safe_divide(table.cancel_volume.fillna(0), table.volume)
    accounted = table.filled_volume.fillna(0) + table.cancel_volume.fillna(0)
    invalid = accounted > table.volume
    complete = accounted.eq(table.volume) & ~invalid
    terminal = table[["last_fill", "last_cancel"]].max(axis=1)
    table["lifetime_seconds"] = (terminal - table.time).dt.total_seconds().where(complete)
    result = table.groupby(level="symbol").agg(fill_rate=("fill_ratio", "mean"), completed_ratio=("lifetime_seconds", "count"), lifetime_seconds=("lifetime_seconds", "mean"))
    result["completed_ratio"] /= table.groupby(level="symbol").size()
    result["overaccounted_orders"] = invalid.groupby(level="symbol").sum()
    result["context_order_count"] = table.groupby(level="symbol").size()
    result["submitted_volume"] = table.volume.groupby(level="symbol").sum()
    result["filled_volume"] = table.filled_volume.fillna(0).groupby(level="symbol").sum()
    result["cancelled_volume"] = table.cancel_volume.fillna(0).groupby(level="symbol").sum()
    result["fill_ratio_sum"] = table.fill_ratio.groupby(level="symbol").sum()
    valid_life = table.lifetime_seconds.where(table.lifetime_seconds >= 0)
    result["lifetime_seconds_sum"] = valid_life.groupby(level="symbol").sum()
    result["lifetime_samples"] = valid_life.groupby(level="symbol").count()
    result["negative_lifetime_count"] = table.lifetime_seconds.lt(0).groupby(level="symbol").sum()
    result["lifetime_seconds"] = table.lifetime_seconds.where(table.lifetime_seconds >= 0).groupby(level="symbol").mean()
    if "status" in orders:
        result["cancel_rate"] = table.cancel_volume.fillna(0).groupby(level="symbol").sum() / table.volume.groupby(level="symbol").sum()
        for side in ("buy", "sell"):
            part = table.loc[table.index.get_level_values("side") == side]
            result[f"{side}_submitted_volume"] = part.volume.groupby(level="symbol").sum()
            result[f"{side}_cancelled_volume"] = part.cancel_volume.fillna(0).groupby(level="symbol").sum()
            result[f"{side}_cancel_rate"] = part.cancel_volume.fillna(0).groupby(level="symbol").sum() / part.volume.groupby(level="symbol").sum()
        result["cancel_side_gap"] = result.buy_cancel_rate - result.sell_cancel_rate
    continuous_keys = pd.MultiIndex.from_frame(data.orders[keys])
    submitted = submissions.loc[pd.MultiIndex.from_frame(submissions[keys]).isin(continuous_keys)]
    return (result, submitted, table.reset_index()) if with_orders else (result, submitted)
