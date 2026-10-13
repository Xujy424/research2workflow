"""Order lifecycle, event returns and distribution-aware research features."""
import numpy as np
import pandas as pd
from ..schema import safe_divide


def lifecycle(data):
    orders = data.orders.copy()
    status = orders.get("status", pd.Series("submit", index=orders.index)).astype(str).str.lower()
    cancels = orders.loc[status.isin(["cancel", "cancelled", "canceled"])]
    submissions = orders.loc[~status.isin(["cancel", "cancelled", "canceled", "fill"])]
    keys = ["symbol", "side", "order_id"]
    if submissions.duplicated(keys).any():
        raise ValueError("Duplicate submitted order keys; order IDs must include channel")
    legs = []
    for side, column in (("buy", "buy_order_id"), ("sell", "sell_order_id")):
        leg = data.trades[["symbol", column, "volume", "time"]].rename(columns={column: "order_id"})
        leg["side"] = side
        legs.append(leg)
    fills = pd.concat(legs).groupby(keys).agg(filled_volume=("volume", "sum"), last_fill=("time", "max"))
    cancel = cancels.groupby(keys).agg(cancel_volume=("volume", "sum"), last_cancel=("time", "max"))
    table = submissions.set_index(keys).join(fills).join(cancel)
    table["fill_ratio"] = safe_divide(table.filled_volume.fillna(0), table.volume)
    table["cancel_ratio"] = safe_divide(table.cancel_volume.fillna(0), table.volume)
    complete = (table.filled_volume.fillna(0) + table.cancel_volume.fillna(0)) >= table.volume
    terminal = table[["last_fill", "last_cancel"]].max(axis=1)
    table["lifetime_seconds"] = (terminal - table.time).dt.total_seconds().where(complete)
    result = table.groupby(level="symbol").agg(fill_rate=("fill_ratio", "mean"), completed_ratio=("lifetime_seconds", "count"), lifetime_seconds=("lifetime_seconds", "mean"))
    result["completed_ratio"] /= table.groupby(level="symbol").size()
    if "status" in orders:
        result["cancel_rate"] = table.cancel_volume.fillna(0).groupby(level="symbol").sum() / table.volume.groupby(level="symbol").sum()
        for side in ("buy", "sell"):
            part = table.loc[table.index.get_level_values("side") == side]
            result[f"{side}_cancel_rate"] = part.cancel_volume.fillna(0).groupby(level="symbol").sum() / part.volume.groupby(level="symbol").sum()
        result["cancel_side_gap"] = result.buy_cancel_rate - result.sell_cancel_rate
    return result, submissions


def event_returns(data):
    results = {}
    for symbol, trades in data.trades.groupby("symbol"):
        trades = trades.sort_values("time")
        row = {}
        for horizon in (1, 5, 10, 30):
            future = trades[["time", "price"]].rename(columns={"time": "future_time", "price": "future_price"})
            events = trades.assign(target=trades.time + pd.Timedelta(minutes=horizon))
            joined = pd.merge_asof(events, future, left_on="target", right_on="future_time", direction="forward", tolerance=pd.Timedelta("30s"))
            # Do not bridge the lunch break or carry signals past the session end.
            same_session = (joined.time.dt.hour < 12) == (joined.future_time.dt.hour < 12)
            ret = (joined.future_price / joined.price - 1).where(same_session)
            signed = ret * joined.direction.map({"buy": 1, "sell": -1})
            for group, mask in {"large": joined["size"].isin(["large", "xl"]), "buy": joined.direction == "buy", "sell": joined.direction == "sell"}.items():
                values = ret.loc[mask].dropna()
                row[f"{group}_ret_{horizon}m"] = values.mean()
                row[f"{group}_samples_{horizon}m"] = len(values)
            valid = signed.loc[joined["size"].isin(["large", "xl"])].dropna()
            row[f"continuation_{horizon}m"] = (valid > 0).mean() if len(valid) else np.nan
            row[f"reversal_{horizon}m"] = (valid < 0).mean() if len(valid) else np.nan
        quote = data.quotes.loc[data.quotes.symbol == symbol, ["time", "bid_price1", "ask_price1"]].copy()
        quote["mid"] = (quote.bid_price1 + quote.ask_price1) / 2
        if not quote.empty:
            matched = pd.merge_asof(trades, quote[["time", "mid"]], on="time", direction="backward", tolerance=pd.Timedelta("30s"))
            sign = matched.direction.map({"buy": 1, "sell": -1})
            row["effective_spread"] = (2 * sign * (matched.price - matched.mid) / matched.mid).mean()
            matched["target"] = matched.time + pd.Timedelta("5min")
            later = pd.merge_asof(matched, quote[["time", "mid"]].rename(columns={"time": "quote_time", "mid": "future_mid"}), left_on="target", right_on="quote_time", direction="forward", tolerance=pd.Timedelta("30s"))
            valid = (later.time.dt.hour < 12) == (later.quote_time.dt.hour < 12)
            row["realized_spread_5m"] = (2 * sign * (later.price - later.future_mid) / later.mid).where(valid).mean()
        results[symbol] = row
    return pd.DataFrame.from_dict(results, orient="index")


def enrich_features(data, bars, frames):
    trade, flow = frames["trade"], frames["flow"]
    for side in ("buy", "sell"):
        amount = data.trades.amount.where(data.trades.direction == side, 0).groupby(data.trades.symbol).sum()
        trade[f"active_{side}_ratio"] = safe_divide(amount, trade.amount)
    trade["xl_ratio"] = safe_divide(data.trades.amount.where(data.trades["size"] == "xl", 0).groupby(data.trades.symbol).sum(), trade.amount)
    trade["kurt"] = data.trades.groupby("symbol").amount.apply(lambda s: s.kurt())
    known = data.trades.direction.isin(["buy", "sell"])
    denominator = data.trades.amount.where(known, 0).groupby(data.trades.symbol).sum()
    flow["imbalance"] = safe_divide(flow[[f"{s}_flow" for s in ("small", "medium", "large", "xl")]].sum(axis=1), denominator)
    for size in ("small", "medium", "large", "xl"):
        amount = data.trades.amount.where(known & (data.trades["size"] == size), 0).groupby(data.trades.symbol).sum()
        flow[size + "_normalized"] = safe_divide(flow[size + "_flow"], amount)
    flow["small_large_gap"] = flow.small_normalized - flow.large_normalized
    flow["large_xl_gap"] = flow.large_normalized - flow.xl_normalized
    flow["small_xl_product"] = flow.small_normalized * flow.xl_normalized
    flow["small_xl_aligned"] = (flow.small_xl_product > 0).astype(float).where(flow.small_xl_product.notna())
    frames["lifecycle"], submissions = lifecycle(data)
    order = frames["order"]
    group = submissions.groupby("symbol").amount
    order["count"], order["amount"] = group.size(), group.sum()
    for statistic in ("mean", "median", "std", "skew"):
        order[statistic] = getattr(group, statistic)()
    for p in (.9, .95, .99):
        order[f"p{int(p*100)}"] = group.quantile(p)
    for side in ("buy", "sell"):
        order[side + "_amount"] = submissions.amount.where(submissions.side == side, 0).groupby(submissions.symbol).sum()
    order["imbalance"] = safe_divide(order.buy_amount - order.sell_amount, order.buy_amount + order.sell_amount)
    order["large_ratio"] = safe_divide(submissions.amount.where(submissions.amount > 100000, 0).groupby(submissions.symbol).sum(), order.amount)
    order["xl_ratio"] = safe_divide(submissions.amount.where(submissions.amount > 500000, 0).groupby(submissions.symbol).sum(), order.amount)
    order["kurt"] = group.apply(lambda s: s.kurt())
    flow["oi_ti_gap"] = order.imbalance - flow.imbalance
    frames["discovery"] = event_returns(data)
    minute = bars["1min"]
    for symbol, part in minute.groupby("symbol"):
        part = part.sort_values("time")
        minute_flows = data.trades.loc[data.trades.symbol == symbol].copy()
        minute_flows["minute"] = minute_flows.time.dt.floor("min")
        minute_flows["signed_amount"] = minute_flows.amount * minute_flows.direction.map({"buy": 1, "sell": -1})
        size_flows = minute_flows.pivot_table(index="minute", columns="size", values="signed_amount", aggfunc="sum", observed=True)
        if {"small", "xl"}.issubset(size_flows.columns):
            product = size_flows.small * size_flows.xl
            product = product.loc[product.notna() & (product != 0)]
            flow.loc[symbol, "direction_agreement_rate"] = (product > 0).mean() if len(product) else np.nan
            flow.loc[symbol, "direction_divergence_rate"] = (product < 0).mean() if len(product) else np.nan
        flow.loc[symbol, "oi_ti_corr"] = part.oi.corr(part.ti)
        same_next_session = (part.time.dt.hour < 12) == (part.time.shift(-1).dt.hour < 12)
        flow.loc[symbol, "oi_ti_lead_corr"] = part.oi.corr(part.ti.shift(-1).where(same_next_session))
        ret = part.close.pct_change(fill_method=None)
        flow.loc[symbol, "flow_return_lead_corr"] = part.ti.corr(ret.shift(-1).where(same_next_session))
        frames["liquidity"].loc[symbol, "amihud"] = safe_divide(ret.abs(), part.amount).mean()
        quote = data.quotes.loc[data.quotes.symbol == symbol]
        if "bid_volume1" in quote:
            frames["liquidity"].loc[symbol, "bid_depth"] = quote.bid_volume1.mean()
            frames["liquidity"].loc[symbol, "ask_depth"] = quote.ask_volume1.mean()
        spread = part.spread.dropna()
        recovery = []
        if len(spread) >= 20 and spread.max() > spread.median():
            threshold, normal = spread.quantile(.9), spread.median()
            for position in range(1, len(part)):
                if part.spread.iloc[position] > threshold and part.spread.iloc[position-1] <= threshold:
                    future = part.iloc[position+1:]
                    same = (future.time.dt.hour < 12) == (part.time.iloc[position].hour < 12)
                    recovered = future.loc[same & (future.spread <= normal)]
                    if not recovered.empty:
                        recovery.append((recovered.time.iloc[0]-part.time.iloc[position]).total_seconds())
        frames["liquidity"].loc[symbol, "spread_recovery_seconds"] = np.mean(recovery) if recovery else np.nan
        frames["liquidity"].loc[symbol, "recovery_samples"] = len(recovery)
        for metric in ("amount", "large_amount", "buy_amount", "spread", "oi", "ti", "volatility"):
            for period in ("open", "mid", "close"):
                values_period = bars["period"].loc[(bars["period"].symbol == symbol) & (bars["period"].period == period), metric]
                frames["intraday"].loc[symbol, metric + "_" + period + "_mean"] = values_period.mean()
                if metric in ("amount", "large_amount", "buy_amount"):
                    total = part[metric].sum()
                    frames["intraday"].loc[symbol, metric + "_" + period + "_share"] = values_period.sum()/total if total else np.nan
            frames["intraday"].loc[symbol, metric + "_close_minus_open"] = frames["intraday"].loc[symbol, metric + "_close_mean"] - frames["intraday"].loc[symbol, metric + "_open_mean"]
            values = part.set_index("time")[metric].dropna()
            if values.empty:
                continue
            for name, value in {"max": values.max(), "min": values.min(), "range": values.max()-values.min(), "argmax_minute": values.idxmax().hour*60+values.idxmax().minute, "argmin_minute": values.idxmin().hour*60+values.idxmin().minute}.items():
                frames["intraday"].loc[symbol, metric + "_" + name] = value
    return frames
