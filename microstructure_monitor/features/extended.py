"""Order lifecycle, event returns and distribution-aware research features."""
import numpy as np
import pandas as pd
from ..schema import safe_divide
from ..bars import minute_returns


from ..preprocess.match_order_trade import lifecycle
from .imbalance import safe_correlation
from .context import compute_event_context


def enrich_features(data, bars, frames, context=None):
    context = context if context is not None else compute_event_context(data)
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
        flow[size + "_classified_amount"] = amount
        for side in ("buy", "sell"):
            flow[f"{size}_{side}_amount"] = data.trades.amount.where((data.trades.direction == side) & (data.trades["size"] == size), 0).groupby(data.trades.symbol).sum()
        flow[size + "_normalized"] = safe_divide(flow[size + "_flow"], amount)
    flow["small_large_gap"] = flow.small_normalized - flow.large_normalized
    flow["large_xl_gap"] = flow.large_normalized - flow.xl_normalized
    flow["small_xl_product"] = flow.small_normalized * flow.xl_normalized
    flow["small_xl_aligned"] = (flow.small_xl_product > 0).astype(float).where(flow.small_xl_product.notna())
    frames["lifecycle"], submissions = context.lifecycle, context.submissions
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
    if "cancel_rate" in frames["lifecycle"]:
        order["cancel_rate"] = frames["lifecycle"].cancel_rate
    frames["discovery"] = context.discovery
    discovery = frames["discovery"]
    frames["liquidity"]["impact"] = discovery.get("signed_price_impact_5m", pd.Series(np.nan, index=discovery.index))
    frames["liquidity"]["trade_impact_proxy"] = discovery.signed_trade_impact_5m
    frames["liquidity"] = frames["liquidity"].join(context.recovery)
    minute = bars["1min"]
    for first, second in (("small", "medium"), ("small", "large"), ("small", "xl"), ("medium", "large"), ("medium", "xl"), ("large", "xl")):
        flow[first+"_"+second+"_agreement"] = np.nan
        flow[first+"_"+second+"_divergence"] = np.nan
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
        for first, second in (("small", "medium"), ("small", "large"), ("small", "xl"), ("medium", "large"), ("medium", "xl"), ("large", "xl")):
            if {first, second}.issubset(size_flows.columns):
                product = size_flows[first]*size_flows[second]
                valid = product.loc[product.notna() & (product != 0)]
                flow.loc[symbol, first+"_"+second+"_agreement"] = (valid > 0).mean() if len(valid) else np.nan
                flow.loc[symbol, first+"_"+second+"_divergence"] = (valid < 0).mean() if len(valid) else np.nan
        flow.loc[symbol, "oi_ti_corr"] = safe_correlation(part.oi, part.ti)
        flow.loc[symbol, "oi_ti_samples"] = (part.oi.notna() & part.ti.notna()).sum()
        same_next_session = (part.time.shift(-1) - part.time).eq(pd.Timedelta("1min"))
        flow.loc[symbol, "oi_ti_lead_corr"] = safe_correlation(part.oi, part.ti.shift(-1).where(same_next_session))
        ret = minute_returns(part)
        flow.loc[symbol, "flow_return_lead_corr"] = safe_correlation(part.ti, ret.shift(-1).where(same_next_session))
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
                signed = metric in ("oi", "ti")
                total = part[metric].abs().sum() if signed else part[metric].sum()
                numerator = values_period.abs().sum() if signed else values_period.sum()
                suffix = "_abs_share" if signed else "_share"
                frames["intraday"].loc[symbol, metric + "_" + period + suffix] = numerator/total if total else np.nan
            frames["intraday"].loc[symbol, metric + "_close_minus_open"] = frames["intraday"].loc[symbol, metric + "_close_mean"] - frames["intraday"].loc[symbol, metric + "_open_mean"]
            values = part.set_index("time")[metric].dropna()
            if values.empty:
                continue
            for name, value in {"max": values.max(), "min": values.min(), "range": values.max()-values.min(), "argmax_minute": values.idxmax().hour*60+values.idxmax().minute, "argmin_minute": values.idxmin().hour*60+values.idxmin().minute}.items():
                frames["intraday"].loc[symbol, metric + "_" + name] = value
            frames["intraday"].loc[symbol, metric + "_argmax_time"] = values.idxmax().strftime("%H:%M")
            frames["intraday"].loc[symbol, metric + "_argmin_time"] = values.idxmin().strftime("%H:%M")
    return frames
