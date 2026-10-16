"""Minute feature series for distributions, flows, events and order cohorts."""
import numpy as np
import pandas as pd
from ..bars import minute_returns
from .context import compute_event_context


def build_feature_minutes(data, bars, context=None):
    context = context if context is not None else compute_event_context(data)
    base = bars["1min"].set_index(["symbol", "time"])
    parts = [base]
    for label, source in (("trade", data.trades), ("order", data.orders)):
        frame = source.copy()
        if label == "order":
            status = frame.get("status", pd.Series("submit", index=frame.index)).astype(str).str.lower()
            frame = frame.loc[~status.isin(["cancel", "cancelled", "canceled", "fill"])]
        frame["time"] = frame.time.dt.floor("min")
        grouped = frame.groupby(["symbol", "time"]).amount
        summary = grouped.agg(count="size", amount="sum", mean="mean", median="median", std="std", skew="skew", max="max")
        summary["kurt"] = grouped.apply(lambda s: s.kurt())
        for p in (.5, .9, .95, .99):
            summary[f"p{int(p*100)}"] = grouped.quantile(p)
        summary["tail_ratio"] = summary.p99/summary.p50.replace(0, np.nan)
        for size in ("small", "medium", "large", "xl"):
            summary[size+"_ratio"] = frame.amount.where(frame["size"] == size, 0).groupby([frame.symbol, frame.time]).sum()/summary.amount.replace(0, np.nan)
        summary["large_ratio"] += summary.xl_ratio
        summary["maximum"] = summary["max"]
        parts.append(summary.add_prefix(label+"."))
    trades = data.trades.assign(time=data.trades.time.dt.floor("min"))
    flow = pd.DataFrame(index=base.index)
    for size in ("small", "medium", "large", "xl"):
        buy = trades.amount.where((trades["size"] == size) & (trades.direction == "buy"), 0).groupby([trades.symbol, trades.time]).sum()
        sell = trades.amount.where((trades["size"] == size) & (trades.direction == "sell"), 0).groupby([trades.symbol, trades.time]).sum()
        flow[size+"_flow"] = buy-sell
        flow[size+"_buy_amount"] = buy
        flow[size+"_sell_amount"] = sell
        flow[size+"_classified_amount"] = buy+sell
        flow[size+"_normalized"] = (buy-sell)/(buy+sell).replace(0, np.nan)
    flow["imbalance"] = base.ti
    flow["oi_ti_gap"] = base.oi-base.ti
    flow["small_large_gap"] = flow.small_normalized-flow.large_normalized
    flow["large_xl_gap"] = flow.large_normalized-flow.xl_normalized
    flow["small_xl_product"] = flow.small_normalized*flow.xl_normalized
    for first, second in (("small", "medium"), ("small", "large"), ("small", "xl"), ("medium", "large"), ("medium", "xl"), ("large", "xl")):
        product = flow[first+"_flow"]*flow[second+"_flow"]
        valid = product.notna() & (product != 0)
        flow[first+"_"+second+"_agreement"] = (product > 0).astype(float).where(valid)
        flow[first+"_"+second+"_divergence"] = (product < 0).astype(float).where(valid)
    parts.append(flow.add_prefix("flow."))
    table = context.orders.copy()
    table["lifetime_seconds"] = table.lifetime_seconds.where(table.lifetime_seconds >= 0)
    table["time"] = table.time.dt.floor("min")
    life = table.groupby(["symbol", "time"])[["fill_ratio", "cancel_ratio", "lifetime_seconds"]].mean().rename(columns={"fill_ratio": "fill_rate", "cancel_ratio": "cancel_rate"})
    life["cancel_rate"] = table.cancel_volume.fillna(0).groupby([table.symbol, table.time]).sum()/table.volume.groupby([table.symbol, table.time]).sum().replace(0, np.nan)
    for side in ("buy", "sell"):
        part = table.loc[table.side == side]
        life[side+"_cancel_rate"] = part.cancel_volume.fillna(0).groupby([part.symbol, part.time]).sum()/part.volume.groupby([part.symbol, part.time]).sum().replace(0, np.nan)
        life[side+"_submitted_volume"] = part.volume.groupby([part.symbol, part.time]).sum()
        life[side+"_cancelled_volume"] = part.cancel_volume.fillna(0).groupby([part.symbol, part.time]).sum()
    life["cancel_side_gap"] = life.buy_cancel_rate-life.sell_cancel_rate
    life["context_order_count"] = table.groupby(["symbol", "time"]).size()
    life["submitted_volume"] = table.volume.groupby([table.symbol, table.time]).sum()
    life["cancelled_volume"] = table.cancel_volume.fillna(0).groupby([table.symbol, table.time]).sum()
    life["fill_ratio_sum"] = table.fill_ratio.groupby([table.symbol, table.time]).sum()
    life["lifetime_seconds_sum"] = table.lifetime_seconds.groupby([table.symbol, table.time]).sum()
    life["lifetime_samples"] = table.lifetime_seconds.groupby([table.symbol, table.time]).count()
    parts.append(life.add_prefix("lifecycle."))
    events = context.trade_events.copy()
    events["time"] = events.time.dt.floor("min")
    aggregation = {c: "sum" if "samples" in c else "mean" for c in events.select_dtypes("number")}
    parts.append(events.groupby(["symbol", "time"]).agg(aggregation).add_prefix("discovery."))
    result = pd.concat(parts, axis=1).reindex(base.index)
    # Rolling minute correlations require enough samples; sessions reset the window.
    result["order.imbalance"] = result.oi
    result["order.buy_amount"] = result.buy_order_amount
    result["order.sell_amount"] = result.sell_order_amount
    result["order.cancel_rate"] = result.get("lifecycle.cancel_rate", np.nan)
    result["trade.active_buy_ratio"] = result.buy_amount/result.amount.replace(0, np.nan)
    result["trade.active_sell_ratio"] = result.sell_amount/result.amount.replace(0, np.nan)
    result["trade.classified_ratio"] = (result.buy_amount+result.sell_amount)/result.amount.replace(0, np.nan)
    result["liquidity.spread"] = result.spread
    for metric in ("depth", "bid_depth", "ask_depth", "depth_imbalance", "paired_bid_depth", "paired_ask_depth"):
        result["liquidity."+metric] = result.get(metric, np.nan)
    result["liquidity.impact"] = result.get("discovery.signed_price_impact_5m", np.nan)
    result["liquidity.trade_impact_proxy"] = result.get("discovery.signed_trade_impact_5m", np.nan)
    reset = result.reset_index()
    returns = minute_returns(reset)
    result["return"] = returns.to_numpy()
    result["liquidity.amihud"] = (returns.abs()/reset.amount.replace(0, np.nan)).to_numpy()
    for symbol, frame in result.groupby(level="symbol"):
        for session in (0, 1):
            subset = frame.loc[(frame.index.get_level_values("time").hour >= 12).astype(int) == session]
            result.loc[subset.index, "flow.oi_ti_corr"] = subset.oi.rolling(20, min_periods=10).corr(subset.ti).to_numpy()
            times = subset.index.get_level_values("time").to_series(index=subset.index)
            adjacent = (times.shift(-1)-times).eq(pd.Timedelta("1min"))
            result.loc[subset.index, "flow.oi_ti_lead_corr"] = subset.oi.rolling(20, min_periods=10).corr(subset.ti.shift(-1).where(adjacent)).to_numpy()
            returns = minute_returns(subset.reset_index())
            future = pd.Series(returns.shift(-1).to_numpy(), index=subset.index).where(adjacent)
            result.loc[subset.index, "flow.flow_return_lead_corr"] = subset.ti.rolling(20, min_periods=10).corr(future).to_numpy()
            spread = subset.spread.dropna()
            if len(spread) >= 20:
                threshold, normal = spread.quantile(.9), spread.median()
                for position in range(1, len(subset)):
                    if subset.spread.iloc[position] > threshold and subset.spread.iloc[position-1] <= threshold:
                        recovered = subset.iloc[position+1:].loc[lambda x: x.spread <= normal]
                        if not recovered.empty:
                            index = subset.index[position]
                            result.loc[index, "liquidity.spread_recovery_seconds"] = (recovered.index[0][1]-index[1]).total_seconds()
    depth_events = context.recovery_events.copy()
    quote = data.quotes.assign(time=data.quotes.time.dt.floor("min"))
    bid = quote.get("bid_volume1", pd.Series(np.nan, index=quote.index))
    ask = quote.get("ask_volume1", pd.Series(np.nan, index=quote.index))
    for side, depth in (("bid", bid), ("ask", ask), ("total", bid+ask)):
        valid = np.isfinite(depth) & depth.ge(0)
        if side == "total":
            valid &= bid.ge(0) & ask.ge(0)
        count = valid.groupby([quote.symbol, quote.time]).sum().reindex(result.index, fill_value=0)
        result[f"liquidity.{side}_depth_valid_quotes"] = count
        result[f"liquidity.{side}_depth_recovery_seconds"] = np.nan
        for metric in ("shock_count", "recovered_count", "censored_count"):
            result[f"liquidity.{side}_depth_{metric}"] = pd.Series(0., index=result.index).where(count > 0)
    if not depth_events.empty:
        depth_events["time"] = pd.to_datetime(depth_events.time).dt.floor("min")
        for side, events in depth_events.groupby("side"):
            recovered = events.loc[events.recovered]
            result["liquidity."+side+"_depth_recovery_seconds"] = recovered.groupby(["symbol", "time"]).duration_seconds.mean()
            for name, values in (("shock_count", events.groupby(["symbol", "time"]).size()), ("recovered_count", recovered.groupby(["symbol", "time"]).size()), ("censored_count", (~events.recovered).groupby([events.symbol, events.time]).sum())):
                key = f"liquidity.{side}_depth_{name}"
                result[key] = values.reindex(result.index).combine_first(result[key])
    return result.reset_index()


def dynamic_intraday_shape(minute):
    results = {}
    for symbol, frame in minute.groupby("symbol"):
        row = {}
        minute_of_day = frame.time.dt.hour*60+frame.time.dt.minute
        masks = {"open": minute_of_day < 600, "mid": (minute_of_day >= 600) & (minute_of_day < 870), "close": minute_of_day >= 870}
        for metric in frame.select_dtypes("number"):
            values = frame[metric]
            valid = values.dropna()
            if valid.empty:
                continue
            total = valid.abs().sum()
            for period, mask in masks.items():
                row[metric+"_"+period+"_mean"] = values.loc[mask].mean()
                row[metric+"_"+period+"_abs_share"] = values.loc[mask].abs().sum()/total if total else np.nan
            row[metric+"_close_minus_open"] = row[metric+"_close_mean"]-row[metric+"_open_mean"]
            row[metric+"_max"], row[metric+"_min"] = valid.max(), valid.min()
            row[metric+"_range"] = valid.max()-valid.min()
            for name, index in (("argmax", valid.idxmax()), ("argmin", valid.idxmin())):
                time = frame.loc[index, "time"]
                row[metric+"_"+name+"_minute"] = time.hour*60+time.minute
                row[metric+"_"+name+"_time"] = time.strftime("%H:%M")
        results[symbol] = row
    return pd.DataFrame.from_dict(results, orient="index")
