"""Clock-time event returns and signed quote impact with coverage."""
import numpy as np
import pandas as pd


def event_returns(data, with_events=False):
    results = {}
    events_out = []
    for symbol, trades in data.trades.groupby("symbol"):
        trades = trades.sort_values("time")
        row = {}
        detail = trades[["symbol", "time"]].reset_index(drop=True).copy()
        for horizon in (1, 5, 10, 30):
            future = trades[["time", "price"]].rename(columns={"time": "future_time", "price": "future_price"})
            events = trades.assign(target=trades.time + pd.Timedelta(minutes=horizon))
            joined = pd.merge_asof(events, future, left_on="target", right_on="future_time", direction="forward", tolerance=pd.Timedelta("30s"))
            # Do not bridge the lunch break or carry signals past the session end.
            same_session = (joined.time.dt.hour < 12) == (joined.future_time.dt.hour < 12)
            ret = (joined.future_price / joined.price - 1).where(same_session)
            signed = ret * joined.direction.map({"buy": 1, "sell": -1})
            row[f"signed_trade_impact_{horizon}m"] = signed.mean()
            for group, mask in {"large": joined["size"].isin(["large", "xl"]), "buy": joined.direction == "buy", "sell": joined.direction == "sell"}.items():
                values = ret.loc[mask].dropna()
                row[f"{group}_ret_{horizon}m"] = values.mean()
                row[f"{group}_samples_{horizon}m"] = len(values)
                detail[f"{group}_ret_{horizon}m"] = ret.where(mask).to_numpy()
                detail[f"{group}_samples_{horizon}m"] = ret.where(mask).notna().astype(int).to_numpy()
            valid = signed.loc[joined["size"].isin(["large", "xl"])].dropna()
            row[f"continuation_{horizon}m"] = (valid > 0).mean() if len(valid) else np.nan
            row[f"reversal_{horizon}m"] = (valid < 0).mean() if len(valid) else np.nan
            row[f"continuation_samples_{horizon}m"] = len(valid)
            row[f"reversal_samples_{horizon}m"] = len(valid)
            detail[f"signed_trade_impact_{horizon}m"] = signed.to_numpy()
            detail[f"continuation_{horizon}m"] = (signed > 0).astype(float).where(signed.notna() & joined["size"].isin(["large", "xl"])).to_numpy()
            detail[f"reversal_{horizon}m"] = (signed < 0).astype(float).where(signed.notna() & joined["size"].isin(["large", "xl"])).to_numpy()
            flag = signed.notna() & joined["size"].isin(["large", "xl"])
            detail[f"continuation_samples_{horizon}m"] = flag.astype(int).to_numpy()
            detail[f"reversal_samples_{horizon}m"] = flag.astype(int).to_numpy()
        quote = data.quotes.loc[data.quotes.symbol == symbol, ["time", "bid_price1", "ask_price1"]].copy()
        quote["mid"] = (quote.bid_price1 + quote.ask_price1) / 2
        row["quote_match_ratio"] = np.nan
        if not quote.empty:
            matched = pd.merge_asof(trades, quote[["time", "mid"]], on="time", direction="backward", tolerance=pd.Timedelta("30s"))
            sign = matched.direction.map({"buy": 1, "sell": -1})
            row["effective_spread"] = (2 * sign * (matched.price - matched.mid) / matched.mid).mean()
            row["quote_match_ratio"] = matched.mid.notna().mean()
            matched["target"] = matched.time + pd.Timedelta("5min")
            later = pd.merge_asof(matched, quote[["time", "mid"]].rename(columns={"time": "quote_time", "mid": "future_mid"}), left_on="target", right_on="quote_time", direction="forward", tolerance=pd.Timedelta("30s"))
            valid = (later.time.dt.hour < 12) == (later.quote_time.dt.hour < 12)
            valid &= later.future_mid.notna() & later.mid.notna() & sign.notna()
            row["realized_spread_samples_5m"] = int(valid.sum())
            row["effective_spread_5m_cohort"] = (2 * sign * (later.price-later.mid)/later.mid).where(valid).mean()
            row["realized_spread_5m"] = (2 * sign * (later.price - later.future_mid) / later.mid).where(valid).mean()
            row["signed_price_impact_5m"] = (2 * sign * (later.future_mid - later.mid) / later.mid).where(valid).mean()
            detail["effective_spread"] = (2*sign*(matched.price-matched.mid)/matched.mid).to_numpy()
            detail["realized_spread_5m"] = (2*sign*(later.price-later.future_mid)/later.mid).where(valid).to_numpy()
            detail["signed_price_impact_5m"] = (2*sign*(later.future_mid-later.mid)/later.mid).where(valid).to_numpy()
        results[symbol] = row
        events_out.append(detail)
    result = pd.DataFrame.from_dict(results, orient="index")
    return (result, pd.concat(events_out, ignore_index=True)) if with_events else result
