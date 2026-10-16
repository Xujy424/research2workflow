"""Observed best-level depth shock/recovery episodes with explicit censoring."""
import numpy as np
import pandas as pd

EVENT_COLUMNS = ["symbol", "side", "time", "end_time", "baseline_depth", "lowest_depth", "recovered", "duration_seconds", "censored_reason"]


def depth_recovery(quotes, drop_fraction=.5, recovery_fraction=.9, horizon_seconds=300, maximum_gap_seconds=60):
    if not 0 < drop_fraction < recovery_fraction <= 1 or horizon_seconds <= 0 or maximum_gap_seconds <= 0:
        raise ValueError("Invalid depth recovery settings")
    symbols = pd.Index(quotes.symbol.unique(), name="symbol")
    summary = pd.DataFrame(index=symbols)
    for side in ("bid", "ask", "total"):
        summary[side+"_depth_recovery_seconds"] = np.nan
        summary[side+"_depth_shock_count"] = np.nan
        summary[side+"_depth_recovered_count"] = np.nan
        summary[side+"_depth_censored_count"] = np.nan
        summary[side+"_depth_valid_quotes"] = 0
    events = []
    def finish(active, end, recovered, reason):
        events.append({**active, "end_time": end, "recovered": recovered,
                       "duration_seconds": (end-active["time"]).total_seconds(), "censored_reason": reason})
    for symbol, frame in quotes.groupby("symbol"):
        frame = frame.sort_values("time")
        bid = frame.get("bid_volume1", pd.Series(np.nan, index=frame.index))
        ask = frame.get("ask_volume1", pd.Series(np.nan, index=frame.index))
        for side, values in (("bid", bid), ("ask", ask), ("total", bid+ask)):
            valid = values.notna() & np.isfinite(values) & (values >= 0)
            if side == "total":
                valid &= (bid >= 0) & (ask >= 0)
            summary.loc[symbol, side+"_depth_valid_quotes"] = int(valid.sum())
            if not valid.any():
                continue
            event_start = len(events)
            previous_time, previous_depth, previous_session, active = None, None, None, None
            for time, depth in zip(frame.loc[valid, "time"], values.loc[valid]):
                session = 0 if time.hour < 12 else 1
                gap = previous_time is not None and (time-previous_time).total_seconds() > maximum_gap_seconds
                changed = previous_session is not None and session != previous_session
                if active and (gap or changed):
                    finish(active, min(previous_time, active["time"]+pd.Timedelta(seconds=horizon_seconds)), False, "session_end" if changed else "observation_gap")
                    active = None
                if active:
                    deadline = active["time"]+pd.Timedelta(seconds=horizon_seconds)
                    if time > deadline:
                        finish(active, deadline, False, "horizon")
                        active = None
                    else:
                        active["lowest_depth"] = min(active["lowest_depth"], depth)
                        if depth >= active["baseline_depth"]*recovery_fraction:
                            finish(active, time, True, "")
                            active = None
                        elif time == deadline:
                            finish(active, deadline, False, "horizon")
                            active = None
                elif previous_depth is not None and not gap and not changed and previous_depth > 0 and depth <= previous_depth*drop_fraction:
                    active = {"symbol": symbol, "side": side, "time": time, "baseline_depth": previous_depth, "lowest_depth": depth}
                previous_time, previous_depth, previous_session = time, depth, session
            if active:
                finish(active, min(previous_time, active["time"]+pd.Timedelta(seconds=horizon_seconds)), False, "end_of_data")
            selected = events[event_start:]
            recovered = [e["duration_seconds"] for e in selected if e["recovered"]]
            summary.loc[symbol, side+"_depth_shock_count"] = len(selected)
            summary.loc[symbol, side+"_depth_recovered_count"] = len(recovered)
            summary.loc[symbol, side+"_depth_censored_count"] = len(selected)-len(recovered)
            summary.loc[symbol, side+"_depth_recovery_seconds"] = np.mean(recovered) if recovered else np.nan
    return summary, pd.DataFrame(events, columns=EVENT_COLUMNS)
