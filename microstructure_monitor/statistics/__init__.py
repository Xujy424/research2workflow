"""Historical baselines use only observations strictly before today."""
import numpy as np
import pandas as pd
from .aggregation import aggregate_features
from .curves import curve_statistics


def compare_with_history(today, history, date):
    history = pd.DataFrame(columns=today.index) if history is None else history.copy()
    history.index = pd.to_datetime(history.index)
    history = history.loc[history.index < pd.Timestamp(date)].sort_index()
    if history.index.has_duplicates:
        raise ValueError("History must contain one market observation per date")
    history = history.reindex(columns=today.index)
    result = today.to_frame("raw")
    result["sample_count"] = history.count()
    for window in (5, 20, 60):
        sample = history.tail(window)
        mean = sample.mean().where(sample.count() >= window)
        std = sample.std().where(sample.count() >= window)
        std = std.where(std > np.finfo(float).eps * mean.abs().clip(lower=1) * 10)
        result[f"ma{window}"] = mean
        if window != 5:
            result[f"std{window}"] = std
            result[f"z{window}"] = (today - mean) / std.replace(0, np.nan)
    annual = history.tail(252)
    result["pct_1y"] = annual.le(today).sum() / annual.count().replace(0, np.nan)
    result["pct_history"] = result.pct_1y.where(annual.count() >= 60)
    result["pct_1y"] = result.pct_1y.where(annual.count() >= 252)
    result["pct_1y_samples"] = annual.count()
    return result


def compare_universes(features, universe_map=None, minute=None):
    groups = {"all": features.index, **(universe_map or {})}
    results = {}
    for name, members in groups.items():
        subset = features.loc[features.index.isin(list(members))]
        if subset.empty:
            summary = subset.mean(numeric_only=True)
            summary["securities"] = 0
        else:
            summary = aggregate_features(subset).iloc[0]
            if minute is not None:
                selected = minute.loc[minute.symbol.isin(members)].copy()
                selected["clock"] = selected.time.dt.strftime("%H:%M")
                _, derived = curve_statistics(aggregate_features(selected, by="clock"))
                summary = pd.concat([summary.drop(derived.index, errors="ignore"), derived])
        results[name] = summary
    return pd.DataFrame(results).T
