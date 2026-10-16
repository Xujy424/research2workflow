"""Empirical distribution and normalized intraday curve differences."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .aggregation import aggregate_features
from .tables import read_features


def ks_distance(first, second):
    first, second = np.sort(first.dropna()), np.sort(second.dropna())
    if not len(first) or not len(second):
        return np.nan
    return float(ks_2samp(first, second, method="asymp").statistic)


def compare_previous(features, minute, directory, date, lookback=20, minimum_samples=20):
    candidates = sorted(p for p in directory.iterdir() if p.is_dir() and p.name.isdigit() and p.name < date.replace("-", "") and (p / "features.csv").exists())
    if not candidates:
        return pd.DataFrame(columns=["reference_date", "ks_distance", "curve_l1"])
    reference = candidates[-1]
    previous = pd.concat([read_features(p / "features.csv") for p in candidates[-lookback:]])
    rows = {}
    # Fixed intersection prevents entry/exit alone changing distribution distances.
    common = features.index.intersection(previous.index)
    for metric in features.select_dtypes("number"):
        if metric in previous:
            current = features.loc[common, metric].dropna()
            historical = previous.loc[previous.index.isin(common), metric].dropna()
            distance = ks_distance(current, historical) if len(current) >= minimum_samples and len(historical) >= minimum_samples else np.nan
            pvalue = ks_2samp(current, historical, method="asymp").pvalue if pd.notna(distance) else np.nan
            rows[metric] = {"reference_date": reference.name, "ks_distance": distance, "ks_pvalue_exploratory": pvalue, "common_securities": len(common), "historical_samples": len(historical), "current_samples": len(current)}
    historical_curves = []
    for day in candidates[-lookback:]:
        feature_path = day / "bars_feature_1min.csv"
        bar_path = feature_path if feature_path.exists() else day / "bars_1min.csv"
        if bar_path.exists():
            frame = pd.read_csv(bar_path, parse_dates=["time"], dtype={"symbol": str})
            frame = frame.loc[frame.symbol.isin(common)].copy()
            frame["clock"] = frame.time.dt.strftime("%H:%M")
            curve = aggregate_features(frame, by="clock")
            historical_curves.append(curve.div(curve.abs().sum().replace(0, 1)))
    if historical_curves and not minute.empty:
        frame = minute.loc[minute.symbol.isin(common)].copy()
        frame["clock"] = frame.time.dt.strftime("%H:%M")
        current_curve = aggregate_features(frame, by="clock")
        reference_curve = pd.concat(historical_curves).groupby(level=0).mean()
        for metric in current_curve:
            if metric not in reference_curve or metric == "securities":
                continue
            aligned = pd.concat([current_curve[metric], reference_curve[metric]], axis=1).dropna()
            if aligned.empty:
                continue
            # Shared observed clocks only; signed curves preserve direction.
            scales = aligned.abs().sum()
            normalized = aligned.div(scales.replace(0, 1))
            row = rows.setdefault(metric, {"reference_date": reference.name})
            row["curve_l1"] = (normalized.iloc[:, 0]-normalized.iloc[:, 1]).abs().sum()
            row["curve_common_minutes"] = len(aligned)
            row["curve_history_days"] = len(historical_curves)
    return pd.DataFrame.from_dict(rows, orient="index")
