"""Empirical distribution and normalized intraday curve differences."""
import numpy as np
import pandas as pd


def ks_distance(first, second):
    first, second = np.sort(first.dropna()), np.sort(second.dropna())
    if not len(first) or not len(second):
        return np.nan
    grid = np.sort(np.concatenate([first, second]))
    return np.max(np.abs(np.searchsorted(first, grid, side="right") / len(first) - np.searchsorted(second, grid, side="right") / len(second)))


def compare_previous(features, minute, directory, date):
    candidates = sorted(p for p in directory.iterdir() if p.is_dir() and p.name.isdigit() and p.name < date.replace("-", "") and (p / "features.csv").exists())
    if not candidates:
        return pd.DataFrame(columns=["reference_date", "ks_distance", "curve_l1"])
    reference = candidates[-1]
    previous = pd.read_csv(reference / "features.csv", index_col=0)
    rows = {}
    # Fixed intersection prevents entry/exit alone changing distribution distances.
    common = features.index.intersection(previous.index)
    for metric in features.select_dtypes("number"):
        if metric in previous:
            rows[metric] = {"reference_date": reference.name, "ks_distance": ks_distance(features.loc[common, metric], previous.loc[common, metric]), "common_securities": len(common)}
    bar_path = reference / "bars_1min.csv"
    if bar_path.exists():
        old = pd.read_csv(bar_path, parse_dates=["time"])
        for metric in ("amount", "large_amount", "buy_amount"):
            curves = []
            for frame in (minute, old):
                frame = frame.loc[frame.symbol.isin(common)]
                curve = frame.groupby(frame.time.dt.strftime("%H:%M"))[metric].sum()
                total = curve.sum()
                curves.append(curve / total if total else curve * np.nan)
            aligned = pd.concat(curves, axis=1).fillna(0)
            if all(curve.notna().any() for curve in curves):
                rows.setdefault(metric, {"reference_date": reference.name})["curve_l1"] = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum()
    return pd.DataFrame.from_dict(rows, orient="index")
