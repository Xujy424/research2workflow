"""Per-security baselines derived from dated feature artifacts."""
from pathlib import Path
import pandas as pd
from . import compare_with_history
from .tables import read_features


def security_history(directory, today, date):
    pieces = []
    for day in sorted(Path(directory).iterdir()):
        if day.is_dir() and day.name.isdigit() and day.name < date.replace("-", "") and (day/"features.csv").exists():
            frame = read_features(day/"features.csv")
            frame.index = frame.index.astype(str)
            frame["date"] = pd.Timestamp(day.name)
            frame["symbol"] = frame.index
            pieces.append(frame.set_index(["date", "symbol"]))
    current = today.copy()
    current["date"], current["symbol"] = pd.Timestamp(date), today.index
    pieces.append(current.set_index(["date", "symbol"]))
    history = pd.concat(pieces).sort_index()
    results = []
    for symbol, row in today.select_dtypes("number").iterrows():
        past = history.xs(str(symbol), level="symbol") if str(symbol) in history.index.get_level_values("symbol") else pd.DataFrame()
        baseline = compare_with_history(row, past.select_dtypes("number"), date)
        baseline["symbol"], baseline["metric"] = symbol, baseline.index
        results.append(baseline.set_index(["symbol", "metric"]))
    return history, pd.concat(results)
