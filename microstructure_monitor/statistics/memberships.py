"""Persist dated memberships and reuse daily aggregation for group minute series."""
import json
from pathlib import Path
import pandas as pd
from .aggregation import aggregate_features
from .curves import curve_statistics


def membership_history(directory, date, groups):
    directory = Path(directory)
    current = directory/date.replace("-", "")
    current.mkdir(parents=True, exist_ok=True)
    snapshot = {name: [str(symbol) for symbol in members] for name, members in groups.items()}
    (current/"memberships.json").write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")
    history = {}
    for day in sorted(directory.iterdir()):
        file = day/"memberships.json"
        if day.is_dir() and day.name.isdigit() and day.name <= date.replace("-", "") and file.exists():
            history[pd.Timestamp(day.name).strftime("%Y-%m-%d")] = json.loads(file.read_text(encoding="utf-8"))
    return history


def group_minute_history(minute, memberships, scopes):
    output = {key: [] for key in scopes}
    for date, frame in minute.groupby("date", sort=True):
        groups = memberships.get(date)
        if groups is None:
            continue
        summary = {name: curve_statistics(aggregate_features(frame.loc[frame.symbol.isin(members)], by="clock"))[0] for name, members in groups.items()}
        for key, scope in scopes.items():
            if scope["kind"] == "group":
                name = scope["name"]
                if name not in summary:
                    continue
                row = summary[name]
            else:
                first, second = scope["firstGroup"], scope["secondGroup"]
                if first not in groups or second not in groups:
                    continue
                row = summary[first]-summary[second]
            output[key].extend({"date": date, **record} for record in json.loads(row.rename_axis("clock").reset_index().to_json(orient="records")))
    return output
