"""Standalone interactive report with all numeric features exposed."""
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
from plotly.offline import get_plotlyjs
from .report.conclusions import render_conclusions
from .statistics.memberships import membership_history, group_minute_history


plotly_bundle = lru_cache(maxsize=1)(get_plotlyjs)


def records(frame):
    return pack_table(frame.reset_index())


def pack_table(frame):
    # Column names are stored once, not repeated for every historical observation.
    return json.loads(frame.to_json(orient="split", index=False, date_format="iso"))


def generate_explorer(result, market_history, directory, date):
    memberships = membership_history(directory, date, result.universe_memberships)
    directories = sorted(p for p in Path(directory).iterdir() if p.is_dir() and p.name.isdigit() and p.name <= date.replace("-", ""))
    bars = []
    for day in directories[-21:]:
        file = day/"bars_1min.csv"
        if file.exists():
            feature_file = day/"bars_feature_1min.csv"
            frame = pd.read_csv(feature_file if feature_file.exists() else file, parse_dates=["time"], dtype={"symbol": str})
            frame["date"] = pd.Timestamp(day.name).strftime("%Y-%m-%d")
            frame["clock"] = frame.time.dt.strftime("%H:%M")
            bars.append(frame)
    minute = pd.concat(bars, ignore_index=True) if bars else result.bars["1min"].assign(date=date, clock=lambda x: x.time.dt.strftime("%H:%M"))
    event_distributions = {}
    edges = np.concatenate(([0], np.geomspace(1, 1e12, 97)))
    for kind in ("orders", "trades"):
        frame = getattr(result.clean_data, kind).copy()
        if kind == "orders":
            status = frame.get("status", pd.Series("submit", index=frame.index)).astype(str).str.lower()
            frame = frame.loc[~status.isin(["cancel", "cancelled", "canceled", "fill"])]
        for symbol in ["all", *sorted(frame.symbol.unique())]:
            values = frame.amount if symbol == "all" else frame.loc[frame.symbol == symbol, "amount"]
            counts, _ = np.histogram(values.dropna(), bins=edges)
            event_distributions[kind+":"+symbol] = {"x": ((edges[:-1]+edges[1:])/2).tolist(), "counts": counts.tolist(), "samples": int(values.notna().sum())}
    dated_distributions = {}
    for day in directories[-21:]:
        file = day/"event_distributions.json"
        if day.name < date.replace("-", "") and file.exists():
            saved = json.loads(file.read_text(encoding="utf-8"))
            dated_distributions[pd.Timestamp(day.name).strftime("%Y-%m-%d")] = saved
    (Path(directory)/date.replace("-", "")/"event_distributions.json").write_text(json.dumps(event_distributions), encoding="utf-8")
    scopes = {"universe."+k: {"kind": "group", "name": k, "members": v} for k, v in result.universe_memberships.items()}
    reference = "hs300" if "hs300" in result.universe_memberships else "all"
    for name in result.universe_gaps.index:
        first = name[:-(len(reference)+1)]
        scopes["gap."+name] = {"kind": "gap", "firstGroup": first, "secondGroup": reference, "first": result.universe_memberships.get(first, []), "second": result.universe_memberships.get(reference, [])}
    scope_bars = {key: pack_table(pd.DataFrame(rows)) for key, rows in group_minute_history(minute, memberships, scopes).items()}
    payload = {"date": date, "source": result.data_source, "market": records(market_history.rename_axis("date")),
               "stocks": records(result.security_history), "features": records(result.features.rename_axis("symbol")),
               "baseline": records(result.baseline.rename_axis("metric")), "stockBaseline": records(result.security_baseline),
               "bars": records(minute), "scopes": scopes, "memberships": memberships, "scopeBars": scope_bars, "datedDistributions": dated_distributions, "eventDistributions": event_distributions, "metrics": list(result.features.select_dtypes("number")),
               "marketMetrics": list(market_history.select_dtypes("number")),
               "barMetrics": [c for c in minute.select_dtypes("number") if c not in ("date", "symbol")]}
    template = (Path(__file__).parent/"report"/"explorer.html").read_text(encoding="utf-8")
    template = template.replace("/*PLOTLY*/", plotly_bundle()).replace("/*PAYLOAD*/", json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace("</", "<\\/"))
    template = template.replace("<!--CONCLUSIONS-->", render_conclusions(market_history, result.features, date, result.regime))
    path = Path(directory)/"monitor.html"
    path.write_text(template, encoding="utf-8")
    result.security_baseline.to_csv(Path(directory)/date.replace("-", "")/"security_baseline.csv")
    return path
