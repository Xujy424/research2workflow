"""Persist daily observations and rebuild historical monitoring without lookahead."""
from pathlib import Path
import pandas as pd
from .statistics import compare_with_history


def load_history(output_dir):
    path = Path(output_dir) / "history.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()


def save_observation(output_dir, date, metrics):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    history = load_history(directory)
    observation = pd.DataFrame([metrics], index=[pd.Timestamp(date)])
    history = pd.concat([history.loc[history.index != pd.Timestamp(date)], observation]).sort_index()
    path = directory / "history.csv"
    temporary = directory / "history.csv.tmp"
    history.to_csv(temporary)
    temporary.replace(path)
    return history


def historical_baselines(history):
    return {str(date.date()): compare_with_history(row, history, date) for date, row in history.iterrows()}


def detect_migration(history, thresholds=None):
    settings = thresholds or {}
    cutoff = float(settings.get("z_warn", 2))
    window = int(settings.get("persistence_window", 5))
    minimum = int(settings.get("persistence_days", 3))
    families_required = int(settings.get("resonance_families", 2))
    if not 1 <= minimum <= window or families_required < 1:
        raise ValueError("Invalid migration persistence settings")
    rows = []
    families = {"liquidity": ["liquidity.spread"], "size": ["trade.large_ratio", "trade.tail_ratio"], "flow": ["flow.imbalance"], "intraday": ["intraday.amount_close_share", "intraday.amount_argmax_minute"], "impact": ["liquidity.impact"],
                "distribution": [c for c in history if c.startswith("structural.ks.")],
                "curve": [c for c in history if c.startswith("structural.curve.")],
                "universe_gap": [c for c in history if c.startswith("gap.") and c.endswith(("liquidity.spread", "liquidity.depth", "liquidity.impact", "trade.large_ratio", "trade.xl_ratio", "flow.imbalance", "order.imbalance", "lifecycle.cancel_rate", "lifecycle.fill_rate"))]}
    required = list(dict.fromkeys(metric for metrics in families.values() for metric in metrics))
    baselines = historical_baselines(history.reindex(columns=required))
    for date, baseline in baselines.items():
        row = {"date": date}
        for family, metrics in families.items():
            z = baseline.z20.reindex(metrics).dropna()
            ma_gap = ((baseline.ma5-baseline.ma20)/baseline.std20.replace(0, float("nan"))).reindex(metrics).dropna()
            if not ma_gap.empty:
                z = pd.concat([z, ma_gap])
            row[family] = z.iloc[z.abs().argmax()] if len(z) else float("nan")
        rows.append(row)
    result = pd.DataFrame(rows).set_index("date")
    # Absolute distances have an interpretable threshold even during Z warmup.
    for family, prefix, default in (("distribution", "structural.ks.", .25), ("curve", "structural.curve.", .3)):
        columns = [c for c in history if c.startswith(prefix)]
        if columns:
            distance = history[columns].max(axis=1)
            distance.index = distance.index.strftime("%Y-%m-%d")
            threshold = float(settings.get(family + "_distance_warn", default))
            result.loc[distance.ge(threshold), family] = cutoff
    signals = result.abs().ge(cutoff)
    sustained = (result.ge(cutoff).rolling(window, min_periods=window).sum().ge(minimum)
                 | result.le(-cutoff).rolling(window, min_periods=window).sum().ge(minimum))
    result["resonance_count"] = signals.sum(axis=1)
    result["persistent_count"] = sustained.sum(axis=1)
    result["overall"] = "NORMAL"
    result.loc[result.resonance_count >= families_required, "overall"] = "TRANSITION"
    result.loc[result.persistent_count >= families_required, "overall"] = "NEW_REGIME"
    result.loc[result[list(families)].notna().sum(axis=1) < families_required, "overall"] = "INSUFFICIENT_HISTORY"
    return result
