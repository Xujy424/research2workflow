"""Coverage and identity checks for reproducible historical comparisons."""
import json
from pathlib import Path
import pandas as pd

METRIC_VERSION = "5.0"


def ensure_history_identity(directory, symbols, universe_id=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    identity = {"metric_version": METRIC_VERSION, "universe_id": universe_id or "fixed:" + ",".join(sorted(symbols))}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("Historical metric version/universe differs; use a separate output directory")
    elif (directory / "history.csv").exists():
        raise ValueError("Legacy history has no identity; use a new output directory and regenerate")
    else:
        path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity


def quality_features(data):
    result = pd.DataFrame(index=sorted(data.trades.symbol.unique()))
    context = data.order_context if data.order_context is not None else data.orders
    status = context.get("status", pd.Series("submit", index=context.index)).astype(str).str.lower()
    submitted = context.loc[~status.isin(["cancel", "cancelled", "canceled", "fill"])]
    keys = ["symbol", "side", "order_id"]
    submitted_index = pd.MultiIndex.from_frame(submitted[keys])
    for side, column in (("buy", "buy_order_id"), ("sell", "sell_order_id")):
        legs = data.trades[["symbol", column]].rename(columns={column: "order_id"}).assign(side=side)
        matched = pd.Series(pd.MultiIndex.from_frame(legs[keys]).isin(submitted_index), index=legs.index)
        result[side + "_order_match_ratio"] = matched.groupby(legs.symbol).mean()
    result["direction_known_ratio"] = data.trades.direction.ne("unknown").groupby(data.trades.symbol).mean()
    result["priced_order_ratio"] = submitted.price.gt(0).groupby(submitted.symbol).mean()
    cancels = context.loc[status.isin(["cancel", "cancelled", "canceled"])]
    matched = pd.Series(pd.MultiIndex.from_frame(cancels[keys]).isin(submitted_index), index=cancels.index)
    result["cancel_order_match_ratio"] = matched.groupby(cancels.symbol).mean()
    result["quotes_count"] = data.quotes.groupby("symbol").size().reindex(result.index, fill_value=0)
    result["duplicate_trade_rows"] = data.trades.duplicated(["symbol", "time", "price", "volume", "buy_order_id", "sell_order_id"]).groupby(data.trades.symbol).sum()
    return result
