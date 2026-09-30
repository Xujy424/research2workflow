"""Shared table contracts used by the microstructure monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


REQUIRED_ORDER_COLUMNS = ("time", "symbol", "order_id", "price", "volume", "side")
REQUIRED_TRADE_COLUMNS = (
    "time",
    "symbol",
    "price",
    "volume",
    "buy_order_id",
    "sell_order_id",
)
REQUIRED_QUOTE_COLUMNS = ("time", "symbol", "bid_price1", "ask_price1")


@dataclass(frozen=True)
class MarketData:
    orders: pd.DataFrame
    trades: pd.DataFrame
    quotes: pd.DataFrame
    daily: pd.DataFrame | None = None


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, pd.NA)


def flatten_feature_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for namespace, frame in frames.items():
        if frame is None or frame.empty:
            continue
        renamed = frame.add_prefix(f"{namespace}.")
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)
