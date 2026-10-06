"""Intraday volatility, jump, volume-shape, and price-path factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
from ..operators import safe_ratio_return
from ...GetData import DataPool
from ...UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class CalendarIntradayConfig:
    lookback_days: int = 20
    min_valid_days: int = 12
    bar_minutes: int = 5

    def __post_init__(self):
        if self.lookback_days < 1 or self.bar_minutes < 1:
            raise ValueError("lookback_days and bar_minutes must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be within lookback_days")


class CalendarIntradayContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=CalendarIntradayConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def window(self, asof, fields):
        end = self.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        return {
            field: np.asarray(self.data.read(field, end, start), dtype=np.float64)
            for field in fields
        }


def _daily_mean(values, min_count):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.divide(
        np.where(valid, values, 0.0).sum(axis=0), count,
        out=np.full(values.shape[1], np.nan), where=count >= min_count,
    )


def _bar_returns(close, minutes):
    endpoints = close[:, minutes - 1::minutes]
    return safe_ratio_return(endpoints[:, 1:], endpoints[:, :-1])


def _moment(values, order):
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    mean = np.divide(
        np.where(valid, values, 0.0).sum(axis=1), count,
        out=np.full((values.shape[0], values.shape[2]), np.nan), where=count > 0,
    )
    centered = np.where(valid, values - mean[:, None, :], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), count,
        out=np.full_like(mean, np.nan), where=count > 1,
    )
    numerator = np.divide(
        np.sum(centered ** order, axis=1), count,
        out=np.full_like(mean, np.nan), where=count > order,
    )
    return np.divide(
        numerator, variance ** (order / 2), out=np.full_like(mean, np.nan),
        where=variance > 0,
    )


def _jump_components(returns):
    valid = np.isfinite(returns)
    enough = valid.sum(axis=1) >= 2
    squared = np.where(valid, returns * returns, 0.0)
    rv = squared.sum(axis=1)
    pairs = valid[:, 1:] & valid[:, :-1]
    bpv = (np.pi / 2.0) * np.where(
        pairs, np.abs(returns[:, 1:] * returns[:, :-1]), 0.0
    ).sum(axis=1)
    jump = np.maximum(rv - bpv, 0.0)
    up_share = np.divide(
        np.where(valid & (returns > 0), squared, 0.0).sum(axis=1), rv,
        out=np.full_like(rv, np.nan), where=rv > 0,
    )
    down_share = np.divide(
        np.where(valid & (returns < 0), squared, 0.0).sum(axis=1), rv,
        out=np.full_like(rv, np.nan), where=rv > 0,
    )
    rv[~enough] = np.nan
    up_share[~enough] = np.nan
    down_share[~enough] = np.nan
    return rv, jump * up_share, jump * down_share


def _realized(returns, selector=None):
    valid = np.isfinite(returns)
    if selector is not None:
        valid &= selector(returns)
    result = np.where(valid, returns * returns, 0.0).sum(axis=1)
    result[np.isfinite(returns).sum(axis=1) < 2] = np.nan
    return result


class _CalendarIntradayFactor(AlphaBase):
    def empty(self):
        return np.full(self.context.data.axis.tick_count, np.nan, dtype=np.float32)

    def aggregate(self, daily):
        return _daily_mean(daily, self.context.config.min_valid_days).astype(np.float32)

    def returns(self, asof):
        raw = self.context.window(asof, ("m_essentials/close",))
        if raw is None:
            return None
        return _bar_returns(raw["m_essentials/close"], self.context.config.bar_minutes)
