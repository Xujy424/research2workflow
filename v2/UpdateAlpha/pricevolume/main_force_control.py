"""Main-force control ability factor from Kaiyuan series (20)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..operators import cross_sectional_rank, time_series_corr
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.UpdateAlpha.operators import cross_sectional_rank, time_series_corr
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class MainForceControlConfig:
    lookback_days: int = 10
    min_valid_days: int = 8
    min_minutes: int = 120

    def __post_init__(self):
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if not 2 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [2, lookback_days]")


class MainForceControlContext(AlphaContext):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        config=MainForceControlConfig(),
        universe="self",
    ):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def read_window(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        previous = start - 1
        if previous < 0:
            return None
        result = {
            "minute_close": np.asarray(
                self.data.read("m_essentials/close", end, start_date=start),
                dtype=np.float64,
            ),
            "minute_volume": np.asarray(
                self.data.read("m_essentials/volume", end, start_date=start),
                dtype=np.float64,
            ),
        }
        for field in ("high", "low", "close"):
            name = f"{field}_ratio_adj"
            result[name] = np.asarray(
                self.data.read(
                    f"d_essentials/{name}", end, start_date=previous
                ),
                dtype=np.float64,
            )
        return result


class MainForceControlFactor(AlphaBase):
    meta = AlphaMeta(
        "main_force_control",
        "Negative rank composite of amplitude and intraday-volatility correlations",
        direction=1,
    )
    dependencies = (
        "m_essentials/close",
        "m_essentials/volume",
        "d_essentials/high_ratio_adj",
        "d_essentials/low_ratio_adj",
        "d_essentials/close_ratio_adj",
    )

    @staticmethod
    def _minute_volatility_features(minute_close, minute_volume, min_minutes):
        close = np.asarray(minute_close, dtype=np.float64)
        returns = np.divide(
            close[:, 1:],
            close[:, :-1],
            out=np.full_like(close[:, 1:], np.nan),
            where=(
                np.isfinite(close[:, 1:])
                & np.isfinite(close[:, :-1])
                & (close[:, 1:] > 0)
                & (close[:, :-1] > 0)
            ),
        ) - 1.0
        return_valid = np.isfinite(returns)
        return_count = return_valid.sum(axis=1)
        return_mean = np.divide(
            np.where(return_valid, returns, 0.0).sum(axis=1),
            return_count,
            out=np.full(return_count.shape, np.nan, dtype=np.float64),
            where=return_count > 0,
        )
        return_centered = np.where(
            return_valid, returns - return_mean[:, None, :], 0.0
        )
        minute_volatility = np.sqrt(np.divide(
            np.sum(return_centered * return_centered, axis=1),
            return_count,
            out=np.full(return_count.shape, np.nan, dtype=np.float64),
            where=return_count >= min_minutes,
        ))

        volume = np.asarray(minute_volume, dtype=np.float64)[:, 1:]
        volume_valid = np.isfinite(volume) & (volume >= 0)
        volume_count = volume_valid.sum(axis=1)
        volume_mean = np.divide(
            np.where(volume_valid, volume, 0.0).sum(axis=1),
            volume_count,
            out=np.full(volume_count.shape, np.nan, dtype=np.float64),
            where=volume_count > 0,
        )
        standardized_volume = np.divide(
            volume,
            volume_mean[:, None, :],
            out=np.full_like(volume, np.nan),
            where=volume_valid & (volume_mean[:, None, :] > 0),
        )
        standardized_valid = np.isfinite(standardized_volume)
        volume_centered = np.where(
            standardized_valid, standardized_volume - 1.0, 0.0
        )
        standardized_count = standardized_valid.sum(axis=1)
        volume_distribution = np.sqrt(np.divide(
            np.sum(volume_centered * volume_centered, axis=1),
            standardized_count,
            out=np.full(volume_count.shape, np.nan, dtype=np.float64),
            where=standardized_count >= min_minutes,
        ))
        return minute_volatility, volume_distribution

    def calculate(self, asof):
        raw = self.context.read_window(asof)
        tick_count = self.context.data.axis.tick_count
        if raw is None:
            return np.full(tick_count, np.nan, dtype=np.float32)
        cfg = self.context.config
        volatility, volume_distribution = self._minute_volatility_features(
            raw["minute_close"], raw["minute_volume"], cfg.min_minutes
        )
        prior_close = raw["close_ratio_adj"][:-1]
        amplitude = np.divide(
            raw["high_ratio_adj"][1:] - raw["low_ratio_adj"][1:],
            prior_close,
            out=np.full_like(prior_close, np.nan),
            where=(
                np.isfinite(raw["high_ratio_adj"][1:])
                & np.isfinite(raw["low_ratio_adj"][1:])
                & np.isfinite(prior_close)
                & (prior_close > 0)
            ),
        )
        correlations = (
            time_series_corr(volatility, amplitude, cfg.min_valid_days),
            time_series_corr(volume_distribution, amplitude, cfg.min_valid_days),
        )
        ranks = np.stack([cross_sectional_rank(item) for item in correlations])
        valid = np.all(np.isfinite(ranks), axis=0)
        return np.where(valid, -ranks.sum(axis=0), np.nan).astype(np.float32)


__all__ = [
    "MainForceControlConfig",
    "MainForceControlContext",
    "MainForceControlFactor",
]
