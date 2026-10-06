"""Trading-sentiment instability factor from Kaiyuan series (20)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..operators import cross_sectional_rank, time_series_range
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.UpdateAlpha.operators import cross_sectional_rank, time_series_range
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class SentimentInstabilityConfig:
    return_volatility_days: int = 10
    volume_smoothing_days: int = 3
    volume_range_days: int = 5
    return_volume_corr_days: int = 6
    min_valid_ratio: float = 0.8
    min_minutes: int = 120

    def __post_init__(self):
        windows = (
            self.return_volatility_days,
            self.volume_smoothing_days,
            self.volume_range_days,
            self.return_volume_corr_days,
        )
        if any(value < 1 for value in windows):
            raise ValueError("all day windows must be positive")
        if not 0 < self.min_valid_ratio <= 1:
            raise ValueError("min_valid_ratio must be in (0, 1]")

    @property
    def lookback_days(self):
        return max(
            self.return_volatility_days,
            self.volume_smoothing_days + self.volume_range_days - 1,
            self.return_volume_corr_days,
        )


class SentimentInstabilityContext(AlphaContext):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        config=SentimentInstabilityConfig(),
        universe="self",
    ):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def read_window(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        return {
            field: np.asarray(
                self.data.read(f"m_essentials/{field}", end, start_date=start),
                dtype=np.float64,
            )
            for field in ("close", "volume")
        }


class SentimentInstabilityFactor(AlphaBase):
    meta = AlphaMeta(
        "sentiment_instability",
        "Rank composite of intraday price-volume instability ranges",
        direction=-1,
    )
    dependencies = ("m_essentials/close", "m_essentials/volume")

    @staticmethod
    def _minute_features(minute_close, minute_volume, min_minutes):
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
        volume = np.asarray(minute_volume, dtype=np.float64)[:, 1:]

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

        paired = return_valid & standardized_valid
        paired_count = paired.sum(axis=1)
        x_mean = np.divide(
            np.where(paired, returns, 0.0).sum(axis=1), paired_count,
            out=np.full(paired_count.shape, np.nan), where=paired_count > 0,
        )
        y_mean = np.divide(
            np.where(paired, standardized_volume, 0.0).sum(axis=1),
            paired_count,
            out=np.full(paired_count.shape, np.nan),
            where=paired_count > 0,
        )
        x_centered = np.where(paired, returns - x_mean[:, None, :], 0.0)
        y_centered = np.where(
            paired, standardized_volume - y_mean[:, None, :], 0.0
        )
        denominator = np.sqrt(
            np.sum(x_centered * x_centered, axis=1)
            * np.sum(y_centered * y_centered, axis=1)
        )
        return_volume_corr = np.divide(
            np.sum(x_centered * y_centered, axis=1),
            denominator,
            out=np.full(paired_count.shape, np.nan),
            where=(paired_count >= min_minutes) & (denominator > 0),
        )
        return minute_volatility, volume_distribution, return_volume_corr

    def calculate(self, asof):
        raw = self.context.read_window(asof)
        tick_count = self.context.data.axis.tick_count
        if raw is None:
            return np.full(tick_count, np.nan, dtype=np.float32)
        cfg = self.context.config
        volatility, volume_distribution, return_volume_corr = self._minute_features(
            raw["close"], raw["volume"], cfg.min_minutes
        )

        smooth = cfg.volume_smoothing_days
        smoothed_volume = np.full_like(volume_distribution, np.nan)
        for row in range(smooth - 1, len(volume_distribution)):
            sample = volume_distribution[row - smooth + 1:row + 1]
            valid = np.isfinite(sample)
            count = valid.sum(axis=0)
            smoothed_volume[row] = np.divide(
                np.where(valid, sample, 0.0).sum(axis=0),
                count,
                out=np.full(tick_count, np.nan),
                where=count == smooth,
            )

        components = (
            time_series_range(
                volatility,
                cfg.return_volatility_days,
                int(np.ceil(cfg.return_volatility_days * cfg.min_valid_ratio)),
            ),
            time_series_range(
                smoothed_volume,
                cfg.volume_range_days,
                int(np.ceil(cfg.volume_range_days * cfg.min_valid_ratio)),
            ),
            time_series_range(
                return_volume_corr,
                cfg.return_volume_corr_days,
                int(np.ceil(cfg.return_volume_corr_days * cfg.min_valid_ratio)),
            ),
        )
        mask = self.context.factor_universe_mask(asof)
        ranks = np.stack([
            cross_sectional_rank(item, mask=mask) for item in components
        ])
        valid = np.all(np.isfinite(ranks), axis=0)
        return np.where(valid, ranks.sum(axis=0), np.nan).astype(np.float32)


__all__ = [
    "SentimentInstabilityConfig",
    "SentimentInstabilityContext",
    "SentimentInstabilityFactor",
]
