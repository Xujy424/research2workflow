"""Improved APM factor from Kaiyuan Securities microstructure series (5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class APMConfig:
    lookback_days: int = 20
    # Current StockPriceOneMin layout has 241 sorted bars; index 121 is the
    # first afternoon bar. Keep configurable if the upstream BarTime changes.
    afternoon_start_bar: int = 121
    min_time_series_observations: int = 24
    min_paired_days: int = 12
    min_cross_section_observations: int = 30


class APMContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=APMConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))


def _ratio_return(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=(np.isfinite(numerator) & np.isfinite(denominator)
               & (numerator > 0) & (denominator > 0)),
    ) - 1.0


def _market_adjusted_residuals(stock_returns, market_returns, min_observations):
    """OLS residuals of each stock on one market series, with intercept."""
    y = np.asarray(stock_returns, float)
    x = np.broadcast_to(np.asarray(market_returns, float)[:, None], y.shape)
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    x_mean = np.divide(np.where(valid, x, 0).sum(0), count,
                       out=np.zeros(y.shape[1]), where=count > 0)
    y_mean = np.divide(np.where(valid, y, 0).sum(0), count,
                       out=np.zeros(y.shape[1]), where=count > 0)
    xc = np.where(valid, x - x_mean, 0)
    yc = np.where(valid, y - y_mean, 0)
    denominator = np.sum(xc * xc, axis=0)
    beta = np.divide(
        np.sum(xc * yc, axis=0), denominator,
        out=np.full(y.shape[1], np.nan),
        where=(count >= min_observations) & (denominator > 0),
    )
    residual = y - ((y_mean - beta * x_mean) + x * beta)
    return np.where(valid & np.isfinite(beta), residual, np.nan)


def _paired_t_stat(first, second, min_days):
    delta = np.asarray(first, float) - np.asarray(second, float)
    valid = np.isfinite(delta)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, delta, 0).sum(0), count,
                     out=np.full(delta.shape[1], np.nan), where=count > 0)
    centered = np.where(valid, delta - mean, 0)
    variance = np.divide(np.sum(centered * centered, axis=0), count - 1,
                         out=np.full(delta.shape[1], np.nan), where=count > 1)
    standard_error = np.sqrt(variance / count)
    return np.divide(mean, standard_error,
                     out=np.full(delta.shape[1], np.nan),
                     where=(count >= min_days) & (standard_error > 0))


def _cross_section_residual(y, x, min_observations):
    y, x = np.asarray(y, float), np.asarray(x, float)
    valid = np.isfinite(x) & np.isfinite(y)
    result = np.full_like(y, np.nan)
    if valid.sum() < min_observations:
        return result
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    coefficient, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    result[valid] = y[valid] - design @ coefficient
    return result


class APMFactor(AlphaBase):
    """APMnew: overnight-versus-afternoon residual-difference t-stat.

    The report uses index returns as the market regressor. Because the local
    factor pool has no aligned index-minute module, this implementation uses
    equal-weight valid-stock segment returns as a reproducible market proxy.
    """

    meta = AlphaMeta("apm", "20D improved overnight-afternoon APM", direction=1)
    dependencies = (
        "d_essentials/open_adj", "d_essentials/close_adj",
        "d_essentials/pct", "m_essentials/open",
        "m_essentials/close", "basic/tradable",
    )

    def calculate(self, asof):
        cfg, data = self.context.config, self.context.data
        axis = data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days + 1
        previous = start - 1
        if previous < 0:
            return np.full(axis.tick_count, np.nan, np.float32)

        open_adj = np.asarray(data.read("d_essentials/open_adj", end, previous), float)
        close_adj = np.asarray(data.read("d_essentials/close_adj", end, previous), float)
        overnight = _ratio_return(open_adj[1:], close_adj[:-1])
        minute_open = np.asarray(data.read("m_essentials/open", end, start), float)
        minute_close = np.asarray(data.read("m_essentials/close", end, start), float)
        bar = cfg.afternoon_start_bar
        if not 0 <= bar < minute_open.shape[1]:
            raise ValueError(f"afternoon_start_bar={bar} outside {minute_open.shape[1]} bars")
        afternoon = _ratio_return(minute_close[:, -1], minute_open[:, bar])  # T，N

        tradable = np.asarray(data.read("basic/tradable", end, start), bool)
        overnight = np.where(tradable, overnight, np.nan)
        afternoon = np.where(tradable, afternoon, np.nan)
        stock_observations = np.concatenate((overnight, afternoon), axis=0)
        market_observations = np.concatenate(
            (np.nanmean(overnight, axis=1), np.nanmean(afternoon, axis=1))
        )
        residual = _market_adjusted_residuals(
            stock_observations, market_observations,
            cfg.min_time_series_observations,
        )
        statistic = _paired_t_stat(
            residual[:cfg.lookback_days], residual[cfg.lookback_days:],
            cfg.min_paired_days,
        )

        daily_pct = np.asarray(data.read("d_essentials/pct", end, start), float) / 100
        valid_daily = np.isfinite(daily_pct) & tradable
        ret20 = np.prod(1 + np.where(valid_daily, daily_pct, 0), axis=0) - 1
        ret20[valid_daily.sum(axis=0) < cfg.min_paired_days] = np.nan
        return _cross_section_residual(
            statistic, ret20, cfg.min_cross_section_observations
        ).astype(np.float32)


__all__ = ["APMConfig", "APMContext", "APMFactor"]
