"""vr_q80_ret_corr: Correlation of 80% conditional-volume residuals and signed returns.

Xingye high-frequency series 10, October 2026.
Independent Config / Context / Factor; only generic math uses operators.
Default: 10-day mean through d-1; preserve raw sign in storage.
241 stored bars: 09:30 plus 240 minute endpoints; use close-to-close returns
at 09:31-11:30 and 13:01-15:00. Population standard deviations are used.
Relative volume = volume / daily mean; conditional regressions include
an intercept, signed minute return, lagged relative volume and opening/closing
30-minute indicators.
See volume_dispersion.md for the backtest entry and time alignment.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.operators import linear_quantile_residual, time_series_corr
from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
from v2.GetData import DataPool
from v2.UpdateData.config import ROOT

@dataclass(frozen=True)
class VRQ80RetCorrConfig:
    lookback_days: int = 10
    lag_days: int = 1
    min_valid_days: int = 10
    min_minutes: int = 120

    def __post_init__(self):
        if self.lookback_days < 1 or self.lag_days < 0:
            raise ValueError('lookback_days must be positive and lag_days nonnegative')
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError('invalid min_valid_days')
        if not 5 <= self.min_minutes <= 240:
            raise ValueError('min_minutes must be in [5, 240]')

class VRQ80RetCorrContext(AlphaContext):
    """Read and cache daily features for vr_q80_ret_corr."""

    def __init__(self, root=ROOT, config=VRQ80RetCorrConfig(), universe='self'):
        self.config = config
        self._daily_cache = {}
        super().__init__(DataPool(root, asset='stock'), universe=universe)

    def _read(self, field, day):
        return np.asarray(self.data.read(field, day), dtype=float)

    def daily_features(self, day):
        key = day
        if key in self._daily_cache:
            return self._daily_cache[key]
        cfg = self.config
        n = self.data.axis.tick_count
        close = self._read('m_essentials/close', day)
        volume = self._read('m_essentials/volume', day)
        if close.shape != volume.shape or close.shape[0] not in (240, 241):
            raise ValueError('expected matching 240/241 by stock minute matrices')
        offset = int(len(close) == 241)
        if offset:
            previous, current = (close[:-1], close[1:])
        else:
            open_ = self._read('m_essentials/open', day)
            previous, current = (np.vstack((open_[:1], close[:-1])), close)
        ret = np.divide(current, previous, out=np.full_like(current, np.nan), where=(current > 0) & (previous > 0) & np.isfinite(current) & np.isfinite(previous)) - 1
        vol = volume[offset:]
        ret[~np.isfinite(vol) | (vol <= 0)] = np.nan
        tradable = self.read_row(self.tradable_field, day)[:n] == 1
        count = (np.isfinite(ret) & tradable).sum(1)
        market = np.divide(np.where(tradable & np.isfinite(ret), ret, 0).sum(1), count, out=np.full(240, np.nan), where=count > 0)
        output = {'vr_q80_ret_corr': np.full(n, np.nan)}
        for stock in np.flatnonzero(tradable):
            r, v = (ret[:, stock], vol[:, stock])
            good = np.isfinite(r) & np.isfinite(market)
            if good.sum() < cfg.min_minutes:
                continue
            mean_volume = np.mean(v[np.isfinite(r)])
            relative = v / mean_volume
            lag = np.r_[np.nan, relative[:-1]]
            qgood = good & np.isfinite(lag) & (lag > 0)
            if qgood.sum() < cfg.min_minutes:
                continue
            t = np.arange(240)
            design = np.column_stack((np.ones(240), r, lag, t < 30, t >= 210))
            high = np.full(240, np.nan)
            high[qgood] = linear_quantile_residual(relative[qgood], design[qgood], 0.8)
            output['vr_q80_ret_corr'][stock] = time_series_corr(high, r, 3)
            continue
        self._daily_cache[key] = output
        oldest = day - cfg.lookback_days - cfg.lag_days
        for item in list(self._daily_cache):
            if item < oldest:
                del self._daily_cache[item]
        return output

class VRQ80RetCorrFactor(AlphaBase):
    meta = AlphaMeta('vr_q80_ret_corr', 'Correlation of 80% conditional-volume residuals and signed returns', direction=-1)
    dependencies = ('m_essentials/close', 'm_essentials/volume', 'basic/tradable')

    def calculate(self, asof):
        cfg, axis = (self.context.config, self.context.data.axis)
        end = axis.date_position(pd.Timestamp(asof).date()) - cfg.lag_days
        start = end - cfg.lookback_days + 1
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)
        rows = np.stack([self.context.daily_features(day)[self.meta.name] for day in range(start, end + 1)])
        valid = np.isfinite(rows)
        count = valid.sum(0)
        return np.divide(np.where(valid, rows, 0).sum(0), count, out=np.full(axis.tick_count, np.nan), where=count >= cfg.min_valid_days).astype(np.float32)
__all__ = ['VRQ80RetCorrConfig', 'VRQ80RetCorrContext', 'VRQ80RetCorrFactor']
