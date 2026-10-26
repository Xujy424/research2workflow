"""bayesHighVol_onset: Empirical-Bayes high-volume future dispersion absorption.

Xingye high-frequency series 10, October 2026.
Independent Config / Context / Factor; only generic math uses operators.
Default: 10-day mean through d-1; preserve raw sign in storage.
241 stored bars: 09:30 plus 240 minute endpoints; use close-to-close returns
at 09:31-11:30 and 13:01-15:00. Population standard deviations are used.
Relative volume = volume / daily mean; conditional regressions include
an intercept, signed minute return, lagged relative volume and opening/closing
30-minute indicators.
Market minute return is the equal-weight mean of tradable stocks.
Sparse support defaults to 24 minutes (configurable; unspecified in report).
Five-minute event windows cross lunch, never trading days.
See volume_dispersion.md for the backtest entry and time alignment.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.operators import linear_quantile_residual, least_trimmed_squares_slope
from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
from v2.GetData import DataPool
from v2.UpdateData.config import ROOT

@dataclass(frozen=True)
class BayesHighVolOnsetConfig:
    lookback_days: int = 10
    lag_days: int = 1
    min_valid_days: int = 10
    min_minutes: int = 120
    sparse_minutes: int = 24
    horizon: int = 5
    min_event_minutes: int = 3
    min_control_minutes: int = 5
    show_progress: bool = True
    workers: int = 0

    def __post_init__(self):
        if self.lookback_days < 1 or self.lag_days < 0:
            raise ValueError('lookback_days must be positive and lag_days nonnegative')
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError('invalid min_valid_days')
        if not 5 <= self.min_minutes <= 240:
            raise ValueError('min_minutes must be in [5, 240]')
        if not 0 <= self.sparse_minutes < self.min_minutes - 1:
            raise ValueError('sparse_minutes must leave at least two ordinary minutes')
        if not 1 <= self.horizon < 240:
            raise ValueError('invalid horizon')
        if min(self.min_event_minutes, self.min_control_minutes) < 1:
            raise ValueError('event/control thresholds must be positive')
        if self.workers < 0:
            raise ValueError('workers must be nonnegative; 0 means automatic')

def sparse_gamma(y, market, sparsity):
    """Report-specific diagonal model; the scalar trimmed LS solver is generic."""
    x = np.abs(np.asarray(market, float))
    y = np.asarray(y, float)
    gamma = np.zeros_like(y)
    positions = np.flatnonzero(x > 0)
    s = min(sparsity, max(0, len(positions) - 1))
    if s == 0:
        return gamma
    beta = least_trimmed_squares_slope(x[positions], y[positions], s)
    residual = y[positions] - beta * x[positions]
    selected = np.argsort(residual ** 2, kind='stable')[-s:]
    output = positions[selected]
    gamma[output] = residual[selected] / x[output] ** 2
    gamma[output[np.abs(residual[selected]) <= 1e-12 * max(1, np.max(np.abs(y)))]] = 0
    return gamma

def onset_counts(gamma, residual, horizon):
    """The selected tail alone determines the event; masks share its design.

    Prefix counts preserve holes in the minute axis and make the complete
    future-window check O(T), independent of the horizon length.
    """
    valid = np.isfinite(gamma) & np.isfinite(residual)
    dispersion = gamma > 0
    prefix_valid = np.r_[0, np.cumsum(valid)]
    prefix_hit = np.r_[0, np.cumsum(dispersion)]
    future_valid = prefix_valid[horizon + 1:] - prefix_valid[1:-horizon]
    hit = prefix_hit[horizon + 1:] - prefix_hit[1:-horizon] > 0
    base = valid[:-horizon] & ~dispersion[:-horizon] & (future_valid == horizon)
    event = residual[:-horizon] > 0
    e, c = (base & event, base & ~event)
    return (e.sum(), (e & hit).sum(), c.sum(), (c & hit).sum())

def posterior_probability(n, h, market_mask):
    valid = market_mask & (n > 0) & np.isfinite(n) & np.isfinite(h)
    if not valid.any():
        return np.full_like(n, np.nan, dtype=float)
    total = n[valid].sum()
    mean = (h[valid].sum() + 0.5) / (total + 1)
    observed = np.sum(n[valid] * (h[valid] / n[valid] - mean) ** 2) / total
    noise = mean * (1 - mean) / (total / valid.sum())
    between = max(observed - noise, 1e-08)
    concentration = np.clip(mean * (1 - mean) / between - 1, 5, 200)
    return (h + mean * concentration) / (n + concentration)

def _stock_feature(task):
    stock, r, v, market, cfg, extra = task
    value = None
    good = np.isfinite(r) & np.isfinite(market)
    if good.sum() < cfg.min_minutes:
        return (stock, value)
    mean_volume = np.mean(v[np.isfinite(r)])
    relative = v / mean_volume
    lag = np.r_[np.nan, relative[:-1]]
    qgood = good & np.isfinite(lag) & (lag > 0)
    if qgood.sum() < cfg.min_minutes:
        return (stock, value)
    t = np.arange(240)
    design = np.column_stack((np.ones(240), r, lag, t < 30, t >= 210))
    residual = np.full(240, np.nan)
    residual[qgood] = linear_quantile_residual(relative[qgood], design[qgood], 0.8)
    gamma = np.full(240, np.nan)
    gamma[good] = sparse_gamma(np.abs(r[good] - market[good]), market[good], cfg.sparse_minutes)
    value = onset_counts(gamma, residual, cfg.horizon)
    return (stock, value)

class BayesHighVolOnsetContext(AlphaContext):
    """Read and cache daily features for bayesHighVol_onset."""

    def __init__(self, root=ROOT, config=BayesHighVolOnsetConfig(), universe='self'):
        self._executor = None
        self.config = config
        self._daily_cache = {}
        super().__init__(DataPool(root, asset='stock'), universe=universe)

    def _read(self, field, day):
        return np.asarray(self.data.read(field, day, copy=False), dtype=float)

    def daily_features(self, day):
        key = day
        if key in self._daily_cache:
            return self._daily_cache[key]
        cfg = self.config
        source_date = str(self.data.axis.trade_dates[day]) if hasattr(self.data.axis, 'trade_dates') else str(day)
        if cfg.show_progress:
            tqdm.write(f'[bayeshighvol_onset] reading source {source_date}', file=sys.stdout)
            sys.stdout.flush()
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
        output = {'bayeshighvol_onset': np.full(n, np.nan)}
        counts = np.zeros((4, n))
        fitted = np.zeros(n, dtype=bool)
        stocks = np.flatnonzero(tradable)
        workers = cfg.workers or (min(8, os.cpu_count() or 1) if len(stocks) >= 256 else 1)
        tasks = ((stock, ret[:, stock], vol[:, stock], market, cfg, None) for stock in stocks)
        if workers > 1:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=workers)
            results = self._executor.map(_stock_feature, tasks, chunksize=32)
        else:
            results = map(_stock_feature, tasks)
        for stock, value in tqdm(results, total=len(stocks), desc=f'bayeshighvol_onset day {day} stocks', file=sys.stdout, leave=False, mininterval=1, disable=not cfg.show_progress or not sys.stdout.isatty(), bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]', position=1):
            if value is not None:
                counts[:, stock] = value
                fitted[stock] = True
        ne, he, nc, hc = counts
        valid = fitted & (ne >= cfg.min_event_minutes) & (nc >= cfg.min_control_minutes)
        eb = posterior_probability(ne, he, fitted & tradable)
        cb = posterior_probability(nc, hc, fitted & tradable)
        output['bayeshighvol_onset'] = np.where(valid, np.log(cb) - np.log(eb), np.nan)
        self._daily_cache[key] = output
        oldest = day - cfg.lookback_days - cfg.lag_days
        for item in list(self._daily_cache):
            if item < oldest:
                del self._daily_cache[item]
        return output

    def close(self):
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
        finally:
            super().close()

class BayesHighVolOnsetFactor(AlphaBase):
    meta = AlphaMeta('bayeshighvol_onset', 'Empirical-Bayes high-volume future dispersion absorption', direction=1)
    dependencies = ('m_essentials/close', 'm_essentials/volume', 'basic/tradable')

    def calculate(self, asof):
        cfg, axis = (self.context.config, self.context.data.axis)
        end = axis.date_position(pd.Timestamp(asof).date()) - cfg.lag_days
        start = end - cfg.lookback_days + 1
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)
        if cfg.show_progress:
            cache = getattr(self.context, '_daily_cache', {})
            missing = sum((day not in cache for day in range(start, end + 1)))
            if missing:
                phase = 'window warm-up' if missing > 1 else 'daily increment'
                tqdm.write(f'[bayeshighvol_onset] {phase}: {missing} new source day(s), {cfg.lookback_days - missing} cached', file=sys.stdout)
                sys.stdout.flush()
        rows = np.stack([self.context.daily_features(day)[self.meta.name] for day in range(start, end + 1)])
        valid = np.isfinite(rows)
        count = valid.sum(0)
        return np.divide(np.where(valid, rows, 0).sum(0), count, out=np.full(axis.tick_count, np.nan), where=count >= cfg.min_valid_days).astype(np.float32)
__all__ = ['BayesHighVolOnsetConfig', 'BayesHighVolOnsetContext', 'BayesHighVolOnsetFactor']
