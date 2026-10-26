"""sr_dispersion_mean_vwap: Mean dispersion-minute VWAP divided by daily mean VWAP.

Xingye high-frequency series 10, October 2026.
Independent Config / Context / Factor; only generic math uses operators.
Default: 10-day mean through d-1; preserve raw sign in storage.
241 stored bars: 09:30 plus 240 minute endpoints; use close-to-close returns
at 09:31-11:30 and 13:01-15:00. Population standard deviations are used.
Market minute return is the equal-weight mean of tradable stocks.
Sparse support defaults to 24 minutes (configurable; unspecified in report).
See volume_dispersion.md for the backtest entry and time alignment.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
from v2.GetData import DataPool
from v2.UpdateData.config import ROOT

@dataclass(frozen=True)
class SRDispersionMeanVWAPConfig:
    lookback_days: int = 10
    lag_days: int = 1
    min_valid_days: int = 10
    min_minutes: int = 120
    sparse_minutes: int = 24

    def __post_init__(self):
        if self.lookback_days < 1 or self.lag_days < 0:
            raise ValueError('lookback_days must be positive and lag_days nonnegative')
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError('invalid min_valid_days')
        if not 5 <= self.min_minutes <= 240:
            raise ValueError('min_minutes must be in [5, 240]')
        if not 0 <= self.sparse_minutes < self.min_minutes - 1:
            raise ValueError('sparse_minutes must leave at least two ordinary minutes')

def sparse_gamma(y, market, sparsity):
    """Global l0 fit of y = beta*abs(market) + diag(market**2)*gamma.

    A selected nonzero-market row can be fit exactly. Thus minimize the sum
    of the T-s smallest squared residuals over scalar beta (trimmed LS).
    Residual ordering changes only at pairwise intersections of squared
    residual lines. Enumerate those intervals, solve their quadratic minima,
    and choose the global minimum. No l1 surrogate or local sparse solver.
    """
    x = np.abs(np.asarray(market, float))
    y = np.asarray(y, float)
    gamma = np.zeros_like(y)
    eligible = x > 0
    a, b = (x[eligible], y[eligible])
    s = min(sparsity, max(0, len(a) - 1))
    if not len(a):
        return gamma
    keep = len(a) - s
    if s == 0:
        return gamma
    scale = a.max()
    a = a / scale
    lo, hi = (np.min(b / a), np.max(b / a))
    i, j = np.triu_indices(len(a), 1)
    roots = []
    for numerator, denominator in ((b[i] - b[j], a[i] - a[j]), (b[i] + b[j], a[i] + a[j])):
        nonzero = denominator != 0
        values = numerator[nonzero] / denominator[nonzero]
        roots.append(values[(values > lo) & (values < hi)])
    edges = np.unique(np.r_[lo, *roots, hi])
    if len(edges) == 1:
        beta = edges[0]
    else:
        best_loss, beta = (np.inf, lo)
        for start in range(0, len(edges) - 1, 256):
            stop = min(start + 256, len(edges) - 1)
            left, right = (edges[start:stop], edges[start + 1:stop + 1])
            mid = left + (right - left) / 2
            squared = (b[None, :] - mid[:, None] * a) ** 2
            selected = np.argpartition(squared, keep - 1, axis=1)[:, :keep]
            ax, by = (a[selected], b[selected])
            candidate = np.clip((ax * by).sum(1) / (ax * ax).sum(1), left, right)
            error = (b[None, :] - candidate[:, None] * a) ** 2
            loss = np.partition(error, keep - 1, axis=1)[:, :keep].sum(1)
            k = np.argmin(loss)
            if loss[k] < best_loss:
                best_loss, beta = (loss[k], candidate[k])
    residual = b - beta * a
    selected = np.argsort(residual ** 2, kind='stable')[-s:]
    positions = np.flatnonzero(eligible)[selected]
    gamma[positions] = residual[selected] / x[positions] ** 2
    gamma[positions[np.abs(residual[selected]) <= 1e-12 * max(1, np.max(np.abs(y)))]] = 0
    return gamma

class SRDispersionMeanVWAPContext(AlphaContext):
    """Read and cache daily features for sr_dispersion_mean_vwap."""

    def __init__(self, root=ROOT, config=SRDispersionMeanVWAPConfig(), universe='self'):
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
        output = {'sr_dispersion_mean_vwap': np.full(n, np.nan)}
        amount = self._read('m_essentials/amount', day)[offset:]
        for stock in np.flatnonzero(tradable):
            r, v = (ret[:, stock], vol[:, stock])
            good = np.isfinite(r) & np.isfinite(market)
            if good.sum() < cfg.min_minutes:
                continue
            gamma = np.full(240, np.nan)
            gamma[good] = sparse_gamma(np.abs(r[good] - market[good]), market[good], cfg.sparse_minutes)
            dispersion = gamma > 0
            vw = np.divide(amount[:, stock], v, out=np.full(240, np.nan), where=(v > 0) & (amount[:, stock] > 0))
            valid_vwap = good & np.isfinite(vw)
            selected = dispersion & valid_vwap
            if selected.any():
                output['sr_dispersion_mean_vwap'][stock] = vw[selected].mean() / vw[valid_vwap].mean()
        self._daily_cache[key] = output
        oldest = day - cfg.lookback_days - cfg.lag_days
        for item in list(self._daily_cache):
            if item < oldest:
                del self._daily_cache[item]
        return output

class SRDispersionMeanVWAPFactor(AlphaBase):
    meta = AlphaMeta('sr_dispersion_mean_vwap', 'Mean dispersion-minute VWAP divided by daily mean VWAP', direction=-1)
    dependencies = ('m_essentials/close', 'm_essentials/volume', 'm_essentials/amount', 'basic/tradable')

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
__all__ = ['SRDispersionMeanVWAPConfig', 'SRDispersionMeanVWAPContext', 'SRDispersionMeanVWAPFactor']
