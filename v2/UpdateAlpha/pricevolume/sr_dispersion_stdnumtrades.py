"""sr_dispersion_stdnumtrades: Dispersion-minute trade-count std divided by daily mean.

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
import polars as pl
from datetime import time
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
from v2.GetData import DataPool
from v2.UpdateData.config import ROOT

@dataclass(frozen=True)
class SRDispersionStdnumtradesConfig:
    lookback_days: int = 10
    lag_days: int = 1
    min_valid_days: int = 10
    min_minutes: int = 120
    sparse_minutes: int = 24
    numtrades_field: str | None = None

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

def l2_minute_trade_counts(l2_root, date, ticks):
    """Count executions, using the same proc parquet schema as SATD.

    Floor timestamps into minute intervals and align to minute bar endpoints.
    11:30/15:00 boundary executions belong to the last minute of that session.
    Includes closing-auction executions; excludes pre-open and lunch. Stocks
    absent from a file remain NaN; zero-trade minutes of observed stocks are 0.
    One unavailable exchange cannot turn into fictitious zero-count stocks.
    """
    folder = Path(l2_root) / 'proc' / pd.Timestamp(date).strftime('%Y%m%d')
    result = np.full((240, len(ticks)), np.nan)
    positions = {str(tick).strip().zfill(6): i for i, tick in enumerate(ticks)}
    paths = [folder / f'{exchange}cj.pq' for exchange in ('sh', 'sz')]
    if not any((path.is_file() for path in paths)):
        raise FileNotFoundError(f'missing L2 executions: {folder}/{{sh,sz}}cj.pq')
    for path in paths:
        if not path.is_file():
            continue
        clock = pl.col('TransactTime')
        minute = clock.cast(pl.Int64) // 60000000000
        frame = pl.scan_parquet(path).select('SecurityID', 'TransactTime', 'Price', 'OrderQty').filter((pl.col('Price') > 0) & (pl.col('OrderQty') > 0) & (clock.is_between(time(9, 30), time(11, 30), closed='both') | clock.is_between(time(13), time(15), closed='both'))).with_columns(pl.col('SecurityID').cast(pl.String).str.pad_start(6, '0').alias('tick'), pl.when(minute <= 690).then((minute - 570).clip(0, 119)).otherwise((minute - 780 + 120).clip(120, 239)).alias('minute')).group_by('tick', 'minute').agg(pl.len().alias('count')).collect(engine='streaming')
        observed = {positions[tick] for tick in frame['tick'].unique() if tick in positions}
        if observed:
            result[:, list(observed)] = 0
        for tick, minute_index, count in frame.iter_rows():
            stock = positions.get(tick)
            if stock is not None:
                result[minute_index, stock] = count
    return result

class SRDispersionStdnumtradesContext(AlphaContext):
    """Read and cache daily features for sr_dispersion_stdnumtrades."""

    def __init__(self, root=ROOT, config=SRDispersionStdnumtradesConfig(), universe='self', l2_root=None):
        self.config = config
        self.l2_root = Path(l2_root) if l2_root is not None else Path(root) / 'stock' / 'l2'
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
        output = {'sr_dispersion_stdnumtrades': np.full(n, np.nan)}
        trades = None
        if cfg.numtrades_field is not None:
            trades = self._read(cfg.numtrades_field, day)[offset:]
        else:
            trades = l2_minute_trade_counts(self.l2_root, self.data.axis.trade_dates[day], self.data.axis.ticks)
        if trades.shape != ret.shape:
            raise ValueError('numtrades field must match the minute price axes')
        for stock in np.flatnonzero(tradable):
            r, v = (ret[:, stock], vol[:, stock])
            good = np.isfinite(r) & np.isfinite(market)
            if good.sum() < cfg.min_minutes:
                continue
            gamma = np.full(240, np.nan)
            gamma[good] = sparse_gamma(np.abs(r[good] - market[good]), market[good], cfg.sparse_minutes)
            dispersion = gamma > 0
            all_trades = trades[:, stock]
            valid_trades = good & np.isfinite(all_trades) & (all_trades >= 0)
            selected = all_trades[dispersion & valid_trades]
            mean = all_trades[valid_trades].mean() if valid_trades.any() else np.nan
            if len(selected) >= 2 and mean > 0:
                output['sr_dispersion_stdnumtrades'][stock] = np.std(selected) / mean
        self._daily_cache[key] = output
        oldest = day - cfg.lookback_days - cfg.lag_days
        for item in list(self._daily_cache):
            if item < oldest:
                del self._daily_cache[item]
        return output

class SRDispersionStdnumtradesFactor(AlphaBase):
    meta = AlphaMeta('sr_dispersion_stdnumtrades', 'Dispersion-minute trade-count std divided by daily mean', direction=-1)
    dependencies = ('m_essentials/close', 'm_essentials/volume', 'basic/tradable', 'l2/proc/shcj.pq', 'l2/proc/szcj.pq')

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
__all__ = ['SRDispersionStdnumtradesConfig', 'SRDispersionStdnumtradesContext', 'SRDispersionStdnumtradesFactor']
