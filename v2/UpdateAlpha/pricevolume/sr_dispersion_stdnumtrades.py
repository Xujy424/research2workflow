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
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl
from datetime import time
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.operators import least_trimmed_squares_slope
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

def _stock_feature(task):
    stock, r, v, market, cfg, extra = task
    value = np.nan
    good = np.isfinite(r) & np.isfinite(market)
    if good.sum() < cfg.min_minutes:
        return (stock, value)
    gamma = np.full(240, np.nan)
    gamma[good] = sparse_gamma(np.abs(r[good] - market[good]), market[good], cfg.sparse_minutes)
    dispersion = gamma > 0
    all_trades = extra
    valid_trades = good & np.isfinite(all_trades) & (all_trades >= 0)
    selected = all_trades[dispersion & valid_trades]
    mean = all_trades[valid_trades].mean() if valid_trades.any() else np.nan
    if len(selected) >= 2 and mean > 0:
        value = np.std(selected) / mean
    return (stock, value)

class SRDispersionStdnumtradesContext(AlphaContext):
    """Read and cache daily features for sr_dispersion_stdnumtrades."""

    def __init__(self, root=ROOT, config=SRDispersionStdnumtradesConfig(), universe='self', l2_root=None):
        self._executor = None
        self.config = config
        self.l2_root = Path(l2_root) if l2_root is not None else Path(root) / 'stock' / 'l2'
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
            tqdm.write(f'[sr_dispersion_stdnumtrades] reading source {source_date}', file=sys.stdout)
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
        output = {'sr_dispersion_stdnumtrades': np.full(n, np.nan)}
        trades = None
        if cfg.numtrades_field is not None:
            trades = self._read(cfg.numtrades_field, day)[offset:]
        else:
            trades = l2_minute_trade_counts(self.l2_root, self.data.axis.trade_dates[day], self.data.axis.ticks)
        if trades.shape != ret.shape:
            raise ValueError('numtrades field must match the minute price axes')
        stocks = np.flatnonzero(tradable)
        workers = cfg.workers or (min(8, os.cpu_count() or 1) if len(stocks) >= 256 else 1)
        tasks = ((stock, ret[:, stock], vol[:, stock], market, cfg, trades[:, stock]) for stock in stocks)
        if workers > 1:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=workers)
            results = self._executor.map(_stock_feature, tasks, chunksize=32)
        else:
            results = map(_stock_feature, tasks)
        for stock, value in tqdm(results, total=len(stocks), desc=f'sr_dispersion_stdnumtrades day {day} stocks', file=sys.stdout, leave=False, mininterval=1, disable=not cfg.show_progress or not sys.stdout.isatty(), bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]', position=1):
            output['sr_dispersion_stdnumtrades'][stock] = value
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

class SRDispersionStdnumtradesFactor(AlphaBase):
    meta = AlphaMeta('sr_dispersion_stdnumtrades', 'Dispersion-minute trade-count std divided by daily mean', direction=-1)
    dependencies = ('m_essentials/close', 'm_essentials/volume', 'basic/tradable', 'l2/proc/shcj.pq', 'l2/proc/szcj.pq')

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
                tqdm.write(f'[sr_dispersion_stdnumtrades] {phase}: {missing} new source day(s), {cfg.lookback_days - missing} cached', file=sys.stdout)
                sys.stdout.flush()
        rows = np.stack([self.context.daily_features(day)[self.meta.name] for day in range(start, end + 1)])
        valid = np.isfinite(rows)
        count = valid.sum(0)
        return np.divide(np.where(valid, rows, 0).sum(0), count, out=np.full(axis.tick_count, np.nan), where=count >= cfg.min_valid_days).astype(np.float32)
__all__ = ['SRDispersionStdnumtradesConfig', 'SRDispersionStdnumtradesContext', 'SRDispersionStdnumtradesFactor']
