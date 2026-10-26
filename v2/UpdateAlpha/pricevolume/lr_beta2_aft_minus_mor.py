"""lr_beta2_aft_minus_mor: CSAD quadratic coefficient after 10:00 minus morning.

Xingye high-frequency series 10, October 2026.
Independent Config / Context / Factor; only generic math uses operators.
Default: 10-day mean through d-1; preserve raw sign in storage.
241 stored bars: 09:30 plus 240 minute endpoints; use close-to-close returns
at 09:31-11:30 and 13:01-15:00. Population standard deviations are used.
Market minute return is the equal-weight mean of tradable stocks.
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
class LRBeta2AftMinusMorConfig:
    lookback_days: int = 10
    lag_days: int = 1
    min_valid_days: int = 10
    min_minutes: int = 120
    min_morning_minutes: int = 20

    def __post_init__(self):
        if self.lookback_days < 1 or self.lag_days < 0:
            raise ValueError('lookback_days must be positive and lag_days nonnegative')
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError('invalid min_valid_days')
        if not 5 <= self.min_minutes <= 240:
            raise ValueError('min_minutes must be in [5, 240]')
        if not 3 <= self.min_morning_minutes <= 30:
            raise ValueError('min_morning_minutes must be in [3, 30]')

class LRBeta2AftMinusMorContext(AlphaContext):
    """Read and cache daily features for lr_beta2_aft_minus_mor."""

    def __init__(self, root=ROOT, config=LRBeta2AftMinusMorConfig(), universe='self'):
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
        output = {'lr_beta2_aft_minus_mor': np.full(n, np.nan)}
        for stock in np.flatnonzero(tradable):
            r, v = (ret[:, stock], vol[:, stock])
            good = np.isfinite(r) & np.isfinite(market)
            if good.sum() < cfg.min_minutes:
                continue
            betas, score = ([], np.nan)
            for segment, threshold in ((np.arange(240) < 30, cfg.min_morning_minutes), (np.arange(240) >= 30, cfg.min_minutes - 30)):
                mask = good & segment
                if mask.sum() < max(3, threshold):
                    betas.append(np.nan)
                    continue
                x = np.abs(market[mask])
                y = np.abs(r[mask] - market[mask])
                scale = x.max()
                if scale <= 0:
                    betas.append(np.nan)
                    continue
                design = np.column_stack((np.ones(len(x)), x / scale, (x / scale) ** 2))
                coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                if rank < 3:
                    betas.append(np.nan)
                    continue
                betas.append(coef[2] / scale ** 2)
            output['lr_beta2_aft_minus_mor'][stock] = betas[1] - betas[0]
            continue
        self._daily_cache[key] = output
        oldest = day - cfg.lookback_days - cfg.lag_days
        for item in list(self._daily_cache):
            if item < oldest:
                del self._daily_cache[item]
        return output

class LRBeta2AftMinusMorFactor(AlphaBase):
    meta = AlphaMeta('lr_beta2_aft_minus_mor', 'CSAD quadratic coefficient after 10:00 minus morning', direction=1)
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
__all__ = ['LRBeta2AftMinusMorConfig', 'LRBeta2AftMinusMorContext', 'LRBeta2AftMinusMorFactor']
