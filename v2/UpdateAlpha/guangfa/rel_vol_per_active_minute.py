"""Guangfa 2026-08-07, table 2 (p8), negative-direction factor.

Explicit reproduction convention (report does not provide full formula):
active minutes = distinct continuous-auction minutes with positive cancellations;
count = cancellation event rows, volume = cancelled shares. No price filter:
Shenzhen cancellation prices may be zero. Missing exchange files raise.
No cancellation observations yield NaN, not an invented zero intensity.
This convention requires confirmation before claiming exact report replication.
"""
from dataclasses import dataclass
from datetime import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
from ...GetData import DataPool
from ...UpdateData.config import ROOT

@dataclass(frozen=True)
class RelVolPerActiveMinuteConfig:
    smoothing_days: int = 1

    def __post_init__(self):
        if not isinstance(self.smoothing_days, int) or self.smoothing_days < 1:
            raise ValueError('smoothing_days must be a positive integer')

class RelVolPerActiveMinuteContext(AlphaContext):
    def __init__(self, root=Path('Z:/') if Path('Z:/axis/dates.npy').is_file() else ROOT,
                 config=RelVolPerActiveMinuteConfig(), l2_root=None, universe='self'):
        self.config = config
        self.l2_root = Path(l2_root) if l2_root else Path(root) / 'stock/l2'
        super().__init__(DataPool(root, asset='stock'), universe=universe)

    @staticmethod
    def aggregate(frame):
        frame = frame.filter(
            (pl.col('OrderQty') > 0)
            & (pl.col('TransactTime').is_between(time(9,30),time(11,30),closed='left')
               | pl.col('TransactTime').is_between(time(13),time(14,57),closed='left'))
        ).with_columns(
            pl.col('SecurityID').cast(pl.String).str.pad_start(6,'0').alias('tick'),
            (pl.col('TransactTime').cast(pl.Int64) // 60_000_000_000).alias('minute'),
        )
        return frame.group_by('tick').agg(
            pl.col('OrderQty').cast(pl.Float64).sum().alias('numerator'),
            pl.col('minute').n_unique().alias('active'),
        ).with_columns((pl.col('numerator') / pl.col('active')).alias('value')).select('tick','value')

    def daily(self, date):
        folder = self.l2_root / 'proc' / pd.Timestamp(date).strftime('%Y%m%d')
        scans = []
        for name in ('shcancel.pq','szcancel.pq'):
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            scans.append(pl.scan_parquet(path).select('SecurityID','TransactTime','OrderQty'))
        result = self.aggregate(pl.concat(scans, how='vertical_relaxed')).collect()
        return self.align(result)

class RelVolPerActiveMinuteFactor(AlphaBase):
    meta = AlphaMeta('gf_rel_vol_per_active_minute', 'rel_vol_per_active_minute (explicit cancellation-event convention)', direction=-1)
    dependencies = ('l2/proc/shcancel.pq','l2/proc/szcancel.pq')

    def calculate(self, asof):
        end = self.context.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - self.context.config.smoothing_days + 1
        if start < 0:
            return np.full(self.context.data.axis.tick_count, np.nan, dtype=np.float32)
        values = [self.context.daily(day) for day in self.context.data.axis.trade_dates[start:end+1]]
        return np.mean(values, axis=0).astype(np.float32)
