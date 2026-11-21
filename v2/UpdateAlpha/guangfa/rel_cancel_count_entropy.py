"""Guangfa 2026-08-07, table 2 (p8), negative-direction factor.

Implementation convention: unnormalized Shannon entropy of minute cancellation counts.
Continuous auction: [09:30,11:30), [13:00,14:57).
Report table 2 provides names and IC, not complete formulas. These choices
are provisional and must not be represented as exact report replication.
Missing exchange input raises; no observed cancellations yields NaN.
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
class RelCancelCountEntropyConfig:
    smoothing_days: int = 1

    def __post_init__(self):
        if not isinstance(self.smoothing_days, int) or self.smoothing_days < 1:
            raise ValueError('smoothing_days must be a positive integer')

class RelCancelCountEntropyContext(AlphaContext):
    def __init__(self, root=Path('Z:/') if Path('Z:/axis/dates.npy').is_file() else ROOT,
                 config=RelCancelCountEntropyConfig(), l2_root=None, universe='self'):
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
        minute = frame.group_by('tick','minute').agg(pl.len().cast(pl.Float64).alias('weight'))
        minute = minute.with_columns((pl.col('weight') / pl.col('weight').sum().over('tick')).alias('p'))
        return minute.group_by('tick').agg((-(pl.col('p') * pl.col('p').log()).sum()).alias('value'))

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

class RelCancelCountEntropyFactor(AlphaBase):
    meta = AlphaMeta('gf_rel_cancel_count_entropy', 'rel_cancel_count_entropy (explicit cancellation-event convention)', direction=-1)
    dependencies = ('l2/proc/shcancel.pq','l2/proc/szcancel.pq')

    def calculate(self, asof):
        end = self.context.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - self.context.config.smoothing_days + 1
        if start < 0:
            return np.full(self.context.data.axis.tick_count, np.nan, dtype=np.float32)
        values = [self.context.daily(day) for day in self.context.data.axis.trade_dates[start:end+1]]
        return np.mean(values, axis=0).astype(np.float32)
