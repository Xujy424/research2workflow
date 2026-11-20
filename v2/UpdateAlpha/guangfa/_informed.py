"""2026-05-11 p8: daily absolute active-side imbalance.

This is the report's daily L2 measure, not the 2022 volume-bucket VPIN.
Normalization uses classified buy plus sell trades; missing exchange input
raises rather than silently treating missing trades as zero.
"""
from dataclasses import dataclass
from datetime import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from ..alphabase import AlphaContext
from ...GetData import DataPool
from ...UpdateData.config import ROOT

@dataclass(frozen=True)
class InformedConfig:
    smoothing_days: int = 1

    def __post_init__(self):
        if self.smoothing_days < 1:
            raise ValueError('smoothing_days must be positive')

class InformedContext(AlphaContext):
    def __init__(self, root=Path('Z:/') if Path('Z:/axis/dates.npy').is_file() else ROOT,
                 config=InformedConfig(), l2_root=None, universe='self'):
        self.config = config
        self.l2_root = Path(l2_root) if l2_root else Path(root) / 'stock/l2'
        super().__init__(DataPool(root, asset='stock'), universe=universe)

    def daily(self, date, measure):
        folder = self.l2_root / 'proc' / pd.Timestamp(date).strftime('%Y%m%d')
        paths = [folder / name for name in ('shcj.pq', 'szcj.pq')]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        trades = pl.concat([pl.scan_parquet(p) for p in paths], how='diagonal_relaxed')
        trades = trades.filter(
            (pl.col('Price') > 0) & (pl.col('OrderQty') > 0)
            & pl.col('Side').is_in([-1, 1])
            & (pl.col('TransactTime').is_between(time(9,30),time(11,30),closed='left')
               | pl.col('TransactTime').is_between(time(13),time(14,57),closed='left'))
        ).with_columns(pl.col('SecurityID').cast(pl.String).str.pad_start(6,'0').alias('tick'))
        value = {'vol': pl.col('OrderQty').cast(pl.Float64),
                 'amt': pl.col('Price').cast(pl.Float64) * pl.col('OrderQty'),
                 'cnt': pl.lit(1.0)}[measure]
        grouped = trades.with_columns(value.alias('weight')).group_by('tick').agg(
            (pl.col('weight') * pl.col('Side')).sum().abs().alias('imbalance'),
            pl.col('weight').sum().alias('total'),
        ).with_columns(pl.when(pl.col('total') > 0)
                       .then(pl.col('imbalance') / pl.col('total'))
                       .otherwise(None).alias('value')).select('tick','value').collect()
        return self.align(grouped)

    def value(self, asof, measure):
        end = self.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.smoothing_days + 1
        if start < 0:
            return np.full(self.data.axis.tick_count, np.nan, dtype=np.float32)
        values = np.stack([self.daily(day, measure)
                           for day in self.data.axis.trade_dates[start:end+1]])
        # Require all smoothing days: no implicit filling or changing window.
        return values.mean(axis=0).astype(np.float32)
