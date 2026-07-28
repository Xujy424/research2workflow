import polars as pl
import numpy as np
from pathlib import Path
import bisect
import pandas as pd

from utils import *


def core(order):
    buy_order, sell_order = order.filter(pl.col('Side')==1), order.filter(pl.col('Side')==-1)

    buy_passionlevel_count = buy_order.group_by(['SecurityID','PassionLevel']).agg(pl.col('PassionLevel').count().alias('LevelCount')).sort(['SecurityID','PassionLevel'])
    buy_passionlevel_count = buy_passionlevel_count.with_columns(
        (pl.col('LevelCount')/pl.col('LevelCount').sum().over('SecurityID')).alias('LevelProportion')
    )

    sell_passionlevel_count = sell_order.group_by(['SecurityID','PassionLevel']).agg(pl.col('PassionLevel').count().alias('LevelCount')).sort(['SecurityID','PassionLevel'])
    sell_passionlevel_count = sell_passionlevel_count.with_columns(
        (pl.col('LevelCount')/pl.col('LevelCount').sum().over('SecurityID')).alias('LevelProportion')
    )

    all_securities = buy_passionlevel_count.select('SecurityID').unique().vstack(
        sell_passionlevel_count.select('SecurityID').unique()
    ).unique()
    all_levels = pl.DataFrame({'PassionLevel': [1, 2, 3, 4, 5]})
    all_combinations = all_securities.join(all_levels, how='cross')

    passionlevel_count = (
        all_combinations
        .join(
            buy_passionlevel_count,
            on=['SecurityID', 'PassionLevel'],
            how='left'
        ).with_columns([
            pl.col('LevelCount').fill_null(0).alias('LevelCount_buy'),
            pl.col('LevelProportion').fill_null(0).alias('LevelProportion_buy')
        ]).drop(['LevelCount', 'LevelProportion'])
        .join(
            sell_passionlevel_count,
            on=['SecurityID', 'PassionLevel'],
            how='left'
        ).with_columns([
            pl.col('LevelCount').fill_null(0).alias('LevelCount_sell'),
            pl.col('LevelProportion').fill_null(0).alias('LevelProportion_sell')
        ]).drop(['LevelCount', 'LevelProportion'])
    ).sort(['SecurityID','PassionLevel'])

    factor_tilt = passionlevel_count.with_columns(
        pl.when(pl.col('PassionLevel')==1).then(5)
        .when(pl.col('PassionLevel')==2).then(4)
        .when(pl.col('PassionLevel')==3).then(3)
        .when(pl.col('PassionLevel')==4).then(2)
        .when(pl.col('PassionLevel')==5).then(1)
        .otherwise(0)
        .alias('LevelScore')
    ).with_columns(
        pl.col('LevelScore')*(pl.col('LevelProportion_buy')-pl.col('LevelProportion_sell')).alias('factor_tilt')
    ).group_by('SecurityID').agg(pl.col('LevelScore').sum())

    factor_tilt = factor_tilt.to_pandas().set_index('SecurityID')
    factor_tilt.index = factor_tilt.index.astype(str).str.zfill(6)

    return factor_tilt




if __name__ == '__main__':
