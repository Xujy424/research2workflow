import polars as pl
import numpy as np
from pathlib import Path
import bisect
import pandas as pd


ROOT = Path("/data/xujiayi/xjy/l2")


def get_tables_per_exchange(date, exchange):
    order = pl.read_parquet(ROOT/"proc"/date/f"{exchange}wt.pq").filter((pl.col('TransactTime')>pl.time(9,30)) & (pl.col('TransactTime')<pl.time(14,57)))
    cancel = pl.read_parquet(ROOT/"proc"/date/f"{exchange}cancel.pq").filter((pl.col('TransactTime')>pl.time(9,30)) & (pl.col('TransactTime')<pl.time(14,57)))
    deal = pl.read_parquet(ROOT/"proc"/date/f"{exchange}cj.pq").filter((pl.col('TransactTime')>pl.time(9,30)) & (pl.col('TransactTime')<pl.time(14,57)))
    shot = pl.read_parquet(ROOT/"proc"/date/f"{exchange}shot_1m.pq").filter((pl.col('BarTime')>=pl.time(9,30)) & (pl.col('BarTime')<=pl.time(14,57)))
    return order, deal, cancel, shot

def get_tables(date):
    key = ['SecurityID','ChannelNo','TransactTime','ApplSeqNum']
    orders, deals, cancels, shots = [],[],[],[]
    for ex in ['sh','sz']:
        order, deal, cancel, shot = get_tables_per_exchange(date, ex)
        orders.append(order.select(['SecurityID','ChannelNo','TransactTime','ApplSeqNum','Side','Price','OrderQty']).with_columns(pl.col("Side").cast(pl.Int8)))
        deals.append(deal.select(['SecurityID','ChannelNo','TransactTime','ApplSeqNum','BidApplSeqNum','OfferApplSeqNum','Side','Price','OrderQty']))
        cancels.append(cancel.select(['SecurityID','ChannelNo','TransactTime','ApplSeqNum','BidApplSeqNum','OfferApplSeqNum','Side','Price','OrderQty']))
        shots.append(shot)
    order = pl.concat(orders).sort(key)
    deal = pl.concat(deals).sort(key)
    cancel = pl.concat(cancels).sort(key)
    shot = pl.concat(shots).sort(['SecurityID','BarTime'])
    return order, deal, cancel, shot

def calc_cluster(df,shot,limit,n):
    qty_bin = df.sort(["SecurityID",'TransactTime','ApplSeqNum']).join_asof(
        shot.select(["BarTime","SecurityID"]),
        left_on="TransactTime",
        right_on="BarTime",
        by="SecurityID",
        strategy="backward",
        check_sortedness=False,
    )
    # 1. 按股票、时间、序号排序，确保移位顺序正确
    minute_qty_bin = qty_bin.group_by(['SecurityID', 'BarTime']).agg(
        pl.col('OrderQty').sum().alias('MinuteQty')
    ).sort(['SecurityID', 'BarTime'])

    # 2. 计算前后分钟均值（固定分母 n）
    surrounding_sum = 0
    for i in range(1, n//2 + 1):
        surrounding_sum += (
            pl.col('MinuteQty').shift(i).over('SecurityID').fill_null(0) +
            pl.col('MinuteQty').shift(-i).over('SecurityID').fill_null(0)
        )
    surround_mean = surrounding_sum / n

    minute_df = minute_qty_bin.with_columns([
        surround_mean.alias('surround_mean'),
        (pl.col('MinuteQty') / surround_mean).alias('ratio'),
        ((pl.col('MinuteQty') / surround_mean) > limit).cast(pl.Int32).alias('Custer')
    ])

    # 3. 回填到原始逐笔数据
    res = qty_bin.join(
        minute_df, on=['SecurityID', 'BarTime'], how='left'
    )
    return res.drop(['BarTime','MinuteQty','surround_mean','ratio'])

def calc_aggress(order, shot):
    buy_order = order.filter(pl.col('Side')==1)
    buy_order = buy_order.join_asof(
        shot,
        left_on="TransactTime",
        right_on="BarTime",
        by="SecurityID",
        strategy="backward",
        check_sortedness=False,
    ).with_columns(
        pl.when((pl.col('Price') >= pl.col('AskPrice1')) & (pl.col('OrderQty') >= pl.col('AskQty1'))).then(1)
        .when((pl.col('Price') >= pl.col('AskPrice1')) & (pl.col('OrderQty') < pl.col('AskQty1'))).then(2)
        .when((pl.col('Price') > pl.col('BidPrice1')) & (pl.col('Price') < pl.col('AskPrice1'))).then(3)
        .when(pl.col('Price') == pl.col('BidPrice1')).then(4)
        .otherwise(5)
        .alias('PassionLevel')
    ).select(order.columns+['PassionLevel'])

    sell_order = order.filter(pl.col('Side')==-1)
    sell_order = sell_order.join_asof(
        shot,
        left_on="TransactTime",
        right_on="BarTime",
        by="SecurityID",
        strategy="backward",
        check_sortedness=False,
    ).with_columns(
        pl.when((pl.col('Price') <= pl.col('BidPrice1')) & (pl.col('OrderQty') >= pl.col('BidQty1'))).then(1)
        .when((pl.col('Price') <= pl.col('BidPrice1')) & (pl.col('OrderQty') < pl.col('BidQty1'))).then(2)
        .when((pl.col('Price') < pl.col('AskPrice1')) & (pl.col('Price') > pl.col('BidPrice1'))).then(3)
        .when(pl.col('Price') == pl.col('AskPrice1')).then(4)
        .otherwise(5)
        .alias('PassionLevel')
    ).select(order.columns+['PassionLevel'])

    return pl.concat([buy_order,sell_order]).sort('ChannelNo','SecurityID','TransactTime','ApplSeqNum','Side')

def calc_activedeal(deal):
    deal = deal.with_columns(
        pl.when((0<=pl.col('OrderQty')) & (pl.col('OrderQty')<50000)).then(1)
        .when((50000<=pl.col('OrderQty')) & (pl.col('OrderQty')<300000)).then(2)
        .when((300000<=pl.col('OrderQty')) & (pl.col('OrderQty')<1000000)).then(3)
        .otherwise(4)
        .alias('Size')
    )
    active_buy = deal.filter(pl.col('BidApplSeqNum')>pl.col('OfferApplSeqNum'))
    active_sell = deal.filter(pl.col('BidApplSeqNum')<pl.col('OfferApplSeqNum'))
    return active_buy, active_sell

