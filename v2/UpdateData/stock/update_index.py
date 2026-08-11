import numpy as np
import polars as pl


sql = f'''
select 
    B.SecuCode as 'index',
    C.SecuCode as 'tick',
    A.EndDate as 'date',
    A.Weight as 'weight'
from LC_IndexComponentsWeight A
left join SecuMain B on A.IndexCode=B.InnerCode
left join SecuMain C on A.InnerCode=C.InnerCode
where B.SecuCode in ('000300','000905','000510','000852','932000','000985')
'''
index_compt = pl.read_database(sql, JY_CONN)
index_compt

index_compt = index_compt.sort(['index','date','tick'])

hs300 = index_compt.filter(pl.col('index')=='000300').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(hs300.index))))
hs300 = hs300.reindex(index=dates)
hs300 = hs300.ffill().reindex(dts).replace(-999,np.nan)
hs300

zz500 = index_compt.filter(pl.col('index')=='000905').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(zz500.index))))
zz500 = zz500.reindex(index=dates)
zz500 = zz500.ffill().reindex(dts).replace(-999,np.nan)
zz500

a500 = index_compt.filter(pl.col('index')=='000510').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(a500.index))))
a500 = a500.reindex(index=dates)
a500 = a500.ffill().reindex(dts).replace(-999,np.nan)
a500

zz1000 = index_compt.filter(pl.col('index')=='000852').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(zz1000.index))))
zz1000 = zz1000.reindex(index=dates)
zz1000 = zz1000.ffill().reindex(dts).replace(-999,np.nan)
zz1000

zz2000 = index_compt.filter(pl.col('index')=='932000').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(zz2000.index))))
zz2000 = zz2000.reindex(index=dates)
zz2000 = zz2000.ffill().reindex(dts).replace(-999,np.nan)
zz2000

zzfull = index_compt.filter(pl.col('index')=='000985').pivot(index='date',columns='tick',values='weight').to_pandas().set_index('date').reindex(columns=ticks).fillna(-999)
dates = sorted(list(set(dts).union(set(zzfull.index))))
zzfull = zzfull.reindex(index=dates)
zzfull = zzfull.ffill().reindex(dts).replace(-999,np.nan)
zzfull



hs300.values.astype(float).tofile('/data/xujiayi/end2end/mask/hs300_weight.bin')
zz500.values.astype(float).tofile('/data/xujiayi/end2end/mask/zz500_weight.bin')
a500.values.astype(float).tofile('/data/xujiayi/end2end/mask/a500_weight.bin')
zz1000.values.astype(float).tofile('/data/xujiayi/end2end/mask/zz1000_weight.bin')
zz2000.values.astype(float).tofile('/data/xujiayi/end2end/mask/zz2000_weight.bin')
zzfull.values.astype(float).tofile('/data/xujiayi/end2end/mask/zzfull_weight.bin')




hs300 = hs300.values.astype(float)
zz500 = zz500.values.astype(float)
a500 = a500.values.astype(float)
zz1000 = zz1000.values.astype(float)
zz2000 = zz2000.values.astype(float)
zzfull = zzfull.values.astype(float)




hs300_mask = ~np.isnan(hs300)
hs300_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/hs300_mask.bin')
zz500_mask = ~np.isnan(zz500)
zz500_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/zz500_mask.bin')
a500_mask = ~np.isnan(a500)
a500_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/a500_mask.bin')
zz1000_mask = ~np.isnan(zz1000)
zz1000_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/zz1000_mask.bin')
zz2000_mask = ~np.isnan(zz2000)
zz2000_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/zz2000_mask.bin')
zzfull_mask = ~np.isnan(zzfull)
zzfull_mask.astype(bool).tofile('/data/xujiayi/end2end/mask/zzfull_mask.bin')


INDEX_CODE_TO_NAME = {
    "000300": "hs300",
    "000905": "zz500",
    "000510": "a500",
    "000852": "zz1000",
    "932000": "zz2000",
    "000985": "zzfull",
}


def _read_latest_index_snapshot(date, conn):
    """读取截至 date 每个指数最近一个完整成分权重快照。"""
    sql_latest = f"""
    with ranked as (
        select
            B.SecuCode as index_code,
            C.SecuCode as tick,
            A.EndDate as snapshot_date,
            A.Weight as weight,
            dense_rank() over (
                partition by B.SecuCode
                order by A.EndDate desc
            ) as date_rank
        from LC_IndexComponentsWeight A
        left join SecuMain B on A.IndexCode = B.InnerCode
        left join SecuMain C on A.InnerCode = C.InnerCode
        where B.SecuCode in ('000300','000905','000510','000852','932000','000985')
          and A.EndDate <= '{date}'
    )
    select index_code, tick, snapshot_date, weight
    from ranked
    where date_rank = 1
    """
    return pl.read_database(sql_latest, conn)


def _write_index_row(date, dates, ticks, root, name, weights):
    """写入一个指数当天的权重和成分 mask。"""
    dt = np.searchsorted(dates, date)
    n_valid = np.count_nonzero(ticks != "")

    weight_arr = np.memmap(
        root / "mask" / f"{name}_weight.bin",
        dtype=float,
        mode="r+",
        shape=(len(dates), len(ticks)),
    )
    weight_arr[dt] = np.nan
    weight_arr[dt,:n_valid] = weights
    weight_arr.flush()

    mask_arr = np.memmap(
        root / "mask" / f"{name}_mask.bin",
        dtype=bool,
        mode="r+",
        shape=(len(dates), len(ticks)),
    )
    mask_arr[dt] = False
    mask_arr[dt,:n_valid] = ~np.isnan(weights)
    mask_arr.flush()


def update_index(date, dates, ticks, conn, root):
    """
    每日更新六个指数最近一期成分权重和成分 mask。

    对每个指数取 date 之前最近的完整快照，而不是对每只股票分别前填；
    这样股票被调出指数后，会在下一次快照中正确变为 NaN/False。
    """
    n_valid = np.count_nonzero(ticks != "")
    valid_ticks = ticks[:n_valid]
    
    latest = _read_latest_index_snapshot(date, conn)
    result = {}

    for index_code, name in INDEX_CODE_TO_NAME.items():
        weight = (
            latest
            .filter(pl.col("index_code") == index_code)
            .select(["tick", "weight"])
        ).sort('tick').to_pandas().set_index('tick').reindex(index=valid_ticks).values.astype(float).flatten()

        _write_index_row(date, dates, ticks, root, name, weight)
        result[name] = weight

    return result




