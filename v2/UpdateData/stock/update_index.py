import numpy as np
import polars as pl

from ..download.update_axis import init_empty_field


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

    path = root / "mask" / f"{name}_weight.bin"
    if not path.exists():
        init_empty_field(dates, ticks, 'mask', f"{name}_weight", np.float32, dim=None)
    weight_arr = np.memmap(
        path,
        dtype=np.float32,
        mode="r+",
        shape=(len(dates), len(ticks)),
    )
    weight_arr[dt] = np.nan
    weight_arr[dt,:n_valid] = weights
    weight_arr.flush()

    path = root / "mask" / f"{name}_mask.bin"
    if not path.exists():
        init_empty_field(dates, ticks, 'mask', f"{name}_mask", bool, dim=None)
    mask_arr = np.memmap(
        path,
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




