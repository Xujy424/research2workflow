import numpy as np
import polars as pl
from pathlib import Path
import pandas as pd
import sys

if __package__:
    from ..utils import (
        asof as _asof,
        date_index as _date_index,
        ensure_memmap as _ensure_memmap,
    )
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.utils import (
        asof as _asof,
        date_index as _date_index,
        ensure_memmap as _ensure_memmap,
    )

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
    dt = _date_index(date, dates)
    n_valid = np.count_nonzero(ticks != "")

    path = root / 'index'/ "weight" / f"{name}_weight.bin"
    weight_arr = _ensure_memmap(
        path, (len(dates), len(ticks)), np.float32, np.nan
    )
    weight_arr[dt] = np.nan
    weight_arr[dt,:n_valid] = weights
    weight_arr.flush()

    path = root / 'index'/ "mask" / f"{name}_mask.bin"
    mask_arr = _ensure_memmap(
        path, (len(dates), len(ticks)), np.bool_, False
    )
    mask_arr[dt] = False
    mask_arr[dt,:n_valid] = ~np.isnan(weights)
    mask_arr.flush()


def _snapshot_weights(latest, index_code, valid_ticks):
    """Return raw snapshot weights and their EndDate."""
    frame = (
        latest.filter(pl.col("index_code") == index_code)
        .select(["tick", "snapshot_date", "weight"])
        .sort("tick")
        .to_pandas()
    )
    if frame.empty:
        return np.full(len(valid_ticks), np.nan), None
    snapshot_date = pd.Timestamp(frame["snapshot_date"].max()).normalize()
    weights = frame.set_index("tick")["weight"].reindex(valid_ticks).to_numpy(dtype=float)
    return weights, snapshot_date


def _drift_weights(previous_weights, pct):
    """Drift weights by constituent daily returns (pct is in percent units)."""
    member = np.isfinite(previous_weights) & (previous_weights > 0)
    daily_return = np.where(np.isfinite(pct), pct / 100, 0.0)
    weights = np.where(member, previous_weights * (1 + daily_return), np.nan)
    total = np.nansum(weights)
    if total > 0:
        weights *= np.nansum(previous_weights) / total
    return weights


def get_init_pct(snapshot_date, date, ticks, conn):
    """Return compounded percent returns over (snapshot_date, date]."""
    start = _asof(snapshot_date).strftime("%Y-%m-%d")
    end = _asof(date).strftime("%Y-%m-%d")
    queries = []
    for table in ("QT_StockPerformance", "LC_STIBPerformance"):
        queries.append(f"""
            select C.SecuCode as tick, A.TradingDay as trading_day,
                   A.ChangePCT as pct
            from {table} A
            left join SecuMain C on A.InnerCode = C.InnerCode
            where A.TradingDay > '{start}' and A.TradingDay <= '{end}'
              and C.SecuMarket in (83,90) and C.SecuCategory = 1
        """)
    frame = pl.read_database(" union all ".join(queries), conn).to_pandas()
    if frame.empty:
        return np.zeros(len(ticks), dtype=float)
    daily_pct = frame["pct"].to_numpy(dtype=float)
    frame["factor"] = 1 + np.where(np.isfinite(daily_pct), daily_pct / 100, 0.0)
    factors = frame.groupby("tick")["factor"].prod().reindex(ticks, fill_value=1.0)
    return (factors.to_numpy(dtype=float) - 1) * 100


def update_index(date, dates, ticks, conn, root):
    """Write daily weights from snapshots and price drift between snapshots."""
    date = _asof(date)
    dt = _date_index(date, dates)

    n_valid = np.count_nonzero(ticks != "")
    valid_ticks = ticks[:n_valid]
    
    latest = _read_latest_index_snapshot(date, conn)
    pct_path = Path(root) / "d_essentials" / "pct.bin"
    matrix_shape = (len(dates), len(ticks))
    expected_size = int(np.prod(matrix_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if dt > 0 and not pct_path.exists():
        raise FileNotFoundError(
            f"{pct_path} is missing; run update_d_essentials before update_index"
        )
    if dt > 0 and pct_path.stat().st_size != expected_size:
        raise ValueError(f"{pct_path} does not match the stock date/tick axis")
    pct = np.memmap(pct_path, dtype=np.float32, mode="r", shape=matrix_shape) if dt > 0 else None
    result = {}
    init_pct = {}

    for index_code, name in INDEX_CODE_TO_NAME.items():
        snapshot, snapshot_date = _snapshot_weights(latest, index_code, valid_ticks)
        if snapshot_date is None or snapshot_date == date:
            weight = snapshot
        elif dt == 0:
            if snapshot_date not in init_pct:
                init_pct[snapshot_date] = get_init_pct(
                    snapshot_date, date, valid_ticks, conn
                )
            weight = _drift_weights(snapshot, init_pct[snapshot_date])
        else:
            weight_path = Path(root) / "index" / "weight" / f"{name}_weight.bin"
            if weight_path.exists() and weight_path.stat().st_size == expected_size:
                previous_weights = np.memmap(
                    weight_path, dtype=np.float32, mode="r", shape=matrix_shape
                )[dt - 1, :n_valid].astype(float)
            else:
                previous_weights = np.full(n_valid, np.nan)

            if np.isfinite(previous_weights).any():
                weight = _drift_weights(
                    previous_weights,
                    np.asarray(pct[dt, :n_valid], dtype=float),
                )
            else:
                raise ValueError(
                    f"Missing previous {name} weights for {date.date()}; "
                    "update index weights in date order from the snapshot date"
                )

        _write_index_row(date, dates, ticks, root, name, weight)
        result[name] = weight

    return result



if __name__ == '__main__':
    from pathlib import Path
    import pymssql

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/stock_ticks.npy', allow_pickle=False)
    root = Path('D:/data')


    JY_CONFIG = {
            "server": "10.10.0.102",
            "user": "jydbReader",
            "password": "jy@9043!Reader",
            "database": "jydb",
            "charset": "cp936",
        }
    jy_conn = pymssql.connect(**JY_CONFIG)

    update_index("2024-06-14", dates, ticks, jy_conn, root)
