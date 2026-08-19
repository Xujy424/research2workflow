import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path


DTYPE = np.float32

def _asof(date):
    return pd.Timestamp(date).normalize()


def _ensure_memmap(path, shape):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    if not path.exists():
        with path.open("wb") as file:
            file.truncate(size)
        array = np.memmap(path, dtype=DTYPE, mode="r+", shape=shape)
        array[:] = np.nan
        array.flush()
    if path.stat().st_size != size:
        raise ValueError(f"{path} size does not match axes {shape}")
    return np.memmap(path, dtype=DTYPE, mode="r+", shape=shape)


def update_d_moneyflow(root, dates, date, ticks):
    date = _asof(date)
    date_str = date.strftime("%Y%m%d")
    shcj = pl.scan_parquet(root/'l2'/'proc'/date_str/'shcj.pq')
    szcj = pl.scan_parquet(root/'l2'/'proc'/date_str/'szcj.pq')
    cols = list(set(shcj.collect().columns).intersection(set(szcj.collect().columns)))

    cj = pl.concat([shcj.select(cols), szcj.select(cols)])
    cj = cj.with_columns(
        pl.col('SecurityID').cast(pl.String).str.pad_start(6, '0'),
        (pl.col('Price')*pl.col('OrderQty')).alias('Amount')
    )
    cj = cj.with_columns(
        pl.when((0<=pl.col('Amount')) & (pl.col('Amount')<50000)).then(1)        # sm
        .when((50000<=pl.col('Amount')) & (pl.col('Amount')<300000)).then(2)     # mid
        .when((300000<=pl.col('Amount')) & (pl.col('Amount')<1000000)).then(3)   # lg
        .otherwise(4)                                                            # elg
        .alias('Size')
    )
    df = cj.group_by(['SecurityID','Size','Side']).agg(
        pl.col('Amount').sum().alias('amount'),
        pl.col('OrderQty').sum().alias('vol')
    ).collect()

    dt = np.searchsorted(dates, date)
    for size, i in dict(zip(['sm','mid','lg','elg'],[1,2,3,4])).items():
        for side, j in dict(zip(['buy','sell'],[1,-1])).items():
            for feat in ['amount','vol']:
                f_name = '_'.join([size,side,feat])
                path = root / "d_moneyflow" / f"{f_name}.bin"
                arr = _ensure_memmap(path, (len(dates), len(ticks)))
                arr[dt] = (df
                           .filter( (pl.col('Size')==i) & (pl.col('Side')==j) )
                           .select(['SecurityID',feat])
                        ).sort('SecurityID').to_pandas().set_index('SecurityID').reindex(index=ticks).values.astype(DTYPE).flatten()
                arr.flush()
    return




if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/ticks.npy',allow_pickle=True)
    root = Path('D:/data')

    update_d_moneyflow(root, dates, date, ticks)
