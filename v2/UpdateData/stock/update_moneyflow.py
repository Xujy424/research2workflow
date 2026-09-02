import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
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

DTYPE = np.float32


def update_d_moneyflow(root, dates, date, ticks, l2_root=None):
    date = _asof(date)
    date_str = date.strftime("%Y%m%d")
    l2_root = Path(l2_root) if l2_root is not None else Path(root) / "l2"
    columns = ["SecurityID", "Price", "OrderQty", "Side"]
    shcj = pl.scan_parquet(
        l2_root/'proc'/date_str/'shcj.pq'
    ).select(columns)
    szcj = pl.scan_parquet(
        l2_root/'proc'/date_str/'szcj.pq'
    ).select(columns)
    cj = pl.concat([shcj, szcj])
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

    dt = _date_index(date, dates)
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

    # date = '2024-06-14'
    # dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    # ticks = np.load('D:/data/axis/stock_ticks.npy', allow_pickle=False)
    # root = Path('D:/data')

    # update_d_moneyflow(root, dates, date, ticks)


    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import tushare as ts

    pro = ts.pro_api('OQa740832ihF7m4ZdjKVKMI0rkF6v9xLvw2Phi-EbXIdjAaCfUCSz_5WQ4l3bBxB')
    pro._DataApi__http_url = "https://goostar.cn/api/dispatch"

    dates = np.load('Z:/axis/dates.npy',allow_pickle=True)
    valid_dates = [date for date in dates if not np.isnat(date)]
    ticks = np.load('Z:/axis/stock_ticks.npy', allow_pickle=True)
    valid_ticks = [tick for tick in ticks if tick!='']

    for date in tqdm(valid_dates[590+679+793+154+357+214+87+67:]):
        df = pro.moneyflow(
            trade_date=pd.Timestamp(date).strftime("%Y%m%d")
        ).drop(['trade_date'], axis=1).set_index('ts_code')
        df.index = df.index.str.split('.').str[0]
        df = df.sort_index().reindex(index=valid_ticks)
        for col in df.columns:
            val = df[col].to_numpy().astype(np.float32)
            val_path = Path('Z:/d_moneyflow') / f"{col}.bin"
            if not val_path.exists():
                arr = np.full((len(dates), len(ticks)), np.nan, dtype=np.float32)
                arr.tofile(val_path)
            else:
                arr = np.memmap(val_path, dtype=np.float32, mode='r+', shape=(len(dates), len(ticks)))
            dt = np.searchsorted(dates,date)
            arr[dt,:len(valid_ticks)] = val
            arr.flush()

    # d = np.memmap('Z:/d_moneyflow/buy_sm_vol.bin', dtype=np.float32, mode='r', shape=(len(dates), len(ticks)))
    # print(d[:591])
            
