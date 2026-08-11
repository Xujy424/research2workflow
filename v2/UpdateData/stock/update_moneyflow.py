import polars as pl
import numpy as np
import pandas as pd



def update_d_moneyflow(shcj, szcj, root, dates, date, ticks):
    cj = pl.concat([shcj,szcj])
    cj = cj.with_columns(
        (pl.col('Price')*pl.col('OrdeQty')).alias('Amount')
    )
    cj = cj.with_columns(
        pl.when((0<=pl.col('Amount')) & (pl.col('Amount')<50000)).then(1)        # sm
        .when((50000<=pl.col('Amount')) & (pl.col('Amount')<300000)).then(2)     # mid
        .when((300000<=pl.col('Amount')) & (pl.col('Amount')<1000000)).then(3)   # lg
        .otherwise(4)                                                            # elg
        .alias('Size')
    )
    df = cj.group_by(['SecruityID','Size','Side']).agg(
        pl.col('Amount').sum().alias('amount'),
        pl.col('OrderQty').sum().alias('vol')
    )

    dt = np.searchsorted(dates, date)
    for size, i in dict(zip(['sm','mid','lg','elg'],[1,2,3,4])).items():
        for side, j in dict(zip(['buy','sell'],[1,-1])).items():
            for feat in ['amount','vol']:
                f_name = '_'.join([size,side,feat])
                path = root / "d_essentials" / f"{f_name}.bin"
                arr = np.memmap(path, dtype=float, mode='r+', shape=(len(dates),len(ticks)))
                arr[dt] = (df
                           .filter( (pl.col('Size')==i) & (pl.col('Side')==j) )
                           .select(['SecruityID',feat])
                        ).sort('SecruityID').to_pandas().set_index('SecruityID').reindex(index=ticks).values.astype(float).flatten()
                arr.flush()
    return

