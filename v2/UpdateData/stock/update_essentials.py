import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import sys

if __package__:
    from ..config import get_jy_conn, get_str_engine
    from ..utils import (
        date_index as _date_index,
        ensure_memmap as _shared_ensure_memmap,
    )
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.config import get_jy_conn, get_str_engine
    from v2.UpdateData.utils import (
        date_index as _date_index,
        ensure_memmap as _shared_ensure_memmap,
    )


DTYPE = np.float32


def _ensure_memmap(path, shape):
    return _shared_ensure_memmap(path, shape, dtype=DTYPE)



def update_adjfactor(date, ticks, conn):
    sql_adjfactor = f"""
    WITH ranked AS (
        SELECT 
            B.SecuCode AS tick,
            A.AdjustingFactor AS scaler,
            A.AdjustingConst AS const,
            A.RatioAdjustingFactor AS ratio_scaler,
            ROW_NUMBER() OVER(PARTITION BY B.SecuCode ORDER BY A.ExDiviDate DESC) AS rn
        FROM QT_AdjustingFactor A
        LEFT JOIN SecuMain B ON A.InnerCode = B.InnerCode
        WHERE A.ExDiviDate <= '{date}'
        UNION ALL
        SELECT 
            B.SecuCode AS tick,
            C.AdjustingFactor AS scaler,
            C.AdjustingConst AS const,
            C.RatioAdjustingFactor AS ratio_scaler,
            ROW_NUMBER() OVER(PARTITION BY B.SecuCode ORDER BY C.ExDiviDate DESC) AS rn
        FROM LC_STIBAdjustingFactor C
        LEFT JOIN SecuMain B ON C.InnerCode = B.InnerCode
        WHERE C.ExDiviDate <= '{date}'
    )
    SELECT tick, scaler, const, ratio_scaler FROM ranked WHERE rn = 1
    """
    recent_df = pl.read_database(sql_adjfactor, conn)
    all_df = pl.DataFrame({'tick':[t for t in ticks if t!='']})
    result = all_df.join(recent_df, on="tick", how="left").with_columns([
        pl.col("scaler").fill_null(1.0),
        pl.col('const').fill_null(0),
        pl.col('ratio_scaler').fill_null(1.0),
    ])
    scaler = result.sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['scaler'].values.astype(float).flatten()
    const = result.sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['const'].values.astype(float).flatten()
    ratio_scaler = result.sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['ratio_scaler'].values.astype(float).flatten()
    return scaler, const, ratio_scaler


def update_dfield(date, ticks, conn):
    sql_dfield = f'''select
        C.SecuCode as "tick",
        A.OpenPrice as "open",
        A.HighPrice as "high",
        A.LowPrice as "low",
        A.ClosePrice as "close",
        A.TurnoverVolume as "volume",
        A.TurnoverValue as "amount",
        A.ChangePCT as "pct",
        A.AvgPrice as "vwap",
        A.TotalMV as "total_mv",
        A.TotalMV/NULLIF(A.ClosePrice,0) as "total_share",
        A.TurnoverRate as "turnover",
        A.NegotiableMV as "circ_mv",
        A.NegotiableMV/NULLIF(A.ClosePrice,0) as "circ_share",
        A.TurnoverVolume/NULLIF(A.NegotiableMV/NULLIF(A.ClosePrice,0),0) as "circ_turnover",
        A.TurnoverRateFreeFloat as "free_turnover",
        A.TurnoverVolume/NULLIF(A.TurnoverRateFreeFloat,0) as "free_share",
        (A.TurnoverVolume/NULLIF(A.TurnoverRateFreeFloat,0))*A.ClosePrice as "free_mv"
    from QT_StockPerformance A
    left join SecuMain C
    on A.InnerCode = C.InnerCode
    where A.TradingDay = '{date}'
        and C.SecuMarket in (83,90)
        and C.SecuCategory=1
    union all
    select
        C.SecuCode as "tick",
        B.OpenPrice as "open",
        B.HighPrice as "high",
        B.LowPrice as "low",
        B.ClosePrice as "close",
        B.TurnoverVolume as "volume",
        B.TurnoverValue as "amount",
        B.ChangePCT as "pct",
        B.AvgPrice as "vwap",
        B.TotalMV as "total_mv",
        B.TotalMV/NULLIF(B.ClosePrice,0) as "total_share",
        B.TurnoverRate as "turnover",
        B.NegotiableMV as "circ_mv",
        B.NegotiableMV/NULLIF(B.ClosePrice,0) as "circ_share",
        B.TurnoverVolume/NULLIF(B.NegotiableMV/NULLIF(B.ClosePrice,0),0) as "circ_turnover",
        B.TurnoverRateFreeFloat as "free_turnover",
        B.TurnoverVolume/NULLIF(B.TurnoverRateFreeFloat,0) as "free_share",
        (B.TurnoverVolume/NULLIF(B.TurnoverRateFreeFloat,0))*B.ClosePrice as "free_mv"
    from LC_STIBPerformance B
    left join SecuMain C
    on B.InnerCode = C.InnerCode
    where B.TradingDay = '{date}'
        and C.SecuMarket in (83,90)
        and C.SecuCategory=1
    '''
    dfield = pl.read_database(sql_dfield, conn).sort('tick').to_pandas().set_index('tick').reindex(index=ticks)
    d_essentials = {}
    for f_name in dfield.columns:
        d_essentials[f_name] = dfield[f_name].values.astype(float).flatten()
    return d_essentials


def update_cjbs(date, ticks, conn):
    sql_cjbs = f'''select
            C.SecuCode as "tick",
            A.TurnoverDeals as "cjbs"
        from QT_DailyQuote A
        left join SecuMain C
        on A.InnerCode = C.InnerCode
        where A.TradingDay = '{date}'
            and C.SecuMarket in (83,90)
            and C.SecuCategory=1
        union all
        select
            C.SecuCode as "tick",
            B.TurnoverDeals as "cjbs"
        from LC_STIBDailyQuote B
        left join SecuMain C
        on B.InnerCode = C.InnerCode
        where B.TradingDay = '{date}'
            and C.SecuMarket in (83,90)
            and C.SecuCategory=1
        '''
    cjbs = pl.read_database(sql_cjbs, conn).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    return cjbs


def update_d_essentials(date, dates, ticks, conn, root):
    scaler, const, ratio_scaler = update_adjfactor(date, ticks, conn)
    d_essentials = update_dfield(date, ticks, conn)
    for f_name in ['open','high','low','close','vwap']:
        d_essentials[f_name+'_adj'] = d_essentials[f_name]*scaler+const
        d_essentials[f_name+'_ratio_adj'] = d_essentials[f_name]*ratio_scaler
    d_essentials['volume_adj'] = np.divide(
        d_essentials['volume'], scaler,
        out=np.full(len(ticks), np.nan), where=scaler != 0,
    )
    d_essentials['adjusting_factor'] = scaler
    d_essentials['adjusting_const'] = const
    d_essentials['ratio_adjusting_factor'] = ratio_scaler
    d_essentials['cjbs'] = update_cjbs(date, ticks, conn)

    dt = _date_index(date, dates)
    for f_name, f in d_essentials.items():
        path = root / "d_essentials" / f"{f_name}.bin"
        arr = _ensure_memmap(path, (len(dates), len(ticks)))
        arr[dt] = f
        arr.flush()
    return
    
    

def update_m_essentials(date, dates, ticks, conn, root):
    sql_mfield = f'''
    select
        SecCode      as "tick",
        BarTime      as "time",
        Open/1000.0    as "open",
        High/1000.0    as "high",
        Low/1000.0     as "low",
        close/1000.0   as "close",
        Volume/1000.0  as "volume",
        Amount/1000.0  as "amount"
    from StockPriceOneMin
    where FDate = '{date}'
    '''
    mfield = pl.read_database(sql_mfield, conn).sort(['tick',"time"])
    dt = _date_index(date, dates)
    factors = {}
    for name in ('adjusting_factor', 'adjusting_const', 'ratio_adjusting_factor'):
        path = root / 'd_essentials' / f'{name}.bin'
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run update_d_essentials before update_m_essentials"
            )
        factors[name] = np.asarray(
            np.memmap(path, dtype=DTYPE, mode='r', shape=(len(dates), len(ticks)))[dt]
        )

    fields = {}
    raw_names = [name for name in mfield.columns if name not in {'tick', 'time'}]
    for name in raw_names:
        pivoted = mfield.pivot(index='time', columns='tick', values=name).to_pandas().set_index('time')
        values = pivoted.reindex(columns=ticks).to_numpy(dtype=DTYPE)
        if values.shape[0] != 241:
            raise ValueError(f"{date} has {values.shape[0]} minute bars, expected 241")
        if name in {'volume', 'amount'}:
            values = np.nan_to_num(values, nan=0.0)
        fields[name] = values

    exact_factor = factors['adjusting_factor'][None, :]
    exact_const = factors['adjusting_const'][None, :]
    ratio_factor = factors['ratio_adjusting_factor'][None, :]
    # for name in ('open', 'high', 'low', 'close'):
    #     fields[f'{name}_adj'] = fields[name] * exact_factor + exact_const
    #     fields[f'{name}_ratio_adj'] = fields[name] * ratio_factor
    # fields['volume_adj'] = np.divide(
    #     fields['volume'], exact_factor,
    #     out=np.full_like(fields['volume'], np.nan), where=exact_factor != 0,
    # )

    for f_name, f in fields.items():
        path = root / "m_essentials" / f"{f_name}.bin"
        arr = _ensure_memmap(path, (len(dates), 241, len(ticks)))
        arr[dt,:,:] = f
        arr.flush()



if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/stock_ticks.npy', allow_pickle=False)
    jy_conn = get_jy_conn()
    str_conn = get_str_engine()
    root = Path('D:/data')

    # update_d_essentials(
    #     date, dates, ticks, jy_conn, root
    # )
    update_m_essentials(
        date, dates, ticks, str_conn, root
    )
