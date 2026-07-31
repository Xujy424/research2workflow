import polars as pl
import numpy as np
import pandas as pd



def update_adjfactor(date, ticks, conn):
    sql_adjfactor = f"""
    WITH ranked AS (
        SELECT 
            B.SecuCode AS tick,
            A.AdjustingFactor AS scaler,
            A.AdjustingConst AS const,
            ROW_NUMBER() OVER(PARTITION BY B.SecuCode ORDER BY A.ExDiviDate DESC) AS rn
        FROM QT_AdjustingFactor A
        LEFT JOIN SecuMain B ON A.InnerCode = B.InnerCode
        WHERE A.ExDiviDate <= '{date}'
        UNION ALL
        SELECT 
            B.SecuCode AS tick,
            C.AdjustingFactor AS scaler,
            C.AdjustingConst AS const,
            ROW_NUMBER() OVER(PARTITION BY B.SecuCode ORDER BY C.ExDiviDate DESC) AS rn
        FROM LC_STIBAdjustingFactor C
        LEFT JOIN SecuMain B ON C.InnerCode = B.InnerCode
        WHERE C.ExDiviDate <= '{date}'
    )
    SELECT tick, scaler, const FROM ranked WHERE rn = 1
    """
    recent_df = pl.read_database(sql_adjfactor, conn)
    all_df = pl.DataFrame({'tick':[t for t in ticks if t!='']})
    result = all_df.join(recent_df, on="tick", how="left").with_columns([
        pl.col("scaler").fill_null(1.0),
        pl.col('const').fill_null(0)
    ])
    scaler = result.sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['scaler'].values.astype(float).flatten()
    const = result.sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['const'].values.astype(float).flatten()
    return scaler, const


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
        A.TotalMV/A.ClosePrice as "total_share",
        A.TurnoverRate as "turnover",
        A.NegotiableMV as "cicr_mv",
        A.NegotiableMV/A.ClosePrice as "cicr_share",
        A.TurnoverVolume/(A.NegotiableMV/A.ClosePrice) as "cicr_turnover",
        A.TurnoverRateFreeFloat as "free_turnover",
        A.TurnoverVolume/A.TurnoverRateFreeFloat as "free_share",
        (A.TurnoverVolume/A.TurnoverRateFreeFloat)*A.ClosePrice as "free_mv"
    from QT_StockPerformance A
    left join SecuMain C
    on A.InnerCode = C.InnerCode
    where A.TradingDay = '{date}'
        and C.SecuMarket in (83,90)
        and C.SecuCategory=1
    union all
    select
        C.SecuCode as "tick",
        B.TradingDay as "date",
        B.OpenPrice as "open",
        B.HighPrice as "high",
        B.LowPrice as "low",
        B.ClosePrice as "close",
        B.TurnoverVolume as "volume",
        B.TurnoverValue as "amount",
        B.ChangePCT as "pct",
        B.AvgPrice as "vwap",
        B.TotalMV as "total_mv",
        B.TotalMV/B.ClosePrice as "total_share",
        B.TurnoverRate as "turnover",
        B.NegotiableMV as "cicr_mv",
        B.NegotiableMV/B.ClosePrice as "cicr_share",
        B.TurnoverVolume/(B.NegotiableMV/B.ClosePrice) as "cicr_turnover",
        B.TurnoverRateFreeFloat as "free_turnover",
        B.TurnoverVolume/B.TurnoverRateFreeFloat as "free_share",
        (B.TurnoverVolume/B.TurnoverRateFreeFloat)*B.ClosePrice as "free_mv"
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
    scaler, const = update_adjfactor(date, ticks, conn)
    d_essentials = update_dfield(date, ticks, conn)
    for f_name in ['open','high','low','close','volume','vwap']:
        d_essentials[f_name+'_adj'] = d_essentials[f_name]*scaler+const
    d_essentials['cjbs'] = update_cjbs(date, ticks, conn)

    dt = np.searchsorted(dates, date)
    for f_name, f in d_essentials.items():
        path = root / "d_essentials" / f"{f_name}.bin"
        arr = np.load(path, mmap_mode="r+", allow_pickle=False)
        arr[dt] = d_essentials[f_name]
        arr.flush()
    return
    
    
