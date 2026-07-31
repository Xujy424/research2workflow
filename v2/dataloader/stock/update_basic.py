import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime



def update_hit(date, ticks, conn):
    sql_hit = f'''
    select distinct 
        B.SecuCode as tick, 
        A.StockBoard as "ceil_only", 
        A.LimitBoard as "floor_only",
        A.SurgedLimit as "hit_ceil",
        A.DeclineLimit as "hit_floor"
    from QT_PerformanceData A
    left join SecuMain B on A.InnerCode = B.InnerCode
    where A.TradingDay = '{date}' 
    and B.SecuMarket in (83,90) 
    and B.SecuCategory=1
    union all
    select distinct 
        B.SecuCode as tick, 
        C.StockBoard as "ceil_only", 
        C.LimitBoard as "floor_only",
        C.SurgedLimit as "hit_ceil",
        C.DeclineLimit as "hit_floor"
    from LC_STIBPerformanceData C
    left join SecuMain B on C.InnerCode = B.InnerCode
    where C.TradingDay = '{date}' 
    and B.SecuMarket in (83,90) 
    and B.SecuCategory=1
    '''
    ceil_only = pl.read_database(sql_hit, conn).select(['tick','ceil_only']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    ceil_only = np.nan_to_num(ceil_only, nan=1)
    floor_only = pl.read_database(sql_hit, conn).select(['tick','floor_only']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    floor_only = np.nan_to_num(floor_only, nan=1)
    hit_ceil = pl.read_database(sql_hit, conn).select(['tick','hit_ceil']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    hit_ceil = np.nan_to_num(hit_ceil, nan=1)
    hit_floor = pl.read_database(sql_hit, conn).select(['tick','hit_floor']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    hit_floor = np.nan_to_num(hit_floor, nan=1)
    return ceil_only, floor_only, hit_ceil, hit_floor


def update_pricelimit(date, ticks, conn):
    sql_pricelimit = f'''
    select distinct 
        B.SecuCode as tick, 
        A.PriceCeiling as "price_ceil", 
        A.PriceFloor as "price_floor"
    from DZ_PriceLimit A
    left join SecuMain B on A.InnerCode = B.InnerCode
    where A.TradingDay = '{date}' 
    and B.SecuMarket in (83,90) 
    and B.SecuCategory=1
    '''
    price_ceil = pl.read_database(sql_pricelimit, conn).select(['tick','price_ceil']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    price_floor = pl.read_database(sql_pricelimit, conn).select(['tick','price_floor']).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    return price_ceil, price_floor


def update_isst(date, ticks, conn):
    sql_isst = f'''
    select distinct 
        B.SecuCode as tick, 
        A.SpecialTradeType as "type"
    from LC_SpecialTrade A
    left join SecuMain B on A.InnerCode = B.InnerCode
    where A.InfoPublDate = '{date}' 
    union all
    select distinct 
        B.SecuCode as tick,
        C.ChangeType as "type"
    from LC_STIBSecuChange C
    left join SecuMain B on C.InnerCode = B.InnerCode
    where C.InfoPublDate = '{date}' 
    '''
    st_df = pl.read_database(sql_isst, conn)
    st_tick_set = set(st_df.filter(pl.col("type").is_not_null())["tick"].to_list())
    return np.array([t in st_tick_set for t in ticks])


def update_issuspend(date, ticks, conn):
    sql_issuspend = f'''
    select distinct 
        C.SecuCode as tick, 
        A.Ifsuspend as "is_suspend"
    from QT_StockPerformance A
    left join SecuMain C on A.InnerCode = C.InnerCode
    where A.TradingDay = '{date}' 
    and C.SecuMarket in (83,90)
    and C.SecuCategory=1
    union all
    select distinct 
        C.SecuCode as tick, 
        B.Ifsuspend as "is_suspend"
    from LC_STIBPerformance B
    left join SecuMain C on B.InnerCode = C.InnerCode
    where B.TradingDay = '{date}' 
    and C.SecuMarket in (83,90) 
    and C.SecuCategory=1
    '''
    suspend_df = pl.read_database(sql_issuspend, conn)
    suspend_dict = dict(zip(suspend_df["tick"], suspend_df["is_suspend"].fill_null(1).to_list()))
    return np.array([suspend_dict.get(tick, 1) == 1 for tick in ticks])


def update_isnew(date, ticks, conn):
    sql_listdate = f'''
    select 
        SecuCode as tick, 
        COALESCE(ListedDate, '1900-01-01') as "list_date"
    from SecuMain
    where SecuMarket in (83,90) 
    and SecuCategory=1
    '''
    list_df = pl.read_database(sql_listdate, conn)
    list_dict = dict(zip(list_df["tick"], list_df["list_date"].to_list())) if list_df.height > 0 else {}
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    return np.array([
        True if tick not in list_dict
        else list_dict[tick] == '1900-01-01' or (date_dt - list_dict[tick]).days < 242
        for tick in ticks
    ])


def update_hastrade(date, ticks, conn):
    sql_volume = f'''
    select
        C.SecuCode as "tick",
        A.TurnoverVolume as "volume"
    from QT_StockPerformance A
    left join SecuMain C
    on A.InnerCode = C.InnerCode
    where A.TradingDay = '{date}' 
        and C.SecuMarket in (83,90)
        and C.SecuCategory=1
    union all
    select
        C.SecuCode as "tick",
        B.TurnoverVolume as "volume"
    from LC_STIBPerformance B
    left join SecuMain C
    on B.InnerCode = C.InnerCode
    where B.TradingDay = '{date}' 
        and C.SecuMarket in (83,90)
        and C.SecuCategory=1
    '''
    volume = pl.read_database(sql_volume, conn).sort('tick').to_pandas().set_index('tick').reindex(index=ticks).values.astype(float).flatten()
    has_trade = pd.notna(volume) & (volume != 0)
    return has_trade



def update_basic(date, dates, ticks, conn, root):
    feats_name = ['ceil_only','floor_only','hit_ceil','hit_floor','hastrade','isst','isnew','issuspend','price_ceil','price_floor']

    ceil_only, floor_only, hit_ceil, hit_floor = update_hit(date, ticks, conn)
    hastrade = update_hastrade(date, ticks, conn)
    isst = update_isst(date, ticks, conn)       
    isnew = update_isnew(date, ticks, conn)
    issuspend = update_issuspend(date, ticks, conn)
    price_ceil, price_floor = update_pricelimit(date, ticks, conn)
    feats = [ceil_only,floor_only,hit_ceil,hit_floor,hastrade,isst,isnew,issuspend,price_ceil,price_floor]
    basic_feats = dict(zip(feats_name, feats))

    dt = np.searchsorted(dates, date)
    for arr_name in feats_name:
        path = root / "basic" / f"{arr_name}.bin"
        arr = np.load(path, mmap_mode="r+", allow_pickle=False)
        arr[dt] = basic_feats[arr_name]
        arr.flush()
    return
        



def get_next_trading_day(date, dates) -> str:
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    dt = np.searchsorted(dates, date_dt)

    n_valid = int(np.count_nonzero(~np.isnat(dates)))
    if dt+1>=n_valid: return date
    next_date_dt = dates[dt+1]
    return datetime.strftime(next_date_dt, "%Y-%m-%d")


def update_tradable(date, dates, ticks, conn):
    ceil_only, floor_only, hit_ceil, hit_floor = update_hit(date, ticks, conn)
    hastrade = update_hastrade(date, ticks, conn)
    isst = update_isst(date, ticks, conn)       
    isnew = update_isnew(date, ticks, conn)
    issuspend= update_issuspend(date, ticks, conn)

    next_date = get_next_trading_day(date, dates)
    next_ceil_only, next_floor_only, next_hit_ceil, next_hit_floor = update_hit(next_date, ticks, conn)
    next_hastrade = update_hastrade(next_date, ticks, conn)

    tradable = hastrade & next_hastrade &\
               ~ceil_only & ~floor_only & ~hit_ceil & ~hit_floor & ~next_ceil_only & ~next_floor_only & ~next_hit_ceil & ~next_hit_floor &\
               ~isst & ~isnew & ~issuspend
    return tradable

