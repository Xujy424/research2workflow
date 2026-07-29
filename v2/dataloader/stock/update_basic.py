import polars as pl
import numpy as np
import pandas as pd



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

sql_pricelimit = f'''
select distinct 
    B.SecuCode as tick, 
    A.PriceCeiling as "price_ceil", 
    A.PriceFloor as "price_floor"
from QT_PriceLimit A
left join SecuMain B on A.InnerCode = B.InnerCode
where A.TradingDay = '{date}' 
and B.SecuMarket in (83,90) 
and B.SecuCategory=1
union all
select distinct 
    B.SecuCode as tick, 
    C.PriceCeiling as "price_ceil", 
    C.PriceFloor as "price_floor"
from LC_STIBPriceLimit C
left join SecuMain B on C.InnerCode = B.InnerCode
where C.TradingDay = '{date}' 
and B.SecuMarket in (83,90) 
and B.SecuCategory=1
'''

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


sql_listdate = f'''
select 
    SecuCode as tick, 
    COALESCE(ListedDate, '1900-01-01') as "list_date"
from SecuMain
where SecuMarket in (83,90) 
and SecuCategory=1
'''
list_df = pl.read_database(sql_listdate, JY_CONN)
list_dict = dict(zip(list_df["tick"], list_df["list_date"].to_list())) if list_df.height > 0 else {}
date_dt = datetime.strptime(date, "%Y-%m-%d")

path = Path(ROOT) / "axis" / "ticks.npy"
ticks = np.load(path, allow_pickle=True)
is_new = np.full((len(ticks),), True)
for tick,list_date in list_dict.items():
    if tick in ticks and list_date!='1900-01-01' and (date_dt - list_date).days >= 242 :
        idx = np.searchsorted(ticks, tick)
        is_new[idx] = False


# has trade

# tradable