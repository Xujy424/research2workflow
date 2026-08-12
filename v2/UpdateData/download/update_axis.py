from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import exchange_calendars as xcals

if __package__:
    from ..config import get_jy_conn
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from v2.UpdateData.config import get_jy_conn


ROOT = Path("/data/xujiayi/xjy/")
DATE_RESERVE = 500
TICK_RESERVE = 1000


def is_tradedate(date=None):
    if date:
        today = date
    else:
        today = datetime.now( ZoneInfo("Asia/Shanghai") ).date()
    calendar = xcals.get_calendar("XSHG")
    if not calendar.is_session(str(today)):
        return False
    else:
        return True

def is_last_tradedate_of_year(date=None):
    if date:
        today = date
    else:
        today = datetime.now( ZoneInfo("Asia/Shanghai") ).date()

    cal = xcals.get_calendar('XSHG')
    end_of_year = today.replace(month=12, day=31)
    schedule = cal.schedule(start_date=today, end_date=end_of_year)
    
    if today not in schedule.index.date:
        return False
    return today == schedule.index[-1].date()


def update_date(date=None):
    if date:
        today = date
    else:
        today = datetime.now( ZoneInfo("Asia/Shanghai") ).date()
        today = np.datetime64(today, "D")

    path = Path(ROOT) / "axis" / "dates.npy"
    dates = np.load(
        path,
        mmap_mode="r+",
        allow_pickle=False,
    )
    n_dates = int(np.count_nonzero(~np.isnat(dates)))
    valid_dates = dates[:n_dates]

    idx = np.searchsorted(valid_dates, today)
    if idx < n_dates and valid_dates[idx] == today:
        print(f"{today} 已存在于 dates.npy")
        return 
    dates[n_dates] = today
    dates.flush()
    return


def get_all_ticks(date):
    JY_CONN = get_jy_conn()
    sql_axis = f'''select
            C.SecuCode as "tick"
        from QT_StockPerformance A
        left join SecuMain C
        on A.InnerCode = C.InnerCode
        where A.TradingDay = '{date}'
            and C.SecuMarket in (83,90)
            and C.SecuCategory=1
        union all
        select
            C.SecuCode as "tick"
        from LC_STIBPerformance B
        left join SecuMain C
        on B.InnerCode = C.InnerCode
        where B.TradingDay = '{date}'
            and C.SecuMarket in (83,90)
            and C.SecuCategory=1
    '''
    df = pl.read_database(sql_axis, JY_CONN).sort('tick')
    return df['tick'].unique().sort().to_numpy()

def update_tick(date):
    current_ticks = get_all_ticks(date)
    path = Path(ROOT) / "axis" / "ticks.npy"
    ticks = np.load(path, allow_pickle=True)

    is_valid = lambda x: x != ""
    n_valid = int(np.count_nonzero(is_valid(ticks)))
    valid_ticks = ticks[:n_valid]

    ipos = sorted(set(current_ticks).difference(set(valid_ticks)))
    if len(ipos)==0: return
    i = 0
    for ipo in ipos:
        ticks[n_valid+i]=ipo
        i += 1

    n_valid = int(np.count_nonzero(is_valid(ticks)))
    ticks[:n_valid] = sorted(ticks[:n_valid])
    np.save(path, ticks, allow_pickle=True)
    return 
        

def _ensure_axis_reverse(path, reserve, is_valid, fill_value):
    arr = np.load(path, allow_pickle=True)

    n_valid = int(np.count_nonzero(is_valid(arr)))
    free = arr.size - n_valid

    if free < reserve:
        new_arr = np.full(
            n_valid + reserve,
            fill_value,
            dtype=arr.dtype,
        )
        new_arr[:n_valid] = arr[:n_valid]
        np.save(path, new_arr, allow_pickle=True)

    return n_valid, len(arr)

def reset_axis():
    '''
        年底重置索引
    '''
    axis_dir = ROOT / "axis"
    date_n_valid, date_old_len = _ensure_axis_reverse(
        axis_dir / "dates.npy",
        DATE_RESERVE,
        lambda x: ~np.isnat(x),
        np.datetime64("NaT"),
    )
    tick_n_valid, tick_old_len = _ensure_axis_reverse(
        axis_dir / "ticks.npy",
        TICK_RESERVE,
        lambda x: x != "",
        "",
    )

    new_arr = np.full((date_n_valid+DATE_RESERVE, tick_n_valid+TICK_RESERVE), np.nan)
    stock_dir = ROOT/"stock"
    for p in stock_dir.rglob('*.bin'):
        arr = np.memmap(p, mode='r', dtype=bool if 'mask' in p else float, shape=(date_old_len, tick_old_len))
        new_arr[:date_old_len, :tick_old_len] = arr
        new_arr.astype(bool if 'mask' in p else float).tofile(p)
    return True


def init_empty_field(dates, ticks, fileshare, name, typ):
    T,N = len(dates), len(ticks)
    arr = np.full(shape=(T,N),fill_value=np.nan)
    arr.astype(typ).tofile(ROOT/f'{fileshare}'/f'{name}.bin')

    



if __name__ == '__main__':
    dates = np.load(ROOT/"axis"/"dates.npy", allow_pickle=True)
    ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)
    print(len(dates),len(ticks))
    _,_ = reset_axis()
    dates = np.load(ROOT/"axis"/"dates.npy", allow_pickle=True)
    ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)
    print(len(dates),len(ticks))


    # date_list = pd.date_range('2026-01-01', '2026-06-30').strftime('%Y-%m-%d').tolist()
    # date_list = np.array(date_list, dtype='datetime64')
    # new_tradeday = []
    # for d in date_list:
    #     if is_tradedate(d):
    #         new_tradeday.append(d)

    # import pymssql
    # JY_CONFIG = {
    #     "server": '10.10.0.102',
    #     "user": 'jydbReader',
    #     "password": 'jy@9043!Reader',
    #     "database": 'jydb',
    #     "charset": 'cp936'
    # }
    # JY_CONN = pymssql.connect(**JY_CONFIG)
    # sql_axis = f'''select
    #         TradingDay as "date"
    #     from QT_StockPerformance
    #     where TradingDay between '{"2026-01-01"}' and '{"2026-06-30"}'
    # '''
    # df = pl.read_database(sql_axis, JY_CONN).sort('date')
    # valid_new_tradeday = df['date'].unique().sort().to_numpy()

    # print(set(valid_new_tradeday).difference(set(new_tradeday)))
    # print(set(new_tradeday).difference(set(valid_new_tradeday)))


    # date_list = pd.date_range('2026-01-01', '2026-06-30').strftime('%Y-%m-%d').tolist()
    # date_list = np.array(date_list, dtype='datetime64')

    # for d in date_list:
    #     if is_tradedate(d):
    #         update_date(d)
    #         update_tick(d)
    #     else:
    #         continue
    # dates = np.load(ROOT/"axis"/"dates.npy", allow_pickle=True)
    # ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)
    # print(len(dates),len([t for t in ticks if t!='']))

    # start_dt = '2008-01-01'     
    # end_dt = '2026-06-30'
    # sql_axis = f'''select
    #                 C.SecuCode as "tick",
    #                 A.TradingDay as "date"
    #             from QT_StockPerformance A
    #             left join SecuMain C
    #             on A.InnerCode = C.InnerCode
    #             where A.TradingDay between '{start_dt}' and '{end_dt}'
    #                 and C.SecuMarket in (83,90)
    #                 and C.SecuCategory=1
    #             union all
    #             select
    #                 C.SecuCode as "tick",
    #                 B.TradingDay as "date"
    #             from LC_STIBPerformance B
    #             left join SecuMain C
    #             on B.InnerCode = C.InnerCode
    #             where B.TradingDay between '{start_dt}' and '{end_dt}'s
    #                 and C.SecuMarket in (83,90)
    #                 and C.SecuCategory=1
    #         '''
    # axis = pl.read_database(sql_axis, JY_CONN).sort('tick','date')

    # tmp = axis['tick'].unique().sort().to_numpy()
    # print(len(tmp))
    
