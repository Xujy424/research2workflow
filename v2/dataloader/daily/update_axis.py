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

    from v2.dataloader.config import get_jy_conn


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

def is_last_tradedate(date=None):
    if date:
        today = date
    else:
        today = datetime.now( ZoneInfo("Asia/Shanghai") ).date()


def update_date(date=None):
    if date:
        today = date
    else:
        today = datetime.now( ZoneInfo("Asia/Shanghai") ).date()
        today_np = np.datetime64(today, "D")

    path = Path(ROOT) / "axis" / "dates.npy"
    dates = np.load(
        path,
        mmap_mode="r+",
        allow_pickle=False,
    )
    n_dates = int(np.count_nonzero(~np.isnat(dates)))
    valid_dates = dates[:n_dates]

    idx = np.searchsorted(valid_dates, today_np)
    if idx < n_dates and valid_dates[idx] == today_np:
        print(f"{today} 已存在于 dates.npy")
        return 
    dates[n_dates] = today_np
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
    new_ticks = get_all_ticks(date)
    ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)

    is_valid = lambda x: x != ""
    n_valid = int(np.count_nonzero(is_valid(ticks)))
    valid_ticks = ticks[:n_valid]
    if n_valid>=new_ticks:
        return False
    else:
        ipos = sorted([t for t in new_ticks if t not in valid_ticks])
        ticks[n_valid:len(ipos)] = ipos
        return True
        

def _ensure_axis_reverse(path, reserve, is_valid, fill_value):
    arr = np.load(path, allow_pickle=True)

    n_valid = int(np.count_nonzero(is_valid(arr)))
    free = arr.size - n_valid
    if free >= reserve:
        return False

    new_arr = np.full(
        n_valid + reserve,
        fill_value,
        dtype=arr.dtype,
    )
    new_arr[:n_valid] = arr[:n_valid]

    np.save(path, new_arr, allow_pickle=True)
    return True

def reset_index():
    axis_dir = ROOT / "axis"
    date_expanded = _ensure_axis_reverse(
        axis_dir / "dates.npy",
        DATE_RESERVE,
        lambda x: ~np.isnat(x),
        np.datetime64("NaT"),
    )
    tick_expanded = _ensure_axis_reverse(
        axis_dir / "ticks.npy",
        TICK_RESERVE,
        lambda x: x != "",
        "",
    )
    return date_expanded, tick_expanded


if __name__ == '__main__':
    dates = np.load(ROOT/"axis"/"dates.npy", allow_pickle=True)
    ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)
    print(len(dates),len(ticks))
    _,_ = reset_index()
    dates = np.load(ROOT/"axis"/"dates.npy", allow_pickle=True)
    ticks = np.load(ROOT/"axis"/"ticks.npy", allow_pickle=True)
    print(dates)
    print(ticks)
    print(len(dates),len(ticks))


    # date_list = pd.date_range('2026-01-01', '2026-06-30').strftime('%Y-%m-%d').tolist()
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
