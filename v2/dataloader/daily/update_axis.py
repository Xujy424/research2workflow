import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import exchange_calendars as xcals


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

def do_update(date=None):
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

def reset_index():
    dates = np.load(
        Path(ROOT) / "axis" / "dates.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    ticks = np.load(
        Path(ROOT) / "axis" / "ticks.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    n_valid_dates = int(np.count_nonzero(~np.isnat(dates)))
    n_valid_ticks = int(np.count_nonzero(ticks!= ""))

    need_date_expand = dates.size()-n_valid_dates < DATE_RESERVE
    need_tick_expand = ticks.size()-n_valid_ticks < TICK_RESERVE

    new_T = (
        n_valid_dates + DATE_RESERVE
        if need_date_expand
        else len(dates)
    )
    new_N = (
        n_valid_ticks + TICK_RESERVE
        if need_tick_expand
        else len(ticks)
    )

    

    new_arr = np.full((new_T, new_N), np.nan, dtype=np.float32)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    if isinstance(self.arr, np.memmap):
        self.arr.flush()

    self.arr = new_arr
    self.T, self.N = new_arr.shape