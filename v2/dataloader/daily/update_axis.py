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


def get_all_ticks():
    

def update_tick()







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

    
