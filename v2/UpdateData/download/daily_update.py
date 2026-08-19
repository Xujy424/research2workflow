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

from .update_axis import *
from ..stock.update_essentials import update_d_essentials, update_m_essentials
from ..stock.update_basic import update_basic, update_tradable
from ..stock.update_industry import update_industry, update_sector
from ..stock.update_index import update_index
from ..stock.update_zyyx import update_zyyx
from ..stock.update_fundamental import update_fundamental
from ..stock.update_dividend import update_dividend
from ..stock.update_barra import update_barra
from ..level2.get_l2data import autoload_l2data
from ..level2.preprocess_l2data import update_l2_basic
from ..level2.generate_bar_snapshot_v2 import update_snapshot
from ..stock.update_moneyflow import update_d_moneyflow

if __package__:
    from ..config import (
        get_jy_conn, get_zyyx_conn, get_str_engine, 
        CIFTABLE_PATTERNS, cifs, L2DATA_PATH, ROOT
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from v2.UpdateData.config import (
        get_jy_conn, get_zyyx_conn, get_str_engine, 
        CIFTABLE_PATTERNS, cifs, L2DATA_PATH, ROOT
    )


jy_conn = get_jy_conn()
zyyx_conn = get_zyyx_conn()
str_conn = get_str_engine()



def update_data(root, date=None):
    dates_path, ticks_path = init_axis(root)

    is_tradeday = is_tradedate(date)
    if not is_tradeday:
        return
    else:
        date = update_date(date)
        update_stockticks(date)

        dates = np.load(dates_path, allow_pickle=False)
        stock_ticks = np.load(ticks_path, allow_pickle=False)

        update_d_essentials(date, dates, stock_ticks, jy_conn, root)
        update_m_essentials(date, dates, stock_ticks, str_conn, root)

        update_basic(date, dates, stock_ticks, jy_conn, root)
        update_tradable(date, dates, stock_ticks, jy_conn, root)

        update_industry(date, dates, ticks, jy_conn, root)
        update_sector(date, dates, ticks, jy_conn, root)

        update_index(date, dates, ticks, jy_conn, root)

        update_zyyx(date, dates, ticks, zyyx_conn, root)

        update_fundamental(date, dates, ticks, jy_conn, root)
        update_dividend(date, dates, ticks, jy_conn, root)

        update_barra(date, dates, ticks, jy_conn, root)

        autoload_l2data(date.replace('-', ''))
        update_l2_basic(L2DATA_PATH, date.replace('-', ''))
        update_snapshot(L2DATA_PATH, date.replace('-', ''), '1m', 10)

        update_d_moneyflow(root, dates, date, ticks)


        is_last_tradeday = is_last_tradedate_of_year(date)
        if is_last_tradeday:
            date_n_valid, date_old_len, tick_n_valid, tick_old_len = reset_axis(root)
            reset_field_axis(root, date_n_valid, tick_n_valid, date_old_len, tick_old_len, dim=241)

    return
