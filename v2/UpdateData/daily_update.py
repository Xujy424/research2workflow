from __future__ import annotations

from datetime import date as date_type, timedelta
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .axis import (
        init_axis,
        is_last_tradedate_of_year,
        is_tradedate,
        reset_axis,
        update_date,
        update_stockticks,
    )
    from .config import (
        L2DATA_PATH,
        ROOT,
        get_jy_conn,
        get_str_engine,
        get_zyyx_conn,
    )
    from .level2.generate_bar_snapshot_v2 import update_snapshot
    from .level2.get_l2data import autoload_l2data
    from .level2.preprocess_l2data import update_l2_basic
    from .stock.update_barra import update_barra
    from .stock.update_basic import update_basic, update_tradable
    from .stock.update_dividend import update_dividend
    from .stock.update_essentials import (
        update_d_essentials,
        update_m_essentials,
    )
    from .stock.update_fundamental import update_fundamental
    from .stock.update_index import update_index
    from .stock.update_industry import update_industry, update_sector
    from .stock.update_moneyflow import update_d_moneyflow
    from .stock.update_zyyx import update_zyyx
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.axis import (
        init_axis,
        is_last_tradedate_of_year,
        is_tradedate,
        reset_axis,
        update_date,
        update_stockticks,
    )
    from v2.UpdateData.config import (
        L2DATA_PATH,
        ROOT,
        get_jy_conn,
        get_str_engine,
        get_zyyx_conn,
    )
    from v2.UpdateData.level2.generate_bar_snapshot_v2 import update_snapshot
    from v2.UpdateData.level2.get_l2data import autoload_l2data
    from v2.UpdateData.level2.preprocess_l2data import update_l2_basic
    from v2.UpdateData.stock.update_barra import update_barra
    from v2.UpdateData.stock.update_basic import update_basic, update_tradable
    from v2.UpdateData.stock.update_dividend import update_dividend
    from v2.UpdateData.stock.update_essentials import (
        update_d_essentials,
        update_m_essentials,
    )
    from v2.UpdateData.stock.update_fundamental import update_fundamental
    from v2.UpdateData.stock.update_index import update_index
    from v2.UpdateData.stock.update_industry import (
        update_industry,
        update_sector,
    )
    from v2.UpdateData.stock.update_moneyflow import update_d_moneyflow
    from v2.UpdateData.stock.update_zyyx import update_zyyx


def _close_connection(connection):
    close = getattr(connection, "close", None)
    if callable(close):
        close()
        return
    dispose = getattr(connection, "dispose", None)
    if callable(dispose):
        dispose()


def update_data(
    root,
    date=None,
    *,
    jy_conn=None,
    zyyx_conn=None,
    str_conn=None,
    update_level2=True,
):
    root = Path(root)
    stock_root = root / "stock"
    stock_root.mkdir(parents=True, exist_ok=True)
    dates_path, ticks_path = init_axis(root)
    print(f'日期:{date}')
    if not is_tradedate(date):
        return {"status": "skipped", "reason": "non-trading day"}

    own_jy = jy_conn is None
    own_zyyx = zyyx_conn is None
    own_str = str_conn is None
    jy_conn = jy_conn or get_jy_conn()
    zyyx_conn = zyyx_conn or get_zyyx_conn()
    str_conn = str_conn or get_str_engine()

    try:
        date = update_date(date, root)
        update_stockticks(date, root, jy_conn)
        dates = np.load(dates_path, allow_pickle=False)
        stock_ticks = np.load(ticks_path, allow_pickle=False)
        print('日期及股票索引更新结束.')

        update_d_essentials(date, dates, stock_ticks, jy_conn, stock_root)
        print('日行情更新结束.')
        update_m_essentials(date, dates, stock_ticks, str_conn, stock_root)
        print('分钟行情更新结束.')
        update_basic(date, dates, stock_ticks, jy_conn, stock_root)
        update_tradable(date, dates, stock_ticks, jy_conn, stock_root)
        print('基础信息更新结束.')
        update_industry(date, dates, stock_ticks, jy_conn, stock_root)
        update_sector(date, dates, stock_ticks, jy_conn, stock_root)
        print('行业与板块分类更新结束.')
        update_index(date, dates, stock_ticks, jy_conn, stock_root)
        print('指数成分更新结束.')
        update_zyyx(date, dates, stock_ticks, zyyx_conn, stock_root)
        print('朝阳永续数据更新结束.')
        update_fundamental(date, dates, stock_ticks, jy_conn, stock_root)
        update_dividend(date, dates, stock_ticks, jy_conn, stock_root)
        print('基本面数据更新结束.')
        update_barra(date, dates, stock_ticks, jy_conn, stock_root)
        print('Barra十因子更新结束.')

        if update_level2:
            compact_date = date.replace("-", "")
            autoload_l2data(compact_date)
            print('level2原始表加载结束.')
            update_l2_basic(L2DATA_PATH, compact_date)
            print('level2处理表更新结束.')
            update_snapshot(L2DATA_PATH, compact_date, "1m", 10)
            print('level2快照表更新结束.')
            update_d_moneyflow(
                stock_root, dates, date, stock_ticks, l2_root=L2DATA_PATH
            )
            print('资金流分类更新结束.')

        print()
        resized = (
            reset_axis(root)
            if is_last_tradedate_of_year(date)
            else None
        )
        return {
            "status": "updated",
            "date": date,
            "axis_resized": bool(resized and resized.changed),
        }
    
    finally:
        if own_jy:
            _close_connection(jy_conn)
        if own_zyyx:
            _close_connection(zyyx_conn)
        if own_str:
            _close_connection(str_conn)


def update_history(
    root,
    start_date="2010-01-01",
    end_date="2026-07-31",
):
    """Build non-Level-2 data sequentially over a historical date range."""
    start = date_type.fromisoformat(str(start_date))
    end = date_type.fromisoformat(str(end_date))
    if start > end:
        raise ValueError("start_date must not be later than end_date")

    jy_conn = None
    zyyx_conn = None
    str_conn = None
    try:
        jy_conn = get_jy_conn()
        zyyx_conn = get_zyyx_conn()
        str_conn = get_str_engine()

        current = start
        while current <= end:
            if is_tradedate(current):
                date_text = current.isoformat()
                print(f"updating {date_text} ...", flush=True)
                result = update_data(
                    root,
                    date_text,
                    jy_conn=jy_conn,
                    zyyx_conn=zyyx_conn,
                    str_conn=str_conn,
                    update_level2=False,
                )
                print(result, flush=True)
            current += timedelta(days=1)
    finally:
        for connection in (jy_conn, zyyx_conn, str_conn):
            if connection is not None:
                _close_connection(connection)


def update_history_l2(
    root,
    start_date="2010-01-01",
    end_date="2026-07-31",
):
    """Download and build only historical Level-2 data and moneyflow."""
    start = date_type.fromisoformat(str(start_date))
    end = date_type.fromisoformat(str(end_date))
    if start > end:
        raise ValueError("start_date must not be later than end_date")

    root = Path(root)
    stock_root = root / "stock"
    stock_root.mkdir(parents=True, exist_ok=True)
    dates_path, ticks_path = init_axis(root)
    dates = np.load(dates_path, allow_pickle=False)
    stock_ticks = np.load(ticks_path, allow_pickle=False)
    valid_dates = {
        str(value)
        for value in dates[~np.isnat(dates)].astype("datetime64[D]")
    }
    if not valid_dates or not np.any(stock_ticks != ""):
        raise ValueError(
            "build the date and stock axes with update_history() before L2"
        )

    current = start
    while current <= end:
        if is_tradedate(current):
            date_text = current.isoformat()
            if date_text not in valid_dates:
                raise ValueError(
                    f"{date_text} is missing from dates.npy; "
                    "run update_history() first"
                )

            compact_date = date_text.replace("-", "")
            print(f"updating L2 {date_text} ...", flush=True)
            autoload_l2data(compact_date)
            print('level2原始表加载结束.')
            update_l2_basic(L2DATA_PATH, compact_date)
            print('level2处理表更新结束.')
            update_snapshot(L2DATA_PATH, compact_date, "1m", 10)
            print('level2快照表更新结束.')
            update_d_moneyflow(
                stock_root,
                dates,
                date_text,
                stock_ticks,
                l2_root=L2DATA_PATH,
            )
            print('资金流分类更新结束.')
            print(f"L2 {date_text} updated", flush=True)
            print()
        current += timedelta(days=1)


if __name__ == "__main__":
    update_history(
        root=ROOT,
        start_date="2010-01-01",
        end_date="2026-07-31",
    )
    # update_history_l2(
    #     root=ROOT,
    #     start_date="2010-01-01",
    #     end_date="2026-07-31",
    # )

    # update_data(
    #     root=ROOT,
    #     date=None,
    #     update_level2=True,
    # )

