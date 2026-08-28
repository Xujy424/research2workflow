from __future__ import annotations

from datetime import date as date_type, timedelta
from pathlib import Path
import sys
import warnings

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


def _run_step(label, function, *args, **kwargs):
    """Run one update step, print one result line, and re-raise failures."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = function(*args, **kwargs)
    except Exception as exc:
        print(
            f"[FAILED] {label}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    print(f"[OK] {label}", flush=True)
    return result


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
    if not is_tradedate(date):
        return {"status": "skipped", "reason": "non-trading day"}

    print(f"date: {date}", flush=True)
    own_jy = jy_conn is None
    own_zyyx = zyyx_conn is None
    own_str = str_conn is None

    try:
        if jy_conn is None:
            jy_conn = get_jy_conn()
        if zyyx_conn is None:
            zyyx_conn = get_zyyx_conn()
        if str_conn is None:
            str_conn = get_str_engine()

        # date = _run_step("date axis", update_date, date, root)
        # _run_step("stock axis", update_stockticks, date, root, jy_conn)
        dates = np.load(dates_path, allow_pickle=False)
        stock_ticks = np.load(ticks_path, allow_pickle=False)

        # _run_step(
        #     "daily essentials",
        #     update_d_essentials,
        #     date, dates, stock_ticks, jy_conn, stock_root,
        # )
        _run_step(
            "minute essentials",
            update_m_essentials,
            date, dates, stock_ticks, str_conn, stock_root,
        )
        # _run_step(
        #     "basic and tradable",
        #     lambda: (
        #         update_basic(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #         update_tradable(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #     ),
        # )
        # _run_step(
        #     "industry and sector",
        #     lambda: (
        #         update_industry(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #         update_sector(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #     ),
        # )
        # _run_step(
        #     "index constituents",
        #     update_index,
        #     date, dates, stock_ticks, jy_conn, stock_root,
        # )
        # _run_step(
        #     "ZYYX consensus",
        #     update_zyyx,
        #     date,
        #     dates,
        #     stock_ticks,
        #     zyyx_conn,
        #     stock_root / "zyyx",
        # )
        # _run_step(
        #     "fundamental and dividend",
        #     lambda: (
        #         update_fundamental(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #         update_dividend(
        #             date, dates, stock_ticks, jy_conn, stock_root
        #         ),
        #     ),
        # )
        # _run_step(
        #     "Barra factors",
        #     update_barra,
        #     date, dates, stock_ticks, jy_conn, stock_root,
        # )

        if update_level2:
            compact_date = date.replace("-", "")
            _run_step("Level-2 download", autoload_l2data, compact_date)
            _run_step(
                "Level-2 preprocessing",
                update_l2_basic,
                L2DATA_PATH,
                compact_date,
            )
            _run_step(
                "Level-2 snapshots",
                update_snapshot,
                L2DATA_PATH,
                compact_date,
                "1m",
                10,
            )
            _run_step(
                "moneyflow",
                update_d_moneyflow,
                stock_root,
                dates,
                date,
                stock_ticks,
                l2_root=L2DATA_PATH,
            )

        resized = None
        if is_last_tradedate_of_year(date):
            resized = _run_step("annual axis resize", reset_axis, root)

        return {
            "status": "updated",
            "date": date,
            "axis_resized": bool(resized and resized.changed),
        }
    finally:
        # if own_jy and jy_conn is not None:
        #     _close_connection(jy_conn)
        # if own_zyyx and zyyx_conn is not None:
        #     _close_connection(zyyx_conn)
        # if own_str and str_conn is not None:
        #     _close_connection(str_conn)
        return
        

def update_history(
    root,
    start_date="2010-01-01",
    end_date="2026-07-31",
):
    """Build non-Level-2 data sequentially and stop on the first failure."""
    start = date_type.fromisoformat(str(start_date))
    end = date_type.fromisoformat(str(end_date))
    if start > end:
        raise ValueError("start_date must not be later than end_date")

    jy_conn = None
    zyyx_conn = None
    str_conn = None
    current = start
    try:
        jy_conn = get_jy_conn()
        zyyx_conn = get_zyyx_conn()
        str_conn = get_str_engine()

        while current <= end:
            if is_tradedate(current):
                update_data(
                    root,
                    current.isoformat(),
                    jy_conn=jy_conn,
                    zyyx_conn=zyyx_conn,
                    str_conn=str_conn,
                    update_level2=False,
                )
            current += timedelta(days=1)
    except Exception:
        print(
            f"[STOPPED] historical update at {current.isoformat()}",
            flush=True,
        )
        raise
    finally:
        for connection in (jy_conn, zyyx_conn, str_conn):
            if connection is not None:
                _close_connection(connection)


def update_history_l2(
    root,
    start_date="2010-01-01",
    end_date="2026-07-31",
):
    """Build only historical Level-2 data and stop on the first failure."""
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
    try:
        while current <= end:
            if is_tradedate(current):
                date_text = current.isoformat()
                if date_text not in valid_dates:
                    raise ValueError(
                        f"{date_text} is missing from dates.npy; "
                        "run update_history() first"
                    )

                compact_date = date_text.replace("-", "")
                print(f"L2 date: {date_text}", flush=True)
                _run_step(
                    "Level-2 download", autoload_l2data, compact_date
                )
                _run_step(
                    "Level-2 preprocessing",
                    update_l2_basic,
                    L2DATA_PATH,
                    compact_date,
                )
                _run_step(
                    "Level-2 snapshots",
                    update_snapshot,
                    L2DATA_PATH,
                    compact_date,
                    "1m",
                    10,
                )
                _run_step(
                    "moneyflow",
                    update_d_moneyflow,
                    stock_root,
                    dates,
                    date_text,
                    stock_ticks,
                    l2_root=L2DATA_PATH,
                )
            current += timedelta(days=1)
    except Exception:
        print(
            f"[STOPPED] historical Level-2 update at "
            f"{current.isoformat()}",
            flush=True,
        )
        raise

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

