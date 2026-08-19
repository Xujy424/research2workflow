from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

if __package__:
    from ..config import ROOT, get_jy_conn
    from .reset_axis import ensure_axis_capacity, load_axes
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.config import ROOT, get_jy_conn
    from v2.UpdateData.axis.reset_axis import (
        ensure_axis_capacity,
        load_axes,
    )


def get_all_ticks(date, conn=None):
    owns_connection = conn is None
    connection = conn or get_jy_conn()
    try:
        sql = f"""
            select C.SecuCode as tick
            from QT_StockPerformance A
            left join SecuMain C on A.InnerCode = C.InnerCode
            where A.TradingDay = '{date}'
              and C.SecuMarket in (83,90)
              and C.SecuCategory=1
            union
            select C.SecuCode as tick
            from LC_STIBPerformance B
            left join SecuMain C on B.InnerCode = C.InnerCode
            where B.TradingDay = '{date}'
              and C.SecuMarket in (83,90)
              and C.SecuCategory=1
        """
        frame = pl.read_database(sql, connection)
    finally:
        if owns_connection:
            connection.close()
    if frame.is_empty():
        return np.asarray([], dtype="<U6")
    return np.asarray(
        sorted({
            str(tick).strip().zfill(6)
            for tick in frame["tick"].drop_nulls().to_list()
            if str(tick).strip()
        }),
        dtype="<U6",
    )


def update_stockticks(date, root=ROOT, conn=None):
    """Append newly observed stocks without reordering existing columns."""
    current = get_all_ticks(date, conn)
    _, ticks_path, _, ticks = load_axes(root)
    n_valid = int(np.count_nonzero(ticks != ""))
    known = set(ticks[:n_valid])
    new_ticks = [tick for tick in current if tick not in known]
    if not new_ticks:
        return []

    ensure_axis_capacity(root, min_tick_free=len(new_ticks))
    ticks = np.load(ticks_path, mmap_mode="r+", allow_pickle=False)
    n_valid = int(np.count_nonzero(ticks != ""))
    ticks[n_valid:n_valid + len(new_ticks)] = new_ticks
    ticks.flush()
    return new_ticks
