from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np

if __package__:
    from ..config import ROOT
    from ..utils import asof
    from .reset_axis import (
        ensure_axis_capacity,
        insert_matrix_date_rows,
        load_axes,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateData.config import ROOT
    from v2.UpdateData.utils import asof
    from v2.UpdateData.axis.reset_axis import (
        ensure_axis_capacity,
        insert_matrix_date_rows,
        load_axes,
    )


def calendar_date(date=None):
    if date is None:
        return datetime.now(ZoneInfo("Asia/Shanghai")).date()
    return asof(date).date()


def is_tradedate(date=None):
    return bool(
        xcals.get_calendar("XSHG").is_session(str(calendar_date(date)))
    )


def is_last_tradedate_of_year(date=None):
    today = calendar_date(date)
    sessions = xcals.get_calendar("XSHG").sessions_in_range(
        str(today), f"{today.year}-12-31"
    )
    return bool(len(sessions) and sessions[-1].date() == today)


def update_date(date=None, root=ROOT):
    """Insert a date in sorted order and keep every matrix row aligned."""
    value = calendar_date(date)
    ensure_axis_capacity(root, min_date_free=1)
    dates_path, _, dates, ticks = load_axes(root)
    n_valid = int(np.count_nonzero(~np.isnat(dates)))
    valid_dates = dates[:n_valid].astype("datetime64[D]")
    target = np.datetime64(value, "D")
    index = int(np.searchsorted(valid_dates, target))
    if index < n_valid and valid_dates[index] == target:
        return value.strftime("%Y-%m-%d")

    if index < n_valid:
        insert_matrix_date_rows(
            root, index, n_valid, len(dates), len(ticks)
        )

    axis = np.load(dates_path, mmap_mode="r+", allow_pickle=False)
    if index < n_valid:
        axis[index + 1:n_valid + 1] = axis[index:n_valid].copy()
    axis[index] = target
    axis.flush()
    return value.strftime("%Y-%m-%d")
