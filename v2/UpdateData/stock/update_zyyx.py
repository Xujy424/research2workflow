"""Daily incremental updates for ZYYX consensus data."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    # Normal package import, e.g. ``python -m v2.UpdateData.stock.update_zyyx``.
    from ..config import get_zyyx_conn
else:
    # IDEs may execute this file directly. Add the repository root so that
    # ``v2.UpdateData.config`` (and its own relative imports) remains valid.
    _repository_root = Path(__file__).resolve().parents[3]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))
    from v2.UpdateData.config import get_zyyx_conn


TABLES = {
    "con_forecast": "con_forecast_stk",
    "con_forecast_roll": "con_forecast_roll_stk",
    "con_rating": "con_rating_stk",
    "con_target_price": "con_target_price_stk",
    "con_forecast_eq": "con_forecast_stk_eq",
    "con_forecast_wgt": "con_forecast_stk_wgt",
    "con_rating_eq": "con_rating_stk_eq",
    "con_rating_wgt": "con_rating_stk_wgt",
    "con_target_price_eq": "con_target_price_stk_eq",
    "con_target_price_wgt": "con_target_price_stk_wgt",
}
META_FIELDS = {
    "id",
    "stock_code",
    "stock_name",
    "con_date",
    "entrytime",
    "updatetime",
    "tmstamp",
    "con_year",
}
FORECAST_YEARS = 4


def _date(value):
    value = pd.Timestamp(value).normalize()
    if pd.isna(value):
        raise ValueError("invalid date")
    return value


def _date_index(date, dates):
    target = np.datetime64(_date(date).date())
    axis = np.asarray(dates, dtype="datetime64[D]")
    index = int(np.searchsorted(axis, target))
    if index >= len(axis) or axis[index] != target:
        raise ValueError(f"{date} is not present in the configured date axis")
    return index


def _tick_axis(ticks):
    """Normalize non-empty ticks without moving their positions on the axis."""
    normalized = []
    positions = []
    for position, tick in enumerate(ticks):
        if tick is None or pd.isna(tick) or str(tick).strip() == "":
            continue
        normalized.append(str(tick).strip().zfill(6))
        positions.append(position)
    return normalized, np.asarray(positions, dtype=int)


def _table_columns(conn, table):
    sql = """
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = %(table)s
        ORDER BY ORDINAL_POSITION
    """
    columns = pd.read_sql(sql, conn, params={"table": table})
    if columns.empty:
        raise ValueError(f"table not found: {table}")
    return dict(zip(columns["COLUMN_NAME"], columns["DATA_TYPE"]))


def _business_fields(columns):
    numeric_types = {
        "bigint", "int", "smallint", "tinyint", "bit",
        "decimal", "numeric", "money", "smallmoney",
        "float", "real",
    }
    return [
        name
        for name, dtype in columns.items()
        if name.lower() not in META_FIELDS and dtype.lower() in numeric_types
    ]


def _load_daily_table(conn, table, date):
    asof = _date(date).date()
    sql = f"SELECT * FROM {table} WHERE con_date = %(date)s"
    frame = pd.read_sql(sql, conn, params={"date": asof})
    if frame.empty:
        return frame
    frame["stock_code"] = frame["stock_code"].astype("string").str.zfill(6)
    order = [
        name
        for name in ("stock_code", "con_year", "create_date", "entrytime", "id")
        if name in frame.columns
    ]
    if order:
        frame = frame.sort_values(order).drop_duplicates(
            [name for name in ("stock_code", "con_year") if name in frame.columns],
            keep="last",
        )
    return frame


def _ensure_memmap(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    size = int(np.prod(shape)) * np.dtype(np.float64).itemsize
    if not path.exists():
        with path.open("wb") as file:
            file.truncate(size)
        array = np.memmap(path, dtype=np.float64, mode="r+", shape=shape)
        array[:] = np.nan
        array.flush()
    if path.stat().st_size != size:
        raise ValueError(f"{path} size does not match axes {shape}")
    return np.memmap(path, dtype=np.float64, mode="r+", shape=shape)


def _aligned_values(frame, field, ticks, positions, axis_size):
    result = np.full(axis_size, np.nan)
    if frame.empty or field not in frame:
        return result
    values = pd.to_numeric(frame.set_index("stock_code")[field], errors="coerce")
    result[positions] = values.reindex(ticks).to_numpy(dtype=np.float64)
    return result


def _save_cross_section(root, table_name, dates, ticks, dt, values):
    shape = (len(dates), len(ticks))
    folder = Path(root) / table_name
    for field, value in values.items():
        array = _ensure_memmap(folder / f"{field}.bin", shape)
        array[dt] = np.nan
        array[dt] = value
        array.flush()


def _save_forecast(root, table_name, dates, ticks, dt, values):
    shape = (len(dates), FORECAST_YEARS, len(ticks))
    folder = Path(root) / table_name
    for field, value in values.items():
        array = _ensure_memmap(folder / f"{field}.bin", shape)
        array[dt] = np.nan
        array[dt, :, : value.shape[1]] = value
        array.flush()


def _update_forecast_table(name, date, dates, ticks, conn, root):
    """Update a con_year table as (date, 4 forecast years, stock tick)."""
    dt = _date_index(date, dates)
    valid_ticks, positions = _tick_axis(ticks)
    table = TABLES[name]
    columns = _table_columns(conn, table)
    fields = _business_fields(columns)
    frame = _load_daily_table(conn, table, date)

    asof = _date(date)
    base_year = asof.year - 1 if asof >= pd.Timestamp(f"{asof.year}-05-01") else asof.year - 2
    result = {}
    for field in fields:
        matrix = np.full((FORECAST_YEARS, len(ticks)), np.nan)
        for offset in range(FORECAST_YEARS):
            year_frame = frame[frame["con_year"] == base_year+offset]
            matrix[offset] = _aligned_values(
                year_frame, field, valid_ticks, positions, len(ticks)
            )
        result[field] = matrix

    _save_forecast(root, name, dates, ticks, dt, result)
    return result


def update_con_forecast(date, dates, ticks, conn, root):
    """Return every forecast field as (4, stock_tick), row 0 being base year."""
    return _update_forecast_table("con_forecast", date, dates, ticks, conn, root)


def _update_single_year_table(name, date, dates, ticks, conn, root):
    dt = _date_index(date, dates)
    valid_ticks, positions = _tick_axis(ticks)
    table = TABLES[name]
    columns = _table_columns(conn, table)
    fields = _business_fields(columns)
    frame = _load_daily_table(conn, table, date)
    result = {
        field: _aligned_values(frame, field, valid_ticks, positions, len(ticks))
        for field in fields
    }
    _save_cross_section(root, name, dates, ticks, dt, result)
    return result


def update_con_forecast_roll(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_forecast_roll", date, dates, ticks, conn, root
    )


def update_con_rating(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_rating", date, dates, ticks, conn, root
    )


def update_con_target_price(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_target_price", date, dates, ticks, conn, root
    )


def update_con_forecast_eq(date, dates, ticks, conn, root):
    return _update_forecast_table("con_forecast_eq", date, dates, ticks, conn, root)


def update_con_forecast_wgt(date, dates, ticks, conn, root):
    return _update_forecast_table("con_forecast_wgt", date, dates, ticks, conn, root)


def update_con_rating_eq(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_rating_eq", date, dates, ticks, conn, root
    )


def update_con_rating_wgt(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_rating_wgt", date, dates, ticks, conn, root
    )


def update_con_target_price_eq(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_target_price_eq", date, dates, ticks, conn, root
    )


def update_con_target_price_wgt(date, dates, ticks, conn, root):
    return _update_single_year_table(
        "con_target_price_wgt", date, dates, ticks, conn, root
    )


def update_zyyx(date, dates, ticks, conn=None, root=None):
    """Update all ten ZYYX consensus tables for one trading day."""
    if conn is None:
        conn = get_zyyx_conn()
    if root is None:
        raise ValueError("root is required")
    return {
        "con_forecast": update_con_forecast(
            date, dates, ticks, conn, root
        ),
        "con_forecast_roll": update_con_forecast_roll(
            date, dates, ticks, conn, root
        ),
        "con_rating": update_con_rating(
            date, dates, ticks, conn, root
        ),
        "con_target_price": update_con_target_price(
            date, dates, ticks, conn, root
        ),
        "con_forecast_eq": update_con_forecast_eq(
            date, dates, ticks, conn, root
        ),
        "con_forecast_wgt": update_con_forecast_wgt(
            date, dates, ticks, conn, root
        ),
        "con_rating_eq": update_con_rating_eq(
            date, dates, ticks, conn, root
        ),
        "con_rating_wgt": update_con_rating_wgt(
            date, dates, ticks, conn, root
        ),
        "con_target_price_eq": update_con_target_price_eq(
            date, dates, ticks, conn, root
        ),
        "con_target_price_wgt": update_con_target_price_wgt(
            date, dates, ticks, conn, root
        ),
    }


__all__ = [
    "FORECAST_YEARS",
    "TABLES",
    "update_con_forecast",
    "update_con_forecast_roll",
    "update_con_rating",
    "update_con_target_price",
    "update_con_forecast_eq",
    "update_con_forecast_wgt",
    "update_con_rating_eq",
    "update_con_rating_wgt",
    "update_con_target_price_eq",
    "update_con_target_price_wgt",
    "update_zyyx",
]



if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/ticks.npy',allow_pickle=True)
    conn = get_zyyx_conn()
    root = Path('D:/data/zyyx')

    update_zyyx(
        date, dates, ticks, conn, root
    )
