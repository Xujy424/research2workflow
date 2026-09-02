"""Write daily StarRocks intraday summaries as T x N float32 matrices."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import polars as pl

if __package__:
    from ..utils import (
        asof as _asof,
        date_index as _date_index,
        ensure_memmap as _ensure_memmap,
        valid_stock_ticks,
    )
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.utils import (
        asof as _asof,
        date_index as _date_index,
        ensure_memmap as _ensure_memmap,
        valid_stock_ticks,
    )


DTYPE = np.float32

TABLES = (
    "StockPriceInfoCloseAverage",
    "StockPriceInfoTHalf",
)


def _snake_case(name):
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    name = re.sub(r"([a-z])(\d+)", r"\1_\2", name)
    return re.sub(r"(\d+)_([a-z])", r"\1\2", name)


def _read_table(conn, table, date):
    sql = f"""
        SELECT *
        FROM {table}
        WHERE FDate = '{date:%Y-%m-%d}'
    """
    frame = pl.read_database(sql, conn, infer_schema_length=None)
    numeric_fields = [
        name for name, dtype in frame.schema.items()
        if name not in {"FDate", "SecCode"} and dtype.is_numeric()
    ]
    frame = frame.select(
        pl.col("SecCode").cast(pl.String).str.zfill(6).alias("tick"),
        *[
            pl.col(field).cast(pl.Float64, strict=False)
            for field in numeric_fields
        ],
    ).unique(maintain_order=True)

    duplicates = (
        frame.group_by("tick")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        examples = duplicates["tick"].head(10).to_list()
        raise ValueError(
            f"{table} has conflicting duplicate rows on {date:%Y-%m-%d}; "
            f"example ticks: {examples}"
        )
    return frame, numeric_fields


def _align_to_axis(aligned, source_field, axis_size):
    result = np.full(axis_size, np.nan, dtype=DTYPE)
    if aligned.is_empty():
        return result
    result[aligned["position"].to_numpy()] = (
        aligned[source_field].to_numpy().astype(DTYPE, copy=False)
    )
    return result


def _write_row(root, field, dates, ticks, date_position, values):
    path = Path(root) / "highfreq" / f"{field}.bin"
    matrix = _ensure_memmap(
        path, (len(dates), len(ticks)), dtype=DTYPE
    )
    matrix[date_position] = values
    matrix.flush()
    del matrix


def update_highfreq(date, dates, ticks, conn, root):
    """Update all 19 CloseAverage and T-half fields for one trade date.

    Root follows the stock updater convention and normally means ROOT / stock.
    Source fields already use yuan, shares and yuan, so no /1000 is applied.
    """
    if conn is None:
        raise ValueError("StarRocks connection is required")
    if root is None:
        raise ValueError("stock root is required")

    asof = _asof(date)
    date_position = _date_index(asof, dates)
    valid_ticks, positions = valid_stock_ticks(ticks)
    axis = pl.DataFrame({"tick": valid_ticks, "position": positions})
    row_counts = {}
    field_count = 0
    for table in TABLES:
        frame, fields = _read_table(conn, table, asof)
        row_counts[table] = frame.height
        field_count += len(fields)
        aligned = axis.join(frame, on="tick", how="left").sort("position")
        for source_field in fields:
            values = _align_to_axis(aligned, source_field, len(ticks))
            _write_row(
                root,
                _snake_case(source_field),
                dates,
                ticks,
                date_position,
                values,
            )

    return {
        "date": asof.strftime("%Y-%m-%d"),
        "rows": row_counts,
        "fields": field_count,
    }


__all__ = [
    "TABLES",
    "update_highfreq",
]



