"""Point-in-time daily financial statement matrices."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__:
    from ..config import get_jy_conn
    from ..utils import asof as _date, ensure_memmap, valid_stock_ticks
else:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from v2.UpdateData.config import get_jy_conn
    from v2.UpdateData.utils import (
        asof as _date,
        ensure_memmap,
        valid_stock_ticks,
    )



STATEMENT_TABLES = {
    "income": ("LC_IncomeStatementAll", "LC_STIBIncomeState"),
    "balance": ("LC_BalanceSheetAll", "LC_STIBBalanceSheet"),
    "cashflow": ("LC_CashFlowStatementAll", "LC_STIBCashFlowState"),
}
META = {
    # Technical columns used only for filtering, point-in-time ordering,
    # deduplication, or stock-code mapping; never write them as factors.
    "id", "infopubldate", "infosource", "companycode", "enddate",
    "ifmerged", "ifadjusted", "ifcomplete", "updatetime", "inserttime",
    "jsid", "bulletintype", "aid", "finrepformat", "infosourcecode",
}
NUMERIC = {
    "bigint", "int", "smallint", "tinyint", "decimal", "numeric",
    "money", "smallmoney", "float", "real", "bit"
}
FUNDAMENTAL_LOOKBACK_YEARS = 3



def _tick_axis(ticks):
    """Return valid tick labels and their positions in the full axis."""
    valid, positions = valid_stock_ticks(ticks)
    return valid.tolist(), positions

def _is_business_field(name):
    normalized = str(name).lower().replace("_", "").replace(" ", "")
    return normalized not in META

def _columns(conn, table):
    sql = """
    SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=%(table)s
    ORDER BY ORDINAL_POSITION
    """
    out = pd.read_sql(sql, conn, params={"table": table})
    if out.empty:
        raise ValueError(f"table not found: {table}")
    return {str(a): str(b).lower() for a, b in zip(out.COLUMN_NAME, out.DATA_TYPE)}

def _shared_business_fields(frame):
    """Return business fields from the normalized, merged statement frame."""
    helper_columns = {
        "company_code", "tick", "end_date", "publish_date",
    }
    return [
        name for name in frame.columns
        if name not in helper_columns and _is_business_field(name)
    ]

def _load(conn, table, date, lookback_years=FUNDAMENTAL_LOOKBACK_YEARS):
    cols = _columns(conn, table)
    names = {x.lower(): x for x in cols}
    code = names.get("companycode")
    if not code or "infopubldate" not in names or "enddate" not in names:
        return pd.DataFrame()
    asof = _date(date)
    start = asof - pd.DateOffset(years=int(lookback_years))
    bulletin_filter = "AND f.BulletinType IN (20, 30)" if "bulletintype" in names else ""
    sql = f"""
        SELECT f.*, s.SecuCode AS tick
        FROM dbo.{table} AS f
        LEFT JOIN dbo.SecuMain AS s
            ON s.CompanyCode = f.CompanyCode
        WHERE InfoPublDate >= %(start_date)s
            AND InfoPublDate <= %(end_date)s
            AND f.IfMerged = 1
            AND f.IfAdjusted = 2
            AND f.IfComplete = 1
            AND s.SecuCategory = 1
            AND s.SecuMarket IN (83, 90)
            {bulletin_filter}
    """
    out = pd.read_sql(
        sql,
        conn,
        params={"start_date": start.date(), "end_date": asof.date()},
    ).rename(columns={
        'OtherNonCurLia':'OtherNonCurrentLiability',
        'TotalNonCurLia':'TotalNonCurrentLiability'
    })
    if out.empty:
        return out
    out = out.rename(columns={
        code: "company_code",
        names["enddate"]: "end_date",
        names["infopubldate"]: "publish_date",
    })
    out["tick"] = out["tick"].astype("string").str.zfill(6)
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
    return out

def _save(root, name, field, dates, dt, values):
    # Each statement now has one point-in-time latest matrix; no extra
    # ``latest`` directory is needed.
    path = Path(root) / "fundamental" / name / f"{field}.bin"
    a = ensure_memmap(
        path, (len(dates), len(values)), dtype=np.float32
    )
    a[dt] = values; a.flush()

def _single_quarter_values(frame, field, valid_ticks, ticks, tick_positions):
    """Derive a quarter from the latest report, without field-level fallback.

    Field-level ffill is correct for accumulated values, but not for a
    difference: an omitted field in the latest report must stay unavailable,
    rather than silently using a prior-year annual-report value.
    """
    reports = frame[["tick", "end_date", "publish_date", field]].copy()
    reports = reports.dropna(subset=["tick", "end_date"]).sort_values(
        ["tick", "end_date", "publish_date"]
    ).drop_duplicates(["tick", "end_date"], keep="last")

    current = reports.drop_duplicates("tick", keep="last").copy()
    if current.empty:
        return np.full(len(ticks), np.nan)
    current[field] = pd.to_numeric(current[field], errors="coerce")
    current["year"] = current["end_date"].dt.year
    current["month"] = current["end_date"].dt.month
    current["prior_month"] = current["month"].map({3: np.nan, 6: 3, 9: 6, 12: 9})

    reports[field] = pd.to_numeric(reports[field], errors="coerce")

    prior = reports.assign(
        year=reports["end_date"].dt.year, 
        month=reports["end_date"].dt.month
    )
    prior = prior[["tick", "year", "month", field]].drop_duplicates(
        ["tick", "year", "month"], keep="last"
    ).rename(columns={"month": "prior_month", field: "prior_value"})

    current = current.merge(prior, on=["tick", "year", "prior_month"], how="left")
    current["quarter_value"] = np.where(
        current["month"].eq(3), current[field], current[field] - current["prior_value"]
    )
    current.loc[~current["month"].isin([3, 6, 9, 12]), "quarter_value"] = np.nan

    aligned = current.set_index("tick")["quarter_value"].reindex(valid_ticks).to_numpy(np.float32)
    values = np.full(len(ticks), np.nan)
    values[tick_positions] = aligned
    return values

def _ttm_values(frame, field, valid_ticks, ticks, tick_positions):
    """Calculate point-in-time TTM from cumulative period-statement values."""
    reports = frame[["tick", "end_date", "publish_date", field]].copy()
    reports = reports.dropna(subset=["tick", "end_date"]).sort_values(
        ["tick", "end_date", "publish_date"]
    ).drop_duplicates(["tick", "end_date"], keep="last")

    current = reports.drop_duplicates("tick", keep="last").copy()
    if current.empty:
        return np.full(len(ticks), np.nan)

    reports[field] = pd.to_numeric(reports[field], errors="coerce")

    current[field] = pd.to_numeric(current[field], errors="coerce")
    current["year"] = current["end_date"].dt.year
    current["month"] = current["end_date"].dt.month

    prior = reports.assign(
        year=reports["end_date"].dt.year,
        month=reports["end_date"].dt.month,
    )[["tick", "year", "month", field]].drop_duplicates(
        ["tick", "year", "month"], keep="last"
    )
    prior_fy = prior[prior["month"].eq(12)][["tick", "year", field]].rename(
        columns={field: "prior_fy"}
    )
    prior_fy["year"] += 1
    prior_same = prior.rename(columns={field: "prior_same"}).copy()
    prior_same["year"] += 1

    current = current.merge(prior_fy, on=["tick", "year"], how="left")
    current = current.merge(prior_same, on=["tick", "year", "month"], how="left")
    current["ttm_value"] = np.where(
        current["month"].eq(12),
        current[field],
        current[field] + current["prior_fy"] - current["prior_same"],
    )
    current.loc[~current["month"].isin([3, 6, 9, 12]), "ttm_value"] = np.nan
    aligned = current.set_index("tick")["ttm_value"].reindex(valid_ticks).to_numpy(np.float32)
    values = np.full(len(ticks), np.nan)
    values[tick_positions] = aligned
    return values

def update_fundamental(date, dates, ticks, conn=None, root=None):
    conn = conn or get_jy_conn()
    if root is None: raise ValueError("root is required")

    date = _date(date)
    dt = int(np.searchsorted(np.asarray(dates, dtype="datetime64[D]"), np.datetime64(date.date())))
    if dt >= len(dates) or np.datetime64(dates[dt], "D") != np.datetime64(date.date()):
        raise ValueError(f"{date.date()} is not in dates")
    
    valid_ticks, tick_positions = _tick_axis(ticks)
    
    result = {}
    for statement, tables in STATEMENT_TABLES.items():
        frames = [x for x in (_load(conn, table, date) for table in tables) if not x.empty]
        if not frames:
            continue
        # Main-board and STAR-market tables have slightly different schemas.
        # Keep only their shared columns before concatenation so every saved
        # field has the same definition and coverage in both markets.
        common = set.intersection(*(set(x.columns) for x in frames))
        columns = [name for name in frames[0].columns if name in common]
        frame = pd.concat([x.loc[:, columns] for x in frames], ignore_index=True)
        if frame.empty: continue

        frame = frame.dropna(subset=["tick"]).sort_values(["tick", "end_date", "publish_date"])
        fields = _shared_business_fields(frame)

        # A preliminary/flash report can be the newest event but omit many
        # fields. Select the latest *non-null* value per field and stock,
        # which is the point-in-time equivalent of a field-level ffill.
        # All rows have already been restricted to publish_date <= date.
        frame = frame.sort_values(["tick", "publish_date", "end_date"])
        for field in tqdm(fields):
            field_frame = frame.dropna(subset=[field]).drop_duplicates("tick", keep="last")  # 所有tick内最新非空数据
            valid_values = pd.to_numeric(
                field_frame.set_index("tick")[field], errors="coerce"
            ).reindex(valid_ticks).to_numpy(np.float32)
            values = np.full(len(ticks), np.nan)
            values[tick_positions] = valid_values

            is_period_statement = statement in {"income", "cashflow"}
            output_name = f"{statement}_accum" if is_period_statement else statement
            _save(root, output_name, field, dates, dt, values)
            
            result[f"{output_name}/{field}"] = values
            if is_period_statement:
                quarterly = _single_quarter_values(
                    frame, field, valid_ticks, ticks, tick_positions
                )
                quarterly_name = f"{statement}_quartly"
                _save(root, quarterly_name, field, dates, dt, quarterly)
                result[f"{quarterly_name}/{field}"] = quarterly

                ttm = _ttm_values(frame, field, valid_ticks, ticks, tick_positions)
                ttm_name = f"{statement}_ttm"
                _save(root, ttm_name, field, dates, dt, ttm)
                result[f"{ttm_name}/{field}"] = ttm
                
    return result

__all__ = ["update_fundamental", "STATEMENT_TABLES"]



if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/stock_ticks.npy', allow_pickle=False)
    conn = get_jy_conn()
    root = Path('D:/data')

    res = update_fundamental(
        date, dates, ticks, conn, root
    )
    print(res)
