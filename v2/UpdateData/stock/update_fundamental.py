"""Point-in-time daily financial statement matrices."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__:
    from ..config import get_jy_conn
else:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from v2.UpdateData.config import get_jy_conn



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



def _date(x):
    return pd.Timestamp(x).normalize()

def _tick_axis(ticks):
    """Return valid tick labels and their positions in the full axis."""
    valid = []
    positions = []
    for pos, tick in enumerate(ticks):
        if tick is None or pd.isna(tick) or str(tick).strip() == "":
            continue
        valid.append(str(tick).strip().zfill(6))
        positions.append(pos)
    return valid, np.asarray(positions, dtype=int)

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

def _shared_business_fields(conn, tables):
    """Shared numeric SQL columns excluding technical metadata."""
    schemas = [_columns(conn, table) for table in tables]
    common = set.intersection(*(set(schema) for schema in schemas))
    return [
        name for name, dtype in schemas[0].items()
        if name in common and _is_business_field(name) and dtype in NUMERIC
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
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    size = len(dates) * len(values) * 8
    if not path.exists():
        with path.open("wb") as f: f.truncate(size)
        a = np.memmap(path, dtype=float, mode="r+", shape=(len(dates), len(values)))
        a[:] = np.nan; a.flush()
    a = np.memmap(path, dtype=float, mode="r+", shape=(len(dates), len(values)))
    a[dt] = values; a.flush()

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
        fields = _shared_business_fields(conn, tables)

        # A preliminary/flash report can be the newest event but omit many
        # fields. Select the latest *non-null* value per field and stock,
        # which is the point-in-time equivalent of a field-level ffill.
        # All rows have already been restricted to publish_date <= date.
        frame = frame.sort_values(["tick", "publish_date", "end_date"])
        for field in fields:
            field_frame = frame.dropna(subset=[field]).drop_duplicates("tick", keep="last")  # 所有tick内最新非空数据
            valid_values = pd.to_numeric(
                field_frame.set_index("tick")[field], errors="coerce"
            ).reindex(valid_ticks).to_numpy(float)
            values = np.full(len(ticks), np.nan)
            values[tick_positions] = valid_values
            _save(root, statement, field, dates, dt, values)
            result[f"{statement}/{field}"] = values
    return result

__all__ = ["update_fundamental", "STATEMENT_TABLES"]



if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/ticks.npy',allow_pickle=True)
    conn = get_jy_conn()
    root = Path('D:/data/fundamental')

    res = update_fundamental(
        date, dates, ticks, conn, root
    )
    print(res)
