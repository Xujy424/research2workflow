"""Point-in-time daily cash-dividend matrices from JY dividend plans."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__:
    from ..config import get_jy_conn
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.config import get_jy_conn


DTYPE = np.float32
ACTUAL_PROCEDURE = 3131       # scheme implemented
FORWARD_PROCEDURE = 1004      # final resolution, not implemented


def _asof(date):
    return pd.Timestamp(date).normalize()


def _date_index(date, dates):
    target = np.datetime64(_asof(date).date())
    axis = np.asarray(dates, dtype="datetime64[D]")
    index = int(np.searchsorted(axis, target))
    if index >= len(axis) or axis[index] != target:
        raise ValueError(f"{target} is not in dates")
    return index


def _tick_axis(ticks):
    valid, positions = [], []
    for position, tick in enumerate(ticks):
        if tick is not None and not pd.isna(tick) and str(tick).strip():
            valid.append(str(tick).strip().zfill(6))
            positions.append(position)
    return valid, np.asarray(positions, dtype=int)


def _load_main(conn, asof):
    sql = """
    SELECT 
        d.ID AS record_id,
        s.SecuCode AS tick,
        d.EndDate AS end_date,
        d.AdvanceDate AS proposal_date,
        d.ProposalSN AS proposal_no,
        d.LatestInfoPublDate AS publish_date,
        d.EventProcedure AS event_procedure,
        d.IfDividend AS if_dividend,
        d.PriceUnit AS price_unit,
        d.DiviObject AS divi_object, 
        d.DiviObjectNew AS divi_object_new,
        d.ExDiviDate AS ex_dividend_date,
        d.CashDiviRMB / 10.0 AS cash_per_share,
        d.CashDiviRMBAdjusted / 10.0 AS cash_per_share_adjusted
    FROM dbo.LC_Dividend AS d
    INNER JOIN dbo.SecuMain AS s ON s.InnerCode=d.InnerCode
    WHERE s.SecuCategory=1 
        AND s.SecuMarket IN (83,90)
        AND d.LatestInfoPublDate <= %(asof)s
    """
    return pd.read_sql(sql, conn, params={"asof": asof.date()})


def _load_stib(conn, asof):
    sql = """
    WITH ranked AS (
        SELECT
            d.ID AS record_id,
            s.SecuCode AS tick,
            d.EndDate AS end_date,
            MIN(d.InfoPublDate) OVER (PARTITION BY d.InnerCode,d.EndDate,d.SchemeNo) AS proposal_date,
            d.SchemeNo AS proposal_no,
            d.InfoPublDate AS publish_date,
            d.EventProcedure AS event_procedure,
            d.IfDividend AS if_dividend,
            CAST(NULL AS int) AS price_unit,
            d.DiviObject AS divi_object,
            d.DiviObjectNew AS divi_object_new,
            d.ExDiviDate AS ex_dividend_date,
            d.CashDiviRMB / 10.0 AS cash_per_share,
            d.CashDiviRMBAdjusted / 10.0 AS cash_per_share_adjusted,
            ROW_NUMBER() OVER (
                PARTITION BY d.InnerCode,d.EndDate,d.SchemeNo
                ORDER BY d.InfoPublDate DESC,d.ID DESC
            ) AS rn
        FROM dbo.LC_STIBDividend AS d
        INNER JOIN dbo.SecuMain AS s ON s.InnerCode=d.InnerCode
        WHERE s.SecuCategory=1
          AND s.SecuMarket IN (83,90)
          AND d.InfoPublDate <= %(asof)s
    )
    SELECT record_id,tick,end_date,proposal_date,proposal_no,publish_date,event_procedure,
           if_dividend,price_unit,divi_object,divi_object_new,
           ex_dividend_date,cash_per_share,cash_per_share_adjusted
    FROM ranked
    WHERE rn=1
    """
    return pd.read_sql(sql, conn, params={"asof": asof.date()})


def _plans(conn, asof):
    asof = _asof(asof)
    frame = pd.concat([_load_main(conn, asof), _load_stib(conn, asof)], ignore_index=True)
    if frame.empty:
        return frame
    
    frame["tick"] = frame["tick"].astype("string").str.zfill(6)
    for column in ("end_date", "proposal_date", "publish_date", "ex_dividend_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["cash_per_share"] = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    frame["cash_per_share_adjusted"] = pd.to_numeric(frame["cash_per_share_adjusted"], errors="coerce")

    # Both SQL queries now return one current valid row per plan.  Keep a
    # defensive deduplication in case the source tables contain duplicate IDs.
    keys = ["tick", "end_date", "proposal_date", "proposal_no"]
    frame = frame.sort_values(
        [*keys, "publish_date", "record_id"]
    ).drop_duplicates(
        keys, keep="last"
    )
    
    return frame


def _valid_cash_plans(frame):
    object_code = frame.divi_object_new.where(
        frame.divi_object_new.notna(), frame.divi_object
    )
    return frame.loc[
        frame.if_dividend.eq(1)
        & object_code.eq(1)
        & frame.price_unit.isna()
        & frame.cash_per_share.gt(0)
    ].copy()


def _values(frame, field, valid_ticks, positions, axis_size):
    result = np.full(axis_size, np.nan, dtype=DTYPE)
    result[positions] = 0.0
    if not frame.empty:
        amounts = frame.groupby("tick", sort=False)[field].sum(min_count=1)
        result[positions] = amounts.reindex(valid_ticks, fill_value=0).to_numpy(DTYPE)
    return result


def _save(root, field, dates, dt, values):
    path = Path(root) / "fundamental" / "dividend" / f"{field}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(dates), len(values))
    size = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    if not path.exists():
        with path.open("wb") as file:
            file.truncate(size)
        array = np.memmap(path, mode="r+", dtype=DTYPE, shape=shape)
        array[:] = np.nan
        array.flush()
    if path.stat().st_size != size:
        raise ValueError(f"{path} size does not match axes {shape}")
    array = np.memmap(path, mode="r+", dtype=DTYPE, shape=shape)
    array[dt] = values
    array.flush()


def update_dividend(date, dates, ticks, conn=None, root=None):
    """Update actual event, TTM and approved-forward cash-dividend matrices."""
    if root is None:
        raise ValueError("root is required")
    asof = _asof(date)
    dt = _date_index(asof, dates)
    valid_ticks, positions = _tick_axis(ticks)
    plans = _plans(conn or get_jy_conn(), asof)

    cash_plans = _valid_cash_plans(plans)
    actual = cash_plans.loc[
        cash_plans.event_procedure.eq(ACTUAL_PROCEDURE)
        & cash_plans.ex_dividend_date.notna()
        & cash_plans.ex_dividend_date.le(asof)
    ]
    event_frame = actual.loc[actual.ex_dividend_date.eq(asof)]
    event = _values(event_frame, "cash_per_share", valid_ticks, positions, len(ticks))
    event_adjusted = _values(event_frame, "cash_per_share_adjusted", valid_ticks, positions, len(ticks))

    ttm_start = asof - pd.DateOffset(years=1)
    ttm_frame = actual.loc[actual.ex_dividend_date.gt(ttm_start)]
    ttm = _values(ttm_frame, "cash_per_share", valid_ticks, positions, len(ticks))
    ttm_adjusted = _values(ttm_frame, "cash_per_share_adjusted", valid_ticks, positions, len(ticks))

    # Forward is defined by the stock's newest disclosed plan.  Select that
    # plan before filtering IfDividend/cash; otherwise an old unresolved 1004
    # can reappear after a newer no-dividend plan is removed.
    latest_plans = plans.sort_values(
        ["tick", "publish_date", "end_date", "proposal_date", "proposal_no", "record_id"],
        na_position="first",
    ).drop_duplicates("tick", keep="last")
    forward_plans = _valid_cash_plans(latest_plans)
    forward_plans = forward_plans.loc[
        forward_plans.event_procedure.eq(FORWARD_PROCEDURE)
        & (forward_plans.ex_dividend_date.isna() | forward_plans.ex_dividend_date.gt(asof))
    ]
    forward = _values(forward_plans, "cash_per_share", valid_ticks, positions, len(ticks))
    forward_adjusted = _values(forward_plans, "cash_per_share_adjusted", valid_ticks, positions, len(ticks))

    result = {
        "cash_dividend_event": event,
        "cash_dividend_event_adjusted": event_adjusted,
        "cash_dividend_ttm": ttm,
        "cash_dividend_ttm_adjusted": ttm_adjusted,
        "cash_dividend_forward": forward,
        "cash_dividend_forward_adjusted": forward_adjusted,
    }
    for field, values in result.items():
        _save(root, field, dates, dt, values)
    return result


__all__ = ["update_dividend"]




if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/ticks.npy',allow_pickle=True)
    conn = get_jy_conn()
    root = Path('D:/data')

    update_dividend(
        date, dates, ticks, conn, root
    )
