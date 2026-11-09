"""盘口快照表清洗、证券轴对齐及通用盘口特征。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _time_key(value):
    text = str(value).strip()
    if len(text) == 5 and text[2] == ":":
        return text + ":00.000000"
    if len(text) == 8 and text[2] == ":":
        return text + ".000000"
    try:
        return pd.Timestamp(f"2000-01-01 {text}").strftime("%H:%M:%S.%f")
    except (TypeError, ValueError):
        return text


def _matrix(indexed, column, shape):
    if column not in indexed:
        return np.full(shape, np.nan)
    return pd.to_numeric(indexed[column], errors="coerce").to_numpy(float).reshape(shape)


def _sum_keep_missing(values, axis):
    result = np.nansum(values, axis=axis)
    result[np.all(~np.isfinite(values), axis=axis)] = np.nan
    return result


def _safe_divide(left, right):
    left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    return np.divide(left, right, out=np.full(left.shape, np.nan),
                     where=np.isfinite(right) & (right != 0))


def align_snapshot(frame, ticks, topn=5):
    """将宽表快照转成[分钟, 股票]矩阵，深度保留topn档、OFI使用一档。"""
    if frame is None or frame.empty:
        return None
    source = frame.copy()
    source["_tick"] = source["SecurityID"].astype(str).str.zfill(6).str[:6]
    source["_time"] = source["BarTime"].map(_time_key)
    trading = (
        source["_time"].between("09:30:00.000000", "11:30:00.000000")
        | source["_time"].between("13:00:00.000000", "14:57:00.000000")
    )
    source = source.loc[trading]
    if source.empty:
        return None

    source = source.sort_values(["_tick", "_time"]).drop_duplicates(
        ["_tick", "_time"], keep="last"
    )
    times = np.array(sorted(source["_time"].unique()), dtype=object)
    shape = (len(times), len(ticks))
    codes = [str(tick).zfill(6)[:6] for tick in ticks]
    indexed = source.set_index(["_time", "_tick"]).reindex(
        pd.MultiIndex.from_product([times, codes])
    )
    columns = {
        "bid_px": "BidPrice", "bid_qty": "BidQty",
        "ask_px": "AskPrice", "ask_qty": "AskQty",
    }
    matrices = {name: [] for name in columns}
    for level in range(1, topn + 1):
        for name, prefix in columns.items():
            matrices[name].append(_matrix(indexed, f"{prefix}{level}", shape))
    bid_px, bid_qty, ask_px, ask_qty = (np.asarray(matrices[name]) for name in columns)

    midpoint = (bid_px[0] + ask_px[0]) / 2
    spread = _safe_divide(ask_px[0] - bid_px[0], midpoint) * 10000
    depth = _sum_keep_missing(bid_qty, 0) + _sum_keep_missing(ask_qty, 0)
    imbalance = _safe_divide(bid_qty[0] - ask_qty[0], bid_qty[0] + ask_qty[0])
    ofi = np.full_like(midpoint, np.nan)
    if len(times) > 1:
        bp, bq, ap, aq = bid_px[0], bid_qty[0], ask_px[0], ask_qty[0]
        bid_flow = np.where(bp[1:] > bp[:-1], bq[1:],
                            np.where(bp[1:] < bp[:-1], -bq[:-1], bq[1:] - bq[:-1]))
        ask_flow = np.where(ap[1:] < ap[:-1], aq[1:],
                            np.where(ap[1:] > ap[:-1], -aq[:-1], aq[1:] - aq[:-1]))
        valid = (np.isfinite(bp[1:]) & np.isfinite(bp[:-1])
                 & np.isfinite(ap[1:]) & np.isfinite(ap[:-1]))
        ofi[1:] = np.where(valid, bid_flow - ask_flow, np.nan)
    return {
        "times": times, "mid": midpoint, "spread": spread, "depth": depth,
        "best_imbalance": imbalance, "ofi": ofi,
        "bid_px": bid_px, "bid_qty": bid_qty,
        "ask_px": ask_px, "ask_qty": ask_qty,
    }
