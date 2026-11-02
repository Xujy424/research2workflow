"""盘口快照的无状态清洗、轴对齐与基础特征计算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import _matrix_from_indexed, _sum_preserve_nan, _time_key, safe_divide


def align_snapshot(frame, ticks, topn=5):
    """将快照表转换成[分钟, 股票]矩阵，深度保留 topn 档、OFI 使用一档。"""
    if frame is None or frame.empty:
        return None

    source = frame.copy()
    source["_tick"] = source["SecurityID"].astype(str).str.zfill(6).str[:6]
    source["_time"] = source["BarTime"].map(_time_key)
    trading_time = (
        source["_time"].between("09:30:00.000000", "11:30:00.000000")
        | source["_time"].between("13:00:00.000000", "14:57:00.000000")
    )
    source = source.loc[trading_time]
    if source.empty:
        return None

    source = source.sort_values(["_tick", "_time"]).drop_duplicates(
        ["_tick", "_time"], keep="last"
    )
    times = np.array(sorted(source["_time"].unique()), dtype=object)
    shape = (len(times), len(ticks))
    stock_codes = [str(tick).zfill(6)[:6] for tick in ticks]
    index = pd.MultiIndex.from_product([times, stock_codes])
    indexed = source.set_index(["_time", "_tick"]).reindex(index)

    columns = {
        "bid_px": "BidPrice", "bid_qty": "BidQty",
        "ask_px": "AskPrice", "ask_qty": "AskQty",
    }
    matrices = {name: [] for name in columns}
    for level in range(1, topn + 1):
        for name, prefix in columns.items():
            matrices[name].append(_matrix_from_indexed(indexed, f"{prefix}{level}", shape))
    bid_px, bid_qty, ask_px, ask_qty = (
        np.asarray(matrices[name]) for name in columns
    )

    midpoint = (bid_px[0] + ask_px[0]) / 2
    spread = safe_divide(ask_px[0] - bid_px[0], midpoint) * 10000
    depth = _sum_preserve_nan(bid_qty, 0) + _sum_preserve_nan(ask_qty, 0)
    best_imbalance = safe_divide(
        bid_qty[0] - ask_qty[0], bid_qty[0] + ask_qty[0]
    )
    ofi = np.full_like(midpoint, np.nan)
    if len(times) > 1:
        bp, bq, ap, aq = bid_px[0], bid_qty[0], ask_px[0], ask_qty[0]
        bid_flow = np.where(
            bp[1:] > bp[:-1], bq[1:],
            np.where(bp[1:] < bp[:-1], -bq[:-1], bq[1:] - bq[:-1]),
        )
        ask_flow = np.where(
            ap[1:] < ap[:-1], aq[1:],
            np.where(ap[1:] > ap[:-1], -aq[:-1], aq[1:] - aq[:-1]),
        )
        valid = (
            np.isfinite(bp[1:]) & np.isfinite(bp[:-1])
            & np.isfinite(ap[1:]) & np.isfinite(ap[:-1])
        )
        ofi[1:] = np.where(valid, bid_flow - ask_flow, np.nan)

    return {
        "times": times, "mid": midpoint, "spread": spread, "depth": depth,
        "best_imbalance": best_imbalance, "ofi": ofi,
        "bid_px": bid_px, "bid_qty": bid_qty,
        "ask_px": ask_px, "ask_qty": ask_qty,
    }
