"""Smart-money 2.0 factor from Kaiyuan Securities microstructure series (3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


@dataclass(frozen=True)
class SmartMoneyConfig:
    lookback_days: int = 10
    smart_volume_share: float = 0.20


class SmartMoneyContext(AlphaContext):
    def __init__(self, root=ROOT, config=SmartMoneyConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))


class SmartMoneyFactor(AlphaBase):
    """
    Q=VWAPsmart/VWAPall using S=abs(minute return)/ln(minute volume).
    如果少量成交能够推动价格发生较大变化，这些成交可能包含更多知情交易信息
    VWAP of smart money / total VWAP
    """

    meta = AlphaMeta(
        "smart_money",
        "10D log-volume smart-money VWAP ratio",
        direction=-1,
    )
    dependencies = (
        "m_essentials/open", "m_essentials/close",
        "m_essentials/volume", "m_essentials/amount",
    )

    def calculate(self, asof):
        cfg = self.context.config
        axis = self.context.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days + 1
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)

        data = self.context.data
        close = np.asarray(data.read("m_essentials/close", end, start), float)
        open_ = np.asarray(data.read("m_essentials/open", end, start), float)
        volume = np.asarray(data.read("m_essentials/volume", end, start), float)
        amount = np.asarray(data.read("m_essentials/amount", end, start), float)

        previous = np.empty_like(close)
        previous[:, 0] = open_[:, 0]
        previous[:, 1:] = close[:, :-1]
        minute_return = np.divide(
            close, previous,
            out=np.full_like(close, np.nan),
            where=np.isfinite(close) & np.isfinite(previous) & (previous > 0),
        ) - 1.0

        # Stored volume is in thousands of shares; restore shares because the
        # logarithmic S definition is sensitive to the unit of V.
        raw_volume = volume * 1000.0
        log_volume = np.log(raw_volume, where=raw_volume>1, out=np.full_like(raw_volume, np.nan))
        smartness = np.divide(
            np.abs(minute_return), log_volume,
            out=np.full_like(minute_return, np.nan),
            where=np.isfinite(minute_return) & np.isfinite(log_volume) & (log_volume > 0) & np.isfinite(amount) & (amount > 0),
        )

        bars = cfg.lookback_days * close.shape[1]
        smartness = smartness.reshape(bars, axis.tick_count)   # 是ticks长度还是valid_tick长度
        volume = volume.reshape(bars, axis.tick_count)
        amount = amount.reshape(bars, axis.tick_count)
        valid = (
            np.isfinite(smartness) & np.isfinite(volume) & (volume > 0) & np.isfinite(amount) & (amount > 0)
        )
        scores = np.where(valid, smartness, -np.inf)
        order = np.argsort(scores, axis=0)[::-1]
        sorted_volume = np.take_along_axis(np.where(valid, volume, 0.0), order, axis=0)
        sorted_amount = np.take_along_axis(np.where(valid, amount, 0.0), order, axis=0)
        total_volume = sorted_volume.sum(axis=0)
        cumulative_before = np.cumsum(sorted_volume, axis=0) - sorted_volume  #计算每个分钟加入之前，已经累计了多少成交量
        smart = cumulative_before < cfg.smart_volume_share * total_volume
        smart_volume = np.sum(np.where(smart, sorted_volume, 0.0), axis=0)
        smart_amount = np.sum(np.where(smart, sorted_amount, 0.0), axis=0)
        total_amount = sorted_amount.sum(axis=0)

        smart_vwap = np.divide(
            smart_amount, smart_volume,
            out=np.full(axis.tick_count, np.nan), where=smart_volume > 0,
        )
        all_vwap = np.divide(
            total_amount, total_volume,
            out=np.full(axis.tick_count, np.nan), where=total_volume > 0,
        )
        result = np.divide(
            smart_vwap, all_vwap,
            out=np.full(axis.tick_count, np.nan),
            where=np.isfinite(all_vwap) & (all_vwap > 0),
        )
        return result.astype(np.float32)


__all__ = [
    "SmartMoneyConfig", "SmartMoneyContext", "SmartMoneyFactor",
    "calculate_smart_money_v2",
]
