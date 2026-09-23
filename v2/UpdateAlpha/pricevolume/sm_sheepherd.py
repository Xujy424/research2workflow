"""Improved retail sheep-herd factor from Kaiyuan series (14) and (18)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
    from .order_activity_moneyflow import (
        OrderActivityMoneyflowConfig,
        _daily_flows,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT
    from v2.UpdateAlpha.pricevolume.order_activity_moneyflow import (
        OrderActivityMoneyflowConfig,
        _daily_flows,
    )


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class SMSHConfig:
    lookback_days: int = 20
    min_valid_days: int = 20
    open_field: str = "d_essentials/open_adj"
    close_field: str = "d_essentials/close_adj"
    small_upper: float = 50_000.0
    large_lower: float = 300_000.0
    flow_end_time: time = time(10, 0)
    cache_days: int = 32

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if not 0 < self.small_upper <= self.large_lower:
            raise ValueError("invalid small/large order thresholds")
        if self.cache_days < self.lookback_days + 1:
            raise ValueError("cache_days must exceed lookback_days")


class SMSHContext(AlphaContext):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        config=SMSHConfig(),
        l2_root=None,
        universe="self",
    ):
        self.config = config
        self.l2_root = (
            Path(l2_root) if l2_root is not None
            else Path(root) / "stock" / "l2"
        )
        self.flow_config = OrderActivityMoneyflowConfig(
            lookback_days=config.lookback_days,
            min_valid_days=config.min_valid_days,
            small_upper=config.small_upper,
            large_lower=config.large_lower,
            cache_days=config.cache_days,
        )
        self._daily_cache = OrderedDict()
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def daily_small_passive(self, date):
        key = pd.Timestamp(date).strftime("%Y%m%d")
        values = self._daily_cache.get(key)
        if values is None:
            frame = _daily_flows(
                self.l2_root,
                date,
                self.flow_config,
                "small",
                "passive",
                self.config.flow_end_time,
            )
            values = {}
            for side, order_side in (("buy", 1), ("sell", -1)):
                selected = frame.filter(
                    pl.col("OrderSide") == order_side
                ).select("tick", pl.col("Amount").alias("value"))
                values[side] = self.align(selected)
            self._daily_cache[key] = values
            while len(self._daily_cache) > self.config.cache_days:
                self._daily_cache.popitem(last=False)
        else:
            self._daily_cache.move_to_end(key)
        return values


def _daily_net_inflow(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Return buy minus sell, treating only one missing side as zero."""
    buy = np.asarray(buy, dtype=np.float64)
    sell = np.asarray(sell, dtype=np.float64)
    available = np.isfinite(buy) | np.isfinite(sell)
    return np.where(
        available,
        np.where(np.isfinite(buy), buy, 0.0)
        - np.where(np.isfinite(sell), sell, 0.0),
        np.nan,
    )


def _rank_columns(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ranked = pd.DataFrame(np.where(valid, values, np.nan)).rank(
        axis=0,
        method="average",
    )
    return ranked.to_numpy(dtype=np.float64)


def _spearman_by_column(
    x: np.ndarray,
    y: np.ndarray,
    min_valid: int,
) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    x_rank = _rank_columns(x, valid)
    y_rank = _rank_columns(y, valid)

    x_mean = np.divide(
        np.where(valid, x_rank, 0.0).sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    y_mean = np.divide(
        np.where(valid, y_rank, 0.0).sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    x_centered = np.where(valid, x_rank - x_mean, 0.0)
    y_centered = np.where(valid, y_rank - y_mean, 0.0)
    numerator = np.sum(x_centered * y_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=0)
        * np.sum(y_centered * y_centered, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=(count >= min_valid) & (denominator > 0),
    ).astype(np.float32)


class SMSHFactor(AlphaBase):
    """20D RankCorr(intraday R_t, next-day morning passive small flow)."""

    meta = AlphaMeta(
        "smsh",
        "20D RankCorr of intraday return and next-day morning passive small flow",
        direction=-1,
    )
    dependencies = (
        "d_essentials/open_adj",
        "d_essentials/close_adj",
    )

    def calculate(self, asof):
        cfg = self.context.config
        axis = self.context.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)

        data = self.context.data
        open_price = np.asarray(
            data.read(cfg.open_field, end, start_date=start),
            dtype=np.float64,
        )
        close_price = np.asarray(
            data.read(cfg.close_field, end, start_date=start),
            dtype=np.float64,
        )
        intraday_return = np.divide(
            close_price,
            open_price,
            out=np.full_like(close_price, np.nan),
            where=(open_price > 0) & np.isfinite(open_price),
        ) - 1.0
        daily_flows = [
            self.context.daily_small_passive(date)
            for date in axis.trade_dates[start:end + 1]
        ]
        buy = np.stack([item["buy"] for item in daily_flows])
        sell = np.stack([item["sell"] for item in daily_flows])
        passive_small_net = _daily_net_inflow(buy, sell)

        return _spearman_by_column(
            intraday_return[:-1],
            passive_small_net[1:],
            cfg.min_valid_days,
        )


__all__ = ["SMSHConfig", "SMSHContext", "SMSHFactor"]
