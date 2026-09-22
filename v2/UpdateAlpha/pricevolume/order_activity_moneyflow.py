"""L2 money-flow strength split by order size and active/passive side."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
    from .moneyflow_strength import (
        _compound_return,
        _cross_section_residual,
        _moneyflow_strength,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT
    from v2.UpdateAlpha.pricevolume.moneyflow_strength import (
        _compound_return,
        _cross_section_residual,
        _moneyflow_strength,
    )


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT
FLOW_KEYS = tuple(
    f"{size}_{activity}_{side}"
    for size in ("large", "small")
    for activity in ("active", "passive")
    for side in ("buy", "sell")
)


@dataclass(frozen=True)
class OrderActivityMoneyflowConfig:
    lookback_days: int = 20
    min_valid_days: int = 20
    min_cross_section_observations: int = 30
    return_field: str = "d_essentials/pct"
    small_upper: float = 50_000.0
    large_lower: float = 300_000.0
    large_upper: float = 1_000_000.0
    cache_days: int = 32

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if self.min_cross_section_observations < 2:
            raise ValueError("min_cross_section_observations must be at least 2")
        if not 0 < self.small_upper <= self.large_lower < self.large_upper:
            raise ValueError("invalid small/large order thresholds")
        if self.cache_days < self.lookback_days:
            raise ValueError("cache_days must be at least lookback_days")


def _order_size(column: str, config: OrderActivityMoneyflowConfig) -> pl.Expr:
    amount = pl.col(column)
    return (
        pl.when((amount >= 0) & (amount < config.small_upper))
        .then(pl.lit("small"))
        .when(
            (amount >= config.large_lower)
            & (amount < config.large_upper)
        )
        .then(pl.lit("large"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def _exchange_flows(
    folder: Path,
    exchange: str,
    config: OrderActivityMoneyflowConfig,
) -> pl.LazyFrame | None:
    trade_path = folder / f"{exchange}cj.pq"
    order_path = folder / f"{exchange}wt.pq"
    if not trade_path.is_file() or not order_path.is_file():
        return None

    trades = pl.scan_parquet(trade_path).select(
        "ChannelNo",
        pl.col("SecurityID").cast(pl.String).str.pad_start(6, "0"),
        "BidApplSeqNum",
        "OfferApplSeqNum",
        pl.col("Side").alias("AggressorSide"),
        (pl.col("Price") * pl.col("OrderQty"))
        .cast(pl.Float64)
        .alias("TradeAmount"),
    )
    orders = pl.scan_parquet(order_path).select(
        "ChannelNo",
        pl.col("SecurityID").cast(pl.String).str.pad_start(6, "0"),
        "ApplSeqNum",
        "Side",
        (pl.col("Price") * pl.col("OrderQty"))
        .cast(pl.Float64)
        .alias("OrderAmount"),
    )
    buy_orders = orders.filter(pl.col("Side") == 1).select(
        "ChannelNo",
        "SecurityID",
        pl.col("ApplSeqNum").alias("BidApplSeqNum"),
        pl.col("OrderAmount").alias("BuyOrderAmount"),
    )
    sell_orders = orders.filter(pl.col("Side") == -1).select(
        "ChannelNo",
        "SecurityID",
        pl.col("ApplSeqNum").alias("OfferApplSeqNum"),
        pl.col("OrderAmount").alias("SellOrderAmount"),
    )
    matched = trades.join(
        buy_orders,
        on=["ChannelNo", "SecurityID", "BidApplSeqNum"],
        how="left",
    ).join(
        sell_orders,
        on=["ChannelNo", "SecurityID", "OfferApplSeqNum"],
        how="left",
    )
    buy = matched.select(
        "SecurityID",
        pl.lit("buy").alias("OrderSide"),
        pl.when(pl.col("AggressorSide") == 1)
        .then(pl.lit("active"))
        .otherwise(pl.lit("passive"))
        .alias("Activity"),
        _order_size("BuyOrderAmount", config).alias("Size"),
        pl.col("TradeAmount").alias("Amount"),
    )
    sell = matched.select(
        "SecurityID",
        pl.lit("sell").alias("OrderSide"),
        pl.when(pl.col("AggressorSide") == -1)
        .then(pl.lit("active"))
        .otherwise(pl.lit("passive"))
        .alias("Activity"),
        _order_size("SellOrderAmount", config).alias("Size"),
        pl.col("TradeAmount").alias("Amount"),
    )
    return pl.concat([buy, sell]).filter(pl.col("Size").is_not_null())


def _daily_flows(
    l2_root: Path,
    date,
    config: OrderActivityMoneyflowConfig,
) -> pl.DataFrame:
    folder = l2_root / "proc" / pd.Timestamp(date).strftime("%Y%m%d")
    scans = [
        scan
        for exchange in ("sh", "sz")
        if (scan := _exchange_flows(folder, exchange, config)) is not None
    ]
    if not scans:
        return pl.DataFrame(
            schema={
                "tick": pl.String,
                "Size": pl.String,
                "Activity": pl.String,
                "OrderSide": pl.String,
                "Amount": pl.Float64,
            }
        )
    return (
        pl.concat(scans)
        .group_by("SecurityID", "Size", "Activity", "OrderSide")
        .agg(pl.col("Amount").sum())
        .rename({"SecurityID": "tick"})
        .collect(engine="streaming")
    )


class OrderActivityMoneyflowContext(AlphaContext):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        config=OrderActivityMoneyflowConfig(),
        l2_root=None,
        universe="self",
    ):
        self.config = config
        self.l2_root = (
            Path(l2_root) if l2_root is not None
            else Path(root) / "stock" / "l2"
        )
        self._daily_cache = OrderedDict()
        self._history_key = None
        self._history_value = None
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def _daily_values(self, date):
        key = pd.Timestamp(date).strftime("%Y%m%d")
        values = self._daily_cache.get(key)
        if values is None:
            frame = _daily_flows(self.l2_root, date, self.config)
            values = {}
            for flow_key in FLOW_KEYS:
                size, activity, side = flow_key.split("_")
                selected = frame.filter(
                    (pl.col("Size") == size)
                    & (pl.col("Activity") == activity)
                    & (pl.col("OrderSide") == side)
                ).select("tick", pl.col("Amount").alias("value"))
                values[flow_key] = self.align(selected)
            self._daily_cache[key] = values
            while len(self._daily_cache) > self.config.cache_days:
                self._daily_cache.popitem(last=False)
        else:
            self._daily_cache.move_to_end(key)
        return values

    def history(self, asof):
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        if start < 0:
            return None
        key = (start, end)
        if key != self._history_key:
            daily = [
                self._daily_values(date)
                for date in axis.trade_dates[start:end + 1]
            ]
            self._history_value = {
                name: np.stack([item[name] for item in daily])
                for name in FLOW_KEYS
            }
            self._history_value["daily_pct"] = np.asarray(
                self.data.read(
                    self.config.return_field,
                    end,
                    start_date=start,
                ),
                dtype=np.float64,
            )
            self._history_key = key
        return self._history_value


class OrderActivityMoneyflowFactor(AlphaBase):
    size: str
    activity: str

    def calculate(self, asof):
        history = self.context.history(asof)
        if history is None:
            return np.full(
                self.context.data.axis.tick_count,
                np.nan,
                dtype=np.float32,
            )
        config = self.context.config
        prefix = f"{self.size}_{self.activity}"
        strength = _moneyflow_strength(
            history[f"{prefix}_buy"],
            history[f"{prefix}_sell"],
            config.min_valid_days,
        )
        ret20 = _compound_return(
            history["daily_pct"], config.min_valid_days
        )
        return _cross_section_residual(
            strength,
            ret20,
            config.min_cross_section_observations,
        )


class LargeActiveMoneyflowFactor(OrderActivityMoneyflowFactor):
    meta = AlphaMeta(
        "large_active_moneyflow_strength",
        "L2 large active-order flow strength residualized against Ret20",
        direction=1,
    )
    dependencies = ("d_essentials/pct",)
    size = "large"
    activity = "active"


class LargePassiveMoneyflowFactor(OrderActivityMoneyflowFactor):
    meta = AlphaMeta(
        "large_passive_moneyflow_strength",
        "L2 large passive-order flow strength residualized against Ret20",
        direction=1,
    )
    dependencies = ("d_essentials/pct",)
    size = "large"
    activity = "passive"


class SmallActiveMoneyflowFactor(OrderActivityMoneyflowFactor):
    meta = AlphaMeta(
        "small_active_moneyflow_strength",
        "L2 small active-order flow strength residualized against Ret20",
        direction=-1,
    )
    dependencies = ("d_essentials/pct",)
    size = "small"
    activity = "active"


class SmallPassiveMoneyflowFactor(OrderActivityMoneyflowFactor):
    meta = AlphaMeta(
        "small_passive_moneyflow_strength",
        "L2 small passive-order flow strength residualized against Ret20",
        direction=-1,
    )
    dependencies = ("d_essentials/pct",)
    size = "small"
    activity = "passive"


ORDER_ACTIVITY_MONEYFLOW_FACTORS = (
    LargeActiveMoneyflowFactor,
    LargePassiveMoneyflowFactor,
    SmallActiveMoneyflowFactor,
    SmallPassiveMoneyflowFactor,
)


__all__ = [
    "OrderActivityMoneyflowConfig",
    "OrderActivityMoneyflowContext",
    "OrderActivityMoneyflowFactor",
    "LargeActiveMoneyflowFactor",
    "LargePassiveMoneyflowFactor",
    "SmallActiveMoneyflowFactor",
    "SmallPassiveMoneyflowFactor",
    "ORDER_ACTIVITY_MONEYFLOW_FACTORS",
]
