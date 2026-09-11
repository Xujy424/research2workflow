"""SATD factors based on a configurable active side at intraday special moments."""
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
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class SATDConfig:
    lookback_days: int = 20
    selection_ratio: float = 0.10
    min_valid_days: int = 20
    min_minutes: int = 20
    cache_days: int = 2

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 0 < self.selection_ratio <= 1:
            raise ValueError("selection_ratio must be in (0, 1]")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if self.min_minutes < 1 or self.cache_days < 1:
            raise ValueError("min_minutes and cache_days must be positive")


class SATDContext(AlphaContext):
    """Load, filter, minute-aggregate, and cache the common L2 input."""
    _MINUTE_SCHEMA = {
        "tick": pl.String, "minute": pl.Int64, "all_amount": pl.Float64,
        "all_count": pl.UInt32, "volume": pl.Int64, "close": pl.Float64,
        "buy_amount": pl.Float64, "buy_count": pl.UInt32,
        "sell_amount": pl.Float64, "sell_count": pl.UInt32,
        "minute_return": pl.Float64,
    }

    def __init__(self, root=DEFAULT_ROOT, config=SATDConfig(), l2_root=None):
        self.config = config
        self.l2_root = Path(l2_root) if l2_root else Path(root) / "stock" / "l2"
        self._minute_cache = OrderedDict()
        super().__init__(DataPool(root, asset="stock"))

    def minute_data(self, date):
        key = pd.Timestamp(date).strftime("%Y%m%d")
        cached = self._minute_cache.get(key)
        if cached is not None:
            self._minute_cache.move_to_end(key)
            return cached

        folder = self.l2_root / "proc" / key
        files = [folder / name for name in ("shcj.pq", "szcj.pq")]
        scans = []
        for path in files:
            if not path.is_file():
                continue
            scans.append(
                pl.scan_parquet(path)
                .select("SecurityID", "TransactTime", "Price", "OrderQty", "Side")
                .filter(
                    (pl.col("Price") > 0) & (pl.col("OrderQty") > 0)
                    & (
                        pl.col("TransactTime").is_between(time(9, 30), time(11, 30), closed="left")
                        | pl.col("TransactTime").is_between(time(13), time(14, 57), closed="left")
                    )
                )
                .with_columns(
                    pl.col("SecurityID").cast(pl.String).str.pad_start(6, "0").alias("tick"),
                    (pl.col("TransactTime").cast(pl.Int64) // 60_000_000_000).alias("minute"),
                    (pl.col("Price") * pl.col("OrderQty")).cast(pl.Float64).alias("amount"),
                )
            )
        if not scans:
            frame = pl.DataFrame(schema=self._MINUTE_SCHEMA)
        else:
            frame = (
                pl.concat(scans)
                .group_by("tick", "minute")
                .agg(
                    pl.col("amount").sum().alias("all_amount"),
                    pl.len().alias("all_count"),
                    pl.col("OrderQty").sum().alias("volume"),
                    pl.col("Price").sort_by("TransactTime").last().alias("close"),
                    pl.col("amount").filter(pl.col("Side") == 1).sum().alias("buy_amount"),
                    (pl.col("Side") == 1).sum().alias("buy_count"),
                    pl.col("amount").filter(pl.col("Side") == -1).sum().alias("sell_amount"),
                    (pl.col("Side") == -1).sum().alias("sell_count"),
                )
                .sort("tick", "minute")
                .with_columns(
                    (pl.col("close") / pl.col("close").shift(1).over("tick") - 1.0)
                    .alias("minute_return")
                )
                .collect(engine="streaming")
            )
        self._minute_cache[key] = frame
        while len(self._minute_cache) > self.config.cache_days:
            self._minute_cache.popitem(last=False)
        return frame


class _SATDFactor(AlphaBase):
    side = -1
    dependencies = ("l2/proc/shcj.pq", "l2/proc/szcj.pq")

    def __init__(self, context):
        super().__init__(context)
        self._daily_cache = OrderedDict()

    def _selected_minutes(self, minute):
        raise NotImplementedError

    def _daily_value(self, date):
        key = pd.Timestamp(date).strftime("%Y%m%d")
        cached = self._daily_cache.get(key)
        if cached is not None:
            self._daily_cache.move_to_end(key)
            return cached
        if self.side not in (-1, 1):
            raise ValueError("factor side must be 1 (active buy) or -1 (active sell)")

        minute = self.context.minute_data(date)
        if minute.is_empty():
            result = np.full(self.context.data.axis.tick_count, np.nan, np.float32)
        else:
            side_name = "buy" if self.side == 1 else "sell"
            selected = self._selected_minutes(minute.lazy())
            daily = (
                selected.group_by("tick")
                .agg(
                    pl.col("all_amount").sum().alias("day_amount"),
                    pl.col("all_count").sum().alias("day_count"),
                    pl.col(f"{side_name}_amount").filter(pl.col("selected")).sum().alias("selected_amount"),
                    pl.col(f"{side_name}_count").filter(pl.col("selected")).sum().alias("selected_count"),
                    pl.len().alias("minute_count"),
                )
                .with_columns(
                    pl.when(
                        (pl.col("minute_count") >= self.context.config.min_minutes)
                        & (pl.col("day_amount") > 0)
                        & (pl.col("selected_count") > 0)
                    )
                    .then(
                        (pl.col("selected_amount") / pl.col("selected_count"))
                        / (pl.col("day_amount") / pl.col("day_count"))
                    )
                    .otherwise(None)
                    .alias("value")
                )
                .select("tick", "value")
                .collect(engine="streaming")
            )
            result = self.context.align(daily)

        self._daily_cache[key] = result
        cache_size = self.context.config.lookback_days + 2
        while len(self._daily_cache) > cache_size:
            self._daily_cache.popitem(last=False)
        return result

    def _window_dates(self, asof):
        cfg = self.context.config
        end = self.context.data.axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days + 1
        if start < 0:
            return None
        return self.context.data["trade_dates"][start:end + 1]

    def _mean_values(self, values):
        valid = np.isfinite(values)
        count = valid.sum(axis=0)
        return np.divide(
            np.where(valid, values, 0.0).sum(axis=0), count,
            out=np.full(values.shape[1], np.nan),
            where=count >= self.context.config.min_valid_days,
        ).astype(np.float32)

    def calculate(self, asof):
        dates = self._window_dates(asof)
        if dates is None:
            return np.full(self.context.data.axis.tick_count, np.nan, np.float32)
        return self._mean_values(np.stack([self._daily_value(date) for date in dates]))

class SATDSellDownRetFactor(_SATDFactor):
    meta = AlphaMeta("satd_selldownret", "20D active-sell SATD at lowest-return minutes", direction=1)

    def _selected_minutes(self, minute):
        return minute.with_columns(
            pl.col("minute_return").rank(method="ordinal").over("tick").alias("rank"),
            pl.col("minute_return").is_not_null().sum().over("tick").alias("valid_count"),
        ).with_columns(
            (pl.col("rank") <= (pl.col("valid_count") * self.context.config.selection_ratio).ceil())
            .fill_null(False).alias("selected")
        )

class SATDSellLowPriceFactor(_SATDFactor):
    meta = AlphaMeta("satd_selllowprice", "20D active-sell SATD at lowest-price minutes", direction=1)

    def _selected_minutes(self, minute):
        return minute.with_columns(
            pl.col("close").rank(method="ordinal").over("tick").alias("rank"),
            pl.len().over("tick").alias("valid_count"),
        ).with_columns(
            (pl.col("rank") <= (pl.col("valid_count") * self.context.config.selection_ratio).ceil())
            .alias("selected")
        )

class SATDSellHighVolumeFactor(_SATDFactor):
    meta = AlphaMeta("satd_sellhighvolume", "20D active-sell SATD at highest-volume minutes", direction=1)

    def _selected_minutes(self, minute):
        return minute.with_columns(
            pl.col("volume").rank(method="ordinal", descending=True).over("tick").alias("rank"),
            pl.len().over("tick").alias("valid_count"),
        ).with_columns(
            (pl.col("rank") <= (pl.col("valid_count") * self.context.config.selection_ratio).ceil())
            .alias("selected")
        )

class SATDBuyFlatFactor(_SATDFactor):
    """Active-buy SATD during minutes whose return is exactly zero."""
    side = 1
    meta = AlphaMeta(
        "satd_buyflat",
        "20D active-buy SATD at flat-return minutes",
        direction=-1,
    )
    def _selected_minutes(self, minute):
        return minute.with_columns(
            (pl.col("minute_return") == 0).fill_null(False).alias("selected")
        )

class SATDCombinationFactor(AlphaBase):
    meta = AlphaMeta("satd_combination", "Equal-weight active-sell SATD combination", direction=1)
    dependencies = _SATDFactor.dependencies
    component_classes = (
        SATDSellDownRetFactor,
        SATDSellLowPriceFactor,
        SATDSellHighVolumeFactor,
    )

    def __init__(self, context):
        super().__init__(context)
        self.components = tuple(cls(context) for cls in self.component_classes)

    def calculate(self, asof):
        dates = self.components[0]._window_dates(asof)
        if dates is None:
            return np.full(self.context.data.axis.tick_count, np.nan, np.float32)

        # Interleave components by date so one small minute cache is enough:
        # each day's L2 parquet is scanned once and reused by all components.
        daily = [
            [factor._daily_value(date) for factor in self.components]
            for date in dates
        ]
        daily = np.asarray(daily)
        values = np.stack([
            factor._mean_values(daily[:, index, :])
            for index, factor in enumerate(self.components)
        ])
        result = np.mean(values, axis=0)
        result[~np.isfinite(values).all(axis=0)] = np.nan
        return result.astype(np.float32)

SATD_FACTORS = (
    SATDSellDownRetFactor, SATDSellLowPriceFactor,
    SATDSellHighVolumeFactor, SATDBuyFlatFactor,
    SATDCombinationFactor,
)

__all__ = [
    "SATDConfig", "SATDContext", "SATDSellDownRetFactor",
    "SATDSellLowPriceFactor", "SATDSellHighVolumeFactor",
    "SATDCombinationFactor",
]
