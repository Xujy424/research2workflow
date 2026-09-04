"""QUA factor based on minute-level average trade amount."""

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
class QUAConfig:
    lookback_days: int = 20
    trim_largest_minutes: int = 10
    quantile: float = 0.10
    min_minutes: int = 20
    min_valid_days: int = 20
    cache_days: int = 32

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if self.trim_largest_minutes < 0:
            raise ValueError("trim_largest_minutes must be non-negative")
        if not 0 <= self.quantile <= 1:
            raise ValueError("quantile must be between 0 and 1")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")


class QUAContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=QUAConfig(), l2_root=None):
        self.config = config
        self.l2_root = (
            Path(l2_root) if l2_root is not None
            else Path(root) / "stock" / "l2"
        )
        self._daily_cache = OrderedDict()
        super().__init__(DataPool(root, asset="stock"))

    def daily_values(self, date, name):
        key = (name, pd.Timestamp(date).strftime("%Y%m%d"))
        cached = self._daily_cache.get(key)
        if cached is None:
            calculators = {
                "qua": _daily_qua,
                "mts": _daily_mts,
                "mte": _daily_mte,
            }
            frame = calculators[name](self.l2_root, date, self.config)
            cached = self.align(frame, name)
            self._daily_cache[key] = cached
            while len(self._daily_cache) > self.config.cache_days:
                self._daily_cache.popitem(last=False)
        else:
            self._daily_cache.move_to_end(key)
        return cached


def _minute_trade_statistics(path: Path) -> pl.LazyFrame:
    """Aggregate trades to average single-trade amount and total amount."""
    return (
        pl.scan_parquet(path)
        .select("SecurityID", "TransactTime", "Price", "OrderQty")
        .filter(
            (pl.col("Price") > 0)
            & (pl.col("OrderQty") > 0)
            & (
                pl.col("TransactTime").is_between(time(9, 30), time(11, 30), closed="left")
                | pl.col("TransactTime").is_between(time(13, 0), time(14, 57), closed="left")
            )
        )
        .with_columns(
            pl.col("SecurityID").cast(pl.String).str.pad_start(6, "0").alias("tick"),
            (pl.col("TransactTime").cast(pl.Int64) // 60_000_000_000).alias("minute"),
            (pl.col("Price") * pl.col("OrderQty")).alias("trade_amount"),
        )
        .group_by("tick", "minute")
        .agg(
            pl.col("trade_amount").mean().alias("average_trade_amount"),
            pl.col("trade_amount").sum().alias("total_amount"),
            pl.col("Price").sort_by("TransactTime").last().alias("close"),   # 这里好像有点问题，收盘价格是否等于最后一笔成交价格？
        )
    )


def _minute_data(l2_root: Path, date) -> pl.LazyFrame | None:
    folder = l2_root / "proc" / pd.Timestamp(date).strftime("%Y%m%d")
    files = [folder / name for name in ("shcj.pq", "szcj.pq")]
    scans = [_minute_trade_statistics(path) for path in files if path.is_file()]
    return pl.concat(scans) if scans else None


def _empty_daily(name) -> pl.DataFrame:
    return pl.DataFrame(schema={"tick": pl.String, name: pl.Float64})


def _daily_qua(l2_root: Path, date, config: QUAConfig) -> pl.DataFrame:
    """Calculate one day's normalized 10% quantile indicator."""
    minute = _minute_data(l2_root, date)
    if minute is None:
        return _empty_daily("qua")

    minute = minute.with_columns(
        pl.col("average_trade_amount").rank(method="ordinal", descending=True).over("tick").alias("large_rank"),
        pl.len().over("tick").alias("minute_count"),
    )
    retained = minute.filter(
        (pl.col("large_rank") > config.trim_largest_minutes)
        & (pl.col("minute_count") >= config.min_minutes + config.trim_largest_minutes)
    )
    return (
        retained.group_by("tick")
        .agg(
            pl.col("average_trade_amount").min().alias("minimum"),
            pl.col("average_trade_amount").max().alias("maximum"),
            pl.col("average_trade_amount").quantile(config.quantile, interpolation="linear").alias("q10"),
        )
        .with_columns(
            pl.when(pl.col("maximum") > pl.col("minimum"))
            .then(
                (pl.col("q10") - pl.col("minimum"))
                / (pl.col("maximum") - pl.col("minimum"))
            )
            .otherwise(None)
            .alias("qua")
        )
        .select("tick", "qua")
        .collect(engine="streaming")
    )


def _daily_correlation(
    l2_root: Path,
    date,
    config: QUAConfig,
    other_column: str,
    name: str,
) -> pl.DataFrame:
    minute = _minute_data(l2_root, date)
    if minute is None:
        return _empty_daily(name)
    return (
        minute.group_by("tick")
        .agg(
            pl.len().alias("minute_count"),
            pl.corr("average_trade_amount", other_column).alias(name),
        )
        .filter(pl.col("minute_count") >= config.min_minutes)
        .select("tick", name)
        .collect(engine="streaming")
    )


def _daily_mts(l2_root: Path, date, config: QUAConfig) -> pl.DataFrame:
    """Correlation of minute average trade amount and minute total amount."""
    return _daily_correlation(l2_root, date, config, "total_amount", "mts")


def _daily_mte(l2_root: Path, date, config: QUAConfig) -> pl.DataFrame:
    """Correlation of minute average trade amount and minute close price."""
    return _daily_correlation(l2_root, date, config, "close", "mte")


class _RollingMinuteFactor(AlphaBase):
    filter_tradable: bool = False
    daily_column: str
    dependencies = ("l2/proc/shcj.pq", "l2/proc/szcj.pq")

    def calculate(self, asof):
        config = self.context.config
        axis = self.context.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - config.lookback_days + 1
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)

        trade_dates = self.context.data["trade_dates"]
        daily = np.stack(
            [
                self.context.daily_values(date, self.daily_column)
                for date in trade_dates[start:end + 1]
            ]
        )
        valid = np.isfinite(daily)
        if self.filter_tradable:
            tradable = np.asarray(
                self.context.data.read(
                    "basic/tradable",
                    start_date=start,
                    end_date=end,
                ),
                dtype=bool,
            )
            valid &= tradable
        count = valid.sum(axis=0)
        result = np.divide(
            np.where(valid, daily, 0).sum(axis=0),
            count,
            out=np.full(axis.tick_count, np.nan),
            where=count >= config.min_valid_days,
        )
        if self.filter_tradable:
            result = np.where(tradable[-1], result, np.nan)
        return result.astype(np.float32)


class QUAFactor(_RollingMinuteFactor):
    """20-day mean of the daily normalized 10% amount quantile."""

    meta = AlphaMeta(
        "qua",
        "20D mean normalized 10% quantile of minute average trade amount",
        direction=-1,
    )
    daily_column = "qua"


class MTSFactor(_RollingMinuteFactor):
    """20-day mean correlation of average trade amount and total amount."""

    filter_tradable = True

    meta = AlphaMeta(
        "mts",
        "20D mean correlation of minute average trade amount and minute turnover",
        direction=1,
    )
    daily_column = "mts"


class MTEFactor(_RollingMinuteFactor):
    """20-day mean correlation of average trade amount and minute close."""

    meta = AlphaMeta(
        "mte",
        "20D mean correlation of minute average trade amount and minute close",
        direction=-1,
    )
    daily_column = "mte"


__all__ = [
    "QUAConfig",
    "QUAContext",
    "QUAFactor",
    "MTSFactor",
    "MTEFactor",
]
