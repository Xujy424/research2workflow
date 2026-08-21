"""AFR, PAFR, expected inertia and expected volatility factors."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from .alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..GetData import DataPool
    from ..UpdateData.config import ROOT, get_zyyx_conn
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn


def _date(value):
    return pd.Timestamp(value).date()


def _residual(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    out = np.full(y.shape, np.nan)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        out[ok] = y[ok]
        return out
    design = np.c_[np.ones(ok.sum()), x[ok]]
    out[ok] = y[ok] - design @ np.linalg.lstsq(design, y[ok], rcond=None)[0]
    return out


def _zscore(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    std = np.nanstd(x[ok], ddof=1) if ok.sum() > 1 else np.nan
    if std > 0:
        out[ok] = (x[ok] - np.nanmean(x[ok])) / std
    return out




def _analyst_values(frame, value):
    """Keep each analyst's latest valid observation for every stock."""
    keys = ["tick", "author_id"]
    order = [
        column
        for column in [*keys, "create_date", "entrytime", "id"]
        if column in frame.columns
    ]
    return (
        frame.filter(pl.col(value).is_finite())
        .sort(order)
        .unique(keys, keep="last", maintain_order=True)
    )


def _equal_weight(frame, value, alias="value"):
    """Use analysts' latest values, then equal-weight analysts and institutions."""
    authors = _analyst_values(frame, value)
    return (
        authors.group_by(["tick", "organ_id"])
        .agg(pl.col(value).mean().alias(value))
        .group_by("tick")
        .agg(pl.col(value).mean().alias(alias))
    )


def _analyst_weight(frame, value, weights, alias="value"):
    """Use latest values, weight analysts, then equal-weight institutions."""
    if not isinstance(weights, pl.DataFrame):
        weights = pl.from_pandas(weights)
    required = {"author_id", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"analyst weights missing columns: {sorted(missing)}")

    weights = weights.select(
        pl.col("author_id").cast(pl.Int64, strict=False),
        pl.col("weight").cast(pl.Float64, strict=False),
    ).unique("author_id", keep="last")
    authors = _analyst_values(frame, value).join(
        weights, on="author_id", how="left"
    ).with_columns(
        pl.when(pl.col("weight").is_finite() & (pl.col("weight") > 0))
        .then(pl.col("weight"))
        .otherwise(0.0)
        .alias("weight")
    )
    institutions = authors.group_by(["tick", "organ_id"]).agg(
        (pl.col(value) * pl.col("weight")).sum().alias("weighted_sum"),
        pl.col("weight").sum().alias("weight_sum"),
        pl.col(value).mean().alias("equal_value"),
    ).with_columns(
        pl.when(pl.col("weight_sum") > 0)
        .then(pl.col("weighted_sum") / pl.col("weight_sum"))
        .otherwise(pl.col("equal_value"))
        .alias(value)
    )
    return institutions.group_by("tick").agg(
        pl.col(value).mean().alias(alias)
    )


def _aggregate(frame, value, weights=None, alias="value"):
    if weights is None:
        return _equal_weight(frame, value, alias)
    return _analyst_weight(frame, value, weights, alias)

@dataclass(frozen=True)
class AFRConfig:
    lookback_days: int = 90
    max_history_days: int = 183
    volatility_days: int = 365
    min_institutions: int = 3
    revision_limit: float = 0.25
    close_field: str = "d_essentials/close_adj"
    market_value_field: str = "d_essentials/circ_mv"
    industry_field: str = "industry/industry"


class AFRContext(AlphaContext):
    """Share point-in-time SQL and local matrix reads among factors."""

    def __init__(
        self, root=ROOT, conn=None, config=AFRConfig(), analyst_weights=None
    ):
        self.config = config
        self.analyst_weights = analyst_weights
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        super().__init__(DataPool(root, asset="stock"))
        self._cache = {}

    def close(self):
        super().close()
        if self._owns_conn:
            self.conn.close()

    def reports(self, asof):
        asof = _date(asof)
        if asof in self._cache:
            return self._cache[asof]
        start = asof - pd.Timedelta(
            days=self.config.volatility_days + self.config.max_history_days
        )
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime, f.report_year, f.forecast_np,
            f.target_price_ceiling, f.target_price_floor, f.current_price
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND f.create_date <= f.entrytime
            AND f.report_quarter = 4
            AND f.report_year = YEAR(f.create_date) + 1
            AND f.forecast_np IS NOT NULL
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
            AND (f.reliability >= 5 OR f.reliability IS NULL)
        """
        frame = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("create_date").cast(pl.Datetime, strict=False).dt.date(),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                *[
                    pl.col(c).cast(pl.Float64, strict=False)
                    for c in (
                        "forecast_np", "target_price_ceiling","target_price_floor", "current_price",
                    )
                ],
            )
            .sort([
                "tick", "author_id", "report_year",
                "create_date", "entrytime", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id", "report_year"],
                keep="last", maintain_order=True,
            )
        )
        self._cache[asof] = frame
        return frame

    def align(self, frame, value="value"):
        """Align a Polars tick/value cross-section to valid local ticks."""
        axis = self.data.axis
        out = np.full(axis.tick_count, np.nan, dtype=np.float32)
        positions = {str(tick): i for i, tick in enumerate(axis.ticks)}
        for tick, item in frame.select("tick", value).iter_rows():
            position = positions.get(str(tick).zfill(6))
            if position is not None and item is not None:
                out[position] = item
        return out

    def empty(self):
        return np.full(self.data.axis.tick_count, np.nan, dtype=np.float32)

    def field_values(self, field, dates, ticks):
        """Read paired date/tick values at the latest available trade date."""
        axes = self.data.axis
        rows = np.searchsorted(
            axes.trade_dates,
            np.asarray(dates, dtype="datetime64[D]"),
            side="right",
        ) - 1
        cols = axes.tick_positions(ticks)
        values = np.full(len(rows), np.nan)
        valid = (rows >= 0) & (rows < axes.date_count)
        matrix = self.data.load(field)
        values[valid] = matrix[rows[valid], cols[valid]]
        return values

    def price(self, dates, ticks):
        return self.field_values(self.config.close_field, dates, ticks)

    def industry(self, asof, ticks):
        return np.asarray(
            self.data.read(self.config.industry_field, asof, ticks=ticks), float
        )


class AFRFactor(AlphaBase):
    meta = AlphaMeta("afr", "analyst forecast net-profit revision")
    dependencies = ("rpt_forecast_stk",)

    def event_values(self, asof):
        cfg = self.context.config
        keys = ["tick", "author_id", "report_year"]
        events = self.context.reports(asof).with_columns(
            pl.col("forecast_np").shift().over(keys).alias("prior_np"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        ).with_columns(
            (pl.col("create_date") - pl.col("prior_date")).dt.total_days().alias("gap_days"),
            ((pl.col("forecast_np") - pl.col("prior_np")) / pl.col("prior_np").abs()).clip(-cfg.revision_limit, cfg.revision_limit).alias("afr_event"),
        )
        return events.filter(
            pl.col("prior_date").is_not_null()  # 去掉首次预测，变相保证一个分析师对某股票至少两次预测
            & (pl.col("create_date") >= _date(asof) - pd.Timedelta(days=cfg.lookback_days))  # 回看90天
            & pl.col("afr_event").is_finite()
            & (pl.col("organ_id").n_unique().over("tick") >= cfg.min_institutions)  # 剔除少于三份预测报告的股票
        )

    def calculate(self, asof):
        asof = _date(asof)
        frame = _aggregate(
            self.event_values(asof), "afr_event", self.context.analyst_weights,
        )
        return self.context.align(frame)


class PAFRFactor(AFRFactor):
    meta = AlphaMeta("pafr", "AFR stripped of two excess-momentum effects")
    dependencies = (
        "rpt_forecast_stk",
        "d_essentials/close_adj",
        "d_essentials/circ_mv",
    )

    def excess_momentum(self, starts, ends, ticks):
        values = np.log(
            self.context.price(ends, ticks)
            / self.context.price(starts, ticks)
        )
        weights = self.context.field_values(
            self.context.config.market_value_field, ends, ticks
        )
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        benchmark = (
            np.average(values[valid], weights=weights[valid])
            if valid.any()
            else np.nan
        )
        return values - benchmark

    def calculate(self, asof):
        asof = _date(asof)
        events = self.event_values(asof)
        if events.is_empty():
            return self.context.empty()
        ticks = events["tick"].to_list()
        between = self.excess_momentum(
            events["prior_date"], events["create_date"], ticks
        )
        post = self.excess_momentum(
            events["create_date"], [asof] * len(events), ticks
        )
        events = events.with_columns(
            pl.Series("raw", _residual(events["afr_event"], between)),
            pl.Series("post", post),
        )
        stock = _aggregate(
            events, "raw", self.context.analyst_weights, "raw",
        ).join(
            _aggregate(
                events, "post", self.context.analyst_weights, "post",
            ),
            on="tick",
        )
        stock = stock.with_columns(
            pl.Series("value", _residual(stock["raw"], stock["post"]))
        )
        return self.context.align(stock)


class ExpectedInertiaFactor(AlphaBase):
    meta = AlphaMeta("expected_inertia", "Implied valuation multiple revision")
    dependencies = ("rpt_forecast_stk",)

    def event_values(self, asof, lookback_days=None):
        cfg = self.context.config
        keys = ["tick", "author_id", "report_year"]
        events = self.context.reports(asof).with_columns(
            pl.mean_horizontal(
                "target_price_ceiling", "target_price_floor"
            ).alias("target_mid")
        ).with_columns(
            pl.col("forecast_np").shift().over(keys).alias("prior_np"),
            pl.col("target_mid").shift().over(keys).alias("prior_target"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        ).with_columns(
            (pl.col("create_date") - pl.col("prior_date"))
            .dt.total_days().alias("gap_days"),
            (
                (pl.col("target_mid") / pl.col("prior_target")).log()
                - (pl.col("forecast_np") / pl.col("prior_np")).log()
            ).alias("inertia_event"),
        )
        days = cfg.lookback_days if lookback_days is None else lookback_days
        return events.filter(
            pl.col("prior_date").is_not_null()
            & pl.col("gap_days").is_between(1, cfg.max_history_days)
            & (
                pl.col("create_date")
                >= _date(asof) - pd.Timedelta(days=days)
            )
            & pl.col("inertia_event").is_finite()
        )

    def calculate(self, asof):
        cfg = self.context.config
        events = self.event_values(asof).filter(
            pl.col("organ_id").n_unique().over("tick")
            >= cfg.min_institutions
        )
        frame = _aggregate(
            events, "inertia_event", self.context.analyst_weights,
        )
        return self.context.align(frame)


class ExpectedVolatilityFactor(ExpectedInertiaFactor):
    meta = AlphaMeta("expected_volatility", "Industry and time-series expected vol")
    dependencies = ("rpt_forecast_stk", "industry/industry")

    def calculate(self, asof):
        asof = _date(asof)
        cfg = self.context.config
        events = self.event_values(asof, cfg.volatility_days)
        recent = events.filter(
            pl.col("create_date") >= asof - pd.Timedelta(days=cfg.lookback_days)
        )
        if recent.is_empty():
            return self.context.empty()
        current = _aggregate(
            recent, "inertia_event", self.context.analyst_weights, "inertia",
        )
        analyst_vol = events.group_by(
            ["tick", "organ_id", "author_id"]
        ).agg(
            pl.col("inertia_event").std(ddof=1).alias("time_vol")
        )
        time_vol = _aggregate(
            analyst_vol, "time_vol", self.context.analyst_weights, "time_vol"
        )
        frame = current.join(time_vol, on="tick", how="left").with_columns(
            pl.Series("industry", self.context.industry(asof, current["tick"]))
        ).with_columns(
            pl.col("inertia").std(ddof=1).over("industry").alias("industry_vol")
        )
        # Buy industry disagreement, penalise unstable stock histories.
        value = (_zscore(frame["industry_vol"]) - _zscore(frame["time_vol"])) / 2
        frame = frame.with_columns(pl.Series("value", value))
        return self.context.align(frame)


def _factor_classes():
    return AFRFactor, PAFRFactor, ExpectedInertiaFactor, ExpectedVolatilityFactor


def calculate_afr_family(
    asof, root=ROOT, conn=None, config=AFRConfig(), analyst_weights=None
):
    """Return four full-axis float32 cross-sections without writing files."""
    with AFRContext(root, conn, config, analyst_weights) as context:
        return {cls.meta.name: cls(context).run(asof) for cls in _factor_classes()}


def update_afr_family(
    asof, root=ROOT, conn=None, config=AFRConfig(),
    folder="alpha", analyst_weights=None,
):
    """Calculate and write all factors into date-by-tick matrices."""
    with AFRContext(root, conn, config, analyst_weights) as context:
        return {cls.meta.name: cls(context).update(asof, folder) for cls in _factor_classes()}


__all__ = [
    "AFRConfig", "AFRContext", "AFRFactor", "PAFRFactor",
    "ExpectedInertiaFactor", "ExpectedVolatilityFactor",
    "calculate_afr_family", "update_afr_family",
]


if __name__ == "__main__":
    # for name, frame in calculate_afr_family(date.today()).items():
    #     print(name, frame.shape)
    
    afr = AFRFactor(AFRContext())
    afr.calclate()