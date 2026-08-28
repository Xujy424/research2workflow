"""Point-in-time analyst coverage factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import bottleneck as bn
import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT, get_zyyx_conn
    from .utils import _date
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import _date


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class COVConfig:
    lookback_days: int = 90
    min_window_analysts: int = 3
    stable_expansion_quantile: float = 0.7
    num_groups: int = 10
    keep_high_groups: int = 2
    close_field: str = "d_essentials/close_adj"
    market_value_field: str = "d_essentials/circ_mv"


def _residual(y, x):
    """Return cross-sectional OLS residuals with an intercept."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() <= 2:
        out[valid] = y[valid]
        return out
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    out[valid] = y[valid] - design @ coefficients
    return out


class COVContext(AlphaContext):
    """Current-FY1 annual forecasts used to measure active analyst coverage."""

    def __init__(self, root=DEFAULT_ROOT, conn=None, config=COVConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof, lookback_days=None, require_forecast=True):
        """Return unique reports carrying a finite current-FY1 annual forecast."""
        asof = _date(asof)
        lookback_days = lookback_days or self.config.lookback_days
        cache_key = (asof, lookback_days, require_forecast)
        if self._cache.get("key") == cache_key:
            return self._cache["reports"]
        start = asof - pd.Timedelta(days=lookback_days)
        forecast_filter = (
            "AND f.forecast_np IS NOT NULL "
            "AND f.gg_rating_code IN ('1','2','3','5','7')"
            if require_forecast else ""
        )
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.report_year, f.report_quarter,
            f.create_date, f.entrytime,
            f.forecast_np, f.gg_rating_code AS rating_score,
            f.target_price_ceiling, f.target_price_floor
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
            {forecast_filter}
        """
        reports = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("create_date").cast(pl.Date, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("forecast_np").cast(pl.Float64, strict=False),
            )
            .filter(
                pl.col("tick").is_not_null()
                & (
                    pl.col("forecast_np").is_finite()
                    if require_forecast else pl.lit(True)
                )
            )
            .sort([
                "tick", "author_id", 
                "report_year", "report_quarter",
                "create_date", "entrytime", 
                "report_id", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id", "report_year"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"key": cache_key, "reports": reports}
        return reports

    def excess_momentum(self, asof):
        """Return endpoint stock log momentum less market log momentum."""
        asof = _date(asof)
        axis = self.data.axis
        dates = axis.trade_dates
        
        end = axis.date_position(asof)
        pre_date = np.datetime64(asof - pd.Timedelta(days=self.config.lookback_days),"D",)
        start = int(np.searchsorted(dates, pre_date, side="right")) - 1

        result = np.full(axis.tick_count, np.nan, dtype=np.float64)
        if start < 0 or start >= end:
            return result

        start_date = dates[start]
        end_date = dates[end]
        start_close = np.asarray(
            self.data.read(self.config.close_field, start_date),
            dtype=np.float64,
        )
        end_close = np.asarray(
            self.data.read(self.config.close_field, end_date),
            dtype=np.float64,
        )
        start_mv = np.asarray(
            self.data.read(self.config.market_value_field, start_date),
            dtype=np.float64,
        )
        end_mv = np.asarray(
            self.data.read(self.config.market_value_field, end_date),
            dtype=np.float64,
        )

        start_market_valid = (
            np.isfinite(start_close)
            & np.isfinite(start_mv)
            & (start_close > 0)
            & (start_mv > 0)
        )
        end_market_valid = (
            np.isfinite(end_close)
            & np.isfinite(end_mv)
            & (end_close > 0)
            & (end_mv > 0)
        )

        start_weight = np.sum(start_mv[start_market_valid])
        end_weight = np.sum(end_mv[end_market_valid])
        if start_weight <= 0 or end_weight <= 0:
            return result
        
        start_market_price = np.sum(
            start_close[start_market_valid] * start_mv[start_market_valid]
        ) / start_weight
        end_market_price = np.sum(
            end_close[end_market_valid] * end_mv[end_market_valid]
        ) / end_weight

        if start_market_price <= 0 or end_market_price <= 0:
            return result
        market_momentum = np.log(end_market_price / start_market_price)

        valid = (
            np.isfinite(start_close)
            & np.isfinite(end_close)
            & (start_close > 0)
            & (end_close > 0)
        )
        result[valid] = (
            np.log(end_close[valid] / start_close[valid])
            - market_momentum
        )
        return result


class COVFactor(AlphaBase):
    """Square root of unique current-FY1 reports published in six months."""

    meta = AlphaMeta(
        "cov",
        "square root of unique reports with a finite current-FY1 forecast",
    )
    dependencies = ("rpt_forecast_stk",)
    column = "cov"
    coverage_field = "report_id"

    def cross_section(self, asof):
        return self.context.reports(asof).group_by("tick").agg(
            pl.col(self.coverage_field).n_unique().sqrt().alias(self.column)
        )

    def calculate(self, asof):
        values = self.context.align(
            self.cross_section(_date(asof)), self.column,
        ).astype(np.float64)

        weights = np.full(values.shape, np.nan, dtype=np.float64)
        if not np.isfinite(values).any():
            return weights

        rank = bn.nanrankdata(values)
        num_signal = np.nanmax(rank)
        stock_each_group = num_signal // self.context.config.num_groups
        cutoff = stock_each_group * (
            self.context.config.num_groups
            - self.context.config.keep_high_groups
        )
        selected = (
            np.isfinite(rank)
            & (rank > cutoff)
            & (rank <= num_signal)
        )
        total = np.sum(values[selected])
        if selected.any() and np.isfinite(total) and total > 0:
            weights[selected] = values[selected] / total
        return weights


class COVAuthorFactor(COVFactor):
    """G9/G10 portfolio based on unique analyst coverage."""

    meta = AlphaMeta(
        "cov_author",
        "top two deciles of square-root unique analyst coverage",
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    coverage_field = "author_id"


class COVOrganFactor(COVFactor):
    """G9/G10 portfolio based on unique institution coverage."""

    meta = AlphaMeta(
        "cov_organ",
        "top two deciles of square-root unique institution coverage",
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    coverage_field = "organ_id"



class _COVAuthorChangeFactor(COVFactor):
    """Base for analyst-set changes across two adjacent 90-day windows."""

    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    metric = None

    def author_stats(self, asof):
        asof = _date(asof)
        days = self.context.config.lookback_days
        split, start = (
            asof - pd.Timedelta(days=days),
            asof - pd.Timedelta(days=2 * days),
        )
        reports = self.context.reports(
            asof, lookback_days=2 * days, require_forecast=False
        )
        recent = reports.filter(
            (pl.col("create_date") > split) & (pl.col("create_date") <= asof)
        ).group_by("tick").agg(
            pl.col("author_id").unique().alias("a1")
        )
        previous = reports.filter(
            (pl.col("create_date") > start) & (pl.col("create_date") <= split)
        ).group_by("tick").agg(
            pl.col("author_id").unique().alias("a0")
        )
        minimum = self.context.config.min_window_analysts
        return recent.join(previous, on="tick", how="inner").with_columns(
            pl.col("a1").list.len().alias("n1"),
            pl.col("a0").list.len().alias("n0"),
            pl.col("a1").list.set_intersection("a0").list.len().alias("nc"),
        ).filter(
            (pl.col("n0") >= minimum) & (pl.col("n1") >= minimum)
        ).with_columns(
            (pl.col("n1") - pl.col("nc")).alias("n_new"),
            (pl.col("n0") - pl.col("nc")).alias("n_exit"),
        ).with_columns(
            (pl.col("nc") / pl.col("n1")).alias("current_overlap"),
            (pl.col("nc") / pl.col("n0")).alias("retention"),
            (pl.col("n_new") / pl.col("n1")).alias("new_ratio"),
            (pl.col("n_new") / (1 + pl.col("n0"))).alias("new_intensity"),
            (pl.col("n_exit") / pl.col("n0")).alias("exit_ratio"),
            (
                pl.col("nc")
                / (pl.col("n0") + pl.col("n1") - pl.col("nc"))
            ).alias("jaccard"),
            (2 * pl.col("nc") / (pl.col("n0") + pl.col("n1"))).alias("dice"),
            (pl.col("n1").log1p() - pl.col("n0").log1p()).alias(
                "coverage_growth"
            ),
        ).with_columns(
            (pl.col("retention") * pl.col("n_new").log1p()).alias(
                "stable_expansion"
            ),
            (
                pl.col("exit_ratio") / (1 + pl.col("n_new"))
            ).alias("coverage_decay"),
        )

    def cross_section(self, asof):
        return self.author_stats(asof).select(
            "tick", pl.col(self.metric).alias(self.column)
        )


class COVAuthorOverlapFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_author_overlap", "recent analyst current-overlap ratio")
    column, metric = "cov_author_overlap", "current_overlap"


class COVCurrentCoverageFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_current_coverage", "log current analyst coverage")
    column, metric = "cov_current_coverage", "n1"

    def cross_section(self, asof):
        return self.author_stats(asof).select(
            "tick", pl.col("n1").log1p().alias(self.column)
        )


class COVCoverageGrowthFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_coverage_growth", "log analyst coverage growth")
    column, metric = "cov_coverage_growth", "coverage_growth"


class COVRetentionFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_retention", "prior analyst retention rate")
    column, metric = "cov_retention", "retention"


class COVNewRatioFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_new_ratio", "new analyst share")
    column, metric = "cov_new_ratio", "new_ratio"


class COVNewIntensityFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_new_intensity", "new analysts relative to prior coverage")
    column, metric = "cov_new_intensity", "new_intensity"


class COVExitRatioFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_exit_ratio", "prior analyst exit rate")
    column, metric = "cov_exit_ratio", "exit_ratio"


class COVStableExpansionFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_stable_expansion", "retention times log new analysts")
    column, metric = "cov_stable_expansion", "stable_expansion"


class COVCoverageDecayFactor(_COVAuthorChangeFactor):
    """Severity of coverage decay conditional on at least one analyst exit."""

    meta = AlphaMeta(
        "cov_coverage_decay",
        "positive analyst exit rate discounted by new analysts",
    )
    column, metric = "cov_coverage_decay", "coverage_decay"

    def cross_section(self, asof):
        return self.author_stats(asof).filter(
            pl.col("n_exit") > 0
        ).select(
            "tick", pl.col(self.metric).alias(self.column)
        )


class _COVCoverageEventFactor(_COVAuthorChangeFactor):
    def event_mask(self, stats, quantile):
        raise NotImplementedError

    def cross_section(self, asof):
        stats = self.author_stats(asof)
        if stats.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        selected = self.event_mask(
            stats, self.context.config.stable_expansion_quantile
        )
        return stats.filter(selected).select("tick").with_columns(
            pl.lit(1.0).alias(self.column)
        )


class COVExpansionEventFactor(_COVCoverageEventFactor):
    meta = AlphaMeta(
        "cov_expansion_event", "high retention with high new-analyst intensity"
    )
    column = "cov_expansion_event"

    def event_mask(self, stats, quantile):
        return (
            (pl.col("retention") > stats["retention"].quantile(quantile))
            & (
                pl.col("new_intensity")
                > stats["new_intensity"].quantile(quantile)
            )
        )


class COVDecayEventFactor(_COVCoverageEventFactor):
    meta = AlphaMeta(
        "cov_decay_event", "high analyst exit with low replacement intensity"
    )
    column = "cov_decay_event"

    def event_mask(self, stats, quantile):
        return (
            (pl.col("exit_ratio") > stats["exit_ratio"].quantile(quantile))
            & (
                pl.col("new_intensity")
                < stats["new_intensity"].quantile(1 - quantile)
            )
        )





__all__ = [
    "COVConfig", "COVContext", "COVFactor",
    "COVAuthorFactor", "COVOrganFactor", "COVAuthorOverlapFactor",
    "COVCurrentCoverageFactor", "COVCoverageGrowthFactor",
    "COVRetentionFactor", "COVNewRatioFactor", "COVNewIntensityFactor",
    "COVExitRatioFactor", "COVStableExpansionFactor",
    "COVCoverageDecayFactor", "COVExpansionEventFactor",
    "COVDecayEventFactor",
]
