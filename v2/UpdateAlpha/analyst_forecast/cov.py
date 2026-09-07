"""Point-in-time analyst coverage factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

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
    close_field: str = "d_essentials/close_adj"
    market_value_field: str = "d_essentials/circ_mv"


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
                & ( pl.col("forecast_np").is_finite() if require_forecast else pl.lit(True) )
            )
            .sort([
                "tick", "author_id", 
                "report_year", "report_quarter",
                "create_date", "entrytime", 
                "report_id", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"key": cache_key, "reports": reports}
        return reports

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
        return values

class COVAuthorFactor(COVFactor):
    meta = AlphaMeta(
        "cov_author",
        "top two deciles of square-root unique analyst coverage",
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    coverage_field = "author_id"

class COVOrganFactor(COVFactor):
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
            pl.col("author_id").unique().alias("cov1")
        )
        previous = reports.filter(
            (pl.col("create_date") > start) & (pl.col("create_date") <= split)
        ).group_by("tick").agg(
            pl.col("author_id").unique().alias("cov0")
        )
        minimum = self.context.config.min_window_analysts
        return recent.join(previous, on="tick", how="inner").with_columns(
            pl.col("cov1").list.len().alias("n1"),
            pl.col("cov0").list.len().alias("n0"),
            pl.col("cov1").list.set_intersection("cov0").list.len().alias("nc"),
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
            (pl.col("nc") / (pl.col("n0") + pl.col("n1") - pl.col("nc"))).alias("jaccard"),
            (2 * pl.col("nc") / (pl.col("n0") + pl.col("n1"))).alias("dice"),
            (pl.col("n1").log1p() - pl.col("n0").log1p()).alias("coverage_growth"),
        ).with_columns(
            (pl.col("retention") * pl.col("n_new").log1p()).alias("stable_expansion"),
            (pl.col("exit_ratio") / (1 + pl.col("n_new"))).alias("coverage_decay"),
        )

    def cross_section(self, asof):
        return self.author_stats(asof).select(
            "tick", pl.col(self.metric).alias(self.column)
        )


class COVCurrentFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta("cov_current", "log current analyst coverage")
    column, metric = "cov_current", "n1"
    def cross_section(self, asof):
        return self.author_stats(asof).select(
            "tick", pl.col("n1").log1p().alias(self.column)
        )


class COVDecayOptimizedFactor(_COVAuthorChangeFactor):
    meta = AlphaMeta(
        "cov_decay_optimized",
        "positive analyst exit rate discounted by new analysts",
    )
    column, metric = "cov_decay_optimized", "coverage_decay"
    def cross_section(self, asof):
        return self.author_stats(asof).filter(
            pl.col("n_exit") > 0
        ).select(
            "tick", pl.col(self.metric).alias(self.column)
        )


__all__ = [
    "COVConfig", "COVContext", "COVFactor",
    "COVAuthorFactor", "COVOrganFactor", "COVCurrentFactor",
    "COVDecayOptimizedFactor",
]