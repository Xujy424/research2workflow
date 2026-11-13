"""Analyst-consensus standardized unexpected earnings, ALSUE0 and ALSUE1.

ALSUE0 splits annual consensus profit evenly across four quarters. ALSUE1
uses each firm's mean quarterly profit shares over the preceding three fully
observed fiscal years, falling back to equal splits unless all shares are
well-defined and historical annual profits are positive. All inputs are
filtered by their publication/entry timestamps at the requested as-of date.
"""

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
    from ...UpdateData.config import ROOT, get_jy_conn, get_zyyx_conn
    from .sue import DEFAULT_ROOT
    from .utils import _date
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_jy_conn, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import _date


@dataclass(frozen=True)
class ALSUEConfig:
    lookback_years: int = 4
    historical_years: int = 3
    min_analysts: int = 2
    report_quarter: int = 4


class ALSUEContext(AlphaContext):
    """Point-in-time annual forecast consensus and quarterly actuals."""

    def __init__(self, root=DEFAULT_ROOT, jy_conn=None, zyyx_conn=None,
                 config=ALSUEConfig(), universe="self"):
        self.config = config
        self.jy_conn = jy_conn or get_jy_conn()
        self.zyyx_conn = zyyx_conn or get_zyyx_conn()
        self._owns_jy = jy_conn is None
        self._owns_zyyx = zyyx_conn is None
        self._cache = {}
        self._forecast_raw = pl.DataFrame()
        self._forecast_through = None
        self._actual_raw = pl.DataFrame()
        self._actual_through = None
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def close(self):
        self.data.close()
        if self._owns_jy and self.jy_conn is not None:
            self.jy_conn.close()
        if self._owns_zyyx and self.zyyx_conn is not None:
            self.zyyx_conn.close()

    def forecast_consensus(self, asof):
        asof = _date(asof)
        key = ("forecast", asof)
        if key in self._cache:
            return self._cache[key]
        overlap_days = 8  # entrytime may lag create_date by up to seven days
        start = (
            asof - pd.Timedelta(days=self.config.lookback_years * 366)
            if self._forecast_through is None or asof < self._forecast_through
            else self._forecast_through - pd.Timedelta(days=overlap_days)
        )
        sql = f"""
        SELECT f.id, f.report_id, f.stock_code AS tick, f.organ_id,
               ra.author_id, f.report_year, f.create_date, f.entrytime,
               f.forecast_np
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id=f.report_id
        WHERE f.report_year BETWEEN {asof.year-self.config.lookback_years}
                                AND {asof.year+1}
          AND f.report_quarter={self.config.report_quarter}
          AND f.create_date BETWEEN '{start}' AND '{asof}'
          AND f.entrytime<DATEADD(day,1,'{asof}')
          AND f.create_date<=f.entrytime
          AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND 7
          AND f.forecast_np IS NOT NULL
          AND (f.reliability>=5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
        """
        delta = pl.read_database(sql, self.zyyx_conn, infer_schema_length=None)
        if not delta.is_empty():
            delta = (delta.with_columns(
                pl.col("tick").cast(pl.String).str.zfill(6),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("report_year").cast(pl.Int32, strict=False),
                pl.col("create_date").cast(pl.Datetime, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("forecast_np").cast(pl.Float64, strict=False),
            ).filter(pl.col("forecast_np").is_finite()))
        if not delta.is_empty():
            self._forecast_raw = pl.concat(
                [self._forecast_raw, delta], how="diagonal_relaxed"
            ).sort(["tick", "report_year", "author_id", "create_date", "entrytime", "id"])
            self._forecast_raw = self._forecast_raw.unique(
                ["report_id", "tick", "organ_id", "author_id", "report_year"],
                keep="last", maintain_order=True,
            )
        self._forecast_through = asof
        if self._forecast_raw.is_empty():
            empty = pl.DataFrame(schema={
                "tick": pl.String, "report_year": pl.Int32,
                "consensus": pl.Float64, "forecast_std": pl.Float64,
                "analyst_count": pl.UInt32,
            })
            self._cache[key] = empty
            return empty
        frame = self._forecast_raw.filter(
            (pl.col("create_date").dt.date() <= asof)
            & (pl.col("entrytime").dt.date() <= asof)
            & (pl.col("report_year") >= asof.year-self.config.lookback_years)
            & (pl.col("report_year") <= asof.year+1)
        ).sort(["tick", "report_year", "author_id", "create_date", "entrytime", "id"])
        if not frame.is_empty():
            frame = frame.unique(
                ["tick", "report_year", "author_id"], keep="last", maintain_order=True
            )
            dispersion = frame.group_by(["tick", "organ_id", "report_year"]).agg(
                pl.col("forecast_np").mean().alias("institution_mean")
            ).group_by(["tick", "report_year"]).agg(
                pl.col("institution_mean").std(ddof=1).alias("forecast_std")
            )
            frame = (frame.group_by(["tick", "organ_id", "report_year"])
                     .agg(pl.col("forecast_np").mean().alias("institution_forecast"))
                     .group_by(["tick", "report_year"])
                     .agg(pl.col("institution_forecast").mean().alias("consensus")))
            frame = frame.join(dispersion, on=["tick", "report_year"], how="left")
            counts = frame.group_by(["tick", "report_year"]).agg(pl.len().alias("analyst_count"))
            frame = frame.join(counts, on=["tick", "report_year"], how="left").filter(
                pl.col("analyst_count") >= self.config.min_analysts
            )
        self._cache[key] = frame
        return frame

    def actual_quarters(self, asof):
        asof = _date(asof)
        key = ("actual", asof)
        if key in self._cache:
            return self._cache[key]
        start = asof - pd.DateOffset(years=self.config.lookback_years + 4)
        overlap_days = 1
        query_start = (
            start
            if self._actual_through is None or asof < self._actual_through
            else self._actual_through - pd.Timedelta(days=overlap_days)
        )
        selects = []
        for table in ("LC_IncomeStatementAll", "LC_STIBIncomeState"):
            bulletin = "AND f.BulletinType IN (20,30)" if table.endswith("All") else ""
            selects.append(f"""
                SELECT f.ID AS id, s.SecuCode AS tick, f.EndDate AS end_date,
                       f.InfoPublDate AS publish_date,
                       f.NPParentCompanyOwners AS net_profit_accum
                FROM dbo.{table} f JOIN dbo.SecuMain s ON f.CompanyCode=s.CompanyCode
                WHERE f.EndDate>='{start.date()}' AND f.InfoPublDate BETWEEN '{query_start}' AND '{asof}'
                  AND f.IfMerged=1 AND f.IfAdjusted=2 AND f.IfComplete=1 {bulletin}
                  AND s.SecuCategory=1 AND s.SecuMarket IN (83,90)
            """)
        delta = pl.read_database(" UNION ALL ".join(selects), self.jy_conn, infer_schema_length=None)
        if not delta.is_empty():
            delta = (delta.with_columns(
                pl.col("tick").cast(pl.String).str.zfill(6),
                pl.col("end_date").cast(pl.Datetime, strict=False).dt.date(),
                pl.col("publish_date").cast(pl.Datetime, strict=False).dt.date(),
                pl.col("net_profit_accum").cast(pl.Float64, strict=False),
            ).filter(pl.col("end_date").dt.month().is_in([3,6,9,12])))
        if not delta.is_empty():
            self._actual_raw = pl.concat(
                [self._actual_raw, delta], how="diagonal_relaxed"
            )
        self._actual_through = asof
        if self._actual_raw.is_empty():
            empty = pl.DataFrame(schema={
                "tick": pl.String, "end_date": pl.Date,
                "publish_date": pl.Date, "net_profit_accum": pl.Float64,
                "year": pl.Int32, "quarter": pl.Int32,
                "quarter_profit": pl.Float64,
            })
            self._cache[key] = empty
            return empty
        frame = self._actual_raw.filter(
            (pl.col("publish_date") <= asof)
            & (pl.col("end_date") >= start.date())
        )
        if not frame.is_empty():
            frame = (frame
              .sort(["tick", "end_date", "publish_date", "id"])
              .group_by(["tick", "end_date"], maintain_order=True)
              .agg(pl.col("publish_date").last(), pl.col("net_profit_accum").drop_nulls().last())
              .with_columns(
                  pl.col("end_date").dt.year().alias("year"),
                  (pl.col("end_date").dt.month()//3).alias("quarter"),
              ))
            lookup = frame.select("tick", "year", "quarter", "net_profit_accum")
            # Accumulated values are differenced against the immediately
            # preceding quarter within the same fiscal year.
            prev = lookup.with_columns(
                (pl.col("quarter")+1).alias("quarter")
            ).rename({"net_profit_accum":"prior_accum"})
            frame = frame.join(prev, on=["tick","year","quarter"], how="left").with_columns(
                pl.when(pl.col("quarter")==1).then(pl.col("net_profit_accum"))
                .otherwise(pl.col("net_profit_accum")-pl.col("prior_accum"))
                .alias("quarter_profit")
            )
        self._cache[key] = frame
        return frame

    def surprises(self, asof, method):
        forecasts = self.forecast_consensus(asof)
        actuals = self.actual_quarters(asof)
        schema = {"tick": pl.String, "value": pl.Float64}
        if forecasts.is_empty() or actuals.is_empty():
            return pl.DataFrame(schema=schema)
        # Select the report period whose actual result first became visible most recently.
        events = actuals.filter(pl.col("publish_date") <= _date(asof)).sort(
            ["tick", "end_date", "publish_date"]
        ).unique("tick", keep="last", maintain_order=True)
        joined = events.join(
            forecasts, left_on=["tick", "year"], right_on=["tick", "report_year"], how="inner"
        ).filter(
            (pl.col("publish_date") <= _date(asof))
            & (pl.col("create_date") <= pl.col("publish_date").cast(pl.Datetime))
            & (pl.col("entrytime") < pl.col("publish_date").cast(pl.Datetime))
        )
        if method == 1:
            history = actuals.filter(
                (pl.col("year") >= pl.col("year").max()-self.config.historical_years)
                & (pl.col("year") < pl.col("year").max())
            )
            shares = history.group_by(["tick", "year"]).agg(
                pl.col("quarter_profit").sum().alias("annual_profit"),
                pl.col("quarter").n_unique().alias("quarter_count"),
            )
            history = history.join(shares, on=["tick", "year"], how="left").filter(
                (pl.col("annual_profit") > 0) & (pl.col("quarter_count") == 4)
                & (pl.col("quarter_profit") > 0)
            ).with_columns((pl.col("quarter_profit")/pl.col("annual_profit")).alias("share"))
            weights = history.group_by(["tick", "quarter"]).agg(
                pl.col("share").mean().alias("historical_share"),
                pl.col("year").n_unique().alias("history_years"),
            )
            joined = joined.join(weights, on=["tick", "quarter"], how="left").with_columns(
                pl.when((pl.col("history_years") >= self.config.historical_years)
                        & pl.col("historical_share").is_finite())
                .then(pl.col("historical_share")).otherwise(0.25).alias("split_weight")
            )
        else:
            joined = joined.with_columns(pl.lit(0.25).alias("split_weight"))
        weight = pl.col("split_weight")
        # Convert annual consensus to quarterly expected profit, then standardize
        # the per-analyst annual consensus dispersion on the same split basis.
        joined = joined.with_columns(
            (pl.col("consensus") * weight).alias("expected_quarter"),
            (pl.col("forecast_std") * weight).alias("expected_std"),
        ).with_columns(
            pl.when(pl.col("expected_std") > 0)
            .then((pl.col("quarter_profit")-pl.col("expected_quarter"))/pl.col("expected_std"))
            .otherwise(None).alias("value")
        )
        return joined.select("tick", "value").filter(pl.col("value").is_finite())


class _ALSUEFactor(AlphaBase):
    method = 0
    column = ""
    dependencies = ("rpt_forecast_stk", "LC_IncomeStatementAll", "LC_STIBIncomeState")

    def calculate(self, asof):
        return self.context.align(self.context.surprises(asof, self.method), "value")


class ALSUE0Factor(_ALSUEFactor):
    meta = AlphaMeta("alsue0", "analyst SUE with equal 25 percent quarterly split")
    method = 0


class ALSUE1Factor(_ALSUEFactor):
    meta = AlphaMeta("alsue1", "analyst SUE with three-year historical quarterly split")
    method = 1


def calculate_alsue_family(asof, root=ROOT, jy_conn=None, zyyx_conn=None,
                            config=ALSUEConfig(), universe="self"):
    with ALSUEContext(root, jy_conn, zyyx_conn, config, universe) as context:
        return {cls.meta.name: cls(context).run(asof) for cls in (ALSUE0Factor, ALSUE1Factor)}


__all__ = ["ALSUEConfig", "ALSUEContext", "ALSUE0Factor", "ALSUE1Factor", "calculate_alsue_family"]
