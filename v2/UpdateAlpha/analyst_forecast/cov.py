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

    def reports(self, asof):
        """Return unique reports carrying a finite current-FY1 annual forecast."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["reports"]
        start = asof - pd.Timedelta(days=self.config.lookback_days)
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
            AND f.forecast_np IS NOT NULL
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
            AND f.gg_rating_code IN ('1','2','3','5','7')
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
            .filter(pl.col("tick").is_not_null() & pl.col("forecast_np").is_finite())
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
        self._cache = {"asof": asof, "reports": reports}
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

    def cross_section(self, asof):
        return self.context.reports(asof).group_by("tick").agg(
            pl.col("report_id").n_unique().sqrt().alias(self.column)
        )

    def calculate(self, asof):
        values = self.context.align(self.cross_section(_date(asof)), self.column)
        return np.nan_to_num(values, nan=0.0)


class CovExMomFactor(COVFactor):
    """Analyst coverage stripped of same-window excess momentum."""

    meta = AlphaMeta(
        "cov_ex_mom",
        "analyst coverage residualized against same-window excess momentum",
    )
    dependencies = (
        "rpt_forecast_stk",
        "rpt_report_author",
        "d_essentials/close_adj",
        "d_essentials/circ_mv",
    )

    def calculate(self, asof):
        asof = _date(asof)
        coverage = self.context.align(
            self.cross_section(asof), self.column,
        ).astype(np.float64)
        momentum = self.context.excess_momentum(asof)
        values = _residual(coverage, momentum)
        return np.nan_to_num(values, nan=0.0)


__all__ = ["COVConfig", "COVContext", "COVFactor", "CovExMomFactor"]
