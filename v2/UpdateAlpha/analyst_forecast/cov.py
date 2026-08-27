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


__all__ = [
    "COVConfig", "COVContext", "COVFactor",
    "COVAuthorFactor", "COVOrganFactor",
]


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    with COVContext() as context:
        cov = COVFactor(context)
        trade_dates = context.data["trade_dates"]

        for trade_date in tqdm(trade_dates, desc="Updating COV"):
            cov.update(trade_date)

        pred = context.data.load("factor_pool/cov").copy()
        pred = pred[
            :context.data.axis.date_count,
            :context.data.axis.tick_count,
        ]
        tradable = context.data.read(
            "basic/tradable",
            start_date=0,
            end_date=pred.shape[0] - 1,
        )
        pct = context.data.read(
            "d_essentials/pct",
            start_date=0,
            end_date=pred.shape[0] - 1,
        ) / 100.0
        circ_mv = context.data.read(
            "d_essentials/circ_mv",
            start_date=0,
            end_date=pred.shape[0] - 1,
        )
        mv = circ_mv.copy()
        mv[1:] = mv[:-1]
        mv[0,:] = np.nan
        m_pct = np.divide(
            np.sum(mv * pct,axis=1),
            np.sum(mv, axis=1),
            out=np.full_like(pct,np.nan),
            where=np.sum(mv,axis=1)!=0
        )

        r = pct.copy()
        r[:-2] = r[2:]
        r[-2:] = np.nan
        mr = m_pct.copy()
        mr[:-2] = mr[2:]
        mr[-2:] = np.nan

        pr = np.cumsum(np.sum(cov*r,axis=1))
        alpha = pr-mr
        plt.plot(alpha)
        plt.show()
        
        
