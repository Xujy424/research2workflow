"""Robust analyst forecast-dispersion factor family."""

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
    from .utils import _date, latest_analyst_values
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import _date, latest_analyst_values


@dataclass(frozen=True)
class DISPConfig:
    lookback_days: int = 180
    min_dispersion_analysts: int = 5
    min_effective_analysts: float = 3.0
    min_institutions: int = 3
    freshness_half_life_days: float = 45.0
    sequence_lookback_days: int = 90
    stable_floor_quantile: float = 0.20



class DISPContext(AlphaContext):
    """Point-in-time FY1 forecasts shared only by the DISP factor family."""

    def __init__(self, root=ROOT, conn=None, config=DISPConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def report_history(self, asof):
        """Return all finite FY1 annual forecasts in the configured lookback."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["history"]
        start = asof - pd.Timedelta(days=self.config.lookback_days)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime, f.forecast_np
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND f.report_year = {asof.year}
            AND f.report_quarter = 4
            AND f.forecast_np IS NOT NULL
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
        """
        history = (
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
                "tick", "author_id", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"asof": asof, "history": history}
        return history

    def reports(self, asof):
        """Return each analyst's latest forecast without discarding cached history."""
        history = self.report_history(asof)
        reports = self._cache.get("reports")
        if reports is None:
            reports = latest_analyst_values(history, "forecast_np")
            self._cache["reports"] = reports
        return reports

class _DISPFactor(AlphaBase):
    column = ""

    def cross_section(self, asof):
        raise NotImplementedError

    def calculate(self, asof):
        frame = self.cross_section(_date(asof))
        return self.context.align(frame, self.column)

    def normalize_dispersion(self, stats, column):
        """Divide dispersion by abs(center), protected by a cross-sectional floor."""
        centers = stats.filter(
            pl.col("center").is_finite() & (pl.col("center").abs() > 0)
        ).select(
            pl.col("center").abs()
            .quantile(self.context.config.stable_floor_quantile)
            .alias("floor")
        )
        if centers.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, column: pl.Float64})
        
        floor = centers.item(0, "floor")
        if floor is None or not np.isfinite(floor):
            return pl.DataFrame(schema={"tick": pl.String, column: pl.Float64})
        
        return stats.with_columns(
            (
                pl.col("dispersion")
                / pl.max_horizontal(pl.col("center").abs(), pl.lit(float(floor)))
            ).alias(column)
        ).select("tick", column)


class DISPFreshnessFactor(_DISPFactor):
    """Stable-floor dispersion with exponentially freshness-weighted analysts."""

    meta = AlphaMeta(
        "disp_freshness",
        "freshness-weighted FY1 analyst dispersion with a stable denominator floor",
        direction=-1,
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp_freshness"

    def cross_section(self, asof):
        config = self.context.config
        analysts = self.context.reports(asof).with_columns(
            (
                -np.log(2.0)
                * (pl.lit(_date(asof)) - pl.col("create_date")).dt.total_days()
                / config.freshness_half_life_days
            ).exp().alias("weight")
        ).with_columns(
            (pl.col("weight") * pl.col("forecast_np")).alias("weighted_value"),
            (pl.col("weight") * pl.col("forecast_np").pow(2)).alias("weighted_square"),
            pl.col("weight").pow(2).alias("weight_square"),
        )
        stats = analysts.group_by("tick").agg(
            pl.col("weight").sum().alias("weight_sum"),
            pl.col("weighted_value").sum().alias("weighted_sum"),
            pl.col("weighted_square").sum().alias("weighted_square_sum"),
            pl.col("weight_square").sum().alias("weight_square_sum"),
            pl.len().alias("count"),
        ).with_columns(
            (pl.col("weighted_sum") / pl.col("weight_sum")).alias("center"),
            (pl.col("weight_sum").pow(2) / pl.col("weight_square_sum")).alias("effective_count"),
        ).with_columns(
            (
                pl.col("weighted_square_sum") / pl.col("weight_sum") - pl.col("center").pow(2)
            ).clip(lower_bound=0.0).sqrt().alias("dispersion")
        ).filter(
            (pl.col("count") >= config.min_dispersion_analysts)
            & (pl.col("effective_count") >= config.min_effective_analysts)
        )
        return self.normalize_dispersion(stats, self.column)


class DISPInstitutionFactor(_DISPFactor):
    """Stable-floor robust dispersion across institution-level forecasts."""

    meta = AlphaMeta(
        "disp_institution",
        "institution-robust FY1 forecast dispersion with a stable denominator floor",
        direction=-1,
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp_institution"

    def cross_section(self, asof):
        institutions = self.context.reports(asof).group_by("tick", "organ_id").agg(
            pl.col("forecast_np").mean().alias("institution_forecast")
        )
        centers = institutions.group_by("tick").agg(
            pl.col("institution_forecast").median().alias("center"),
            pl.len().alias("institution_count"),
        ).filter(
            pl.col("institution_count") >= self.context.config.min_institutions
        )
        stats = institutions.join(centers, on="tick", how="inner").with_columns(
            (pl.col("institution_forecast") - pl.col("center")).abs().alias("deviation")
        ).group_by("tick").agg(
            pl.col("center").first(),
            (1.4826 * pl.col("deviation").median()).alias("dispersion"),
        )
        return self.normalize_dispersion(stats, self.column)


class DISPMidFactor(_DISPFactor):
    """Analyst EWMA90, institution equal-weighting and ordinary dispersion."""

    meta = AlphaMeta(
        "disp_mid",
        "analyst EWMA90 followed by institution-equal FY1 mean/std dispersion",
        direction=-1,
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp_mid"

    def cross_section(self, asof):
        config = self.context.config
        asof = _date(asof)
        history = self.context.report_history(asof).filter(
            pl.col("create_date")
            >= asof - pd.Timedelta(days=config.sequence_lookback_days)
        ).with_columns(
            (
                -np.log(2.0)
                * (pl.lit(asof) - pl.col("create_date")).dt.total_days()
                / config.freshness_half_life_days
            ).exp().alias("weight")
        ).with_columns(
            (pl.col("weight") * pl.col("forecast_np")).alias("weighted_value")
        )

        analysts = history.group_by("tick", "organ_id", "author_id").agg(
            pl.col("weighted_value").sum().alias("weighted_sum"),
            pl.col("weight").sum().alias("weight_sum"),
        ).with_columns(
            (pl.col("weighted_sum") / pl.col("weight_sum"))
            .alias("analyst_forecast")
        )
        institutions = analysts.group_by("tick", "organ_id").agg(
            pl.col("analyst_forecast").mean().alias("institution_forecast")
        )
        stats = institutions.group_by("tick").agg(
            pl.col("institution_forecast").mean().alias("center"),
            pl.col("institution_forecast").std(ddof=1).alias("dispersion"),
            pl.len().alias("institution_count"),
        ).filter(
            pl.col("institution_count") >= config.min_institutions
        )
        return self.normalize_dispersion(stats, self.column)

class DISPSeqFactor(_DISPFactor):
    """Sequential freshness, institution equal-weighting and robust dispersion."""

    meta = AlphaMeta(
        "disp_seq",
        "analyst EWMA90 followed by institution-equal robust FY1 dispersion",
        direction=-1,
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp_seq"

    def cross_section(self, asof):
        config = self.context.config
        asof = _date(asof)
        history = self.context.report_history(asof).filter(
            pl.col("create_date")
            >= asof - pd.Timedelta(days=config.sequence_lookback_days)
        ).with_columns(
            (
                -np.log(2.0)
                * (pl.lit(asof) - pl.col("create_date")).dt.total_days()
                / config.freshness_half_life_days
            ).exp().alias("weight")
        ).with_columns(
            (pl.col("weight") * pl.col("forecast_np")).alias("weighted_value")
        )

        analysts = history.group_by("tick", "organ_id", "author_id").agg(
            pl.col("weighted_value").sum().alias("weighted_sum"),
            pl.col("weight").sum().alias("weight_sum"),
        ).with_columns(
            (pl.col("weighted_sum") / pl.col("weight_sum"))
            .alias("analyst_forecast")
        )

        institutions = analysts.group_by("tick", "organ_id").agg(
            pl.col("analyst_forecast").mean().alias("institution_forecast")
        )

        centers = institutions.group_by("tick").agg(
            pl.col("institution_forecast").median().alias("center"),
            pl.len().alias("institution_count"),
        ).filter(
            pl.col("institution_count") >= config.min_institutions
        )
        
        stats = institutions.join(centers, on="tick", how="inner").with_columns(
            (pl.col("institution_forecast") - pl.col("center"))
            .abs().alias("deviation")
        ).group_by("tick").agg(
            pl.col("center").first(),
            (1.4826 * pl.col("deviation").median()).alias("dispersion"),
        )
        return self.normalize_dispersion(stats, self.column)


class DISPEqualFactor(_DISPFactor):
    """Equal-weight rank blend of the two independently calculated DISP branches."""

    meta = AlphaMeta(
        "disp",
        "equal-rank blend of freshness-weighted and institution-robust dispersion",
        direction=-1,
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp"

    def cross_section(self, asof):
        freshness = DISPFreshnessFactor(self.context).cross_section(asof)
        institution = DISPInstitutionFactor(self.context).cross_section(asof)
        frame = freshness.join(institution, on="tick", how="inner")
        if frame.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        return frame.with_columns(
            pl.col("disp_freshness").rank(method="average").alias("fresh_rank"),
            pl.col("disp_institution").rank(method="average").alias("institution_rank"),
        ).with_columns(
            (
                (pl.col("fresh_rank") - pl.col("fresh_rank").mean())
                / pl.col("fresh_rank").std(ddof=0)
            ).alias("fresh_value"),
            (
                (pl.col("institution_rank") - pl.col("institution_rank").mean())
                / pl.col("institution_rank").std(ddof=0)
            ).alias("institution_value"),
        ).select(
            "tick",
            (
                0.5 * pl.col("fresh_value")
                + 0.5 * pl.col("institution_value")
            ).alias(self.column),
        )

def _factor_classes():
    return DISPFreshnessFactor, DISPInstitutionFactor, DISPMidFactor, DISPSeqFactor, DISPEqualFactor


def _backtest():
    from tqdm import tqdm
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret

    start, end = pd.Timestamp("2012-02-13"), pd.Timestamp("2015-02-13")
    output = Path(__file__).resolve().parents[1] / "output" / "disp"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with DISPContext() as context:
        dates = pd.DatetimeIndex(context.data.axis.trade_dates)
        positions = np.flatnonzero((dates >= start) & (dates <= end))
        trade_dates = dates[positions]
        factors = [factor(context) for factor in _factor_classes()]
        predictions = {
            factor.meta.name: np.full(
                (len(positions), context.data.axis.tick_count), np.nan, np.float32
            )
            for factor in factors
        }
        for row, trade_date in enumerate(tqdm(trade_dates, desc="Updating DISP family")):
            for factor in factors:
                frame = factor.cross_section(trade_date)
                predictions[factor.meta.name][row] = context.align(frame, factor.column)

        daily_return = context.data.read(
            "d_essentials/pct", context.data.axis.date_count - 1, 0
        ) / 100.0
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for name, pred in predictions.items():
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
            for ax, horizon in zip(axes.flat, (1, 5, 10, 20)):
                windows = np.lib.stride_tricks.sliding_window_view(
                    daily_return[2:], horizon, axis=0
                )
                forward_return = np.prod(1.0 + windows, axis=-1) - 1.0
                label = np.full(pred.shape, np.nan)
                valid = positions < len(forward_return)
                label[valid] = forward_return[positions[valid]]
                ic, rank_ic = IC(pred, label), rankIC(pred, label)
                group_return = calc_group_ret(pred, label, 10)
                means = np.nanmean(group_return, axis=1)
                long_short = group_return[0] - group_return[-1]
                rows.append({
                    "factor": name,
                    "horizon": horizon,
                    "coverage": np.mean(np.isfinite(pred).sum(axis=1)),
                    "mean_ic": np.nanmean(ic),
                    "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(252),
                    "mean_rank_ic": np.nanmean(rank_ic),
                    "rank_icir": np.nanmean(rank_ic) / np.nanstd(rank_ic) * np.sqrt(252),
                    "long_short_bps": np.nanmean(long_short) * 1e4,
                    "long_short_sharpe": np.nanmean(long_short) / np.nanstd(long_short) * np.sqrt(252),
                    **{f"g{k + 1}_bps": value * 1e4 for k, value in enumerate(means)},
                })
                for group, values in enumerate(np.nancumsum(group_return, axis=1), 1):
                    ax.plot(trade_dates[:len(values)], values, color=colors[group - 1], linewidth=1.1, label=f"G{group}")
                ax.set_title(
                    f"{name} {horizon}D | IC={np.nanmean(ic):.4f}, "
                    f"RankIC={np.nanmean(rank_ic):.4f}"
                )
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                ax.grid(alpha=0.25)
                ax.legend(ncol=2, fontsize=8)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            fig.suptitle(f"{name} Decile Cumulative Excess Returns | No Tradable Filter")
            fig.tight_layout()
            fig.savefig(output / f"{name}_cumulative.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"saved: {output}")


if __name__ == "__main__":
    _backtest()