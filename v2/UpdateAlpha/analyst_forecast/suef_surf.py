"""Consensus-forecast standardized unexpected earnings and revenue."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT, get_jy_conn, get_zyyx_conn
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_jy_conn, get_zyyx_conn
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT

@dataclass(frozen=True)
class SUEFSURFConfig:
    financial_lookback_years: int = 4
    history_quarters: int = 8
    min_history: int = 4
    report_lookback_days: int = 90
    report_half_life_days: float = 45.0
    local_np_field: str = "zyyx/con_forecast/con_np"
    local_revenue_field: str = "zyyx/con_forecast/con_or"

def _growth(expected_remainder, prior_remainder):
    ratio = expected_remainder / prior_remainder
    return np.where(prior_remainder >= 0, ratio - 1.0, 1.0 - ratio)

def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return np.datetime64(value, "D").astype(object)

class SUEFSURFContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, jy_conn=None, zyyx_conn=None, config=SUEFSURFConfig()):
        self.config=config
        self.jy_conn=jy_conn or get_jy_conn()
        self.zyyx_conn=zyyx_conn
        self._owns_jy=jy_conn is None
        self._owns_zyyx=False; self._cache={}
        super().__init__(DataPool(root, asset="stock"))

    def close(self):
        self.data.close()
        if self._owns_jy and self.jy_conn is not None: self.jy_conn.close()
        if self._owns_zyyx and self.zyyx_conn is not None: self.zyyx_conn.close()

    def actual_events(self, asof, history_start=None):
        asof=_date(asof)
        if self._cache.get("asof") != asof:
            self._cache.clear()
            self._cache["asof"] = asof
        if history_start is None:
            try:
                history_start=asof.replace(
                    year=asof.year-self.config.financial_lookback_years
                )
            except ValueError:
                history_start=asof.replace(
                    year=asof.year-self.config.financial_lookback_years,day=28
                )
        else:
            history_start=_date(history_start)
        key=("actual",history_start,asof)
        if key in self._cache: return self._cache[key]
        selects=[]
        for table in ("LC_IncomeStatementAll","LC_STIBIncomeState"):
            bulletin="AND f.BulletinType IN (20,30)" if table.endswith("All") else ""
            selects.append(
                f"""SELECT 
                    f.ID id, s.SecuCode tick,
                    f.EndDate end_date, f.InfoPublDate publish_date,
                    f.NPParentCompanyOwners net_profit_accum,
                    f.OperatingRevenue revenue_accum 
                    FROM dbo.{table} f 
                    JOIN dbo.SecuMain s
                        ON f.CompanyCode=s.CompanyCode 
                    WHERE f.EndDate>='{history_start}'
                        AND f.InfoPublDate>='{history_start}'
                        AND f.InfoPublDate<='{asof}' 
                        AND f.IfMerged=1 
                        AND f.IfAdjusted=2
                        AND f.IfComplete=1 {bulletin} 
                        AND s.SecuCategory=1 
                        AND s.SecuMarket IN (83,90)"""
            )
        x=pl.read_database(
            " UNION ALL ".join(selects),self.jy_conn,infer_schema_length=None
        ).with_columns(
            pl.col("tick").cast(pl.String).str.zfill(6),
            pl.col("end_date").cast(pl.Datetime,strict=False).dt.date(),
            pl.col("publish_date").cast(pl.Datetime,strict=False).dt.date(),
            pl.col("net_profit_accum").cast(pl.Float64,strict=False),
            pl.col("revenue_accum").cast(pl.Float64,strict=False),
        )
        if x.is_empty(): self._cache[key]=x; return x
        x=(
            x.filter(pl.col("end_date").dt.month().is_in([3,6,9,12]))
           .sort(["tick","end_date","publish_date","id"])
           .unique(["tick","end_date"],keep="first",maintain_order=True)
           .with_columns(
               pl.col("end_date").dt.year().alias("year"),
               (pl.col("end_date").dt.month()//3).alias("quarter"),
            )
        )
        self._cache[key]=x
        return x

    def local_consensus(self, events, field):
        axis=self.data.axis
        dt=events["cutoff"].to_numpy().astype("datetime64[D]")
        rows=np.searchsorted(axis.trade_dates,dt,side="left")-1
        cols=np.array([axis._tick_positions.get(x,-1) for x in events["tick"]])
        cutoff_year=events["cutoff"].dt.year().to_numpy()
        base=np.where(events["cutoff"].dt.month().to_numpy()<5,cutoff_year-2,cutoff_year-1)
        levels=events["year"].to_numpy()-base
        out=np.full(len(events),np.nan)
        ok=(rows>=0)&(rows<axis.date_count)&(cols>=0)&(levels>=0)&(levels<4)
        matrix=self.data.load(field)
        out[ok]=matrix[rows[ok],levels[ok],cols[ok]]
        return out

    def report_consensus(self, events, forecast_column):
        if self.zyyx_conn is None:
            self.zyyx_conn=get_zyyx_conn()
            self._owns_zyyx=True
        start=events["cutoff"].min()-timedelta(days=self.config.report_lookback_days)
        end=events["cutoff"].max()
        years=sorted(events["year"].unique().to_list())
        sql=f"""
        SELECT 
        f.id,f.report_id,f.stock_code,f.organ_id,ra.author_id,
        f.create_date,f.entrytime,f.report_year,
        f.{forecast_column} forecast
        FROM rpt_forecast_stk f 
        JOIN rpt_report_author ra ON ra.report_id=f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{end}'
        AND f.report_year IN ({','.join(map(str,years))}) AND f.report_quarter=4
        AND f.{forecast_column} IS NOT NULL 
        AND (f.reliability>=5 OR f.reliability IS NULL)
        AND f.organ_id IS NOT NULL 
        AND ra.author_id IS NOT NULL
        """
        r=pl.read_database(sql,self.zyyx_conn,infer_schema_length=None)
        if r.is_empty(): return np.full(len(events),np.nan)
        r=(r.with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
            pl.col("create_date").cast(pl.Datetime,strict=False).dt.date(),
            pl.col("entrytime").cast(pl.Datetime,strict=False),
            pl.col("forecast").cast(pl.Float64,strict=False),
        ).sort(["tick","author_id","report_year","create_date","entrytime","id"])
        .unique(["report_id","tick","organ_id","author_id","report_year"],
                keep="last",maintain_order=True))
        e=events.with_row_index("event_id")
        j=(
            e.select("event_id","tick","year","cutoff")
            .join(r,left_on=["tick","year"],right_on=["tick","report_year"],how="left")
            .with_columns((pl.col("cutoff")-pl.col("create_date")).dt.total_days().alias("age"))
            .filter(
               pl.col("age").is_between(0,self.config.report_lookback_days)
               & (pl.col("entrytime")<pl.col("cutoff").cast(pl.Datetime)+pl.duration(days=1))
               & pl.col("forecast").is_finite()
            ).with_columns(
               (-np.log(2.0)*pl.col("age")/self.config.report_half_life_days).exp().alias("weight")
            ).with_columns(
               (pl.col("forecast")*pl.col("weight")).alias("weighted"))
        )
        values=(
            j
            .group_by(["event_id","organ_id","author_id"])
            .agg(pl.col("weighted").sum(), pl.col("weight").sum())
                  .with_columns((pl.col("weighted")/pl.col("weight")).alias("value"))
                  .group_by(["event_id","organ_id"]).agg(pl.col("value").mean())
                  .group_by("event_id").agg(pl.col("value").mean()))
        return (
            e
            .select("event_id").join(values,on="event_id",how="left")
            .select("value").to_numpy().ravel().astype(float)
        )

class _ConsensusSurpriseFactor(AlphaBase):
    """Daily calculation base; Context only supplies filtered PIT inputs.
    先从全年一致预期中扣掉已经公布的累计业绩，
    得到分析师对剩余期间的预期；
    再将这个预期增长率应用于去年同期单季度业绩，
    估计本季度市场原本期待实现多少，
    最后用实际值减去预期值
    """

    column = ""
    source = "local"
    actual_column = ""
    local_field = ""
    report_column = ""
    dependencies = (
        "LC_IncomeStatementAll", "LC_STIBIncomeState",
        "zyyx/con_forecast/con_np", "zyyx/con_forecast/con_or",
    )

    @staticmethod
    def _quarter_fields(x,value):
        base=x.with_row_index("_row")
        lookup=base.select("tick","year","quarter", pl.col(value).alias("_value"))
        # 上个季度
        prior=(
            lookup.with_columns(
                (pl.col("quarter")+1).alias("quarter")
            )
            .rename({"_value":"prior"})
        )
        # 去年年报
        prior_fy=(
            lookup.filter(pl.col("quarter")==4)
            .with_columns((pl.col("year")+1).alias("year"))
            .select("tick","year",pl.col("_value").alias("prior_fy"))
        )
        # 去年同期
        prior_year=(
            lookup.with_columns((pl.col("year")+1).alias("year"))
            .rename({"_value":"prior_year"})
        )
        # 去年上个季度
        prior_before=(
            lookup.with_columns(
                (pl.col("year")+1).alias("year"),
                (pl.col("quarter")+1).alias("quarter"),
            ).rename({"_value":"prior_before"})
        )
        fields=(
            base
            .join(prior,on=["tick","year","quarter"],how="left")        # 上个季度
            .join(prior_fy,on=["tick","year"],how="left")               # 去年年报
            .join(prior_year,on=["tick","year","quarter"],how="left")   # 去年同期
            .join(prior_before,on=["tick","year","quarter"],how="left") # 去年上个季度
            .with_columns(
                pl.when(pl.col("quarter")==1).then(0.0).otherwise(pl.col("prior")).alias("prior"),
                pl.when(pl.col("quarter")==1).then(0.0).otherwise(pl.col("prior_before")).alias("prior_before"),
            ).with_columns(
                (pl.col(value)-pl.col("prior")).alias("actual_q"),
                (pl.col("prior_year")-pl.col("prior_before")).alias("prior_q"),
            ).sort("_row")
        )
        return tuple(fields[name].to_numpy() for name in (
            "prior","actual_q","prior_fy","prior_before","prior_q"
        ))


    def _forecast(self, events):
        if self.source == "local":
            return self.context.local_consensus(events, self.local_field)
        return self.context.report_consensus(events, self.report_column)

    def cross_section(self, asof):
        events = self.context.actual_events(asof)
        if events.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        events = events.with_columns(pl.col("publish_date").alias("cutoff"))
        prior, actual_q, prior_fy, prior_before, prior_q = self._quarter_fields(
            events, self.actual_column
        )
        forecast = self._forecast(events)
        error = actual_q - prior_q * (
            1.0 + _growth(forecast - prior, prior_fy - prior_before)
        )
        error_col = f"_{self.column}_error"
        return (
            events.with_columns(pl.Series(error_col, error))
            .sort(["tick", "end_date", "publish_date"])
            .with_columns(
                (
                    pl.col(error_col)
                    / pl.col(error_col).shift(1).rolling_std(
                        window_size=self.context.config.history_quarters,
                        min_samples=self.context.config.min_history,
                        ddof=1,
                    ).over("tick")
                ).alias(self.column)
            )
            .sort(["tick", "publish_date", "end_date"])
            .unique("tick", keep="last", maintain_order=True)
            .select("tick", self.column)
        )

    def calculate(self, asof):
        return self.context.align(self.cross_section(_date(asof)), self.column)


class _SimpleConsensusSurpriseFactor(_ConsensusSurpriseFactor):
    """Simple surprise: (actual quarter - expected quarter) / |expected quarter|."""

    def cross_section(self, asof):
        events = self.context.actual_events(asof)
        if events.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        events = events.with_columns(pl.col("publish_date").alias("cutoff"))
        prior, actual_q, prior_fy, prior_before, prior_q = self._quarter_fields(
            events, self.actual_column
        )
        forecast = self._forecast(events)
        expected_q = prior_q * (
            1.0 + _growth(forecast - prior, prior_fy - prior_before)
        )
        surprise = np.divide(
            actual_q - expected_q,
            np.abs(expected_q),
            out=np.full_like(expected_q, np.nan, dtype=float),
            where=np.isfinite(expected_q) & (expected_q != 0),
        )
        return (
            events.with_columns(pl.Series(self.column, surprise))
            .sort(["tick", "publish_date", "end_date"])
            .unique("tick", keep="last", maintain_order=True)
            .select("tick", self.column)
        )


class SUEFFactor(_ConsensusSurpriseFactor):
    meta = AlphaMeta("suef", "local-consensus standardized unexpected earnings")
    column = "suef"
    actual_column = "net_profit_accum"
    local_field = SUEFSURFConfig.local_np_field
    report_column = "forecast_np"


class SURFFactor(_ConsensusSurpriseFactor):
    meta = AlphaMeta("surf", "local-consensus standardized unexpected revenue")
    column = "surf"
    actual_column = "revenue_accum"
    local_field = SUEFSURFConfig.local_revenue_field
    report_column = "forecast_or"


class SUEFReportFactor(SUEFFactor):
    meta = AlphaMeta(
        "suef_reports", "90D report-consensus standardized unexpected earnings"
    )
    source = "reports"


class SURFReportFactor(SURFFactor):
    meta = AlphaMeta(
        "surf_reports", "90D report-consensus standardized unexpected revenue"
    )
    source = "reports"


class SUEFSimpleFactor(_SimpleConsensusSurpriseFactor):
    meta = AlphaMeta("suef_simple", "local-consensus earnings surprise / expected")
    column = "suef_simple"
    actual_column = "net_profit_accum"
    local_field = SUEFSURFConfig.local_np_field
    report_column = "forecast_np"


class SURFSimpleFactor(_SimpleConsensusSurpriseFactor):
    meta = AlphaMeta("surf_simple", "local-consensus revenue surprise / expected")
    column = "surf_simple"
    actual_column = "revenue_accum"
    local_field = SUEFSURFConfig.local_revenue_field
    report_column = "forecast_or"


class SUEFReportSimpleFactor(SUEFSimpleFactor):
    meta = AlphaMeta(
        "suef_reports_simple", "90D report-consensus earnings surprise / expected"
    )
    source = "reports"


class SURFReportSimpleFactor(SURFSimpleFactor):
    meta = AlphaMeta(
        "surf_reports_simple", "90D report-consensus revenue surprise / expected"
    )
    source = "reports"


__all__ = [
    "SUEFSURFConfig", "SUEFSURFContext", "SUEFFactor", "SURFFactor",
    "SUEFReportFactor", "SURFReportFactor",
    "SUEFSimpleFactor", "SURFSimpleFactor",
    "SUEFReportSimpleFactor", "SURFReportSimpleFactor",
]


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Inclusive date range. Use None to select the first/last available date.
    START_DATE = "2017-01-01"
    END_DATE = "2026-06-30"

    with SUEFSURFContext() as context:
        alpha = SUEFFactor(context)
        trade_dates = context.data["trade_dates"]

        date_index = pd.DatetimeIndex(trade_dates)
        start_date = pd.Timestamp(START_DATE) if START_DATE else date_index[0]
        end_date = pd.Timestamp(END_DATE) if END_DATE else date_index[-1]
        if start_date > end_date:
            raise ValueError("START_DATE must not be later than END_DATE")
        selected = np.flatnonzero(
            (date_index >= start_date) & (date_index <= end_date)
        )
        if selected.size == 0:
            raise ValueError("no trade dates found in the requested range")
        start_idx, end_idx = selected[0], selected[-1]
        selected_dates = trade_dates[start_idx:end_idx + 1]

        for trade_date in tqdm(selected_dates, desc="Updating SUEF"):
            alpha.update(trade_date)

        pred = context.data.load("factor_pool/suef").copy()
        pred = pred[
            start_idx:end_idx + 1,
            :context.data.axis.tick_count,
        ]
        daily_return = context.data.read(
            "d_essentials/pct",
            start_date=0,
            end_date=context.data.axis.date_count - 1,
        ) / 100.0

        tradable = context.data.read(
            "basic/tradable",
            start_date=start_idx,
            end_date=end_idx,
        )
        pred = np.where(tradable, pred, np.nan)

        horizons = (1, 5, 10, 20)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for ax, horizon in zip(axes.flat, horizons):
            # pct[t] is the return from t - 1 to t. For a signal formed on
            # t, skip t + 1 and compound t + 2 ... t + 1 + horizon.
            windows = np.lib.stride_tricks.sliding_window_view(
                daily_return[2:], horizon, axis=0
            )
            forward_return = np.prod(1.0 + windows, axis=-1) - 1.0

            label = np.full(pred.shape, np.nan)
            range_forward_return = forward_return[start_idx:end_idx + 1]
            label[:len(range_forward_return)] = range_forward_return
            ic = IC(pred, label)
            rank_ic = rankIC(pred, label)
            group_return = calc_group_ret(pred, label, 10)
            cumulative_return = np.nancumsum(group_return, axis=1)

            for group, values in enumerate(cumulative_return, start=1):
                suffix = " (Low)" if group == 1 else " (High)" if group == 10 else ""
                ax.plot(
                    selected_dates[:len(values)],
                    values,
                    color=colors[group - 1],
                    linewidth=1.2,
                    label=f"Group {group}{suffix}",
                )

            mean_ic = np.nanmean(ic)
            mean_rank_ic = np.nanmean(rank_ic)
            ax.set_title(
                f"SUEF {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("SUEF Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        range_tag = f"{date_index[start_idx]:%Y%m%d}_{date_index[end_idx]:%Y%m%d}"
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"suef_group_ret_{range_tag}.png"
        )
        # output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")
