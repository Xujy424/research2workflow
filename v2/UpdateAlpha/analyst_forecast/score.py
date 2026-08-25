"""Point-in-time analyst-rating level, bias and revision-event factors."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase,AlphaContext,AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT,get_zyyx_conn
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from .utils import _date,aggregate,latest_analyst_values
else:
    PROJECT_ROOT=Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase,AlphaContext,AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT,get_zyyx_conn
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.UpdateAlpha.analyst_forecast.utils import _date,aggregate,latest_analyst_values

@dataclass(frozen=True)
class ScoreConfig:
    lookback_days:int=90
    history_days:int=365
    min_history_ratings:int=3
    max_revision_gap_days:int=180
    event_positive_threshold:float=0.0
    event_negative_threshold:float=0.0

class ScoreContext(AlphaContext):
    """Point-in-time standardized ratings; neutralization belongs downstream."""
    def __init__(self,root=ROOT,conn=None,config=ScoreConfig()):
        self.config=config;self.conn=conn or get_zyyx_conn();self._owns_conn=conn is None;self._cache={}
        super().__init__(DataPool(root,asset='stock'))

    def reports(self,asof):
        asof=_date(asof)
        if self._cache.get('asof')==asof:return self._cache['frame']
        start=asof-pd.Timedelta(days=self.config.history_days)
        sql=f"""
        SELECT f.id,f.report_id,f.stock_code,f.organ_id,ra.author_id,
               f.create_date,f.entrytime,
               f.gg_rating_code AS rating_score
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id=f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
          AND f.entrytime<='{asof} 23:59:59'
          AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND 7
          AND (f.reliability>=5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
          AND f.gg_rating_code IN ('1','2','3','5','7')
        """
        frame=(pl.read_database(sql,self.conn,infer_schema_length=None).with_columns(
            pl.col('stock_code').cast(pl.String).str.zfill(6).alias('tick'),
            pl.col('organ_id').cast(pl.Int64,strict=False),pl.col('author_id').cast(pl.Int64,strict=False),
            pl.col('create_date').cast(pl.Date,strict=False),pl.col('entrytime').cast(pl.Datetime,strict=False),
            pl.col('rating_score').cast(pl.Float64,strict=False),
        ).filter(
            pl.col('tick').is_not_null()
        ).sort(
            ['tick','organ_id','author_id','create_date','entrytime','report_id','id']
        ).unique(
            ['report_id','tick','organ_id','author_id'],
            keep='last',maintain_order=True))
        self._cache={'asof':asof,'frame':frame};return frame

    def current_analysts(self,asof):
        asof=_date(asof)
        recent=self.reports(asof).filter(pl.col('create_date')>=asof-pd.Timedelta(days=self.config.lookback_days))
        return latest_analyst_values(recent,'rating_score')

    def previous_trade_date(self,asof):
        asof=pd.Timestamp(asof).normalize()
        dates=pd.DatetimeIndex(self.data.axis.trade_dates).normalize()
        position=dates.searchsorted(asof,side='left')
        return dates[position-1] if position>0 else asof-pd.Timedelta(days=1)

class _ScoreFactor(AlphaBase):
    column=''
    def cross_section(self,asof):raise NotImplementedError
    def calculate(self,asof):return self.context.align(self.cross_section(_date(asof)),self.column)

class ScoreLevelFactor(_ScoreFactor):
    """Current consensus rating level; higher standardized codes are better."""
    meta=AlphaMeta('score_level','current institution-balanced analyst rating level')
    dependencies=('rpt_forecast_stk','rpt_report_author')
    column='score_level'
    def cross_section(self,asof):
        reports=self.context.reports(asof).filter(
            (pl.col('create_date')>=asof-pd.Timedelta(days=self.context.config.lookback_days))&
            pl.col('rating_score').is_finite()
        )
        return aggregate(reports,'rating_score',alias=self.column)

class ScoreBiasFactor(_ScoreFactor):
    """Current analyst-stock rating relative to that analyst's older history."""
    meta=AlphaMeta('score_bias','current rating minus analyst-stock historical mean')
    dependencies=('rpt_forecast_stk','rpt_report_author')
    column='score_bias'
    def cross_section(self,asof):
        cutoff=asof-pd.Timedelta(days=self.context.config.lookback_days)
        reports=(self.context.reports(asof).filter(
            pl.col('rating_score').is_finite()
        ).sort(
            ['tick','organ_id','author_id','create_date','entrytime','report_id','id']
        ).unique(
            ['report_id','tick','organ_id','author_id'],
            keep='last',maintain_order=True))
        history=(
            reports.filter(pl.col('create_date')<cutoff).group_by(['author_id','tick']).agg(
            pl.col('rating_score').mean().alias('history_mean'),
            pl.len().alias('history_count')
        ).filter(
            pl.col('history_count')>=self.context.config.min_history_ratings)
        )
        current=self.context.current_analysts(asof).join(
            history,on=['author_id','tick'],how='inner'
        ).with_columns(
            (pl.col('rating_score')-pl.col('history_mean')).alias(self.column)
        )
        return aggregate(current,self.column,alias=self.column)

class ScoreRevisionEventFactor(_ScoreFactor):
    """Sparse rating-revision magnitude for newly available reports; zero otherwise."""
    meta=AlphaMeta('score_revision_event','new analyst rating-revision magnitude event')
    dependencies=('rpt_forecast_stk','rpt_report_author')
    column='revision'
    def cross_section(self,asof):
        reports=(
            self.context.reports(asof).filter(
                pl.col('rating_score').is_finite()
            ).sort(
                ['tick','organ_id','author_id','create_date','entrytime','report_id','id']
            ).unique(
            ['report_id','tick','organ_id','author_id'],
            keep='last',maintain_order=True
            )
        )
        keys=['tick','organ_id','author_id']
        events=reports.with_columns(
            pl.col('rating_score').shift().over(keys).alias('prior_score'),
            pl.col('create_date').shift().over(keys).alias('prior_date'),
        ).with_columns(
            (pl.col('create_date')-pl.col('prior_date')).dt.total_days().alias('gap_days'),
            (pl.col('rating_score')-pl.col('prior_score')).alias('revision'),
        ).filter(
            (pl.col("create_date") >= asof - pd.Timedelta(days=self.context.config.lookback_days))
            & pl.col("prior_score").is_not_null()
            & pl.col("gap_days").is_between(1, self.context.config.max_revision_gap_days)
        )
        return aggregate(events, self.column, alias=self.column)

    def calculate(self,asof):
        return np.nan_to_num(super().calculate(asof),nan=0.0)

def _factor_classes():return (ScoreLevelFactor,ScoreBiasFactor,ScoreRevisionEventFactor)

def calculate_score_family(asof,root=ROOT,conn=None,config=ScoreConfig()):
    with ScoreContext(root,conn,config) as context:return {cls.meta.name:cls(context).run(asof) for cls in _factor_classes()}

def update_score_family(asof,root=ROOT,conn=None,config=ScoreConfig(),folder='factor_pool'):
    with ScoreContext(root,conn,config) as context:return {cls.meta.name:cls(context).update(asof,folder) for cls in _factor_classes()}

__all__=['ScoreConfig','ScoreContext','ScoreLevelFactor','ScoreBiasFactor','ScoreRevisionEventFactor','calculate_score_family','update_score_family']




if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    with ScoreContext() as context:
        score = _ScoreFactor(context)
        trade_dates = context.data["trade_dates"][:-1000]

        for trade_date in tqdm(trade_dates, desc="Updating SCORE"):
            score.update(trade_date)

        pred = context.data.load("factor_pool/score").copy()
        pred = pred[
            :context.data.axis.date_count,
            :context.data.axis.tick_count,
        ]
        daily_return = context.data.read(
            "d_essentials/pct",
            start_date=0,
            end_date=pred.shape[0] - 1,
        ) / 100.0

        tradable = context.data.read(
            "basic/tradable",
            start_date=0,
            end_date=pred.shape[0] - 1,
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
            label[:len(forward_return)] = forward_return
            ic = IC(pred, label)
            rank_ic = rankIC(pred, label)
            group_return = calc_group_ret(pred, label, 10)
            cumulative_return = np.nancumsum(group_return, axis=1)

            for group, values in enumerate(cumulative_return, start=1):
                suffix = " (Low)" if group == 1 else " (High)" if group == 10 else ""
                ax.plot(
                    context.data["trade_dates"][:len(values)],
                    values,
                    color=colors[group - 1],
                    linewidth=1.2,
                    label=f"Group {group}{suffix}",
                )

            mean_ic = np.nanmean(ic)
            mean_rank_ic = np.nanmean(rank_ic)
            ax.set_title(
                f"SCORE {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("SCORE Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        output = Path(__file__).resolve().parents[1] / "output" / "score_group_returns.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")