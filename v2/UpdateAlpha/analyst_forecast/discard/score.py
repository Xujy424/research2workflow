"""Point-in-time analyst-rating level, bias and revision-event factors."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ...alphabase import AlphaBase,AlphaContext,AlphaMeta
    from ....GetData import DataPool
    from ....UpdateData.config import ROOT,get_zyyx_conn
    from ....ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from ..utils import _date,aggregate
else:
    PROJECT_ROOT=Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase,AlphaContext,AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT,get_zyyx_conn
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.UpdateAlpha.analyst_forecast.utils import _date,aggregate


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
        self.config=config
        self.conn=conn or get_zyyx_conn()
        self._owns_conn=conn is None
        self._cache={}
        super().__init__(DataPool(root,asset='stock'))

    def reports(self,start,end):
        """Load and cache raw rating reports for an inclusive date range."""
        start,end=_date(start),_date(end)
        if (
            self._cache
            and self._cache['start']<=start
            and self._cache['end']>=end
        ):
            return self._cache['frame']
        sql=f"""
        SELECT f.id,f.report_id,f.stock_code,f.organ_id,ra.author_id,
               f.create_date,f.entrytime,
               f.gg_rating_code AS rating_code
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id=f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{end}'
          AND f.entrytime<='{end} 23:59:59'
          AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND 7
          AND (f.reliability>=5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL
          AND ra.author_id IS NOT NULL
          AND f.gg_rating_code IN ('1','2','3','5','7')
        """
        frame=(pl.read_database(sql,self.conn,infer_schema_length=None).with_columns(
            pl.col('stock_code').cast(pl.String).str.zfill(6).alias('tick'),
            pl.col('organ_id').cast(pl.Int64,strict=False),pl.col('author_id').cast(pl.Int64,strict=False),
            pl.col('create_date').cast(pl.Date,strict=False),pl.col('entrytime').cast(pl.Datetime,strict=False),
            pl.col('rating_code').cast(pl.Int64,strict=False),
        ).with_columns(
            pl.col('rating_code').replace_strict(
                {7:7,5:6,3:5,2:3,1:1},
                default=None,
                return_dtype=pl.Float64,
            ).alias('rating_score')
        ).filter(
            pl.col('tick').is_not_null()
        ).sort(
            ['tick','organ_id','author_id','create_date','entrytime','report_id','id']
        ).unique(
            ['report_id','tick','organ_id','author_id'],
            keep='last',maintain_order=True))
        self._cache={'start':start,'end':end,'frame':frame}
        return frame

    def previous_trade_date(self,asof):
        asof=pd.Timestamp(asof).normalize()
        dates=pd.DatetimeIndex(self.data.axis.trade_dates).normalize()
        position=dates.searchsorted(asof,side='left')
        return dates[position-1] if position>0 else asof-pd.Timedelta(days=1)

class _ScoreFactor(AlphaBase):
    column=''
    def factor_reports(self,asof):
        """Apply point-in-time and history-window filters to raw reports."""
        asof=_date(asof)
        start=asof-pd.Timedelta(days=self.context.config.history_days)
        cutoff=pd.Timestamp(asof)+pd.Timedelta(days=1)
        return self.context.reports(start,asof).filter(
            pl.col('create_date').is_between(start,asof)
            &(pl.col('entrytime')<cutoff)
        )
    def cross_section(self,asof):
        raise NotImplementedError
    def calculate(self,asof):
        return self.context.align(self.cross_section(_date(asof)),self.column)


class ScoreLevelFactor(_ScoreFactor):
    """Current consensus rating level; higher standardized codes are better."""
    meta=AlphaMeta('score_level','current institution-balanced analyst rating level')
    dependencies=('rpt_forecast_stk','rpt_report_author')
    column='score_level'
    def cross_section(self,asof):
        reports=self.factor_reports(asof).filter(
            (pl.col('create_date')>=asof-pd.Timedelta(days=self.context.config.lookback_days))&
            pl.col('rating_score').is_finite()
        )
        return aggregate(reports,'rating_score',alias=self.column)


class _ScoreAdjustmentMeanFactor(_ScoreFactor):
    """Latest rating minus an exponentially weighted historical mean."""
    dependencies=('rpt_forecast_stk','rpt_report_author')
    half_lives=(30,60,90)

    def cross_section(self,asof):
        asof=_date(asof)
        cached=getattr(self.context,'_adjustment_cache',None)
        if cached is not None and cached['asof']==asof:
            return cached['frames'][self.column]
        reports=(
            self.factor_reports(asof)
            .filter(pl.col('rating_score').is_finite())
            .sort([
                'tick','organ_id','author_id','create_date','entrytime','report_id','id',
            ])
            .with_columns(
                pl.col('create_date').cast(pl.Int32).alias('date_ordinal'),
                pl.lit(1.0).alias('observation'),
            )
            .with_columns(
                *[
                    (
                        pl.lit(2.0).log()*pl.col('date_ordinal')/half_life
                    ).exp().alias(f'decay_{half_life}')
                    for half_life in self.half_lives
                ]
            )
            .with_columns(
                *[
                    (pl.col('rating_score')*pl.col(f'decay_{half_life}'))
                    .alias(f'weighted_score_{half_life}')
                    for half_life in self.half_lives
                ]
            )
        )
        keys=['tick','organ_id','author_id']
        window=f'{self.context.config.history_days}d'
        reports=reports.with_columns(
            pl.col('observation')
            .rolling_sum_by('create_date',window_size=window,closed='left')
            .over(keys)
            .alias('history_count'),
            *[
                expression
                for half_life in self.half_lives
                for expression in (
                    pl.col(f'decay_{half_life}')
                    .rolling_sum_by('create_date',window_size=window,closed='left')
                    .over(keys).alias(f'weight_sum_{half_life}'),
                    pl.col(f'weighted_score_{half_life}')
                    .rolling_sum_by('create_date',window_size=window,closed='left')
                    .over(keys).alias(f'weighted_score_sum_{half_life}'),
                )
            ],
        ).with_columns(
            *[
                (
                    pl.col('rating_score')
                    -pl.col(f'weighted_score_sum_{half_life}')
                    /pl.col(f'weight_sum_{half_life}')
                ).alias(f'score_adjustment_{half_life}')
                for half_life in self.half_lives
            ]
        )
        recent=reports.filter(
            (pl.col('create_date')>=asof-pd.Timedelta(days=self.context.config.lookback_days))
            &(pl.col('history_count')>=self.context.config.min_history_ratings)
        )
        frames={}
        for half_life in self.half_lives:
            column=f'score_adjustment_{half_life}'
            adjustments=(
                recent.filter(
                    (pl.col(f'weight_sum_{half_life}')>0)
                    &pl.col(column).is_finite()
                )
                .unique(keys,keep='last',maintain_order=True)
            )
            frames[column]=aggregate(adjustments,column,alias=column)
        self.context._adjustment_cache={'asof':asof,'frames':frames}
        return frames[self.column]


class ScoreAdjustment30Factor(_ScoreAdjustmentMeanFactor):
    meta=AlphaMeta('score_adjustment_30','rating minus 30D-half-life mean')
    column='score_adjustment_30'
    half_life_days=30


class ScoreAdjustment60Factor(_ScoreAdjustmentMeanFactor):
    meta=AlphaMeta('score_adjustment_60','rating minus 60D-half-life mean')
    column='score_adjustment_60'
    half_life_days=60


class ScoreAdjustment90Factor(_ScoreAdjustmentMeanFactor):
    meta=AlphaMeta('score_adjustment_90','rating minus 90D-half-life mean')
    column='score_adjustment_90'
    half_life_days=90


def _factor_classes():
    return (
        ScoreLevelFactor,ScoreAdjustment30Factor,
        ScoreAdjustment60Factor,ScoreAdjustment90Factor,
    )


def calculate_score_family(asof,root=ROOT,conn=None,config=ScoreConfig()):
    with ScoreContext(root,conn,config) as context:
        return {
            factor_class.meta.name:factor_class(context).run(asof)
            for factor_class in _factor_classes()
        }


def update_score_family(
    asof,root=ROOT,conn=None,config=ScoreConfig(),folder='factor_pool'
):
    with ScoreContext(root,conn,config) as context:
        return {
            factor_class.meta.name:factor_class(context).update(asof,folder)
            for factor_class in _factor_classes()
        }






__all__=[
    'ScoreConfig','ScoreContext','ScoreLevelFactor',
    'ScoreAdjustment30Factor','ScoreAdjustment60Factor',
    'ScoreAdjustment90Factor',
    'calculate_score_family','update_score_family',
]




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
