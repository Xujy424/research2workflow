"""Generate a reproducible three-year COV backtest report."""

from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.Backtest.singleFactor import (
    BacktestConfig, ExecutionConfig, PortfolioConfig,
    compare_rebalance_frequencies, factor_data_from_arrays,
)
from v2.Backtest.singleFactor.config import ActiveSide, Method, Weighting
from v2.Backtest.singleFactor.metrics import cross_sectional_ic
from v2.GetData import DataPool
from v2.UpdateData.config import get_zyyx_conn


ROOT = Path("Z:/")
OUTPUT = Path(__file__).resolve().parent / "cov_three_year"


def markdown_table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.itertuples(index=False, name=None):
        values = [f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)
                  for x in record]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(summary, dates, signal, returns, tradable):
    signal_df = pd.DataFrame(np.where(tradable, signal, np.nan),
                             index=pd.to_datetime(dates))
    return_df = pd.DataFrame(np.where(tradable, returns, np.nan),
                             index=signal_df.index)
    rank_ic = cross_sectional_ic(signal_df, return_df.shift(-2))
    coverage = np.isfinite(signal).sum(1)
    best = summary.loc[summary.groupby("strategy")["sharpe"].idxmax(),
                       ["strategy", "rebalance_frequency", "annual_return",
                        "sharpe", "max_drawdown", "average_turnover"]]
    report = [
        "# COV因子最近三年回测报告", "",
        f"- 区间：{dates[0]} 至 {dates[-1]}", f"- 交易日：{len(dates)}",
        "- 信号滞后：2个交易日", "- 交易成本：单边5bps",
        "- 多头：最高两个十分位组",
        f"- 日均有信号股票数：{coverage.mean():.1f}",
        f"- 日均RankIC：{rank_ic.mean():.4f}", "",
        "## 各策略最优调仓频率", "", markdown_table(best), "",
        "## 完整绩效汇总", "", markdown_table(summary), "",
        "## 结果解读", "",
        "- 纯多头收益包含市场方向、行业配置和选股三部分，不能单独证明COV存在纯alpha。",
        "- 行业对齐指数对冲组合的行业主动权重接近零，更接近行业内选股检验。",
        "- 调仓频率差异同时反映信号衰减和交易成本；不能仅凭样本内最优频率定上线参数。",
        "- 本报告未加入涨跌停成交、冲击成本和融券约束，容量结论需另做执行模拟。", "",
        "## 输出文件", "", "- `performance.csv`：完整绩效指标",
        "- `daily_returns.csv`：策略日收益", "- `net_value.png`：净值曲线",
        "- `cov_signal.npy`：原始COV因子矩阵",
        "- `dates.npy`、`ticks.npy`：本次回测使用的轴快照", "",
        "## 口径说明", "",
        "COV为过去90个自然日内有效报告数的平方根。报告必须在信号日结束前入库。",
        "收益从信号日后第2个交易日开始计入。指数对冲组合限制在对应指数成分股内，",
        "逐行业复制指数权重，并在行业内部选择高COV股票。",
    ]
    (OUTPUT / "报告.md").write_text("\n".join(report), encoding="utf-8")


def load_reports(start, end):
    warmup = pd.Timestamp(start) - pd.Timedelta(days=90)
    sql = f"""
    SELECT f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
           f.report_year, f.report_quarter, f.create_date, f.entrytime,
           f.forecast_np
    FROM rpt_forecast_stk f
    JOIN rpt_report_author ra ON ra.report_id = f.report_id
    WHERE f.create_date BETWEEN '{warmup.date()}' AND '{pd.Timestamp(end).date()}'
      AND f.entrytime <= '{pd.Timestamp(end).date()} 23:59:59'
      AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
      AND f.forecast_np IS NOT NULL
      AND (f.reliability >= 5 OR f.reliability IS NULL)
      AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
      AND f.gg_rating_code IN ('1','2','3','5','7')
    """
    conn = get_zyyx_conn()
    try:
        return pl.read_database(sql, conn, infer_schema_length=None).with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
            pl.col("create_date").cast(pl.Date, strict=False),
            pl.col("entrytime").cast(pl.Datetime, strict=False),
            pl.col("forecast_np").cast(pl.Float64, strict=False),
        ).filter(
            pl.col("tick").is_not_null() & pl.col("forecast_np").is_finite()
        ).sort([
            "tick", "author_id", "report_year", "report_quarter",
            "create_date", "entrytime", "report_id", "id",
        ])
    finally:
        conn.close()


def build_cov_matrix(reports, dates, ticks):
    positions = {str(tick): i for i, tick in enumerate(ticks)}
    signal = np.full((len(dates), len(ticks)), np.nan, dtype=np.float32)
    for row, date in enumerate(dates):
        asof = pd.Timestamp(date)
        frame = reports.filter(
            pl.col("create_date").is_between(
                (asof - pd.Timedelta(days=90)).date(), asof.date())
            & (pl.col("entrytime") <= asof.replace(hour=23, minute=59, second=59))
        ).unique(
            ["report_id", "tick", "organ_id", "author_id", "report_year"],
            keep="last", maintain_order=True,
        ).group_by("tick").agg(
            pl.col("report_id").n_unique().sqrt().alias("cov"))
        for tick, value in frame.iter_rows():
            col = positions.get(tick)
            if col is not None:
                signal[row, col] = value
    return signal


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = DataPool(ROOT, asset="stock")
    try:
        all_dates = data.axis.trade_dates
        end = all_dates[-1]
        start_target = end - np.timedelta64(3 * 365, "D")
        start_row = int(np.searchsorted(all_dates, start_target, side="left"))
        dates, ticks = all_dates[start_row:], data.axis.ticks
        reports = load_reports(dates[0], dates[-1])
        signal = build_cov_matrix(reports, dates, ticks)
        rows, cols = slice(start_row, data.axis.date_count), slice(0, data.axis.tick_count)
        returns = np.asarray(data.load("d_essentials/pct")[rows, cols], float) / 100
        tradable = np.asarray(data.load("basic/tradable")[rows, cols], bool)
        industry = np.asarray(data.load("industry/industry")[rows, cols])
        np.save(OUTPUT / "cov_signal.npy", signal)
        np.save(OUTPUT / "dates.npy", dates)
        np.save(OUTPUT / "ticks.npy", ticks)

        summaries, result_sets = [], {}
        long_data = factor_data_from_arrays(
            signal, returns, dates, ticks, tradable=tradable, industry=industry)
        long_config = BacktestConfig(
            portfolio=PortfolioConfig(
                method=Method.QUANTILE_LONG_ONLY, quantiles=10, top_groups=2,
                weighting=Weighting.SIGNAL),
            execution=ExecutionConfig(signal_lag=2, cost_bps=5))
        table, results = compare_rebalance_frequencies(long_data, long_config)
        table.insert(0, "strategy", "cov_long_only")
        summaries.append(table.reset_index())
        result_sets.update({f"cov_long_only_{k}": v for k, v in results.items()})

        for benchmark in ("hs300", "zz500", "zz1000"):
            index_weight = np.asarray(data.load(
                f"index/weight/{benchmark}_weight")[rows, cols], float)
            aligned_data = factor_data_from_arrays(
                signal, returns, dates, ticks, tradable=tradable,
                industry=industry, benchmark_weight=index_weight)
            aligned_config = BacktestConfig(
                portfolio=PortfolioConfig(
                method=Method.BENCHMARK_HEDGED, quantiles=10,
                top_groups=2, weighting=Weighting.SIGNAL,
                    industry_align=True, active_side=ActiveSide.LONG,
                    active_gross=1.0),
                execution=ExecutionConfig(signal_lag=2, cost_bps=5))
            table, results = compare_rebalance_frequencies(
                aligned_data, aligned_config)
            table.insert(0, "strategy", f"cov_industry_aligned_minus_{benchmark}")
            summaries.append(table.reset_index())
            result_sets.update({f"cov_{benchmark}_{k}": v for k, v in results.items()})

        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(OUTPUT / "performance.csv", index=False)
        daily = pd.concat(
            {name: result.returns["net"] for name, result in result_sets.items()},
            axis=1)
        daily.to_csv(OUTPUT / "daily_returns.csv")
        nav = (1 + daily.fillna(0)).cumprod()
        ax = nav.plot(figsize=(16, 9), linewidth=1.1)
        ax.set_title("COV three-year backtest: net value")
        ax.set_ylabel("Net value")
        ax.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(OUTPUT / "net_value.png", dpi=160)
        plt.close()

        write_report(summary, dates, signal, returns, tradable)
        print(summary.to_string(index=False))
        print(OUTPUT / "报告.md")
    finally:
        data.close()


if __name__ == "__main__":
    main()
