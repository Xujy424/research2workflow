"""Reusable single-factor research for continuous and sparse event signals."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from tqdm import tqdm

from v2.GetData import DataPool
from v2.UpdateAlpha.analyst_forecast.score import (
    ScoreBiasFactor,
    ScoreConfig,
    ScoreContext,
    ScoreLevelFactor,
    ScoreRevisionEventFactor,
)
from v2.UpdateData.config import get_zyyx_conn

DATA_ROOT = Path(r"D:\data")
DEFAULT_START = pd.Timestamp("2012-02-13")
DEFAULT_END = pd.Timestamp("2016-02-13")
DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_EVENT_REBALANCE_DAYS = (1, 5, 10, 20)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    factor_class: type
    kind: str  # continuous | event
    neutralize: bool
    event_holding_days: int = 5


SPECS = {
    "score_level": FactorSpec("score_level", ScoreLevelFactor, "continuous", True),
    "score_bias": FactorSpec("score_bias", ScoreBiasFactor, "continuous", True),
    "score_revision_event": FactorSpec(
        "score_revision_event", ScoreRevisionEventFactor, "event", False, 5
    ),
}


class InMemoryScoreContext(ScoreContext):
    """Use one bulk point-in-time report table instead of one SQL query per day."""

    def __init__(self, data, reports, config=ScoreConfig()):
        self.data = data
        self.config = config
        self._reports = reports
        self._owns_conn = False

    def reports(self, asof):
        asof = pd.Timestamp(asof)
        return self._reports.filter(
            (pl.col("create_date") >= asof - pd.Timedelta(days=self.config.history_days))
            & (pl.col("create_date") <= asof)
            & (pl.col("entrytime") < asof + pd.Timedelta(days=1))
        )


def load_reports(start, end, history_days=365):
    """Bulk-load point-in-time rating reports in bounded SQL chunks."""
    conn = get_zyyx_conn()
    frames = []
    left = start - pd.Timedelta(days=history_days)
    while left <= end:
        right = min(left + pd.DateOffset(months=6) - pd.Timedelta(days=1), end)
        sql = f"""
        SELECT f.id,f.report_id,f.stock_code,f.organ_id,ra.author_id,
               f.report_year,f.report_quarter,f.create_date,f.entrytime,
               f.gg_rating_code AS rating_score
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id=f.report_id
        WHERE f.create_date BETWEEN '{left:%Y-%m-%d}' AND '{right:%Y-%m-%d}'
          AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND 7
          AND (f.reliability>=5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
          AND f.gg_rating_code IN ('1','2','3','5','7')
        """
        frames.append(pl.read_database(sql, conn, infer_schema_length=None))
        left = right + pd.Timedelta(days=1)
    conn.close()
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
            pl.col("organ_id").cast(pl.Int64, strict=False),
            pl.col("author_id").cast(pl.Int64, strict=False),
            pl.col("report_year").cast(pl.Int64, strict=False),
            pl.col("report_quarter").cast(pl.Int64, strict=False),
            pl.col("create_date").cast(pl.Date, strict=False),
            pl.col("entrytime").cast(pl.Datetime, strict=False),
            pl.col("rating_score").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("tick").is_not_null())
        .sort([
            "tick", "organ_id", "author_id", "report_year", "report_quarter",
            "create_date", "entrytime", "report_id", "id",
        ])
        .unique([
            "report_id", "tick", "organ_id", "author_id",
            "report_year", "report_quarter",
        ], keep="last", maintain_order=True)
    )


def build_signals(context, specs, dates):
    """Calculate production factor classes into date-by-stock matrices."""
    shape = (len(dates), context.data.axis.tick_count)
    matrices = {spec.name: np.full(shape, np.nan, np.float32) for spec in specs}
    factors = {spec.name: spec.factor_class(context) for spec in specs}
    for row, asof in enumerate(tqdm(dates, desc="building score factors")):
        for spec in specs:
            matrices[spec.name][row] = factors[spec.name].calculate(asof)
    return matrices


def robust_standardize_cross_section(values, mad_limit=5.0):
    """Median-MAD winsorization followed by cross-sectional Z-score."""
    output = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 20:
        return output
    sample = values[valid].astype(float)
    median = np.median(sample)
    mad = np.median(np.abs(sample - median))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 1e-12:
        sample = np.clip(sample, median-mad_limit*robust_sigma, median+mad_limit*robust_sigma)
    else:
        lower, upper = np.quantile(sample, (0.01, 0.99))
        sample = np.clip(sample, lower, upper)
    standard_deviation = sample.std()
    if standard_deviation > 1e-12:
        output[valid] = (sample-sample.mean())/standard_deviation
    return output


def robust_standardize_matrix(signal, mad_limit=5.0):
    return np.asarray(
        [robust_standardize_cross_section(row, mad_limit) for row in signal],
        dtype=np.float32,
    )


def neutralize_cross_section(values, industry, market_value):
    """OLS residual on industry dummies and standardized log float market value."""
    valid = (
        np.isfinite(values)
        & np.isfinite(industry)
        & np.isfinite(market_value)
        & (market_value > 0)
    )
    residual = np.full_like(values, np.nan, dtype=float)
    if valid.sum() < 30:
        return residual
    codes = industry[valid]
    log_size = np.log(market_value[valid])
    log_size = (log_size - log_size.mean()) / (log_size.std() + 1e-12)
    columns = [np.ones(valid.sum()), log_size]
    columns.extend((codes == code).astype(float) for code in np.unique(codes)[1:])
    design = np.column_stack(columns)
    residual[valid] = values[valid] - design @ np.linalg.lstsq(
        design, values[valid], rcond=None
    )[0]
    return residual


def neutralize_matrix(signal, industry, market_value):
    return np.asarray(
        [neutralize_cross_section(x, ind, mv) for x, ind, mv in zip(signal, industry, market_value)],
        dtype=np.float32,
    )


def forward_returns(daily_return, positions, horizons):
    """Signal on t, skip t+1, then compound t+2 onward."""
    result = {}
    for horizon in horizons:
        values = np.full((len(positions), daily_return.shape[1]), np.nan, np.float32)
        for row, position in enumerate(positions):
            start = position + 2
            if start + horizon <= len(daily_return):
                values[row] = np.prod(1 + daily_return[start : start + horizon], axis=0) - 1
        result[horizon] = values
    return result


def pearson_by_date(signal, returns):
    values = np.full(len(signal), np.nan)
    for i, (x, y) in enumerate(zip(signal, returns)):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 20 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
            values[i] = np.corrcoef(x[valid], y[valid])[0, 1]
    return values


def rank_ic_by_date(signal, returns):
    ranked = np.full_like(signal, np.nan)
    for i, row in enumerate(signal):
        valid = np.isfinite(row)
        if valid.sum() >= 20:
            ranked[i, valid] = rankdata(row[valid], method="average")
    return pearson_by_date(ranked, returns)


def hac_t_stat(values, lag):
    """Newey-West t-stat for an overlapping daily mean series."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < max(20, lag + 2):
        return np.nan
    centered = x - x.mean()
    long_run_variance = np.dot(centered, centered) / len(x)
    for k in range(1, min(lag, len(x) - 2) + 1):
        covariance = np.dot(centered[k:], centered[:-k]) / len(x)
        long_run_variance += 2 * (1 - k / (lag + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / len(x))
    return x.mean() / standard_error if standard_error > 0 else np.nan


def signal_masks(signal, kind, tail=0.2):
    """Positive/negative masks: tails for continuous factors, signs for events."""
    positive = np.zeros(signal.shape, bool)
    negative = np.zeros(signal.shape, bool)
    for i, row in enumerate(signal):
        valid = np.isfinite(row)
        if kind == "event":
            positive[i] = valid & (row > 0)
            negative[i] = valid & (row < 0)
        elif valid.sum() >= 20:
            ranks = np.empty(valid.sum())
            ranks[rankdata(row[valid], method="average").astype(int) - 1] = np.arange(valid.sum())
            percentile = np.full(len(row), np.nan)
            percentile[valid] = rankdata(row[valid], method="average") / valid.sum()
            positive[i] = valid & (percentile > 1 - tail)
            negative[i] = valid & (percentile <= tail)
    return positive, negative


def car_statistics(signal, labels, tradable, kind):
    """Daily positive/negative portfolio CAR relative to the equal-weight market."""
    positive, negative = signal_masks(signal, kind)
    rows = []
    for horizon, returns in labels.items():
        market = np.nanmean(np.where(tradable, returns, np.nan), axis=1)
        series = {}
        for name, mask in (("positive", positive), ("negative", negative)):
            portfolio = np.array([
                np.nanmean(r[m & t]) if np.any(m & t & np.isfinite(r)) else np.nan
                for r, m, t in zip(returns, mask, tradable)
            ])
            series[name] = portfolio - market
        spread = series["positive"] - series["negative"]
        for name, values in (*series.items(), ("positive_minus_negative", spread)):
            rows.append({
                "horizon": horizon,
                "leg": name,
                "mean_car": np.nanmean(values),
                "t_value_hac": hac_t_stat(values, horizon - 1),
                "observations": int(np.isfinite(values).sum()),
            })
    return pd.DataFrame(rows)


def target_weights(signal, tradable, kind, event_holding_days=5, tail=0.2,
                   event_value_mode="sign"):
    """Build benchmark, long-only and long-short target weights."""
    benchmark = np.zeros_like(signal, float)
    factor = np.zeros_like(signal, float)
    long_short = np.zeros_like(signal, float)
    active = signal.copy()
    if kind == "event":
        active = np.zeros_like(signal, float)
        for i in range(len(signal)):
            window_sum = np.nansum(
                signal[max(0, i-event_holding_days+1):i+1], axis=0
            )
            active[i] = np.sign(window_sum) if event_value_mode == "sign" else window_sum
    positive, negative = signal_masks(active, kind, tail)
    for i in range(len(signal)):
        universe = tradable[i]
        valid = universe & np.isfinite(active[i])
        if universe.any():
            benchmark[i, universe] = 1 / universe.sum()
        if kind == "continuous":
            selected = positive[i] & valid
            if selected.any():
                factor[i, selected] = 1 / selected.sum()
        else:
            weights = valid.astype(float)
            strength = np.abs(active[i])
            weights[positive[i]] *= 1 + strength[positive[i]]
            weights[negative[i]] = 0
            if weights.sum() > 0:
                factor[i] = weights / weights.sum()
        pos = positive[i] & valid
        neg = negative[i] & valid
        if pos.any() and neg.any():
            if kind == "event" and event_value_mode == "magnitude":
                pos_strength = np.abs(active[i, pos])
                neg_strength = np.abs(active[i, neg])
                long_short[i, pos] = 0.5 * pos_strength / pos_strength.sum()
                long_short[i, neg] = -0.5 * neg_strength / neg_strength.sum()
            else:
                long_short[i, pos] = 0.5 / pos.sum()
                long_short[i, neg] = -0.5 / neg.sum()
    return {"benchmark": benchmark, "factor_portfolio": factor, "long_short": long_short}
def portfolio_returns(weights, daily_return, positions):
    result = np.zeros(len(weights))
    for i, position in enumerate(positions):
        result[i] = np.nansum(weights[i] * daily_return[position + 2])
    return result


def rebalance_weights(targets, interval):
    """Update targets every ``interval`` trading days and hold otherwise."""
    if interval < 1:
        raise ValueError("rebalance interval must be positive")
    held = np.empty_like(targets)
    for start in range(0, len(targets), interval):
        held[start:min(start + interval, len(targets))] = targets[start]
    return held


def max_drawdown(returns):
    nav = np.cumprod(1 + np.nan_to_num(returns))
    return np.min(nav / np.maximum.accumulate(nav) - 1)


def performance_metrics(returns, weights, benchmark=None):
    """Gross performance and target-weight turnover; no transaction-cost deduction."""
    returns = np.asarray(returns)
    turnover = np.r_[0.0, 0.5*np.abs(np.diff(weights, axis=0)).sum(axis=1)]
    metrics = {
        "annual_return": np.prod(1+returns)**(252/len(returns))-1,
        "annual_volatility": np.nanstd(returns)*np.sqrt(252),
        "sharpe": np.nanmean(returns)/np.nanstd(returns)*np.sqrt(252),
        "max_drawdown": max_drawdown(returns),
        "average_daily_turnover": np.nanmean(turnover),
    }
    if benchmark is not None:
        excess = returns-benchmark
        metrics["annual_excess_return"] = np.nanmean(excess)*252
        metrics["information_ratio"] = np.nanmean(excess)/np.nanstd(excess)*np.sqrt(252)
    return metrics


def grouped_returns(signal, returns, tradable, kind, groups=10):
    """Daily group excess returns without arbitrary splitting of zero event signals."""
    if kind == "event":
        masks = [signal < 0, signal == 0, signal > 0]
        names = ["negative", "no_event", "positive"]
    else:
        masks = [np.zeros(signal.shape, bool) for _ in range(groups)]
        for i, row in enumerate(signal):
            valid = tradable[i] & np.isfinite(row)
            if valid.sum() < groups:
                continue
            order = np.flatnonzero(valid)[np.argsort(row[valid], kind="stable")]
            for group, indices in enumerate(np.array_split(order, groups)):
                masks[group][i, indices] = True
        names = [f"G{i}" for i in range(1, groups + 1)]
    market = np.nanmean(np.where(tradable, returns, np.nan), axis=1)
    output = []
    for mask in masks:
        group_return = np.array([
            np.nanmean(r[m & t]) if np.any(m & t & np.isfinite(r)) else np.nan
            for r, m, t in zip(returns, mask, tradable)
        ])
        output.append(group_return - market)
    return names, np.asarray(output)


def subplot_grid(count, width=8, height=5, sharex=True, sharey=False):
    """Create a compact, automatically sized grid and hide unused axes."""
    columns = min(2, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows),
        sharex=sharex, sharey=sharey, squeeze=False,
    )
    flat = axes.flat
    for ax in list(flat)[count:]:
        ax.set_visible(False)
    return fig, axes

def save_plots(name, signal, labels, tradable, dates, portfolio_series, kind, output):
    horizons = list(labels)
    fig, axes = subplot_grid(len(horizons), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, horizon in zip(axes.flat, horizons):
        names, groups = grouped_returns(signal, labels[horizon], tradable, kind)
        for i, (label, values) in enumerate(zip(names, groups)):
            ax.plot(dates, np.nancumsum(values), label=label, color=colors[i % 10])
        ax.set_title(f"{name}: {horizon}D forward excess return")
        ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle(f"{name} grouped cumulative excess returns")
    fig.tight_layout(); fig.savefig(output / f"{name}_groups.png", dpi=160); plt.close(fig)

    periods = sorted(portfolio_series) if kind == "event" else [None]
    if kind == "event":
        fig, axes = subplot_grid(len(periods), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(figsize=(12, 7)); axes = np.asarray([axes])
    for ax, period in zip(axes.flat, periods):
        series = portfolio_series[period] if period is not None else portfolio_series
        for label, values in series.items():
            ax.plot(dates, np.cumprod(1 + values), label=label)
        title = f"{period}D rebalance" if period is not None else "portfolio NAV"
        ax.set_title(f"{name}: {title}")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle(f"{name}: portfolio NAV versus equal-weight benchmark")
    fig.tight_layout()
    fig.savefig(output / f"{name}_portfolios.png", dpi=160); plt.close(fig)


def save_horizon_diagnostics(name, ic, car, output):
    """Show IC and CAR/HAC significance together for every selected horizon."""
    horizons = ic["horizon"].tolist()
    fig, axes = subplot_grid(len(horizons), sharex=False, sharey=True)
    for ax, horizon in zip(axes.flat, horizons):
        ic_row = ic.loc[ic["horizon"] == horizon].iloc[0]
        rows = car.loc[car["horizon"] == horizon]
        labels = rows["leg"].str.replace("positive_minus_negative", "spread").tolist()
        bars = ax.bar(labels, rows["mean_car"], color=["#2ca02c", "#d62728", "#1f77b4"])
        for bar, t_value in zip(bars, rows["t_value_hac"]):
            y = bar.get_height()
            ax.annotate(f"HAC t={t_value:.2f}", (bar.get_x()+bar.get_width()/2, y),
                        xytext=(0, 4 if y >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{horizon}D | IC={ic_row['mean_ic']:.4f}, "
            f"ICIR={ic_row['icir_nonoverlap']:.2f}, IC HAC t={ic_row['ic_t_value_hac']:.2f}"
        )
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{name}: horizon diagnostics (CAR and HAC significance)")
    fig.tight_layout()
    fig.savefig(output / f"{name}_horizon_diagnostics.png", dpi=160)
    plt.close(fig)

def backtest_factor(spec, signal, labels, daily_return, tradable, positions, dates,
                    output, event_rebalance_days=DEFAULT_EVENT_REBALANCE_DAYS,
                    event_value_mode="sign"):
    ic_rows = []
    for horizon, returns in labels.items():
        ic = pearson_by_date(signal, returns)
        rank_ic = rank_ic_by_date(signal, returns)
        ic_rows.append({
            "factor": spec.name,
            "horizon": horizon,
            "mean_ic": np.nanmean(ic),
            "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(252),
            "mean_rank_ic": np.nanmean(rank_ic),
            "rank_icir": np.nanmean(rank_ic) / np.nanstd(rank_ic) * np.sqrt(252),
            "icir_nonoverlap": np.nanmean(ic[::horizon]) / np.nanstd(ic[::horizon]) * np.sqrt(252/horizon),
            "ic_t_value_hac": hac_t_stat(ic, horizon-1),
        })
    car = car_statistics(signal, labels, tradable, spec.kind)
    car.insert(0, "factor", spec.name)
    ic_table = pd.DataFrame(ic_rows)
    save_horizon_diagnostics(spec.name, ic_table, car, output)

    targets = target_weights(
        signal, tradable, spec.kind, spec.event_holding_days,
        event_value_mode=event_value_mode,
    )
    periods = event_rebalance_days if spec.kind == "event" else (1,)
    portfolio_series = {}
    performance_rows = []
    for period in periods:
        weights = {name: rebalance_weights(value, period) for name, value in targets.items()}
        series = {name: portfolio_returns(value, daily_return, positions)
                  for name, value in weights.items()}
        portfolio_series[period] = series
        benchmark = series["benchmark"]
        for name, returns in series.items():
            row = {"factor": spec.name, "rebalance_days": period, "portfolio": name}
            comparison = benchmark if name == "factor_portfolio" else None
            row.update(performance_metrics(returns, weights[name], comparison))
            performance_rows.append(row)
    plot_series = portfolio_series if spec.kind == "event" else portfolio_series[1]
    save_plots(spec.name, signal, labels, tradable, dates, plot_series, spec.kind, output)
    return ic_table, car, pd.DataFrame(performance_rows)
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=pd.Timestamp, default=DEFAULT_START)
    parser.add_argument("--end", type=pd.Timestamp, default=DEFAULT_END)
    parser.add_argument("--factors", nargs="+", choices=SPECS, default=list(SPECS))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "score")
    parser.add_argument("--no-neutralize", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--mad-limit", type=float, default=5.0)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
                        help="forward-return/CAR horizons in trading days")
    parser.add_argument("--event-value-modes", nargs="+",
                        choices=("sign", "magnitude"), default=["sign", "magnitude"],
                        help="event direction-only or magnitude-weighted expression")
    parser.add_argument("--event-rebalance-days", nargs="+", type=int,
                        default=list(DEFAULT_EVENT_REBALANCE_DAYS),
                        help="event-portfolio rebalance intervals in trading days")
    return parser.parse_args()


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = DataPool(DATA_ROOT, asset="stock"); data.asset_root = DATA_ROOT
    all_dates = pd.DatetimeIndex(data.axis.trade_dates)
    positions = np.flatnonzero((all_dates >= args.start) & (all_dates <= args.end))
    dates = all_dates[positions]
    if not len(dates):
        raise ValueError("selected date range contains no trading dates")

    specs = [SPECS[name] for name in args.factors]
    reports = load_reports(args.start, args.end)
    context = InMemoryScoreContext(data, reports)
    signals = build_signals(context, specs, dates)

    industry = data.read("industry/industry", data.axis.date_count - 1, 0)[positions]
    market_value = data.read("d_essentials/circ_mv", data.axis.date_count - 1, 0)[positions]
    tradable = data.read("basic/tradable", data.axis.date_count - 1, 0)[positions].astype(bool)
    daily_return = data.read("d_essentials/pct", data.axis.date_count - 1, 0) / 100
    horizons = tuple(dict.fromkeys(args.horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must contain positive integers")
    labels = forward_returns(daily_return, positions, horizons)

    ic_tables, car_tables, performance_tables = [], [], []
    for spec in specs:
        signal = signals[spec.name]
        if spec.kind == "continuous" and not args.no_preprocess:
            signal = robust_standardize_matrix(signal, args.mad_limit)
        if spec.neutralize and not args.no_neutralize:
            signal = neutralize_matrix(signal, industry, market_value)
            if not args.no_preprocess:
                signal = robust_standardize_matrix(signal, args.mad_limit)
        signal = np.where(tradable, signal, np.nan)
        modes = args.event_value_modes if spec.kind == "event" else [None]
        for mode in dict.fromkeys(modes):
            test_spec = spec if mode is None else FactorSpec(
                f"{spec.name}_{mode}", spec.factor_class, spec.kind,
                spec.neutralize, spec.event_holding_days,
            )
            test_signal = np.sign(signal) if mode == "sign" else signal
            ic, car, performance = backtest_factor(
                test_spec, test_signal, labels, daily_return, tradable,
                positions, dates, args.output,
                tuple(dict.fromkeys(args.event_rebalance_days)),
                event_value_mode=mode or "sign",
            )
            ic_tables.append(ic)
            car_tables.append(car)
            performance_tables.append(performance)

    ic = pd.concat(ic_tables, ignore_index=True)
    car = pd.concat(car_tables, ignore_index=True)
    performance = pd.concat(performance_tables, ignore_index=True)
    ic.to_csv(args.output / "single_factor_ic.csv", index=False, encoding="utf-8-sig")
    car.to_csv(args.output / "single_factor_car.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(args.output / "single_factor_performance.csv", index=False, encoding="utf-8-sig")
    print("\nIC\n", ic.to_string(index=False))
    print("\nCAR\n", car.to_string(index=False))
    print("\nPERFORMANCE\n", performance.to_string(index=False))
    print("saved:", args.output)


if __name__ == "__main__":
    main()