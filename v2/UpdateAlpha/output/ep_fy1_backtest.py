"""Latest-three-year decile backtest for EPFY1 and inverse PEG."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
from v2.UpdateAlpha.analyst_forecast.ep_fy1 import (
    EPFY1Context,
    EPFY1Factor,
    PEGInverseFactor,
)
from v2.UpdateAlpha.analyst_forecast.utils import _consensus_fy1_year, _date

DATA_ROOT = Path("D:/data")
OUTPUT = Path(__file__).resolve().parent / "ep_fy1"
HORIZONS = (1, 5, 10, 20)
FACTORS = (EPFY1Factor, PEGInverseFactor)
YEARS = 3


class BacktestContext(EPFY1Context):
    """Bulk-loaded daily snapshots with production-context semantics."""

    def preload(self, start, end):
        start, end = _date(start), _date(end)
        segments = []
        cursor = start
        while cursor <= end:
            may_first = pd.Timestamp(cursor.year, 5, 1).date()
            if cursor < may_first:
                segment_end = min(end, may_first - pd.Timedelta(days=1))
            else:
                segment_end = min(
                    end, pd.Timestamp(cursor.year + 1, 5, 1).date()
                    - pd.Timedelta(days=1)
                )
            fy1_year = _consensus_fy1_year(cursor)
            sql = f"""
            SELECT
                id, stock_code, con_date, con_year, con_np,
                con_npcgrate_2y, entrytime
            FROM con_forecast_stk
            WHERE con_date BETWEEN '{cursor}' AND '{segment_end}'
                AND con_year = {fy1_year}
                AND entrytime < DATEADD(day, 1, CAST(con_date AS datetime))
                AND con_np IS NOT NULL
            """
            print(
                f"  loading {cursor} to {segment_end}, FY1={fy1_year}",
                flush=True,
            )
            segments.append(
                pl.read_database(sql, self.conn, infer_schema_length=None)
            )
            cursor = segment_end + pd.Timedelta(days=1)

        self._all_consensus = (
            pl.concat(segments, how="vertical_relaxed")
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("con_date").cast(pl.Date, strict=False),
                pl.col("con_year").cast(pl.Int32, strict=False),
                pl.col("con_np").cast(pl.Float64, strict=False),
                pl.col("con_npcgrate_2y").cast(pl.Float64, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
            )
            .filter(pl.col("tick").is_not_null() & pl.col("con_np").is_finite())
            .sort(["con_date", "tick", "con_year", "entrytime", "id"])
            .unique(
                ["con_date", "tick", "con_year"],
                keep="last", maintain_order=True,
            )
            .select(
                "tick", "con_date", "con_year", "con_np", "con_npcgrate_2y"
            )
        )
    def consensus(self, asof):
        asof = _date(asof)
        fy1_year = _consensus_fy1_year(asof)
        return (
            self._all_consensus.filter(
                (pl.col("con_date") == asof)
                & (pl.col("con_year") == fy1_year)
            )
            .select("tick", "con_np", "con_npcgrate_2y")
        )


def forward_return(daily_return, horizon):
    label = np.full_like(daily_return, np.nan, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(
        daily_return[2:], horizon, axis=0
    )
    label[:len(windows)] = np.prod(1.0 + windows, axis=-1) - 1.0
    return label


def calculate_signals(context, dates):
    factors = [factor_class(context) for factor_class in FACTORS]
    signals = {
        factor.meta.name: np.full(
            (len(dates), context.data.axis.tick_count), np.nan, dtype=np.float32
        )
        for factor in factors
    }
    for row, date in enumerate(tqdm(dates, desc="Calculating EP/PEG")):
        for factor in factors:
            signals[factor.meta.name][row] = factor.calculate(date)
    return signals

def winsorize_zscore(values):
    """Cross-sectional 1%/99% winsorization followed by z-score."""
    output = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 20:
        return output
    sample = values[valid]
    lower, upper = np.nanquantile(sample, [0.01, 0.99])
    sample = np.clip(sample, lower, upper)
    std = np.nanstd(sample)
    if not np.isfinite(std) or std <= 0:
        return output
    output[valid] = (sample - np.nanmean(sample)) / std
    return output


def neutralize(signal, industry, market_value):
    """Remove daily industry dummies and log-market-value exposure."""
    result = np.full_like(signal, np.nan, dtype=float)
    for row in tqdm(range(len(signal)), desc="Neutralizing"):
        y = winsorize_zscore(signal[row])
        ind = industry[row]
        mv = market_value[row]
        valid = (
            np.isfinite(y) & np.isfinite(ind) & np.isfinite(mv) & (mv > 0)
        )
        if valid.sum() < 50:
            continue
        y_valid = y[valid]
        log_mv = np.log(mv[valid])
        log_mv = (log_mv - log_mv.mean()) / log_mv.std()
        codes, inverse = np.unique(ind[valid], return_inverse=True)
        dummies = np.eye(len(codes), dtype=float)[inverse]
        design = np.column_stack([np.ones(valid.sum()), log_mv, dummies[:, 1:]])
        residual = y_valid - design @ np.linalg.lstsq(
            design, y_valid, rcond=None
        )[0]
        residual_std = residual.std()
        if np.isfinite(residual_std) and residual_std > 0:
            result[row, valid] = (residual - residual.mean()) / residual_std
    return result

def evaluate(signal, daily_return, dates, factor_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rows = []
    for ax, horizon in zip(axes.flat, HORIZONS):
        label = forward_return(daily_return, horizon)
        ic = IC(signal, label)
        rank_ic = rankIC(signal, label)
        group_return = calc_group_ret(signal, label, 10)
        cumulative = np.nancumsum(group_return, axis=1)
        group_mean = np.nanmean(group_return, axis=1)
        for group, values in enumerate(cumulative, 1):
            ax.plot(
                dates[:len(values)], values, color=colors[group - 1],
                linewidth=1.1, label=f"G{group}",
            )
        rows.append({
            "factor": factor_name,
            "horizon": horizon,
            "mean_ic": np.nanmean(ic),
            "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(242),
            "mean_rank_ic": np.nanmean(rank_ic),
            "rank_icir": np.nanmean(rank_ic) / np.nanstd(rank_ic) * np.sqrt(242),
            "mean_g10_g1": group_mean[-1] - group_mean[0],
            "monotonic_corr": np.corrcoef(np.arange(1, 11), group_mean)[0, 1],
            "adjacent_order_rate": np.mean(np.diff(group_mean) > 0),
            "mean_coverage": np.mean(np.sum(np.isfinite(signal), axis=1)),
        })
        ax.set_title(
            f"{factor_name} {horizon}D | IC={np.nanmean(ic):.4f}, "
            f"RankIC={np.nanmean(rank_ic):.4f}"
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=7)

    fig.suptitle(f"{factor_name}: Decile Cumulative Excess Returns (Latest 3Y)")
    fig.supxlabel("Trade Date")
    fig.supylabel("Cumulative Group Excess Return")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"{factor_name}_group_returns_3y.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with BacktestContext(root=DATA_ROOT) as context:
        context.data.asset_root = DATA_ROOT
        all_dates = np.asarray(context.data["trade_dates"])
        end_pos = context.data.axis.date_count - 1
        end = pd.Timestamp(all_dates[end_pos])
        start_pos = int(np.searchsorted(
            all_dates, np.datetime64((end - pd.DateOffset(years=YEARS)).date())
        ))
        dates = all_dates[start_pos:end_pos + 1]
        daily_return = np.asarray(context.data.read(
            "d_essentials/pct", start_date=start_pos, end_date=end_pos
        ), dtype=float) / 100.0
        tradable = np.asarray(context.data.read(
            "basic/tradable", start_date=start_pos, end_date=end_pos
        ), dtype=bool)
        market_value = np.asarray(context.data.read(
            "d_essentials/total_mv", start_date=start_pos, end_date=end_pos
        ), dtype=float)
        industry = np.asarray(context.data.read(
            "industry/industry", start_date=start_pos, end_date=end_pos
        ), dtype=float)
        print("Bulk-loading ZYYX FY1 snapshots...", flush=True)
        context.preload(dates[0], dates[-1])
        signals = calculate_signals(context, dates)

    for factor_name in tuple(signals):
        raw = np.where(tradable, signals[factor_name], np.nan)
        signals[factor_name] = raw
        signals[f"{factor_name}_neutral"] = neutralize(
            raw, industry, market_value
        )

    rows = []
    for factor_name, signal in signals.items():
        rows.extend(evaluate(signal, daily_return, dates, factor_name))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "ep_peg_summary_3y.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()