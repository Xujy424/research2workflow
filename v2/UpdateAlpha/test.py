"""Compare daily, weekly, and monthly rebalancing over 5/3/1-year windows."""

from pathlib import Path

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Data paths and factor direction. direction=-1 means long the lowest-ranked
# bucket and short the highest-ranked bucket.
ROOT = Path("Z:/")
DATES_PATH = ROOT / "axis" / "dates.npy"
TICKS_PATH = ROOT / "axis" / "stock_ticks.npy"
FACTOR_PATH = ROOT / "stock" / "factor_pool" / "gf_ratio_realupvar.bin"
FACTOR_NAME = FACTOR_PATH.stem
RETURN_PATH = ROOT / "stock" / "d_essentials" / "pct.bin"
DIRECTION = 1
END_DATE = "2026-09-30"
WINDOWS = {
    "5年 (2021-01-01)": "2021-01-01",
    "3年 (2023-01-01)": "2023-01-01",
    "1年 (2025-09-30)": "2025-09-30",
}
FREQUENCIES = ("daily", "weekly", "monthly")
OUTPUT_DIR = Path(__file__).resolve().parent


def make_groups(alpha, label, num_groups=10):
    """Calculate daily demeaned equal-weight returns for factor buckets."""
    rank = bn.nanrankdata(alpha, axis=1)
    n_valid = np.isfinite(alpha).sum(axis=1)
    bucket_size = n_valid // num_groups
    result = np.full((num_groups, len(alpha)), np.nan, dtype=np.float64)
    for group in range(num_groups):
        lower = (bucket_size * group)[:, None]
        upper = (
            n_valid if group == num_groups - 1
            else bucket_size * (group + 1)
        )[:, None]
        mask = (rank > lower) & (rank <= upper) & np.isfinite(alpha)
        selected = np.where(mask, label, np.nan)
        with np.errstate(invalid="ignore"):
            result[group] = np.nanmean(selected, axis=1)
    with np.errstate(invalid="ignore"):
        result -= np.nanmean(result, axis=0)
    return result


def period_first_trading_days(dates, frequency):
    dates = pd.DatetimeIndex(dates)
    if frequency == "daily":
        return np.arange(len(dates))
    periods = dates.to_period("W-FRI" if frequency == "weekly" else "M")
    return np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]])


def make_long_short(alpha, returns, dates, frequency, direction):
    """Hold each rebalance-date top/bottom decile portfolio until next rebalance."""
    rebalance_rows = period_first_trading_days(dates, frequency)
    strategy_returns = np.full(len(dates), np.nan, dtype=np.float64)
    for k, start in enumerate(rebalance_rows):
        end = rebalance_rows[k + 1] if k + 1 < len(rebalance_rows) else len(dates)
        signal = alpha[start]
        valid = np.isfinite(signal)
        if valid.sum() < 10:
            continue
        ranks = bn.nanrankdata(signal)
        n_valid = int(valid.sum())
        bucket_size = n_valid // 10
        if direction == -1:
            long_mask = (ranks > 0) & (ranks <= bucket_size)
            short_mask = (ranks > bucket_size * 9) & (ranks <= n_valid)
        elif direction == 1:
            long_mask = (ranks > bucket_size * 9) & (ranks <= n_valid)
            short_mask = (ranks > 0) & (ranks <= bucket_size)
        else:
            raise ValueError("direction must be either -1 or 1")
        long_mask &= valid
        short_mask &= valid
        if not long_mask.any() or not short_mask.any():
            continue
        with np.errstate(invalid="ignore"):
            long_daily = np.nanmean(returns[start:end, long_mask], axis=1)
            short_daily = np.nanmean(returns[start:end, short_mask], axis=1)
        strategy_returns[start:end] = long_daily - short_daily
    return pd.Series(strategy_returns, index=dates, name="long_short")


def summarize(returns, annual_days=242):
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.Series({
            "total_return": np.nan, "annual_return": np.nan,
            "annual_volatility": np.nan, "sharpe": np.nan,
            "max_drawdown": np.nan, "win_rate": np.nan,
        }), clean, clean
    nav = (1.0 + clean).cumprod()
    years = len(clean) / annual_days
    annual_return = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    volatility = clean.std(ddof=1) * np.sqrt(annual_days)
    sharpe = clean.mean() / clean.std(ddof=1) * np.sqrt(annual_days) if clean.std(ddof=1) > 0 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    stats = pd.Series({
        "total_return": nav.iloc[-1] - 1.0,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "win_rate": (clean > 0).mean(),
    })
    return stats, nav, drawdown


def main():
    dates_raw = np.load(DATES_PATH, allow_pickle=True)
    ticks = np.load(TICKS_PATH, allow_pickle=True)
    all_dates = pd.DatetimeIndex(pd.to_datetime(dates_raw))
    factor = np.memmap(
        FACTOR_PATH, dtype="float32", mode="r", shape=(len(all_dates), len(ticks))
    )
    pct = np.memmap(
        RETURN_PATH, dtype="float32", mode="r", shape=(len(all_dates), len(ticks))
    )
    end_pos = int(np.searchsorted(all_dates, pd.Timestamp(END_DATE), side="right"))
    if end_pos == 0:
        raise ValueError(f"No trading dates on or before {END_DATE}")
    # Existing convention: shift returns by two trading sessions for execution.
    stop_pos = min(end_pos + 2, len(all_dates))
    returns_all = np.asarray(pct[:stop_pos], dtype=np.float64) / 100.0
    returns_all[:-2] = returns_all[2:]
    returns_all[-2:] = np.nan
    axis_dates = all_dates[:end_pos]
    factor_all = np.asarray(factor[:end_pos], dtype=np.float64)
    returns_all = returns_all[:end_pos]

    outputs = {}
    group_outputs = {}
    for window_name, start_date in WINDOWS.items():
        start = int(np.searchsorted(axis_dates, pd.Timestamp(start_date), side="left"))
        if start >= len(axis_dates):
            raise ValueError(f"No trading dates found for window {window_name}")
        dates = axis_dates[start:]
        alpha = factor_all[start:]
        returns = returns_all[start:]
        groups = make_groups(alpha, returns, num_groups=10)
        group_outputs[window_name] = pd.DataFrame(
            groups.T, index=dates, columns=[f"G{i}" for i in range(1, 11)]
        ).cumsum()
        for frequency in FREQUENCIES:
            series = make_long_short(alpha, returns, dates, frequency, DIRECTION)
            stats, nav, drawdown = summarize(series)
            outputs[(window_name, frequency)] = {
                "returns": series, "stats": stats, "nav": nav,
                "drawdown": drawdown,
            }

    # Summary table: metrics down rows, 5/3/1-year windows across columns,
    # all using the same weekly rebalancing rule.
    table = pd.DataFrame({
        window_name: outputs[(window_name, "weekly")]["stats"]
        for window_name in WINDOWS
    })
    table.columns.name = "回测区间（周度调仓）"
    table_path = OUTPUT_DIR / f"{FACTOR_NAME}_rebalance_5y_3y_1y.csv"
    table.to_csv(table_path, encoding="utf-8-sig", float_format="%.6f")
    print("绩效汇总（周度调仓；列为 5/3/1 年区间）")
    percent_metrics = {"total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"}
    formatted_table = table.copy()
    for metric in formatted_table.index:
        if metric in percent_metrics:
            formatted_table.loc[metric] = formatted_table.loc[metric].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else "nan"
            )
        else:
            formatted_table.loc[metric] = formatted_table.loc[metric].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else "nan"
            )
    print(formatted_table.to_string())

    # Original two-column layout: NAV on the left, factor groups on the right.
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharex=False, constrained_layout=True)
    colors = {"daily": "#176B87", "weekly": "#D95F02", "monthly": "#5B8E7D"}
    for row, window_name in enumerate(WINDOWS):
        nav_axis, group_axis = axes[row]
        for frequency in FREQUENCIES:
            nav = outputs[(window_name, frequency)]["nav"]
            if not nav.empty:
                nav_axis.plot(nav.index, nav.values, label=frequency, color=colors[frequency], linewidth=1.25)
        nav_axis.set_title(f"{window_name} · Long-short NAV")
        nav_axis.set_ylabel("Net value")
        nav_axis.grid(alpha=0.25)
        nav_axis.legend(title="Rebalance", ncol=3)
        group_cumret = group_outputs[window_name]
        group_axis.plot(group_cumret.index, group_cumret.values, linewidth=0.9)
        group_axis.set_title(f"{window_name} · Daily Factor Group Cumulative Returns")
        group_axis.set_ylabel("Cumulative demeaned return")
        group_axis.legend(group_cumret.columns, ncol=2, fontsize=8)
        group_axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")
    fig.suptitle(f"{FACTOR_NAME}: 5/3/1-Year Results", fontsize=15)
    chart_path = OUTPUT_DIR / f"{FACTOR_NAME}_rebalance_5y_3y_1y.png"
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    print(f"表格已保存: {table_path}")
    print(f"图表已保存: {chart_path}")
    plt.show()


if __name__ == "__main__":
    main()
