"""HS300 price-direction regimes and separate IV risk overlay."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.GetData import DataPool
from v2.UpdateData.config import ROOT


def load_inputs(root, chunk_size=256):
    """Reconstruct a proxy index from previous-day normalized index weights.

    Weight units may be percent or decimals. Missing constituent returns are
    excluded and remaining weights renormalized. This is not the official index.
    """
    root = Path(root)
    with DataPool(root, asset="stock") as data:
        dates = pd.DatetimeIndex(data.axis.trade_dates)
        shape = (len(data.axis.full_dates), 1)
        iv = np.memmap(root / "option/HS300_IV30.bin", shape=shape, dtype=np.float32, mode="r")
        values = np.array(iv[:len(dates), 0], dtype=float)
        del iv
        returns = np.full(len(dates), np.nan)
        for start in range(1, len(dates), chunk_size):
            end = min(len(dates) - 1, start + chunk_size - 1)
            pct = np.asarray(data.read("d_essentials/pct", end, start), dtype=float) / 100
            weights = np.asarray(data.read("index/weight/hs300_weight", end - 1, start - 1), dtype=float)
            valid = np.isfinite(pct) & np.isfinite(weights) & (weights > 0)
            weights = np.where(valid, weights, 0)
            total = weights.sum(axis=1)
            numerator = (np.where(valid, pct, 0) * weights).sum(axis=1)
            returns[start:end + 1] = np.divide(numerator, total, out=np.full(len(total), np.nan), where=total > 0)
    # Start the reconstructed NAV at the first actual return, not at zero-filled missing dates.
    result = pd.DataFrame({"se": values, "index_daily": returns}, index=dates)
    result.index.name = "date"
    if result.index.has_duplicates or result.index.isna().any():
        raise ValueError("local date axis contains duplicate or missing populated dates")
    return result


def build_regimes(df, fast=20, slow=60, confirm=3):
    """Close-of-day signals; IV describes risk, price trend describes direction."""
    df = df.copy()
    df["se"] = df.se.where(df.se > 0).ffill(limit=5)
    for window, name in ((20, "ma1"), (126, "ma2"), (252, "ma3")):
        df[name] = df.se.rolling(window).mean()
    df["index_nav"] = (1 + df.index_daily).cumprod()
    df["price_fast"] = df.index_nav.ewm(span=fast, adjust=False, min_periods=fast).mean()
    df["price_slow"] = df.index_nav.ewm(span=slow, adjust=False, min_periods=slow).mean()
    valid = df.index_nav.notna() & df.price_slow.notna()
    up = valid & (df.index_nav > df.price_slow) & (df.price_fast > df.price_slow)
    down = valid & (df.index_nav < df.price_slow) & (df.price_fast < df.price_slow)
    # Mixed trends remain neutral; no retrospective relabeling of confirmation days.
    df["regime"] = np.select([up.rolling(confirm).sum() == confirm,
                               down.rolling(confirm).sum() == confirm], [1, -1], default=0)
    df["iv_rising_risk"] = (df.ma1 > df.ma3) & (df.ma1.diff(5) > 0)
    # Preserve the old IV-only rule for comparison, using the same prediction sample.
    df["old_iv_regime"] = pd.Series(np.where(df.iv_rising_risk, -1, 1), index=df.index).where(df.ma3.notna())
    for horizon in (5, 20):
        df[f"forward_{horizon}d"] = df.index_nav.shift(-horizon) / df.index_nav - 1
    return df


def hysteresis_state(score, threshold=0.5):
    """Switch immediately at opposite bands, retain state inside the buffer.

    No future-state forecast, consecutive-day confirmation or minimum hold.
    Missing scores preserve an established state but cannot trigger changes.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    state = 0
    values = np.zeros(len(score), dtype=np.int8)
    for t, value in enumerate(score):
        if np.isfinite(value):
            if value >= threshold:
                state = 1
            elif value <= -threshold:
                state = -1
        values[t] = state
    return pd.Series(values, index=score.index)


def add_early_signal(df, threshold=0.5, iv_weight=0.7):
    """Experimental risk-on score from faster IV and price momentum.

    IV falling relative to its short trend contributes positively; rising IV
    negatively. Scale using trailing realized changes, not full-sample moments.
    This heuristic is not a demonstrated forecast of subsequent returns.
    """
    if not 0 <= iv_weight <= 1:
        raise ValueError("iv_weight must be between 0 and 1")
    df = df.copy()
    log_iv = np.log(df.se.where(df.se > 0))
    fast_iv = log_iv.ewm(span=5, adjust=False, min_periods=5).mean()
    slow_iv = log_iv.ewm(span=20, adjust=False, min_periods=20).mean()
    iv_scale = log_iv.diff().rolling(60, min_periods=40).std() * np.sqrt(10)
    df["iv_momentum_score"] = -(fast_iv - slow_iv) / iv_scale.where(iv_scale > 1e-8)
    price_scale = df.index_daily.rolling(20).std() * np.sqrt(5)
    df["price_momentum_score"] = df.index_nav.pct_change(5, fill_method=None) / price_scale.where(price_scale > 1e-8)
    score = iv_weight * df.iv_momentum_score + (1 - iv_weight) * df.price_momentum_score
    # Clip only extreme scores, then smooth without delaying for confirmation.
    df["signal_score"] = score.clip(-3, 3).ewm(span=3, adjust=False, min_periods=3).mean().where(score.notna())
    df["price_regime"] = df.regime
    df["regime"] = hysteresis_state(df.signal_score, threshold)
    return df


def switching_statistics(df, threshold=0.5):
    """Count observed runs; short runs describe instability, not trade losses."""
    rows = []
    for sample, selected in (("all", df), ("2024_onward", df.loc["2024-01-01":])):
        valid = selected.signal_score.notna()
        for name, states in (("iv_only", selected.old_iv_regime),
                             ("price_confirm3", selected.price_regime),
                             ("score_zero_threshold", np.sign(selected.signal_score)),
                             ("score_hysteresis", hysteresis_state(df.signal_score, threshold).reindex(selected.index))):
            states = states.loc[valid].dropna()
            if states.empty:
                continue
            cuts = np.r_[0, np.flatnonzero(states.to_numpy()[1:] != states.to_numpy()[:-1]) + 1, len(states)]
            runs = np.diff(cuts)
            # First and last runs may be censored by the sample boundaries.
            complete = runs[1:-1]
            rows.append(dict(sample=sample, model=name, switches=len(runs) - 1,
                             median_complete_run=np.median(complete) if len(complete) else np.nan,
                             complete_runs_le5=int((complete <= 5).sum())))
    return pd.DataFrame(rows)


def plot_signal_score(df, threshold=0.5):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df.index, df.signal_score, lw=.9, color="#4a6581", label="IV / price momentum score")
    ax.axhline(threshold, color="#54875e", ls="--", label="Switch to risk-on")
    ax.axhline(-threshold, color="#b55d55", ls="--", label="Switch to defensive")
    ax.axhspan(-threshold, threshold, color="gray", alpha=.1, label="Keep previous state")
    ax.axhline(0, color="gray", lw=.5)
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("Immediate threshold triggers; no waiting for future 3/5-day states")
    ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


def annual_stability(df):
    """Group by signal year; overlapping forward returns are descriptive only."""
    rows = []
    switched = df.regime.ne(df.regime.shift()) & df.regime.shift().notna()
    for year, group in df.groupby(df.index.year):
        row = dict(year=year, first=str(group.index[0].date()), last=str(group.index[-1].date()),
                   days=len(group), switches=int(switched.loc[group.index].sum()),
                   risk_on_days=int(group.regime.eq(1).sum()), defensive_days=int(group.regime.eq(-1).sum()))
        for h in (5, 20):
            for state, name in ((1, "risk_on"), (-1, "defensive")):
                values = group.loc[group.regime.eq(state), f"forward_{h}d"].dropna()
                row[f"{name}_count_{h}d"] = len(values)
                row[f"{name}_mean_{h}d"] = values.mean()
                row[f"{name}_up_fraction_{h}d"] = (values > 0).mean() if len(values) else np.nan
            row[f"separation_{h}d"] = row[f"risk_on_mean_{h}d"] - row[f"defensive_mean_{h}d"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def plot_annual_stability(annual):
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    annual.switches.plot.bar(ax=axes[0], color="#617d9d", rot=0)
    axes[0].set_title("State switches by calendar year (partial years included)")
    annual[["risk_on_mean_20d", "defensive_mean_20d"]].plot.bar(ax=axes[1], rot=0, color=["#679276", "#b56c61"])
    axes[1].set_title("Mean subsequent 20-day return by signal year and state")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axhline(0, color="gray", linewidth=.7)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    return fig


def regime_statistics(df):
    """Descriptive forward performance, not optimized thresholds or strategy PnL."""
    rows = []
    for model in ("old_iv_regime", "price_regime", "regime"):
        for sample, mask in (("all", np.ones(len(df), dtype=bool)),
                             ("before_2024", df.index < "2024-01-01"),
                             ("2024_onward", df.index >= "2024-01-01")):
            for state, group in df.loc[mask].groupby(model):
                for horizon in (5, 20):
                    values = group[f"forward_{horizon}d"].dropna()
                    rows.append(dict(model=model, sample=sample, state=state, horizon=horizon,
                                     count=len(values), mean_forward_return=values.mean(),
                                     up_fraction=(values > 0).mean() if len(values) else np.nan))
    return pd.DataFrame(rows)


def plot_regimes(df):
    """Dual date axes share numeric coordinates; backgrounds never obscure NAV."""
    fig, (ax, ax_price) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    ax_ret = ax.twinx()
    x = mdates.date2num(df.index.to_numpy(dtype="datetime64[us]"))
    colors = {1: "#c7e2ce", -1: "#efd0cc", 0: "#eeeeee"}
    states = df.regime.to_numpy()
    bounds = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(df)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        for target in (ax, ax_price):
            target.axvspan(x[left], x[right] if right < len(x) else x[-1],
                           color=colors[states[left]], alpha=.4, linewidth=0, zorder=0)
    for col, label, color in (("se", "IV30", "#687687"), ("ma1", "IV MA20", "#dc9c3e"),
                               ("ma2", "IV MA126", "#5483b0"), ("ma3", "IV MA252", "#8c70a5")):
        ax.plot(x, df[col], lw=1, label=label, color=color)
    nav = df.index_nav / df.index_nav.iloc[0]
    ax_ret.plot(x, nav - 1, color="black", lw=1.6, label="HS300 proxy cumulative return")
    ax_ret.patch.set_visible(False)
    ax_ret.set_zorder(ax.get_zorder() + 1)
    ax.set_ylabel("Implied volatility (%)")
    ax_ret.set_ylabel("Cumulative return (decimal)")
    for column, label in (("index_nav", "HS300 proxy NAV"), ("price_fast", "Price EMA20"), ("price_slow", "Price EMA60")):
        ax_price.plot(x, df[column] / df.index_nav.iloc[0], lw=1, label=label)
    ax_price.set_ylabel("Rebased NAV")
    ax_price.legend(loc="upper left", ncol=3)
    for target in (ax, ax_price):
        target.grid(alpha=.15)
        target.xaxis_date()
    locator = mdates.AutoDateLocator()
    ax_price.xaxis.set_major_locator(locator)
    ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlim(x[0], x[-1])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_ret.get_legend_handles_labels()
    ax.legend(h1 + h2 + [Patch(facecolor=colors[s]) for s in (1, 0, -1)],
              l1 + l2 + ["Risk-on", "Not initialized", "Defensive"], loc="upper left", ncol=4)
    ax.set_title("HS300 research signal: IV / price momentum + hysteresis; immediate triggers")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    started = perf_counter()
    start_date = None  # Use all locally available, sufficiently warmed-up data.
    output = Path(__file__).resolve().parent / "output/iv_regime"
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(ROOT)
    observed_iv = inputs.se.where(inputs.se > 0).dropna()
    print(f"Price axis: {inputs.index[0].date()} to {inputs.index[-1].date()}")
    print(f"Observed IV: {observed_iv.index[0].date()} to {observed_iv.index[-1].date()}, observations={len(observed_iv)}")
    df = add_early_signal(build_regimes(inputs))
    if start_date is not None:
        df = df.loc[start_date:]
    # The score needs roughly 60 IV observations, not a full 252-day IV MA.
    # Keep the early sample; long IV MAs may legitimately be missing on the plot.
    df = df.dropna(subset=["se", "signal_score", "index_nav", "price_slow"])
    if len(df) < 2:
        raise ValueError("insufficient overlapping IV and price history")
    df["index_ret"] = df.index_nav / df.index_nav.iloc[0] - 1
    summary = regime_statistics(df)
    df.to_csv(output / "hs300_iv_regimes.csv")
    summary.to_csv(output / "regime_forward_returns.csv", index=False)
    annual = annual_stability(df)
    annual.to_csv(output / "annual_stability.csv")
    print(annual[["days", "switches", "risk_on_mean_20d", "defensive_mean_20d", "separation_20d"]].to_string())
    annual_figure = plot_annual_stability(annual)
    annual_figure.savefig(output / "annual_stability.png", dpi=160)
    switches = switching_statistics(df)
    switches.to_csv(output / "switching_statistics.csv", index=False)
    print(switches.to_string(index=False))
    figure = plot_regimes(df)
    figure.savefig(output / "hs300_iv_regimes.png", dpi=160)
    score_figure = plot_signal_score(df)
    score_figure.savefig(output / "hs300_signal_score.png", dpi=160)
    print(summary.loc[(summary['sample'] == 'all') & (summary.horizon == 20)].to_string(index=False))
    print(f"range={df.index[0].date()} to {df.index[-1].date()}, saved={output}, elapsed={perf_counter()-started:.2f}s")
    if plt.get_backend().lower() != "agg":
        plt.show()
