from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from v2.GetData import DataPool
from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
from v2.UpdateData.config import get_jy_conn, get_zyyx_conn

DATA_ROOT = Path(r"D:\data")
OUT = Path(__file__).resolve().parent / "cov"
START = pd.Timestamp("2012-02-13")
END = pd.Timestamp("2015-02-13")
LOOKBACK = 180


def _chunks(left, right, months=6):
    while left <= right:
        end = min(left + pd.DateOffset(months=months) - pd.Timedelta(days=1), right)
        yield left, end
        left = end + pd.Timedelta(days=1)


def load_reports():
    conn = get_zyyx_conn()
    frames = []
    left0 = START - pd.Timedelta(days=LOOKBACK)
    for left, right in _chunks(left0, END):
        print(f"reports {left:%Y-%m-%d} .. {right:%Y-%m-%d}", flush=True)
        sql = f"""
        SELECT f.report_id, f.stock_code, f.organ_id, f.create_date,
               MAX(f.entrytime) AS entrytime,
               MAX(CASE WHEN ISNULL(f.gg_rating_code, '0') <> '0' THEN 1 ELSE 0 END) AS has_rating
        FROM rpt_forecast_stk f
        WHERE f.create_date BETWEEN '{left:%Y-%m-%d}' AND '{right:%Y-%m-%d}'
          AND f.entrytime <= '{END:%Y-%m-%d} 23:59:59'
          AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
          AND (f.reliability >= 5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL
        GROUP BY f.report_id, f.stock_code, f.organ_id, f.create_date
        """
        frames.append(pd.read_sql(sql, conn))
    out = pd.concat(frames, ignore_index=True)
    out["tick"] = out.stock_code.astype(str).str.zfill(6)
    out["create_date"] = pd.to_datetime(out.create_date).dt.normalize()
    out["entrytime"] = pd.to_datetime(out.entrytime)
    return out.drop_duplicates(["report_id", "tick"]).sort_values(["tick", "create_date", "report_id"])


def load_first_events(reports):
    """Identify first, first-rated, and >1Y restarted institution coverage."""
    conn = get_zyyx_conn()
    boundary = START - pd.Timedelta(days=LOOKBACK)
    sql = f"""
    SELECT f.stock_code, f.organ_id, MAX(f.create_date) AS last_date,
           MAX(CASE WHEN ISNULL(f.gg_rating_code, '0') <> '0' THEN 1 ELSE 0 END) AS prior_rated
    FROM rpt_forecast_stk f
    WHERE f.create_date < '{boundary:%Y-%m-%d}'
      AND f.entrytime < '{boundary:%Y-%m-%d}'
      AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
      AND (f.reliability >= 5 OR f.reliability IS NULL)
      AND f.organ_id IS NOT NULL
    GROUP BY f.stock_code, f.organ_id
    """
    prior = pd.read_sql(sql, conn)
    prior["tick"] = prior.stock_code.astype(str).str.zfill(6)
    prior["last_date"] = pd.to_datetime(prior.last_date).dt.normalize()
    state = {(r.tick, r.organ_id): [r.last_date, bool(r.prior_rated)] for r in prior.itertuples()}
    events = []
    ordered = reports.sort_values(["create_date", "report_id"])
    for r in ordered.itertuples():
        key = (r.tick, r.organ_id)
        last, rated = state.get(key, [pd.NaT, False])
        first = pd.isna(last)
        restarted = (not first) and (r.create_date - last).days > 365
        first_rated = bool(r.has_rating) and not rated
        if first or restarted or first_rated:
            events.append((r.report_id, r.tick, r.organ_id, r.create_date, r.entrytime))
        state[key] = [r.create_date if first or r.create_date > last else last,
                      rated or bool(r.has_rating)]
    return pd.DataFrame(events, columns=["report_id", "tick", "organ_id", "create_date", "entrytime"])

def load_announcements():
    conn = get_jy_conn()
    start = (START - pd.Timedelta(days=LOOKBACK + 7)).strftime("%Y-%m-%d")
    end = END.strftime("%Y-%m-%d")
    sql = f"""
    SELECT s.SecuCode AS tick, x.InfoPublDate AS announce_date
    FROM (
      SELECT CompanyCode, InfoPublDate FROM LC_IncomeStatementAll
       WHERE InfoPublDate BETWEEN '{start}' AND '{end}'
         AND IfMerged=1 AND IfAdjusted=2 AND IfComplete=1 AND BulletinType IN (20,30)
      UNION ALL
      SELECT CompanyCode, InfoPublDate FROM LC_STIBIncomeState
       WHERE InfoPublDate BETWEEN '{start}' AND '{end}'
         AND IfMerged=1 AND IfAdjusted=2 AND IfComplete=1
      UNION ALL
      SELECT CompanyCode, InfoPublDate FROM LC_PerformanceForecast
       WHERE InfoPublDate BETWEEN '{start}' AND '{end}'
      UNION ALL
      SELECT CompanyCode, InfoPublDate FROM LC_STIBPerformForecast
       WHERE InfoPublDate BETWEEN '{start}' AND '{end}'
      UNION ALL
      SELECT CompanyCode, InfoPublDate FROM LC_PerformanceLetters
       WHERE InfoPublDate BETWEEN '{start}' AND '{end}'
    ) x
    JOIN SecuMain s ON x.CompanyCode=s.CompanyCode
    WHERE s.SecuCategory=1 AND s.SecuMarket IN (83,90)
    """
    out = pd.read_sql(sql, conn)
    out["tick"] = out.tick.astype(str).str.zfill(6)
    out["announce_date"] = pd.to_datetime(out.announce_date).dt.normalize()
    return out.dropna().drop_duplicates(["tick", "announce_date"]).sort_values(["tick", "announce_date"])


def load_holdings():
    conn = get_jy_conn()
    sql = f"""
    SELECT s.SecuCode AS tick, h.StatDate AS available_date, h.EndDate,
           h.InstitutionsHoldProp AS institution_hold
    FROM LC_StockHoldingSt h
    JOIN SecuMain s ON h.InnerCode=s.InnerCode
    WHERE h.StatDate <= '{END:%Y-%m-%d}'
      AND h.StatDate >= '2010-01-01'
      AND h.InstitutionsHoldProp IS NOT NULL
      AND s.SecuCategory=1 AND s.SecuMarket IN (83,90)
    """
    out = pd.read_sql(sql, conn)
    out["tick"] = out.tick.astype(str).str.zfill(6)
    out["available_date"] = pd.to_datetime(out.available_date).dt.normalize()
    out["EndDate"] = pd.to_datetime(out.EndDate)
    return (out.sort_values(["tick", "available_date", "EndDate"])
               .drop_duplicates(["tick", "available_date"], keep="last"))


def mark_post_announcement(reports, announcements):
    qualified = np.zeros(len(reports), dtype=bool)
    ann_groups = {k: g.announce_date.values.astype("datetime64[D]") for k, g in announcements.groupby("tick")}
    for tick, ix in reports.groupby("tick").groups.items():
        dates = ann_groups.get(tick)
        if dates is None:
            continue
        report_dates = reports.loc[ix, "create_date"].values.astype("datetime64[D]")
        pos = np.searchsorted(dates, report_dates, side="right") - 1
        ok = pos >= 0
        delta = np.full(len(pos), 99, dtype=int)
        delta[ok] = (report_dates[ok] - dates[pos[ok]]).astype("timedelta64[D]").astype(int)
        qualified[np.asarray(ix)] = ok & (delta <= 7)
    out = reports.copy()
    out["post_announcement"] = qualified
    return out


def build_factors(trade_dates, ticks, reports, first_events, announcements, holdings):
    tickpos = {str(t): i for i, t in enumerate(ticks)}
    shape = (len(trade_dates), len(ticks))
    anncov = np.full(shape, np.nan, np.float32)
    firstcov = np.zeros(shape, np.float32)
    cov = np.zeros(shape, np.float32)
    ortho = np.full(shape, np.nan, np.float32)

    ann_by_tick = {k: g.announce_date.to_numpy() for k, g in announcements.groupby("tick")}
    hold_by_tick = {k: (g.available_date.to_numpy(), g.institution_hold.to_numpy(float)) for k, g in holdings.groupby("tick")}

    for i, date in enumerate(tqdm(trade_dates, desc="Building improved COV")):
        cutoff = date - pd.Timedelta(days=LOOKBACK)
        visible = reports[(reports.create_date >= cutoff) & (reports.create_date <= date)
                          & (reports.entrytime <= date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
        counts = visible.groupby("tick").report_id.nunique()
        for tick, n in counts.items():
            j = tickpos.get(tick)
            if j is not None:
                cov[i, j] = np.sqrt(n)

        # No announcement -> missing; announcement but no responsive report -> zero.
        announced_ticks = []
        for tick, dates in ann_by_tick.items():
            left = np.searchsorted(dates, cutoff.to_datetime64(), side="left")
            right = np.searchsorted(dates, date.to_datetime64(), side="right")
            if right > left:
                j = tickpos.get(tick)
                if j is not None:
                    anncov[i, j] = 0.0
                    announced_ticks.append(tick)
        responsive = visible[visible.post_announcement]
        for tick, n in responsive.groupby("tick").report_id.nunique().items():
            j = tickpos.get(tick)
            if j is not None and np.isfinite(anncov[i, j]):
                anncov[i, j] = np.sqrt(n)

        events = first_events[(first_events.create_date >= cutoff) & (first_events.create_date <= date)
                              & (first_events.entrytime <= date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
        for tick, n in events.groupby("tick").organ_id.nunique().items():
            j = tickpos.get(tick)
            if j is not None:
                firstcov[i, j] = np.sqrt(n)

        hold = np.full(len(ticks), np.nan)
        dt64 = date.to_datetime64()
        for tick, (dates, values) in hold_by_tick.items():
            p = np.searchsorted(dates, dt64, side="right") - 1
            j = tickpos.get(tick)
            if p >= 0 and j is not None:
                hold[j] = values[p]
        valid = np.isfinite(hold)
        if valid.sum() >= 30:
            x = np.column_stack([np.ones(valid.sum()), hold[valid]])
            ortho[i, valid] = cov[i, valid] - x @ np.linalg.lstsq(x, cov[i, valid], rcond=None)[0]
    return {"anncov": anncov, "firstcov": firstcov, "cov_ortho_inshold": ortho}


def evaluate(factors, data, positions, trade_dates):
    OUT.mkdir(parents=True, exist_ok=True)
    pct = data.read("d_essentials/pct", data.axis.date_count - 1, 0) / 100.0
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rows = []
    rng = np.random.default_rng(0)
    tie = rng.uniform(0, 1e-6, data.axis.tick_count)
    for name, pred in factors.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        for ax, horizon in zip(axes.flat, (1, 5, 10, 20)):
            windows = np.lib.stride_tricks.sliding_window_view(pct[2:], horizon, axis=0)
            fwd = np.prod(1.0 + windows, axis=-1) - 1.0
            label = np.full(pred.shape, np.nan)
            valid_rows = positions < len(fwd)
            label[valid_rows] = fwd[positions[valid_rows]]
            grouped = np.where(np.isfinite(pred), pred + tie[None, :], np.nan)
            ic, ric = IC(pred, label), rankIC(pred, label)
            group = calc_group_ret(grouped, label, 10)
            means = np.nanmean(group, axis=1)
            high_low = group[-1] - group[0]
            mono = spearmanr(np.arange(1, 11), means).statistic
            rows.append({
                "factor": name, "horizon": horizon,
                "coverage": np.nanmean(np.isfinite(pred).sum(axis=1)),
                "nonzero": np.nanmean((pred != 0).sum(axis=1)),
                "mean_ic": np.nanmean(ic),
                "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(252),
                "rank_ic": np.nanmean(ric),
                "rank_icir": np.nanmean(ric) / np.nanstd(ric) * np.sqrt(252),
                "monotonicity": mono,
                "high_minus_low_bps": np.nanmean(high_low) * 1e4,
                "high_minus_low_sharpe": np.nanmean(high_low) / np.nanstd(high_low) * np.sqrt(252),
                **{f"g{k+1}_bps": v * 1e4 for k, v in enumerate(means)},
            })
            for k, values in enumerate(np.nancumsum(group, axis=1)):
                ax.plot(trade_dates[:len(values)], values, color=colors[k], linewidth=1.1, label=f"G{k+1}")
            ax.set_title(f"{name} {horizon}D | IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(ric):.4f}")
            ax.axhline(0, color="black", linewidth=.8)
            ax.grid(alpha=.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.suptitle(f"{name} Decile Cumulative Excess Returns | No Industry/Size Neutralization")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_cumulative.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "improved_cov_summary.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False), flush=True)


def main():
    data = DataPool(DATA_ROOT, asset="stock")
    data.asset_root = DATA_ROOT
    dates = pd.DatetimeIndex(data.axis.trade_dates)
    positions = np.flatnonzero((dates >= START) & (dates <= END))
    trade_dates = dates[positions]
    reports = load_reports()
    announcements = load_announcements()
    reports = mark_post_announcement(reports.reset_index(drop=True), announcements)
    first_events = load_first_events(reports)
    holdings = load_holdings()
    factors = build_factors(trade_dates, data.axis.ticks, reports, first_events, announcements, holdings)
    evaluate(factors, data, positions, trade_dates)
    print(f"saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()

