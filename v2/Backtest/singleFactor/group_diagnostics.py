"""分组股票画像诊断脚本。

直接修改下方参数后运行：
    python -m v2.Backtest.singleFactor.group_diagnostics

代码板块划分：
    创业板：300、301 开头
    科创板：688 开头
    中小板历史代码：002、003 开头

收益类指标使用“每日组内截面均值，再对时间取均值”。
市值保留中位数；成交额同时输出均值和中位数。
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT

# =========================
# 可调整参数
# =========================
ROOT_PATH = DEFAULT_ROOT
FACTOR_NAME = "pafr"
START_DATE = "2025-01-01"
END_DATE = "2026-06-30"
BENCHMARK = "zz1000"
UNIVERSE = "benchmark"  # "benchmark" 或 "tradable"
GROUPS = 10
GROUP = 10
FORWARD_OFFSET = 2
FORWARD_HORIZON = 5
STANDARDIZE_WITHIN_UNIVERSE = True
WORST_DAYS = 10
SAMPLE_SIZE = 20
SAMPLE_DATE = None  # 例如 "2025-03-27"；None 表示自动选该组未来收益最差日
TOP_EXPOSURES = 10


class Config:
    root = ROOT_PATH
    factor = FACTOR_NAME
    start = START_DATE
    end = END_DATE
    benchmark = BENCHMARK
    universe = UNIVERSE
    groups = GROUPS
    group = GROUP
    offset = FORWARD_OFFSET
    horizon = FORWARD_HORIZON
    standardize = STANDARDIZE_WITHIN_UNIVERSE
    worst_days = WORST_DAYS
    sample_size = SAMPLE_SIZE
    sample_date = SAMPLE_DATE
    top_exposures = TOP_EXPOSURES


def _zscore(frame: np.ndarray) -> np.ndarray:
    valid = np.isfinite(frame)
    count = valid.sum(axis=1, keepdims=True)
    mean = np.divide(
        np.nansum(np.where(valid, frame, 0.0), axis=1, keepdims=True),
        count,
        out=np.full((frame.shape[0], 1), np.nan),
        where=count > 0,
    )
    centered = frame - mean
    std = np.sqrt(np.divide(
        np.nansum(np.where(valid, centered * centered, 0.0), axis=1, keepdims=True),
        count,
        out=np.full((frame.shape[0], 1), np.nan),
        where=count > 1,
    ))
    return np.divide(centered, std, out=np.full_like(frame, np.nan), where=std > 0)


def _forward_return(daily_return: np.ndarray, length: int, offset: int, horizon: int) -> np.ndarray:
    out = np.full((length, daily_return.shape[1]), np.nan)
    for i in range(length):
        window = daily_return[i + offset:i + offset + horizon]
        if len(window) == horizon:
            out[i] = np.prod(1.0 + window, axis=0) - 1.0
    return out


def _group_mask(score: np.ndarray, valid: np.ndarray, group: int, groups: int) -> np.ndarray:
    selected = np.zeros_like(valid, dtype=bool)
    for t in range(len(score)):
        ok = valid[t] & np.isfinite(score[t])
        if ok.sum() < groups:
            continue
        order = np.argsort(score[t, ok], kind="stable")
        positions = np.flatnonzero(ok)[order]
        edges = np.linspace(0, len(positions), groups + 1)
        lo = int(np.floor(edges[group - 1]))
        hi = int(np.floor(edges[group])) if group < groups else len(positions)
        selected[t, positions[lo:hi]] = True
    return selected


def _nanmean_row(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.nansum(np.where(valid, values, 0.0), axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def _nanmedian_row(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        row = values[t, mask[t]]
        row = row[np.isfinite(row)]
        if len(row):
            out[t] = np.nanmedian(row)
    return out


def _board_mask(ticks: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(ticks), dtype=bool)
    ticks = ticks.astype(str)
    for prefix in prefixes:
        result |= np.char.startswith(ticks, prefix)
    return result


def _category_exposure(codes: np.ndarray, selected: np.ndarray, universe: np.ndarray, top_n: int) -> pd.DataFrame:
    records = []
    finite_codes = np.unique(codes[np.isfinite(codes)])
    for code in finite_codes:
        member = codes == code
        selected_pct = np.divide(
            (selected & member).sum(axis=1),
            selected.sum(axis=1),
            out=np.full(len(selected), np.nan, dtype=float),
            where=selected.sum(axis=1) > 0,
        )
        universe_pct = np.divide(
            (universe & member).sum(axis=1),
            universe.sum(axis=1),
            out=np.full(len(universe), np.nan, dtype=float),
            where=universe.sum(axis=1) > 0,
        )
        records.append({
            "code": int(code),
            "selected_pct": np.nanmean(selected_pct),
            "universe_pct": np.nanmean(universe_pct),
            "active_pct": np.nanmean(selected_pct - universe_pct),
        })
    if not records:
        return pd.DataFrame(columns=["code", "selected_pct", "universe_pct", "active_pct"])
    frame = pd.DataFrame(records)
    return frame.reindex(frame.active_pct.abs().sort_values(ascending=False).index).head(top_n)


def load_inputs(config: Config):
    with DataPool(config.root, asset="stock") as data:
        dates = pd.DatetimeIndex(data.axis.trade_dates)
        selected = np.flatnonzero(
            (dates >= pd.Timestamp(config.start)) & (dates <= pd.Timestamp(config.end))
        )
        if selected.size == 0:
            raise ValueError("no trade dates found in the requested range")
        start, end = int(selected[0]), int(selected[-1])
        future_end = min(end + config.offset + config.horizon, data.axis.date_count - 1)
        n = data.axis.tick_count
        close_adj = np.asarray(data.read("d_essentials/close_adj", end, start - 1), float)[:, :n]
        values = {
            "dates": dates[start:end + 1],
            "ticks": np.asarray(data.axis.ticks[:n]).astype(str),
            "factor": np.asarray(data.read(f"factor_pool/{config.factor}", end, start), float)[:, :n],
            "tradable": np.asarray(data.read("basic/tradable", end, start), bool)[:, :n],
            "benchmark": np.asarray(data.read(f"index/weight/{config.benchmark}_weight", end, start), float)[:, :n],
            "amount": np.asarray(data.read("d_essentials/amount", end, start), float)[:, :n],
            "market_cap": np.asarray(data.read("d_essentials/circ_mv", end, start), float)[:, :n],
            "industry": np.asarray(data.read("industry/industry", end, start), float)[:, :n],
            "sector": np.asarray(data.read("industry/sector", end, start), float)[:, :n],
            "pct": np.asarray(data.read("d_essentials/pct", future_end, start), float)[:, :n] / 100.0,
            "open_adj": np.asarray(data.read("d_essentials/open_adj", end, start), float)[:, :n],
            "close_adj": close_adj[1:],
            "prev_close_adj": close_adj[:-1],
        }
    return values


def analyse(config: Config = Config()):
    if not 1 <= config.group <= config.groups:
        raise ValueError("require 1 <= GROUP <= GROUPS")
    if config.horizon < 1 or config.offset < 0:
        raise ValueError("FORWARD_HORIZON must be positive and FORWARD_OFFSET must be non-negative")

    data = load_inputs(config)
    universe = data["tradable"] & np.isfinite(data["factor"])
    if config.universe == "benchmark":
        universe &= np.isfinite(data["benchmark"]) & (data["benchmark"] > 0)
    elif config.universe != "tradable":
        raise ValueError("UNIVERSE must be 'benchmark' or 'tradable'")

    score = np.where(universe, data["factor"], np.nan)
    if config.standardize:
        score = _zscore(score)

    selected = _group_mask(score, universe, config.group, config.groups)
    forward = _forward_return(data["pct"], len(score), config.offset, config.horizon)
    overnight = data["open_adj"] / data["prev_close_adj"] - 1.0
    intraday = data["close_adj"] / data["open_adj"] - 1.0

    cyb = _board_mask(data["ticks"], ("300", "301"))
    kcb = _board_mask(data["ticks"], ("688",))
    sme = _board_mask(data["ticks"], ("002", "003"))

    rows = pd.DataFrame({
        "date": data["dates"],
        "n": selected.sum(axis=1),
        "group_fwd_mean": _nanmean_row(forward, selected),
        "universe_fwd_mean": _nanmean_row(forward, universe),
        "amount_mean": _nanmean_row(data["amount"], selected),
        "universe_amount_mean": _nanmean_row(data["amount"], universe),
        "amount_median": _nanmedian_row(data["amount"], selected),
        "universe_amount_median": _nanmedian_row(data["amount"], universe),
        "mv_median": _nanmedian_row(data["market_cap"], selected),
        "universe_mv_median": _nanmedian_row(data["market_cap"], universe),
        "overnight_mean": _nanmean_row(overnight, selected),
        "universe_overnight_mean": _nanmean_row(overnight, universe),
        "intraday_mean": _nanmean_row(intraday, selected),
        "universe_intraday_mean": _nanmean_row(intraday, universe),
        "bad_rate": _nanmean_row((forward < 0).astype(float), selected),
        "cyb_pct": selected[:, cyb].sum(axis=1) / np.maximum(selected.sum(axis=1), 1),
        "kcb_pct": selected[:, kcb].sum(axis=1) / np.maximum(selected.sum(axis=1), 1),
        "sme_pct": selected[:, sme].sum(axis=1) / np.maximum(selected.sum(axis=1), 1),
    })
    rows["relative_fwd_mean"] = rows["group_fwd_mean"] - rows["universe_fwd_mean"]

    group_returns = []
    for group in range(1, config.groups + 1):
        mask = _group_mask(score, universe, group, config.groups)
        group_returns.append(_nanmean_row(forward, mask))
    group_table = pd.Series(
        [np.nanmean(item) for item in group_returns],
        index=[f"G{i}" for i in range(1, config.groups + 1)],
        name=f"mean_forward_{config.horizon}d",
    )

    industry_table = _category_exposure(data["industry"], selected, universe, config.top_exposures)
    sector_table = _category_exposure(data["sector"], selected, universe, config.top_exposures)

    summary = rows.drop(columns="date").mean(numeric_only=True)
    print(f"factor: {config.factor}")
    print(f"period: {data['dates'][0].date()} ~ {data['dates'][-1].date()}")
    print(f"universe: {config.universe} ({config.benchmark}), group: G{config.group}/{config.groups}")
    print("\nsummary")
    print(summary.to_string(float_format=lambda value: f"{value:.6g}"))
    print("\nmean group forward returns")
    print(group_table.to_string(float_format=lambda value: f"{value:.6g}"))
    print("\nindustry exposure top active pct")
    print(industry_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nsector exposure top active pct")
    print(sector_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nworst dates by selected group forward return")
    print(rows.nsmallest(config.worst_days, "group_fwd_mean").to_string(index=False))

    if config.sample_date:
        sample_date = pd.Timestamp(config.sample_date)
        matches = np.flatnonzero(data["dates"] == sample_date)
        if not len(matches):
            raise ValueError(f"sample date {config.sample_date} is not in selected range")
        sample_pos = int(matches[0])
    else:
        sample_pos = int(rows["group_fwd_mean"].idxmin())
    positions = np.flatnonzero(selected[sample_pos])
    sample = pd.DataFrame({
        "tick": data["ticks"][positions],
        "score": score[sample_pos, positions],
        "fwd": forward[sample_pos, positions],
        "amount": data["amount"][sample_pos, positions],
        "market_cap": data["market_cap"][sample_pos, positions],
        "overnight": overnight[sample_pos, positions],
        "intraday": intraday[sample_pos, positions],
        "industry": data["industry"][sample_pos, positions],
        "sector": data["sector"][sample_pos, positions],
    }).sort_values("fwd").head(config.sample_size)
    print(f"\nsample constituents on {data['dates'][sample_pos].date()} sorted by forward return")
    print(sample.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    analyse(Config())
