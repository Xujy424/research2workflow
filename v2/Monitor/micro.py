"""微观结构监控指标。盘口快照与分钟行情先形成个股因子，再做域内市值加权。"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from .context import MarketContext
    from .utils import historical_percentile, safe_divide, weighted_mean
except ImportError:  # 直接以脚本路径运行
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.context import MarketContext
    from v2.Monitor.utils import historical_percentile, safe_divide, weighted_mean

PERIODS = (1, 5, 10, 20)
INDEX_DOMAINS = ("沪深300", "中证500", "中证1000", "中证2000")
STYLE_DOMAINS = tuple(f"barra{i}" for i in range(1, 11))
MICRO_METRICS = (
    "有效买卖价差", "五档盘口深度", "OFI流动性冲击", "买卖成交不平衡",
    "订单流自相关", "冲击衰减", "路径结构", "波动率冲击衰减",
    "异常价格跳跃频率", "ACD预期跳跃强度", "标准化跳跃强度", "跳跃-成交量偏离",
)


def _rolling_mean(x, window):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        block = x[i - window + 1:i + 1]
        if np.isfinite(block).any():
            out[i] = np.nanmean(block)
    return out


def _weighted_rows(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    return np.array([weighted_mean(v, w) for v, w in zip(values, weights)])


def _domain_members(context, start, end, domain):
    weights = context.domain_weights(start, end, domain)
    return np.isfinite(weights) & (weights > 0), weights


def _percentile_rows(daily, membership, weights, history, period):
    """先日频形成域内因子，再平滑并取历史分位。"""
    series = _weighted_rows(daily, np.where(membership, weights, np.nan))
    return historical_percentile(_rolling_mean(series, period)[-history:])


def _as_time_string(values):
    return np.array([str(v) for v in values], dtype=object)


class MicroContext(MarketContext):
    """微观结构取数上下文，读取 L2 生成的日内盘口快照 parquet。"""

    def __init__(self, *args, snapshot_root=None, snapshot_pattern="proc/{date}/{exchange}_shot_1m.pq",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_root = Path(snapshot_root or (Path(self.data.asset_root) / "l2"))
        self.snapshot_pattern = snapshot_pattern
        self._snapshot_cache = {}

    def snapshots(self, date, exchanges=("sh", "sz")):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        if day in self._snapshot_cache:
            return self._snapshot_cache[day]
        frames = []
        for exchange in exchanges:
            rel = self.snapshot_pattern.format(date=day, exchange=exchange)
            path = self.snapshot_root / rel
            if not path.exists():
                continue
            try:
                import polars as pl
                frame = pl.read_parquet(path).to_pandas()
            except Exception as exc:
                warnings.warn(f"无法读取盘口快照 {path}: {exc}", RuntimeWarning)
                continue
            if not frame.empty:
                frame["SecurityID"] = frame["SecurityID"].astype(str).str.zfill(6)
                frames.append(frame)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._snapshot_cache[day] = result
        return result

    def close(self):
        self._snapshot_cache.clear()
        super().close()


def _snapshot_arrays(frame, ticks, topn=5):
    """Align one daily snapshot to context tick order; return per-minute fields."""
    n = len(ticks)
    if frame.empty:
        return None
    frame = frame.copy()
    frame["_tick"] = frame["SecurityID"].astype(str).str.zfill(6).str[:6]
    frame["_time"] = _as_time_string(frame["BarTime"])
    frame = frame.sort_values(["_tick", "_time"]).drop_duplicates(["_tick", "_time"], keep="last")
    times = np.array(sorted(frame["_time"].unique()), dtype=object)
    # [minutes, securities]; absent quote levels remain NaN.
    shape = (len(times), n)
    target = pd.MultiIndex.from_product([times, [str(x).zfill(6)[:6] for x in ticks]])
    indexed = frame.set_index(["_time", "_tick"]).reindex(target)
    def mat(col):
        if col not in indexed:
            return np.full(shape, np.nan)
        return pd.to_numeric(indexed[col], errors="coerce").to_numpy(dtype=float).reshape(shape)
    bp, bq, ap, aq = [], [], [], []
    for level in range(1, topn + 1):
        bp.append(mat(f"BidPrice{level}")); bq.append(mat(f"BidQty{level}"))
        ap.append(mat(f"AskPrice{level}")); aq.append(mat(f"AskQty{level}"))
    bp, bq, ap, aq = map(np.asarray, (bp, bq, ap, aq))
    bid1, ask1 = bp[0], ap[0]
    mid = (bid1 + ask1) / 2
    spread = safe_divide(ask1 - bid1, mid) * 10000
    bid_depth = np.nansum(np.where(np.isfinite(bq), bq, 0.0), axis=0)
    ask_depth = np.nansum(np.where(np.isfinite(aq), aq, 0.0), axis=0)
    bid_depth[(~np.isfinite(bq)).all(axis=0)] = np.nan
    ask_depth[(~np.isfinite(aq)).all(axis=0)] = np.nan
    depth = bid_depth + ask_depth
    # Top-five depth imbalance; also the observable quote-change component
    # used as an OFI proxy, since the saved snapshots are minute endpoints.
    imbalance = safe_divide(bid_depth - ask_depth, bid_depth + ask_depth)
    ofi = np.full_like(imbalance, np.nan)
    if len(times) > 1:
        ofi[1:] = np.diff(bid_depth, axis=0) - np.diff(ask_depth, axis=0)
    return {"times": times, "mid": mid, "spread": spread, "depth": depth,
            "imbalance": imbalance, "ofi": ofi, "bid_depth": bid_depth,
            "ask_depth": ask_depth}


def _daily_snapshot_metric(snap, name):
    if snap is None:
        return None
    values = snap[name]
    if name in {"spread", "depth"}:
        return np.nanmedian(values, axis=0)
    if name == "imbalance":
        return np.nanmean(values, axis=0)
    if name == "ofi":
        gross = np.nansum(np.abs(values), axis=0)
        net = np.nansum(values, axis=0)
        return safe_divide(net, gross)
    return np.nanmean(values, axis=0)


def _daily_ohlcv(context, index):
    date = context.dates[index]
    def read(field):
        return np.asarray(context.row(field, index), float)
    close = read("m_essentials/close")
    high = read("m_essentials/high")
    low = read("m_essentials/low")
    amount = read("m_essentials/amount")
    volume = read("m_essentials/volume")
    return close, high, low, amount, volume


def _historical_daily(context, start, end, snapshot_cache):
    n = len(context.ticks)
    out = {name: np.full((end - start + 1, n), np.nan) for name in (
        "spread", "depth", "ofi", "imbalance", "order_auto", "impact_decay",
        "path", "vol_decay", "jump_freq", "acd", "acd_z", "jump_volume")}
    jump_history = []
    for t, idx in enumerate(range(start, end + 1)):
        snap = snapshot_cache.get(context.dates[idx].strftime("%Y%m%d"))
        if snap is None:
            snap = _snapshot_arrays(context.snapshots(context.dates[idx]), context.ticks)
            snapshot_cache[context.dates[idx].strftime("%Y%m%d")] = snap
        for field in ("spread", "depth", "ofi", "imbalance"):
            if snap is not None:
                out[field][t] = _daily_snapshot_metric(snap, field)
        close, high, low, amount, volume = _daily_ohlcv(context, idx)
        if snap is not None:
            mid = snap["mid"]
            minute_ret = safe_divide(np.diff(mid, axis=0), mid[:-1])
            quote_flow = snap["ofi"]
            # Minute quote changes only: autocorrelation is a proxy for signed order-flow persistence.
            for j in range(n):
                x = quote_flow[:, j]
                valid = np.isfinite(x[1:]) & np.isfinite(x[:-1])
                if valid.sum() >= 5:
                    a, b = x[1:][valid], x[:-1][valid]
                    if np.std(a) > 0 and np.std(b) > 0:
                        out["order_auto"][t, j] = np.corrcoef(a, b)[0, 1]
                if len(minute_ret) > 2:
                    # Future/initial return ratio by horizon; average signed retention is daily proxy.
                    r1 = minute_ret[:, j]
                    valid_r = np.isfinite(r1)
                    if valid_r.sum() > 3:
                        first = np.nanmean(r1[:max(1, len(r1)//5)])
                        later = np.nanmean(r1[max(1, len(r1)//5):])
                        out["impact_decay"][t, j] = later / first if abs(first) > 1e-10 else np.nan
                    # Realized volatility persistence: next-window / first-window variance ratio.
                    chunks = np.array_split(r1, 4)
                    vv = [np.nanvar(chunk) if np.isfinite(chunk).sum() >= 3 else np.nan for chunk in chunks]
                    out["vol_decay"][t, j] = safe_divide(vv[-1], vv[0])
                    # Return-path structure / efficiency ratio: endpoint move over total variation.
                    out["path"][t, j] = safe_divide(np.nansum(r1), np.nansum(np.abs(r1)))
                # Jump events are robust 3-sigma minute returns, ACD fitted to event waiting time.
                finite = minute_ret[:, j][np.isfinite(minute_ret[:, j])]
                if len(finite) >= 30:
                    sigma = np.nanstd(finite)
                    events = np.flatnonzero(np.abs(minute_ret[:, j]) > 3 * sigma) if sigma > 0 else np.array([], int)
                    out["jump_freq"][t, j] = len(events) / 240.0
                    if len(events) >= 2:
                        gaps = np.diff(events).astype(float)
                        wait = np.full(1, np.nanmean(gaps))
                        # Stable ACD(1,1) proxy; forecast intensity is inverse expected wait.
                        w = np.r_[wait, gaps]
                        omega = max(float(np.nanmean(gaps)) * 0.05, 1e-3)
                        alpha, beta = 0.15, 0.75
                        psi = np.full(len(w), np.nan)
                        psi[0] = max(float(np.nanmean(gaps)), 1.0)
                        for k in range(1, len(w)):
                            psi[k] = omega + alpha * w[k - 1] + beta * psi[k - 1]
                        out["acd"][t, j] = 1.0 / psi[-1]
                    if len(jump_history) >= 20:
                        hist = np.asarray(jump_history[-20:], float)[:, j]
                        if np.isfinite(hist).sum() >= 10:
                            sd = np.nanstd(hist)
                            out["acd_z"][t, j] = ((out["jump_freq"][t, j] - np.nanmean(hist)) / sd
                                                  if sd > 0 else np.nan)
        jump_history.append(out["jump_freq"][t].copy())
        # Jump intensity relative to turnover proxy; high intensity without amount confirmation is divergence.
        daily_volume = np.nansum(volume, axis=0) if volume.ndim > 1 else volume
        out["jump_volume"][t] = safe_divide(out["jump_freq"][t], daily_volume)
    return out


class MicroMonitor:
    """微观结构因子计算与展示，返回指标×域×周期的历史分位表。"""

    def __init__(self, context):
        self.context = context

    def snapshot(self, asof, domains=("全市场",), history=252, periods=PERIODS,
                 metrics=MICRO_METRICS):
        c = self.context
        end = c.position(asof)
        lookback = max(history, 252) + max(periods) + 5
        start = max(0, end - lookback)
        daily = _historical_daily(c, start, end, {})
        metric_map = {
            "有效买卖价差": ("spread", -1), "五档盘口深度": ("depth", 1),
            "OFI流动性冲击": ("ofi", 1), "买卖成交不平衡": ("imbalance", 1),
            "订单流自相关": ("order_auto", 1), "冲击衰减": ("impact_decay", 1),
            "路径结构": ("path", 1), "波动率冲击衰减": ("vol_decay", -1),
            "异常价格跳跃频率": ("jump_freq", 1), "ACD预期跳跃强度": ("acd", 1),
            "标准化跳跃强度": ("acd_z", 1), "跳跃-成交量偏离": ("jump_volume", 1),
        }
        all_domains = ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(c.industry_map.keys())
        requested = None if tuple(domains) == ("全市场",) else set(domains)
        rows = []
        for metric in metrics:
            field, direction = metric_map[metric]
            candidates = all_domains
            if requested is not None:
                candidates = tuple(d for d in candidates if d in requested)
            for domain in candidates:
                membership, weights = _domain_members(c, start, end, domain)
                values = daily[field]
                if direction < 0:
                    values = -values
                for period in periods:
                    value = _percentile_rows(values, membership, weights, history, period)
                    rows.append({"指标": metric, "域": domain, "周期": period, "历史分位": value})
        raw = pd.DataFrame(rows)
        if raw.empty:
            return raw
        table = raw.pivot(index=["指标", "域"], columns="周期", values="历史分位")
        table.columns = [f"{int(x)}日历史分位" for x in table.columns]
        return table

    @staticmethod
    def table(result, decimals=4):
        return result.round(decimals)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    context = MicroContext()
    try:
        result = MicroMonitor(context).snapshot("2026-09-30")
        print(result.to_string())
    finally:
        context.close()
