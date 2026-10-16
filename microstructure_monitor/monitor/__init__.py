"""Transparent provisional rules, with unavailable history marked explicitly."""
import pandas as pd


def detect_regime(baseline, thresholds):
    z = baseline.z20.dropna()
    warn = float(thresholds.get("z_warn", 2))
    anomalies = baseline.loc[baseline.z20.abs() >= warn].copy()
    anomalies["severity"] = anomalies.z20.abs().apply(lambda value: "alert" if value >= float(thresholds.get("z_alert", 3)) else "warn")
    def state(metric, low, normal, high, threshold=1.5):
        if pd.isna(baseline.raw.get(metric, float("nan"))):
            return "unavailable"
        value = z.get(metric)
        if value is None:
            return "insufficient_history"
        return high if value >= threshold else low if value <= -threshold else normal
    regime = {
        "liquidity_regime": state("liquidity.spread", "loose", "normal", "tight", float(thresholds.get("liquidity_tight_z", 1.5))),
        "trade_size_regime": state("trade.large_ratio", "fragmented", "normal", "concentrated"),
        "orderflow_regime": state("flow.imbalance", "sell", "neutral", "buy"),
        "impact_regime": state("liquidity.impact", "low", "normal", "high", float(thresholds.get("impact_high_z", 1.5))),
        "institution_regime": state("trade.large_ratio", "weak", "normal", "strong", float(thresholds.get("institution_strong_z", 1.5))),
        "institution_proxy_regime": state("trade.large_ratio", "weak", "normal", "strong", float(thresholds.get("institution_strong_z", 1.5))),
        "intraday_regime": state("intraday.amount_close_share", "opening_dominant", "balanced", "closing_dominant"),
        # A single-day signal cannot establish a persistent new regime.
        "overall": "INSUFFICIENT_HISTORY" if z.empty else "TRANSITION" if len(anomalies) >= 2 else "NORMAL",
    }
    def raw(metric):
        return baseline.raw.get(metric, float("nan"))
    flow = raw("flow.imbalance")
    direction_threshold = float(thresholds.get("flow_direction_threshold", .05))
    regime["orderflow_regime"] = ("unavailable" if pd.isna(flow) else "buy" if flow > direction_threshold else "sell" if flow < -direction_threshold else "neutral")
    opening, closing = raw("intraday.amount_open_mean"), raw("intraday.amount_close_mean")
    gap = (closing-opening)/(closing+opening) if closing+opening > 0 else float("nan")
    threshold = float(thresholds.get("intraday_dominance_threshold", .1))
    regime["intraday_regime"] = "unavailable" if pd.isna(gap) else "closing_dominant" if gap > threshold else "opening_dominant" if gap < -threshold else "balanced"
    return regime, anomalies
