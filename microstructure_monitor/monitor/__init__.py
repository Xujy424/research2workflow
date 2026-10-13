"""Transparent provisional rules, with unavailable history marked explicitly."""
import pandas as pd


def detect_regime(baseline, thresholds):
    z = baseline.z20.dropna()
    warn = float(thresholds.get("z_warn", 2))
    anomalies = baseline.loc[baseline.z20.abs() >= warn].copy()
    anomalies["severity"] = anomalies.z20.abs().apply(lambda value: "alert" if value >= float(thresholds.get("z_alert", 3)) else "warn")
    def state(metric, low, normal, high, threshold=1.5):
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
    return regime, anomalies
