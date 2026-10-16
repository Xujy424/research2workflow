"""Public daily orchestration entry point."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .schema import MarketData, flatten_feature_frames
from .config import PACKAGE_ROOT, DEFAULT_OUTPUT_DIR, read_simple_yaml
from .data import load_market_data
from .preprocess import preprocess
from .bars import build_bars
from .features import compute_features
from .statistics import compare_with_history, compare_universes
from .monitor import detect_regime
from .report import generate_daily_report
from .statistics.distribution import summarize_distribution, universe_differences
from .history import load_history, save_observation, detect_migration
from .report.dashboard import generate_dashboard
from .data.universes import build_universes
from .statistics.migration import compare_previous
from .quality import ensure_history_identity, quality_features
from .statistics.security_history import security_history
from .features.minute_features import build_feature_minutes, dynamic_intraday_shape
from .features.context import compute_event_context


@dataclass
class MonitorResult:
    features: pd.DataFrame
    baseline: pd.DataFrame
    cross_section: pd.DataFrame
    anomalies: pd.DataFrame
    regime: dict
    bars: dict
    report_path: Path | None = None
    distribution: pd.DataFrame | None = None
    universe_gaps: pd.DataFrame | None = None
    migration: pd.DataFrame | None = None
    dashboard_path: Path | None = None
    structural_change: pd.DataFrame | None = None
    quality: pd.DataFrame | None = None
    cleaning_quality: pd.DataFrame | None = None
    auctions: dict | None = None
    clean_data: MarketData | None = None
    security_history: pd.DataFrame | None = None
    security_baseline: pd.DataFrame | None = None
    universe_memberships: dict | None = None
    data_source: str = "caller_supplied"
    depth_recovery_events: pd.DataFrame | None = None


def run_daily_monitor(date, market_data: MarketData | None = None, path_config=None,
                      history=None, universe_map=None, output_dir=None, thresholds=None, universe_id=None, data_source="caller_supplied", static_dashboard=False):
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    paths = path_config or read_simple_yaml(PACKAGE_ROOT / "config" / "paths.yaml")
    if market_data is None:
        market_data = load_market_data(date.replace("-", ""), paths)
    cleaned = preprocess(market_data, date)
    if cleaned.trades.empty:
        raise ValueError("No valid continuous-session trades for requested date")
    bars = build_bars(cleaned)
    context = compute_event_context(cleaned)
    frames = compute_features(cleaned, bars, context)
    bars["feature_1min"] = build_feature_minutes(cleaned, bars, context)
    frames["minute_shape"] = dynamic_intraday_shape(bars["feature_1min"])
    features = flatten_feature_frames(frames)
    quality = quality_features(cleaned)
    features = features.join(quality.add_prefix("quality."))
    output_dir = output_dir or paths.get("output_dir", DEFAULT_OUTPUT_DIR)
    ensure_history_identity(output_dir, sorted(cleaned.trades.symbol.unique()), universe_id)
    stored_history = load_history(output_dir)
    history = stored_history if history is None else history
    metrics = features.mean(numeric_only=True).to_dict()
    structural = compare_previous(features, bars["feature_1min"], Path(output_dir), date)
    for metric, row in structural.iterrows():
        if pd.notna(row.get("ks_distance")):
            metrics[f"structural.ks.{metric}"] = row.ks_distance
        if pd.notna(row.get("curve_l1")):
            metrics[f"structural.curve.{metric}"] = row.curve_l1
    # Pooled trade quantiles and totals have explicit namespaces, separate from stock means.
    amounts = cleaned.trades.amount
    for name, value in {"mean": amounts.mean(), "median": amounts.median(), "std": amounts.std(), "skew": amounts.skew(), "kurt": amounts.kurt(), "max": amounts.max()}.items():
        metrics["pooled_trade." + name] = value
    for p in (.5, .9, .95, .99):
        metrics[f"pooled_trade.p{int(p*100)}"] = amounts.quantile(p)
    metrics["pooled_trade.tail_ratio"] = metrics["pooled_trade.p99"] / metrics["pooled_trade.p50"]
    metrics["market.trade_count"] = len(amounts)
    metrics["market.trade_amount"] = amounts.sum()
    universe_map = {**build_universes(cleaned.daily), **(universe_map or {})}
    cross_section = compare_universes(features, universe_map, minute=bars["feature_1min"])
    for universe, row in cross_section.iterrows():
        for metric, value in row.items():
            metrics[f"universe.{universe}.{metric}"] = value
    gaps = universe_differences(cross_section)
    for group, row in gaps.iterrows():
        for metric, value in row.items():
            metrics[f"gap.{group}.{metric}"] = value
    distribution = summarize_distribution(features)
    for metric, row in distribution.iterrows():
        # Shapes already have their own history; avoid a second layer of meta-statistics.
        if metric.startswith("minute_shape."):
            continue
        for statistic, value in row.items():
            metrics[f"distribution.{metric}.{statistic}"] = value
    metrics = pd.Series(metrics, dtype=float)
    baseline = compare_with_history(metrics, history, date)
    settings = read_simple_yaml(PACKAGE_ROOT / "config" / "thresholds.yaml")
    settings.update(thresholds or {})
    regime, anomalies = detect_regime(baseline, settings)
    result = MonitorResult(features, baseline, cross_section, anomalies, regime, bars)
    result.distribution = distribution
    result.universe_gaps = gaps
    result.quality = quality
    result.cleaning_quality = cleaned.quality
    result.auctions = cleaned.auctions
    result.clean_data = cleaned
    result.data_source = data_source
    result.depth_recovery_events = context.recovery_events
    result.universe_memberships = {"all": features.index.tolist(), **{k: list(v) for k, v in universe_map.items()}}
    result.security_history, result.security_baseline = security_history(output_dir, features, date)
    accumulated = save_observation(output_dir, date, metrics)
    combined = pd.concat([history.loc[~history.index.isin(accumulated.index)], accumulated]).sort_index()
    result.migration = detect_migration(combined, settings)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.structural_change = structural
    result.regime["overall"] = result.migration.loc[date, "overall"]
    result.report_path = generate_daily_report(date, result, output_dir)
    if static_dashboard:
        generate_dashboard(combined, result.migration, output_dir, bars["1min"], features, structural, regime)
    from .explorer import generate_explorer
    result.dashboard_path = generate_explorer(result, combined, output_dir, date)
    return result
