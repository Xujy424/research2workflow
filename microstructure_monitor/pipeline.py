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


def run_daily_monitor(date, market_data: MarketData | None = None, path_config=None,
                      history=None, universe_map=None, output_dir=None, thresholds=None):
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    paths = path_config or read_simple_yaml(PACKAGE_ROOT / "config" / "paths.yaml")
    if market_data is None:
        market_data = load_market_data(date.replace("-", ""), paths)
    cleaned = preprocess(market_data, date)
    if cleaned.trades.empty:
        raise ValueError("No valid continuous-session trades for requested date")
    bars = build_bars(cleaned)
    features = flatten_feature_frames(compute_features(cleaned, bars))
    output_dir = output_dir or paths.get("output_dir", DEFAULT_OUTPUT_DIR)
    stored_history = load_history(output_dir)
    history = stored_history if history is None else history
    metrics = features.mean(numeric_only=True)
    baseline = compare_with_history(metrics, history, date)
    universe_map = {**build_universes(cleaned.daily), **(universe_map or {})}
    cross_section = compare_universes(features, universe_map)
    for universe, row in cross_section.iterrows():
        for metric, value in row.items():
            metrics[f"universe.{universe}.{metric}"] = value
    gaps = universe_differences(cross_section)
    for group, row in gaps.iterrows():
        for metric, value in row.items():
            metrics[f"gap.{group}.{metric}"] = value
    distribution = summarize_distribution(features)
    for metric, row in distribution.iterrows():
        for statistic, value in row.items():
            metrics[f"distribution.{metric}.{statistic}"] = value
    baseline = compare_with_history(metrics, history, date)
    settings = read_simple_yaml(PACKAGE_ROOT / "config" / "thresholds.yaml")
    settings.update(thresholds or {})
    regime, anomalies = detect_regime(baseline, settings)
    result = MonitorResult(features, baseline, cross_section, anomalies, regime, bars)
    result.distribution = distribution
    result.universe_gaps = gaps
    accumulated = save_observation(output_dir, date, metrics)
    combined = pd.concat([history.loc[~history.index.isin(accumulated.index)], accumulated]).sort_index()
    result.migration = detect_migration(combined, settings)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.structural_change = compare_previous(features, bars["1min"], Path(output_dir), date)
    result.regime["overall"] = result.migration.loc[date, "overall"]
    result.report_path = generate_daily_report(date, result, output_dir)
    result.dashboard_path = generate_dashboard(combined, result.migration, output_dir)
    return result
