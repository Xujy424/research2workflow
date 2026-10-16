"""Portable CSV outputs and a readable daily Markdown report."""
from pathlib import Path


def generate_daily_report(date, result, output_dir):
    directory = Path(output_dir) / str(date).replace("-", "")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("features", "baseline", "cross_section", "anomalies", "distribution", "universe_gaps", "migration", "structural_change", "quality", "cleaning_quality", "depth_recovery_events"):
        getattr(result, name).to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
    for name, frame in (result.auctions or {}).items():
        frame.to_csv(directory / f"auction_{name}.csv", index=False)
    if result.clean_data is not None:
        for name in ("orders", "trades", "quotes"):
            getattr(result.clean_data, name).to_csv(directory / f"clean_{name}.csv", index=False)
    for name, frame in result.bars.items():
        frame.to_csv(directory / f"bars_{name}.csv", index=name in ("daily", "period_summary"))
    lines = [f"# Microstructure monitor: {date}", "", "Market metrics are equal-weight security averages.", ""]
    lines.extend(f"- {key}: {value}" for key, value in result.regime.items())
    core = [c for c in result.features if c.startswith(("trade.", "lifecycle.", "quality."))]
    lines += ["", "## Core Features", "", "```text", result.features[core].T.round(4).to_string(), "```", "", "## Core Baselines", "", "```text", result.baseline.loc[result.baseline.index.str.startswith(("trade.", "flow.", "liquidity.", "pooled_trade."))].round(4).to_string(), "```", "", "Full tables: features.csv, baseline.csv, quality.csv.", "Interactive report: [monitor.html](../monitor.html)"]
    lines += ["", f"Securities: {len(result.features)}", f"Anomalies: {len(result.anomalies)}", "", "Large trade size is a proxy, not an identification of institutional activity.", "Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded."]
    path = directory / "daily_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
