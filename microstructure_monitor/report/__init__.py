"""Portable CSV outputs and a readable daily Markdown report."""
from pathlib import Path


def generate_daily_report(date, result, output_dir):
    directory = Path(output_dir) / str(date).replace("-", "")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("features", "baseline", "cross_section", "anomalies", "distribution", "universe_gaps", "migration", "structural_change"):
        getattr(result, name).to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
    for name, frame in result.bars.items():
        frame.to_csv(directory / f"bars_{name}.csv", index=name in ("daily", "period_summary"))
    lines = [f"# Microstructure monitor: {date}", "", "Market metrics are equal-weight security averages.", ""]
    lines.extend(f"- {key}: {value}" for key, value in result.regime.items())
    lines += ["", "## Features", "", "```text", result.features.round(4).to_string(), "```", "", "## Baselines", "", "```text", result.baseline.round(4).to_string(), "```", "", "Trend report: ../trends.html"]
    lines += ["", f"Securities: {len(result.features)}", f"Anomalies: {len(result.anomalies)}", "", "Large trade size is a proxy, not an identification of institutional activity.", "Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded."]
    path = directory / "daily_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
