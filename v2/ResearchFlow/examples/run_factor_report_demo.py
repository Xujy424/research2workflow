"""Demo launcher for the FactorTest Streamlit report.

Run from the repository root:

    python v2/ResearchFlow/examples/run_factor_report_demo.py

The demo starts the report on http://localhost:8501 without opening a browser.
Stop it with Ctrl+C.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "v2" / "ResearchFlow" / "FactorTest" / "run_report.py"
    command = [
        sys.executable,
        str(script),
        "--port",
        "8501",
        "--host",
        "localhost",
        "--no-browser",
    ]
    print("Starting FactorTest report demo:")
    print(" ".join(command))
    print("Open http://localhost:8501 after Streamlit starts. Press Ctrl+C to stop.")
    return subprocess.call(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
