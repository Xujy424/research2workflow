"""Launch the FactorTest Streamlit report.

Examples
--------
python v2/ResearchFlow/FactorTest/run_report.py
python v2/ResearchFlow/FactorTest/run_report.py --port 8502 --host 0.0.0.0 --no-browser
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the FactorTest factor report web app.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")
    parser.add_argument("--host", default="localhost", help="Streamlit server address.")
    parser.add_argument("--browser", dest="browser", action="store_true", help="Open browser automatically.")
    parser.add_argument("--no-browser", dest="browser", action="store_false", help="Do not open browser automatically.")
    parser.set_defaults(browser=True)
    parser.add_argument("streamlit_args", nargs=argparse.REMAINDER, help="Extra arguments passed after '--' to streamlit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = Path(__file__).with_name("report_app.py").resolve()
    v2_root = Path(__file__).resolve().parents[2]

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(v2_root) if not old_pythonpath else f"{v2_root}{os.pathsep}{old_pythonpath}"

    passthrough = list(args.streamlit_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "false" if args.browser else "true",
        *passthrough,
    ]
    try:
        return subprocess.call(command, env=env)
    except ModuleNotFoundError:
        print("streamlit is not installed in the current Python environment.", file=sys.stderr)
        print("Install it or switch to the environment used by FactorTest, then rerun this script.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
