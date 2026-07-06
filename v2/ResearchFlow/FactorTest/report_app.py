"""Streamlit entrypoint for the FactorTest report app.

This wrapper keeps package imports stable when Streamlit executes a file path.
Run it through run_report.py rather than invoking this file directly.
"""

from __future__ import annotations

from ResearchFlow.FactorTest.web import main


if __name__ == "__main__":
    main()
