"""Xingye (2020), seasonal random walk surprise with eight-quarter drift.

This financial-statement implementation uses the existing SUE data adapter.
It does not yet include guidance/express reports or the paper's monthly
multi-report-period selection, and must not be labelled an exact replication.
"""
from dataclasses import replace

from .sue import SUEContext, SUEConfig, _SeasonalSurpriseFactor
from ..alphabase import AlphaMeta


class TSSUEContext(SUEContext):
    def __init__(self, *args, config=SUEConfig(), **kwargs):
        # Eight *preceding* year-on-year changes require 13 consecutive
        # quarterly profits, including the observation being tested.
        config = replace(config, lookback_years=5, history_quarters=8,
                         min_quarter_observations=13,
                         min_change_observations=8)
        super().__init__(*args, config=config, **kwargs)


class TSSUEFactor(_SeasonalSurpriseFactor):
    meta = AlphaMeta("tssue", "Eight-quarter drift standardized earnings surprise")
    column = "sue0"


if __name__ == "__main__":
    from ..backtest import run_from_ide
    run_from_ide("tssue", start_date="2021-01-01", end_date="2026-09-30")
