"""Deterministic and historical portfolio stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quant_shared.contracts import RiskModelOutput



@dataclass(frozen=True)
class StressResult:
    scenario_pnl: pd.Series
    factor_pnl: pd.DataFrame
    stressed_volatility: pd.Series


class StressTester:
    '''
        1. 给定一组因子冲击，计算组合在每个压力情景下的预期损益；
        2. 给定特异风险放大倍数，计算压力情景下组合波动率会变成多少。
    '''
    def run(
        self,
        weights: pd.Series,
        risk: RiskModelOutput,
        factor_shocks: Mapping[str, Mapping[str, float]],     # 每个情景下各因子的冲击幅度
        specific_vol_multipliers: Mapping[str, float] | None = None, # 每个压力情景中特异波动率放大多少倍
    ) -> StressResult:
        aligned = weights.reindex(risk.assets).fillna(0.0)
        portfolio_exposure = risk.exposures.T @ aligned
        scenario_pnl: dict[str, float] = {}
        factor_rows: list[pd.Series] = []
        stressed_vol: dict[str, float] = {}
        for scenario, shocks in factor_shocks.items():
            shock = pd.Series(shocks, dtype=float).reindex(portfolio_exposure.index).fillna(0.0)
            contributions = portfolio_exposure * shock
            scenario_pnl[scenario] = float(contributions.sum())
            contributions.name = scenario
            factor_rows.append(contributions)
            multiplier = (
                specific_vol_multipliers.get(scenario, 1.0)
                if specific_vol_multipliers is not None
                else 1.0
            )
            covariance = risk.stock_covariance.to_numpy(float).copy()
            covariance += np.diag(
                risk.specific_variance.to_numpy(float) * (multiplier**2 - 1.0)
            )
            w = aligned.to_numpy(float)
            stressed_vol[scenario] = float(np.sqrt(max(w @ covariance @ w, 0.0)))
        return StressResult(
            pd.Series(scenario_pnl, name="scenario_pnl"),
            pd.DataFrame(factor_rows),
            pd.Series(stressed_vol, name="stressed_volatility"),
        )
