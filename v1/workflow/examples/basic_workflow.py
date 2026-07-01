"""Synthetic end-to-end example; replace the data adapter with production PIT data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_workflow import (
    AlphaConfig,
    CompositeConfig,
    DailyProductionWorkflow,
    OptimizerConfig,
    PanelData,
    PreprocessConfig,
    RiskConfig,
    ResearchArtifact,
    TransformConfig,
)


# 中文说明：`make_demo_data` 是本示例的函数入口。
def make_demo_data(seed: int = 7) -> PanelData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=160)
    assets = pd.Index([f"S{i:03d}" for i in range(60)], name="asset")
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factors = pd.DataFrame(
        rng.normal(size=(len(index), 4)),
        index=index,
        columns=["value", "quality", "momentum", "reversal"],
    )
    industries = np.arange(len(assets)) % 6
    industry = np.eye(6)[industries]
    style = rng.normal(size=(len(assets), 2))
    current_exposure = np.column_stack([industry, style])
    exposure_values = np.tile(current_exposure, (len(dates), 1))
    exposures = pd.DataFrame(
        exposure_values,
        index=index,
        columns=[f"industry_{i}" for i in range(6)] + ["size", "beta"],
    )
    true_beta = np.array([0.0010, 0.0006, 0.0008, -0.0004])
    signal_return = factors.to_numpy() @ true_beta
    forward_returns = pd.Series(
        signal_return + rng.normal(scale=0.02, size=len(index)),
        index=index,
        name="forward_return",
    )
    market_caps = pd.Series(
        np.tile(np.exp(rng.normal(22, 1, len(assets))), len(dates)),
        index=index,
        name="market_cap",
    )
    tradable = pd.Series(True, index=index, name="tradable")
    return PanelData(factors, forward_returns, exposures, market_caps, tradable)


if __name__ == "__main__":
    panel = make_demo_data()
    artifact = ResearchArtifact(
        artifact_id="demo-production-v1",
        created_at=pd.Timestamp("2026-06-14", tz="UTC"),
        selected_factors=tuple(panel.factors.columns),
        preprocess=PreprocessConfig(neutralize=True, min_observations=20),
        transform=TransformConfig(),
        composite=CompositeConfig(lookback=40, min_periods=15),
        alpha=AlphaConfig(lookback=80, min_periods=30, ridge=5.0),
        risk=RiskConfig(factor_halflife=40, specific_halflife=40),
        optimizer=OptimizerConfig(max_weight=0.05, max_turnover=None),
    )
    workflow = DailyProductionWorkflow(artifact)
    result = workflow.run(panel, panel.dates[-1])
    print(result.portfolio.weights.sort_values(ascending=False).head(10))
    print(result.portfolio.constraint_usage)
