"""Minimal daily production example using a published research artifact."""

from __future__ import annotations

import pandas as pd

from basic_workflow import make_demo_data
from quant_workflow import (
    AlphaConfig,
    CompositeConfig,
    DailyProductionWorkflow,
    OptimizerConfig,
    PreprocessConfig,
    ResearchArtifact,
    RiskConfig,
    TransformConfig,
)


# 中文说明：`main` 是本示例的函数入口。
def main() -> None:
    panel = make_demo_data(seed=42)
    artifact = ResearchArtifact(
        artifact_id="demo-approved-v1",
        created_at=pd.Timestamp("2026-06-14", tz="UTC"),
        selected_factors=tuple(panel.factors.columns),
        preprocess=PreprocessConfig(neutralize=True, min_observations=15),
        transform=TransformConfig(method="none"),
        composite=CompositeConfig(method="icir", lookback=30, min_periods=10),
        alpha=AlphaConfig(method="ridge", lookback=60, min_periods=20),
        risk=RiskConfig(factor_halflife=30, specific_halflife=30),
        optimizer=OptimizerConfig(max_weight=0.08, max_turnover=None),
    )
    result = DailyProductionWorkflow(artifact).run(panel, panel.dates[-1])
    print(result.portfolio.weights.sort_values(ascending=False).head())


if __name__ == "__main__":
    main()
