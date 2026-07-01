"""Versioned hand-off contracts between factor research and daily production."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from .config import (
    AlphaConfig,
    CompositeConfig,
    OptimizerConfig,
    PreprocessConfig,
    RiskConfig,
    TransformConfig,
    StrategyType,
)


# 中文说明：定义 `ResearchArtifact`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ResearchArtifact:
    """Immutable model specification approved by the research workflow.

    Clustering assignments and selected factors are research decisions. Daily
    production consumes them and must not re-run factor admission or clustering.
    """

    artifact_id: str
    created_at: pd.Timestamp
    selected_factors: tuple[str, ...]
    preprocess: PreprocessConfig
    transform: TransformConfig
    composite: CompositeConfig
    alpha: AlphaConfig
    risk: RiskConfig
    optimizer: OptimizerConfig
    effective_from: pd.Timestamp | None = None
    strategy_optimizers: Mapping[StrategyType, OptimizerConfig] = field(default_factory=dict)
    cluster_assignments: Mapping[str, int] = field(default_factory=dict)
    factor_families: Mapping[str, str] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(self) -> "ResearchArtifact":
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if not self.selected_factors:
            raise ValueError("selected_factors must not be empty")
        if len(set(self.selected_factors)) != len(self.selected_factors):
            raise ValueError("selected_factors must be unique")
        if self.effective_from is not None:
            pd.Timestamp(self.effective_from)
        for strategy, config in self.strategy_optimizers.items():
            if config.strategy != strategy:
                raise ValueError(
                    f"strategy optimizer key {strategy.value} does not match "
                    f"config strategy {config.strategy.value}"
                )
        return self
