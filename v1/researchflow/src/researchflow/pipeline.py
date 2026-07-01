"""Research workflow separated from the repeatable daily production line."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping

import pandas as pd

from quant_shared.artifacts import ResearchArtifact
from quant_shared.config import (
    AlphaConfig,
    CompositeConfig,
    OptimizerConfig,
    PreprocessConfig,
    RiskConfig,
    TransformConfig,
    StrategyType,
)
from quant_shared.contracts import FactorResearchReport, PanelData
from quant_shared.preprocessing import CrossSectionalPreprocessor
from quant_shared.transforms import FactorTransformer, TransformResult

from .adapters import ResearchScorecard
from .clustering import ClusterResult, FactorClusterer
from .research import FactorResearchEngine, ResearchConfig
from .validation import (
    FactorRobustnessValidator,
    RobustnessReport,
    incremental_value,
)


# 中文说明：定义 `ResearchFlowConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ResearchFlowConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    strategy_optimizers: Mapping[StrategyType, OptimizerConfig] = field(
        default_factory=dict
    )
    cluster_threshold: float = 0.30
    accepted_conclusions: tuple[str, ...] = ("Production", "Shadow", "Research")
    run_robustness: bool = True


# 中文说明：定义 `ResearchFlowResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ResearchFlowResult:
    processed_factors: pd.DataFrame
    single_factor_report: FactorResearchReport
    scorecard: pd.DataFrame
    upgrade_advice: pd.DataFrame
    clusters: ClusterResult
    selected_factors: tuple[str, ...]
    incremental_ic: pd.DataFrame
    model_transform: TransformResult
    robustness: RobustnessReport | None
    artifact: ResearchArtifact
    stage_order: tuple[str, ...]


# 中文说明：定义 `FactorResearchWorkflow`，封装本模块对应的数据、配置与行为。
class FactorResearchWorkflow:
    """Validate factors and publish an immutable production specification.

    Orthogonalisation, residualisation, PCA and PLS are model construction
    options. They deliberately run after single-factor analysis and clustering.
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: ResearchFlowConfig | None = None) -> None:
        self.config = config or ResearchFlowConfig()
        self.preprocessor = CrossSectionalPreprocessor(self.config.preprocess)
        self.research_engine = FactorResearchEngine(self.config.research)
        self.scorecard = ResearchScorecard()
        self.clusterer = FactorClusterer()
        self.transformer = FactorTransformer(self.config.transform)

    # 中文说明：`run`：执行主流程并返回结构化结果。
    def run(
        self,
        data: PanelData,
        *,
        factor_families: Mapping[str, str] | None = None,
        regimes: pd.Series | None = None,
        created_at: str | pd.Timestamp | None = None,
    ) -> ResearchFlowResult:
        data.validate()
        processed = self.preprocessor.transform(
            data.factors, data.exposures, data.market_caps
        )
        research_data = PanelData(
            processed,
            data.forward_returns,
            data.exposures,
            data.market_caps,
            data.tradable,
            data.metadata,
        )

        # Analyzer-equivalent diagnostics run on interpretable factor columns.
        report = self.research_engine.analyze(research_data)
        scorecard = self.scorecard.score(report)
        advice = self._upgrade_advice(scorecard)
        quality = (
            scorecard.groupby("factor", sort=False)["factor_score"].first()   # 因子评价总分
        )

        # Clustering must see preprocessed factors, never orthogonal/residual ones.
        clusters = self.clusterer.cluster(
            report.factor_correlations,
            quality,
            threshold=self.config.cluster_threshold,
        )
        eligible = self._eligible_factors(scorecard)
        selected = tuple(
            factor for factor in clusters.representatives if factor in eligible
        )
        if not selected:
            selected = (str(quality.idxmax()),)

        selected_frame = processed.loc[:, list(selected)]
        incremental_ic = self._incremental_ic(
            selected_frame,
            data.forward_returns,
        )
        model_transform = self.transformer.transform(
            selected_frame, data.forward_returns
        )
        robustness = None
        if self.config.run_robustness:
            robustness = FactorRobustnessValidator(
                self.research_engine
            ).validate(research_data, regimes=regimes)

        timestamp = pd.Timestamp.now(tz="UTC") if created_at is None else pd.Timestamp(created_at)
        artifact = ResearchArtifact(
            artifact_id=self._artifact_id(timestamp, selected),
            created_at=timestamp,
            selected_factors=selected,
            preprocess=self.config.preprocess,
            transform=self.config.transform,
            composite=self.config.composite,
            alpha=self.config.alpha,
            risk=self.config.risk,
            optimizer=self.config.optimizer,
            effective_from=timestamp,
            strategy_optimizers=dict(self.config.strategy_optimizers),
            cluster_assignments={
                str(name): int(label)
                for name, label in clusters.assignments.items()
            },
            factor_families=dict(factor_families or {}),
            diagnostics={
                "source": "researchflow",
                "single_factor_input": "uniformly_preprocessed_factors",
                "clustering_input": "uniformly_preprocessed_factor_correlations",
                "transform_position": "after_clustering_and_representative_selection",
                "selected_factor_count": len(selected),
            },
        ).validate()
        return ResearchFlowResult(
            processed_factors=processed,
            single_factor_report=report,
            scorecard=scorecard,
            upgrade_advice=advice,
            clusters=clusters,
            selected_factors=selected,
            incremental_ic=incremental_ic,
            model_transform=model_transform,
            robustness=robustness,
            artifact=artifact,
            stage_order=(
                "validate",
                "preprocess",
                "single_factor_analyzer",
                "score",
                "upgrade_diagnostics",
                "economic_and_risk_diagnostics",
                "correlation_clustering",
                "representative_selection",
                "incremental_value",
                "robustness_validation",
                "optional_model_transform",
                "publish_research_artifact",
            ),
        )

    # 中文说明：`_incremental_ic`：计算每个入选因子相对其余因子组合的边际 RankIC。
    @staticmethod
    def _incremental_ic(
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        rows: dict[str, pd.Series] = {}
        for factor in factors.columns:
            peers = factors.drop(columns=factor)
            if peers.empty:
                rows[factor] = FactorResearchEngine().information_coefficients(
                    factors[[factor]], forward_returns
                )[factor]
                continue
            base = peers.mean(axis=1)
            rows[factor] = incremental_value(
                base,
                factors[factor],
                forward_returns,
            )
        return pd.DataFrame(rows).sort_index()

    # 中文说明：`_eligible_factors`：内部辅助步骤，不作为稳定公共接口。
    def _eligible_factors(self, scorecard: pd.DataFrame) -> set[str]:
        conclusions = scorecard.groupby("factor", sort=False)["conclusion"].first()
        return set(
            conclusions[conclusions.isin(self.config.accepted_conclusions)].index
        )

    # 中文说明：`_upgrade_advice`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _upgrade_advice(scorecard: pd.DataFrame) -> pd.DataFrame:
        failed = scorecard.loc[~scorecard["passed"]].copy()
        if failed.empty:
            return pd.DataFrame(
                columns=["factor", "metric", "actual", "recommendation"]
            )
        recommendations = {
            "ic_mean": "复核方向、标签时点和经济逻辑，避免仅靠变换制造表面 IC。",
            "icir": "检查子样本稳定性、参数平台和衰减速度。",
            "ic_positive_ratio": "检查市场状态依赖和失效区间。",
            "long_short_sharpe": "加入换手、冲击成本和容量后重新评估。",
            "coverage": "修复数据可得性与股票池覆盖，不用填充值掩盖缺失。",
        }
        failed["recommendation"] = failed["metric"].map(recommendations).fillna(
            "回到单因子诊断定位原因后再提交复核。"
        )
        return failed[["factor", "metric", "actual", "recommendation"]].reset_index(
            drop=True
        )

    # 中文说明：`_artifact_id`：内部辅助步骤，不作为稳定公共接口。
    def _artifact_id(
        self, timestamp: pd.Timestamp, selected: tuple[str, ...]
    ) -> str:
        payload = "|".join(
            [
                timestamp.isoformat(),
                *selected,
                repr(self.config.preprocess),
                repr(self.config.transform),
                repr(self.config.alpha),
            ]
        )
        return f"research-{sha256(payload.encode('utf-8')).hexdigest()[:12]}"
