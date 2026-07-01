"""Deprecated mixed research/production orchestration.

This is the main application service of the package.  It joins data validation,
factor preprocessing, single-factor research, factor combination, alpha
calibration, risk estimation and constrained portfolio optimisation.  Optional
research tools such as robustness tests and sleeve allocation remain separate
so that the production path stays explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from quant_shared.alpha import (
    FamaMacBethAlpha,
    DynamicLinearAlpha,
    MonotonicScoreCalibrator,
    WalkForwardRidgeAlpha,
    WalkForwardSklearnAlpha,
)
from quant_shared.combination import FactorCombiner
from quant_shared.config import (
    AlphaConfig,
    CompositeConfig,
    OptimizerConfig,
    PreprocessConfig,
    RiskConfig,
    StrategyType,
    TransformConfig,
)
from quant_shared.contracts import (
    FactorResearchReport,
    OptimizationResult,
    PanelData,
    RiskModelOutput,
)
from quant_workflow.costs import TransactionCostModel
from quant_workflow.portfolio import PortfolioOptimizer
from quant_shared.preprocessing import CrossSectionalPreprocessor
from quant_workflow.risk import EquityFactorRiskModel, risk_attribution
from quant_shared.transforms import FactorTransformer

from .research import FactorResearchEngine, ResearchConfig


# 中文说明：定义 `WorkflowResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class WorkflowResult:
    """All auditable artefacts produced for one rebalance and one strategy."""

    as_of: pd.Timestamp
    processed_factors: pd.DataFrame
    transformed_factors: pd.DataFrame
    transform_loadings: pd.DataFrame
    research: FactorResearchReport
    composite_score: pd.Series
    factor_weights: pd.DataFrame
    expected_returns: pd.Series
    alpha_coefficients: pd.DataFrame | pd.Series
    risk: RiskModelOutput
    portfolio: OptimizationResult
    risk_attribution: pd.DataFrame


# 中文说明：定义 `MultiStrategyWorkflowResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class MultiStrategyWorkflowResult:
    """Shared research/risk result plus independently optimised strategy books.

    ``base`` contains the full factor and risk-model audit trail. ``portfolios``
    contains one :class:`OptimizationResult` per requested strategy.  Models are
    estimated once and reused, ensuring that weight differences come only from
    portfolio mandates and constraints.
    """

    base: WorkflowResult
    portfolios: Mapping[StrategyType, OptimizationResult]


# 中文说明：定义 `FactorToPortfolioWorkflow`，封装本模块对应的数据、配置与行为。
class FactorToPortfolioWorkflow:
    """Composable production entry point.

    Input labels are expected to be point-in-time aligned. At rebalance date
    ``as_of``, all model estimation uses dates strictly before ``as_of``.
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        preprocess: PreprocessConfig | None = None,
        transform: TransformConfig | None = None,
        composite: CompositeConfig | None = None,
        alpha: AlphaConfig | None = None,
        risk: RiskConfig | None = None,
        optimizer: OptimizerConfig | None = None,
        research: ResearchConfig | None = None,
        costs: TransactionCostModel | None = None,
    ) -> None:
        self.preprocessor = CrossSectionalPreprocessor(preprocess)   # 截面标准化+中性化
        self.transformer = FactorTransformer(transform)              # 正交
        self.research_engine = FactorResearchEngine(research)        # 因子评分
        self.combiner = FactorCombiner(composite)                    # 滚动icir加权
        self.alpha_config = alpha or AlphaConfig()
        self.risk_model = EquityFactorRiskModel(risk)
        self.optimizer = PortfolioOptimizer(optimizer, costs)

    # 中文说明：`run`：执行主流程并返回结构化结果。
    def run(
        self,
        data: PanelData,
        as_of: str | pd.Timestamp,
        current_weights: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
        adv_fraction: pd.Series | None = None,
        optimizer: PortfolioOptimizer | None = None,
    ) -> WorkflowResult:
        """Run one complete rebalance for the configured strategy mandate.

        ``optimizer`` is normally omitted.  It exists for orchestration layers
        that need to reuse the same factor workflow with several independent
        portfolio mandates without mutating shared state.
        """

        data.validate()
        rebalance_date = pd.Timestamp(as_of)
        selected_optimizer = optimizer or self.optimizer
        if (
            selected_optimizer.config.strategy == StrategyType.INDEX_ENHANCED
            and benchmark_weights is None
        ):
            raise ValueError("benchmark_weights are required for index enhancement")
        available = data.factors.index.get_level_values(0) <= rebalance_date
        factors = data.factors.loc[available]
        labels = data.forward_returns.loc[available]
        exposures = data.exposures.loc[available] if data.exposures is not None else None
        caps = data.market_caps.loc[available] if data.market_caps is not None else None
        tradable = data.tradable.loc[available] if data.tradable is not None else None
        processed = self.preprocessor.transform(factors, exposures, caps)
        # Single-factor diagnostics preserve factor meaning. Model transforms
        # belong to the combination layer and therefore run only afterwards.
        research_data = PanelData(
            processed,
            labels.reindex(processed.index),
            exposures.reindex(processed.index) if exposures is not None else None,
            caps.reindex(processed.index) if caps is not None else None,
            tradable.reindex(processed.index) if tradable is not None else None,
            data.metadata,
        )
        report = self.research_engine.analyze(research_data)
        transform_result = self.transformer.transform(processed, labels)
        transformed = transform_result.values
        transformed_labels = labels.reindex(transformed.index)
        model_ic = self.research_engine.information_coefficients(
            transformed, transformed_labels
        )
        composite_score, factor_weights = self.combiner.combine(transformed, model_ic)
        expected_returns, alpha_coefficients = self._calibrate_alpha(
            transformed, composite_score, transformed_labels
        )
        alpha_now = self._cross_section(expected_returns, rebalance_date)
        if exposures is None:
            raise ValueError("risk exposures are required for portfolio optimisation")
        risk_output = self._fit_risk(
            labels, exposures, caps, rebalance_date, alpha_now.index
        )
        tradable_now = (
            self._cross_section(tradable, rebalance_date) if tradable is not None else None
        )
        portfolio = selected_optimizer.optimize(
            alpha=alpha_now,
            risk=risk_output,
            current_weights=current_weights,
            benchmark_weights=benchmark_weights,
            adv_fraction=adv_fraction,
            tradable=tradable_now,
        )
        return WorkflowResult(
            as_of=rebalance_date,
            processed_factors=processed,
            transformed_factors=transformed,
            transform_loadings=transform_result.loadings,
            research=report,
            composite_score=composite_score,
            factor_weights=factor_weights,
            expected_returns=expected_returns,
            alpha_coefficients=alpha_coefficients,
            risk=risk_output,
            portfolio=portfolio,
            risk_attribution=risk_attribution(portfolio.weights, risk_output),
        )

    # 中文说明：`run_strategies`：执行主流程并返回结构化结果。
    def run_strategies(
        self,
        data: PanelData,
        as_of: str | pd.Timestamp,
        strategy_configs: Mapping[StrategyType, OptimizerConfig],
        current_weights: Mapping[StrategyType, pd.Series] | None = None,
        benchmark_weights: pd.Series | None = None,
        adv_fraction: pd.Series | None = None,
    ) -> MultiStrategyWorkflowResult:
        """Generate several mandate-specific portfolios from one model snapshot.

        The expensive research, alpha and risk calculations run only once.
        Long-only, index-enhanced and market-neutral portfolios are then solved
        independently from the same expected returns and covariance matrix.

        Parameters
        ----------
        strategy_configs:
            Mapping from strategy type to its optimiser configuration.  The
            configuration's ``strategy`` field must match the mapping key.
        current_weights:
            Optional current holdings for each strategy book.
        benchmark_weights:
            Required when an index-enhanced strategy is requested.
        """

        if not strategy_configs:
            raise ValueError("strategy_configs must contain at least one strategy")
        for strategy, config in strategy_configs.items():
            if config.strategy != strategy:
                raise ValueError(
                    f"strategy key {strategy.value} does not match "
                    f"OptimizerConfig.strategy={config.strategy.value}"
                )
        if (
            StrategyType.INDEX_ENHANCED in strategy_configs
            and benchmark_weights is None
        ):
            raise ValueError("benchmark_weights are required for index enhancement")

        holdings = current_weights or {}
        first_strategy = next(iter(strategy_configs))
        first_optimizer = PortfolioOptimizer(
            strategy_configs[first_strategy],
            self.optimizer.cost_model,
        )
        base = self.run(
            data=data,
            as_of=as_of,
            current_weights=holdings.get(first_strategy),
            benchmark_weights=(
                benchmark_weights
                if first_strategy == StrategyType.INDEX_ENHANCED
                else None
            ),
            adv_fraction=adv_fraction,
            optimizer=first_optimizer,
        )

        portfolios: dict[StrategyType, OptimizationResult] = {
            first_strategy: base.portfolio
        }
        for strategy, config in strategy_configs.items():
            if strategy == first_strategy:
                continue
            optimizer = PortfolioOptimizer(config, self.optimizer.cost_model)
            portfolios[strategy] = optimizer.optimize(
                alpha=self._cross_section(base.expected_returns, base.as_of),
                risk=base.risk,
                current_weights=holdings.get(strategy),
                benchmark_weights=(
                    benchmark_weights
                    if strategy == StrategyType.INDEX_ENHANCED
                    else None
                ),
                adv_fraction=adv_fraction,
                tradable=(
                    self._cross_section(data.tradable, base.as_of)
                    if data.tradable is not None
                    else None
                ),
            )
        return MultiStrategyWorkflowResult(base=base, portfolios=portfolios)

    # 中文说明：`_calibrate_alpha`：内部辅助步骤，不作为稳定公共接口。
    def _calibrate_alpha(
        self,
        factors: pd.DataFrame,
        composite_score: pd.Series,
        labels: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series]:
        if self.alpha_config.method == "ridge":
            return WalkForwardRidgeAlpha(self.alpha_config).fit_predict(factors, labels)
        if self.alpha_config.method == "score_slope":
            return MonotonicScoreCalibrator(self.alpha_config).fit_predict(
                composite_score, labels
            )
        if self.alpha_config.method == "fama_macbeth":
            return FamaMacBethAlpha(self.alpha_config).fit_predict(factors, labels)
        if self.alpha_config.method == "dynamic_linear":
            return DynamicLinearAlpha(self.alpha_config).fit_predict(factors, labels)
        if self.alpha_config.method in {
            "elastic_net",
            "lasso",
            "bayesian_ridge",
            "pls",
            "random_forest",
            "gbdt",
            "hist_gbdt",
            "rank_gbdt",
            "mlp",
        }:
            return WalkForwardSklearnAlpha(self.alpha_config).fit_predict(
                factors, labels
            )
        raise ValueError(f"unsupported alpha method: {self.alpha_config.method}")

    # 中文说明：`_fit_risk`：内部辅助步骤，不作为稳定公共接口。
    def _fit_risk(
        self,
        labels: pd.Series,
        exposures: pd.DataFrame,
        caps: pd.Series | None,
        as_of: pd.Timestamp,
        assets: pd.Index,
    ) -> RiskModelOutput:
        historical_dates = pd.Index(
            labels.index.get_level_values(0)[
                labels.index.get_level_values(0) < as_of
            ].unique()
        ).sort_values()
        returns = labels.loc[
            labels.index.get_level_values(0).isin(historical_dates)    # T,N
        ].unstack(level=1)
        exposure_history = exposures.loc[
            exposures.index.get_level_values(0).isin(historical_dates)  # T,N,F
        ]
        cap_history = (
            caps.loc[caps.index.get_level_values(0).isin(historical_dates)].unstack(level=1)  # T,N
            if caps is not None
            else None
        )
        current_exposures = exposures.xs(as_of, level=0).reindex(assets)   # N,F
        return self.risk_model.fit(
            returns,
            exposure_history,
            current_exposures,
            cap_history,
        )

    # 中文说明：`_cross_section`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _cross_section(
        value: pd.Series | pd.DataFrame,
        date: pd.Timestamp,
    ) -> pd.Series | pd.DataFrame:
        try:
            return value.xs(date, level=0)
        except KeyError as exc:
            raise ValueError(f"no data available on rebalance date {date.date()}") from exc
