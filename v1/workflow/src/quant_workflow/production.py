"""Daily production orchestration consuming an approved research artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from quant_shared.alpha import (
    DynamicLinearAlpha,
    FamaMacBethAlpha,
    MonotonicScoreCalibrator,
    WalkForwardRidgeAlpha,
    WalkForwardSklearnAlpha,
)
from quant_shared.artifacts import ResearchArtifact
from quant_shared.combination import FactorCombiner, information_coefficients
from quant_shared.config import OptimizerConfig, StrategyType
from quant_shared.contracts import OptimizationResult, PanelData, RiskModelOutput
from .costs import TransactionCostModel
from .portfolio import PortfolioOptimizer
from quant_shared.preprocessing import CrossSectionalPreprocessor
from .risk import EquityFactorRiskModel, risk_attribution
from quant_shared.transforms import FactorTransformer


# 中文说明：定义 `DailyProductionResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class DailyProductionResult:
    as_of: pd.Timestamp
    artifact_id: str
    processed_factors: pd.DataFrame
    model_factors: pd.DataFrame
    composite_score: pd.Series
    factor_weights: pd.DataFrame
    expected_returns: pd.Series
    alpha_coefficients: pd.DataFrame | pd.Series
    risk: RiskModelOutput
    portfolio: OptimizationResult
    risk_attribution: pd.DataFrame
    diagnostics: Mapping[str, object]


# 中文说明：定义 `MultiStrategyProductionResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class MultiStrategyProductionResult:
    """One shared Alpha/risk snapshot and independent strategy portfolios."""

    base: DailyProductionResult
    portfolios: Mapping[StrategyType, OptimizationResult]


# 中文说明：定义 `DailyProductionWorkflow`，封装本模块对应的数据、配置与行为。
class DailyProductionWorkflow:
    """Run the repeatable daily line without research admission or clustering."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        artifact: ResearchArtifact,
        costs: TransactionCostModel | None = None,
    ) -> None:
        self.artifact = artifact.validate()
        self.preprocessor = CrossSectionalPreprocessor(artifact.preprocess)
        self.transformer = FactorTransformer(artifact.transform)
        self.combiner = FactorCombiner(artifact.composite)
        self.risk_model = EquityFactorRiskModel(artifact.risk)
        self.optimizer = PortfolioOptimizer(artifact.optimizer, costs)

    # 中文说明：`run`：执行主流程并返回结构化结果。
    def run(
        self,
        data: PanelData,
        as_of: str | pd.Timestamp,
        current_weights: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
        adv_fraction: pd.Series | None = None,
        optimizer: PortfolioOptimizer | None = None,
    ) -> DailyProductionResult:
        data.validate()
        rebalance_date = pd.Timestamp(as_of)
        if self.artifact.effective_from is not None:
            effective_time = pd.Timestamp(self.artifact.effective_from)
            effective_utc = (
                effective_time.tz_localize("UTC")
                if effective_time.tzinfo is None
                else effective_time.tz_convert("UTC")
            )
            rebalance_utc = (
                rebalance_date.tz_localize("UTC")
                if rebalance_date.tzinfo is None
                else rebalance_date.tz_convert("UTC")
            )
            if effective_utc > rebalance_utc:
                raise ValueError(
                    "research artifact is not effective on the production as_of date"
                )
        selected_optimizer = optimizer or self.optimizer
        missing = set(self.artifact.selected_factors) - set(data.factors.columns)
        if missing:
            raise ValueError(f"production factor input is missing: {sorted(missing)}")

        available = data.factors.index.get_level_values(0) <= rebalance_date
        factors = data.factors.loc[available, list(self.artifact.selected_factors)]
        labels = data.forward_returns.loc[available]
        exposures = data.exposures.loc[available] if data.exposures is not None else None
        caps = data.market_caps.loc[available] if data.market_caps is not None else None
        tradable = data.tradable.loc[available] if data.tradable is not None else None
        if exposures is None:
            raise ValueError("risk exposures are required for daily production")
        if rebalance_date not in factors.index.get_level_values(0):
            raise ValueError(
                f"factor snapshot is missing on rebalance date {rebalance_date.date()}"
            )
        if rebalance_date not in exposures.index.get_level_values(0):
            raise ValueError(
                f"risk exposure snapshot is missing on rebalance date {rebalance_date.date()}"
            )

        processed = self.preprocessor.transform(factors, exposures, caps)
        transformed = self.transformer.transform(processed, labels)
        model_factors = transformed.values
        model_labels = labels.reindex(model_factors.index)
        if self.artifact.composite.method == "icir":
            ic = information_coefficients(model_factors, model_labels)
            composite_score, factor_weights = self.combiner.combine(model_factors, ic)
        else:
            composite_score, factor_weights = self.combiner.combine(model_factors)
        expected_returns, alpha_coefficients = self._calibrate_alpha(
            model_factors, composite_score, model_labels
        )
        alpha_now = self._cross_section(expected_returns, rebalance_date)
        risk_output = self._fit_risk(
            labels, exposures, caps, rebalance_date, alpha_now.index
        )
        tradable_now = (
            self._cross_section(tradable, rebalance_date)
            if tradable is not None
            else None
        )
        portfolio = selected_optimizer.optimize(
            alpha=alpha_now,
            risk=risk_output,
            current_weights=current_weights,
            benchmark_weights=benchmark_weights,
            adv_fraction=adv_fraction,
            tradable=tradable_now,
        )
        return DailyProductionResult(
            as_of=rebalance_date,
            artifact_id=self.artifact.artifact_id,
            processed_factors=processed,
            model_factors=model_factors,
            composite_score=composite_score,
            factor_weights=factor_weights,
            expected_returns=expected_returns,
            alpha_coefficients=alpha_coefficients,
            risk=risk_output,
            portfolio=portfolio,
            risk_attribution=risk_attribution(portfolio.weights, risk_output),
            diagnostics={
                "research_steps_executed": False,
                "clustering_executed": False,
                "selected_factors": self.artifact.selected_factors,
            },
        )

    # 中文说明：`run_strategies`：执行主流程并返回结构化结果。
    def run_strategies(
        self,
        data: PanelData,
        as_of: str | pd.Timestamp,
        strategy_configs: Mapping[StrategyType, OptimizerConfig] | None = None,
        current_weights: Mapping[StrategyType, pd.Series] | None = None,
        benchmark_weights: pd.Series | None = None,
        adv_fraction: pd.Series | None = None,
    ) -> MultiStrategyProductionResult:
        """Generate strategy books from one approved Alpha and risk snapshot."""

        selected_configs = dict(
            strategy_configs or self.artifact.strategy_optimizers
        )
        if not selected_configs:
            raise ValueError("strategy_configs must contain at least one strategy")
        for strategy, config in selected_configs.items():
            if config.strategy != strategy:
                raise ValueError(
                    f"strategy key {strategy.value} does not match "
                    f"OptimizerConfig.strategy={config.strategy.value}"
                )
        if (
            StrategyType.INDEX_ENHANCED in selected_configs
            and benchmark_weights is None
        ):
            raise ValueError(
                "benchmark_weights are required for index enhancement"
            )

        holdings = current_weights or {}
        first_strategy = next(iter(selected_configs))
        first_optimizer = PortfolioOptimizer(
            selected_configs[first_strategy],
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
        alpha_now = self._cross_section(base.expected_returns, base.as_of)
        tradable_now = (
            self._cross_section(data.tradable, base.as_of)
            if data.tradable is not None
            else None
        )
        for strategy, config in selected_configs.items():
            if strategy == first_strategy:
                continue
            optimizer = PortfolioOptimizer(config, self.optimizer.cost_model)
            portfolios[strategy] = optimizer.optimize(
                alpha=alpha_now,
                risk=base.risk,
                current_weights=holdings.get(strategy),
                benchmark_weights=(
                    benchmark_weights
                    if strategy == StrategyType.INDEX_ENHANCED
                    else None
                ),
                adv_fraction=adv_fraction,
                tradable=tradable_now,
            )
        return MultiStrategyProductionResult(base=base, portfolios=portfolios)

    # 中文说明：`_calibrate_alpha`：内部辅助步骤，不作为稳定公共接口。
    def _calibrate_alpha(
        self,
        factors: pd.DataFrame,
        composite_score: pd.Series,
        labels: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series]:
        config = self.artifact.alpha
        if config.method == "ridge":
            return WalkForwardRidgeAlpha(config).fit_predict(factors, labels)
        if config.method == "score_slope":
            return MonotonicScoreCalibrator(config).fit_predict(
                composite_score, labels
            )
        if config.method == "fama_macbeth":
            return FamaMacBethAlpha(config).fit_predict(factors, labels)
        if config.method == "dynamic_linear":
            return DynamicLinearAlpha(config).fit_predict(factors, labels)
        if config.method in {
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
            return WalkForwardSklearnAlpha(config).fit_predict(factors, labels)
        raise ValueError(f"unsupported alpha method: {config.method}")

    # 中文说明：`_fit_risk`：内部辅助步骤，不作为稳定公共接口。
    def _fit_risk(
        self,
        labels: pd.Series,
        exposures: pd.DataFrame,
        caps: pd.Series | None,
        as_of: pd.Timestamp,
        assets: pd.Index,
    ) -> RiskModelOutput:
        dates = labels.index.get_level_values(0)
        historical_dates = pd.Index(dates[dates < as_of].unique()).sort_values()
        returns = labels.loc[dates.isin(historical_dates)].unstack(level=1)
        exposure_dates = exposures.index.get_level_values(0)
        exposure_history = exposures.loc[exposure_dates.isin(historical_dates)]
        cap_history = None
        if caps is not None:
            cap_dates = caps.index.get_level_values(0)
            cap_history = caps.loc[cap_dates.isin(historical_dates)].unstack(level=1)
        current_exposures = exposures.xs(as_of, level=0).reindex(assets)
        return self.risk_model.fit(
            returns, exposure_history, current_exposures, cap_history
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
            raise ValueError(
                f"no data available on rebalance date {date.date()}"
            ) from exc
