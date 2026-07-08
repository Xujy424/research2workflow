"""End-to-end v2 flow from registry-approved factors to stock weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .FactorRegistry import FactorRegistry, FactorStatus
from .config import PortfolioMethod, PortfolioRoute, ResearchFlowV2Config
from .matrix_math import cross_sectional_zscore
from .matrix_store import MatrixStore
from .FactorComb.alpha import UnifiedAlphaPath, UnifiedAlphaResult
from .FactorComb.family import FactorFamilyBuilder, FamilyBuildResult
from .Portfolio import CvxPortfolioOptimizer, FactorRiskModel, OptimizationResult, PortfolioProjectionResult, StockWeightProjector
from .FactorComb.sleeve import SleevePath, SleeveResult


@dataclass(frozen=True)
class ResearchToPortfolioResult:
    route: PortfolioRoute
    factor_names: tuple[str, ...]
    family_result: FamilyBuildResult
    alpha: np.ndarray
    stock_weights: np.ndarray
    projection: PortfolioProjectionResult | None
    optimization: tuple[OptimizationResult | None, ...] | None
    branch_result: SleeveResult | UnifiedAlphaResult


class ResearchToPortfolioWorkflow:
    """Excel-aligned v2 workflow.

    The single-factor test is deliberately not part of this class. It starts
    from factors that humans have admitted through ``FactorRegistry``.
    """

    def __init__(
        self,
        config: ResearchFlowV2Config | None = None,
        *,
        store: MatrixStore | None = None,
        registry: FactorRegistry | None = None,
    ) -> None:
        self.config = config or ResearchFlowV2Config()
        self.store = store or MatrixStore(self.config.data_root)
        self.registry = registry or FactorRegistry(self.config.registry_path)

    def run(self, *, save: bool = True) -> ResearchToPortfolioResult:
        metas = self.registry.by_status(FactorStatus.PRODUCTION, FactorStatus.SHADOW)
        if not metas:
            raise ValueError("no production or shadow factors are registered")
        names = tuple(meta.storage_field for meta in metas)
        families = {meta.storage_field: meta.family for meta in metas}
        factors = self._load_factor_cube(metas)
        tradable = self._read_or_default(self.config.tradable_category, self.config.tradable_field, True).astype(bool)
        industry = self._read_or_default(self.config.industry_category, self.config.industry_field, np.nan)
        labels = self._read_or_default(self.config.label_category, self.config.label_field, np.nan)

        family = FactorFamilyBuilder(self.config.family).build(
            factors,
            labels,
            factor_names=names,
            families=families,
            tradable=tradable,
        )
        if self.config.route == PortfolioRoute.SLEEVE:
            branch = SleevePath(self.config.sleeve).run(family.family_scores, labels=labels, tradable=tradable)
            alpha = cross_sectional_zscore(branch.merged_score, mask=tradable)
        elif self.config.route == PortfolioRoute.UNIFIED_ALPHA:
            branch = UnifiedAlphaPath(self.config.alpha).run(family.family_scores, labels, tradable=tradable)
            alpha = branch.alpha
        else:
            raise ValueError(f"unsupported route: {self.config.route}")

        current = self._optional(self.config.current_weight_category, self.config.current_weight_field)
        adv = self._optional(self.config.adv_category, self.config.adv_field)
        benchmark = self._optional(self.config.benchmark_category, self.config.benchmark_field)
        benchmark_member = self._optional(self.config.benchmark_member_category, self.config.benchmark_member_field)
        exposures = self._optional_cube(self.config.exposure_category, self.config.exposure_field)
        market_cap = self._optional(self.config.market_cap_category, self.config.market_cap_field)
        if self.config.optimizer.method == PortfolioMethod.PROJECT:
            projection = self._project_weights(alpha, tradable, current, adv, industry, benchmark)
            optimization = None
            weights = projection.weights
        elif self.config.optimizer.method == PortfolioMethod.OPTIMIZE:
            projection = None
            weights, optimization = self._optimize_weights(alpha, labels, tradable, current, adv, benchmark, benchmark_member, exposures, market_cap)
        else:
            raise ValueError(f"unsupported portfolio method: {self.config.optimizer.method}")

        if save:
            self.store.write_matrix(self.config.output_alpha_category, self.config.output_alpha_field, alpha)
            self.store.write_matrix(self.config.output_category, self.config.output_weight_field, weights)
        return ResearchToPortfolioResult(
            route=self.config.route,
            factor_names=names,
            family_result=family,
            alpha=alpha,
            stock_weights=weights,
            projection=projection,
            optimization=optimization,
            branch_result=branch,
        )

    def _project_weights(
        self,
        alpha: np.ndarray,
        tradable: np.ndarray,
        current: np.ndarray | None,
        adv: np.ndarray | None,
        industry: np.ndarray | None,
        benchmark: np.ndarray | None,
    ) -> PortfolioProjectionResult:
        return StockWeightProjector(self.config.optimizer).project(
            alpha,
            tradable=tradable,
            current_weight=current,
            benchmark_weight=benchmark,
            adv=adv,
            industry=industry,
        )

    def _optimize_weights(
        self,
        alpha: np.ndarray,
        labels: np.ndarray,
        tradable: np.ndarray,
        current: np.ndarray | None,
        adv: np.ndarray | None,
        benchmark: np.ndarray | None,
        benchmark_member: np.ndarray | None,
        exposures: np.ndarray | None,
        market_cap: np.ndarray | None,
    ) -> tuple[np.ndarray, tuple[OptimizationResult | None, ...]]:
        weights = np.zeros_like(alpha, dtype=float)
        results: list[OptimizationResult | None] = []
        if exposures is None:
            raise ValueError("optimizer.method='optimize' requires risk exposure cube, e.g. D:/data/barra/*.bin")
        optimizer = CvxPortfolioOptimizer(self.config.optimizer)
        risk_cfg = self.config.risk
        risk_model = FactorRiskModel(
            factor_halflife=risk_cfg.factor_halflife,
            specific_halflife=risk_cfg.specific_halflife,
            newey_west_lags=risk_cfg.newey_west_lags,
            covariance_shrinkage=risk_cfg.covariance_shrinkage,
            specific_shrinkage=risk_cfg.specific_shrinkage,
            variance_floor=risk_cfg.variance_floor,
            annualization=risk_cfg.annualization,
        )
        prev = np.zeros(alpha.shape[1], dtype=float) if current is None else np.nan_to_num(current[0], nan=0.0)
        for t in range(alpha.shape[0]):
            valid = tradable[t] & np.isfinite(alpha[t]) & np.isfinite(exposures[t]).all(axis=1)
            if valid.sum() < 2:
                results.append(None)
                continue
            start = max(0, t - self.config.family.lookback)
            try:
                estimate = risk_model.fit(
                    labels[start:t, valid],
                    exposures[start:t, valid],
                    exposures[t, valid],
                    market_cap_history=None if market_cap is None else market_cap[start:t, valid],
                    mask=tradable[start:t, valid],
                )
                covariance = estimate.stock_covariance
            except ValueError:
                results.append(None)
                weights[t] = prev
                continue
            result = optimizer.optimize(
                alpha[t, valid],
                covariance,
                current_weight=prev[valid],
                benchmark_weight=None if benchmark is None else benchmark[t, valid],
                benchmark_member_mask=None if benchmark_member is None else benchmark_member[t, valid].astype(bool),
                adv_weight=None if adv is None else adv[t, valid],
                tradable=tradable[t, valid],
                exposures=exposures[t, valid],
            )
            row = np.zeros(alpha.shape[1], dtype=float)
            row[valid] = result.weights
            weights[t] = row
            prev = row
            results.append(result)
        return weights, tuple(results)

    def _load_factor_cube(self, metas: list[object]) -> np.ndarray:
        arrays = [
            np.asarray(self.store.open_matrix(
                meta.category, meta.storage_field, dtype=meta.dtype
                ), dtype=float)
            for meta in metas
        ]
        return np.stack(arrays, axis=2)

    def _read_or_default(self, category: str, field: str, default: float | bool) -> np.ndarray:
        axis = self.store.load_axis()
        try:
            return np.asarray(self.store.open_matrix(category, field), dtype=float)
        except FileNotFoundError:
            return np.full(axis.shape, default)

    def _optional(self, category: str, field: str) -> np.ndarray | None:
        try:
            return np.asarray(self.store.open_matrix(category, field), dtype=float)
        except FileNotFoundError:
            return None

    def _optional_cube(self, category: str, field: str) -> np.ndarray | None:
        try:
            return np.asarray(self.store.open_cube(category, field), dtype=float)
        except FileNotFoundError:
            return None


