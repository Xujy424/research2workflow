"""End-to-end v2 flow from registry-approved factors to stock weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .FactorRegistry import FactorRegistry, FactorStatus
from .config import PortfolioRoute, ResearchFlowV2Config
from .matrix_math import cross_sectional_zscore
from .matrix_store import MatrixStore
from .FactorComb.alpha import UnifiedAlphaPath, UnifiedAlphaResult
from .FactorComb.family import FactorFamilyBuilder, FamilyBuildResult
from .Portfolio import PortfolioProjectionResult, StockWeightProjector
from .FactorComb.preprocess import FactorPoolPreprocessor
from .FactorComb.sleeve import SleevePath, SleeveResult


@dataclass(frozen=True)
class ResearchToPortfolioResult:
    route: PortfolioRoute
    factor_names: tuple[str, ...]
    family_result: FamilyBuildResult
    alpha: np.ndarray
    stock_weights: np.ndarray
    projection: PortfolioProjectionResult
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
        raw = self._load_factor_cube(metas)
        tradable = self._read_or_default(self.config.tradable_category, self.config.tradable_field, True).astype(bool)
        industry = self._read_or_default(self.config.industry_category, self.config.industry_field, np.nan)
        market_cap = self._read_or_default(self.config.market_cap_category, self.config.market_cap_field, np.nan)
        labels = self._read_or_default(self.config.label_category, self.config.label_field, np.nan)
        factors = self._preprocess(raw, tradable, industry, market_cap)

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
        projection = StockWeightProjector(self.config.optimizer).project(
            alpha,
            tradable=tradable,
            current_weight=current,
            adv=adv,
            industry=industry,
        )
        if save:
            self.store.write_matrix(self.config.output_alpha_category, self.config.output_alpha_field, alpha)
            self.store.write_matrix(self.config.output_category, self.config.output_weight_field, projection.weights)
        return ResearchToPortfolioResult(
            route=self.config.route,
            factor_names=names,
            family_result=family,
            alpha=alpha,
            stock_weights=projection.weights,
            projection=projection,
            branch_result=branch,
        )

    def _load_factor_cube(self, metas: list[object]) -> np.ndarray:
        arrays = [
            np.asarray(self.store.open_matrix(meta.category, meta.storage_field, dtype=meta.dtype), dtype=float)
            for meta in metas
        ]
        return np.stack(arrays, axis=2)

    def _preprocess(self, raw: np.ndarray, tradable: np.ndarray, industry: np.ndarray, market_cap: np.ndarray) -> np.ndarray:
        return FactorPoolPreprocessor(
            winsor_method=self.config.preprocess.winsor_method,
            standardize=self.config.preprocess.standardize,
            neutralize=self.config.preprocess.neutralize,
        ).transform(
            raw,
            tradable=tradable,
            industry=industry,
            market_cap=market_cap,
        )

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





