"""Classify, transform, re-test, and combine approved factors inside families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config import FamilyConfig
from ..matrix_math import calc_icir, cross_sectional_zscore, factor_ic_history
from .clustering import FactorClusterer
from .combination import equal_weights, rolling_icir_weights
from .family_transform import FamilyTransform, FamilyTransformResult


@dataclass(frozen=True)
class FamilyBuildResult:
    family_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    representatives: dict[str, tuple[str, ...]]
    factor_to_family: dict[str, str]
    member_weights: dict[str, np.ndarray]
    transform_diagnostics: dict[str, dict[str, object]]
    family_scores: np.ndarray
    ic_history: np.ndarray


class FactorFamilyBuilder:
    """Build family composites from approved, already preprocessed factors.

    Flow: economic family grouping -> intra-family redundancy selection ->
    optional transform (raw/orthogonal/PCA/PLS) -> transformed-factor validation ->
    family-level combination.
    """

    def __init__(self, config: FamilyConfig | None = None) -> None:
        self.config = config or FamilyConfig()
        self._transform = FamilyTransform(self.config)

    def build(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        factor_names: tuple[str, ...],
        families: Mapping[str, str],
        tradable: np.ndarray | None = None,
    ) -> FamilyBuildResult:
        self._validate_inputs(factors, factor_names, families)
        mask = np.ones(factors.shape[:2], dtype=bool) if tradable is None else tradable.astype(bool)

        raw_ic = factor_ic_history(factors, labels, mask=mask)
        raw_icir = calc_icir(raw_ic)
        raw_quality = np.abs(raw_icir)
        factor_corr = self._factor_corr(factors, mask)  # K,K
        ic_corr = self._ic_corr(raw_ic)                 # K,K

        family_names = tuple(sorted(set(families[name] for name in factor_names)))
        family_scores = np.full((factors.shape[0], factors.shape[1], len(family_names)), np.nan, dtype=float)
        reps: dict[str, tuple[str, ...]] = {}
        member_weights: dict[str, np.ndarray] = {}
        transform_diagnostics: dict[str, dict[str, object]] = {}

        for j, family in enumerate(family_names):
            member_idx = [i for i, name in enumerate(factor_names) if families[name] == family]
            selected = self._select_representatives(member_idx, factor_corr, ic_corr, raw_quality)
            selected_names = tuple(factor_names[i] for i in selected)
            reps[family] = selected_names

            transform_result = self._transform_family(factors[:, :, selected], labels, mask, raw_icir[selected])
            transform_diagnostics[family] = transform_result.diagnostics
            candidates, candidate_ic = self._keep_explanatory_candidates(transform_result.values, labels, mask)
            weights = self._member_weights(candidate_ic)

            member_weights[family] = weights
            family_scores[:, :, j] = combine_factor_cube(candidates, weights, mask=mask)

        family_scores = np.stack(
            [cross_sectional_zscore(family_scores[:, :, j], mask=mask) for j in range(len(family_names))],
            axis=2,
        )
        return FamilyBuildResult(
            family_names=family_names,
            factor_names=factor_names,
            representatives=reps,
            factor_to_family=dict(families),
            member_weights=member_weights,
            transform_diagnostics=transform_diagnostics,
            family_scores=family_scores,
            ic_history=raw_ic,
        )

    @staticmethod
    def _validate_inputs(factors: np.ndarray, factor_names: tuple[str, ...], families: Mapping[str, str]) -> None:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if factors.shape[2] != len(factor_names):
            raise ValueError("factor_names length must match factors.shape[2]")
        missing = [name for name in factor_names if name not in families]
        if missing:
            raise KeyError(f"missing family mapping for factors: {missing}")

    @staticmethod
    def _corr_by_feature(values: np.ndarray, min_obs: int = 20) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        valid = np.isfinite(x)
        count = valid.sum(axis=1)
        mean = np.divide(
            np.nansum(np.where(valid, x, 0.0), axis=1),
            count,
            out=np.zeros(x.shape[0], dtype=float),
            where=count > 0,
        )
        centered = np.where(valid, x - mean[:, None], 0.0)
        gram = centered @ centered.T
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        denom = norm[:, None] * norm[None, :]
        out = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 1e-12)
        pair_count = valid.astype(float) @ valid.T.astype(float)
        out = np.where(pair_count >= min_obs, out, 0.0)
        np.fill_diagonal(out, 1.0)
        return out

    @classmethod
    def _factor_corr(cls, factors: np.ndarray, mask: np.ndarray, min_obs: int = 20) -> np.ndarray:
        values = np.where(mask.reshape(-1, 1), factors.reshape(-1, factors.shape[2]), np.nan).T  
        return cls._corr_by_feature(values, min_obs=min_obs)

    @classmethod
    def _ic_corr(cls, ic: np.ndarray, min_obs: int = 20) -> np.ndarray:
        return cls._corr_by_feature(ic.T, min_obs=min_obs)

    def _select_representatives(
        self,
        idx: list[int],
        factor_corr: np.ndarray,
        ic_corr: np.ndarray,
        icir: np.ndarray,
    ) -> list[int]:
        local = np.asarray(idx, dtype=int)
        result = FactorClusterer().select(
            factor_corr[np.ix_(local, local)],
            icir[local],
            ic_corr=ic_corr[np.ix_(local, local)],
            method=self.config.clustering_method,
            corr_threshold=self.config.corr_threshold,
            ic_corr_threshold=self.config.ic_corr_threshold,
            distance_threshold=self.config.distance_threshold,
        )
        return [int(local[i]) for i in result.representatives]

    def _transform_family(
        self,
        family_factors: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        icir: np.ndarray,
    ) -> FamilyTransformResult:
        return self._transform.run(family_factors, labels, mask=mask, quality=icir)

    def _keep_explanatory_candidates(
        self,
        candidates: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ic = factor_ic_history(candidates, labels, mask=mask)
        mean_ic = np.nanmean(ic, axis=0)
        icir = calc_icir(ic)
        enough_obs = np.isfinite(ic).sum(axis=0) >= self.config.min_ic_obs
        explanatory = (
            enough_obs
            & (np.abs(mean_ic) >= self.config.min_component_abs_ic)
            & (np.abs(icir) >= self.config.min_component_abs_icir)
        )
        if not explanatory.any():
            best = int(np.nanargmax(np.abs(icir))) if np.isfinite(icir).any() else 0
            explanatory[best] = True
        kept = candidates[:, :, explanatory].copy()
        kept_ic = ic[:, explanatory].copy()
        direction = np.sign(np.nanmean(kept_ic, axis=0))
        direction = np.where(direction==0, 1.0, direction)
        return kept * direction[None, None, :], kept_ic * direction[None, :]

    def _member_weights(self, ic: np.ndarray) -> np.ndarray:
        if self.config.composite_method == "equal":
            return equal_weights(ic.shape[0], ic.shape[1])
        if self.config.composite_method == "icir":
            return rolling_icir_weights(
                ic,
                lookback=self.config.lookback,
                min_periods=self.config.min_ic_obs,
                max_weight=self.config.max_member_weight,
            )
        raise ValueError(f"unsupported composite method: {self.config.composite_method}")


def combine_factor_cube(factors: np.ndarray, weights: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(factors)
    weighted = np.where(valid, factors * weights[:, None, :], 0.0)
    denom = np.where(valid, weights[:, None, :], 0.0).sum(axis=2)
    out = np.divide(weighted.sum(axis=2), denom, out=np.full(factors.shape[:2], np.nan), where=denom > 0)
    return cross_sectional_zscore(out, mask=mask)

