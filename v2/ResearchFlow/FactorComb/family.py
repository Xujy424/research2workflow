"""Classify, de-duplicate, and combine approved factors inside families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config import FamilyConfig
from ..matrix_math import cap_and_renormalize, cross_sectional_zscore, nan_corr_by_row
from .clustering import FactorClusterer, factor_correlation, time_series_corr


@dataclass(frozen=True)
class FamilyBuildResult:
    family_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    representatives: dict[str, tuple[str, ...]]
    factor_to_family: dict[str, str]
    member_weights: dict[str, np.ndarray]
    family_scores: np.ndarray
    ic_history: np.ndarray


class FactorFamilyBuilder:
    """Matrix implementation of factor classification, pruning, and composite building."""

    def __init__(self, config: FamilyConfig | None = None) -> None:
        self.config = config or FamilyConfig()

    def build(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        factor_names: tuple[str, ...],
        families: Mapping[str, str],
        tradable: np.ndarray | None = None,
    ) -> FamilyBuildResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if factors.shape[2] != len(factor_names):
            raise ValueError("factor_names length must match factors.shape[2]")
        missing = [name for name in factor_names if name not in families]
        if missing:
            raise KeyError(f"missing family mapping for factors: {missing}")

        mask = np.ones(factors.shape[:2], dtype=bool) if tradable is None else tradable.astype(bool)
        ic = self._ic_history(factors, labels, mask)
        quality = self._quality(ic)
        factor_corr = factor_correlation(factors, mask=mask)
        ic_corr = time_series_corr(ic)

        family_names = tuple(sorted(set(families[name] for name in factor_names)))
        family_scores = np.full((factors.shape[0], factors.shape[1], len(family_names)), np.nan, dtype=float)
        reps: dict[str, tuple[str, ...]] = {}
        member_weights: dict[str, np.ndarray] = {}
        for j, family in enumerate(family_names):
            idx = [i for i, name in enumerate(factor_names) if families[name] == family]
            selected = self._select_representatives(idx, factor_corr, ic_corr, quality)
            reps[family] = tuple(factor_names[i] for i in selected)
            weights = self._member_weights(ic[:, selected])
            member_weights[family] = weights
            family_scores[:, :, j] = combine_factor_cube(factors[:, :, selected], weights, mask=mask)
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
            family_scores=family_scores,
            ic_history=ic,
        )

    @staticmethod
    def _ic_history(factors: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ic = np.full((factors.shape[0], factors.shape[2]), np.nan, dtype=float)
        y = np.where(mask, labels, np.nan)
        for k in range(factors.shape[2]):
            ic[:, k] = nan_corr_by_row(factors[:, :, k], y, rank=True)
        return ic

    @staticmethod
    def _quality(ic: np.ndarray) -> np.ndarray:
        mean = np.nanmean(ic, axis=0)
        std = np.nanstd(ic, axis=0)
        return np.divide(mean, std, out=np.zeros_like(mean), where=std > 1e-12)

    def _select_representatives(
        self,
        idx: list[int],
        factor_corr: np.ndarray,
        ic_corr: np.ndarray,
        quality: np.ndarray,
    ) -> list[int]:
        local = np.asarray(idx, dtype=int)
        result = FactorClusterer().select(
            factor_corr[np.ix_(local, local)],
            quality[local],
            ic_corr=ic_corr[np.ix_(local, local)],
            method=self.config.clustering_method,
            corr_threshold=self.config.corr_threshold,
            ic_corr_threshold=self.config.ic_corr_threshold,
        )
        return [int(local[i]) for i in result.representatives]

    def _member_weights(self, ic: np.ndarray) -> np.ndarray:
        n_dates, n_members = ic.shape
        weights = np.full((n_dates, n_members), 1.0 / n_members, dtype=float)
        if self.config.composite_method == "equal":
            return weights
        if self.config.composite_method != "icir":
            raise ValueError(f"unsupported composite method: {self.config.composite_method}")
        for t in range(n_dates):
            history = ic[:t]
            valid_count = np.isfinite(history).sum(axis=0)
            if len(history) < self.config.min_ic_obs or valid_count.max(initial=0) < self.config.min_ic_obs:
                continue
            mean = np.nanmean(history, axis=0)
            std = np.nanstd(history, axis=0)
            raw = np.divide(mean, std, out=np.zeros(n_members), where=std > 1e-12)
            raw = np.maximum(raw, 0.0)
            if raw.sum() > 1e-12:
                weights[t] = cap_and_renormalize(raw / raw.sum(), max_weight=self.config.max_member_weight)
        return weights


def combine_factor_cube(factors: np.ndarray, weights: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(factors)
    weighted = np.where(valid, factors * weights[:, None, :], 0.0)
    denom = np.where(valid, weights[:, None, :], 0.0).sum(axis=2)
    out = np.divide(weighted.sum(axis=2), denom, out=np.full(factors.shape[:2], np.nan), where=denom > 0)
    return cross_sectional_zscore(out, mask=mask)


