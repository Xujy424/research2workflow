"""Factor redundancy clustering and representative selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    representatives: np.ndarray
    method: str
    linkage_matrix: np.ndarray | None = None


class FactorClusterer:
    """Select representative and complementary factors inside a family."""

    def select(
        self,
        factor_corr: np.ndarray,
        quality: np.ndarray,
        *,
        ic_corr: np.ndarray | None = None,
        method: str = "greedy",
        corr_threshold: float = 0.85,
        ic_corr_threshold: float = 0.80,
        distance_threshold: float = 0.30,
    ) -> ClusterResult:
        if method == "greedy":
            return self.greedy_select(
                factor_corr,
                quality,
                ic_corr=ic_corr,
                corr_threshold=corr_threshold,
                ic_corr_threshold=ic_corr_threshold,
            )
        if method == "hierarchical":
            return self.hierarchical_select(
                factor_corr,
                quality,
                distance_threshold=distance_threshold,
            )
        raise ValueError(f"unsupported clustering method: {method}")

    def greedy_select(
        self,
        factor_corr: np.ndarray,
        quality: np.ndarray,
        *,
        ic_corr: np.ndarray | None = None,
        corr_threshold: float = 0.85,
        ic_corr_threshold: float = 0.80,
    ) -> ClusterResult:
        score = np.nan_to_num(np.asarray(quality, dtype=float), nan=-np.inf)
        n = len(score)
        ordered = sorted(range(n), key=lambda i: score[i], reverse=True)
        selected: list[int] = []
        labels = np.zeros(n, dtype=int)
        for i in ordered:
            duplicate_of = None
            for cluster_id, j in enumerate(selected, start=1):
                redundant = abs(factor_corr[i, j]) >= corr_threshold
                if ic_corr is not None:
                    redundant = redundant or abs(ic_corr[i, j]) >= ic_corr_threshold
                if redundant:
                    duplicate_of = cluster_id
                    break
            if duplicate_of is None:
                selected.append(i)
                labels[i] = len(selected)
            else:
                labels[i] = duplicate_of
        if not selected and n:
            selected = [int(np.nanargmax(score))]
            labels[selected[0]] = 1
        return ClusterResult(
            labels=labels,
            representatives=np.asarray(selected, dtype=int),
            method="greedy",
        )

    def hierarchical_select(
        self,
        factor_corr: np.ndarray,
        quality: np.ndarray,
        *,
        distance_threshold: float = 0.30,
    ) -> ClusterResult:
        score = np.nan_to_num(np.asarray(quality, dtype=float), nan=-np.inf)
        n = len(score)
        if n == 0:
            return ClusterResult(np.array([], dtype=int), np.array([], dtype=int), "hierarchical")
        if n == 1:
            return ClusterResult(np.array([1]), np.array([0]), "hierarchical", np.empty((0, 4)))
        distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - np.abs(factor_corr))))
        np.fill_diagonal(distance, 0.0)
        z = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(z, t=distance_threshold, criterion="distance")
        reps = []
        for cluster_id in sorted(np.unique(labels)):
            idx = np.flatnonzero(labels == cluster_id)
            reps.append(idx[np.nanargmax(score[idx])])
        return ClusterResult(
            labels=labels.astype(int),
            representatives=np.asarray(reps, dtype=int),
            method="hierarchical",
            linkage_matrix=z,
        )
