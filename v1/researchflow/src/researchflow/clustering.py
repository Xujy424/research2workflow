"""Correlation clustering and factor-family hierarchical composites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# 中文说明：定义 `ClusterResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ClusterResult:
    assignments: pd.Series
    representatives: list[str]
    linkage_matrix: np.ndarray


# 中文说明：定义 `FactorClusterer`，封装本模块对应的数据、配置与行为。
class FactorClusterer:
    # 中文说明：`cluster`：执行该名称对应的业务计算，并返回调用方所需结果。
    def cluster(
        self,
        correlation: pd.DataFrame,
        quality: pd.Series | None = None,
        threshold: float = 0.30,
    ) -> ClusterResult:
        corr = correlation.loc[correlation.index, correlation.index].fillna(0.0)
        if len(corr) == 0:
            raise ValueError("correlation must contain at least one factor")
        if len(corr) == 1:
            assignments = pd.Series(1, index=corr.index, name="cluster")
            return ClusterResult(
                assignments=assignments,
                representatives=[str(corr.index[0])],
                linkage_matrix=np.empty((0, 4), dtype=float),
            )
        distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr.abs().to_numpy(float))))
        np.fill_diagonal(distance, 0.0)
        linkage_matrix = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
        assignments = pd.Series(labels, index=corr.index, name="cluster")
        score = quality.reindex(corr.index).fillna(0.0) if quality is not None else pd.Series(0.0, index=corr.index)
        representatives = [
            str(score.loc[assignments[assignments == cluster].index].idxmax())
            for cluster in sorted(assignments.unique())
        ]
        return ClusterResult(assignments, representatives, linkage_matrix)


# 中文说明：定义 `HierarchicalFactorComposite`，封装本模块对应的数据、配置与行为。
class HierarchicalFactorComposite:
    """Combine factors inside economic families, then combine family scores."""

    # 中文说明：`combine`：组合多个输入并生成统一结果。
    def combine(
        self,
        factors: pd.DataFrame,
        families: Mapping[str, str],
        factor_weights: pd.Series | None = None,
        family_weights: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        missing = set(factors.columns) - set(families)
        if missing:
            raise KeyError(f"missing family mapping for factors: {sorted(missing)}")
        weights = (
            factor_weights.reindex(factors.columns).fillna(0.0)
            if factor_weights is not None
            else pd.Series(1.0, index=factors.columns)
        )
        family_scores: dict[str, pd.Series] = {}
        for family in sorted(set(families.values())):
            members = [factor for factor in factors.columns if families[factor] == family]
            member_weights = weights.loc[members]
            if member_weights.abs().sum() == 0:
                member_weights[:] = 1.0
            member_weights = member_weights / member_weights.abs().sum()
            numerator = factors[members].mul(member_weights, axis=1).sum(axis=1, min_count=1)
            denominator = factors[members].notna().mul(member_weights.abs(), axis=1).sum(axis=1)
            family_scores[family] = numerator / denominator.replace(0.0, np.nan)
        family_frame = pd.DataFrame(family_scores)
        fw = (
            family_weights.reindex(family_frame.columns).fillna(0.0)
            if family_weights is not None
            else pd.Series(1.0, index=family_frame.columns)
        )
        fw = fw / fw.abs().sum()
        composite = family_frame.mul(fw, axis=1).sum(axis=1, min_count=1)
        return family_frame, composite.rename("hierarchical_composite")
