"""Numerical utilities with stable defaults for noisy financial estimates."""

from __future__ import annotations

import numpy as np


# 中文说明：`exponential_weights`：执行该名称对应的业务计算，并返回调用方所需结果。
def exponential_weights(length: int, halflife: float) -> np.ndarray:
    if length <= 0 or halflife <= 0:
        raise ValueError("length and halflife must be positive")
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / halflife)
    return weights / weights.sum()


# 中文说明：`nearest_psd`：执行该名称对应的业务计算，并返回调用方所需结果。
def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    repaired = (vectors * values) @ vectors.T
    return (repaired + repaired.T) / 2


# 中文说明：`normalize_weights`：规范化输入或权重。
def normalize_weights(
    values: np.ndarray,
    allow_negative: bool,
    max_weight: float,
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if not allow_negative:
        raw = np.maximum(raw, 0.0)
    if np.allclose(raw, 0):
        raw = np.ones_like(raw)
    denominator = np.abs(raw).sum() if allow_negative else raw.sum()
    weights = raw / denominator
    if max_weight <= 0:
        return weights
    eligible = np.abs(raw) > 1e-15
    # A cap below 1 / n_active is infeasible. Relax it within the active set
    # instead of assigning weight to factors with no estimated efficacy.
    effective_max = max(max_weight, 1.0 / max(int(eligible.sum()), 1))
    for _ in range(len(weights) + 1):
        clipped = np.clip(
            weights,
            -effective_max if allow_negative else 0.0,
            effective_max,
        )
        residual = 1.0 - (np.abs(clipped).sum() if allow_negative else clipped.sum())
        free = eligible & (np.abs(clipped) < effective_max - 1e-12)
        if abs(residual) < 1e-12 or not free.any():
            weights = clipped
            break
        base = np.abs(raw[free]) if allow_negative else np.maximum(raw[free], 0)
        if base.sum() == 0:
            base = np.ones(free.sum())
        clipped[free] += residual * base / base.sum()
        weights = clipped
    scale = np.abs(weights).sum() if allow_negative else weights.sum()
    return weights / scale if scale else weights
