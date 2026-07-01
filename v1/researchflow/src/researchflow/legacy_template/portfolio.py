"""Named covariance helper preserved from the original Portfolio script."""

from __future__ import annotations

import numpy as np


# 中文说明：`newey_west_cov`：执行该名称对应的业务计算，并返回调用方所需结果。
def newey_west_cov(data: np.ndarray, q: int = 2) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    sample_count = len(values)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / sample_count
    for lag in range(1, min(q, sample_count - 1) + 1):
        weight = 1.0 - lag / (q + 1.0)
        cross = centered[lag:].T @ centered[:-lag] / sample_count
        covariance += weight * (cross + cross.T)
    return covariance
