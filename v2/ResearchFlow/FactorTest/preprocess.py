"""Single-factor preprocessing for the analyst review page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import sys

import numpy as np

try:
    from ..matrix_math import cross_sectional_zscore, industry_size_neutralize, winsorize
    from ..matrix_store import MatrixStore
except ImportError:  # Allows direct execution from an IDE without package context.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ResearchFlow.matrix_math import cross_sectional_zscore, industry_size_neutralize, winsorize
    from ResearchFlow.matrix_store import MatrixStore


WinsorMethod = Literal["mad", "sigma", "quantile"]


@dataclass(frozen=True)
class PreprocessConfig:
    winsor_method: WinsorMethod = "mad"
    quantile_p: float = 0.01
    n_sigma: float = 3.0
    standardize: bool = True
    neutralize: bool = True


@dataclass(frozen=True)
class FactorDataStats:
    shape: tuple[int, int]
    finite_ratio: float
    nan_ratio: float
    inf_count: int
    mean_abs_cross_median: float         # 横截面中心是否偏离0
    median_cross_std: float              # 横截面离散度是否稳定
    extreme_ratio_mad3: float            # 极值比例
    max_abs_robust_z: float              # 最严重异常点强度


def describe_factor_values(factor: np.ndarray, *, mask: np.ndarray | None = None) -> FactorDataStats:
    values = np.asarray(factor, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    median = np.nanmedian(work, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(work - median), axis=1, keepdims=True)
    robust_z = np.divide(work - median, 1.4826 * mad, out=np.full_like(work, np.nan), where=mad > 0)
    cross_std = np.nanstd(work, axis=1)
    return FactorDataStats(
        shape=values.shape,
        finite_ratio=float(valid.mean()),
        nan_ratio=float(np.isnan(values).mean()),
        inf_count=int(np.isinf(values).sum()),
        mean_abs_cross_median=float(np.nanmean(np.abs(median))),
        median_cross_std=float(np.nanmedian(cross_std)),
        extreme_ratio_mad3=float(np.nanmean(np.abs(robust_z) > 3.0)),
        max_abs_robust_z=float(np.nanmax(np.abs(robust_z))) if np.isfinite(robust_z).any() else np.nan,
    )


def preprocess_factor_matrix(
    factor: np.ndarray,
    *,
    config: PreprocessConfig | None = None,
    tradable: np.ndarray | None = None,
    industry: np.ndarray | None = None,
    market_cap: np.ndarray | None = None,
) -> np.ndarray:
    """Winsorize, standardize, and optionally neutralize a ``T x N`` factor."""

    cfg = config or PreprocessConfig()
    mask = None if tradable is None else np.asarray(tradable, dtype=bool)
    out = winsorize(
        factor,
        method=cfg.winsor_method,
        p=cfg.quantile_p,
        n_sigma=cfg.n_sigma,
        mask=mask,
    )
    if cfg.standardize:
        out = cross_sectional_zscore(out, mask=mask)
    if cfg.neutralize:
        if industry is None or market_cap is None:
            raise ValueError("industry and market_cap are required when neutralize=True")
        out = industry_size_neutralize(out, industry, market_cap, mask=mask, standardize=cfg.standardize)
    return out


class FactorPreprocessor:
    """UI-friendly wrapper for data diagnostics and transformation."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def describe(self, factor: np.ndarray, *, mask: np.ndarray | None = None) -> FactorDataStats:
        return describe_factor_values(factor, mask=mask)

    def transform(
        self,
        factor: np.ndarray,
        *,
        tradable: np.ndarray | None = None,
        industry: np.ndarray | None = None,
        market_cap: np.ndarray | None = None,
    ) -> np.ndarray:
        return preprocess_factor_matrix(
            factor,
            config=self.config,
            tradable=tradable,
            industry=industry,
            market_cap=market_cap,
        )



if __name__ == '__main__':
    
    import pandas as pd
    import bisect

    ms = MatrixStore()
    axis = ms.load_axis()
    start_idx = bisect.bisect_left(axis.dates,pd.to_datetime('2021-01-01'))
    end_idx = bisect.bisect_right(axis.dates,pd.to_datetime('2025-12-31'))
    dts = axis.dates[start_idx:end_idx]

    tradable = ms.read_slice('mask','tradable',dtype=bool,dates=dts)
    industry = ms.read_slice('mask','industry',dtype=float,dates=dts)
    mv = ms.read_slice('d_field','mv',dtype=float,dates=dts)

    factor_path = Path(__file__).resolve().parents[1] / 'examples' / 'alpha_merge_20210104_20251231.csv'
    factor = pd.read_csv(factor_path, index_col=0)
    val = factor.to_numpy()

    preprocess = FactorPreprocessor()
    factor_stats = preprocess.describe(val)
    factor_trans = preprocess.transform(val, tradable=tradable, industry=industry, market_cap=mv)
    factor_trans = pd.DataFrame(factor_trans, index=factor.index, columns=factor.columns)
    factor_trans.to_csv(Path(__file__).resolve().parents[1] / 'examples' /'GRU.csv')
    print('hahahah')
