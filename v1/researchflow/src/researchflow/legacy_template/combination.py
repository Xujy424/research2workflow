"""Named compatibility methods from the original Combination scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


# 中文说明：`rolling_icir`：执行该名称对应的业务计算，并返回调用方所需结果。
def rolling_icir(ic_series: pd.Series, window: int = 12) -> pd.Series:
    roll_mean = ic_series.rolling(window, min_periods=max(1, window // 2)).mean()
    roll_std = ic_series.rolling(window, min_periods=max(1, window // 2)).std()
    return roll_mean / roll_std.replace(0.0, np.nan)


# 中文说明：定义 `Orthogonalization`，封装本模块对应的数据、配置与行为。
class Orthogonalization:
    """Legacy method names backed by explicit array inputs.

    ``align_index`` requires the caller to provide the pool arrays as instance
    attributes, preserving the old API without loading institution paths at
    import time.
    """

    poolfactor_dates: object
    poolfactor_names: object
    poolfactor_ticks: object
    poolfactors: np.ndarray

    # 中文说明：`align_index`：执行该名称对应的业务计算，并返回调用方所需结果。
    def align_index(self, alpha_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        valid_dates = np.intersect1d(
            pd.to_datetime(self.poolfactor_dates), alpha_df.index
        )
        pool_date_index = np.searchsorted(
            pd.to_datetime(self.poolfactor_dates), valid_dates
        )
        alpha_date_index = np.searchsorted(alpha_df.index, valid_dates)
        alpha = alpha_df.to_numpy(float)[alpha_date_index]
        pool = self.poolfactors[pool_date_index]
        _, pool_tick_index, alpha_tick_index = np.intersect1d(
            self.poolfactor_ticks, alpha_df.columns, return_indices=True
        )
        return (
            alpha[:, alpha_tick_index],
            pool.transpose(2, 0, 1)[pool_tick_index].transpose(1, 0, 2),
        )

    # 中文说明：`ortho_for_t`：执行该名称对应的业务计算，并返回调用方所需结果。
    def ortho_for_t(
        self,
        t: int,
        y_t: np.ndarray,
        x_t: np.ndarray,
    ) -> np.ndarray:
        del t
        valid = np.isfinite(y_t) & np.isfinite(x_t).all(axis=1)
        result = np.full_like(y_t, np.nan, dtype=float)
        if valid.sum() <= x_t.shape[1]:
            result[valid] = y_t[valid]
            return result
        design = np.column_stack([np.ones(valid.sum()), x_t[valid]])
        beta = np.linalg.lstsq(design, y_t[valid], rcond=None)[0]
        result[valid] = y_t[valid] - design @ beta
        return result

    # 中文说明：`ortho_newalpha_parallel`：执行该名称对应的业务计算，并返回调用方所需结果。
    def ortho_newalpha_parallel(
        self,
        y: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [self.ortho_for_t(t, y[t], x[t]) for t in range(len(y))]
        )

    # 中文说明：`run`：执行主流程并返回结构化结果。
    def run(self, alpha_df: pd.DataFrame) -> np.ndarray:
        alpha, pool = self.align_index(alpha_df)
        return self.ortho_newalpha_parallel(alpha, pool)
