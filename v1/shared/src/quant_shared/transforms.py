"""Factor transforms: residualisation, orthogonalisation, PCA, and PLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from .config import TransformConfig


# 中文说明：定义 `TransformResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class TransformResult:
    values: pd.DataFrame
    loadings: pd.DataFrame
    diagnostics: dict[str, object]


# 中文说明：定义 `FactorTransformer`，封装本模块对应的数据、配置与行为。
class FactorTransformer:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: TransformConfig | None = None) -> None:
        self.config = config or TransformConfig()

    # 中文说明：`transform`：转换输入数据并保持索引对齐。
    def transform(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series | None = None,
    ) -> TransformResult:
        method = self.config.method.lower()
        if method == "none":
            return TransformResult(factors.copy(), pd.DataFrame(), {"method": "none"})
        if method == "orthogonal":
            return self.orthogonalize(factors)
        if method == "pca":
            return self.walk_forward_pca(factors)
        if method == "pls":
            if forward_returns is None:
                raise ValueError("PLS requires forward returns")
            return self.walk_forward_pls(factors, forward_returns)
        raise ValueError(f"unsupported transform method: {self.config.method}")

    # 中文说明：`orthogonalize`：执行该名称对应的业务计算，并返回调用方所需结果。
    def orthogonalize(self, factors: pd.DataFrame) -> TransformResult:
        frames: list[pd.DataFrame] = []
        loading_rows: list[pd.DataFrame] = []
        for date, frame in factors.groupby(level=0, sort=True):
            cross = frame.droplevel(0)
            valid = cross.notna().all(axis=1)
            output = pd.DataFrame(np.nan, index=cross.index, columns=cross.columns)
            if valid.sum() >= cross.shape[1] + 2:
                x = cross.loc[valid].to_numpy(float)
                centered = x - x.mean(axis=0)
                if self.config.orthogonalization == "sequential":
                    q, r = np.linalg.qr(centered, mode="reduced")
                    transformed = q * np.sqrt(len(q))
                    loadings = np.linalg.pinv(r)
                elif self.config.orthogonalization == "symmetric":
                    covariance = centered.T @ centered / len(centered)
                    values, vectors = np.linalg.eigh(covariance)
                    inverse_sqrt = (vectors * np.maximum(values, self.config.ridge) ** -0.5) @ vectors.T
                    transformed = centered @ inverse_sqrt
                    loadings = inverse_sqrt
                else:
                    raise ValueError("orthogonalization must be sequential or symmetric")
                output.loc[valid] = transformed
                loading_rows.append(
                    pd.DataFrame(
                        loadings,
                        index=cross.columns,
                        columns=cross.columns,
                    ).assign(date=pd.Timestamp(date))
                )
            output.index = pd.MultiIndex.from_product(
                [[date], output.index], names=factors.index.names
            )
            frames.append(output)
        loading = (
            pd.concat(loading_rows).set_index("date", append=True).swaplevel(0, 1)
            if loading_rows
            else pd.DataFrame()
        )
        return TransformResult(
            pd.concat(frames).reindex(factors.index),
            loading,
            {"method": self.config.orthogonalization},
        )

    # 中文说明：`residualize`：执行该名称对应的业务计算，并返回调用方所需结果。
    def residualize(
        self,
        target: pd.Series,
        controls: pd.DataFrame,
    ) -> pd.Series:
        pieces: list[pd.Series] = []
        for date, y_group in target.groupby(level=0, sort=True):
            y = y_group.droplevel(0)
            x = controls.xs(date, level=0)
            common = y.index.intersection(x.index)
            valid = y.loc[common].notna() & x.loc[common].notna().all(axis=1)
            residual = pd.Series(np.nan, index=y.index)
            if valid.sum() >= x.shape[1] + 2:
                xv = x.loc[common[valid]].to_numpy(float)
                yv = y.loc[common[valid]].to_numpy(float)
                design = np.column_stack([np.ones(len(xv)), xv])
                penalty = self.config.ridge * np.eye(design.shape[1])
                penalty[0, 0] = 0.0
                beta = np.linalg.solve(design.T @ design + penalty, design.T @ yv)
                residual.loc[common[valid]] = yv - design @ beta
            residual.index = pd.MultiIndex.from_product(
                [[date], residual.index], names=target.index.names
            )
            pieces.append(residual)
        return pd.concat(pieces).reindex(target.index).rename(target.name)

    # 中文说明：`walk_forward_pca`：执行该名称对应的业务计算，并返回调用方所需结果。
    def walk_forward_pca(self, factors: pd.DataFrame) -> TransformResult:
        return self._walk_forward_projection(factors, None, supervised=False)

    # 中文说明：`walk_forward_pls`：执行该名称对应的业务计算，并返回调用方所需结果。
    def walk_forward_pls(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> TransformResult:
        return self._walk_forward_projection(factors, forward_returns, supervised=True)

    # 中文说明：`_walk_forward_projection`：内部辅助步骤，不作为稳定公共接口。
    def _walk_forward_projection(
        self,
        factors: pd.DataFrame,
        returns: pd.Series | None,
        supervised: bool,
    ) -> TransformResult:
        dates = pd.Index(factors.index.get_level_values(0).unique()).sort_values()
        projected: list[pd.DataFrame] = []
        loadings: list[pd.DataFrame] = []
        explained: dict[pd.Timestamp, list[float]] = {}
        n_components = min(self.config.n_components, factors.shape[1])
        columns = [f"{'PLS' if supervised else 'PC'}{i + 1}" for i in range(n_components)]
        for position, date in enumerate(dates):
            train_dates = dates[max(0, position - self.config.lookback) : position]
            if len(train_dates) < self.config.min_periods:
                continue
            train_mask = factors.index.get_level_values(0).isin(train_dates)
            x_train = factors.loc[train_mask]
            valid = x_train.notna().all(axis=1)
            if supervised:
                assert returns is not None
                valid &= returns.loc[train_mask].notna()
            if valid.sum() <= n_components:
                continue
            x_values = x_train.loc[valid].to_numpy(float)
            x_mean = x_values.mean(axis=0)
            x_std = x_values.std(axis=0, ddof=0)
            x_std[x_std == 0] = 1.0
            x_scaled = (x_values - x_mean) / x_std
            if supervised:
                model = PLSRegression(n_components=n_components, scale=False)
                y = returns.loc[train_mask].loc[valid].to_numpy(float)
                model.fit(x_scaled, y)
                rotation = model.x_rotations_
                explained[pd.Timestamp(date)] = []
            else:
                model = PCA(n_components=n_components, svd_solver="full")
                model.fit(x_scaled)
                rotation = model.components_.T
                explained[pd.Timestamp(date)] = model.explained_variance_ratio_.tolist()
            current = factors.xs(date, level=0)
            current_scaled = (current - x_mean) / x_std
            values = current_scaled.fillna(0.0).to_numpy(float) @ rotation
            index = pd.MultiIndex.from_product(
                [[date], current.index], names=factors.index.names
            )
            projected.append(pd.DataFrame(values, index=index, columns=columns))
            loadings.append(
                pd.DataFrame(rotation, index=factors.columns, columns=columns).assign(
                    date=pd.Timestamp(date)
                )
            )
        loading_frame = (
            pd.concat(loadings).set_index("date", append=True).swaplevel(0, 1)
            if loadings
            else pd.DataFrame()
        )
        return TransformResult(
            pd.concat(projected).sort_index() if projected else pd.DataFrame(columns=columns),
            loading_frame,
            {"method": "pls" if supervised else "pca", "explained_variance": explained},
        )
