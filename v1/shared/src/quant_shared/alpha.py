"""Alpha layer: convert dimensionless signals into expected-return units."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neural_network import MLPRegressor

from .config import AlphaConfig
from .math_utils import exponential_weights


# 中文说明：定义 `WalkForwardRidgeAlpha`，封装本模块对应的数据、配置与行为。
class WalkForwardRidgeAlpha:
    """Walk-forward ridge calibration with decayed observations.

    The coefficient used on date ``t`` is estimated only from dates strictly
    before ``t``. This is intentionally slower than an in-sample fit but keeps
    the research and production timing contract honest.
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: AlphaConfig | None = None) -> None:
        self.config = config or AlphaConfig()

    # 中文说明：`fit_predict`：拟合模型参数。
    def fit_predict(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        dates = pd.Index(features.index.get_level_values(0).unique()).sort_values()
        predictions: list[pd.Series] = []
        coefficients: list[pd.Series] = []
        for position, date in enumerate(dates):
            start = max(0, position - self.config.lookback)
            train_dates = dates[start:position]
            if len(train_dates) < self.config.min_periods:
                continue
            train_mask = features.index.get_level_values(0).isin(train_dates)
            x_train = features.loc[train_mask]
            y_train = forward_returns.loc[train_mask]
            beta = self._fit(x_train, y_train)
            x_now = features.xs(date, level=0)
            pred = x_now.fillna(0.0).to_numpy(float) @ beta
            pred = np.clip(
                pred,
                -self.config.clip_sigma * np.nanstd(y_train),
                self.config.clip_sigma * np.nanstd(y_train),
            )
            index = pd.MultiIndex.from_product(
                [[date], x_now.index], names=features.index.names
            )
            predictions.append(pd.Series(pred, index=index))
            coefficients.append(pd.Series(beta, index=features.columns, name=date))
        if not predictions:
            return pd.Series(dtype=float, name="expected_return"), pd.DataFrame(
                columns=features.columns
            )
        return (
            pd.concat(predictions).sort_index().rename("expected_return"),
            pd.DataFrame(coefficients),
        )

    # 中文说明：`_fit`：内部辅助步骤，不作为稳定公共接口。
    def _fit(self, features: pd.DataFrame, returns: pd.Series) -> np.ndarray:
        valid = features.notna().all(axis=1) & returns.notna()
        x = features.loc[valid].to_numpy(float)
        y = returns.loc[valid].to_numpy(float)
        date_codes = pd.factorize(features.loc[valid].index.get_level_values(0))[0]
        n_dates = date_codes.max() + 1
        date_weights = exponential_weights(n_dates, self.config.decay_halflife)
        observation_weights = np.sqrt(date_weights[date_codes])
        xw = x * observation_weights[:, None]
        yw = y * observation_weights
        penalty = self.config.ridge * np.eye(x.shape[1])
        return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


# 中文说明：定义 `MonotonicScoreCalibrator`，封装本模块对应的数据、配置与行为。
class MonotonicScoreCalibrator:
    """Fast score-to-return calibration using historical cross-sectional slope."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: AlphaConfig | None = None) -> None:
        self.config = config or AlphaConfig(method="score_slope")

    # 中文说明：`fit_predict`：拟合模型参数。
    def fit_predict(
        self, score: pd.Series, forward_returns: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        daily_slope: dict[pd.Timestamp, float] = {}
        for date, signal in score.groupby(level=0, sort=True):
            x = signal.droplevel(0)
            y = forward_returns.xs(date, level=0)
            valid = x.notna() & y.notna()
            denominator = float(np.dot(x[valid], x[valid]))
            daily_slope[pd.Timestamp(date)] = (
                float(np.dot(x[valid], y[valid]) / denominator) if denominator > 0 else np.nan
            )
        slope = pd.Series(daily_slope).rolling(
            self.config.lookback, min_periods=self.config.min_periods
        ).median().shift(1)
        aligned = slope.reindex(score.index.get_level_values(0)).set_axis(score.index)
        return (score * aligned).rename("expected_return"), slope


# 中文说明：定义 `FamaMacBethAlpha`，封装本模块对应的数据、配置与行为。
class FamaMacBethAlpha:
    """Rolling Fama-MacBeth estimator with coefficient shrinkage."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: AlphaConfig | None = None) -> None:
        self.config = config or AlphaConfig(method="fama_macbeth")

    # 中文说明：`fit_predict`：拟合模型参数。
    def fit_predict(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        daily_beta = self._daily_coefficients(features, forward_returns)
        rolling_beta = daily_beta.rolling(
            self.config.lookback, min_periods=self.config.min_periods
        ).mean().shift(1)
        predictions: list[pd.Series] = []
        for date, beta in rolling_beta.dropna(how="all").iterrows():
            if date not in features.index.get_level_values(0):
                continue
            current = features.xs(date, level=0)
            values = current.fillna(0.0).to_numpy(float) @ beta.fillna(0.0).to_numpy(float)
            index = pd.MultiIndex.from_product(
                [[date], current.index], names=features.index.names
            )
            predictions.append(pd.Series(values, index=index))
        result = (
            pd.concat(predictions).sort_index()
            if predictions
            else pd.Series(dtype=float)
        )
        return result.rename("expected_return"), rolling_beta

    # 中文说明：`_daily_coefficients`：内部辅助步骤，不作为稳定公共接口。
    def _daily_coefficients(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        rows: list[pd.Series] = []
        for date, frame in features.groupby(level=0, sort=True):
            x = frame.droplevel(0)
            y = returns.xs(date, level=0)
            valid = x.notna().all(axis=1) & y.notna()
            if valid.sum() <= x.shape[1] + 2:
                continue
            xv = x.loc[valid].to_numpy(float)
            yv = y.loc[valid].to_numpy(float)
            penalty = self.config.ridge * np.eye(xv.shape[1])
            beta = np.linalg.solve(xv.T @ xv + penalty, xv.T @ yv)
            rows.append(pd.Series(beta, index=features.columns, name=date))
        return pd.DataFrame(rows)


# 中文说明：定义 `WalkForwardSklearnAlpha`，封装本模块对应的数据、配置与行为。
class WalkForwardSklearnAlpha:
    """Walk-forward PLS, Elastic Net, random forest, or histogram GBDT."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: AlphaConfig | None = None) -> None:
        self.config = config or AlphaConfig(method="elastic_net")

    # 中文说明：`fit_predict`：拟合模型参数。
    def fit_predict(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        dates = pd.Index(features.index.get_level_values(0).unique()).sort_values()
        predictions: list[pd.Series] = []
        importance_rows: list[pd.Series] = []
        for position, date in enumerate(dates):
            train_dates = dates[max(0, position - self.config.lookback) : position]
            if len(train_dates) < self.config.min_periods:
                continue
            mask = features.index.get_level_values(0).isin(train_dates)
            x_train = features.loc[mask]
            y_train = forward_returns.loc[mask]
            valid = x_train.notna().all(axis=1) & y_train.notna()
            if valid.sum() <= features.shape[1] + 5:
                continue
            x = x_train.loc[valid].to_numpy(float)
            y = y_train.loc[valid].to_numpy(float)
            if self.config.method == "rank_gbdt":
                y = (
                    pd.Series(y, index=x_train.loc[valid].index)
                    .groupby(level=0)
                    .rank(pct=True)
                    .sub(0.5)
                    .to_numpy(float)
                )
            mean = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            model = self._make_model()
            model.fit((x - mean) / std, y)
            current = features.xs(date, level=0)
            pred = model.predict((current.fillna(0.0).to_numpy(float) - mean) / std)
            index = pd.MultiIndex.from_product(
                [[date], current.index], names=features.index.names
            )
            predictions.append(pd.Series(np.asarray(pred).reshape(-1), index=index))
            importance_rows.append(
                pd.Series(self._importance(model, features.shape[1]), index=features.columns, name=date)
            )
        output = pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)
        return output.rename("expected_return"), pd.DataFrame(importance_rows)

    # 中文说明：`_make_model`：内部辅助步骤，不作为稳定公共接口。
    def _make_model(self) -> object:
        method = self.config.method.lower()
        if method in {"elastic_net", "lasso"}:
            return ElasticNet(
                alpha=self.config.ridge,
                l1_ratio=1.0 if method == "lasso" else self.config.l1_ratio,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
        if method == "bayesian_ridge":
            return BayesianRidge(n_iter=self.config.max_iter)
        if method == "pls":
            return PLSRegression(
                n_components=self.config.n_components,
                scale=False,
                max_iter=self.config.max_iter,
            )
        if method == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=self.config.random_state,
            )
        if method in {"gbdt", "hist_gbdt", "rank_gbdt"}:
            return HistGradientBoostingRegressor(
                max_iter=min(self.config.max_iter, 200),
                max_leaf_nodes=15,
                l2_regularization=self.config.ridge,
                random_state=self.config.random_state,
            )
        if method == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=self.config.ridge,
                early_stopping=True,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
        raise ValueError(f"unsupported sklearn alpha method: {self.config.method}")

    # 中文说明：`_importance`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _importance(model: object, n_features: int) -> np.ndarray:
        if hasattr(model, "coef_"):
            values = np.asarray(getattr(model, "coef_"), dtype=float).reshape(-1)
            return values[:n_features]
        if hasattr(model, "feature_importances_"):
            return np.asarray(getattr(model, "feature_importances_"), dtype=float)
        return np.full(n_features, np.nan)


# 中文说明：定义 `DynamicLinearAlpha`，封装本模块对应的数据、配置与行为。
class DynamicLinearAlpha:
    """Online state-space approximation using recursive least squares.

    Coefficients follow a random walk. ``forgetting_factor`` controls state
    drift: values near one are stable, lower values adapt faster.
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        config: AlphaConfig | None = None,
        forgetting_factor: float = 0.99,
        initial_uncertainty: float = 100.0,
    ) -> None:
        self.config = config or AlphaConfig(method="dynamic_linear")
        self.forgetting_factor = forgetting_factor
        self.initial_uncertainty = initial_uncertainty

    # 中文说明：`fit_predict`：拟合模型参数。
    def fit_predict(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        columns = features.columns
        beta = np.zeros(len(columns))
        covariance = np.eye(len(columns)) * self.initial_uncertainty
        predictions: list[pd.Series] = []
        coefficient_rows: list[pd.Series] = []
        dates = pd.Index(features.index.get_level_values(0).unique()).sort_values()
        for position, date in enumerate(dates):
            current = features.xs(date, level=0)
            if position >= self.config.min_periods:
                values = current.fillna(0.0).to_numpy(float) @ beta
                index = pd.MultiIndex.from_product(
                    [[date], current.index], names=features.index.names
                )
                predictions.append(pd.Series(values, index=index))
                coefficient_rows.append(pd.Series(beta.copy(), index=columns, name=date))
            x_day = current
            y_day = forward_returns.xs(date, level=0)
            valid = x_day.notna().all(axis=1) & y_day.notna()
            for x, y in zip(
                x_day.loc[valid].to_numpy(float),
                y_day.loc[valid].to_numpy(float),
            ):
                projected = covariance @ x
                denominator = self.forgetting_factor + x @ projected
                gain = projected / max(denominator, 1e-12)
                beta = beta + gain * (y - x @ beta)
                covariance = (
                    covariance - np.outer(gain, x) @ covariance
                ) / self.forgetting_factor
        output = pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)
        return output.rename("expected_return"), pd.DataFrame(coefficient_rows)
