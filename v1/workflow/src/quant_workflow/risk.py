"""Equity factor risk model: exposures, factor covariance, and specific risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_shared.config import RiskConfig
from quant_shared.contracts import RiskModelOutput
from quant_shared.math_utils import exponential_weights, nearest_psd


# 中文说明：定义 `FactorReturnEstimate`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class FactorReturnEstimate:
    factor_returns: pd.DataFrame
    residual_returns: pd.DataFrame
    diagnostics: dict[str, float]


# 中文说明：定义 `EquityFactorRiskModel`，封装本模块对应的数据、配置与行为。
class EquityFactorRiskModel:
    """Cross-sectional factor risk model with robust covariance estimates."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    # 中文说明：`fit`：拟合模型参数。
    def fit(
        self,
        asset_returns: pd.DataFrame,
        exposure_history: pd.DataFrame,
        current_exposures: pd.DataFrame,
        market_cap_history: pd.DataFrame | None = None,
    ) -> RiskModelOutput:
        estimate = self.estimate_factor_returns(
            asset_returns, exposure_history, market_cap_history
        )
        factor_covariance = self._factor_covariance(estimate.factor_returns)  # F,F
        specific_variance = self._specific_variance(estimate.residual_returns)  # N,
        assets = current_exposures.index
        factor_covariance = factor_covariance.reindex(
            index=current_exposures.columns, columns=current_exposures.columns
        ).fillna(0.0)
        specific_variance = specific_variance.reindex(assets)
        fallback = float(specific_variance.median())
        specific_variance = specific_variance.fillna(fallback).clip(
            lower=self.config.variance_floor
        )
        x = current_exposures.fillna(0.0).to_numpy(float)
        sigma = x @ factor_covariance.to_numpy(float) @ x.T
        sigma += np.diag(specific_variance.to_numpy(float))
        sigma = nearest_psd(sigma, self.config.variance_floor)
        output = RiskModelOutput(
            assets=assets,
            exposures=current_exposures,
            factor_covariance=factor_covariance,
            specific_variance=specific_variance,
            stock_covariance=pd.DataFrame(sigma, index=assets, columns=assets),
            diagnostics={
                **estimate.diagnostics,
                "condition_number": float(np.linalg.cond(sigma)),
                "minimum_eigenvalue": float(np.linalg.eigvalsh(sigma).min()),
            },
        )
        return output.validate()

    # 中文说明：`estimate_factor_returns`：估计模型量或交易成本。
    def estimate_factor_returns(
        self,
        asset_returns: pd.DataFrame,                     # T,N
        exposure_history: pd.DataFrame,                  # T,N,F
        market_cap_history: pd.DataFrame | None = None,  # T,N
    ) -> FactorReturnEstimate:
        if not isinstance(exposure_history.index, pd.MultiIndex):
            raise ValueError("exposure_history must be indexed by (date, asset)")
        factor_rows: list[pd.Series] = []
        residual_rows: list[pd.Series] = []
        skipped = 0
        for date in asset_returns.index.intersection(
            exposure_history.index.get_level_values(0).unique()
        ):
            x = exposure_history.xs(date, level=0)
            y = asset_returns.loc[date].reindex(x.index)
            valid = x.notna().all(axis=1) & y.notna()
            if valid.sum() <= x.shape[1] + 2:
                skipped += 1
                continue
            xv = x.loc[valid].to_numpy(float)
            yv = y.loc[valid].to_numpy(float)
            if market_cap_history is None:
                weights = np.ones(len(yv))
            else:
                cap = market_cap_history.loc[date].reindex(x.index).loc[valid]
                weights = np.sqrt(cap.clip(lower=1.0).to_numpy(float))
                weights /= np.nanmedian(weights)
            xw = xv * weights[:, None]
            yw = yv * weights
            ridge = self.config.variance_floor * np.eye(xv.shape[1])
            beta = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
            residual = pd.Series(np.nan, index=asset_returns.columns, name=date)
            residual.loc[x.loc[valid].index] = yv - xv @ beta
            factor_rows.append(pd.Series(beta, index=x.columns, name=date))
            residual_rows.append(residual)
        factor_returns = pd.DataFrame(factor_rows).sort_index()
        residual_returns = pd.DataFrame(residual_rows).sort_index()
        return FactorReturnEstimate(
            factor_returns=factor_returns,
            residual_returns=residual_returns,
            diagnostics={
                "n_factor_return_dates": float(len(factor_returns)),
                "n_skipped_dates": float(skipped),
            },
        )

    # 中文说明：`_factor_covariance`：内部辅助步骤，不作为稳定公共接口。
    def _factor_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        clean = returns.dropna(how="any")
        if len(clean) < 2:
            raise ValueError("insufficient history to estimate factor covariance")
        values = clean.to_numpy(float)
        weights = exponential_weights(len(values), self.config.factor_halflife)
        centered = values - np.average(values, axis=0, weights=weights)
        covariance = (centered * weights[:, None]).T @ centered
        for lag in range(1, min(self.config.newey_west_lags, len(values) - 1) + 1):
            kernel = 1.0 - lag / (self.config.newey_west_lags + 1.0)
            cross = (centered[lag:] * weights[lag:, None]).T @ centered[:-lag]
            covariance += kernel * (cross + cross.T)
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - self.config.covariance_shrinkage) * covariance
            + self.config.covariance_shrinkage * diagonal
        )
        covariance *= self.config.annualization
        covariance = nearest_psd(covariance, self.config.variance_floor)
        return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)

    # 中文说明：`_specific_variance`：内部辅助步骤，不作为稳定公共接口。
    def _specific_variance(self, residuals: pd.DataFrame) -> pd.Series:
        values = residuals.to_numpy(float)
        weights = exponential_weights(len(values), self.config.specific_halflife)
        valid = np.isfinite(values)
        denominator = (valid * weights[:, None]).sum(axis=0)
        means = np.divide(
            np.nansum(values * weights[:, None], axis=0),
            denominator,
            out=np.zeros(values.shape[1]),
            where=denominator > 0,
        )
        variances = np.divide(
            np.nansum(((values - means) ** 2) * weights[:, None], axis=0),
            denominator,
            out=np.full(values.shape[1], np.nan),
            where=denominator > 0,
        )
        cross_median = np.nanmedian(variances)
        shrunk = (
            (1.0 - self.config.specific_shrinkage) * variances
            + self.config.specific_shrinkage * cross_median
        )
        return pd.Series(
            np.maximum(shrunk * self.config.annualization, self.config.variance_floor),
            index=residuals.columns,
            name="specific_variance",
        )


# 中文说明：`risk_attribution`：执行该名称对应的业务计算，并返回调用方所需结果。
def risk_attribution(
    weights: pd.Series,
    risk: RiskModelOutput,
) -> pd.DataFrame:
    """Euler decomposition of total variance into factor and specific components."""
    aligned = weights.reindex(risk.assets).fillna(0.0)
    w = aligned.to_numpy(float)
    x = risk.exposures.loc[risk.assets].fillna(0.0).to_numpy(float)
    factor_cov = risk.factor_covariance.to_numpy(float)
    factor_exposure = x.T @ w
    factor_marginal = factor_cov @ factor_exposure
    factor_contribution = factor_exposure * factor_marginal
    specific_contribution = (w**2) * risk.specific_variance.to_numpy(float)
    rows = {
        **{
            f"factor:{name}": value
            for name, value in zip(risk.factor_covariance.index, factor_contribution)
        },
        "specific": float(specific_contribution.sum()),
    }
    total = sum(rows.values())
    result = pd.DataFrame({"variance_contribution": pd.Series(rows)})
    result["share"] = result["variance_contribution"] / total if total > 0 else np.nan
    return result
