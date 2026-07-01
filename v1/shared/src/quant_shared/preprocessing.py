"""Cross-sectional factor cleaning, standardisation, and neutralisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PreprocessConfig


# 中文说明：定义 `CrossSectionalPreprocessor`，封装本模块对应的数据、配置与行为。
class CrossSectionalPreprocessor:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    # 中文说明：`transform`：转换输入数据并保持索引对齐。
    def transform(
        self,
        factors: pd.DataFrame,
        exposures: pd.DataFrame | None = None,
        market_caps: pd.Series | None = None,
    ) -> pd.DataFrame:
        cleaned = factors.replace([np.inf, -np.inf], np.nan).astype(float)
        result = cleaned.groupby(level=0, group_keys=False, sort=False).apply(
            self._clean_cross_section
        )
        if self.config.neutralize and (exposures is not None or market_caps is not None):
            result = self._neutralize_panel(result, exposures, market_caps)
            result = result.groupby(level=0, group_keys=False, sort=False).apply(
                self._standardize_cross_section
            )
        return result.sort_index()

    # 中文说明：`_clean_cross_section`：内部辅助步骤，不作为稳定公共接口。
    def _clean_cross_section(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._standardize_cross_section(self._winsorize_cross_section(frame))

    # 中文说明：`_winsorize_cross_section`：内部辅助步骤，不作为稳定公共接口。
    def _winsorize_cross_section(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.config.winsor_method == "quantile":
            lower = frame.quantile(0.01)
            upper = frame.quantile(0.99)
        elif self.config.winsor_method == "mad":
            median = frame.median()
            mad = (frame - median).abs().median()
            robust_sigma = 1.4826 * mad.replace(0.0, np.nan)
            lower = median - self.config.winsor_limit * robust_sigma
            upper = median + self.config.winsor_limit * robust_sigma
        else:
            raise ValueError(f"unsupported winsor method: {self.config.winsor_method}")
        return frame.clip(lower=lower, upper=upper, axis=1)

    # 中文说明：`_standardize_cross_section`：内部辅助步骤，不作为稳定公共接口。
    def _standardize_cross_section(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.config.standardize == "rank":
            ranked = frame.rank(pct=True, method="average")
            return ranked.sub(0.5).mul(np.sqrt(12.0))
        if self.config.standardize != "zscore":
            raise ValueError(f"unsupported standardization: {self.config.standardize}")
        mean = frame.mean()
        std = frame.std(ddof=0).replace(0.0, np.nan)
        return frame.sub(mean).div(std)

    # 中文说明：`_neutralize_panel`：内部辅助步骤，不作为稳定公共接口。
    def _neutralize_panel(
        self,
        factors: pd.DataFrame,
        exposures: pd.DataFrame | None,
        market_caps: pd.Series | None,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for date, y_frame in factors.groupby(level=0, sort=False):
            design_parts: list[pd.DataFrame] = []
            if exposures is not None:
                design_parts.append(exposures.xs(date, level=0).astype(float))
            if market_caps is not None:
                cap = market_caps.xs(date, level=0).clip(lower=1.0)
                design_parts.append(np.log(cap).rename("log_market_cap").to_frame())
            design = pd.concat(design_parts, axis=1)
            y = y_frame.droplevel(0)
            common = y.index.intersection(design.index)
            residuals = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
            if len(common) >= self.config.min_observations:
                x = design.loc[common].replace([np.inf, -np.inf], np.nan)
                for column in y.columns:
                    valid = x.notna().all(axis=1) & y.loc[common, column].notna()
                    if valid.sum() < max(self.config.min_observations, x.shape[1] + 2):
                        continue
                    xv = x.loc[valid].to_numpy(float)
                    yv = y.loc[common[valid], column].to_numpy(float)
                    xv = np.column_stack([np.ones(len(xv)), xv])
                    penalty = self.config.ridge * np.eye(xv.shape[1])
                    penalty[0, 0] = 0.0
                    beta = np.linalg.solve(xv.T @ xv + penalty, xv.T @ yv)
                    residuals.loc[common[valid], column] = yv - xv @ beta
            residuals.index = pd.MultiIndex.from_product(
                [[date], residuals.index], names=factors.index.names
            )
            parts.append(residuals)
        return pd.concat(parts).reindex(factors.index)

