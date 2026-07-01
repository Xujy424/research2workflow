"""Robustness, out-of-sample, regime, and implementation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_shared.contracts import FactorResearchReport, PanelData
from .research import FactorResearchEngine


# 中文说明：定义 `RobustnessReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class RobustnessReport:
    perturbation_summary: pd.DataFrame
    lag_sensitivity: pd.DataFrame
    subperiod_summary: pd.DataFrame
    regime_summary: pd.DataFrame
    diagnostics: Mapping[str, object]


# 中文说明：定义 `FactorRobustnessValidator`，封装本模块对应的数据、配置与行为。
class FactorRobustnessValidator:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, research_engine: FactorResearchEngine | None = None) -> None:
        self.research_engine = research_engine or FactorResearchEngine()

    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(
        self,
        data: PanelData,
        perturbations: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
        lags: Sequence[int] = (0, 1, 2),
        regimes: pd.Series | None = None,
    ) -> RobustnessReport:
        perturbations = perturbations or {
            "base": lambda frame: frame,
            "rank": self._rank_transform,
            "clip_2_5pct": lambda frame: frame.groupby(level=0, group_keys=False).apply(
                lambda cross: cross.clip(
                    cross.quantile(0.025), cross.quantile(0.975), axis=1
                )
            ),
        }
        perturbation_rows: list[pd.DataFrame] = []
        for name, transform in perturbations.items():
            transformed = transform(data.factors.copy()).reindex(data.factors.index)
            report = self.research_engine.analyze(
                PanelData(
                    transformed,
                    data.forward_returns,
                    data.exposures,
                    data.market_caps,
                    data.tradable,
                    data.metadata,
                )
            )
            frame = report.summary[["ic_mean", "icir", "long_short_sharpe"]].copy()
            frame["scenario"] = name
            perturbation_rows.append(frame.reset_index(names="factor"))
        lag_rows: list[pd.DataFrame] = []
        for lag in lags:
            lagged = data.factors.groupby(level=1).shift(lag)
            ic = self.research_engine.information_coefficients(
                lagged, data.forward_returns
            )
            lag_rows.append(
                pd.DataFrame(
                    {
                        "factor": ic.columns,
                        "lag": lag,
                        "ic_mean": ic.mean().to_numpy(),
                        "icir": (ic.mean() / ic.std(ddof=1)).to_numpy(),
                    }
                )
            )
        subperiod = self._subperiod_analysis(data)
        regime_summary = self._regime_analysis(data, regimes)
        return RobustnessReport(
            perturbation_summary=pd.concat(perturbation_rows, ignore_index=True),
            lag_sensitivity=pd.concat(lag_rows, ignore_index=True),
            subperiod_summary=subperiod,
            regime_summary=regime_summary,
            diagnostics={
                "parameter_plateau_ratio": self._plateau_ratio(perturbation_rows),
                "tested_lags": list(lags),
            },
        )

    # 中文说明：`_rank_transform`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _rank_transform(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(level=0, group_keys=False).rank(pct=True)

    # 中文说明：`_subperiod_analysis`：内部辅助步骤，不作为稳定公共接口。
    def _subperiod_analysis(self, data: PanelData) -> pd.DataFrame:
        dates = data.dates
        splits = np.array_split(dates, min(4, len(dates)))
        rows: list[pd.DataFrame] = []
        for position, period in enumerate(splits):
            if len(period) == 0:
                continue
            mask = data.factors.index.get_level_values(0).isin(period)
            ic = self.research_engine.information_coefficients(
                data.factors.loc[mask], data.forward_returns.loc[mask]
            )
            rows.append(
                pd.DataFrame(
                    {
                        "factor": ic.columns,
                        "subperiod": position + 1,
                        "start": period[0],
                        "end": period[-1],
                        "ic_mean": ic.mean().to_numpy(),
                        "ic_positive_ratio": (ic > 0).mean().to_numpy(),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    # 中文说明：`_regime_analysis`：内部辅助步骤，不作为稳定公共接口。
    def _regime_analysis(
        self,
        data: PanelData,
        regimes: pd.Series | None,
    ) -> pd.DataFrame:
        if regimes is None:
            return pd.DataFrame()
        ic = self.research_engine.information_coefficients(
            data.factors, data.forward_returns
        )
        aligned = regimes.reindex(ic.index)
        rows: list[dict[str, object]] = []
        for regime in aligned.dropna().unique():
            mask = aligned == regime
            for factor in ic.columns:
                values = ic.loc[mask, factor].dropna()
                rows.append(
                    {
                        "regime": regime,
                        "factor": factor,
                        "ic_mean": values.mean(),
                        "icir": values.mean() / values.std(ddof=1)
                        if values.std(ddof=1) > 0
                        else np.nan,
                        "observations": len(values),
                    }
                )
        return pd.DataFrame(rows)

    # 中文说明：`_plateau_ratio`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _plateau_ratio(frames: list[pd.DataFrame]) -> pd.Series:
        values = pd.concat(frames, ignore_index=True)
        if "factor" not in values.columns:
            values = values.reset_index(names="factor")
        sign_consistency = values.groupby("factor")["ic_mean"].apply(
            lambda series: max((series > 0).mean(), (series < 0).mean())
        )
        return sign_consistency


# 中文说明：定义 `WalkForwardSplitter`，封装本模块对应的数据、配置与行为。
class WalkForwardSplitter:
    """Purged expanding/rolling time splits for overlapping labels."""

    # 中文说明：`split`：执行该名称对应的业务计算，并返回调用方所需结果。
    def split(
        self,
        dates: pd.Index,
        train_size: int,
        test_size: int,
        purge: int = 0,
        embargo: int = 0,
        expanding: bool = False,
    ) -> list[tuple[pd.Index, pd.Index]]:
        ordered = pd.Index(dates).sort_values().unique()
        splits: list[tuple[pd.Index, pd.Index]] = []
        test_start = train_size + purge
        while test_start + test_size <= len(ordered):
            train_end = test_start - purge
            train_start = 0 if expanding else max(0, train_end - train_size)
            train = ordered[train_start:train_end]
            test = ordered[test_start : test_start + test_size]
            splits.append((train, test))
            test_start += test_size + embargo
        return splits


# 中文说明：`incremental_value`：执行该名称对应的业务计算，并返回调用方所需结果。
def incremental_value(
    base_signal: pd.Series,
    candidate_signal: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Measure candidate residual IC after removing the existing signal."""
    rows: dict[pd.Timestamp, float] = {}
    for date in base_signal.index.get_level_values(0).unique():
        base = base_signal.xs(date, level=0)
        candidate = candidate_signal.xs(date, level=0)
        future = returns.xs(date, level=0)
        valid = base.notna() & candidate.notna() & future.notna()
        if valid.sum() < 10:
            rows[pd.Timestamp(date)] = np.nan
            continue
        x = np.column_stack([np.ones(valid.sum()), base.loc[valid].to_numpy(float)])
        y = candidate.loc[valid].to_numpy(float)
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        rows[pd.Timestamp(date)] = float(
            spearmanr(residual, future.loc[valid].to_numpy(float)).statistic
        )
    return pd.Series(rows, name="incremental_rank_ic")
