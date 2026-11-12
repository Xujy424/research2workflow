"""作者：xujiayi | 创建日期：2026-11-11
整体表现（截面）：组内等权的研究超额收益、覆盖率和换手。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import maths as m


def calculate(base, construction=None, group_numbers=None, domain="universe"):
    """统一整体研究口径；Params: 基类/构建方式/组号/域；Return: 日频DataFrame。

    多头：多头组-域内等权；空头：域内等权-空头组；多空：多头组-空头组。
    多空为每腿100%的收益差，与交易组合默认总敞口100%有明确区别。
    """
    construction = construction or base.construction
    base.cal_target_holding(construction, group_numbers, "equal", domain)
    long = m.weighted_returns(base.long_target.shift(1).fillna(0), base.pct)
    short = m.weighted_returns(base.short_target.shift(1).fillna(0), base.pct)
    benchmark = base.cal_benchmark(domain=domain)
    excess = long - short if construction == "longshort" else long - benchmark if construction == "long" else benchmark - short
    weights = base.long_target - base.short_target
    previous = weights.shift(1).fillna(0)
    # 漂移后的旧权重与当日目标比较，日期移至实际收益日。
    drift = previous * (1 + base.pct)
    gross_ret = m.weighted_returns(previous, base.pct)
    drift = drift.div(1 + gross_ret, axis=0)
    turnover = weights.sub(drift).abs().sum(axis=1).div(2).shift(1)
    pool = base.mask(domain)
    coverage = base.raw_alpha.where(pool).count(axis=1) / pool.sum(axis=1).replace(0, np.nan)
    valid = (base.long_target.sum(axis=1).gt(0) if construction == "long" else
             base.short_target.sum(axis=1).gt(0) if construction == "short" else
             base.long_target.sum(axis=1).gt(0) & base.short_target.sum(axis=1).gt(0))
    excess = excess.where(valid.shift(1, fill_value=False))
    result = pd.DataFrame({"excess_return": excess, "coverage": coverage, "turnover": turnover})
    base.results["overall"] = result
    base.results["overall_spec"] = (construction, group_numbers, domain)
    return result


def plot(base, construction=None, group_numbers=None, domain="universe"):
    """画超额累计收益/覆盖率/换手；Params: 同calculate；Return: Figure。"""
    data = calculate(base, construction, group_numbers, domain).loc[base.report_start:]
    fig, ax = plt.subplots(figsize=(12, 5))
    (m.nav(data.excess_return) - 1).plot(ax=ax, label="Excess cumulative return")
    right = ax.twinx()
    data[["coverage", "turnover"]].plot(ax=right, alpha=.5)
    ax.legend(loc="upper left")
    right.legend(loc="upper right")
    fig.tight_layout()
    return fig


def yearly_table(base, construction=None, group_numbers=None, domain="universe"):
    """逐自然年及Overall表现；Params: 同calculate；Return: DataFrame。"""
    data = calculate(base, construction, group_numbers, domain)
    table = pd.DataFrame({y: m.performance(data.excess_return.loc[mask], data.turnover.loc[mask])
                          for y, mask in m.periods(data.index).items()}).T
    base.results["overall_yearly"] = table
    return table


def monthly_table(base, construction=None, group_numbers=None, domain="universe"):
    """各年月复合超额收益；Params: 同calculate；Return: 年份×1-12月，未来/缺失月为空。"""
    r = calculate(base, construction, group_numbers, domain).excess_return
    r = r.loc[str(r.index[-1].year - 5):]
    rows = []
    for (year, month), values in r.groupby([r.index.year, r.index.month]):
        rows.append((year, month, (1 + values).prod(min_count=1) - 1))
    result = pd.DataFrame(rows, columns=["year", "month", "return"]).pivot(index="year", columns="month", values="return")
    return result.reindex(index=range(r.index[-1].year - 5, r.index[-1].year + 1), columns=range(1, 13))
