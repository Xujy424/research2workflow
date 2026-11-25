"""截面因子整体表现：超额收益、覆盖率、换手率及分年统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __package__:
    from ..base import maths as m
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base import maths as m


def basic_info(
        base, 
        construction="longshort", 
        num_groups=10,
        group_numbers=(10, 1), 
        domain="universe", 
        name='equal'
    ):
    """计算多空/多头/空头组合的日超额收益、因子覆盖率和换手率。
    Params: 
        base: FactorBase实例；
        construction: longshort/long/short；
        group_numbers: 多头和空头组号；
        domain/name: 分组域及名称。
    Return: 
        含 excess_return、coverage、turnover 的日频 DataFrame。
    """
    if not hasattr(base, "alpha"):
        base.process_alpha()
    if not hasattr(base, "benchmark"):
        base.cal_benchmark()

    base.cal_groups(
        num_groups=num_groups, domain=domain, name=name, horizon=1
    )
    name = "equal" if domain == "universe" else name
    benchmark_ret = base.benchmark["ret"][domain][name].shift(-2)
    group_returns = base.group_returns[1]
    if construction == "longshort":
        long_mask = base.group_codes.eq(group_numbers[0])
        short_mask = base.group_codes.eq(group_numbers[-1])
        long_count = long_mask.sum(axis=1).replace(0, np.nan)
        short_count = short_mask.sum(axis=1).replace(0, np.nan)
        excess = group_returns[f"G{group_numbers[0]}"] - group_returns[f"G{group_numbers[-1]}"]
        holds = long_mask.div(long_count, axis=0).fillna(0) - short_mask.div(short_count, axis=0).fillna(0)
    elif construction == "long":
        long_mask = base.group_codes.eq(group_numbers[0])
        long_count = long_mask.sum(axis=1).replace(0, np.nan)
        excess = group_returns[f"G{group_numbers[0]}"] - benchmark_ret
        holds = long_mask.div(long_count, axis=0).fillna(0)
    elif construction == "short":
        short_mask = base.group_codes.eq(group_numbers[-1])
        short_count = short_mask.sum(axis=1).replace(0, np.nan)
        excess = benchmark_ret - group_returns[f"G{group_numbers[-1]}"]
        holds = short_mask.div(short_count, axis=0).fillna(0)
    else:
        raise ValueError("construction 必须为 longshort、long 或 short")

    turnover = holds.sub(holds.shift(1).fillna(0)).abs().sum(axis=1) / 2

    coverage = np.sum(~np.isnan(base.alpha), axis=1)/np.sum(base.fields['tradable_t'],axis=1)

    result = pd.DataFrame({"excess_return": excess, "coverage": coverage, "turnover": turnover})
    return result


def plot_basic_info(basic_info, start):
    """绘制超额累积收益、因子覆盖率和因子换手率时序。
    """
    res = basic_info.loc[start:]

    fig, ax = plt.subplots(figsize=(12, 5))
    (1 + res.excess_return.fillna(0)).cumprod().sub(1).plot(ax=ax, label="超额累积收益")
    right = ax.twinx()
    res[["coverage", "turnover"]].plot(ax=right, alpha=.7)
    ax.legend(loc="upper left")
    right.legend(loc="upper right")
    fig.tight_layout()
    return fig


def table_yearly_performance(basic_info, start):
    """计算最近六个自然年及Overall的超额收益统计。

    Params: base: FactorBase实例；construction: 构建方式；group_numbers: 多空组号；domain/name: 分组域及名称。
    Return: 年份为行、指标为列的统计 DataFrame。
    """
    res = basic_info.loc[start:]
    periods = m.periods(res.index, years=5)
    result = pd.DataFrame({
        year: m.performance(res.excess_return.loc[mask], res.turnover.loc[mask])
        for year, mask in periods.items()
    }).T
    result["Overall"] = m.performance(res.excess_return, res.turnover)
    return result


def table_monthly_return(basic_info, start):
    """计算最近六个自然年逐月的超额累计收益。
    """
    returns = basic_info.loc[start:].excess_return
    end_year = returns.index.max().year
    returns = returns.loc[returns.index.year >= end_year - 5]
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda values: (1 + values).prod(min_count=1) - 1
    )
    result = monthly.unstack().reindex(
        index=range(end_year - 5, end_year + 1), columns=range(1, 13)
    )
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base.create_base import create_base

    base = create_base()

    result = basic_info(
        base, 
        num_groups=10, 
        group_numbers=(10,1),
        domain='universe', 
        construction='longshort',
        name=None,
    )
    fig = plot_basic_info(result, base.report_start)
    yearly = table_yearly_performance(result, base.report_start)
    monthly = table_monthly_return(result, base.report_start)
    print("Yearly performance:")
    percent_columns = ["annual_return", "annual_vol", "max_drawdown", "turnover"]
    print(yearly.to_string(formatters={
        column: lambda value: f"{value:.2%}" for column in percent_columns
    } | {"sharpe": lambda value: f"{value:.2f}"}), flush=True)
    print("Monthly excess return:")
    print(monthly.to_string(formatters={
        column: lambda value: f"{value:.2%}" for column in monthly.columns
    }), flush=True)
    plt.show()
