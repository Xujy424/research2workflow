"""宏观环境指标。每个指标使用独立函数，取数代码和计算口径在函数中可见。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.Monitor"

from .context import MonitorContext


def _daily_series(context, values, start, end):
    """将已完成计算的序列对齐交易日，并在日频层面向前填充。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    dates = context.dates[(context.dates >= start) & (context.dates <= end)]
    if values.empty or dates.empty:
        return pd.Series(index=dates, dtype=float, name=values.name)
    index = values.index.union(dates).sort_values()
    result = values.reindex(index).ffill().reindex(dates)
    result.name = values.name
    return result


def _monthly_values(values):
    """整理月频观测，并先在月频索引上填补缺失月份。"""
    if values.empty:
        return pd.Series(dtype=float, name=values.name, index=pd.PeriodIndex([], freq="M"))
    monthly = values.groupby(values.index.to_period("M")).last()
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    return monthly.reindex(months).ffill()


def _monthly_to_daily(context, monthly, start, end):
    """月频计算完成后按月末日期对齐交易日，再向前填充。"""
    if not monthly.empty:
        monthly = monthly.copy()
        monthly.index = monthly.index.to_timestamp(how="end").normalize()
    return _daily_series(context, monthly, start, end)


def _spread(context, start, end, left_code, right_code, scale=100, name=None):
    """读取两条原始序列、分别对齐日频后计算利差。"""
    history_start = min(pd.Timestamp(start), context.dates[0]) if len(context.dates) else start
    left = context.read("edb", left_code, history_start, end, kind="series")
    right = context.read("edb", right_code, history_start, end, kind="series")
    left = _daily_series(context, left, start, end)
    right = _daily_series(context, right, start, end)
    return left.subtract(right).mul(scale).rename(name or f"{left_code}-{right_code}")


def dr007(context, start, end):
    """DR007日度序列，保留EDB百分数口径。"""
    history_start = min(pd.Timestamp(start), context.dates[0]) if len(context.dates) else start
    values = context.read("edb", "M1006337", history_start, end, kind="series")
    return _daily_series(context, values, start, end)


def policy_rate_spread_dr007(context, start, end):
    """DR007减7天逆回购政策利率，利差以bp表示。"""
    return _spread(
        context, start, end, "M1006337", "M0041371",
        name="DR007-7天逆回购利差_bp",
    )


def shibor3m(context, start, end):
    """3个月固定期限Shibor报价日序列，保留EDB百分数口径。"""
    history_start = min(pd.Timestamp(start), context.dates[0]) if len(context.dates) else start
    values = context.read("edb", "M0017142", history_start, end, kind="series")
    # Shibor 为正利率；将 EDB 中的零值/负值占位异常记为缺失，避免形成假跳点。
    values = _daily_series(context, values.where(values > 0), start, end)
    return values.rename("Shibor3M")


def shibor3m_change(context, start, end, lag=5):
    """3个月Shibor较 lag 个交易日前的变化，单位为bp。"""
    history_start = context.dates[0] if len(context.dates) else start
    values = shibor3m(context, history_start, end).diff(lag).mul(100)
    return values.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()].rename(
        f"Shibor3M_{lag}日变化_bp"
    )


def credit_spread_aa_aaa(context, start, end):
    """中短期票据AA减AAA信用利差。"""
    return _spread(
        context, start, end, "S0059717", "S0059738",
        name="中短票据AA-AAA利差_bp",
    )


def credit_spread_aa_cdb(context, start, end):
    """中短期票据AA减3年国开债收益率。"""
    return _spread(
        context, start, end, "S0059717", "M1004265",
        name="中短票据AA-3年国开债利差_bp",
    )


def bond_10y(context, start, end):
    """10年期国债收益率日度序列，保留EDB百分数口径。"""
    # Wind 个别日期以0占位；收益率非正值视为无效观测。
    history_start = min(pd.Timestamp(start), context.dates[0]) if len(context.dates) else start
    values = context.read("edb", "S0059749", history_start, end, kind="series")
    return _daily_series(context, values.where(values > 0), start, end)


def bond_10y_change(context, start, end, lag=20):
    """10年期国债收益率较 lag 个交易日前的变化，单位为bp。"""
    history_start = context.dates[0] if len(context.dates) else start
    values = bond_10y(context, history_start, end).diff(lag).mul(100)
    return values.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()].rename(
        f"10Y国债_{lag}日变化_bp"
    )


def term_spread(context, start, end):
    """10年减1年国债收益率期限利差。"""
    ten_year = bond_10y(context, start, end)
    history_start = min(pd.Timestamp(start), context.dates[0]) if len(context.dates) else start
    one_year = context.read("edb", "S0059744", history_start, end, kind="series")
    one_year = _daily_series(context, one_year.where(one_year > 0), start, end)
    return ten_year.subtract(one_year).mul(100).rename("10Y-1Y国债利差_bp")


def real_rate(context, start, end):
    """10年国债收益率减Wind一致预期通胀率的实际利率代理。"""
    inflation_raw = context.read(
        "edb", "M0061676", context.dates[0], end, kind="series"
    )
    expected_inflation = _monthly_to_daily(context, _monthly_values(inflation_raw), start, end)
    return bond_10y(context, start, end).subtract(expected_inflation).rename("实际利率代理")


def _monthly_yoy(context, start, end, yoy_code, balance_code):
    """优先取EDB同比；缺失时由月频余额计算同比，再对齐到交易日。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    # 多取前两个月，保证区间起点可沿用最近一条已发布观测。
    fetch_start = start - pd.DateOffset(months=2)
    values = _monthly_values(
        context.read("edb", yoy_code, fetch_start, end, kind="series")
    )
    if values.empty:
        balance = _monthly_values(
            context.read(
                "edb", balance_code, fetch_start - pd.DateOffset(years=1), end,
                kind="series",
            )
        )
        if not balance.empty:
            values = balance.pct_change(12) * 100
    # 先完成月频缺失填充和同比计算，再按EDB统计期末对齐日频。
    values.name = yoy_code
    return _monthly_to_daily(context, values, start, end)

def _yoy_change(series, months=3):
    """计算同比序列相对指定自然月前的变化，保留同比的百分点口径。"""
    lagged_dates = series.index - pd.DateOffset(months=months)
    lagged = series.reindex(lagged_dates, method="ffill")
    lagged.index = series.index
    return series - lagged

def m2_yoy(context, start, end):
    """M2同比，优先使用Wind同比指标，缺失时由M2余额计算。"""
    return _monthly_yoy(context, start, end, "M0001385", "M0001384").rename("M2同比")


def m2_yoy_change(context, start, end, months=3):
    """M2同比较指定自然月前的变化，默认3个月，单位为百分点。"""
    warm_start = pd.Timestamp(start) - pd.DateOffset(months=months + 1)
    values = _yoy_change(m2_yoy(context, warm_start, end), months)
    return values.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()].rename(
        f"M2同比_{months}月变化"
    )


def tsf_yoy(context, start, end):
    """社融存量同比，优先使用Wind同比指标，缺失时由社融余额计算。"""
    return _monthly_yoy(context, start, end, "M5525763", "M5525755").rename("社融存量同比")


def tsf_yoy_change(context, start, end, months=3):
    """社融存量同比较指定自然月前的变化，默认3个月，单位为百分点。"""
    warm_start = pd.Timestamp(start) - pd.DateOffset(months=months + 1)
    values = _yoy_change(tsf_yoy(context, warm_start, end), months)
    return values.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()].rename(
        f"社融存量同比_{months}月变化"
    )


def credit_impulse(context, start, end, window=12, lag=3):
    """过去12个月新增社融/过去12个月名义GDP的3个月变化，单位为百分点。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    data_start = start - pd.DateOffset(months=window + lag + 2)
    tsf = context.read("edb", "M5206730", data_start, end, kind="series")
    gdp = context.read("edb", "M5567876", data_start, end, kind="series")
    if tsf.empty or gdp.empty:
        return pd.Series(dtype=float, name="信用脉冲")

    tsf = _monthly_values(tsf)
    tsf_months = pd.period_range(tsf.index.min(), end.to_period("M"), freq="M")
    tsf = tsf.reindex(tsf_months)

    # EDB GDP 是季度流量：先补齐季度缺口，构造TTM后映射到月末。
    gdp = gdp.groupby(gdp.index.to_period("Q")).last()
    quarters = pd.period_range(gdp.index.min(), gdp.index.max(), freq="Q")
    gdp = gdp.reindex(quarters).ffill()
    gdp_ttm = gdp.rolling(4, min_periods=4).sum()
    gdp_ttm.index = gdp_ttm.index.asfreq("M", how="end")
    months = tsf_months
    gdp_ttm = gdp_ttm.reindex(months).ffill()
    if len(months) < window + lag:
        return pd.Series(dtype=float, name="信用脉冲")
    ratio = tsf.rolling(window, min_periods=window).sum().div(gdp_ttm)
    monthly = ratio - ratio.shift(lag)
    daily = _monthly_to_daily(context, monthly, start, end)
    return daily.mul(100).rename("信用脉冲")



MACRO_INDICATORS = {
    function.__name__: function
    for function in (
        dr007, policy_rate_spread_dr007, shibor3m, shibor3m_change,
        credit_spread_aa_aaa, credit_spread_aa_cdb, bond_10y,
        bond_10y_change, term_spread, real_rate, m2_yoy, tsf_yoy,
        m2_yoy_change, tsf_yoy_change, credit_impulse,
    )
}

if __name__ == "__main__":
    from v2.UpdateData.config import ROOT
    from v2.UpdateData.utils import ensure_memmap

    # 从有效日期轴首尾取数；每个输出文件与完整日期轴一一对齐。
    context = MonitorContext(root=ROOT)
    try:
        axis_dates = context.data.axis.full_dates
        valid_date = axis_dates[~np.isnat(axis_dates)]
        if valid_date.size == 0:
            raise ValueError("日期轴没有有效交易日")
        start, end = pd.Timestamp(valid_date[0]), pd.Timestamp(valid_date[-1])
        output_root = Path(ROOT) / "median_variables" / "monitor" / "macro"

        for name, calculate in MACRO_INDICATORS.items():
            values = calculate(context, start, end).reindex(context.dates)
            array = ensure_memmap(
                output_root / f"{name}.bin", (len(axis_dates), 1), dtype=np.float32
            )
            array[:] = np.nan
            array[:len(valid_date), 0] = values.to_numpy(dtype=np.float32)
            array.flush()
            del array
            print(f"[OK] {name}: {start.date()} - {end.date()}", flush=True)
    finally:
        context.close()
