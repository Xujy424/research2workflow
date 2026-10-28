"""宏观环境监控：Context 取数 -> 独立指标函数 -> 汇总表。

数据代码不在本模块写死。通过 ``series_map`` 注入 EDB/WSD 代码，或直接注入
``series_reader(name, start, end)``，便于按数据商口径替换。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .context import MarketContext
    from .utils import historical_percentile, safe_divide
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.Monitor.dailyupdate.context import MarketContext
    from v2.Monitor.dailyupdate.utils import historical_percentile, safe_divide

PERIODS = (1, 5, 20)
RATE_METRICS = {
    "DR007", "DR007-政策利率利差", "Shibor3M", "中短期票据AA-AAA信用利差",
    "中短期票据AA-国开债信用利差", "10Y国债收益率", "国债期限利差(10Y-1Y)",
    "10Y实际利率代理(Wind一致预期)",
}
SPREAD_METRICS = {
    "DR007-政策利率利差", "中短期票据AA-AAA信用利差",
    "中短期票据AA-国开债信用利差", "国债期限利差(10Y-1Y)",
}
DEFAULT_SERIES = {
    # 2026-10-21 Wind EDB 指标表确认的代码。
    "7D政策利率": {"source": "edb", "code": "M0041653"},
    "DR007": {"source": "edb", "code": "M1006337"},
    "Shibor3M": {"source": "edb", "code": "M0017142"},
    "AAA信用收益率": {"source": "edb", "code": "S0059738"},
    "AA信用收益率": {"source": "edb", "code": "S0059717"},
    "国开3Y收益率": {"source": "edb", "code": "M1004265"},
    "10Y国债收益率": {"source": "edb", "code": "S0059749"},
    "1Y国债收益率": {"source": "edb", "code": "S0059744"},
    "M2同比": {"source": "edb", "code": "M0001385"},
    "M2余额": {"source": "edb", "code": "M0001384"},
    "社融存量": {"source": "edb", "code": "M5525755"},
    "社融存量同比": {"source": "edb", "code": "M5525763"},
    "新增社融": {"source": "edb", "code": "M5206730"},
    "名义GDP": {"source": "edb", "code": "M5567876"},
    # Wind 一致预期代码需要按实际使用的数据商序列补入；支持通过 series_map 覆盖。
    "预期通胀_Wind一致预期": {"source": "edb", "code": "M0061676"},
}


def _clean(series):
    if series is None:
        return pd.Series(dtype=float)
    x = pd.Series(series, dtype=float)
    x.index = pd.to_datetime(x.index).normalize()
    return x[~x.index.duplicated(keep="last")].sort_index()


def level(series):
    """最新有效水平。"""
    x = _clean(series).dropna()
    return float(x.iloc[-1]) if len(x) else np.nan


def change(series, periods=1):
    """当前值减去 periods 个观测日前值。"""
    x = _clean(series).dropna()
    return float(x.iloc[-1] - x.iloc[-1 - periods]) if len(x) > periods else np.nan


def change_months(series, months=3):
    """月度同比序列的月度变化：YoY_t - YoY_(t-months)。"""
    return change(series, months)


def yoy(series, months=12):
    """月度余额同比，按可用观测序列计算。"""
    x = _clean(series).dropna()
    return float(safe_divide(x.iloc[-1], x.iloc[-1 - months]) - 1) if len(x) > months else np.nan


def yoy_series(series, months=12):
    """返回完整的月度同比序列，便于历史分位计算。"""
    x = _clean(series).dropna()
    return x.div(x.shift(months)).sub(1).dropna()


def spread(left, right):
    """两条同日序列的利差。"""
    return _clean(left).subtract(_clean(right), fill_value=np.nan).dropna()


def aligned_spread(left, right):
    """按日期对齐两条序列，低频序列向前填充到高频序列日期。"""
    lhs, rhs = _clean(left), _clean(right)
    if lhs.empty or rhs.empty:
        return pd.Series(dtype=float)
    index = lhs.index.union(rhs.index).sort_values()
    # 只向后填充已发布值，避免将未来数据回填到过去造成前视偏差。
    lhs = lhs.reindex(index).ffill()
    rhs = rhs.reindex(index).ffill()
    return lhs.subtract(rhs).dropna()


def percentile(series, history=756):
    """最新值在历史有效样本中的分位。"""
    return historical_percentile(_clean(series).dropna().iloc[-history:].to_numpy())


def rolling_change_percentile(series, periods=20, history=756):
    """变化值的历史分位。"""
    x = _clean(series).dropna()
    if len(x) <= periods:
        return np.nan
    return percentile(x.diff(periods).dropna(), history)


def credit_impulse(tsf_stock, nominal_gdp, window=12, lag=3):
    """信用脉冲：过去12个月新增社融/GDP，再减3个月前值。

    GDP 为季度序列时按月线性插值到社融月份，仅用于形成同比尺度；
    不使用未来季度值。
    """
    tsf, gdp = _clean(tsf_stock), _clean(nominal_gdp)
    if tsf.empty or gdp.empty:
        return pd.Series(dtype=float)
    # EDB月度值可能标记在月末，先统一到月初，避免直接 reindex 全部丢失。
    tsf_monthly = tsf.groupby(tsf.index.to_period("M")).last()
    tsf_monthly.index = tsf_monthly.index.to_timestamp(how="start")
    gdp_monthly = gdp.groupby(gdp.index.to_period("M")).last()
    gdp_monthly.index = gdp_monthly.index.to_timestamp(how="start")
    month_index = pd.date_range(tsf_monthly.index.min(), tsf_monthly.index.max(), freq="MS")
    tsf = tsf_monthly.reindex(month_index).ffill()
    gdp_index = gdp_monthly.index.union(month_index).sort_values()
    gdp = gdp_monthly.reindex(gdp_index).interpolate(method="time").reindex(month_index).ffill()
    frame = pd.concat({"tsf": tsf, "gdp": gdp}, axis=1).dropna()
    if len(frame) < window + lag + 1:
        return pd.Series(dtype=float)
    ratio = frame["tsf"].rolling(window).sum().div(frame["gdp"])
    return ratio - ratio.shift(lag)


class MacroContext(MarketContext):
    """宏观序列取数上下文。``series_reader`` 可返回 DatetimeIndex Series。"""

    def __init__(self, *args, series_map=None, series_reader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.series_map = {**DEFAULT_SERIES, **(series_map or {})}
        self.series_reader = series_reader
        self._series_cache = {}

    def series(self, name, start, end):
        key = (name, pd.Timestamp(start), pd.Timestamp(end))
        if key in self._series_cache:
            return self._series_cache[key]
        spec = self.series_map.get(name)
        if self.series_reader is not None:
            value = self.series_reader(name, start, end)
        elif not spec or not spec.get("code"):
            value = pd.Series(dtype=float)
        elif spec.get("source", "edb") == "wsd":
            value = self.wsd(spec["code"], spec.get("field", "close"), start, end)
        else:
            value = self.edb(spec["code"], start, end)
        value = _clean(value).loc[pd.Timestamp(start):pd.Timestamp(end)]
        self._series_cache[key] = value
        return value

    def close(self):
        self._series_cache.clear()
        super().close()


class MacroMonitor:
    """计算宏观指标当前值、变化和历史分位。"""

    def __init__(self, context):
        self.context = context

    def _series(self, name, start, end):
        return self.context.series(name, start, end)

    def snapshot(self, asof, history=252, periods=PERIODS, metrics=None):
        end = pd.Timestamp(asof)
        start = end - pd.DateOffset(years=3)
        c = self.context
        raw = {name: self._series(name, start, end) for name in c.series_map}
        # 所有原始频率先统一到交易日：将原始日期与交易日并集后前向填充，
        # 再取交易日索引。这样月频、季频和年频数据在派生指标计算前已统一。
        trading_index = pd.DatetimeIndex(c.dates)
        trading_index = trading_index[(trading_index >= start.normalize()) & (trading_index <= end.normalize())]
        if len(trading_index) == 0:
            trading_index = pd.date_range(start.normalize(), end.normalize(), freq="B")
        raw = {
            name: _clean(value).reindex(_clean(value).index.union(trading_index).sort_values()).ffill()
            .reindex(trading_index)
            for name, value in raw.items()
        }
        derived = {
            "DR007-政策利率利差": aligned_spread(raw["DR007"], raw["7D政策利率"]),
            "中短期票据AA-AAA信用利差": aligned_spread(raw["AA信用收益率"], raw["AAA信用收益率"]),
            "中短期票据AA-国开债信用利差": aligned_spread(raw["AA信用收益率"], raw["国开3Y收益率"]),
            "国债期限利差(10Y-1Y)": aligned_spread(raw["10Y国债收益率"], raw["1Y国债收益率"]),
            "M2同比": raw["M2同比"] if len(raw["M2同比"]) else yoy_series(raw["M2余额"]),
            "社融存量同比": raw["社融存量同比"] if len(raw["社融存量同比"]) else yoy_series(raw["社融存量"]),
            "信用脉冲": credit_impulse(raw["新增社融"], raw["名义GDP"])
                .reindex(trading_index).ffill(),
            "10Y实际利率代理(Wind一致预期)": aligned_spread(raw["10Y国债收益率"], raw["预期通胀_Wind一致预期"]),
        }
        series = {**raw, **derived}
        default_metrics = ("DR007", "DR007-政策利率利差", "Shibor3M", "中短期票据AA-AAA信用利差",
                           "中短期票据AA-国开债信用利差", "10Y国债收益率", "国债期限利差(10Y-1Y)",
                           "10Y实际利率代理(Wind一致预期)",
                           "M2同比", "社融存量同比", "信用脉冲")
        selected = tuple(metrics or default_metrics)
        rows = []
        for name in selected:
            x = series.get(name, pd.Series(dtype=float))
            value = level(x) if len(x) else np.nan
            if name in RATE_METRICS:
                unit = "bp" if name in SPREAD_METRICS else "%"
            elif name in {"M2同比", "社融存量同比"}:
                unit = "%"
            elif name == "信用脉冲":
                unit = "%GDP"
            else:
                unit = "比例"
            for label, lag in (("当前值", 0), ("5日变化", 5), ("20日变化", 20)):
                raw_value = value if lag == 0 else change(x, lag)
                hist = percentile(x, history) if lag == 0 else rolling_change_percentile(x, lag, history)
                display_value = raw_value
                if name in SPREAD_METRICS and np.isfinite(display_value):
                    display_value *= 100.0
                elif name in RATE_METRICS and lag > 0 and np.isfinite(display_value):
                    display_value *= 100.0
                elif name == "信用脉冲" and np.isfinite(display_value):
                    display_value *= 100.0
                display_unit = unit if lag == 0 else ("bp" if name in RATE_METRICS else unit)
                rows.append({"指标": name, "周期": label, "数值": display_value,
                             "单位": display_unit, "历史分位": hist})
        return pd.DataFrame(rows).set_index(["指标", "周期"]) if rows else pd.DataFrame()

    @staticmethod
    def table(result, decimals=6):
        return result.round(decimals)


if __name__ == "__main__":
    context = MacroContext()
    try:
        result = MacroMonitor(context).snapshot("2026-09-30")
        print(result.to_string(float_format=lambda value: f"{value:.6f}"))
    finally:
        context.close()





