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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.context import MarketContext
    from v2.Monitor.utils import historical_percentile, safe_divide

PERIODS = (1, 5, 20)
DEFAULT_SERIES = {
    # 已从图中确认的 Wind EDB ID。
    "1Y国债收益率": {"source": "edb", "code": "M1001940"},
    "5Y国债收益率": {"source": "edb", "code": "M1001943"},
    "10Y国债收益率": {"source": "edb", "code": "M1001946"},
    "M2同比": {"source": "edb", "code": "M0001391"},
    "社融存量": {"source": "edb", "code": "M5525755"},
    "社融存量同比": {"source": "edb", "code": "M5525763"},
    "名义GDP": {"source": "edb", "code": "M5567876"},
    # 以下代码需按数据商最终确认的同口径序列替换。
    "DR007": {"source": "edb", "code": None, "search": "DR007"},
    "7D政策利率": {"source": "edb", "code": None, "search": "7天逆回购操作利率"},
    "Shibor3M": {"source": "edb", "code": None, "search": "Shibor:3个月"},
    "AA信用收益率": {"source": "edb", "code": None, "search": "中债中短期票据收益率曲线 AA"},
    "AAA信用收益率": {"source": "edb", "code": None, "search": "中债中短期票据收益率曲线 AAA"},
    "民企信用收益率": {"source": "edb", "code": None, "search": "民营企业信用债收益率"},
    "国企信用收益率": {"source": "edb", "code": None, "search": "国有企业信用债收益率"},
    "实际利率代理": None,
    "M2余额": None,
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
    """信用脉冲：过去12个月社融存量增量/GDP，再减3个月前值。"""
    tsf, gdp = _clean(tsf_stock), _clean(nominal_gdp)
    frame = pd.concat({"tsf": tsf, "gdp": gdp}, axis=1).dropna()
    if len(frame) < window + lag + 1:
        return pd.Series(dtype=float)
    ratio = frame["tsf"].diff(window).div(frame["gdp"])
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
        elif not spec:
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

    def snapshot(self, asof, history=756, periods=PERIODS, metrics=None):
        end = pd.Timestamp(asof)
        start = end - pd.Timedelta(days=max(history * 2, 3650))
        c = self.context
        raw = {name: self._series(name, start, end) for name in c.series_map}
        derived = {
            "DR007-7D政策利率": spread(raw["DR007"], raw["7D政策利率"]),
            "AA-AAA信用利差": spread(raw["AA信用收益率"], raw["AAA信用收益率"]),
            "民企-国企信用利差": spread(raw["民企信用收益率"], raw["国企信用收益率"]),
            "10Y-1Y国债期限利差": spread(raw["10Y国债收益率"], raw["1Y国债收益率"]),
            "M2同比": raw["M2同比"] if len(raw["M2同比"]) else yoy_series(raw["M2余额"]),
            "社融存量同比": yoy_series(raw["社融存量"]),
            "信用脉冲": credit_impulse(raw["社融存量"], raw["名义GDP"]),
        }
        series = {**raw, **derived}
        default_metrics = ("DR007", "DR007-7D政策利率", "Shibor3M", "AA-AAA信用利差",
                           "民企-国企信用利差", "10Y国债收益率", "10Y-1Y国债期限利差",
                           "M2同比", "社融存量同比", "信用脉冲")
        selected = tuple(metrics or default_metrics)
        rows = []
        for name in selected:
            x = series.get(name, pd.Series(dtype=float))
            if not len(x):
                continue
            for period in periods:
                value = level(x) if period == 1 else change(x, period)
                rows.append({"指标": name, "周期": period, "当前值/变化": value,
                             "历史分位": percentile(x, history) if period == 1 else rolling_change_percentile(x, period, history)})
        return pd.DataFrame(rows).set_index(["指标", "周期"]) if rows else pd.DataFrame()

    @staticmethod
    def table(result, decimals=4):
        return result.round(decimals)


if __name__ == "__main__":
    context = MacroContext()
    try:
        print(MacroMonitor(context).snapshot("2026-09-30").to_string())
    finally:
        context.close()
