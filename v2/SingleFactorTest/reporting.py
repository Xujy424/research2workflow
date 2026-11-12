"""作者：xujiayi | 创建日期：2026-11-11
将单因子数值结果转换为可序列化的表格及ECharts配置，不生成Matplotlib图片。
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from . import maths as m, overall_cross_section as cross, distribution, ic_test, domains, capacity


LABELS = {
    "annual_return": "年化收益", "annual_vol": "年化波动", "sharpe": "夏普比率",
    "max_drawdown": "最大回撤", "calmar": "卡玛比率", "turnover": "日均换手",
    "max": "最大值", "min": "最小值", "max_avg": "截面最大值均值", "min_avg": "截面最小值均值",
    "skew_avg": "截面偏度均值", "kurt_avg": "截面峰度均值", "mean": "均值", "std": "标准差",
    "avg_ic": "平均IC", "ic_ir": "IC IR", "avg_pos_ic": "上涨子样本IC", "avg_neg_ic": "下跌子样本IC",
    "pos_icir": "上涨子样本IR", "neg_icir": "下跌子样本IR", "avg_rankic": "平均RankIC",
    "rank_icir": "RankIC IR", "count": "有效天数", "beta": "市场β", "BP": "账面市值比",
    "IC_IR": "组内IC IR", "crowding": "拥挤度代理", "rankIC": "域内RankIC",
    "regression_beta": "回归β", "return_contribution": "收益贡献", "exposure": "因子暴露",
    "fill_ratio": "金额成交率", "fill_ratio_p05": "成交率P05", "tracking_gap": "持仓偏离",
    "gross_exposure": "总敞口", "net_exposure": "净敞口", "cash_ratio": "现金比例",
    "st_remaining_max": "ST残余持仓(元)", "stock_cap_excess_max": "权重超限",
    "annual_commission": "年化佣金", "annual_stamp_tax": "年化印花税", "annual_impact": "年化冲击",
    "annual_borrow": "年化借券费", "capacity_pass": "容量达标",
}
PERCENT = {"annual_return", "annual_vol", "max_drawdown", "turnover", "return_contribution",
           "fill_ratio", "fill_ratio_p05", "tracking_gap", "gross_exposure", "net_exposure",
           "cash_ratio", "stock_cap_excess_max", "annual_commission", "annual_stamp_tax",
           "annual_impact", "annual_borrow"}


def scalar(value):
    """JSON安全标量；Params: 单元值；Return: 原生标量，缺失/无穷为None。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, int, np.number)):
        return float(value) if np.isfinite(value) else None
    if value is None or pd.isna(value):
        return None
    return str(value)


def table(frame, index_label="年份", monthly=False):
    """表格协议；Params: DataFrame/索引标题/月度标记；Return: JSON兼容dict。"""
    return {"type": "table", "indexLabel": index_label,
            "columns": [{"key": str(c), "label": f"{c}月" if monthly else LABELS.get(str(c), str(c)),
                         "format": "percent" if monthly or c in PERCENT else "number"} for c in frame.columns],
            "rows": [{"label": "整体" if str(i) == "Overall" else str(i),
                      "values": [scalar(x) for x in row]} for i, row in frame.iterrows()],
            "monthly": monthly}


def chart(frame, percent=False, bars=(), right=(), zoom=True, area=False):
    """ECharts折线/柱状配置；Params: 序列矩阵及样式；Return: 图表协议dict。"""
    dates = isinstance(frame.index, pd.DatetimeIndex)
    x = frame.index.strftime("%Y-%m-%d").tolist() if dates else [str(x) for x in frame.index]
    series = []
    for c in frame:
        item = {"name": str(c), "type": "bar" if c in bars else "line", "showSymbol": False,
                "data": [scalar(v * 100 if percent else v) for v in frame[c]],
                "yAxisIndex": 1 if c in right else 0, "connectNulls": False,
                "lineStyle": {"width": 1.7}, "emphasis": {"focus": "series"}}
        if area:
            item["areaStyle"] = {"opacity": .15}
        series.append(item)
    axis = {"type": "value", "scale": True, "splitLine": {"lineStyle": {"color": "#edf0f4"}},
            "axisLabel": {"formatter": "{value}%" if percent else "{value}"}}
    option = {"animation": False, "color": ["#246bfd", "#00a99a", "#f2a93b", "#a36bdf", "#ed6971", "#526880",
                                            "#72b5e8", "#78ad58", "#d68451", "#ca83af"],
              "tooltip": {"trigger": "axis", "confine": True},
              "legend": {"type": "scroll", "top": 0, "textStyle": {"color": "#66758a"}},
              "grid": {"left": 55, "right": 55 if right else 24, "top": 45, "bottom": 65 if zoom else 40},
              "xAxis": {"type": "category", "data": x, "boundaryGap": bool(bars),
                        "axisLine": {"lineStyle": {"color": "#dae0ea"}}, "axisLabel": {"color": "#7a8799"}},
              "yAxis": [axis, {**axis, "splitLine": {"show": False}}] if right else axis, "series": series}
    if zoom:
        option["dataZoom"] = [{"type": "inside"}, {"type": "slider", "height": 17, "bottom": 8, "borderColor": "#e4e9f0"}]
    return {"type": "chart", "option": option}


def window(base, years):
    """轻量只读视图；Params: 基类/年数；Return: 独立派生状态的FactorBase浅拷贝。"""
    view = copy.copy(base)
    view.report_start = max(base.dates[0], base.dates[-1] - pd.DateOffset(years=years))
    view.results = {}
    return view


def histogram(values, bins=60):
    """全量有限样本直方密度；Params: ndarray/bins；Return: Series。"""
    x = np.asarray(values).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return pd.Series(dtype=float)
    counts, edges = np.histogram(x, bins=bins, density=True)
    return pd.Series(counts, index=[f"{v:.3f}" for v in (edges[:-1] + edges[1:]) / 2])


def overall(base, options):
    """整体表现模块；Params: base/构建参数；Return: 三个图表协议。"""
    args = {k: options[k] for k in ("construction", "group_numbers", "domain")}
    data = cross.calculate(base, **args)
    yearly = pd.DataFrame({y: m.performance(data.excess_return.loc[mask], data.turnover.loc[mask])
                          for y, mask in m.periods(data.index).items()}).T
    curve = data.loc[base.report_start:].copy()
    curve["excess_return"] = m.nav(curve.excess_return) - 1
    curve = curve.rename(columns={"excess_return": "累计超额收益", "coverage": "覆盖率", "turnover": "换手率"})
    return {"overall_table": table(yearly), "overall_chart": chart(curve, True, right=("覆盖率", "换手率")),
            "monthly_table": table(cross.monthly_table(base, **args), monthly=True)}


def value_distribution(base, options):
    """因子分布模块；Params: base/processed及year；Return: 统计表、直方图。"""
    processed = options.get("processed", True)
    x = base.alpha if processed else base.raw_alpha
    year = str(options.get("year", "Overall"))
    if year != "Overall":
        x = x[x.index.year == int(year)]
    density = histogram(x.to_numpy()).to_frame("因子值密度")
    return {"distribution_table": table(distribution.value_table(base, processed)),
            "distribution_chart": chart(density, bars=("因子值密度",), zoom=False)}


def groups(base, options):
    """分组模块；Params: base/domain；Return: 表格、累计收益图。"""
    domain = options.get("domain", "universe")
    stats = distribution.group_table(base, domain)
    curves = m.nav(base.group_returns.drop(columns="Average").loc[base.report_start:]) - 1
    return {"groups_table": table(stats, "分组"), "groups_chart": chart(curves, True)}


def ic(base, options):
    """IC模块；Params: base/horizon/kind/years；Return: 四个图表协议。"""
    horizon, years = int(options.get("horizon", 1)), int(options.get("years", 5))
    base.cal_labels((horizon,))
    view = window(base, years)
    # 年表与整体均只统计选择窗口，避免控件改变后仍显示未筛选的Overall。
    selection = base.dates >= view.report_start
    view.dates = base.dates[selection]
    view.ics, view.rankics = base.ics.loc[selection], base.rankics.loc[selection]
    view.sign_ics = {h: frame.loc[selection] for h, frame in base.sign_ics.items()}
    stats = ic_test.table(view, horizon, options.get("kind", "IC"))
    current = pd.DataFrame({"IC": view.ics[horizon], "RankIC": view.rankics[horizon]})
    bins = np.linspace(-1, 1, 61)
    densities = pd.DataFrame(index=[f"{v:.3f}" for v in (bins[1:] + bins[:-1]) / 2])
    for name, s in current.items():
        x = s.dropna().to_numpy()
        densities[name] = np.histogram(x, bins=bins, density=True)[0] if len(x) else np.nan
    key = ("ic_decay", years)
    if key not in base.results:
        base.results[key] = ic_test.decay_table(window(base, years))
    decay = base.results[key].rename(columns={"IC_autocorrelation": "IC序列相关", "rankIC_autocorrelation": "RankIC序列相关",
                                             "rank_autocorrelation": "因子Rank自相关"})
    cumulative = ic_test.cumulative(view, years, horizon).rename(columns={"ic": "IC", "pos_ic": "上涨子样本IC",
                    "neg_ic": "下跌子样本IC", "rank_ic": "RankIC", "rank_pos_ic": "上涨子样本RankIC", "rank_neg_ic": "下跌子样本RankIC"})
    dist = chart(densities, zoom=False)
    dist["note"] = "；".join(f"{c} 均值 {current[c].mean():.4f} / 标准差 {current[c].std():.4f}" for c in current)
    return {"ic_table": table(stats), "ic_distribution": dist,
            "ic_cumulative": chart(cumulative), "ic_decay": chart(decay, zoom=False)}


def domain_report(base, options):
    """分域模块；Params: base/years/domain_type及构建参数；Return: 表格、柱图。"""
    dtype, years = options.get("domain_type", "industry"), int(options.get("years", 5))
    result = domains.table(base, years, dtype, options["construction"], options["group_numbers"])
    # 使用已有Monitor行业名称映射，不触发DataPool或数据库导入。
    if dtype == "industry":
        from v2.Monitor.config import INDUSTRY_MAP
        labels = {str(float(v)): k for k, v in INDUSTRY_MAP.items()}
        labels.update({str(v): k for k, v in INDUSTRY_MAP.items()})
        result = result.rename(index=labels)
    field = options.get("metric", "exposure")
    bars = result[[field]].rename(columns={field: LABELS.get(field, field)})
    return {"domains_table": table(result, "域分组"),
            "domains_chart": chart(bars, field in PERCENT, bars=tuple(bars.columns), zoom=False)}


def portfolio(base, options):
    """Excel约定的四张组合跟踪图；Params: base/组合参数；Return: 四图及结果缓存。"""
    p = base.cal_portfolio(**options)
    base.cal_benchmark(options["benchmark_method"], options["domain"], options["benchmark_index"])
    daily = p.daily.loc[base.report_start:]
    curves = pd.DataFrame({"组合（扣费前）": m.nav(daily.gross_return) - 1,
                           "组合（扣费后）": m.nav(daily.net_return) - 1,
                           "基准": m.nav(base.benchmark_return.reindex(daily.index)) - 1})
    previous_nav = p.daily.equity.shift(1).fillna(options["capital"]).reindex(daily.index)
    pnl = pd.DataFrame({"扣费前PnL（万元）": (daily.gross_return * previous_nav).cumsum() / 1e4,
                         "扣费后PnL（万元）": daily.pnl.cumsum() / 1e4})
    h = options["rebalance_days"]
    base.cal_labels((h,))
    rankic = base.rankics[h].shift(1).reindex(daily.index).where(daily.rebalance)
    ric = pd.DataFrame({"调仓日RankIC": rankic, "累计RankIC": rankic.fillna(0).cumsum()})
    net = m.nav(daily.net_return)
    drawdown = (net / net.cummax().clip(lower=1) - 1).to_frame("扣费后回撤")
    return {"portfolio_return": chart(curves, True), "portfolio_pnl": chart(pnl),
            "portfolio_ic": chart(ric, bars=("调仓日RankIC",), right=("累计RankIC",)),
            "portfolio_drawdown": chart(drawdown, True, area=True)}


def capacity_report(base, options):
    """容量模块；Params: base/资金档位及成交参数；Return: 表格、净值及成交率。"""
    options = dict(options)
    options.pop("capital", None)
    capitals = options.pop("capitals", (1e7, 5e7, 1e8, 5e8))
    stats = capacity.table(base, capitals=capitals, **options)
    curves = pd.DataFrame({f"{c/1e4:,.0f}万": m.nav(p.daily.net_return.loc[base.report_start:]) - 1
                           for c, p in base.capacity_results.items()})
    stats.index = [f"{c/1e4:,.0f}万" for c in stats.index]
    return {"capacity_table": table(stats, "初始资金"), "capacity_chart": chart(curves, True)}
