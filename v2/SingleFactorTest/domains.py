"""作者：xujiayi | 创建日期：2026-11-11
分域表现：事前域构建、域内选股和回归暴露归因。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from . import maths as m
from .overall_cross_section import calculate


def universe(base):
    """股票池域；Params: base；Return: 名称→bool DataFrame。"""
    return {"universe": base.mask(), **{k: v for k, v in base.domain_masks.items() if k.startswith("universe:")}}


def industry(base):
    """行业域；Params: base；Return: 行业代码→当日mask。"""
    base.require("industry")
    x = base.fields["industry"]
    return {str(c): x.eq(c) for c in pd.unique(x.to_numpy().ravel()) if pd.notna(c)}


def index(base):
    """指数成分域；Params: base；Return: 名称→正权重mask。"""
    return {name: w.gt(0) for name, w in base.domain_weights.items()}


def barra(base):
    """Barra正负域；Params: base；Return: 风格名称±→mask，缺失值不入域。"""
    return {f"{name}{suffix}": base.domain_masks.get(f"barra:{name}{suffix}",
                                                    x.gt(0) if suffix == "+" else x.lt(0))
            for name, x in base.barra.items() for suffix in ("+", "-")}


def _tails(x):
    """截面20/80分位域；Params: x；Return: high/low字典。"""
    return {"high": x.ge(x.quantile(.8, axis=1), axis=0) & x.notna(),
            "low": x.le(x.quantile(.2, axis=1), axis=0) & x.notna()}


def volume(base):
    """过去20日平均成交量域；Params: base；Return: high/low mask。"""
    base.require("volume")
    return _tails(base.fields["volume"].where(base.mask()).rolling(20, min_periods=20).mean())


def vol(base):
    """过去20日收益波动域；Params: base；Return: high/low mask。"""
    return _tails(base.pct.where(base.mask()).rolling(20, min_periods=20).std())


def momentum(base):
    """过去20日复合动量域；Params: base；Return: high/low mask。"""
    x = (1 + base.pct).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    return _tails(x.where(base.mask()))


def regime(base):
    """沿用analyzer的60日均线±5%牛/熊/震荡域，无全样本阈值。

    Params: base；Return: 三种市场状态mask，最初不足20日不分类。
    """
    nav = m.nav(base.cal_benchmark())
    ma = nav.rolling(60, min_periods=20).mean()
    bull, bear = nav > ma * 1.05, nav < ma * .95
    states = {"bull": bull, "bear": bear, "sideways": ma.notna() & ~bull & ~bear}
    return {name: pd.DataFrame(np.broadcast_to(s.to_numpy()[:, None], base.pct.shape), base.dates, base.ticks)
            for name, s in states.items()}


BUILDERS = {"universe": universe, "industry": industry, "index": index, "barra": barra,
            "volume": volume, "vol": vol, "momentum": momentum, "regime": regime}


def domain_returns(base, domain="industry"):
    """域收益：前日域内等权；Params: base/domain类型；Return: DataFrame，并注册mask。"""
    masks = BUILDERS[domain](base)
    if not masks:
        raise ValueError(f"{domain}没有域数据，请注入domain_masks/domain_weights/barra")
    output = {}
    for name, mask in masks.items():
        key = f"{domain}:{name}"
        base.domain_masks[key] = mask & base.mask()
        w = m.normalize(base.domain_masks[key].astype(float))
        output[name] = m.weighted_returns(w.shift(1).fillna(0), base.pct).where(w.shift(1).sum(axis=1) > 0)
    result = pd.DataFrame(output)
    base.results[f"{domain}_returns"] = result
    return result


def table(base, years=5, domain="industry", construction=None, group_numbers=None):
    """域内IC/回归暴露/收益贡献和选股表现。

    Params: base/years:1或3或5/domain:八种类型/construction/group_numbers；Return: 域×指标表。
    beta为全市场因子组合超额对域超额收益的单变量OLS斜率；贡献=beta×域日均超额×252，
    域有重叠，贡献不可相加。Barra暴露为标准化因子组合对标准化风格的主动权重暴露。
    """
    if years not in (1, 3, 5) or domain not in BUILDERS:
        raise ValueError("years/domain无效")
    base.cal_labels((1,))
    start = base.dates[-1] - pd.DateOffset(years=years)
    dr = domain_returns(base, domain)
    market = base.cal_benchmark()
    overall = calculate(base, construction, group_numbers).excess_return
    market_mean = base.alpha.where(base.mask()).mean(axis=1)
    rows = {}
    for name in dr:
        key = f"{domain}:{name}"
        mask = base.domain_masks[key]
        # 分组只处理该域曾出现的股票；保留前一交易日用于收益对齐。
        positions = np.flatnonzero(base.dates >= start)
        begin = max(0, positions[0] - 1) if len(positions) else 0
        dates = base.dates[begin:]
        columns = mask.loc[dates].any(axis=0)
        view = copy.copy(base)
        view.dates, view.ticks = dates, base.ticks[columns]
        view.raw_alpha = base.raw_alpha.loc[dates, columns]
        view.alpha, view.pct = base.alpha.loc[dates, columns], base.pct.loc[dates, columns]
        view.fields = {k: x.loc[dates, columns] for k, x in base.fields.items()}
        view.domain_masks = {key: mask.loc[dates, columns]}
        view.results, view._group_cache_key = {}, None
        report = calculate(view, construction, group_numbers, key).loc[start:]
        row = m.performance(report.excess_return, report.turnover)
        row["rankIC"] = m.cross_ic(base.alpha.where(mask), base.labels[1], True).loc[start:].mean()
        x, y = (dr[name] - market).loc[start:], overall.loc[start:]
        joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(joined) > 2 and joined.x.var() > 0:
            beta = joined.x.cov(joined.y) / joined.x.var()
            row["regression_beta"] = beta
            row["return_contribution"] = beta * joined.x.mean() * 252
        else:
            row["regression_beta"], row["return_contribution"] = np.nan, np.nan
        row["exposure"] = (base.alpha.where(mask).mean(axis=1) - market_mean).loc[start:].mean()
        if domain == "barra":
            style = m.standardize(base.barra[name[:-1]]).loc[dates, columns]
            row["exposure"] = (view.active_target * style).sum(axis=1, min_count=1).loc[start:].mean()
        rows[name] = row
    result = pd.DataFrame(rows).T
    result.attrs["title"] = f"{years}年期{domain}域对比"
    base.results[f"domain_{years}_{domain}"] = result
    return result


def plot(base, years=5, domain="industry", construction=None, group_numbers=None):
    """域暴露柱图；Params: 同table；Return: Figure。"""
    result = table(base, years, domain, construction, group_numbers)
    fig, ax = plt.subplots(figsize=(12, 5))
    result.exposure.plot.bar(ax=ax)
    ax.set_title(result.attrs["title"])
    fig.tight_layout()
    return fig
