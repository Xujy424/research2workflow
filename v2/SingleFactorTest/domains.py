"""浣滆€咃細xujiayi | 鍒涘缓鏃ユ湡锛?026-11-11
鍒嗗煙琛ㄧ幇锛氫簨鍓嶅煙鏋勫缓銆佸煙鍐呴€夎偂鍜屽洖褰掓毚闇插綊鍥犮€?"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from . import maths as m
from .overall_cross_section import calculate


def universe(base):
    """鑲＄エ姹犲煙锛汸arams: base锛汻eturn: 鍚嶇О鈫抌ool DataFrame銆?""
    return {"universe": base.cal_mask(), **{k: v for k, v in base.domain_masks.items() if k.startswith("universe:")}}


def industry(base):
    """琛屼笟鍩燂紱Params: base锛汻eturn: 琛屼笟浠ｇ爜鈫掑綋鏃ask銆?""
    base.require("industry")
    x = base.fields["industry"]
    return {str(c): x.eq(c) for c in pd.unique(x.to_numpy().ravel()) if pd.notna(c)}


def index(base):
    """鎸囨暟鎴愬垎鍩燂紱Params: base锛汻eturn: 鍚嶇О鈫掓鏉冮噸mask銆?""
    return {name: w.gt(0) for name, w in base.domain_weights.items()}


def barra(base):
    """Barra姝ｈ礋鍩燂紱Params: base锛汻eturn: 椋庢牸鍚嶇О卤鈫抦ask锛岀己澶卞€间笉鍏ュ煙銆?""
    return {f"{name}{suffix}": base.domain_masks.get(f"barra:{name}{suffix}",
                                                    x.gt(0) if suffix == "+" else x.lt(0))
            for name, x in base.barra.items() for suffix in ("+", "-")}


def _tails(x):
    """鎴潰20/80鍒嗕綅鍩燂紱Params: x锛汻eturn: high/low瀛楀吀銆?""
    return {"high": x.ge(x.quantile(.8, axis=1), axis=0) & x.notna(),
            "low": x.le(x.quantile(.2, axis=1), axis=0) & x.notna()}


def volume(base):
    """杩囧幓20鏃ュ钩鍧囨垚浜ら噺鍩燂紱Params: base锛汻eturn: high/low mask銆?""
    base.require("volume")
    return _tails(base.fields["volume"].where(base.cal_mask()).rolling(20, min_periods=20).mean())


def vol(base):
    """杩囧幓20鏃ユ敹鐩婃尝鍔ㄥ煙锛汸arams: base锛汻eturn: high/low mask銆?""
    return _tails(base.pct.where(base.cal_mask()).rolling(20, min_periods=20).std())


def momentum(base):
    """杩囧幓20鏃ュ鍚堝姩閲忓煙锛汸arams: base锛汻eturn: high/low mask銆?""
    x = (1 + base.pct).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    return _tails(x.where(base.cal_mask()))


def regime(base):
    """娌跨敤analyzer鐨?0鏃ュ潎绾柯?%鐗?鐔?闇囪崱鍩燂紝鏃犲叏鏍锋湰闃堝€笺€?
    Params: base锛汻eturn: 涓夌甯傚満鐘舵€乵ask锛屾渶鍒濅笉瓒?0鏃ヤ笉鍒嗙被銆?    """
    nav = m.nav(base.cal_benchmark())
    ma = nav.rolling(60, min_periods=20).mean()
    bull, bear = nav > ma * 1.05, nav < ma * .95
    states = {"bull": bull, "bear": bear, "sideways": ma.notna() & ~bull & ~bear}
    return {name: pd.DataFrame(np.broadcast_to(s.to_numpy()[:, None], base.pct.shape), base.dates, base.ticks)
            for name, s in states.items()}


BUILDERS = {"universe": universe, "industry": industry, "index": index, "barra": barra,
            "volume": volume, "vol": vol, "momentum": momentum, "regime": regime}


def domain_returns(base, domain="industry"):
    """鍩熸敹鐩婏細鍓嶆棩鍩熷唴绛夋潈锛汸arams: base/domain绫诲瀷锛汻eturn: DataFrame锛屽苟娉ㄥ唽mask銆?""
    masks = BUILDERS[domain](base)
    if not masks:
        raise ValueError(f"{domain}娌℃湁鍩熸暟鎹紝璇锋敞鍏omain_masks/domain_weights/barra")
    output = {}
    for name, mask in masks.items():
        key = f"{domain}:{name}"
        base.domain_masks[key] = mask & base.cal_mask()
        w = m.normalize(base.domain_masks[key].astype(float))
        output[name] = m.weighted_returns(w.shift(1).fillna(0), base.pct).where(w.shift(1).sum(axis=1) > 0)
    result = pd.DataFrame(output)
    base.results[f"{domain}_returns"] = result
    return result


def table(base, years=5, domain="industry", construction=None, group_numbers=None):
    """鍩熷唴IC/鍥炲綊鏆撮湶/鏀剁泭璐＄尞鍜岄€夎偂琛ㄧ幇銆?
    Params: base/years:1鎴?鎴?/domain:鍏绫诲瀷/construction/group_numbers锛汻eturn: 鍩熋楁寚鏍囪〃銆?    beta涓哄叏甯傚満鍥犲瓙缁勫悎瓒呴瀵瑰煙瓒呴鏀剁泭鐨勫崟鍙橀噺OLS鏂滅巼锛涜础鐚?beta脳鍩熸棩鍧囪秴棰澝?52锛?    鍩熸湁閲嶅彔锛岃础鐚笉鍙浉鍔犮€侭arra鏆撮湶涓烘爣鍑嗗寲鍥犲瓙缁勫悎瀵规爣鍑嗗寲椋庢牸鐨勪富鍔ㄦ潈閲嶆毚闇层€?    """
    if years not in (1, 3, 5) or domain not in BUILDERS:
        raise ValueError("years/domain鏃犳晥")
    base.cal_labels((1,))
    start = base.dates[-1] - pd.DateOffset(years=years)
    dr = domain_returns(base, domain)
    market = base.cal_benchmark()
    overall = calculate(base, construction, group_numbers).excess_return
    market_mean = base.alpha.where(base.cal_mask()).mean(axis=1)
    rows = {}
    for name in dr:
        key = f"{domain}:{name}"
        mask = base.domain_masks[key]
        # 鍒嗙粍鍙鐞嗚鍩熸浘鍑虹幇鐨勮偂绁紱淇濈暀鍓嶄竴浜ゆ槗鏃ョ敤浜庢敹鐩婂榻愩€?        positions = np.flatnonzero(base.dates >= start)
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
    result.attrs["title"] = f"{years}骞存湡{domain}鍩熷姣?
    base.results[f"domain_{years}_{domain}"] = result
    return result


def plot(base, years=5, domain="industry", construction=None, group_numbers=None):
    """鍩熸毚闇叉煴鍥撅紱Params: 鍚宼able锛汻eturn: Figure銆?""
    result = table(base, years, domain, construction, group_numbers)
    fig, ax = plt.subplots(figsize=(12, 5))
    result.exposure.plot.bar(ax=ax)
    ax.set_title(result.attrs["title"])
    fig.tight_layout()
    return fig

