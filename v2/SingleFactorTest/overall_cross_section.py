"""浣滆€咃細xujiayi | 鍒涘缓鏃ユ湡锛?026-11-11
鏁翠綋琛ㄧ幇锛堟埅闈級锛氱粍鍐呯瓑鏉冪殑鐮旂┒瓒呴鏀剁泭銆佽鐩栫巼鍜屾崲鎵嬨€?"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import maths as m


def calculate(base, construction=None, group_numbers=None, domain="universe"):
    """缁熶竴鏁翠綋鐮旂┒鍙ｅ緞锛汸arams: 鍩虹被/鏋勫缓鏂瑰紡/缁勫彿/鍩燂紱Return: 鏃ラDataFrame銆?
    澶氬ご锛氬澶寸粍-鍩熷唴绛夋潈锛涚┖澶达細鍩熷唴绛夋潈-绌哄ご缁勶紱澶氱┖锛氬澶寸粍-绌哄ご缁勩€?    澶氱┖涓烘瘡鑵?00%鐨勬敹鐩婂樊锛屼笌浜ゆ槗缁勫悎榛樿鎬绘暈鍙?00%鏈夋槑纭尯鍒€?    """
    construction = construction or base.construction
    base.cal_target_holding(construction, group_numbers, "equal", domain)
    long = m.weighted_returns(base.long_target.shift(1).fillna(0), base.pct)
    short = m.weighted_returns(base.short_target.shift(1).fillna(0), base.pct)
    benchmark = base.cal_benchmark(domain=domain)
    excess = long - short if construction == "longshort" else long - benchmark if construction == "long" else benchmark - short
    weights = base.long_target - base.short_target
    previous = weights.shift(1).fillna(0)
    # 婕傜Щ鍚庣殑鏃ф潈閲嶄笌褰撴棩鐩爣姣旇緝锛屾棩鏈熺Щ鑷冲疄闄呮敹鐩婃棩銆?    drift = previous * (1 + base.pct)
    gross_ret = m.weighted_returns(previous, base.pct)
    drift = drift.div(1 + gross_ret, axis=0)
    turnover = weights.sub(drift).abs().sum(axis=1).div(2).shift(1)
    pool = base.cal_mask(domain[0], domain[1])
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
    """鐢昏秴棰濈疮璁℃敹鐩?瑕嗙洊鐜?鎹㈡墜锛汸arams: 鍚宑alculate锛汻eturn: Figure銆?""
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
    """閫愯嚜鐒跺勾鍙奜verall琛ㄧ幇锛汸arams: 鍚宑alculate锛汻eturn: DataFrame銆?""
    data = calculate(base, construction, group_numbers, domain)
    table = pd.DataFrame({y: m.performance(data.excess_return.loc[mask], data.turnover.loc[mask])
                          for y, mask in m.periods(data.index).items()}).T
    base.results["overall_yearly"] = table
    return table


def monthly_table(base, construction=None, group_numbers=None, domain="universe"):
    """鍚勫勾鏈堝鍚堣秴棰濇敹鐩婏紱Params: 鍚宑alculate锛汻eturn: 骞翠唤脳1-12鏈堬紝鏈潵/缂哄け鏈堜负绌恒€?""
    r = calculate(base, construction, group_numbers, domain).excess_return
    r = r.loc[str(r.index[-1].year - 5):]
    rows = []
    for (year, month), values in r.groupby([r.index.year, r.index.month]):
        rows.append((year, month, (1 + values).prod(min_count=1) - 1))
    result = pd.DataFrame(rows, columns=["year", "month", "return"]).pivot(index="year", columns="month", values="return")
    return result.reindex(index=range(r.index[-1].year - 5, r.index[-1].year + 1), columns=range(1, 13))

