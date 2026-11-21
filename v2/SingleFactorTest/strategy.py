"""将因子分组映射为目标持仓；不计算交易成本、不改变实际成交。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(weights):
    total = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def _raw_leg(groups, sign, codes, benchmark, normalization, market_cap, alpha):
    """从指定分组构建未归一化的非负sleeve权重。"""
    if not groups:
        return np.zeros_like(benchmark)
    chosen = np.isin(codes, groups) & (benchmark > 0)
    if normalization == "equal":
        return chosen.astype(float)
    if normalization == "market_cap":
        return np.where(chosen & np.isfinite(market_cap) & (market_cap > 0), market_cap, 0.)
    score = alpha * sign
    good = chosen & np.isfinite(score)
    floor = np.where(good, score, np.inf).min(axis=1)
    floor = np.where(np.isfinite(floor), floor, 0.)
    return np.where(good, score - floor[:, None] + 1e-8, 0.)


def absolute(long_w, short_w, construction):
    """绝对策略初始持仓：L / -S / 0.5(L-S)。"""
    if construction == "long":
        return long_w.copy()
    if construction == "short":
        return -short_w.copy()
    return 0.5 * (long_w - short_w)


def enhanced(long_w, short_w, benchmark, construction):
    """指数增强的主动部分（尚未按active_gross缩放）。"""
    if construction == "long":
        return long_w - benchmark
    if construction == "short":
        return benchmark - short_w
    return 0.5 * (long_w - short_w)


def market_neutral(long_w, short_w, benchmark, construction, hedge_ratio):
    """用当前Alpha域基准作对冲腿；hedge_ratio可按历史Beta估计。"""
    hedge = hedge_ratio[:, None] * benchmark
    if construction == "long":
        return long_w - hedge
    if construction == "short":
        return hedge - short_w
    return 0.5 * (long_w - short_w) - hedge


def estimate_hedge_ratio(base, long_w, short_w, benchmark, construction,
                         window=60, min_periods=40):
    """滚动回归：T日h只使用截至T日已实现的历史组合收益。

    收益序列对应历史逐日构建的理论sleeve，不是成交后的实际组合Beta；
    如需实盘动态估计，应以实际成交持仓的历史收益重新估计。
    """
    exposure = (long_w if construction == "long" else short_w
                if construction == "short" else 0.5 * (long_w - short_w))
    pct = base.pct.to_numpy(dtype=float, copy=False)
    dates = base.dates
    b_prev = pd.DataFrame(benchmark, index=dates).shift(1).to_numpy()
    leg_prev = pd.DataFrame(exposure, index=dates).shift(1).to_numpy()
    valid_pct = np.isfinite(pct)
    bm_ret = np.sum(np.nan_to_num(b_prev) * np.where(valid_pct, pct, 0.), axis=1)
    leg_ret = np.sum(np.nan_to_num(leg_prev) * np.where(valid_pct, pct, 0.), axis=1)
    missing_b = ((np.abs(b_prev) > 0) & ~valid_pct).any(axis=1)
    missing_leg = ((np.abs(leg_prev) > 0) & ~valid_pct).any(axis=1)
    rb = pd.Series(bm_ret, index=dates).mask(missing_b)
    rl = pd.Series(leg_ret, index=dates).mask(missing_leg)
    # 首日没有前一交易日形成的组合，不能视为一条真实的0收益观测。
    rb.iloc[0] = np.nan
    rl.iloc[0] = np.nan
    cov = rl.rolling(window, min_periods=min_periods).cov(rb)
    var = rb.rolling(window, min_periods=min_periods).var()
    h = (cov / var.where(var > 1e-12)).to_numpy()
    return np.where(np.isfinite(h), h, 0.), np.isfinite(h)


def build_target(
    base, strategy="enhanced", construction=None,
        group_numbers=None, normalization=None,
        domain="universe", name=None, industry_neutral=False,
        active_gross=None, num_groups=None,
        hedge_method="dollar", hedge_window=60, hedge_min_periods=40):
    """将T日Alpha分组映射为T日目标权重，不处理成交。

    absolute:       L / -S / 0.5(L-S)
    enhanced:       B + c*(L-B / B-S / 0.5(L-S))，要求最终非负
    market_neutral: 对冲组合，可选择dollar(资金中性)或beta(滚动基准Beta中性)
    hedge_method仅影响market_neutral；beta按已实现的历史sleeve收益估计，
    最早需hedge_min_periods个有效日，样本不足时不建立市场中性仓位。

    industry_neutral=True会使L/S sleeve按基准行业比例配置，缺少候选的
    行业持有原基准篮子。beta对冲会重新引入行业净偏离，不能同时
    声称最终仓位严格行业中性。
    """
    strategy = str(strategy).lower()
    construction = construction or base.construction
    normalization = normalization or base.normalization
    num_groups = base.groups if num_groups is None else num_groups
    if strategy not in ("absolute", "enhanced", "market_neutral"):
        raise ValueError("strategy须为absolute/enhanced/market_neutral")
    if construction not in ("long", "short", "longshort"):
        raise ValueError("construction须为long/short/longshort")
    if normalization not in ("equal", "market_cap", "factor"):
        raise ValueError("normalization须为equal/market_cap/factor")
    if domain not in ("universe", "index", "industry"):
        raise ValueError("domain须为universe/index/industry")
    if not isinstance(num_groups, (int, np.integer)) or num_groups < 2:
        raise ValueError("num_groups须为至少2的整数")
    if active_gross is None:
        active_gross = 0.1 if strategy == "enhanced" else 1.0
    if not np.isfinite(active_gross) or active_gross < 0:
        raise ValueError("active_gross须为非负有限数")
    if hedge_method not in ("dollar", "beta"):
        raise ValueError("hedge_method须为dollar/beta")
    if not (isinstance(hedge_window, int) and isinstance(hedge_min_periods, int)
            and 3 <= hedge_min_periods <= hedge_window):
        raise ValueError("需满足3 <= hedge_min_periods <= hedge_window")

    # 直接复用现有基准权重；universe与cal_benchmark口径一致。
    if domain == "universe":
        name = "equal" if name is None else name
        if name not in ("equal", "market_cap"):
            raise ValueError("universe的name须为equal/market_cap")
        mv_df = base.fields["market_cap"]
        eligible = base.pct.notna() & np.isfinite(mv_df) & mv_df.gt(0)
        raw = eligible.astype(float) if name == "equal" else mv_df.where(eligible, 0).fillna(0)
        b_frame = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    else:
        if name is None:
            raise ValueError("index/industry须传入name")
        key = str(int(name)) if domain == "industry" else name
        b_frame = base.domain_weights[domain][key]

    base.cal_groups(num_groups=num_groups, domain=domain,
                    name=None if domain == "universe" else name)
    codes = base.group_codes.to_numpy(dtype=float, copy=False)
    b = b_frame.reindex(index=base.dates, columns=base.ticks).to_numpy(dtype=float, copy=True)
    if not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("基准权重包含负数或非有限数")
    total_b = b.sum(axis=1)
    valid_b = total_b > 1e-12
    if np.any(valid_b & ~np.isclose(total_b, 1., atol=1e-4)):
        raise ValueError("有效基准日权重和须为1，请先检查指数权重单位")

    if group_numbers is None:
        group_numbers = (((num_groups,), (1,)) if construction == "longshort"
                         else (num_groups,) if construction == "long" else (1,))
    if construction == "longshort":
        if len(group_numbers) != 2:
            raise ValueError("多空分组形式须为((10,), (1,))")
        lg, sg = group_numbers
    elif construction == "long":
        lg, sg = group_numbers, ()
    else:
        lg, sg = (), group_numbers
    lg, sg = tuple(lg), tuple(sg)
    if (not set(lg + sg).issubset(range(1, num_groups + 1))
            or set(lg) & set(sg) or (not lg and not sg)):
        raise ValueError("分组无效或多空组重叠")

    mv = base.fields["market_cap"].to_numpy(dtype=float, copy=False)
    alpha = base.alpha.to_numpy(dtype=float, copy=False)

    raw_l = _raw_leg(lg, 1, codes, b, normalization, mv, alpha)
    raw_s = _raw_leg(sg, -1, codes, b, normalization, mv, alpha)
    relative = strategy != "absolute"
    if not industry_neutral:
        wl, ws = _normalize(raw_l), _normalize(raw_s)
        # 没选中股票的日期：相对组合退化为基准，不凭空引入敞口。
        if relative:
            if lg:
                empty = raw_l.sum(axis=1) == 0
                wl[empty] = b[empty]
            if sg:
                empty = raw_s.sum(axis=1) == 0
                ws[empty] = b[empty]
    else:
        industry = base.fields["industry"].to_numpy(dtype=float, copy=False)
        invalid = (b > 0) & (~np.isfinite(industry) | (industry < 0)
                             | (industry >= 31) | (industry != np.floor(industry)))
        if np.any(invalid):
            raise ValueError("基准股票存在无效行业编码")
        wl, ws = np.zeros_like(b), np.zeros_like(b)
        b_ind = np.zeros((len(base.dates), 31))
        l_ind = np.zeros_like(b_ind)
        s_ind = np.zeros_like(b_ind)
        for g in range(31):
            where = industry == g
            b_ind[:, g] = np.where(where, b, 0).sum(axis=1)
            l_ind[:, g] = np.where(where, raw_l, 0).sum(axis=1)
            s_ind[:, g] = np.where(where, raw_s, 0).sum(axis=1)
        if relative:
            budget = b_ind  # 无选中股票的行业退回原基准篮子
        else:
            covered = ((l_ind > 0) & (s_ind > 0) if construction == "longshort"
                       else (l_ind > 0) if construction == "long" else (s_ind > 0))
            budget = _normalize(b_ind * covered)
        for g in range(31):
            where = industry == g
            for raw, total, out, enabled in ((raw_l, l_ind[:, g], wl, bool(lg)),
                                              (raw_s, s_ind[:, g], ws, bool(sg))):
                if not enabled:
                    continue
                partial = np.divide(np.where(where, raw, 0) * budget[:, g, None],
                                    total[:, None], out=np.zeros_like(b), where=total[:, None] > 0)
                if relative:
                    partial += np.where((total <= 0)[:, None] & where, b, 0.)
                out += partial


    if strategy == "absolute":
        init = absolute(wl, ws, construction)
    elif strategy == "enhanced":
        init = enhanced(wl, ws, b, construction)
    else:
        init = None

    # dollar模式：单边以1倍Benchmark对冲，longshort已自对冲，不额外加基准。
    hedge = np.full(len(base.dates), 0. if construction == "longshort" else 1.)
    hedge_valid = np.ones(len(base.dates), dtype=bool)
    if strategy == "market_neutral":
        if hedge_method == "beta":
            hedge, hedge_valid = estimate_hedge_ratio(
                base, wl, ws, b, construction, hedge_window, hedge_min_periods)
        init = market_neutral(wl, ws, b, construction, hedge)

    valid = valid_b & hedge_valid
    if strategy == "absolute":
        if construction == "long":
            valid &= wl.sum(axis=1) > 1e-12
        elif construction == "short":
            valid &= ws.sum(axis=1) > 1e-12
        else:
            valid &= (wl.sum(axis=1) > 1e-12) & (ws.sum(axis=1) > 1e-12)
    if construction == "longshort":
        valid &= (wl.sum(axis=1) > 1e-12) & (ws.sum(axis=1) > 1e-12)
    init[~valid] = 0.
    gross = np.abs(init).sum(axis=1)
    scale = np.divide(active_gross, gross, out=np.zeros_like(gross), where=gross > 1e-12)
    if strategy == "enhanced":
        # 保留主动方向统一缩放，防止个股低配成为实际负仓位。
        feasible = np.divide(b, -init, out=np.full_like(b, np.inf), where=init < -1e-12)
        scale = np.minimum(scale, feasible.min(axis=1))
    active = init * scale[:, None]
    target = b + active if strategy == "enhanced" else active

    def as_frame(values):
        return pd.DataFrame(values, index=base.dates, columns=base.ticks)
    base.long_target = as_frame(wl)
    base.short_target = as_frame(ws)
    base.benchmark_weight = as_frame(b)
    base.active_target = as_frame(active)
    base.target_holding = as_frame(target)
    base.hedge_ratio = pd.Series(
        np.where(hedge_valid & valid_b, hedge, np.nan)
        if strategy == "market_neutral" else np.full(len(base.dates), np.nan),
        index=base.dates, name="hedge_ratio"
    )
    base.realized_active_gross = pd.Series(np.abs(active).sum(axis=1), index=base.dates)
    base.target_spec = (strategy, construction, group_numbers, normalization, domain, name,
                        industry_neutral, active_gross, num_groups, hedge_method,
                        hedge_window, hedge_min_periods)
    return base.target_holding

