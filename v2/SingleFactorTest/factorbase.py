"""作者：xujiayi | 创建日期：2026-11-11
单因子研究上下文：数据、预处理、标签、分组、持仓及统一成交入口。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from . import maths as m


# 使用已有数据生产脚本中的真实字段名；不将缺失交易状态静默替换为可交易。
LOCAL_FIELDS = {
    "pct": "d_essentials/pct", "market_cap": "d_essentials/circ_mv",
    "volume": "d_essentials/volume", "amount": "d_essentials/amount",
    "open": "d_essentials/open", "vwap": "d_essentials/vwap",
    "close": "d_essentials/close", "industry": "industry/industry",
    "tradable": "basic/tradable", "tradable_t": "basic/tradable_t", "limit_up": "basic/ceil_only",
    "limit_down": "basic/floor_only", "suspended": "basic/issuspend",
    "st": "basic/isst", "bp_lf": "d_financial/bp_lf",
    "adjustment": "d_essentials/ratio_adjusting_factor",
    "hastrade": "basic/hastrade",
}


class FactorBase:
    """单因子检验基类，所有研究结果按明确名称保存在实例中。

    Params: factor_name: 因子名称/本地字段；construction: long/short/longshort；
        normalization: equal/market_cap/factor；transform: 因子压缩方法；
        neutralize_industry/neutralize_size: 是否中性化；standardize: 是否Z-score；
        groups: 分组数；direction: 1表示大值看多，-1表示小值看多；
        start/end: 报告日期；history_years: 至少6年加载窗口；root/data: DataPool入口。
    Return: FactorBase，实例化不读盘、不计算、不改全局绘图设置。
    """

    def __init__(self, factor_name, construction="longshort", normalization="equal",
                 transform="asinh", neutralize_industry=True, neutralize_size=True,
                 standardize=True, groups=10, direction=1, start=None, end=None,
                 history_years=6, root=None, data=None):
        if construction not in ("long", "short", "longshort"):
            raise ValueError("construction须为long/short/longshort")
        if normalization not in ("equal", "market_cap", "factor"):
            raise ValueError("normalization须为equal/market_cap/factor")
        if groups < 2 or direction not in (-1, 1) or history_years < 6:
            raise ValueError("groups>=2，direction为±1，history_years>=6")
        self.factor_name, self.construction = factor_name, construction
        self.normalization, self.transform = normalization, transform
        self.neutralize_industry, self.neutralize_size = neutralize_industry, neutralize_size
        self.standardize, self.groups, self.direction = standardize, groups, direction
        self.start, self.end, self.history_years = start, end, history_years
        self.root, self.data = root, data
        self.fields, self.domain_masks, self.domain_weights, self.barra = {}, {}, {}, {}
        self.results = {}

    def load(self, alpha=None, pct=None, fields=None, domain_masks=None,
             domain_weights=None, barra=None, field_paths=None, pct_scale=.01,
             strict_history=True, progress=None):
        """读取DataPool或注入研究数据，严格对齐日期和股票轴。

        Params: alpha/pct: DataFrame，注入pct必须为小数；fields: 原始市场字段映射；
            domain_masks/domain_weights/barra: 名称→同轴DataFrame；field_paths: 本地字段覆盖；
            pct_scale: 仅本地pct转换倍数，默认百分数转小数；strict_history: 不足5年时报错。
        Return: self；保留raw_alpha/pct/fields、PE=1/BP及完整加载日期。
        """
        progress = progress or (lambda message: None)
        if alpha is None:
            if self.data is None:
                from v2.GetData import DataPool
                if self.root is None:
                    from v2.UpdateData.config import ROOT
                    self.root = ROOT
                self.data = DataPool(self.root, asset="stock")
            paths = {**LOCAL_FIELDS, **(field_paths or {})}
            dates = pd.DatetimeIndex(self.data.axis.trade_dates)
            end = pd.Timestamp(self.end) if self.end else dates[-1]
            begin = min(pd.Timestamp(self.start), end - pd.DateOffset(years=self.history_years)) if self.start else end - pd.DateOffset(years=self.history_years)
            selected = dates[(dates >= begin) & (dates <= end)]
            if len(selected) == 0:
                raise ValueError("请求区间没有交易日")

            def read(path):
                progress(f"读取 {path}")
                values = self.data.read(path, selected[-1], selected[0], copy=False)
                return pd.DataFrame(values, selected, pd.Index(self.data.axis.ticks).astype(str))

            alpha = read(self.factor_name)
            pct = read(paths["pct"]) * pct_scale
            fields = dict(fields or {})
            self.missing_fields = []
            for name, path in paths.items():
                if name != "pct" and name not in fields:
                    try:
                        fields[name] = read(path)
                    except FileNotFoundError:
                        self.missing_fields.append(name)
            if self.missing_fields:
                warnings.warn(f"本地缺少字段（相关功能调用时将检查）: {self.missing_fields}")
        if pct is None:
            raise ValueError("注入alpha时必须同时提供小数日收益pct")
        progress("对齐因子矩阵")
        alpha = alpha.copy()
        alpha.index = pd.DatetimeIndex(alpha.index)
        alpha.columns = alpha.columns.astype(str)
        if not alpha.index.is_monotonic_increasing or alpha.index.has_duplicates or alpha.columns.has_duplicates:
            raise ValueError("因子日期须递增，日期/股票不得重复")
        if self.end is not None:
            alpha = alpha.loc[:pd.Timestamp(self.end)]
        if alpha.empty:
            raise ValueError("因子为空")
        self.dates, self.ticks = alpha.index, alpha.columns

        def align(frame):
            frame = frame.copy()
            frame.index, frame.columns = pd.DatetimeIndex(frame.index), frame.columns.astype(str)
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError("输入轴重复")
            return frame.reindex(index=self.dates, columns=self.ticks)

        self.raw_alpha = alpha.replace([np.inf, -np.inf], np.nan).astype(float)
        self.pct = align(pct).replace([np.inf, -np.inf], np.nan).astype(float)
        self.fields = {}
        for k, v in (fields or {}).items():
            progress(f"对齐市场数据：{k}")
            self.fields[k] = align(v)
        self.domain_masks = {k: align(v).eq(True) for k, v in (domain_masks or {}).items()}
        self.domain_weights = {k: align(v) for k, v in (domain_weights or {}).items()}
        self.barra = {k: align(v) for k, v in (barra or {}).items()}
        if "bp_lf" in self.fields:
            self.fields["pe_lf"] = 1 / self.fields["bp_lf"].replace(0, np.nan)
        span = (self.dates[-1] - self.dates[0]).days
        if strict_history and (span < 365 * 5 or len(self.dates) < 252 * 5):
            raise ValueError("至少需要5年有效交易日；建议加载6年以上。短样本测试可显式strict_history=False")
        self.report_start = pd.Timestamp(self.start) if self.start else self.dates[-1] - pd.DateOffset(years=5)
        self.results.clear()
        # 重载时清除全部派生数据，避免旧结果污染新样本。
        for name in ("alpha", "labels", "ics", "rankics", "sign_ics", "group_codes", "group_returns",
                     "group_weights", "target_holding", "portfolio", "capacity_results"):
            self.__dict__.pop(name, None)
        return self

    def load_domains(self, indices=("hs300", "zz500", "zz1000", "zz2000"),
                     styles=("beta", "btop", "size", "nonlinear_size", "momentom",
                             "residual_vol", "liquidity", "leverage", "growth1", "earnings_yield")):
        """按需加载本地指数权重、Barra暴露及正负分域，避免初始加载占用过多内存。

        Params: indices/styles: 本地指数/风格字段名序列；Return: self。
        缺失文件明确报错；可使用load的domain_weights/barra参数直接注入替代。
        """
        if self.data is None:
            raise ValueError("需要本地DataPool；或通过load注入域数据")

        def read(path):
            value = self.data.read(path, self.dates[-1], self.dates[0], ticks=self.ticks, copy=False)
            return pd.DataFrame(value, self.dates, self.ticks)

        for name in indices:
            weight = read(f"index/weight/{name}_weight")
            self.domain_weights[name] = weight
            self.domain_masks[f"index:{name}"] = weight.gt(0)
        for name in styles:
            self.barra[name] = read(f"barra/{name}")
            mask = read(f"barra/{name}_mask")
            self.domain_masks[f"barra:{name}+"] = mask.gt(0)
            self.domain_masks[f"barra:{name}-"] = mask.lt(0)
        return self

    def require(self, *names):
        """检查市场字段；Params: names: 字段名；Return: None，缺失则明确报错。"""
        missing = [x for x in names if x not in self.fields]
        if missing:
            raise ValueError(f"需要数据字段: {missing}；通过fields或field_paths提供")

    def mask(self, domain="universe"):
        """取得事前研究股票池；Params: domain: 注册域名称；Return: bool DataFrame。

        不使用本地basic/tradable建研究股票池：该字段包含次日状态，存在前视信息。
        """
        base = self.pct.notna()
        if "tradable_t" in self.fields:
            base &= self.fields["tradable_t"].eq(True)
        for field, allowed in (("hastrade", True), ("st", False), ("suspended", False)):
            if field in self.fields:
                base &= self.fields[field].eq(allowed)
        if domain == "universe":
            return base
        if domain not in self.domain_masks:
            raise KeyError(f"未注册域: {domain}")
        return base & self.domain_masks[domain]

    def process_alpha(self):
        """稳健压缩→行业/log市值中性化→截面Z-score。

        Params: 无，使用显式构造参数；Return: DataFrame，并保留self.alpha。
        """
        if self.neutralize_industry:
            self.require("industry")
        if self.neutralize_size:
            self.require("market_cap")
        values = m.transform(self.raw_alpha.where(self.mask()) * self.direction, self.transform)
        if self.neutralize_industry or self.neutralize_size:
            values = m.neutralize(values, self.fields.get("industry") if self.neutralize_industry else None,
                                  self.fields.get("market_cap") if self.neutralize_size else None)
        self.alpha = m.standardize(values) if self.standardize else values
        self._group_cache_key = None
        for name in ("labels", "ics", "rankics", "sign_ics"):
            self.__dict__.pop(name, None)
        return self.alpha

    def cal_labels(self, horizons=(1, 3, 5, 10, 20)):
        """未来复合收益和IC；Params: horizons: 交易日期限；Return: 标签字典。

        t行对应t+1开始的未来收益，尾部不完整窗口保留NaN。
        """
        if not hasattr(self, "alpha"):
            self.process_alpha()
        if not hasattr(self, "labels"):
            self.labels, self.sign_ics = {}, {}
            self.ics, self.rankics = pd.DataFrame(index=self.dates), pd.DataFrame(index=self.dates)
        for h in horizons:
            if h in self.labels:
                continue
            y = self.labels[h] = m.forward_returns(self.pct, h)
            self.ics[h], self.rankics[h] = m.cross_ic(self.alpha, y), m.cross_ic(self.alpha, y, True)
            self.sign_ics[h] = pd.DataFrame({
            "pos_ic": m.cross_ic(self.alpha.where(y > 0), y),
            "neg_ic": m.cross_ic(self.alpha.where(y < 0), y),
            "rank_pos_ic": m.cross_ic(self.alpha.where(y > 0), y, True),
            "rank_neg_ic": m.cross_ic(self.alpha.where(y < 0), y, True),
            })
        return self.labels

    def cal_benchmark(self, method="equal", domain="universe", index_name=None):
        """基准收益与净值（收益索引为实际收益日，使用前日权重）。

        Params: method: equal/market_cap/index；domain: 域名；index_name: 注册指数权重名。
        Return: Series，同时保存benchmark_weight/benchmark_return/benchmark_nav。
        """
        mask = self.mask(domain)
        if method == "equal":
            weight = mask.astype(float)
        elif method == "market_cap":
            self.require("market_cap")
            weight = self.fields["market_cap"].where(mask, 0)
        elif method == "index":
            weight = self.domain_weights[index_name].where(mask, 0)
        else:
            raise ValueError("未知基准方法")
        self.benchmark_weight = m.normalize(weight)
        self.benchmark_return = m.weighted_returns(self.benchmark_weight.shift(1).fillna(0), self.pct)
        self.benchmark_nav = m.nav(self.benchmark_return)
        return self.benchmark_return

    def cal_groups(self, domain="universe"):
        """域内分组及组内等权收益。

        Params: domain: 域名；Return: DataFrame，G1低值→G10高值，Average为可用组等权。
        并列因子保持同组，不按股票代码强行拆分；常数截面不分组。
        """
        if not hasattr(self, "alpha"):
            self.process_alpha()
        key = (id(self.alpha), domain, self.groups, id(self.domain_masks.get(domain)))
        if getattr(self, "_group_cache_key", None) == key:
            return self.group_returns
        x = self.alpha.where(self.mask(domain))
        count = x.count(axis=1)
        codes = np.ceil((x.rank(axis=1, method="average") - .5).div(count, axis=0) * self.groups).clip(1, self.groups)
        self.group_codes = codes.where(x.nunique(axis=1).gt(1), axis=0)
        self.group_weights, returns = {}, {}
        for g in range(1, self.groups + 1):
            w = m.normalize(self.group_codes.eq(g).astype(float))
            self.group_weights[g] = w
            returns[f"G{g}"] = m.weighted_returns(w.shift(1).fillna(0), self.pct).where(w.shift(1).sum(axis=1) > 0)
        self.group_returns = pd.DataFrame(returns)
        self.group_returns["Average"] = self.group_returns.mean(axis=1)
        self.group_domain = domain
        self._group_cache_key = key
        return self.group_returns

    def cal_target_holding(self, construction=None, group_numbers=None, normalization=None,
                           domain="universe", benchmark_enhanced=False, active_gross=1.,
                           benchmark_method="equal", benchmark_index=None):
        """选择组并归一化；默认高组多头、低组空头，多空每腿各100%。

        Params: construction: long/short/longshort；group_numbers: 单腿组号序列，
            多空时为(多头组号序列, 空头组号序列)；normalization: equal/market_cap/factor；
            domain: 域；benchmark_enhanced: 是否基准+主动权重；active_gross: 主动总敞口。
        Return: 目标权重DataFrame（信号日）；空头有符号，不与空头组的原股票收益混淆。
        """
        construction = construction or self.construction
        normalization = normalization or self.normalization
        if construction not in ("long", "short", "longshort") or active_gross < 0:
            raise ValueError("构建方式/active_gross无效")
        self.cal_groups(domain)
        if group_numbers is None:
            group_numbers = ((self.groups,), (1,)) if construction == "longshort" else ((self.groups,) if construction == "long" else (1,))
        long_groups, short_groups = (group_numbers if construction == "longshort" else
                                     (group_numbers, ()) if construction == "long" else ((), group_numbers))
        if set(long_groups) & set(short_groups):
            raise ValueError("多空组不可重叠")
        if not set(long_groups).union(short_groups).issubset(range(1, self.groups + 1)):
            raise ValueError("组号超出范围")

        def leg(groups, sign):
            selected = self.group_codes.isin(groups)
            if normalization == "equal":
                raw = selected.astype(float)
            elif normalization == "market_cap":
                self.require("market_cap")
                raw = self.fields["market_cap"].where(selected, 0)
            elif normalization == "factor":
                # 保证腿内权重非负且随信号方向单调；避免负分数直接相除导致反向持仓。
                score = (self.alpha * sign).where(selected)
                raw = score.sub(score.min(axis=1), axis=0).add(1e-8).where(selected, 0)
            else:
                raise ValueError("未知归一化方法")
            return m.normalize(raw)

        self.long_target, self.short_target = leg(long_groups, 1), leg(short_groups, -1)
        scale = active_gross / 2 if construction == "longshort" else active_gross
        self.active_target = (self.long_target - self.short_target) * scale
        self.target_holding = self.active_target.copy()
        if benchmark_enhanced:
            self.cal_benchmark(benchmark_method, domain, benchmark_index)
            self.target_holding += self.benchmark_weight
        self.target_spec = (construction, group_numbers, normalization, domain, benchmark_enhanced, active_gross)
        return self.target_holding

    def cal_portfolio(self, turnover_limit=1., stock_deviation=None, stock_cap=1.,
                      industry_deviation=None, commission=.0003, stamp_tax=.0005,
                      execution_price="vwap", capital=1e8, max_participation=.1,
                      impact_coefficient=.001, lot_size=100, borrow_rate=0.,
                      construction=None, group_numbers=None, normalization=None,
                      domain="universe", benchmark_enhanced=False, active_gross=1.,
                      rebalance_days=1, benchmark_method="equal", benchmark_index=None):
        """统一真实成交与容量模拟入口。

        Params: 换手/偏离/上限均为NAV比例；费用为小数；成交为次日open/vwap；
            capital: 初始资金元；max_participation: 当日成交额参与上限；
            impact_coefficient: 平方根冲击系数；lot_size: 整手股数；borrow_rate: 年借券成本；
            其余构建参数同cal_target_holding。
        Return: PortfolioResult；保存在self.portfolio，并保留全部持仓及成本。
        """
        from .execution import simulate
        self.cal_target_holding(construction, group_numbers, normalization, domain,
                                benchmark_enhanced, active_gross, benchmark_method, benchmark_index)
        if stock_deviation is not None or industry_deviation is not None:
            self.cal_benchmark(benchmark_method, domain, benchmark_index)
        self.portfolio = simulate(self, turnover_limit, stock_deviation, stock_cap,
                                  industry_deviation, commission, stamp_tax, execution_price,
                                  capital, max_participation, impact_coefficient, lot_size, borrow_rate,
                                  rebalance_days=rebalance_days)
        return self.portfolio

    def cal_porfolio(self, *args, **kwargs):
        """兼容需求中的拼写；Params: 同cal_portfolio；Return: PortfolioResult。"""
        return self.cal_portfolio(*args, **kwargs)


factorbase = FactorBase
