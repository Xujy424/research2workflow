"""作者：xujiayi | 创建日期：2026-11-11
单因子研究上下文：数据、预处理、标签、分组、持仓和成交模拟。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from . import maths as m


from v2.GetData import DataPool
from v2.UpdateData.config import ROOT


# 本地字段路径映射。
LOCAL_FIELDS = {
    "pct": "d_essentials/pct",
    "market_cap": "d_essentials/circ_mv",
    "volume": "d_essentials/volume",
    "amount": "d_essentials/amount",
    "open": "d_essentials/open",
    "vwap": "d_essentials/vwap",
    "close": "d_essentials/close",
    "industry": "industry/industry",
    "tradable": "basic/tradable",
    "tradable_t": "basic/tradable_t",
    "limit_up": "basic/ceil_only",
    "limit_down": "basic/floor_only",
    "suspended": "basic/issuspend",
    "st": "basic/isst",
    "bp_lf": "d_financial/bp_lf",
    "adjustment": "d_essentials/ratio_adjusting_factor",
    "hastrade": "basic/hastrade",
}

INDEX_FIELDS = {name: f"index/weight/{name}_weight"
                for name in ("hs300", "zz500", "a500", "zz1000", "zz2000")}
BARRA_FIELDS = {name: f"barra/{name}"
                for name in ("beta", "btop", "size", "nonlinear_size", "momentom",
                             "residual_vol", "liquidity", "leverage", "growth1", "earnings_yield")}


class FactorBase:
    """单因子检验基类，计算结果保存在实例中。
    Params: factor_name: 因子名称；construction: long/short/longshort；
        normalization: equal/market_cap/factor；transform: 因子变换方法；
        neutralize: 是否中性化；standardize: 是否标准化；groups: 分组数；
        direction: 1为大值看多，-1为小值看多；start/end: 报告日期；
        history_years: 历史加载年数；root/data: 数据根目录和数据源。
    Return: FactorBase实例。
    """
    def __init__(
            self,
            factor_name,
            construction="longshort",  # long / short / longshort
            normalization="equal",    # equal / market_cap / factor
            transform="asinh",        # softsign / piecewise / rank_gaussian / yeo_johnson / none
            neutralize=True,
            standardize=True,
            groups=10,
            direction=1,
            start=None, end=None, history_years=6,
            root=ROOT, data=None
        ):
        self.factor_name, self.construction = factor_name, construction
        self.normalization, self.transform = normalization, transform
        self.neutralize, self.standardize = neutralize, standardize
        self.groups, self.direction = groups, direction
        self.start, self.end, self.history_years = start, end, history_years
        self.root, self.data = root, data

        self.fields, self.domain_masks, self.domain_weights, self.barra = {}, {}, {}, {}
        self.results = {}

    def _read_field(self, path):
        """读取当前区间的DataPool字段。
        Params: path: 字段路径；Return: 日期×股票的DataFrame。
        """
        values = self.data.read(path, self.dates[-1], self.dates[0], copy=False)
        return pd.DataFrame(values, index=self.dates, columns=self.ticks)

    def load(self, indices=(), styles=()):
        """创建DataPool并读取因子、行情和指定域数据。
        Params: indices/styles: 指数和Barra名称。
        Return: self；请求至少5年、默认6年的历史窗口，实际可用历史取决于本地数据。
        """
        self.data = DataPool(self.root, asset="stock")
        end = pd.Timestamp(self.end) if self.end else pd.Timestamp(self.data.axis.trade_dates[-1])
        begin = end - pd.DateOffset(years=max(5, self.history_years))
        dates = pd.DatetimeIndex(self.data.axis.trade_dates)
        self.dates = dates[(dates >= begin) & (dates <= end)]
        self.ticks = pd.Index(self.data.axis.ticks).astype(str)

        self.raw_alpha = self._read_field('factor_pool/'+self.factor_name)
        self.fields = {name: self._read_field(path) for name, path in LOCAL_FIELDS.items()}
        self.pct = self.fields['pct']*.01
        self.fields["pe_lf"] = 1 / self.fields["bp_lf"].replace(0, np.nan)
        self.barra = {name: self._read_field(path) for name, path in BARRA_FIELDS.items()}
        self.cal_mask()
        self.cal_weight()
        self.report_start = pd.Timestamp(self.start) if self.start else self.dates[-1] - pd.DateOffset(years=5)
        self.results.clear()
        # 重载时清除派生结果。
        for name in ("alpha", "labels", "ics", "rankics", "sign_ics", "group_codes", "group_returns",
                     "group_weights", "target_holding", "portfolio", "capacity_results"):
            self.__dict__.pop(name, None)
        return self
    
    def cal_mask(self):
        """读取本地指数、行业和Barra掩码，不附加交易状态过滤。
        Params: 无；Return: 域掩码字典，行业为编码矩阵，Barra为正负编码矩阵。
        """
        self.domain_masks = {
            "index": {name: self._read_field(f"index/mask/{name}_mask")
                      for name in INDEX_FIELDS},
            "industry": self.fields["industry"],
            "barra": {name: self._read_field(f"{path}_mask")
                      for name, path in BARRA_FIELDS.items()},
        }
        return self.domain_masks

    def cal_weight(self):
        """计算各域权重，指数读取本地权重，行业和Barra按流通市值归一。
        Params: 无；Return: 域权重字典。行业编码0—30，共31类；Barra为正组减负组。
        """
        market_cap = self.fields["market_cap"]
        self.domain_weights = {
            "index": {name: self._read_field(INDEX_FIELDS[name])
                      for name in self.domain_masks["index"]},
            "industry": {}, "barra": {},
        }
        for code in range(31):
            raw = market_cap.where(self.domain_masks["industry"].eq(code), 0)
            self.domain_weights["industry"][str(code)] = raw.div(
                raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        for name, mask in self.domain_masks["barra"].items():
            positive = market_cap.where(mask.eq(1), 0)
            negative = market_cap.where(mask.eq(-1), 0)
            positive = positive.div(positive.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            negative = negative.div(negative.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            self.domain_weights["barra"][name] = positive - negative
        return self.domain_weights

    def cal_barra_ret(self):
        """以昨日风格暴露及行业哑变量解释当日股票收益，计算风格因子收益。
        模型：r_t = X_{t-1} f_t + D_{t-1} g_t + e_t。
        WLS最小化sum(sqrt(mv_{t-1}) * e_t**2)，使用昨日流通市值。
        行业采用完整哑变量、不另加截距；仅保留当日样本中出现的行业列。
        Params: 无；
        Return: 日期×十大风格的DataFrame，并保存barra_returns。
        首日、有效样本不足或设计矩阵不满秩时保留NaN。
        """
        styles = list(BARRA_FIELDS)
        returns = self.pct.to_numpy(dtype=float, copy=False)
        exposures = np.stack([self.barra[name].to_numpy(dtype=float, copy=False)
                              for name in styles], axis=2)
        industry = self.fields["industry"].to_numpy(dtype=float, copy=False)
        market_cap = self.fields["market_cap"].to_numpy(dtype=float, copy=False)
        factor_returns = np.full((len(self.dates), len(styles)), np.nan)
        for day in range(1, len(self.dates)):
            x = exposures[day - 1]
            y = returns[day]
            codes, mv = industry[day - 1], market_cap[day - 1]
            valid = (np.isfinite(y) & np.isfinite(x).all(axis=1)
                     & np.isin(codes, np.arange(31)) & np.isfinite(mv) & (mv > 0))
            codes = codes[valid]
            dummies = codes[:, None] == np.unique(codes)[None, :]
            design = np.column_stack((x[valid], dummies))
            if design.shape[0] <= design.shape[1]:
                continue
            # lstsq两侧乘sqrt(W)：W=sqrt(mv)，因此缩放因子为mv**0.25。
            scale = mv[valid] ** .25
            coefficients, _, rank, _ = np.linalg.lstsq(
                design * scale[:, None], y[valid] * scale, rcond=None)
            if rank == design.shape[1]:
                factor_returns[day] = coefficients[:len(styles)]
        result = pd.DataFrame(factor_returns, index=self.dates, columns=styles)
        self.barra_returns = result
        return result

    
    


    def process_alpha(self):
        """执行因子变换、行业市值中性化和截面标准化。
        Params: 无，使用实例配置；Return: DataFrame，并保存self.alpha。
        """
        if self.neutralize_industry:
            self.require("industry")
        if self.neutralize_size:
            self.require("market_cap")
        values = m.transform(self.raw_alpha.where(self.pct.notna()) * self.direction, self.transform)
        if self.neutralize_industry or self.neutralize_size:
            values = m.neutralize(values, self.fields.get("industry") if self.neutralize_industry else None,
                                  self.fields.get("market_cap") if self.neutralize_size else None)
        self.alpha = m.standardize(values) if self.standardize else values
        self._group_cache_key = None
        for name in ("labels", "ics", "rankics", "sign_ics"):
            self.__dict__.pop(name, None)
        return self.alpha

    def cal_labels(self, horizons=(1, 3, 5, 10, 20)):
        """计算未来复合收益和IC。
        Params: horizons: 交易日窗口；Return: 标签字典。
        t行对应t+1起的未来收益，尾部不完整窗口保留NaN。
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
        """计算基准收益和净值，使用前日权重。
        Params: method: equal/market_cap/index；domain: 域名；index_name: 指数名称。
        Return: 收益Series，并保存benchmark_weight、benchmark_return、benchmark_nav。
        """
        if domain == "universe":
            mask = self.pct.notna()
        else:
            kind, name = domain if isinstance(domain, tuple) else domain.split(":", 1)
            if kind == "industry":
                mask = self.domain_masks[kind].eq(float(name))
            elif kind == "barra":
                mask = (self.domain_masks[kind][name[:-1]].eq(1 if name[-1] == "+" else -1)
                        if name.endswith(("+", "-")) else self.domain_masks[kind][name].isin([1, -1]))
            else:
                mask = self.domain_masks[kind][name].eq(1)
        if method == "equal":
            weight = mask.astype(float)
        elif method == "market_cap":
            self.require("market_cap")
            weight = self.fields["market_cap"].where(mask, 0)
        elif method == "index":
            weight = self.domain_weights["index"][index_name].where(mask, 0)
        else:
            raise ValueError('未知基准方法')
        self.benchmark_weight = m.normalize(weight)
        self.benchmark_return = m.weighted_returns(self.benchmark_weight.shift(1).fillna(0), self.pct)
        self.benchmark_nav = m.nav(self.benchmark_return)
        return self.benchmark_return

    def cal_groups(self, domain="universe"):
        """计算域内分组和组内等权收益。
        Params: domain: 域名；Return: 分组收益DataFrame，Average为有效分组的等权平均。
        并列因子保持同组，常数截面不分组。
        """
        if not hasattr(self, "alpha"):
            self.process_alpha()
        key = (id(self.alpha), domain, self.groups, id(self.domain_masks.get(domain)))
        if getattr(self, "_group_cache_key", None) == key:
            return self.group_returns
        if domain == "universe":
            mask = self.pct.notna()
        else:
            kind, name = domain if isinstance(domain, tuple) else domain.split(":", 1)
            if kind == "industry":
                mask = self.domain_masks[kind].eq(float(name))
            elif kind == "barra":
                mask = (self.domain_masks[kind][name[:-1]].eq(1 if name[-1] == "+" else -1)
                        if name.endswith(("+", "-")) else self.domain_masks[kind][name].isin([1, -1]))
            else:
                mask = self.domain_masks[kind][name].eq(1)
        x = self.alpha.where(mask)
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
        """选择分组并生成归一化目标持仓。
        Params: construction: long/short/longshort；group_numbers: 分组编号；
            normalization: equal/market_cap/factor；domain: 域名；
            benchmark_enhanced: 是否叠加基准；active_gross: 主动总敞口；
            benchmark_method/benchmark_index: 基准配置。
        Return: 信号日目标权重DataFrame，空头权重为负。
        """
        construction = construction or self.construction
        normalization = normalization or self.normalization
        if construction not in ("long", "short", "longshort") or active_gross < 0:
            raise ValueError('构建方式或active_gross无效')
        self.cal_groups(domain)
        if group_numbers is None:
            group_numbers = ((self.groups,), (1,)) if construction == "longshort" else ((self.groups,) if construction == "long" else (1,))
        long_groups, short_groups = (group_numbers if construction == "longshort" else
                                     (group_numbers, ()) if construction == "long" else ((), group_numbers))
        if set(long_groups) & set(short_groups):
            raise ValueError('多空分组不能重叠')
        if not set(long_groups).union(short_groups).issubset(range(1, self.groups + 1)):
            raise ValueError('组号超出范围')

        def leg(groups, sign):
            selected = self.group_codes.isin(groups)
            if normalization == "equal":
                raw = selected.astype(float)
            elif normalization == "market_cap":
                self.require("market_cap")
                raw = self.fields["market_cap"].where(selected, 0)
            elif normalization == "factor":
                # 组内分数平移为非负值，保留因子方向。
                score = (self.alpha * sign).where(selected)
                raw = score.sub(score.min(axis=1), axis=0).add(1e-8).where(selected, 0)
            else:
                raise ValueError('未知归一化方法')
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
        """成交和容量模拟入口。
        Params: 换手、偏离和持仓上限为NAV比例；费用为小数；
            execution_price: 次日open/vwap；capital: 初始资金（元）；
            max_participation: 成交额参与上限；impact_coefficient: 平方根冲击系数；
            lot_size: 每手股数；borrow_rate: 年化借券成本；其余参数见cal_target_holding。
        Return: PortfolioResult，保存至self.portfolio。
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
        """兼容原需求的函数拼写。
        Params: 同cal_portfolio；Return: PortfolioResult。
        """
        return self.cal_portfolio(*args, **kwargs)


factorbase = FactorBase
