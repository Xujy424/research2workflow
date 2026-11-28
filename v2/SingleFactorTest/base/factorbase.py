"""作者：xujiayi | 创建日期：2026-11-11
单因子研究上下文：数据、预处理、标签、分组、持仓和成交模拟。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
if __package__:
    from . import maths as m
    from .backtest import simulate
    from .backtest import PortfolioResult
    from .strategy import build_target
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base import maths as m
    from v2.SingleFactorTest.base.backtest import simulate
    from v2.SingleFactorTest.base.backtest import PortfolioResult
    from v2.SingleFactorTest.base.strategy import build_target
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
    "pb_lf": "fundamental/valuation/pb_lf",
    "adjustment": "d_essentials/ratio_adjusting_factor",
    "hastrade": "basic/hastrade",
}

INDEX_FIELDS = {name: f"index/weight/{name}_weight"
                for name in ("hs300", "zz500", "a500", "zz1000", "zz2000")}
BARRA_FIELDS = {name: f"barra/{name}"
                for name in (
                    "beta", "btop", "size", "nonlinear_size", "momentom",
                    "residual_vol", "liquidity", "leverage", "growth1", "earnings_yield"
                )}


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
            normalization="factor",    # equal / market_cap / factor
            transform="asinh",        # softsign / piecewise / rank_gaussian / yeo_johnson / none
            neutralize=True,
            standardize=True,
            num_groups=10,
            direction=1,
            start=None, end=None, history_years=6,
            root=ROOT, data=None
        ):
        # 因子名称、单边或双边、因子方向、分组数
        self.factor_name, self.construction = factor_name, construction 
        self.direction, self.num_groups = direction, num_groups
        # 归一化、变换、中性化、标准化
        self.normalization, self.transform = normalization, transform
        self.neutralize, self.standardize = neutralize, standardize
        # 报告日期、历史窗口
        self.start, self.end, self.history_years = start, end, history_years
        # 数据根目录、数据源
        self.root, self.data = root, data

        # 读取的字段、域掩码、域权重、Barra头尾组掩码
        self.fields, self.domain_masks, self.domain_weights = {}, {}, {}

    def _read_field(self, path):
        """读取当前区间的DataPool字段。
        Params: path: 字段路径；Return: 日期×股票的DataFrame。
        """
        values = self.data.read(path, self.dates[-1], self.dates[0], copy=False)
        return pd.DataFrame(values, index=self.dates, columns=self.ticks)

    def load(self):
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
        self.fields['barra'] = {name: self._read_field(path) for name, path in BARRA_FIELDS.items()}
        self.cal_mask()
        self.cal_weight()
        self.report_start = pd.Timestamp(self.start) if self.start else pd.Timestamp(self.dates[-1].year - 5, 1, 1)
        # 重载时清除派生结果。
        for name in ("alpha", "labels", "ics", "rankics", "sign_ics", "group_codes", "group_returns",
                     "group_weights", "target_holding", "portfolio", "capacity_results"):
            self.__dict__.pop(name, None)
        return self
    
    def cal_mask(self):
        """读取本地股票池、指数、行业和Barra掩码。
        Params: 无；Return: 域掩码字典，行业为编码矩阵，Barra为正负编码矩阵。
        """
        self.domain_masks = {
            "universe": self.fields["tradable_t"].eq(1),
            "index": {name: self._read_field(f"index/mask/{name}_mask")
                      for name in INDEX_FIELDS},
            "industry": self.fields["industry"],
            "barra": {name: self._read_field(f"{path}_mask")
                      for name, path in BARRA_FIELDS.items()},
        }
        return self.domain_masks

    def cal_weight(self):
        """计算各域权重；股票池按等权归一，行业和Barra按流通市值归一，指数读取本地权重。
        Params: 无；Return: 域权重字典。行业编码0—30，共31类；Barra为正组减负组。
        """
        market_cap = self.fields["market_cap"]
        self.domain_weights = {
            "universe": {},
            "index": {name: self._read_field(INDEX_FIELDS[name])
                      for name in self.domain_masks["index"]},
            "industry": {}, "barra": {},
        }
        universe = self.domain_masks["universe"].astype(float)
        self.domain_weights["universe"] = universe.div(
            universe.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0)
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
        """计算Barra因子收益，使用前日因子暴露和当日收益，行业和市值中性化。"""
        styles = list(BARRA_FIELDS)
        returns = self.pct.to_numpy(dtype=float, copy=False)
        exposures = np.stack([
            self.fields["barra"][name].to_numpy(dtype=float, copy=False)
            for name in styles
        ], axis=2)
        industry = self.fields["industry"].to_numpy(dtype=float, copy=False)
        market_cap = self.fields["market_cap"].to_numpy(dtype=float, copy=False)

        factor_returns = np.full((len(self.dates), len(styles)), np.nan)
        for day in range(1, len(self.dates)):
            x, y = exposures[day - 1], returns[day]
            codes, mv = industry[day - 1], market_cap[day - 1]
            valid = (
                np.isfinite(y) 
                & np.isfinite(x).all(axis=1)
                & np.isin(codes, np.arange(31)) 
                & np.isfinite(mv) & (mv > 0)
            )
            x, y, codes, mv = x[valid], y[valid], codes[valid].astype(np.int64), mv[valid]
            # 与包含行业哑变量的 WLS 保持一致：样本数须大于参数数。
            if len(y) <= len(styles) + np.unique(codes).size:
                continue
            w = np.sqrt(mv)
            w_sum = np.bincount(codes, weights=w, minlength=31)
            y_sum = np.bincount(codes, weights=w*y, minlength=31)
            y_mean = np.divide(
                y_sum, w_sum,
                out=np.zeros_like(y_sum),
                where=w_sum > 0
            )
            y_res = y - y_mean[codes]
            x_res = np.empty_like(x)

            for k in range(len(styles)):
                x_sum = np.bincount(codes, weights=w*x[:, k], minlength=31)
                x_mean = np.divide(
                    x_sum, w_sum,
                    out=np.zeros_like(x_sum),
                    where=w_sum > 0
                )
                x_res[:, k] = x[:, k] - x_mean[codes]

            scale = np.sqrt(w)
            xw = x_res * scale[:, None]
            yw = y_res * scale
            coefficients, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
            if rank == len(styles):
                factor_returns[day] = coefficients

        result = pd.DataFrame(factor_returns, index=self.dates, columns=styles)
        self.barra_returns = result
        return result


    def process_alpha(self):
        """执行因子变换、行业市值中性化和截面标准化。
        Params: 无，使用实例配置；Return: DataFrame，并保存self.alpha。
        """
        values = m.transform(
            self.raw_alpha.where(self.pct.notna()) * self.direction, 
            self.transform
        )
        if self.neutralize:
            values = m.neutralize(
                values, 
                self.fields.get("industry"),
                self.fields.get("market_cap")
            )
        self.alpha = m.standardize(values) if self.standardize else values
        self.alpha = pd.DataFrame(self.alpha, index=self.dates, columns=self.ticks)

        # 清除旧因子计算结果对应的缓存，防止重新执行 process_alpha() 后，后续回测仍然使用上一次的结果
        self._group_cache_key = None
        for name in ("labels", "ics", "rankics", "sign_ics"):
            self.__dict__.pop(name, None)

        return self.alpha


    def cal_labels(self, horizons=(1, 3, 5, 10, 20)):
        """t行对应t+2/t+1起的未来收益，尾部不完整窗口保留NaN。
        Params: horizons: 交易日窗口；
        Return: 标签字典。
        """
        if not hasattr(self, "labels"):
            self.labels = {}
        for h in horizons:
            if h in self.labels: continue
            values = m.forward_returns(self.pct, horizon=h, start=2)
            self.labels[h] = pd.DataFrame(values, index=self.dates, columns=self.ticks)
        return self.labels
        
    def cal_ics(self, horizons=(1, 3, 5, 10, 20)):
        """计算IC和RankIC。
        Params: horizons: 交易日窗口；
        Return: DataFrame，列为horizon，行索引为日期。
        """
        if not hasattr(self, "alpha"):
            self.process_alpha()
        if not hasattr(self, "labels"):
            self.cal_labels(horizons)

        self.ics, self.rankics, self.sign_ics = {}, {}, {}
        for h in horizons:
            if (h in self.ics and h in self.rankics and h in self.sign_ics): continue
            y = self.labels[h]

            self.ics[h] = m.cross_ic(self.alpha, y)
            self.rankics[h] = m.cross_ic(self.alpha, y, True)
            self.sign_ics[h] = pd.DataFrame({
                "pos_ic": m.cross_ic(self.alpha.where(y.to_numpy() > 0).to_numpy(), y.to_numpy()),
                "neg_ic": m.cross_ic(self.alpha.where(y.to_numpy() < 0).to_numpy(), y.to_numpy()),
                "rank_pos_ic": m.cross_ic(self.alpha.where(y.to_numpy() > 0).to_numpy(), y.to_numpy(), True),
                "rank_neg_ic": m.cross_ic(self.alpha.where(y.to_numpy() < 0).to_numpy(), y.to_numpy(), True),
            })

        return self.ics, self.rankics, self.sign_ics


    def cal_benchmark(self):
        """计算全市场、指数、行业及Barra基准收益和净值。
        权重来源：
            universe: 全市场等权、市值加权
            index:    self.domain_weights["index"]
            industry: self.domain_weights["industry"]
            barra:    self.domain_weights["barra"]
        时序：
            T日基准收益 = T-1日权重 × T日股票收益
        Return:
            self.benchmark = {
                "weight": {domain: {name: DataFrame}},
                "ret": {domain: {name: Series}},
                "nav": {domain: {name: Series}}
            }
        """
        pct = self.pct
        weights = self.domain_weights

        self.benchmark = {
            "weight": {},
            "ret": {},
            "nav": {},
        }
        for domain, domain_weights in weights.items():
            self.benchmark["weight"][domain] = {}
            self.benchmark["ret"][domain] = {}
            self.benchmark["nav"][domain] = {}

            if isinstance(domain_weights, pd.DataFrame):
                named_weights = {"equal": domain_weights}
                if domain == "universe":
                    market_cap = self.fields["market_cap"]
                    valid = pct.notna() & np.isfinite(market_cap) & (market_cap > 0)
                    cap_weight = market_cap.where(valid, 0)
                    named_weights["market_cap"] = cap_weight.div(
                        cap_weight.sum(axis=1).replace(0, np.nan), axis=0
                    ).fillna(0)
            else:
                named_weights = domain_weights
            for name, weight in named_weights.items():
                self.benchmark["weight"][domain][name] = weight.copy()
                # 使用前一交易日权重
                w = weight.shift(1).fillna(0)
                # 检查是否存在有效持仓
                active = w.abs().sum(axis=1) > 0
                # 缺失收益按0计入，避免单只股票缺数导致整个基准日收益缺失。
                # Barra支持正负权重；长仓域基准权重已在前一日归一化。
                ret = (w * pct.fillna(0)).sum(axis=1)
                ret = ret.where(active)
                # 计算累计净值
                nav = (1 + ret.fillna(0)).cumprod()
                nav = nav.where(ret.notna())
                # 保存结果
                self.benchmark["ret"][domain][name] = ret
                self.benchmark["nav"][domain][name] = nav

        return self.benchmark


    def cal_groups(self, num_groups=10, domain="universe", name=None, horizon=(1,5,10,20)):
        """计算域内因子分组及各组未来 horizon 日等权累计收益。
        G1 为最小的方向调整后 Alpha，G{num_groups} 为最大的一组。
        T 日分组直接匹配 cal_labels() 中 T 日的未来标签；
        Params: 
            num_groups: 分组数；
            domain/name: 分组域及名称；
            horizon: 单个周期或周期序列。
        Return: horizon 到分组收益 DataFrame 的字典。
        """
        if not hasattr(self, "alpha"):
            self.process_alpha()
        horizons = (horizon,) if isinstance(horizon, (int, np.integer)) else tuple(horizon)
        if not horizons or any(not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
            raise ValueError("horizon 必须是正整数或正整数序列")
        if not isinstance(num_groups, (int, np.integer)) or num_groups < 2:
            raise ValueError("num_groups 必须是大于等于 2 的整数")

        # 1. 分组域（与原版本一致）
        if domain == "universe":
            domain_mask = self.domain_masks["universe"].eq(1)
        elif domain == "index":
            domain_mask = self.domain_masks["index"][name].eq(1)
        elif domain == "industry":
            domain_mask = self.domain_masks["industry"].eq(int(name))
        else:
            raise ValueError(f"未知分组域: {domain}")

        # 2. 缓存必须包含 horizon；同组号在不同 horizon 下对应不同收益
        key = (id(self.alpha), domain, name, num_groups, horizons)
        if getattr(self, "_group_cache_key", None) == key:
            return self.group_returns

        # 3. 默认升序：因子值越大，排名及组号越大
        x = self.alpha.where(domain_mask)
        count = x.count(axis=1)
        rank = x.rank(axis=1, method="average", ascending=True)
        codes = np.ceil(
            rank.sub(0.5).div(count.replace(0, np.nan), axis=0) * num_groups
        ).clip(1, num_groups)
        self.group_codes = codes.where(x.nunique(axis=1).gt(1), axis=0)

        # 4. T 日组号匹配 T 日未来 horizon 日收益，不再 shift(1)
        if not hasattr(self, "labels") or any(h not in self.labels for h in horizons):
            self.cal_labels(horizons)
        self.group_returns = {}
        for h in horizons:
            future_ret = self.labels[h]
            returns = {
                f"G{g}": future_ret.where(self.group_codes.eq(g)).mean(axis=1)
                for g in range(1, num_groups + 1)
            }
            result = pd.DataFrame(returns, index=self.dates)
            result["Average"] = result.mean(axis=1)
            self.group_returns[h] = result
        self._group_cache_key = key
        return self.group_returns


    def cal_target_holding(
            self,
            strategy="enhanced",
            construction="longshort",
            num_groups=None,
            group_numbers=(10, 1),
            normalization="factor",
            domain="universe",
            name=None,
            industry_neutral=True,
            active_gross=1.0,
            hedge_method="dollar",
            hedge_window=60,
            hedge_min_periods=40,
    ):
        """根据因子分组生成目标持仓，并保存为实例变量。"""
        self.target_holding = build_target(
            self,
            strategy=strategy,
            construction=construction,
            group_numbers=group_numbers,
            normalization=normalization,
            domain=domain,
            name=name,
            industry_neutral=industry_neutral,
            active_gross=active_gross,
            num_groups=num_groups if num_groups is not None else self.num_groups,
            hedge_method=hedge_method,
            hedge_window=hedge_window,
            hedge_min_periods=hedge_min_periods,
        )
        # Shift signal weights to the next trading day here, so simulate()
        # receives execution-day targets and does not need to shift them again.
        target = self.target_holding.shift(1)
        if strategy == "enhanced":
            # Enhanced portfolios start with the index benchmark already held.
            benchmark = self.benchmark_weight.reindex(
                index=self.target_holding.index,
                columns=self.target_holding.columns,
            )
            target.iloc[0] = benchmark.iloc[0].fillna(0)
        elif strategy in ("absolute", "market_neutral"):
            target.iloc[0] = 0.
        self.target_holding = target.fillna(0)
        return self.target_holding

    def cal_portfolio(
            self,
            strategy="enhanced",
            construction="longshort",
            num_groups=None,
            group_numbers=(10, 1),
            normalization="factor",
            domain="universe",
            name=None,
            industry_neutral=True,
            active_gross=1.0,
            hedge_method="dollar",
            hedge_window=60,
            hedge_min_periods=40,
            commission=0.003,
            stamp_tax=0.0005,
            execution_price="vwap",
            capital=1e8,
            max_participation=0.1,
            impact_coefficient=0.001,
            lot_size=100,
            borrow_rate=0.0,
            rebalance_days=5,
    ):
        """T日生成目标权重，T+1执行；不进行权重优化。"""

        # Rebuild target weights from the parameters of every portfolio run.
        # This prevents stale holdings when a FactorBase instance is reused.
        self.cal_target_holding(
            strategy=strategy,
            construction=construction,
            num_groups=num_groups,
            group_numbers=group_numbers,
            normalization=normalization,
            domain=domain,
            name=name,
            industry_neutral=industry_neutral,
            active_gross=active_gross,
            hedge_method=hedge_method,
            hedge_window=hedge_window,
            hedge_min_periods=hedge_min_periods,
        )
        self.strategy, self.domain, self.name = strategy, domain, name
        self.capital = capital

        # 2. 模拟实际成交及收益
        self.portfolio = simulate(
            self,
            self.target_holding,
            commission=commission,
            stamp_tax=stamp_tax,
            execution_price=execution_price,
            capital=capital,
            max_participation=max_participation,
            impact_coefficient=impact_coefficient,
            lot_size=lot_size,
            borrow_rate=borrow_rate,
            rebalance_days=rebalance_days,
        )
        return self.portfolio


factorbase = FactorBase
