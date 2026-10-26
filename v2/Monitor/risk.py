"""Risk pressure monitor."""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
try:
    from .context import MarketContext
    from .utils import historical_percentile, weighted_mean
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.context import MarketContext
    from v2.Monitor.utils import historical_percentile, weighted_mean

PERIODS=(1,5,10,20)

def _pct(x): return np.asarray(x,float)/100.0
def _p(x,h=252): return historical_percentile(np.asarray(x,float)[-h:])

def cross_section_dispersion(returns, period=1, history=252):
    """横截面收益离散度：每日股票收益标准差，再按周期取均值并计算历史分位。"""
    r=_pct(returns); daily=np.nanstd(r,axis=1,ddof=1)
    return float(np.nanmean(daily[-period:])), _p(daily,history)

def _summarize(vals, period, history):
    return (float(np.nanmean(vals[-period:])), _p(vals, history)) if len(vals) else (np.nan, np.nan)


def _mean_pair_correlation(x):
    """Exact off-diagonal mean in O(T*N), without an N*N matrix.

    For centered unit-norm columns z_j, sum(i != j) corr(i,j)
    equals ||sum_j z_j||**2 - n. Constant/empty columns are excluded,
    matching nanmean(corrcoef(...)) after per-column median imputation.
    """
    x = np.asarray(x, dtype=float)
    x = x[:, np.isfinite(x).any(axis=0)]
    if x.shape[1] < 2:
        return np.nan
    if not np.isfinite(x).all():
        finite = np.isfinite(x)
        medians = np.nanmedian(np.where(finite, x, np.nan), axis=0)
        x = np.where(finite, x, medians)
    x = x - x.mean(axis=0)
    norms = np.linalg.norm(x, axis=0)
    valid = np.isfinite(norms) & (norms > 0)
    n = int(valid.sum())
    if n < 2:
        return np.nan
    zsum = (x[:, valid] / norms[valid]).sum(axis=1)
    return float(np.clip((zsum @ zsum - n) / (n * (n - 1)), -1, 1))


def _correlation_series(returns, window, downside=False, market_returns=None):
    r = _pct(returns)
    vals = []
    if downside:
        r = np.where(np.isfinite(r), r, np.nan)
    market = (np.nanmean(r, axis=1) if market_returns is None
              else np.asarray(market_returns, dtype=float)) if downside else None
    for i in range(window, len(r) + 1):
        x = r[i-window:i]
        if downside:
            m = market[i-window:i]
            finite = np.isfinite(m)
            x = x[finite & (m <= np.quantile(m[finite], 0.20))] if finite.any() else x[:0]
            if len(x) < 3:
                vals.append(np.nan)
                continue
        vals.append(_mean_pair_correlation(x))
    return np.asarray(vals)


def average_correlation(returns, window=60, period=1, history=252):
    """Rolling mean pair correlation via normalized column sums."""
    return _summarize(_correlation_series(returns, window), period, history)


def downside_correlation(returns, window=120, period=1, history=252, market_returns=None):
    """Rolling correlation on the lowest 20% of benchmark-return days.

    market_returns, when supplied, must be aligned daily decimal returns.
    """
    return _summarize(_correlation_series(returns, window, downside=True,
                                        market_returns=market_returns), period, history)


def _pca_series(returns, window):
    r = _pct(returns)
    vals = []
    for i in range(window, len(r) + 1):
        x = r[i-window:i]
        finite = np.isfinite(x)
        # Require at least 80% finite observations (round up).
        keep = finite.sum(axis=0) >= max(1, int(np.ceil(window * .8)))
        x = x[:, keep]
        finite = finite[:, keep]
        if x.shape[1] < 2:
            vals.append(np.nan)
            continue
        # Fill missing returns with each stock's finite-observation mean.
        # After centering, those entries are zero; observed deviations survive.
        means = np.where(finite, x, 0.0).sum(axis=0) / finite.sum(axis=0)
        x = np.where(finite, x, means) - means
        x -= x.mean(axis=0)
        norms = np.linalg.norm(x, axis=0)
        valid = np.isfinite(norms) & (norms > 0)
        if valid.sum() < 2:
            vals.append(np.nan)
            continue
        # Unit-norm columns give correlation PCA, rather than covariance PCA.
        x = x[:, valid] / norms[valid]
        # X.T @ X and X @ X.T have the same nonzero eigenvalues.
        # The covariance normalization cancels in lambda_max / trace.
        gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
        total = np.trace(gram)
        vals.append(float(np.linalg.eigvalsh(gram)[-1] / total) if total > 0 else np.nan)
    return np.asarray(vals)


def pca_explained_ratio(returns, window=60, period=1, history=252):
    """Leading correlation eigenvalue share via the smaller Gram matrix."""
    return _summarize(_pca_series(returns, window), period, history)


def downside_resonance(returns, hit_floor=None, has_trade=None, period=1, history=252,
                      limit_eligible=None):
    """下行共振：负收益、极端下跌和跌停扩散三个分量等权平均。"""
    r = _pct(returns)
    valid = np.isfinite(r)
    if has_trade is not None:
        valid &= np.isfinite(has_trade) & (np.asarray(has_trade) > 0)
    counts = valid.sum(axis=1)
    neg = np.divide(((r < 0) & valid).sum(1), counts,
                    out=np.full(len(r), np.nan), where=counts > 0)
    extreme = np.divide(((r < -.05) & valid).sum(1), counts,
                        out=np.full(len(r), np.nan), where=counts > 0)
    limit = (np.full(len(r), np.nan) if hit_floor is None else
             _limit_down_series(hit_floor, has_trade, limit_eligible))
    # All three components are required by the research definition.
    vals = (neg + extreme + limit) / 3
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def worst_rolling_loss(index_returns, lookback=252, loss_window=5, history=252):
    """最差滚动5日收益：过去回看窗口内5日复合收益的最低值。"""
    r=_pct(index_returns).reshape(-1)
    five=pd.Series(r).rolling(loss_window).apply(lambda x: np.prod(1+x)-1).to_numpy()
    vals=pd.Series(five).rolling(lookback).min().to_numpy()
    return float(vals[-1]) if len(vals) else np.nan,_p(vals,history)

def _limit_down_series(hit_floor, has_trade=None, limit_eligible=None):
    hit = np.asarray(hit_floor, dtype=float)
    valid = np.isfinite(hit)
    if has_trade is not None:
        valid &= np.isfinite(has_trade) & (np.asarray(has_trade) > 0)
    if limit_eligible is not None:
        valid &= np.isfinite(limit_eligible) & (np.asarray(limit_eligible) > 0)
    counts = valid.sum(1)
    return np.divide(((hit > 0) & valid).sum(1), counts,
                     out=np.full(len(hit), np.nan), where=counts > 0)


def limit_down_ratio(hit_floor, has_trade=None, period=1, history=252, limit_eligible=None):
    """跌停扩散：跌停股票数 / 有效交易且适用涨跌幅限制股票数。"""
    vals = _limit_down_series(hit_floor, has_trade, limit_eligible)
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def volatility_risk_premium(iv, rv, period=1, history=252):
    """原表风险溢价代理 IV30²-RV30²；输入均为年化小数。"""
    x=np.asarray(iv,float)**2-np.asarray(rv,float)**2
    return float(np.nanmean(x[-period:])),_p(x,history)

def liquidity_pressure_percentile(spread, depth_decline, amihud, history=252):
    """先等权合成三个已标准化压力分位，再计算综合历史分位。"""
    frame = pd.concat([pd.Series(spread), pd.Series(depth_decline), pd.Series(amihud)], axis=1)
    score = frame.mean(axis=1, skipna=False)
    return (float(score.iloc[-1]), _p(score.to_numpy(), history)) if len(score) else (np.nan, np.nan)

def realized_volatility(index_returns, window=20, period=1, history=252):
    """实现波动率：窗口内日收益标准差年化，返回当前值及历史分位。"""
    vals=pd.Series(_pct(index_returns).reshape(-1)).rolling(window).std().to_numpy()*np.sqrt(252)
    return float(np.nanmean(vals[-period:])),_p(vals,history)


def _rv_metrics(decimal_returns):
    """原表交易日窗口；完整窗口才输出，零分母不生成无穷值。"""
    r = pd.Series(decimal_returns, dtype=float)
    rv = {h: r.rolling(h).std(ddof=1) * np.sqrt(252) for h in (5,20,60)}
    return {
        '实现波动率RV5': rv[5],
        '实现波动率RV20': rv[20],
        '实现波动率RV60': rv[60],
        '波动率期限结构RV5/RV20': rv[5] / rv[20].where(rv[20] > 0),
        '波动率期限结构RV20/RV60': rv[20] / rv[60].where(rv[60] > 0),
    }


class RiskContext(MarketContext):
    """风险模块数据上下文。

    风险指标使用 MarketContext 的交易日、股票矩阵、域权重和历史数据接口，
    当前没有额外的远端数据源，因此不覆盖父类取数逻辑。
    """

class RiskMonitor:
    DEFAULT_FIELDS={"returns":"d_essentials/pct", "hit_floor":"basic/hit_floor",
                    "has_trade":"basic/hastrade", "price_floor":"basic/price_floor"}
    def __init__(self, context, field_map=None):
        self.context = context
        self.fields = {**self.DEFAULT_FIELDS, **(field_map or {})}

    def option_volatility(self, asof, start, end, index_codes=("000300.SH", "000852.SH"), *,
                          surface_field="品种隐含波动率曲面(%)",
                          surface_options="Period=D;期限={tenor}天;价值状态={moneyness}%",
                          surface_scale=0.01, points=((30,100),)):
        """固定期限/价值状态曲面，列为(code, tenor, moneyness)，值为年化小数。

        价值状态方向须由 Wind 元数据确认；这里原样传入，不推断看涨/看跌。
        缺失数据保留 NaN，不跨日期前向填充。asof 限制查询截止日期。
        字段与参数可按 Wind 生成的 API 调用覆盖；surface_scale 将百分数转为小数。
        """
        end = min(pd.Timestamp(asof), pd.Timestamp(end))
        out = {}
        for code in index_codes:
            missing = []
            for tenor, moneyness in points:
                options = surface_options.format(tenor=tenor, moneyness=moneyness)
                try:
                    x = self.context.wsd(code, surface_field, start, end,
                                         options=options, strict=True)
                except Exception as exc:
                    warnings.warn(f'IV unavailable: {code}, {tenor}, {moneyness}: {exc}',
                                  RuntimeWarning, stacklevel=2)
                    x = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
                x = pd.to_numeric(x, errors='coerce') * surface_scale
                x = x.where(np.isfinite(x) & (x >= 0))
                out[(code,tenor,moneyness)] = x
                if not x.notna().any():
                    missing.append((tenor, moneyness))
            if missing:
                warnings.warn(f'WSD surface unavailable: {code}, points={missing}; '
                              f'check field/options/permissions: {surface_field}',
                              RuntimeWarning, stacklevel=2)
        return pd.DataFrame(out)

    @staticmethod
    def _volatility_metrics(surface, close):
        """原表：RV5/20/60、期限比值、IV30 与方差风险溢价代理。

        RV 按交易日、简单收益样本标准差计算，统一年化小数。
        代理的 RV30 为30个交易日，与 IV30 的日历期限并非严格匹配。
        """
        close = close.where(np.isfinite(close) & (close > 0)).sort_index()
        returns = close.pct_change(fill_method=None)
        metrics = _rv_metrics(returns)
        atm = surface[(30,100)]
        rv30 = returns.rolling(30).std(ddof=1) * np.sqrt(252)
        metrics['指数隐含波动率IV30'] = atm
        metrics['波动率风险溢价代理IV30²-RV30²'] = atm**2 - rv30**2
        return metrics
    def snapshot(self,asof,history=252,periods=PERIODS,domains=("全市场",), micro_result=None):
        c=self.context; 
        end=c.position(asof); 
        start=max(0,end-history-260); 
        r=c.matrix(self.fields["returns"],start,end)
        floor=c.matrix(self.fields["hit_floor"],start,end)
        trade=c.matrix(self.fields["has_trade"],start,end)
        price_floor=c.matrix(self.fields["price_floor"],start,end)
        eligible = np.isfinite(price_floor) & (price_floor > 0)
        dates = pd.DatetimeIndex(c.dates[start:end+1])
        rows=[]
        for domain in domains:
            w=c.domain_weights(start,end,domain)
            member = np.isfinite(w) & (w > 0)
            active = member & np.isfinite(trade) & (trade > 0)
            dr = np.where(active & np.isfinite(r), r, np.nan)
            domain_trade = np.where(member, trade, 0)
            # Domain benchmark uses current returns with prior-day weights.
            ir=np.array([weighted_mean(_pct(dr[t]),w[t-1]) if t else np.nan for t in range(len(r))])
            correlation = _correlation_series(dr, 60)
            downside = _correlation_series(dr, 120, downside=True, market_returns=ir)
            pca = _pca_series(dr, 60)
            for p in periods:
                for n,fn in (
                    ("横截面收益离散度",lambda:cross_section_dispersion(dr,p,history)),
                    ("平均收益相关性",lambda:_summarize(correlation,p,history)),
                    ("下跌条件相关性",lambda:_summarize(downside,p,history)),
                    ("首主成分解释率",lambda:_summarize(pca,p,history)),
                    ("市场下行共振",lambda:downside_resonance(dr,floor,domain_trade,p,history,eligible)),
                    ("跌停扩散比例",lambda:limit_down_ratio(floor,domain_trade,p,history,eligible))
                ):
                    v,h=fn(); rows.append({"指标":n,"域":domain,"周期":p,"数值":v,"历史分位":h})
            # worst_rolling_loss takes percent returns; ir is already decimal.
            v,h = worst_rolling_loss(ir * 100, history, 5, history)
            rows.append({"指标":"最差滚动5日收益","域":domain,"周期":1,"数值":v,"历史分位":h})
            for name, series in _rv_metrics(pd.Series(ir, index=dates)).items():
                rows.append({"指标":name,"域":domain,"周期":1,
                             "数值":float(series.iloc[-1]),"历史分位":_p(series.to_numpy(),history)})
        # Use constant-tenor surface points and official index close returns.
        for code, label in (("000300.SH", "沪深300"), ("000852.SH", "中证1000")):
            vol = self.option_volatility(asof, c.dates[start], c.dates[end], (code,))
            surface = vol[code].reindex(dates)
            try:
                close = c.wsd(code, 'close', c.dates[max(0, start-65)], c.dates[end], strict=True)
            except Exception as exc:
                warnings.warn(f'Index close unavailable: {code}: {exc}', RuntimeWarning, stacklevel=2)
                close = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            metrics = self._volatility_metrics(surface, close)
            for name, series in metrics.items():
                if label in domains and name in _rv_metrics(pd.Series(dtype=float)):
                    continue
                x = series.reindex(dates)
                # Missing as-of observations stay unavailable, rather than
                # presenting an older quote as today's value.
                rows.append({"指标": name, "域": label, "周期": 1,
                             "数值": float(x.iloc[-1]) if len(x) else np.nan,
                             "历史分位": _p(x.to_numpy(), history)})
        if micro_result is not None:
            required = ("相对价差分位", "深度下降分位", "Amihud分位")
            if all(k in micro_result for k in required):
                v,h=liquidity_pressure_percentile(*(micro_result[k] for k in required), history)
                rows.append({"指标":"流动性压力历史分位","域":"全市场","周期":1,"数值":v,"历史分位":h})
        return pd.DataFrame(rows).set_index(["指标","域","周期"])

    @staticmethod
    def table(result, decimals=6):
        return result.round(decimals)


if __name__ == "__main__":
    context = RiskContext()
    try:
        result = RiskMonitor(context).snapshot(
            "2026-09-30",
            domains=("全市场",),
        )
        print(RiskMonitor.table(result).to_string(
            float_format=lambda value: f"{value:.6f}"
        ))
    finally:
        context.close()


