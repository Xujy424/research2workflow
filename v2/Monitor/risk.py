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
    r=_pct(returns); daily=np.nanstd(r,axis=1)
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


def _correlation_series(returns, window, downside=False):
    r = _pct(returns)
    vals = []
    if downside:
        r = np.where(np.isfinite(r), r, np.nan)
    market = np.nanmean(r, axis=1) if downside else None
    for i in range(window, len(r) + 1):
        x = r[i-window:i]
        if downside:
            m = market[i-window:i]
            x = x[m <= np.nanquantile(m, 0.10)]
            if len(x) < 3:
                vals.append(np.nan)
                continue
        vals.append(_mean_pair_correlation(x))
    return np.asarray(vals)


def average_correlation(returns, window=60, period=1, history=252):
    """Rolling mean pair correlation via normalized column sums."""
    return _summarize(_correlation_series(returns, window), period, history)


def downside_correlation(returns, window=120, period=1, history=252):
    """Rolling correlation on the lowest 10% of market-return days."""
    return _summarize(_correlation_series(returns, window, downside=True), period, history)


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
        # X.T @ X and X @ X.T have the same nonzero eigenvalues.
        # The covariance normalization cancels in lambda_max / trace.
        gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
        total = np.trace(gram)
        vals.append(float(np.linalg.eigvalsh(gram)[-1] / total) if total > 0 else np.nan)
    return np.asarray(vals)


def pca_explained_ratio(returns, window=60, period=1, history=252):
    """Leading covariance eigenvalue share via the smaller Gram matrix."""
    return _summarize(_pca_series(returns, window), period, history)


def downside_resonance(returns, hit_floor=None, has_trade=None, period=1, history=252):
    """下行共振：负收益、极端下跌和跌停扩散三个分量等权平均。"""
    r=_pct(returns); neg=np.nanmean(r<0,axis=1); extreme=np.nanmean(r<-0.05,axis=1)
    if hit_floor is None: limit=np.full(len(r), np.nan)
    else:
        hit=np.asarray(hit_floor,bool); valid=np.ones_like(hit,bool) if has_trade is None else np.asarray(has_trade,bool)
        limit=np.divide((hit&valid).sum(1),valid.sum(1),out=np.full(len(hit),np.nan),where=valid.sum(1)>0)
    vals=np.nanmean(np.vstack([neg,extreme,limit]),axis=0)
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def worst_rolling_loss(index_returns, lookback=252, loss_window=5, history=252):
    """最差滚动5日收益：过去回看窗口内5日复合收益的最低值。"""
    r=_pct(index_returns).reshape(-1)
    five=pd.Series(r).rolling(loss_window).apply(lambda x: np.prod(1+x)-1).to_numpy()
    vals=pd.Series(five).rolling(lookback).min().to_numpy()
    return float(vals[-1]) if len(vals) else np.nan,_p(vals,history)

def limit_down_ratio(hit_floor, has_trade=None, period=1, history=252):
    """跌停扩散比例：一字跌停且有效交易股票数占有效交易股票数的比例。"""
    hit=np.asarray(hit_floor,bool); valid=np.ones_like(hit,bool) if has_trade is None else np.asarray(has_trade,bool)
    vals=np.divide((hit&valid).sum(1),valid.sum(1),out=np.full(hit.shape[0],np.nan),where=valid.sum(1)>0)
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def volatility_risk_premium(iv, rv, period=1, history=252):
    """波动率风险溢价：隐含波动率减历史实现波动率。"""
    x=np.asarray(iv,float)-np.asarray(rv,float)
    return float(np.nanmean(x[-period:])),_p(x,history)

def liquidity_pressure_percentile(spread, depth_decline, amihud, history=252):
    """流动性压力综合分位：对价差、深度下降、Amihud三个压力序列等权标准化。"""
    frame = pd.concat([pd.Series(spread), pd.Series(depth_decline), pd.Series(amihud)], axis=1)
    z = frame.apply(lambda x: (x - x.rolling(history, min_periods=20).mean()) /
                    x.rolling(history, min_periods=20).std())
    score = z.mean(axis=1, skipna=True).dropna()
    return (float(score.iloc[-1]), _p(score.to_numpy(), history)) if len(score) else (np.nan, np.nan)

def realized_volatility(index_returns, window=20, period=1, history=252):
    """实现波动率：窗口内日收益标准差年化，返回当前值及历史分位。"""
    vals=pd.Series(_pct(index_returns).reshape(-1)).rolling(window).std().to_numpy()*np.sqrt(252)
    return float(np.nanmean(vals[-period:])),_p(vals,history)


class RiskContext(MarketContext):
    """风险模块数据上下文。

    风险指标使用 MarketContext 的交易日、股票矩阵、域权重和历史数据接口，
    当前没有额外的远端数据源，因此不覆盖父类取数逻辑。
    """

class RiskMonitor:
    DEFAULT_FIELDS={"returns":"d_essentials/pct","hit_floor":"basic/floor_only","has_trade":"basic/hastrade"}
    def __init__(self, context, field_map=None, *,
                 surface_field="品种隐含波动率曲面(%)",
                 surface_options="Period=D;期限={tenor}天;价值状态={moneyness}%",
                 surface_scale=0.01):
        self.context = context
        self.fields = {**self.DEFAULT_FIELDS, **(field_map or {})}
        # User-supplied Wind display name/options; replace with API-generated
        # identifiers if the data service requires them. Percent -> decimal.
        self.surface_field = surface_field
        self.surface_options = surface_options
        self.surface_scale = surface_scale

    def option_volatility(self, asof, start, end, index_codes=("000300.SH", "000852.SH")):
        """固定期限/价值状态曲面，列为(code, tenor, moneyness)，值为年化小数。

        价值状态方向须由 Wind 元数据确认；这里原样传入，不推断看涨/看跌。
        缺失数据保留 NaN，不跨日期前向填充。asof 限制查询截止日期。
        """
        end = min(pd.Timestamp(asof), pd.Timestamp(end))
        points = ((7,100),(30,90),(30,95),(30,100),(30,105),(60,100))
        out = {}
        for code in index_codes:
            missing = []
            for tenor, moneyness in points:
                options = self.surface_options.format(tenor=tenor, moneyness=moneyness)
                x = self.context.wsd(code, self.surface_field, start, end,
                                     options=options, strict=True)
                x = pd.to_numeric(x, errors='coerce') * self.surface_scale
                x = x.where(np.isfinite(x) & (x >= 0))
                out[(code,tenor,moneyness)] = x
                if not x.notna().any():
                    missing.append((tenor, moneyness))
            if missing:
                warnings.warn(f'WSD surface unavailable: {code}, points={missing}; '
                              f'check field/options/permissions: {self.surface_field}',
                              RuntimeWarning, stacklevel=2)
        return pd.DataFrame(out)

    @staticmethod
    def _volatility_metrics(surface, close):
        """Same-date surface spreads and trailing calendar-30-day RV proxy.

        RV uses daily log returns with 252-day annualization. It is a backward
        looking comparison, not an estimate of expected future variance premium.
        """
        metrics = {}
        for tenor, m in ((7,100),(30,90),(30,95),(30,100),(30,105),(60,100)):
            metrics[f'隐含波动率{tenor}天_{m}%'] = surface[(tenor,m)]
        atm = surface[(30,100)]
        metrics['95%-100%隐波差_30天'] = surface[(30,95)] - atm
        metrics['105%-100%隐波差_30天'] = surface[(30,105)] - atm
        metrics['95%-105%隐波差_30天'] = surface[(30,95)] - surface[(30,105)]
        metrics['隐波期限差7天-30天'] = surface[(7,100)] - atm
        metrics['隐波期限差30天-60天'] = atm - surface[(60,100)]
        close = close.where(np.isfinite(close) & (close > 0)).sort_index()
        log_returns = np.log(close / close.shift(1))
        rv = log_returns.rolling('30D', min_periods=15).std() * np.sqrt(252)
        metrics['历史实现波动率_过去30日历天'] = rv
        metrics['平值隐波30天-历史实现波动率'] = atm - rv
        return metrics
    def snapshot(self,asof,history=252,periods=PERIODS,domains=("全市场",), micro_result=None):
        c=self.context; 
        end=c.position(asof); 
        start=max(0,end-history-260); 
        r=c.matrix(self.fields["returns"],start,end)
        floor=c.matrix(self.fields["hit_floor"],start,end).astype(bool)
        trade=c.matrix(self.fields["has_trade"],start,end).astype(bool)
        # These unweighted market series are shared across periods and domains.
        correlation = _correlation_series(r, 60)
        downside = _correlation_series(r, 120, downside=True)
        pca = _pca_series(r, 60)
        rows=[]
        for domain in domains:
            w=c.domain_weights(start,end,domain)
            ir=np.array([weighted_mean(_pct(r[t]),w[t-1]) if t else np.nan for t in range(len(r))])
            for p in periods:
                for n,fn in (
                    ("横截面收益离散度",lambda:cross_section_dispersion(r,p,history)),
                    ("平均收益相关性",lambda:_summarize(correlation,p,history)),
                    ("下跌条件相关性",lambda:_summarize(downside,p,history)),
                    ("首主成分解释率",lambda:_summarize(pca,p,history)),
                    ("下行共振",lambda:downside_resonance(r,floor,trade,p,history)),
                    ("跌停扩散比例",lambda:limit_down_ratio(floor,trade,p,history)),
                    ("最差滚动5日收益",lambda:worst_rolling_loss(ir,history,5,history))
                ):
                    v,h=fn(); rows.append({"指标":n,"域":domain,"周期":p,"数值":v,"历史分位":h})
        # Use constant-tenor surface points and official index close returns.
        for code, label in (("000300.SH", "\u6caa\u6df1300"), ("000852.SH", "\u4e2d\u8bc11000")):
            vol = self.option_volatility(asof, c.dates[start], c.dates[end], (code,))
            dates = pd.DatetimeIndex(c.dates[start:end+1])
            surface = vol[code].reindex(dates)
            close = c.wsd(code, 'close', c.dates[max(0, start-40)], c.dates[end], strict=True)
            metrics = self._volatility_metrics(surface, close)
            for name, series in metrics.items():
                x = series.reindex(dates)
                # Missing as-of observations stay unavailable, rather than
                # presenting an older quote as today's value.
                rows.append({"\u6307\u6807": name, "\u57df": label, "\u5468\u671f": 1,
                             "\u6570\u503c": float(x.iloc[-1]) if len(x) else np.nan,
                             "\u5386\u53f2\u5206\u4f4d": _p(x.to_numpy(), history)})
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


