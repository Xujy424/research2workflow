"""Risk pressure monitor."""
from __future__ import annotations
import numpy as np
import pandas as pd
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

def average_correlation(returns, window=60, period=1, history=252):
    """平均收益相关性：滚动60日股票两两收益相关系数的非对角均值。"""
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; x=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0)); c=np.corrcoef(x.T); vals.append(np.nanmean(c[np.triu_indices_from(c,1)]))
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

def downside_correlation(returns, window=120, period=1, history=252):
    """在滚动120日中筛选市场收益最低10%的日期后计算相关性。"""
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; market=np.nanmean(x,axis=1); cutoff=np.nanquantile(market,0.10); x=x[market<=cutoff]
        if len(x)<3: vals.append(np.nan); continue
        x=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0)); c=np.corrcoef(x.T); vals.append(np.nanmean(c[np.triu_indices_from(c,1)]))
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

def pca_explained_ratio(returns, window=60, period=1, history=252):
    """首主成分解释率：滚动60日收益协方差矩阵第一特征值占比。"""
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; x=x[:,np.isfinite(x).sum(0)>=int(window*.8)]
        if x.shape[1]<2: vals.append(np.nan); continue
        x=np.nan_to_num(x-x.mean(0)); s=np.linalg.svd(np.cov(x,rowvar=False),compute_uv=False)
        vals.append(float(s[0]/s.sum()) if s.sum()>0 else np.nan)
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

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
    def __init__(self,context,field_map=None): self.context=context; self.fields={**self.DEFAULT_FIELDS,**(field_map or {})}

    def option_volatility(self, asof, start, end, index_codes=("000300.SH", "000852.SH")):
        """读取中证300/1000的 WSD 历史、隐含波动率及交易所隐含波动率。"""
        fields=("underlyinghisvol_30d","underlyinghisvol_60d","underlyinghisvol_90d",
                "us_impliedvol","mid_impliedvol","impliedvol_exch")
        out={}
        for code in index_codes:
            for field in fields:
                out[(code,field)]=self.context.wsd(code,field,start,end)
        return pd.DataFrame(out)
    def snapshot(self,asof,history=252,periods=PERIODS,domains=("全市场",), micro_result=None):
        c=self.context; end=c.position(asof); start=max(0,end-history-260); r=c.matrix(self.fields["returns"],start,end)
        floor=c.matrix(self.fields["hit_floor"],start,end).astype(bool)
        trade=c.matrix(self.fields["has_trade"],start,end).astype(bool)
        if r.shape[1] > 300:
            keep=np.linspace(0,r.shape[1]-1,300,dtype=int); r=r[:,keep]; floor=floor[:,keep]; trade=trade[:,keep]
        rows=[]
        for domain in domains:
            w=c.domain_weights(start,end,domain)
            if r.shape[1] != w.shape[1]: w=w[:,keep]
            ir=np.array([weighted_mean(_pct(r[t]),w[t-1]) if t else np.nan for t in range(len(r))])
            for p in periods:
                for n,fn in (("横截面收益离散度",lambda:cross_section_dispersion(r,p,history)),("平均收益相关性",lambda:average_correlation(r,60,p,history)),("下跌条件相关性",lambda:downside_correlation(r,120,p,history)),("首主成分解释率",lambda:pca_explained_ratio(r,60,p,history)),("下行共振",lambda:downside_resonance(r,floor,trade,p,history)),("跌停扩散比例",lambda:limit_down_ratio(floor,trade,p,history)),("最差滚动5日收益",lambda:worst_rolling_loss(ir,history,5,history))):
                    v,h=fn(); rows.append({"指标":n,"域":domain,"周期":p,"数值":v,"历史分位":h})
        # Wind 指数波动率：直接使用 WSD 指标，避免用本地收益代理替代。
        for code, label in (("000300.SH","中证300"),("000852.SH","中证1000")):
            vol = self.option_volatility(asof, c.dates[start], c.dates[end], (code,))
            for field, name in (("underlyinghisvol_30d","历史波动率30日"),("underlyinghisvol_60d","历史波动率60日"),("underlyinghisvol_90d","历史波动率90日"),("us_impliedvol","隐含波动率US"),("mid_impliedvol","隐含波动率中值"),("impliedvol_exch","隐含波动率交易所")):
                x=vol.get((code,field),pd.Series(dtype=float)).reindex(c.dates[start:end+1]).ffill().dropna()
                if len(x):
                    rows.append({"指标":name,"域":label,"周期":1,"数值":float(x.iloc[-1]),"历史分位":_p(x.to_numpy(),history)})
            iv=vol.get((code,"us_impliedvol"),pd.Series(dtype=float)); rv=vol.get((code,"underlyinghisvol_30d"),pd.Series(dtype=float))
            if len(iv) and len(rv):
                x=iv.reindex(iv.index.union(rv.index)).ffill().subtract(rv.reindex(iv.index.union(rv.index)).ffill()).dropna()
                if len(x): rows.append({"指标":"波动率风险溢价","域":label,"周期":1,"数值":float(x.iloc[-1]),"历史分位":_p(x.to_numpy(),history)})
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
