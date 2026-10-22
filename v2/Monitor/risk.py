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
    r=_pct(returns); daily=np.nanstd(r,axis=1)
    return float(np.nanmean(daily[-period:])), _p(daily,history)

def average_correlation(returns, window=60, period=1, history=252):
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; x=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0)); c=np.corrcoef(x.T); vals.append(np.nanmean(c[np.triu_indices_from(c,1)]))
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

def downside_correlation(returns, window=120, period=1, history=252):
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; x=x[np.nanmean(x,axis=1)<0]
        if len(x)<3: vals.append(np.nan); continue
        x=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0)); c=np.corrcoef(x.T); vals.append(np.nanmean(c[np.triu_indices_from(c,1)]))
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

def pca_explained_ratio(returns, window=60, period=1, history=252):
    r=_pct(returns); vals=[]
    for i in range(window,len(r)+1):
        x=r[i-window:i]; x=x[:,np.isfinite(x).sum(0)>=int(window*.8)]
        if x.shape[1]<2: vals.append(np.nan); continue
        x=np.nan_to_num(x-x.mean(0)); s=np.linalg.svd(np.cov(x,rowvar=False),compute_uv=False)
        vals.append(float(s[0]/s.sum()) if s.sum()>0 else np.nan)
    return (float(np.nanmean(vals[-period:])),_p(vals,history)) if vals else (np.nan,np.nan)

def downside_resonance(returns, period=1, history=252):
    r=_pct(returns); vals=np.nanmean(r < -0.05,axis=1)
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def worst_rolling_loss(index_returns, lookback=252, loss_window=5, history=252):
    r=_pct(index_returns).reshape(-1)
    five=pd.Series(r).rolling(loss_window).apply(lambda x: np.prod(1+x)-1).to_numpy()
    vals=pd.Series(five).rolling(lookback).min().to_numpy()
    return float(vals[-1]) if len(vals) else np.nan,_p(vals,history)

def limit_down_ratio(hit_floor, has_trade=None, period=1, history=252):
    hit=np.asarray(hit_floor,bool); valid=np.ones_like(hit,bool) if has_trade is None else np.asarray(has_trade,bool)
    vals=np.divide((hit&valid).sum(1),valid.sum(1),out=np.full(hit.shape[0],np.nan),where=valid.sum(1)>0)
    return float(np.nanmean(vals[-period:])),_p(vals,history)

def volatility_risk_premium(iv, rv, period=1, history=252):
    x=np.asarray(iv,float)-np.asarray(rv,float)
    return float(np.nanmean(x[-period:])),_p(x,history)

def realized_volatility(index_returns, window=20, period=1, history=252):
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
    def snapshot(self,asof,history=252,periods=PERIODS,domains=("全市场",)):
        c=self.context; end=c.position(asof); start=max(0,end-history-260); r=c.matrix(self.fields["returns"],start,end)
        floor=c.matrix(self.fields["hit_floor"],start,end).astype(bool)
        trade=c.matrix(self.fields["has_trade"],start,end).astype(bool)
        rows=[]
        for domain in domains:
            w=c.domain_weights(start,end,domain); ir=np.array([weighted_mean(_pct(r[t]),w[t-1]) if t else np.nan for t in range(len(r))])
            for p in periods:
                for n,fn in (("横截面收益离散度",lambda:cross_section_dispersion(r,p,history)),("平均收益相关性",lambda:average_correlation(r,60,p,history)),("下跌条件相关性",lambda:downside_correlation(r,120,p,history)),("首主成分解释率",lambda:pca_explained_ratio(r,60,p,history)),("下行共振",lambda:downside_resonance(r,p,history)),("跌停扩散比例",lambda:limit_down_ratio(floor,trade,p,history)),("最差滚动5日收益",lambda:worst_rolling_loss(ir,history,5,history))):
                    v,h=fn(); rows.append({"指标":n,"域":domain,"周期":p,"数值":v,"历史分位":h})
            for window in (30,60,90):
                v,h=realized_volatility(ir,window,1,history); rows.append({"指标":f"历史波动率{window}日", "域":domain,"周期":1,"数值":v,"历史分位":h})
        return pd.DataFrame(rows).set_index(["指标","域","周期"])
