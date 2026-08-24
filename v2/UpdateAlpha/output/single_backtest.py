from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from v2.GetData import DataPool
from v2.ResearchFlow.FactorTest.metrics import IC,rankIC,calc_group_ret
from v2.UpdateData.config import get_zyyx_conn

DATA_ROOT=Path(r"D:\data")
OUTPUT=ROOT/"disp_optimization"
START=pd.Timestamp("2012-02-13");END=pd.Timestamp("2015-02-13");LOOKBACK=180

def load_reports():
    conn=get_zyyx_conn();chunks=[];left=START-pd.Timedelta(days=LOOKBACK)
    while left<=END:
        right=min(left+pd.DateOffset(months=6)-pd.Timedelta(days=1),END)
        print(f"report chunk {left:%Y-%m-%d} .. {right:%Y-%m-%d}",flush=True)
        conn.exec_driver_sql("IF OBJECT_ID('tempdb..#d') IS NOT NULL DROP TABLE #d")
        conn.exec_driver_sql(f"""
        SELECT f.id,f.report_id,f.stock_code,f.organ_id,f.create_date,f.entrytime,
               f.report_year,f.report_quarter,f.forecast_np INTO #d
        FROM rpt_forecast_stk f
        WHERE f.create_date BETWEEN '{left:%Y-%m-%d}' AND '{right:%Y-%m-%d}'
          AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND 7
          AND (f.reliability>=5 OR f.reliability IS NULL)
          AND f.organ_id IS NOT NULL AND f.forecast_np IS NOT NULL;
        CREATE INDEX ix_d_report ON #d(report_id);
        """)
        chunks.append(pd.read_sql("""SELECT f.*,ra.author_id FROM #d f
          JOIN rpt_report_author ra ON ra.report_id=f.report_id WHERE ra.author_id IS NOT NULL""",conn))
        left=right+pd.Timedelta(days=1)
    d=pd.concat(chunks,ignore_index=True);d["tick"]=d.stock_code.astype(str).str.zfill(6)
    d["create_date"]=pd.to_datetime(d.create_date);d["entrytime"]=pd.to_datetime(d.entrytime)
    d["forecast_np"]=pd.to_numeric(d.forecast_np,errors="coerce")
    return d.sort_values(["tick","organ_id","author_id","report_year","create_date","entrytime","report_id","id"]).drop_duplicates(
      ["report_id","tick","organ_id","author_id","report_year","report_quarter"],keep="last")

def weighted_median(x,w):
    order=np.argsort(x);x=x[order];w=w[order];return x[np.searchsorted(np.cumsum(w),w.sum()/2)]

def section(reports,asof):
    asof=pd.Timestamp(asof)
    d=reports[(reports.create_date>=asof-pd.Timedelta(days=LOOKBACK))&(reports.create_date<=asof)
      &(reports.entrytime<=asof+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))
      &(reports.report_year==asof.year)&(reports.report_quarter==4)&np.isfinite(reports.forecast_np)].copy()
    d=d.sort_values(["tick","author_id","create_date","entrytime","report_id","id"]).drop_duplicates(["tick","author_id"],keep="last")
    d["w"]=np.exp(-np.log(2)*(asof-d.create_date).dt.days/45);d["wx"]=d.w*d.forecast_np
    org=d.groupby(["tick","organ_id"]).agg(sw=("w","sum"),swx=("wx","sum"),fresh_w=("w","max"),analysts=("author_id","nunique"))
    org["forecast"]=org.swx/org.sw;org=org.reset_index()
    rows=[]
    for tick,g in org.groupby("tick"):
        if len(g)<3:continue
        x=g.forecast.to_numpy(float);w=g.fresh_w.to_numpy(float)
        eff=w.sum()**2/np.sum(w*w)
        if eff<2.5:continue
        center=weighted_median(x,w);mad=weighted_median(np.abs(x-center),w)
        rows.append((tick,center,1.4826*mad,len(g),eff))
    if not rows:return pd.Series(dtype=float)
    s=pd.DataFrame(rows,columns=["tick","center","disp","count","eff"]).set_index("tick")
    floor=np.nanquantile(np.abs(s.center[np.isfinite(s.center)&(s.center!=0)]),.20)
    return s.disp/s.center.abs().clip(lower=floor)

def turnover(pred):
    hs=[]
    for row in pred:
        valid=np.flatnonzero(np.isfinite(row));h=set()
        if len(valid)>=20:
            order=valid[np.argsort(row[valid])];n=max(len(order)//10,1);h=set(order[:n])|set(-order[-n:]-1)
        hs.append(h)
    return np.mean([len(a.symmetric_difference(b))/max(len(a)+len(b),1) for a,b in zip(hs[:-1],hs[1:])])

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);data=DataPool(DATA_ROOT,asset="stock");data.asset_root=DATA_ROOT
    dates=pd.DatetimeIndex(data.axis.trade_dates);pos=np.flatnonzero((dates>=START)&(dates<=END));td=dates[pos]
    reports=load_reports();pred=np.zeros((len(pos),data.axis.tick_count),np.float32);tickpos=data.axis._tick_positions
    for i,date in enumerate(tqdm(td,desc="Combined DISP")):
        for tick,value in section(reports,date).dropna().items():
            if (j:=tickpos.get(str(tick))) is not None and np.isfinite(value):pred[i,j]=value
    pct=data.read("d_essentials/pct",data.axis.date_count-1,0)/100;rows=[]
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True);colors=plt.cm.tab10(np.linspace(0,1,10))
    for ax,h in zip(axes.flat,(1,5,10,20)):
        win=np.lib.stride_tricks.sliding_window_view(pct[2:],h,axis=0);fwd=np.prod(1+win,axis=-1)-1
        label=np.full(pred.shape,np.nan);valid=pos<len(fwd);label[valid]=fwd[pos[valid]]
        ic,ric=IC(pred,label),rankIC(pred,label);group=calc_group_ret(pred,label,10);means=np.nanmean(group,axis=1);ls=group[0]-group[-1]
        rows.append({"factor":"fresh_institution_robust","horizon":h,"coverage":np.mean((pred>0).sum(1)),
          "mean_ic":np.nanmean(ic),"icir":np.nanmean(ic)/np.nanstd(ic)*np.sqrt(252),"rank_ic":np.nanmean(ric),
          "rank_icir":np.nanmean(ric)/np.nanstd(ric)*np.sqrt(252),"monotonicity":-spearmanr(range(1,11),means).statistic,
          "low_group_bps":means[0]*1e4,"high_group_short_bps":-means[-1]*1e4,"long_short_bps":np.nanmean(ls)*1e4,
          "long_short_sharpe":np.nanmean(ls)/np.nanstd(ls)*np.sqrt(252),"turnover":turnover(pred),
          **{f"g{k+1}_bps":v*1e4 for k,v in enumerate(means)}})
        for k,v in enumerate(np.nancumsum(group,axis=1)):ax.plot(td[:len(v)],v,color=colors[k],linewidth=1.1,label=f"G{k+1}")
        ax.set_title(f"Combined {h}D | IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(ric):.4f}");ax.axhline(0,color="black",linewidth=.8)
        ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);ax.xaxis.set_major_locator(mdates.AutoDateLocator());ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle("DISP Freshness + Stable Floor + Institution Robust | No Tradable Filter");fig.tight_layout()
    fig.savefig(OUTPUT/"fresh_institution_robust_cumulative.png",dpi=160,bbox_inches="tight");plt.close(fig)
    combined=pd.DataFrame(rows);base=pd.read_csv(OUTPUT/"summary.csv");comparison=pd.concat([base,combined],ignore_index=True)
    comparison.to_csv(OUTPUT/"summary_with_combined.csv",index=False,encoding="utf-8-sig")
    print(combined.to_string(index=False),flush=True);print(f"saved: {OUTPUT}",flush=True)

def independent_sections(reports,asof):
    asof=pd.Timestamp(asof)
    d=reports[(reports.create_date>=asof-pd.Timedelta(days=LOOKBACK))&(reports.create_date<=asof)
      &(reports.entrytime<=asof+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))
      &(reports.report_year==asof.year)&(reports.report_quarter==4)&np.isfinite(reports.forecast_np)].copy()
    d=d.sort_values(["tick","author_id","create_date","entrytime","report_id","id"]).drop_duplicates(["tick","author_id"],keep="last")
    stable=d.groupby("tick").forecast_np.agg(["mean","std","count"])
    stable_eligible=stable["count"]>=5
    stable_floor=np.nanquantile(np.abs(stable.loc[stable_eligible,"mean"]),.20)
    stable_floor_factor=(stable["std"]/stable["mean"].abs().clip(lower=stable_floor)).where(stable_eligible)
    d["w"]=np.exp(-np.log(2)*(asof-d.create_date).dt.days/45);d["wx"]=d.w*d.forecast_np
    fresh=d.groupby("tick").agg(sw=("w","sum"),swx=("wx","sum"),count=("forecast_np","size"));fresh["mean"]=fresh.swx/fresh.sw
    temp=d.join(fresh["mean"],on="tick");temp["wv"]=temp.w*(temp.forecast_np-temp["mean"])**2
    fresh["std"]=np.sqrt(temp.groupby("tick").wv.sum()/fresh.sw);fresh["eff_n"]=fresh.sw**2/d.assign(w2=d.w**2).groupby("tick").w2.sum()
    eligible=(fresh["count"]>=5)&(fresh.eff_n>=3);floor=np.nanquantile(np.abs(fresh.loc[eligible,"mean"]),.20)
    freshness=(fresh["std"]/fresh["mean"].abs().clip(lower=floor)).where(eligible)
    org=d.groupby(["tick","organ_id"],as_index=False).forecast_np.mean();rows=[]
    for tick,g in org.groupby("tick"):
        if len(g)<3:continue
        x=g.forecast_np.to_numpy(float);center=np.median(x);rows.append((tick,center,1.4826*np.median(np.abs(x-center))))
    inst=pd.DataFrame(rows,columns=["tick","center","disp"]).set_index("tick") if rows else pd.DataFrame(columns=["center","disp"])
    if len(inst):
        ifloor=np.nanquantile(np.abs(inst.center[np.isfinite(inst.center)&(inst.center!=0)]),.20);institution=inst.disp/inst.center.abs().clip(lower=ifloor)
    else:institution=pd.Series(dtype=float)
    return stable_floor_factor,freshness,institution

def orthogonal_main():
    OUTPUT.mkdir(parents=True,exist_ok=True);data=DataPool(DATA_ROOT,asset="stock");data.asset_root=DATA_ROOT
    dates=pd.DatetimeIndex(data.axis.trade_dates);pos=np.flatnonzero((dates>=START)&(dates<=END));td=dates[pos];reports=load_reports()
    names=("stable_floor","freshness_weighted","institution_robust")
    base={n:np.full((len(pos),data.axis.tick_count),np.nan,np.float32) for n in names};tickpos=data.axis._tick_positions
    for i,date in enumerate(tqdm(td,desc="Three-way Orthogonal DISP")):
        for name,values in zip(names,independent_sections(reports,date)):
            for tick,value in values.dropna().items():
                if (j:=tickpos.get(str(tick))) is not None and np.isfinite(value):base[name][i,j]=value
    candidate_names=("disp_freshness","disp_institution","disp","threeway_equal_rank","stable_floor_orthogonal_others","freshness_orthogonal_others","institution_orthogonal_others")
    candidates={n:np.full_like(base["stable_floor"],np.nan) for n in candidate_names}
    for i in range(len(pos)):
        raw=[base[n][i] for n in names];valid=np.logical_and.reduce([np.isfinite(x) for x in raw])
        if valid.sum()<20:continue
        ranks=[]
        for x in raw:
            r=pd.Series(x[valid]).rank(pct=True).to_numpy();ranks.append((r-r.mean())/(r.std()+1e-12))
        r=np.column_stack(ranks);candidates["disp_freshness"][i]=base["freshness_weighted"][i];candidates["disp_institution"][i]=base["institution_robust"][i]
        candidates["disp"][i,valid]=(r[:,1]+r[:,2])/2;candidates["threeway_equal_rank"][i,valid]=r.mean(axis=1)
        residual_names=candidate_names[4:]
        for j,out_name in enumerate(residual_names):
            other=[k for k in range(3) if k!=j];x=np.column_stack([np.ones(len(r)),r[:,other]])
            candidates[out_name][i,valid]=r[:,j]-x@np.linalg.lstsq(x,r[:,j],rcond=None)[0]
    pct=data.read("d_essentials/pct",data.axis.date_count-1,0)/100;rows=[]
    for name,pred in candidates.items():
        fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True);colors=plt.cm.tab10(np.linspace(0,1,10))
        for ax,h in zip(axes.flat,(1,5,10,20)):
            win=np.lib.stride_tricks.sliding_window_view(pct[2:],h,axis=0);fwd=np.prod(1+win,axis=-1)-1;label=np.full(pred.shape,np.nan);valid=pos<len(fwd);label[valid]=fwd[pos[valid]]
            ic,ric=IC(pred,label),rankIC(pred,label);group=calc_group_ret(pred,label,10);means=np.nanmean(group,axis=1);ls=group[0]-group[-1]
            rows.append({"factor":name,"horizon":h,"coverage":np.mean((pred>0).sum(1)),"mean_ic":np.nanmean(ic),"icir":np.nanmean(ic)/np.nanstd(ic)*np.sqrt(252),"rank_ic":np.nanmean(ric),"rank_icir":np.nanmean(ric)/np.nanstd(ric)*np.sqrt(252),"monotonicity":-spearmanr(range(1,11),means).statistic,"low_group_bps":means[0]*1e4,"high_group_short_bps":-means[-1]*1e4,"long_short_bps":np.nanmean(ls)*1e4,"long_short_sharpe":np.nanmean(ls)/np.nanstd(ls)*np.sqrt(252),"turnover":turnover(pred),**{f"g{k+1}_bps":v*1e4 for k,v in enumerate(means)}})
            for k,v in enumerate(np.nancumsum(group,axis=1)):ax.plot(td[:len(v)],v,color=colors[k],linewidth=1.1,label=f"G{k+1}")
            ax.set_title(f"{name} {h}D | IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(ric):.4f}");ax.axhline(0,color="black",linewidth=.8);ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);ax.xaxis.set_major_locator(mdates.AutoDateLocator());ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.suptitle(f"DISP {name} | No Tradable Filter");fig.tight_layout();fig.savefig(OUTPUT/f"{name}_cumulative.png",dpi=160,bbox_inches="tight");plt.close(fig)
    result=pd.DataFrame(rows);result.to_csv(OUTPUT/"summary_current.csv",index=False,encoding="utf-8-sig")
    print(result.to_string(index=False),flush=True);print(f"saved: {OUTPUT}",flush=True)
def hierarchical_freshness_section(reports,asof):
    asof=pd.Timestamp(asof);start=asof-pd.Timedelta(days=90)
    d=reports[(reports.create_date>=start)&(reports.create_date<=asof)
      &(reports.entrytime<=asof+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))
      &(reports.report_year==asof.year)&(reports.report_quarter==4)
      &np.isfinite(reports.forecast_np)].copy()
    if d.empty:return pd.Series(dtype=float)
    d["w"]=np.exp(-np.log(2)*(asof-d.create_date).dt.days/45);d["wx"]=d.w*d.forecast_np
    analyst=d.groupby(["tick","organ_id","author_id"],as_index=False).agg(sw=("w","sum"),swx=("wx","sum"))
    analyst["analyst_forecast"]=analyst.swx/analyst.sw
    institution=analyst.groupby(["tick","organ_id"],as_index=False).analyst_forecast.mean()
    rows=[]
    for tick,g in institution.groupby("tick"):
        if len(g)<3:continue
        x=g.analyst_forecast.to_numpy(float);center=np.median(x)
        rows.append((tick,center,1.4826*np.median(np.abs(x-center))))
    if not rows:return pd.Series(dtype=float)
    stats=pd.DataFrame(rows,columns=["tick","center","dispersion"]).set_index("tick")
    centers=np.abs(stats.center);centers=centers[np.isfinite(centers)&(centers>0)]
    if centers.empty:return pd.Series(dtype=float)
    floor=np.nanquantile(centers,.20)
    return stats.dispersion/stats.center.abs().clip(lower=floor)

def hierarchical_main():
    out=Path(__file__).resolve().parent/"disp";out.mkdir(parents=True,exist_ok=True)
    data=DataPool(DATA_ROOT,asset="stock");data.asset_root=DATA_ROOT
    dates=pd.DatetimeIndex(data.axis.trade_dates);pos=np.flatnonzero((dates>=START)&(dates<=END));td=dates[pos]
    reports=load_reports();pred=np.zeros((len(pos),data.axis.tick_count),np.float32);tickpos=data.axis._tick_positions
    for i,date in enumerate(tqdm(td,desc="Hierarchical Freshness DISP")):
        for tick,value in hierarchical_freshness_section(reports,date).dropna().items():
            if (j:=tickpos.get(str(tick))) is not None and np.isfinite(value):pred[i,j]=value
    pct=data.read("d_essentials/pct",data.axis.date_count-1,0)/100;rows=[]
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True);colors=plt.cm.tab10(np.linspace(0,1,10))
    for ax,h in zip(axes.flat,(1,5,10,20)):
        win=np.lib.stride_tricks.sliding_window_view(pct[2:],h,axis=0);fwd=np.prod(1+win,axis=-1)-1
        label=np.full(pred.shape,np.nan);valid=pos<len(fwd);label[valid]=fwd[pos[valid]]
        ic,ric=IC(pred,label),rankIC(pred,label);group=calc_group_ret(pred,label,10);means=np.nanmean(group,axis=1);ls=group[0]-group[-1]
        rows.append({"factor":"disp_analyst_fresh_institution_robust","horizon":h,"coverage":np.mean((pred>0).sum(1)),"mean_ic":np.nanmean(ic),"icir":np.nanmean(ic)/np.nanstd(ic)*np.sqrt(252),"rank_ic":np.nanmean(ric),"rank_icir":np.nanmean(ric)/np.nanstd(ric)*np.sqrt(252),"monotonicity":-spearmanr(range(1,11),means).statistic,"low_group_bps":means[0]*1e4,"high_group_short_bps":-means[-1]*1e4,"long_short_bps":np.nanmean(ls)*1e4,"long_short_sharpe":np.nanmean(ls)/np.nanstd(ls)*np.sqrt(252),"turnover":turnover(pred),**{f"g{k+1}_bps":v*1e4 for k,v in enumerate(means)}})
        for k,v in enumerate(np.nancumsum(group,axis=1)):ax.plot(td[:len(v)],v,color=colors[k],linewidth=1.1,label=f"G{k+1}")
        ax.set_title(f"Hierarchical Freshness {h}D | IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(ric):.4f}");ax.axhline(0,color="black",linewidth=.8);ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);ax.xaxis.set_major_locator(mdates.AutoDateLocator());ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle("DISP Analyst EWMA90 -> Institution Equal Weight -> Cross-Institution Median/MAD");fig.tight_layout();fig.savefig(out/"disp_analyst_fresh_institution_robust_cumulative.png",dpi=160,bbox_inches="tight");plt.close(fig)
    result=pd.DataFrame(rows);base_path=out/"summary.csv";base=pd.read_csv(base_path) if base_path.exists() else pd.DataFrame();pd.concat([base,result],ignore_index=True).to_csv(out/"summary_with_hierarchical.csv",index=False,encoding="utf-8-sig")
    print(result.to_string(index=False),flush=True);print(f"saved: {out}",flush=True)

def cov_fy1_section(reports,asof):
    asof=pd.Timestamp(asof)
    d=reports[(reports.create_date>=asof-pd.Timedelta(days=180))&(reports.create_date<=asof)
      &(reports.entrytime<=asof+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))
      &(reports.report_year==asof.year)&(reports.report_quarter==4)
      &np.isfinite(reports.forecast_np)].copy()
    return np.sqrt(d.groupby("tick").report_id.nunique())

def cov_fy1_main():
    out=Path(__file__).resolve().parent/"cov";out.mkdir(parents=True,exist_ok=True)
    data=DataPool(DATA_ROOT,asset="stock");data.asset_root=DATA_ROOT
    dates=pd.DatetimeIndex(data.axis.trade_dates);pos=np.flatnonzero((dates>=START)&(dates<=END));td=dates[pos]
    reports=load_reports();pred=np.zeros((len(pos),data.axis.tick_count),np.float32);tickpos=data.axis._tick_positions
    for i,date in enumerate(tqdm(td,desc="Report-count FY1 COV")):
        for tick,value in cov_fy1_section(reports,date).dropna().items():
            if (j:=tickpos.get(str(tick))) is not None and np.isfinite(value):pred[i,j]=value
    pct=data.read("d_essentials/pct",data.axis.date_count-1,0)/100;rows=[]
    # Only the grouping copy gets a stable, return-independent tie break.
    tie_break=np.random.default_rng(0).uniform(0.0,1e-6,pred.shape[1])
    grouped_pred=pred+tie_break[None,:]
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True);colors=plt.cm.tab10(np.linspace(0,1,10))
    for ax,h in zip(axes.flat,(1,5,10,20)):
        win=np.lib.stride_tricks.sliding_window_view(pct[2:],h,axis=0);fwd=np.prod(1+win,axis=-1)-1
        label=np.full(pred.shape,np.nan);valid=pos<len(fwd);label[valid]=fwd[pos[valid]]
        ic,ric=IC(pred,label),rankIC(pred,label);group=calc_group_ret(grouped_pred,label,10);means=np.nanmean(group,axis=1);ls=group[0]-group[-1]
        rows.append({"factor":"cov_report_fy1_zero","horizon":h,"coverage":np.mean((pred>0).sum(1)),"mean_ic":np.nanmean(ic),"icir":np.nanmean(ic)/np.nanstd(ic)*np.sqrt(252),"rank_ic":np.nanmean(ric),"rank_icir":np.nanmean(ric)/np.nanstd(ric)*np.sqrt(252),"monotonicity":-spearmanr(range(1,11),means).statistic,"low_group_bps":means[0]*1e4,"high_group_short_bps":-means[-1]*1e4,"low_minus_high_bps":np.nanmean(ls)*1e4,"low_minus_high_sharpe":np.nanmean(ls)/np.nanstd(ls)*np.sqrt(252),"turnover":turnover(pred),**{f"g{k+1}_bps":v*1e4 for k,v in enumerate(means)}})
        for k,v in enumerate(np.nancumsum(group,axis=1)):ax.plot(td[:len(v)],v,color=colors[k],linewidth=1.1,label=f"G{k+1}")
        ax.set_title(f"Report-count FY1 COV {h}D | IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(ric):.4f}");ax.axhline(0,color="black",linewidth=.8);ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);ax.xaxis.set_major_locator(mdates.AutoDateLocator());ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle("Report-count FY1 Coverage (Zero-filled) Decile Cumulative Excess Returns | No Tradable Filter");fig.tight_layout();fig.savefig(out/"cov_report_fy1_zero_cumulative.png",dpi=160,bbox_inches="tight");plt.close(fig)
    result=pd.DataFrame(rows);result.to_csv(out/"summary_report_fy1_zero.csv",index=False,encoding="utf-8-sig");print(result.to_string(index=False),flush=True);print(f"saved: {out}",flush=True)

if __name__=="__main__":cov_fy1_main()





