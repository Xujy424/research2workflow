"""Point-in-time ZYYX factors: COV, DISP, EP_FY1, PEG, SCORE, TPER, WFR."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import numpy as np
import pandas as pd

FACTOR_NAMES = ("cov", "disp", "ep_fy1", "peg", "score", "tper", "wfr")
WeightFunction = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]

@dataclass(frozen=True)
class AnalystFactorConfig:
    lookback_days: int = 90
    max_entry_delay_days: int = 7
    min_reliability: int = 5
    min_dispersion_orgs: int = 5
    revision_gap_days: int = 30
    revision_history_days: int = 180
    # ZYYX forecast profit fields are in RMB 10,000; RMB 1m therefore equals 100.
    min_profit_denominator: float = 100.0

def aggregate_report_authors(authors: pd.DataFrame) -> pd.DataFrame:
    """XCF=max (any New Fortune author); other numeric attributes=author mean."""
    if authors.empty:
        return pd.DataFrame(columns=["report_id"])
    if not {"report_id", "xcf"}.issubset(authors.columns):
        raise ValueError("authors must contain report_id and xcf")
    cols = [c for c in authors.select_dtypes("number") if c not in {"report_id", "author_id", "xcf"}]
    return authors.groupby("report_id", as_index=False).agg({**{c: "mean" for c in cols}, "xcf": "max"})

def make_accwt2_weight_fn(coefficients: Mapping[str, float]) -> WeightFunction:
    """Create Accwt2 weights from annually fitted PMAFE coefficients."""
    def weight(frame: pd.DataFrame, _asof: pd.Timestamp) -> pd.Series:
        data=frame.copy()
        if "horizon" in coefficients and "horizon" not in data:
            start=pd.to_datetime(data.report_year.astype("Int64").astype(str)+"-01-01")
            data["horizon"]=(pd.to_datetime(data.entrytime).dt.normalize()-start).dt.days
        missing=set(coefficients)-set(data.columns)
        if missing:
            raise ValueError(f"Accwt2 attributes missing: {sorted(missing)}")
        p=sum(pd.to_numeric(data[x],errors="coerce")*b for x,b in coefficients.items())
        keys=[data.stock_code,data.report_year]
        z = (p - p.groupby(keys).transform("mean")) / p.groupby(keys).transform("std").replace(0, np.nan)
        sole = frame.groupby(["stock_code", "report_year"]).organ_id.transform("nunique").eq(1)
        return (-z).clip(lower=0).where(~sole, 1).fillna(0)
    return weight

ACCURACY_FEATURES=("horizon","acc","gfexp","ninds","naut")



def fit_accwt2_weight_fn(forecasts,actuals):
    """Fit the report's no-intercept PMAFE model and return Accwt2 weights."""
    required={"stock_code","report_year","forecast_np","entrytime",*ACCURACY_FEATURES[1:]}
    missing=required-set(forecasts.columns)
    if missing: raise ValueError(f"accuracy training columns missing: {sorted(missing)}")
    x=forecasts.copy(); x.stock_code=x.stock_code.astype("string").str.zfill(6)
    x=x.merge(actuals[["stock_code","report_year","actual_np"]],
              on=["stock_code","report_year"],how="inner",validate="many_to_one")
    x["abs_error"]=(pd.to_numeric(x.forecast_np,errors="coerce")-x.actual_np).abs()
    keep=x.groupby("report_year").abs_error.transform(
        lambda v: v.between(v.quantile(.01),v.quantile(.99)))
    x=x.loc[keep].copy()
    keys=["stock_code","report_year"]
    mean_error=x.groupby(keys).abs_error.transform("mean").replace(0,np.nan)
    x["pmafe"]=(x.abs_error-mean_error)/mean_error
    start=pd.to_datetime(x.report_year.astype("Int64").astype(str)+"-01-01")
    x["horizon"]=(pd.to_datetime(x.entrytime).dt.normalize()-start).dt.days
    for name in ACCURACY_FEATURES:
        x[name]=pd.to_numeric(x[name],errors="coerce")
        x[name]-=x.groupby(keys)[name].transform("mean")
    valid=x[["pmafe",*ACCURACY_FEATURES]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(valid)<=len(ACCURACY_FEATURES): raise ValueError("insufficient PMAFE observations")
    beta=np.linalg.lstsq(valid[list(ACCURACY_FEATURES)].to_numpy(float),
                         valid.pmafe.to_numpy(float),rcond=None)[0]
    return make_accwt2_weight_fn(dict(zip(ACCURACY_FEATURES,beta,strict=True)))

def _load_accuracy_forecasts(conn,year,cfg):
    sql=f"""
    SELECT f.id,f.report_id,f.stock_code,f.organ_id,f.create_date,f.entrytime,
           f.report_year,f.report_quarter,f.forecast_np
    FROM rpt_forecast_stk f
    WHERE EXISTS (SELECT 1 FROM rpt_organ_information o
      WHERE o.organ_id=f.organ_id AND o.organ_type=5) AND f.report_year={year} AND f.report_quarter=4
      AND f.entrytime>='{year}-01-01' AND f.entrytime<'{year+1}-05-01'
      AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND {cfg.max_entry_delay_days}
      AND f.reliability>={cfg.min_reliability}
    """
    x=pd.read_sql(sql,conn)
    x["stock_code"]=x.stock_code.astype("string").str.zfill(6)
    x["create_date"]=pd.to_datetime(x.create_date,errors="coerce").dt.normalize()
    x["entrytime"]=pd.to_datetime(x.entrytime,errors="coerce")
    x["forecast_np"]=pd.to_numeric(x.forecast_np,errors="coerce")
    return x.loc[x.stock_code.str.match(r"^(00|30|60|68|92)")&x.forecast_np.notna()]

def _authors_for_reports(conn,reports):
    ids=pd.to_numeric(reports.report_id,errors="coerce").dropna().astype("int64").unique()
    if not len(ids): return pd.DataFrame(columns=["report_id","author_id"])
    parts=[]
    for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/1000)))):
        values=",".join(map(str,chunk))
        parts.append(pd.read_sql(f"SELECT report_id,author_id,organ_id FROM rpt_report_author "
            f"WHERE report_id IN ({values}) AND entrytime<='{reports.entrytime.max()}'",conn))
    return pd.concat(parts,ignore_index=True).drop_duplicates(["report_id","author_id"],keep="last")

def _author_accuracy(conn,jy_conn,year,cfg):
    forecasts=_load_accuracy_forecasts(conn,year,cfg)
    actuals=load_annual_actuals(jy_conn,year,pd.Timestamp(f"{year+1}-04-30"))
    x=forecasts.merge(actuals[["stock_code","report_year","actual_np"]],
                      on=["stock_code","report_year"],how="inner")
    x["error"]=(x.forecast_np-x.actual_np).abs()
    mean=x.groupby(["stock_code","report_year"]).error.transform("mean").replace(0,np.nan)
    x["pmafe"]=(x.error-mean)/mean
    authors=_authors_for_reports(conn,x)
    relative=authors.merge(x[["report_id","pmafe"]],on="report_id",how="inner")
    return -relative.groupby("author_id").pmafe.mean()

def _attach_analyst_attributes(conn,reports,acc):
    if reports.empty: return reports.assign(acc=np.nan,gfexp=np.nan,ninds=np.nan,naut=np.nan)
    authors=_authors_for_reports(conn,reports)
    targets=authors.merge(reports[["report_id","stock_code","create_date"]].drop_duplicates("report_id"),
                          on="report_id",how="inner")
    author_ids=authors.author_id.dropna().astype("int64").unique()
    if not len(author_ids): return reports.assign(acc=np.nan,gfexp=np.nan,ninds=np.nan,naut=np.nan)
    start=(reports.create_date.min()-pd.Timedelta(days=365)).date()
    end=(reports.create_date.max()+pd.Timedelta(days=1)).date()
    values=",".join(map(str,author_ids))
    history=pd.read_sql(f"""
      SELECT ra.author_id,ra.organ_id,f.stock_code,f.create_date,ic.industry_code
      FROM rpt_report_author ra JOIN rpt_forecast_stk f ON f.report_id=ra.report_id
      OUTER APPLY (SELECT TOP 1 industry_code FROM qt_indus_constituents q
        WHERE q.stock_code=f.stock_code AND q.standard_code=908 AND q.industry_level=2
          AND q.into_date<=f.create_date AND (q.out_date IS NULL OR q.out_date>f.create_date)) ic
      WHERE ra.author_id IN ({values}) AND f.create_date>='{start}' AND f.create_date<'{end}'
    """,conn)
    first=pd.read_sql(f"""SELECT ra.author_id,MIN(f.create_date) first_general
      FROM rpt_report_author ra JOIN rpt_forecast_stk f ON f.report_id=ra.report_id
      WHERE ra.author_id IN ({values}) GROUP BY ra.author_id""",conn).set_index("author_id").first_general
    first_stock=pd.read_sql(f"""SELECT ra.author_id,f.stock_code,MIN(f.create_date) first_stock
      FROM rpt_report_author ra JOIN rpt_forecast_stk f ON f.report_id=ra.report_id
      WHERE ra.author_id IN ({values}) GROUP BY ra.author_id,f.stock_code""",conn).set_index(
          ["author_id","stock_code"]).first_stock
    history["create_date"]=pd.to_datetime(history.create_date,errors="coerce").dt.normalize()
    organ_ids=targets.organ_id.dropna().astype("int64").unique()
    organs=",".join(map(str,organ_ids))
    employer_history=pd.read_sql(f"""SELECT ra.author_id,ra.organ_id,f.create_date
      FROM rpt_report_author ra JOIN rpt_forecast_stk f ON f.report_id=ra.report_id
      WHERE ra.organ_id IN ({organs}) AND f.create_date>='{start}' AND f.create_date<'{end}'""",conn)
    employer_history["create_date"]=pd.to_datetime(employer_history.create_date,errors="coerce").dt.normalize()
    rows=[]
    for row in targets.itertuples():
        date=pd.Timestamp(row.create_date)
        period=history.create_date.between(date-pd.Timedelta(days=365),date)
        own=history.loc[history.author_id.eq(row.author_id)&period]
        employer_period=employer_history.create_date.between(date-pd.Timedelta(days=365),date)
        employer=employer_history.loc[employer_history.organ_id.eq(row.organ_id)&employer_period]
        g=np.sqrt(max((date-pd.Timestamp(first.get(row.author_id,date))).days,0))
        f=np.sqrt(max((date-pd.Timestamp(first_stock.get((row.author_id,row.stock_code),date))).days,0))
        rows.append({"report_id":row.report_id,"acc":acc.get(row.author_id,np.nan),
                     "gfexp":(g+f)/2,"ninds":np.sqrt(own.industry_code.nunique()),
                     "naut":np.sqrt(employer.author_id.nunique())})
    attrs=pd.DataFrame(rows).groupby("report_id",as_index=False)[["acc","gfexp","ninds","naut"]].mean()
    return reports.merge(attrs,on="report_id",how="left",validate="many_to_one")

_ACCWT2_CACHE={}

def build_raw_accwt2_weight_fn(conn,jy_conn,asof,config=AnalystFactorConfig()):
    year=accuracy_model_year(asof)
    key=(id(conn),id(jy_conn),year,config.min_reliability)
    if key in _ACCWT2_CACHE: return _ACCWT2_CACHE[key]
    train=_load_accuracy_forecasts(conn,year,config)
    train=_attach_analyst_attributes(conn,train,_author_accuracy(conn,jy_conn,year-1,config))
    fitted=fit_accwt2_weight_fn(train,load_annual_actuals(
        jy_conn,year,pd.Timestamp(f"{year+1}-04-30")))
    current_acc=_author_accuracy(conn,jy_conn,year,config)
    def weight(frame,weight_asof):
        return fitted(_attach_analyst_attributes(conn,frame,current_acc),weight_asof)
    _ACCWT2_CACHE[key]=weight
    return weight

def _date(x):
    x = pd.Timestamp(x).normalize()
    if pd.isna(x): raise ValueError("invalid date")
    return x

def _clean(df, asof, cfg, lookback_days=None):
    """Apply report-level point-in-time and universe filters."""
    need = {"id","report_id","stock_code","reliability","organ_id","create_date","entrytime", "report_year","report_quarter"}
    if need - set(df.columns): raise ValueError(f"reports missing columns: {sorted(need-set(df.columns))}")
    x = df.copy(); x.stock_code = x.stock_code.astype("string").str.zfill(6)
    x.create_date = pd.to_datetime(x.create_date, errors="coerce").dt.normalize()
    x.entrytime = pd.to_datetime(x.entrytime, errors="coerce")
    delay = (x.entrytime.dt.normalize() - x.create_date).dt.days
    days=cfg.lookback_days if lookback_days is None else lookback_days
    ok = x.entrytime.between(asof-pd.Timedelta(days=days), asof+pd.Timedelta(days=1), inclusive="left")
    ok &= delay.between(0, cfg.max_entry_delay_days)
    ok &= pd.to_numeric(x.reliability,errors="coerce") >= cfg.min_reliability
    ok &= x.stock_code.str.match(r"^(00|30|60|68|92)")
    keys = ["report_id","stock_code","report_year","report_quarter"]
    return x.loc[ok].sort_values(keys+["entrytime","id"]).drop_duplicates(keys, keep="last")

def _latest(df, extra=()):
    if df.empty: return df
    keys = ["stock_code","organ_id",*extra]
    return df.sort_values(keys+["entrytime","report_id"]).drop_duplicates(keys, keep="last")

def _mean(df,value):
    values=pd.to_numeric(df[value],errors="coerce")
    weights=pd.to_numeric(df["_weight"],errors="coerce")
    ok=values.notna()&np.isfinite(values)&weights.gt(0)
    x=df.loc[ok,["stock_code",value,"_weight"]].copy()
    if x.empty: return pd.Series(dtype=float)
    x[value]=values.loc[ok]; x["_weight"]=weights.loc[ok]
    x["_wv"]=x[value]*x._weight; g=x.groupby("stock_code")
    return g._wv.sum()/g._weight.sum()

def _rating_score(content):
    rules=((1.,("sell","\u5356\u51fa")),
           (2.,("underperform","underweight","\u51cf\u6301")),
           (3.,("neutral","hold","\u4e2d\u6027","\u6301\u6709","\u89c2\u671b")),
           (5.,("outperform","overweight","\u589e\u6301","\u8c28\u614e\u63a8\u8350","\u5ba1\u614e\u63a8\u8350")),
           (7.,("strong buy","buy","\u5f3a\u70c8\u63a8\u8350","\u4e70\u5165","\u63a8\u8350")))
    def score(value):
        if pd.isna(value): return np.nan
        text=str(value).strip()
        try: text=text.encode("latin1").decode("gb18030")
        except (UnicodeEncodeError,UnicodeDecodeError): pass
        text=text.casefold()
        return next((n for n,words in rules if any(word in text for word in words)),np.nan)
    return content.map(score).astype(float)


def factor_cov(reports):
    return np.log1p(reports.groupby("stock_code").organ_id.nunique().astype(float))

def factor_disp(annual,asof,cfg):
    x=annual.loc[annual.report_year.eq(asof.year)]
    d=x.groupby("stock_code").forecast_np.agg(["count","mean","std"])
    return (d["std"]/d["mean"].replace(0,np.nan)).where(d["count"]>=cfg.min_dispersion_orgs)

def factor_ep_fy1(fy1_np,market):
    return fy1_np/market.total_mv.replace(0,np.nan)

def factor_peg(consensus,fy1,fy1_np,market):
    growth=np.sqrt(consensus[fy1+1]/market.actual_np_fy0)-1
    return (market.total_mv/fy1_np.replace(0,np.nan))/growth.where(growth>0)

def factor_score(reports,weights):
    x=_latest(reports.loc[reports.organ_rating_content.notna()]).copy()
    x["_rating"]=_rating_score(x.organ_rating_content)
    x=x.drop(columns="_weight",errors="ignore").merge(
        weights,on=["stock_code","organ_id"],how="inner",validate="many_to_one")
    return _mean(x,"_rating")

def factor_tper(reports,weights,market):
    x=_latest(reports.loc[
        reports.target_price_ceiling.notna()|reports.target_price_floor.notna()]).copy()
    x["_target"]=x[["target_price_ceiling","target_price_floor"]].mean(axis=1)
    current_share=x.stock_code.map(market.total_share)
    report_share=pd.to_numeric(x.report_capital,errors="coerce")
    x["_target"]*=(report_share/current_share).where(
        current_share.gt(0)&report_share.gt(0),1.0)
    x=x.drop(columns="_weight",errors="ignore").merge(
        weights,on=["stock_code","organ_id"],how="inner",validate="many_to_one")
    return _mean(x,"_target")/market.close.replace(0,np.nan)-1

def factor_wfr(annual,asof,cfg):
    x=annual.loc[annual.report_year.eq(asof.year)].sort_values(
        ["stock_code","organ_id","entrytime","report_id"])
    current=x.loc[x.entrytime.ge(asof-pd.Timedelta(days=cfg.lookback_days))]
    current=current.drop_duplicates(["stock_code","organ_id"],keep="last")
    current=current.rename(columns={"_weight":"weight"})
    rows=[]
    for row in current.itertuples():
        previous=x.loc[x.stock_code.eq(row.stock_code)&x.organ_id.eq(row.organ_id)
            &x.entrytime.le(row.entrytime-pd.Timedelta(days=cfg.revision_gap_days))
            &x.entrytime.ge(row.entrytime-pd.Timedelta(days=cfg.revision_history_days))]
        if previous.empty: continue
        old=float(previous.iloc[-1].forecast_np)
        rows.append({"stock_code":row.stock_code,"_weight":row.weight,
                     "_revision":(float(row.forecast_np)-old)/max(old,cfg.min_profit_denominator)})
    return _mean(pd.DataFrame(rows),"_revision") if rows else pd.Series(dtype=float)


def calculate_analyst_factors(reports,asof,market,*,config=AnalystFactorConfig(),weight_fn):
    """Calculate seven independent factors from raw point-in-time inputs."""
    asof=_date(asof)
    history=_clean(reports,asof,config,config.revision_history_days)
    current=history.loc[history.entrytime.ge(asof-pd.Timedelta(days=config.lookback_days))].copy()
    idx=pd.Index(sorted(current.stock_code.unique()),name="tick")
    out=pd.DataFrame(index=idx)
    if current.empty: return out.reindex(columns=FACTOR_NAMES).reset_index()
    if market is None: raise ValueError("market is required")
    m=market.copy()
    if "stock_code" in m:
        m.stock_code=m.stock_code.astype("string").str.zfill(6); m=m.set_index("stock_code")
    required={"close","total_mv","total_share","actual_np_fy0"}
    if missing:=required-set(m.columns): raise ValueError(f"market columns missing: {sorted(missing)}")
    m=m.reindex(idx)
    for c in required|{"actual_np","actual_np_year"}:
        if c not in m: m[c]=np.nan
        m[c]=pd.to_numeric(m[c],errors="coerce")

    out["cov"]=factor_cov(current)
    annual=history.loc[pd.to_numeric(history.report_quarter,errors="coerce").eq(4)].copy()
    annual["forecast_np"]=pd.to_numeric(annual.forecast_np,errors="coerce")
    annual["forecast_np"]=annual.forecast_np.fillna(
        pd.to_numeric(annual.forecast_eps,errors="coerce")
        *pd.to_numeric(annual.report_capital,errors="coerce"))
    annual_history=annual.loc[annual.forecast_np.notna()].copy()
    current_annual=_latest(annual_history.loc[
        annual_history.entrytime.ge(asof-pd.Timedelta(days=config.lookback_days))],("report_year",))
    current_annual["_weight"]=weight_fn(current_annual,asof)
    weight_lookup=current_annual[["stock_code","organ_id","report_year","_weight"]]
    wfr_history=annual_history.merge(weight_lookup,
        on=["stock_code","organ_id","report_year"],how="left",validate="many_to_one")
    out["disp"]=factor_disp(current_annual,asof,config)

    fy1=consensus_fy1_year(asof)
    consensus={y:_mean(current_annual.loc[current_annual.report_year.eq(y)],"forecast_np").reindex(idx)
               for y in (fy1-1,fy1,fy1+1)}
    fy1_np=consensus[fy1].copy()
    if "actual_np_info_date" in m:
        known=m.actual_np_year.eq(fy1)&pd.to_datetime(
            m.actual_np_info_date,errors="coerce").le(asof)&m.actual_np.notna()
        fy1_np=fy1_np.where(~known,m.actual_np)
    fy1_weights=current_annual.loc[current_annual.report_year.eq(fy1),
        ["stock_code","organ_id","_weight"]].drop_duplicates(["stock_code","organ_id"],keep="last")
    out["ep_fy1"]=factor_ep_fy1(fy1_np,m)
    out["peg"]=factor_peg(consensus,fy1,fy1_np,m)
    out["score"]=factor_score(current,fy1_weights)
    out["tper"]=factor_tper(current,fy1_weights,m).reindex(idx)
    out["wfr"]=factor_wfr(wfr_history,asof,config)
    return out.reindex(columns=FACTOR_NAMES).replace([np.inf,-np.inf],np.nan).reset_index()

def load_zyyx_inputs(conn, asof, config=AnalystFactorConfig()):
    """Read only the point-in-time windows needed by one daily update."""
    asof=_date(asof)
    start=(asof-pd.Timedelta(days=max(config.lookback_days,config.revision_history_days))).date()
    end=(asof+pd.Timedelta(days=1)).date()
    history=(asof-pd.Timedelta(days=config.revision_history_days)).date()
    sql=f"""
    SELECT
        f.id,
        f.report_id,
        f.stock_code,
        f.report_type,
        f.reliability,
        f.organ_id,
        f.create_date,
        f.entrytime,
        f.report_year,
        f.report_quarter,
        f.forecast_np,
        f.forecast_eps,
        f.organ_rating_code,
        f.organ_rating_content,
        f.target_price_ceiling,
        f.target_price_floor,
        f.current_price,
        f.refered_capital,
        cap.report_capital,
        cap.real_capital,
        cap.is_capital_change
    FROM rpt_forecast_stk f
    LEFT JOIN rpt_report_capital cap
        ON cap.report_id=f.report_id AND cap.report_year=f.report_year
    WHERE EXISTS (SELECT 1 FROM rpt_organ_information oi
        WHERE oi.organ_id=f.organ_id AND oi.organ_type=5)
        AND f.entrytime>='{start}' AND f.entrytime<'{end}'
        AND DATEDIFF(day,f.create_date,f.entrytime) BETWEEN 0 AND {config.max_entry_delay_days}
        AND f.reliability>={config.min_reliability}
    """
    reports=pd.read_sql(sql,conn)
    return reports,None


def load_market_snapshot(date,ticks,jy_conn):
    sql=f"""
    SELECT
        C.SecuCode as stock_code,
        A.ClosePrice as close,
        A.TotalMV as total_mv,
        A.TotalMV/NULLIF(A.ClosePrice,0) as total_share
    FROM QT_StockPerformance A
    JOIN SecuMain C ON C.InnerCode=A.InnerCode
    WHERE A.TradingDay='{date}' AND C.SecuMarket IN (83,90) AND C.SecuCategory=1
    UNION ALL
    SELECT
        C.SecuCode as stock_code,
        B.ClosePrice as close,
        B.TotalMV as total_mv,
        B.TotalMV/NULLIF(B.ClosePrice,0) as total_share
    FROM LC_STIBPerformance B
    JOIN SecuMain C ON C.InnerCode=B.InnerCode
    WHERE B.TradingDay='{date}' AND C.SecuMarket IN (83,90) AND C.SecuCategory=1
    """
    out=pd.read_sql(sql,jy_conn)
    out["stock_code"]=out.stock_code.astype("string").str.zfill(6)
    return out.set_index('stock_code').reindex([t for t in ticks if t!=''])


def load_annual_actuals(date, jy_conn):
    asof=_date(asof)
    fy0 = asof.year if (asof.month, asof.day) >= (5, 1) else asof.year - 1

    sql=f"""
    WITH ranked AS (
        SELECT 
            S.SecuCode as stock_code,
            A.EndDate,
            A.InfoPublDate,
            A.NetProfit/10000.0 as actual_np,
            ROW_NUMBER() OVER(
                PARTITION BY S.SecuCode, A.EndDate 
                ORDER BY
                    CASE WHEN A.BulletinType=20 THEN 0 ELSE 1 END,
                    A.InfoPublDate,
                    A.ID
            ) rn
        FROM LC_IncomeStatementAll A 
        JOIN SecuMain S ON S.CompanyCode=A.CompanyCode
        WHERE A.EndDate='{fy0}-12-31' AND A.InfoPublDate<='{date}'
            AND A.IfMerged=1 AND A.IfAdjusted=2 AND A.IfComplete=1
            AND A.BulletinType IN (20,30) 
            AND S.SecuCategory=1 AND S.SecuMarket in (83,90)
    ) SELECT stock_code,EndDate,InfoPublDate,actual_np FROM ranked WHERE rn=1
    """
    out=pd.read_sql(sql,jy_conn)
    out["stock_code"] = out["stock_code"].astype("string").str.zfill(6)
    out["report_year"] = fy0
    return out.rename(columns={"EndDate": "end_date", "InfoPublDate": "info_pub_date"})


def get_zyyx_factors(date,ticks=None,conn=None,market=None,*,jy_conn=None,config=AnalystFactorConfig()):
    if conn is None:
        from ..config import get_zyyx_conn
        conn=get_zyyx_conn()
    if jy_conn is None:
        from ..config import get_jy_conn
        jy_conn=get_jy_conn()
    reports,_=load_zyyx_inputs(conn,date,config)
    market=load_market_snapshot(jy_conn,market,date)
    weight_fn=build_raw_accwt2_weight_fn(conn,jy_conn,date,config)
    out=calculate_analyst_factors(reports,date,market,config=config,weight_fn=weight_fn).set_index("tick")
    if ticks is not None:
        out=out.reindex(pd.Index([str(t).zfill(6) for t in ticks],name="tick"))
        out["cov"]=out["cov"].fillna(0.0)
    return out

def update_zyyx(date,dates,ticks,conn,root,market=None,*,jy_conn=None,config=AnalystFactorConfig()):
    """Persist one new trading-day row in seven binary matrices."""
    if jy_conn is None:
        from ..config import get_jy_conn
        jy_conn=get_jy_conn()
    if market is None:
        market=load_market_snapshot(date,dates,ticks,jy_conn,root)
    factors=get_zyyx_factors(date,ticks,conn,market,jy_conn=jy_conn,config=config)
    positions=np.flatnonzero(np.asarray([str(pd.Timestamp(d).date()) for d in dates])==str(_date(date).date()))
    if not len(positions): raise ValueError(f"{date} is not present in dates")
    folder=Path(root)/"zyyx_factors"; folder.mkdir(parents=True,exist_ok=True)
    for name in FACTOR_NAMES:
        path=folder/f"{name}.bin"
        arr=np.memmap(path,dtype=float,mode="r+" if path.exists() else "w+",shape=(len(dates),len(ticks)))
        arr[int(positions[0])]=factors[name].to_numpy(float); arr.flush()

__all__=["AnalystFactorConfig","FACTOR_NAMES","aggregate_report_authors",
         "build_raw_accwt2_weight_fn","calculate_analyst_factors","consensus_fy1_year","get_zyyx_factors","load_market_snapshot","query_market_snapshot","load_zyyx_inputs",
         "fit_accwt2_weight_fn","load_annual_actuals","make_accwt2_weight_fn","update_zyyx"]
