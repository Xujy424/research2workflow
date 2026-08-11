from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import numpy as np
import pandas as pd
import polars as pl



@dataclass(frozen=True)
class AnalystFactorConfig:
    lookback_days: int = 90
    max_entry_delay_days: int = 7
    min_reliability: int = 5
    min_dispersion_orgs: int = 5
    revision_gap_days: int = 30
    revision_history_days: int = 180
    min_profit_denominator: float = 100.0


def _date(x):
    x = pd.Timestamp(x).normalize()
    if pd.isna(x): raise ValueError("invalid date")
    return x


def load_stock_marketinfo(date,ticks,jy_conn):
    """Load stock close, market value, and total shares."""
    sql=f"""
    SELECT
        C.SecuCode as stock_code,
        A.ClosePrice as [close],
        A.TotalMV as total_mv,
        A.TotalMV/NULLIF(A.ClosePrice,0) as total_share
    FROM QT_StockPerformance A
    JOIN SecuMain C ON C.InnerCode=A.InnerCode
    WHERE A.TradingDay='{date}' AND C.SecuMarket IN (83,90) AND C.SecuCategory=1
    UNION ALL
    SELECT
        C.SecuCode as stock_code,
        B.ClosePrice as [close],
        B.TotalMV as total_mv,
        B.TotalMV/NULLIF(B.ClosePrice,0) as total_share
    FROM LC_STIBPerformance B
    JOIN SecuMain C ON C.InnerCode=B.InnerCode
    WHERE B.TradingDay='{date}' AND C.SecuMarket IN (83,90) AND C.SecuCategory=1
    """
    out=pd.read_sql(sql,jy_conn)
    out["stock_code"]=out.stock_code.astype("string").str.zfill(6)
    return out.set_index('stock_code').reindex([t for t in ticks if t!=''])



def load_annual_actuals(fy,date,ticks,jy_conn):
    """Load annual actual profit and align it to the stock axis."""
    asof=_date(date)
    sql=f"""
    WITH ranked AS (
        SELECT
            S.SecuCode AS stock_code,
            A.EndDate AS end_date,
            A.InfoPublDate AS info_pub_date,
            A.NetProfit/10000.0 AS actual_np,
            ROW_NUMBER() OVER (
                PARTITION BY S.SecuCode,A.EndDate
                ORDER BY
                    CASE WHEN A.BulletinType=20 THEN 0 ELSE 1 END,
                    A.InfoPublDate,
                    A.ID
            ) AS rn
        FROM LC_IncomeStatementAll A
        JOIN SecuMain S ON S.CompanyCode=A.CompanyCode
        WHERE A.EndDate='{int(fy)}-12-31'
          AND A.InfoPublDate<='{asof.date()}'
          AND A.IfMerged=1
          AND A.IfAdjusted=2
          AND A.IfComplete=1
          AND A.BulletinType IN (20,30)
          AND S.SecuCategory=1
          AND S.SecuMarket IN (83,90)
    )
    SELECT stock_code,end_date,info_pub_date,actual_np
    FROM ranked
    WHERE rn=1
    """
    actual=pl.read_database(
        sql,jy_conn,infer_schema_length=None
    ).with_columns(
        pl.col("stock_code").cast(pl.String).str.zfill(6),
        pl.col("actual_np").cast(pl.Float64,strict=False),
        pl.lit(int(fy)).alias("report_year"),
    )
    stocks=[str(t).zfill(6) for t in ticks if t!=""]
    return pl.DataFrame({
        "stock_code":stocks,
        "_order":range(len(stocks)),
    }).join(
        actual,on="stock_code",how="left"
    ).sort("_order").drop("_order")

def load_forecasts(fy,conn,date,cfg,latest_only=False):
    """Load annual forecasts, optionally keeping the latest row per stock and institution."""
    asof=_date(date)
    sql=f"""
    SELECT
        f.id,
        f.report_id,
        f.stock_code,
        f.organ_id,
        f.create_date,
        f.entrytime,
        f.report_year,
        f.report_quarter,
        f.forecast_np
    FROM rpt_forecast_stk f
    WHERE f.report_year={int(fy)}
      AND f.report_quarter=4
      AND f.entrytime<='{asof.date()}'
      AND f.create_date<=f.entrytime
      AND f.entrytime<=DATEADD(day,{cfg.max_entry_delay_days},f.create_date)
      AND f.reliability>={cfg.min_reliability}
    """
    out=(
        pl.read_database(sql,conn)
        .with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6),
            pl.col("create_date").str.to_datetime(),
        )
        .filter(
            pl.col("stock_code").str.contains(r"^(00|30|60|68|92)")
            &pl.col("forecast_np").is_not_null()
        )
        .sort(["stock_code","organ_id","entrytime","report_id","id"])
    )
    if latest_only:
        out=out.unique(
            subset=["stock_code","organ_id"],
            keep="last",
            maintain_order=True,
        )
    return out

def calc_afe(date,ticks,conn,jy_conn,cfg,latest_only=True,fy=None):
    asof=_date(date)
    if fy is None:
        fy=asof.year-1 if (asof.month,asof.day)>=(5,1) else asof.year-2
    actual=load_annual_actuals(fy,date,ticks,jy_conn)
    forecast=load_forecasts(fy,conn,date,cfg, latest_only)
    out=forecast.join(
        actual.select("stock_code","actual_np","info_pub_date"),
        on="stock_code",
        how="left",
    ).filter(
        pl.col("create_date")<pl.col("info_pub_date")
    ).filter(
            pl.col("create_date")>=pl.col("info_pub_date")-pl.duration(days=cfg.lookback_days)
    ).sort(
        ["stock_code","organ_id","entrytime","report_id","id"]
    )
    if latest_only:
        out=out.unique(
            subset=["stock_code","organ_id"],keep="last",maintain_order=True
        )
    else:
        out=out.unique(
            subset=["report_id","stock_code","report_year","report_quarter"],
            keep="last",maintain_order=True
        )
    return out.with_columns(
        (pl.col("forecast_np")-pl.col("actual_np")).abs().alias("afe")
    )

def calc_pafe(date,ticks,conn,jy_conn,cfg,latest_only=True,fy=None):
    pafe=calc_afe(date,ticks,conn,jy_conn,cfg,latest_only,fy)
    mean_afe=pl.col("afe").mean().over("stock_code")
    return pafe.with_columns(
        ((pl.col("afe")-mean_afe)/mean_afe).alias("pafe")
    )


def get_fit_data(date,ticks,conn,jy_conn,cfg):
    asof=_date(date)
    dates=[asof-pd.DateOffset(years=2), asof-pd.DateOffset(years=1), asof]
    outs=[]
    for date in dates:
        outs.append(calc_pafe(date,ticks,conn,jy_conn,cfg,False))
    data=pl.concat(outs).select(["id","report_id","stock_code","organ_id","report_year","pafe","create_date","entrytime"])
    data = data.with_columns(pl.col('create_date').dt.offset_by('-1y').alias('last_yr_date'))
    return data


def load_analyst_history(start_date,end_date,conn,report_ids,cfg):
    """Load history for authors appearing in the target reports."""
    start,end=_date(start_date),_date(end_date)

    reports=",".join(map(str,pl.Series(report_ids).drop_nulls().unique().to_list())) or "NULL"

    sql = f"""
        SELECT DISTINCT author_id
        FROM rpt_report_author
        WHERE report_id IN ({reports})
    """
    author_ids=pl.read_database(sql,conn)["author_id"].drop_nulls().unique().to_list()    
    authors=",".join(map(str,author_ids)) or "NULL"

    sql=f"""
    SELECT
        f.report_id,
        f.stock_code,
        f.report_year,
        f.create_date,
        f.entrytime,
        ra.author_id,
        ra.organ_id,
        CASE WHEN ai.y1 NOT BETWEEN 1753 AND 9999
               OR COALESCE(NULLIF(ai.m1,0),1) NOT BETWEEN 1 AND 12
             THEN NULL ELSE CONVERT(datetime,
            CAST(ai.y1 AS varchar(4))+'-'+RIGHT('0'+CAST(COALESCE(NULLIF(ai.m1,0),1) AS varchar(2)),2)+'-01')
        END AS author_start_date,
        CASE WHEN ai.y2<=1900 OR ai.y2>9999
               OR COALESCE(ai.m2,0) NOT BETWEEN 1 AND 12
             THEN NULL ELSE CONVERT(datetime,
            CAST(ai.y2 AS varchar(4))+'-'+RIGHT('0'+CAST(COALESCE(ai.m2,1) AS varchar(2)),2)+'-01')
        END AS author_end_date,
        ind.industry_code,
        CASE WHEN EXISTS (
            SELECT 1 FROM der_new_fortune_author nf
            WHERE nf.author_id=ra.author_id
              AND nf.report_year=YEAR(f.create_date)-1
        ) THEN 1 ELSE 0 END AS xcf
    FROM rpt_forecast_stk f
    JOIN rpt_report_author ra ON ra.report_id=f.report_id
    OUTER APPLY (
        SELECT TOP 1 a.y1,a.m1,a.y2,a.m2 FROM rpt_author_information a
        WHERE a.author_id=ra.author_id ORDER BY a.y1,a.m1,a.id
    ) ai
    OUTER APPLY (
        SELECT TOP 1 q.industry_code FROM qt_indus_constituents q
        WHERE q.stock_code=f.stock_code
          AND q.standard_code='908' AND q.industry_level=2
          AND q.into_date<=f.create_date
          AND (q.out_date IS NULL OR q.out_date>f.create_date)
        ORDER BY q.into_date DESC,q.id DESC
    ) ind
    WHERE ra.author_id IN ({authors})
      AND f.report_quarter=4
      AND f.create_date>='{start.date()}' AND f.create_date<='{end.date()}'
      AND f.create_date<=f.entrytime
      AND f.entrytime<=DATEADD(day,{cfg.max_entry_delay_days},f.create_date)
      AND f.reliability>={cfg.min_reliability}
    """
    history=pl.read_database(sql,conn)
    first=pl.read_database(f"""
        SELECT ra.author_id,f.stock_code,MIN(f.create_date) AS first_stock_date
        FROM rpt_report_author ra
        JOIN rpt_forecast_stk f ON f.report_id=ra.report_id
        WHERE ra.author_id IN ({authors}) AND f.create_date<='{end.date()}'
          AND f.report_quarter=4
          AND f.create_date<=f.entrytime
          AND f.entrytime<=DATEADD(day,{cfg.max_entry_delay_days},f.create_date)
          AND f.reliability>={cfg.min_reliability}
        GROUP BY ra.author_id,f.stock_code
    """,conn)
    history=history.join(first,on=["author_id","stock_code"],how="left")    
    return history.with_columns(
        pl.col("stock_code").cast(pl.String).str.zfill(6),
        pl.col("create_date").cast(pl.String).str.to_datetime(strict=False),
        pl.col("entrytime").cast(pl.Datetime,strict=False),
        pl.col("author_start_date").cast(pl.String).str.to_datetime(strict=False),
        pl.col("author_end_date").cast(pl.String).str.to_datetime(strict=False),
        pl.col("first_stock_date").cast(pl.String).str.to_datetime(strict=False),
    )

def calc_author_accuracy(date,ticks,conn,jy_conn,cfg,pafe_data):
    target_years=pafe_data["report_year"].drop_nulls().unique().to_list()
    source_years={int(year)-1 for year in target_years}
    available_years=set(pafe_data["report_year"].drop_nulls().unique().to_list())
    accuracy_data=pafe_data.select("report_id","report_year","pafe")

    missing_years=source_years-available_years
    if missing_years:
        prior=[
            calc_pafe(date,ticks,conn,jy_conn,cfg,False,fy=year).select("report_id","report_year","pafe") 
            for year in sorted(missing_years)
        ]
        accuracy_data=pl.concat([*prior,accuracy_data])
    reports=",".join(map(str,accuracy_data["report_id"].drop_nulls().unique().to_list())) or "NULL"
    authors=pl.read_database(f"""
        SELECT DISTINCT report_id,author_id
        FROM rpt_report_author
        WHERE report_id IN ({reports})
    """,conn)
    return (accuracy_data.join(authors,on="report_id",how="inner")
        .group_by("author_id","report_year").agg(
            (-pl.col("pafe").mean()).alias("acc")
        ).with_columns(
            (pl.col("report_year")+1).alias("report_year")
        ))

def merge_analyst_attributes(fit_data,history,author_accuracy):
    targets=(fit_data.join(
        history.select(
            "report_id","author_id","author_start_date","author_end_date","xcf"
        ).unique(["report_id","author_id"]),
        on="report_id",how="left",
    ).join(
        history.select(
            "author_id","stock_code","first_stock_date"
        ).unique(["author_id","stock_code"]),
        on=["author_id","stock_code"],how="left",
    ).with_row_index("sample_id"))
    targets=targets.with_columns(
        ((pl.min_horizontal("create_date","author_end_date")-pl.col("author_start_date"))
         .dt.total_days().clip(lower_bound=0).sqrt()).alias("gexp"),
        ((pl.col("create_date")-pl.col("first_stock_date")).dt.total_days().clip(lower_bound=0).sqrt()).alias("fexp"),
        (pl.col("create_date")-pl.datetime(pl.col("report_year"),5,1)).dt.total_days().alias("horizon"),
    )
    targets=targets.join(
        author_accuracy,on=["author_id","report_year"],how="left"
    )

    #history锛氫竴绡囨姤鍛?脳 涓€鍙偂绁?脳 涓€浣嶄綔鑰?
    events=history.select(
        "report_id","author_id","organ_id","stock_code","industry_code","create_date"
    ).unique(["report_id","author_id","stock_code"])
    author_window=(targets.select(
        "sample_id","author_id","last_yr_date","create_date"
    ).join(events,on="author_id",how="left",suffix="_event").filter(
        pl.col("create_date_event").is_between(pl.col("last_yr_date"),pl.col("create_date"),closed="both"
        )
    ).group_by("sample_id").agg(
        pl.col("stock_code").n_unique().sqrt().alias("ncomp"),
        pl.col("industry_code").drop_nulls().n_unique().sqrt().alias("ninds"),
    ))
    organ_window=(targets.select(
        "sample_id","organ_id","last_yr_date","create_date"
    ).join(events,on="organ_id",how="left",suffix="_event").filter(
        pl.col("create_date_event").is_between(pl.col("last_yr_date"),pl.col("create_date"),closed="both"
        )
    ).group_by("sample_id").agg(
        pl.col("author_id").drop_nulls().n_unique().sqrt().alias("naut")
    ))
    targets=targets.join(
        author_window,on="sample_id",how="left"
    ).join(
        organ_window,on="sample_id",how="left"
    )

    report_attributes=targets.group_by("id").agg(
        pl.col("gexp","fexp","ncomp","ninds","naut","acc","horizon").mean(),
        pl.col("xcf").max(),
    )
    return fit_data.join(report_attributes,on="id",how="left")


def build_pmafe_samples(date,ticks,conn,jy_conn,cfg,return_author_accuracy=False):
    """Build three-year PMAFE samples and attach report-level analyst attributes."""
    fit_data=get_fit_data(date,ticks,conn,jy_conn,cfg)
    history_start=fit_data.select(pl.col("last_yr_date").min()).item()
    history_end=fit_data.select(pl.col("create_date").max()).item()
    history=load_analyst_history(
        history_start,history_end,conn,fit_data["report_id"],cfg
    )
    author_accuracy=calc_author_accuracy(
        date,ticks,conn,jy_conn,cfg,fit_data
    )
    acc = author_accuracy.filter(pl.col('report_year')<pl.col('report_year').max())
    samples=merge_analyst_attributes(fit_data,history,acc)
    if return_author_accuracy:
        return samples,author_accuracy
    return samples

PMAFE_MODEL_FEATURES=("horizon","acc","gfexp","ninds","naut","ncomp","xcf")
def fit_pmafe_model(samples):
    """Fit the report's stock-year demeaned PMAFE regression without an intercept."""
    data=samples.with_columns(
        ((pl.col("gexp")+pl.col("fexp"))/2).alias("gfexp")
    )
    keys=["stock_code","report_year"]
    regressors=[]
    for name in PMAFE_MODEL_FEATURES:
        if name=="xcf":
            regressors.append(name)
            continue
        column=name+"_dm"
        data=data.with_columns(
            (pl.col(name)-pl.col(name).mean().over(keys)).alias(column)
        )
        regressors.append(column)
    data=data.with_columns(
        (pl.col("pafe")-pl.col("pafe").mean().over(keys)).alias("pafe_dm")
    )
    valid=data.drop_nulls(["pafe_dm",*regressors]).filter(
        pl.all_horizontal(pl.col(name).is_finite() for name in ["pafe_dm",*regressors])
    )
    if valid.height<=len(PMAFE_MODEL_FEATURES):
        raise ValueError("insufficient complete PMAFE regression samples")
    x=valid.select(regressors).to_numpy().astype(float)
    y=valid["pafe_dm"].to_numpy().astype(float)
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    coefficients=dict(zip(PMAFE_MODEL_FEATURES,beta,strict=True))
    fitted=sum(
        pl.col(name if name=="xcf" else name+"_dm")*value
        for name,value in coefficients.items()
    )
    return coefficients,valid.with_columns(fitted.alias("fitted_pmafe"))


def predict_next_pmafe(next_samples,coefficients):
    """Predict next-period PMAFE from raw attributes using fitted coefficients."""
    missing=set(PMAFE_MODEL_FEATURES)-set(coefficients)
    if missing:
        raise ValueError(f"missing PMAFE coefficients: {sorted(missing)}")
    data=next_samples.with_columns(
        ((pl.col("gexp")+pl.col("fexp"))/2).alias("gfexp")
    )
    prediction=sum(pl.col(name)*float(coefficients[name]) for name in PMAFE_MODEL_FEATURES)
    return data.with_columns(prediction.alias("predicted_pmafe"))


def calc_last_accwt(date,ticks,conn,jy_conn,cfg):
    accwt=calc_pafe(date,ticks,conn,jy_conn,cfg, True)
    mean_pafe=pl.col("pafe").mean().over("stock_code")
    std_pafe=pl.col("pafe").std().over("stock_code")
    accwt=accwt.with_columns(
        ((pl.col("pafe")-mean_pafe)/std_pafe).alias("pafe_z")
    ).with_columns(
        pl.when(pl.col("pafe_z")<0)
        .then(-pl.col("pafe_z"))
        .otherwise(0.0)
        .alias("accwt")
    )
    return accwt.with_columns(
        (pl.col("accwt")/pl.col("accwt").sum().over("stock_code")).alias("accwt")
    )

def calc_accwt(date,ticks,conn,jy_conn,cfg):
    samples,author_accuracy=build_pmafe_samples(
        date,ticks,conn,jy_conn,cfg,return_author_accuracy=True
    )
    coefficients,_=fit_pmafe_model(samples)

    asof=_date(date)
    fy=asof.year if (asof.month,asof.day)>=(5,1) else asof.year-1
    current=load_forecasts(fy,conn,date,cfg,latest_only=True).filter(
        pl.col("entrytime")>=pl.lit(
            asof.to_pydatetime()-pd.Timedelta(days=cfg.lookback_days)
        )
    ).with_columns(
        pl.lit(None,dtype=pl.Float64).alias("pafe"),
        pl.col("create_date").dt.offset_by("-1y").alias("last_yr_date"),
    )

    history=load_analyst_history(
        asof-pd.DateOffset(years=1),asof,conn,current["report_id"],cfg
    )

    author_accuracy=author_accuracy.filter(
        pl.col("report_year")==fy
    )

    next_samples=merge_analyst_attributes(current,history,author_accuracy)
    predicted=predict_next_pmafe(next_samples,coefficients)

    mean=pl.col("predicted_pmafe").mean().over("stock_code")
    std=pl.col("predicted_pmafe").std().over("stock_code")
    count=pl.len().over("stock_code")
    return predicted.with_columns(
        ((pl.col("predicted_pmafe")-mean)/std).alias("pmafe_z")
    ).with_columns(
        pl.when(count==1).then(1.0)
        .when(std.is_null() | (std==0)).then(1.0/count)
        .when(pl.col("pmafe_z")<0).then(-pl.col("pmafe_z"))
        .otherwise(0.0).alias("accwt")
    ).with_columns(
        (pl.col("accwt")/pl.col("accwt").sum().over("stock_code")).alias("accwt")
    )


def calc_next_accwt(date,ticks,conn,jy_conn,cfg):
    samples,author_accuracy=build_pmafe_samples(
        date,ticks,conn,jy_conn,cfg,return_author_accuracy=True
    )
    coefficients,_=fit_pmafe_model(samples)

    asof=_date(date)
    fy=asof.year if (asof.month,asof.day)>=(5,1) else asof.year-1
    current=load_forecasts(fy,conn,date,cfg,latest_only=True).with_columns(
        pl.lit(None,dtype=pl.Float64).alias("pafe"),
        pl.col("create_date").dt.offset_by("-1y").alias("last_yr_date"),
    )

    history=load_analyst_history(
        current["last_yr_date"].min(),
        current["create_date"].max(),
        conn,current["report_id"],cfg,
    )
    author_accuracy=author_accuracy.filter(
        pl.col("report_year")==fy
    )

    next_samples=merge_analyst_attributes(current,history,author_accuracy)
    predicted=predict_next_pmafe(next_samples,coefficients)

    mean=pl.col("predicted_pmafe").mean().over("stock_code")
    std=pl.col("predicted_pmafe").std().over("stock_code")
    count=pl.len().over("stock_code")
    return predicted.with_columns(
        ((pl.col("predicted_pmafe")-mean)/std).alias("pmafe_z")
    ).with_columns(
        pl.when(count==1).then(1.0)
        .when(std.is_null() | (std==0)).then(1.0/count)
        .when(pl.col("pmafe_z")<0).then(-pl.col("pmafe_z"))
        .otherwise(0.0).alias("accwt")
    ).with_columns(
        (pl.col("accwt")/pl.col("accwt").sum().over("stock_code")).alias("accwt")
    )

def calc_con_forecast(date,ticks,conn,jy_conn,cfg):
    return calc_next_accwt(date,ticks,conn,jy_conn,cfg).with_columns(
        (pl.col("forecast_np")*pl.col("accwt")).alias("weighted_forecast")
    ).group_by("stock_code").agg(
        pl.col("weighted_forecast").sum().alias("con_forecast"),
        pl.len().alias("num_forecasts"),
    )





if __name__=="__main__":
    from pathlib import Path
    import sys
    import polars as pl
    import numpy as np
    import pandas as pd
    import bottleneck as bn
    import bisect
    from datetime import datetime, timedelta
    from dataclasses import dataclass
    import pymssql
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    ticks = np.load(r"D:\data\axis\ticks.npy", allow_pickle=True)
    date = '2024-06-30'
    cfg = AnalystFactorConfig()

    zyyx_url = URL.create(drivername="mssql+pymssql",
             username="zyyxReader",
             password="zyyx!5893@Fund",
             host="10.110.0.106",
             database="zyyx",
             query={"charset": "utf8"})
    zyyx_engine = create_engine(
        zyyx_url,
        connect_args={
            "tds_version": "7.0",
            "charset": "utf8"
        }
    )
    conn = zyyx_engine.connect()

    JY_CONFIG = {
        "server": "10.10.0.102",
        "user": "jydbReader",
        "password": "jy@9043!Reader",
        "database": "jydb",
        "charset": "cp936",
    }
    jy_conn = pymssql.connect(**JY_CONFIG)


    con_np = calc_con_forecast(date,ticks,conn,jy_conn,cfg)
    print(con_np)

    sql=f"""
    SELECT
        f.id,
        f.stock_code,
        f.con_np,
        f.con_np_type
    FROM con_forecast_stk f
    WHERE f.con_date='{date}' AND f.con_year={int(date[:4])}
    """
    correct_con_np = pl.read_database(sql,conn).sort('stock_code')
    print(correct_con_np)


































