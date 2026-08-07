import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pymssql
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

import update_zyyx_xjy as model


def month_ends(start, end, conn):
    sql = f"""
    SELECT MAX(TradingDay) trade_date
    FROM QT_StockPerformance
    WHERE TradingDay BETWEEN '{start}' AND '{end}'
    GROUP BY YEAR(TradingDay),MONTH(TradingDay)
    ORDER BY trade_date
    """
    return [pd.Timestamp(x) for x in pl.read_database(sql, conn)["trade_date"]]


def gogoal(date, conn):
    date = pd.Timestamp(date)
    fy = date.year if (date.month, date.day) >= (5, 1) else date.year - 1
    return pl.read_database(f"""
        SELECT stock_code,con_np gogoal_forecast
        FROM con_forecast_stk
        WHERE con_date='{date.date()}' AND con_year={fy}
    """, conn).with_columns(
        pl.col("stock_code").cast(pl.String).str.zfill(6),
        pl.col("gogoal_forecast").cast(pl.Float64, strict=False),
    )


def forward_return(start, end, conn):
    sql = f"""
    SELECT S.SecuCode stock_code,Q.ChangePCT FROM QT_StockPerformance Q
    JOIN SecuMain S ON S.InnerCode=Q.InnerCode
    WHERE Q.TradingDay>'{start.date()}' AND Q.TradingDay<='{end.date()}'
      AND S.SecuCategory=1 AND S.SecuMarket IN (83,90)
    UNION ALL
    SELECT S.SecuCode stock_code,Q.ChangePCT FROM LC_STIBPerformance Q
    JOIN SecuMain S ON S.InnerCode=Q.InnerCode
    WHERE Q.TradingDay>'{start.date()}' AND Q.TradingDay<='{end.date()}'
      AND S.SecuCategory=1 AND S.SecuMarket IN (83,90)
    """
    return (pl.read_database(sql, conn).with_columns(
        pl.col("stock_code").cast(pl.String).str.zfill(6),
        pl.col("ChangePCT").cast(pl.Float64, strict=False) / 100,
    ).group_by("stock_code").agg(
        ((pl.col("ChangePCT") + 1).product() - 1).alias("forward_return")
    ))


def run_period(date, next_date, ticks, conn, jy_conn, cfg, groups=5):
    weighted = model.calc_accwt(date, ticks, conn, jy_conn, cfg)
    print(f"[{date.date()}] Accwt rows: {weighted.height}",flush=True)
    consensus = weighted.group_by("stock_code").agg(
        (pl.col("forecast_np") * pl.col("accwt")).sum().alias("accwt_forecast"),
        pl.col("forecast_np").mean().alias("equal_forecast"),
    ).join(gogoal(date, conn), on="stock_code", how="left")
    print(f"[{date.date()}] loading market and returns",flush=True)
    market = pl.from_pandas(model.load_stock_marketinfo(date, ticks, jy_conn).reset_index())
    data = (consensus.join(market, on="stock_code", how="left")
            .join(forward_return(date, next_date, jy_conn), on="stock_code", how="left"))

    valid = data.select("accwt_forecast", "equal_forecast", "gogoal_forecast").drop_nulls().to_pandas()
    corr = {"date": date, "n": len(valid)}
    for left in ("accwt", "equal"):
        corr[f"pearson_{left}_gogoal"] = valid[f"{left}_forecast"].corr(valid.gogoal_forecast)
        corr[f"spearman_{left}_gogoal"] = valid[f"{left}_forecast"].corr(
            valid.gogoal_forecast, method="spearman")

    frame = data.select("forward_return", "total_mv", "accwt_forecast",
                        "equal_forecast", "gogoal_forecast").to_pandas()
    rows = []
    for method in ("accwt", "equal", "gogoal"):
        factor = frame[f"{method}_forecast"] / frame.total_mv
        ok = factor.notna() & frame.forward_return.notna() & np.isfinite(factor)
        part = frame.loc[ok, ["forward_return"]].copy()
        part["group"] = pd.qcut(factor[ok].rank(method="first"), groups, labels=False) + 1
        for group, value in part.groupby("group").forward_return.mean().items():
            rows.append({"date": date, "method": method, "group": int(group), "return": value})
    print(f"[{date.date()}] period completed",flush=True)
    return pl.DataFrame([corr]), pl.from_pandas(pd.DataFrame(rows)), data


def main(start, end, output):
    ticks = np.load(r"D:\data\axis\ticks.npy", allow_pickle=True)
    zyyx = URL.create("mssql+pymssql", username="zyyxReader",
                      password="zyyx!5893@Fund", host="10.110.0.106",
                      database="zyyx", query={"charset": "utf8"})
    engine = create_engine(zyyx, connect_args={"tds_version": "7.0", "charset": "utf8"})
    conn = engine.connect()
    jy_conn = pymssql.connect(server="10.10.0.102", user="jydbReader",
                              password="jy@9043!Reader", database="jydb", charset="cp936")
    dates = month_ends(start, end, jy_conn)
    output.mkdir(parents=True, exist_ok=True)
    for date, next_date in zip(dates[:-1], dates[1:]):
        corr, groups, snapshot = run_period(
            date, next_date, ticks, conn, jy_conn, model.AnalystFactorConfig())
        stamp = date.strftime("%Y%m%d")
        corr.write_csv(output / f"correlation_{stamp}.csv")
        groups.write_csv(output / f"groups_{stamp}.csv")
        snapshot.write_parquet(output / f"snapshot_{stamp}.parquet")
        print(corr)
        print(groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("zyyx_backtest"))
    args = parser.parse_args()
    main(args.start, args.end, args.output)

