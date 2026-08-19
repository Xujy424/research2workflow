import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime

from ..download.update_axis import init_empty_field


# 1.0 申万行业分类命名更改映射
update_map = {
    "黑色金属": "钢铁",
    "建筑建材": "建筑材料",
    "化工": "基础化工",
    "交运设备": "汽车",
    "信息服务": "传媒",
    "信息设备": "电子",
    "商业贸易": "商贸零售",
    "纺织服装": "纺织服饰",
    "休闲服务": "社会服务",
    "餐饮旅游": "社会服务",
    "电气设备": "电力设备",
}

# 1.1 行业→行业值映射
industry_to_id = {
    "商贸零售": 0, "轻工制造": 1, "汽车": 2, "美容护理": 3, "房地产": 4,
    "国防军工": 5, "通信": 6, "煤炭": 7, "交通运输": 8, "公用事业": 9,
    "机械设备": 10, "电力设备": 11, "环保": 12, "食品饮料": 13, "计算机": 14,
    "纺织服饰": 15, "家用电器": 16, "医药生物": 17, "钢铁": 18, "社会服务": 19,
    "有色金属": 20, "非银金融": 21, "综合": 22, "建筑装饰": 23, "农林牧渔": 24,
    "银行": 25, "传媒": 26, "基础化工": 27, "建筑材料": 28, "石油石化": 29, "电子": 30
}

# 1.2 行业→板块映射
industry_to_sector = {
    "商贸零售": "消费", "轻工制造": "消费", "汽车": "制造", "美容护理": "消费", "房地产": "金融地产",
    "国防军工": "制造", "通信": "科技", "煤炭": "周期", "交通运输": "周期", "公用事业": "周期",
    "机械设备": "制造", "电力设备": "制造", "环保": "制造", "食品饮料": "消费", "计算机": "科技",
    "纺织服饰": "消费", "家用电器": "消费", "医药生物": "消费", "钢铁": "周期", "社会服务": "消费",
    "有色金属": "周期", "非银金融": "金融地产", "综合": "制造", "建筑装饰": "周期", "农林牧渔": "周期",
    "银行": "金融地产", "传媒": "科技", "基础化工": "周期", "建筑材料": "周期", "石油石化": "周期", "电子": "科技"
}

# 1.3 板块→板块ID映射
sector_to_id = {
    "消费": 0,
    "制造": 1,
    "金融地产": 2,
    "科技": 3,
    "周期": 4
}



def _daily_industry_codes(date, ticks, conn):
    """按有效期读取一天的申万行业，并返回行业、板块编码。"""
    sql = f'''
    select
        A.InfoPublDate   as 'start_date',
        A.CancelDate     as 'end_date',
        B.SecuCode       as 'tick',
        B.CompanyCode    as 'code',
        A.FirstIndustryName as 'industry'
    from DZ_ExgIndustry A
    left join SecuMain B
        on A.CompanyCode = B.CompanyCode
    where A.Standard=38
        and B.SecuMarket in (83,90)
        and B.SecuCategory=1
    union all 
    select
        C.InfoPublDate   as 'start_date',
        C.CancelDate     as 'end_date',
        B.SecuCode       as 'tick',
        B.CompanyCode    as 'code',
        C.FirstIndustryName as 'industry'
    from LC_STIBExgIndustry C
    left join SecuMain B
        on C.CompanyCode = B.CompanyCode
    where C.Standard=38
        and B.SecuMarket in (83,90)
        and B.SecuCategory=1
    '''
    history = pl.read_database(sql, conn).filter(pl.col("tick").is_in(ticks)).unique()
    if history.is_empty():
        empty = np.full(len(ticks), np.nan, dtype=float)
        return empty, empty.copy()
    history = (
        history.with_columns([
            pl.col("tick").cast(pl.String),
            pl.col("start_date").cast(pl.Datetime),
            pl.col("end_date").cast(pl.Datetime).fill_null(pl.datetime(2099, 12, 31)),
            pl.col("industry").replace(update_map),
        ]).sort(["tick", "start_date"])
    )

    # 申万分类变化前的合并行业，用股票完整历史中的新分类拆分。
    bank = set(history.filter(pl.col("industry") == "银行")["tick"].to_list())
    financial = set(history.filter(pl.col("industry") == "非银金融")["tick"].to_list())
    coal = set(history.filter(pl.col("industry") == "煤炭")["tick"].to_list())
    petro = set(history.filter(pl.col("industry") == "石油石化")["tick"].to_list())
    history = history.with_columns(
        pl.when(pl.col("industry") == "金融服务")
        .then(pl.when(pl.col("tick").is_in(bank)).then(pl.lit("银行"))
              .when(pl.col("tick").is_in(financial)).then(pl.lit("非银金融"))
              .otherwise(pl.lit("非银金融")))
        .when(pl.col("industry") == "采掘")
        .then(pl.when(pl.col("tick").is_in(petro)).then(pl.lit("石油石化"))
              .when(pl.col("tick").is_in(coal)).then(pl.lit("煤炭"))
              .otherwise(pl.lit("煤炭")))
        .otherwise(pl.col("industry"))
        .alias("industry")
    )

    history = (
        history
        .with_columns([
            pl.col("industry").replace(industry_to_id).alias("industry_id"),
            pl.col("industry").replace(industry_to_sector).alias("sector"),
        ])
        .with_columns(
            pl.col("sector").replace(sector_to_id).alias("sector_id")
        )
    )

    target = pd.Timestamp(date).to_pydatetime()
    active = (
        history.filter(
            (pl.col("start_date") <= target) & (target < pl.col("end_date"))
        )
        .sort(["tick", "start_date"])
        #.unique(subset="tick", keep="last")
        .select(["tick", "industry_id", "sector_id"])
    )
    cross_section = (
        active.to_pandas()
        .set_index("tick")
        .reindex(index=ticks)
    )
    industry = cross_section["industry_id"].values.astype(float).flatten()
    sector = cross_section["sector_id"].values.astype(float).flatten()
    return industry, sector


def _write_daily_mask(date, dates, ticks, root, name, values):
    dt = np.searchsorted(dates, pd.to_datetime(date))
    n_valid = np.count_nonzero(ticks != "")
    path = root / "mask" / f"{name}.bin"
    if not path.exists():
        init_empty_field(dates, ticks, 'mask', name, np.float32, dim=None)
    arr = np.memmap(path, dtype=np.float32, mode="r+", shape=(len(dates), len(ticks)))
    arr[dt,:n_valid] = values
    arr.flush()


def update_industry(date, dates, ticks, conn, root):
    """每日更新申万一级行业编码。"""
    n_valid = np.count_nonzero(ticks != "")
    valid_ticks = ticks[:n_valid]
    
    values, _ = _daily_industry_codes(date, valid_ticks, conn)
    _write_daily_mask(date, dates, ticks, root, "industry", values)
    return values


def update_sector(date, dates, ticks, conn, root):
    """每日更新由申万一级行业映射得到的板块编码。"""
    n_valid = np.count_nonzero(ticks != "")
    valid_ticks = ticks[:n_valid]

    _, values = _daily_industry_codes(date, valid_ticks, conn)
    _write_daily_mask(date, dates, ticks, root, "sector", values)
    return values




if __name__ == '__main__':
    from pathlib import Path
    import pymssql

    dates = np.load('/data/xujiayi/xjy/axis/dates.npy', allow_pickle=True)
    ticks = np.load('/data/xujiayi/xjy/axis/ticks.npy', allow_pickle=True)
    root = Path('/data/xujiayi/xjy/')


    JY_CONFIG = {
            "server": "10.10.0.102",
            "user": "jydbReader",
            "password": "jy@9043!Reader",
            "database": "jydb",
            "charset": "cp936",
        }
    jy_conn = pymssql.connect(**JY_CONFIG)

    update_industry("2024-06-14", dates, ticks, jy_conn, root)
