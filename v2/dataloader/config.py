"""Configuration and lazy connection factories for dataloader."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pymssql

from .cifsLoader import CIFSLoader
import exchange_calendars as xcals



START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
@lru_cache(maxsize=1)
def get_xshg_calendar():
    return xcals.get_calendar("XSHG")


ROOT = 'D:/data/'


JY_CONFIG = {
    "server": "10.10.0.102",
    "user": "jydbReader",
    "password": "jy@9043!Reader",
    "database": "jydb",
    "charset": "cp936",
}
@lru_cache(maxsize=1)
def get_jy_conn():
    return pymssql.connect(**JY_CONFIG)


STR_CONN_URL = "mysql+pymysql://QuantReader:Quant%40Reader%21zsfund.com@10.10.6.101:9030/HighFrequency"
@lru_cache(maxsize=1)
def get_str_engine():
    return create_engine(STR_CONN_URL)


ZYYX_URL = URL.create(
    drivername="mssql+pymssql",
    username="zyyxReader",
    password="zyyx!5893@Fund",
    host="10.110.0.106",
    database="zyyx",
    query={"charset": "utf8"},
)
@lru_cache(maxsize=1)
def get_zyyx_conn():
    engine = create_engine(
        ZYYX_URL,
        connect_args={
            "tds_version": "7.0",
            "charset": "utf8",
        },
    )
    return engine.connect()


CIFTABLE_PATTERNS = {
    "szwt": "mdl_6_33_0",
    "szcj": "mdl_6_36_0",
    "sh": "mdl_4_24_0",
    "szshot": "mdl_6_28_0",
    "shshot": "MarketData",
}
cifs = CIFSLoader("xujiayi", "ZSfund.com@202606")
L2DATA_PATH = ROOT + 'l2/'


