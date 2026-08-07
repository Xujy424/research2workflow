import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime


def update_industry(date, ticks, conn):
    sql_industry = f"""
    SELECT 
        B.SecuCode as tick,
        C.IndustryCode as industry
    FROM SecuMain B
    LEFT JOIN LC_ExgIndustry C ON B.InnerCode = C.InnerCode
    WHERE C.EndDate = '{date}'
    """
    industry = pl.read_database(sql_industry, conn).sort('tick').to_pandas().set_index('tick').reindex(index=ticks)['industry'].values.astype(str).flatten()
    return industry