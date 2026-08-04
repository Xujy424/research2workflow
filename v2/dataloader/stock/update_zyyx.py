from __future__ import annotations

from pathlib import Path
import sys
import polars as pl
import numpy as np
import pandas as pd
import bottleneck as bn
import bisect


if __package__:
    from ..config import *
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from v2.dataloader.config import *

conn = get_zyyx_conn()
date = '2026-08-04'


sql = f"""
SELECT * FROM rpt_forecast_stk 
WHERE entrytime BETWEEN DATEADD(day, -90, '{date}') AND '{date}';
"""
dff = pl.read_database(sql, conn)
print(dff)