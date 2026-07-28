from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import polars as pl

try:
    from . import CIFTABLE_PATTERNS, cifs, L2DATA_PATH
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataloader import CIFTABLE_PATTERNS, cifs, L2DATA_PATH


def normalize_date(date: dt.date | dt.datetime | str) -> str:
    if isinstance(date, (dt.datetime, dt.date)):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "").replace(".", "").strip("/")


def autoload_l2data(date: dt.date | dt.datetime | str, mode: str = "offline"):
    date = normalize_date(date)
    all_filenames = cifs.get_filenames(f"L2Data/{date}") or []
    matched_filenames = {
        key: filename
        for filename in all_filenames
        for key, pattern in CIFTABLE_PATTERNS.items()
        if pattern in filename
    }

    if mode == "online":
        return {
            key: cifs.get_data_csv(cifs._raw_share_name, os.path.join("L2Data", date, filename))
            for key, filename in matched_filenames.items()
        }

    if mode == "offline":
        for key, filename in matched_filenames.items():
            df = cifs.get_data_csv(cifs._raw_share_name, os.path.join("L2Data", date, filename))
            if df is None:
                continue
            save_path = Path(L2DATA_PATH) / "raw" / date / f"{key}.pq"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(save_path, compression="gzip")
            print(f'{date}日期{key}表搬运完毕！')
        return None

    raise ValueError(f"Unsupported mode: {mode}")



def manulpull_l2data(date: dt.date | dt.datetime | str) -> None:
    date = normalize_date(date)
    out_dir = Path(L2DATA_PATH) / "raw" / date
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, pattern in CIFTABLE_PATTERNS.items():
        zip_path = Path(L2DATA_PATH) / f"{date}_{pattern}.csv.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Local zip file not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            members = [info for info in z.infolist() if not info.is_dir() and info.file_size > 0]
            if not members:
                raise ValueError(f"No non-empty CSV file in zip: {zip_path}")

            with tempfile.TemporaryDirectory() as tmp:
                tmp_csv = Path(tmp) / members[0].filename
                tmp_csv.parent.mkdir(parents=True, exist_ok=True)

                with z.open(members[0]) as src, open(tmp_csv, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                pl.scan_csv(tmp_csv, truncate_ragged_lines=True).sink_parquet(
                    out_dir / f"{key}.pq",
                    compression="gzip",
                )


if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import bisect

    dates = np.load("/data/xujiayi/xjy/axis/dates.npy", allow_pickle=True)
    start_dt, end_dt = bisect.bisect_left(dates, pd.to_datetime('2025-08-11')), bisect.bisect_right(dates, pd.to_datetime('2025-12-31'))
    for date in dates[start_dt:end_dt]:
        date = date.astype('datetime64[D]').astype(str).replace('-', '')
        print(date)
        autoload_l2data(date)
        print()




