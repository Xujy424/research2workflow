"""读取并绘制本地保存的单条监控指标序列。"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.UpdateData.config import ROOT
from v2.Monitor.config import INDUSTRY_MAP


def plot_series(name, category="macro", *, root=ROOT, start_date=None, end_date=None,
                ylabel="Value", save_path=None):
    """按共享日期轴读取指定类别的指标二进制序列并绘图。"""
    root = Path(root)
    dates = np.load(root / "axis" / "dates.npy", allow_pickle=False)
    valid_dates = dates[~np.isnat(dates)]
    path = root / "median_variables" / "monitor" / category / f"{name}.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    row_bytes = len(dates) * np.dtype(np.float32).itemsize
    if path.stat().st_size % row_bytes:
        raise ValueError(f"二进制文件大小与日期轴不匹配: {path}")
    columns = path.stat().st_size // row_bytes
    if columns < 1:
        raise ValueError(f"二进制文件没有数据列: {path}")

    values = np.memmap(path, dtype=np.float32, mode="r", shape=(len(dates), columns))
    frame = pd.DataFrame(
        np.asarray(values[:len(valid_dates)]).copy(),
        index=pd.to_datetime(valid_dates),
    )
    del values
    if columns == 1:
        series = frame.iloc[:, 0].rename(name)
    else:
        micro_columns = {'impact_term_structure': ('1m', '5m', '15m', '30m'),
                         'jump_impact_quadrant': ('jump_z', 'retention_5m')}
        frame.columns = (micro_columns[name] if category == 'micro' and name in micro_columns
                         else tuple(INDUSTRY_MAP) if category == "style"
                         and name == "industry_relative_return"
                         and columns == len(INDUSTRY_MAP)
                         else [f"{name}_{i + 1}" for i in range(columns)])
        series = frame

    if start_date is not None:
        series = series.loc[pd.Timestamp(start_date):]
    if end_date is not None:
        series = series.loc[:pd.Timestamp(end_date)]
    if series.empty:
        raise ValueError("所选日期范围没有有效日期")

    multi = isinstance(series, pd.DataFrame)
    ax = series.plot(figsize=(16, 7) if multi else (14, 5),
                     linewidth=.8 if multi else 1, alpha=.75 if multi else 1)
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if multi:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2,
                  fontsize=7, frameon=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=160)
    plt.show()
    return series


if __name__ == "__main__":
    # 在 IDE 中修改类别（macro/risk 等）、指标函数名及日期范围。
    CATEGORY = "moneyflow"
    SERIES_NAME = "etf_net_subscription_ratio"
    START_DATE = None
    END_DATE = None
    plot_series(SERIES_NAME, CATEGORY, start_date=START_DATE, end_date=END_DATE)

    

