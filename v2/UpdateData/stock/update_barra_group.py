"""Daily cross-sectional long/short constituent masks for Barra factors."""

from pathlib import Path
import sys

import numpy as np

if __package__:
    from .update_barra import BARRA_NAMES
    from ..utils import date_index, ensure_memmap
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.stock.update_barra import BARRA_NAMES
    from v2.UpdateData.utils import date_index, ensure_memmap


def _group_mask(values, ticks):
    """Top/bottom floor(N / 10) finite stocks get +1/-1; others get 0.

    Empty stock-axis slots are excluded. Ties follow stock-axis order,
    so both tails have the same size and never overlap.
    """
    mask = np.zeros(len(ticks), dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(values) & (ticks != ""))
    count = len(valid) // 10
    if count:
        ordered = valid[np.argsort(values[valid], kind="stable")]
        mask[ordered[:count]] = -1
        mask[ordered[-count:]] = 1
    return mask


def update_barra_group(date, dates, ticks, conn, root):
    """Overwrite one daily row in stock/index/mask/{factor}_mask.bin.

    root is the stock directory; conn is accepted for the UpdateData
    calling convention. Inputs are float64 Barra files; outputs are
    float32 matrices compatible with axis resizing and pruning.
    """
    dt = date_index(date, dates)
    root = Path(root)
    shape = (len(dates), len(ticks))
    result = {}
    for name in BARRA_NAMES:
        source = root / "barra" / f"{name}.bin"
        expected_size = int(np.prod(shape)) * np.dtype(np.float64).itemsize
        if source.stat().st_size != expected_size:
            raise ValueError(f"{source} size does not match axes")
        factors = np.memmap(source, dtype=np.float64, mode="r", shape=shape)
        mask = _group_mask(factors[dt], ticks)
        output = ensure_memmap(
            root / "index" / "mask" / f"{name}_mask.bin",
            shape, np.float32, np.nan,
        )
        output[dt] = mask
        output.flush()
        result[name] = mask
    return result


def update_history(root, start_date=None, end_date=None):
    """Build masks from existing Barra data without database queries.

    root is the data directory containing axis/ and stock/. Omitted
    bounds include all populated dates; reserved NaT rows are skipped.
    """
    root = Path(root)
    dates = np.load(root / "axis" / "dates.npy", allow_pickle=False)
    ticks = np.load(root / "axis" / "stock_ticks.npy", allow_pickle=False)
    start = np.datetime64(start_date, "D") if start_date is not None else None
    end = np.datetime64(end_date, "D") if end_date is not None else None
    if (start is not None and np.isnat(start)) or (end is not None and np.isnat(end)):
        raise ValueError("date bounds must not be NaT")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be later than end_date")
    for date in dates[~np.isnat(dates)]:
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        update_barra_group(date, dates, ticks, None, root / "stock")
        print(f"[OK] Barra groups: {date}", flush=True)


if __name__ == "__main__":
    from v2.UpdateData.config import ROOT

    update_history(root=ROOT)
