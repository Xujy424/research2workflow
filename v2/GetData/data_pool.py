"""Read-only access to the axis-backed local research database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..UpdateData.config import ROOT
except ImportError:
    ROOT = Path("/data/shanghai/xujiayi/workflow/data")


MIDDLE_BY_FOLDER = {
    "m_essentials": 241,
    "con_forecast": 4,
    "con_forecast_eq": 4,
    "con_forecast_wgt": 4,
}
DTYPE_BY_ITEMSIZE = {
    1: np.dtype(np.bool_),
    2: np.dtype(np.float16),
    4: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


@dataclass(frozen=True)
class FieldSpec:
    """Physical dtype and shape of one binary field."""
    path: Path
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    middle: int


class AxisLoader:
    """Load the shared date axis and one asset code axis."""

    def __init__(self, root=ROOT, asset="stock"):
        self.root = Path(root)
        self.asset = str(asset).strip()
        self.reload()

    def reload(self):
        axis_root = self.root / "axis"
        dates_path = axis_root / "dates.npy"
        ticks_path = axis_root / f"{self.asset}_ticks.npy"
        if not dates_path.is_file() or not ticks_path.is_file():
            raise FileNotFoundError(
                f"axis files are missing: {dates_path}, {ticks_path}"
            )

        self.full_dates = np.load(dates_path, allow_pickle=False)
        self.full_ticks = np.load(ticks_path, allow_pickle=False)

        if np.any(
            np.isnat(self.full_dates[:-1])
            & ~np.isnat(self.full_dates[1:])
        ):
            raise ValueError("dates.npy has a gap before reserve rows")
        if np.any(
            (self.full_ticks[:-1] == "")
            & (self.full_ticks[1:] != "")
        ):
            raise ValueError(
                f"{ticks_path.name} has a gap before reserve columns"
            )

        self.date_count = int(np.count_nonzero(~np.isnat(self.full_dates)))
        self.tick_count = int(np.count_nonzero(self.full_ticks != ""))
        self.trade_dates = self.full_dates[:self.date_count].astype("datetime64[D]", copy=False)
        self.ticks = self.full_ticks[:self.tick_count]
        self._tick_positions = {
            str(tick): position
            for position, tick in enumerate(self.ticks)
        }
        return self
    
    def date_position(self, value) -> int:
        if isinstance(value, (int, np.integer)):
            position = int(value)
            if position < 0:
                position += self.date_count
            if position < 0 or position >= self.date_count:
                raise IndexError(f"date index out of range: {value}")
            return position

        target = np.datetime64(pd.Timestamp(value).date(), "D")
        position = int(np.searchsorted(self.trade_dates, target))
        if (
            position >= self.date_count or self.trade_dates[position] != target
        ):
            raise KeyError(f"date is not present in axis: {target}")
        return position

    def tick_positions(self, ticks) -> np.ndarray:
        if isinstance(ticks, str):
            ticks = [ticks]
        normalized = [str(tick).strip() for tick in ticks]
        if self.asset == "stock":
            normalized = [tick.zfill(6) for tick in normalized]
        missing = [
            tick for tick in normalized
            if tick not in self._tick_positions
        ]
        if missing:
            raise KeyError(f"instrument codes are not present in axis: {missing}")
        return np.asarray(
            [self._tick_positions[tick] for tick in normalized],
            dtype=np.int64,
        )


class DataPool:
    """Read fields for one maintained asset module."""

    def __init__(self, root=ROOT, asset="stock"):
        self.root = Path(root)
        self.asset = str(asset).strip()
        self.asset_root = self.root / self.asset
        self.axis = AxisLoader(self.root, self.asset)
        self._specs: dict[str, FieldSpec] = {}
        self._arrays: dict[str, np.memmap] = {}

    def _normalize(self, field) -> str:
        name = str(field).replace(chr(92), "/").strip("/")
        prefix = f"{self.asset}/"
        if name.startswith(prefix):
            name = name[len(prefix):]
        if name.endswith(".bin"):
            name = name[:-4]
        if not name:
            raise ValueError("field path must not be empty")
        return name
    
    def field_spec(self, field) -> FieldSpec:
        name = self._normalize(field)
        if name in self._specs:
            return self._specs[name]

        path = self.asset_root / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"field does not exist: {path}")

        middle = MIDDLE_BY_FOLDER.get(path.parent.name.lower(), 1)
        elements = (
            len(self.axis.full_dates)
            * middle
            * len(self.axis.full_ticks)
        )
        size = path.stat().st_size
        if elements <= 0 or size % elements:
            raise ValueError(
                f"{path} size={size} is incompatible with axes "
                f"({len(self.axis.full_dates)}, {middle}, "
                f"{len(self.axis.full_ticks)})"
            )

        itemsize = size // elements
        if itemsize not in DTYPE_BY_ITEMSIZE:
            raise ValueError(
                f"unsupported itemsize {itemsize} for {path}"
            )

        shape = (
            (
                len(self.axis.full_dates),
                len(self.axis.full_ticks),
            )
            if middle == 1
            else (
                len(self.axis.full_dates),
                middle,
                len(self.axis.full_ticks),
            )
        )
        spec = FieldSpec(
            path=path,
            name=name,
            dtype=DTYPE_BY_ITEMSIZE[itemsize],
            shape=shape,
            middle=middle,
        )
        self._specs[name] = spec
        return spec

    def load(self, field) -> np.memmap:
        """Return the full reserved read-only memmap for a field."""
        name = self._normalize(field)
        if name not in self._arrays:
            spec = self.field_spec(name)
            self._arrays[name] = np.memmap(
                spec.path,
                dtype=spec.dtype,
                mode="r",
                shape=spec.shape,
            )
        return self._arrays[name]

    def read(self, field, end_date, start_date=None, *, ticks=None, copy=True,):
        """Read one date or an inclusive date range on valid axes."""
        array = self.load(field)
        end = self.axis.date_position(end_date)
        if start_date is None:
            result = array[end]
        else:
            start = self.axis.date_position(start_date)
            if start > end:
                raise ValueError("start_date must not be after end_date")
            result = array[start:end + 1]

        if ticks is None:
            result = result[..., :self.axis.tick_count]
        else:
            positions = self.axis.tick_positions(ticks)
            result = np.take(result, positions, axis=-1)
        return np.array(result, copy=True) if copy else result

    def get_field(self, field) -> pd.DataFrame:
        """Return a complete valid 2-D field as a labeled DataFrame."""
        spec = self.field_spec(field)
        if spec.middle != 1:
            raise ValueError(
                "get_field supports 2-D fields only; use read() "
                "for minute and forecast matrices"
            )
        values = self.read(
            field,
            end_date=self.axis.date_count - 1,
            start_date=0,
            copy=False,
        )
        return pd.DataFrame(
            values,
            index=pd.Index(self.axis.trade_dates, name="date"),
            columns=pd.Index(self.axis.ticks, name="tick"),
        )

    def fields(self) -> list[str]:
        """List every axis-backed field, excluding Level-2 files."""
        if not self.asset_root.exists():
            return []
        return sorted(
            path.relative_to(self.asset_root)
            .with_suffix("")
            .as_posix()
            for path in self.asset_root.rglob("*.bin")
            if "l2" not in {
                part.lower()
                for part in path.relative_to(self.asset_root).parts
            }
        )

    def close(self):
        """Close all cached memmap file handles."""
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def refresh(self):
        """Close mappings and reload axes after UpdateData runs."""
        self.close()
        self.axis.reload()
        self._specs.clear()
        return self

    def __getitem__(self, key):
        if key in {"dates", "trade_dates"}:
            return self.axis.trade_dates
        if key in {
            "ticks", f"{self.asset}_tick", f"{self.asset}_ticks"
        }:
            return self.axis.ticks
        return self.load(key)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["AxisLoader", "DataPool", "FieldSpec"]


if __name__ == "__main__":
    with DataPool(ROOT) as data:
        end_date = data["trade_dates"][-1]
        ticks = data["ticks"][:3]

        close = data.read(
            "d_essentials/close",
            end_date,
            ticks=ticks,
        )
        forecast_eps = data.read(
            "zyyx/con_forecast/eps",
            end_date,
            ticks=ticks,
        )
        raw_net_profit = data.load(
            "fundamental/income_ttm/NetProfit"
        )

        print("close:", close.shape)
        print("forecast EPS:", forecast_eps.shape)
        print("raw net profit:", raw_net_profit.shape)