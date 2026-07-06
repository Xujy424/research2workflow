"""Binary matrix storage for the local ``D:/data`` layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt


MemmapMode = Literal["r", "r+", "w+", "c"]


@dataclass(frozen=True)
class MatrixAxis:
    dates: npt.NDArray[np.generic]
    ticks: npt.NDArray[np.generic]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.dates), len(self.ticks))


class MatrixStore:
    """Memmap-based reader/writer for daily ``T x N`` matrices."""

    def __init__(self, root: str | Path = "D:/data", *, dtype: npt.DTypeLike = "float64") -> None:
        self.root = Path(root)
        self.default_dtype = np.dtype(dtype)

    def load_axis(self, *, mmap_mode: str | None = "r") -> MatrixAxis:
        return MatrixAxis(
            dates=self._load_axis("date", mmap_mode=mmap_mode),
            ticks=self._load_axis("tick", mmap_mode=mmap_mode),
        )

    def open_matrix(
        self,
        category: str,
        field: str,
        *,
        dtype: npt.DTypeLike | None = None,
        mode: MemmapMode = "r",
        shape: tuple[int, int] | None = None,
        validate_size: bool = True,
    ) -> np.memmap:
        final_shape = shape or self.load_axis().shape
        final_dtype = np.dtype(dtype or self.default_dtype)
        path = self.path(category, field)
        if mode == "r" and not path.exists():
            raise FileNotFoundError(path)
        if mode in {"r+", "w+"}:
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and mode != "w+" and validate_size:
            self._validate_size(path, dtype=final_dtype, shape=final_shape)
        return np.memmap(path, dtype=final_dtype, mode=mode, shape=final_shape)

    def ensure_matrix(
        self,
        category: str,
        field: str,
        *,
        dtype: npt.DTypeLike | None = None,
        fill_value: float | int | None = np.nan,
    ) -> Path:
        axis = self.load_axis()
        path = self.path(category, field)
        final_dtype = np.dtype(dtype or self.default_dtype)
        if path.exists():
            self._validate_size(path, dtype=final_dtype, shape=axis.shape)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.memmap(path, dtype=final_dtype, mode="w+", shape=axis.shape)
        if fill_value is not None:
            arr[...] = fill_value
        arr.flush()
        return path

    def write_matrix(
        self,
        category: str,
        field: str,
        values: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> Path:
        axis = self.load_axis()
        arr = np.asarray(values, dtype=dtype or self.default_dtype)
        if arr.shape != axis.shape:
            raise ValueError(f"expected matrix shape {axis.shape}, got {arr.shape}")
        path = self.path(category, field)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.memmap(path, dtype=arr.dtype, mode="w+", shape=axis.shape)
        out[...] = arr
        out.flush()
        return path

    def path(self, category: str, field: str) -> Path:
        return self.root / category / f"{field}.bin"

    def positions(self, axis_values: Sequence[object], labels: Sequence[object]) -> np.ndarray:
        lookup = {str(value): i for i, value in enumerate(axis_values)}
        missing = [str(label) for label in labels if str(label) not in lookup]
        if missing:
            raise KeyError(f"labels not found in axis: {missing[:10]}")
        return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)

    def _load_axis(self, stem: str, *, mmap_mode: str | None) -> npt.NDArray[np.generic]:
        axis_dir = self.root / "axis"
        candidates = (axis_dir / f"{stem}.npy", axis_dir / f"{stem}s.npy")
        for path in candidates:
            if path.exists():
                return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        raise FileNotFoundError(f"axis file not found, tried: {', '.join(map(str, candidates))}")

    @staticmethod
    def _validate_size(path: Path, *, dtype: npt.DTypeLike, shape: tuple[int, ...]) -> None:
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"unexpected matrix size for {path}: expected {expected} bytes "
                f"for shape={shape}, dtype={np.dtype(dtype)}, got {actual} bytes"
            )

