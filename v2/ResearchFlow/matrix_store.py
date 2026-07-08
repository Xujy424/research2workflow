"""Binary matrix storage for the local ``D:/data`` layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt


MemmapMode = Literal["r", "r+", "w+", "c"]
AxisSelector = object | Sequence[object] | None


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

    def load_axis(self) -> MatrixAxis:
        return MatrixAxis(
            dates=self._load_axis("date"),
            ticks=self._load_axis("tick"),
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

    def open_cube(
        self,
        category: str,
        field: str,
        *,
        dtype: npt.DTypeLike | None = None,
        fields: Sequence[str] | None = None,
    ) -> np.ndarray:
        axis = self.load_axis()
        final_dtype = np.dtype(dtype or self.default_dtype)
        directory = self.root / category / field
        paths = [directory / f"{name}.bin" for name in fields] if fields is not None else sorted(directory.glob("*.bin")) if directory.is_dir() else []
        if not paths:
            directory = self.root / category
            paths = [directory / f"{name}.bin" for name in fields] if fields is not None else sorted(directory.glob("*.bin")) if directory.is_dir() else []
        if not paths:
            raise FileNotFoundError(self.root / category / field)
        arrays = []
        for path in paths:
            self._validate_size(path, dtype=final_dtype, shape=axis.shape)
            arrays.append(np.asarray(np.memmap(path, dtype=final_dtype, mode="r", shape=axis.shape), dtype=final_dtype))
        return np.stack(arrays, axis=2)

    def read_slice(
        self,
        category: str,
        field: str,
        *,
        dates: AxisSelector = None,
        ticks: AxisSelector = None,
        dtype: npt.DTypeLike | None = None,
        paired: bool = False,
    ) -> np.ndarray:
        axis = self.load_axis()
        matrix = self.open_matrix(category, field, dtype=dtype, mode="r", shape=axis.shape)
        index = self._matrix_index(self._loc(axis.dates, dates), self._loc(axis.ticks, ticks), paired=paired)
        return np.asarray(matrix[index]).copy()

    def update_slice(
        self,
        category: str,
        field: str,
        values: npt.ArrayLike,
        *,
        dates: AxisSelector = None,
        ticks: AxisSelector = None,
        dtype: npt.DTypeLike | None = None,
        fill_value: float | int | None = np.nan,
        paired: bool = False,
    ) -> Path:
        axis = self.load_axis()
        final_dtype = np.dtype(dtype or self.default_dtype)
        path = self.ensure_matrix(category, field, dtype=final_dtype, fill_value=fill_value)
        matrix = np.memmap(path, dtype=final_dtype, mode="r+", shape=axis.shape)
        index = self._matrix_index(self._loc(axis.dates, dates), self._loc(axis.ticks, ticks), paired=paired)
        arr = np.asarray(values, dtype=final_dtype)
        expected = matrix[index].shape
        if arr.shape != expected:
            raise ValueError(f"expected slice shape {expected}, got {arr.shape}")
        matrix[index] = arr
        matrix.flush()
        return path

    def positions(self, axis_values: Sequence[object], labels: Sequence[object]) -> np.ndarray:
        lookup = {str(value): i for i, value in enumerate(axis_values)}
        missing = [str(label) for label in labels if str(label) not in lookup]
        if missing:
            raise KeyError(f"labels not found in axis: {missing[:10]}")
        return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)

    def _loc(self, axis_values: Sequence[object], labels: AxisSelector) -> slice | int | np.ndarray:
        if labels is None:
            return slice(None)
        if self._is_scalar_label(labels):
            return int(self.positions(axis_values, [labels])[0])
        return self.positions(axis_values, list(labels))

    def _matrix_index(
        self,
        rows: slice | int | np.ndarray,
        cols: slice | int | np.ndarray,
        *,
        paired: bool,
    ) -> tuple[object, object]:
        if paired:
            if not isinstance(rows, np.ndarray) or not isinstance(cols, np.ndarray):
                raise ValueError("paired=True requires sequence dates and sequence ticks")
            if len(rows) != len(cols):
                raise ValueError("dates and ticks must have the same length for paired access")
            return rows, cols
        if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
            return np.ix_(rows, cols)
        return rows, cols

    @staticmethod
    def _is_scalar_label(value: object) -> bool:
        if isinstance(value, (str, bytes, np.datetime64, np.integer, np.floating)):
            return True
        return np.ndim(value) == 0

    def _load_axis(self, stem: str) -> npt.NDArray[np.generic]:
        axis_dir = self.root / "axis"
        candidates = (axis_dir / f"{stem}.npy", axis_dir / f"{stem}s.npy")
        for path in candidates:
            if path.exists():
                return np.load(path, allow_pickle=True)
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
        
    def path(self, category: str, field: str) -> Path:
            return self.root / category / f"{field}.bin"
