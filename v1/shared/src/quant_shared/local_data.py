"""Local binary market data store shared by researchflow and workflow.

The local data store is intentionally independent from both flow packages.
``researchflow`` should read from it for research panels; ``workflow`` should
write newly pulled SQL results into it, compute online factors from it, and
then continue downstream from the same binary source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .contracts import PanelData


DataFrequency = Literal["daily", "minute"]
MemmapMode = Literal["r", "r+", "w+", "c"]


class LocalDataStoreError(RuntimeError):
    """Base exception for local data store failures."""


class AxisNotFoundError(LocalDataStoreError, FileNotFoundError):
    """Raised when an axis ``.npy`` file is missing."""


class MatrixNotFoundError(LocalDataStoreError, FileNotFoundError):
    """Raised when a binary matrix file is missing."""


class L2TableNotFoundError(LocalDataStoreError, FileNotFoundError):
    """Raised when a requested L2 table cannot be found."""


@dataclass(frozen=True)
class LocalDataConfig:
    root: Path = Path("D:/data")
    axis_dir: str = "axis"
    date_axis: str = "date"
    tick_axis: str = "tick"
    daily_dir: str = 'd_field'
    minute_dir: str = "m_field"
    l2_dir: str = "L2"
    default_dtype: str = "float64"
    minute_bars_per_day: int = 241

    def axis_path(self, name: str) -> Path:
        return self.root / self.axis_dir / f"{name}.npy"

    def matrix_path(self, category: str, field: str) -> Path:
        return self.root / category / f"{field}.bin"

    def l2_day_dir(self, trading_date: object) -> Path:
        return self.root / self.l2_dir / format_trading_date(trading_date)


@dataclass(frozen=True)
class MarketAxis:
    """Shared matrix indexes for local binary files."""

    dates: npt.NDArray[np.generic]
    ticks: npt.NDArray[np.generic]
    minute_bars_per_day: int = 241

    @property
    def daily_shape(self) -> tuple[int, int]:
        return (len(self.dates), len(self.ticks))

    @property
    def minute_shape(self) -> tuple[int, int, int]:
        return (len(self.dates), len(self.ticks), self.minute_bars_per_day)


@dataclass(frozen=True)
class L2TablePaths:
    """Resolved L2 table paths for one trading day."""

    trading_date: str
    tables: Mapping[str, Path]

    def require(self, exchange: str, table: str) -> Path:
        key = f"{normalize_exchange(exchange)}.{table.lower()}"
        try:
            return self.tables[key]
        except KeyError as exc:
            raise L2TableNotFoundError(f"L2 table is not resolved: {key}") from exc


class SqlReader(Protocol):
    """Minimal database client protocol used by workflow data updates."""

    def read_sql(
        self,
        sql: str,
        params: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame for ``sql`` and optional bind parameters."""


@dataclass(frozen=True)
class LocalPanelSpec:
    """Matrix fields required to assemble a flow-ready ``PanelData`` object."""

    factor_category: str = "research_factors"
    factor_fields: tuple[str, ...] = ()
    label_category: str = "label"
    label_field: str = "forward_return"
    exposure_category: str | None = "barra"
    exposure_fields: tuple[str, ...] = ()
    market_cap_category: str | None = "d_field"
    market_cap_field: str | None = "market_cap"
    tradable_category: str | None = "mask"
    tradable_field: str | None = "tradable"


@dataclass(frozen=True)
class SqlDailyUpdateSpec:
    """One SQL-to-local-daily-matrix update task for workflow."""

    name: str
    category: str
    field: str
    sql: str
    value_column: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    dtype: npt.DTypeLike | None = None
    create_if_missing: bool = True


DailyMatrixInput = Mapping[str, np.memmap]
OnlineFactorCompute = Callable[[DailyMatrixInput, MarketAxis, int], npt.ArrayLike]


@dataclass(frozen=True)
class DailyMatrixRef:
    """Reference to one daily local matrix."""

    category: str
    field: str
    dtype: npt.DTypeLike | None = None


@dataclass(frozen=True)
class OnlineFactorSpec:
    """Compute one online factor from already-updated local matrices."""

    name: str
    output_field: str
    inputs: Mapping[str, DailyMatrixRef]
    compute: OnlineFactorCompute
    output_category: str = "online_factors"
    dtype: npt.DTypeLike | None = None
    create_if_missing: bool = True


@dataclass(frozen=True)
class WorkflowDataUpdateResult:
    """Audit result for one workflow local-data update run."""

    as_of: str
    sql_updates: tuple[str, ...]
    online_factors: tuple[str, ...]
    l2_tables: L2TablePaths | None = None

class LocalMarketDataStore:
    """Memmap-based reader/writer for the shared local market data store."""

    def __init__(self, config: LocalDataConfig | str | Path | None = None) -> None:
        if config is None:
            self.config = LocalDataConfig()
        elif isinstance(config, LocalDataConfig):
            self.config = config
        else:
            self.config = LocalDataConfig(root=Path(config))

    @property
    def root(self) -> Path:
        return self.config.root

    def load_axis(self, *, mmap_mode: MemmapMode | None = "r") -> MarketAxis:
        return MarketAxis(
            dates=self.load_axis_array(self.config.date_axis, mmap_mode=mmap_mode),
            ticks=self.load_axis_array(self.config.tick_axis, mmap_mode=mmap_mode),
            minute_bars_per_day=self.config.minute_bars_per_day,
        )

    def load_axis_array(
        self,
        name: str,
        *,
        mmap_mode: MemmapMode | None = "r",
    ) -> npt.NDArray[np.generic]:
        path = self.config.axis_path(name)
        if not path.exists():
            raise AxisNotFoundError(f"axis file does not exist: {path}")
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

    def save_axis(
        self,
        *,
        dates: npt.ArrayLike | None = None,
        ticks: npt.ArrayLike | None = None,
    ) -> None:
        """Write axis arrays. Existing arrays are replaced atomically by NumPy."""

        axis_dir = self.root / self.config.axis_dir
        axis_dir.mkdir(parents=True, exist_ok=True)
        if dates is not None:
            np.save(self.config.axis_path(self.config.date_axis), np.asarray(dates))
        if ticks is not None:
            np.save(self.config.axis_path(self.config.tick_axis), np.asarray(ticks))

    def open_daily(
        self,
        category: str,
        field: str,
        *,
        dtype: npt.DTypeLike | None = None,
        mode: MemmapMode = "r",
        shape: tuple[int, int] | None = None,
        validate_size: bool = True,
    ) -> np.memmap:
        """Open a daily ``T x N`` raw binary field."""

        final_shape = shape or self.load_axis().daily_shape
        return self._open_matrix(
            self.config.matrix_path(category, field),
            dtype=dtype or self.config.default_dtype,
            mode=mode,
            shape=final_shape,
            validate_size=validate_size,
        )

    def open_minute(
        self,
        field: str,
        *,
        dtype: npt.DTypeLike | None = None,
        mode: MemmapMode = "r",
        shape: tuple[int, int, int] | None = None,
        validate_size: bool = True,
    ) -> np.memmap:
        """Open a minute ``T x N x 241`` raw binary field."""

        final_shape = shape or self.load_axis().minute_shape
        return self._open_matrix(
            self.config.matrix_path(self.config.minute_dir, field),
            dtype=dtype or self.config.default_dtype,
            mode=mode,
            shape=final_shape,
            validate_size=validate_size,
        )

    def ensure_matrix(
        self,
        category: str,
        field: str,
        *,
        frequency: DataFrequency = "daily",
        dtype: npt.DTypeLike | None = None,
        fill_value: float | int | None = np.nan,
    ) -> Path:
        """Create a binary matrix aligned to the current axis if it is missing."""

        axis = self.load_axis()
        target_category = self.config.minute_dir if frequency == "minute" else category
        shape = axis.minute_shape if frequency == "minute" else axis.daily_shape
        path = self.config.matrix_path(target_category, field)
        if path.exists():
            self._validate_file_size(
                path,
                dtype=dtype or self.config.default_dtype,
                shape=shape,
            )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.memmap(
            path,
            dtype=dtype or self.config.default_dtype,
            mode="w+",
            shape=shape,
        )
        if fill_value is not None:
            matrix[...] = fill_value
        matrix.flush()
        return path

    def read_panel(
        self,
        category: str,
        fields: Sequence[str],
        *,
        dtype: npt.DTypeLike | None = None,
        dates: Sequence[object] | None = None,
        ticks: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read daily fields into a ``(date, asset)`` indexed DataFrame."""

        axis = self.load_axis()
        date_positions = (
            self.date_positions(dates, axis=axis) if dates else np.arange(len(axis.dates))
        )
        tick_positions = (
            self.tick_positions(ticks, axis=axis) if ticks else np.arange(len(axis.ticks))
        )
        index = pd.MultiIndex.from_product(
            [axis.dates[date_positions], axis.ticks[tick_positions]],
            names=["date", "asset"],
        )
        frame = pd.DataFrame(index=index)
        rows = np.ix_(date_positions, tick_positions)
        for field in fields:
            matrix = self.open_daily(category, field, dtype=dtype)
            frame[field] = np.asarray(matrix[rows]).reshape(-1)
        return frame

    def write_daily_frame(
        self,
        category: str,
        field: str,
        frame: pd.Series | pd.DataFrame,
        *,
        value_column: str | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Update a daily matrix from SQL-style long data.

        ``frame`` must be indexed by ``(date, asset)`` or contain ``date`` and
        ``asset`` columns. A DataFrame with more than one value column must pass
        ``value_column`` explicitly.
        """

        values = normalize_daily_values(frame, value_column=value_column)
        axis = self.load_axis()
        matrix = self.open_daily(category, field, dtype=dtype, mode="r+")
        date_lookup = axis_lookup(axis.dates, date_like=True)
        tick_lookup = axis_lookup(axis.ticks)
        for (date_value, tick), value in values.items():
            matrix[date_lookup[format_trading_date(date_value)], tick_lookup[str(tick)]] = value
        matrix.flush()

    def write_daily_slice(
        self,
        category: str,
        field: str,
        trading_date: object,
        values: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        axis = self.load_axis()
        row = self.date_position(trading_date, axis=axis)
        array = np.asarray(values, dtype=dtype or self.config.default_dtype)
        if array.shape != (len(axis.ticks),):
            raise ValueError(f"expected daily slice shape {(len(axis.ticks),)}, got {array.shape}")
        matrix = self.open_daily(category, field, dtype=dtype, mode="r+")
        matrix[row, :] = array
        matrix.flush()

    def write_minute_slice(
        self,
        field: str,
        trading_date: object,
        values: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        axis = self.load_axis()
        row = self.date_position(trading_date, axis=axis)
        expected = (len(axis.ticks), self.config.minute_bars_per_day)
        array = np.asarray(values, dtype=dtype or self.config.default_dtype)
        if array.shape != expected:
            raise ValueError(f"expected minute slice shape {expected}, got {array.shape}")
        matrix = self.open_minute(field, dtype=dtype, mode="r+")
        matrix[row, :, :] = array
        matrix.flush()

    def resolve_l2_tables(
        self,
        trading_date: object,
        *,
        exchanges: Iterable[str] = ("SSE", "SZSE"),
        tables: Iterable[str] = ("orders", "trades"),
        suffixes: Iterable[str] = (".parquet", ".csv", ".feather", ".ftr", ".pkl", ".pickle"),
    ) -> L2TablePaths:
        """Resolve L2 daily exchange tables under ``L2/YYYYMMDD``.

        Supported path conventions:

        - ``L2/YYYYMMDD/SSE/orders.parquet``
        - ``L2/YYYYMMDD/SSE_orders.parquet``
        - ``L2/YYYYMMDD/orders_SSE.parquet``
        """

        day = format_trading_date(trading_date)
        day_dir = self.config.l2_day_dir(day)
        resolved: dict[str, Path] = {}
        missing: list[str] = []
        for exchange in exchanges:
            normalized_exchange = normalize_exchange(exchange)
            for table in tables:
                table_name = table.lower()
                path = self._find_l2_table(day_dir, normalized_exchange, table_name, suffixes)
                key = f"{normalized_exchange}.{table_name}"
                if path is None:
                    missing.append(key)
                else:
                    resolved[key] = path
        if missing:
            raise L2TableNotFoundError(f"missing L2 tables under {day_dir}: {missing}")
        return L2TablePaths(trading_date=day, tables=resolved)

    def read_l2_table(
        self,
        trading_date: object,
        exchange: str,
        table: str,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Read one resolved L2 table with pandas based on file suffix."""

        path = self.resolve_l2_tables(
            trading_date,
            exchanges=(exchange,),
            tables=(table,),
        ).require(exchange, table)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        if suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)
        if suffix in {".feather", ".ftr"}:
            return pd.read_feather(path, **kwargs)
        if suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(path, **kwargs)
        raise ValueError(f"unsupported L2 table format: {path}")

    def date_position(self, trading_date: object, *, axis: MarketAxis | None = None) -> int:
        return int(self.date_positions((trading_date,), axis=axis)[0])

    def date_positions(
        self,
        dates: Sequence[object],
        *,
        axis: MarketAxis | None = None,
    ) -> npt.NDArray[np.int64]:
        lookup = axis_lookup((axis or self.load_axis()).dates, date_like=True)
        missing = [
            format_trading_date(value)
            for value in dates
            if format_trading_date(value) not in lookup
        ]
        if missing:
            raise KeyError(f"dates are not present in axis/{self.config.date_axis}.npy: {missing}")
        return np.asarray([lookup[format_trading_date(value)] for value in dates], dtype=np.int64)

    def tick_positions(
        self,
        ticks: Sequence[str],
        *,
        axis: MarketAxis | None = None,
    ) -> npt.NDArray[np.int64]:
        lookup = axis_lookup((axis or self.load_axis()).ticks)
        missing = [str(value) for value in ticks if str(value) not in lookup]
        if missing:
            raise KeyError(f"ticks are not present in axis/{self.config.tick_axis}.npy: {missing}")
        return np.asarray([lookup[str(value)] for value in ticks], dtype=np.int64)

    def _open_matrix(
        self,
        path: Path,
        *,
        dtype: npt.DTypeLike,
        mode: MemmapMode,
        shape: tuple[int, ...],
        validate_size: bool,
    ) -> np.memmap:
        if mode == "r" and not path.exists():
            raise MatrixNotFoundError(f"matrix file does not exist: {path}")
        if mode in {"r+", "w+"}:
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and mode != "w+" and validate_size:
            self._validate_file_size(path, dtype=dtype, shape=shape)
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    @staticmethod
    def _validate_file_size(
        path: Path,
        *,
        dtype: npt.DTypeLike,
        shape: tuple[int, ...],
    ) -> None:
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"unexpected matrix size for {path}: expected {expected} bytes "
                f"for shape={shape}, dtype={np.dtype(dtype)}, got {actual} bytes"
            )

    @staticmethod
    def _find_l2_table(
        day_dir: Path,
        exchange: str,
        table: str,
        suffixes: Iterable[str],
    ) -> Path | None:
        candidates: list[Path] = []
        for suffix in suffixes:
            candidates.extend(
                [
                    day_dir / exchange / f"{table}{suffix}",
                    day_dir / f"{exchange}_{table}{suffix}",
                    day_dir / f"{table}_{exchange}{suffix}",
                    day_dir / exchange.lower() / f"{table}{suffix}",
                ]
            )
        for path in candidates:
            if path.exists():
                return path
        return None


def normalize_daily_values(
    frame: pd.Series | pd.DataFrame,
    *,
    value_column: str | None = None,
) -> pd.Series:
    if isinstance(frame, pd.Series):
        values = frame.copy()
    else:
        data = frame.copy()
        if not isinstance(data.index, pd.MultiIndex):
            required = {"date", "asset"}
            if not required.issubset(data.columns):
                raise ValueError("daily data must use a MultiIndex or contain date/asset columns")
            data = data.set_index(["date", "asset"])
        if value_column is None:
            columns = list(data.columns)
            if len(columns) != 1:
                raise ValueError("value_column is required when daily data has multiple columns")
            value_column = columns[0]
        values = data[value_column]
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError("daily values must be indexed by (date, asset)")
    if values.index.has_duplicates:
        raise ValueError("daily values contain duplicate (date, asset) rows")
    return values


def axis_lookup(axis: Sequence[object], *, date_like: bool = False) -> dict[str, int]:
    keys = [format_trading_date(value) if date_like else str(value) for value in axis]
    if len(set(keys)) != len(keys):
        raise ValueError("axis contains duplicate labels after normalization")
    return {key: position for position, key in enumerate(keys)}


def normalize_exchange(exchange: str) -> str:
    text = exchange.strip().upper()
    aliases = {
        "SH": "SSE",
        "SHSE": "SSE",
        "XSHG": "SSE",
        "SZ": "SZSE",
        "XSHE": "SZSE",
    }
    return aliases.get(text, text)


def format_trading_date(value: object) -> str:
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[D]")).replace("-", "")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if len(text) >= 10 and ("-" in text[:10] or "/" in text[:10]):
        return text[:10].replace("-", "").replace("/", "")
    return text.replace("-", "").replace("/", "")[:8]

class LocalResearchPanelLoader:
    """Build researchflow/production ``PanelData`` from the local data store."""

    def __init__(self, store: LocalMarketDataStore) -> None:
        self.store = store

    def load(
        self,
        spec: LocalPanelSpec,
        *,
        dates: Sequence[object] | None = None,
        ticks: Sequence[str] | None = None,
    ) -> PanelData:
        if not spec.factor_fields:
            raise ValueError("factor_fields must contain at least one factor")
        factors = self.store.read_panel(
            spec.factor_category,
            spec.factor_fields,
            dates=dates,
            ticks=ticks,
        )
        labels = self.store.read_panel(
            spec.label_category,
            (spec.label_field,),
            dates=dates,
            ticks=ticks,
        )[spec.label_field]
        exposures = None
        if spec.exposure_category is not None and spec.exposure_fields:
            exposures = self.store.read_panel(
                spec.exposure_category,
                spec.exposure_fields,
                dates=dates,
                ticks=ticks,
            )
        market_caps = None
        if spec.market_cap_category is not None and spec.market_cap_field is not None:
            market_caps = self.store.read_panel(
                spec.market_cap_category,
                (spec.market_cap_field,),
                dates=dates,
                ticks=ticks,
            )[spec.market_cap_field]
        tradable = None
        if spec.tradable_category is not None and spec.tradable_field is not None:
            tradable = self.store.read_panel(
                spec.tradable_category,
                (spec.tradable_field,),
                dates=dates,
                ticks=ticks,
            )[spec.tradable_field].astype(bool)
        return PanelData(
            factors=factors,
            forward_returns=labels,
            exposures=exposures,
            market_caps=market_caps,
            tradable=tradable,
            metadata={
                "source": "local_binary_store",
                "root": str(self.store.root),
                "factor_category": spec.factor_category,
            },
        ).validate()


class LocalWorkflowDataUpdater:
    """Workflow bridge: SQL updates, local matrix writes, then online factors."""

    def __init__(self, store: LocalMarketDataStore, sql_reader: SqlReader) -> None:
        self.store = store
        self.sql_reader = sql_reader

    def update_daily_sql_fields(
        self,
        specs: Sequence[SqlDailyUpdateSpec],
        *,
        as_of: object,
    ) -> tuple[str, ...]:
        updated: list[str] = []
        for spec in specs:
            if spec.create_if_missing:
                self.store.ensure_matrix(spec.category, spec.field, dtype=spec.dtype)
            params = {"as_of": format_trading_date(as_of), **dict(spec.params)}
            frame = self.sql_reader.read_sql(spec.sql, params=params)
            self.store.write_daily_frame(
                spec.category,
                spec.field,
                frame,
                value_column=spec.value_column,
                dtype=spec.dtype,
            )
            updated.append(spec.name)
        return tuple(updated)

    def update_online_factors(
        self,
        specs: Sequence[OnlineFactorSpec],
        *,
        as_of: object,
    ) -> tuple[str, ...]:
        axis = self.store.load_axis()
        row = self.store.date_position(as_of, axis=axis)
        updated: list[str] = []
        for spec in specs:
            if spec.create_if_missing:
                self.store.ensure_matrix(
                    spec.output_category,
                    spec.output_field,
                    dtype=spec.dtype,
                )
            inputs = {
                name: self.store.open_daily(
                    ref.category,
                    ref.field,
                    dtype=ref.dtype,
                )
                for name, ref in spec.inputs.items()
            }
            values = np.asarray(
                spec.compute(inputs, axis, row),
                dtype=spec.dtype or self.store.config.default_dtype,
            )
            if values.shape != (len(axis.ticks),):
                raise ValueError(
                    f"online factor {spec.name!r} returned shape {values.shape}; "
                    f"expected {(len(axis.ticks),)}"
                )
            output = self.store.open_daily(
                spec.output_category,
                spec.output_field,
                dtype=spec.dtype,
                mode="r+",
            )
            output[row, :] = values
            output.flush()
            updated.append(spec.name)
        return tuple(updated)

    def run_daily_update(
        self,
        *,
        as_of: object,
        sql_updates: Sequence[SqlDailyUpdateSpec] = (),
        online_factors: Sequence[OnlineFactorSpec] = (),
        require_l2: bool = False,
        l2_exchanges: Iterable[str] = ("SSE", "SZSE"),
        l2_tables: Iterable[str] = ("orders", "trades"),
    ) -> WorkflowDataUpdateResult:
        sql_names = self.update_daily_sql_fields(sql_updates, as_of=as_of)
        factor_names = self.update_online_factors(online_factors, as_of=as_of)
        l2_paths = None
        if require_l2:
            l2_paths = self.store.resolve_l2_tables(
                as_of,
                exchanges=l2_exchanges,
                tables=l2_tables,
            )
        return WorkflowDataUpdateResult(
            as_of=format_trading_date(as_of),
            sql_updates=sql_names,
            online_factors=factor_names,
            l2_tables=l2_paths,
        )