"""Versioned factor metadata and lifecycle governance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_shared.local_data import LocalMarketDataStore, LocalPanelSpec


class FactorStatus(str, Enum):
    RESEARCH = "research"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PRODUCTION = "production"
    DECAYING = "decaying"
    RETIRED = "retired"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class FactorMetadata:
    """Governance metadata for one versioned factor matrix."""

    factor_id: str
    name: str
    version: str
    family: str
    owner: str
    economic_rationale: str
    horizon_days: int
    data_delay_days: int
    status: FactorStatus = FactorStatus.RESEARCH
    parameters: Mapping[str, object] = field(default_factory=dict)
    risk_tags: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    storage_category: str = "research_factors"
    storage_field: str | None = None
    dtype: str = "float64"
    shape: tuple[int, ...] | None = None
    file_path: str | None = None
    file_size_bytes: int | None = None
    last_modified_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    validation: Mapping[str, object] = field(default_factory=dict)
    update_enabled: bool = False
    update_frequency: str = "daily"
    update_dependencies: tuple[str, ...] = ()
    notes: str = ""

    def validate(self) -> "FactorMetadata":
        if not self.factor_id or not self.version:
            raise ValueError("factor_id and version are required")
        if not self.name:
            raise ValueError("name is required")
        if self.horizon_days <= 0 or self.data_delay_days < 0:
            raise ValueError("invalid horizon or data delay")
        if self.storage_field is not None and not self.storage_field:
            raise ValueError("storage_field must not be empty")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.factor_id, self.version)

    @property
    def field_name(self) -> str:
        return self.storage_field or self.factor_id

    def with_status(
        self,
        status: FactorStatus,
        *,
        validation: Mapping[str, object] | None = None,
        notes: str | None = None,
        update_enabled: bool | None = None,
    ) -> "FactorMetadata":
        return replace(
            self,
            status=status,
            validation=dict(validation or self.validation),
            notes=self.notes if notes is None else notes,
            update_enabled=self.update_enabled if update_enabled is None else update_enabled,
            updated_at=utc_now(),
        )


class FactorRegistry:
    """Persistent registry for local binary factor matrices.

    The registry maps logical factor versions to physical local-data files such
    as ``D:/data/research_factors/momentum.bin``. Research can scan and admit
    factors from this registry; workflow should consume only statuses that have
    explicitly passed governance, typically ``production`` or ``shadow``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: dict[tuple[str, str], FactorMetadata] = {}
        if self.path is not None and self.path.exists():
            self.load()

    @classmethod
    def for_store(
        cls,
        store: LocalMarketDataStore,
        *,
        filename: str = "factors.json",
    ) -> "FactorRegistry":
        return cls(store.root / "registry" / filename)

    def register(self, metadata: FactorMetadata) -> None:
        item = metadata.validate()
        if item.key in self._records:
            raise ValueError(f"factor version already exists: {item.key}")
        self._records[item.key] = item

    def upsert(self, metadata: FactorMetadata) -> None:
        item = metadata.validate()
        self._records[item.key] = item

    def get(self, factor_id: str, version: str = "v1") -> FactorMetadata:
        return self._records[(factor_id, version)]

    def latest(self, factor_id: str) -> FactorMetadata:
        candidates = [item for item in self._records.values() if item.factor_id == factor_id]
        if not candidates:
            raise KeyError(factor_id)
        return sorted(candidates, key=lambda item: item.version)[-1]

    def all(self) -> list[FactorMetadata]:
        return sorted(self._records.values(), key=lambda item: item.key)

    def by_status(self, *statuses: FactorStatus) -> list[FactorMetadata]:
        wanted = set(statuses)
        return [item for item in self.all() if item.status in wanted]

    def production_factors(self) -> list[FactorMetadata]:
        return self.by_status(FactorStatus.PRODUCTION)

    def workflow_update_factors(
        self,
        *,
        statuses: tuple[FactorStatus, ...] = (FactorStatus.PRODUCTION,),
        categories: tuple[str, ...] | None = None,
    ) -> list[FactorMetadata]:
        """Return factors that workflow should update from the registry JSON."""

        items = [item for item in self.by_status(*statuses) if item.update_enabled]
        if categories is not None:
            allowed = set(categories)
            items = [item for item in items if item.storage_category in allowed]
        return items
    def update_status(
        self,
        factor_id: str,
        version: str,
        status: FactorStatus,
        *,
        validation: Mapping[str, object] | None = None,
        notes: str | None = None,
        update_enabled: bool | None = None,
    ) -> FactorMetadata:
        item = self.get(factor_id, version).with_status(
            status,
            validation=validation,
            notes=notes,
            update_enabled=update_enabled,
        )
        self._records[item.key] = item
        return item

    def scan_local_factors(
        self,
        store: LocalMarketDataStore,
        *,
        categories: Iterable[str] = ("research_factors", "online_factors"),
        default_owner: str = "unknown",
        default_family: str = "unknown",
        default_horizon_days: int = 1,
        default_data_delay_days: int = 0,
        version: str = "v1",
    ) -> list[FactorMetadata]:
        """Register missing ``*.bin`` factors discovered in local data folders."""

        axis = store.load_axis()
        discovered: list[FactorMetadata] = []
        for category in categories:
            folder = store.root / category
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.bin")):
                field_name = path.stem
                factor_id = field_name
                key = (factor_id, version)
                stat = path.stat()
                existing = self._records.get(key)
                base = existing or FactorMetadata(
                    factor_id=factor_id,
                    name=field_name,
                    version=version,
                    family=default_family,
                    owner=default_owner,
                    economic_rationale="pending review",
                    horizon_days=default_horizon_days,
                    data_delay_days=default_data_delay_days,
                    status=FactorStatus.RESEARCH,
                    created_at=utc_now(),
                )
                item = replace(
                    base,
                    storage_category=category,
                    storage_field=field_name,
                    dtype=store.config.default_dtype,
                    shape=axis.daily_shape,
                    file_path=str(path),
                    file_size_bytes=stat.st_size,
                    last_modified_at=pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
                    updated_at=utc_now(),
                ).validate()
                self._records[key] = item
                discovered.append(item)
        return discovered

    def to_panel_spec(
        self,
        *,
        statuses: tuple[FactorStatus, ...] = (FactorStatus.PRODUCTION,),
        label_category: str = "label",
        label_field: str = "forward_return",
        exposure_category: str | None = "barra",
        exposure_fields: tuple[str, ...] = (),
        market_cap_category: str | None = "d_field",
        market_cap_field: str | None = "market_cap",
        tradable_category: str | None = "mask",
        tradable_field: str | None = "tradable",
    ) -> LocalPanelSpec:
        factors = self.by_status(*statuses)
        if not factors:
            raise ValueError(f"no factors registered with statuses={statuses}")
        categories = {item.storage_category for item in factors}
        if len(categories) != 1:
            raise ValueError(
                "LocalPanelSpec requires factors from one storage category; "
                f"got {sorted(categories)}"
            )
        return LocalPanelSpec(
            factor_category=next(iter(categories)),
            factor_fields=tuple(item.field_name for item in factors),
            label_category=label_category,
            label_field=label_field,
            exposure_category=exposure_category,
            exposure_fields=exposure_fields,
            market_cap_category=market_cap_category,
            market_cap_field=market_cap_field,
            tradable_category=tradable_category,
            tradable_field=tradable_field,
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("registry path is not configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "factors": [metadata_to_dict(item) for item in self.all()],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.path = target
        return target

    def load(self, path: str | Path | None = None) -> "FactorRegistry":
        source = Path(path) if path is not None else self.path
        if source is None:
            raise ValueError("registry path is not configured")
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = {}
        for raw in payload.get("factors", []):
            item = metadata_from_dict(raw).validate()
            records[item.key] = item
        self._records = records
        self.path = source
        return self


def metadata_to_dict(metadata: FactorMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["status"] = metadata.status.value
    data["risk_tags"] = list(metadata.risk_tags)
    data["failure_modes"] = list(metadata.failure_modes)
    data["shape"] = list(metadata.shape) if metadata.shape is not None else None
    return data


def metadata_from_dict(data: Mapping[str, Any]) -> FactorMetadata:
    values = dict(data)
    values["status"] = FactorStatus(values.get("status", FactorStatus.RESEARCH.value))
    values["risk_tags"] = tuple(values.get("risk_tags", ()))
    values["failure_modes"] = tuple(values.get("failure_modes", ()))
    if values.get("shape") is not None:
        values["shape"] = tuple(values["shape"])
    return FactorMetadata(**values)


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
