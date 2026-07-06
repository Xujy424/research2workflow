"""Versioned factor registry and lifecycle governance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dataclass_field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..matrix_store import MatrixStore


class FactorStatus(str, Enum):
    RESEARCH = "research"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"


ALLOWED_TRANSITIONS: dict[FactorStatus, set[FactorStatus]] = {
    FactorStatus.RESEARCH: {FactorStatus.RESEARCH, FactorStatus.CANDIDATE, FactorStatus.RETIRED},
    FactorStatus.CANDIDATE: {FactorStatus.RESEARCH, FactorStatus.CANDIDATE, FactorStatus.SHADOW, FactorStatus.RETIRED},
    FactorStatus.SHADOW: {FactorStatus.CANDIDATE, FactorStatus.SHADOW, FactorStatus.PRODUCTION, FactorStatus.RETIRED},
    FactorStatus.PRODUCTION: {FactorStatus.SHADOW, FactorStatus.PRODUCTION, FactorStatus.RETIRED},
    FactorStatus.RETIRED: {FactorStatus.RETIRED},
}


@dataclass(frozen=True)
class FactorMetadata:
    factor_id: str
    name: str
    version: str
    owner: str
    family: str
    horizon_days: int
    status: FactorStatus = FactorStatus.RESEARCH
    category: str = "factorpool"
    field: str | None = None
    dtype: str = "float64"
    shape: tuple[int, int] | None = None
    path: str | None = None
    formula: str = ""
    data_source: str = ""
    universe: str = ""
    code_commit: str = "unknown"
    approved_by: str | None = None
    update_enabled: bool = False
    retired: bool = False
    validation: Mapping[str, Any] = dataclass_field(default_factory=dict)
    monitoring: Mapping[str, Any] = dataclass_field(default_factory=dict)
    notes: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    last_monitor_at: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.factor_id, self.version

    @property
    def storage_field(self) -> str:
        return self.field or f"{self.factor_id}.{self.version}"

    def validate(self) -> "FactorMetadata":
        if not self.factor_id or not self.version or not self.name:
            raise ValueError("factor_id, version and name are required")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.retired and self.status != FactorStatus.RETIRED:
            raise ValueError("retired factors must use RETIRED status")
        return self


@dataclass(frozen=True)
class FactorLifecycleEvent:
    factor_id: str
    version: str
    old_status: FactorStatus
    new_status: FactorStatus
    reason: str = ""
    decision_by: str = "system"
    evidence_path: str = ""
    created_at: str = dataclass_field(default_factory=lambda: utc_now())


@dataclass(frozen=True)
class FactorDecisionLog:
    factor_id: str
    version: str
    current_status: FactorStatus
    suggested_status: FactorStatus
    action: str
    reason: str
    metrics_snapshot: Mapping[str, Any] = dataclass_field(default_factory=dict)
    operator: str = "system"
    created_at: str = dataclass_field(default_factory=lambda: utc_now())


@dataclass(frozen=True)
class FactorMonitorRecord:
    factor_id: str
    version: str
    metrics_snapshot: Mapping[str, Any]
    alert_level: str = "info"
    message: str = ""
    created_at: str = dataclass_field(default_factory=lambda: utc_now())


class FactorRegistry:
    """Persistent JSON registry for factor versions, decisions and monitoring."""

    def __init__(self, path: str | Path = "D:/data/factorpool/registry.json") -> None:
        self.path = Path(path)
        self._records: dict[tuple[str, str], FactorMetadata] = {}
        self._status_log: list[FactorLifecycleEvent] = []
        self._decision_log: list[FactorDecisionLog] = []
        self._monitor_log: list[FactorMonitorRecord] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            self.load()

    def register(self, metadata: FactorMetadata, *, reason: str = "register", decision_by: str = "system") -> None:
        item = metadata.validate()
        if item.key in self._records:
            raise ValueError(f"factor version already exists: {item.key}")
        item = replace(item, created_at=item.created_at or utc_now(), updated_at=utc_now())
        self._records[item.key] = item
        self._status_log.append(FactorLifecycleEvent(item.factor_id, item.version, item.status, item.status, reason, decision_by))

    def upsert(self, metadata: FactorMetadata) -> None:
        item = metadata.validate()
        created_at = self._records.get(item.key, item).created_at
        self._records[item.key] = replace(item, created_at=created_at or utc_now(), updated_at=utc_now())

    def get(self, factor_id: str, version: str = "v1") -> FactorMetadata:
        return self._records[(factor_id, version)]

    def all(self) -> list[FactorMetadata]:
        return sorted(self._records.values(), key=lambda item: item.key)

    def by_status(self, *statuses: FactorStatus) -> list[FactorMetadata]:
        wanted = set(statuses)
        return [item for item in self.all() if item.status in wanted and not item.retired]

    def allowed_next_statuses(self, status: FactorStatus | str) -> set[FactorStatus]:
        return set(ALLOWED_TRANSITIONS[FactorStatus(status)])

    def promote(
        self,
        factor_id: str,
        version: str,
        status: FactorStatus,
        *,
        approved_by: str | None = None,
        validation: Mapping[str, Any] | None = None,
        notes: str | None = None,
        update_enabled: bool | None = None,
        reason: str = "manual lifecycle change",
        evidence_path: str = "",
        validate_transition: bool = True,
    ) -> FactorMetadata:
        old = self.get(factor_id, version)
        new_status = FactorStatus(status)
        if validate_transition and new_status not in ALLOWED_TRANSITIONS[old.status]:
            raise ValueError(f"invalid transition: {old.status.value} -> {new_status.value}")
        item = replace(
            old,
            status=new_status,
            approved_by=approved_by or old.approved_by,
            validation={**dict(old.validation), **dict(validation or {})},
            notes=old.notes if notes is None else notes,
            update_enabled=old.update_enabled if update_enabled is None else update_enabled,
            retired=new_status == FactorStatus.RETIRED,
            updated_at=utc_now(),
        ).validate()
        self._records[item.key] = item
        self._status_log.append(
            FactorLifecycleEvent(
                factor_id=factor_id,
                version=version,
                old_status=old.status,
                new_status=new_status,
                reason=reason,
                decision_by=approved_by or "system",
                evidence_path=evidence_path,
            )
        )
        return item

    def retire(self, factor_id: str, version: str, *, notes: str = "", decision_by: str = "system", evidence_path: str = "") -> FactorMetadata:
        return self.promote(
            factor_id,
            version,
            FactorStatus.RETIRED,
            approved_by=decision_by,
            notes=notes,
            update_enabled=False,
            reason=notes or "retire factor",
            evidence_path=evidence_path,
            validate_transition=False,
        )

    def append_decision(
        self,
        factor_id: str,
        version: str,
        current_status: FactorStatus | str,
        suggested_status: FactorStatus | str,
        action: str,
        reason: str,
        *,
        metrics_snapshot: Mapping[str, Any] | None = None,
        operator: str = "system",
    ) -> FactorDecisionLog:
        row = FactorDecisionLog(
            factor_id=factor_id,
            version=version,
            current_status=FactorStatus(current_status),
            suggested_status=FactorStatus(suggested_status),
            action=action,
            reason=reason,
            metrics_snapshot=dict(metrics_snapshot or {}),
            operator=operator,
        )
        self._decision_log.append(row)
        return row

    def record_monitoring(
        self,
        factor_id: str,
        version: str,
        metrics_snapshot: Mapping[str, Any],
        *,
        alert_level: str = "info",
        message: str = "",
    ) -> FactorMonitorRecord:
        item = self.get(factor_id, version)
        now = utc_now()
        snapshot = dict(metrics_snapshot)
        record = FactorMonitorRecord(factor_id, version, snapshot, alert_level, message, now)
        self._monitor_log.append(record)
        self._records[item.key] = replace(item, monitoring=snapshot, last_monitor_at=now, updated_at=now)
        return record

    def status_log(self) -> list[FactorLifecycleEvent]:
        return list(self._status_log)

    def decision_log(self) -> list[FactorDecisionLog]:
        return list(self._decision_log)

    def monitor_log(self) -> list[FactorMonitorRecord]:
        return list(self._monitor_log)

    def scan_factorpool(
        self,
        store: MatrixStore | None = None,
        *,
        default_owner: str = "unknown",
        default_family: str = "unknown",
        version: str = "v1",
    ) -> list[FactorMetadata]:
        store = store or MatrixStore(self.path.parents[1] if self.path.parent.name == "factorpool" else "D:/data")
        axis = store.load_axis()
        folder = store.root / "factorpool"
        discovered: list[FactorMetadata] = []
        if not folder.exists():
            return discovered
        for path in sorted(folder.glob("*.bin")):
            factor_id = path.stem
            key = (factor_id, version)
            base = self._records.get(key) or FactorMetadata(
                factor_id=factor_id,
                name=factor_id,
                version=version,
                owner=default_owner,
                family=default_family,
                horizon_days=1,
            )
            stat = path.stat()
            item = replace(
                base,
                category="factorpool",
                field=factor_id,
                dtype=str(store.default_dtype),
                shape=axis.shape,
                path=str(path),
                validation={**dict(base.validation), "file_size_bytes": stat.st_size},
                updated_at=utc_now(),
            ).validate()
            self._records[key] = item
            discovered.append(item)
        return discovered

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "updated_at": utc_now(),
            "factors": [metadata_to_dict(item) for item in self.all()],
            "status_log": [lifecycle_event_to_dict(item) for item in self._status_log],
            "decision_log": [decision_log_to_dict(item) for item in self._decision_log],
            "monitor_log": [monitor_record_to_dict(item) for item in self._monitor_log],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        self.path = target
        return target

    def load(self, path: str | Path | None = None) -> "FactorRegistry":
        source = Path(path) if path is not None else self.path
        payload = json.loads(source.read_text(encoding="utf-8"))
        self._records = {}
        for raw in payload.get("factors", []):
            item = metadata_from_dict(raw).validate()
            self._records[item.key] = item
        self._status_log = [lifecycle_event_from_dict(raw) for raw in payload.get("status_log", [])]
        self._decision_log = [decision_log_from_dict(raw) for raw in payload.get("decision_log", [])]
        self._monitor_log = [monitor_record_from_dict(raw) for raw in payload.get("monitor_log", [])]
        self.path = source
        return self


def metadata_to_dict(metadata: FactorMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["status"] = metadata.status.value
    data["shape"] = list(metadata.shape) if metadata.shape is not None else None
    return data


def metadata_from_dict(data: Mapping[str, Any]) -> FactorMetadata:
    allowed = {item.name for item in fields(FactorMetadata)}
    values = {key: value for key, value in dict(data).items() if key in allowed}
    values["status"] = FactorStatus(values.get("status", FactorStatus.RESEARCH.value))
    if values.get("shape") is not None:
        values["shape"] = tuple(values["shape"])
    return FactorMetadata(**values)


def lifecycle_event_to_dict(event: FactorLifecycleEvent) -> dict[str, Any]:
    data = asdict(event)
    data["old_status"] = event.old_status.value
    data["new_status"] = event.new_status.value
    return data


def lifecycle_event_from_dict(data: Mapping[str, Any]) -> FactorLifecycleEvent:
    values = dict(data)
    values["old_status"] = FactorStatus(values.get("old_status", FactorStatus.RESEARCH.value))
    values["new_status"] = FactorStatus(values.get("new_status", FactorStatus.RESEARCH.value))
    return FactorLifecycleEvent(**values)


def decision_log_to_dict(row: FactorDecisionLog) -> dict[str, Any]:
    data = asdict(row)
    data["current_status"] = row.current_status.value
    data["suggested_status"] = row.suggested_status.value
    return data


def decision_log_from_dict(data: Mapping[str, Any]) -> FactorDecisionLog:
    values = dict(data)
    values["current_status"] = FactorStatus(values.get("current_status", FactorStatus.RESEARCH.value))
    values["suggested_status"] = FactorStatus(values.get("suggested_status", values["current_status"]))
    return FactorDecisionLog(**values)


def monitor_record_to_dict(record: FactorMonitorRecord) -> dict[str, Any]:
    return asdict(record)


def monitor_record_from_dict(data: Mapping[str, Any]) -> FactorMonitorRecord:
    return FactorMonitorRecord(**dict(data))


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
