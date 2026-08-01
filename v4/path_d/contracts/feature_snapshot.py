"""FeatureSnapshotV1: deterministic, source-neutral model input evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence

from ._base import canonical_json, require_choice, require_exact_keys, require_nonempty, require_sha256, require_utc_timestamp, semantic_sha256
from .execution_intent import ContractIdentityV1, DECISION_CLOCK, CONTRACT_IDENTITY_SCHEMA


SCHEMA_VERSION = "pathd.feature_snapshot.v1"


@dataclass(frozen=True)
class FeatureValueV1:
    name: str
    value: float

    def __post_init__(self) -> None:
        require_nonempty(self.name, "feature name")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool) or not math.isfinite(float(self.value)):
            raise ValueError(f"feature {self.name!r} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": float(self.value)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureValueV1":
        require_exact_keys(payload, required=("name", "value"), name=cls.__name__)
        return cls(name=str(payload["name"]), value=float(payload["value"]))


@dataclass(frozen=True)
class FeatureSnapshotV1:
    schema_version: str
    snapshot_id: str
    session_date: str
    decision_clock: str
    decision_available_at_utc: str
    option_watermark_received_timestamp_utc: str
    spx_watermark_received_timestamp_utc: str
    feature_contract_version: str
    features: tuple[FeatureValueV1, ...]
    selected_contract: ContractIdentityV1
    position_state: str

    def __post_init__(self) -> None:
        require_choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        require_sha256(self.snapshot_id, "snapshot_id")
        try:
            date.fromisoformat(self.session_date)
        except ValueError as exc:
            raise ValueError("session_date must be YYYY-MM-DD") from exc
        require_choice(self.decision_clock, (DECISION_CLOCK,), "decision_clock")
        for name in ("decision_available_at_utc", "option_watermark_received_timestamp_utc", "spx_watermark_received_timestamp_utc"):
            require_utc_timestamp(getattr(self, name), name)
        require_nonempty(self.feature_contract_version, "feature_contract_version")
        require_choice(self.position_state, ("FLAT", "LONG_ONE"), "position_state")
        names = [item.name for item in self.features]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("features must be unique and lexicographically sorted")
        if self.snapshot_id != semantic_sha256(self.semantic_payload()):
            raise ValueError("snapshot_id does not match semantic payload")

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "session_date": self.session_date,
            "decision_clock": self.decision_clock, "decision_available_at_utc": self.decision_available_at_utc,
            "option_watermark_received_timestamp_utc": self.option_watermark_received_timestamp_utc,
            "spx_watermark_received_timestamp_utc": self.spx_watermark_received_timestamp_utc,
            "feature_contract_version": self.feature_contract_version,
            "features": [item.to_dict() for item in self.features],
            "selected_contract": self.selected_contract.to_dict(), "position_state": self.position_state,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, **self.semantic_payload()}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def feature_map(self) -> dict[str, float]:
        return {item.name: float(item.value) for item in self.features}

    @classmethod
    def create(cls, *, features: Mapping[str, float] | Sequence[FeatureValueV1], **values: Any) -> "FeatureSnapshotV1":
        items = (
            tuple(sorted((FeatureValueV1(str(name), float(value)) for name, value in features.items()), key=lambda item: item.name))
            if isinstance(features, Mapping)
            else tuple(sorted(features, key=lambda item: item.name))
        )
        values = {"schema_version": SCHEMA_VERSION, "features": items, **values}
        payload = {key: value for key, value in values.items() if key != "snapshot_id"}
        payload["features"] = [item.to_dict() for item in items]
        payload["selected_contract"] = values["selected_contract"].to_dict()
        values["snapshot_id"] = semantic_sha256(payload)
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSnapshotV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        values = dict(payload)
        values["features"] = tuple(FeatureValueV1.from_dict(item) for item in values["features"])
        values["selected_contract"] = ContractIdentityV1.from_dict(values["selected_contract"])
        return cls(**values)


FEATURE_VALUE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["name", "value"],
    "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
}
FEATURE_SNAPSHOT_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_VERSION,
    "type": "object", "additionalProperties": False,
    "required": list(FeatureSnapshotV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "snapshot_id": {"type": "string"},
        "session_date": {"type": "string"}, "decision_clock": {"const": DECISION_CLOCK},
        "decision_available_at_utc": {"type": "string"},
        "option_watermark_received_timestamp_utc": {"type": "string"},
        "spx_watermark_received_timestamp_utc": {"type": "string"},
        "feature_contract_version": {"type": "string"},
        "features": {"type": "array", "items": FEATURE_VALUE_SCHEMA},
        "selected_contract": CONTRACT_IDENTITY_SCHEMA,
        "position_state": {"enum": ["FLAT", "LONG_ONE"]},
    },
}
