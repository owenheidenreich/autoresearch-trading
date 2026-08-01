"""GovernorDecisionV1: the only contract that can authorize submission."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._base import canonical_json, require_choice, require_exact_keys, require_nonempty, require_sha256, require_utc_timestamp, semantic_sha256


SCHEMA_VERSION = "pathd.governor_decision.v1"


@dataclass(frozen=True)
class GovernorDecisionV1:
    schema_version: str
    decision_id: str
    intent_id: str
    disposition: str
    reason_codes: tuple[str, ...]
    evaluated_at_utc: str
    broker_state_version: str
    authorization_token: str | None

    def __post_init__(self) -> None:
        require_choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        require_sha256(self.decision_id, "decision_id")
        require_sha256(self.intent_id, "intent_id")
        require_choice(self.disposition, ("ALLOW", "BLOCK"), "disposition")
        require_utc_timestamp(self.evaluated_at_utc, "evaluated_at_utc")
        require_nonempty(self.broker_state_version, "broker_state_version")
        if not self.reason_codes:
            raise ValueError("reason_codes must not be empty")
        for reason in self.reason_codes:
            require_nonempty(reason, "reason_code")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("reason_codes must be sorted and unique")
        expected_token = semantic_sha256(self.authorization_payload()) if self.disposition == "ALLOW" else None
        if self.authorization_token != expected_token:
            raise ValueError("authorization_token is absent or invalid for disposition")
        if self.decision_id != semantic_sha256(self.semantic_payload()):
            raise ValueError("decision_id does not match semantic payload")

    def authorization_payload(self) -> dict[str, Any]:
        return {
            "contract": SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "broker_state_version": self.broker_state_version,
            "evaluated_at_utc": self.evaluated_at_utc,
            "disposition": self.disposition,
        }

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "intent_id": self.intent_id,
            "disposition": self.disposition, "reason_codes": list(self.reason_codes),
            "evaluated_at_utc": self.evaluated_at_utc, "broker_state_version": self.broker_state_version,
            "authorization_token": self.authorization_token,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, **self.semantic_payload()}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        intent_id: str,
        disposition: str,
        reason_codes: Sequence[str],
        evaluated_at_utc: str,
        broker_state_version: str,
    ) -> "GovernorDecisionV1":
        reasons = tuple(sorted(set(str(reason) for reason in reason_codes)))
        provisional = {
            "contract": SCHEMA_VERSION, "intent_id": intent_id,
            "broker_state_version": broker_state_version, "evaluated_at_utc": evaluated_at_utc,
            "disposition": disposition,
        }
        token = semantic_sha256(provisional) if disposition == "ALLOW" else None
        semantic = {
            "schema_version": SCHEMA_VERSION, "intent_id": intent_id, "disposition": disposition,
            "reason_codes": list(reasons), "evaluated_at_utc": evaluated_at_utc,
            "broker_state_version": broker_state_version, "authorization_token": token,
        }
        return cls(decision_id=semantic_sha256(semantic), authorization_token=token, reason_codes=reasons, schema_version=SCHEMA_VERSION, intent_id=intent_id, disposition=disposition, evaluated_at_utc=evaluated_at_utc, broker_state_version=broker_state_version)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GovernorDecisionV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        values = dict(payload)
        values["reason_codes"] = tuple(values["reason_codes"])
        return cls(**values)


GOVERNOR_DECISION_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_VERSION,
    "type": "object", "additionalProperties": False, "required": list(GovernorDecisionV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "decision_id": {"type": "string"}, "intent_id": {"type": "string"},
        "disposition": {"enum": ["ALLOW", "BLOCK"]}, "reason_codes": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "evaluated_at_utc": {"type": "string"}, "broker_state_version": {"type": "string"},
        "authorization_token": {"type": ["string", "null"]},
    },
}
