"""ExecutionEventV1 transcript row."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._base import canonical_json, require_exact_keys, require_int, require_nonempty, require_sha256, require_utc_timestamp, semantic_sha256


SCHEMA_VERSION = "pathd.execution_event.v1"
ORDER_STATES = (
    "CREATED", "SUBMIT_AUTHORIZED", "SUBMITTED", "ACKNOWLEDGED", "WORKING",
    "PARTIALLY_FILLED", "FILLED", "CANCEL_REQUESTED", "CANCEL_CONFIRMED",
    "LATE_FILL_AFTER_CANCEL", "REJECTED", "EXPIRED", "DISCONNECTED",
    "UNKNOWN_RECONCILE", "RECONCILING", "RECONCILED_FILLED", "RECONCILED_CANCELLED",
)


@dataclass(frozen=True)
class ExecutionEventV1:
    schema_version: str
    event_id: str
    order_id: str
    intent_id: str
    event_type: str
    state_from: str | None
    state_to: str
    event_at_utc: str
    monotonic_ns: int
    filled_quantity: int
    fill_price_micros: int | None
    reason_code: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        require_sha256(self.event_id, "event_id")
        require_nonempty(self.order_id, "order_id")
        require_sha256(self.intent_id, "intent_id")
        require_nonempty(self.event_type, "event_type")
        require_nonempty(self.state_to, "state_to")
        if self.state_to not in ORDER_STATES or (self.state_from is not None and self.state_from not in ORDER_STATES):
            raise ValueError("execution event contains an unknown state")
        if self.event_type != self.state_to:
            raise ValueError("event_type must equal state_to in ExecutionEventV1")
        require_utc_timestamp(self.event_at_utc, "event_at_utc")
        require_int(self.monotonic_ns, "monotonic_ns", minimum=0)
        require_int(self.filled_quantity, "filled_quantity", minimum=0)
        if self.fill_price_micros is not None:
            require_int(self.fill_price_micros, "fill_price_micros", minimum=0)
        require_nonempty(self.reason_code, "reason_code")
        if self.event_id != semantic_sha256(self.semantic_payload()):
            raise ValueError("event_id does not match execution event payload")

    def semantic_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "event_id"}

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, **self.semantic_payload()}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(cls, **values: Any) -> "ExecutionEventV1":
        values = {"schema_version": SCHEMA_VERSION, **values}
        values["event_id"] = semantic_sha256({key: value for key, value in values.items() if key != "event_id"})
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionEventV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


EXECUTION_EVENT_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_VERSION,
    "type": "object", "additionalProperties": False, "required": list(ExecutionEventV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "event_id": {"type": "string"},
        "order_id": {"type": "string"}, "intent_id": {"type": "string"}, "event_type": {"enum": list(ORDER_STATES)},
        "state_from": {"enum": [None, *ORDER_STATES]}, "state_to": {"enum": list(ORDER_STATES)},
        "event_at_utc": {"type": "string"}, "monotonic_ns": {"type": "integer", "minimum": 0},
        "filled_quantity": {"type": "integer", "minimum": 0},
        "fill_price_micros": {"type": ["integer", "null"], "minimum": 0}, "reason_code": {"type": "string"},
    },
}
