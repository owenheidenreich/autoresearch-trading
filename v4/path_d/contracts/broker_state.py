"""BrokerStateSnapshotV1 used by the deterministic governor and fake gateway."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._base import canonical_json, require_choice, require_exact_keys, require_int, require_nonempty, require_utc_timestamp


SCHEMA_VERSION = "pathd.broker_state_snapshot.v1"


@dataclass(frozen=True)
class PositionV1:
    osi_symbol: str
    quantity: int
    average_cost_micros: int
    opened_at_utc: str

    def __post_init__(self) -> None:
        if len(self.osi_symbol) != 21:
            raise ValueError("position osi_symbol must be exactly 21 characters")
        require_int(self.quantity, "quantity", minimum=1)
        require_int(self.average_cost_micros, "average_cost_micros", minimum=0)
        require_utc_timestamp(self.opened_at_utc, "opened_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositionV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class BrokerStateSnapshotV1:
    schema_version: str
    snapshot_version: str
    captured_at_utc: str
    account_id_redacted: str
    available_funds_micros: int
    daily_pnl_micros: int
    open_positions: tuple[PositionV1, ...]
    connectivity: str
    source: str

    def __post_init__(self) -> None:
        require_choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        require_nonempty(self.snapshot_version, "snapshot_version")
        require_utc_timestamp(self.captured_at_utc, "captured_at_utc")
        require_nonempty(self.account_id_redacted, "account_id_redacted")
        require_int(self.available_funds_micros, "available_funds_micros", minimum=0)
        require_int(self.daily_pnl_micros, "daily_pnl_micros")
        require_choice(self.connectivity, ("CONNECTED", "DISCONNECTED", "UNKNOWN"), "connectivity")
        require_choice(self.source, ("FAKE_GATEWAY", "IBKR"), "source")
        symbols = [position.osi_symbol for position in self.open_positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("open positions must have unique OSI symbols")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "snapshot_version": self.snapshot_version,
            "captured_at_utc": self.captured_at_utc, "account_id_redacted": self.account_id_redacted,
            "available_funds_micros": self.available_funds_micros, "daily_pnl_micros": self.daily_pnl_micros,
            "open_positions": [position.to_dict() for position in self.open_positions],
            "connectivity": self.connectivity, "source": self.source,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BrokerStateSnapshotV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        values = dict(payload)
        values["open_positions"] = tuple(PositionV1.from_dict(item) for item in values["open_positions"])
        return cls(**values)


POSITION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": list(PositionV1.__dataclass_fields__),
    "properties": {"osi_symbol": {"type": "string", "minLength": 21, "maxLength": 21}, "quantity": {"type": "integer", "minimum": 1}, "average_cost_micros": {"type": "integer", "minimum": 0}, "opened_at_utc": {"type": "string"}},
}
BROKER_STATE_SNAPSHOT_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_VERSION,
    "type": "object", "additionalProperties": False, "required": list(BrokerStateSnapshotV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "snapshot_version": {"type": "string"},
        "captured_at_utc": {"type": "string"}, "account_id_redacted": {"type": "string"},
        "available_funds_micros": {"type": "integer", "minimum": 0}, "daily_pnl_micros": {"type": "integer"},
        "open_positions": {"type": "array", "items": POSITION_SCHEMA},
        "connectivity": {"enum": ["CONNECTED", "DISCONNECTED", "UNKNOWN"]},
        "source": {"enum": ["FAKE_GATEWAY", "IBKR"]},
    },
}

