"""Canonical market-event wire contract for historical and future live feeds."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

from ._base import (
    canonical_json,
    require_choice,
    require_exact_keys,
    require_int,
    require_sha256,
    require_utc_timestamp,
    semantic_sha256,
)


SCHEMA_VERSION = "pathd.canonical_market_event.v1"


@dataclass(frozen=True)
class CanonicalMarketEventV1:
    schema_version: str
    event_id: str
    source: str
    event_type: str
    session_date: str
    source_timestamp_utc: str
    received_timestamp_utc: str
    osi_symbol: str | None
    bid_price_micros: int | None
    ask_price_micros: int | None
    bid_size: int | None
    ask_size: int | None
    index_symbol: str | None
    index_price_micros: int | None
    volume: int | None

    def __post_init__(self) -> None:
        require_choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        require_sha256(self.event_id, "event_id")
        require_choice(
            self.source,
            ("DATABENTO_OPRA", "THETADATA_SPX", "IBKR_EXECUTION_FIXTURE"),
            "source",
        )
        require_choice(self.event_type, ("OPTION_QUOTE", "SPX_INDEX", "BROKER_QUOTE"), "event_type")
        try:
            date.fromisoformat(self.session_date)
        except ValueError as exc:
            raise ValueError("session_date must be YYYY-MM-DD") from exc
        require_utc_timestamp(self.source_timestamp_utc, "source_timestamp_utc")
        require_utc_timestamp(self.received_timestamp_utc, "received_timestamp_utc")
        if _timestamp(self.received_timestamp_utc) < _timestamp(self.source_timestamp_utc):
            raise ValueError("received timestamp cannot precede source timestamp")
        for name in ("bid_price_micros", "ask_price_micros", "bid_size", "ask_size", "index_price_micros", "volume"):
            value = getattr(self, name)
            if value is not None:
                require_int(value, name, minimum=0)
        if self.event_type in {"OPTION_QUOTE", "BROKER_QUOTE"}:
            if self.osi_symbol is None or len(self.osi_symbol) != 21:
                raise ValueError("option/broker quote requires an exact 21-character OSI symbol")
            if self.bid_price_micros is None or self.ask_price_micros is None:
                raise ValueError("option/broker quote requires bid and ask")
            if self.ask_price_micros < self.bid_price_micros:
                raise ValueError("ask must be >= bid")
            if self.index_symbol is not None or self.index_price_micros is not None:
                raise ValueError("option/broker quote cannot carry index fields")
            expected_source = "DATABENTO_OPRA" if self.event_type == "OPTION_QUOTE" else "IBKR_EXECUTION_FIXTURE"
            if self.source != expected_source:
                raise ValueError(f"{self.event_type} requires source={expected_source}")
        if self.event_type == "SPX_INDEX":
            if self.index_symbol != "SPX" or self.index_price_micros is None:
                raise ValueError("SPX_INDEX requires index_symbol=SPX and a price")
            if any(value is not None for value in (self.osi_symbol, self.bid_price_micros, self.ask_price_micros, self.bid_size, self.ask_size)):
                raise ValueError("SPX_INDEX cannot carry option fields")
            if self.source != "THETADATA_SPX":
                raise ValueError("SPX_INDEX requires source=THETADATA_SPX")
        if self.event_id != semantic_sha256(self.semantic_payload()):
            raise ValueError("event_id does not match canonical event payload")

    def semantic_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "event_id"}

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(cls, **values: Any) -> "CanonicalMarketEventV1":
        values = {"schema_version": SCHEMA_VERSION, **values}
        values.setdefault("osi_symbol", None)
        values.setdefault("bid_price_micros", None)
        values.setdefault("ask_price_micros", None)
        values.setdefault("bid_size", None)
        values.setdefault("ask_size", None)
        values.setdefault("index_symbol", None)
        values.setdefault("index_price_micros", None)
        values.setdefault("volume", None)
        payload = {key: value for key, value in values.items() if key != "event_id"}
        values["event_id"] = semantic_sha256(payload)
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalMarketEventV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


_NULLABLE_INTEGER = {"type": ["integer", "null"], "minimum": 0}
_NULLABLE_STRING = {"type": ["string", "null"]}
CANONICAL_MARKET_EVENT_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_VERSION,
    "type": "object", "additionalProperties": False,
    "required": list(CanonicalMarketEventV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "event_id": {"type": "string"},
        "source": {"enum": ["DATABENTO_OPRA", "THETADATA_SPX", "IBKR_EXECUTION_FIXTURE"]},
        "event_type": {"enum": ["OPTION_QUOTE", "SPX_INDEX", "BROKER_QUOTE"]},
        "session_date": {"type": "string"}, "source_timestamp_utc": {"type": "string"},
        "received_timestamp_utc": {"type": "string"}, "osi_symbol": _NULLABLE_STRING,
        "bid_price_micros": _NULLABLE_INTEGER, "ask_price_micros": _NULLABLE_INTEGER,
        "bid_size": _NULLABLE_INTEGER, "ask_size": _NULLABLE_INTEGER,
        "index_symbol": _NULLABLE_STRING, "index_price_micros": _NULLABLE_INTEGER,
        "volume": _NULLABLE_INTEGER,
    },
}


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
