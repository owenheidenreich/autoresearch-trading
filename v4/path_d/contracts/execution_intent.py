"""ExecutionIntentV1 and its semantic identity law."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

from ._base import (
    canonical_json,
    require_choice,
    require_exact_keys,
    require_int,
    require_nonempty,
    require_sha256,
    require_utc_timestamp,
    semantic_sha256,
)


SCHEMA_VERSION = "pathd.execution_intent.v1"
DECISION_CLOCK = "received_timestamp_utc"


@dataclass(frozen=True)
class ContractIdentityV1:
    osi_symbol: str
    underlying: str
    trading_class: str
    expiry: str
    strike_milli: int
    right: str
    multiplier: int = 100
    settlement: str = "PM"
    currency: str = "USD"

    def __post_init__(self) -> None:
        if len(self.osi_symbol) != 21:
            raise ValueError("osi_symbol must be the exact padded 21-character OSI symbol")
        require_choice(self.underlying, ("SPX",), "underlying")
        require_choice(self.trading_class, ("SPXW",), "trading_class")
        try:
            date.fromisoformat(self.expiry)
        except ValueError as exc:
            raise ValueError("expiry must be YYYY-MM-DD") from exc
        require_int(self.strike_milli, "strike_milli", minimum=1)
        require_choice(self.right, ("C", "P"), "right")
        if self.multiplier != 100 or self.settlement != "PM" or self.currency != "USD":
            raise ValueError("Path-D v1 requires multiplier=100, PM settlement, and USD")
        root = self.osi_symbol[:6].strip()
        if root != self.trading_class:
            raise ValueError("OSI root must match trading_class")
        if self.osi_symbol[12] != self.right or int(self.osi_symbol[13:21]) != self.strike_milli:
            raise ValueError("OSI right/strike must match contract fields")
        if datetime.strptime(self.osi_symbol[6:12], "%y%m%d").date().isoformat() != self.expiry:
            raise ValueError("OSI expiry must match contract expiry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "osi_symbol": self.osi_symbol,
            "underlying": self.underlying,
            "trading_class": self.trading_class,
            "expiry": self.expiry,
            "strike_milli": self.strike_milli,
            "right": self.right,
            "multiplier": self.multiplier,
            "settlement": self.settlement,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractIdentityV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class ProducerIdentityV1:
    component_version: str
    strategy_id: str
    artifact_sha256: str | None
    feature_contract_version: str
    feature_snapshot_sha256: str

    def __post_init__(self) -> None:
        require_nonempty(self.component_version, "component_version")
        require_nonempty(self.strategy_id, "strategy_id")
        if self.artifact_sha256 is not None:
            require_sha256(self.artifact_sha256, "artifact_sha256")
        require_nonempty(self.feature_contract_version, "feature_contract_version")
        require_sha256(self.feature_snapshot_sha256, "feature_snapshot_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_version": self.component_version,
            "strategy_id": self.strategy_id,
            "artifact_sha256": self.artifact_sha256,
            "feature_contract_version": self.feature_contract_version,
            "feature_snapshot_sha256": self.feature_snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProducerIdentityV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class DecisionDirectiveV1:
    action: str
    side: str
    position_effect: str
    urgency: str
    quantity: int
    reason_code: str

    def __post_init__(self) -> None:
        require_choice(self.action, ("OPEN_LONG", "CLOSE_LONG"), "action")
        require_choice(self.side, ("BUY", "SELL"), "side")
        require_choice(self.position_effect, ("OPEN", "CLOSE"), "position_effect")
        require_choice(
            self.urgency,
            ("NORMAL_ENTRY", "NORMAL_EXIT", "PROTECTIVE_EXIT", "FORCED_FLAT"),
            "urgency",
        )
        require_int(self.quantity, "quantity", minimum=1)
        require_nonempty(self.reason_code, "reason_code")
        if self.action == "OPEN_LONG" and (self.side, self.position_effect) != ("BUY", "OPEN"):
            raise ValueError("OPEN_LONG requires BUY/OPEN")
        if self.action == "CLOSE_LONG" and (self.side, self.position_effect) != ("SELL", "CLOSE"):
            raise ValueError("CLOSE_LONG requires SELL/CLOSE")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionDirectiveV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class PriceBudgetV1:
    unit: str
    reference_vendor: str
    reference_bid_micros: int
    reference_ask_micros: int
    max_adverse_move_micros: int
    hard_limit_micros: int

    def __post_init__(self) -> None:
        require_choice(self.unit, ("USD_OPTION_PRICE_MICROS",), "unit")
        require_choice(self.reference_vendor, ("DATABENTO_OPRA", "IBKR_SAFETY"), "reference_vendor")
        for name in (
            "reference_bid_micros",
            "reference_ask_micros",
            "max_adverse_move_micros",
            "hard_limit_micros",
        ):
            require_int(getattr(self, name), name, minimum=0)
        if self.reference_ask_micros < self.reference_bid_micros:
            raise ValueError("reference ask must be >= reference bid")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriceBudgetV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class IntentClocksV1:
    decision_clock: str
    event_interval_end_utc: str
    option_received_watermark_utc: str
    spx_received_watermark_utc: str
    decision_available_at_utc: str
    model_started_at_utc: str
    model_finished_at_utc: str
    intent_emitted_at_utc: str
    valid_until_utc: str

    def __post_init__(self) -> None:
        require_choice(self.decision_clock, (DECISION_CLOCK,), "decision_clock")
        for name in self.__dataclass_fields__:
            if name != "decision_clock":
                require_utc_timestamp(getattr(self, name), name)
        if _timestamp(self.decision_available_at_utc) < max(
            _timestamp(self.option_received_watermark_utc),
            _timestamp(self.spx_received_watermark_utc),
        ):
            raise ValueError("decision availability must not precede either received watermark")
        if not (
            _timestamp(self.model_started_at_utc)
            <= _timestamp(self.model_finished_at_utc)
            <= _timestamp(self.intent_emitted_at_utc)
            <= _timestamp(self.valid_until_utc)
        ):
            raise ValueError("measurement and validity clocks are out of order")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentClocksV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class PositionPreconditionV1:
    expected_position: str
    position_snapshot_version: str
    held_osi_symbol: str | None

    def __post_init__(self) -> None:
        require_choice(self.expected_position, ("FLAT", "LONG_ONE"), "expected_position")
        require_nonempty(self.position_snapshot_version, "position_snapshot_version")
        if self.expected_position == "FLAT" and self.held_osi_symbol is not None:
            raise ValueError("FLAT precondition cannot name a held contract")
        if self.expected_position == "LONG_ONE" and (
            self.held_osi_symbol is None or len(self.held_osi_symbol) != 21
        ):
            raise ValueError("LONG_ONE precondition requires an exact OSI symbol")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositionPreconditionV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True)
class ExecutionIntentV1:
    schema_version: str
    intent_id: str
    trace_id: str
    parent_intent_id: str | None
    origin: str
    producer: ProducerIdentityV1
    decision: DecisionDirectiveV1
    contract: ContractIdentityV1
    price_budget: PriceBudgetV1
    clocks: IntentClocksV1
    state_precondition: PositionPreconditionV1
    execution_profile_version: str

    def __post_init__(self) -> None:
        require_choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        require_sha256(self.intent_id, "intent_id")
        require_nonempty(self.trace_id, "trace_id")
        if self.parent_intent_id is not None:
            require_sha256(self.parent_intent_id, "parent_intent_id")
        require_choice(self.origin, ("MODEL", "DETERMINISTIC_EXIT", "RISK_GOVERNOR"), "origin")
        require_nonempty(self.execution_profile_version, "execution_profile_version")
        if self.origin == "RISK_GOVERNOR" and self.decision.urgency != "FORCED_FLAT":
            raise ValueError("RISK_GOVERNOR origin is reserved for FORCED_FLAT")
        if self.decision.urgency == "FORCED_FLAT" and self.origin != "RISK_GOVERNOR":
            raise ValueError("FORCED_FLAT may originate only from RISK_GOVERNOR")
        if self.decision.side == "SELL" and self.price_budget.hard_limit_micros > self.price_budget.reference_bid_micros:
            raise ValueError("sell hard limit cannot be above the decision reference bid")
        if self.decision.side == "BUY" and self.price_budget.hard_limit_micros < self.price_budget.reference_ask_micros:
            raise ValueError("buy hard limit cannot be below the decision reference ask")
        expected = semantic_sha256(self.semantic_payload())
        if self.intent_id != expected:
            raise ValueError(f"intent_id does not match semantic payload: expected {expected}")

    def semantic_payload(self) -> dict[str, Any]:
        """Identity-bearing fields; trace and measurement-only timings are excluded."""
        return {
            "schema_version": self.schema_version,
            "parent_intent_id": self.parent_intent_id,
            "origin": self.origin,
            "producer": self.producer.to_dict(),
            "decision": self.decision.to_dict(),
            "contract": self.contract.to_dict(),
            "price_budget": self.price_budget.to_dict(),
            "clocks": {
                "decision_clock": self.clocks.decision_clock,
                "event_interval_end_utc": self.clocks.event_interval_end_utc,
                "option_received_watermark_utc": self.clocks.option_received_watermark_utc,
                "spx_received_watermark_utc": self.clocks.spx_received_watermark_utc,
                "decision_available_at_utc": self.clocks.decision_available_at_utc,
                "valid_until_utc": self.clocks.valid_until_utc,
            },
            "state_precondition": self.state_precondition.to_dict(),
            "execution_profile_version": self.execution_profile_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "trace_id": self.trace_id,
            "parent_intent_id": self.parent_intent_id,
            "origin": self.origin,
            "producer": self.producer.to_dict(),
            "decision": self.decision.to_dict(),
            "contract": self.contract.to_dict(),
            "price_budget": self.price_budget.to_dict(),
            "clocks": self.clocks.to_dict(),
            "state_precondition": self.state_precondition.to_dict(),
            "execution_profile_version": self.execution_profile_version,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        trace_id: str,
        parent_intent_id: str | None,
        origin: str,
        producer: ProducerIdentityV1,
        decision: DecisionDirectiveV1,
        contract: ContractIdentityV1,
        price_budget: PriceBudgetV1,
        clocks: IntentClocksV1,
        state_precondition: PositionPreconditionV1,
        execution_profile_version: str,
    ) -> "ExecutionIntentV1":
        provisional = object.__new__(cls)
        values = {
            "schema_version": SCHEMA_VERSION,
            "intent_id": "sha256:" + "0" * 64,
            "trace_id": trace_id,
            "parent_intent_id": parent_intent_id,
            "origin": origin,
            "producer": producer,
            "decision": decision,
            "contract": contract,
            "price_budget": price_budget,
            "clocks": clocks,
            "state_precondition": state_precondition,
            "execution_profile_version": execution_profile_version,
        }
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        values["intent_id"] = semantic_sha256(provisional.semantic_payload())
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionIntentV1":
        require_exact_keys(payload, required=cls.__dataclass_fields__, name=cls.__name__)
        values = dict(payload)
        values["producer"] = ProducerIdentityV1.from_dict(values["producer"])
        values["decision"] = DecisionDirectiveV1.from_dict(values["decision"])
        values["contract"] = ContractIdentityV1.from_dict(values["contract"])
        values["price_budget"] = PriceBudgetV1.from_dict(values["price_budget"])
        values["clocks"] = IntentClocksV1.from_dict(values["clocks"])
        values["state_precondition"] = PositionPreconditionV1.from_dict(values["state_precondition"])
        return cls(**values)


_STRICT = {"additionalProperties": False}
_STRING = {"type": "string"}
_NULLABLE_STRING = {"type": ["string", "null"]}
CONTRACT_IDENTITY_SCHEMA = {
    "type": "object", **_STRICT,
    "required": list(ContractIdentityV1.__dataclass_fields__),
    "properties": {
        "osi_symbol": {"type": "string", "minLength": 21, "maxLength": 21},
        "underlying": {"const": "SPX"}, "trading_class": {"const": "SPXW"},
        "expiry": _STRING, "strike_milli": {"type": "integer", "minimum": 1},
        "right": {"enum": ["C", "P"]}, "multiplier": {"const": 100},
        "settlement": {"const": "PM"}, "currency": {"const": "USD"},
    },
}
PRODUCER_SCHEMA = {
    "type": "object", **_STRICT, "required": list(ProducerIdentityV1.__dataclass_fields__),
    "properties": {name: (_NULLABLE_STRING if name == "artifact_sha256" else _STRING) for name in ProducerIdentityV1.__dataclass_fields__},
}
DECISION_SCHEMA = {
    "type": "object", **_STRICT, "required": list(DecisionDirectiveV1.__dataclass_fields__),
    "properties": {
        "action": {"enum": ["OPEN_LONG", "CLOSE_LONG"]}, "side": {"enum": ["BUY", "SELL"]},
        "position_effect": {"enum": ["OPEN", "CLOSE"]},
        "urgency": {"enum": ["NORMAL_ENTRY", "NORMAL_EXIT", "PROTECTIVE_EXIT", "FORCED_FLAT"]},
        "quantity": {"type": "integer", "minimum": 1}, "reason_code": _STRING,
    },
}
PRICE_BUDGET_SCHEMA = {
    "type": "object", **_STRICT, "required": list(PriceBudgetV1.__dataclass_fields__),
    "properties": {
        "unit": {"const": "USD_OPTION_PRICE_MICROS"},
        "reference_vendor": {"enum": ["DATABENTO_OPRA", "IBKR_SAFETY"]},
        **{name: {"type": "integer", "minimum": 0} for name in ("reference_bid_micros", "reference_ask_micros", "max_adverse_move_micros", "hard_limit_micros")},
    },
}
CLOCKS_SCHEMA = {
    "type": "object", **_STRICT, "required": list(IntentClocksV1.__dataclass_fields__),
    "properties": {name: ({"const": DECISION_CLOCK} if name == "decision_clock" else _STRING) for name in IntentClocksV1.__dataclass_fields__},
}
PRECONDITION_SCHEMA = {
    "type": "object", **_STRICT, "required": list(PositionPreconditionV1.__dataclass_fields__),
    "properties": {"expected_position": {"enum": ["FLAT", "LONG_ONE"]}, "position_snapshot_version": _STRING, "held_osi_symbol": _NULLABLE_STRING},
}
EXECUTION_INTENT_V1_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_VERSION,
    "type": "object", **_STRICT,
    "required": list(ExecutionIntentV1.__dataclass_fields__),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION}, "intent_id": _STRING, "trace_id": _STRING,
        "parent_intent_id": _NULLABLE_STRING, "origin": {"enum": ["MODEL", "DETERMINISTIC_EXIT", "RISK_GOVERNOR"]},
        "producer": PRODUCER_SCHEMA, "decision": DECISION_SCHEMA, "contract": CONTRACT_IDENTITY_SCHEMA,
        "price_budget": PRICE_BUDGET_SCHEMA, "clocks": CLOCKS_SCHEMA,
        "state_precondition": PRECONDITION_SCHEMA, "execution_profile_version": _STRING,
    },
}


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
