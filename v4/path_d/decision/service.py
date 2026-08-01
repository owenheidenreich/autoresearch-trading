"""Offline canonical-event replay to deterministic fixture intent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Iterable

from v4.path_d.contracts import (
    ContractIdentityV1,
    DecisionDirectiveV1,
    ExecutionIntentV1,
    FeatureSnapshotV1,
    IntentClocksV1,
    PositionPreconditionV1,
    PriceBudgetV1,
    ProducerIdentityV1,
    CanonicalMarketEventV1,
)
from v4.path_d.features.source_neutral import FEATURE_CONTRACT_VERSION, SourceNeutralFeatureBuilder

from .market_state import CanonicalMarketState


@dataclass(frozen=True)
class ExitFixtureState:
    held_osi_symbol: str
    position_snapshot_version: str
    entry_bid_micros: int
    running_max_bid_micros: int
    opened_at_utc: str


@dataclass(frozen=True)
class OfflineDecisionResult:
    snapshot: FeatureSnapshotV1
    intent: ExecutionIntentV1 | None


@dataclass(frozen=True)
class DeterministicExitFixtureProducer:
    floor_fraction_micros: int = 800_000
    max_giveback_micros: int = 500_000
    max_hold_seconds: int = 900
    max_adverse_execution_micros: int = 100_000

    def reason(self, *, state: ExitFixtureState, current_bid_micros: int, now_utc: str) -> str | None:
        running_max = max(state.entry_bid_micros, state.running_max_bid_micros, current_bid_micros)
        upward_floor = running_max * self.floor_fraction_micros // 1_000_000
        giveback_floor = max(0, running_max - self.max_giveback_micros)
        floor = max(upward_floor, giveback_floor)
        if current_bid_micros <= floor:
            return "DETERMINISTIC_UPWARD_FLOOR"
        if _timestamp(now_utc) >= _timestamp(state.opened_at_utc) + timedelta(seconds=self.max_hold_seconds):
            return "DETERMINISTIC_TIME_STOP"
        return None


class OfflineDecisionService:
    """Replay historical canonical events and emit no-op or serialized exit intent."""

    component_version = "pathd.offline-decision-service.v1"
    strategy_id = "pathd.deterministic-exit-fixture.v1"

    def __init__(
        self,
        *,
        builder: SourceNeutralFeatureBuilder | None = None,
        producer: DeterministicExitFixtureProducer | None = None,
    ) -> None:
        self.builder = builder or SourceNeutralFeatureBuilder()
        self.producer = producer or DeterministicExitFixtureProducer()

    def replay(
        self,
        events: Iterable[CanonicalMarketEventV1],
        *,
        exit_state: ExitFixtureState,
    ) -> OfflineDecisionResult:
        state = CanonicalMarketState()
        ordered = sorted(events, key=lambda event: _timestamp(event.received_timestamp_utc))
        if not ordered:
            raise ValueError("offline replay requires canonical events")
        for event in ordered:
            state.ingest(event)
        quote = state.option_quote(exit_state.held_osi_symbol)
        if quote.source != "DATABENTO_OPRA":
            raise ValueError("offline decision fixture requires a historical Databento option quote")
        if not state.spx_observations:
            raise ValueError("offline decision fixture requires historical ThetaData SPX context")
        assert quote.bid_price_micros is not None and quote.ask_price_micros is not None
        contract = contract_from_osi(exit_state.held_osi_symbol)
        decision_time = _utc_iso(max(_timestamp(quote.received_timestamp_utc), _timestamp(state.spx_watermark)))
        feature_map = self.builder.feature_map(
            spx=state.spx_observations,
            vix=(),
            decision_available_at_utc=decision_time,
            option_quote={
                "bid_price_micros": quote.bid_price_micros,
                "ask_price_micros": quote.ask_price_micros,
                "bid_size": quote.bid_size or 0,
                "ask_size": quote.ask_size or 0,
            },
            strike=contract.strike_milli / 1000.0,
        )
        feature_map = {name: value for name, value in feature_map.items() if math.isfinite(value)}
        snapshot = FeatureSnapshotV1.create(
            session_date=quote.session_date,
            decision_clock="received_timestamp_utc",
            decision_available_at_utc=decision_time,
            option_watermark_received_timestamp_utc=quote.received_timestamp_utc,
            spx_watermark_received_timestamp_utc=state.spx_watermark,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            features=feature_map,
            selected_contract=contract,
            position_state="LONG_ONE",
        )
        reason = self.producer.reason(state=exit_state, current_bid_micros=quote.bid_price_micros, now_utc=decision_time)
        if reason is None:
            return OfflineDecisionResult(snapshot=snapshot, intent=None)
        deadline = _timestamp(decision_time) + timedelta(seconds=2)
        clocks = IntentClocksV1(
            decision_clock="received_timestamp_utc",
            event_interval_end_utc=quote.source_timestamp_utc,
            option_received_watermark_utc=quote.received_timestamp_utc,
            spx_received_watermark_utc=state.spx_watermark,
            decision_available_at_utc=decision_time,
            model_started_at_utc=decision_time,
            model_finished_at_utc=decision_time,
            intent_emitted_at_utc=decision_time,
            valid_until_utc=_utc_iso(deadline),
        )
        intent = ExecutionIntentV1.create(
            trace_id=f"offline-{snapshot.snapshot_id[7:23]}",
            parent_intent_id=None,
            origin="DETERMINISTIC_EXIT",
            producer=ProducerIdentityV1(
                component_version=self.component_version,
                strategy_id=self.strategy_id,
                artifact_sha256=None,
                feature_contract_version=FEATURE_CONTRACT_VERSION,
                feature_snapshot_sha256=snapshot.snapshot_id,
            ),
            decision=DecisionDirectiveV1(
                action="CLOSE_LONG", side="SELL", position_effect="CLOSE",
                urgency="PROTECTIVE_EXIT", quantity=1, reason_code=reason,
            ),
            contract=contract,
            price_budget=PriceBudgetV1(
                unit="USD_OPTION_PRICE_MICROS", reference_vendor="DATABENTO_OPRA",
                reference_bid_micros=quote.bid_price_micros, reference_ask_micros=quote.ask_price_micros,
                max_adverse_move_micros=self.producer.max_adverse_execution_micros,
                hard_limit_micros=max(0, quote.bid_price_micros - self.producer.max_adverse_execution_micros),
            ),
            clocks=clocks,
            state_precondition=PositionPreconditionV1(
                expected_position="LONG_ONE", position_snapshot_version=exit_state.position_snapshot_version,
                held_osi_symbol=exit_state.held_osi_symbol,
            ),
            execution_profile_version="pathd.offline-limit-profile.v1",
        )
        return OfflineDecisionResult(snapshot=snapshot, intent=intent)


def contract_from_osi(osi_symbol: str) -> ContractIdentityV1:
    if len(osi_symbol) != 21:
        raise ValueError("OSI symbol must have exact 21-character padded form")
    root = osi_symbol[:6].strip()
    expiry_code = osi_symbol[6:12]
    right = osi_symbol[12]
    strike_digits = osi_symbol[13:21]
    if root != "SPXW" or right not in {"C", "P"} or not expiry_code.isdigit() or not strike_digits.isdigit():
        raise ValueError(f"unsupported Path-D OSI symbol: {osi_symbol!r}")
    expiry = datetime.strptime(expiry_code, "%y%m%d").date().isoformat()
    return ContractIdentityV1(
        osi_symbol=osi_symbol, underlying="SPX", trading_class="SPXW", expiry=expiry,
        strike_milli=int(strike_digits), right=right,
    )


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
