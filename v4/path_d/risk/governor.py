"""Deterministic pre-submit governor; no broker connectivity or learned policy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from v4.path_d.contracts import (
    BrokerStateSnapshotV1,
    ContractIdentityV1,
    DecisionDirectiveV1,
    ExecutionIntentV1,
    GovernorDecisionV1,
    IntentClocksV1,
    PositionPreconditionV1,
    PositionV1,
    PriceBudgetV1,
    ProducerIdentityV1,
)


@dataclass(frozen=True)
class GovernorConfigV1:
    max_quantity: int = 1
    max_open_positions: int = 1
    max_daily_loss_micros: int = 100_000_000
    max_feed_age_ms: int = 2_000
    max_intent_age_ms: int = 2_000
    floor_fraction_micros: int = 800_000
    max_giveback_micros: int = 500_000
    max_hold_seconds: int = 900
    forced_flat_adverse_move_micros: int = 200_000
    allowed_forced_flat_reference_vendors: tuple[str, ...] = ("IBKR_SAFETY",)
    daily_loss_blocks_close: bool = True

    def __post_init__(self) -> None:
        vendors = self.allowed_forced_flat_reference_vendors
        if type(self.daily_loss_blocks_close) is not bool:
            raise TypeError("daily_loss_blocks_close must be an exact bool")
        if (
            type(vendors) is not tuple
            or not vendors
            or any(type(value) is not str for value in vendors)
            or len(set(vendors)) != len(vendors)
            or any(
                value not in {"IBKR_SAFETY", "DATABENTO_OPRA"}
                for value in vendors
            )
        ):
            raise ValueError("forced-flat reference-vendor allowlist is invalid")


@dataclass(frozen=True)
class FeedHealthV1:
    option_received_timestamp_utc: str
    spx_received_timestamp_utc: str
    option_feed_available: bool = True
    spx_feed_available: bool = True


@dataclass(frozen=True)
class LifecycleStateV1:
    contract: ContractIdentityV1
    position_snapshot_version: str
    entry_bid_micros: int
    running_max_bid_micros: int
    current_bid_micros: int
    current_ask_micros: int
    opened_at_utc: str
    feature_contract_version: str
    feature_snapshot_sha256: str


class DeterministicGovernor:
    """The sole producer of a valid GovernorDecisionV1 authorization token."""

    def __init__(self, config: GovernorConfigV1 | None = None) -> None:
        self.config = config or GovernorConfigV1()

    def evaluate(
        self,
        intent: ExecutionIntentV1,
        *,
        broker_state: BrokerStateSnapshotV1,
        feed_health: FeedHealthV1,
        now_utc: str,
    ) -> GovernorDecisionV1:
        now = _timestamp(now_utc)
        reasons: list[str] = []
        forced_flat = intent.origin == "RISK_GOVERNOR" and intent.decision.urgency == "FORCED_FLAT"
        if broker_state.connectivity != "CONNECTED":
            reasons.append("BROKER_NOT_CONNECTED")
        if intent.decision.quantity > self.config.max_quantity:
            reasons.append("QUANTITY_LIMIT")
        if (
            broker_state.daily_pnl_micros <= -self.config.max_daily_loss_micros
            and (
                self.config.daily_loss_blocks_close
                or intent.decision.position_effect == "OPEN"
            )
        ):
            reasons.append("DAILY_LOSS_LIMIT")
        if now > _timestamp(intent.clocks.valid_until_utc):
            reasons.append("INTENT_EXPIRED")
        if _age_ms(now, _timestamp(intent.clocks.decision_available_at_utc)) > self.config.max_intent_age_ms:
            reasons.append("INTENT_STALE")
        if not feed_health.option_feed_available and not forced_flat:
            reasons.append("OPTION_FEED_UNAVAILABLE")
        if not feed_health.spx_feed_available and intent.decision.position_effect == "OPEN":
            reasons.append("SPX_FEED_UNAVAILABLE")
        if (
            _age_ms(now, _timestamp(feed_health.option_received_timestamp_utc)) > self.config.max_feed_age_ms
            and not forced_flat
        ):
            reasons.append("OPTION_FEED_STALE")
        if intent.decision.position_effect == "OPEN" and _age_ms(now, _timestamp(feed_health.spx_received_timestamp_utc)) > self.config.max_feed_age_ms:
            reasons.append("SPX_FEED_STALE")

        positions = {position.osi_symbol: position for position in broker_state.open_positions}
        precondition = intent.state_precondition
        if precondition.position_snapshot_version != broker_state.snapshot_version:
            reasons.append("POSITION_SNAPSHOT_VERSION_MISMATCH")
        if precondition.expected_position == "FLAT":
            if positions:
                reasons.append("EXPECTED_FLAT")
            if len(positions) >= self.config.max_open_positions:
                reasons.append("POSITION_LIMIT")
            required = intent.price_budget.hard_limit_micros * intent.decision.quantity * intent.contract.multiplier
            if required > broker_state.available_funds_micros:
                reasons.append("INSUFFICIENT_FUNDS")
        else:
            if precondition.held_osi_symbol not in positions:
                reasons.append("EXPECTED_HELD_POSITION_MISSING")
            if precondition.held_osi_symbol != intent.contract.osi_symbol:
                reasons.append("HELD_CONTRACT_MISMATCH")
        if (
            forced_flat
            and intent.price_budget.reference_vendor
            not in self.config.allowed_forced_flat_reference_vendors
        ):
            reasons.append("FORCED_FLAT_REFERENCE_VENDOR_NOT_ALLOWED")
        if forced_flat and intent.producer.component_version != "pathd.deterministic-governor.v1":
            reasons.append("FORCED_FLAT_REQUIRES_GOVERNOR_PRODUCER")
        disposition = "ALLOW" if not reasons else "BLOCK"
        return GovernorDecisionV1.create(
            intent_id=intent.intent_id,
            disposition=disposition,
            reason_codes=reasons or ("ALL_GOVERNOR_CHECKS_PASSED",),
            evaluated_at_utc=_utc_iso(now),
            broker_state_version=broker_state.snapshot_version,
        )

    def lifecycle_exit_intent(
        self,
        state: LifecycleStateV1,
        *,
        feed_health: FeedHealthV1,
        now_utc: str,
    ) -> ExecutionIntentV1 | None:
        """Apply only the preregistered upward-floor/giveback/time/feed-loss law."""
        now = _timestamp(now_utc)
        running_max = max(state.entry_bid_micros, state.running_max_bid_micros, state.current_bid_micros)
        floor = max(
            running_max * self.config.floor_fraction_micros // 1_000_000,
            max(0, running_max - self.config.max_giveback_micros),
        )
        feed_lost = (
            not feed_health.option_feed_available
            or _age_ms(now, _timestamp(feed_health.option_received_timestamp_utc)) > self.config.max_feed_age_ms
        )
        if feed_lost:
            reason, urgency = "HOLDING_FEED_LOSS_FORCED_FLAT", "FORCED_FLAT"
        elif state.current_bid_micros <= floor:
            reason, urgency = "GOVERNOR_UPWARD_FLOOR", "PROTECTIVE_EXIT"
        elif now >= _timestamp(state.opened_at_utc) + timedelta(seconds=self.config.max_hold_seconds):
            reason, urgency = "GOVERNOR_TIME_STOP", "PROTECTIVE_EXIT"
        else:
            return None
        emitted = _utc_iso(now)
        valid_until = _utc_iso(now + timedelta(seconds=2))
        contract = state.contract
        return ExecutionIntentV1.create(
            trace_id=f"governor-{state.feature_snapshot_sha256[7:23]}",
            parent_intent_id=None,
            origin="RISK_GOVERNOR" if urgency == "FORCED_FLAT" else "DETERMINISTIC_EXIT",
            producer=ProducerIdentityV1(
                component_version="pathd.deterministic-governor.v1",
                strategy_id="pathd.preregistered-floor-time.v1",
                artifact_sha256=None,
                feature_contract_version=state.feature_contract_version,
                feature_snapshot_sha256=state.feature_snapshot_sha256,
            ),
            decision=DecisionDirectiveV1(
                action="CLOSE_LONG", side="SELL", position_effect="CLOSE", urgency=urgency,
                quantity=1, reason_code=reason,
            ),
            contract=contract,
            price_budget=PriceBudgetV1(
                unit="USD_OPTION_PRICE_MICROS", reference_vendor="IBKR_SAFETY",
                reference_bid_micros=state.current_bid_micros, reference_ask_micros=state.current_ask_micros,
                max_adverse_move_micros=self.config.forced_flat_adverse_move_micros,
                hard_limit_micros=max(0, state.current_bid_micros - self.config.forced_flat_adverse_move_micros),
            ),
            clocks=IntentClocksV1(
                decision_clock="received_timestamp_utc", event_interval_end_utc=feed_health.option_received_timestamp_utc,
                option_received_watermark_utc=feed_health.option_received_timestamp_utc,
                spx_received_watermark_utc=feed_health.spx_received_timestamp_utc,
                decision_available_at_utc=emitted, model_started_at_utc=emitted,
                model_finished_at_utc=emitted, intent_emitted_at_utc=emitted, valid_until_utc=valid_until,
            ),
            state_precondition=PositionPreconditionV1(
                expected_position="LONG_ONE", position_snapshot_version=state.position_snapshot_version,
                held_osi_symbol=contract.osi_symbol,
            ),
            execution_profile_version="pathd.governed-limit-profile.v1",
        )

    def forced_flat_intent(
        self,
        state: LifecycleStateV1,
        /,
        *,
        feed_health: FeedHealthV1,
        now_utc: str,
        reference_vendor: str,
        reference_bid_micros: int,
        reference_ask_micros: int,
        reason_code: str,
    ) -> ExecutionIntentV1:
        """Create a governor-owned terminal close from one fresh actionable BBO."""

        now = _timestamp(now_utc)
        if reason_code != "PATHD_RESEARCH_TERMINAL":
            raise ValueError("forced-flat reason is not the frozen research terminal")
        if reference_vendor not in self.config.allowed_forced_flat_reference_vendors:
            raise ValueError("forced-flat reference vendor is not allowed")
        if not feed_health.option_feed_available:
            raise ValueError("forced-flat option feed is unavailable")
        option_watermark = _timestamp(feed_health.option_received_timestamp_utc)
        if option_watermark > now:
            raise ValueError("forced-flat option watermark is in the future")
        if _age_ms(now, option_watermark) > self.config.max_feed_age_ms:
            raise ValueError("forced-flat option BBO is stale")
        if (
            type(reference_bid_micros) is not int
            or type(reference_ask_micros) is not int
            or reference_bid_micros <= 0
            or reference_ask_micros <= 0
            or reference_bid_micros >= reference_ask_micros
        ):
            raise ValueError("forced-flat BBO must be positive, unlocked, and uncrossed")

        adverse_move = 50_000 if reference_bid_micros < 3_000_000 else 100_000
        emitted = _utc_iso(now)
        valid_until = emitted
        contract = state.contract
        return ExecutionIntentV1.create(
            trace_id=f"governor-terminal-{state.feature_snapshot_sha256[7:23]}",
            parent_intent_id=None,
            origin="RISK_GOVERNOR",
            producer=ProducerIdentityV1(
                component_version="pathd.deterministic-governor.v1",
                strategy_id="pathd.research-terminal-flat.v1",
                artifact_sha256=None,
                feature_contract_version=state.feature_contract_version,
                feature_snapshot_sha256=state.feature_snapshot_sha256,
            ),
            decision=DecisionDirectiveV1(
                action="CLOSE_LONG",
                side="SELL",
                position_effect="CLOSE",
                urgency="FORCED_FLAT",
                quantity=1,
                reason_code=reason_code,
            ),
            contract=contract,
            price_budget=PriceBudgetV1(
                unit="USD_OPTION_PRICE_MICROS",
                reference_vendor=reference_vendor,
                reference_bid_micros=reference_bid_micros,
                reference_ask_micros=reference_ask_micros,
                max_adverse_move_micros=adverse_move,
                hard_limit_micros=max(0, reference_bid_micros - adverse_move),
            ),
            clocks=IntentClocksV1(
                decision_clock="received_timestamp_utc",
                event_interval_end_utc=feed_health.option_received_timestamp_utc,
                option_received_watermark_utc=feed_health.option_received_timestamp_utc,
                spx_received_watermark_utc=feed_health.spx_received_timestamp_utc,
                decision_available_at_utc=emitted,
                model_started_at_utc=emitted,
                model_finished_at_utc=emitted,
                intent_emitted_at_utc=emitted,
                valid_until_utc=valid_until,
            ),
            state_precondition=PositionPreconditionV1(
                expected_position="LONG_ONE",
                position_snapshot_version=state.position_snapshot_version,
                held_osi_symbol=contract.osi_symbol,
            ),
            execution_profile_version="pathd.governed-limit-profile.v1",
        )


def fake_broker_state(
    *,
    captured_at_utc: str,
    snapshot_version: str = "fake-state-1",
    available_funds_micros: int = 1_000_000_000,
    daily_pnl_micros: int = 0,
    positions: tuple[PositionV1, ...] = (),
    connectivity: str = "CONNECTED",
) -> BrokerStateSnapshotV1:
    return BrokerStateSnapshotV1(
        schema_version="pathd.broker_state_snapshot.v1", snapshot_version=snapshot_version,
        captured_at_utc=captured_at_utc, account_id_redacted="FAKE-***",
        available_funds_micros=available_funds_micros, daily_pnl_micros=daily_pnl_micros,
        open_positions=positions, connectivity=connectivity, source="FAKE_GATEWAY",
    )


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _age_ms(later: datetime, earlier: datetime) -> int:
    return max(0, int((later - earlier).total_seconds() * 1000))
