"""Hash-chained one-account replay primitives for quarantined Path-D research.

The journal is the economic source of truth.  Summary fields are always rebuilt
from its validated transitions; caller-authored PnL or PASS flags are never inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from v4.path_d.contracts import (
    BrokerStateSnapshotV1,
    ContractIdentityV1,
    DecisionDirectiveV1,
    ExecutionEventV1,
    ExecutionIntentV1,
    IntentClocksV1,
    PositionPreconditionV1,
    PositionV1,
    PriceBudgetV1,
    ProducerIdentityV1,
)
from v4.path_d.execution.research_fill_law import (
    ResearchFillLawV1,
    ResearchQuoteV1,
    evaluate_research_arrival,
    option_tick_micros,
    research_fill_law_from_preregistration,
    seal_research_quote,
)
from v4.path_d.execution.simulated import (
    ExecutionScenario,
    SimulatedExecutor,
    SimulatedQuote,
    VirtualMonotonicClock,
)
from v4.path_d.risk.governor import (
    DeterministicGovernor,
    FeedHealthV1,
    GovernorConfigV1,
    LifecycleStateV1,
)
from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_dataset import (
    EntryEvidenceDatasetV1,
    EntryExampleV1,
    VerifiedOfficialSpxRowV1,
    VerifiedResearchQuoteRowV1,
    read_verified_official_spx_row,
    read_verified_research_quote_row,
    validate_entry_evidence_dataset,
)
from v4.research.pathd_entry_models import EntryActionDecisionV1


assert_entry_evidence_authorization_current = (
    prereg.assert_entry_evidence_authorization_current
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _jsonable(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite replay value")
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _plain(value: Any) -> Any:
    return _jsonable(value)


def _exact_hash(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} is not a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} is not a lowercase SHA-256") from exc
    return value


def _utc_from_ns(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("research clock must be a nonnegative integer nanosecond")
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if nanoseconds:
        return f"{base}.{nanoseconds:09d}Z"
    return base + "Z"


def _datetime_from_ns(value: int) -> datetime:
    if type(value) is not int or value < 0 or value % 1_000 != 0:
        raise ValueError("simulator clock must be an exact microsecond boundary")
    return datetime.fromtimestamp(value // 1_000_000_000, tz=timezone.utc) + timedelta(
        microseconds=(value % 1_000_000_000) // 1_000
    )


def _terminal_time_ns(session: str) -> int:
    try:
        local = datetime.fromisoformat(f"{session}T15:55:00").replace(
            tzinfo=ZoneInfo("America/New_York")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("research session is not an ISO date") from exc
    return int(local.astimezone(timezone.utc).timestamp()) * 1_000_000_000


def _validate_fill_law(law: Any) -> ResearchFillLawV1:
    expected = research_fill_law_from_preregistration(
        prereg.preregistration_payload()[0]
    )
    if type(law) is not ResearchFillLawV1 or law != expected:
        raise ValueError("research replay fill-law drift")
    return law


def _validate_dataset_authority(
    dataset: Any, authorization: Any
) -> tuple[Any, EntryEvidenceDatasetV1, str]:
    current = assert_entry_evidence_authorization_current(authorization)
    if type(dataset) is not EntryEvidenceDatasetV1:
        raise TypeError("research replay requires EntryEvidenceDatasetV1")
    sealed = validate_entry_evidence_dataset(dataset, authorization=current)
    authorization_sha256 = prereg.stable_hash(current.to_dict())
    if sealed.authorization_sha256 != authorization_sha256:
        raise ValueError("research replay dataset authority drift")
    return current, sealed, authorization_sha256


def _exact_dataset_example(
    example: Any, *, dataset: EntryEvidenceDatasetV1
) -> EntryExampleV1:
    if type(example) is not EntryExampleV1:
        raise TypeError("research replay example is mistyped")
    matches = [
        row
        for row in dataset.examples
        if row.canonical_sha256() == example.canonical_sha256()
    ]
    if len(matches) != 1 or matches[0] is not example:
        raise ValueError("research replay example is not the exact sealed member")
    return example


def _session_source_receipt(
    dataset: EntryEvidenceDatasetV1, *, session: str
) -> dict[str, Any]:
    matches = [row for row in dataset.source_receipts if row.get("session") == session]
    if len(matches) != 1:
        raise ValueError("research replay session source receipt drift")
    receipt = matches[0]
    # validate_entry_evidence_dataset has already checked the complete receipt,
    # but keep the two intentionally different roots explicit here.  Action
    # facts bind the non-circular source-file root; verified selectors bind the
    # enclosing session receipt, whose digest also commits the example hashes.
    if (
        type(receipt) is not dict
        or receipt.get("source_files_sha256")
        != prereg.stable_hash(receipt.get("source_files"))
        or receipt.get("receipt_sha256")
        != prereg.stable_hash(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise ValueError("research replay session source receipt seal drift")
    return receipt


def _validate_selected_source_binding(
    row: Any,
    *,
    dataset: EntryEvidenceDatasetV1,
    session: str,
) -> dict[str, Any]:
    receipt = _session_source_receipt(dataset, session=session)
    files = {
        item["relative_path"]: item
        for item in receipt["source_files"]
        if type(item) is dict and "relative_path" in item
    }
    source = files.get(row.source_relative_path)
    if (
        row.source_receipt_sha256 != receipt["receipt_sha256"]
        or type(source) is not dict
        or source.get("sha256") != row.source_file_sha256
    ):
        raise ValueError("verified selector source is outside the sealed session receipt")
    return receipt


def _decision_semantic(decision: EntryActionDecisionV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(decision, field.name))
        for field in fields(decision)
        if field.name != "decision_sha256"
    }


def _validate_action_decision(
    decision: Any,
    *,
    dataset: EntryEvidenceDatasetV1,
    authorization_sha256: str,
    example: EntryExampleV1,
) -> EntryActionDecisionV1:
    if type(decision) is not EntryActionDecisionV1:
        raise TypeError("research action decision is mistyped")
    physical = tuple(bool(value) for value in example.model_input.physical_action_mask)
    dynamic = tuple(decision.dynamic_account_mask)
    combined = tuple(decision.combined_action_mask)
    terminal_ns = _terminal_time_ns(example.model_input.session)
    arrival_ns = example.model_input.decision_time_ns + 60_000_000_000
    expected_horizons = () if arrival_ns >= terminal_ns else tuple(
        [
            f"h{minutes}"
            for minutes in (10, 20, 45, 90)
            if arrival_ns + minutes * 60_000_000_000 <= terminal_ns
        ]
        + ["remaining_session"]
    )
    if (
        decision.schema_version != decision.SCHEMA_VERSION
        or decision.decision_sha256 != prereg.stable_hash(_decision_semantic(decision))
        or decision.authorization_sha256 != authorization_sha256
        or decision.dataset_sha256 != dataset.dataset_sha256
        or decision.example_sha256 != example.canonical_sha256()
        or decision.model_input_sha256
        != example.model_input.canonical_sha256()
        or len(decision.physical_action_mask) != 42
        or len(decision.dynamic_account_mask) != 42
        or len(decision.combined_action_mask) != 42
        or any(type(value) is not bool for value in decision.physical_action_mask)
        or any(type(value) is not bool for value in dynamic)
        or any(type(value) is not bool for value in combined)
        or tuple(decision.physical_action_mask) != physical
        or combined
        != tuple(left and right for left, right in zip(physical, dynamic, strict=True))
        or tuple(decision.available_horizons) != expected_horizons
        or decision.fill_law_hash
        != prereg.preregistration_payload()[0]["fill_law"]["fill_law_hash"]
    ):
        raise ValueError("research action decision identity drift")
    if decision.action == "WAIT":
        if any(
            value is not None
            for value in (
                decision.selected_action_index,
                decision.selected_source_neutral_contract_id,
                decision.selected_contract,
                decision.reference_bid_micros,
                decision.reference_ask_micros,
                decision.buy_hard_limit_micros,
            )
        ) or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in (
                decision.mean_lcb_dollars,
                decision.mean_lcb_return,
                decision.q10_dollars,
                decision.q10_return,
            )
        ):
            raise ValueError("WAIT decision carries an action payload")
        return decision
    if decision.action != "ENTER" or type(decision.selected_action_index) is not int:
        raise ValueError("research action decision is neither WAIT nor ENTER")
    index = decision.selected_action_index
    if not 0 <= index < len(example.action_execution_facts):
        raise ValueError("research action index is outside the sealed ladder")
    fact = example.action_execution_facts[index]
    fact_semantic = {
        field.name: _plain(getattr(fact, field.name))
        for field in fields(fact)
        if field.name != "fact_sha256"
    }
    if fact.fact_sha256 != prereg.stable_hash(fact_semantic):
        raise ValueError("selected execution fact seal drift")
    expected_limit = fact.ask_micros + option_tick_micros(fact.ask_micros)
    fact_contract = (
        fact.contract
        if type(fact.contract) is ContractIdentityV1
        else ContractIdentityV1.from_dict(fact.contract)
    )
    decision_contract = (
        decision.selected_contract
        if type(decision.selected_contract) is ContractIdentityV1
        else ContractIdentityV1.from_dict(decision.selected_contract)
    )
    if (
        decision.selected_source_neutral_contract_id
        != fact.source_neutral_contract_id
        or decision_contract != fact_contract
        or decision.reference_bid_micros != fact.bid_micros
        or decision.reference_ask_micros != fact.ask_micros
        or decision.buy_hard_limit_micros != expected_limit
        or fact.physical_eligible is not True
        or decision.combined_action_mask[index] is not True
        or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in (
                decision.mean_lcb_dollars,
                decision.mean_lcb_return,
                decision.q10_dollars,
                decision.q10_return,
            )
        )
    ):
        raise ValueError("ENTER decision differs from its sealed execution fact")
    return decision


def _verified_spx_row(
    row: Any,
    *,
    session: str,
    at_or_before_ns: int,
    maximum_age_ms: int,
) -> VerifiedOfficialSpxRowV1:
    if type(row) is not VerifiedOfficialSpxRowV1:
        raise TypeError("research context SPX row is mistyped")
    semantic = {
        field.name: _plain(getattr(row, field.name))
        for field in fields(row)
        if field.name != "record_sha256"
    }
    if (
        row.schema_version != row.SCHEMA_VERSION
        or row.source_vendor != "THETADATA"
        or row.session != session
        or row.query_at_or_before_ns != at_or_before_ns
        or row.query_maximum_age_ms != maximum_age_ms
        or row.available_at_ns > at_or_before_ns
        or at_or_before_ns - row.available_at_ns > maximum_age_ms * 1_000_000
        or not (row.represented_interval_end_ns <= row.ts_recv_ns <= row.available_at_ns)
        or row.record_sha256 != prereg.stable_hash(semantic)
        or type(row.eligible_row_count) is not int
        or row.eligible_row_count < 1
        or tuple(row.selection_key)
        != (
            row.available_at_ns,
            row.ts_recv_ns,
            row.represented_interval_end_ns,
            row.source_relative_path,
            row.row_group,
            row.row_index,
        )
    ):
        raise ValueError("research context SPX row drift")
    _exact_hash(row.selection_proof_sha256, name="SPX selection proof")
    return row


def _context_semantic(context: ResearchDecisionContextV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(context, field.name))
        for field in fields(context)
        if field.name != "context_sha256"
    }


def _validate_context_seal(context: Any) -> ResearchDecisionContextV1:
    if type(context) is not ResearchDecisionContextV1:
        raise TypeError("research decision context is mistyped")
    if (
        context.schema_version != context.SCHEMA_VERSION
        or context.context_sha256 != prereg.stable_hash(_context_semantic(context))
        or type(context.decision_quote) is not ResearchQuoteV1
        or type(context.official_spx_row) is not VerifiedOfficialSpxRowV1
        or context.session != context.decision_quote.session
        or context.contract != context.decision_quote.contract
        or context.option_source_receipt_sha256
        != context.decision_quote.source_receipt_sha256
        or context.spx_source_receipt_sha256
        != context.official_spx_row.source_receipt_sha256
        or not (
            context.event_interval_end_ns
            <= context.option_watermark_ns
            <= context.decision_time_ns
        )
        or context.spx_watermark_ns > context.decision_time_ns
    ):
        raise ValueError("research decision context seal drift")
    return context


@dataclass(frozen=True)
class EntryActionCalibrationObservationV1:
    SCHEMA_VERSION = "pathd.entry_action_calibration_observation.v1"

    schema_version: str
    outer_fold: int
    action: str
    session: str
    trajectory_or_episode_id: str
    predicted_mean_micros: int
    predicted_lower_micros: int
    realized_micros: int
    source_policy_evaluation_sha256: str
    source_transition_sha256: str
    observation_sha256: str


@dataclass(frozen=True)
class EntryTime300TrajectoryEvidenceV1:
    SCHEMA_VERSION = "pathd.entry_time300_trajectory_evidence.v1"

    schema_version: str
    outer_fold: int
    session: str
    trajectory_id: str
    source_policy_evaluation_sha256: str
    entry_fill_transition_sha256: str
    h300_mark_receipt_sha256: str
    evidence_sha256: str


@dataclass(frozen=True)
class EntrySurvivalAuditRecordV1:
    SCHEMA_VERSION = "pathd.entry_survival_audit_record.v1"

    schema_version: str
    outer_fold: int
    session: str
    policy_id: str
    overlap: int
    unaffordable_fill: int
    quantity_breach: int
    nonflat_close: int
    duplicate_terminal_consumption: int
    d48_breach: int
    d49_breach: int
    source_transition_sha256s: tuple[str, ...]
    record_sha256: str


@dataclass(frozen=True)
class EntryActionObservationV1:
    SCHEMA_VERSION = "pathd.entry_action_observation.v1"

    schema_version: str
    authorization_sha256: str
    dataset_sha256: str
    session: str
    trajectory_id: str
    episode_id: str
    example_sha256: str
    decision_sha256: str
    action: str
    predicted_mean_cash_micros: int
    predicted_lower_cash_micros: int
    realized_value_cash_micros: int
    coverage: bool
    filled: bool
    execution_result_sha256: str | None
    observation_sha256: str


@dataclass(frozen=True)
class EntryFrameCoverageV1:
    SCHEMA_VERSION = "pathd.entry_frame_coverage.v1"

    schema_version: str
    authorization_sha256: str
    dataset_sha256: str
    session: str
    decision_time_ns: int
    example_sha256: str
    disposition: str
    reason_codes: tuple[str, ...]
    prior_journal_root_sha256: str
    coverage_sha256: str


@dataclass(frozen=True)
class ResearchExecutionResultV1:
    SCHEMA_VERSION = "pathd.research_execution_result.v1"

    schema_version: str
    prepared_execution_sha256: str
    decision_quote_sha256: str
    arrival_quote_sha256: str | None
    source_receipt_sha256: str
    delay_ms: int
    events: tuple[Any, ...]
    final_state: str
    filled_quantity: int
    fill_price_micros: int | None
    fee_micros: int
    cash_delta_micros: int
    position_before: Any
    position_after: Any
    prior_ledger_sha256: str
    result_ledger: Any
    result_sha256: str


@dataclass(frozen=True)
class ResearchDecisionContextV1:
    SCHEMA_VERSION = "pathd.research_decision_context.v1"

    schema_version: str
    intent_kind: str
    session: str
    contract: ContractIdentityV1
    decision_time_ns: int
    event_interval_end_ns: int
    option_watermark_ns: int
    spx_watermark_ns: int
    authorization_sha256: str
    dataset_sha256: str
    entry_example_sha256: str | None
    entry_action_decision_sha256: str | None
    option_source_receipt_sha256: str
    spx_source_receipt_sha256: str
    decision_quote: Any
    official_spx_row: Any
    context_sha256: str


@dataclass(frozen=True)
class ResearchLedgerStateV1:
    SCHEMA_VERSION = "pathd.research_ledger_state.v1"

    schema_version: str
    policy_id: str
    fee_path: int
    authorization_sha256: str
    sessions_sha256_newline: str
    account_scope_sha256: str
    account_session_index: int
    session_index: int
    session: str
    session_start_equity_micros: int
    cash_micros: int
    realized_session_pnl_micros: int
    position: Any
    pending_intent_id: str | None
    sequence: int
    last_decision_time_ns: int | None
    prior_transition_sha256: str
    ledger_sha256: str


@dataclass(frozen=True)
class ResearchLedgerControlEventV1:
    SCHEMA_VERSION = "pathd.research_ledger_control_event.v1"

    schema_version: str
    event_kind: str
    event_time_ns: int
    prior_authorization_sha256: str | None
    next_authorization_sha256: str
    prior_session: str | None
    next_session: str
    prior_session_index: int | None
    next_session_index: int
    prior_account_session_index: int | None
    next_account_session_index: int
    intervening_skip_receipt_sha256s: tuple[str, ...]
    reason: str
    event_sha256: str


@dataclass(frozen=True)
class ResearchLedgerTransitionV1:
    SCHEMA_VERSION = "pathd.research_ledger_transition.v1"

    schema_version: str
    transition_kind: str
    event_time_ns: int
    prior_authorization_sha256: str | None
    next_authorization_sha256: str
    prior_ledger_sha256: str
    event_schema_version: str
    event_sha256: str
    event_payload: dict[str, Any]
    next_ledger: ResearchLedgerStateV1
    transition_sha256: str


@dataclass(frozen=True)
class ResearchLedgerJournalV1:
    SCHEMA_VERSION = "pathd.research_ledger_journal.v1"

    schema_version: str
    account_scope_sha256: str
    policy_id: str
    fee_path: int
    transitions: tuple[ResearchLedgerTransitionV1, ...]
    tip_ledger: ResearchLedgerStateV1
    journal_root_sha256: str


@dataclass(frozen=True)
class ResearchPreparedExecutionV1:
    SCHEMA_VERSION = "pathd.research_prepared_execution.v1"

    schema_version: str
    intent: Any
    broker_state: Any
    feed_health: Any
    authorization: Any
    decision_context: ResearchDecisionContextV1
    decision_context_sha256: str
    decision_quote_sha256: str
    source_receipt_sha256: str
    prior_journal: ResearchLedgerJournalV1
    prior_journal_root_sha256: str
    governor_config_sha256: str
    arrival_base_time_ns: int
    prepared_sha256: str


@dataclass(frozen=True)
class ResearchResultEnvelopeV1:
    SCHEMA_VERSION = "pathd.research_result_envelope.v1"

    schema_version: str
    authorization_sha256: str
    sessions_sha256_newline: str
    dataset_sha256: str
    preregistration_sha256: str
    plan_sha256: str
    fill_law_hash: str
    quarantine_labels: tuple[str, ...]
    claim_boundary: str
    holdout_caveat: str
    holdout_open_count: int
    payload_sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class NonOrderSafetyEventV1:
    SCHEMA_VERSION = "pathd.research.non_order_safety_event.v1"

    schema_version: str
    event_id: str
    event_type: str
    session: str
    policy_id: str
    fee_path: int
    authorization_sha256: str
    event_time_utc: str
    event_time_ns: int
    source_neutral_contract_id: str
    osi_symbol: str
    state_from: str
    state_to: str
    invalid_reason: str
    provenance: str
    last_actual_option_watermark_utc: str | None
    zero_proceeds_cash_micros: int
    modeled_close_penalty_cash_micros: int
    cash_before_micros: int
    cash_after_micros: int
    equity_before_micros: int
    equity_after_micros: int
    pending_intent_id_consumed: str | None
    fill_law_hash: str
    prior_ledger_sha256: str
    result_ledger: ResearchLedgerStateV1
    event_sha256: str


@dataclass(frozen=True)
class ResearchExecutionOutcomeV1:
    SCHEMA_VERSION = "pathd.research_execution_outcome.v1"

    schema_version: str
    execution_result: ResearchExecutionResultV1
    result_journal: ResearchLedgerJournalV1
    outcome_sha256: str


def _ledger_semantic(state: ResearchLedgerStateV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(state, field.name))
        for field in fields(state)
        if field.name != "ledger_sha256"
    }


def _seal_ledger(**values: Any) -> ResearchLedgerStateV1:
    semantic = {"schema_version": ResearchLedgerStateV1.SCHEMA_VERSION, **values}
    return ResearchLedgerStateV1(**semantic, ledger_sha256=prereg.stable_hash(_plain(semantic)))


def _event_semantic(event: ResearchLedgerControlEventV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(event, field.name))
        for field in fields(event)
        if field.name != "event_sha256"
    }


def _seal_control_event(**values: Any) -> ResearchLedgerControlEventV1:
    semantic = {"schema_version": ResearchLedgerControlEventV1.SCHEMA_VERSION, **values}
    return ResearchLedgerControlEventV1(
        **semantic, event_sha256=prereg.stable_hash(_plain(semantic))
    )


def _transition_semantic(transition: ResearchLedgerTransitionV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(transition, field.name))
        for field in fields(transition)
        if field.name != "transition_sha256"
    }


def _seal_transition(
    *,
    transition_kind: str,
    event_time_ns: int,
    prior_authorization_sha256: str | None,
    next_authorization_sha256: str,
    prior_ledger_sha256: str,
    event_schema_version: str,
    event_sha256: str,
    event_payload: dict[str, Any],
    next_ledger: ResearchLedgerStateV1,
) -> ResearchLedgerTransitionV1:
    values = {
        "schema_version": ResearchLedgerTransitionV1.SCHEMA_VERSION,
        "transition_kind": transition_kind,
        "event_time_ns": event_time_ns,
        "prior_authorization_sha256": prior_authorization_sha256,
        "next_authorization_sha256": next_authorization_sha256,
        "prior_ledger_sha256": prior_ledger_sha256,
        "event_schema_version": event_schema_version,
        "event_sha256": event_sha256,
        "event_payload": event_payload,
        "next_ledger": next_ledger,
    }
    return ResearchLedgerTransitionV1(
        **values, transition_sha256=prereg.stable_hash(_plain(values))
    )


def _journal_root(
    *,
    account_scope_sha256: str,
    policy_id: str,
    fee_path: int,
    transitions: tuple[ResearchLedgerTransitionV1, ...],
    tip_ledger: ResearchLedgerStateV1,
) -> str:
    return prereg.stable_hash(
        {
            "schema_version": ResearchLedgerJournalV1.SCHEMA_VERSION,
            "account_scope_sha256": account_scope_sha256,
            "policy_id": policy_id,
            "fee_path": fee_path,
            "transition_sha256s": [row.transition_sha256 for row in transitions],
            "tip_ledger_sha256": tip_ledger.ledger_sha256,
        }
    )


def _seal_journal(
    *,
    account_scope_sha256: str,
    policy_id: str,
    fee_path: int,
    transitions: tuple[ResearchLedgerTransitionV1, ...],
    tip_ledger: ResearchLedgerStateV1,
) -> ResearchLedgerJournalV1:
    return ResearchLedgerJournalV1(
        schema_version=ResearchLedgerJournalV1.SCHEMA_VERSION,
        account_scope_sha256=account_scope_sha256,
        policy_id=policy_id,
        fee_path=fee_path,
        transitions=transitions,
        tip_ledger=tip_ledger,
        journal_root_sha256=_journal_root(
            account_scope_sha256=account_scope_sha256,
            policy_id=policy_id,
            fee_path=fee_path,
            transitions=transitions,
            tip_ledger=tip_ledger,
        ),
    )


def _coerce_ledger(value: Any) -> ResearchLedgerStateV1:
    if type(value) is ResearchLedgerStateV1:
        return value
    if type(value) is dict:
        return ResearchLedgerStateV1(**value)
    raise TypeError("journal ledger is mistyped")


def _coerce_transition(value: Any) -> ResearchLedgerTransitionV1:
    if type(value) is ResearchLedgerTransitionV1:
        return value
    if type(value) is dict:
        row = dict(value)
        row["next_ledger"] = _coerce_ledger(row["next_ledger"])
        return ResearchLedgerTransitionV1(**row)
    raise TypeError("journal transition is mistyped")


def _coerce_journal(value: Any) -> ResearchLedgerJournalV1:
    if type(value) is ResearchLedgerJournalV1:
        return value
    if type(value) is dict:
        row = dict(value)
        row["transitions"] = tuple(_coerce_transition(item) for item in row["transitions"])
        row["tip_ledger"] = _coerce_ledger(row["tip_ledger"])
        return ResearchLedgerJournalV1(**row)
    raise TypeError("journal is mistyped")


def _nested_account_scope_sha256(
    *, policy_id: str, fee_path: int, outer_fold: int, assignments: dict[str, Any]
) -> str:
    if policy_id not in {"HGB", "NEURAL"}:
        raise ValueError("nested replay account policy must be HGB or NEURAL")
    rows = assignments["folds"][outer_fold - 1]["inner_forward_folds"][
        "scored_forward_folds"
    ]
    valid_rows = [row for row in rows if row.get("calibration_valid") is True]
    nested_sessions = tuple(
        session for row in valid_rows for session in row["validation"]
    )
    if not valid_rows or not nested_sessions or len(nested_sessions) != len(
        set(nested_sessions)
    ):
        raise ValueError("nested replay account scope is malformed")
    return prereg.stable_hash(
        {
            "policy_id": policy_id,
            "fee_path": fee_path,
            "outer_fold": outer_fold,
            "ordered_calibration_valid_nested_validation_sessions": list(
                nested_sessions
            ),
        }
    )


def genesis_research_ledger_journal(
    authorization: Any, /, *, policy_id: str, fee_path: int
) -> ResearchLedgerJournalV1:
    current = assert_entry_evidence_authorization_current(authorization)
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id must be nonempty")
    if type(fee_path) is not int or fee_path not in (3, 4):
        raise ValueError("fee_path must be 3 or 4")
    sessions = tuple(current.sessions)
    if not sessions:
        raise ValueError("evidence authorization has no sessions")
    authorization_sha = prereg.stable_hash(current.to_dict())
    if current.role == "nested_validation":
        assignments = prereg.session_assignments()
        rows = assignments["folds"][current.outer_fold - 1][
            "inner_forward_folds"
        ]["scored_forward_folds"]
        valid_rows = [row for row in rows if row.get("calibration_valid") is True]
        if not valid_rows or current.inner_fold != valid_rows[0]["inner_fold"]:
            raise ValueError(
                "nested account genesis is allowed only at the first calibration-valid block"
            )
        account_scope = _nested_account_scope_sha256(
            policy_id=policy_id,
            fee_path=fee_path,
            outer_fold=current.outer_fold,
            assignments=assignments,
        )
    else:
        account_scope = prereg.stable_hash(
            {
                "policy_id": policy_id,
                "fee_path": fee_path,
                "outer_fold": current.outer_fold,
                "inner_fold": current.inner_fold,
                "sessions_sha256_newline": current.sessions_sha256_newline,
            }
        )
    state = _seal_ledger(
        policy_id=policy_id,
        fee_path=fee_path,
        authorization_sha256=authorization_sha,
        sessions_sha256_newline=current.sessions_sha256_newline,
        account_scope_sha256=account_scope,
        account_session_index=0,
        session_index=0,
        session=sessions[0],
        session_start_equity_micros=10_000_000_000,
        cash_micros=10_000_000_000,
        realized_session_pnl_micros=0,
        position=None,
        pending_intent_id=None,
        sequence=0,
        last_decision_time_ns=None,
        prior_transition_sha256="0" * 64,
    )
    event = _seal_control_event(
        event_kind="GENESIS",
        event_time_ns=0,
        prior_authorization_sha256=None,
        next_authorization_sha256=authorization_sha,
        prior_session=None,
        next_session=sessions[0],
        prior_session_index=None,
        next_session_index=0,
        prior_account_session_index=None,
        next_account_session_index=0,
        intervening_skip_receipt_sha256s=(),
        reason="PATHD_RESEARCH_ACCOUNT_GENESIS",
    )
    transition = _seal_transition(
        transition_kind="CONTROL",
        event_time_ns=0,
        prior_authorization_sha256=None,
        next_authorization_sha256=authorization_sha,
        prior_ledger_sha256="0" * 64,
        event_schema_version=event.SCHEMA_VERSION,
        event_sha256=event.event_sha256,
        event_payload=_plain(event),
        next_ledger=state,
    )
    return _seal_journal(
        account_scope_sha256=account_scope,
        policy_id=policy_id,
        fee_path=fee_path,
        transitions=(transition,),
        tip_ledger=state,
    )


def _validate_event_payload(transition: ResearchLedgerTransitionV1) -> None:
    payload = transition.event_payload
    if type(payload) is not dict or not payload:
        raise ValueError("journal event payload must be a complete typed mapping")
    schema = transition.event_schema_version
    hash_fields = {
        ResearchLedgerControlEventV1.SCHEMA_VERSION: "event_sha256",
        EntryFrameCoverageV1.SCHEMA_VERSION: "coverage_sha256",
        EntryActionObservationV1.SCHEMA_VERSION: "observation_sha256",
        EntryActionCalibrationObservationV1.SCHEMA_VERSION: "observation_sha256",
        EntryTime300TrajectoryEvidenceV1.SCHEMA_VERSION: "evidence_sha256",
        EntrySurvivalAuditRecordV1.SCHEMA_VERSION: "record_sha256",
        ResearchExecutionResultV1.SCHEMA_VERSION: "result_sha256",
        NonOrderSafetyEventV1.SCHEMA_VERSION: "event_sha256",
    }
    if schema == "pathd.entry_action_decision.v1":
        hash_field = "decision_sha256"
    else:
        hash_field = hash_fields.get(schema)
    if hash_field is None or set(payload) == {"forged_hash_only_event"}:
        raise ValueError("unregistered or incomplete journal event schema")
    if payload.get("schema_version") != schema or hash_field not in payload:
        raise ValueError("journal event schema/hash field drift")
    semantic = dict(payload)
    observed = semantic.pop(hash_field)
    if observed != prereg.stable_hash(semantic) or transition.event_sha256 != observed:
        raise ValueError("journal event payload hash drift")


def _validated_execution_events(rows: Any) -> tuple[ExecutionEventV1, ...]:
    if type(rows) not in (tuple, list) or not rows:
        raise ValueError("research execution transcript is empty or mistyped")
    events = tuple(
        row if type(row) is ExecutionEventV1 else ExecutionEventV1.from_dict(row)
        for row in rows
    )
    intent_ids = {row.intent_id for row in events}
    order_ids = {row.order_id for row in events}
    if len(intent_ids) != 1 or len(order_ids) != 1:
        raise ValueError("research execution transcript identity drift")
    for left, right in zip(events, events[1:]):
        if (
            left.state_to != right.state_from
            or left.monotonic_ns > right.monotonic_ns
            or left.event_at_utc > right.event_at_utc
        ):
            raise ValueError("research execution transcript chain drift")
    return events


def _validate_execution_transition(
    *,
    prior: ResearchLedgerStateV1,
    state: ResearchLedgerStateV1,
    payload: dict[str, Any],
) -> None:
    if set(payload) != {field.name for field in fields(ResearchExecutionResultV1)}:
        raise ValueError("research execution result schema drift")
    events = _validated_execution_events(payload["events"])
    filled_quantity = max(row.filled_quantity for row in events)
    fill_prices = [
        row.fill_price_micros
        for row in events
        if row.filled_quantity == filled_quantity and row.fill_price_micros is not None
    ]
    fill_price = fill_prices[-1] if fill_prices else None
    if (
        payload["final_state"] != events[-1].state_to
        or payload["filled_quantity"] != filled_quantity
        or payload["fill_price_micros"] != fill_price
        or payload["prior_ledger_sha256"] != prior.ledger_sha256
        or payload["position_before"] != _plain(prior.position)
        or payload["position_after"] != _plain(state.position)
        or payload["result_ledger"] != _plain(state)
        or state.cash_micros != prior.cash_micros + payload["cash_delta_micros"]
        or state.pending_intent_id is not None
    ):
        raise ValueError("research execution result ledger binding drift")
    fee = payload["fee_micros"]
    cash_delta = payload["cash_delta_micros"]
    if type(fee) is not int or fee < 0 or type(cash_delta) is not int:
        raise TypeError("research execution result economics are mistyped")
    if filled_quantity == 0:
        if (
            fill_price is not None
            or fee != 0
            or cash_delta != 0
            or state.position != prior.position
            or state.realized_session_pnl_micros
            != prior.realized_session_pnl_micros
        ):
            raise ValueError("research no-fill changed account economics")
        return
    if filled_quantity != 1 or type(fill_price) is not int:
        raise ValueError("research replay quantity drift")
    if prior.position is None and type(state.position) is dict:
        if (
            cash_delta != -(fill_price * 100 + fee)
            or state.position.get("entry_fill_price_micros") != fill_price
            or state.position.get("entry_fee_micros") != fee
            or state.realized_session_pnl_micros
            != prior.realized_session_pnl_micros - fee
        ):
            raise ValueError("research BUY economics drift")
    elif type(prior.position) is dict and state.position is None:
        expected_realized = (
            prior.realized_session_pnl_micros
            + (fill_price - prior.position["entry_fill_price_micros"]) * 100
            - fee
        )
        if (
            cash_delta != fill_price * 100 - fee
            or state.realized_session_pnl_micros != expected_realized
        ):
            raise ValueError("research SELL economics drift")
    else:
        raise ValueError("research fill did not perform one legal position transition")


def _non_order_event_id_semantic(payload: dict[str, Any]) -> dict[str, Any]:
    names = (
        "schema_version", "event_type", "session", "policy_id", "fee_path",
        "authorization_sha256", "event_time_utc", "event_time_ns",
        "source_neutral_contract_id", "osi_symbol", "state_from", "state_to",
        "invalid_reason", "provenance", "last_actual_option_watermark_utc",
        "pending_intent_id_consumed", "fill_law_hash", "prior_ledger_sha256",
    )
    return {name: payload[name] for name in names}


def _validate_non_order_safety_transition(
    *,
    prior: ResearchLedgerStateV1,
    state: ResearchLedgerStateV1,
    payload: dict[str, Any],
) -> None:
    if set(payload) != {field.name for field in fields(NonOrderSafetyEventV1)}:
        raise ValueError("non-order safety event schema drift")
    position = prior.position
    if type(position) is not dict:
        raise ValueError("non-order safety event requires a replay-owned long")
    contract_value = position.get("contract")
    contract = (
        contract_value
        if type(contract_value) is ContractIdentityV1
        else ContractIdentityV1.from_dict(contract_value)
    )
    pending = prior.pending_intent_id
    state_from = "EXIT_PENDING" if pending is not None else "LONG_ONE"
    fee = _fee_per_filled_side(prior.fee_path)
    expected_realized = (
        prior.realized_session_pnl_micros
        - position["entry_fill_price_micros"] * contract.multiplier
        - fee
    )
    if (
        payload["event_type"]
        not in {"FEED_LOSS_ZERO_WRITE_DOWN", "TERMINAL_ZERO_WRITE_DOWN"}
        or payload["session"] != prior.session
        or payload["policy_id"] != prior.policy_id
        or payload["fee_path"] != prior.fee_path
        or payload["authorization_sha256"] != prior.authorization_sha256
        or payload["event_time_utc"] != _utc_from_ns(payload["event_time_ns"])
        or payload["source_neutral_contract_id"]
        != position["source_neutral_contract_id"]
        or payload["osi_symbol"] != contract.osi_symbol
        or payload["state_from"] != state_from
        or payload["state_to"] != "FLAT_TERMINAL"
        or type(payload["invalid_reason"]) is not str
        or not payload["invalid_reason"]
        or payload["provenance"]
        != (
            "DERIVED_FEED_LOSS"
            if payload["event_type"] == "FEED_LOSS_ZERO_WRITE_DOWN"
            else "DERIVED_NO_ACTIONABLE_BBO"
        )
        or payload["zero_proceeds_cash_micros"] != 0
        or payload["modeled_close_penalty_cash_micros"] != fee
        or payload["cash_before_micros"] != prior.cash_micros
        or payload["cash_after_micros"] != prior.cash_micros - fee
        or payload["equity_before_micros"] != prior.cash_micros
        or payload["equity_after_micros"] != prior.cash_micros - fee
        or payload["pending_intent_id_consumed"] != pending
        or payload["fill_law_hash"]
        != prereg.preregistration_payload()[0]["fill_law"]["fill_law_hash"]
        or payload["prior_ledger_sha256"] != prior.ledger_sha256
        or payload["event_id"] != prereg.stable_hash(
            _non_order_event_id_semantic(payload)
        )
        or payload["result_ledger"] != _plain(state)
        or state.cash_micros != prior.cash_micros - fee
        or state.realized_session_pnl_micros != expected_realized
        or state.position is not None
        or state.pending_intent_id is not None
        or state.last_decision_time_ns != payload["event_time_ns"]
    ):
        raise ValueError("non-order safety event economics/state drift")
    if (
        payload["event_type"] == "TERMINAL_ZERO_WRITE_DOWN"
        and payload["event_time_ns"] != _terminal_time_ns(prior.session)
    ):
        raise ValueError("terminal zero write-down clock drift")
    if (
        payload["event_type"] == "FEED_LOSS_ZERO_WRITE_DOWN"
        and payload["event_time_ns"] >= _terminal_time_ns(prior.session)
    ):
        raise ValueError("feed-loss write-down did not precede terminal handling")


def _validate_research_ledger_journal_bound(
    journal: Any, /, *, current: Any
) -> ResearchLedgerJournalV1:
    if type(current) is not prereg.FrozenEvidenceAuthorization:
        raise TypeError("journal validation requires exact evidence authorization")
    value = _coerce_journal(journal)
    if value.schema_version != value.SCHEMA_VERSION or not value.transitions:
        raise ValueError("journal schema or transition count drift")
    if value.policy_id != value.tip_ledger.policy_id or value.fee_path != value.tip_ledger.fee_path:
        raise ValueError("journal policy/fee drift")
    authorization_sha = prereg.stable_hash(current.to_dict())
    prior_transition_sha = "0" * 64
    prior_ledger_sha = "0" * 64
    previous_state: ResearchLedgerStateV1 | None = None
    for index, transition in enumerate(value.transitions):
        if transition.schema_version != transition.SCHEMA_VERSION:
            raise ValueError("journal transition schema drift")
        state = transition.next_ledger
        if state.schema_version != state.SCHEMA_VERSION:
            raise ValueError("journal ledger schema drift")
        if state.ledger_sha256 != prereg.stable_hash(_ledger_semantic(state)):
            raise ValueError("journal ledger seal drift")
        if state.policy_id != value.policy_id or state.fee_path != value.fee_path:
            raise ValueError("journal state policy/fee drift")
        if state.account_scope_sha256 != value.account_scope_sha256:
            raise ValueError("journal account-scope drift")
        if state.prior_transition_sha256 != prior_transition_sha:
            raise ValueError("journal ledger prior-transition drift")
        if transition.prior_ledger_sha256 != prior_ledger_sha:
            raise ValueError("journal transition prior-ledger drift")
        if transition.transition_sha256 != prereg.stable_hash(_transition_semantic(transition)):
            raise ValueError("journal transition seal drift")
        if transition.next_authorization_sha256 != state.authorization_sha256:
            raise ValueError("journal transition next-authorization drift")
        _validate_event_payload(transition)
        if index == 0:
            if (
                transition.event_schema_version != ResearchLedgerControlEventV1.SCHEMA_VERSION
                or transition.event_payload.get("event_kind") != "GENESIS"
                or state.sequence != 0
                or state.cash_micros != 10_000_000_000
                or state.session_start_equity_micros != 10_000_000_000
                or state.realized_session_pnl_micros != 0
                or state.position is not None
                or state.pending_intent_id is not None
            ):
                raise ValueError("journal genesis drift")
        else:
            assert previous_state is not None
            if state.sequence != previous_state.sequence + 1:
                raise ValueError("journal sequence drift")
            if transition.prior_authorization_sha256 != previous_state.authorization_sha256:
                raise ValueError("journal authorization chain drift")
            if transition.event_schema_version == ResearchExecutionResultV1.SCHEMA_VERSION:
                _validate_execution_transition(
                    prior=previous_state,
                    state=state,
                    payload=transition.event_payload,
                )
            elif transition.event_schema_version == EntryFrameCoverageV1.SCHEMA_VERSION:
                if (
                    previous_state.last_decision_time_ns is not None
                    and transition.event_payload["decision_time_ns"]
                    < previous_state.last_decision_time_ns
                ):
                    raise ValueError("frame coverage moved the account clock backward")
                allowed = replace(
                    previous_state,
                    sequence=state.sequence,
                    last_decision_time_ns=transition.event_payload["decision_time_ns"],
                    prior_transition_sha256=state.prior_transition_sha256,
                    ledger_sha256=state.ledger_sha256,
                )
                if state != allowed:
                    raise ValueError("frame coverage changed non-clock account state")
            elif transition.event_schema_version in {
                EntryActionDecisionV1.SCHEMA_VERSION,
                EntryActionObservationV1.SCHEMA_VERSION,
            }:
                allowed = replace(
                    previous_state,
                    sequence=state.sequence,
                    prior_transition_sha256=state.prior_transition_sha256,
                    ledger_sha256=state.ledger_sha256,
                )
                if state != allowed:
                    raise ValueError("action event changed account state")
            elif transition.event_schema_version == NonOrderSafetyEventV1.SCHEMA_VERSION:
                _validate_non_order_safety_transition(
                    prior=previous_state,
                    state=state,
                    payload=transition.event_payload,
                )
            elif transition.event_schema_version == ResearchLedgerControlEventV1.SCHEMA_VERSION:
                kind = transition.event_payload.get("event_kind")
                if kind == "SESSION_TERMINAL":
                    allowed = replace(
                        previous_state,
                        sequence=state.sequence,
                        prior_transition_sha256=state.prior_transition_sha256,
                        ledger_sha256=state.ledger_sha256,
                    )
                    if state != allowed or state.position is not None or state.pending_intent_id is not None:
                        raise ValueError("session terminal changed or retained account risk")
                elif kind == "SESSION_ADVANCE":
                    if (
                        state.account_session_index
                        != previous_state.account_session_index + 1
                        or state.session_index != previous_state.session_index + 1
                        or state.session_start_equity_micros != previous_state.cash_micros
                        or state.cash_micros != previous_state.cash_micros
                        or state.realized_session_pnl_micros != 0
                        or state.position is not None
                        or state.pending_intent_id is not None
                        or state.last_decision_time_ns is not None
                    ):
                        raise ValueError("session advance economics/state drift")
                elif kind == "AUTHORIZATION_HANDOFF":
                    payload = transition.event_payload
                    allowed = replace(
                        previous_state,
                        authorization_sha256=payload["next_authorization_sha256"],
                        sessions_sha256_newline=state.sessions_sha256_newline,
                        account_session_index=previous_state.account_session_index + 1,
                        session_index=0,
                        session=payload["next_session"],
                        session_start_equity_micros=previous_state.cash_micros,
                        realized_session_pnl_micros=0,
                        sequence=state.sequence,
                        last_decision_time_ns=None,
                        prior_transition_sha256=state.prior_transition_sha256,
                        ledger_sha256=state.ledger_sha256,
                    )
                    if (
                        state != allowed
                        or previous_state.position is not None
                        or previous_state.pending_intent_id is not None
                        or payload["prior_authorization_sha256"]
                        != previous_state.authorization_sha256
                        or payload["next_authorization_sha256"]
                        != state.authorization_sha256
                        or payload["prior_session"] != previous_state.session
                        or payload["next_session"] != state.session
                        or payload["prior_session_index"]
                        != previous_state.session_index
                        or payload["next_session_index"] != 0
                        or payload["prior_account_session_index"]
                        != previous_state.account_session_index
                        or payload["next_account_session_index"]
                        != state.account_session_index
                        or payload["reason"]
                        != "PATHD_NESTED_VALIDATION_AUTHORIZATION_HANDOFF"
                    ):
                        raise ValueError("authorization handoff economics/state drift")
                elif kind == "POSITION_CLOCK_ADVANCE":
                    allowed = replace(
                        previous_state,
                        sequence=state.sequence,
                        last_decision_time_ns=transition.event_time_ns,
                        prior_transition_sha256=state.prior_transition_sha256,
                        ledger_sha256=state.ledger_sha256,
                    )
                    payload = transition.event_payload
                    if (
                        state != allowed
                        or type(previous_state.position) is not dict
                        or previous_state.pending_intent_id is not None
                        or type(previous_state.last_decision_time_ns) is not int
                        or transition.event_time_ns
                        != previous_state.last_decision_time_ns + 1_000_000_000
                        or transition.event_time_ns
                        >= _terminal_time_ns(previous_state.session)
                        or payload["prior_authorization_sha256"]
                        != previous_state.authorization_sha256
                        or payload["next_authorization_sha256"]
                        != previous_state.authorization_sha256
                        or payload["prior_session"] != previous_state.session
                        or payload["next_session"] != previous_state.session
                        or payload["prior_session_index"]
                        != previous_state.session_index
                        or payload["next_session_index"]
                        != previous_state.session_index
                        or payload["prior_account_session_index"]
                        != previous_state.account_session_index
                        or payload["next_account_session_index"]
                        != previous_state.account_session_index
                        or payload["intervening_skip_receipt_sha256s"] not in ((), [])
                        or type(payload["reason"]) is not str
                        or not payload["reason"]
                    ):
                        raise ValueError("position clock advance state drift")
                else:
                    raise ValueError("unregistered research control event")
        prior_transition_sha = transition.transition_sha256
        prior_ledger_sha = state.ledger_sha256
        previous_state = state
    if value.tip_ledger != value.transitions[-1].next_ledger:
        raise ValueError("journal tip differs from transition chain")
    if value.tip_ledger.authorization_sha256 != authorization_sha:
        raise ValueError("journal authorization is stale")
    if value.tip_ledger.sessions_sha256_newline != current.sessions_sha256_newline:
        raise ValueError("journal session partition drift")
    if value.journal_root_sha256 != _journal_root(
        account_scope_sha256=value.account_scope_sha256,
        policy_id=value.policy_id,
        fee_path=value.fee_path,
        transitions=value.transitions,
        tip_ledger=value.tip_ledger,
    ):
        raise ValueError("journal root drift")
    return value


def validate_research_ledger_journal(
    journal: Any, /, *, authorization: Any
) -> ResearchLedgerJournalV1:
    try:
        current = assert_entry_evidence_authorization_current(authorization)
    except RuntimeError:
        # Immutable result validation necessarily occurs after the live decode
        # capability has become terminal.  It may inspect (never decode with)
        # the exact durable authorization identity recorded on disk.
        if type(authorization) is not prereg.FrozenEvidenceAuthorization:
            raise
        from v4.research.pathd_evidence_gate import (
            read_frozen_entry_evidence_authorization,
        )

        current = read_frozen_entry_evidence_authorization(
            role=authorization.role,
            outer_fold=authorization.outer_fold,
            inner_fold=authorization.inner_fold,
        )
        if current != authorization:
            raise ValueError("journal authorization differs from durable evidence identity")
    return _validate_research_ledger_journal_bound(journal, current=current)


def _append_event(
    journal: ResearchLedgerJournalV1,
    *,
    event_payload: dict[str, Any],
    event_schema_version: str,
    event_hash: str,
    event_time_ns: int,
    transition_kind: str,
    next_state_changes: dict[str, Any] | None = None,
) -> ResearchLedgerJournalV1:
    tip = journal.tip_ledger
    changes = dict(next_state_changes or {})
    state_values = _ledger_semantic(tip)
    state_values.update(changes)
    state_values["sequence"] = tip.sequence + 1
    state_values["prior_transition_sha256"] = journal.transitions[-1].transition_sha256
    next_state = _seal_ledger(**state_values)
    transition = _seal_transition(
        transition_kind=transition_kind,
        event_time_ns=event_time_ns,
        prior_authorization_sha256=tip.authorization_sha256,
        next_authorization_sha256=next_state.authorization_sha256,
        prior_ledger_sha256=tip.ledger_sha256,
        event_schema_version=event_schema_version,
        event_sha256=event_hash,
        event_payload=event_payload,
        next_ledger=next_state,
    )
    transitions = (*journal.transitions, transition)
    return _seal_journal(
        account_scope_sha256=journal.account_scope_sha256,
        policy_id=journal.policy_id,
        fee_path=journal.fee_path,
        transitions=transitions,
        tip_ledger=next_state,
    )


def _entry_frame_state_ineligibility_reasons(
    *, example: EntryExampleV1, journal: ResearchLedgerJournalV1
) -> tuple[str, ...]:
    tip = journal.tip_ledger
    decision_time_ns = example.model_input.decision_time_ns
    if type(decision_time_ns) is not int or decision_time_ns < 0:
        raise ValueError("frame coverage decision clock is mistyped")
    if (
        tip.last_decision_time_ns is not None
        and decision_time_ns < tip.last_decision_time_ns
    ):
        raise ValueError("frame coverage would move the causal account clock backward")
    physical = np.asarray(example.model_input.physical_action_mask)
    facts = tuple(example.action_execution_facts)
    if (
        physical.shape != (42,)
        or physical.dtype != np.bool_
        or len(facts) != 42
        or tuple(bool(value) for value in physical)
        != tuple(bool(fact.physical_eligible) for fact in facts)
    ):
        raise ValueError("frame coverage physical ladder drift")
    reasons: list[str] = []
    if any(
        transition.event_schema_version == ResearchLedgerControlEventV1.SCHEMA_VERSION
        and transition.event_payload.get("event_kind") == "SESSION_TERMINAL"
        for transition in journal.transitions
        if transition.next_ledger.session == tip.session
    ):
        reasons.append("SESSION_TERMINAL")
    if any(
        transition.event_schema_version == NonOrderSafetyEventV1.SCHEMA_VERSION
        for transition in journal.transitions
        if transition.next_ledger.session == tip.session
    ):
        reasons.append("FLAT_TERMINAL")
    if tip.pending_intent_id is not None:
        reasons.append("INTENT_PENDING")
    if tip.position is not None:
        reasons.append("POSITION_OCCUPIED")
    loss_budget = math.floor(0.05 * tip.session_start_equity_micros)
    if tip.realized_session_pnl_micros <= -loss_budget:
        reasons.append("DAILY_STOPPED")
    local = _datetime_from_ns(decision_time_ns).astimezone(
        ZoneInfo("America/New_York")
    )
    if (local.hour, local.minute, local.second) >= (15, 30, 0):
        reasons.append("ENTRY_CUTOFF")
    if (
        tip.last_decision_time_ns is not None
        and decision_time_ns <= tip.last_decision_time_ns
    ):
        reasons.append("ACCOUNT_CLOCK_NOT_STRICTLY_LATER")
    return tuple(reasons)


def entry_frame_coverage_from_verified_example(
    example: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
    journal: Any,
) -> EntryFrameCoverageV1:
    current, dataset, _authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    if type(example) is not EntryExampleV1:
        raise TypeError("frame coverage requires sealed evidence example/dataset")
    active = validate_research_ledger_journal(journal, authorization=current)
    matching = [
        row for row in dataset.examples
        if row.canonical_sha256() == example.canonical_sha256()
    ]
    if len(matching) != 1 or matching[0] is not example:
        raise ValueError("frame coverage example is absent from dataset")
    already = [
        row.event_payload.get("example_sha256")
        for row in active.transitions
        if row.event_schema_version == EntryFrameCoverageV1.SCHEMA_VERSION
    ]
    expected_index = len(already)
    if (
        expected_index >= len(dataset.examples)
        or dataset.examples[expected_index] is not example
    ):
        raise ValueError("frame coverage is not in exact dataset order")
    if example.model_input.session != active.tip_ledger.session:
        raise ValueError("frame coverage session differs from ledger")
    reasons = _entry_frame_state_ineligibility_reasons(
        example=example, journal=active
    )
    disposition = "ACTION_DECISION" if not reasons else "STATE_INELIGIBLE"
    semantic = {
        "schema_version": EntryFrameCoverageV1.SCHEMA_VERSION,
        "authorization_sha256": prereg.stable_hash(current.to_dict()),
        "dataset_sha256": dataset.dataset_sha256,
        "session": example.model_input.session,
        "decision_time_ns": example.model_input.decision_time_ns,
        "example_sha256": example.canonical_sha256(),
        "disposition": disposition,
        "reason_codes": reasons,
        "prior_journal_root_sha256": active.journal_root_sha256,
    }
    return EntryFrameCoverageV1(
        **semantic, coverage_sha256=prereg.stable_hash(_plain(semantic))
    )


def append_entry_frame_coverage(
    journal: Any,
    coverage: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
) -> ResearchLedgerJournalV1:
    current, dataset, _authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    active = validate_research_ledger_journal(journal, authorization=current)
    if type(coverage) is not EntryFrameCoverageV1:
        raise TypeError("coverage event is mistyped")
    matches = [
        example for example in dataset.examples
        if example.canonical_sha256() == coverage.example_sha256
    ]
    if len(matches) != 1:
        raise ValueError("coverage example binding drift")
    expected = entry_frame_coverage_from_verified_example(
        matches[0], dataset=dataset, authorization=authorization, journal=active
    )
    if coverage != expected:
        raise ValueError("coverage event is not independently reconstructible")
    return _append_event(
        active,
        event_payload=_plain(coverage),
        event_schema_version=coverage.SCHEMA_VERSION,
        event_hash=coverage.coverage_sha256,
        event_time_ns=coverage.decision_time_ns,
        transition_kind="FRAME_COVERAGE",
        next_state_changes={"last_decision_time_ns": coverage.decision_time_ns},
    )


def append_research_action_decision(
    journal: Any,
    decision: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
) -> ResearchLedgerJournalV1:
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    active = validate_research_ledger_journal(journal, authorization=current)
    if type(decision) is not EntryActionDecisionV1:
        raise TypeError("action decision is mistyped")
    matches = [
        row for row in sealed.examples
        if row.canonical_sha256() == decision.example_sha256
    ]
    if len(matches) != 1:
        raise ValueError("action decision example binding drift")
    checked = _validate_action_decision(
        decision,
        dataset=sealed,
        authorization_sha256=authorization_sha256,
        example=matches[0],
    )
    payload = _plain(checked)
    decision_hash = checked.decision_sha256
    if checked.prior_journal_root_sha256 != active.journal_root_sha256:
        raise ValueError("action decision prior-journal binding drift")
    if checked.example_sha256 != active.transitions[-1].event_payload.get(
        "example_sha256"
    ) or active.transitions[-1].event_payload.get("disposition") != "ACTION_DECISION":
        raise ValueError("action decision does not immediately follow its frame coverage")
    if matches[0].model_input.session != active.tip_ledger.session:
        raise ValueError("action decision session differs from the ledger")
    return _append_event(
        active,
        event_payload=payload,
        event_schema_version=payload["schema_version"],
        event_hash=decision_hash,
        event_time_ns=active.tip_ledger.last_decision_time_ns or 0,
        transition_kind="ACTION_DECISION",
    )


def _validate_all_action_targets(example: EntryExampleV1) -> None:
    horizons = ("h3", "h5", "h10", "h20", "h45", "h90", "session")
    axes = (
        "mfe_dollars", "mfe_return",
        "profit_area_dollars", "profit_area_return",
    )
    expected = [f"{horizon}_{axis}" for horizon in horizons for axis in axes]
    if (
        type(example.targets) is not dict
        or type(example.target_validity) is not dict
        or list(example.targets) != expected
        or list(example.target_validity) != expected
    ):
        raise ValueError("replay requires the canonical all-42-action target layout")
    for name in expected:
        values = example.targets[name]
        validity = example.target_validity[name]
        if (
            type(values) is not list
            or type(validity) is not list
            or len(values) != 42
            or len(validity) != 42
            or any(type(flag) is not bool for flag in validity)
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError("all-action target vector schema drift")


def _realized_action_composite_micros(
    example: EntryExampleV1,
    *,
    action_index: int,
    available_horizons: tuple[str, ...],
) -> int:
    if type(action_index) is not int or not 0 <= action_index < 42:
        raise ValueError("realized action index drift")
    allowed = ("h10", "h20", "h45", "h90", "remaining_session")
    if (
        not available_horizons
        or any(value not in allowed for value in available_horizons)
        or tuple(value for value in allowed if value in available_horizons)
        != available_horizons
    ):
        raise ValueError("realized action horizon order drift")
    _validate_all_action_targets(example)
    components: list[float] = []
    for horizon in available_horizons:
        prefix = "session" if horizon == "remaining_session" else horizon
        for suffix in ("mfe_dollars", "profit_area_dollars"):
            name = f"{prefix}_{suffix}"
            if example.target_validity[name][action_index] is not True:
                raise ValueError("realized action requires an invalid target component")
            components.append(float(example.targets[name][action_index]))
    return int(round(math.fsum(components) / len(components) * 1_000_000))


def entry_action_observation_from_verified_dataset(
    decision: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
    trajectory_id: str,
    episode_id: str,
    execution_outcome: Any,
) -> EntryActionObservationV1:
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    if type(decision) is not EntryActionDecisionV1:
        raise TypeError("action observation decision is mistyped")
    matches = [
        row for row in sealed.examples
        if row.canonical_sha256() == decision.example_sha256
    ]
    if len(matches) != 1:
        raise ValueError("action observation example binding drift")
    example = matches[0]
    checked = _validate_action_decision(
        decision,
        dataset=sealed,
        authorization_sha256=authorization_sha256,
        example=example,
    )
    if any(type(value) is not str or not value for value in (trajectory_id, episode_id)):
        raise ValueError("action observation trajectory/episode identity is missing")
    result: ResearchExecutionResultV1 | None = None
    if checked.action == "ENTER":
        outcome = validate_research_execution_outcome(
            execution_outcome,
            authorization=current,
            dataset=sealed,
            law=research_fill_law_from_preregistration(
                prereg.preregistration_payload()[0]
            ),
        )
        result = outcome.execution_result
        decisions = [
            transition.event_payload
            for transition in outcome.result_journal.transitions
            if transition.event_schema_version == EntryActionDecisionV1.SCHEMA_VERSION
            and transition.event_payload.get("decision_sha256") == checked.decision_sha256
        ]
        if len(decisions) != 1:
            raise ValueError("action observation outcome lacks its causal decision")
        result_transition = outcome.result_journal.transitions[-1]
        decision_index = next(
            index
            for index, transition in enumerate(outcome.result_journal.transitions)
            if transition.event_schema_version == EntryActionDecisionV1.SCHEMA_VERSION
            and transition.event_payload.get("decision_sha256")
            == checked.decision_sha256
        )
        if (
            result_transition.event_schema_version
            != ResearchExecutionResultV1.SCHEMA_VERSION
            or result.position_before is not None
            or (
                result.filled_quantity == 0
                and result.position_after is not None
            )
            or (
                result.filled_quantity == 1
                and (
                    type(result.position_after) is not dict
                    or result.position_after.get("entry_decision_sha256")
                    != checked.decision_sha256
                    or result.position_after.get("entry_example_sha256")
                    != checked.example_sha256
                )
            )
            or any(
                transition.event_schema_version
                in {
                    EntryActionDecisionV1.SCHEMA_VERSION,
                    ResearchExecutionResultV1.SCHEMA_VERSION,
                }
                for transition in outcome.result_journal.transitions[
                    decision_index + 1 : -1
                ]
            )
            or result.delay_ms != 60_000
            or result.source_receipt_sha256
            != _session_source_receipt(
                sealed, session=example.model_input.session
            )["receipt_sha256"]
            or (
                result.filled_quantity == 1
                and result.fill_price_micros != checked.buy_hard_limit_micros
            )
        ):
            raise ValueError("action observation consumed a non-entry execution outcome")
        realized_micros = _realized_action_composite_micros(
            example,
            action_index=checked.selected_action_index,
            available_horizons=tuple(checked.available_horizons),
        )
    else:
        if execution_outcome is not None:
            raise ValueError("WAIT action observation cannot consume an execution outcome")
        legal = [
            index for index, allowed in enumerate(checked.combined_action_mask)
            if allowed
        ]
        realized_micros = max(
            [0],
            *(
                _realized_action_composite_micros(
                    example,
                    action_index=index,
                    available_horizons=tuple(checked.available_horizons),
                )
                for index in legal
            ),
        )
    predicted_mean = int(round(float(checked.mean_lcb_dollars) * 1_000_000))
    predicted_lower = int(round(float(checked.q10_dollars) * 1_000_000))
    filled = result is not None and result.filled_quantity == 1
    semantic = {
        "schema_version": EntryActionObservationV1.SCHEMA_VERSION,
        "authorization_sha256": authorization_sha256,
        "dataset_sha256": sealed.dataset_sha256,
        "session": example.model_input.session,
        "trajectory_id": trajectory_id,
        "episode_id": episode_id,
        "example_sha256": example.canonical_sha256(),
        "decision_sha256": checked.decision_sha256,
        "action": checked.action,
        "predicted_mean_cash_micros": predicted_mean,
        "predicted_lower_cash_micros": predicted_lower,
        "realized_value_cash_micros": realized_micros,
        "coverage": realized_micros >= predicted_lower,
        "filled": filled,
        "execution_result_sha256": None if result is None else result.result_sha256,
    }
    return EntryActionObservationV1(
        **semantic, observation_sha256=prereg.stable_hash(semantic)
    )


def append_entry_action_observation(
    journal: Any,
    observation: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
) -> ResearchLedgerJournalV1:
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    active = validate_research_ledger_journal(journal, authorization=current)
    if type(observation) is not EntryActionObservationV1:
        raise TypeError("action observation is mistyped")
    semantic = {
        field.name: _plain(getattr(observation, field.name))
        for field in fields(observation)
        if field.name != "observation_sha256"
    }
    if observation.observation_sha256 != prereg.stable_hash(semantic):
        raise ValueError("action observation hash drift")
    if (
        observation.authorization_sha256 != authorization_sha256
        or observation.dataset_sha256 != sealed.dataset_sha256
        or observation.session != active.tip_ledger.session
        or sum(
            row.canonical_sha256() == observation.example_sha256
            for row in sealed.examples
        )
        != 1
    ):
        raise ValueError("action observation authority/dataset binding drift")
    decisions = [
        transition for transition in active.transitions
        if transition.event_schema_version == EntryActionDecisionV1.SCHEMA_VERSION
        and transition.event_payload.get("decision_sha256")
        == observation.decision_sha256
        and transition.event_payload.get("example_sha256")
        == observation.example_sha256
    ]
    if len(decisions) != 1 or decisions[0].event_payload.get("action") != observation.action:
        raise ValueError("action observation decision binding drift")
    if any(
        transition.event_schema_version == EntryActionObservationV1.SCHEMA_VERSION
        and transition.event_payload.get("decision_sha256")
        == observation.decision_sha256
        for transition in active.transitions
    ):
        raise ValueError("action decision already has an observation")
    results = [
        transition.event_payload
        for transition in active.transitions
        if transition.event_schema_version == ResearchExecutionResultV1.SCHEMA_VERSION
        and transition.event_payload.get("result_sha256")
        == observation.execution_result_sha256
    ]
    if observation.action == "ENTER" and len(results) != 1:
        raise ValueError("ENTER observation lacks its execution result")
    if observation.action == "WAIT" and observation.execution_result_sha256 is not None:
        raise ValueError("WAIT observation cannot name an execution result")
    return _append_event(
        active,
        event_payload=_plain(observation),
        event_schema_version=observation.SCHEMA_VERSION,
        event_hash=observation.observation_sha256,
        event_time_ns=active.tip_ledger.last_decision_time_ns or 0,
        transition_kind="ACTION_OBSERVATION",
    )


def mark_research_session_terminal(
    journal: Any,
    /,
    *,
    authorization: Any,
    reason_code: str,
) -> ResearchLedgerJournalV1:
    current = assert_entry_evidence_authorization_current(authorization)
    active = validate_research_ledger_journal(journal, authorization=current)
    if reason_code not in prereg.RESEARCH_SESSION_TERMINAL_REASON_CODES:
        raise ValueError("unregistered research session terminal reason")
    if active.tip_ledger.position is not None or active.tip_ledger.pending_intent_id is not None:
        raise ValueError("cannot mark a nonflat/pending session terminal")
    event = _seal_control_event(
        event_kind="SESSION_TERMINAL",
        event_time_ns=active.tip_ledger.last_decision_time_ns or 0,
        prior_authorization_sha256=active.tip_ledger.authorization_sha256,
        next_authorization_sha256=active.tip_ledger.authorization_sha256,
        prior_session=active.tip_ledger.session,
        next_session=active.tip_ledger.session,
        prior_session_index=active.tip_ledger.session_index,
        next_session_index=active.tip_ledger.session_index,
        prior_account_session_index=active.tip_ledger.account_session_index,
        next_account_session_index=active.tip_ledger.account_session_index,
        intervening_skip_receipt_sha256s=(),
        reason=reason_code,
    )
    return _append_event(
        active,
        event_payload=_plain(event),
        event_schema_version=event.SCHEMA_VERSION,
        event_hash=event.event_sha256,
        event_time_ns=event.event_time_ns,
        transition_kind="CONTROL",
    )


def advance_research_ledger_session(
    journal: Any, /, *, authorization: Any
) -> ResearchLedgerJournalV1:
    current = assert_entry_evidence_authorization_current(authorization)
    active = validate_research_ledger_journal(journal, authorization=current)
    last = active.transitions[-1]
    if last.event_payload.get("event_kind") != "SESSION_TERMINAL":
        raise ValueError("session advance requires terminal prior session")
    next_index = active.tip_ledger.session_index + 1
    if next_index >= len(current.sessions):
        raise ValueError("no next authorized session")
    next_session = current.sessions[next_index]
    event = _seal_control_event(
        event_kind="SESSION_ADVANCE",
        event_time_ns=0,
        prior_authorization_sha256=active.tip_ledger.authorization_sha256,
        next_authorization_sha256=active.tip_ledger.authorization_sha256,
        prior_session=active.tip_ledger.session,
        next_session=next_session,
        prior_session_index=active.tip_ledger.session_index,
        next_session_index=next_index,
        prior_account_session_index=active.tip_ledger.account_session_index,
        next_account_session_index=active.tip_ledger.account_session_index + 1,
        intervening_skip_receipt_sha256s=(),
        reason="PATHD_RESEARCH_NEXT_SESSION",
    )
    return _append_event(
        active,
        event_payload=_plain(event),
        event_schema_version=event.SCHEMA_VERSION,
        event_hash=event.event_sha256,
        event_time_ns=0,
        transition_kind="CONTROL",
        next_state_changes={
            "account_session_index": active.tip_ledger.account_session_index + 1,
            "session_index": next_index,
            "session": next_session,
            "session_start_equity_micros": active.tip_ledger.cash_micros,
            "realized_session_pnl_micros": 0,
            "last_decision_time_ns": None,
        },
    )


def advance_research_position_clock(
    journal: Any,
    /,
    *,
    authorization: Any,
    decision_time_ns: int,
    reason_code: str,
) -> ResearchLedgerJournalV1:
    """Record one exact held-position second at which no order was emitted."""

    current = assert_entry_evidence_authorization_current(authorization)
    active = validate_research_ledger_journal(journal, authorization=current)
    event = _position_clock_advance_event(
        journal=active,
        decision_time_ns=decision_time_ns,
        reason_code=reason_code,
    )
    return append_research_position_clock_event(
        active, event, authorization=current
    )


def _position_clock_advance_event(
    *,
    journal: ResearchLedgerJournalV1,
    decision_time_ns: int,
    reason_code: str,
) -> ResearchLedgerControlEventV1:
    tip = journal.tip_ledger
    if (
        type(tip.position) is not dict
        or tip.pending_intent_id is not None
        or type(tip.last_decision_time_ns) is not int
        or type(decision_time_ns) is not int
        or decision_time_ns != tip.last_decision_time_ns + 1_000_000_000
        or decision_time_ns >= _terminal_time_ns(tip.session)
        or type(reason_code) is not str
        or not reason_code
    ):
        raise ValueError("position clock advance is not one causal held second")
    return _seal_control_event(
        event_kind="POSITION_CLOCK_ADVANCE",
        event_time_ns=decision_time_ns,
        prior_authorization_sha256=tip.authorization_sha256,
        next_authorization_sha256=tip.authorization_sha256,
        prior_session=tip.session,
        next_session=tip.session,
        prior_session_index=tip.session_index,
        next_session_index=tip.session_index,
        prior_account_session_index=tip.account_session_index,
        next_account_session_index=tip.account_session_index,
        intervening_skip_receipt_sha256s=(),
        reason=reason_code,
    )


def append_research_position_clock_event(
    journal: Any,
    event: Any,
    /,
    *,
    authorization: Any,
) -> ResearchLedgerJournalV1:
    current = assert_entry_evidence_authorization_current(authorization)
    active = validate_research_ledger_journal(journal, authorization=current)
    if type(event) is not ResearchLedgerControlEventV1:
        raise TypeError("position clock event is mistyped")
    expected = _position_clock_advance_event(
        journal=active,
        decision_time_ns=event.event_time_ns,
        reason_code=event.reason,
    )
    if event != expected:
        raise ValueError("position clock event is not independently reconstructible")
    advanced = _append_event(
        active,
        event_payload=_plain(event),
        event_schema_version=event.SCHEMA_VERSION,
        event_hash=event.event_sha256,
        event_time_ns=event.event_time_ns,
        transition_kind="CONTROL",
        next_state_changes={"last_decision_time_ns": event.event_time_ns},
    )
    return validate_research_ledger_journal(advanced, authorization=current)


def handoff_research_ledger_authorization(
    journal: Any,
    /,
    *,
    current_authorization: Any,
    next_authorization: Any,
) -> ResearchLedgerJournalV1:
    if (
        type(current_authorization) is not prereg.FrozenEvidenceAuthorization
        or type(next_authorization) is not prereg.FrozenEvidenceAuthorization
        or current_authorization.role != "nested_validation"
        or next_authorization.role != "nested_validation"
        or type(current_authorization.outer_fold) is not int
        or current_authorization.outer_fold != next_authorization.outer_fold
        or type(current_authorization.inner_fold) is not int
        or type(next_authorization.inner_fold) is not int
        or next_authorization.inner_fold <= current_authorization.inner_fold
    ):
        raise ValueError("authorization handoff requires ordered same-outer nested blocks")
    next_current = assert_entry_evidence_authorization_current(next_authorization)
    assignments = prereg.read_json(prereg.SESSION_PATH)
    payload = prereg.read_json(prereg.PREREG_PATH)
    outer_fold = current_authorization.outer_fold
    rows = assignments["folds"][outer_fold - 1]["inner_forward_folds"][
        "scored_forward_folds"
    ]
    valid_rows = [row for row in rows if row.get("calibration_valid") is True]
    current_rows = [
        row for row in valid_rows
        if row.get("inner_fold") == current_authorization.inner_fold
    ]
    next_rows = [
        row for row in valid_rows
        if row.get("inner_fold") == next_current.inner_fold
    ]
    if (
        len(current_rows) != 1
        or len(next_rows) != 1
        or valid_rows.index(next_rows[0]) != valid_rows.index(current_rows[0]) + 1
        or tuple(current_rows[0]["validation"])
        != tuple(current_authorization.sessions)
        or tuple(next_rows[0]["validation"]) != tuple(next_current.sessions)
        or current_authorization.sessions_sha256_newline
        != prereg.canonical_session_hash(current_authorization.sessions)
        or next_current.sessions_sha256_newline
        != prereg.canonical_session_hash(next_current.sessions)
    ):
        raise ValueError("authorization handoff skipped or changed a valid nested block")

    active = _validate_research_ledger_journal_bound(
        journal, current=current_authorization
    )
    last = active.transitions[-1]
    expected_current_global_index = sum(
        len(row["validation"])
        for row in valid_rows[: valid_rows.index(current_rows[0])]
    ) + len(current_authorization.sessions) - 1
    if (
        active.policy_id not in {"HGB", "NEURAL"}
        or active.tip_ledger.session != current_authorization.sessions[-1]
        or active.tip_ledger.session_index != len(current_authorization.sessions) - 1
        or active.tip_ledger.account_session_index != expected_current_global_index
        or active.tip_ledger.position is not None
        or active.tip_ledger.pending_intent_id is not None
        or last.event_schema_version != ResearchLedgerControlEventV1.SCHEMA_VERSION
        or last.event_payload.get("event_kind") != "SESSION_TERMINAL"
    ):
        raise ValueError("authorization handoff requires the exact flat terminal block tip")
    expected_scope = _nested_account_scope_sha256(
        policy_id=active.policy_id,
        fee_path=active.fee_path,
        outer_fold=outer_fold,
        assignments=assignments,
    )
    if active.account_scope_sha256 != expected_scope:
        raise ValueError("authorization handoff account scope drift")

    # This validates the frozen result, its complete policy evaluation, and the
    # terminal journal/root before any authority is changed.
    prereg._validate_nested_result_receipt_one(
        payload,
        assignments,
        outer_fold,
        current_authorization.inner_fold,
    )
    result_path = prereg._outer_fold_artifact_path(
        outer_fold,
        f"nested_inner_{current_authorization.inner_fold}_result.json",
    )
    result = prereg.read_json(result_path)
    evaluation = result.get(active.policy_id.lower())
    if (
        type(evaluation) is not dict
        or evaluation.get("terminal_journal_sha256")
        != active.journal_root_sha256
        or evaluation.get("terminal_journal") != _plain(active)
    ):
        raise ValueError("authorization handoff journal is not the frozen block result")

    skip_hashes: list[str] = []
    for inner_fold in range(
        current_authorization.inner_fold + 1, next_current.inner_fold
    ):
        prereg._validate_nested_skip_receipt_one(
            payload, assignments, outer_fold, inner_fold
        )
        skip_hashes.append(
            prereg.sha256_path(
                prereg._outer_fold_artifact_path(
                    outer_fold, f"nested_inner_{inner_fold}_skip_receipt.json"
                )
            )
        )

    next_authorization_sha = prereg.stable_hash(next_current.to_dict())
    event = _seal_control_event(
        event_kind="AUTHORIZATION_HANDOFF",
        event_time_ns=0,
        prior_authorization_sha256=active.tip_ledger.authorization_sha256,
        next_authorization_sha256=next_authorization_sha,
        prior_session=active.tip_ledger.session,
        next_session=next_current.sessions[0],
        prior_session_index=active.tip_ledger.session_index,
        next_session_index=0,
        prior_account_session_index=active.tip_ledger.account_session_index,
        next_account_session_index=active.tip_ledger.account_session_index + 1,
        intervening_skip_receipt_sha256s=tuple(skip_hashes),
        reason="PATHD_NESTED_VALIDATION_AUTHORIZATION_HANDOFF",
    )
    handed = _append_event(
        active,
        event_payload=_plain(event),
        event_schema_version=event.SCHEMA_VERSION,
        event_hash=event.event_sha256,
        event_time_ns=0,
        transition_kind="CONTROL",
        next_state_changes={
            "authorization_sha256": next_authorization_sha,
            "sessions_sha256_newline": next_current.sessions_sha256_newline,
            "account_session_index": active.tip_ledger.account_session_index + 1,
            "session_index": 0,
            "session": next_current.sessions[0],
            "session_start_equity_micros": active.tip_ledger.cash_micros,
            "realized_session_pnl_micros": 0,
            "last_decision_time_ns": None,
        },
    )
    return _validate_research_ledger_journal_bound(handed, current=next_current)


def validate_research_ledger_journal_against_dataset(
    journal: Any,
    /,
    *,
    authorization: Any,
    dataset: Any,
) -> ResearchLedgerJournalV1:
    active = validate_research_ledger_journal(journal, authorization=authorization)
    validate_entry_evidence_dataset(dataset, authorization=authorization)
    authorization_sha256 = prereg.stable_hash(authorization.to_dict())
    segment_start = _authorization_segment_start_index(
        active, authorization_sha256=authorization_sha256
    )
    expected = [example.canonical_sha256() for example in dataset.examples]
    observed: list[str] = []
    disposition_by_example: dict[str, str] = {}
    decision_counts = {digest: 0 for digest in expected}
    observation_counts = {digest: 0 for digest in expected}
    for transition in active.transitions[segment_start:]:
        if transition.event_schema_version == EntryFrameCoverageV1.SCHEMA_VERSION:
            digest = transition.event_payload.get("example_sha256")
            observed.append(digest)
            disposition_by_example[digest] = transition.event_payload.get("disposition")
        elif transition.event_schema_version == "pathd.entry_action_decision.v1":
            digest = transition.event_payload.get("example_sha256")
            if digest in decision_counts:
                decision_counts[digest] += 1
        elif transition.event_schema_version == EntryActionObservationV1.SCHEMA_VERSION:
            digest = transition.event_payload.get("example_sha256")
            if digest in observation_counts:
                observation_counts[digest] += 1
    if observed != expected:
        raise ValueError("EntryFrameCoverageV1 rows do not visit dataset.examples exactly once")
    for digest in expected:
        disposition = disposition_by_example[digest]
        if disposition == "ACTION_DECISION":
            if decision_counts[digest] != 1 or observation_counts[digest] != 1:
                raise ValueError("ACTION_DECISION requires one decision and one observation")
        elif disposition == "STATE_INELIGIBLE":
            if decision_counts[digest] or observation_counts[digest]:
                raise ValueError("STATE_INELIGIBLE forbids decision/observation")
        else:
            raise ValueError("unknown frame disposition")
    return active


def reconstruct_entry_policy_economics_from_validated_transitions(
    *, transitions: Any, starting_cash_micros: int
) -> dict[str, Any]:
    if type(starting_cash_micros) is not int or starting_cash_micros < 0:
        raise ValueError("starting cash must be a nonnegative exact integer")
    rows = tuple(transitions)
    cash = starting_cash_micros
    open_trade: str | None = None
    completed: list[str] = []
    for index, row in enumerate(rows, start=1):
        if type(row) is not dict:
            raise TypeError("economic transition must be an exact mapping")
        required = {
            "sequence", "kind", "trade_id", "cash_before_micros",
            "cash_delta_micros", "cash_after_micros", "position_after",
        }
        if set(row) != required or row["sequence"] != index:
            raise ValueError("economic transition schema/sequence drift")
        if any(type(row[name]) is not int for name in (
            "cash_before_micros", "cash_delta_micros", "cash_after_micros", "position_after"
        )):
            raise TypeError("economic transition integer field drift")
        if row["cash_before_micros"] != cash:
            raise ValueError("economic cash chain drift")
        if row["cash_after_micros"] != row["cash_before_micros"] + row["cash_delta_micros"]:
            raise ValueError("economic cash arithmetic drift")
        trade_id = row["trade_id"]
        if not isinstance(trade_id, str) or not trade_id:
            raise ValueError("economic trade identity drift")
        if row["kind"] == "BUY_FILL":
            if open_trade is not None or row["position_after"] != 1 or row["cash_delta_micros"] >= 0:
                raise ValueError("invalid BUY_FILL transition")
            open_trade = trade_id
        elif row["kind"] == "SELL_FILL":
            if open_trade != trade_id or row["position_after"] != 0 or row["cash_delta_micros"] <= 0:
                raise ValueError("invalid SELL_FILL transition")
            completed.append(trade_id)
            open_trade = None
        else:
            raise ValueError("unregistered economic transition kind")
        cash = row["cash_after_micros"]
    if open_trade is not None:
        raise ValueError("economic transition chain ends nonflat")
    return {
        "session_pnl_micros": cash - starting_cash_micros,
        "completed_trade_ids": completed,
        "terminal_cash_micros": cash,
    }


def _terminal_control_rows(journal: ResearchLedgerJournalV1) -> list[ResearchLedgerTransitionV1]:
    return [
        row for row in journal.transitions
        if row.event_schema_version == ResearchLedgerControlEventV1.SCHEMA_VERSION
        and row.event_payload.get("event_kind") == "SESSION_TERMINAL"
    ]


def _authorization_segment_start_index(
    journal: ResearchLedgerJournalV1,
    /,
    *,
    authorization_sha256: str,
) -> int:
    """Return the first event wholly governed by the current authorization.

    A nested-validation journal deliberately retains the complete serial-account
    history across authorization handoffs.  Dataset coverage and per-block
    economics, however, belong only to the authorization at the journal tip.
    The unique GENESIS or AUTHORIZATION_HANDOFF transition is the authenticated
    boundary between those two scopes.
    """

    if (
        type(authorization_sha256) is not str
        or len(authorization_sha256) != 64
        or any(character not in "0123456789abcdef" for character in authorization_sha256)
    ):
        raise ValueError("authorization segment hash is malformed")
    boundaries: list[int] = []
    for index, transition in enumerate(journal.transitions):
        if (
            transition.event_schema_version
            != ResearchLedgerControlEventV1.SCHEMA_VERSION
            or transition.next_authorization_sha256 != authorization_sha256
        ):
            continue
        kind = transition.event_payload.get("event_kind")
        if kind == "GENESIS":
            if index != 0 or transition.prior_authorization_sha256 is not None:
                raise ValueError("authorization genesis boundary drift")
            boundaries.append(index)
        elif kind == "AUTHORIZATION_HANDOFF":
            if transition.prior_authorization_sha256 == authorization_sha256:
                raise ValueError("authorization handoff did not change authority")
            boundaries.append(index)
    if len(boundaries) != 1:
        raise ValueError("journal does not contain one current-authorization boundary")
    start = boundaries[0] + 1
    for transition in journal.transitions[start:]:
        if (
            transition.prior_authorization_sha256 != authorization_sha256
            or transition.next_authorization_sha256 != authorization_sha256
        ):
            raise ValueError("current authorization segment is not contiguous")
    return start


def reconstruct_entry_policy_evaluation_from_journal(
    journal: Any,
    /,
    *,
    authorization: Any,
    dataset_sha256: str,
    sessions: Any,
    evidence_role: str,
    outer_fold: int,
    inner_fold: int | None,
    policy_id: str,
) -> dict[str, Any]:
    terminal_journal = validate_research_ledger_journal(
        journal, authorization=authorization
    )
    expected_sessions = tuple(sessions)
    if expected_sessions != tuple(authorization.sessions):
        raise ValueError("policy evaluation session partition drift")
    if terminal_journal.policy_id != policy_id:
        raise ValueError("policy evaluation policy drift")
    authorization_sha256 = prereg.stable_hash(authorization.to_dict())
    segment_start = _authorization_segment_start_index(
        terminal_journal, authorization_sha256=authorization_sha256
    )
    indexed_segment = tuple(
        enumerate(terminal_journal.transitions[segment_start:], start=segment_start)
    )
    terminal_rows = [
        row
        for _index, row in indexed_segment
        if row.event_schema_version == ResearchLedgerControlEventV1.SCHEMA_VERSION
        and row.event_payload.get("event_kind") == "SESSION_TERMINAL"
    ]
    if len(terminal_rows) != len(expected_sessions):
        raise ValueError("policy evaluation requires one terminal per session")
    if [row.next_ledger.session for row in terminal_rows] != list(expected_sessions):
        raise ValueError("policy evaluation terminal session order drift")
    pnl = [row.next_ledger.realized_session_pnl_micros for row in terminal_rows]
    terminal_hashes: list[str] = []
    terminal_index_by_sha256 = {
        row.transition_sha256: index
        for index, row in indexed_segment
        if row.event_schema_version == ResearchLedgerControlEventV1.SCHEMA_VERSION
        and row.event_payload.get("event_kind") == "SESSION_TERMINAL"
    }
    for row in terminal_rows:
        if row is terminal_journal.transitions[-1]:
            terminal_hashes.append(terminal_journal.journal_root_sha256)
        else:
            index = terminal_index_by_sha256[row.transition_sha256]
            prefix = terminal_journal.transitions[: index + 1]
            terminal_hashes.append(
                _journal_root(
                    account_scope_sha256=terminal_journal.account_scope_sha256,
                    policy_id=terminal_journal.policy_id,
                    fee_path=terminal_journal.fee_path,
                    transitions=prefix,
                    tip_ledger=row.next_ledger,
                )
            )
    completed_ids: list[str] = []
    completed_sessions: list[str] = []
    for transition_index, transition in indexed_segment:
        payload = transition.event_payload
        if transition.event_schema_version == ResearchExecutionResultV1.SCHEMA_VERSION:
            if payload.get("filled_quantity") == 1 and payload.get("position_after") is None:
                position_before = payload.get("position_before")
                trade_id = (
                    position_before.get("trade_id")
                    if type(position_before) is dict
                    else None
                )
                if isinstance(trade_id, str) and trade_id not in completed_ids:
                    completed_ids.append(trade_id)
                    completed_sessions.append(transition.next_ledger.session)
        elif transition.event_schema_version == NonOrderSafetyEventV1.SCHEMA_VERSION:
            prior_position = (
                terminal_journal.transitions[transition_index - 1].next_ledger.position
                if transition_index > 0
                else None
            )
            trade_id = (
                prior_position.get("trade_id")
                if type(prior_position) is dict
                else None
            )
            if isinstance(trade_id, str) and trade_id not in completed_ids:
                completed_ids.append(trade_id)
                completed_sessions.append(transition.next_ledger.session)
    zero = [session for session in expected_sessions if session not in set(completed_sessions)]
    trace_root = prereg.stable_hash(
        [row.transition_sha256 for row in terminal_journal.transitions]
    )
    coverage_hash = prereg.stable_hash(
        {
            "sessions": list(expected_sessions),
            "session_pnl_micros": pnl,
            "session_terminal_journal_sha256s": terminal_hashes,
            "completed_trade_ids": completed_ids,
            "completed_trade_sessions": completed_sessions,
            "zero_trade_sessions": zero,
            "execution_trace_root_sha256": trace_root,
        }
    )
    value = {
        "schema_version": "pathd.entry_policy_evaluation.v1",
        "holdout_caveat": prereg.HOLDOUT_CAVEAT,
        "evidence_role": evidence_role,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "authorization_sha256": prereg.stable_hash(authorization.to_dict()),
        "dataset_sha256": dataset_sha256,
        "policy_id": policy_id,
        "sessions": list(expected_sessions),
        "sessions_sha256_newline": prereg.canonical_session_hash(expected_sessions),
        "session_pnl_micros": pnl,
        "session_terminal_journal_sha256s": terminal_hashes,
        "completed_trade_ids": completed_ids,
        "completed_trade_sessions": completed_sessions,
        "zero_trade_sessions": zero,
        "execution_trace_root_sha256": trace_root,
        "terminal_journal": _plain(terminal_journal),
        "terminal_journal_sha256": terminal_journal.journal_root_sha256,
        "session_coverage_sha256": coverage_hash,
        "metrics": {
            "unique_session_count": len(expected_sessions),
            "completed_trade_count": len(completed_ids),
            "total_net_pnl_micros": sum(pnl),
            "zero_trade_session_count": len(zero),
        },
    }
    value["result_sha256"] = prereg.stable_hash(value)
    return value


def entry_decision_context_from_verified_example(
    example: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
    decision: Any,
) -> ResearchDecisionContextV1:
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    member = _exact_dataset_example(example, dataset=sealed)
    checked = _validate_action_decision(
        decision,
        dataset=sealed,
        authorization_sha256=authorization_sha256,
        example=member,
    )
    if checked.action != "ENTER":
        raise ValueError("only an ENTER decision has an entry execution context")
    decision_time_ns = member.model_input.decision_time_ns
    fact = member.action_execution_facts[checked.selected_action_index]
    contract = (
        fact.contract
        if type(fact.contract) is ContractIdentityV1
        else ContractIdentityV1.from_dict(fact.contract)
    )
    quote_row = read_verified_research_quote_row(
        current,
        dataset=sealed,
        session=member.model_input.session,
        contract=contract,
        at_or_before_ns=decision_time_ns,
        maximum_age_ms=90_000,
        require_actionable=True,
    )
    quote = seal_research_quote(quote_row)
    spx = _verified_spx_row(
        read_verified_official_spx_row(
            current,
            dataset=sealed,
            session=member.model_input.session,
            at_or_before_ns=decision_time_ns,
            maximum_age_ms=90_000,
        ),
        session=member.model_input.session,
        at_or_before_ns=decision_time_ns,
        maximum_age_ms=90_000,
    )
    session_receipt = _validate_selected_source_binding(
        quote_row, dataset=sealed, session=member.model_input.session
    )
    _validate_selected_source_binding(
        spx, dataset=sealed, session=member.model_input.session
    )
    if (
        not quote.actionable
        or quote.contract != contract
        or quote.session != fact.session
        or quote.bid_micros != fact.bid_micros
        or quote.ask_micros != fact.ask_micros
        or quote.represented_interval_end_ns != fact.represented_interval_end_ns
        or quote.ts_recv_ns != fact.ts_recv_ns
        or quote.available_at_ns != fact.available_at_ns
        or fact.source_receipt_sha256 != session_receipt["source_files_sha256"]
    ):
        raise ValueError("entry context selector differs from the sealed execution fact")
    semantic = {
        "schema_version": ResearchDecisionContextV1.SCHEMA_VERSION,
        "intent_kind": "BUY",
        "session": member.model_input.session,
        "contract": contract,
        "decision_time_ns": decision_time_ns,
        "event_interval_end_ns": quote.represented_interval_end_ns,
        "option_watermark_ns": quote.ts_recv_ns,
        "spx_watermark_ns": spx.ts_recv_ns,
        "authorization_sha256": authorization_sha256,
        "dataset_sha256": sealed.dataset_sha256,
        "entry_example_sha256": member.canonical_sha256(),
        "entry_action_decision_sha256": checked.decision_sha256,
        "option_source_receipt_sha256": quote.source_receipt_sha256,
        "spx_source_receipt_sha256": spx.source_receipt_sha256,
        "decision_quote": quote,
        "official_spx_row": spx,
    }
    return ResearchDecisionContextV1(
        **semantic, context_sha256=prereg.stable_hash(_plain(semantic))
    )


def _terminal_zero_write_down_event(
    *,
    journal: ResearchLedgerJournalV1,
    authorization_sha256: str,
    fill_law: ResearchFillLawV1,
    quote: ResearchQuoteV1 | None,
) -> NonOrderSafetyEventV1:
    tip = journal.tip_ledger
    position = tip.position
    if type(position) is not dict:
        raise ValueError("terminal zero write-down requires a replay-owned long")
    contract_value = position["contract"]
    contract = (
        contract_value
        if type(contract_value) is ContractIdentityV1
        else ContractIdentityV1.from_dict(contract_value)
    )
    terminal_ns = _terminal_time_ns(tip.session)
    if quote is not None and (
        quote.session != tip.session
        or quote.contract != contract
        or quote.available_at_ns > terminal_ns
        or quote.actionable
        or quote.invalid_reason is None
    ):
        raise ValueError("terminal safety quote is not an invalid causal held-contract BBO")
    fee = _fee_per_filled_side(tip.fee_path)
    state_values = _ledger_semantic(tip)
    state_values.update(
        {
            "cash_micros": tip.cash_micros - fee,
            "realized_session_pnl_micros": (
                tip.realized_session_pnl_micros
                - position["entry_fill_price_micros"] * contract.multiplier
                - fee
            ),
            "position": None,
            "pending_intent_id": None,
            "sequence": tip.sequence + 1,
            "last_decision_time_ns": terminal_ns,
            "prior_transition_sha256": journal.transitions[-1].transition_sha256,
        }
    )
    result_ledger = _seal_ledger(**state_values)
    identity = {
        "schema_version": NonOrderSafetyEventV1.SCHEMA_VERSION,
        "event_type": "TERMINAL_ZERO_WRITE_DOWN",
        "session": tip.session,
        "policy_id": tip.policy_id,
        "fee_path": tip.fee_path,
        "authorization_sha256": authorization_sha256,
        "event_time_utc": _utc_from_ns(terminal_ns),
        "event_time_ns": terminal_ns,
        "source_neutral_contract_id": position["source_neutral_contract_id"],
        "osi_symbol": contract.osi_symbol,
        "state_from": (
            "EXIT_PENDING" if tip.pending_intent_id is not None else "LONG_ONE"
        ),
        "state_to": "FLAT_TERMINAL",
        "invalid_reason": (
            "NO_ELIGIBLE_BBO" if quote is None else quote.invalid_reason
        ),
        "provenance": "DERIVED_NO_ACTIONABLE_BBO",
        "last_actual_option_watermark_utc": (
            None if quote is None else _utc_from_ns(quote.ts_recv_ns)
        ),
        "pending_intent_id_consumed": tip.pending_intent_id,
        "fill_law_hash": fill_law.fill_law_hash,
        "prior_ledger_sha256": tip.ledger_sha256,
    }
    semantic = {
        **identity,
        "event_id": prereg.stable_hash(identity),
        "zero_proceeds_cash_micros": 0,
        "modeled_close_penalty_cash_micros": fee,
        "cash_before_micros": tip.cash_micros,
        "cash_after_micros": tip.cash_micros - fee,
        "equity_before_micros": tip.cash_micros,
        "equity_after_micros": tip.cash_micros - fee,
        "result_ledger": result_ledger,
    }
    return NonOrderSafetyEventV1(
        **semantic, event_sha256=prereg.stable_hash(_plain(semantic))
    )


def append_research_non_order_safety_event(
    journal: Any,
    event: Any,
    /,
    *,
    authorization: Any,
    law: Any,
) -> ResearchLedgerJournalV1:
    current = assert_entry_evidence_authorization_current(authorization)
    fill_law = _validate_fill_law(law)
    active = validate_research_ledger_journal(journal, authorization=current)
    if type(event) is not NonOrderSafetyEventV1:
        raise TypeError("non-order safety event is mistyped")
    semantic = {
        field.name: _plain(getattr(event, field.name))
        for field in fields(event)
        if field.name != "event_sha256"
    }
    if (
        event.event_sha256 != prereg.stable_hash(semantic)
        or event.authorization_sha256 != prereg.stable_hash(current.to_dict())
        or event.fill_law_hash != fill_law.fill_law_hash
        or event.prior_ledger_sha256 != active.tip_ledger.ledger_sha256
    ):
        raise ValueError("non-order safety event seal/authority drift")
    appended = _append_event(
        active,
        event_payload=_plain(event),
        event_schema_version=event.SCHEMA_VERSION,
        event_hash=event.event_sha256,
        event_time_ns=event.event_time_ns,
        transition_kind="NON_ORDER_SAFETY",
        next_state_changes={
            "cash_micros": event.result_ledger.cash_micros,
            "realized_session_pnl_micros": (
                event.result_ledger.realized_session_pnl_micros
            ),
            "position": None,
            "pending_intent_id": None,
            "last_decision_time_ns": event.event_time_ns,
        },
    )
    if appended.tip_ledger != event.result_ledger:
        raise RuntimeError("non-order safety event ledger/journal cycle drift")
    return validate_research_ledger_journal(appended, authorization=current)


def exit_decision_context_from_verified_journal(
    journal: Any, /, *, dataset: Any, authorization: Any
) -> (
    ResearchDecisionContextV1
    | NonOrderSafetyEventV1
    | ResearchLedgerControlEventV1
):
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    active = validate_research_ledger_journal(journal, authorization=current)
    position = active.tip_ledger.position
    required = {
        "contract", "source_neutral_contract_id", "entry_fill_price_micros",
        "entry_fee_micros", "opened_at_ns", "entry_intent_id",
        "entry_decision_sha256", "entry_example_sha256", "trade_id",
    }
    if type(position) is not dict or set(position) != required:
        raise ValueError("exit context requires one exact replay-owned long position")
    contract_value = position["contract"]
    contract = (
        contract_value
        if type(contract_value) is ContractIdentityV1
        else ContractIdentityV1.from_dict(contract_value)
    )
    last_time = active.tip_ledger.last_decision_time_ns
    if type(last_time) is not int:
        raise ValueError("exit context has no causal entry/fill clock")
    terminal_ns = _terminal_time_ns(active.tip_ledger.session)
    if last_time >= terminal_ns:
        raise ValueError("exit context cannot start after the terminal boundary")
    decision_time_ns = min(last_time + 1_000_000_000, terminal_ns)
    intent_kind = (
        "TERMINAL_SELL" if decision_time_ns == terminal_ns else "ORDINARY_SELL"
    )
    try:
        quote_row = read_verified_research_quote_row(
            current,
            dataset=sealed,
            session=active.tip_ledger.session,
            contract=contract,
            at_or_before_ns=decision_time_ns,
            maximum_age_ms=2_000,
            require_actionable=False,
        )
    except RuntimeError as exc:
        if str(exc) != "verified quote selector found no eligible row":
            raise
        if intent_kind == "TERMINAL_SELL":
            return _terminal_zero_write_down_event(
                journal=active,
                authorization_sha256=authorization_sha256,
                fill_law=research_fill_law_from_preregistration(
                    prereg.preregistration_payload()[0]
                ),
                quote=None,
            )
        return _position_clock_advance_event(
            journal=active,
            decision_time_ns=decision_time_ns,
            reason_code="PATHD_RESEARCH_HOLD_NO_ELIGIBLE_BBO",
        )
    quote = seal_research_quote(quote_row)
    _validate_selected_source_binding(
        quote_row, dataset=sealed, session=active.tip_ledger.session
    )
    if quote.session != active.tip_ledger.session or quote.contract != contract:
        raise ValueError("exit selector crossed the held contract/session")
    if not quote.actionable:
        if intent_kind == "TERMINAL_SELL":
            return _terminal_zero_write_down_event(
                journal=active,
                authorization_sha256=authorization_sha256,
                fill_law=research_fill_law_from_preregistration(
                    prereg.preregistration_payload()[0]
                ),
                quote=quote,
            )
        return _position_clock_advance_event(
            journal=active,
            decision_time_ns=decision_time_ns,
            reason_code=(
                "PATHD_RESEARCH_HOLD_NON_ACTIONABLE_"
                + str(quote.invalid_reason)
            ),
        )
    spx = _verified_spx_row(
        read_verified_official_spx_row(
            current,
            dataset=sealed,
            session=active.tip_ledger.session,
            at_or_before_ns=decision_time_ns,
            maximum_age_ms=90_000,
        ),
        session=active.tip_ledger.session,
        at_or_before_ns=decision_time_ns,
        maximum_age_ms=90_000,
    )
    _validate_selected_source_binding(
        spx, dataset=sealed, session=active.tip_ledger.session
    )
    if not quote.actionable or quote.contract != contract:
        raise ValueError("exit context does not contain an actionable held-contract BBO")
    semantic = {
        "schema_version": ResearchDecisionContextV1.SCHEMA_VERSION,
        "intent_kind": intent_kind,
        "session": active.tip_ledger.session,
        "contract": contract,
        "decision_time_ns": decision_time_ns,
        "event_interval_end_ns": quote.represented_interval_end_ns,
        "option_watermark_ns": quote.ts_recv_ns,
        "spx_watermark_ns": spx.ts_recv_ns,
        "authorization_sha256": authorization_sha256,
        "dataset_sha256": sealed.dataset_sha256,
        "entry_example_sha256": position["entry_example_sha256"],
        "entry_action_decision_sha256": position["entry_decision_sha256"],
        "option_source_receipt_sha256": quote.source_receipt_sha256,
        "spx_source_receipt_sha256": spx.source_receipt_sha256,
        "decision_quote": quote,
        "official_spx_row": spx,
    }
    return ResearchDecisionContextV1(
        **semantic, context_sha256=prereg.stable_hash(_plain(semantic))
    )


def _governor_config(
    *, intent_kind: str, session_start_equity_micros: int
) -> GovernorConfigV1:
    if type(session_start_equity_micros) is not int or session_start_equity_micros <= 0:
        raise ValueError("research governor requires positive session-start equity")
    common = {
        "max_quantity": 1,
        "max_open_positions": 1,
        "max_daily_loss_micros": math.floor(
            0.05 * session_start_equity_micros
        ),
        "max_intent_age_ms": 2_000,
    }
    if intent_kind == "BUY":
        return GovernorConfigV1(
            **common, max_feed_age_ms=90_000, daily_loss_blocks_close=True
        )
    if intent_kind == "ORDINARY_SELL":
        return GovernorConfigV1(
            **common, max_feed_age_ms=2_000, daily_loss_blocks_close=False
        )
    if intent_kind == "TERMINAL_SELL":
        return GovernorConfigV1(
            **common,
            max_feed_age_ms=2_000,
            daily_loss_blocks_close=False,
            allowed_forced_flat_reference_vendors=("DATABENTO_OPRA",),
        )
    raise ValueError("unregistered research intent kind")


def _broker_snapshot(
    *, context: ResearchDecisionContextV1, journal: ResearchLedgerJournalV1
) -> BrokerStateSnapshotV1:
    tip = journal.tip_ledger
    position = tip.position
    snapshot_semantic = {
        "policy_id": tip.policy_id,
        "fee_path": tip.fee_path,
        "session": tip.session,
        "decision_time_ns": context.decision_time_ns,
        "cash_micros": tip.cash_micros,
        "realized_session_pnl_micros": tip.realized_session_pnl_micros,
        "position_or_null": _plain(position),
        "pending_intent_id_or_null": tip.pending_intent_id,
    }
    positions: tuple[PositionV1, ...] = ()
    if position is not None:
        if type(position) is not dict:
            raise ValueError("research ledger position is not replay-owned")
        contract_value = position.get("contract")
        contract = (
            contract_value
            if type(contract_value) is ContractIdentityV1
            else ContractIdentityV1.from_dict(contract_value)
        )
        positions = (
            PositionV1(
                osi_symbol=contract.osi_symbol,
                quantity=1,
                average_cost_micros=position["entry_fill_price_micros"],
                opened_at_utc=_utc_from_ns(position["opened_at_ns"]),
            ),
        )
    return BrokerStateSnapshotV1(
        schema_version="pathd.broker_state_snapshot.v1",
        snapshot_version=prereg.stable_hash(snapshot_semantic),
        captured_at_utc=_utc_from_ns(context.decision_time_ns),
        account_id_redacted="PATHD-TIER-S-OFFLINE-***",
        available_funds_micros=max(0, tip.cash_micros),
        daily_pnl_micros=tip.realized_session_pnl_micros,
        open_positions=positions,
        connectivity="CONNECTED",
        source="FAKE_GATEWAY",
    )


def _feed_health(context: ResearchDecisionContextV1) -> FeedHealthV1:
    return FeedHealthV1(
        option_received_timestamp_utc=_utc_from_ns(context.option_watermark_ns),
        spx_received_timestamp_utc=_utc_from_ns(context.spx_watermark_ns),
        option_feed_available=True,
        spx_feed_available=True,
    )


def _prepared_semantic(value: ResearchPreparedExecutionV1) -> dict[str, Any]:
    return {
        field.name: _plain(getattr(value, field.name))
        for field in fields(value)
        if field.name != "prepared_sha256"
    }


def prepare_research_execution(
    *,
    intent_kind: str,
    context: Any,
    journal: Any,
    dataset: Any,
    authorization: Any,
    law: Any,
    reason_code: str,
) -> ResearchPreparedExecutionV1:
    current, sealed, authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    fill_law = _validate_fill_law(law)
    active = validate_research_ledger_journal(journal, authorization=current)
    supplied = _validate_context_seal(context)
    if (
        supplied.intent_kind != intent_kind
        or supplied.authorization_sha256 != authorization_sha256
        or supplied.dataset_sha256 != sealed.dataset_sha256
        or supplied.session != active.tip_ledger.session
        or supplied.decision_time_ns < (active.tip_ledger.last_decision_time_ns or 0)
    ):
        raise ValueError("prepared execution context/ledger binding drift")
    decision: EntryActionDecisionV1 | None = None
    if intent_kind == "BUY":
        matches = [
            row for row in sealed.examples
            if row.canonical_sha256() == supplied.entry_example_sha256
        ]
        if len(matches) != 1:
            raise ValueError("prepared BUY example binding drift")
        decisions = [
            row.event_payload for row in active.transitions
            if row.event_schema_version == EntryActionDecisionV1.SCHEMA_VERSION
            and row.event_payload.get("decision_sha256")
            == supplied.entry_action_decision_sha256
        ]
        if len(decisions) != 1:
            raise ValueError("prepared BUY decision is absent from the journal")
        decision = EntryActionDecisionV1(**decisions[0])
        _validate_action_decision(
            decision,
            dataset=sealed,
            authorization_sha256=authorization_sha256,
            example=matches[0],
        )
        expected_context = entry_decision_context_from_verified_example(
            matches[0], dataset=sealed, authorization=current, decision=decision
        )
        if reason_code != decision.reason:
            raise ValueError("prepared BUY reason differs from the composed decision")
    else:
        expected_context = exit_decision_context_from_verified_journal(
            active, dataset=sealed, authorization=current
        )
        if intent_kind == "TERMINAL_SELL" and reason_code != "PATHD_RESEARCH_TERMINAL":
            raise ValueError("terminal replay reason drift")
        if type(reason_code) is not str or not reason_code:
            raise ValueError("research SELL reason must be nonempty")
    if supplied != expected_context:
        raise ValueError("research context is not independently reconstructible")
    config = _governor_config(
        intent_kind=intent_kind,
        session_start_equity_micros=active.tip_ledger.session_start_equity_micros,
    )
    governor = DeterministicGovernor(config)
    broker = _broker_snapshot(context=supplied, journal=active)
    feed = _feed_health(supplied)
    now_utc = _utc_from_ns(supplied.decision_time_ns)
    quote = supplied.decision_quote
    if intent_kind == "TERMINAL_SELL":
        position = active.tip_ledger.position
        if type(position) is not dict:
            raise ValueError("terminal replay requires a held long")
        lifecycle = LifecycleStateV1(
            contract=supplied.contract,
            position_snapshot_version=broker.snapshot_version,
            entry_bid_micros=position["entry_fill_price_micros"],
            running_max_bid_micros=max(
                position["entry_fill_price_micros"], quote.bid_micros
            ),
            current_bid_micros=quote.bid_micros,
            current_ask_micros=quote.ask_micros,
            opened_at_utc=_utc_from_ns(position["opened_at_ns"]),
            feature_contract_version="pathd.research_decision_context.v1",
            feature_snapshot_sha256="sha256:" + supplied.context_sha256,
        )
        intent = governor.forced_flat_intent(
            lifecycle,
            feed_health=feed,
            now_utc=now_utc,
            reference_vendor="DATABENTO_OPRA",
            reference_bid_micros=quote.bid_micros,
            reference_ask_micros=quote.ask_micros,
            reason_code=reason_code,
        )
    else:
        if intent_kind == "BUY":
            assert decision is not None
            directive = DecisionDirectiveV1(
                action="OPEN_LONG", side="BUY", position_effect="OPEN",
                urgency="NORMAL_ENTRY", quantity=1, reason_code=reason_code,
            )
            hard_limit = decision.buy_hard_limit_micros
            adverse = hard_limit - quote.ask_micros
            origin = "MODEL"
            producer = ProducerIdentityV1(
                component_version="pathd.entry-composer.v1",
                strategy_id=active.policy_id,
                artifact_sha256="sha256:" + decision.composer_sha256,
                feature_contract_version="pathd.entry-signed17.v1",
                feature_snapshot_sha256="sha256:" + decision.model_input_sha256,
            )
            expected_position = "FLAT"
            held_symbol = None
            valid_until_ns = supplied.decision_time_ns + 60_000_000_000
        else:
            directive = DecisionDirectiveV1(
                action="CLOSE_LONG", side="SELL", position_effect="CLOSE",
                urgency=(
                    "PROTECTIVE_EXIT"
                    if "FLOOR" in reason_code or "PROTECT" in reason_code
                    else "NORMAL_EXIT"
                ),
                quantity=1, reason_code=reason_code,
            )
            adverse = option_tick_micros(quote.bid_micros)
            hard_limit = max(0, quote.bid_micros - adverse)
            origin = "DETERMINISTIC_EXIT"
            producer = ProducerIdentityV1(
                component_version="pathd.research-exit.v1",
                strategy_id=active.policy_id,
                artifact_sha256=None,
                feature_contract_version="pathd.research_decision_context.v1",
                feature_snapshot_sha256="sha256:" + supplied.context_sha256,
            )
            expected_position = "LONG_ONE"
            held_symbol = supplied.contract.osi_symbol
            valid_until_ns = (
                supplied.decision_time_ns + fill_law.headline_delay_ms * 1_000_000
            )
        intent = ExecutionIntentV1.create(
            trace_id=f"pathd-research-{supplied.context_sha256[:24]}",
            parent_intent_id=None,
            origin=origin,
            producer=producer,
            decision=directive,
            contract=supplied.contract,
            price_budget=PriceBudgetV1(
                unit="USD_OPTION_PRICE_MICROS",
                reference_vendor="DATABENTO_OPRA",
                reference_bid_micros=quote.bid_micros,
                reference_ask_micros=quote.ask_micros,
                max_adverse_move_micros=adverse,
                hard_limit_micros=hard_limit,
            ),
            clocks=IntentClocksV1(
                decision_clock="received_timestamp_utc",
                event_interval_end_utc=_utc_from_ns(
                    supplied.event_interval_end_ns
                ),
                option_received_watermark_utc=_utc_from_ns(
                    supplied.option_watermark_ns
                ),
                spx_received_watermark_utc=_utc_from_ns(
                    supplied.spx_watermark_ns
                ),
                decision_available_at_utc=now_utc,
                model_started_at_utc=now_utc,
                model_finished_at_utc=now_utc,
                intent_emitted_at_utc=now_utc,
                valid_until_utc=_utc_from_ns(valid_until_ns),
            ),
            state_precondition=PositionPreconditionV1(
                expected_position=expected_position,
                position_snapshot_version=broker.snapshot_version,
                held_osi_symbol=held_symbol,
            ),
            execution_profile_version="pathd.governed-limit-profile.v1",
        )
    governor_authorization = governor.evaluate(
        intent, broker_state=broker, feed_health=feed, now_utc=now_utc
    )
    if governor_authorization.disposition != "ALLOW":
        raise RuntimeError(
            "research intent blocked by deterministic governor: "
            + ",".join(governor_authorization.reason_codes)
        )
    semantic = {
        "schema_version": ResearchPreparedExecutionV1.SCHEMA_VERSION,
        "intent": intent,
        "broker_state": broker,
        "feed_health": feed,
        "authorization": governor_authorization,
        "decision_context": supplied,
        "decision_context_sha256": supplied.context_sha256,
        "decision_quote_sha256": quote.quote_sha256,
        "source_receipt_sha256": quote.source_receipt_sha256,
        "prior_journal": active,
        "prior_journal_root_sha256": active.journal_root_sha256,
        "governor_config_sha256": prereg.stable_hash(_plain(config)),
        "arrival_base_time_ns": supplied.decision_time_ns,
    }
    return ResearchPreparedExecutionV1(
        **semantic, prepared_sha256=prereg.stable_hash(_plain(semantic))
    )


def _fee_per_filled_side(fee_path: int) -> int:
    if fee_path not in (3, 4):
        raise ValueError("research replay fee path drift")
    return fee_path * 500_000


def _prepared_execution_for_delay(
    prepared: ResearchPreparedExecutionV1,
    *,
    delay_ms: int,
) -> ResearchPreparedExecutionV1:
    """Derive the exact ordinary-SELL sensitivity intent and authorization."""

    if prepared.decision_context.intent_kind != "ORDINARY_SELL":
        return prepared
    valid_until_ns = prepared.arrival_base_time_ns + delay_ms * 1_000_000
    clocks = replace(
        prepared.intent.clocks,
        valid_until_utc=_utc_from_ns(valid_until_ns),
    )
    intent = ExecutionIntentV1.create(
        trace_id=prepared.intent.trace_id,
        parent_intent_id=prepared.intent.parent_intent_id,
        origin=prepared.intent.origin,
        producer=prepared.intent.producer,
        decision=prepared.intent.decision,
        contract=prepared.intent.contract,
        price_budget=prepared.intent.price_budget,
        clocks=clocks,
        state_precondition=prepared.intent.state_precondition,
        execution_profile_version=prepared.intent.execution_profile_version,
    )
    config = _governor_config(
        intent_kind="ORDINARY_SELL",
        session_start_equity_micros=(
            prepared.prior_journal.tip_ledger.session_start_equity_micros
        ),
    )
    if prepared.governor_config_sha256 != prereg.stable_hash(_plain(config)):
        raise ValueError("research delay replay governor-config drift")
    authorization = DeterministicGovernor(config).evaluate(
        intent,
        broker_state=prepared.broker_state,
        feed_health=prepared.feed_health,
        now_utc=_utc_from_ns(prepared.arrival_base_time_ns),
    )
    if authorization.disposition != "ALLOW":
        raise RuntimeError(
            "delay-specific research intent blocked by deterministic governor: "
            + ",".join(authorization.reason_codes)
        )
    semantic = {
        **_prepared_semantic(prepared),
        "intent": intent,
        "authorization": authorization,
    }
    return ResearchPreparedExecutionV1(
        **semantic, prepared_sha256=prereg.stable_hash(_plain(semantic))
    )


def replay_research_intent(
    prepared: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
    law: Any,
    delay_ms: int,
) -> ResearchExecutionOutcomeV1:
    current, sealed, _authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    fill_law = _validate_fill_law(law)
    if type(prepared) is not ResearchPreparedExecutionV1:
        raise TypeError("research replay requires ResearchPreparedExecutionV1")
    if (
        prepared.schema_version != prepared.SCHEMA_VERSION
        or prepared.prepared_sha256
        != prereg.stable_hash(_prepared_semantic(prepared))
        or prepared.decision_context_sha256
        != prepared.decision_context.context_sha256
        or prepared.decision_quote_sha256
        != prepared.decision_context.decision_quote.quote_sha256
        or prepared.source_receipt_sha256
        != prepared.decision_context.option_source_receipt_sha256
    ):
        raise ValueError("research prepared execution seal drift")
    prior = validate_research_ledger_journal(
        prepared.prior_journal, authorization=current
    )
    if prepared.prior_journal_root_sha256 != prior.journal_root_sha256:
        raise ValueError("research prepared journal-root drift")
    kind = prepared.decision_context.intent_kind
    if type(delay_ms) is not int:
        raise TypeError("research delay must be an exact integer millisecond")
    if (
        (kind == "BUY" and delay_ms != 60_000)
        or (
            kind == "ORDINARY_SELL"
            and delay_ms not in tuple(fill_law.delay_sensitivity_ms)
        )
        or (kind == "TERMINAL_SELL" and delay_ms != 0)
        or kind not in {"BUY", "ORDINARY_SELL", "TERMINAL_SELL"}
    ):
        raise ValueError("research delay is not a frozen delay rung for this intent")
    rebuilt = prepare_research_execution(
        intent_kind=kind,
        context=prepared.decision_context,
        journal=prior,
        dataset=sealed,
        authorization=current,
        law=fill_law,
        reason_code=prepared.intent.decision.reason_code,
    )
    if rebuilt != prepared:
        raise ValueError("research prepared execution is not reconstructible")
    effective = _prepared_execution_for_delay(prepared, delay_ms=delay_ms)
    if (
        kind == "ORDINARY_SELL"
        and delay_ms == fill_law.headline_delay_ms
        and effective != prepared
    ):
        raise RuntimeError("headline SELL delay failed to reproduce prepared execution")
    arrival_time_ns = effective.arrival_base_time_ns + delay_ms * 1_000_000
    if kind == "ORDINARY_SELL" and arrival_time_ns >= _terminal_time_ns(
        effective.decision_context.session
    ):
        raise ValueError("ordinary SELL arrival cannot consume the terminal boundary")
    query_age = 90_000 if kind == "BUY" else 2_000
    arrival_row = read_verified_research_quote_row(
        current,
        dataset=sealed,
        session=effective.decision_context.session,
        contract=effective.intent.contract,
        at_or_before_ns=arrival_time_ns,
        maximum_age_ms=query_age,
        require_actionable=False,
    )
    if type(arrival_row) is not VerifiedResearchQuoteRowV1:
        raise TypeError("research arrival selector returned a mistyped row")
    arrival_quote = seal_research_quote(arrival_row)
    _validate_selected_source_binding(
        arrival_row,
        dataset=sealed,
        session=effective.decision_context.session,
    )
    if (
        arrival_quote.session != effective.decision_context.session
        or arrival_quote.contract != effective.intent.contract
        or arrival_quote.source_receipt_sha256
        != effective.source_receipt_sha256
        or arrival_row.query_at_or_before_ns != arrival_time_ns
        or arrival_row.query_maximum_age_ms != query_age
        or arrival_row.query_require_actionable is not False
        or arrival_quote.available_at_ns > arrival_time_ns
    ):
        raise ValueError("research arrival quote binding drift")
    if kind == "TERMINAL_SELL":
        if arrival_quote != effective.decision_context.decision_quote:
            raise ValueError("terminal replay must reuse the exact boundary BBO")
    elif kind == "ORDINARY_SELL" and delay_ms == 0:
        if arrival_quote != effective.decision_context.decision_quote:
            raise ValueError("zero-delay SELL must reuse the exact decision BBO")
    elif (
        arrival_quote.represented_interval_end_ns
        <= effective.decision_context.event_interval_end_ns
    ):
        raise ValueError("research arrival quote is not strictly post-decision")
    quote_tape = (
        (
            SimulatedQuote(
                offset_ms=delay_ms,
                bid_micros=arrival_quote.bid_micros,
                ask_micros=arrival_quote.ask_micros,
            ),
        )
        if arrival_quote.actionable
        else ()
    )
    executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(
            _datetime_from_ns(effective.arrival_base_time_ns)
        ),
        quote_tape=quote_tape,
        scenario=ExecutionScenario(
            outcome="FULL",
            submit_latency_ms=delay_ms,
            acknowledge_latency_ms=0,
            cancel_after_ms=0,
            cancel_confirm_latency_ms=0,
            fill_price_mode=fill_law.fill_price_mode,
        ),
    )
    events = tuple(executor.submit(effective.intent, effective.authorization))
    if not events:
        raise RuntimeError("research simulator emitted no execution transcript")
    fill_decision = evaluate_research_arrival(
        effective.intent, arrival=arrival_quote, law=fill_law
    )
    filled_quantity = max(event.filled_quantity for event in events)
    fill_price = next(
        (
            event.fill_price_micros for event in reversed(events)
            if event.filled_quantity == filled_quantity and event.fill_price_micros is not None
        ),
        None,
    )
    if (
        filled_quantity not in (0, 1)
        or (filled_quantity == 1) is not fill_decision.filled
        or fill_price != fill_decision.fill_price_micros
    ):
        raise ValueError("simulator transcript differs from the shared fill law")
    tip = prior.tip_ledger
    position_before = tip.position
    position_after = position_before
    cash_delta = 0
    fee_micros = 0
    realized = tip.realized_session_pnl_micros
    if filled_quantity == 1:
        assert fill_price is not None
        fee_micros = _fee_per_filled_side(tip.fee_path)
        if effective.intent.decision.side == "BUY":
            if position_before is not None:
                raise ValueError("research BUY filled while already occupied")
            decision = effective.decision_context
            cash_delta = -(fill_price * effective.intent.contract.multiplier + fee_micros)
            realized -= fee_micros
            position_after = {
                "contract": _plain(effective.intent.contract),
                "source_neutral_contract_id": next(
                    row.event_payload["selected_source_neutral_contract_id"]
                    for row in reversed(prior.transitions)
                    if row.event_schema_version == EntryActionDecisionV1.SCHEMA_VERSION
                    and row.event_payload.get("decision_sha256")
                    == decision.entry_action_decision_sha256
                ),
                "entry_fill_price_micros": fill_price,
                "entry_fee_micros": fee_micros,
                "opened_at_ns": arrival_time_ns,
                "entry_intent_id": effective.intent.intent_id,
                "entry_decision_sha256": decision.entry_action_decision_sha256,
                "entry_example_sha256": decision.entry_example_sha256,
                "trade_id": effective.intent.intent_id,
            }
        else:
            if type(position_before) is not dict:
                raise ValueError("research SELL filled without a held replay position")
            cash_delta = fill_price * effective.intent.contract.multiplier - fee_micros
            realized += (
                (fill_price - position_before["entry_fill_price_micros"])
                * effective.intent.contract.multiplier
                - fee_micros
            )
            position_after = None
    state_values = _ledger_semantic(tip)
    state_values.update(
        {
            "cash_micros": tip.cash_micros + cash_delta,
            "realized_session_pnl_micros": realized,
            "position": position_after,
            "pending_intent_id": None,
            "sequence": tip.sequence + 1,
            "last_decision_time_ns": arrival_time_ns,
            "prior_transition_sha256": prior.transitions[-1].transition_sha256,
        }
    )
    result_ledger = _seal_ledger(**state_values)
    result_semantic = {
        "schema_version": ResearchExecutionResultV1.SCHEMA_VERSION,
        "prepared_execution_sha256": effective.prepared_sha256,
        "decision_quote_sha256": effective.decision_quote_sha256,
        "arrival_quote_sha256": arrival_quote.quote_sha256,
        "source_receipt_sha256": arrival_quote.source_receipt_sha256,
        "delay_ms": delay_ms,
        "events": events,
        "final_state": events[-1].state_to,
        "filled_quantity": filled_quantity,
        "fill_price_micros": fill_price,
        "fee_micros": fee_micros,
        "cash_delta_micros": cash_delta,
        "position_before": position_before,
        "position_after": position_after,
        "prior_ledger_sha256": tip.ledger_sha256,
        "result_ledger": result_ledger,
    }
    result = ResearchExecutionResultV1(
        **result_semantic,
        result_sha256=prereg.stable_hash(_plain(result_semantic)),
    )
    result_journal = _append_event(
        prior,
        event_payload=_plain(result),
        event_schema_version=result.SCHEMA_VERSION,
        event_hash=result.result_sha256,
        event_time_ns=arrival_time_ns,
        transition_kind="EXECUTION_RESULT",
        next_state_changes={
            "cash_micros": result_ledger.cash_micros,
            "realized_session_pnl_micros": result_ledger.realized_session_pnl_micros,
            "position": result_ledger.position,
            "pending_intent_id": None,
            "last_decision_time_ns": arrival_time_ns,
        },
    )
    if result_journal.tip_ledger != result_ledger:
        raise RuntimeError("research execution result ledger/journal cycle drift")
    outcome_semantic = {
        "schema_version": ResearchExecutionOutcomeV1.SCHEMA_VERSION,
        "execution_result": result,
        "result_journal": result_journal,
    }
    outcome = ResearchExecutionOutcomeV1(
        **outcome_semantic,
        outcome_sha256=prereg.stable_hash(_plain(outcome_semantic)),
    )
    return validate_research_execution_outcome(
        outcome, authorization=current, dataset=sealed, law=fill_law
    )


def validate_research_execution_outcome(
    outcome: Any, /, *, authorization: Any, dataset: Any, law: Any
) -> ResearchExecutionOutcomeV1:
    current, sealed, _authorization_sha256 = _validate_dataset_authority(
        dataset, authorization
    )
    _validate_fill_law(law)
    if type(outcome) is not ResearchExecutionOutcomeV1:
        raise TypeError("research execution outcome is mistyped")
    semantic = {
        "schema_version": outcome.schema_version,
        "execution_result": _plain(outcome.execution_result),
        "result_journal": _plain(outcome.result_journal),
    }
    if outcome.schema_version != outcome.SCHEMA_VERSION or outcome.outcome_sha256 != prereg.stable_hash(semantic):
        raise ValueError("research execution outcome seal drift")
    result = outcome.execution_result
    if type(result) is not ResearchExecutionResultV1:
        raise TypeError("research execution result is mistyped")
    result_semantic = {
        field.name: _plain(getattr(result, field.name))
        for field in fields(result)
        if field.name != "result_sha256"
    }
    if (
        result.schema_version != result.SCHEMA_VERSION
        or result.result_sha256 != prereg.stable_hash(result_semantic)
    ):
        raise ValueError("research execution result seal drift")
    active = validate_research_ledger_journal(
        outcome.result_journal, authorization=current
    )
    last = active.transitions[-1]
    if (
        last.event_schema_version != ResearchExecutionResultV1.SCHEMA_VERSION
        or last.event_sha256 != result.result_sha256
        or last.event_payload != _plain(result)
        or active.tip_ledger != result.result_ledger
        or _exact_hash(
            result.source_receipt_sha256, name="execution source receipt"
        )
        != result.source_receipt_sha256
    ):
        raise ValueError("research execution result journal/dataset binding drift")
    return outcome


def _current_preregistration_sha256() -> str:
    if prereg.PREREG_PATH.exists():
        return prereg.sha256_path(prereg.PREREG_PATH)
    return prereg.stable_hash(prereg.preregistration_payload()[0])


def seal_research_result(
    *, authorization: Any, dataset: Any, fill_law: Any, payload: Any
) -> ResearchResultEnvelopeV1:
    current = assert_entry_evidence_authorization_current(authorization)
    validate_entry_evidence_dataset(dataset, authorization=current)
    prereg._assert_strict_json_value(payload)
    if type(payload) is not dict:
        raise TypeError("research result payload must be an exact mapping")
    if getattr(fill_law, "fill_law_hash", None) != prereg.fill_law()["fill_law_hash"]:
        raise ValueError("research result fill law drift")
    from v4.research.pathd_holdout_gate import inspect_protected_holdout_state

    holdout_count = inspect_protected_holdout_state().holdout_open_count
    if type(holdout_count) is not int or holdout_count != 0:
        raise RuntimeError("research result blocked after holdout access")
    return ResearchResultEnvelopeV1(
        schema_version=ResearchResultEnvelopeV1.SCHEMA_VERSION,
        authorization_sha256=prereg.stable_hash(current.to_dict()),
        sessions_sha256_newline=current.sessions_sha256_newline,
        dataset_sha256=dataset.dataset_sha256,
        preregistration_sha256=_current_preregistration_sha256(),
        plan_sha256=prereg.sha256_path(prereg.PLAN_PATH),
        fill_law_hash=fill_law.fill_law_hash,
        quarantine_labels=tuple(prereg.QUARANTINE_LABELS),
        claim_boundary=prereg.CLAIM_BOUNDARY,
        holdout_caveat=prereg.HOLDOUT_CAVEAT,
        holdout_open_count=0,
        payload_sha256=prereg.stable_hash(payload),
        payload=dict(payload),
    )


def validate_research_result(
    envelope: Any,
    /,
    *,
    authorization: Any,
    dataset: Any,
    fill_law: Any,
) -> ResearchResultEnvelopeV1:
    current = assert_entry_evidence_authorization_current(authorization)
    validate_entry_evidence_dataset(dataset, authorization=current)
    if type(envelope) is not ResearchResultEnvelopeV1:
        raise TypeError("research result envelope is mistyped")
    prereg._assert_strict_json_value(envelope.payload)
    expected = seal_research_result(
        authorization=current, dataset=dataset, fill_law=fill_law, payload=envelope.payload
    )
    if envelope != expected:
        raise ValueError("research result envelope drift")
    return envelope


__all__ = [
    "EntryActionCalibrationObservationV1",
    "EntryTime300TrajectoryEvidenceV1",
    "EntrySurvivalAuditRecordV1",
    "EntryActionObservationV1",
    "EntryFrameCoverageV1",
    "ResearchExecutionResultV1",
    "ResearchDecisionContextV1",
    "ResearchLedgerStateV1",
    "ResearchLedgerControlEventV1",
    "ResearchLedgerTransitionV1",
    "ResearchLedgerJournalV1",
    "ResearchPreparedExecutionV1",
    "ResearchResultEnvelopeV1",
    "NonOrderSafetyEventV1",
    "ResearchExecutionOutcomeV1",
    "replay_research_intent",
    "prepare_research_execution",
    "entry_decision_context_from_verified_example",
    "exit_decision_context_from_verified_journal",
    "genesis_research_ledger_journal",
    "append_research_action_decision",
    "entry_frame_coverage_from_verified_example",
    "append_entry_frame_coverage",
    "entry_action_observation_from_verified_dataset",
    "append_entry_action_observation",
    "append_research_non_order_safety_event",
    "mark_research_session_terminal",
    "advance_research_ledger_session",
    "advance_research_position_clock",
    "append_research_position_clock_event",
    "handoff_research_ledger_authorization",
    "validate_research_ledger_journal",
    "validate_research_ledger_journal_against_dataset",
    "reconstruct_entry_policy_economics_from_validated_transitions",
    "reconstruct_entry_policy_evaluation_from_journal",
    "validate_research_execution_outcome",
    "seal_research_result",
    "validate_research_result",
]
