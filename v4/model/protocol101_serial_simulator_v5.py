"""Owner-signed Protocol101 serial simulator v5.

Version 5 prices PnL from the selected source quote while releasing account
occupancy only at the separately persisted realized exit time.  The complete
candidate stream is validated before hashes or economic output are produced.
Simulator v4 remains untouched for historical audit.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from v4.model.protocol101_regimen_repair import (
    ExitReason,
    InvalidReason,
    Protocol101DuplicateCanonicalSlotError,
    Protocol101DuplicateContractIdentityError,
    Protocol101DuplicateDecisionIdentityError,
    Protocol101RegimenRepairError,
    ny_session_for_ns,
)


PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION = (
    "protocol101_serial_simulator_v5_label_realized_exit_"
    "account_continuity_fee_reserve"
)
PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION = (
    "protocol101_serial_simulator_v4_account_continuity_fee_reserve"
)
EXIT_TIME_SEMANTICS = "label_realized_exit_time"
DAILY_LOSS_BASIS = "raw_realized_net_pnl_at_occupancy_exit"
CASH_BASIS = "raw_realized_net_pnl_at_occupancy_exit"
STRESS_APPLICATION = "metrics_only"
ACCOUNT_CONTINUITY = "cash_compounds_across_sessions_within_split"
NY = ZoneInfo("America/New_York")


class Protocol101ReplayContractError(Protocol101RegimenRepairError):
    blocker_code = "P101_REPLAY_CONTRACT_ERROR"


class Protocol101InvalidRealizedExitError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_INVALID_REALIZED_EXIT"


class Protocol101MissingRealizedExitError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_MISSING_REALIZED_EXIT"


class Protocol101CrossSessionExitError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_CROSS_SESSION_EXIT"


class Protocol101ExitAfterDeadlineError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_EXIT_AFTER_DEADLINE"


class Protocol101LegacySyntheticExitArtifactError(
    Protocol101ReplayContractError
):
    blocker_code = "P101_REPLAY_LEGACY_SYNTHETIC_EXIT_ARTIFACT"


class Protocol101PnlSourceQuoteMismatchError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_PNL_SOURCE_QUOTE_MISMATCH"


class Protocol101QuoteAgeMismatchError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_QUOTE_AGE_MISMATCH"


class Protocol101MixedSimulatorVersionError(Protocol101ReplayContractError):
    blocker_code = "P101_REPLAY_MIXED_SIMULATOR_VERSIONS"


@dataclass(frozen=True)
class SerialSimulatorV5Config:
    starting_cash: float = 10_000.0
    contract_multiplier: float = 100.0
    campaign_round_trip_fee_dollars: float = 3.0
    affordability_reserve_per_trade: float = 3.0
    max_trades_per_session: int = 0
    max_daily_loss_fraction_of_session_start_equity: float = 0.05
    stress_per_trade_dollars: float = 0.0
    enforce_affordability: bool = True
    no_new_entries_after_et: str = "15:30"
    forced_flat_before_et: str = "15:55"
    split_order: tuple[str, ...] = ()
    lexical_split_order_declared: bool = True
    simulator_version: str = PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    exit_time_semantics: str = EXIT_TIME_SEMANTICS
    daily_loss_basis: str = DAILY_LOSS_BASIS
    cash_basis: str = CASH_BASIS
    stress_application: str = STRESS_APPLICATION
    account_continuity: str = ACCOUNT_CONTINUITY

    def semantics(self) -> dict[str, Any]:
        return {
            "simulator_version": self.simulator_version,
            "exit_time_semantics": self.exit_time_semantics,
            "starting_cash": float(self.starting_cash),
            "contract_multiplier": float(self.contract_multiplier),
            "campaign_round_trip_fee_dollars": float(
                self.campaign_round_trip_fee_dollars
            ),
            "affordability_reserve_per_trade": float(
                self.affordability_reserve_per_trade
            ),
            "max_trades_per_session": int(self.max_trades_per_session),
            "max_daily_loss_fraction_of_session_start_equity": float(
                self.max_daily_loss_fraction_of_session_start_equity
            ),
            "stress_per_trade_dollars": float(
                self.stress_per_trade_dollars
            ),
            "enforce_affordability": bool(self.enforce_affordability),
            "no_new_entries_after": self.no_new_entries_after_et,
            "forced_flat_before": self.forced_flat_before_et,
            "split_order": list(self.split_order),
            "lexical_split_order_declared": bool(
                self.lexical_split_order_declared
            ),
            "daily_loss_basis": self.daily_loss_basis,
            "cash_basis": self.cash_basis,
            "stress_application": self.stress_application,
            "account_continuity": self.account_continuity,
        }


@dataclass(frozen=True)
class SerialCandidateV5:
    split: str
    session: str
    decision_time_ns: int
    contract_id: str
    right: str
    canonical_strike_slot: int
    policy_index: int
    entry_ask: float
    score: float
    raw_label_pnl_after_campaign_fee: float
    label_mid_pnl_before_campaign_fee: float
    label_realized_exit_time_ns: int
    label_source_exit_quote_time_ns: int
    label_exit_quote_age_ms: float
    label_exit_reason_code: int
    label_executable_exit_bid: float
    label_policy_deadline_ns: int
    label_invalid_reason_code: int = int(InvalidReason.NONE)
    feature_hash: str = ""
    source_quote_time_ns: int = 0
    source_context_time_ns: int = 0
    strategy: str = ""
    source_simulator_version: str = PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    fold: str = ""


@dataclass(frozen=True)
class SerialReplayTradeV5:
    split: str
    fold: str
    session: str
    decision_time_ns: int
    contract_id: str
    right: str
    canonical_strike_slot: int
    policy_index: int
    entry_ask: float
    premium_at_risk: float
    premium_plus_fee_required: float
    score: float
    raw_label_pnl_after_campaign_fee: float
    stressed_pnl: float
    label_realized_exit_time_ns: int
    label_source_exit_quote_time_ns: int
    label_exit_quote_age_ms: float
    label_exit_reason_code: int
    label_executable_exit_bid: float
    label_policy_deadline_ns: int
    cash_before: float
    cash_after: float
    session_start_equity: float
    session_realized_pnl_before: float
    session_realized_pnl_after: float
    feature_hash: str
    source_quote_time_ns: int
    source_context_time_ns: int
    strategy: str
    simulator_version: str
    exit_time_semantics: str
    simulator_config_hash: str
    candidate_stream_hash: str
    candidate_payload_hash: str


@dataclass(frozen=True)
class SerialSkippedEventV5:
    candidate_identity: tuple[Any, ...]
    reason_code: str
    decision_time_ns: int
    pending_contract_id_or_null: str | None
    pending_source_quote_time_ns_or_null: int | None
    pending_realized_exit_time_ns_or_null: int | None
    pending_exit_quote_age_ms_or_null: float | None
    cash: float
    session_start_equity: float
    session_realized_pnl: float
    premium_plus_fee_required: float
    candidate_stream_hash: str
    candidate_payload_hash: str
    simulator_version: str


@dataclass(frozen=True)
class SerialReplayStateV5:
    skipped: dict[str, int]
    skipped_events: tuple[SerialSkippedEventV5, ...]
    cash_by_account: dict[str, float]
    session_start_equity: dict[str, float]
    realized_raw_pnl_by_split_session: dict[str, float]
    equity_events_by_account: dict[str, tuple[dict[str, Any], ...]]
    semantics: dict[str, Any]
    simulator_config_hash: str
    candidate_stream_hash: str
    candidate_payload_hash: str
    trade_identity_hash: str


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Protocol101ReplayContractError(
                "nonfinite value cannot enter canonical simulator hash",
                boundary="candidate hashing",
            )
        return value
    return _canonical(asdict(value)) if hasattr(value, "__dataclass_fields__") else str(value)


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_stream_payload(
    candidates: Iterable[SerialCandidateV5],
) -> list[dict[str, Any]]:
    return [
        {
            "split": item.split,
            "session": item.session,
            "decision_time_ns": int(item.decision_time_ns),
            "contract_id": item.contract_id,
            "right": item.right,
            "canonical_strike_slot": int(item.canonical_strike_slot),
            "policy_index": int(item.policy_index),
        }
        for item in candidates
    ]


def candidate_payload(
    candidates: Iterable[SerialCandidateV5],
) -> list[dict[str, Any]]:
    return [
        {
            **candidate_stream_payload([item])[0],
            "entry_ask": float(item.entry_ask),
            "score": float(item.score),
            "raw_label_pnl_after_campaign_fee": float(
                item.raw_label_pnl_after_campaign_fee
            ),
            "label_mid_pnl_before_campaign_fee": float(
                item.label_mid_pnl_before_campaign_fee
            ),
            "label_realized_exit_time_ns": int(
                item.label_realized_exit_time_ns
            ),
            "label_source_exit_quote_time_ns": int(
                item.label_source_exit_quote_time_ns
            ),
            "label_exit_quote_age_ms": float(item.label_exit_quote_age_ms),
            "label_exit_reason_code": int(item.label_exit_reason_code),
            "label_executable_exit_bid": float(
                item.label_executable_exit_bid
            ),
            "label_policy_deadline_ns": int(item.label_policy_deadline_ns),
            "label_invalid_reason_code": int(item.label_invalid_reason_code),
            "feature_hash": item.feature_hash,
            "source_quote_time_ns": int(item.source_quote_time_ns),
            "source_context_time_ns": int(item.source_context_time_ns),
            "strategy": item.strategy,
            "metadata": item.metadata,
        }
        for item in candidates
    ]


def trade_identity_payload(
    trades: Iterable[SerialReplayTradeV5],
) -> list[dict[str, Any]]:
    return [
        {
            "split": item.split,
            "fold": item.fold,
            "session": item.session,
            "decision_time_ns": int(item.decision_time_ns),
            "contract_id": item.contract_id,
            "policy_index": int(item.policy_index),
            "label_source_exit_quote_time_ns": int(
                item.label_source_exit_quote_time_ns
            ),
            "label_realized_exit_time_ns": int(
                item.label_realized_exit_time_ns
            ),
        }
        for item in trades
    ]


def _identity(candidate: SerialCandidateV5) -> tuple[Any, ...]:
    return (
        candidate.split,
        candidate.session,
        int(candidate.decision_time_ns),
        candidate.contract_id,
        int(candidate.policy_index),
    )


def _split_rank(
    split: str,
    config: SerialSimulatorV5Config,
) -> tuple[int, str]:
    if config.split_order:
        if split not in config.split_order:
            raise Protocol101ReplayContractError(
                f"split absent from explicit order: {split}",
                canonical_key=split,
                boundary="candidate preflight ordering",
            )
        return config.split_order.index(split), split
    if not config.lexical_split_order_declared:
        raise Protocol101ReplayContractError(
            "split order is neither explicit nor declared lexical",
            boundary="candidate preflight ordering",
        )
    return 0, split


def _assert_candidate_identities(candidates: list[SerialCandidateV5]) -> None:
    decisions: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    contracts: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    slots: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for index, item in enumerate(candidates):
        locator = ("candidate_index", index, _identity(item))
        decision_key = (
            item.split,
            item.session,
            int(item.decision_time_ns),
        )
        contract_key = (*decision_key, item.contract_id)
        slot_key = (
            *decision_key,
            int(item.canonical_strike_slot),
            item.right,
        )
        for key, seen, error_type in (
            (
                decision_key,
                decisions,
                Protocol101DuplicateDecisionIdentityError,
            ),
            (
                contract_key,
                contracts,
                Protocol101DuplicateContractIdentityError,
            ),
            (
                slot_key,
                slots,
                Protocol101DuplicateCanonicalSlotError,
            ),
        ):
            if key in seen:
                raise error_type(
                    f"{error_type.blocker_code}: duplicate selected intent",
                    canonical_key=key,
                    first_source_locator=seen[key],
                    duplicate_source_locator=locator,
                    observed_count=2,
                    boundary="serial replay preflight before hashing",
                )
            seen[key] = locator


def _forced_flat_ns(session: str, config: SerialSimulatorV5Config) -> int:
    hour, minute = [int(value) for value in config.forced_flat_before_et.split(":")]
    local = datetime.fromisoformat(session).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
        tzinfo=NY,
    )
    return int(local.timestamp()) * 1_000_000_000


def _preflight_candidate(
    candidate: SerialCandidateV5,
    config: SerialSimulatorV5Config,
) -> None:
    key = _identity(candidate)
    if candidate.source_simulator_version != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        if candidate.source_simulator_version == PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION:
            raise Protocol101LegacySyntheticExitArtifactError(
                "v4 synthetic-exit artifact cannot enter repaired evidence",
                canonical_key=key,
                boundary="serial replay preflight",
            )
        raise Protocol101MixedSimulatorVersionError(
            "candidate stream contains a non-v5 simulator marker",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    clocks = (
        candidate.decision_time_ns,
        candidate.label_source_exit_quote_time_ns,
        candidate.label_realized_exit_time_ns,
        candidate.label_policy_deadline_ns,
    )
    if any(not isinstance(value, int) for value in clocks):
        raise Protocol101MissingRealizedExitError(
            "two-clock nanosecond metadata is missing or non-integer",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    decision, source, realized, deadline = [int(value) for value in clocks]
    if not decision < source <= realized:
        raise Protocol101InvalidRealizedExitError(
            "required clock ordering decision < source <= realized failed",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if realized > deadline:
        raise Protocol101ExitAfterDeadlineError(
            "realized exit exceeds policy deadline",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if deadline > _forced_flat_ns(candidate.session, config):
        raise Protocol101ExitAfterDeadlineError(
            "policy deadline exceeds same-session forced-flat cap",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    sessions = {
        ny_session_for_ns(decision),
        ny_session_for_ns(source),
        ny_session_for_ns(realized),
        ny_session_for_ns(deadline),
    }
    if sessions != {candidate.session}:
        raise Protocol101CrossSessionExitError(
            "entry/source/realized/deadline clocks cross New York session",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    try:
        reason = ExitReason(int(candidate.label_exit_reason_code))
    except ValueError as exc:
        raise Protocol101InvalidRealizedExitError(
            "unknown terminal exit reason",
            canonical_key=key,
            boundary="serial replay preflight",
        ) from exc
    if int(candidate.label_invalid_reason_code) != int(InvalidReason.NONE):
        raise Protocol101InvalidRealizedExitError(
            "selected intent carries an invalid label reason",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    threshold_reasons = {
        ExitReason.STOP_LOSS,
        ExitReason.TAKE_PROFIT,
        ExitReason.NO_BID_STOP,
    }
    deadline_reasons = {ExitReason.MAX_HOLD, ExitReason.FORCED_FLAT}
    if reason in threshold_reasons and source != realized:
        raise Protocol101InvalidRealizedExitError(
            "threshold/no-bid source and occupancy clocks differ",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if reason in deadline_reasons and realized != deadline:
        raise Protocol101InvalidRealizedExitError(
            "max-hold/forced-flat occupancy does not end at deadline",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if reason not in threshold_reasons | deadline_reasons:
        raise Protocol101InvalidRealizedExitError(
            "selected intent lacks a terminal exit reason",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    expected_age = (realized - source) / 1_000_000.0
    if (
        not math.isfinite(float(candidate.label_exit_quote_age_ms))
        or float(candidate.label_exit_quote_age_ms) < 0.0
        or not math.isclose(
            float(candidate.label_exit_quote_age_ms),
            expected_age,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise Protocol101QuoteAgeMismatchError(
            "quote age does not equal the two-clock difference",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if (
        not math.isfinite(float(candidate.entry_ask))
        or float(candidate.entry_ask) <= 0.0
        or not math.isfinite(float(candidate.label_executable_exit_bid))
        or float(candidate.label_executable_exit_bid) < 0.0
    ):
        raise Protocol101InvalidRealizedExitError(
            "entry ask or executable source exit bid is invalid",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    expected_pnl = (
        (
            float(candidate.label_executable_exit_bid)
            - float(candidate.entry_ask)
        )
        * float(config.contract_multiplier)
        - float(config.campaign_round_trip_fee_dollars)
    )
    if not math.isclose(
        float(candidate.raw_label_pnl_after_campaign_fee),
        expected_pnl,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise Protocol101PnlSourceQuoteMismatchError(
            "candidate PnL does not match source bid, entry ask, and fee",
            canonical_key=key,
            boundary="serial replay preflight",
        )
    if candidate.right not in {"C", "P"} or not 0 <= int(candidate.policy_index) <= 6:
        raise Protocol101InvalidRealizedExitError(
            "right or policy index is outside the signed axes",
            canonical_key=key,
            boundary="serial replay preflight",
        )


def preflight_serial_candidates_v5(
    candidates: Iterable[SerialCandidateV5],
    *,
    config: SerialSimulatorV5Config | None = None,
) -> list[SerialCandidateV5]:
    cfg = config or SerialSimulatorV5Config()
    if cfg.simulator_version != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        raise Protocol101MixedSimulatorVersionError(
            "v5 preflight requires the exact signed simulator version",
            boundary="serial replay preflight",
        )
    items = list(candidates)
    _assert_candidate_identities(items)
    for item in items:
        _preflight_candidate(item, cfg)
    return sorted(
        items,
        key=lambda item: (
            _split_rank(item.split, cfg),
            item.session,
            int(item.decision_time_ns),
            item.contract_id,
            int(item.policy_index),
        ),
    )


def _is_after_cutoff(value_ns: int, cutoff: str) -> bool:
    hour, minute = [int(value) for value in cutoff.split(":")]
    local = datetime.fromtimestamp(value_ns / 1_000_000_000, tz=NY)
    return (local.hour, local.minute) > (hour, minute)


def simulate_serial_candidates_v5(
    candidates: Iterable[SerialCandidateV5],
    *,
    config: SerialSimulatorV5Config | None = None,
) -> tuple[list[SerialReplayTradeV5], SerialReplayStateV5]:
    """Replay a fully preflighted selected-intent stream under v5."""

    cfg = config or SerialSimulatorV5Config()
    ordered = preflight_serial_candidates_v5(candidates, config=cfg)
    # Identity and clock validation intentionally precede every hash.
    stream_hash = canonical_json_hash(candidate_stream_payload(ordered))
    payload_hash = canonical_json_hash(candidate_payload(ordered))
    config_hash = canonical_json_hash(cfg.semantics())
    semantics = {
        **cfg.semantics(),
        "simulator_config_hash": config_hash,
        "candidate_stream_hash": stream_hash,
        "candidate_payload_hash": payload_hash,
        "quote_age_gate": False,
        "quote_age_rejection_threshold_ms": None,
    }

    skipped = {
        "after_entry_cutoff": 0,
        "overlap": 0,
        "daily_loss_stop": 0,
        "session_trade_cap": 0,
        "nonpositive_ask": 0,
        "unaffordable": 0,
    }
    skipped_events: list[SerialSkippedEventV5] = []
    trades: list[SerialReplayTradeV5] = []
    pending_by_account: dict[str, SerialReplayTradeV5 | None] = {}
    cash_by_account: dict[str, float] = {}
    active_session: dict[str, str] = {}
    session_start: dict[tuple[str, str], float] = {}
    realized_session: dict[tuple[str, str], float] = {}
    trades_session: dict[tuple[str, str], int] = {}
    equity_events: dict[str, list[dict[str, Any]]] = {}

    def realize(account: str) -> None:
        pending = pending_by_account.get(account)
        if pending is None:
            return
        key = (pending.split, pending.session)
        pnl = float(pending.raw_label_pnl_after_campaign_fee)
        cash = cash_by_account.setdefault(account, float(cfg.starting_cash)) + pnl
        cash_by_account[account] = cash
        realized_session[key] = realized_session.get(key, 0.0) + pnl
        equity_events.setdefault(
            account,
            [{"event_time_ns": None, "equity": float(cfg.starting_cash)}],
        ).append(
            {
                "event_time_ns": int(pending.label_realized_exit_time_ns),
                "equity": float(cash),
                "contract_id": pending.contract_id,
                "session": pending.session,
            }
        )
        pending_by_account[account] = None

    def record_skip(
        reason: str,
        item: SerialCandidateV5,
        *,
        cash: float,
        start_equity: float,
        session_pnl: float,
        required_cash: float,
    ) -> None:
        skipped[reason] += 1
        pending = pending_by_account.get(item.split)
        skipped_events.append(
            SerialSkippedEventV5(
                candidate_identity=_identity(item),
                reason_code=reason,
                decision_time_ns=int(item.decision_time_ns),
                pending_contract_id_or_null=(
                    pending.contract_id if pending is not None else None
                ),
                pending_source_quote_time_ns_or_null=(
                    int(pending.label_source_exit_quote_time_ns)
                    if pending is not None
                    else None
                ),
                pending_realized_exit_time_ns_or_null=(
                    int(pending.label_realized_exit_time_ns)
                    if pending is not None
                    else None
                ),
                pending_exit_quote_age_ms_or_null=(
                    float(pending.label_exit_quote_age_ms)
                    if pending is not None
                    else None
                ),
                cash=float(cash),
                session_start_equity=float(start_equity),
                session_realized_pnl=float(session_pnl),
                premium_plus_fee_required=float(required_cash),
                candidate_stream_hash=stream_hash,
                candidate_payload_hash=payload_hash,
                simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
            )
        )

    for item in ordered:
        account = item.split
        cash = cash_by_account.setdefault(account, float(cfg.starting_cash))
        equity_events.setdefault(
            account,
            [{"event_time_ns": None, "equity": float(cfg.starting_cash)}],
        )
        pending = pending_by_account.get(account)
        if (
            pending is not None
            and int(pending.label_realized_exit_time_ns)
            <= int(item.decision_time_ns)
        ):
            realize(account)
            cash = cash_by_account[account]
            pending = None
        previous_session = active_session.get(account)
        if previous_session is not None and previous_session != item.session:
            if pending_by_account.get(account) is not None:
                raise Protocol101CrossSessionExitError(
                    "pending position survived into a later session",
                    canonical_key=_identity(item),
                    boundary="serial replay state transition",
                )
        active_session[account] = item.session
        session_key = (item.split, item.session)
        session_start.setdefault(session_key, float(cash))
        start_equity = float(session_start[session_key])
        session_pnl = float(realized_session.get(session_key, 0.0))
        premium = float(item.entry_ask) * float(cfg.contract_multiplier)
        required_cash = premium + float(cfg.affordability_reserve_per_trade)

        if _is_after_cutoff(item.decision_time_ns, cfg.no_new_entries_after_et):
            record_skip(
                "after_entry_cutoff",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue
        if pending_by_account.get(account) is not None:
            record_skip(
                "overlap",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue
        daily_limit = (
            float(cfg.max_daily_loss_fraction_of_session_start_equity)
            * start_equity
        )
        if daily_limit > 0.0 and session_pnl <= -daily_limit:
            record_skip(
                "daily_loss_stop",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue
        if (
            cfg.max_trades_per_session > 0
            and trades_session.get(session_key, 0)
            >= int(cfg.max_trades_per_session)
        ):
            record_skip(
                "session_trade_cap",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue
        if item.entry_ask <= 0.0:
            record_skip(
                "nonpositive_ask",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue
        if cfg.enforce_affordability and required_cash > cash + 1e-9:
            record_skip(
                "unaffordable",
                item,
                cash=cash,
                start_equity=start_equity,
                session_pnl=session_pnl,
                required_cash=required_cash,
            )
            continue

        raw_pnl = float(item.raw_label_pnl_after_campaign_fee)
        trade = SerialReplayTradeV5(
            split=item.split,
            fold=item.fold,
            session=item.session,
            decision_time_ns=int(item.decision_time_ns),
            contract_id=item.contract_id,
            right=item.right,
            canonical_strike_slot=int(item.canonical_strike_slot),
            policy_index=int(item.policy_index),
            entry_ask=float(item.entry_ask),
            premium_at_risk=float(premium),
            premium_plus_fee_required=float(required_cash),
            score=float(item.score),
            raw_label_pnl_after_campaign_fee=raw_pnl,
            stressed_pnl=raw_pnl - float(cfg.stress_per_trade_dollars),
            label_realized_exit_time_ns=int(
                item.label_realized_exit_time_ns
            ),
            label_source_exit_quote_time_ns=int(
                item.label_source_exit_quote_time_ns
            ),
            label_exit_quote_age_ms=float(item.label_exit_quote_age_ms),
            label_exit_reason_code=int(item.label_exit_reason_code),
            label_executable_exit_bid=float(
                item.label_executable_exit_bid
            ),
            label_policy_deadline_ns=int(item.label_policy_deadline_ns),
            cash_before=float(cash),
            cash_after=float(cash + raw_pnl),
            session_start_equity=start_equity,
            session_realized_pnl_before=session_pnl,
            session_realized_pnl_after=float(session_pnl + raw_pnl),
            feature_hash=item.feature_hash,
            source_quote_time_ns=int(item.source_quote_time_ns),
            source_context_time_ns=int(item.source_context_time_ns),
            strategy=item.strategy,
            simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
            exit_time_semantics=EXIT_TIME_SEMANTICS,
            simulator_config_hash=config_hash,
            candidate_stream_hash=stream_hash,
            candidate_payload_hash=payload_hash,
        )
        trades.append(trade)
        trades_session[session_key] = trades_session.get(session_key, 0) + 1
        pending_by_account[account] = trade

    for account in sorted(pending_by_account):
        realize(account)

    trade_hash = canonical_json_hash(trade_identity_payload(trades))
    state = SerialReplayStateV5(
        skipped=skipped,
        skipped_events=tuple(skipped_events),
        cash_by_account={
            key: float(value) for key, value in sorted(cash_by_account.items())
        },
        session_start_equity={
            f"{split}:{session}": float(value)
            for (split, session), value in sorted(session_start.items())
        },
        realized_raw_pnl_by_split_session={
            f"{split}:{session}": float(value)
            for (split, session), value in sorted(realized_session.items())
        },
        equity_events_by_account={
            key: tuple(values) for key, values in sorted(equity_events.items())
        },
        semantics=semantics,
        simulator_config_hash=config_hash,
        candidate_stream_hash=stream_hash,
        candidate_payload_hash=payload_hash,
        trade_identity_hash=trade_hash,
    )
    return trades, state
