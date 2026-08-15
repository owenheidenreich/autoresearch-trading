"""Protocol101 Stage-1 reference and multiplicity machinery.

This module is deliberately campaign-economics agnostic. It accepts repaired
two-clock decisions or synthetic per-session PnL fixtures, routes every
reference replay through simulator v5, and requires a frozen synchronized
schedule before maxT statistics can be evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    FEATURE_NAMES,
    HYPOTHESES,
)
from v4.model.protocol101_regimen_repair import (
    InvalidReason,
    assert_alpha_feature_names,
)
from v4.model.protocol101_repair_artifacts import (
    REQUIRED_MANIFEST_HASHES,
    canonical_json_bytes,
    semantic_payload_hashes,
    sha256_file,
    write_immutable_replay_packet,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    RepairedCanonicalDecision,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialReplayStateV5,
    SerialReplayTradeV5,
    SerialSimulatorV5Config,
    canonical_json_hash,
    simulate_serial_candidates_v5,
)


REFERENCE_SCHEMA = "Protocol101Stage1ReferenceV5ContractV1"
MATCHED_RANDOM_DRAWS = 200
MATCHED_RANDOM_SEED = 101
MATCHED_RANDOM_PRNG = "NumPy PCG64"
FIXED_HEURISTIC_POLICY = 5
D1_NORMAL_ROWS = 359
D1_INCLUDED_ROWS = 330
D1_BLOCK_SIZE = 30
D1_COMPLETE_BLOCKS = 11
D1_TRAILING_EXCLUDED = 29
D1_SEEDS = tuple(range(8600, 8620))
D6_ROUTE = (
    "SIGNED_SPLIT_FAMILY_SYNCHRONIZATION_SUFFICIENT_FOR_OFFLINE_"
    "LABEL_ONLY_REPAIR"
)
MAXT_ROWS = tuple(
    f"H{hypothesis}/P{policy}"
    for hypothesis in range(4)
    for policy in range(7)
)
MAXT_SEEDS = (42, 43, 44)
MAXT_REPLICATES = 20_000
MAXT_BLOCK_SIZE = 5
MAXT_MASTER_SEED = 2_026_072_601
MAXT_ALPHA = 0.05
MAXT_PRNG = "NumPy PCG64DXSM"
MAXT_DENOMINATOR_DDOF = 1


class Protocol101ReferenceMultiplicityError(RuntimeError):
    """Fail-closed reference/multiplicity contract error."""

    blocker_code = "P101_REFERENCE_MULTIPLICITY_CONTRACT_ERROR"

    def __init__(self, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.payload = {"blocker_code": self.blocker_code, **payload}


class Protocol101ReferenceIdentityError(
    Protocol101ReferenceMultiplicityError
):
    blocker_code = "P101_REFERENCE_IDENTITY_ERROR"


class Protocol101ReferenceNonfiniteError(
    Protocol101ReferenceMultiplicityError
):
    blocker_code = "P101_REFERENCE_NONFINITE_INPUT"


class Protocol101D1ContractError(Protocol101ReferenceMultiplicityError):
    blocker_code = "P101_D1_CONTRACT_ERROR"


class Protocol101D6AuthorityError(Protocol101ReferenceMultiplicityError):
    blocker_code = "P101_D6_AUTHORITY_ERROR"


class Protocol101MaxTContractError(Protocol101ReferenceMultiplicityError):
    blocker_code = "P101_MAXT_CONTRACT_ERROR"


@dataclass(frozen=True)
class ReferenceOpportunity:
    """One governed decision opportunity for a single fixed policy."""

    campaign_id: str
    fold: str
    split: str
    policy_index: int
    repaired: RepairedCanonicalDecision
    vwap_side: str
    decision_ordinal: int

    @property
    def identity(self) -> tuple[Any, ...]:
        base = self.repaired.base
        return (
            self.campaign_id,
            self.fold,
            self.split,
            base.session,
            int(base.decision_time.value),
            int(self.decision_ordinal),
            int(self.policy_index),
        )


@dataclass(frozen=True)
class MatchedRandomSchedule:
    policy_index: int
    draws: int
    seed: int
    prng: str
    opportunity_identities: tuple[tuple[Any, ...], ...]
    selected_indices: tuple[tuple[int, ...], ...]
    schedule_hash: str


@dataclass(frozen=True)
class D1TargetOverride:
    """Target-only row; it cannot be submitted to a replay API."""

    session: str
    destination_decision_time_ns: int
    destination_row_index: int
    source_row_index: int
    net_labels: np.ndarray
    mid_labels: np.ndarray
    role: str
    seed: int | None
    non_candidate: bool = True
    replay_permitted: bool = False


@dataclass(frozen=True)
class SessionPnl:
    session: str
    fold: str
    pnl: float


@dataclass(frozen=True)
class MaxTGrid:
    rows: tuple[str, ...]
    seeds: tuple[int, ...]
    sessions: tuple[str, ...]
    folds: tuple[str, ...]
    pnl: np.ndarray
    grid_hash: str


@dataclass(frozen=True)
class MaxTConfig:
    replicates: int = MAXT_REPLICATES
    block_size: int = MAXT_BLOCK_SIZE
    master_seed: int = MAXT_MASTER_SEED
    alpha: float = MAXT_ALPHA
    prng: str = MAXT_PRNG
    denominator_ddof: int = MAXT_DENOMINATOR_DDOF

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenMaxTSchedule:
    schedule_path: Path
    manifest_path: Path
    indices: np.ndarray
    schedule_sha256: str
    manifest_sha256: str
    grid_hash: str
    config: MaxTConfig


@dataclass(frozen=True)
class MaxTResult:
    observed_z_by_row_seed: np.ndarray
    observed_t_by_row: np.ndarray
    denominator_by_row_seed: np.ndarray
    null_t_by_replicate_row: np.ndarray
    max_null_by_replicate: np.ndarray
    exceedance_counts: np.ndarray
    p_fwer: np.ndarray
    hard_pass: np.ndarray
    exceedance_limit: int
    max_null_sha256: str


def _array_finite(name: str, values: np.ndarray, identity: Any) -> None:
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise Protocol101ReferenceNonfiniteError(
            f"nonfinite {name} before reference replay",
            identity=identity,
            field=name,
        )


def assert_reference_opportunities(
    opportunities: Sequence[ReferenceOpportunity],
) -> tuple[ReferenceOpportunity, ...]:
    """Validate the complete opportunity grid before selection or hashing."""

    if not opportunities:
        raise Protocol101ReferenceIdentityError(
            "reference opportunity grid is empty"
        )
    ordered = tuple(
        sorted(
            opportunities,
            key=lambda item: (
                item.fold,
                item.repaired.base.session,
                int(item.repaired.base.decision_time.value),
                int(item.decision_ordinal),
            ),
        )
    )
    seen_opportunities: set[tuple[Any, ...]] = set()
    seen_contracts: set[tuple[Any, ...]] = set()
    seen_slots: set[tuple[Any, ...]] = set()
    for item in ordered:
        identity = item.identity
        if identity in seen_opportunities:
            raise Protocol101ReferenceIdentityError(
                "duplicate campaign/fold/session/decision/policy identity",
                identity=identity,
            )
        seen_opportunities.add(identity)
        if not item.campaign_id or not item.fold or not item.split:
            raise Protocol101ReferenceIdentityError(
                "reference identity fields must be nonempty",
                identity=identity,
            )
        if item.vwap_side not in {"C", "P"}:
            raise Protocol101ReferenceIdentityError(
                "vwap side must be C or P",
                identity=identity,
                observed=item.vwap_side,
            )
        if int(item.policy_index) not in range(7):
            raise Protocol101ReferenceIdentityError(
                "reference policy is outside P0-P6",
                identity=identity,
            )
        repaired = item.repaired
        base = repaired.base
        count = len(base.contract_ids)
        arrays = {
            "features": base.features,
            "labels": base.labels,
            "mid_labels": base.mid_labels,
            "entry_asks": base.entry_asks,
            "offsets": base.offsets,
            "rights": base.rights,
            "contract_ids": base.contract_ids,
            "strike_indices": base.strike_indices,
            "right_indices": base.right_indices,
            "realized_exit_time_ns": repaired.realized_exit_time_ns,
            "source_exit_quote_time_ns": repaired.source_exit_quote_time_ns,
            "exit_quote_age_ms": repaired.exit_quote_age_ms,
            "exit_reason_codes": repaired.exit_reason_codes,
            "executable_exit_bids": repaired.executable_exit_bids,
            "policy_deadline_ns": repaired.policy_deadline_ns,
            "invalid_reason_codes": repaired.invalid_reason_codes,
            "canonical_strike_slots": repaired.canonical_strike_slots,
            "source_quote_time_ns": repaired.source_quote_time_ns,
            "source_context_time_ns": repaired.source_context_time_ns,
        }
        if count <= 0 or any(len(values) != count for values in arrays.values()):
            raise Protocol101ReferenceIdentityError(
                "reference candidate axes are empty or misaligned",
                identity=identity,
                candidate_count=count,
            )
        for name in (
            "features",
            "labels",
            "mid_labels",
            "entry_asks",
            "offsets",
            "exit_quote_age_ms",
            "executable_exit_bids",
        ):
            _array_finite(name, np.asarray(arrays[name]), identity)
        if np.any(np.asarray(base.entry_asks, dtype=float) <= 0.0):
            raise Protocol101ReferenceNonfiniteError(
                "nonpositive entry ask before reference replay",
                identity=identity,
            )
        if np.any(
            np.asarray(repaired.invalid_reason_codes, dtype=int)
            != int(InvalidReason.NONE)
        ):
            raise Protocol101ReferenceNonfiniteError(
                "invalid label metadata before reference replay",
                identity=identity,
            )
        for candidate_index in range(count):
            right = str(base.rights[candidate_index])
            contract_id = str(base.contract_ids[candidate_index])
            contract_key = (*identity, contract_id)
            slot_key = (
                *identity,
                int(repaired.canonical_strike_slots[candidate_index]),
                right,
            )
            if right not in {"C", "P"} or not contract_id:
                raise Protocol101ReferenceIdentityError(
                    "invalid contract identity",
                    identity=identity,
                    candidate_index=candidate_index,
                )
            if contract_key in seen_contracts:
                raise Protocol101ReferenceIdentityError(
                    "duplicate reference contract identity",
                    identity=contract_key,
                )
            if slot_key in seen_slots:
                raise Protocol101ReferenceIdentityError(
                    "duplicate reference canonical slot identity",
                    identity=slot_key,
                )
            seen_contracts.add(contract_key)
            seen_slots.add(slot_key)
    return ordered


def _candidate_from_opportunity(
    opportunity: ReferenceOpportunity,
    candidate_index: int,
    *,
    strategy: str,
    score: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> SerialCandidateV5:
    repaired = opportunity.repaired
    base = repaired.base
    index = int(candidate_index)
    if index < 0 or index >= len(base.contract_ids):
        raise Protocol101ReferenceIdentityError(
            "sampled candidate index is outside the opportunity ladder",
            identity=opportunity.identity,
            candidate_index=index,
        )
    return SerialCandidateV5(
        split=opportunity.split,
        fold=opportunity.fold,
        session=base.session,
        decision_time_ns=int(base.decision_time.value),
        contract_id=str(base.contract_ids[index]),
        right=str(base.rights[index]),
        canonical_strike_slot=int(
            repaired.canonical_strike_slots[index]
        ),
        policy_index=int(opportunity.policy_index),
        entry_ask=float(base.entry_asks[index]),
        score=float(score),
        raw_label_pnl_after_campaign_fee=(
            float(base.labels[index]) - 3.0
        ),
        label_mid_pnl_before_campaign_fee=float(base.mid_labels[index]),
        label_realized_exit_time_ns=int(
            repaired.realized_exit_time_ns[index]
        ),
        label_source_exit_quote_time_ns=int(
            repaired.source_exit_quote_time_ns[index]
        ),
        label_exit_quote_age_ms=float(
            repaired.exit_quote_age_ms[index]
        ),
        label_exit_reason_code=int(repaired.exit_reason_codes[index]),
        label_executable_exit_bid=float(
            repaired.executable_exit_bids[index]
        ),
        label_policy_deadline_ns=int(
            repaired.policy_deadline_ns[index]
        ),
        label_invalid_reason_code=int(
            repaired.invalid_reason_codes[index]
        ),
        feature_hash=CONTRACT_ID,
        source_quote_time_ns=int(
            repaired.source_quote_time_ns[index]
        ),
        source_context_time_ns=int(
            repaired.source_context_time_ns[index]
        ),
        strategy=strategy,
        source_simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        metadata={
            "campaign_id": opportunity.campaign_id,
            "fold": opportunity.fold,
            "decision_ordinal": int(opportunity.decision_ordinal),
            "candidate_index": index,
            **dict(metadata or {}),
        },
    )


def reference_simulator_config(
    *,
    fee: float = 3.0,
) -> SerialSimulatorV5Config:
    return SerialSimulatorV5Config(
        starting_cash=10_000.0,
        campaign_round_trip_fee_dollars=float(fee),
        affordability_reserve_per_trade=float(fee),
        max_trades_per_session=0,
        max_daily_loss_fraction_of_session_start_equity=0.05,
        no_new_entries_after_et="15:30",
        forced_flat_before_et="15:55",
    )


def replay_reference_v5(
    candidates: Iterable[SerialCandidateV5],
    *,
    fee: float = 3.0,
) -> tuple[list[SerialReplayTradeV5], SerialReplayStateV5]:
    """The sole fresh reference replay terminus."""

    return simulate_serial_candidates_v5(
        candidates,
        config=reference_simulator_config(fee=fee),
    )


def generate_matched_random_schedule(
    opportunities: Sequence[ReferenceOpportunity],
    *,
    policy_index: int,
    draws: int = MATCHED_RANDOM_DRAWS,
    seed: int = MATCHED_RANDOM_SEED,
) -> MatchedRandomSchedule:
    ordered = assert_reference_opportunities(opportunities)
    if int(draws) <= 0:
        raise Protocol101ReferenceMultiplicityError(
            "matched-random draw count must be positive"
        )
    if any(item.policy_index != int(policy_index) for item in ordered):
        raise Protocol101ReferenceIdentityError(
            "matched-random opportunity policy differs from requested policy",
            requested_policy=int(policy_index),
        )
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    selected = tuple(
        tuple(
            int(rng.integers(0, len(item.repaired.base.contract_ids)))
            for item in ordered
        )
        for _draw in range(int(draws))
    )
    payload = {
        "schema_version": REFERENCE_SCHEMA,
        "kind": "matched_random_schedule",
        "policy_index": int(policy_index),
        "draws": int(draws),
        "seed": int(seed),
        "prng": MATCHED_RANDOM_PRNG,
        "opportunity_identities": [list(item.identity) for item in ordered],
        "selected_indices": [list(row) for row in selected],
    }
    return MatchedRandomSchedule(
        policy_index=int(policy_index),
        draws=int(draws),
        seed=int(seed),
        prng=MATCHED_RANDOM_PRNG,
        opportunity_identities=tuple(item.identity for item in ordered),
        selected_indices=selected,
        schedule_hash=canonical_json_hash(payload),
    )


def matched_random_draw_candidates(
    opportunities: Sequence[ReferenceOpportunity],
    schedule: MatchedRandomSchedule,
    *,
    draw_index: int,
) -> list[SerialCandidateV5]:
    ordered = assert_reference_opportunities(opportunities)
    identities = tuple(item.identity for item in ordered)
    if identities != schedule.opportunity_identities:
        raise Protocol101ReferenceIdentityError(
            "matched-random schedule opportunity grid changed"
        )
    if draw_index < 0 or draw_index >= schedule.draws:
        raise Protocol101ReferenceIdentityError(
            "matched-random draw index is outside the frozen schedule",
            draw_index=int(draw_index),
        )
    return [
        _candidate_from_opportunity(
            item,
            schedule.selected_indices[draw_index][index],
            strategy="matched_random_selection_v5",
            metadata={
                "draw_index": int(draw_index),
                "schedule_hash": schedule.schedule_hash,
            },
        )
        for index, item in enumerate(ordered)
    ]


def run_matched_random_reference(
    opportunities: Sequence[ReferenceOpportunity],
    schedule: MatchedRandomSchedule,
) -> dict[str, Any]:
    """Replay every frozen draw and retain exact sampled trade identities."""

    draw_results: list[dict[str, Any]] = []
    for draw_index in range(schedule.draws):
        candidates = matched_random_draw_candidates(
            opportunities,
            schedule,
            draw_index=draw_index,
        )
        trades, state = replay_reference_v5(candidates)
        metrics = replay_metrics(trades, state)
        draw_results.append(
            {
                "draw_index": draw_index,
                "metrics": metrics,
                "sampled_candidate_identities": [
                    {
                        "campaign_id": item.metadata["campaign_id"],
                        "fold": item.fold,
                        "session": item.session,
                        "decision_time_ns": int(item.decision_time_ns),
                        "contract_id": item.contract_id,
                        "canonical_strike_slot": int(
                            item.canonical_strike_slot
                        ),
                        "policy_index": int(item.policy_index),
                    }
                    for item in candidates
                ],
            }
        )
    pnl = np.asarray(
        [item["metrics"]["net_pnl"] for item in draw_results],
        dtype=float,
    )
    return {
        "schema_version": REFERENCE_SCHEMA,
        "kind": "matched_random_selection_v5",
        "policy_index": schedule.policy_index,
        "draws": schedule.draws,
        "seed": schedule.seed,
        "prng": schedule.prng,
        "schedule_hash": schedule.schedule_hash,
        "opportunity_grid_complete": all(
            len(item["sampled_candidate_identities"])
            == len(schedule.opportunity_identities)
            for item in draw_results
        ),
        "draw_results": draw_results,
        "pooled_net_pnl_distribution": {
            "count": int(len(pnl)),
            "mean": float(np.mean(pnl)),
            "sample_std": (
                float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
            ),
            "minimum": float(np.min(pnl)),
            "maximum": float(np.max(pnl)),
        },
    }


def fixed_heuristic_candidates(
    opportunities: Sequence[ReferenceOpportunity],
) -> list[SerialCandidateV5]:
    """Build the unchanged P5 VWAP-side nearest-ATM heuristic."""

    ordered = assert_reference_opportunities(opportunities)
    if any(item.policy_index != FIXED_HEURISTIC_POLICY for item in ordered):
        raise Protocol101ReferenceIdentityError(
            "fixed heuristic is frozen at policy P5"
        )
    candidates: list[SerialCandidateV5] = []
    for item in ordered:
        base = item.repaired.base
        side_indices = [
            index
            for index, right in enumerate(base.rights)
            if str(right) == item.vwap_side
        ]
        if not side_indices:
            continue
        selected = min(
            side_indices,
            key=lambda index: (
                abs(float(base.offsets[index])),
                float(base.offsets[index]),
                0 if str(base.rights[index]) == "C" else 1,
                str(base.contract_ids[index]),
            ),
        )
        candidates.append(
            _candidate_from_opportunity(
                item,
                selected,
                strategy="vwap_side_nearest_atm_policy5_v5",
                metadata={"heuristic_policy_fixed": 5},
            )
        )
    return candidates


def replay_metrics(
    trades: Sequence[SerialReplayTradeV5],
    state: SerialReplayStateV5,
) -> dict[str, Any]:
    pnl = np.asarray(
        [item.raw_label_pnl_after_campaign_fee for item in trades],
        dtype=float,
    )
    equity = np.asarray(
        [
            float(event["equity"])
            for events in state.equity_events_by_account.values()
            for event in events
        ]
        or [10_000.0],
        dtype=float,
    )
    peak = np.maximum.accumulate(equity)
    return {
        "entry_intents": int(
            len(trades) + sum(int(value) for value in state.skipped.values())
        ),
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "max_drawdown": float(np.max(peak - equity)),
        "minimum_equity": float(np.min(equity)),
        "skipped": dict(state.skipped),
        "simulator_version": state.semantics["simulator_version"],
        "simulator_config_hash": state.simulator_config_hash,
        "candidate_stream_hash": state.candidate_stream_hash,
        "candidate_payload_hash": state.candidate_payload_hash,
        "trade_identity_hash": state.trade_identity_hash,
    }


def run_fixed_heuristic_reference(
    opportunities: Sequence[ReferenceOpportunity],
) -> dict[str, Any]:
    candidates = fixed_heuristic_candidates(opportunities)
    trades, state = replay_reference_v5(candidates)
    pooled = replay_metrics(trades, state)
    candidate_by_trade_identity = {
        (
            item.fold,
            item.session,
            int(item.decision_time_ns),
            item.contract_id,
            int(item.policy_index),
        ): item
        for item in candidates
    }
    per_fold: dict[str, Any] = {}
    for fold in sorted({item.fold for item in candidates}):
        fold_candidates = [item for item in candidates if item.fold == fold]
        fold_trades, fold_state = replay_reference_v5(fold_candidates)
        per_fold[fold] = replay_metrics(fold_trades, fold_state)
    return {
        "schema_version": REFERENCE_SCHEMA,
        "kind": "fixed_heuristic_policy5",
        "ordered_candidate_intents": [asdict(item) for item in candidates],
        "ordered_trade_identities": [
            {
                "campaign_id": candidate_by_trade_identity[
                    (
                        item.fold,
                        item.session,
                        int(item.decision_time_ns),
                        item.contract_id,
                        int(item.policy_index),
                    )
                ].metadata["campaign_id"],
                "fold": item.fold,
                "session": item.session,
                "decision_time_ns": int(item.decision_time_ns),
                "contract_id": item.contract_id,
                "canonical_strike_slot": int(item.canonical_strike_slot),
                "policy_index": int(item.policy_index),
                "label_source_exit_quote_time_ns": int(
                    item.label_source_exit_quote_time_ns
                ),
                "label_realized_exit_time_ns": int(
                    item.label_realized_exit_time_ns
                ),
            }
            for item in trades
        ],
        "per_fold_metrics": per_fold,
        "continuous_pooled_metrics": pooled,
        "candidates": candidates,
        "trades": trades,
        "state": state,
    }


def _session_metrics(
    trades: Sequence[SerialReplayTradeV5],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for trade in trades:
        item = result.setdefault(
            trade.session,
            {"trades": 0, "net_pnl": 0.0},
        )
        item["trades"] += 1
        item["net_pnl"] += float(trade.raw_label_pnl_after_campaign_fee)
    return result


def write_immutable_reference_packet(
    packet_dir: Path,
    *,
    candidates: Sequence[SerialCandidateV5],
    trades: Sequence[SerialReplayTradeV5],
    state: SerialReplayStateV5,
    metrics: Mapping[str, Any],
    provenance_hashes: Mapping[str, str],
    attempt_id: str,
) -> dict[str, Any]:
    """Commit one fresh v5 reference replay packet manifest-last."""

    payloads = {
        "candidate_intents.jsonl": [asdict(item) for item in candidates],
        "exit_quote_age_report.json": {
            "schema_version": "Protocol101ExitQuoteAgeReportV1",
            "gate": False,
            "rejection_threshold_ms": None,
            "ages_ms": [
                float(item.label_exit_quote_age_ms) for item in trades
            ],
        },
        "fold_metrics.json": dict(metrics.get("per_fold_metrics") or {}),
        "pooled_metrics.json": dict(
            metrics.get("continuous_pooled_metrics") or metrics
        ),
        "session_metrics.json": _session_metrics(trades),
        "skipped_events.jsonl": [
            asdict(item) for item in state.skipped_events
        ],
        "trades.jsonl": [asdict(item) for item in trades],
    }
    semantic = semantic_payload_hashes(payloads)
    required_provenance = set(REQUIRED_MANIFEST_HASHES) - {
        "simulator_config_hash",
        "candidate_stream_hash",
        "candidate_payload_hash",
        "trade_identity_hash",
        "exit_quote_age_report_hash",
    }
    missing = sorted(required_provenance - set(provenance_hashes))
    if missing:
        raise Protocol101ReferenceMultiplicityError(
            "immutable reference provenance is incomplete",
            missing=missing,
        )
    manifest = {
        **{name: provenance_hashes[name] for name in required_provenance},
        **semantic,
        "simulator_config_hash": state.simulator_config_hash,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "attempt_id": attempt_id,
        "reference_schema": REFERENCE_SCHEMA,
    }
    result = write_immutable_replay_packet(
        Path(packet_dir),
        payloads=payloads,
        manifest_fields=manifest,
    )
    return asdict(result)


def _ordered_full_session(
    decisions: Sequence[RepairedCanonicalDecision],
) -> tuple[RepairedCanonicalDecision, ...]:
    ordered = tuple(
        sorted(decisions, key=lambda item: int(item.base.decision_time.value))
    )
    if len(ordered) != D1_NORMAL_ROWS:
        raise Protocol101D1ContractError(
            "D1 requires exactly 359 rows in every full session",
            observed_rows=len(ordered),
        )
    sessions = {item.base.session for item in ordered}
    times = [int(item.base.decision_time.value) for item in ordered]
    if len(sessions) != 1 or len(set(times)) != len(times):
        raise Protocol101D1ContractError(
            "D1 session identities are mixed or duplicated",
            sessions=sorted(sessions),
        )
    h2_feature_count = len(HYPOTHESES["H2"])
    if any(
        item.base.features.ndim != 2
        or item.base.features.shape[1] != h2_feature_count
        for item in ordered
    ):
        raise Protocol101D1ContractError(
            "D1 requires the exact H2-shaped feature matrix",
            expected_feature_count=h2_feature_count,
        )
    return ordered


def build_d1_target_overrides(
    decisions: Sequence[RepairedCanonicalDecision],
    *,
    role: str,
    seed: int | None = None,
    policy_index: int = FIXED_HEURISTIC_POLICY,
) -> tuple[
    tuple[RepairedCanonicalDecision, ...],
    tuple[D1TargetOverride, ...],
    dict[str, Any],
]:
    """Build target-only complete-block overrides without moving inputs."""

    if role not in {"fit", "calibration", "validation"}:
        raise Protocol101D1ContractError("invalid D1 split role", role=role)
    if int(policy_index) != FIXED_HEURISTIC_POLICY:
        raise Protocol101D1ContractError(
            "D1 is frozen at policy P5",
            policy_index=int(policy_index),
        )
    ordered = _ordered_full_session(decisions)
    included = ordered[:D1_INCLUDED_ROWS]
    if len(ordered[D1_INCLUDED_ROWS:]) != D1_TRAILING_EXCLUDED:
        raise Protocol101D1ContractError(
            "D1 trailing-row geometry differs from exactly 29"
        )
    if role in {"fit", "calibration"}:
        if seed not in D1_SEEDS:
            raise Protocol101D1ContractError(
                "D1 fit/calibration seed must be 8600 through 8619",
                seed=seed,
            )
        rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
        block_order = tuple(
            int(value) for value in rng.permutation(D1_COMPLETE_BLOCKS)
        )
    else:
        if seed is not None:
            raise Protocol101D1ContractError(
                "D1 validation is unpermuted and takes no seed"
            )
        block_order = tuple(range(D1_COMPLETE_BLOCKS))
    source_rows = tuple(
        source_block * D1_BLOCK_SIZE + within_block
        for source_block in block_order
        for within_block in range(D1_BLOCK_SIZE)
    )
    overrides = tuple(
        D1TargetOverride(
            session=destination.base.session,
            destination_decision_time_ns=int(
                destination.base.decision_time.value
            ),
            destination_row_index=destination_index,
            source_row_index=source_index,
            net_labels=np.asarray(
                included[source_index].base.labels,
                dtype=float,
            ).copy(),
            mid_labels=np.asarray(
                included[source_index].base.mid_labels,
                dtype=float,
            ).copy(),
            role=role,
            seed=seed,
        )
        for destination_index, (destination, source_index) in enumerate(
            zip(included, source_rows)
        )
    )
    payload = {
        "schema_version": "Protocol101D1TargetOverrideV1",
        "role": role,
        "seed": seed,
        "policy_index": int(policy_index),
        "normal_rows": D1_NORMAL_ROWS,
        "included_rows": D1_INCLUDED_ROWS,
        "trailing_rows_excluded": D1_TRAILING_EXCLUDED,
        "block_size": D1_BLOCK_SIZE,
        "complete_blocks": D1_COMPLETE_BLOCKS,
        "block_order": list(block_order),
        "source_rows": list(source_rows),
        "targets": [
            {
                "destination_decision_time_ns": (
                    item.destination_decision_time_ns
                ),
                "destination_row_index": item.destination_row_index,
                "source_row_index": item.source_row_index,
                "net_labels": item.net_labels.tolist(),
                "mid_labels": item.mid_labels.tolist(),
                "non_candidate": item.non_candidate,
                "replay_permitted": item.replay_permitted,
            }
            for item in overrides
        ],
    }
    receipt = {
        "block_order": list(block_order),
        "source_rows": list(source_rows),
        "override_hash": canonical_json_hash(payload),
        "features_moved": False,
        "identities_moved": False,
        "entry_asks_moved": False,
        "canonical_exit_metadata_moved": False,
        "net_and_mid_labels_moved_together": True,
        "permuted_fit_rows_replayable": False,
    }
    return included, overrides, receipt


def d1_target_arrays(
    included: Sequence[RepairedCanonicalDecision],
    overrides: Sequence[D1TargetOverride],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Expose only target arrays to the future non-candidate fit layer."""

    if len(included) != D1_INCLUDED_ROWS or len(overrides) != D1_INCLUDED_ROWS:
        raise Protocol101D1ContractError(
            "D1 target override count must be exactly 330"
        )
    for decision, override in zip(included, overrides):
        if int(decision.base.decision_time.value) != (
            override.destination_decision_time_ns
        ):
            raise Protocol101D1ContractError(
                "D1 target override destination identity changed"
            )
        if not override.non_candidate or override.replay_permitted:
            raise Protocol101D1ContractError(
                "D1 target override became replayable"
            )
    return (
        [np.asarray(item.net_labels, dtype=float) for item in overrides],
        [np.asarray(item.mid_labels, dtype=float) for item in overrides],
    )


def _sha256_path(path: Path) -> str:
    return sha256_file(Path(path))


def build_d6_authority_receipt(workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace)
    paths = {
        "signed_synchronization_decision": (
            "v4/docs/protocol101/synchronization/contracts/"
            "PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md"
        ),
        "repair_amendment": (
            "v4/docs/protocol101/training/contracts/"
            "PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md"
        ),
        "campaign_contract": (
            "v4/audit/autoresearch/"
            "protocol101_full_trader_entry_campaign_preregistration_attempt001/"
            "campaign_contract.json"
        ),
        "feature_firewall_source": (
            "v4/model/protocol101_canonical_stage1_contract.py"
        ),
    }
    missing = [
        relative for relative in paths.values()
        if not (workspace / relative).is_file()
    ]
    if missing:
        raise Protocol101D6AuthorityError(
            "D6 authority input is missing",
            missing=missing,
        )
    assert_alpha_feature_names(
        FEATURE_NAMES,
        allowed_feature_sets=HYPOTHESES.values(),
        boundary="D6 exact ordered 17-feature receipt",
    )
    receipt = {
        "schema_version": "Protocol101D6AuthorityReceiptV1",
        "route": D6_ROUTE,
        "offline_17_feature_rebuild_allowed": True,
        "repaired_exit_label_audit_metadata_as_alpha_forbidden": True,
        "exact_candidate_transfer": (
            "deferred_to_mandatory_no_order_shadow"
        ),
        "sealed_evidence_access": "forbidden",
        "broker_paper_authority": "none",
        "ordered_17_feature_firewall": list(FEATURE_NAMES),
        "ordered_17_feature_firewall_hash": canonical_json_hash(
            list(FEATURE_NAMES)
        ),
        "bound_sources": {
            name: {
                "path": relative,
                "sha256": _sha256_path(workspace / relative),
            }
            for name, relative in paths.items()
        },
        "alpha_prohibitions": [
            "repaired exit times",
            "source exit quote times",
            "exit quote age",
            "exit reason",
            "executable exit bid",
            "policy deadline",
            "invalid reason",
            "net labels",
            "mid labels",
            "future path metadata",
        ],
        "new_synchronization_claim": False,
    }
    receipt["receipt_hash"] = canonical_json_hash(receipt)
    return receipt


def build_maxT_grid(
    series: Mapping[str, Mapping[int, Sequence[SessionPnl]]],
) -> MaxTGrid:
    """Require the exact 28-by-3 family on one identical session/fold grid."""

    if tuple(series.keys()) != MAXT_ROWS:
        raise Protocol101MaxTContractError(
            "maxT requires the exact ordered 28-row family",
            expected=list(MAXT_ROWS),
            observed=list(series.keys()),
        )
    baseline_grid: tuple[tuple[str, str], ...] | None = None
    row_values: list[list[list[float]]] = []
    for row in MAXT_ROWS:
        seed_map = series[row]
        if tuple(seed_map.keys()) != MAXT_SEEDS:
            raise Protocol101MaxTContractError(
                "maxT requires exact ordered seeds 42,43,44",
                row=row,
                observed=list(seed_map.keys()),
            )
        seed_values: list[list[float]] = []
        for seed in MAXT_SEEDS:
            records = tuple(seed_map[seed])
            grid = tuple(
                (str(item.session), str(item.fold)) for item in records
            )
            if baseline_grid is None:
                baseline_grid = grid
            elif grid != baseline_grid:
                raise Protocol101MaxTContractError(
                    "session/fold grids differ across rows or seeds",
                    row=row,
                    seed=seed,
                )
            seed_values.append([float(item.pnl) for item in records])
        row_values.append(seed_values)
    if not baseline_grid:
        raise Protocol101MaxTContractError("maxT session grid is empty")
    pnl = np.asarray(row_values, dtype=float)
    if pnl.shape[:2] != (28, 3) or pnl.shape[2] != len(baseline_grid):
        raise Protocol101MaxTContractError(
            "maxT input has an invalid 28-by-3 grid shape",
            shape=list(pnl.shape),
        )
    if not np.isfinite(pnl).all():
        raise Protocol101MaxTContractError(
            "nonfinite per-session PnL entered maxT"
        )
    sessions = tuple(item[0] for item in baseline_grid)
    folds = tuple(item[1] for item in baseline_grid)
    if len(set(sessions)) != len(sessions):
        raise Protocol101MaxTContractError(
            "maxT session identities are duplicated"
        )
    fold_order = tuple(dict.fromkeys(folds))
    if len(fold_order) != 5:
        raise Protocol101MaxTContractError(
            "maxT requires exactly five chronological folds",
            folds=list(fold_order),
        )
    observed_fold_sequence = tuple(
        fold for index, fold in enumerate(folds)
        if index == 0 or fold != folds[index - 1]
    )
    if observed_fold_sequence != fold_order:
        raise Protocol101MaxTContractError(
            "maxT folds are not contiguous chronological sequences"
        )
    for fold in fold_order:
        fold_sessions = [
            session for session, item_fold in baseline_grid
            if item_fold == fold
        ]
        if not fold_sessions or fold_sessions != sorted(fold_sessions):
            raise Protocol101MaxTContractError(
                "maxT sessions are not chronological within fold",
                fold=fold,
            )
    grid_payload = {
        "rows": list(MAXT_ROWS),
        "seeds": list(MAXT_SEEDS),
        "session_fold_grid": [list(item) for item in baseline_grid],
        "pnl": pnl.tolist(),
    }
    return MaxTGrid(
        rows=MAXT_ROWS,
        seeds=MAXT_SEEDS,
        sessions=sessions,
        folds=folds,
        pnl=pnl,
        grid_hash=canonical_json_hash(grid_payload),
    )


def _fold_index_arrays(grid: MaxTGrid) -> tuple[np.ndarray, ...]:
    fold_order = tuple(dict.fromkeys(grid.folds))
    return tuple(
        np.asarray(
            [index for index, fold in enumerate(grid.folds) if fold == name],
            dtype=np.uint32,
        )
        for name in fold_order
    )


def generate_maxT_schedule(
    grid: MaxTGrid,
    *,
    config: MaxTConfig | None = None,
) -> np.ndarray:
    cfg = config or MaxTConfig()
    if cfg.replicates <= 1 or cfg.block_size != MAXT_BLOCK_SIZE:
        raise Protocol101MaxTContractError(
            "maxT requires positive replicates and frozen block size five",
            config=cfg.to_dict(),
        )
    if cfg.prng != MAXT_PRNG or cfg.denominator_ddof != 1:
        raise Protocol101MaxTContractError(
            "maxT PRNG or denominator law changed",
            config=cfg.to_dict(),
        )
    folds = _fold_index_arrays(grid)
    children = np.random.SeedSequence(cfg.master_seed).spawn(cfg.replicates)
    schedule = np.empty(
        (cfg.replicates, len(grid.sessions)),
        dtype=np.uint32,
    )
    for replicate, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        cursor = 0
        for fold_indices in folds:
            n_fold = len(fold_indices)
            starts = rng.integers(
                0,
                n_fold,
                size=math.ceil(n_fold / cfg.block_size),
            )
            local = np.concatenate(
                [
                    (int(start) + np.arange(cfg.block_size)) % n_fold
                    for start in starts
                ]
            )[:n_fold]
            sampled = fold_indices[local]
            schedule[replicate, cursor : cursor + n_fold] = sampled
            cursor += n_fold
    return schedule


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _npy_bytes(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def freeze_maxT_schedule(
    schedule_path: Path,
    manifest_path: Path,
    grid: MaxTGrid,
    *,
    config: MaxTConfig | None = None,
) -> FrozenMaxTSchedule:
    """Persist every sampled index and its manifest before observation."""

    cfg = config or MaxTConfig()
    schedule = generate_maxT_schedule(grid, config=cfg)
    schedule_payload = _npy_bytes(schedule)
    schedule_hash = hashlib.sha256(schedule_payload).hexdigest()
    manifest = {
        "schema_version": "Protocol101FrozenMaxTScheduleV1",
        "algorithm": "synchronized_five_session_moving_block_maxT",
        "grid_hash": grid.grid_hash,
        "config": cfg.to_dict(),
        "schedule_file": Path(schedule_path).name,
        "schedule_sha256": schedule_hash,
        "shape": list(schedule.shape),
        "dtype": str(schedule.dtype),
        "every_sampled_index_persisted": True,
        "same_schedule_for_all_28_rows_and_three_seeds": True,
        "frozen_before_observed_adjusted_p_values": True,
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    schedule_path = Path(schedule_path)
    manifest_path = Path(manifest_path)
    if schedule_path.exists() or manifest_path.exists():
        if not schedule_path.is_file() or not manifest_path.is_file():
            raise Protocol101MaxTContractError(
                "partial maxT schedule artifact exists"
            )
        if (
            sha256_file(schedule_path) != schedule_hash
            or sha256_file(manifest_path) != manifest_hash
        ):
            raise Protocol101MaxTContractError(
                "existing maxT schedule artifact differs from frozen schedule"
            )
    else:
        _atomic_bytes(schedule_path, schedule_payload)
        _atomic_bytes(manifest_path, manifest_payload)
    return load_frozen_maxT_schedule(
        schedule_path,
        manifest_path,
        grid,
        config=cfg,
    )


def load_frozen_maxT_schedule(
    schedule_path: Path,
    manifest_path: Path,
    grid: MaxTGrid,
    *,
    config: MaxTConfig | None = None,
) -> FrozenMaxTSchedule:
    """Verify hashes and exact SeedSequence reproduction before evaluation."""

    cfg = config or MaxTConfig()
    schedule_path = Path(schedule_path)
    manifest_path = Path(manifest_path)
    if not schedule_path.is_file() or not manifest_path.is_file():
        raise Protocol101MaxTContractError(
            "maxT schedule must be frozen before evaluation"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule_hash = sha256_file(schedule_path)
    if (
        manifest.get("grid_hash") != grid.grid_hash
        or manifest.get("config") != cfg.to_dict()
        or manifest.get("schedule_sha256") != schedule_hash
        or manifest.get("frozen_before_observed_adjusted_p_values") is not True
        or manifest.get("same_schedule_for_all_28_rows_and_three_seeds")
        is not True
    ):
        raise Protocol101MaxTContractError(
            "maxT frozen schedule manifest failed verification"
        )
    with schedule_path.open("rb") as handle:
        schedule = np.load(handle, allow_pickle=False)
    expected = generate_maxT_schedule(grid, config=cfg)
    if schedule.dtype != expected.dtype or not np.array_equal(
        schedule, expected
    ):
        raise Protocol101MaxTContractError(
            "maxT schedule cannot be exactly reproduced"
        )
    return FrozenMaxTSchedule(
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        indices=schedule,
        schedule_sha256=schedule_hash,
        manifest_sha256=sha256_file(manifest_path),
        grid_hash=grid.grid_hash,
        config=cfg,
    )


def fwer_p_values(
    observed_t: np.ndarray,
    max_null: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_t, dtype=float)
    maxima = np.asarray(max_null, dtype=float)
    if (
        observed.shape != (28,)
        or maxima.ndim != 1
        or not np.isfinite(observed).all()
        or not np.isfinite(maxima).all()
    ):
        raise Protocol101MaxTContractError(
            "invalid observed or null maxT arrays"
        )
    counts = np.asarray(
        [(maxima >= value).sum() for value in observed],
        dtype=np.int64,
    )
    return counts, (1.0 + counts) / (len(maxima) + 1.0)


def _numeric_array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def evaluate_maxT(
    grid: MaxTGrid,
    frozen: FrozenMaxTSchedule,
    *,
    chunk_size: int = 256,
) -> MaxTResult:
    """Evaluate observed and null statistics only from a frozen schedule."""

    cfg = frozen.config
    if (
        frozen.grid_hash != grid.grid_hash
        or sha256_file(frozen.schedule_path) != frozen.schedule_sha256
        or sha256_file(frozen.manifest_path) != frozen.manifest_sha256
    ):
        raise Protocol101MaxTContractError(
            "maxT schedule changed before observed evaluation"
        )
    if frozen.indices.shape != (cfg.replicates, len(grid.sessions)):
        raise Protocol101MaxTContractError(
            "maxT schedule shape differs from frozen session grid"
        )
    centered = np.asarray(grid.pnl, dtype=float).copy()
    for fold_indices in _fold_index_arrays(grid):
        centered[:, :, fold_indices] -= centered[:, :, fold_indices].mean(
            axis=2,
            keepdims=True,
        )
    null_pooled = np.empty(
        (cfg.replicates, len(MAXT_ROWS), len(MAXT_SEEDS)),
        dtype=float,
    )
    for start in range(0, cfg.replicates, int(chunk_size)):
        stop = min(start + int(chunk_size), cfg.replicates)
        indices = frozen.indices[start:stop]
        sampled = np.take(centered, indices, axis=2)
        null_pooled[start:stop] = sampled.sum(axis=3).transpose(2, 0, 1)
    denominator = np.std(
        null_pooled,
        axis=0,
        ddof=cfg.denominator_ddof,
    )
    if (
        not np.isfinite(denominator).all()
        or np.any(denominator <= 0.0)
    ):
        raise Protocol101MaxTContractError(
            "zero or nonfinite variance prevents all-row maxT statistic"
        )
    observed_pooled = np.asarray(grid.pnl, dtype=float).sum(axis=2)
    observed_z = observed_pooled / denominator
    observed_t = np.median(observed_z, axis=1)
    null_z = null_pooled / denominator[None, :, :]
    null_t = np.median(null_z, axis=2)
    max_null = np.max(null_t, axis=1)
    counts, p_values = fwer_p_values(observed_t, max_null)
    exceedance_limit = math.floor(
        cfg.alpha * (cfg.replicates + 1) - 1.0
    )
    return MaxTResult(
        observed_z_by_row_seed=observed_z,
        observed_t_by_row=observed_t,
        denominator_by_row_seed=denominator,
        null_t_by_replicate_row=null_t,
        max_null_by_replicate=max_null,
        exceedance_counts=counts,
        p_fwer=p_values,
        hard_pass=p_values <= cfg.alpha,
        exceedance_limit=exceedance_limit,
        max_null_sha256=_numeric_array_sha256(max_null),
    )


def maxT_contract() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101HardMaxTContractV1",
        "rows": list(MAXT_ROWS),
        "family_size": 28,
        "seeds": list(MAXT_SEEDS),
        "input": "repaired per-session OOF net PnL on one grid",
        "purpose": "one-sided FWER control of positive OOF net PnL",
        "observed_statistic": (
            "median_seed(pooled_OOF_net_PnL / frozen_resampled_SD)"
        ),
        "null_statistic": (
            "max_row(median_seed(centered_block_resampled_pooled_PnL / "
            "same_frozen_SD))"
        ),
        "config": MaxTConfig().to_dict(),
        "replicate_seeds": (
            "SeedSequence(master_seed).spawn(20000), order 0..19999"
        ),
        "tie_rule": "greater_than_or_equal",
        "p_fwer": "(1 + count(max_null >= observed_T)) / 20001",
        "hard_pass": "p_FWER <= 0.05 and existing per-row G2 later",
        "exceedance_limit": 999,
        "bonferroni_selectable": False,
        "post_result_method_switching": False,
    }
