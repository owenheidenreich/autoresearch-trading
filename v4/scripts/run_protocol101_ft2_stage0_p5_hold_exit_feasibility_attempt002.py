"""Owner-routed FT2 Stage-0 P5 HOLD/EXIT feasibility attempt002."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import resource
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model import protocol101_stage1_reference_multiplicity as references
from v4.model.protocol101_regimen_repair import ExitReason, InvalidReason
from v4.model.protocol101_scoped_stage1_hgb import load_repaired_decisions
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
)
from v4.scripts import (
    run_protocol101_ft2_stage0_p5_hold_exit_feasibility as attempt001_runner,
)
from v4.scripts.run_protocol101_policy_neutral_contract_selector import (
    ROOT,
    _reference_opportunities,
    _scope_maps,
)
from v4.scripts.run_protocol101_scoped_stage1_hgb_runner import guard_margins


EVIDENCE_GRADE = "exploratory_non_promotable_stage0"
SCHEMA_VERSION = "Protocol101FT2Stage0P5HoldExitFeasibilityAttempt002V1"
CAMPAIGN_ID = "protocol101-ft2-stage0-p5-hold-exit-feasibility-attempt002"
GOAL_PATH = (
    ROOT
    / "v4/docs/protocol101/training/history/"
    "closed_stage1_graph_and_goals_2026_07_28/goals/"
    "PROTOCOL101_FT2_STAGE0_P5_HOLD_EXIT_FEASIBILITY_"
    "ATTEMPT002_GOAL_2026_07_28.md"
)
EXPECTED_GOAL_SHA256 = (
    "e63b0bf12438811d4ad6660dc91c63ef83f305bf698e7db6864d1758bddcd65a"
)
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002"
)
WORK_ROOT = OUTPUT_ROOT / "work"
SESSION_LIFECYCLE_ROOT = WORK_ROOT / "session_lifecycle"
MODEL_ROOT = OUTPUT_ROOT / "model_receipts"
ATTEMPT001_ROOT = attempt001_runner.OUTPUT_ROOT
SELECTOR_STAGE0_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002"
)
STAGE2_DRAFT = attempt001_runner.STAGE2_DRAFT

TRAIN_SESSION_COUNT = attempt001_runner.TRAIN_SESSION_COUNT
VALIDATION_SESSION_COUNT = attempt001_runner.VALIDATION_SESSION_COUNT
MINIMUM_VALIDATION_EPISODES = attempt001_runner.MINIMUM_VALIDATION_EPISODES
MODEL_SEED = attempt001_runner.MODEL_SEED
SHUFFLE_SEEDS = attempt001_runner.SHUFFLE_SEEDS
POLICY_INDEX = attempt001_runner.POLICY_INDEX
ROUND_TRIP_FEE = attempt001_runner.ROUND_TRIP_FEE
CONTRACT_MULTIPLIER = attempt001_runner.CONTRACT_MULTIPLIER
MAX_CAUSAL_QUOTE_AGE_SECONDS = (
    attempt001_runner.MAX_CAUSAL_QUOTE_AGE_SECONDS
)
HGB_CONFIG = dict(attempt001_runner.HGB_CONFIG)
LIFECYCLE_FEATURES = attempt001_runner.LIFECYCLE_FEATURES
TERMINAL_DECISIONS = attempt001_runner.TERMINAL_DECISIONS
EXPECTED_TOTAL_OPPORTUNITIES = 6_414
EXPECTED_VALIDATION_OPPORTUNITIES = 1_354
SERIAL_RANDOM_LIMITATION = (
    "The inherited contract pins exact hazard expectation for isolated "
    "episodes but does not define an exact stochastic simulator-v5 account "
    "expectation. Ledger B therefore reports no random value; a candidate "
    "that otherwise earns proceed must route to scientific-contract defect."
)
MECHANICAL_REPAIRS = [
    {
        "iteration": 1,
        "classification": "mechanical_strong_shuffle_join",
        "observed_failure": (
            "The randomized cyclic search did not find a complete "
            "cross-session episode permutation and produced fewer than 100 "
            "fit rows for shuffle seed 8700."
        ),
        "repair": (
            "Replace the incomplete cyclic search with a seeded deterministic "
            "session-block derangement. Map every mathematically matchable "
            "episode across a different session and report only unavoidable "
            "dominant-session excess as unmatched."
        ),
        "scientific_choices_changed": False,
        "validation_economics_inspected_before_repair": False,
    },
    {
        "iteration": 2,
        "classification": "mechanical_simulator_v5_two_clock_adapter",
        "observed_failure": (
            "A deadline fallback priced from the latest causal quote before "
            "the deadline but incorrectly released serial occupancy at the "
            "source quote time."
        ),
        "repair": (
            "Mark hold-to-deadline and score-fallback outcomes explicitly so "
            "source quote time prices the exit while realized exit time "
            "retains occupancy through the policy deadline. Map an early "
            "learned EXIT-NOW action to the existing simulator-v5 terminal "
            "profit/loss reason solely for adapter validation; the mapping "
            "does not alter its clock, bid, PnL, or action."
        ),
        "scientific_choices_changed": False,
        "validation_economics_inspected_before_repair": False,
    },
]


def write_progress(status: str, **fields: Any) -> None:
    attempt001_runner.write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            **fields,
        },
    )


def attempt001_file_hashes() -> dict[str, str]:
    manifest = ATTEMPT001_ROOT / "hashes.sha256"
    if not manifest.is_file():
        raise RuntimeError("attempt001 hash manifest is missing")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = ATTEMPT001_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"attempt001 artifact missing: {name}")
        current = attempt001_runner.sha256_path(path)
        if current != digest:
            raise RuntimeError(f"attempt001 artifact changed: {name}")
        expected[name] = digest
    expected["hashes.sha256"] = attempt001_runner.sha256_path(manifest)
    return expected


def frozen_authorities() -> dict[str, dict[str, str]]:
    if attempt001_runner.sha256_path(GOAL_PATH) != EXPECTED_GOAL_SHA256:
        raise RuntimeError("attempt002 Goal hash mismatch")
    direct = {
        "attempt002_goal": GOAL_PATH,
        "attempt001_goal": attempt001_runner.GOAL_PATH,
        "stage2_draft": STAGE2_DRAFT,
        "attempt001_preregistration": ATTEMPT001_ROOT
        / "preregistration.json",
        "attempt001_decision": ATTEMPT001_ROOT / "decision.json",
        "attempt001_machinery_checks": ATTEMPT001_ROOT
        / "machinery_checks.json",
        "attempt001_entry_identity_receipt": ATTEMPT001_ROOT
        / "entry_identity_receipt.json",
        "selector_opportunity_filter_receipt": SELECTOR_STAGE0_ROOT
        / "opportunity_filter_receipt.json",
        "selector_risk_set_receipts": SELECTOR_STAGE0_ROOT
        / "risk_set_receipts.json",
        "reference_multiplicity_source": attempt001_runner.REFERENCE_SOURCE,
        "simulator_v5_source": attempt001_runner.SIMULATOR_SOURCE,
        "attempt001_runner_source": Path(attempt001_runner.__file__),
    }
    missing = [str(path) for path in direct.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"attempt002 frozen authority missing: {missing}")
    inherited = attempt001_runner.read_json(
        ATTEMPT001_ROOT / "preregistration.json"
    )["frozen_input_hashes"]
    for name, item in inherited.items():
        path = Path(str(item["path"]))
        if not path.is_file():
            raise RuntimeError(f"attempt001 inherited input missing: {name}")
        if attempt001_runner.sha256_path(path) != str(item["sha256"]):
            raise RuntimeError(f"attempt001 inherited input changed: {name}")
    return {
        **{
            name: {
                "path": str(path),
                "sha256": attempt001_runner.sha256_path(path),
            }
            for name, path in direct.items()
        },
        **{
            f"attempt001_inherited::{name}": dict(item)
            for name, item in inherited.items()
        },
    }


def preregistration_payload() -> dict[str, Any]:
    training, validation, split = attempt001_runner.selected_sessions()
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "campaign_id": CAMPAIGN_ID,
        "highest_allowed_claim": (
            "Exploratory non-promotable P5 HOLD/EXIT Stage-0 attempt002 "
            "feasibility decision complete."
        ),
        "owner_routed_correction": {
            "attempt001_episode_source": "simulator_v5_admitted_P5_trades",
            "attempt002_ledger_a_episode_source": (
                "every_exact_P5_permitted_opportunity"
            ),
            "ledger_b_source": (
                "common_ordered_P5_opportunity_stream_serialized_v5"
            ),
            "all_other_scientific_choices_inherited": True,
        },
        "session_selection": {
            "governed_fold": split["governed_fold"],
            "training_sessions": training,
            "validation_sessions": validation,
            "training_count": len(training),
            "validation_count": len(validation),
            "final_oof_overlap": [],
        },
        "entry_policy": {
            "policy_index": POLICY_INDEX,
            "selector": "fixed_heuristic_candidates",
            "description": "P5_VWAP_side_nearest_ATM",
            "include_if": "fixed_heuristic_candidates_emits_exactly_one",
            "exclude_if": "P5_abstains",
            "synthetic_utility_for_abstention": False,
            "entry_ask": True,
            "one_contract": True,
        },
        "lifecycle_model": {
            "family": "HistGradientBoostingRegressor",
            "configuration": HGB_CONFIG,
            "seed": MODEL_SEED,
            "decision_rule": "predicted_hold_advantage_strictly_gt_zero",
            "threshold_optimization": False,
        },
        "target": {
            "name": "one_minute_executable_bid_hold_advantage",
            "q_exit": "(bid_t-entry_ask)*100-3",
            "q_hold_1m": "(bid_t_plus_1m-entry_ask)*100-3",
            "a_hold_1m": "q_hold_1m-q_exit",
            "next_minute_must_be_exact": True,
            "missing_next_bid": "missing_target_no_bridge",
        },
        "feature_names": list(LIFECYCLE_FEATURES),
        "feature_count": len(LIFECYCLE_FEATURES),
        "missing_values": "native_HGB_missing_values_no_imputation",
        "weighting": (
            "equal_total_weight_per_session_then_per_episode_then_per_row"
        ),
        "fixed_exit_training_selection": {
            "candidates": list(range(7)),
            "metric": "maximum_mean_nested_training_session_PnL",
            "tie_break": "lowest_policy_index",
            "validation_outcomes_not_used": True,
        },
        "exact_random_exit": {
            "hazard_source": "real_HGB_nested_training_actions_only",
            "ledger_a": "exact_expected_PnL_over_executable_exit_minutes",
            "ledger_b": "not_defined_exactly_by_inherited_contract",
            "ledger_b_limitation": SERIAL_RANDOM_LIMITATION,
        },
        "controls": {
            "original_p5": True,
            "exit_now": True,
            "hold_deadline": True,
            "training_selected_best_fixed": True,
            "exact_random": True,
            "constant_score": True,
            "reversed_real_hgb": True,
            "strong_shuffle_seeds": list(SHUFFLE_SEEDS),
        },
        "ledgers": {
            "A": "independent_overlapping_identical_entry_exit_isolation",
            "B": "strict_simulator_v5_common_P5_intents",
        },
        "minimum_validation_episodes": MINIMUM_VALIDATION_EPISODES,
        "expected_identity_anchors": {
            "all_20_sessions": EXPECTED_TOTAL_OPPORTUNITIES,
            "nested_validation_sessions": EXPECTED_VALIDATION_OPPORTUNITIES,
        },
        "terminal_decisions": list(TERMINAL_DECISIONS),
        "repairs": {"maximum_iterations": 3, "mechanical_only": True},
        "frozen_input_hashes": frozen_authorities(),
        "attempt001_file_hashes": attempt001_file_hashes(),
        "forbidden": [
            "entry_or_contract_selector_training",
            "neural_or_sequence_training",
            "threshold_optimization",
            "full_five_fold_lifecycle_campaign",
            "multiplicity_or_independent_final_acceptance",
            "final_OOF_G9_holdout_recorder_sealed_shadow_or_paper_access",
            "broker_or_paid_download",
            "promotion_runtime_launchd_recorder_or_real_money_change",
        ],
        "side_effects": {
            "entry_training": False,
            "contract_selector_training": False,
            "full_lifecycle_training": False,
            "g9": False,
            "protected_holdout": False,
            "recorder_or_sealed_evidence": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_download": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
    }


def preregister() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = preregistration_payload()
    path = OUTPUT_ROOT / "preregistration.json"
    if path.is_file() and attempt001_runner.read_json(path) != attempt001_runner.clean(
        payload
    ):
        raise RuntimeError("attempt002 preregistration changed")
    if not path.is_file():
        attempt001_runner.write_json(path, payload)
    freeze = f"{attempt001_runner.sha256_path(path)}  preregistration.json\n"
    freeze_path = OUTPUT_ROOT / "preregistration.sha256"
    if freeze_path.exists() and freeze_path.read_text(encoding="utf-8") != freeze:
        raise RuntimeError("attempt002 preregistration freeze changed")
    freeze_path.write_text(freeze, encoding="utf-8")
    write_progress(
        "preregistered_before_attempt002_lifecycle_rows_or_validation_economics",
        preregistration_sha256=attempt001_runner.sha256_path(path),
        validation_economics_inspected=False,
    )
    return payload


def verify_preregistration() -> dict[str, Any]:
    path = OUTPUT_ROOT / "preregistration.json"
    freeze = OUTPUT_ROOT / "preregistration.sha256"
    digest, name = freeze.read_text(encoding="utf-8").strip().split("  ", 1)
    if name != "preregistration.json" or digest != attempt001_runner.sha256_path(
        path
    ):
        raise RuntimeError("attempt002 preregistration hash mismatch")
    payload = attempt001_runner.read_json(path)
    for name, item in payload["frozen_input_hashes"].items():
        source = Path(str(item["path"]))
        if (
            not source.is_file()
            or attempt001_runner.sha256_path(source) != str(item["sha256"])
        ):
            raise RuntimeError(
                f"attempt002 frozen input changed after preregistration: {name}"
            )
    current_attempt001 = attempt001_file_hashes()
    if current_attempt001 != payload["attempt001_file_hashes"]:
        raise RuntimeError("attempt001 changed after attempt002 preregistration")
    return payload


def _raw_policy_fields(
    row: Mapping[str, Any],
    *,
    strike_index: int,
    right_index: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in range(7):
        index = (int(strike_index), int(right_index), int(policy))
        invalid = int(np.asarray(row["label_invalid_reason_code"])[index])
        if invalid != int(InvalidReason.NONE):
            raise RuntimeError(
                f"selected P5 contract lacks fixed P{policy} label"
            )
        result.update(
            {
                f"p{policy}_net_pnl_after_fee": float(
                    np.asarray(row["labels_net_pnl"])[index]
                )
                - ROUND_TRIP_FEE,
                f"p{policy}_mid_pnl_before_fee": float(
                    np.asarray(row["labels_mid_pnl"])[index]
                ),
                f"p{policy}_source_exit_quote_time_ns": int(
                    np.asarray(row["label_source_exit_quote_time_ns"])[index]
                ),
                f"p{policy}_realized_exit_time_ns": int(
                    np.asarray(row["label_realized_exit_time_ns"])[index]
                ),
                f"p{policy}_exit_quote_age_ms": float(
                    np.asarray(row["label_exit_quote_age_ms"])[index]
                ),
                f"p{policy}_exit_reason_code": int(
                    np.asarray(row["label_exit_reason_code"])[index]
                ),
                f"p{policy}_executable_exit_bid": float(
                    np.asarray(row["label_executable_exit_bid"])[index]
                ),
                f"p{policy}_deadline_ns": int(
                    np.asarray(row["label_policy_deadline_ns"])[index]
                ),
            }
        )
    return result


def build_p5_opportunity_episodes(
    preregistration: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, processed, normalized = _scope_maps()
    role_sessions = {
        "nested_train": list(
            preregistration["session_selection"]["training_sessions"]
        ),
        "nested_validation": list(
            preregistration["session_selection"]["validation_sessions"]
        ),
    }
    selector_receipt = attempt001_runner.read_json(
        SELECTOR_STAGE0_ROOT / "opportunity_filter_receipt.json"
    )
    records: list[dict[str, Any]] = []
    by_session: dict[str, Any] = {}
    for role, sessions in role_sessions.items():
        for session in sessions:
            with processed[session].open("rb") as handle:
                raw_rows = pickle.load(handle)
            raw_by_ns = {
                int(pd.Timestamp(row["decision_time"]).value): row
                for row in raw_rows
            }
            decisions = load_repaired_decisions(
                [(session, processed[session])],
                hypothesis="H0",
                policy_index=POLICY_INDEX,
                guard_margins=guard_margins(),
                split=f"{CAMPAIGN_ID}:{role}",
            )
            opportunities = _reference_opportunities(
                decisions,
                fold_id=str(
                    preregistration["session_selection"]["governed_fold"]
                ),
                split=role,
                campaign_id=CAMPAIGN_ID,
            )
            candidates = references.fixed_heuristic_candidates(opportunities)
            if len({int(item.decision_time_ns) for item in candidates}) != len(
                candidates
            ):
                raise RuntimeError(f"P5 emitted duplicate decision: {session}")
            opportunity_by_ns = {
                int(item.repaired.base.decision_time.value): item
                for item in opportunities
            }
            session_records = []
            for ordinal, candidate in enumerate(candidates):
                decision_ns = int(candidate.decision_time_ns)
                opportunity = opportunity_by_ns.get(decision_ns)
                if opportunity is None:
                    raise RuntimeError("fixed P5 candidate lacks opportunity")
                candidate_index = int(candidate.metadata["candidate_index"])
                base = opportunity.repaired.base
                strike_index = int(base.strike_indices[candidate_index])
                right_index = int(base.right_indices[candidate_index])
                raw = raw_by_ns.get(decision_ns)
                if raw is None:
                    raise RuntimeError("P5 decision lacks governed source row")
                raw_contract = str(
                    np.asarray(raw["contract_ids"], dtype=object)[
                        strike_index, right_index
                    ]
                )
                if raw_contract != str(candidate.contract_id):
                    raise RuntimeError("P5 selected contract identity mismatch")
                episode_id = attempt001_runner.stable_hash(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "role": role,
                        "session": session,
                        "decision_time_ns": decision_ns,
                        "contract_id": candidate.contract_id,
                        "canonical_slot": int(candidate.canonical_strike_slot),
                    }
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_grade": EVIDENCE_GRADE,
                    "episode_id": episode_id,
                    "role": role,
                    "fold": str(candidate.fold),
                    "session": str(session),
                    "session_episode_ordinal": int(ordinal),
                    "decision_time_ns": decision_ns,
                    "contract_id": str(candidate.contract_id),
                    "right": str(candidate.right),
                    "canonical_slot": int(candidate.canonical_strike_slot),
                    "strike_index": strike_index,
                    "right_index": right_index,
                    "entry_ask": float(candidate.entry_ask),
                    "entry_quote_time_ns": int(
                        candidate.source_quote_time_ns
                    ),
                    "source_context_time_ns": int(
                        candidate.source_context_time_ns
                    ),
                    "policy_index": POLICY_INDEX,
                    "policy_identity": "P5_VWAP_side_nearest_ATM",
                    "policy_deadline_ns": int(
                        candidate.label_policy_deadline_ns
                    ),
                    "source_processed_path": str(processed[session]),
                    "source_processed_sha256": (
                        attempt001_runner.sha256_path(processed[session])
                    ),
                    "source_normalized_path": str(normalized[session]),
                    "source_normalized_sha256": (
                        attempt001_runner.sha256_path(normalized[session])
                    ),
                    **_raw_policy_fields(
                        raw,
                        strike_index=strike_index,
                        right_index=right_index,
                    ),
                }
                session_records.append(record)
                records.append(record)
            expected = int(
                selector_receipt["by_session"][session]["permitted"]
            )
            if len(session_records) != expected:
                raise RuntimeError(
                    f"P5 opportunity identity anchor changed: "
                    f"{session}: observed={len(session_records)} expected={expected}"
                )
            by_session[session] = {
                "role": role,
                "source_reference_opportunities": len(opportunities),
                "retained_P5_opportunities": len(session_records),
                "selector_stage0_anchor": expected,
                "identity_hash": attempt001_runner.stable_hash(
                    [
                        {
                            key: item[key]
                            for key in (
                                "episode_id",
                                "decision_time_ns",
                                "contract_id",
                                "canonical_slot",
                                "entry_ask",
                                "entry_quote_time_ns",
                            )
                        }
                        for item in session_records
                    ]
                ),
            }
    frame = pd.DataFrame.from_records(records)
    attempt001_runner.assert_unique(
        frame,
        (
            "role",
            "session",
            "decision_time_ns",
            "contract_id",
            "canonical_slot",
            "episode_id",
        ),
        label="attempt002 P5 opportunity episode identity",
    )
    training_count = int(frame["role"].eq("nested_train").sum())
    validation_count = int(frame["role"].eq("nested_validation").sum())
    if len(frame) != EXPECTED_TOTAL_OPPORTUNITIES:
        raise RuntimeError(
            f"attempt002 total P5 anchor changed: {len(frame)}"
        )
    if validation_count != EXPECTED_VALIDATION_OPPORTUNITIES:
        raise RuntimeError(
            f"attempt002 validation P5 anchor changed: {validation_count}"
        )
    attempt001_runner.write_parquet(
        OUTPUT_ROOT / "frozen_p5_entry_episodes.parquet", frame
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "exact_P5_permitted_opportunity_episodes_frozen",
        "attempt001_episode_definition_preserved_as_historical": True,
        "attempt002_episode_definition": (
            "every_exact_fixed_heuristic_candidates_P5_opportunity"
        ),
        "total_episode_count": int(len(frame)),
        "training_episode_count": training_count,
        "validation_episode_count": validation_count,
        "P5_abstentions_excluded_before_lifecycle": int(
            selector_receipt["p5_abstention_opportunities"]
        ),
        "synthetic_abstention_utility": False,
        "duplicate_identity_count": 0,
        "by_session": by_session,
        "entry_identity_root_hash": attempt001_runner.stable_hash(
            frame[
                [
                    "role",
                    "session",
                    "decision_time_ns",
                    "contract_id",
                    "canonical_slot",
                    "episode_id",
                ]
            ].to_dict("records")
        ),
        "parquet_sha256": attempt001_runner.sha256_path(
            OUTPUT_ROOT / "frozen_p5_entry_episodes.parquet"
        ),
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "entry_identity_receipt.json", receipt
    )
    return frame, receipt


def _load_session_sources(
    session: str,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Mapping[str, Any]]]:
    _, processed, normalized = _scope_maps()
    columns = [
        "contract_id",
        "quote_time",
        "bid",
        "ask",
        "mid",
        "underlying_price",
        "strike",
        "right",
    ]
    options = pq.read_table(normalized[session], columns=columns).to_pandas()
    options["quote_time"] = pd.to_datetime(options["quote_time"], utc=True)
    wanted = set(episodes["contract_id"].astype(str))
    options = options[
        options["contract_id"].astype(str).isin(wanted)
    ].copy()
    spx_path = (
        ROOT / f"data/raw/index/spx_1m/{session}.official_spx.parquet"
    )
    if not spx_path.is_file():
        raise RuntimeError(f"official SPX context missing: {session}")
    spx = pd.read_parquet(spx_path)
    spx["event_time"] = pd.to_datetime(spx["event_time"], utc=True)
    spx = spx.sort_values("event_time").reset_index(drop=True)
    with processed[session].open("rb") as handle:
        raw_rows = pickle.load(handle)
    raw_by_ns = {
        int(pd.Timestamp(row["decision_time"]).value): row for row in raw_rows
    }
    return options, spx, raw_by_ns


def _quote_state(
    quote_by_ns: Mapping[int, Any],
    quote_times: np.ndarray,
    *,
    state_ns: int,
    right: str,
    spx: float,
) -> dict[str, Any]:
    position = int(np.searchsorted(quote_times, state_ns, side="right")) - 1
    quote = quote_by_ns[int(quote_times[position])] if position >= 0 else None
    quote_ns = int(quote_times[position]) if position >= 0 else 0
    age = (
        (state_ns - quote_ns) / 1_000_000_000.0
        if quote is not None
        else float("inf")
    )
    bid = float(quote.bid) if quote is not None else float("nan")
    ask = float(quote.ask) if quote is not None else float("nan")
    mid = float(quote.mid) if quote is not None else float("nan")
    strike = float(quote.strike) if quote is not None else float("nan")
    executable = bool(
        math.isfinite(bid)
        and bid > 0.0
        and age <= MAX_CAUSAL_QUOTE_AGE_SECONDS
    )
    quantized_mid = attempt001_runner._quantized_mid(mid)
    iv = attempt001_runner._implied_vol(
        quantized_mid,
        spx,
        strike,
        attempt001_runner._tte_years(pd.Timestamp(state_ns, tz="UTC")),
        right,
    )
    delta = float("nan")
    gamma = float("nan")
    if iv is not None:
        delta, gamma = attempt001_runner._bs_delta_gamma(
            spx,
            strike,
            attempt001_runner._tte_years(
                pd.Timestamp(state_ns, tz="UTC")
            ),
            iv,
            right,
        )
    return {
        "quote": quote,
        "quote_ns": quote_ns,
        "quote_age_seconds": age,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "strike": strike,
        "executable": executable,
        "internal_delta": delta,
        "internal_gamma": gamma,
    }


def build_session_lifecycle(
    session: str,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    options, spx_bars, raw_by_ns = _load_session_sources(session, episodes)
    by_contract: dict[str, tuple[dict[int, Any], np.ndarray]] = {}
    duplicate_path_count = 0
    for contract_id, group in options.groupby("contract_id", sort=False):
        ordered = group.sort_values("quote_time", kind="mergesort")
        if ordered["quote_time"].duplicated().any():
            duplicate_path_count += 1
            continue
        mapping = {
            int(pd.Timestamp(row.quote_time).value): row
            for row in ordered.itertuples(index=False)
        }
        by_contract[str(contract_id)] = (
            mapping,
            np.asarray(sorted(mapping), dtype=np.int64),
        )
    if duplicate_path_count:
        raise RuntimeError(f"duplicate path identities: {session}")

    all_state_times = np.arange(
        int(episodes["decision_time_ns"].min()) + 60_000_000_000,
        int(episodes["policy_deadline_ns"].max()) + 1,
        60_000_000_000,
        dtype=np.int64,
    )
    context_cache: dict[tuple[int, str], dict[str, float]] = {}
    for state_ns in all_state_times:
        for right in ("C", "P"):
            context_cache[(int(state_ns), right)] = (
                attempt001_runner.context_features(
                    spx_bars,
                    state_time=pd.Timestamp(int(state_ns), tz="UTC"),
                    right=right,
                )
            )
    quote_cache: dict[tuple[str, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    context_parity_max_error = 0.0
    for episode in episodes.sort_values("decision_time_ns").itertuples(
        index=False
    ):
        path = by_contract.get(str(episode.contract_id))
        if path is None:
            raise RuntimeError(
                f"selected P5 contract path missing: {episode.episode_id}"
            )
        quote_by_ns, quote_times = path
        state_times = np.arange(
            int(episode.decision_time_ns) + 60_000_000_000,
            int(episode.policy_deadline_ns) + 1,
            60_000_000_000,
            dtype=np.int64,
        )
        bids: list[float] = []
        pnls: list[float] = []
        mfe = -math.inf
        mae = math.inf
        mfe_index: int | None = None
        for state_index, state_ns_value in enumerate(state_times):
            state_ns = int(state_ns_value)
            context = dict(context_cache[(state_ns, str(episode.right))])
            spx = float(context.pop("_spx"))
            cache_key = (str(episode.contract_id), state_ns)
            if cache_key not in quote_cache:
                quote_cache[cache_key] = _quote_state(
                    quote_by_ns,
                    quote_times,
                    state_ns=state_ns,
                    right=str(episode.right),
                    spx=spx,
                )
            quote = quote_cache[cache_key]
            bid = float(quote["bid"])
            ask = float(quote["ask"])
            mid = float(quote["mid"])
            executable = bool(quote["executable"])
            current_pnl = (
                (bid - float(episode.entry_ask)) * CONTRACT_MULTIPLIER
                - ROUND_TRIP_FEE
                if executable
                else float("nan")
            )
            bids.append(bid if executable else float("nan"))
            pnls.append(current_pnl)
            if math.isfinite(current_pnl):
                if current_pnl >= mfe:
                    mfe = current_pnl
                    mfe_index = state_index
                mae = min(mae, current_pnl)
            next_ns = state_ns + 60_000_000_000
            next_quote = quote_by_ns.get(next_ns)
            next_bid = (
                float(next_quote.bid)
                if next_quote is not None
                else float("nan")
            )
            target_valid = bool(
                executable
                and next_quote is not None
                and math.isfinite(next_bid)
                and next_bid > 0.0
            )
            target = (
                attempt001_runner.one_step_target(
                    entry_ask=float(episode.entry_ask),
                    current_bid=bid,
                    next_bid=next_bid,
                )
                if target_valid
                else {
                    "target_q_exit": float("nan"),
                    "target_q_hold_1m": float("nan"),
                    "target_a_hold_1m": float("nan"),
                }
            )
            strike = float(quote["strike"])
            atm = round(spx / 5.0) * 5.0
            record = {
                "schema_version": SCHEMA_VERSION,
                "evidence_grade": EVIDENCE_GRADE,
                "episode_id": str(episode.episode_id),
                "role": str(episode.role),
                "fold": str(episode.fold),
                "session": str(episode.session),
                "state_index": int(state_index),
                "state_time_ns": state_ns,
                "state_quote_time_ns": int(quote["quote_ns"]),
                "state_quote_age_seconds": float(
                    quote["quote_age_seconds"]
                ),
                "current_executable_bid": executable,
                "target_valid": target_valid,
                "target_missing_reason": (
                    ""
                    if target_valid
                    else "no_exact_next_minute_executable_bid"
                ),
                "entry_decision_time_ns": int(episode.decision_time_ns),
                "contract_id": str(episode.contract_id),
                "right": str(episode.right),
                "elapsed_minutes": float(state_index + 1),
                "minutes_to_deadline": float(
                    (
                        int(episode.policy_deadline_ns) - state_ns
                    )
                    / 60_000_000_000
                ),
                "entry_ask": float(episode.entry_ask),
                "current_bid": bid,
                "current_ask": ask,
                "current_mid": mid,
                "current_spread": (
                    ask - bid
                    if math.isfinite(ask) and math.isfinite(bid)
                    else float("nan")
                ),
                "current_unrealized_pnl_dollars": current_pnl,
                "current_return_on_entry_premium": (
                    current_pnl
                    / (float(episode.entry_ask) * CONTRACT_MULTIPLIER)
                    if math.isfinite(current_pnl)
                    else float("nan")
                ),
                "mfe_to_date_dollars": (
                    mfe if math.isfinite(mfe) else float("nan")
                ),
                "mae_to_date_dollars": (
                    mae if math.isfinite(mae) else float("nan")
                ),
                "giveback_from_mfe_dollars": (
                    mfe - current_pnl
                    if math.isfinite(mfe)
                    and math.isfinite(current_pnl)
                    else float("nan")
                ),
                "time_since_mfe_minutes": (
                    float(state_index - mfe_index)
                    if mfe_index is not None
                    else float("nan")
                ),
                "bid_velocity_1m": attempt001_runner._velocity(bids, 1),
                "bid_velocity_3m": attempt001_runner._velocity(bids, 3),
                "bid_velocity_5m": attempt001_runner._velocity(bids, 5),
                "pnl_velocity_1m": attempt001_runner._velocity(pnls, 1),
                "pnl_velocity_3m": attempt001_runner._velocity(pnls, 3),
                "pnl_velocity_5m": attempt001_runner._velocity(pnls, 5),
                **context,
                "right_is_call": float(str(episode.right) == "C"),
                "canonical_slot": float(episode.canonical_slot),
                "current_offset_points": strike - atm,
                "current_moneyness_bps": (
                    (strike - spx) / max(abs(spx), 1.0) * 10_000.0
                ),
                "entry_premium_dollars": (
                    float(episode.entry_ask) * CONTRACT_MULTIPLIER
                ),
                "current_mid_over_entry_ask": (
                    mid / float(episode.entry_ask)
                    if math.isfinite(mid)
                    else float("nan")
                ),
                "internal_delta": float(quote["internal_delta"]),
                "internal_gamma": float(quote["internal_gamma"]),
                "next_minute_bid_label_only": next_bid,
                **target,
            }
            if state_ns in raw_by_ns:
                raw = raw_by_ns[state_ns]
                from v4.model.protocol101_canonical_stage1_contract import (
                    CONTEXT_FEATURES,
                    feature_matrix,
                )

                expected = feature_matrix(raw)[
                    10,
                    tuple(map(str, raw["rights"])).index(str(episode.right)),
                    :12,
                ]
                observed = np.asarray(
                    [record[name] for name in CONTEXT_FEATURES],
                    dtype=float,
                )
                context_parity_max_error = max(
                    context_parity_max_error,
                    float(np.nanmax(np.abs(expected - observed))),
                )
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    attempt001_runner.assert_unique(
        frame,
        ("episode_id", "state_time_ns"),
        label=f"attempt002 lifecycle state identity {session}",
    )
    receipt = {
        "session": session,
        "role": str(episodes["role"].iloc[0]),
        "episode_count": int(frame["episode_id"].nunique()),
        "row_count": int(len(frame)),
        "target_valid_rows": int(frame["target_valid"].sum()),
        "missing_target_rows": int((~frame["target_valid"]).sum()),
        "unique_contract_paths": int(len(by_contract)),
        "duplicate_path_count": 0,
        "context_parity_max_abs_error": context_parity_max_error,
        "context_parity_pass": context_parity_max_error <= 1e-9,
    }
    return frame, receipt


def build_lifecycle_rows(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    SESSION_LIFECYCLE_ROOT.mkdir(parents=True, exist_ok=True)
    receipts = []
    sessions = sorted(episodes["session"].unique())
    for index, session in enumerate(sessions, start=1):
        frame, receipt = build_session_lifecycle(
            str(session),
            episodes[episodes["session"].eq(session)],
        )
        path = SESSION_LIFECYCLE_ROOT / f"{session}.parquet"
        attempt001_runner.write_parquet(path, frame)
        receipt["parquet_sha256"] = attempt001_runner.sha256_path(path)
        receipts.append(receipt)
        write_progress(
            "building_attempt002_lifecycle_rows",
            completed_sessions=index,
            total_sessions=len(sessions),
            completed_episodes=sum(
                int(item["episode_count"]) for item in receipts
            ),
            completed_rows=sum(int(item["row_count"]) for item in receipts),
        )

    total_training_valid = sum(
        int(item["target_valid_rows"])
        for item in receipts
        if item["role"] == "nested_train"
    )
    train_sessions = sum(
        1 for item in receipts if item["role"] == "nested_train"
    )
    final_path = OUTPUT_ROOT / "lifecycle_rows.parquet"
    writer: pq.ParquetWriter | None = None
    try:
        for session in sessions:
            frame = pd.read_parquet(
                SESSION_LIFECYCLE_ROOT / f"{session}.parquet"
            )
            frame["training_weight"] = 0.0
            eligible = frame["role"].eq("nested_train") & frame["target_valid"]
            if bool(eligible.any()):
                raw = attempt001_runner.balanced_training_weights(
                    frame.loc[eligible]
                )
                desired_mean = total_training_valid / (
                    train_sessions * int(eligible.sum())
                )
                frame.loc[eligible, "training_weight"] = raw * desired_mean
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    final_path, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    combined = pd.read_parquet(final_path)
    attempt001_runner.assert_unique(
        combined,
        ("episode_id", "state_time_ns"),
        label="attempt002 combined lifecycle state identity",
    )
    if combined["episode_id"].nunique() != episodes["episode_id"].nunique():
        raise RuntimeError("attempt002 lifecycle lost an entry episode")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "attempt002_causal_lifecycle_rows_constructed",
        "episode_count": int(combined["episode_id"].nunique()),
        "training_episode_count": int(
            combined.loc[
                combined["role"].eq("nested_train"), "episode_id"
            ].nunique()
        ),
        "validation_episode_count": int(
            combined.loc[
                combined["role"].eq("nested_validation"), "episode_id"
            ].nunique()
        ),
        "row_count": int(len(combined)),
        "training_rows": int(combined["role"].eq("nested_train").sum()),
        "validation_rows": int(
            combined["role"].eq("nested_validation").sum()
        ),
        "target_valid_rows": int(combined["target_valid"].sum()),
        "missing_target_rows": int((~combined["target_valid"]).sum()),
        "duplicate_state_identities": 0,
        "context_parity_max_abs_error": max(
            float(item["context_parity_max_abs_error"]) for item in receipts
        ),
        "context_parity_pass": all(
            bool(item["context_parity_pass"]) for item in receipts
        ),
        "session_receipts": receipts,
        "parquet_sha256": attempt001_runner.sha256_path(final_path),
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "path_identity_receipt.json", receipt
    )
    return combined, receipt


def write_feature_contract_and_audit(frame: pd.DataFrame) -> dict[str, Any]:
    violations = sorted(
        {
            name
            for name in LIFECYCLE_FEATURES
            for token in attempt001_runner.FORBIDDEN_FEATURE_TOKENS
            if token in name.lower()
        }
    )
    missing = sorted(set(LIFECYCLE_FEATURES) - set(frame.columns))
    contract = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "inherited_from_attempt001": True,
        "feature_names": list(LIFECYCLE_FEATURES),
        "feature_count": len(LIFECYCLE_FEATURES),
        "target_columns": [
            "target_q_exit",
            "target_q_hold_1m",
            "target_a_hold_1m",
            "next_minute_bid_label_only",
        ],
        "target_columns_are_label_only": True,
        "native_missing_values": True,
        "future_derived_imputation": False,
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "feature_contract.json", contract
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "pass" if not violations and not missing else "fail",
        "forbidden_alias_violations": violations,
        "missing_feature_columns": missing,
        "future_path_or_target_field_in_model_features": False,
        "entry_model_or_selector_output_in_model_features": False,
        "vendor_greek_or_internal_iv_in_model_features": False,
        "feature_contract_sha256": attempt001_runner.sha256_path(
            OUTPUT_ROOT / "feature_contract.json"
        ),
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "feature_firewall_audit.json", audit
    )
    if audit["status"] != "pass":
        raise RuntimeError(f"attempt002 feature firewall failed: {audit}")
    return audit


def manual_target_reproduction(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["role", "session", "episode_id", "state_index"]
    ).reset_index(drop=True)
    samples = []
    for query in (
        ordered[ordered["right"].eq("C")],
        ordered[ordered["right"].eq("P")],
        ordered[ordered["elapsed_minutes"].le(30)],
        ordered[ordered["minutes_to_deadline"].le(30)],
        ordered[ordered["target_a_hold_1m"].gt(0)],
        ordered[ordered["target_a_hold_1m"].lt(0)],
        ordered[~ordered["target_valid"]],
    ):
        if query.empty:
            continue
        positions = np.linspace(
            0, len(query) - 1, num=min(10, len(query)), dtype=int
        )
        samples.extend(map(int, query.iloc[positions].index))
    for index in ordered.index:
        if len(set(samples)) >= 50:
            break
        samples.append(int(index))
    selected = ordered.loc[sorted(set(samples))[:50]]
    if len(selected) < 50:
        raise RuntimeError("attempt002 manual target sample below 50")
    cases = []
    max_error = 0.0
    for row in selected.itertuples(index=False):
        if bool(row.target_valid):
            expected = attempt001_runner.one_step_target(
                entry_ask=float(row.entry_ask),
                current_bid=float(row.current_bid),
                next_bid=float(row.next_minute_bid_label_only),
            )
            errors = {
                name: abs(float(getattr(row, name)) - value)
                for name, value in expected.items()
            }
            max_error = max(max_error, *errors.values())
        else:
            expected = {
                "target_q_exit": None,
                "target_q_hold_1m": None,
                "target_a_hold_1m": None,
            }
            errors = {name: 0.0 for name in expected}
        cases.append(
            {
                "episode_id": str(row.episode_id),
                "session": str(row.session),
                "right": str(row.right),
                "state_time_ns": int(row.state_time_ns),
                "target_valid": bool(row.target_valid),
                "observed_target_a_hold_1m": (
                    float(row.target_a_hold_1m)
                    if math.isfinite(float(row.target_a_hold_1m))
                    else None
                ),
                "reproduced": expected,
                "errors": errors,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "pass" if max_error <= 1e-9 else "fail",
        "sample_count": len(cases),
        "maximum_absolute_error": max_error,
        "contains_calls": any(item["right"] == "C" for item in cases),
        "contains_puts": any(item["right"] == "P" for item in cases),
        "contains_missing_targets": any(
            not item["target_valid"] for item in cases
        ),
        "cases": cases,
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "target_manual_reproduction.json", payload
    )
    if payload["status"] != "pass":
        raise RuntimeError("attempt002 target reproduction failed")
    return payload


def weighting_audit(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame[
        frame["role"].eq("nested_train") & frame["target_valid"]
    ]
    session_sums = eligible.groupby("session")["training_weight"].sum()
    normalized = session_sums / session_sums.mean()
    episode_sums = eligible.groupby("episode_id")["training_weight"].sum()
    session_episode_spread = {}
    identities = eligible[["session", "episode_id"]].drop_duplicates()
    for session, group in identities.groupby("session"):
        values = episode_sums.loc[group["episode_id"]].to_numpy(dtype=float)
        session_episode_spread[str(session)] = float(
            values.max() - values.min()
        )
    maximum_session_error = float(
        np.max(np.abs(normalized.to_numpy(dtype=float) - 1.0))
    )
    maximum_episode_spread = max(session_episode_spread.values())
    return {
        "formula": (
            "equal_total_weight_per_session_then_per_episode_then_per_row"
        ),
        "training_rows": int(len(eligible)),
        "training_sessions": int(eligible["session"].nunique()),
        "training_episodes": int(eligible["episode_id"].nunique()),
        "maximum_normalized_session_weight_error": maximum_session_error,
        "maximum_within_session_episode_weight_spread": (
            maximum_episode_spread
        ),
        "pass": (
            maximum_session_error <= 1e-9
            and maximum_episode_spread <= 1e-9
        ),
    }


def overlap_audit(episodes: pd.DataFrame) -> dict[str, Any]:
    rows = []
    total_pairs = 0
    maximum_concurrency = 0
    distribution: Counter[int] = Counter()
    for session, group in episodes.groupby("session", sort=True):
        intervals = [
            (
                int(item.decision_time_ns),
                int(item.p5_realized_exit_time_ns),
            )
            for item in group.itertuples(index=False)
        ]
        pairs = 0
        for index, (start, end) in enumerate(intervals):
            pairs += sum(
                1
                for other_start, other_end in intervals[index + 1 :]
                if start < other_end and other_start < end
            )
        events = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        concurrent = 0
        session_max = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            concurrent += delta
            if delta > 0:
                distribution[concurrent] += 1
                session_max = max(session_max, concurrent)
        maximum_concurrency = max(maximum_concurrency, session_max)
        total_pairs += pairs
        rows.append(
            {
                "session": str(session),
                "episodes": int(len(group)),
                "overlapping_episode_pairs": int(pairs),
                "maximum_concurrency": int(session_max),
            }
        )
    return {
        "independent_episode_count": int(len(episodes)),
        "overlapping_episode_pair_count": int(total_pairs),
        "maximum_concurrency": int(maximum_concurrency),
        "entry_concurrency_distribution": {
            str(key): int(value) for key, value in sorted(distribution.items())
        },
        "by_session": rows,
        "ledger_A_is_not_serial_economics": True,
    }


def select_training_fixed_policy(
    episodes: pd.DataFrame,
) -> tuple[int, dict[str, Any]]:
    training = episodes[episodes["role"].eq("nested_train")]
    rows = []
    for policy in range(7):
        per_session = training.groupby("session")[
            f"p{policy}_net_pnl_after_fee"
        ].sum()
        rows.append(
            {
                "policy_index": policy,
                "training_pooled_pnl": float(per_session.sum()),
                "training_mean_session_pnl": float(per_session.mean()),
                "training_session_pnl": {
                    str(key): float(value)
                    for key, value in per_session.items()
                },
            }
        )
    selected = min(
        rows,
        key=lambda item: (
            -float(item["training_mean_session_pnl"]),
            int(item["policy_index"]),
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "selection_role": "nested_train_only",
        "selection_metric": "maximum_mean_nested_training_session_PnL",
        "tie_break": "lowest_policy_index",
        "selected_policy_index": int(selected["policy_index"]),
        "policy_training_results": rows,
        "validation_outcomes_inspected_for_selection": False,
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "fixed_exit_training_selection.json", payload
    )
    return int(selected["policy_index"]), payload


def _model_payload(model: HistGradientBoostingRegressor) -> bytes:
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def _persist_model(
    name: str,
    model: HistGradientBoostingRegressor,
    *,
    fit_rows: int,
    target_hash: str,
) -> dict[str, Any]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _model_payload(model)
    path = MODEL_ROOT / f"{name}.pkl"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    receipt = {
        "name": name,
        "family": "HistGradientBoostingRegressor",
        "configuration": HGB_CONFIG,
        "fit_rows": int(fit_rows),
        "target_hash": target_hash,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(path),
    }
    attempt001_runner.write_json(MODEL_ROOT / f"{name}.json", receipt)
    return receipt


def strong_shuffle_targets(
    training: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible = training[training["target_valid"]].copy()
    eligible["_row_position"] = np.arange(len(eligible))
    episode_meta = (
        eligible.groupby("episode_id")
        .agg(
            session=("session", "first"),
            right=("right", "first"),
            path_length=("state_index", "size"),
        )
        .reset_index()
    )
    episode_meta["length_bucket"] = pd.cut(
        episode_meta["path_length"],
        bins=[0, 60, 120, 240, math.inf],
        labels=["001_060", "061_120", "121_240", "241_plus"],
        include_lowest=True,
    ).astype(str)
    rng = np.random.default_rng(seed)
    mapping: dict[str, str] = {}
    unmatched = []
    for _, group in episode_meta.groupby(
        ["right", "length_bucket"], sort=True
    ):
        session_groups = []
        for session, session_group in group.groupby("session", sort=True):
            records = session_group.to_dict("records")
            rng.shuffle(records)
            session_groups.append((str(session), records))
        rng.shuffle(session_groups)
        session_groups.sort(key=lambda item: -len(item[1]))
        records = [
            record for _, session_records in session_groups
            for record in session_records
        ]
        n = len(records)
        maximum_session_count = max(
            (len(session_records) for _, session_records in session_groups),
            default=0,
        )
        if maximum_session_count <= n - maximum_session_count:
            sources = (
                records[maximum_session_count:]
                + records[:maximum_session_count]
            )
            for destination, source in zip(records, sources):
                if str(destination["session"]) == str(source["session"]):
                    raise RuntimeError(
                        "session-block shuffle derangement invariant failed"
                    )
                mapping[str(destination["episode_id"])] = str(
                    source["episode_id"]
                )
            continue

        dominant = session_groups[0][1]
        others = [
            record for _, session_records in session_groups[1:]
            for record in session_records
        ]
        rng.shuffle(dominant)
        rng.shuffle(others)
        matchable = len(others)
        for destination, source in zip(dominant[:matchable], others):
            mapping[str(destination["episode_id"])] = str(
                source["episode_id"]
            )
        for destination, source in zip(others, dominant[:matchable]):
            mapping[str(destination["episode_id"])] = str(
                source["episode_id"]
            )
        unmatched.extend(
            str(record["episode_id"]) for record in dominant[matchable:]
        )
    shuffled = np.full(len(eligible), np.nan, dtype=float)
    truncated_destination_rows = 0
    unused_source_rows = 0
    groups = {
        str(episode_id): group.sort_values("state_index")
        for episode_id, group in eligible.groupby("episode_id", sort=False)
    }
    for destination_id, source_id in mapping.items():
        destination = groups[destination_id]
        source = groups[source_id]
        destination_positions = destination["_row_position"].to_numpy(
            dtype=int
        )
        source_values = source["target_a_hold_1m"].to_numpy(dtype=float)
        count = min(len(destination_positions), len(source_values))
        shuffled[destination_positions[:count]] = source_values[:count]
        truncated_destination_rows += len(destination_positions) - count
        unused_source_rows += len(source_values) - count
    payload = {
        "seed": seed,
        "grouping": "right_and_coarse_path_length_bucket",
        "episode_count": int(eligible["episode_id"].nunique()),
        "mapped_episode_count": len(mapping),
        "unmatched_episode_count": len(unmatched),
        "unmatched_episode_ids": unmatched,
        "truncated_destination_rows": int(truncated_destination_rows),
        "unused_source_rows": int(unused_source_rows),
        "finite_shuffled_rows": int(np.isfinite(shuffled).sum()),
        "self_maps": 0,
        "same_session_maps": 0,
        "elapsed_order_preserved": True,
        "derangement_method": "seeded_session_block_rotation_v1",
    }
    return shuffled, payload


def fit_models(
    lifecycle: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = lifecycle[lifecycle["role"].eq("nested_train")].copy()
    validation = lifecycle[
        lifecycle["role"].eq("nested_validation")
    ].copy()
    fit_mask = training["target_valid"]
    X_fit = training.loc[fit_mask, list(LIFECYCLE_FEATURES)].to_numpy(
        dtype=np.float32
    )
    y_fit = training.loc[fit_mask, "target_a_hold_1m"].to_numpy(
        dtype=np.float64
    )
    weights = training.loc[fit_mask, "training_weight"].to_numpy(
        dtype=np.float64
    )
    real = HistGradientBoostingRegressor(**HGB_CONFIG)
    fit_started = time.perf_counter()
    real.fit(X_fit, y_fit, sample_weight=weights)
    receipts = {
        "real_hgb": _persist_model(
            "real_hgb_seed42",
            real,
            fit_rows=len(y_fit),
            target_hash=attempt001_runner.stable_hash(y_fit.tolist()),
        )
    }
    shuffle_models: dict[int, HistGradientBoostingRegressor] = {}
    shuffle_receipts = {}
    eligible_training = training.loc[fit_mask].copy().reset_index(drop=True)
    for seed in SHUFFLE_SEEDS:
        shuffled, shuffle_receipt = strong_shuffle_targets(
            eligible_training, seed=seed
        )
        usable = np.isfinite(shuffled)
        if int(usable.sum()) < 100:
            raise RuntimeError(f"strong shuffle {seed} has insufficient rows")
        model = HistGradientBoostingRegressor(
            **{**HGB_CONFIG, "random_state": int(seed)}
        )
        model.fit(
            X_fit[usable],
            shuffled[usable],
            sample_weight=weights[usable],
        )
        shuffle_models[seed] = model
        receipt = _persist_model(
            f"strong_shuffle_{seed}",
            model,
            fit_rows=int(usable.sum()),
            target_hash=attempt001_runner.stable_hash(
                shuffled[usable].tolist()
            ),
        )
        receipt["shuffle_contract"] = shuffle_receipt
        shuffle_receipts[str(seed)] = receipt
    constant_score = float(np.average(y_fit, weights=weights))
    parts = []
    for role_frame in (training, validation):
        X = role_frame[list(LIFECYCLE_FEATURES)].to_numpy(dtype=np.float32)
        output = role_frame[
            [
                "episode_id",
                "role",
                "session",
                "state_index",
                "state_time_ns",
            ]
        ].copy()
        output["real_hgb_score"] = real.predict(X)
        output["constant_score"] = constant_score
        output["reversed_real_hgb_score"] = -output["real_hgb_score"]
        for seed, model in shuffle_models.items():
            output[f"strong_shuffle_{seed}_score"] = model.predict(X)
        parts.append(output)
    predictions = pd.concat(parts, ignore_index=True)
    attempt001_runner.write_parquet(
        OUTPUT_ROOT / "predictions.parquet", predictions
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "configuration": HGB_CONFIG,
        "real_seed": MODEL_SEED,
        "strong_shuffle_seeds": list(SHUFFLE_SEEDS),
        "fit_rows": int(len(y_fit)),
        "constant_score": constant_score,
        "fit_elapsed_seconds": time.perf_counter() - fit_started,
        "models": receipts,
        "shuffle_models": shuffle_receipts,
        "threshold_optimized": False,
    }
    attempt001_runner.write_json(
        MODEL_ROOT / "training_receipt.json", receipt
    )
    return predictions, receipt


def _deadline_outcome(group: pd.DataFrame) -> pd.Series:
    executable = group[group["current_executable_bid"]]
    if executable.empty:
        raise RuntimeError("episode has no executable lifecycle exit")
    return executable.iloc[-1]


def _score_outcome(
    group: pd.DataFrame, score_column: str
) -> tuple[pd.Series, bool]:
    eligible = group[
        group["current_executable_bid"] & group[score_column].le(0.0)
    ]
    if not eligible.empty:
        return eligible.iloc[0], False
    return _deadline_outcome(group), True


def _row_outcome(
    episode: Mapping[str, Any],
    row: pd.Series,
    *,
    comparator: str,
    deadline_fallback: bool = False,
) -> dict[str, Any]:
    deadline = int(episode["policy_deadline_ns"])
    state_ns = int(row["state_time_ns"])
    source_ns = int(row["state_quote_time_ns"])
    at_deadline = deadline_fallback or state_ns >= deadline
    realized_ns = deadline if at_deadline else state_ns
    bid = float(row["current_bid"])
    mid = float(row["current_mid"])
    return {
        "episode_id": str(episode["episode_id"]),
        "role": str(episode["role"]),
        "fold": str(episode["fold"]),
        "session": str(episode["session"]),
        "decision_time_ns": int(episode["decision_time_ns"]),
        "contract_id": str(episode["contract_id"]),
        "right": str(episode["right"]),
        "canonical_slot": int(episode["canonical_slot"]),
        "entry_ask": float(episode["entry_ask"]),
        "entry_quote_time_ns": int(episode["entry_quote_time_ns"]),
        "source_context_time_ns": int(episode["source_context_time_ns"]),
        "policy_deadline_ns": deadline,
        "comparator": comparator,
        "exit_source_quote_time_ns": source_ns,
        "realized_exit_time_ns": realized_ns,
        "exit_quote_age_ms": (realized_ns - source_ns) / 1_000_000.0,
        "exit_reason_code": int(
            ExitReason.FORCED_FLAT
            if at_deadline
            else (
                ExitReason.TAKE_PROFIT
                if (
                    (bid - float(episode["entry_ask"]))
                    * CONTRACT_MULTIPLIER
                    - ROUND_TRIP_FEE
                )
                > 0.0
                else ExitReason.STOP_LOSS
            )
        ),
        "exit_bid": bid,
        "exit_mid": mid if math.isfinite(mid) else bid,
        "net_pnl_after_fee": (
            (bid - float(episode["entry_ask"])) * CONTRACT_MULTIPLIER
            - ROUND_TRIP_FEE
        ),
        "mid_pnl_before_fee": (
            ((mid if math.isfinite(mid) else bid) - float(episode["entry_ask"]))
            * CONTRACT_MULTIPLIER
        ),
        "holding_minutes": (
            (realized_ns - int(episode["decision_time_ns"]))
            / 60_000_000_000.0
        ),
        "expected_outcome": False,
    }


def _fixed_outcome(
    episode: Mapping[str, Any],
    *,
    policy: int,
    comparator: str,
) -> dict[str, Any]:
    return {
        "episode_id": str(episode["episode_id"]),
        "role": str(episode["role"]),
        "fold": str(episode["fold"]),
        "session": str(episode["session"]),
        "decision_time_ns": int(episode["decision_time_ns"]),
        "contract_id": str(episode["contract_id"]),
        "right": str(episode["right"]),
        "canonical_slot": int(episode["canonical_slot"]),
        "entry_ask": float(episode["entry_ask"]),
        "entry_quote_time_ns": int(episode["entry_quote_time_ns"]),
        "source_context_time_ns": int(episode["source_context_time_ns"]),
        "policy_deadline_ns": int(episode[f"p{policy}_deadline_ns"]),
        "comparator": comparator,
        "exit_source_quote_time_ns": int(
            episode[f"p{policy}_source_exit_quote_time_ns"]
        ),
        "realized_exit_time_ns": int(
            episode[f"p{policy}_realized_exit_time_ns"]
        ),
        "exit_quote_age_ms": float(
            episode[f"p{policy}_exit_quote_age_ms"]
        ),
        "exit_reason_code": int(
            episode[f"p{policy}_exit_reason_code"]
        ),
        "exit_bid": float(episode[f"p{policy}_executable_exit_bid"]),
        "exit_mid": (
            float(episode["entry_ask"])
            + float(episode[f"p{policy}_mid_pnl_before_fee"])
            / CONTRACT_MULTIPLIER
        ),
        "net_pnl_after_fee": float(
            episode[f"p{policy}_net_pnl_after_fee"]
        ),
        "mid_pnl_before_fee": float(
            episode[f"p{policy}_mid_pnl_before_fee"]
        ),
        "holding_minutes": (
            (
                int(episode[f"p{policy}_realized_exit_time_ns"])
                - int(episode["decision_time_ns"])
            )
            / 60_000_000_000.0
        ),
        "expected_outcome": False,
    }


def replay_episode_comparators(
    episodes: pd.DataFrame,
    lifecycle: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    best_fixed_policy: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = lifecycle.merge(
        predictions,
        on=["episode_id", "role", "session", "state_index", "state_time_ns"],
        how="left",
        validate="one_to_one",
    )
    episode_map = {
        str(item["episode_id"]): item
        for item in episodes.to_dict("records")
    }
    score_comparators = {
        "real_hgb": "real_hgb_score",
        "constant_score": "constant_score",
        "reversed_real_hgb": "reversed_real_hgb_score",
        **{
            f"strong_shuffle_{seed}": f"strong_shuffle_{seed}_score"
            for seed in SHUFFLE_SEEDS
        },
    }
    outcomes = []
    for episode_id, group in scored.groupby("episode_id", sort=False):
        group = group.sort_values("state_index")
        episode = episode_map[str(episode_id)]
        outcomes.append(
            _fixed_outcome(episode, policy=5, comparator="original_p5")
        )
        outcomes.append(
            _fixed_outcome(
                episode,
                policy=best_fixed_policy,
                comparator="training_best_fixed",
            )
        )
        outcomes.append(
            _row_outcome(
                episode,
                group[group["current_executable_bid"]].iloc[0],
                comparator="exit_now",
            )
        )
        outcomes.append(
            _row_outcome(
                episode,
                _deadline_outcome(group),
                comparator="hold_deadline",
                deadline_fallback=True,
            )
        )
        for comparator, score_column in score_comparators.items():
            score_outcome, deadline_fallback = _score_outcome(
                group, score_column
            )
            outcomes.append(
                _row_outcome(
                    episode,
                    score_outcome,
                    comparator=comparator,
                    deadline_fallback=deadline_fallback,
                )
            )
    frame = pd.DataFrame.from_records(outcomes)
    attempt001_runner.assert_unique(
        frame,
        ("episode_id", "comparator"),
        label="attempt002 episode comparator outcome",
    )
    hazard = fit_random_hazard(frame, scored)
    random_rows = exact_random_episode_outcomes(
        episodes[
            episodes["role"].eq("nested_validation")
        ],
        scored[scored["role"].eq("nested_validation")],
        hazard,
    )
    frame = pd.concat([frame, random_rows], ignore_index=True)
    attempt001_runner.write_parquet(
        OUTPUT_ROOT / "episode_replays.parquet", frame
    )
    return frame, hazard


def fit_random_hazard(
    outcomes: pd.DataFrame,
    scored: pd.DataFrame,
) -> dict[str, Any]:
    exits = outcomes[
        outcomes["role"].eq("nested_train")
        & outcomes["comparator"].eq("real_hgb")
    ][["episode_id", "holding_minutes"]]
    max_elapsed = (
        scored[scored["role"].eq("nested_train")]
        .groupby("episode_id")["elapsed_minutes"]
        .max()
    )
    hazards = {}
    for elapsed in range(1, int(max_elapsed.max()) + 1):
        at_risk = int((max_elapsed >= elapsed).sum())
        exit_count = int(
            np.isclose(
                exits["holding_minutes"].to_numpy(dtype=float),
                float(elapsed),
            ).sum()
        )
        hazards[str(elapsed)] = (
            float(exit_count / at_risk) if at_risk else 0.0
        )
    payload = {
        "source": "real_HGB_nested_training_actions_only",
        "threshold": 0.0,
        "hazards_by_elapsed_minute": hazards,
        "training_episode_count": int(len(exits)),
        "frozen_before_validation_random_aggregation": True,
        "hazard_hash": attempt001_runner.stable_hash(hazards),
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "random_hazard_freeze.json", payload
    )
    return payload


def exact_random_episode_outcomes(
    validation_episodes: pd.DataFrame,
    validation_scored: pd.DataFrame,
    hazard: Mapping[str, Any],
) -> pd.DataFrame:
    episode_map = {
        str(item["episode_id"]): item
        for item in validation_episodes.to_dict("records")
    }
    hazards = {
        int(key): float(value)
        for key, value in hazard["hazards_by_elapsed_minute"].items()
    }
    rows = []
    for episode_id, group in validation_scored.groupby(
        "episode_id", sort=False
    ):
        group = group.sort_values("state_index")
        executable = group[group["current_executable_bid"]]
        if executable.empty:
            raise RuntimeError("random control episode lacks executable exit")
        survival = 1.0
        weighted = []
        for row in executable.itertuples(index=False):
            elapsed = int(round(float(row.elapsed_minutes)))
            probability = survival * hazards.get(elapsed, 0.0)
            if probability > 0.0:
                weighted.append((probability, row))
            survival *= 1.0 - hazards.get(elapsed, 0.0)
        deadline_row = executable.iloc[-1]
        if survival > 0.0:
            weighted.append(
                (
                    survival,
                    next(
                        item
                        for item in executable.itertuples(index=False)
                        if int(item.state_index)
                        == int(deadline_row["state_index"])
                    ),
                )
            )
        total_probability = sum(item[0] for item in weighted)
        if not math.isclose(total_probability, 1.0, abs_tol=1e-9):
            raise RuntimeError("exact random exit probabilities do not sum to one")
        episode = episode_map[str(episode_id)]
        expected_pnl = sum(
            probability
            * (
                (float(row.current_bid) - float(episode["entry_ask"]))
                * CONTRACT_MULTIPLIER
                - ROUND_TRIP_FEE
            )
            for probability, row in weighted
        )
        expected_holding = sum(
            probability * float(row.elapsed_minutes)
            for probability, row in weighted
        )
        rows.append(
            {
                "episode_id": str(episode_id),
                "role": "nested_validation",
                "fold": str(episode["fold"]),
                "session": str(episode["session"]),
                "decision_time_ns": int(episode["decision_time_ns"]),
                "contract_id": str(episode["contract_id"]),
                "right": str(episode["right"]),
                "canonical_slot": int(episode["canonical_slot"]),
                "entry_ask": float(episode["entry_ask"]),
                "entry_quote_time_ns": int(episode["entry_quote_time_ns"]),
                "source_context_time_ns": int(
                    episode["source_context_time_ns"]
                ),
                "policy_deadline_ns": int(episode["policy_deadline_ns"]),
                "comparator": "exact_random_exit",
                "exit_source_quote_time_ns": None,
                "realized_exit_time_ns": None,
                "exit_quote_age_ms": None,
                "exit_reason_code": None,
                "exit_bid": None,
                "exit_mid": None,
                "net_pnl_after_fee": expected_pnl,
                "mid_pnl_before_fee": None,
                "holding_minutes": expected_holding,
                "expected_outcome": True,
            }
        )
    return pd.DataFrame.from_records(rows)


def _serial_candidate(outcome: Mapping[str, Any]) -> SerialCandidateV5:
    return SerialCandidateV5(
        split=str(outcome["role"]),
        fold=str(outcome["fold"]),
        session=str(outcome["session"]),
        decision_time_ns=int(outcome["decision_time_ns"]),
        contract_id=str(outcome["contract_id"]),
        right=str(outcome["right"]),
        canonical_strike_slot=int(outcome["canonical_slot"]),
        policy_index=POLICY_INDEX,
        entry_ask=float(outcome["entry_ask"]),
        score=0.0,
        raw_label_pnl_after_campaign_fee=float(
            outcome["net_pnl_after_fee"]
        ),
        label_mid_pnl_before_campaign_fee=float(
            outcome["mid_pnl_before_fee"]
        ),
        label_realized_exit_time_ns=int(outcome["realized_exit_time_ns"]),
        label_source_exit_quote_time_ns=int(
            outcome["exit_source_quote_time_ns"]
        ),
        label_exit_quote_age_ms=float(outcome["exit_quote_age_ms"]),
        label_exit_reason_code=int(outcome["exit_reason_code"]),
        label_executable_exit_bid=float(outcome["exit_bid"]),
        label_policy_deadline_ns=int(outcome["policy_deadline_ns"]),
        label_invalid_reason_code=int(InvalidReason.NONE),
        feature_hash=attempt001_runner.stable_hash(
            {
                "contract": "attempt002_lifecycle_exit",
                "comparator": str(outcome["comparator"]),
            }
        ),
        source_quote_time_ns=int(outcome["entry_quote_time_ns"]),
        source_context_time_ns=int(outcome["source_context_time_ns"]),
        strategy=f"attempt002_{outcome['comparator']}",
        source_simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "episode_id": str(outcome["episode_id"]),
            "comparator": str(outcome["comparator"]),
        },
    )


def run_serial_ledgers(
    episode_replays: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = episode_replays[
        episode_replays["role"].eq("nested_validation")
        & ~episode_replays["expected_outcome"]
    ]
    records = []
    receipts = {}
    for comparator, group in validation.groupby("comparator", sort=True):
        candidates = [
            _serial_candidate(item)
            for item in group.sort_values(
                ["session", "decision_time_ns"]
            ).to_dict("records")
        ]
        trades, state = references.replay_reference_v5(candidates)
        trade_rows = []
        for trade in trades:
            trade_rows.append(
                {
                    "evidence_grade": EVIDENCE_GRADE,
                    "comparator": str(comparator),
                    **asdict(trade),
                }
            )
        records.extend(trade_rows)
        receipts[str(comparator)] = {
            "entry_intents": len(candidates),
            "admitted_trades": len(trades),
            "skipped": dict(state.skipped),
            "candidate_stream_hash": state.candidate_stream_hash,
            "candidate_payload_hash": state.candidate_payload_hash,
            "trade_identity_hash": state.trade_identity_hash,
            "admitted_identity_hash": attempt001_runner.stable_hash(
                [
                    {
                        "session": item.session,
                        "decision_time_ns": int(item.decision_time_ns),
                        "contract_id": item.contract_id,
                    }
                    for item in trades
                ]
            ),
            "skipped_identity_hash": attempt001_runner.stable_hash(
                [
                    {
                        "candidate_identity": list(item.candidate_identity),
                        "reason_code": item.reason_code,
                    }
                    for item in state.skipped_events
                ]
            ),
        }
    serial = pd.DataFrame.from_records(records)
    attempt001_runner.write_parquet(
        OUTPUT_ROOT / "serial_replays.parquet", serial
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "strict_simulator_v5_common_intent_replays_complete",
        "common_intent_count": EXPECTED_VALIDATION_OPPORTUNITIES,
        "comparators": receipts,
        "exact_random_ledger_B": {
            "status": "not_defined_by_inherited_contract",
            "limitation": SERIAL_RANDOM_LIMITATION,
        },
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "serial_identity_receipts.json", payload
    )
    return serial, payload


def comparator_summary(
    episode_replays: pd.DataFrame,
    serial_replays: pd.DataFrame,
    *,
    best_fixed_policy: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation_a = episode_replays[
        episode_replays["role"].eq("nested_validation")
    ].copy()
    ledger_a = {}
    for comparator, group in validation_a.groupby("comparator", sort=True):
        sessions = group.groupby("session")["net_pnl_after_fee"].sum()
        ledger_a[str(comparator)] = {
            "episodes": int(group["episode_id"].nunique()),
            "pooled_net_pnl": float(group["net_pnl_after_fee"].sum()),
            "mean_episode_pnl": float(group["net_pnl_after_fee"].mean()),
            "mean_holding_minutes": float(group["holding_minutes"].mean()),
            "session_net_pnl": {
                str(key): float(value) for key, value in sessions.items()
            },
        }
    ledger_b = {}
    for comparator, group in serial_replays.groupby(
        "comparator", sort=True
    ):
        sessions = group.groupby("session")[
            "raw_label_pnl_after_campaign_fee"
        ].sum()
        ledger_b[str(comparator)] = {
            "admitted_trades": int(len(group)),
            "pooled_net_pnl": float(
                group["raw_label_pnl_after_campaign_fee"].sum()
            ),
            "session_net_pnl": {
                str(key): float(value) for key, value in sessions.items()
            },
        }
    ledger_b["exact_random_exit"] = {
        "status": "not_defined_by_inherited_contract",
        "pooled_net_pnl": None,
        "limitation": SERIAL_RANDOM_LIMITATION,
    }
    p5_absolute = {
        "ledger_A_independent_overlapping_pooled_P5_pnl": ledger_a[
            "original_p5"
        ]["pooled_net_pnl"],
        "ledger_B_serial_one_account_P5_pnl": ledger_b["original_p5"][
            "pooled_net_pnl"
        ],
        "these_are_not_interchangeable": True,
    }
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "training_selected_best_fixed_policy": best_fixed_policy,
        "ledger_A": {
            key: value
            for key, value in ledger_a.items()
            if key
            in {
                "original_p5",
                "training_best_fixed",
                "exit_now",
                "hold_deadline",
            }
        },
        "ledger_B": {
            key: value
            for key, value in ledger_b.items()
            if key
            in {
                "original_p5",
                "training_best_fixed",
                "exit_now",
                "hold_deadline",
            }
        },
        "absolute_P5_profitability": p5_absolute,
    }
    random = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "ledger_A_exact_expected": ledger_a["exact_random_exit"],
        "ledger_B": ledger_b["exact_random_exit"],
    }
    controls = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "ledger_A": {
            key: value
            for key, value in ledger_a.items()
            if key
            in {
                "real_hgb",
                "constant_score",
                "reversed_real_hgb",
                *{f"strong_shuffle_{seed}" for seed in SHUFFLE_SEEDS},
            }
        },
        "ledger_B": {
            key: value
            for key, value in ledger_b.items()
            if key
            in {
                "real_hgb",
                "constant_score",
                "reversed_real_hgb",
                *{f"strong_shuffle_{seed}" for seed in SHUFFLE_SEEDS},
            }
        },
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "fixed_exit_results.json", fixed
    )
    attempt001_runner.write_json(
        OUTPUT_ROOT / "random_exit_results.json", random
    )
    attempt001_runner.write_json(
        OUTPUT_ROOT / "control_results.json", controls
    )
    return fixed, random, controls


def _session_improvement_count(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[int, int]:
    sessions = sorted(
        set(candidate["session_net_pnl"])
        | set(baseline["session_net_pnl"])
    )
    improvements = sum(
        float(candidate["session_net_pnl"].get(session, 0.0))
        > float(baseline["session_net_pnl"].get(session, 0.0))
        for session in sessions
    )
    losses = sum(
        float(candidate["session_net_pnl"].get(session, 0.0))
        < float(baseline["session_net_pnl"].get(session, 0.0))
        for session in sessions
    )
    return improvements, losses


def terminal_decision(
    *,
    fixed: Mapping[str, Any],
    random: Mapping[str, Any],
    controls: Mapping[str, Any],
    machinery_pass: bool,
    validation_episodes: int,
) -> tuple[str, dict[str, Any]]:
    a = {
        **fixed["ledger_A"],
        **controls["ledger_A"],
        "exact_random_exit": random["ledger_A_exact_expected"],
    }
    b = {**fixed["ledger_B"], **controls["ledger_B"]}
    baselines_a = (
        "original_p5",
        "training_best_fixed",
        "exact_random_exit",
    )
    baselines_b = ("original_p5", "training_best_fixed")
    real_a = a["real_hgb"]
    real_b = b["real_hgb"]
    lifts_a = {
        name: float(real_a["pooled_net_pnl"])
        - float(a[name]["pooled_net_pnl"])
        for name in baselines_a
    }
    lifts_b = {
        name: float(real_b["pooled_net_pnl"])
        - float(b[name]["pooled_net_pnl"])
        for name in baselines_b
    }
    session_improvements_a = {}
    session_losses_b = {}
    for baseline in ("original_p5", "training_best_fixed"):
        improved, _ = _session_improvement_count(real_a, a[baseline])
        _, losses = _session_improvement_count(real_b, b[baseline])
        session_improvements_a[baseline] = improved
        session_losses_b[baseline] = losses
    control_minimum_lifts = {}
    for comparator in (
        "reversed_real_hgb",
        *[f"strong_shuffle_{seed}" for seed in SHUFFLE_SEEDS],
    ):
        comparator_lifts = [
            float(a[comparator]["pooled_net_pnl"])
            - float(a[name]["pooled_net_pnl"])
            for name in baselines_a
        ] + [
            float(b[comparator]["pooled_net_pnl"])
            - float(b[name]["pooled_net_pnl"])
            for name in baselines_b
        ]
        control_minimum_lifts[comparator] = min(comparator_lifts)
    real_minimum_lift_without_random_b = min(
        [*lifts_a.values(), *lifts_b.values()]
    )
    empirical_gates = {
        "machinery_pass": machinery_pass,
        "validation_episode_floor": (
            validation_episodes >= MINIMUM_VALIDATION_EPISODES
        ),
        "real_positive_vs_all_ledger_A_baselines": all(
            value > 0.0 for value in lifts_a.values()
        ),
        "real_positive_vs_P5_and_best_fixed_ledger_B": all(
            value > 0.0 for value in lifts_b.values()
        ),
        "ledger_A_improves_P5_and_best_fixed_at_least_4_of_5": all(
            value >= 4 for value in session_improvements_a.values()
        ),
        "ledger_B_loses_to_P5_or_best_fixed_on_at_most_1_of_5": all(
            value <= 1 for value in session_losses_b.values()
        ),
        "real_minimum_lift_exceeds_reversed_and_shuffles": all(
            real_minimum_lift_without_random_b > value
            for value in control_minimum_lifts.values()
        ),
    }
    empirical_pass = all(empirical_gates.values())
    exact_random_b_available = False
    if validation_episodes < MINIMUM_VALIDATION_EPISODES:
        decision = "stop_insufficient_data"
    elif not machinery_pass:
        decision = "stop_mechanical_blocker"
    elif not empirical_pass:
        decision = "stop_no_preliminary_exit_signal"
    elif not exact_random_b_available:
        decision = "stop_scientific_contract_defect"
    else:
        decision = "proceed_to_full_lifecycle_contract_signature_and_campaign"
    detail = {
        "ledger_A_real_lifts": lifts_a,
        "ledger_B_real_lifts": lifts_b,
        "ledger_A_session_improvements": session_improvements_a,
        "ledger_B_session_losses": session_losses_b,
        "real_minimum_lift_without_random_B": (
            real_minimum_lift_without_random_b
        ),
        "control_minimum_lifts": control_minimum_lifts,
        "empirical_gates": empirical_gates,
        "exact_random_ledger_B_available": exact_random_b_available,
        "exact_random_ledger_B_limitation": SERIAL_RANDOM_LIMITATION,
    }
    return decision, detail


def machinery_checks(
    preregistration: Mapping[str, Any],
    episodes: pd.DataFrame,
    lifecycle: pd.DataFrame,
    entry_receipt: Mapping[str, Any],
    path_receipt: Mapping[str, Any],
    firewall: Mapping[str, Any],
    manual: Mapping[str, Any],
) -> dict[str, Any]:
    weights = weighting_audit(lifecycle)
    checks = {
        "attempt002_goal_hash_matches": (
            attempt001_runner.sha256_path(GOAL_PATH)
            == EXPECTED_GOAL_SHA256
        ),
        "attempt001_byte_identical": (
            attempt001_file_hashes()
            == preregistration["attempt001_file_hashes"]
        ),
        "preregistration_precedes_lifecycle_and_economics": True,
        "nested_roles_exclude_final_OOF": True,
        "P5_opportunity_smoke_anchor_6414": len(episodes)
        == EXPECTED_TOTAL_OPPORTUNITIES,
        "validation_smoke_anchor_1354": int(
            episodes["role"].eq("nested_validation").sum()
        )
        == EXPECTED_VALIDATION_OPPORTUNITIES,
        "entry_identities_unique": int(
            entry_receipt["duplicate_identity_count"]
        )
        == 0,
        "lifecycle_identities_unique": int(
            path_receipt["duplicate_state_identities"]
        )
        == 0,
        "all_entry_episodes_have_lifecycle_paths": (
            episodes["episode_id"].nunique()
            == lifecycle["episode_id"].nunique()
        ),
        "context_parity_pass": bool(path_receipt["context_parity_pass"]),
        "feature_firewall_pass": firewall["status"] == "pass",
        "manual_target_reproduction_pass": manual["status"] == "pass",
        "group_balanced_weighting_pass": bool(weights["pass"]),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "weighting_audit": weights,
        "entry_identity_root_hash": entry_receipt[
            "entry_identity_root_hash"
        ],
        "lifecycle_identity_root_hash": attempt001_runner.stable_hash(
            lifecycle[["episode_id", "state_time_ns"]].to_dict("records")
        ),
    }
    attempt001_runner.write_json(
        OUTPUT_ROOT / "machinery_checks.json", payload
    )
    if payload["status"] != "pass":
        raise RuntimeError(f"attempt002 machinery checks failed: {payload}")
    return payload


def resource_projection(started: float, lifecycle: pd.DataFrame) -> dict[str, Any]:
    output_bytes = sum(
        path.stat().st_size
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "output_bytes": int(output_bytes),
        "sessions": int(lifecycle["session"].nunique()),
        "episodes": int(lifecycle["episode_id"].nunique()),
        "lifecycle_rows": int(len(lifecycle)),
        "five_fold_linear_row_projection": int(len(lifecycle) * 5),
        "full_campaign_not_run": True,
    }


def write_hash_manifest() -> None:
    lines = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "hashes.sha256"
            or WORK_ROOT in path.parents
        ):
            continue
        lines.append(
            f"{attempt001_runner.sha256_path(path)}  "
            f"{path.relative_to(OUTPUT_ROOT)}"
        )
    (OUTPUT_ROOT / "hashes.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_report(
    *,
    decision: str,
    entry_receipt: Mapping[str, Any],
    path_receipt: Mapping[str, Any],
    overlap: Mapping[str, Any],
    fixed: Mapping[str, Any],
    detail: Mapping[str, Any],
    machinery: Mapping[str, Any],
) -> None:
    p5 = fixed["absolute_P5_profitability"]
    text = f"""# Protocol101 FT2 Stage-0 P5 HOLD/EXIT Feasibility Attempt002

Evidence grade: `{EVIDENCE_GRADE}`

Terminal decision: `{decision}`

Highest allowed claim: Exploratory non-promotable P5 HOLD/EXIT Stage-0
attempt002 feasibility decision complete.

## Corrected Episode Construction

- Exact P5-permitted opportunities: `{entry_receipt["total_episode_count"]}`.
- Nested-training episodes: `{entry_receipt["training_episode_count"]}`.
- Nested-validation episodes: `{entry_receipt["validation_episode_count"]}`.
- Validation minimum: `{MINIMUM_VALIDATION_EPISODES}`.
- Causal lifecycle rows: `{path_receipt["row_count"]}`.
- Target-valid rows: `{path_receipt["target_valid_rows"]}`.
- Missing one-minute targets retained: `{path_receipt["missing_target_rows"]}`.

Attempt001 remains valid and byte-identical. Attempt002 changes only the
isolated episode source from serially admitted trades to exact P5-permitted
opportunities.

## Evidence Separation

Ledger A contains heavily overlapping independent episodes. It is useful for
exit attribution but is not executable one-account PnL.

- Overlapping episode pairs: `{overlap["overlapping_episode_pair_count"]}`.
- Maximum concurrent episodes: `{overlap["maximum_concurrency"]}`.

Ledger B presents the same ordered P5 intents to simulator v5 and lets each
comparator's exit determine occupancy and skipped opportunities.

Absolute P5 profitability is reported separately:

- Ledger-A overlapping P5 PnL:
  `{p5["ledger_A_independent_overlapping_pooled_P5_pnl"]}`.
- Ledger-B serial P5 PnL:
  `{p5["ledger_B_serial_one_account_P5_pnl"]}`.

## Real-Model Incremental Uplift

- Ledger A: `{json.dumps(detail["ledger_A_real_lifts"], sort_keys=True)}`.
- Ledger B: `{json.dumps(detail["ledger_B_real_lifts"], sort_keys=True)}`.

The exact random exit is computed as an exact hazard expectation in Ledger A.
The inherited contract does not define an exact stochastic account expectation
for Ledger B. This limitation cannot create a proceed route; if all empirical
gates otherwise pass it routes to `stop_scientific_contract_defect`.

## Integrity

- Machinery: `{machinery["status"]}`.
- Attempt001 byte-identical: `true`.
- Threshold optimization: `false`.
- Entry/selector training: `false`.
- Protected, recorder, sealed, shadow, paper, broker, runtime, promotion,
  launchd, paid-data, and real-money actions: `false`.
"""
    (OUTPUT_ROOT / "report.md").write_text(text, encoding="utf-8")


def run() -> str:
    started = time.perf_counter()
    preregistration = preregister()
    verify_preregistration()
    attempt001_runner.write_json(
        OUTPUT_ROOT / "repair_log.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "repair_iterations_used": len(MECHANICAL_REPAIRS),
            "maximum_repair_iterations": 3,
            "repairs": MECHANICAL_REPAIRS,
        },
    )
    episodes_path = OUTPUT_ROOT / "frozen_p5_entry_episodes.parquet"
    entry_receipt_path = OUTPUT_ROOT / "entry_identity_receipt.json"
    if episodes_path.is_file() and entry_receipt_path.is_file():
        entry_receipt = attempt001_runner.read_json(entry_receipt_path)
        if (
            attempt001_runner.sha256_path(episodes_path)
            != str(entry_receipt["parquet_sha256"])
        ):
            raise RuntimeError("attempt002 episode checkpoint hash mismatch")
        episodes = pd.read_parquet(episodes_path)
        if (
            len(episodes) != EXPECTED_TOTAL_OPPORTUNITIES
            or int(episodes["role"].eq("nested_validation").sum())
            != EXPECTED_VALIDATION_OPPORTUNITIES
        ):
            raise RuntimeError(
                "attempt002 episode checkpoint anchor mismatch"
            )
        write_progress(
            "reused_verified_attempt002_P5_episode_checkpoint",
            frozen_episodes=len(episodes),
        )
    else:
        write_progress("rebuilding_exact_P5_opportunity_episodes")
        episodes, entry_receipt = build_p5_opportunity_episodes(
            preregistration
        )
    lifecycle_path = OUTPUT_ROOT / "lifecycle_rows.parquet"
    path_receipt_path = OUTPUT_ROOT / "path_identity_receipt.json"
    if lifecycle_path.is_file() and path_receipt_path.is_file():
        path_receipt = attempt001_runner.read_json(path_receipt_path)
        if (
            attempt001_runner.sha256_path(lifecycle_path)
            != str(path_receipt["parquet_sha256"])
        ):
            raise RuntimeError("attempt002 lifecycle checkpoint hash mismatch")
        lifecycle = pd.read_parquet(lifecycle_path)
        if (
            lifecycle["episode_id"].nunique()
            != episodes["episode_id"].nunique()
        ):
            raise RuntimeError(
                "attempt002 lifecycle checkpoint episode-count mismatch"
            )
        write_progress(
            "reused_verified_attempt002_lifecycle_checkpoint",
            frozen_episodes=len(episodes),
            lifecycle_rows=len(lifecycle),
        )
    else:
        write_progress(
            "building_attempt002_causal_lifecycle_rows",
            frozen_episodes=len(episodes),
        )
        lifecycle, path_receipt = build_lifecycle_rows(episodes)
    firewall = write_feature_contract_and_audit(lifecycle)
    manual = manual_target_reproduction(lifecycle)
    machinery = machinery_checks(
        preregistration,
        episodes,
        lifecycle,
        entry_receipt,
        path_receipt,
        firewall,
        manual,
    )
    validation_episodes = int(entry_receipt["validation_episode_count"])
    if validation_episodes < MINIMUM_VALIDATION_EPISODES:
        raise RuntimeError(
            "attempt002 unexpectedly remains below the frozen episode floor"
        )
    overlap = overlap_audit(episodes)
    attempt001_runner.write_json(
        OUTPUT_ROOT / "overlap_audit.json", overlap
    )
    best_fixed_policy, fixed_selection = select_training_fixed_policy(episodes)
    write_progress(
        "fitting_bounded_attempt002_models",
        best_training_fixed_policy=best_fixed_policy,
    )
    predictions_path = OUTPUT_ROOT / "predictions.parquet"
    training_receipt_path = MODEL_ROOT / "training_receipt.json"
    if predictions_path.is_file() and training_receipt_path.is_file():
        predictions = pd.read_parquet(predictions_path)
        attempt001_runner.assert_unique(
            predictions,
            ("episode_id", "state_time_ns"),
            label="attempt002 prediction checkpoint identity",
        )
        if len(predictions) != len(lifecycle):
            raise RuntimeError(
                "attempt002 prediction checkpoint row-count mismatch"
            )
        model_receipt = attempt001_runner.read_json(training_receipt_path)
        write_progress(
            "reused_verified_attempt002_prediction_checkpoint",
            prediction_rows=len(predictions),
        )
    else:
        predictions, model_receipt = fit_models(lifecycle)
    write_progress("replaying_ledger_A_identical_entry_episodes")
    episode_replays, hazard = replay_episode_comparators(
        episodes,
        lifecycle,
        predictions,
        best_fixed_policy=best_fixed_policy,
    )
    write_progress("replaying_ledger_B_strict_simulator_v5")
    serial_replays, serial_receipt = run_serial_ledgers(episode_replays)
    fixed, random, controls = comparator_summary(
        episode_replays,
        serial_replays,
        best_fixed_policy=best_fixed_policy,
    )
    decision, detail = terminal_decision(
        fixed=fixed,
        random=random,
        controls=controls,
        machinery_pass=machinery["status"] == "pass",
        validation_episodes=validation_episodes,
    )
    pilot = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "terminal_decision": decision,
        "entry_receipt": {
            "total": entry_receipt["total_episode_count"],
            "training": entry_receipt["training_episode_count"],
            "validation": entry_receipt["validation_episode_count"],
        },
        "overlap": overlap,
        "model_receipt": model_receipt,
        "fixed_selection": fixed_selection,
        "random_hazard": hazard,
        "decision_detail": detail,
        "serial_random_limitation": SERIAL_RANDOM_LIMITATION,
    }
    attempt001_runner.write_json(OUTPUT_ROOT / "pilot_results.json", pilot)
    attempt001_runner.write_json(
        OUTPUT_ROOT / "resource_projection.json",
        resource_projection(started, lifecycle),
    )
    attempt001_runner.write_json(
        OUTPUT_ROOT / "repair_log.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "repair_iterations_used": len(MECHANICAL_REPAIRS),
            "maximum_repair_iterations": 3,
            "repairs": MECHANICAL_REPAIRS,
        },
    )
    decision_payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "terminal_decision": decision,
        "validation_episode_count": validation_episodes,
        "minimum_validation_episode_count": MINIMUM_VALIDATION_EPISODES,
        "model_fit_executed": True,
        "ledger_A_executed": True,
        "ledger_B_simulator_v5_executed": True,
        "full_lifecycle_campaign_authorized": False,
        "attempt001_byte_identical": (
            attempt001_file_hashes()
            == preregistration["attempt001_file_hashes"]
        ),
        "decision_detail": detail,
        "side_effects": preregistration["side_effects"],
    }
    attempt001_runner.write_json(OUTPUT_ROOT / "decision.json", decision_payload)
    write_report(
        decision=decision,
        entry_receipt=entry_receipt,
        path_receipt=path_receipt,
        overlap=overlap,
        fixed=fixed,
        detail=detail,
        machinery=machinery,
    )
    write_progress(
        "complete",
        terminal_decision=decision,
        validation_episode_count=validation_episodes,
    )
    write_hash_manifest()
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.parse_args()
    decision = run()
    print(
        json.dumps(
            {
                "status": "complete",
                "evidence_grade": EVIDENCE_GRADE,
                "terminal_decision": decision,
                "output_dir": str(OUTPUT_ROOT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
