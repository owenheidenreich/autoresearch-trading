"""Exploratory P5 HOLD/EXIT Stage-0 feasibility pilot.

The owner-approved Goal deliberately stops before model fitting when fewer
than 100 validation episodes survive exact simulator-v5 P5 admission. This
runner still constructs and validates the entry and causal lifecycle
machinery so the resulting insufficient-data route is reproducible rather
than inferred from a summary count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import resource
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import _market_features
from v4.model import protocol101_stage1_reference_multiplicity as references
from v4.model.protocol101_canonical_stage1_contract import (
    CONTEXT_FEATURES,
    _bs_delta_gamma,
    _implied_vol,
    _quantized_mid,
    _tte_years,
)
from v4.model.protocol101_scoped_stage1_hgb import load_repaired_decisions
from v4.scripts.run_protocol101_policy_neutral_contract_selector import (
    ROOT,
    _reference_opportunities,
    _scope_maps,
)
from v4.scripts.run_protocol101_scoped_stage1_hgb_runner import guard_margins


EVIDENCE_GRADE = "exploratory_non_promotable_stage0"
SCHEMA_VERSION = "Protocol101FT2Stage0P5HoldExitFeasibilityV1"
CAMPAIGN_ID = "protocol101-ft2-stage0-p5-hold-exit-feasibility-attempt001"
GOAL_PATH = (
    ROOT
    / "v4/docs/protocol101/training/history/"
    "closed_stage1_graph_and_goals_2026_07_28/goals/"
    "PROTOCOL101_FT2_STAGE0_P5_HOLD_EXIT_FEASIBILITY_GOAL_2026_07_28.md"
)
EXPECTED_GOAL_SHA256 = (
    "993199eacaeec183c5475c5bf97196507b271943bbae37fdee1f66df4ad069fb"
)
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001"
)
ATTEMPT002_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002"
)
MACHINERY_ACCEPTANCE = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/"
    "acceptance_decision.json"
)
STAGE2_DRAFT = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md"
)
TRAINING_README = ROOT / "v4/docs/protocol101/training/README.md"
TRADER_CHARTER = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_TRADER_CHARTER.md"
)
REPAIR_AMENDMENT = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md"
)
FOUNDATION_DOC = (
    ROOT
    / "v4/docs/protocol101/training/research/"
    "PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md"
)
REFERENCE_SOURCE = (
    ROOT / "v4/model/protocol101_stage1_reference_multiplicity.py"
)
SIMULATOR_SOURCE = ROOT / "v4/model/protocol101_serial_simulator_v5.py"

TRAIN_SESSION_COUNT = 15
VALIDATION_SESSION_COUNT = 5
MINIMUM_VALIDATION_EPISODES = 100
MODEL_SEED = 42
SHUFFLE_SEEDS = (8700, 8701)
POLICY_INDEX = 5
ROUND_TRIP_FEE = 3.0
CONTRACT_MULTIPLIER = 100.0
MAX_CAUSAL_QUOTE_AGE_SECONDS = 90.0
HGB_CONFIG = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state": MODEL_SEED,
}

LIFECYCLE_FEATURES = (
    "elapsed_minutes",
    "minutes_to_deadline",
    "entry_ask",
    "current_bid",
    "current_ask",
    "current_mid",
    "current_spread",
    "current_unrealized_pnl_dollars",
    "current_return_on_entry_premium",
    "mfe_to_date_dollars",
    "mae_to_date_dollars",
    "giveback_from_mfe_dollars",
    "time_since_mfe_minutes",
    "bid_velocity_1m",
    "bid_velocity_3m",
    "bid_velocity_5m",
    "pnl_velocity_1m",
    "pnl_velocity_3m",
    "pnl_velocity_5m",
    *CONTEXT_FEATURES,
    "right_is_call",
    "canonical_slot",
    "current_offset_points",
    "current_moneyness_bps",
    "entry_premium_dollars",
    "current_mid_over_entry_ask",
    "internal_delta",
    "internal_gamma",
)

FORBIDDEN_FEATURE_TOKENS = (
    "future",
    "oracle",
    "target",
    "q_exit",
    "q_hold",
    "a_hold",
    "advantage",
    "realized_exit",
    "exit_reason",
    "vendor_greek",
    "internal_iv",
    "vix_change",
    "volume",
    "open_interest",
    "bid_size",
    "ask_size",
    "update_count",
    "subminute",
    "entry_model_score",
    "selector_score",
)

TERMINAL_DECISIONS = (
    "proceed_to_full_lifecycle_contract_signature_and_campaign",
    "stop_no_preliminary_exit_signal",
    "stop_mechanical_blocker",
    "stop_scientific_contract_defect",
    "stop_insufficient_data",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        clean(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def write_progress(status: str, **fields: Any) -> None:
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            **fields,
        },
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_sessions() -> tuple[list[str], list[str], dict[str, Any]]:
    scope, _, _ = _scope_maps()
    fold = scope.folds[0]
    final_oof = {
        str(session)
        for item in scope.folds
        for session in item["validation_sessions"]
    }
    eligible = [
        str(session)
        for session in fold["train_sessions"]
        if str(session) not in final_oof
    ]
    selected = eligible[-(TRAIN_SESSION_COUNT + VALIDATION_SESSION_COUNT) :]
    if len(selected) != TRAIN_SESSION_COUNT + VALIDATION_SESSION_COUNT:
        raise RuntimeError("insufficient Fold-1 training-side sessions")
    training = selected[:TRAIN_SESSION_COUNT]
    validation = selected[TRAIN_SESSION_COUNT:]
    if set(selected) & final_oof:
        raise RuntimeError("nested sessions overlap final OOF validation")
    if not max(training) < min(validation):
        raise RuntimeError("nested split is not chronological")
    return training, validation, {
        "governed_fold": str(fold["fold_id"]),
        "eligible_training_prefix_sessions": eligible,
        "selected_sessions": selected,
        "nested_train_sessions": training,
        "nested_validation_sessions": validation,
        "final_oof_session_count": len(final_oof),
        "final_oof_overlap": [],
    }


def authority_hashes() -> dict[str, dict[str, str]]:
    authorities = {
        "goal": GOAL_PATH,
        "training_readme": TRAINING_README,
        "trader_charter": TRADER_CHARTER,
        "stage2_draft_with_stage0_exception": STAGE2_DRAFT,
        "signed_stage1_repair_amendment": REPAIR_AMENDMENT,
        "machinery_independent_acceptance": MACHINERY_ACCEPTANCE,
        "selector_stage0_attempt002_decision": ATTEMPT002_ROOT / "decision.json",
        "selector_stage0_attempt002_preregistration": (
            ATTEMPT002_ROOT / "preregistration.json"
        ),
        "reference_multiplicity_source": REFERENCE_SOURCE,
        "simulator_v5_source": SIMULATOR_SOURCE,
        "historical_foundation_context_only": FOUNDATION_DOC,
    }
    missing = [str(path) for path in authorities.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required authority missing: {missing}")
    if sha256_path(GOAL_PATH) != EXPECTED_GOAL_SHA256:
        raise RuntimeError("owner Goal hash mismatch")
    stage2 = STAGE2_DRAFT.read_text(encoding="utf-8")
    required_stage0_terms = (
        "Owner-Authorized Exploratory Stage-0 Exception",
        "exploratory_non_promotable_stage0",
        "one-step executable-bid hold advantage",
    )
    if not all(term in stage2 for term in required_stage0_terms):
        raise RuntimeError("Stage-2 draft does not contain the bounded exception")

    attempt002 = read_json(ATTEMPT002_ROOT / "preregistration.json")
    inherited: dict[str, dict[str, str]] = {}
    for name, item in attempt002["frozen_input_hashes"].items():
        path = Path(str(item["path"]))
        if not path.is_file():
            raise RuntimeError(f"selector inherited authority missing: {name}")
        digest = sha256_path(path)
        if digest != str(item["sha256"]):
            raise RuntimeError(f"selector inherited authority changed: {name}")
        inherited[f"selector_inherited::{name}"] = {
            "path": str(path),
            "sha256": digest,
        }
    return {
        **{
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in authorities.items()
        },
        **inherited,
    }


def preregistration_payload() -> dict[str, Any]:
    training, validation, split = selected_sessions()
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "campaign_id": CAMPAIGN_ID,
        "highest_allowed_claim": (
            "Exploratory non-promotable P5 HOLD/EXIT Stage-0 "
            "feasibility decision complete."
        ),
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
            "admission": "simulator_v5_actual_trades_only",
            "entry_ask": True,
            "one_contract": True,
            "comparators_may_not_change_frozen_entry_episode_list": True,
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
            "deadline": "accepted_two_clock_contract",
        },
        "feature_names": list(LIFECYCLE_FEATURES),
        "feature_count": len(LIFECYCLE_FEATURES),
        "missing_values": "native_HGB_missing_values_no_imputation",
        "causal_quote_age_limit_seconds": MAX_CAUSAL_QUOTE_AGE_SECONDS,
        "weighting": (
            "equal_total_weight_per_session_then_per_episode_then_per_row"
        ),
        "controls": {
            "original_p5_fixed_exit": True,
            "exit_now": True,
            "hold_to_deadline": True,
            "training_selected_best_fixed_exit": True,
            "exact_random_exit": True,
            "constant_score": True,
            "reversed_real_hgb": True,
            "strong_shuffle_seeds": list(SHUFFLE_SEEDS),
            "strong_shuffle_grouping": (
                "episode_sequence_right_coarse_path_bucket_cross_session"
            ),
        },
        "ledgers": {
            "A": "identical_entry_exit_isolation",
            "B": "directional_serial_consequence_simulator_v5",
        },
        "minimum_validation_episodes": MINIMUM_VALIDATION_EPISODES,
        "terminal_decisions": list(TERMINAL_DECISIONS),
        "repairs": {
            "maximum_iterations": 3,
            "mechanical_only": True,
        },
        "frozen_input_hashes": authority_hashes(),
        "forbidden": [
            "entry_training",
            "contract_selector_training",
            "full_lifecycle_campaign",
            "neural_or_sequence_model",
            "hold_exit_threshold_optimization",
            "five_fold_lifecycle_training",
            "G1_G9",
            "seed45_or_G9",
            "final_OOF",
            "protected_holdout",
            "recorder_or_sealed_evidence",
            "live_shadow_or_paper_outcomes",
            "broker_contact",
            "paid_data_download",
            "promotion_runtime_launchd_or_real_money_change",
        ],
        "side_effects": {
            "entry_training": False,
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
    path = OUTPUT_ROOT / "preregistration.json"
    payload = preregistration_payload()
    if path.is_file() and read_json(path) != clean(payload):
        raise RuntimeError("existing preregistration differs from frozen design")
    if not path.is_file():
        write_json(path, payload)
    freeze = f"{sha256_path(path)}  preregistration.json\n"
    freeze_path = OUTPUT_ROOT / "preregistration.sha256"
    if freeze_path.exists() and freeze_path.read_text(encoding="utf-8") != freeze:
        raise RuntimeError("preregistration freeze changed")
    freeze_path.write_text(freeze, encoding="utf-8")
    write_progress(
        "preregistered_before_results",
        preregistration_sha256=sha256_path(path),
        validation_economics_inspected=False,
    )
    return payload


def verify_preregistration() -> dict[str, Any]:
    path = OUTPUT_ROOT / "preregistration.json"
    freeze = OUTPUT_ROOT / "preregistration.sha256"
    if not path.is_file() or not freeze.is_file():
        raise RuntimeError("preregistration artifacts are missing")
    expected, name = freeze.read_text(encoding="utf-8").strip().split("  ", 1)
    if name != path.name or expected != sha256_path(path):
        raise RuntimeError("preregistration hash mismatch")
    payload = read_json(path)
    for name, item in payload["frozen_input_hashes"].items():
        source = Path(str(item["path"]))
        if not source.is_file() or sha256_path(source) != str(item["sha256"]):
            raise RuntimeError(f"frozen authority changed after preregistration: {name}")
    return payload


def assert_unique(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    if frame.empty:
        raise RuntimeError(f"{label} is empty")
    if frame.duplicated(list(columns)).any():
        duplicate = frame.loc[
            frame.duplicated(list(columns), keep=False), list(columns)
        ].iloc[0]
        raise RuntimeError(f"duplicate {label}: {duplicate.to_dict()}")


def _candidate_identity(candidate: Any) -> tuple[Any, ...]:
    return (
        str(candidate.split),
        str(candidate.fold),
        str(candidate.session),
        int(candidate.decision_time_ns),
        str(candidate.contract_id),
        int(candidate.canonical_strike_slot),
        int(candidate.policy_index),
    )


def freeze_p5_episodes(
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
    records: list[dict[str, Any]] = []
    role_receipts: dict[str, Any] = {}
    for role, sessions in role_sessions.items():
        candidates = []
        source_decisions = 0
        for session in sessions:
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
            source_decisions += len(opportunities)
            candidates.extend(references.fixed_heuristic_candidates(opportunities))
        candidate_map = {
            _candidate_identity(candidate): candidate for candidate in candidates
        }
        if len(candidate_map) != len(candidates):
            raise RuntimeError(f"duplicate P5 candidate intent identity: {role}")
        trades, state = references.replay_reference_v5(candidates)
        role_records = []
        for ordinal, trade in enumerate(trades):
            key = (
                str(trade.split),
                str(trade.fold),
                str(trade.session),
                int(trade.decision_time_ns),
                str(trade.contract_id),
                int(trade.canonical_strike_slot),
                int(trade.policy_index),
            )
            candidate = candidate_map.get(key)
            if candidate is None:
                raise RuntimeError(f"simulator trade lacks source P5 intent: {key}")
            episode_id = stable_hash(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "role": role,
                    "session": trade.session,
                    "decision_time_ns": int(trade.decision_time_ns),
                    "contract_id": trade.contract_id,
                    "canonical_slot": int(trade.canonical_strike_slot),
                    "policy_index": int(trade.policy_index),
                }
            )
            item = {
                "schema_version": SCHEMA_VERSION,
                "evidence_grade": EVIDENCE_GRADE,
                "episode_id": episode_id,
                "role": role,
                "fold": str(trade.fold),
                "session": str(trade.session),
                "role_episode_ordinal": int(ordinal),
                "decision_time_ns": int(trade.decision_time_ns),
                "contract_id": str(trade.contract_id),
                "right": str(trade.right),
                "canonical_slot": int(trade.canonical_strike_slot),
                "entry_ask": float(trade.entry_ask),
                "entry_quote_time_ns": int(trade.source_quote_time_ns),
                "source_context_time_ns": int(trade.source_context_time_ns),
                "policy_index": int(trade.policy_index),
                "policy_identity": "P5_VWAP_side_nearest_ATM",
                "p5_exit_source_quote_time_ns": int(
                    trade.label_source_exit_quote_time_ns
                ),
                "p5_realized_exit_time_ns": int(
                    trade.label_realized_exit_time_ns
                ),
                "p5_exit_reason_code": int(trade.label_exit_reason_code),
                "p5_executable_exit_bid": float(
                    trade.label_executable_exit_bid
                ),
                "policy_deadline_ns": int(trade.label_policy_deadline_ns),
                "p5_net_pnl_after_fee": float(
                    trade.raw_label_pnl_after_campaign_fee
                ),
                "source_simulator_version": str(trade.simulator_version),
                "source_processed_path": str(processed[str(trade.session)]),
                "source_processed_sha256": sha256_path(
                    processed[str(trade.session)]
                ),
                "source_normalized_path": str(normalized[str(trade.session)]),
                "source_normalized_sha256": sha256_path(
                    normalized[str(trade.session)]
                ),
            }
            role_records.append(item)
            records.append(item)
        role_receipts[role] = {
            "sessions": sessions,
            "source_decision_opportunities": int(source_decisions),
            "p5_candidate_intents": int(len(candidates)),
            "simulator_admitted_trades": int(len(role_records)),
            "simulator_skipped": dict(state.skipped),
            "simulator_config_hash": state.simulator_config_hash,
            "candidate_stream_hash": state.candidate_stream_hash,
            "candidate_payload_hash": state.candidate_payload_hash,
            "trade_identity_hash": state.trade_identity_hash,
            "admitted_entry_identity_hash": stable_hash(
                [
                    {
                        key: item[key]
                        for key in (
                            "episode_id",
                            "session",
                            "decision_time_ns",
                            "contract_id",
                            "canonical_slot",
                            "entry_ask",
                            "entry_quote_time_ns",
                        )
                    }
                    for item in role_records
                ]
            ),
        }
    frame = pd.DataFrame.from_records(records)
    identity_columns = (
        "role",
        "session",
        "decision_time_ns",
        "contract_id",
        "canonical_slot",
        "episode_id",
    )
    assert_unique(frame, identity_columns, label="frozen P5 episode identity")
    if set(frame["policy_index"].astype(int)) != {POLICY_INDEX}:
        raise RuntimeError("frozen entry stream contains a non-P5 policy")
    write_parquet(OUTPUT_ROOT / "frozen_p5_entry_episodes.parquet", frame)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "frozen_P5_entries_reconstructed_through_simulator_v5",
        "role_receipts": role_receipts,
        "frozen_episode_count": int(len(frame)),
        "training_episode_count": int(
            frame["role"].eq("nested_train").sum()
        ),
        "validation_episode_count": int(
            frame["role"].eq("nested_validation").sum()
        ),
        "identity_columns": list(identity_columns),
        "duplicate_identity_count": 0,
        "all_entries_produced_by_fixed_heuristic_candidates": True,
        "all_entries_admitted_by_simulator_v5": True,
        "comparators_must_use_identical_episode_list": True,
        "entry_identity_root_hash": stable_hash(
            frame[list(identity_columns)].to_dict("records")
        ),
        "parquet_sha256": sha256_path(
            OUTPUT_ROOT / "frozen_p5_entry_episodes.parquet"
        ),
    }
    write_json(OUTPUT_ROOT / "entry_identity_receipt.json", receipt)
    return frame, receipt


def context_features(
    spx_bars: pd.DataFrame,
    *,
    state_time: pd.Timestamp,
    right: str,
) -> dict[str, float]:
    context_time = state_time - pd.Timedelta(minutes=1)
    empty_vix = pd.DataFrame(columns=["event_time", "close"])
    market = _market_features(spx_bars, empty_vix, context_time)
    spx, _, vwap, omar, session_range, momentum_5, momentum_15 = map(
        float, market
    )
    spx_denom = max(abs(spx), 1.0)
    range_denom = max(abs(session_range), 1.0)
    gap = spx - vwap
    return {
        "spx_vwap_gap_points": gap,
        "spx_vwap_gap_bps": gap / spx_denom * 10_000.0,
        "spx_vwap_gap_over_session_range": gap / range_denom,
        "session_range_bps": session_range / spx_denom * 10_000.0,
        "momentum_5m_bps": momentum_5 / spx_denom * 10_000.0,
        "momentum_15m_bps": momentum_15 / spx_denom * 10_000.0,
        "momentum_5m_over_session_range": momentum_5 / range_denom,
        "momentum_15m_over_session_range": momentum_15 / range_denom,
        "omar_clipped_neg3_pos3": min(max(omar, -3.0), 3.0),
        "vwap_side_alignment_flag": float(
            (right == "C" and gap > 0.0) or (right == "P" and gap < 0.0)
        ),
        "omar_side_alignment_flag": float(
            (right == "C" and omar > 0.0)
            or (right == "P" and omar < 0.0)
        ),
        "momentum15_side_alignment_flag": float(
            (right == "C" and momentum_15 > 0.0)
            or (right == "P" and momentum_15 < 0.0)
        ),
        "_spx": spx,
    }


def balanced_training_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    counts = (
        frame.groupby(["session", "episode_id"], sort=False)
        .size()
        .rename("episode_rows")
    )
    episode_counts = (
        frame[["session", "episode_id"]]
        .drop_duplicates()
        .groupby("session", sort=False)
        .size()
        .rename("session_episodes")
    )
    keyed = frame[["session", "episode_id"]].join(
        counts, on=["session", "episode_id"]
    )
    keyed = keyed.join(episode_counts, on="session")
    raw = 1.0 / (
        keyed["episode_rows"].to_numpy(dtype=float)
        * keyed["session_episodes"].to_numpy(dtype=float)
    )
    return raw * (len(raw) / raw.sum())


def one_step_target(
    *,
    entry_ask: float,
    current_bid: float,
    next_bid: float,
    fee: float = ROUND_TRIP_FEE,
) -> dict[str, float]:
    q_exit = (float(current_bid) - float(entry_ask)) * CONTRACT_MULTIPLIER - fee
    q_hold = (float(next_bid) - float(entry_ask)) * CONTRACT_MULTIPLIER - fee
    return {
        "target_q_exit": q_exit,
        "target_q_hold_1m": q_hold,
        "target_a_hold_1m": q_hold - q_exit,
    }


def _velocity(values: Sequence[float], lag: int) -> float:
    if len(values) <= lag:
        return float("nan")
    current = float(values[-1])
    previous = float(values[-1 - lag])
    if not math.isfinite(current) or not math.isfinite(previous):
        return float("nan")
    return (current - previous) / float(lag)


def build_lifecycle_rows(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    invalid_episodes: list[dict[str, Any]] = []
    context_parity_max_error = 0.0
    _, processed, normalized = _scope_maps()
    for session, session_episodes in episodes.groupby("session", sort=True):
        option_columns = [
            "contract_id",
            "quote_time",
            "bid",
            "ask",
            "mid",
            "underlying_price",
            "strike",
            "right",
        ]
        option_frame = pq.read_table(
            normalized[str(session)], columns=option_columns
        ).to_pandas()
        option_frame["quote_time"] = pd.to_datetime(
            option_frame["quote_time"], utc=True
        )
        wanted = set(session_episodes["contract_id"].astype(str))
        option_frame = option_frame[
            option_frame["contract_id"].astype(str).isin(wanted)
        ].copy()
        spx_path = (
            ROOT / f"data/raw/index/spx_1m/{session}.official_spx.parquet"
        )
        if not spx_path.is_file():
            raise RuntimeError(f"official SPX context missing: {session}")
        spx_bars = pd.read_parquet(spx_path)
        spx_bars["event_time"] = pd.to_datetime(
            spx_bars["event_time"], utc=True
        )
        spx_bars = spx_bars.sort_values("event_time").reset_index(drop=True)

        with processed[str(session)].open("rb") as handle:
            processed_rows = pickle.load(handle)
        processed_by_ns = {
            int(pd.Timestamp(row["decision_time"]).value): row
            for row in processed_rows
        }
        for episode in session_episodes.itertuples(index=False):
            quotes = option_frame[
                option_frame["contract_id"].astype(str)
                == str(episode.contract_id)
            ].sort_values("quote_time", kind="mergesort")
            if quotes.empty or quotes["quote_time"].duplicated().any():
                invalid_episodes.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "reason": (
                            "missing_contract_path"
                            if quotes.empty
                            else "duplicate_contract_quote_time"
                        ),
                    }
                )
                continue
            quote_by_ns = {
                int(pd.Timestamp(row.quote_time).value): row
                for row in quotes.itertuples(index=False)
            }
            quote_times = np.asarray(sorted(quote_by_ns), dtype=np.int64)
            entry_ns = int(episode.decision_time_ns)
            deadline_ns = int(episode.policy_deadline_ns)
            state_times = np.arange(
                entry_ns + 60_000_000_000,
                deadline_ns + 1,
                60_000_000_000,
                dtype=np.int64,
            )
            if not len(state_times):
                invalid_episodes.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "reason": "empty_lifecycle_state_grid",
                    }
                )
                continue
            bids: list[float] = []
            pnls: list[float] = []
            mfe = -math.inf
            mae = math.inf
            mfe_index: int | None = None
            episode_records: list[dict[str, Any]] = []
            for state_index, state_ns in enumerate(state_times):
                state_time = pd.Timestamp(int(state_ns), tz="UTC")
                position = int(
                    np.searchsorted(quote_times, int(state_ns), side="right")
                ) - 1
                quote = (
                    quote_by_ns[int(quote_times[position])]
                    if position >= 0
                    else None
                )
                quote_ns = int(quote_times[position]) if position >= 0 else 0
                quote_age_seconds = (
                    (int(state_ns) - quote_ns) / 1_000_000_000.0
                    if quote is not None
                    else float("inf")
                )
                bid = float(quote.bid) if quote is not None else float("nan")
                ask = float(quote.ask) if quote is not None else float("nan")
                mid = float(quote.mid) if quote is not None else float("nan")
                executable = bool(
                    math.isfinite(bid)
                    and bid > 0.0
                    and quote_age_seconds <= MAX_CAUSAL_QUOTE_AGE_SECONDS
                )
                current_pnl = (
                    (bid - float(episode.entry_ask))
                    * CONTRACT_MULTIPLIER
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

                context = context_features(
                    spx_bars,
                    state_time=state_time,
                    right=str(episode.right),
                )
                spx = float(context.pop("_spx"))
                strike = (
                    float(quote.strike)
                    if quote is not None and math.isfinite(float(quote.strike))
                    else float("nan")
                )
                atm = round(spx / 5.0) * 5.0
                quantized_mid = _quantized_mid(mid)
                iv = _implied_vol(
                    quantized_mid,
                    spx,
                    strike,
                    _tte_years(state_time),
                    str(episode.right),
                )
                delta = float("nan")
                gamma = float("nan")
                if iv is not None:
                    delta, gamma = _bs_delta_gamma(
                        spx,
                        strike,
                        _tte_years(state_time),
                        iv,
                        str(episode.right),
                    )

                next_ns = int(state_ns) + 60_000_000_000
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
                    one_step_target(
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
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_grade": EVIDENCE_GRADE,
                    "episode_id": str(episode.episode_id),
                    "role": str(episode.role),
                    "fold": str(episode.fold),
                    "session": str(episode.session),
                    "state_index": int(state_index),
                    "state_time_ns": int(state_ns),
                    "state_quote_time_ns": int(quote_ns),
                    "state_quote_age_seconds": float(quote_age_seconds),
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
                        (deadline_ns - int(state_ns)) / 60_000_000_000
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
                        if math.isfinite(mfe) and math.isfinite(current_pnl)
                        else float("nan")
                    ),
                    "time_since_mfe_minutes": (
                        float(state_index - mfe_index)
                        if mfe_index is not None
                        else float("nan")
                    ),
                    "bid_velocity_1m": _velocity(bids, 1),
                    "bid_velocity_3m": _velocity(bids, 3),
                    "bid_velocity_5m": _velocity(bids, 5),
                    "pnl_velocity_1m": _velocity(pnls, 1),
                    "pnl_velocity_3m": _velocity(pnls, 3),
                    "pnl_velocity_5m": _velocity(pnls, 5),
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
                    "internal_delta": delta,
                    "internal_gamma": gamma,
                    "next_minute_bid_label_only": next_bid,
                    **target,
                }
                if int(state_ns) in processed_by_ns:
                    raw = processed_by_ns[int(state_ns)]
                    rights = tuple(map(str, raw["rights"]))
                    right_index = rights.index(str(episode.right))
                    from v4.model.protocol101_canonical_stage1_contract import (
                        feature_matrix,
                    )

                    expected = feature_matrix(raw)[10, right_index, :12]
                    observed = np.asarray(
                        [record[name] for name in CONTEXT_FEATURES],
                        dtype=float,
                    )
                    context_parity_max_error = max(
                        context_parity_max_error,
                        float(np.nanmax(np.abs(expected - observed))),
                    )
                episode_records.append(record)
            if not episode_records:
                invalid_episodes.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "reason": "no_lifecycle_rows",
                    }
                )
                continue
            records.extend(episode_records)
    frame = pd.DataFrame.from_records(records)
    assert_unique(
        frame,
        ("episode_id", "state_time_ns"),
        label="lifecycle state identity",
    )
    if not bool(frame.groupby("episode_id")["state_time_ns"].apply(
        lambda values: bool(
            np.all(np.diff(np.asarray(values, dtype=np.int64)) > 0)
        )
    ).all()):
        raise RuntimeError("non-monotonic lifecycle state identity")
    frame["training_weight"] = 0.0
    training_mask = frame["role"].eq("nested_train") & frame["target_valid"]
    frame.loc[training_mask, "training_weight"] = balanced_training_weights(
        frame.loc[training_mask]
    )
    write_parquet(OUTPUT_ROOT / "lifecycle_rows.parquet", frame)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "causal_lifecycle_rows_constructed",
        "row_count": int(len(frame)),
        "episode_count": int(frame["episode_id"].nunique()),
        "training_rows": int(frame["role"].eq("nested_train").sum()),
        "validation_rows": int(
            frame["role"].eq("nested_validation").sum()
        ),
        "usable_target_rows": int(frame["target_valid"].sum()),
        "missing_target_rows": int((~frame["target_valid"]).sum()),
        "invalid_episodes": invalid_episodes,
        "invalid_episode_count": len(invalid_episodes),
        "duplicate_state_identities": 0,
        "non_monotonic_state_paths": 0,
        "context_parity_max_abs_error_on_entry_grid": context_parity_max_error,
        "context_parity_pass": context_parity_max_error <= 1e-9,
        "lifecycle_parquet_sha256": sha256_path(
            OUTPUT_ROOT / "lifecycle_rows.parquet"
        ),
    }
    return frame, receipt


def feature_contract_and_firewall(frame: pd.DataFrame) -> dict[str, Any]:
    lowered = [name.lower() for name in LIFECYCLE_FEATURES]
    violations = sorted(
        {
            name
            for name, lower in zip(LIFECYCLE_FEATURES, lowered)
            for token in FORBIDDEN_FEATURE_TOKENS
            if token in lower
        }
    )
    missing_columns = sorted(set(LIFECYCLE_FEATURES) - set(frame.columns))
    feature_contract = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
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
    write_json(OUTPUT_ROOT / "feature_contract.json", feature_contract)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": (
            "pass" if not violations and not missing_columns else "fail"
        ),
        "forbidden_alias_violations": violations,
        "missing_feature_columns": missing_columns,
        "future_path_or_target_field_in_model_features": False,
        "entry_model_or_selector_output_in_model_features": False,
        "vendor_greek_or_internal_iv_in_model_features": False,
        "feature_contract_sha256": sha256_path(
            OUTPUT_ROOT / "feature_contract.json"
        ),
    }
    write_json(OUTPUT_ROOT / "feature_firewall_audit.json", audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"lifecycle feature firewall failed: {audit}")
    return audit


def manual_target_reproduction(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["role", "session", "episode_id", "state_index"]
    ).reset_index(drop=True)
    categories = [
        ordered[ordered["right"].eq("C")],
        ordered[ordered["right"].eq("P")],
        ordered[ordered["elapsed_minutes"].le(30)],
        ordered[ordered["minutes_to_deadline"].le(30)],
        ordered[ordered["target_a_hold_1m"].gt(0)],
        ordered[ordered["target_a_hold_1m"].lt(0)],
        ordered[~ordered["target_valid"]],
    ]
    chosen: list[int] = []
    for category in categories:
        if category.empty:
            continue
        positions = np.linspace(
            0,
            len(category) - 1,
            num=min(8, len(category)),
            dtype=int,
        )
        chosen.extend(map(int, category.iloc[positions].index))
    for index in ordered.index:
        if len(set(chosen)) >= 50:
            break
        chosen.append(int(index))
    selected = ordered.loc[sorted(set(chosen))[:50]]
    if len(selected) < 50:
        raise RuntimeError("fewer than 50 lifecycle targets for manual check")
    cases = []
    max_error = 0.0
    for row in selected.itertuples(index=False):
        if bool(row.target_valid):
            reproduced = one_step_target(
                entry_ask=float(row.entry_ask),
                current_bid=float(row.current_bid),
                next_bid=float(row.next_minute_bid_label_only),
            )
            errors = {
                name: abs(float(getattr(row, name)) - float(value))
                for name, value in reproduced.items()
            }
            max_error = max(max_error, *errors.values())
        else:
            reproduced = {
                "target_q_exit": None,
                "target_q_hold_1m": None,
                "target_a_hold_1m": None,
            }
            errors = {
                "target_q_exit": 0.0,
                "target_q_hold_1m": 0.0,
                "target_a_hold_1m": 0.0,
            }
            if not all(
                math.isnan(float(getattr(row, name)))
                for name in reproduced
            ):
                raise RuntimeError("missing target row contains a finite target")
        cases.append(
            {
                "episode_id": str(row.episode_id),
                "session": str(row.session),
                "right": str(row.right),
                "state_time_ns": int(row.state_time_ns),
                "elapsed_minutes": float(row.elapsed_minutes),
                "minutes_to_deadline": float(row.minutes_to_deadline),
                "target_valid": bool(row.target_valid),
                "entry_ask": float(row.entry_ask),
                "current_bid": (
                    float(row.current_bid)
                    if math.isfinite(float(row.current_bid))
                    else None
                ),
                "next_minute_bid": (
                    float(row.next_minute_bid_label_only)
                    if math.isfinite(float(row.next_minute_bid_label_only))
                    else None
                ),
                "observed": {
                    name: (
                        float(getattr(row, name))
                        if math.isfinite(float(getattr(row, name)))
                        else None
                    )
                    for name in reproduced
                },
                "reproduced": reproduced,
                "absolute_errors": errors,
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
        "contains_early_minutes": any(
            item["elapsed_minutes"] <= 30 for item in cases
        ),
        "contains_late_minutes": any(
            item["minutes_to_deadline"] <= 30 for item in cases
        ),
        "contains_winners": any(
            (item["observed"]["target_a_hold_1m"] or 0.0) > 0.0
            for item in cases
        ),
        "contains_losers": any(
            (item["observed"]["target_a_hold_1m"] or 0.0) < 0.0
            for item in cases
        ),
        "contains_missing_targets": any(
            not item["target_valid"] for item in cases
        ),
        "cases": cases,
    }
    write_json(OUTPUT_ROOT / "target_manual_reproduction.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError("manual target reproduction failed")
    return payload


def weighting_audit(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame[
        frame["role"].eq("nested_train") & frame["target_valid"]
    ].copy()
    if eligible.empty:
        raise RuntimeError("no target-valid nested training rows")
    sums = eligible.groupby("session")["training_weight"].sum()
    normalized = sums / sums.mean()
    return {
        "formula": (
            "equal_total_weight_per_session_then_per_episode_then_per_row"
        ),
        "training_rows": int(len(eligible)),
        "training_sessions": int(eligible["session"].nunique()),
        "training_episodes": int(eligible["episode_id"].nunique()),
        "minimum_normalized_session_weight": float(normalized.min()),
        "maximum_normalized_session_weight": float(normalized.max()),
        "maximum_session_weight_deviation": float(
            np.max(np.abs(normalized.to_numpy(dtype=float) - 1.0))
        ),
        "pass": bool(
            np.max(np.abs(normalized.to_numpy(dtype=float) - 1.0)) <= 1e-9
        ),
    }


def machinery_checks(
    episodes: pd.DataFrame,
    lifecycle: pd.DataFrame,
    entry_receipt: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    firewall: Mapping[str, Any],
    manual: Mapping[str, Any],
) -> dict[str, Any]:
    weights = weighting_audit(lifecycle)
    checks = {
        "goal_hash_matches_owner_value": (
            sha256_path(GOAL_PATH) == EXPECTED_GOAL_SHA256
        ),
        "preregistration_precedes_result_inspection": True,
        "nested_roles_exclude_final_oof": True,
        "all_entries_exact_P5": bool(
            entry_receipt["all_entries_produced_by_fixed_heuristic_candidates"]
        ),
        "all_entries_admitted_by_simulator_v5": bool(
            entry_receipt["all_entries_admitted_by_simulator_v5"]
        ),
        "entry_identities_unique": (
            int(entry_receipt["duplicate_identity_count"]) == 0
        ),
        "lifecycle_identities_unique_and_monotonic": (
            int(lifecycle_receipt["duplicate_state_identities"]) == 0
            and int(lifecycle_receipt["non_monotonic_state_paths"]) == 0
        ),
        "context_reproduces_signed_entry_grid": bool(
            lifecycle_receipt["context_parity_pass"]
        ),
        "feature_firewall_pass": firewall["status"] == "pass",
        "manual_target_reproduction_pass": manual["status"] == "pass",
        "group_balanced_weighting_pass": bool(weights["pass"]),
        "entry_episode_count_matches_lifecycle": (
            int(episodes["episode_id"].nunique())
            == int(lifecycle["episode_id"].nunique())
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "weighting_audit": weights,
        "entry_identity_root_hash": entry_receipt["entry_identity_root_hash"],
        "lifecycle_identity_root_hash": stable_hash(
            lifecycle[["episode_id", "state_time_ns"]].to_dict("records")
        ),
    }
    write_json(OUTPUT_ROOT / "machinery_checks.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"lifecycle machinery check failed: {payload}")
    return payload


def write_not_run_artifacts(
    *,
    validation_episodes: int,
) -> None:
    status = (
        "not_run_goal_requires_stop_insufficient_data_before_"
        "performance_interpretation"
    )
    (OUTPUT_ROOT / "model_receipts").mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_ROOT / "model_receipts" / "not_run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "status": status,
            "validation_episodes": validation_episodes,
            "minimum_required": MINIMUM_VALIDATION_EPISODES,
            "model_fit_executed": False,
        },
    )
    empty = pd.DataFrame(
        {
            "evidence_grade": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
        }
    )
    write_parquet(OUTPUT_ROOT / "predictions.parquet", empty)
    write_parquet(OUTPUT_ROOT / "episode_replays.parquet", empty)
    write_parquet(OUTPUT_ROOT / "serial_replays.parquet", empty)
    common = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": status,
        "validation_episodes": validation_episodes,
        "minimum_required": MINIMUM_VALIDATION_EPISODES,
        "validation_economics_inspected": False,
    }
    for name in (
        "serial_identity_receipts.json",
        "fixed_exit_results.json",
        "random_exit_results.json",
        "control_results.json",
        "pilot_results.json",
    ):
        write_json(OUTPUT_ROOT / name, common)


def resource_projection(
    *,
    started: float,
    lifecycle: pd.DataFrame,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output_bytes = sum(
        path.stat().st_size
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    )
    session_factor = 271.0 / float(
        lifecycle["session"].nunique()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "preflight_resource_projection_only_no_model_fit",
        "elapsed_seconds": elapsed,
        "peak_memory_mib": float(peak_kib) / 1024.0,
        "current_output_bytes": int(output_bytes),
        "lifecycle_rows": int(len(lifecycle)),
        "sessions": int(lifecycle["session"].nunique()),
        "linear_full_corpus_session_factor": session_factor,
        "projected_full_corpus_lifecycle_rows": int(
            math.ceil(len(lifecycle) * session_factor)
        ),
        "projected_full_corpus_preflight_output_bytes": int(
            math.ceil(output_bytes * session_factor)
        ),
        "limitations": (
            "No model was fit because the frozen validation episode floor "
            "failed; training-time projection is therefore unavailable."
        ),
    }


def write_hash_manifest() -> None:
    lines = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "hashes.sha256":
            continue
        lines.append(f"{sha256_path(path)}  {path.relative_to(OUTPUT_ROOT)}")
    (OUTPUT_ROOT / "hashes.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_report(
    *,
    decision: str,
    entry_receipt: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    machinery: Mapping[str, Any],
    preregistration_sha256: str,
) -> None:
    validation_episodes = int(entry_receipt["validation_episode_count"])
    text = f"""# Protocol101 FT2 Stage-0 P5 HOLD/EXIT Feasibility

Evidence grade: `{EVIDENCE_GRADE}`

Terminal decision: `{decision}`

Highest allowed claim: Exploratory non-promotable P5 HOLD/EXIT Stage-0
feasibility decision complete.

## Result

Exact simulator-v5 P5 reconstruction admitted
`{entry_receipt["training_episode_count"]}` nested-training episodes and
`{validation_episodes}` nested-validation episodes. The frozen Goal requires
at least `{MINIMUM_VALIDATION_EPISODES}` validation episodes before performance
may be interpreted.

The pilot therefore stopped before HGB fitting, fixed-exit comparison, random
exit comparison, Ledger-A economics, or Ledger-B economics. This is an
insufficient-data result for the signed pilot design, not evidence for or
against learned HOLD/EXIT skill.

## Mechanical Evidence

- Entry identity checks: pass.
- Exact P5 selector and simulator-v5 admission: pass.
- Causal lifecycle rows: `{lifecycle_receipt["row_count"]}`.
- Target-valid lifecycle rows: `{lifecycle_receipt["usable_target_rows"]}`.
- Missing one-minute targets retained and reported:
  `{lifecycle_receipt["missing_target_rows"]}`.
- Feature firewall: pass.
- Manual target reproduction: pass.
- Session/episode-balanced weighting: pass.
- Signed entry-grid context parity maximum absolute error:
  `{lifecycle_receipt["context_parity_max_abs_error_on_entry_grid"]}`.
- Overall machinery checks: `{machinery["status"]}`.

## Interpretation

P5 policy 5 normally holds the first admitted position until the 15:55 ET
deadline. Under one-account simulator-v5 occupancy this yields approximately
one standardized entry episode per session. Five validation sessions therefore
cannot supply the Goal's required 100 independent episodes.

No model was selected, promoted, or interpreted. No full lifecycle campaign,
entry training, G9, protected holdout, recorder/sealed evidence, broker,
paper, runtime, promotion, launchd, paid-data, or real-money action occurred.

## Frozen Receipts

- Preregistration SHA-256: `{preregistration_sha256}`
- Entry identity root:
  `{entry_receipt["entry_identity_root_hash"]}`
- Lifecycle identity root:
  `{machinery["lifecycle_identity_root_hash"]}`
"""
    (OUTPUT_ROOT / "report.md").write_text(text, encoding="utf-8")


def run() -> str:
    started = time.perf_counter()
    preregistration = preregister()
    verify_preregistration()
    write_progress("freezing_exact_P5_simulator_v5_entries")
    episodes, entry_receipt = freeze_p5_episodes(preregistration)
    write_progress(
        "building_causal_lifecycle_rows",
        frozen_episodes=int(len(episodes)),
    )
    lifecycle, lifecycle_receipt = build_lifecycle_rows(episodes)
    firewall = feature_contract_and_firewall(lifecycle)
    manual = manual_target_reproduction(lifecycle)
    machinery = machinery_checks(
        episodes,
        lifecycle,
        entry_receipt,
        lifecycle_receipt,
        firewall,
        manual,
    )
    validation_episodes = int(entry_receipt["validation_episode_count"])
    if validation_episodes < MINIMUM_VALIDATION_EPISODES:
        decision = "stop_insufficient_data"
        write_not_run_artifacts(validation_episodes=validation_episodes)
    else:
        raise RuntimeError(
            "current corpus unexpectedly reaches the model-fit branch; "
            "bounded implementation must be extended without changing the "
            "preregistered contract"
        )
    resource = resource_projection(started=started, lifecycle=lifecycle)
    write_json(OUTPUT_ROOT / "resource_projection.json", resource)
    write_json(
        OUTPUT_ROOT / "repair_log.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "repair_iterations_used": 0,
            "maximum_repair_iterations": 3,
            "repairs": [],
        },
    )
    decision_payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "terminal_decision": decision,
        "validation_episode_count": validation_episodes,
        "minimum_validation_episode_count": MINIMUM_VALIDATION_EPISODES,
        "performance_interpreted": False,
        "model_fit_executed": False,
        "ledger_a_economics_executed": False,
        "ledger_b_economics_executed": False,
        "full_lifecycle_campaign_authorized": False,
        "next_owner_decision_required": True,
        "highest_allowed_claim": (
            "Exploratory non-promotable P5 HOLD/EXIT Stage-0 "
            "feasibility decision complete."
        ),
        "side_effects": preregistration["side_effects"],
    }
    write_json(OUTPUT_ROOT / "decision.json", decision_payload)
    write_report(
        decision=decision,
        entry_receipt=entry_receipt,
        lifecycle_receipt=lifecycle_receipt,
        machinery=machinery,
        preregistration_sha256=sha256_path(
            OUTPUT_ROOT / "preregistration.json"
        ),
    )
    write_progress(
        "complete",
        terminal_decision=decision,
        validation_episode_count=validation_episodes,
        minimum_validation_episode_count=MINIMUM_VALIDATION_EPISODES,
        model_fit_executed=False,
    )
    write_hash_manifest()
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute mutable result artifacts while preserving the frozen "
            "preregistration. The dedicated output directory only is used."
        ),
    )
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
