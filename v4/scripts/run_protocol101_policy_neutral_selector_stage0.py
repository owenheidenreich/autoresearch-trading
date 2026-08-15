"""Exploratory, non-promotable Stage-0 selector feasibility pilot."""
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import (
    NeuralDatasetConfig,
    _contract_quote_path,
    _prepare_options,
)
from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_policy_neutral_selector import (
    BARRIERS,
    BARRIER_COLUMNS,
    BREAKEVEN_COLUMNS,
    M0_FEATURES,
    M1_FEATURES,
    MODEL_CONFIG,
    PATH_COLUMNS,
    RAW_TARGET_COLUMNS,
    RETURN_COLUMNS,
    candidate_order,
    fit_selector,
    group_balanced_weights,
    numeric_hash,
    permute_targets_within_decision,
    score_selector,
    select_position,
    stable_hash,
)
from v4.scripts.run_protocol101_policy_neutral_contract_selector import (
    FROZEN_INPUTS,
    ROOT,
    _noise_model,
    _scope_maps,
    _session_frame,
    read_json,
    sha256_path,
    write_json,
)


EVIDENCE_GRADE = "exploratory_non_promotable_stage0"
SCHEMA_VERSION = "Protocol101PolicyNeutralSelectorStage0V2"
CAMPAIGN_ID = "protocol101-policy-neutral-selector-stage0-attempt002"
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002"
)
SESSION_ROOT = OUTPUT_ROOT / "risk_sets_by_session"
MODEL_ROOT = OUTPUT_ROOT / "model_receipts"
ATTEMPT001_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt001"
)
ATTEMPT002_GOAL = Path(
    "/Users/gduby/.codex/attachments/"
    "74283e86-dd94-4318-9220-96e580fbce38/pasted-text-1.txt"
)
OWNER_SELECTOR_GOAL = Path(
    "/Users/gduby/.codex/attachments/"
    "fff2e3d8-66e1-4cd2-99a9-27b44962e5be/pasted-text.txt"
)
FROZEN_AUTHORITIES = {
    "attempt002_goal": ATTEMPT002_GOAL,
    "owner_approved_selector_goal": OWNER_SELECTOR_GOAL,
    "attempt001_decision": ATTEMPT001_ROOT / "decision.json",
    "attempt001_contract_audit": (
        ATTEMPT001_ROOT / "scientific_contract_audit.json"
    ),
    "attempt001_preregistration": ATTEMPT001_ROOT / "preregistration.json",
    "attempt001_hash_manifest": ATTEMPT001_ROOT / "hashes.sha256",
    **{
        f"governed_{name}": path
        for name, path in FROZEN_INPUTS.items()
        if name not in {"goal"}
    },
}
TRAIN_SESSION_COUNT = 15
VALIDATION_SESSION_COUNT = 5
TOTAL_SESSION_COUNT = TRAIN_SESSION_COUNT + VALIDATION_SESSION_COUNT
PILOT_MODEL_SEED = 42
PILOT_SHUFFLE_SEEDS = (8600, 8601)
MINIMUM_VALIDATION_SESSIONS = 3
MINIMUM_VALIDATION_OPPORTUNITIES = 300


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_progress(status: str, **fields: Any) -> None:
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


def _fold_sessions() -> tuple[list[str], list[str], dict[str, Any]]:
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
    if len(eligible) < TOTAL_SESSION_COUNT:
        raise RuntimeError("insufficient non-OOF Fold-1 training sessions")
    selected = eligible[-TOTAL_SESSION_COUNT:]
    training = selected[:TRAIN_SESSION_COUNT]
    validation = selected[TRAIN_SESSION_COUNT:]
    if set(selected) & final_oof:
        raise RuntimeError("Stage-0 session leaked into final OOF validation")
    if not max(training) < min(validation):
        raise RuntimeError("nested split is not chronological")
    return training, validation, {
        "fold_id": str(fold["fold_id"]),
        "eligible_training_prefix_sessions": eligible,
        "selected_sessions": selected,
        "nested_train_sessions": training,
        "nested_validation_sessions": validation,
        "final_oof_session_count": len(final_oof),
        "final_oof_overlap": [],
    }


def preregistration_payload() -> dict[str, Any]:
    training, validation, split = _fold_sessions()
    frozen = {
        name: {"path": str(path), "sha256": sha256_path(path)}
        for name, path in FROZEN_AUTHORITIES.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "campaign_id": CAMPAIGN_ID,
        "purpose": "mechanical_and_directional_compute_feasibility_only",
        "non_promotable": True,
        "may_select_or_validate_model": False,
        "may_change_full_campaign_contract": False,
        "attempt001_artifacts_may_be_reused": False,
        "attempt001_model_results_may_be_inspected": False,
        "governed_fold": split["fold_id"],
        "session_selection": {
            "rule": (
                "last_20_chronological_sessions_from_fold1_training_side_"
                "excluding_union_of_all_final_oof_validation_sessions"
            ),
            "training_sessions": training,
            "validation_sessions": validation,
            "training_count": TRAIN_SESSION_COUNT,
            "validation_count": VALIDATION_SESSION_COUNT,
        },
        "models": {
            "M0": list(M0_FEATURES),
            "M1": list(M1_FEATURES),
            "family": "HistGradientBoostingRegressor",
            "configuration": MODEL_CONFIG,
            "seed": PILOT_MODEL_SEED,
            "noise": "same_frozen_1x_divergence_noise_law",
        },
        "controls": {
            "deterministic": "P5_VWAP_side_nearest_ATM",
            "exact_random": "mean_utility_complete_risk_set",
            "constant_score": "frozen_tie_break",
            "reversed": ["M0", "M1"],
            "strong_shuffle_seeds": list(PILOT_SHUFFLE_SEEDS),
            "strong_shuffle_rule": (
                "permute_primary_utility_within_each_training_decision_group"
            ),
        },
        "opportunity_contract": {
            "timing_engine": "frozen_P5",
            "include_if": (
                "byte_equivalent_fixed_heuristic_candidates_emits_exactly_one"
            ),
            "exclude_if": "P5_abstains",
            "excluded_utility": "none_no_synthetic_no_trade_value",
            "retained_risk_set": "all_governed_eligible_calls_and_puts",
            "only_selected_contract_may_differ": True,
        },
        "ledger": "Ledger_A_decision_local_only",
        "forbidden": [
            "full_five_fold_training",
            "model_seeds_43_or_44",
            "remaining_18_D1_shuffles",
            "simulator_v5_serial_replay",
            "20k_inference",
            "campaign_multiplicity",
            "independent_final_verification",
            "G9",
            "protected_holdout",
            "HOLD_EXIT_training",
            "broker_or_runtime_changes",
        ],
        "machinery_checks": [
            "H0_12_feature_reference_to_H3_17_feature_identity_join",
            "candidate_membership_exact_governed_risk_set",
            "HGB_native_missing_value_handling_without_future_imputation",
            "feature_firewall_excludes_future_paths_and_targets",
            "group_balanced_weights",
            "manual_target_reproduction",
            "runtime_disk_memory_projection",
        ],
        "directional_metric": (
            "mean_session(mean_opportunity(selected_utility-baseline_utility))"
        ),
        "comparable_control_definition": (
            "a_reversed_or_shuffle_control_is_comparable_when_its_minimum_"
            "lift_over_deterministic_and_exact_random_is_greater_than_or_"
            "equal_to_the_best_real_model_minimum_lift"
        ),
        "proceed_rule": [
            "all_machinery_checks_pass",
            "M0_or_M1_positive_vs_deterministic_and_exact_random",
            "best_real_minimum_lift_strictly_exceeds_every_reversed_and_"
            "shuffle_control_minimum_lift",
            "no_frozen_scientific_contract_change_required",
        ],
        "insufficient_data_floor": {
            "validation_sessions": MINIMUM_VALIDATION_SESSIONS,
            "validation_opportunities": MINIMUM_VALIDATION_OPPORTUNITIES,
            "routing": (
                "stop_no_preliminary_signal_with_failure_class_"
                "insufficient_data"
            ),
        },
        "terminal_decisions": [
            "proceed_to_full_campaign",
            "stop_no_preliminary_signal",
            "stop_mechanical_blocker",
            "stop_scientific_contract_defect",
        ],
        "frozen_input_hashes": frozen,
        "side_effects": {
            "full_campaign_training": False,
            "simulator_replay": False,
            "g9": False,
            "protected_holdout": False,
            "broker": False,
            "paper_submit": False,
            "paid_download": False,
            "promotion_or_default_change": False,
            "runtime_or_launchd_change": False,
        },
    }


def preregister() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = preregistration_payload()
    path = OUTPUT_ROOT / "preregistration.json"
    if path.is_file():
        if read_json(path) != _clean(payload):
            raise RuntimeError("Stage-0 preregistration changed")
    else:
        write_json(path, payload)
    freeze = f"{sha256_path(path)}  preregistration.json\n"
    freeze_path = OUTPUT_ROOT / "preregistration.sha256"
    if freeze_path.exists() and freeze_path.read_text() != freeze:
        raise RuntimeError("Stage-0 preregistration freeze changed")
    freeze_path.write_text(freeze, encoding="utf-8")
    _write_progress(
        "preregistered",
        preregistration_sha256=sha256_path(path),
        model_results_inspected=False,
    )


def verify_preregistration() -> dict[str, Any]:
    path = OUTPUT_ROOT / "preregistration.json"
    freeze = OUTPUT_ROOT / "preregistration.sha256"
    if not path.is_file() or not freeze.is_file():
        raise RuntimeError("Stage-0 preregistration missing")
    digest, name = freeze.read_text().strip().split("  ", 1)
    if name != path.name or sha256_path(path) != digest:
        raise RuntimeError("Stage-0 preregistration hash mismatch")
    payload = read_json(path)
    current = {
        name: {"path": str(source), "sha256": sha256_path(source)}
        for name, source in FROZEN_AUTHORITIES.items()
    }
    if payload["frozen_input_hashes"] != current:
        raise RuntimeError("Stage-0 frozen input changed")
    return payload


def build_risk_sets(
    *,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prereg = verify_preregistration()
    training_sessions = prereg["session_selection"]["training_sessions"]
    validation_sessions = prereg["session_selection"]["validation_sessions"]
    _, processed, normalized = _scope_maps()
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    receipts = []
    rebuilt_sessions = 0
    reused_sessions = 0
    build_started = time.perf_counter()
    for index, session in enumerate(
        [*training_sessions, *validation_sessions],
        start=1,
    ):
        role = "nested_train" if session in training_sessions else "nested_validation"
        path = SESSION_ROOT / f"{session}.parquet"
        receipt_path = SESSION_ROOT / f"{session}.json"
        if path.is_file() and receipt_path.is_file() and not force_rebuild:
            receipt = read_json(receipt_path)
            if receipt["parquet_sha256"] != sha256_path(path):
                raise RuntimeError(f"Stage-0 risk-set checkpoint changed: {session}")
            reused_sessions += 1
        else:
            frame, receipt = _session_frame(
                session=session,
                processed_path=processed[session],
                normalized_path=normalized[session],
                fold_id="expanding_fold_01_stage0",
                split=role,
                campaign_id=CAMPAIGN_ID,
                require_fixed_p5_opportunity=True,
            )
            frame["evidence_grade"] = EVIDENCE_GRADE
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, path)
            receipt["evidence_grade"] = EVIDENCE_GRADE
            receipt["role"] = role
            receipt["parquet_path"] = str(path)
            receipt["parquet_sha256"] = sha256_path(path)
            write_json(receipt_path, receipt)
            rebuilt_sessions += 1
        receipts.append(receipt)
        _write_progress(
            "building_stage0_risk_sets",
            completed_sessions=index,
            total_sessions=TOTAL_SESSION_COUNT,
        )
    training = pd.concat(
        [pd.read_parquet(SESSION_ROOT / f"{session}.parquet") for session in training_sessions],
        ignore_index=True,
    )
    validation = pd.concat(
        [pd.read_parquet(SESSION_ROOT / f"{session}.parquet") for session in validation_sessions],
        ignore_index=True,
    )
    combined_path = OUTPUT_ROOT / "risk_sets.parquet"
    combined = pd.concat([training, validation], ignore_index=True)
    temporary = combined_path.with_name(
        f".{combined_path.name}.tmp-{os.getpid()}"
    )
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, combined_path)
    elapsed = time.perf_counter() - build_started
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": "stage0_risk_sets_complete",
        "training_sessions": len(training_sessions),
        "validation_sessions": len(validation_sessions),
        "training_opportunities": int(
            training[["session", "decision_time_ns"]].drop_duplicates().shape[0]
        ),
        "validation_opportunities": int(
            validation[["session", "decision_time_ns"]]
            .drop_duplicates()
            .shape[0]
        ),
        "training_candidates": int(len(training)),
        "validation_candidates": int(len(validation)),
        "build_elapsed_seconds": elapsed,
        "build_mode": (
            "clean_source_rebuild"
            if rebuilt_sessions == TOTAL_SESSION_COUNT
            else "checkpoint_reload"
        ),
        "rebuilt_sessions": rebuilt_sessions,
        "reused_sessions": reused_sessions,
        "clean_source_build_elapsed_seconds": (
            elapsed if rebuilt_sessions == TOTAL_SESSION_COUNT else None
        ),
        "risk_sets_sha256": sha256_path(combined_path),
        "session_receipts": receipts,
    }
    write_json(OUTPUT_ROOT / "risk_set_receipts.json", receipt)
    opportunity_filter_receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "filter": (
            "byte_equivalent_fixed_heuristic_candidates_emits_exactly_one"
        ),
        "source_opportunities": int(
            sum(int(item["source_opportunities"]) for item in receipts)
        ),
        "p5_permitted_opportunities": int(
            sum(int(item["p5_permitted_opportunities"]) for item in receipts)
        ),
        "p5_abstention_opportunities": int(
            sum(int(item["p5_abstention_opportunities"]) for item in receipts)
        ),
        "abstention_reasons": {
            "no_eligible_vwap_side_contract": int(
                sum(
                    int(
                        item["p5_abstention_reasons"][
                            "no_eligible_vwap_side_contract"
                        ]
                    )
                    for item in receipts
                )
            )
        },
        "by_session": {
            str(item["session"]): {
                "role": str(item["role"]),
                "source": int(item["source_opportunities"]),
                "permitted": int(item["p5_permitted_opportunities"]),
                "abstained": int(item["p5_abstention_opportunities"]),
            }
            for item in receipts
        },
        "excluded_before_target_aggregation_fitting_scoring": True,
        "synthetic_no_trade_utility_assigned": False,
        "attempt001_artifacts_reused": False,
    }
    write_json(
        OUTPUT_ROOT / "opportunity_filter_receipt.json",
        opportunity_filter_receipt,
    )
    return training, validation, receipt


def _manual_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros(len(values), dtype=float)
    count = int(finite.sum())
    if count == 0:
        return result
    if count == 1:
        result[np.flatnonzero(finite)[0]] = 0.5
        return result
    finite_values = values[finite]
    ranks = np.asarray(
        [
            1.0
            + float(np.sum(finite_values < value))
            + 0.5 * float(np.sum(finite_values == value) - 1)
            for value in finite_values
        ]
    )
    result[finite] = (ranks - 1.0) / float(count - 1)
    return result


def _manual_raw_target(
    quote_ns: np.ndarray,
    bids: np.ndarray,
    decision_ns: int,
    ask: float,
) -> dict[str, float]:
    premium = float(ask) * 100.0
    end_ns = int(decision_ns) + 15 * 60_000_000_000
    start = int(np.searchsorted(quote_ns, int(decision_ns), side="right"))
    stop = int(np.searchsorted(quote_ns, end_ns, side="right"))
    times = quote_ns[start:stop]
    path_bids = bids[start:stop]
    valid = np.isfinite(path_bids) & (path_bids >= 0.0)
    times = times[valid]
    path_bids = path_bids[valid]
    returns = (
        ((path_bids - ask) * 100.0 - 3.0) / premium
        if len(path_bids)
        else np.asarray([], dtype=float)
    )
    out: dict[str, float] = {}
    for minutes in (1, 2, 5, 10, 15):
        horizon = int(decision_ns) + minutes * 60_000_000_000
        position = int(np.searchsorted(quote_ns, horizon, side="right")) - 1
        if (
            position < 0
            or int(quote_ns[position]) <= int(decision_ns)
            or horizon - int(quote_ns[position]) > 90_000_000_000
            or not math.isfinite(float(bids[position]))
            or float(bids[position]) < 0.0
        ):
            value = float("nan")
        else:
            value = ((float(bids[position]) - ask) * 100.0 - 3.0) / premium
        out[f"return_{minutes}m"] = value
    out["mfe_15m"] = float(np.max(returns)) if len(returns) else float("nan")
    out["mae_15m"] = float(np.min(returns)) if len(returns) else float("nan")
    hits = np.flatnonzero(returns >= 0.0)
    out["time_to_breakeven_quality"] = (
        -float((int(times[int(hits[0])]) - int(decision_ns)) / 60_000_000_000)
        if len(hits)
        else -16.0
    )
    for name, favorable, adverse in BARRIERS:
        if not len(returns):
            out[f"barrier_{name}"] = float("nan")
            continue
        positive = np.flatnonzero(returns >= favorable)
        negative = np.flatnonzero((returns <= adverse) | (path_bids <= 0.0))
        pi = int(positive[0]) if len(positive) else None
        ni = int(negative[0]) if len(negative) else None
        if pi is not None and (ni is None or int(times[pi]) < int(times[ni])):
            out[f"barrier_{name}"] = 1.0
        elif ni is not None:
            out[f"barrier_{name}"] = -1.0
        else:
            out[f"barrier_{name}"] = 0.0
    return out


def manual_target_verification(
    combined: pd.DataFrame,
    *,
    sample_decisions: int = 8,
) -> dict[str, Any]:
    _, _, normalized = _scope_maps()
    identities = (
        combined[["session", "decision_time_ns"]]
        .drop_duplicates()
        .sort_values(["session", "decision_time_ns"])
    )
    positions = np.linspace(
        0,
        len(identities) - 1,
        num=min(sample_decisions, len(identities)),
        dtype=int,
    )
    selected = identities.iloc[positions]
    cases = []
    maximum_raw_error = 0.0
    maximum_utility_error = 0.0
    for session, decision_ns in selected.itertuples(index=False):
        group = combined[
            (combined["session"] == session)
            & (combined["decision_time_ns"] == decision_ns)
        ].copy()
        table = pq.read_table(normalized[str(session)])
        options = _prepare_options(table, NeuralDatasetConfig())
        paths = {
            str(contract_id): _contract_quote_path(
                rows,
                enforce_unique_path=True,
            )
            for contract_id, rows in options.groupby("contract_id", sort=False)
        }
        raw_columns: dict[str, list[float]] = {
            column: [] for column in RAW_TARGET_COLUMNS
        }
        for row in group.itertuples(index=False):
            path = paths[str(row.contract_id)]
            manual = _manual_raw_target(
                path.quote_ns,
                path.bid,
                int(row.decision_time_ns),
                float(row.entry_ask),
            )
            for column in RAW_TARGET_COLUMNS:
                observed = float(getattr(row, column))
                expected = float(manual[column])
                if math.isnan(observed) and math.isnan(expected):
                    error = 0.0
                else:
                    error = abs(observed - expected)
                maximum_raw_error = max(maximum_raw_error, error)
                raw_columns[column].append(expected)
        pct = {
            column: _manual_percentiles(np.asarray(values, dtype=float))
            for column, values in raw_columns.items()
        }
        utility = 0.25 * (
            np.mean([pct[column] for column in RETURN_COLUMNS], axis=0)
            + np.mean([pct[column] for column in PATH_COLUMNS], axis=0)
            + np.mean([pct[column] for column in BREAKEVEN_COLUMNS], axis=0)
            + np.mean([pct[column] for column in BARRIER_COLUMNS], axis=0)
        )
        utility_error = float(
            np.max(
                np.abs(
                    utility
                    - group["primary_utility"].to_numpy(dtype=float)
                )
            )
        )
        maximum_utility_error = max(maximum_utility_error, utility_error)
        cases.append(
            {
                "session": str(session),
                "decision_time_ns": int(decision_ns),
                "candidates": int(len(group)),
                "maximum_raw_target_error": maximum_raw_error,
                "maximum_utility_error": utility_error,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "sampled_decisions": len(cases),
        "cases": cases,
        "maximum_raw_target_error": maximum_raw_error,
        "maximum_utility_error": maximum_utility_error,
        "pass": bool(maximum_raw_error <= 1e-12 and maximum_utility_error <= 1e-12),
        "oracle": "independent_explicit_formula_and_manual_midrank",
    }
    write_json(OUTPUT_ROOT / "target_manual_reproduction.json", result)
    return result


def machinery_checks(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    risk_receipt: Mapping[str, Any],
    target_verification: Mapping[str, Any],
) -> dict[str, Any]:
    noise = _noise_model()
    sample = training.iloc[: min(10_000, len(training))].copy()
    missing_results = {}
    for model_name, features in (("M0", M0_FEATURES), ("M1", M1_FEATURES)):
        source = sample.loc[:, list(features)].copy()
        source["abs_offset"] = np.abs(sample["offset"].to_numpy(dtype=float))
        injected = noise.inject_dataframe(
            source,
            feature_columns=features,
            seed=PILOT_MODEL_SEED,
            scale=1.0,
        )
        before = np.isnan(source[list(features)].to_numpy(dtype=float))
        after = np.isnan(injected[list(features)].to_numpy(dtype=float))
        missing_results[model_name] = {
            "missing_cells": int(before.sum()),
            "missing_mask_preserved": bool(np.array_equal(before, after)),
            "imputation_used": False,
            "hgb_native_missing_support": True,
        }
    weights = group_balanced_weights(training.reset_index(drop=True))
    weight_frame = training[["session", "decision_time_ns"]].reset_index(drop=True)
    weight_frame["weight"] = weights
    session_totals = weight_frame.groupby("session")["weight"].sum()
    decision_totals = weight_frame.groupby(
        ["session", "decision_time_ns"]
    )["weight"].sum()
    within_session_spread = (
        decision_totals.groupby(level=0).max()
        - decision_totals.groupby(level=0).min()
    )
    forbidden_tokens = (
        "return_",
        "mfe",
        "mae",
        "breakeven",
        "barrier",
        "primary_utility",
        "label",
        "future",
        "path",
    )
    feature_firewall_pass = not any(
        token in feature.lower()
        for feature in (*M0_FEATURES, *M1_FEATURES)
        for token in forbidden_tokens
    )
    receipts = risk_receipt["session_receipts"]
    combined = pd.concat([training, validation], ignore_index=True)
    deterministic_matches = (
        (
            combined["contract_id"].astype(str)
            == combined["deterministic_p5_contract_id"].astype(str)
        )
        & (
            combined["right"].astype(str)
            == combined["deterministic_p5_right"].astype(str)
        )
        & (
            combined["canonical_slot"].astype(int)
            == combined["deterministic_p5_canonical_slot"].astype(int)
        )
    )
    deterministic_match_counts = deterministic_matches.groupby(
        [combined["session"], combined["decision_time_ns"]]
    ).sum()
    group_sizes = combined.groupby(
        ["session", "decision_time_ns"]
    ).size()
    declared_sizes = combined.groupby(
        ["session", "decision_time_ns"]
    )["candidate_count"].first()
    risk_hash_counts = combined.groupby(
        ["session", "decision_time_ns"]
    )["risk_set_hash"].nunique()
    ordered_candidate_identities_valid = all(
        group["candidate_index"].astype(int).tolist()
        == list(range(len(group)))
        for _, group in combined.sort_values(
            ["session", "decision_time_ns", "candidate_index"]
        ).groupby(["session", "decision_time_ns"], sort=False)
    )
    checks = {
        "H0_H3_identity_join": all(
            item["reference_opportunities_validated"]
            and item["reference_feature_view"] == "finite_H0_context_features"
            and item["selector_feature_join"]
            == "identity_exact_H3_17_feature_tensor"
            for item in receipts
        ),
        "p5_opportunity_filter": bool(
            all(item["p5_opportunity_filter_applied"] for item in receipts)
            and all(
                int(item["source_opportunities"])
                == int(item["p5_permitted_opportunities"])
                + int(item["p5_abstention_opportunities"])
                for item in receipts
            )
        ),
        "deterministic_baseline_unique_and_in_risk_set": bool(
            len(deterministic_match_counts)
            and (deterministic_match_counts == 1).all()
        ),
        "exact_governed_risk_set": bool(
            not training.duplicated(
                ["session", "decision_time_ns", "contract_id"]
            ).any()
            and not validation.duplicated(
                ["session", "decision_time_ns", "contract_id"]
            ).any()
            and group_sizes.equals(declared_sizes)
            and (risk_hash_counts == 1).all()
        ),
        "shared_ordered_candidate_identities": bool(
            ordered_candidate_identities_valid
        ),
        "missing_values_native_HGB": all(
            item["missing_mask_preserved"] and not item["imputation_used"]
            for item in missing_results.values()
        ),
        "feature_firewall": feature_firewall_pass,
        "group_balanced_weights": bool(
            float(session_totals.max() - session_totals.min()) <= 1e-8
            and float(within_session_spread.max()) <= 1e-8
        ),
        "manual_target_reproduction": bool(target_verification["pass"]),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "checks": checks,
        "all_pass": all(checks.values()),
        "missing_value_diagnostics": missing_results,
        "feature_names": {"M0": list(M0_FEATURES), "M1": list(M1_FEATURES)},
        "feature_target_overlap": sorted(
            set((*M0_FEATURES, *M1_FEATURES))
            & set((*RAW_TARGET_COLUMNS, "primary_utility"))
        ),
        "weight_diagnostics": {
            "mean": float(np.mean(weights)),
            "minimum": float(np.min(weights)),
            "maximum": float(np.max(weights)),
            "session_total_spread": float(
                session_totals.max() - session_totals.min()
            ),
            "maximum_within_session_decision_total_spread": float(
                within_session_spread.max()
            ),
            "weight_hash": numeric_hash(weights),
        },
    }
    write_json(OUTPUT_ROOT / "machinery_checks.json", result)
    return result


def scientific_contract_audit(validation: pd.DataFrame) -> dict[str, Any]:
    """Check whether the frozen deterministic comparator exists everywhere."""

    missing: list[dict[str, Any]] = []
    for (session, decision_ns), group in validation.groupby(
        ["session", "decision_time_ns"],
        sort=False,
    ):
        side = str(group["vwap_side"].iloc[0])
        available = sorted(set(group["right"].astype(str)))
        if side not in available:
            missing.append(
                {
                    "session": str(session),
                    "decision_time_ns": int(decision_ns),
                    "vwap_side": side,
                    "available_rights": available,
                    "candidate_count": int(len(group)),
                }
            )
    total = int(
        validation[["session", "decision_time_ns"]]
        .drop_duplicates()
        .shape[0]
    )
    by_session = {}
    for item in missing:
        by_session[item["session"]] = by_session.get(item["session"], 0) + 1
    defect = bool(missing)
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "status": (
            "scientific_contract_defect"
            if defect
            else "scientific_contract_consistent"
        ),
        "validation_opportunities": total,
        "deterministic_baseline_undefined_opportunities": len(missing),
        "undefined_share": len(missing) / total if total else 0.0,
        "by_session": by_session,
        "examples": missing[:50],
        "defect": (
            "The frozen opportunity rule admits any decision with at least "
            "one eligible contract, but fixed_heuristic_candidates skips a "
            "decision when no eligible contract matches VWAP side. Ledger A "
            "requires every selector to choose one contract on the same grid, "
            "so the deterministic baseline is undefined on those decisions."
            if defect
            else None
        ),
        "why_not_mechanical": (
            "Forcing the heuristic onto the opposite side, assigning a "
            "synthetic utility to no-trade, or dropping decisions would each "
            "change the frozen opportunity/comparator contract."
            if defect
            else None
        ),
        "smallest_owner_decision": (
            "Amend the full campaign opportunity stream to include only "
            "decisions where fixed_heuristic_candidates emits a contract, "
            "or explicitly define and preregister no-trade utility before "
            "any selector result is inspected."
            if defect
            else None
        ),
        "model_scores_inspected_to_make_this_finding": False,
    }
    write_json(OUTPUT_ROOT / "scientific_contract_audit.json", result)
    return result


def _pickle_model(path: Path, model: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _fit_and_score(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    model_name: str,
    features: Sequence[str],
    target_override: np.ndarray | None,
    namespace: str,
    extra: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    noise = _noise_model()
    started = time.perf_counter()
    model, fit_receipt = fit_selector(
        training.reset_index(drop=True),
        model_name=model_name,
        feature_names=features,
        seed=PILOT_MODEL_SEED,
        fold_id="expanding_fold_01_stage0",
        noise_model=noise,
        target_override=target_override,
    )
    scores = score_selector(
        model,
        validation,
        feature_names=features,
        noise_model=noise,
        noise_seed=PILOT_MODEL_SEED + 200_000,
        noise_scale=1.0,
    )
    elapsed = time.perf_counter() - started
    root = MODEL_ROOT / namespace / model_name
    model_path = root / "model.pkl"
    receipt_path = root / "receipt.json"
    model_hash = _pickle_model(model_path, model)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        **asdict(fit_receipt),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "validation_prediction_hash": numeric_hash(scores),
        "validation_rows": int(len(validation)),
        "elapsed_seconds": elapsed,
        **dict(extra or {}),
    }
    write_json(receipt_path, receipt)
    return scores, receipt


def _deterministic_position(group: pd.DataFrame) -> int:
    side = str(group["vwap_side"].iloc[0])
    eligible = group[group["right"].astype(str) == side]
    if eligible.empty:
        raise RuntimeError("deterministic P5 side absent from risk set")
    selected_index = min(
        eligible.index,
        key=lambda index: (
            abs(float(group.loc[index, "offset"])),
            float(group.loc[index, "offset"]),
            0 if str(group.loc[index, "right"]) == "C" else 1,
            str(group.loc[index, "contract_id"]),
        ),
    )
    return int(np.flatnonzero(group.index.to_numpy() == selected_index)[0])


def _selected_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["session", "decision_time_ns"]
    rows = []
    for identity, group in frame.groupby(keys, sort=False):
        local = group.reset_index(drop=True)
        deterministic = _deterministic_position(local)
        constant = int(candidate_order(local)[0])
        row = {
            "evidence_grade": EVIDENCE_GRADE,
            "session": str(identity[0]),
            "decision_time_ns": int(identity[1]),
            "fold": "expanding_fold_01_stage0",
            "deterministic_utility": float(
                local.iloc[deterministic]["primary_utility"]
            ),
            "exact_random_utility": float(local["primary_utility"].mean()),
            "constant_utility": float(local.iloc[constant]["primary_utility"]),
            "candidate_count": int(len(local)),
        }
        for model_name in ("M0", "M1"):
            scores = local[f"{model_name}_score"].to_numpy(dtype=float)
            selected = select_position(local, scores)
            reversed_position = select_position(local, -scores)
            row[f"{model_name}_utility"] = float(
                local.iloc[selected]["primary_utility"]
            )
            row[f"{model_name}_reversed_utility"] = float(
                local.iloc[reversed_position]["primary_utility"]
            )
            row[f"{model_name}_contract_id"] = str(
                local.iloc[selected]["contract_id"]
            )
            for shuffle_seed in PILOT_SHUFFLE_SEEDS:
                shuffled_scores = local[
                    f"{model_name}_shuffle{shuffle_seed}_score"
                ].to_numpy(dtype=float)
                position = select_position(local, shuffled_scores)
                row[
                    f"{model_name}_shuffle{shuffle_seed}_utility"
                ] = float(local.iloc[position]["primary_utility"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _session_mean_effect(
    selected: pd.DataFrame,
    column: str,
    baseline: str,
) -> dict[str, Any]:
    selected = selected.copy()
    selected["effect"] = (
        selected[column].to_numpy(dtype=float)
        - selected[baseline].to_numpy(dtype=float)
    )
    per_session = selected.groupby("session")["effect"].mean()
    return {
        "mean_session_effect": float(per_session.mean()),
        "pooled_opportunity_effect": float(selected["effect"].mean()),
        "per_session": {
            str(session): float(value)
            for session, value in per_session.items()
        },
        "positive_session_count": int((per_session > 0.0).sum()),
        "session_count": int(len(per_session)),
    }


def train_and_evaluate(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = validation.copy()
    model_receipts = []
    training_started = time.perf_counter()
    for model_name, features in (("M0", M0_FEATURES), ("M1", M1_FEATURES)):
        scores, receipt = _fit_and_score(
            training,
            validation,
            model_name=model_name,
            features=features,
            target_override=None,
            namespace="real",
        )
        predictions[f"{model_name}_score"] = scores
        model_receipts.append(receipt)
    permutation_receipts = []
    for shuffle_seed in PILOT_SHUFFLE_SEEDS:
        shuffled, permutation = permute_targets_within_decision(
            training.reset_index(drop=True),
            seed=shuffle_seed,
        )
        permutation["evidence_grade"] = EVIDENCE_GRADE
        permutation_path = (
            MODEL_ROOT / f"shuffle{shuffle_seed}" / "permutation_receipt.json"
        )
        write_json(permutation_path, permutation)
        permutation["path"] = str(permutation_path)
        permutation["sha256"] = sha256_path(permutation_path)
        permutation_receipts.append(permutation)
        for model_name, features in (("M0", M0_FEATURES), ("M1", M1_FEATURES)):
            scores, receipt = _fit_and_score(
                training,
                validation,
                model_name=model_name,
                features=features,
                target_override=shuffled,
                namespace=f"shuffle{shuffle_seed}",
                extra={
                    "shuffle_seed": shuffle_seed,
                    "permutation_sha256": permutation["sha256"],
                },
            )
            predictions[
                f"{model_name}_shuffle{shuffle_seed}_score"
            ] = scores
            model_receipts.append(receipt)
    training_elapsed = time.perf_counter() - training_started
    prediction_path = OUTPUT_ROOT / "predictions.parquet"
    temporary = prediction_path.with_name(
        f".{prediction_path.name}.tmp-{os.getpid()}"
    )
    predictions.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, prediction_path)
    selected = _selected_opportunities(predictions)
    selected_path = OUTPUT_ROOT / "ledger_a_selected_opportunities.parquet"
    temporary = selected_path.with_name(
        f".{selected_path.name}.tmp-{os.getpid()}"
    )
    selected.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, selected_path)

    selectors = [
        "M0",
        "M1",
        "M0_reversed",
        "M1_reversed",
        *[
            f"{model}_shuffle{seed}"
            for seed in PILOT_SHUFFLE_SEEDS
            for model in ("M0", "M1")
        ],
    ]
    effects = {}
    for selector in selectors:
        column = f"{selector}_utility"
        effects[selector] = {
            "versus_deterministic": _session_mean_effect(
                selected,
                column,
                "deterministic_utility",
            ),
            "versus_exact_random": _session_mean_effect(
                selected,
                column,
                "exact_random_utility",
            ),
        }
        effects[selector]["minimum_baseline_lift"] = min(
            effects[selector]["versus_deterministic"]["mean_session_effect"],
            effects[selector]["versus_exact_random"]["mean_session_effect"],
        )
    baseline = {
        "deterministic_mean_utility": float(
            selected.groupby("session")["deterministic_utility"].mean().mean()
        ),
        "exact_random_mean_utility": float(
            selected.groupby("session")["exact_random_utility"].mean().mean()
        ),
        "constant_mean_utility": float(
            selected.groupby("session")["constant_utility"].mean().mean()
        ),
    }
    constant_effects = {
        "versus_deterministic": _session_mean_effect(
            selected,
            "constant_utility",
            "deterministic_utility",
        ),
        "versus_exact_random": _session_mean_effect(
            selected,
            "constant_utility",
            "exact_random_utility",
        ),
    }
    constant_effects["minimum_baseline_lift"] = min(
        constant_effects["versus_deterministic"]["mean_session_effect"],
        constant_effects["versus_exact_random"]["mean_session_effect"],
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "ledger": "Ledger_A_decision_local_only",
        "validation_sessions": int(selected["session"].nunique()),
        "validation_opportunities": int(len(selected)),
        "baselines": baseline,
        "selector_effects": effects,
        "no_inference_or_multiplicity_claim": True,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_path(prediction_path),
        "selected_path": str(selected_path),
        "selected_sha256": sha256_path(selected_path),
    }
    write_json(OUTPUT_ROOT / "ledger_a_results.json", results)
    write_json(
        OUTPUT_ROOT / "selector_results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "ledger": "Ledger_A_decision_local_only",
            "validation_sessions": int(selected["session"].nunique()),
            "validation_opportunities": int(len(selected)),
            "candidate_rows": int(len(predictions)),
            "real_selectors": {
                name: effects[name] for name in ("M0", "M1")
            },
            "performance_interpreted": True,
        },
    )
    control_names = [
        name for name in selectors if name not in {"M0", "M1"}
    ]
    write_json(
        OUTPUT_ROOT / "control_results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "baselines": baseline,
            "constant_score_is_feature_independent": True,
            "exact_random_is_analytic_risk_set_mean": True,
            "constant_score": constant_effects,
            "controls": {
                name: effects[name] for name in control_names
            },
        },
    )
    receipts = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "real_models": 2,
        "shuffle_models": 4,
        "model_receipts": model_receipts,
        "permutation_receipts": permutation_receipts,
        "training_elapsed_seconds": training_elapsed,
    }
    write_json(OUTPUT_ROOT / "training_receipts.json", receipts)
    return results, receipts


def resource_projection(
    risk_receipt: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_bytes = sum(
        path.stat().st_size
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    )
    model_files = list(MODEL_ROOT.rglob("model.pkl"))
    mean_model_bytes = (
        float(np.mean([path.stat().st_size for path in model_files]))
        if model_files
        else 0.0
    )
    pilot_models = 6
    full_models = 2 * 3 * 5 + 2 * 3 * 20 * 5
    clean_build = risk_receipt.get("clean_source_build_elapsed_seconds")
    build_seconds = float(
        clean_build
        if clean_build is not None
        else risk_receipt["build_elapsed_seconds"]
    )
    training_seconds = float(training_receipt["training_elapsed_seconds"])
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports ru_maxrss in bytes.
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "pilot": {
            "sessions": TOTAL_SESSION_COUNT,
            "models": pilot_models,
            "risk_set_build_seconds": build_seconds,
            "risk_set_build_mode": str(risk_receipt["build_mode"]),
            "training_seconds": training_seconds,
            "artifact_bytes": artifact_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "full_campaign_projection": {
            "scope": "core_risk_set_build_model_fit_and_scoring_only",
            "sessions": 271,
            "models": full_models,
            "risk_set_build_seconds_linear": (
                build_seconds / TOTAL_SESSION_COUNT * 271
            ),
            "model_training_seconds_linear": (
                training_seconds / pilot_models * full_models
            ),
            "model_artifact_bytes_linear": mean_model_bytes * full_models,
            "risk_set_bytes_linear": (
                artifact_bytes / TOTAL_SESSION_COUNT * 271
            ),
            "expected_peak_rss_bytes_single_process": peak_rss_bytes,
            "projection_is_directional": True,
            "not_total_end_to_end_campaign_runtime": True,
            "excluded_later_costs": [
                "remaining_18_strong_shuffles",
                "20k_session_block_inference",
                "campaign_multiplicity",
                "simulator_v5_serial_replay",
                "independent_final_verification",
            ],
        },
    }
    write_json(OUTPUT_ROOT / "resource_projection.json", result)
    return result


def decide(
    machinery: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    contract_audit: Mapping[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    effects = results["selector_effects"]
    real = {name: float(effects[name]["minimum_baseline_lift"]) for name in ("M0", "M1")}
    controls = {
        name: float(payload["minimum_baseline_lift"])
        for name, payload in effects.items()
        if name not in {"M0", "M1"}
    }
    best_model = max(real, key=real.get)
    best_real = real[best_model]
    best_control_name = max(controls, key=controls.get)
    best_control = controls[best_control_name]
    enough_data = (
        int(results["validation_sessions"]) >= MINIMUM_VALIDATION_SESSIONS
        and int(results["validation_opportunities"])
        >= MINIMUM_VALIDATION_OPPORTUNITIES
    )
    contract_change_required = bool(
        contract_audit and contract_audit.get("status") == "scientific_contract_defect"
    )
    if not machinery["all_pass"]:
        route = "stop_mechanical_blocker"
        failure_class = "mechanical"
    elif contract_change_required:
        route = "stop_scientific_contract_defect"
        failure_class = "scientific_contract_defect"
    elif not enough_data:
        route = "stop_no_preliminary_signal"
        failure_class = "insufficient_data"
    elif best_real <= 0.0:
        route = "stop_no_preliminary_signal"
        failure_class = "no_signal"
    elif best_control >= best_real:
        route = "stop_no_preliminary_signal"
        failure_class = "control_comparable_to_real"
    else:
        route = "proceed_to_full_campaign"
        failure_class = None
    performance_interpreted = route not in {
        "stop_mechanical_blocker",
        "stop_scientific_contract_defect",
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "terminal_decision": route,
        "failure_class": failure_class,
        "performance_interpreted": performance_interpreted,
        "best_real_model": best_model if performance_interpreted else None,
        "best_real_minimum_baseline_lift": (
            best_real if performance_interpreted else None
        ),
        "best_control": best_control_name if performance_interpreted else None,
        "best_control_minimum_baseline_lift": (
            best_control if performance_interpreted else None
        ),
        "all_machinery_checks_pass": bool(machinery["all_pass"]),
        "enough_directional_data": enough_data,
        "scientific_contract_change_required": contract_change_required,
        "scientific_contract_defect": (
            contract_audit.get("defect")
            if contract_change_required and contract_audit
            else None
        ),
        "non_promotable": True,
        "full_campaign_contract_unchanged": True,
        "may_select_or_promote_model": False,
        "may_access_g9": False,
        "may_train_hold_exit": False,
    }
    if persist:
        write_json(OUTPUT_ROOT / "decision.json", decision)
    return decision


def report_text(
    decision: Mapping[str, Any],
    results: Mapping[str, Any],
    machinery: Mapping[str, Any],
    resources: Mapping[str, Any],
    contract_audit: Mapping[str, Any],
    opportunity_filter: Mapping[str, Any],
) -> str:
    effects = results["selector_effects"]
    projection = resources["full_campaign_projection"]
    return "\n".join(
        [
            "# Protocol101 Policy-Neutral Selector Stage-0",
            "",
            f"Evidence grade: `{EVIDENCE_GRADE}`.",
            "",
            f"Terminal decision: `{decision['terminal_decision']}`.",
            "",
            "This pilot is mechanical and directional only. It did not select, "
            "validate, promote, or authorize a model.",
            "",
            "## Frozen P5 Opportunities",
            "",
            (
                f"- P5 permitted `{opportunity_filter['p5_permitted_opportunities']}` "
                "of "
                f"`{opportunity_filter['source_opportunities']}` sampled "
                "decision opportunities."
            ),
            (
                f"- P5 abstained from "
                f"`{opportunity_filter['p5_abstention_opportunities']}` "
                "opportunities because no eligible contract matched its "
                "frozen VWAP side."
            ),
            "- Abstentions were excluded before target aggregation, fitting, "
            "scoring, and evaluation; no synthetic utility was assigned.",
            "",
            "## Directional Result",
            "",
            (
                "- Directional effects were not interpreted because the "
                "frozen deterministic comparator is undefined on part of the "
                "validation grid."
                if contract_audit["status"] == "scientific_contract_defect"
                else (
                    f"- M0 minimum lift over deterministic/random: "
                    f"`{effects['M0']['minimum_baseline_lift']:.6f}`."
                )
            ),
            *(
                []
                if contract_audit["status"] == "scientific_contract_defect"
                else [
                    (
                        f"- M1 minimum lift over deterministic/random: "
                        f"`{effects['M1']['minimum_baseline_lift']:.6f}`."
                    ),
                    (
                        "- M0 versus deterministic P5 / exact random: "
                        f"`{effects['M0']['versus_deterministic']['mean_session_effect']:.6f}` "
                        "/ "
                        f"`{effects['M0']['versus_exact_random']['mean_session_effect']:.6f}`."
                    ),
                    (
                        "- M1 versus deterministic P5 / exact random: "
                        f"`{effects['M1']['versus_deterministic']['mean_session_effect']:.6f}` "
                        "/ "
                        f"`{effects['M1']['versus_exact_random']['mean_session_effect']:.6f}`."
                    ),
                    (
                        f"- Best reversed/shuffle control: "
                        f"`{decision['best_control']}` at "
                        f"`{decision['best_control_minimum_baseline_lift']:.6f}`."
                    ),
                    (
                        "- M0 beat both deterministic P5 and exact random: "
                        f"`{effects['M0']['minimum_baseline_lift'] > 0.0}`."
                    ),
                    (
                        "- M1 beat both deterministic P5 and exact random: "
                        f"`{effects['M1']['minimum_baseline_lift'] > 0.0}`."
                    ),
                    (
                        "- A reversed or shuffled control was comparable to "
                        "the best real selector: "
                        f"`{decision['failure_class'] == 'control_comparable_to_real'}`."
                    ),
                ]
            ),
            (
                f"- Validation evidence: `{results['validation_sessions']}` "
                f"sessions and `{results['validation_opportunities']}` "
                "decision-local opportunities."
            ),
            "",
            "## Scientific Contract Audit",
            "",
            (
                f"- Deterministic baseline undefined on "
                f"`{contract_audit['deterministic_baseline_undefined_opportunities']}` "
                f"of `{contract_audit['validation_opportunities']}` opportunities "
                f"(`{contract_audit['undefined_share']:.2%}`)."
            ),
            (
                f"- Finding: {contract_audit['defect']}"
                if contract_audit["defect"]
                else "- No opportunity/comparator contradiction found."
            ),
            (
                f"- Smallest repair: {contract_audit['smallest_owner_decision']}"
                if contract_audit["smallest_owner_decision"]
                else "- No contract repair required."
            ),
            f"- Route rationale: `{decision['failure_class'] or 'all_proceed_rules_passed'}`.",
            "",
            "## Machinery",
            "",
            *[
                f"- {name}: `{'pass' if value else 'fail'}`"
                for name, value in machinery["checks"].items()
            ],
            "",
            "## Full-Campaign Projection",
            "",
            "- These are core build/fit/scoring projections, not total "
            "end-to-end campaign runtime.",
            (
                f"- Risk-set build: approximately "
                f"`{projection['risk_set_build_seconds_linear'] / 3600:.2f}` hours."
            ),
            (
                f"- Model training: approximately "
                f"`{projection['model_training_seconds_linear'] / 3600:.2f}` hours."
            ),
            (
                f"- Model artifacts: approximately "
                f"`{projection['model_artifact_bytes_linear'] / 2**30:.2f}` GiB."
            ),
            (
                f"- Observed peak process memory: "
                f"`{projection['expected_peak_rss_bytes_single_process'] / 2**30:.2f}` GiB."
            ),
            "",
            "## Boundaries",
            "",
            "- No final OOF session was used.",
            "- No full five-fold campaign was run.",
            "- No simulator replay, G9, protected holdout, HOLD/EXIT training, "
            "broker, paper, paid-data, promotion, runtime, or launchd action occurred.",
            "- Pilot results cannot change the frozen full-campaign contract.",
            "",
        ]
    )


def write_hashes() -> None:
    path = OUTPUT_ROOT / "hashes.sha256"
    files = sorted(
        item
        for item in OUTPUT_ROOT.rglob("*")
        if item.is_file()
        and item != path
        and "model.pkl" not in item.name
    )
    path.write_text(
        "\n".join(
            f"{sha256_path(item)}  {item.relative_to(OUTPUT_ROOT)}"
            for item in files
        )
        + "\n",
        encoding="utf-8",
    )


def run(*, force_risk_sets: bool = False) -> dict[str, Any]:
    preregister()
    verify_preregistration()
    started = time.perf_counter()
    training, validation, risk_receipt = build_risk_sets(
        force_rebuild=force_risk_sets,
    )
    opportunity_filter = read_json(
        OUTPUT_ROOT / "opportunity_filter_receipt.json"
    )
    combined = pd.concat([training, validation], ignore_index=True)
    target = manual_target_verification(combined)
    machinery = machinery_checks(training, validation, risk_receipt, target)
    contract_audit = scientific_contract_audit(validation)
    if (
        not machinery["all_pass"]
        or contract_audit["status"] == "scientific_contract_defect"
    ):
        results = {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "validation_sessions": int(validation["session"].nunique()),
            "validation_opportunities": int(
                validation[["session", "decision_time_ns"]]
                .drop_duplicates()
                .shape[0]
            ),
            "selector_effects": {
                name: {"minimum_baseline_lift": float("-inf")}
                for name in (
                    "M0",
                    "M1",
                    "M0_reversed",
                    "M1_reversed",
                    "M0_shuffle8600",
                    "M1_shuffle8600",
                    "M0_shuffle8601",
                    "M1_shuffle8601",
                )
            },
        }
        existing_receipts = [
            read_json(path)
            for path in MODEL_ROOT.rglob("receipt.json")
            if path.is_file()
        ]
        training_receipt = {
            "training_elapsed_seconds": float(
                sum(float(item.get("elapsed_seconds", 0.0)) for item in existing_receipts)
            ),
            "existing_exploratory_models": len(existing_receipts),
            "performance_not_interpreted": True,
        }
    else:
        results, training_receipt = train_and_evaluate(training, validation)
    resources = resource_projection(risk_receipt, training_receipt)
    decision = decide(
        machinery,
        results,
        contract_audit=contract_audit,
    )
    write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_grade": EVIDENCE_GRADE,
            "status": "stage0_complete",
            "terminal_decision": decision["terminal_decision"],
            "elapsed_seconds": time.perf_counter() - started,
            "side_effects": {
                "full_campaign_training": False,
                "simulator_replay": False,
                "g9": False,
                "protected_holdout": False,
                "hold_exit_training": False,
                "broker": False,
                "paper_submit": False,
                "paid_download": False,
                "promotion_or_default_change": False,
                "runtime_or_launchd_change": False,
            },
        },
    )
    (OUTPUT_ROOT / "report.md").write_text(
        report_text(
            decision,
            results,
            machinery,
            resources,
            contract_audit,
            opportunity_filter,
        ),
        encoding="utf-8",
    )
    _write_progress(
        "stage0_complete",
        terminal_decision=decision["terminal_decision"],
    )
    write_hashes()
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preregister", "run"),
        default="run",
    )
    parser.add_argument(
        "--force-risk-sets",
        action="store_true",
        help="Rebuild all sampled risk sets from governed source artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "preregister":
        preregister()
        print(
            json.dumps(
                {
                    "status": "preregistered",
                    "evidence_grade": EVIDENCE_GRADE,
                },
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            run(force_risk_sets=bool(args.force_risk_sets)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
