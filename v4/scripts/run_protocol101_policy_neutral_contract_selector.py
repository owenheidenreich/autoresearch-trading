"""Run the preregistered Protocol101 policy-neutral selector campaign.

The runner is resumable by phase.  It never changes the frozen P5 opportunity
engine, simulator-v5 economics, protected evidence, runtime, or broker state.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import (
    NeuralDatasetConfig,
    _contract_quote_path,
    _prepare_options,
)
from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_policy_neutral_selector import (
    BARRIER_COLUMNS,
    BREAKEVEN_COLUMNS,
    M0_FEATURES,
    M1_FEATURES,
    MODEL_CONFIG,
    MODEL_SEEDS,
    PATH_COLUMNS,
    RAW_TARGET_COLUMNS,
    RETURN_COLUMNS,
    SCHEMA_VERSION,
    SHUFFLE_SEEDS,
    add_primary_utility,
    block_schedule,
    candidate_order,
    deterministic_group_sample,
    fit_selector,
    numeric_hash,
    path_target_record,
    permute_targets_within_decision,
    score_selector,
    select_position,
    simultaneous_inference,
    stable_hash,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    RepairedCanonicalDecision,
    load_repaired_decisions,
)
from v4.model.protocol101_serial_simulator_v5 import SerialReplayTradeV5
from v4.model import protocol101_stage1_reference_multiplicity as references
from v4.scripts import materialize_protocol101_ft1d_two_clock_rows as materializer
from v4.scripts.run_protocol101_scoped_stage1_hgb_runner import (
    NOISE_DISTRIBUTION,
    guard_margins,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_attempt001"
)
INDEPENDENT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_policy_neutral_contract_selector_independent_audit_attempt001"
)
WORK_ROOT = OUTPUT_ROOT / "work"
SESSION_ROOT = WORK_ROOT / "session_risk_sets"
MODEL_ROOT = OUTPUT_ROOT / "model_receipts"
RANDOM_ROOT = OUTPUT_ROOT / "random_control_receipts"
SHUFFLE_ROOT = OUTPUT_ROOT / "shuffle_receipts"
GOAL_PATH = Path(
    "/Users/gduby/.codex/attachments/"
    "c021fa0d-7a55-4735-9d4a-f70d4a97690f/pasted-text-1.txt"
)
ENTRY_AUDIT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_entry_objective_audit_attempt001"
)
D1_CORRECTED_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_shuffled_profit_causal_attribution_"
    "inference_correction_attempt011"
)
CAMPAIGN_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
CAMPAIGN_ID = "protocol101-policy-neutral-contract-selector-attempt001"
POLICY_INDEX = 5
MAX_TRAIN_CANDIDATES = 350_000
RANDOM_DRAWS = 1_000
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BLOCK_SIZE = 5
BOOTSTRAP_SEED = 2_026_072_801
MINIMUM_EFFECT = 0.01
D1_MARGIN = 0.0025
MATERIALIZED_RECEIPT = materializer.GLOBAL_RECEIPT
GOVERNED_MANIFEST = materializer.MANIFEST_PATH
PREREGISTRATION_FILES = (
    "preregistration.json",
    "target_definition.json",
    "model_contract.json",
    "selector_d1_contract.json",
    "statistical_contract.json",
)
REQUIRED_PRODUCER_FILES = (
    *PREREGISTRATION_FILES,
    "preregistration.sha256",
    "decision_risk_sets.parquet",
    "risk_set_receipts.json",
    "training_manifest.json",
    "oof_predictions.parquet",
    "oof_ranking_results.json",
    "paired_predictive_results.json",
    "paired_economic_results.json",
    "control_results.json",
    "multiplicity_results.json",
    "contract_selector_decision.json",
    "report.md",
    "progress.json",
    "hashes.sha256",
)
FROZEN_INPUTS = {
    "goal": GOAL_PATH,
    "entry_audit_report": ENTRY_AUDIT_ROOT / "report.md",
    "entry_route": ENTRY_AUDIT_ROOT / "entry_campaign_route_decision.json",
    "recommended_entry_objective": (
        ENTRY_AUDIT_ROOT / "recommended_entry_objective.json"
    ),
    "score_path_quality": ENTRY_AUDIT_ROOT / "score_path_quality_results.json",
    "paired_policy_model": ENTRY_AUDIT_ROOT / "paired_policy_model_results.json",
    "d1_corrected_report": D1_CORRECTED_ROOT / "report.md",
    "d1_corrected_attribution": (
        D1_CORRECTED_ROOT / "causal_attribution_corrected.json"
    ),
    "d1_corrected_trust": (
        D1_CORRECTED_ROOT / "entry_model_trust_decision_corrected.json"
    ),
    "reference_multiplicity_source": (
        ROOT / "v4/model/protocol101_stage1_reference_multiplicity.py"
    ),
    "simulator_v5_source": (
        ROOT / "v4/model/protocol101_serial_simulator_v5.py"
    ),
    "canonical_contract_source": (
        ROOT / "v4/model/protocol101_canonical_stage1_contract.py"
    ),
    "materialized_receipt": MATERIALIZED_RECEIPT,
    "governed_manifest": GOVERNED_MANIFEST,
    "noise_distribution": (
        Path(NOISE_DISTRIBUTION)
        if Path(NOISE_DISTRIBUTION).is_absolute()
        else ROOT / Path(NOISE_DISTRIBUTION)
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_progress(status: str, **fields: Any) -> None:
    write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            **fields,
        },
    )


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _required_input_hashes() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in FROZEN_INPUTS.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"frozen inputs missing: {missing}")
    return {
        name: {"path": _relative(path), "sha256": sha256_path(path)}
        for name, path in FROZEN_INPUTS.items()
    }


def _side_effects(*, training: bool = False) -> dict[str, bool]:
    return {
        "offline_selector_training_executed": bool(training),
        "entry_timing_training_executed": False,
        "hold_exit_training_executed": False,
        "g9_accessed": False,
        "protected_holdout_accessed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "real_order_submit_allowed": False,
        "paid_data_downloaded": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "recorder_changed": False,
        "live_state_changed": False,
        "simulator_v5_economics_changed": False,
    }


def preregistration_payloads() -> dict[str, dict[str, Any]]:
    input_hashes = _required_input_hashes()
    common = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "evidence_grade": "design_grade_reused_sessions",
        "policy_index": POLICY_INDEX,
        "input_hashes": input_hashes,
        "highest_allowed_claim": (
            "Design-grade policy-neutral contract-selector candidate "
            "eligible for separate G9 authorization."
        ),
        "hard_boundaries": [
            "do_not_select_or_promote_existing_420_models",
            "do_not_resume_old_score_threshold_campaign",
            "do_not_train_entry_timing",
            "do_not_train_hold_exit",
            "do_not_access_g9",
            "do_not_access_protected_holdout",
            "do_not_contact_broker",
            "do_not_submit_orders",
            "do_not_download_paid_data",
            "do_not_modify_runtime_or_promotion_state",
            "do_not_treat_candidate_rows_as_iid_evidence",
        ],
    }
    target = {
        **common,
        "target_name": "policy_neutral_contract_quality_utility_v1",
        "entry": "executable_ask_at_decision",
        "exit": "latest_executable_bid_at_or_before_horizon",
        "horizons_minutes": [1, 2, 5, 10, 15],
        "maximum_quote_age_seconds": 90,
        "future_interpolation": False,
        "round_trip_fee_dollars": 3.0,
        "path_metrics": ["mfe_15m", "mae_15m"],
        "time_to_breakeven": {
            "quality": "negative_minutes_to_first_fee_adjusted_breakeven",
            "not_reached_value": -16.0,
        },
        "barriers": [
            {"name": "+10_before_-10", "favorable": 0.10, "adverse": -0.10},
            {"name": "+25_before_-20", "favorable": 0.25, "adverse": -0.20},
            {"name": "+50_before_-35", "favorable": 0.50, "adverse": -0.35},
        ],
        "barrier_encoding": {
            "favorable_first": 1,
            "neither": 0,
            "adverse_first": -1,
            "no_bid_before_favorable": -1,
        },
        "within_decision_percentile": {
            "method": "deterministic_midrank",
            "formula": "(midrank-1)/(finite_candidate_count-1)",
            "single_finite_candidate": 0.5,
            "missing_component_score": 0.0,
        },
        "families": {
            "return_family": {
                "weight": 0.25,
                "components": list(RETURN_COLUMNS),
            },
            "path_family": {
                "weight": 0.25,
                "components": list(PATH_COLUMNS),
            },
            "breakeven_family": {
                "weight": 0.25,
                "components": list(BREAKEVEN_COLUMNS),
            },
            "barrier_family": {
                "weight": 0.25,
                "components": list(BARRIER_COLUMNS),
            },
        },
        "primary_utility": (
            "0.25*(return_family+path_family+breakeven_family+barrier_family)"
        ),
        "utility_bounds": [0.0, 1.0],
    }
    model = {
        **common,
        "models": {
            "M0": {"features": list(M0_FEATURES)},
            "M1": {"features": list(M1_FEATURES)},
        },
        "model_family": "sklearn_HistGradientBoostingRegressor",
        "configuration": MODEL_CONFIG,
        "maximum_training_candidates_per_fold": MAX_TRAIN_CANDIDATES,
        "cap_sampling": (
            "complete decision groups sorted by sha256("
            "schema_version|fold_id|session|decision_time_ns)"
        ),
        "model_seeds": list(MODEL_SEEDS),
        "folds": "same governed five expanding chronological folds",
        "training_noise": "approved_1x_divergence_noise_law",
        "training_weight": (
            "1/(candidate_count_in_selected_decision*"
            "selected_decision_count_in_session), rescaled to mean 1"
        ),
        "primary_oof_score": "median_across_seeds_42_43_44",
        "selection": "highest_median_score",
        "tie_break": [
            "smallest_absolute_offset",
            "lowest_strike_index",
            "call_before_put",
            "lexicographically_smallest_contract_id",
        ],
        "no_hyperparameter_search": True,
        "no_score_threshold": True,
        "no_calibration_pnl_threshold_optimization": True,
        "neural_or_sequence_challenger_allowed": False,
    }
    selector_d1 = {
        **common,
        "contract": "Selector-D1-v1",
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "shuffle_unit": "within_each_training_decision_group",
        "shuffle_action": (
            "jointly_permute_complete_primary_utility_labels_among_"
            "candidate_identities"
        ),
        "preserved": [
            "target_multiset",
            "candidate_count",
            "features",
            "candidate_identities",
            "entry_asks",
            "sessions",
            "timestamps",
        ],
        "destroyed_relationships": [
            "right",
            "strike",
            "slot",
            "premium",
            "moneyness",
            "delta",
            "gamma",
            "contract_id",
        ],
        "validation_labels_shuffled": False,
        "model_seeds_per_shuffle": list(MODEL_SEEDS),
        "null_materiality_margin": D1_MARGIN,
        "pass_requirements": [
            "no_shuffled_model_passes_real_predictive_gate",
            "maximum_adjusted_upper_ci_shuffle_minus_random_le_0.0025",
            "real_selector_beats_synchronized_max_shuffle_with_"
            "adjusted_lower_ci_gt_0",
            "all_receipts_validate",
        ],
        "insufficient_precision_route": (
            "selector_d1_insufficient_precision_when_shuffle_effect_is_"
            "indistinguishable_from_zero_but_upper_ci_exceeds_margin"
        ),
        "old_three_dollar_equivalence_bound_applies": False,
    }
    statistics = {
        **common,
        "primary_unit": "session",
        "paired_effect": (
            "mean_session(mean_opportunity(U_selected-U_baseline))"
        ),
        "moving_block": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_size_sessions": BOOTSTRAP_BLOCK_SIZE,
            "prng": "NumPy PCG64DXSM",
            "master_seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "folds_resampled_separately": True,
            "circular_blocks": True,
            "same_indices_for_all_contrasts": True,
        },
        "multiplicity": {
            "method": "synchronized_max_abs_t_intervals_and_one_sided_maxT",
            "alpha_fwer": 0.05,
            "family": [
                "M0_vs_deterministic",
                "M0_vs_exact_random",
                "M1_vs_deterministic",
                "M1_vs_exact_random",
                "M1_vs_M0",
                "each_real_model_vs_synchronized_strong_shuffle_family",
            ],
        },
        "minimum_meaningful_predictive_effect": MINIMUM_EFFECT,
        "predictive_requirements": [
            "effect_ge_0.01_vs_deterministic_and_exact_random",
            "adjusted_lower_ci_gt_0_vs_both",
            "positive_every_oof_fold",
            "positive_each_model_seed",
            "selector_d1_passes",
            "reversed_worse",
            "single_session_contribution_le_20_percent",
            "not_confined_to_one_right_time_premium_or_moneyness_stratum",
            "identity_feature_risk_set_causality_checks_pass",
        ],
        "economic_route_a": [
            "positive_simulator_v5_increment_vs_deterministic",
            "adjusted_session_block_lower_ci_gt_0",
            "no_worse_maximum_drawdown",
            "no_affordability_identity_or_safety_failure",
        ],
    }
    preregistration = {
        **common,
        "objective": (
            "Given a frozen P5 opportunity, choose a better eligible SPXW "
            "contract than VWAP-side nearest ATM."
        ),
        "opportunity_stream": {
            "loader": "load_repaired_decisions",
            "identity": "ReferenceOpportunity",
            "validator": "assert_reference_opportunities",
            "policy": 5,
            "vwap_side": (
                "C when spx_vwap_gap_points>=0, otherwise P"
            ),
            "deterministic_baseline": (
                "byte-equivalent fixed_heuristic_candidates"
            ),
            "learned_risk_set": "all_eligible_calls_and_puts",
        },
        "ledgers": {
            "A": "decision_local_exact_selector_attribution",
            "B": "simulator_v5_serial_economic_diagnostic",
        },
        "target_definition_sha256": stable_hash(target),
        "model_contract_sha256": stable_hash(model),
        "selector_d1_contract_sha256": stable_hash(selector_d1),
        "statistical_contract_sha256": stable_hash(statistics),
        "terminal_routes": ["A", "B", "C", "D", "E"],
        "side_effects": _side_effects(),
    }
    return {
        "preregistration.json": preregistration,
        "target_definition.json": target,
        "model_contract.json": model,
        "selector_d1_contract.json": selector_d1,
        "statistical_contract.json": statistics,
    }


def preregister(*, force: bool) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = preregistration_payloads()
    existing = [OUTPUT_ROOT / name for name in PREREGISTRATION_FILES]
    if any(path.exists() for path in existing) and not all(
        path.is_file() for path in existing
    ):
        raise RuntimeError("partial preregistration exists")
    for name, payload in payloads.items():
        path = OUTPUT_ROOT / name
        if path.exists():
            if read_json(path) != _clean(payload):
                raise RuntimeError(f"existing preregistration changed: {name}")
        else:
            write_json(path, payload)
    lines = [
        f"{sha256_path(OUTPUT_ROOT / name)}  {name}"
        for name in PREREGISTRATION_FILES
    ]
    freeze = "\n".join(lines) + "\n"
    freeze_path = OUTPUT_ROOT / "preregistration.sha256"
    if freeze_path.exists() and freeze_path.read_text() != freeze:
        raise RuntimeError("preregistration freeze changed")
    freeze_path.write_text(freeze, encoding="utf-8")
    write_progress(
        "preregistered",
        preregistration_hashes={
            name: sha256_path(OUTPUT_ROOT / name)
            for name in PREREGISTRATION_FILES
        },
        result_inspection_performed=False,
    )


def verify_preregistration() -> dict[str, str]:
    if not (OUTPUT_ROOT / "preregistration.sha256").is_file():
        raise RuntimeError("preregistration freeze is missing")
    expected: dict[str, str] = {}
    for line in (OUTPUT_ROOT / "preregistration.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = OUTPUT_ROOT / name
        if not path.is_file() or sha256_path(path) != digest:
            raise RuntimeError(f"preregistration hash mismatch: {name}")
        expected[name] = digest
    if tuple(expected) != PREREGISTRATION_FILES:
        raise RuntimeError("preregistration file set changed")
    frozen = read_json(OUTPUT_ROOT / "preregistration.json")["input_hashes"]
    current = _required_input_hashes()
    if frozen != current:
        raise RuntimeError("frozen input hashes changed after preregistration")
    return expected


def _scope_maps() -> tuple[Any, dict[str, Path], dict[str, Path]]:
    scope = materializer.load_training_scope()
    receipt = read_json(MATERIALIZED_RECEIPT)
    processed = {
        str(item["session"]): ROOT / str(item["output_path"])
        for item in receipt["sessions"]
    }
    manifest = read_json(GOVERNED_MANIFEST)
    normalized = {
        str(item["session"]): (
            ROOT / str(item["normalized_official_context_file"])
        )
        for item in manifest["included_sessions"]
    }
    used_sessions = {
        str(session)
        for fold in scope.folds
        for key in ("train_sessions", "validation_sessions")
        for session in fold[key]
    }
    missing = sorted(
        (used_sessions - set(processed)) | (used_sessions - set(normalized))
    )
    if missing:
        raise RuntimeError(f"campaign path map incomplete: {missing[:5]}")
    return scope, processed, normalized


def _fold_for_validation_session(scope: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for fold in scope.folds:
        fold_id = str(fold["fold_id"])
        for session in fold["validation_sessions"]:
            if str(session) in out:
                raise RuntimeError("validation session appears in multiple folds")
            out[str(session)] = fold_id
    if len(out) != 225:
        raise RuntimeError(f"expected 225 validation sessions, got {len(out)}")
    return out


def _reference_opportunities(
    decisions: Sequence[RepairedCanonicalDecision],
    *,
    fold_id: str,
    split: str,
    campaign_id: str = CAMPAIGN_ID,
) -> tuple[references.ReferenceOpportunity, ...]:
    out = []
    for ordinal, repaired in enumerate(decisions):
        gap = float(repaired.base.features[0, 0])
        out.append(
            references.ReferenceOpportunity(
                campaign_id=str(campaign_id),
                fold=str(fold_id),
                split=str(split),
                policy_index=POLICY_INDEX,
                repaired=repaired,
                vwap_side="C" if gap >= 0.0 else "P",
                decision_ordinal=int(ordinal),
            )
        )
    return references.assert_reference_opportunities(out)


def _risk_set_hash(
    opportunity: references.ReferenceOpportunity,
) -> str:
    base = opportunity.repaired.base
    payload = {
        "opportunity_identity": list(opportunity.identity),
        "contracts": [
            {
                "contract_id": str(base.contract_ids[index]),
                "right": str(base.rights[index]),
                "offset": float(base.offsets[index]),
                "strike_index": int(base.strike_indices[index]),
                "right_index": int(base.right_indices[index]),
                "entry_ask": float(base.entry_asks[index]),
            }
            for index in range(len(base.contract_ids))
        ],
    }
    return stable_hash(payload)


def _session_frame(
    *,
    session: str,
    processed_path: Path,
    normalized_path: Path,
    fold_id: str,
    split: str,
    campaign_id: str = CAMPAIGN_ID,
    require_fixed_p5_opportunity: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    margins = guard_margins()
    h0 = load_repaired_decisions(
        [(session, processed_path)],
        hypothesis="H0",
        policy_index=POLICY_INDEX,
        guard_margins=margins,
        split=f"selector-reference:{split}:{fold_id}",
    )
    h3 = load_repaired_decisions(
        [(session, processed_path)],
        hypothesis="H3",
        policy_index=POLICY_INDEX,
        guard_margins=margins,
        split=f"selector-features:{split}:{fold_id}",
    )
    if len(h0) != len(h3):
        raise RuntimeError(f"H0/H3 decision mismatch: {session}")
    for left, right in zip(h0, h3):
        left_ids = (
            int(left.base.decision_time.value),
            tuple(map(str, left.base.contract_ids)),
            tuple(map(int, left.base.strike_indices)),
            tuple(map(int, left.base.right_indices)),
        )
        right_ids = (
            int(right.base.decision_time.value),
            tuple(map(str, right.base.contract_ids)),
            tuple(map(int, right.base.strike_indices)),
            tuple(map(int, right.base.right_indices)),
        )
        if left_ids != right_ids:
            raise RuntimeError(f"H0/H3 candidate identity mismatch: {session}")
    opportunities = _reference_opportunities(
        h0,
        fold_id=fold_id,
        split=split,
        campaign_id=campaign_id,
    )
    source_opportunity_count = len(opportunities)
    fixed_candidates_by_decision: dict[int, Any] = {}
    if require_fixed_p5_opportunity:
        for candidate in references.fixed_heuristic_candidates(opportunities):
            decision_ns = int(candidate.decision_time_ns)
            if decision_ns in fixed_candidates_by_decision:
                raise RuntimeError(
                    f"multiple fixed P5 candidates at one decision: "
                    f"{session}:{decision_ns}"
                )
            fixed_candidates_by_decision[decision_ns] = candidate
        paired = [
            (opportunity, full)
            for opportunity, full in zip(opportunities, h3)
            if int(opportunity.repaired.base.decision_time.value)
            in fixed_candidates_by_decision
        ]
        opportunities = tuple(item[0] for item in paired)
        h3 = tuple(item[1] for item in paired)
        if not opportunities:
            raise RuntimeError(f"P5 emitted no session opportunities: {session}")
    table = pq.read_table(normalized_path)
    options = _prepare_options(table, NeuralDatasetConfig())
    paths = {
        str(contract_id): _contract_quote_path(
            group,
            enforce_unique_path=True,
        )
        for contract_id, group in options.groupby("contract_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    missing_paths = 0
    for opportunity, full in zip(opportunities, h3):
        base = opportunity.repaired.base
        risk_hash = _risk_set_hash(opportunity)
        deterministic_candidate = fixed_candidates_by_decision.get(
            int(base.decision_time.value)
        )
        count = len(base.contract_ids)
        for candidate_index in range(count):
            contract_id = str(base.contract_ids[candidate_index])
            path = paths.get(contract_id)
            if path is None:
                missing_paths += 1
                continue
            target = path_target_record(
                quote_ns=path.quote_ns,
                bids=path.bid,
                decision_ns=int(base.decision_time.value),
                entry_ask=float(base.entry_asks[candidate_index]),
            )
            features = np.asarray(
                full.base.features[candidate_index],
                dtype=np.float64,
            )
            record = {
                "campaign_id": str(campaign_id),
                "fold": str(fold_id),
                "split": str(split),
                "session": str(session),
                "decision_time_ns": int(base.decision_time.value),
                "decision_ordinal": int(opportunity.decision_ordinal),
                "policy_index": POLICY_INDEX,
                "vwap_side": str(opportunity.vwap_side),
                "candidate_index": int(candidate_index),
                "candidate_count": int(count),
                "contract_id": contract_id,
                "right": str(base.rights[candidate_index]),
                "offset": float(base.offsets[candidate_index]),
                "strike_index": int(base.strike_indices[candidate_index]),
                "right_index": int(base.right_indices[candidate_index]),
                "canonical_slot": int(
                    opportunity.repaired.canonical_strike_slots[candidate_index]
                ),
                "entry_ask": float(base.entry_asks[candidate_index]),
                "entry_label_net_pnl_p5": float(base.labels[candidate_index]),
                "label_realized_exit_time_ns_p5": int(
                    opportunity.repaired.realized_exit_time_ns[candidate_index]
                ),
                "risk_set_hash": risk_hash,
                "eligibility_reason": (
                    "signed_boundary_stable_guard_and_complete_p5_label"
                ),
                **(
                    {
                        "deterministic_p5_contract_id": str(
                            deterministic_candidate.contract_id
                        ),
                        "deterministic_p5_right": str(
                            deterministic_candidate.right
                        ),
                        "deterministic_p5_canonical_slot": int(
                            deterministic_candidate.canonical_strike_slot
                        ),
                    }
                    if deterministic_candidate is not None
                    else {}
                ),
                **{
                    name: float(features[index])
                    for index, name in enumerate(FEATURE_NAMES)
                },
                **target,
            }
            records.append(record)
    if missing_paths:
        raise RuntimeError(f"missing normalized contract paths: {session}:{missing_paths}")
    frame = add_primary_utility(pd.DataFrame.from_records(records))
    identity = [
        "session",
        "decision_time_ns",
        "contract_id",
        "canonical_slot",
        "right_index",
    ]
    if frame.duplicated(identity).any():
        raise RuntimeError(f"duplicate selector candidate identity: {session}")
    if frame[["session", "decision_time_ns"]].drop_duplicates().shape[0] != len(
        opportunities
    ):
        raise RuntimeError(f"selector opportunity loss: {session}")
    if require_fixed_p5_opportunity:
        deterministic_matches = (
            (
                frame["contract_id"].astype(str)
                == frame["deterministic_p5_contract_id"].astype(str)
            )
            & (
                frame["right"].astype(str)
                == frame["deterministic_p5_right"].astype(str)
            )
            & (
                frame["canonical_slot"].astype(int)
                == frame["deterministic_p5_canonical_slot"].astype(int)
            )
        )
        match_counts = deterministic_matches.groupby(
            [frame["session"], frame["decision_time_ns"]]
        ).sum()
        if not bool((match_counts == 1).all()):
            raise RuntimeError(
                f"fixed P5 candidate missing or duplicated in risk set: {session}"
            )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "session": str(session),
        "fold": str(fold_id),
        "split": str(split),
        "opportunities": int(len(opportunities)),
        "source_opportunities": int(source_opportunity_count),
        "p5_opportunity_filter_applied": bool(require_fixed_p5_opportunity),
        "p5_permitted_opportunities": int(len(opportunities)),
        "p5_abstention_opportunities": int(
            source_opportunity_count - len(opportunities)
        ),
        "p5_abstention_reasons": {
            "no_eligible_vwap_side_contract": int(
                source_opportunity_count - len(opportunities)
            )
        },
        "deterministic_candidate_identity_hash": (
            stable_hash(
                [
                    {
                        "session": str(candidate.session),
                        "decision_time_ns": int(candidate.decision_time_ns),
                        "contract_id": str(candidate.contract_id),
                        "right": str(candidate.right),
                        "canonical_slot": int(
                            candidate.canonical_strike_slot
                        ),
                    }
                    for candidate in fixed_candidates_by_decision.values()
                ]
            )
            if require_fixed_p5_opportunity
            else None
        ),
        "candidate_rows": int(len(frame)),
        "minimum_candidates": int(frame["candidate_count"].min()),
        "maximum_candidates": int(frame["candidate_count"].max()),
        "risk_set_root_hash": stable_hash(
            frame[
                ["session", "decision_time_ns", "risk_set_hash"]
            ].drop_duplicates().to_dict("records")
        ),
        "candidate_identity_hash": stable_hash(
            frame[identity].to_dict("records")
        ),
        "target_hash": numeric_hash(
            frame["primary_utility"].to_numpy(dtype=np.float64)
        ),
        "feature_hash": numeric_hash(
            frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        ),
        "reference_opportunities_validated": True,
        "reference_feature_view": "finite_H0_context_features",
        "selector_feature_join": "identity_exact_H3_17_feature_tensor",
        "both_calls_and_puts_present": bool(
            set(frame["right"].astype(str)) == {"C", "P"}
        ),
    }
    return frame, receipt


def build_risk_sets(*, force: bool) -> dict[str, Any]:
    verify_preregistration()
    scope, processed, normalized = _scope_maps()
    validation_fold = _fold_for_validation_session(scope)
    all_sessions = sorted(
        {
            str(session)
            for fold in scope.folds
            for key in ("train_sessions", "validation_sessions")
            for session in fold[key]
        }
    )
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for index, session in enumerate(all_sessions, start=1):
        path = SESSION_ROOT / f"{session}.parquet"
        receipt_path = SESSION_ROOT / f"{session}.json"
        fold_id = validation_fold.get(session, "TRAIN_PREFIX")
        split = "validation" if session in validation_fold else "train_prefix"
        if path.is_file() and receipt_path.is_file() and not force:
            receipt = read_json(receipt_path)
            if receipt.get("parquet_sha256") == sha256_path(path):
                receipts.append(receipt)
                continue
        frame, receipt = _session_frame(
            session=session,
            processed_path=processed[session],
            normalized_path=normalized[session],
            fold_id=fold_id,
            split=split,
        )
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        receipt["parquet_path"] = _relative(path)
        receipt["parquet_sha256"] = sha256_path(path)
        write_json(receipt_path, receipt)
        receipts.append(receipt)
        write_progress(
            "building_risk_sets",
            completed_sessions=index,
            total_sessions=len(all_sessions),
            candidate_rows=sum(int(item["candidate_rows"]) for item in receipts),
        )

    validation_frames = []
    for session in sorted(validation_fold):
        frame = pd.read_parquet(SESSION_ROOT / f"{session}.parquet")
        frame["fold"] = validation_fold[session]
        frame["split"] = "validation"
        validation_frames.append(frame)
    oof = pd.concat(validation_frames, ignore_index=True)
    output = OUTPUT_ROOT / "decision_risk_sets.parquet"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    oof.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)
    risk_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "risk_sets_constructed_and_validated",
        "sessions_all_roles": len(all_sessions),
        "oof_validation_sessions": int(oof["session"].nunique()),
        "oof_opportunities": int(
            oof[["session", "decision_time_ns"]].drop_duplicates().shape[0]
        ),
        "oof_candidate_rows": int(len(oof)),
        "decision_risk_sets_sha256": sha256_path(output),
        "session_receipts": receipts,
        "reference_contract": {
            "loader": "load_repaired_decisions",
            "identity": "ReferenceOpportunity",
            "validator": "assert_reference_opportunities",
            "baseline": "fixed_heuristic_candidates",
        },
        "duplicate_or_missing_identity_count": 0,
    }
    write_json(OUTPUT_ROOT / "risk_set_receipts.json", risk_receipt)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "training_inputs_ready_pending_independent_machinery_validation",
        "campaign_id": CAMPAIGN_ID,
        "evidence_grade": "design_grade_reused_sessions",
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "folds": [
            {
                "fold_id": str(fold["fold_id"]),
                "train_sessions": list(map(str, fold["train_sessions"])),
                "validation_sessions": list(
                    map(str, fold["validation_sessions"])
                ),
                "embargo_sessions": sorted(
                    set(map(str, fold.get("test_sessions", [])))
                    - set(map(str, fold["validation_sessions"]))
                ),
            }
            for fold in scope.folds
        ],
        "all_session_files": {
            session: {
                "path": _relative(SESSION_ROOT / f"{session}.parquet"),
                "sha256": sha256_path(SESSION_ROOT / f"{session}.parquet"),
            }
            for session in all_sessions
        },
        "protected_holdout_accessed": False,
        "g9_accessed": False,
        "side_effects": _side_effects(),
    }
    write_json(OUTPUT_ROOT / "training_manifest.json", manifest)
    write_progress(
        "risk_sets_complete_pending_independent_machinery_validation",
        oof_sessions=risk_receipt["oof_validation_sessions"],
        oof_opportunities=risk_receipt["oof_opportunities"],
        oof_candidate_rows=risk_receipt["oof_candidate_rows"],
    )
    return risk_receipt


def _noise_model() -> DivergenceNoiseModel:
    path = FROZEN_INPUTS["noise_distribution"]
    return DivergenceNoiseModel.from_parquet(path)


def _load_sessions(sessions: Iterable[str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(SESSION_ROOT / f"{session}.parquet")
        for session in sessions
    ]
    if not frames:
        raise RuntimeError("no selector session frames requested")
    frame = pd.concat(frames, ignore_index=True)
    identity = ["session", "decision_time_ns", "contract_id"]
    if frame.duplicated(identity).any():
        raise RuntimeError("duplicate candidate identity across session frames")
    return frame


def _pickle_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _load_model_with_receipt(
    model_path: Path,
    receipt_path: Path,
) -> tuple[Any, dict[str, Any]]:
    receipt = read_json(receipt_path)
    if sha256_path(model_path) != receipt["model_sha256"]:
        raise RuntimeError(f"model receipt hash mismatch: {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle), receipt


def _fit_or_load_model(
    training: pd.DataFrame,
    *,
    model_name: str,
    feature_names: Sequence[str],
    seed: int,
    fold_id: str,
    noise_model: DivergenceNoiseModel,
    root: Path,
    target_override: np.ndarray | None = None,
    extra_receipt: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_path = root / "model.pkl"
    receipt_path = root / "receipt.json"
    if model_path.is_file() and receipt_path.is_file():
        return _load_model_with_receipt(model_path, receipt_path)
    if model_path.exists() or receipt_path.exists():
        raise RuntimeError(f"partial model checkpoint exists: {root}")
    model, fit_receipt = fit_selector(
        training,
        model_name=model_name,
        feature_names=feature_names,
        seed=seed,
        fold_id=fold_id,
        noise_model=noise_model,
        target_override=target_override,
    )
    model_hash = _pickle_atomic(model_path, model)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        **asdict(fit_receipt),
        "model_path": _relative(model_path),
        "model_sha256": model_hash,
        "noise_distribution_sha256": sha256_path(
            FROZEN_INPUTS["noise_distribution"]
        ),
        "training_noise_scale": 1.0,
        "validation_noise_scale": 1.0,
        "validation_noise_seed": int(seed) + 200_000,
        **dict(extra_receipt or {}),
    }
    write_json(receipt_path, receipt)
    return model, receipt


def _score_fold_real_models(
    *,
    fold: Mapping[str, Any],
    training: pd.DataFrame,
    validation: pd.DataFrame,
    noise_model: DivergenceNoiseModel,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fold_id = str(fold["fold_id"])
    out = validation.copy()
    receipts: list[dict[str, Any]] = []
    for model_name, feature_names in (
        ("M0", M0_FEATURES),
        ("M1", M1_FEATURES),
    ):
        score_columns = []
        for seed in MODEL_SEEDS:
            root = (
                MODEL_ROOT
                / model_name
                / f"seed{seed}"
                / fold_id
            )
            model, receipt = _fit_or_load_model(
                training,
                model_name=model_name,
                feature_names=feature_names,
                seed=seed,
                fold_id=fold_id,
                noise_model=noise_model,
                root=root,
            )
            scores = score_selector(
                model,
                validation,
                feature_names=feature_names,
                noise_model=noise_model,
                noise_seed=int(seed) + 200_000,
                noise_scale=1.0,
            )
            column = f"{model_name}_score_seed{seed}"
            out[column] = scores
            score_columns.append(column)
            receipt["validation_rows"] = int(len(validation))
            receipt["validation_prediction_hash"] = numeric_hash(scores)
            write_json(root / "receipt.json", receipt)
            receipts.append(receipt)
        out[f"{model_name}_score_median"] = out[score_columns].median(axis=1)
    return out, receipts


def _shuffle_receipt_path(shuffle_seed: int, fold_id: str) -> Path:
    return (
        SHUFFLE_ROOT
        / f"seed{shuffle_seed}"
        / fold_id
        / "permutation_receipt.json.gz"
    )


def _write_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(_clean(payload), handle, sort_keys=True)
    os.replace(temporary, path)


def _score_fold_shuffles(
    *,
    fold: Mapping[str, Any],
    training: pd.DataFrame,
    validation: pd.DataFrame,
    noise_model: DivergenceNoiseModel,
) -> list[dict[str, Any]]:
    fold_id = str(fold["fold_id"])
    fold_receipts: list[dict[str, Any]] = []
    group_keys = ["session", "decision_time_ns"]
    group_indexes = [
        np.asarray(list(indexes), dtype=np.int64)
        for indexes in validation.groupby(group_keys, sort=False).groups.values()
    ]
    for shuffle_index, shuffle_seed in enumerate(SHUFFLE_SEEDS, start=1):
        shuffled, permutation_receipt = permute_targets_within_decision(
            training.reset_index(drop=True),
            seed=int(shuffle_seed),
        )
        permutation_path = _shuffle_receipt_path(shuffle_seed, fold_id)
        if permutation_path.is_file():
            with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
                existing = json.load(handle)
            if (
                existing["source_target_hash"]
                != permutation_receipt["source_target_hash"]
                or existing["destination_target_hash"]
                != permutation_receipt["destination_target_hash"]
                or existing["group_receipt_root_hash"]
                != permutation_receipt["group_receipt_root_hash"]
            ):
                raise RuntimeError(
                    f"Selector-D1 permutation checkpoint changed: "
                    f"{shuffle_seed}:{fold_id}"
                )
        else:
            _write_gzip_json(permutation_path, permutation_receipt)
        seed_result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "shuffle_seed": int(shuffle_seed),
            "fold_id": fold_id,
            "permutation_receipt_path": _relative(permutation_path),
            "permutation_receipt_sha256": sha256_path(permutation_path),
            "models": [],
        }
        selected = validation[
            [
                "session",
                "decision_time_ns",
                "contract_id",
                "primary_utility",
            ]
        ].copy()
        for model_name, feature_names in (
            ("M0", M0_FEATURES),
            ("M1", M1_FEATURES),
        ):
            columns = []
            for model_seed in MODEL_SEEDS:
                root = (
                    SHUFFLE_ROOT
                    / f"seed{shuffle_seed}"
                    / fold_id
                    / model_name
                    / f"model_seed{model_seed}"
                )
                model, receipt = _fit_or_load_model(
                    training,
                    model_name=model_name,
                    feature_names=feature_names,
                    seed=model_seed,
                    fold_id=fold_id,
                    noise_model=noise_model,
                    root=root,
                    target_override=shuffled,
                    extra_receipt={
                        "selector_d1_version": "Selector-D1-v1",
                        "shuffle_seed": int(shuffle_seed),
                        "permutation_receipt_sha256": sha256_path(
                            permutation_path
                        ),
                    },
                )
                scores = score_selector(
                    model,
                    validation,
                    feature_names=feature_names,
                    noise_model=noise_model,
                    noise_seed=int(model_seed) + 200_000,
                    noise_scale=1.0,
                )
                column = f"_score_{model_name}_{model_seed}"
                selected[column] = scores
                columns.append(column)
                receipt["validation_rows"] = int(len(validation))
                receipt["validation_prediction_hash"] = numeric_hash(scores)
                write_json(root / "receipt.json", receipt)
                seed_result["models"].append(receipt)
            median = selected[columns].median(axis=1).to_numpy(dtype=float)
            picked_contract: list[str] = []
            picked_utility: list[float] = []
            picked_score: list[float] = []
            for indexes in group_indexes:
                local = validation.iloc[indexes]
                position = select_position(local, median[indexes])
                global_index = int(indexes[position])
                picked_contract.append(
                    str(validation.iloc[global_index]["contract_id"])
                )
                picked_utility.append(
                    float(validation.iloc[global_index]["primary_utility"])
                )
                picked_score.append(float(median[global_index]))
            opportunity = (
                validation[group_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if len(opportunity) != len(picked_contract):
                raise RuntimeError("Selector-D1 selected opportunity loss")
            opportunity[f"{model_name}_selected_contract_id"] = picked_contract
            opportunity[f"{model_name}_selected_utility"] = picked_utility
            opportunity[f"{model_name}_selected_score"] = picked_score
            if model_name == "M0":
                selected_opportunities = opportunity
            else:
                selected_opportunities = selected_opportunities.merge(
                    opportunity,
                    on=group_keys,
                    how="inner",
                    validate="one_to_one",
                )
            selected.drop(columns=columns, inplace=True)
        selected_path = (
            SHUFFLE_ROOT
            / f"seed{shuffle_seed}"
            / fold_id
            / "selected_oof.parquet"
        )
        temporary = selected_path.with_name(
            f".{selected_path.name}.tmp-{os.getpid()}"
        )
        selected_opportunities.to_parquet(
            temporary,
            index=False,
            compression="zstd",
        )
        os.replace(temporary, selected_path)
        seed_result["selected_oof_path"] = _relative(selected_path)
        seed_result["selected_oof_sha256"] = sha256_path(selected_path)
        seed_result["opportunities"] = int(len(selected_opportunities))
        seed_result["receipt_hash"] = stable_hash(seed_result)
        write_json(
            selected_path.with_name("shuffle_fold_receipt.json"),
            seed_result,
        )
        fold_receipts.append(seed_result)
        write_progress(
            "training_selector_d1",
            fold_id=fold_id,
            completed_shuffle_seeds=shuffle_index,
            total_shuffle_seeds=len(SHUFFLE_SEEDS),
            completed_models=shuffle_index * 2 * len(MODEL_SEEDS),
            total_models=len(SHUFFLE_SEEDS) * 2 * len(MODEL_SEEDS),
        )
    return fold_receipts


def train_models() -> dict[str, Any]:
    verify_preregistration()
    if not (OUTPUT_ROOT / "training_manifest.json").is_file():
        raise RuntimeError("risk-set machinery is not ready")
    machinery = (
        INDEPENDENT_ROOT / "independent_machinery_verification.json"
    )
    if not machinery.is_file():
        raise RuntimeError(
            "independent machinery validation must pass before training"
        )
    machinery_result = read_json(machinery)
    if machinery_result.get("status") != "independent_machinery_pass":
        raise RuntimeError("independent machinery validation did not pass")
    scope, _, _ = _scope_maps()
    noise_model = _noise_model()
    all_receipts: list[dict[str, Any]] = []
    shuffle_receipts: list[dict[str, Any]] = []
    fold_outputs: list[Path] = []
    for fold_index, fold in enumerate(scope.folds, start=1):
        fold_id = str(fold["fold_id"])
        training_full = _load_sessions(fold["train_sessions"])
        training = deterministic_group_sample(
            training_full,
            maximum_candidates=MAX_TRAIN_CANDIDATES,
            fold_id=fold_id,
        ).reset_index(drop=True)
        validation = _load_sessions(fold["validation_sessions"]).reset_index(
            drop=True
        )
        validation["fold"] = fold_id
        training_receipt = {
            "fold_id": fold_id,
            "training_sessions": int(training["session"].nunique()),
            "training_decisions_before_cap": int(
                training_full[
                    ["session", "decision_time_ns"]
                ].drop_duplicates().shape[0]
            ),
            "training_candidates_before_cap": int(len(training_full)),
            "training_decisions_after_cap": int(
                training[
                    ["session", "decision_time_ns"]
                ].drop_duplicates().shape[0]
            ),
            "training_candidates_after_cap": int(len(training)),
            "training_candidate_identity_hash": stable_hash(
                training[
                    ["session", "decision_time_ns", "contract_id"]
                ].to_dict("records")
            ),
            "validation_sessions": int(validation["session"].nunique()),
            "validation_candidates": int(len(validation)),
        }
        scored, receipts = _score_fold_real_models(
            fold=fold,
            training=training,
            validation=validation,
            noise_model=noise_model,
        )
        all_receipts.extend(receipts)
        fold_output = WORK_ROOT / f"oof_predictions_{fold_id}.parquet"
        fold_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = fold_output.with_name(
            f".{fold_output.name}.tmp-{os.getpid()}"
        )
        scored.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, fold_output)
        training_receipt["oof_prediction_path"] = _relative(fold_output)
        training_receipt["oof_prediction_sha256"] = sha256_path(fold_output)
        write_json(
            WORK_ROOT / f"training_receipt_{fold_id}.json",
            training_receipt,
        )
        fold_outputs.append(fold_output)
        write_progress(
            "real_selector_models_complete_training_d1",
            completed_folds=fold_index - 1,
            total_folds=len(scope.folds),
            active_fold=fold_id,
            real_models_completed=len(receipts),
        )
        shuffle_receipts.extend(
            _score_fold_shuffles(
                fold=fold,
                training=training,
                validation=validation,
                noise_model=noise_model,
            )
        )
        write_progress(
            "fold_training_complete",
            completed_folds=fold_index,
            total_folds=len(scope.folds),
            active_fold=fold_id,
        )

    oof = pd.concat(
        [pd.read_parquet(path) for path in fold_outputs],
        ignore_index=True,
    )
    oof_path = OUTPUT_ROOT / "oof_predictions.parquet"
    temporary = oof_path.with_name(f".{oof_path.name}.tmp-{os.getpid()}")
    oof.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, oof_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "training_complete_pending_evaluation",
        "real_model_count": len(all_receipts),
        "selector_d1_model_count": sum(
            len(item["models"]) for item in shuffle_receipts
        ),
        "oof_candidate_rows": int(len(oof)),
        "oof_prediction_sha256": sha256_path(oof_path),
        "real_model_receipt_root_hash": stable_hash(all_receipts),
        "shuffle_receipt_root_hash": stable_hash(shuffle_receipts),
        "side_effects": _side_effects(training=True),
    }
    write_json(OUTPUT_ROOT / "training_completion.json", summary)
    write_progress("training_complete_pending_evaluation", **summary)
    return summary
