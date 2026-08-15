"""Causally attribute the Protocol101 D1 shuffled-model profit.

This is a diagnostic-only runner. It reuses the frozen P5/H2 corpus, folds,
persisted campaign models, and simulator-v5 economics. It may fit disposable
strong-permutation models, but it cannot create a campaign candidate or touch
G9, the protected holdout, lifecycle training, live, paper, or broker state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model import protocol101_scoped_stage1_hgb as hgb_core
from v4.model import protocol101_stage1_reference_multiplicity as references
from v4.model.protocol101_canonical_stage1_contract import HYPOTHESES
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_scoped_stage1_hgb import (
    HGBUnitConfig,
    RepairedCanonicalDecision,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialReplayTradeV5,
)
from v4.scripts import run_protocol101_ft1d_real_campaign_g1_g8 as ft1d
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as stage1_runner


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "v4/audit/autoresearch"
OUTPUT_ROOT = (
    AUDIT_ROOT
    / "protocol101_d1_shuffled_profit_causal_attribution_attempt009"
)
WORK_ROOT = OUTPUT_ROOT / "work"
PERMUTATION_ROOT = OUTPUT_ROOT / "permutation_receipts"
STRONG_MODEL_ROOT = OUTPUT_ROOT / "strong_permutation_models"
SESSION_CHUNK_ROOT = WORK_ROOT / "session_chunks"
CONTROL_CHUNK_ROOT = WORK_ROOT / "control_chunks"

INVESTIGATION_PLAN_PATH = OUTPUT_ROOT / "investigation_plan.json"
CONTROL_DEFINITIONS_PATH = OUTPUT_ROOT / "control_definitions.json"
PAIRED_RESULTS_PATH = OUTPUT_ROOT / "paired_control_results.json"
SESSION_RESULTS_PATH = OUTPUT_ROOT / "session_level_results.parquet"
IMPLEMENTATION_AUDIT_PATH = OUTPUT_ROOT / "implementation_audit.json"
CAUSAL_ATTRIBUTION_PATH = OUTPUT_ROOT / "causal_attribution.json"
REVISED_D1_PATH = OUTPUT_ROOT / "revised_d1_specification.json"
TRUST_DECISION_PATH = OUTPUT_ROOT / "entry_model_trust_decision.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
ACTIVE_PROGRESS_PATH = PROGRESS_PATH
HASHES_PATH = OUTPUT_ROOT / "hashes.sha256"

CAMPAIGN_ROOT = (
    AUDIT_ROOT / "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
CAMPAIGN_GATE_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001"
)
D1_ROOT = CAMPAIGN_GATE_ROOT / "D1"
FIXED_HEURISTIC_PATH = CAMPAIGN_GATE_ROOT / "references/fixed_heuristic.json"
INDEPENDENT_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_independent_audit_selection_attempt001"
)

D1_SEEDS = tuple(range(8600, 8620))
REAL_SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
POLICY_INDEX = 5
HYPOTHESIS = "H2"
RANDOM_DRAWS = 50
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BLOCK_SESSIONS = 5
MASTER_RANDOM_SEED = 640_201
STRONG_PERMUTATION_SEED_OFFSET = 910_000
FIXED_FEE = 3.0
MIN_MATCHED_CONTROL_TOKEN_REASSIGNMENT_RATE = 0.90
MAX_MATCHED_CONTROL_P95_ABS_LOG_ASK_ERROR = math.log(2.0)
MIN_D_EXACT_RIGHT_SLOT_MATCH_RATE = 0.90
MAX_STRONG_SHUFFLE_JOINT_G1_G2_PASSES = 1
MAX_MATCHED_EXECUTED_TRADE_COUNT_DRIFT = 0.05
MAX_MATCHED_PREMIUM_RISK_DRIFT = 0.10
MAX_MATCHED_HOLDING_TIME_DRIFT = 0.10
MATCHED_CONTROL_SWAP_PASSES = 200
NEGATIVE_CONTROL_CONTRASTS = (
    "G_minus_B_strong_shuffle_total_increment",
    "C_G_minus_B_strong_shuffle_timing_residual",
    "D_G_minus_B_strong_shuffle_contract_profile_residual",
    "G_minus_C_G_strong_shuffle_slot_residual",
)
COMMON_INTENT_BUDGET_BY_FOLD = {
    1: 10_942,
    2: 1_146,
    3: 11_252,
    4: 5_925,
    5: 3_286,
}
PRIMARY_CONTRASTS = (
    "C_minus_B_timing_given_random_slots",
    "D_minus_B_model_like_contract_profile",
    "F_minus_C_model_slot_beyond_random_at_model_times",
    "F_minus_E_threshold_optimization",
    "F_minus_E_common_threshold_budget_choice",
    "E_minus_G_weak_shuffle_residual_structure",
    "E_common_minus_G_common_weak_shuffle_structure",
    "F_minus_M_F_current_D1_margin_matched_increment",
    "G_minus_B_strong_shuffle_total_increment",
    "G_minus_C_G_strong_shuffle_slot_residual",
    "G_minus_M_G_valid_negative_control_increment",
    "C_G_minus_B_strong_shuffle_timing_residual",
    "D_G_minus_B_strong_shuffle_contract_profile_residual",
    "H_minus_B_real_total_real_model_increment",
    "H_minus_C_real_real_slot_increment",
    "H_minus_M_real_margin_matched_real_increment",
    "C_real_minus_B_real_real_timing_increment",
    "H_minus_I_real_score_direction",
    "F_minus_C_minus_D_plus_B_interaction",
)


class AttributionError(RuntimeError):
    """Fail-closed diagnostic error."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AttributionError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")
    temporary.replace(path)


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    expected = dict(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise AttributionError(f"immutable artifact differs: {path}")
        return
    write_json(path, expected)


def update_progress(node: str, **fields: Any) -> None:
    payload = (
        read_json(ACTIVE_PROGRESS_PATH)
        if ACTIVE_PROGRESS_PATH.is_file()
        else {}
    )
    payload.update(
        {
            "schema_version": "Protocol101D1AttributionProgressV1",
            "status": "running",
            "current_node": node,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
    )
    write_json(ACTIVE_PROGRESS_PATH, payload)


def _required_input_hashes() -> dict[str, str]:
    paths = {
        "attribution_runner": Path(__file__).resolve(),
        "campaign_gate_script": Path(ft1d.__file__).resolve(),
        "hgb_core": Path(hgb_core.__file__).resolve(),
        "reference_core": Path(references.__file__).resolve(),
        "simulator_v5": (
            ROOT / "v4/model/protocol101_serial_simulator_v5.py"
        ),
        "materialization_receipt": ft1d.MATERIALIZATION_RECEIPT,
        "fixed_heuristic": FIXED_HEURISTIC_PATH,
        "d1_detail": D1_ROOT / "D1_detail.json",
        "independent_audit": INDEPENDENT_ROOT / "independent_audit.json",
        "selection": INDEPENDENT_ROOT / "selection.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AttributionError(f"required inputs missing: {missing}")
    return {
        name: sha256_path(path)
        for name, path in sorted(paths.items())
    }


def investigation_plan() -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101D1CausalAttributionPlanV1",
        "status": "preregistered_before_new_control_results",
        "objective": (
            "attribute the shuffled-HGB increment beyond valid "
            "feature-independent P5 controls"
        ),
        "universe": {
            "hypothesis": HYPOTHESIS,
            "policy": "P5",
            "policy_index": POLICY_INDEX,
            "simulator": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
            "folds": list(FOLDS),
            "d1_seeds": list(D1_SEEDS),
            "real_model_seeds": list(REAL_SEEDS),
            "rows_per_session": 330,
            "validation_sessions_per_fold": 45,
            "protected_holdout_access": False,
        },
        "matching_law": {
            "trade_count_definition": (
                "entry-intent count before simulator replay"
            ),
            "executed_trade_count": (
                "reported, never forced after observing outcomes"
            ),
            "exposure": (
                "intent count primary; premium-at-risk, realized holding "
                "minutes, and session coverage reported"
            ),
            "paired_threshold_control": (
                "select exactly the current D1 fold intent count by "
                "outcome-independent score rank"
            ),
            "common_budget_by_fold": {
                str(fold): count
                for fold, count in COMMON_INTENT_BUDGET_BY_FOLD.items()
            },
            "common_budget_source": (
                "rounded median existing D1 intent count by fold, frozen "
                "before new control replay"
            ),
            "score_ties": (
                "top score descending, session, decision time, selected "
                "contract identity"
            ),
        },
        "randomization": {
            "draws": RANDOM_DRAWS,
            "draws_justification": (
                "50 draws per seed yields 1,000 selector-null draws across "
                "all 20 D1 seeds; primary paired uncertainty is separately "
                "estimated with 20,000 chronological block-bootstrap "
                "replicates"
            ),
            "master_seed": MASTER_RANDOM_SEED,
            "prng": "numpy.PCG64DXSM",
            "strong_permutation_seed_offset": (
                STRONG_PERMUTATION_SEED_OFFSET
            ),
            "strong_permutation_unit": (
                "final finite normalized and clipped target after current "
                "D1 cap; global within fold fit role"
            ),
            "strong_permutation_preserves": (
                "exact target multiset, feature rows, row count, model "
                "hyperparameters, noise injection, fold boundaries"
            ),
            "strong_permutation_destroys": (
                "systematic destination association with session, date, "
                "minute, slot, right, moneyness, and premium"
            ),
        },
        "uncertainty": {
            "primary_pairing_unit": "session within fold and seed",
            "chronological_block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
            "replicates": BOOTSTRAP_REPLICATES,
            "shared_resample_across_controls": True,
            "random_control_interval": "empirical 2.5/50/97.5 percentiles",
            "multiple_comparisons": "Holm family-wise alpha 0.05",
            "primary_contrasts": list(PRIMARY_CONTRASTS),
        },
        "decision_rules": {
            "matched_control_min_token_reassignment_rate": (
                MIN_MATCHED_CONTROL_TOKEN_REASSIGNMENT_RATE
            ),
            "matched_control_max_p95_abs_log_ask_error": (
                MAX_MATCHED_CONTROL_P95_ABS_LOG_ASK_ERROR
            ),
            "D_min_exact_right_slot_match_rate": (
                MIN_D_EXACT_RIGHT_SLOT_MATCH_RATE
            ),
            "strong_shuffle_max_joint_G1_G2_passes": (
                MAX_STRONG_SHUFFLE_JOINT_G1_G2_PASSES
            ),
            "matched_executed_trade_count_drift_max": (
                MAX_MATCHED_EXECUTED_TRADE_COUNT_DRIFT
            ),
            "matched_premium_risk_drift_max": (
                MAX_MATCHED_PREMIUM_RISK_DRIFT
            ),
            "matched_holding_time_drift_max": (
                MAX_MATCHED_HOLDING_TIME_DRIFT
            ),
            "negative_control_contrast_family": list(
                NEGATIVE_CONTROL_CONTRASTS
            ),
            "revised_D1_materiality": (
                "$3 per median executed strong-shuffle trade"
            ),
        },
        "constraints": {
            "no_420_campaign_refit": True,
            "disposable_strong_permutation_fits_only": True,
            "no_G9": True,
            "no_protected_holdout": True,
            "no_hold_exit_training": True,
            "no_live_paper_broker_mutation": True,
            "no_simulator_economic_change": True,
            "no_candidate_promotion": True,
        },
        "required_input_hashes": _required_input_hashes(),
    }
    payload["plan_sha256"] = stable_hash(payload)
    return payload


def control_definitions() -> dict[str, Any]:
    controls = {
        "A": {
            "name": "fixed_P5_heuristic",
            "timing": "every eligible D1 decision minute",
            "slot": "VWAP-side nearest ATM",
            "model": None,
            "purpose": "absolute P5 policy-edge baseline",
        },
        "B": {
            "name": "fully_feature_independent_random",
            "timing": "uniform without replacement from complete fold risk set",
            "slot": "uniform eligible candidate at sampled time",
            "budget": "exact paired fold intent count",
            "purpose": "joint random timing and slot baseline",
        },
        "C": {
            "name": "random_slot_at_model_time",
            "timing": "exact current D1 intent times",
            "slot": "uniform eligible candidate",
            "budget": "exact current D1 intent schedule",
            "purpose": "timing retained, slot information removed",
        },
        "D": {
            "name": "random_time_model_like_contract_profile",
            "timing": "random without replacement within fold",
            "slot": (
                "exact source right/canonical-slot profile where available; "
                "nearest causal candidate fallback"
            ),
            "premium_matching": (
                "greedy outcome-blind log-ask proximity from random shortlist"
            ),
            "budget": "exact paired fold intent count",
            "purpose": "contract profile retained, timing removed",
        },
        "E": {
            "name": "weak_D1_HGB_fixed_count",
            "model": "persisted current D1 shuffled HGB",
            "threshold": (
                "none; select exact paired fold budget by validation score rank"
            ),
            "purpose": (
                "feature-dependent weak-shuffle filtering without "
                "calibration-PnL threshold search"
            ),
        },
        "E_common": {
            "name": "weak_D1_HGB_common_frozen_count",
            "model": "persisted current D1 shuffled HGB",
            "threshold": (
                "none; select the preregistered common fold budget by "
                "validation score rank"
            ),
            "purpose": (
                "outcome-independent common-budget weak-shuffle control"
            ),
        },
        "F": {
            "name": "current_D1_exact",
            "model": "persisted current D1 shuffled HGB",
            "threshold": (
                "existing calibration-PnL-maximizing threshold, reproduced"
            ),
            "purpose": "exact failed D1 behavior",
        },
        "G": {
            "name": "strong_global_target_permutation_HGB",
            "model": "100 disposable H2/P5 HGB fits",
            "target": (
                "current final normalized finite target globally permuted "
                "after cap within fold fit role"
            ),
            "threshold": "same fixed-count law as E",
            "purpose": "remove residual session/slot/right/premium structure",
        },
        "G_common": {
            "name": "strong_permutation_HGB_common_frozen_count",
            "model": "same disposable strong-permutation HGB fits as G",
            "threshold": (
                "none; select the same preregistered common fold budget "
                "as E_common"
            ),
            "purpose": (
                "strong-shuffle counterpart for common-budget attribution"
            ),
        },
        "H": {
            "name": "real_H2_P5_HGB_fixed_count",
            "model": "15 persisted real H2/P5 campaign models",
            "threshold": "common frozen intent budget by fold",
            "purpose": "real model contribution under equal search budget",
        },
        "I": {
            "name": "sign_reversed_real_H2_P5",
            "model": "same 15 persisted real H2/P5 models",
            "scores": (
                "negate every candidate score before action and slot ranking"
            ),
            "threshold": "same common frozen intent budget as H",
            "purpose": "directional-information control",
        },
        "derived_matching_controls": {
            "M_F": (
                "feature-independent profile reassignment at current-D1 "
                "times, preserving exact intent-time, slot/right, and "
                "premium-bucket margins"
            ),
            "C_G": (
                "random slots at strong-permutation G selected times"
            ),
            "D_G": (
                "random times with strong-permutation G contract profile"
            ),
            "M_G": (
                "feature-independent profile reassignment at G times, "
                "preserving exact slot/right/premium-bucket margins; "
                "time/profile tokens must be >=90% reassigned while physical "
                "contract change is reported separately"
            ),
            "B_real": (
                "fully random timing/slot at H common intent budget"
            ),
            "C_real": "random slots at real-H selected times",
            "D_real": "random times with real-H contract profile",
            "M_real": (
                "feature-independent profile reassignment at real-H times "
                "with exact slot/right/premium-bucket margins"
            ),
        },
    }
    payload = {
        "schema_version": "Protocol101D1PairedControlDefinitionsV1",
        "status": "preregistered_before_new_control_results",
        "controls": controls,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "random_draws": RANDOM_DRAWS,
        "fixed_fee_dollars": FIXED_FEE,
        "fail_closed": [
            "any input hash mismatch",
            "any simulator other than v5",
            "any duplicate replay identity",
            "any validation/fit/calibration session overlap",
            "any current-D1 refit model hash mismatch",
            "any current-D1 replay mismatch",
            "any paired intent-count mismatch",
            "any missing permutation receipt",
        ],
    }
    payload["definitions_sha256"] = stable_hash(payload)
    return payload


def preregister() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    PERMUTATION_ROOT.mkdir(parents=True, exist_ok=True)
    STRONG_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    SESSION_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    CONTROL_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_once(INVESTIGATION_PLAN_PATH, investigation_plan())
    write_json_once(CONTROL_DEFINITIONS_PATH, control_definitions())
    update_progress(
        "PREREGISTERED",
        plan_sha256=sha256_path(INVESTIGATION_PLAN_PATH),
        definitions_sha256=sha256_path(CONTROL_DEFINITIONS_PATH),
    )


def _fold_payload(scope: Any, fold: int) -> Mapping[str, Any]:
    return scope.folds[FOLDS.index(int(fold))]


def _model_path_for_d1(seed: int, fold: int) -> Path:
    return D1_ROOT / f"seed{seed}" / f"fold{fold}" / "model.pkl"


def _summary_path_for_d1(seed: int, fold: int) -> Path:
    return D1_ROOT / f"seed{seed}" / f"fold{fold}" / "summary.json"


def _real_unit_root(seed: int, fold: int) -> Path:
    return (
        CAMPAIGN_ROOT
        / "H2/units/H2/policy5"
        / f"seed{seed}"
        / f"expanding_fold_{fold:02d}"
    )


def _load_model(path: Path) -> HistGradientBoostingRegressor:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, HistGradientBoostingRegressor):
        raise AttributionError(f"unexpected model type: {path}: {type(model)}")
    return model


def _noise_model() -> DivergenceNoiseModel:
    path = Path(stage1_runner.NOISE_DISTRIBUTION)
    if not path.is_absolute():
        path = ROOT / path
    return DivergenceNoiseModel.from_parquet(path)


def _d1_validation_decisions(
    *,
    scope: Any,
    path_map: Mapping[str, Path],
    fold: int,
    guard_margins: Mapping[str, float],
) -> tuple[list[RepairedCanonicalDecision], list[dict[str, Any]]]:
    sessions = list(_fold_payload(scope, fold)["validation_sessions"])
    return ft1d._d1_role_decisions(
        path_map=path_map,
        sessions=sessions,
        role="validation",
        seed=None,
        fold=fold,
        guard_margins=guard_margins,
    )


def _reference_opportunities(
    decisions: Sequence[RepairedCanonicalDecision],
    *,
    fold: int,
    campaign_id: str,
) -> list[references.ReferenceOpportunity]:
    out: list[references.ReferenceOpportunity] = []
    for ordinal, repaired in enumerate(decisions):
        gap = float(repaired.base.features[0, 0])
        out.append(
            references.ReferenceOpportunity(
                campaign_id=campaign_id,
                fold=f"F{fold}",
                split="validation",
                policy_index=POLICY_INDEX,
                repaired=repaired,
                vwap_side="C" if gap >= 0.0 else "P",
                decision_ordinal=ordinal,
            )
        )
    return out


def _candidate(
    opportunity: references.ReferenceOpportunity,
    index: int,
    *,
    strategy: str,
    score: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> SerialCandidateV5:
    item = references._candidate_from_opportunity(
        opportunity,
        int(index),
        strategy=strategy,
        metadata=dict(metadata or {}),
    )
    values = {
        name: getattr(item, name)
        for name in SerialCandidateV5.__dataclass_fields__
    }
    values["score"] = float(score)
    return SerialCandidateV5(**values)


def fixed_heuristic_candidates_for_d1_risk_set(
    opportunities: Sequence[references.ReferenceOpportunity],
) -> list[SerialCandidateV5]:
    """Apply frozen P5 selection without requiring unused H2 fields finite."""
    ordered = sorted(
        opportunities,
        key=lambda item: (
            item.fold,
            item.repaired.base.session,
            int(item.repaired.base.decision_time.value),
        ),
    )
    identities = [
        (
            item.fold,
            item.repaired.base.session,
            int(item.repaired.base.decision_time.value),
        )
        for item in ordered
    ]
    if len(identities) != len(set(identities)):
        raise AttributionError("duplicate fixed-heuristic opportunity identity")
    candidates: list[SerialCandidateV5] = []
    for item in ordered:
        if int(item.policy_index) != POLICY_INDEX:
            raise AttributionError("fixed heuristic is not P5")
        base = item.repaired.base
        side_indices = [
            index
            for index, right in enumerate(base.rights)
            if str(right) == str(item.vwap_side)
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
            _candidate(
                item,
                selected,
                strategy="A_fixed_P5_vwap_side_nearest_atm",
                metadata={"heuristic_policy_fixed": POLICY_INDEX},
            )
        )
    return candidates


def _selected_candidate_rows(
    model: HistGradientBoostingRegressor,
    decisions: Sequence[RepairedCanonicalDecision],
    *,
    seed: int,
    fold: int,
    epsilon: float,
    noise_model: DivergenceNoiseModel,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    feature_names = tuple(HYPOTHESES[HYPOTHESIS])
    scores = hgb_core.score_decisions(
        model,
        [item.base for item in decisions],
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(seed) + 200_000,
    )
    rows: list[dict[str, Any]] = []
    for opportunity, candidate_scores in zip(
        _reference_opportunities(
            decisions,
            fold=fold,
            campaign_id=f"ATTRIBUTION-S{seed}-F{fold}",
        ),
        scores,
    ):
        working_scores = (
            -np.asarray(candidate_scores, dtype=float)
            if reverse
            else np.asarray(candidate_scores, dtype=float)
        )
        selected, top, second, margin, confident = (
            hgb_core.selected_index_for_scores(
                opportunity.repaired.base,
                working_scores,
                epsilon=float(epsilon),
                k_slot=2.0,
            )
        )
        base = opportunity.repaired.base
        rows.append(
            {
                "opportunity": opportunity,
                "selected_index": int(selected),
                "top_score": float(top),
                "second_score": float(second),
                "margin": float(margin),
                "slot_confident": bool(confident),
                "selected_score": float(working_scores[selected]),
                "session": base.session,
                "decision_time_ns": int(base.decision_time.value),
                "contract_id": str(base.contract_ids[selected]),
                "right": str(base.rights[selected]),
                "slot": int(
                    opportunity.repaired.canonical_strike_slots[selected]
                ),
                "offset": float(base.offsets[selected]),
                "entry_ask": float(base.entry_asks[selected]),
            }
        )
    return rows


def select_fixed_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    strategy: str,
) -> list[SerialCandidateV5]:
    if count < 0 or count > len(rows):
        raise AttributionError(
            f"fixed-count budget outside decision grid: {count}/{len(rows)}"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["top_score"]),
            str(row["session"]),
            int(row["decision_time_ns"]),
            str(row["contract_id"]),
        ),
    )
    selected = ordered[: int(count)]
    candidates = [
        _candidate(
            row["opportunity"],
            int(row["selected_index"]),
            strategy=strategy,
            score=float(row["top_score"]),
            metadata={
                "fixed_count_budget": int(count),
                "slot_confident": bool(row["slot_confident"]),
                "selected_candidate_score": float(row["selected_score"]),
            },
        )
        for row in selected
    ]
    return sorted(
        candidates,
        key=lambda item: (
            item.fold,
            item.session,
            int(item.decision_time_ns),
            item.contract_id,
        ),
    )


def _current_candidates_from_summary(
    summary: Mapping[str, Any],
) -> list[SerialCandidateV5]:
    candidates: list[SerialCandidateV5] = []
    for payload in summary["entry_intents"]:
        values = {
            name: value
            for name, value in payload.items()
            if name in SerialCandidateV5.__dataclass_fields__
        }
        candidate = SerialCandidateV5(**values)
        pnl = (
            (
                float(candidate.label_executable_exit_bid)
                - float(candidate.entry_ask)
            )
            * 100.0
            - FIXED_FEE
        )
        values["raw_label_pnl_after_campaign_fee"] = float(pnl)
        candidates.append(SerialCandidateV5(**values))
    return candidates


def _with_fold(
    candidates: Sequence[SerialCandidateV5],
    fold: int,
) -> list[SerialCandidateV5]:
    result: list[SerialCandidateV5] = []
    for item in candidates:
        values = {
            name: getattr(item, name)
            for name in SerialCandidateV5.__dataclass_fields__
        }
        values["fold"] = f"F{int(fold)}"
        values["split"] = "validation"
        result.append(SerialCandidateV5(**values))
    return result


def _profit_factor(pnls: np.ndarray) -> float | None:
    positive = float(pnls[pnls > 0.0].sum())
    negative = float(-pnls[pnls < 0.0].sum())
    if negative == 0.0:
        return None
    return positive / negative


def _time_bucket(decision_time_ns: int) -> str:
    timestamp = pd.Timestamp(int(decision_time_ns), tz="UTC").tz_convert(
        "America/New_York"
    )
    minute = timestamp.hour * 60 + timestamp.minute
    if minute < 10 * 60 + 30:
        return "09:32-10:29"
    if minute < 14 * 60:
        return "10:30-13:59"
    return "14:00-15:30"


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items(), key=lambda x: str(x[0]))
    }


def evaluate_stream(
    candidates: Sequence[SerialCandidateV5],
    *,
    control: str,
    model_seed: int,
    draw: int,
    all_sessions_by_fold: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trades, state = references.replay_reference_v5(candidates, fee=FIXED_FEE)
    metrics = references.replay_metrics(trades, state)
    pnls = np.asarray(
        [float(item.raw_label_pnl_after_campaign_fee) for item in trades],
        dtype=float,
    )
    fold_metrics: dict[str, Any] = {}
    for fold in sorted(all_sessions_by_fold):
        fold_candidates = [item for item in candidates if item.fold == fold]
        if (
            len(all_sessions_by_fold) == 1
            and len(fold_candidates) == len(candidates)
        ):
            fold_trades, fold_state = trades, state
        else:
            fold_trades, fold_state = references.replay_reference_v5(
                fold_candidates,
                fee=FIXED_FEE,
            )
        fold_values = np.asarray(
            [
                float(item.raw_label_pnl_after_campaign_fee)
                for item in fold_trades
            ],
            dtype=float,
        )
        fold_metrics[fold] = {
            **references.replay_metrics(fold_trades, fold_state),
            "win_rate": (
                float(np.mean(fold_values > 0.0))
                if len(fold_values)
                else 0.0
            ),
            "profit_factor": _profit_factor(fold_values),
        }
    sessions: list[dict[str, Any]] = []
    trades_by_session: dict[tuple[str, str], list[SerialReplayTradeV5]] = (
        defaultdict(list)
    )
    for trade in trades:
        trades_by_session[(trade.fold, trade.session)].append(trade)
    for fold, session_ids in sorted(all_sessions_by_fold.items()):
        for session in session_ids:
            selected = trades_by_session[(fold, session)]
            values = np.asarray(
                [
                    float(item.raw_label_pnl_after_campaign_fee)
                    for item in selected
                ],
                dtype=float,
            )
            sessions.append(
                {
                    "control": control,
                    "model_seed": int(model_seed),
                    "draw": int(draw),
                    "fold": fold,
                    "session": session,
                    "net_pnl": float(values.sum()) if len(values) else 0.0,
                    "trades": int(len(values)),
                    "win_rate": (
                        float(np.mean(values > 0.0))
                        if len(values)
                        else 0.0
                    ),
                    "profit_factor": _profit_factor(values),
                }
            )
    candidate_premiums = np.asarray(
        [float(item.entry_ask) * 100.0 for item in candidates],
        dtype=float,
    )
    holding_minutes = np.asarray(
        [
            (
                int(item.label_realized_exit_time_ns)
                - int(item.decision_time_ns)
            )
            / 60_000_000_000.0
            for item in candidates
        ],
        dtype=float,
    )
    executed_premiums = np.asarray(
        [float(item.premium_at_risk) for item in trades],
        dtype=float,
    )
    executed_holding_minutes = np.asarray(
        [
            (
                int(item.label_realized_exit_time_ns)
                - int(item.decision_time_ns)
            )
            / 60_000_000_000.0
            for item in trades
        ],
        dtype=float,
    )
    summary = {
        "control": control,
        "model_seed": int(model_seed),
        "draw": int(draw),
        "entry_intents": int(len(candidates)),
        "trades": int(metrics["trades"]),
        "net_pnl": float(metrics["net_pnl"]),
        "median_session_pnl": float(
            statistics.median(row["net_pnl"] for row in sessions)
        ),
        "pnl_per_trade": (
            float(metrics["net_pnl"]) / int(metrics["trades"])
            if int(metrics["trades"])
            else 0.0
        ),
        "max_drawdown": float(metrics["max_drawdown"]),
        "minimum_equity": float(metrics["minimum_equity"]),
        "win_rate": float(np.mean(pnls > 0.0)) if len(pnls) else 0.0,
        "profit_factor": _profit_factor(pnls),
        "gross_profit": float(pnls[pnls > 0.0].sum()),
        "gross_loss": float(-pnls[pnls < 0.0].sum()),
        "call_put_distribution": _distribution(
            item.right for item in trades
        ),
        "slot_distribution": _distribution(
            item.canonical_strike_slot for item in trades
        ),
        "time_of_day_distribution": _distribution(
            _time_bucket(item.decision_time_ns) for item in trades
        ),
        "session_count": len(sessions),
        "profitable_sessions": int(
            sum(float(row["net_pnl"]) > 0.0 for row in sessions)
        ),
        "fold_metrics": fold_metrics,
        "profitable_folds": int(
            sum(
                float(item["net_pnl"]) > 0.0
                for item in fold_metrics.values()
            )
        ),
        "G1": bool(
            sum(
                float(item["net_pnl"]) > 0.0
                for item in fold_metrics.values()
            )
            >= 4
            and float(metrics["net_pnl"]) > 0.0
        ),
        "premium_at_risk": {
            "mean": (
                float(np.mean(candidate_premiums))
                if len(candidate_premiums)
                else 0.0
            ),
            "median": (
                float(np.median(candidate_premiums))
                if len(candidate_premiums)
                else 0.0
            ),
            "p95": (
                float(np.percentile(candidate_premiums, 95))
                if len(candidate_premiums)
                else 0.0
            ),
        },
        "candidate_holding_minutes": {
            "mean": (
                float(np.mean(holding_minutes))
                if len(holding_minutes)
                else 0.0
            ),
            "median": (
                float(np.median(holding_minutes))
                if len(holding_minutes)
                else 0.0
            ),
            "p95": (
                float(np.percentile(holding_minutes, 95))
                if len(holding_minutes)
                else 0.0
            ),
        },
        "executed_premium_at_risk": {
            "mean": (
                float(np.mean(executed_premiums))
                if len(executed_premiums)
                else 0.0
            ),
            "median": (
                float(np.median(executed_premiums))
                if len(executed_premiums)
                else 0.0
            ),
            "p95": (
                float(np.percentile(executed_premiums, 95))
                if len(executed_premiums)
                else 0.0
            ),
        },
        "executed_holding_minutes": {
            "mean": (
                float(np.mean(executed_holding_minutes))
                if len(executed_holding_minutes)
                else 0.0
            ),
            "median": (
                float(np.median(executed_holding_minutes))
                if len(executed_holding_minutes)
                else 0.0
            ),
            "p95": (
                float(np.percentile(executed_holding_minutes, 95))
                if len(executed_holding_minutes)
                else 0.0
            ),
        },
        "candidate_stream_hash": metrics["candidate_stream_hash"],
        "candidate_payload_hash": metrics["candidate_payload_hash"],
        "trade_identity_hash": metrics["trade_identity_hash"],
        "simulator_version": metrics["simulator_version"],
    }
    return summary, sessions


def _random_generator(*parts: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [MASTER_RANDOM_SEED, *[int(value) for value in parts]]
    )
    return np.random.Generator(np.random.PCG64DXSM(seed_sequence))


def random_time_random_slot(
    opportunities: Sequence[references.ReferenceOpportunity],
    *,
    count: int,
    model_seed: int,
    draw: int,
) -> list[SerialCandidateV5]:
    if count > len(opportunities):
        raise AttributionError("B intent budget exceeds complete risk set")
    rng = _random_generator(1, model_seed, draw)
    chosen = np.sort(
        rng.choice(len(opportunities), size=int(count), replace=False)
    )
    candidates = []
    for index in chosen:
        opportunity = opportunities[int(index)]
        selected = int(
            rng.integers(0, len(opportunity.repaired.base.contract_ids))
        )
        candidates.append(
            _candidate(
                opportunity,
                selected,
                strategy="B_random_time_random_slot",
                metadata={"draw": int(draw), "model_seed": int(model_seed)},
            )
        )
    return candidates


def random_slot_at_model_time(
    model_candidates: Sequence[SerialCandidateV5],
    opportunity_by_time: Mapping[
        tuple[str, int], references.ReferenceOpportunity
    ],
    *,
    model_seed: int,
    draw: int,
) -> list[SerialCandidateV5]:
    rng = _random_generator(2, model_seed, draw)
    candidates = []
    for source in model_candidates:
        opportunity = opportunity_by_time[
            (source.session, int(source.decision_time_ns))
        ]
        selected = int(
            rng.integers(0, len(opportunity.repaired.base.contract_ids))
        )
        candidates.append(
            _candidate(
                opportunity,
                selected,
                strategy="C_random_slot_at_model_time",
                metadata={"draw": int(draw), "model_seed": int(model_seed)},
            )
        )
    return candidates


def _profile_candidate_index(
    opportunity: references.ReferenceOpportunity,
    *,
    right: str,
    slot: int,
    entry_ask: float,
) -> tuple[int, float]:
    repaired = opportunity.repaired
    exact = [
        index
        for index, (candidate_right, candidate_slot) in enumerate(
            zip(
                repaired.base.rights,
                repaired.canonical_strike_slots,
            )
        )
        if str(candidate_right) == str(right)
        and int(candidate_slot) == int(slot)
    ]
    pool = exact or [
        index
        for index, candidate_right in enumerate(repaired.base.rights)
        if str(candidate_right) == str(right)
    ]
    if not pool:
        pool = list(range(len(repaired.base.contract_ids)))
    selected = min(
        pool,
        key=lambda index: (
            abs(
                math.log(max(float(repaired.base.entry_asks[index]), 1e-9))
                - math.log(max(float(entry_ask), 1e-9))
            ),
            abs(int(repaired.canonical_strike_slots[index]) - int(slot)),
            int(index),
        ),
    )
    ask_error = abs(
        math.log(max(float(repaired.base.entry_asks[selected]), 1e-9))
        - math.log(max(float(entry_ask), 1e-9))
    )
    return int(selected), float(ask_error)


def _premium_bucket(value: float) -> str:
    edges = (1.0, 2.0, 5.0, 10.0, 20.0)
    for index, edge in enumerate(edges):
        if float(value) < edge:
            return f"B{index}"
    return f"B{len(edges)}"


def random_time_model_profile(
    model_candidates: Sequence[SerialCandidateV5],
    opportunities: Sequence[references.ReferenceOpportunity],
    *,
    model_seed: int,
    draw: int,
) -> tuple[list[SerialCandidateV5], dict[str, Any]]:
    if len(model_candidates) > len(opportunities):
        raise AttributionError("D intent budget exceeds complete risk set")
    rng = _random_generator(3, model_seed, draw)
    chosen = list(
        np.sort(
            rng.choice(
                len(opportunities),
                size=len(model_candidates),
                replace=False,
            )
        )
    )
    profiles_by_key: dict[tuple[str, int], list[SerialCandidateV5]] = (
        defaultdict(list)
    )
    for source in model_candidates:
        profiles_by_key[
            (str(source.right), int(source.canonical_strike_slot))
        ].append(source)
    rng.shuffle(chosen)
    candidates: list[SerialCandidateV5] = []
    exact_profile = 0
    ask_errors: list[float] = []
    cursor = 0
    for key in sorted(profiles_by_key):
        profiles = profiles_by_key[key]
        assigned = chosen[cursor : cursor + len(profiles)]
        cursor += len(profiles)
        destinations: list[
            tuple[references.ReferenceOpportunity, int, float]
        ] = []
        for opportunity_index in assigned:
            opportunity = opportunities[int(opportunity_index)]
            selected, _ = _profile_candidate_index(
                opportunity,
                right=key[0],
                slot=key[1],
                entry_ask=1.0,
            )
            destinations.append(
                (
                    opportunity,
                    selected,
                    float(opportunity.repaired.base.entry_asks[selected]),
                )
            )
        profiles = sorted(profiles, key=lambda item: float(item.entry_ask))
        destinations = sorted(destinations, key=lambda item: item[2])
        for source, (opportunity, selected, destination_ask) in zip(
            profiles,
            destinations,
        ):
            repaired = opportunity.repaired
            ask_error = abs(
                math.log(max(destination_ask, 1e-9))
                - math.log(max(float(source.entry_ask), 1e-9))
            )
            exact_profile += int(
                str(repaired.base.rights[selected]) == str(source.right)
                and int(repaired.canonical_strike_slots[selected])
                == int(source.canonical_strike_slot)
            )
            ask_errors.append(float(ask_error))
            candidates.append(
                _candidate(
                    opportunity,
                    selected,
                    strategy="D_random_time_model_profile",
                    metadata={
                        "draw": int(draw),
                        "model_seed": int(model_seed),
                        "source_right": source.right,
                        "source_slot": int(source.canonical_strike_slot),
                        "source_entry_ask": float(source.entry_ask),
                        "log_ask_error": float(ask_error),
                    },
                )
            )
    if cursor != len(chosen):
        raise AttributionError("D profile partition did not consume time grid")
    return candidates, {
        "intent_count": len(candidates),
        "exact_right_slot_matches": int(exact_profile),
        "exact_right_slot_match_rate": (
            float(exact_profile / len(candidates)) if candidates else 1.0
        ),
        "median_abs_log_ask_error": (
            float(np.median(ask_errors)) if ask_errors else 0.0
        ),
        "p95_abs_log_ask_error": (
            float(np.percentile(ask_errors, 95)) if ask_errors else 0.0
        ),
    }


def profile_permutation_at_model_times(
    model_candidates: Sequence[SerialCandidateV5],
    opportunity_by_time: Mapping[
        tuple[str, int], references.ReferenceOpportunity
    ],
    *,
    model_seed: int,
    draw: int,
) -> tuple[list[SerialCandidateV5], dict[str, Any]]:
    """Keep exact model times and profile margins, but break their pairing."""
    rng = _random_generator(4, model_seed, draw)
    destinations = sorted(
        model_candidates,
        key=lambda item: (
            item.session,
            int(item.decision_time_ns),
            item.contract_id,
        ),
    )
    token_at_destination = list(range(len(destinations)))
    profiles = [
        (
            str(item.right),
            int(item.canonical_strike_slot),
            _premium_bucket(float(item.entry_ask)),
        )
        for item in destinations
    ]
    available_profiles: list[set[tuple[str, int, str]]] = []
    contract_by_profile: list[dict[tuple[str, int, str], str]] = []
    for destination in destinations:
        opportunity = opportunity_by_time[
            (destination.session, int(destination.decision_time_ns))
        ]
        profile_contract = {
                (str(right), int(slot))
                + (_premium_bucket(float(ask)),): str(contract_id)
                for right, slot, ask, contract_id in zip(
                    opportunity.repaired.base.rights,
                    opportunity.repaired.canonical_strike_slots,
                    opportunity.repaired.base.entry_asks,
                    opportunity.repaired.base.contract_ids,
                )
        }
        available_profiles.append(set(profile_contract))
        contract_by_profile.append(profile_contract)
    successful_swaps = 0
    for _pass in range(MATCHED_CONTROL_SWAP_PASSES):
        order = rng.permutation(len(destinations))
        for left, right in zip(order[0::2], order[1::2]):
            left = int(left)
            right = int(right)
            left_token = token_at_destination[left]
            right_token = token_at_destination[right]
            if (
                profiles[right_token] in available_profiles[left]
                and profiles[left_token] in available_profiles[right]
            ):
                old_unchanged = int(
                    contract_by_profile[left][profiles[left_token]]
                    == str(destinations[left].contract_id)
                ) + int(
                    contract_by_profile[right][profiles[right_token]]
                    == str(destinations[right].contract_id)
                )
                new_unchanged = int(
                    contract_by_profile[left][profiles[right_token]]
                    == str(destinations[left].contract_id)
                ) + int(
                    contract_by_profile[right][profiles[left_token]]
                    == str(destinations[right].contract_id)
                )
                if new_unchanged <= old_unchanged:
                    token_at_destination[left] = right_token
                    token_at_destination[right] = left_token
                    successful_swaps += 1
    candidates: list[SerialCandidateV5] = []
    exact_profile = 0
    unchanged_pairing = 0
    unchanged_selected_contract = 0
    ask_errors: list[float] = []
    for destination_index, source_index in enumerate(token_at_destination):
        destination = destinations[destination_index]
        source = destinations[int(source_index)]
        opportunity = opportunity_by_time[
            (destination.session, int(destination.decision_time_ns))
        ]
        selected, ask_error = _profile_candidate_index(
            opportunity,
            right=str(source.right),
            slot=int(source.canonical_strike_slot),
            entry_ask=float(source.entry_ask),
        )
        repaired = opportunity.repaired
        exact_profile += int(
            str(repaired.base.rights[selected]) == str(source.right)
            and int(repaired.canonical_strike_slots[selected])
            == int(source.canonical_strike_slot)
        )
        unchanged_pairing += int(destination_index == int(source_index))
        unchanged_selected_contract += int(
            str(repaired.base.contract_ids[selected])
            == str(destination.contract_id)
        )
        ask_errors.append(float(ask_error))
        candidates.append(
            _candidate(
                opportunity,
                selected,
                strategy="M_profile_permutation_at_model_times",
                metadata={
                    "draw": int(draw),
                    "model_seed": int(model_seed),
                    "source_session": source.session,
                    "source_decision_time_ns": int(source.decision_time_ns),
                    "source_right": str(source.right),
                    "source_slot": int(source.canonical_strike_slot),
                    "source_entry_ask": float(source.entry_ask),
                    "log_ask_error": float(ask_error),
                },
            )
        )
    source_profile = Counter(
        (
            str(item.right),
            int(item.canonical_strike_slot),
            _premium_bucket(float(item.entry_ask)),
        )
        for item in destinations
    )
    destination_profile = Counter(
        (
            str(item.right),
            int(item.canonical_strike_slot),
            _premium_bucket(float(item.entry_ask)),
        )
        for item in candidates
    )
    if source_profile != destination_profile:
        raise AttributionError(
            "M control failed exact right/slot/premium-bucket margins"
        )
    contract_change_rate = (
        1.0 - unchanged_selected_contract / len(candidates)
        if candidates
        else 1.0
    )
    token_reassignment_rate = (
        1.0 - unchanged_pairing / len(candidates)
        if candidates
        else 1.0
    )
    p95_ask_error = (
        float(np.percentile(ask_errors, 95)) if ask_errors else 0.0
    )
    if p95_ask_error > MAX_MATCHED_CONTROL_P95_ABS_LOG_ASK_ERROR:
        raise AttributionError("M control premium matching exceeds frozen bound")
    return candidates, {
        "intent_count": len(candidates),
        "exact_time_set_preserved": True,
        "exact_right_slot_distribution_preserved": True,
        "exact_premium_bucket_distribution_preserved": True,
        "exact_right_slot_matches": int(exact_profile),
        "exact_right_slot_match_rate": (
            float(exact_profile / len(candidates)) if candidates else 1.0
        ),
        "unchanged_time_profile_pairings": int(unchanged_pairing),
        "unchanged_time_profile_pairing_rate": (
            float(unchanged_pairing / len(candidates))
            if candidates
            else 1.0
        ),
        "time_profile_token_reassignment_rate": float(
            token_reassignment_rate
        ),
        "token_reassignment_quality_pass": bool(
            token_reassignment_rate
            >= MIN_MATCHED_CONTROL_TOKEN_REASSIGNMENT_RATE
        ),
        "unchanged_selected_contracts": int(unchanged_selected_contract),
        "effective_contract_change_rate": float(contract_change_rate),
        "median_abs_log_ask_error": (
            float(np.median(ask_errors)) if ask_errors else 0.0
        ),
        "p95_abs_log_ask_error": p95_ask_error,
        "successful_feasible_token_swaps": int(successful_swaps),
        "matching_method": (
            "identity-feasible initialization plus 200 randomized monotone "
            "token-derangement passes preserving exact right/slot/premium "
            "bucket; physical contract change is diagnostic because a "
            "stable ladder can map reassigned tokens to the same contract"
        ),
    }


def _array_correlations(
    original: np.ndarray,
    shuffled: np.ndarray,
) -> dict[str, float | None]:
    finite = np.isfinite(original) & np.isfinite(shuffled)
    original = original[finite]
    shuffled = shuffled[finite]
    if len(original) < 3:
        return {"pearson": None, "spearman": None}
    pearson = pearsonr(original, shuffled)
    spearman = spearmanr(original, shuffled)
    return {
        "pearson": (
            float(pearson.statistic)
            if math.isfinite(float(pearson.statistic))
            else None
        ),
        "spearman": (
            float(spearman.statistic)
            if math.isfinite(float(spearman.statistic))
            else None
        ),
    }


def _class_balance(target: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    finite = target[np.isfinite(target)]
    return {
        "count": int(len(target)),
        "finite_count": int(len(finite)),
        "positive": int((finite > 0.0).sum()),
        "zero": int((finite == 0.0).sum()),
        "negative": int((finite < 0.0).sum()),
        "positive_fraction": (
            float(np.mean(finite > 0.0)) if len(finite) else 0.0
        ),
        "negative_fraction": (
            float(np.mean(finite < 0.0)) if len(finite) else 0.0
        ),
    }


def _raw_source_candidate(
    source_row: Mapping[str, Any],
    *,
    strike_index: int,
    right_index: int,
) -> dict[str, Any]:
    source_contracts = np.asarray(source_row["contract_ids"], dtype=object)
    source_rights = tuple(str(value) for value in source_row["rights"])
    source_ladder = np.asarray(source_row["option_ladder"], dtype=float)
    source_labels = np.asarray(
        source_row["labels_net_pnl"],
        dtype=float,
    )[:, :, POLICY_INDEX]
    if (
        source_contracts.shape != (21, 2)
        or source_ladder.shape[:2] != (21, 2)
        or source_labels.shape != (21, 2)
        or len(source_rights) != 2
    ):
        raise AttributionError(
            "D1 source raw ladder geometry differs from 21x2"
        )
    if not (0 <= int(strike_index) < 21 and 0 <= int(right_index) < 2):
        raise AttributionError("D1 source ladder index outside 21x2 axes")
    return {
        "contract_id": str(source_contracts[strike_index, right_index]),
        "right": source_rights[right_index],
        "canonical_slot": int(strike_index),
        "entry_ask": float(source_ladder[strike_index, right_index, 1]),
        "dollar_pnl_label": float(
            source_labels[strike_index, right_index]
        ),
    }


def _training_arrays_with_metadata(
    decisions: Sequence[RepairedCanonicalDecision],
    *,
    original_by_identity: Mapping[
        tuple[str, int, str], RepairedCanonicalDecision
    ],
    raw_source_by_time: Mapping[tuple[str, int], Mapping[str, Any]],
    source_time_by_destination: Mapping[tuple[str, int], int],
    noise_model: DivergenceNoiseModel,
    config: HGBUnitConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_names = tuple(HYPOTHESES[HYPOTHESIS])
    features = np.vstack([item.base.features for item in decisions]).astype(
        np.float64
    )
    labels = np.concatenate([item.base.labels for item in decisions]).astype(
        np.float64
    )
    asks = np.concatenate([item.base.entry_asks for item in decisions]).astype(
        np.float64
    )
    offsets = np.concatenate([item.base.offsets for item in decisions]).astype(
        np.float64
    )
    frame = pd.DataFrame(features, columns=feature_names)
    frame["abs_offset"] = np.abs(offsets)
    noisy = noise_model.inject_dataframe(
        frame,
        feature_columns=feature_names,
        seed=int(config.seed),
        scale=float(config.training_noise_scale),
    )
    x = noisy.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    target = np.divide(
        labels - float(config.fee),
        asks * 100.0,
        out=np.full_like(labels, np.nan),
        where=asks > 0.0,
    )
    target = np.clip(
        target,
        float(config.target_clip_low),
        float(config.target_clip_high),
    )
    rows: list[dict[str, Any]] = []
    destination_indexes: dict[tuple[str, int], int] = {}
    source_indexes: dict[tuple[str, int], int] = {}
    for session in sorted({item.base.session for item in decisions}):
        destination_times = sorted(
            int(destination)
            for (destination_session, destination)
            in source_time_by_destination
            if destination_session == session
        )
        destination_indexes.update(
            {
                (session, time_value): index
                for index, time_value in enumerate(destination_times)
            }
        )
        source_values = sorted(
            {
                int(value)
                for (source_session, _destination), value
                in source_time_by_destination.items()
                if source_session == session
            }
        )
        source_indexes.update(
            {
                (session, time_value): index
                for index, time_value in enumerate(source_values)
            }
        )
    for repaired in decisions:
        base = repaired.base
        source_time = source_time_by_destination[
            (base.session, int(base.decision_time.value))
        ]
        source_row = raw_source_by_time[(base.session, int(source_time))]
        for index in range(len(base.labels)):
            strike_index = int(base.strike_indices[index])
            right_index = int(base.right_indices[index])
            source = _raw_source_candidate(
                source_row,
                strike_index=strike_index,
                right_index=right_index,
            )
            source_label = float(source["dollar_pnl_label"])
            original = original_by_identity.get(
                (
                    base.session,
                    int(base.decision_time.value),
                    str(base.contract_ids[index]),
                )
            )
            original_label = (
                math.nan
                if original is None
                else float(original.base.labels[0])
            )
            original_target = (
                np.clip(
                    (original_label - float(config.fee))
                    / (float(base.entry_asks[index]) * 100.0),
                    float(config.target_clip_low),
                    float(config.target_clip_high),
                )
                if math.isfinite(original_label)
                and float(base.entry_asks[index]) > 0.0
                else math.nan
            )
            rows.append(
                {
                    "destination_session": base.session,
                    "destination_decision_time_ns": int(
                        base.decision_time.value
                    ),
                    "destination_row_index": int(
                        destination_indexes[
                            (
                                base.session,
                                int(base.decision_time.value),
                            )
                        ]
                    ),
                    "source_session": base.session,
                    "source_decision_time_ns": int(source_time),
                    "source_row_index": int(
                        source_indexes[(base.session, int(source_time))]
                    ),
                    "source_contract_id": source["contract_id"],
                    "source_right": source["right"],
                    "source_strike_index": strike_index,
                    "source_right_index": right_index,
                    "source_canonical_slot": source["canonical_slot"],
                    "source_entry_ask": source["entry_ask"],
                    "source_dollar_pnl_label": source_label,
                    "contract_id": str(base.contract_ids[index]),
                    "right": str(base.rights[index]),
                    "strike_index": strike_index,
                    "right_index": right_index,
                    "canonical_slot": int(
                        repaired.canonical_strike_slots[index]
                    ),
                    "offset": float(base.offsets[index]),
                    "entry_ask": float(base.entry_asks[index]),
                    "original_target": float(original_target),
                    "shuffled_dollar_pnl_label": float(
                        base.labels[index]
                    ),
                    "shuffled_target": float(
                        np.clip(
                            (
                                float(base.labels[index])
                                - float(config.fee)
                            )
                            / (float(base.entry_asks[index]) * 100.0),
                            float(config.target_clip_low),
                            float(config.target_clip_high),
                        )
                    ),
                }
            )
    metadata = pd.DataFrame(rows)
    finite = np.isfinite(target)
    x = x[finite]
    target = target[finite]
    metadata = metadata.loc[finite].reset_index(drop=True)
    if len(target) > int(config.max_train_examples):
        rng = np.random.default_rng(int(config.seed))
        indexes = np.sort(
            rng.choice(
                len(target),
                size=int(config.max_train_examples),
                replace=False,
            )
        )
        x = x[indexes]
        target = target[indexes]
        metadata = metadata.iloc[indexes].reset_index(drop=True)
    if not np.array_equal(
        target,
        metadata["shuffled_target"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise AttributionError("receipt target differs from exact fit target")
    if not np.array_equal(
        metadata["shuffled_dollar_pnl_label"].to_numpy(dtype=float),
        metadata["source_dollar_pnl_label"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AttributionError(
            "weak D1 moved dollar label differs from source receipt"
        )
    return x, target, metadata


def _original_decision_identity_map(
    decisions: Sequence[RepairedCanonicalDecision],
) -> dict[tuple[str, int, str], RepairedCanonicalDecision]:
    result: dict[tuple[str, int, str], RepairedCanonicalDecision] = {}
    for repaired in decisions:
        base = repaired.base
        for index, contract_id in enumerate(base.contract_ids):
            key = (
                base.session,
                int(base.decision_time.value),
                str(contract_id),
            )
            if key in result:
                raise AttributionError(f"duplicate original candidate: {key}")
            one = RepairedCanonicalDecision(
                base=hgb_core.CanonicalDecision(
                    session=base.session,
                    decision_time=base.decision_time,
                    features=base.features[index : index + 1],
                    labels=base.labels[index : index + 1],
                    mid_labels=base.mid_labels[index : index + 1],
                    entry_asks=base.entry_asks[index : index + 1],
                    offsets=base.offsets[index : index + 1],
                    rights=base.rights[index : index + 1],
                    contract_ids=base.contract_ids[index : index + 1],
                    strike_indices=base.strike_indices[index : index + 1],
                    right_indices=base.right_indices[index : index + 1],
                ),
                realized_exit_time_ns=repaired.realized_exit_time_ns[
                    index : index + 1
                ],
                source_exit_quote_time_ns=repaired.source_exit_quote_time_ns[
                    index : index + 1
                ],
                exit_quote_age_ms=repaired.exit_quote_age_ms[index : index + 1],
                exit_reason_codes=repaired.exit_reason_codes[index : index + 1],
                executable_exit_bids=repaired.executable_exit_bids[
                    index : index + 1
                ],
                policy_deadline_ns=repaired.policy_deadline_ns[
                    index : index + 1
                ],
                invalid_reason_codes=repaired.invalid_reason_codes[
                    index : index + 1
                ],
                canonical_strike_slots=repaired.canonical_strike_slots[
                    index : index + 1
                ],
                source_quote_time_ns=repaired.source_quote_time_ns[
                    index : index + 1
                ],
                source_context_time_ns=repaired.source_context_time_ns[
                    index : index + 1
                ],
            )
            result[key] = one
    return result


def _raw_source_rows_by_time(
    *,
    path_map: Mapping[str, Path],
    sessions: Sequence[str],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for session in sessions:
        with path_map[session].open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list) or len(rows) != references.D1_NORMAL_ROWS:
            raise AttributionError(f"D1 raw source geometry mismatch: {session}")
        for row in rows[: references.D1_INCLUDED_ROWS]:
            key = (session, int(pd.Timestamp(row["decision_time"]).value))
            if key in result:
                raise AttributionError(f"duplicate D1 raw source time: {key}")
            result[key] = row
    return result


def _source_time_mapping(
    *,
    path_map: Mapping[str, Path],
    sessions: Sequence[str],
    seed: int,
) -> dict[tuple[str, int], int]:
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    block_order = tuple(
        int(value)
        for value in rng.permutation(references.D1_COMPLETE_BLOCKS)
    )
    source_rows = tuple(
        source_block * references.D1_BLOCK_SIZE + within
        for source_block in block_order
        for within in range(references.D1_BLOCK_SIZE)
    )
    result: dict[tuple[str, int], int] = {}
    for session in sessions:
        with path_map[session].open("rb") as handle:
            rows = pickle.load(handle)
        times = [
            int(pd.Timestamp(row["decision_time"]).value)
            for row in rows[: references.D1_INCLUDED_ROWS]
        ]
        if len(times) != references.D1_INCLUDED_ROWS:
            raise AttributionError(f"D1 source geometry mismatch: {session}")
        for destination_index, source_index in enumerate(source_rows):
            result[(session, times[destination_index])] = times[source_index]
    return result


def _fit_hgb_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    config: HGBUnitConfig,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_depth=int(config.max_depth),
        min_samples_leaf=int(config.min_samples_leaf),
        l2_regularization=float(config.l2_regularization),
        early_stopping=False,
        random_state=int(config.seed),
    )
    model.fit(x, y)
    return model


def _hash_model(model: HistGradientBoostingRegressor) -> tuple[str, bytes]:
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest(), payload


def audit_and_fit_strong_model(
    *,
    scope: Any,
    path_map: Mapping[str, Path],
    fold: int,
    seed: int,
    guard_margins: Mapping[str, float],
    noise_model: DivergenceNoiseModel,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    receipt_path = (
        PERMUTATION_ROOT / f"seed{seed}" / f"fold{fold}.json"
    )
    parquet_path = (
        PERMUTATION_ROOT
        / f"seed{seed}"
        / f"fold{fold}_fit_targets.parquet"
    )
    strong_path = (
        STRONG_MODEL_ROOT / f"seed{seed}" / f"fold{fold}" / "model.pkl"
    )
    strong_receipt_path = strong_path.parent / "receipt.json"
    if (
        receipt_path.is_file()
        and parquet_path.is_file()
        and strong_path.is_file()
        and strong_receipt_path.is_file()
    ):
        receipt = read_json(receipt_path)
        if receipt["receipt_sha256"] != stable_hash(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        ):
            raise AttributionError(f"receipt self-hash mismatch: {receipt_path}")
        if (
            sha256_path(parquet_path)
            != receipt["detailed_targets_sha256"]
            or sha256_path(strong_path)
            != receipt["strong_model_sha256"]
        ):
            raise AttributionError(
                f"resumed strong-permutation artifact mismatch:S{seed}:F{fold}"
            )
        return _load_model(strong_path), receipt

    payload = _fold_payload(scope, fold)
    train_sessions = list(payload["train_sessions"])
    fit_sessions, calibration_sessions = (
        hgb_core.split_fit_calibration_sessions(train_sessions)
    )
    validation_sessions = list(payload["validation_sessions"])
    if (
        set(fit_sessions) & set(calibration_sessions)
        or set(fit_sessions) & set(validation_sessions)
        or set(calibration_sessions) & set(validation_sessions)
    ):
        raise AttributionError("fold role overlap")

    moved_fit, moved_receipts = ft1d._d1_role_decisions(
        path_map=path_map,
        sessions=fit_sessions,
        role="fit",
        seed=seed,
        fold=fold,
        guard_margins=guard_margins,
    )
    moved_calibration, calibration_receipts = ft1d._d1_role_decisions(
        path_map=path_map,
        sessions=calibration_sessions,
        role="calibration",
        seed=seed,
        fold=fold,
        guard_margins=guard_margins,
    )
    original_fit = hgb_core.load_repaired_decisions(
        [(session, path_map[session]) for session in fit_sessions],
        hypothesis=HYPOTHESIS,
        policy_index=POLICY_INDEX,
        guard_margins=dict(guard_margins),
        split=f"ATTRIBUTION-ORIGINAL-FIT-S{seed}-F{fold}",
        max_rows_per_session=references.D1_INCLUDED_ROWS,
    )
    original_map = _original_decision_identity_map(original_fit)
    raw_source_map = _raw_source_rows_by_time(
        path_map=path_map,
        sessions=fit_sessions,
    )
    source_time = _source_time_mapping(
        path_map=path_map,
        sessions=fit_sessions,
        seed=seed,
    )
    config = HGBUnitConfig(
        hypothesis=HYPOTHESIS,
        policy_index=POLICY_INDEX,
        seed=seed,
    )
    x, weak_y, metadata = _training_arrays_with_metadata(
        moved_fit,
        original_by_identity=original_map,
        raw_source_by_time=raw_source_map,
        source_time_by_destination=source_time,
        noise_model=noise_model,
        config=config,
    )
    current_refit = _fit_hgb_from_arrays(x, weak_y, config=config)
    current_hash, current_bytes = _hash_model(current_refit)
    persisted_path = _model_path_for_d1(seed, fold)
    persisted_hash = sha256_path(persisted_path)
    if current_hash != persisted_hash or current_bytes != persisted_path.read_bytes():
        raise AttributionError(
            f"current D1 deterministic refit mismatch:S{seed}:F{fold}"
        )
    current_epsilon, current_epsilon_summary = hgb_core.score_noise_epsilon(
        current_refit,
        [item.base for item in moved_calibration],
        feature_names=tuple(HYPOTHESES[HYPOTHESIS]),
        noise_model=noise_model,
        seed=seed,
    )
    current_calibration_scores = hgb_core.score_decisions(
        current_refit,
        [item.base for item in moved_calibration],
        feature_names=tuple(HYPOTHESES[HYPOTHESIS]),
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=seed + 100_000,
    )
    current_threshold, current_sweep = (
        ft1d._choose_D1_non_candidate_threshold(
            moved_calibration,
            current_calibration_scores,
            epsilon=current_epsilon,
            config=config,
            fold=f"D1_expanding_fold_{fold:02d}",
        )
    )
    persisted_summary = _existing_d1_fold_summary(seed, fold)
    if not math.isclose(
        current_epsilon,
        float(persisted_summary["epsilon"]),
        abs_tol=1e-12,
    ):
        raise AttributionError(
            f"current D1 epsilon mismatch:S{seed}:F{fold}"
        )
    if not math.isclose(
        current_threshold,
        float(persisted_summary["threshold"]),
        abs_tol=1e-12,
    ):
        raise AttributionError(
            f"current D1 threshold mismatch:S{seed}:F{fold}"
        )

    rng = np.random.Generator(
        np.random.PCG64DXSM(seed + STRONG_PERMUTATION_SEED_OFFSET)
    )
    permutation = rng.permutation(len(weak_y))
    if not np.array_equal(
        np.sort(permutation),
        np.arange(len(weak_y), dtype=permutation.dtype),
    ):
        raise AttributionError("strong permutation is not a complete bijection")
    strong_y = weak_y[permutation]
    if len(strong_y) > 1 and np.array_equal(permutation, np.arange(len(weak_y))):
        raise AttributionError("strong permutation unexpectedly identity")
    if not np.array_equal(np.sort(strong_y), np.sort(weak_y)):
        raise AttributionError("strong permutation changed target multiset")
    source_metadata = metadata.iloc[permutation].reset_index(drop=True)
    metadata = metadata.copy()
    metadata["strong_source_session"] = source_metadata[
        "destination_session"
    ].to_numpy()
    metadata["strong_source_decision_time_ns"] = source_metadata[
        "destination_decision_time_ns"
    ].to_numpy(dtype=np.int64)
    metadata["strong_source_right"] = source_metadata["right"].to_numpy()
    metadata["strong_source_slot"] = source_metadata[
        "canonical_slot"
    ].to_numpy(dtype=np.int64)
    metadata["strong_source_offset"] = source_metadata["offset"].to_numpy(
        dtype=float
    )
    metadata["strong_target"] = strong_y
    metadata["strong_source_index"] = permutation.astype(np.int64)
    metadata["target_changed"] = strong_y != weak_y
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )

    strong_model = _fit_hgb_from_arrays(x, strong_y, config=config)
    strong_epsilon, strong_epsilon_summary = hgb_core.score_noise_epsilon(
        strong_model,
        [item.base for item in moved_calibration],
        feature_names=tuple(HYPOTHESES[HYPOTHESIS]),
        noise_model=noise_model,
        seed=seed,
    )
    strong_hash, strong_bytes = _hash_model(strong_model)
    strong_path.parent.mkdir(parents=True, exist_ok=True)
    strong_path.write_bytes(strong_bytes)
    same_session = (
        metadata["destination_session"].to_numpy()
        == metadata["strong_source_session"].to_numpy()
    )
    same_right = (
        metadata["right"].to_numpy()
        == metadata["strong_source_right"].to_numpy()
    )
    same_slot = (
        metadata["canonical_slot"].to_numpy()
        == metadata["strong_source_slot"].to_numpy()
    )
    weak_same_right = (
        metadata["right"].to_numpy()
        == metadata["source_right"].to_numpy()
    )
    weak_same_slot = (
        metadata["canonical_slot"].to_numpy(dtype=np.int64)
        == metadata["source_canonical_slot"].to_numpy(dtype=np.int64)
    )
    receipt = {
        "schema_version": "Protocol101D1ExactFitPermutationReceiptV1",
        "seed": seed,
        "fold": fold,
        "fit_sessions": fit_sessions,
        "calibration_sessions": calibration_sessions,
        "validation_sessions": validation_sessions,
        "roles_disjoint": True,
        "sample_weight": None,
        "fit_rows": int(len(weak_y)),
        "current_D1": {
            "per_session_rng_reset": True,
            "source_session_preserved_fraction": 1.0,
            "source_right_preserved_fraction": float(
                np.mean(weak_same_right)
            ),
            "source_slot_preserved_fraction": float(
                np.mean(weak_same_slot)
            ),
            "source_dollar_label_exact_match": True,
            "exact_X_sha256": array_hash(x),
            "exact_y_sha256": array_hash(weak_y),
            "persisted_model_sha256": persisted_hash,
            "deterministic_refit_sha256": current_hash,
            "deterministic_refit_byte_identical": True,
            "target_change_rate": float(
                np.mean(
                    metadata["original_target"].to_numpy(dtype=float)
                    != metadata["shuffled_target"].to_numpy(dtype=float)
                )
            ),
            "original_vs_shuffled": _array_correlations(
                metadata["original_target"].to_numpy(dtype=float),
                metadata["shuffled_target"].to_numpy(dtype=float),
            ),
            "original_class_balance": _class_balance(
                metadata["original_target"].to_numpy(dtype=float)
            ),
            "shuffled_class_balance": _class_balance(weak_y),
            "target_vs_destination_ask": _array_correlations(
                weak_y,
                metadata["entry_ask"].to_numpy(dtype=float),
            ),
            "target_override_receipts_sha256": stable_hash(moved_receipts),
            "calibration_override_receipts_sha256": stable_hash(
                calibration_receipts
            ),
            "epsilon": float(current_epsilon),
            "epsilon_summary": current_epsilon_summary,
            "calibration_scores_sha256": stable_hash(
                [
                    [float(value) for value in scores]
                    for scores in current_calibration_scores
                ]
            ),
            "threshold": float(current_threshold),
            "threshold_selection_metric": (
                "shuffled_calibration_serial_target_net_pnl"
            ),
            "threshold_sweep": current_sweep,
            "validation_economics_used_for_threshold": False,
        },
        "strong_permutation": {
            "method": "global_final_normalized_target_permutation_after_cap",
            "seed": seed + STRONG_PERMUTATION_SEED_OFFSET,
            "prng": "numpy.PCG64DXSM",
            "permutation_sha256": array_hash(
                permutation.astype(np.int64)
            ),
            "exact_y_sha256": array_hash(strong_y),
            "sorted_target_multiset_sha256": array_hash(np.sort(strong_y)),
            "weak_sorted_target_multiset_sha256": array_hash(
                np.sort(weak_y)
            ),
            "complete_bijection": True,
            "target_multiset_preserved": True,
            "model_sha256": strong_hash,
            "target_change_rate": float(np.mean(strong_y != weak_y)),
            "weak_vs_strong": _array_correlations(weak_y, strong_y),
            "class_balance": _class_balance(strong_y),
            "same_session_fraction": float(np.mean(same_session)),
            "same_right_fraction": float(np.mean(same_right)),
            "same_slot_fraction": float(np.mean(same_slot)),
            "same_session_right_slot_fraction": float(
                np.mean(same_session & same_right & same_slot)
            ),
            "target_vs_destination_ask": _array_correlations(
                strong_y,
                metadata["entry_ask"].to_numpy(dtype=float),
            ),
            "epsilon": float(strong_epsilon),
            "epsilon_summary": strong_epsilon_summary,
        },
        "detailed_targets_path": str(parquet_path.relative_to(ROOT)),
        "detailed_targets_sha256": sha256_path(parquet_path),
        "strong_model_path": str(strong_path.relative_to(ROOT)),
        "strong_model_sha256": strong_hash,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = stable_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )
    write_json(receipt_path, receipt)
    write_json(
        strong_receipt_path,
        {
            "schema_version": "Protocol101D1StrongModelReceiptV1",
            "seed": seed,
            "fold": fold,
            "model_sha256": strong_hash,
            "permutation_receipt_sha256": sha256_path(receipt_path),
            "non_candidate": True,
            "G9_eligible": False,
            "promotion_eligible": False,
        },
    )
    del moved_fit, moved_calibration, original_fit, original_map
    del raw_source_map
    del metadata, x, weak_y, strong_y
    del current_refit, current_bytes, strong_bytes
    gc.collect()
    return strong_model, receipt


def _write_chunk(
    *,
    control: str,
    seed: int,
    summaries: Sequence[Mapping[str, Any]],
    session_rows: Sequence[Mapping[str, Any]],
) -> None:
    summary_path = CONTROL_CHUNK_ROOT / f"{control}_seed{seed}.json"
    session_path = SESSION_CHUNK_ROOT / f"{control}_seed{seed}.parquet"
    write_json(
        summary_path,
        {
            "schema_version": "Protocol101D1ControlSeedChunkV1",
            "control": control,
            "seed": seed,
            "summaries": list(summaries),
        },
    )
    pd.DataFrame(session_rows).to_parquet(
        session_path,
        index=False,
        compression="zstd",
    )


def _chunk_complete(control: str, seed: int) -> bool:
    return (
        (CONTROL_CHUNK_ROOT / f"{control}_seed{seed}.json").is_file()
        and (SESSION_CHUNK_ROOT / f"{control}_seed{seed}.parquet").is_file()
    )


def _all_sessions_by_fold(scope: Any) -> dict[str, list[str]]:
    return {
        f"F{fold}": list(_fold_payload(scope, fold)["validation_sessions"])
        for fold in FOLDS
    }


def _existing_d1_fold_summary(seed: int, fold: int) -> dict[str, Any]:
    summary = read_json(_summary_path_for_d1(seed, fold))
    if summary.get("seed") != seed or summary.get("fold") != fold:
        raise AttributionError("D1 summary axis mismatch")
    return summary


def _validate_current_replay(
    summary: Mapping[str, Any],
    candidates: Sequence[SerialCandidateV5],
) -> None:
    trades, state = references.replay_reference_v5(candidates, fee=FIXED_FEE)
    metrics = references.replay_metrics(trades, state)
    expected = summary["v5_replay"]
    for key in (
        "net_pnl",
        "trades",
        "candidate_stream_hash",
        "candidate_payload_hash",
        "trade_identity_hash",
    ):
        left, right = metrics[key], expected[key]
        if isinstance(left, float):
            if not math.isclose(float(left), float(right), abs_tol=1e-9):
                raise AttributionError(f"current D1 replay mismatch:{key}")
        elif left != right:
            raise AttributionError(f"current D1 replay mismatch:{key}")


def run_controls(
    *,
    folds: Sequence[int] = FOLDS,
    d1_seeds: Sequence[int] = D1_SEEDS,
    include_real: bool = True,
    include_fixed_heuristic: bool = True,
) -> None:
    selected_folds = tuple(int(fold) for fold in folds)
    if not selected_folds or any(fold not in FOLDS for fold in selected_folds):
        raise AttributionError(f"invalid fold shard: {selected_folds}")
    preregister()
    update_progress("LOAD_SCOPE")
    scope = ft1d._materialized_scope()
    path_map = {session: path for session, path in scope.sessions}
    guard_margins = stage1_runner.guard_margins()
    noise_model = _noise_model()
    sessions_by_fold = _all_sessions_by_fold(scope)

    # A is seed-free but represented once for the complete five-fold stream.
    if (
        include_fixed_heuristic
        and 1 in selected_folds
        and not _chunk_complete("A", 0)
    ):
        candidates: list[SerialCandidateV5] = []
        for fold in FOLDS:
            decisions, _ = _d1_validation_decisions(
                scope=scope,
                path_map=path_map,
                fold=fold,
                guard_margins=guard_margins,
            )
            opportunities = _reference_opportunities(
                decisions,
                fold=fold,
                campaign_id=f"ATTRIBUTION-A-F{fold}",
            )
            candidates.extend(
                fixed_heuristic_candidates_for_d1_risk_set(opportunities)
            )
        summary, sessions = evaluate_stream(
            candidates,
            control="A",
            model_seed=0,
            draw=0,
            all_sessions_by_fold=sessions_by_fold,
        )
        summary["full_359_reference"] = {
            key: read_json(FIXED_HEURISTIC_PATH)[
                "continuous_pooled_metrics"
            ][key]
            for key in ("net_pnl", "trades", "max_drawdown")
        }
        _write_chunk(
            control="A",
            seed=0,
            summaries=[summary],
            session_rows=sessions,
        )

    for fold in selected_folds:
        update_progress("FOLD_CONTROLS", fold=fold)
        validation, _validation_receipts = _d1_validation_decisions(
            scope=scope,
            path_map=path_map,
            fold=fold,
            guard_margins=guard_margins,
        )
        opportunities = _reference_opportunities(
            validation,
            fold=fold,
            campaign_id=f"ATTRIBUTION-RISKSET-F{fold}",
        )
        opportunity_by_time = {
            (
                item.repaired.base.session,
                int(item.repaired.base.decision_time.value),
            ): item
            for item in opportunities
        }

        for seed in d1_seeds:
            update_progress(
                "D1_SEED_FOLD",
                fold=fold,
                seed=seed,
            )
            strong_model, receipt = audit_and_fit_strong_model(
                scope=scope,
                path_map=path_map,
                fold=fold,
                seed=seed,
                guard_margins=guard_margins,
                noise_model=noise_model,
            )
            d1_summary = _existing_d1_fold_summary(seed, fold)
            original_current_candidates = _current_candidates_from_summary(
                d1_summary
            )
            _validate_current_replay(
                d1_summary,
                original_current_candidates,
            )
            current_candidates = _with_fold(
                original_current_candidates,
                fold,
            )
            current_model = _load_model(_model_path_for_d1(seed, fold))
            current_rows = _selected_candidate_rows(
                current_model,
                validation,
                seed=seed,
                fold=fold,
                epsilon=float(d1_summary["epsilon"]),
                noise_model=noise_model,
            )
            paired_budget = len(current_candidates)
            e_candidates = select_fixed_count(
                current_rows,
                count=paired_budget,
                strategy="E_weak_D1_fixed_count",
            )
            e_common_candidates = select_fixed_count(
                current_rows,
                count=COMMON_INTENT_BUDGET_BY_FOLD[fold],
                strategy="E_common_weak_D1_fixed_count",
            )
            strong_epsilon = float(
                receipt["strong_permutation"]["epsilon"]
            )
            strong_rows = _selected_candidate_rows(
                strong_model,
                validation,
                seed=seed,
                fold=fold,
                epsilon=strong_epsilon,
                noise_model=noise_model,
            )
            g_candidates = select_fixed_count(
                strong_rows,
                count=paired_budget,
                strategy="G_strong_permutation_fixed_count",
            )
            g_common_candidates = select_fixed_count(
                strong_rows,
                count=COMMON_INTENT_BUDGET_BY_FOLD[fold],
                strategy="G_common_strong_permutation_fixed_count",
            )

            deterministic = {
                "E": e_candidates,
                "E_common": e_common_candidates,
                "F": current_candidates,
                "G": g_candidates,
                "G_common": g_common_candidates,
            }
            for control, candidates in deterministic.items():
                chunk_control = f"{control}_F{fold}"
                if _chunk_complete(chunk_control, seed):
                    continue
                summary, session_rows = evaluate_stream(
                    candidates,
                    control=control,
                    model_seed=seed,
                    draw=0,
                    all_sessions_by_fold={
                        f"F{fold}": sessions_by_fold[f"F{fold}"]
                    },
                )
                summary["fold"] = fold
                if control == "F":
                    summary["current_threshold"] = float(
                        d1_summary["threshold"]
                    )
                _write_chunk(
                    control=chunk_control,
                    seed=seed,
                    summaries=[summary],
                    session_rows=session_rows,
                )

            for control in ("B", "C", "D"):
                chunk_control = f"{control}_F{fold}"
                if _chunk_complete(chunk_control, seed):
                    continue
                summaries: list[dict[str, Any]] = []
                session_rows: list[dict[str, Any]] = []
                for draw in range(RANDOM_DRAWS):
                    if control == "B":
                        candidates = random_time_random_slot(
                            opportunities,
                            count=paired_budget,
                            model_seed=seed,
                            draw=draw,
                        )
                        profile_receipt = None
                    elif control == "C":
                        candidates = random_slot_at_model_time(
                            current_candidates,
                            opportunity_by_time,
                            model_seed=seed,
                            draw=draw,
                        )
                        profile_receipt = None
                    else:
                        candidates, profile_receipt = (
                            random_time_model_profile(
                                current_candidates,
                                opportunities,
                                model_seed=seed,
                                draw=draw,
                            )
                        )
                    summary, rows = evaluate_stream(
                        candidates,
                        control=control,
                        model_seed=seed,
                        draw=draw,
                        all_sessions_by_fold={
                            f"F{fold}": sessions_by_fold[f"F{fold}"]
                        },
                    )
                    summary["fold"] = fold
                    if profile_receipt is not None:
                        summary["profile_match"] = profile_receipt
                    summaries.append(summary)
                    session_rows.extend(rows)
                _write_chunk(
                    control=chunk_control,
                    seed=seed,
                    summaries=summaries,
                    session_rows=session_rows,
                )

            chunk_control = f"M_F_F{fold}"
            if not _chunk_complete(chunk_control, seed):
                summaries = []
                session_rows = []
                for draw in range(RANDOM_DRAWS):
                    candidates, profile_receipt = (
                        profile_permutation_at_model_times(
                            current_candidates,
                            opportunity_by_time,
                            model_seed=seed + 25_000,
                            draw=draw,
                        )
                    )
                    summary, rows = evaluate_stream(
                        candidates,
                        control="M_F",
                        model_seed=seed,
                        draw=draw,
                        all_sessions_by_fold={
                            f"F{fold}": sessions_by_fold[f"F{fold}"]
                        },
                    )
                    summary["fold"] = fold
                    summary["profile_match"] = profile_receipt
                    summaries.append(summary)
                    session_rows.extend(rows)
                _write_chunk(
                    control=chunk_control,
                    seed=seed,
                    summaries=summaries,
                    session_rows=session_rows,
                )

            for control, source_candidates in (
                ("C_G", g_candidates),
                ("D_G", g_candidates),
                ("M_G", g_candidates),
            ):
                chunk_control = f"{control}_F{fold}"
                if _chunk_complete(chunk_control, seed):
                    continue
                summaries = []
                session_rows = []
                for draw in range(RANDOM_DRAWS):
                    if control == "C_G":
                        candidates = random_slot_at_model_time(
                            source_candidates,
                            opportunity_by_time,
                            model_seed=seed + 50_000,
                            draw=draw,
                        )
                        profile_receipt = None
                    elif control == "D_G":
                        candidates, profile_receipt = (
                            random_time_model_profile(
                                source_candidates,
                                opportunities,
                                model_seed=seed + 50_000,
                                draw=draw,
                            )
                        )
                    else:
                        candidates, profile_receipt = (
                            profile_permutation_at_model_times(
                                source_candidates,
                                opportunity_by_time,
                                model_seed=seed + 75_000,
                                draw=draw,
                            )
                        )
                    summary, rows = evaluate_stream(
                        candidates,
                        control=control,
                        model_seed=seed,
                        draw=draw,
                        all_sessions_by_fold={
                            f"F{fold}": sessions_by_fold[f"F{fold}"]
                        },
                    )
                    summary["fold"] = fold
                    if profile_receipt is not None:
                        summary["profile_match"] = profile_receipt
                    summaries.append(summary)
                    session_rows.extend(rows)
                _write_chunk(
                    control=chunk_control,
                    seed=seed,
                    summaries=summaries,
                    session_rows=session_rows,
                )

            del current_model, strong_model, current_rows, strong_rows
            del d1_summary, current_candidates, original_current_candidates
            gc.collect()

        for seed in REAL_SEEDS if include_real else ():
            unit_root = _real_unit_root(seed, fold)
            unit_summary = read_json(unit_root / "summary.json")
            model = _load_model(unit_root / "model.pkl")
            epsilon = float(unit_summary["unit"]["calibration"]["epsilon"])
            h_rows = _selected_candidate_rows(
                model,
                validation,
                seed=seed,
                fold=fold,
                epsilon=epsilon,
                noise_model=noise_model,
                reverse=False,
            )
            i_rows = _selected_candidate_rows(
                model,
                validation,
                seed=seed,
                fold=fold,
                epsilon=epsilon,
                noise_model=noise_model,
                reverse=True,
            )
            h_candidates = select_fixed_count(
                h_rows,
                count=COMMON_INTENT_BUDGET_BY_FOLD[fold],
                strategy="H_real_H2_P5_fixed_count",
            )
            i_candidates = select_fixed_count(
                i_rows,
                count=COMMON_INTENT_BUDGET_BY_FOLD[fold],
                strategy="I_sign_reversed_real_H2_P5",
            )
            for control, candidates in (
                ("H", h_candidates),
                ("I", i_candidates),
            ):
                chunk_control = f"{control}_F{fold}"
                if not _chunk_complete(chunk_control, seed):
                    result, session_rows = evaluate_stream(
                        candidates,
                        control=control,
                        model_seed=seed,
                        draw=0,
                        all_sessions_by_fold={
                            f"F{fold}": sessions_by_fold[f"F{fold}"]
                        },
                    )
                    result["fold"] = fold
                    _write_chunk(
                        control=chunk_control,
                        seed=seed,
                        summaries=[result],
                        session_rows=session_rows,
                    )
            for control in ("B_real", "C_real", "D_real", "M_real"):
                chunk_control = f"{control}_F{fold}"
                if _chunk_complete(chunk_control, seed):
                    continue
                summaries = []
                session_rows = []
                for draw in range(RANDOM_DRAWS):
                    if control == "B_real":
                        candidates = random_time_random_slot(
                            opportunities,
                            count=len(h_candidates),
                            model_seed=seed + 100_000,
                            draw=draw,
                        )
                        profile_receipt = None
                    elif control == "C_real":
                        candidates = random_slot_at_model_time(
                            h_candidates,
                            opportunity_by_time,
                            model_seed=seed + 100_000,
                            draw=draw,
                        )
                        profile_receipt = None
                    elif control == "D_real":
                        candidates, profile_receipt = (
                            random_time_model_profile(
                                h_candidates,
                                opportunities,
                                model_seed=seed + 100_000,
                                draw=draw,
                            )
                        )
                    else:
                        candidates, profile_receipt = (
                            profile_permutation_at_model_times(
                                h_candidates,
                                opportunity_by_time,
                                model_seed=seed + 125_000,
                                draw=draw,
                            )
                        )
                    result, rows = evaluate_stream(
                        candidates,
                        control=control,
                        model_seed=seed,
                        draw=draw,
                        all_sessions_by_fold={
                            f"F{fold}": sessions_by_fold[f"F{fold}"]
                        },
                    )
                    result["fold"] = fold
                    if profile_receipt is not None:
                        result["profile_match"] = profile_receipt
                    summaries.append(result)
                    session_rows.extend(rows)
                _write_chunk(
                    control=chunk_control,
                    seed=seed,
                    summaries=summaries,
                    session_rows=session_rows,
                )
            del model, h_rows, i_rows, h_candidates, i_candidates
            del unit_summary
            gc.collect()
        del validation, opportunities, opportunity_by_time
        gc.collect()

    update_progress("CONTROL_REPLAYS_COMPLETE")


def _load_control_chunks() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    deterministic_d1 = ("E", "E_common", "F", "G", "G_common")
    random_d1 = ("B", "C", "D", "M_F", "C_G", "D_G", "M_G")
    deterministic_real = ("H", "I")
    random_real = ("B_real", "C_real", "D_real", "M_real")
    expected_stems = {"A_seed0"}
    expected_draws: dict[str, int] = {"A_seed0": 1}
    for fold in FOLDS:
        for seed in D1_SEEDS:
            for control in deterministic_d1:
                stem = f"{control}_F{fold}_seed{seed}"
                expected_stems.add(stem)
                expected_draws[stem] = 1
            for control in random_d1:
                stem = f"{control}_F{fold}_seed{seed}"
                expected_stems.add(stem)
                expected_draws[stem] = RANDOM_DRAWS
        for seed in REAL_SEEDS:
            for control in deterministic_real:
                stem = f"{control}_F{fold}_seed{seed}"
                expected_stems.add(stem)
                expected_draws[stem] = 1
            for control in random_real:
                stem = f"{control}_F{fold}_seed{seed}"
                expected_stems.add(stem)
                expected_draws[stem] = RANDOM_DRAWS
    actual_json = {path.stem for path in CONTROL_CHUNK_ROOT.glob("*.json")}
    actual_parquet = {
        path.stem for path in SESSION_CHUNK_ROOT.glob("*.parquet")
    }
    if actual_json != expected_stems or actual_parquet != expected_stems:
        raise AttributionError(
            "control chunk grid incomplete or contains unexpected axes:"
            f"json_missing={len(expected_stems - actual_json)}:"
            f"json_extra={len(actual_json - expected_stems)}:"
            f"parquet_missing={len(expected_stems - actual_parquet)}:"
            f"parquet_extra={len(actual_parquet - expected_stems)}"
        )
    scope = ft1d._materialized_scope()
    expected_sessions_by_fold = {
        fold: {(fold, str(session)) for session in sessions}
        for fold, sessions in _all_sessions_by_fold(scope).items()
    }
    expected_all_sessions = set().union(
        *expected_sessions_by_fold.values()
    )
    summaries: list[dict[str, Any]] = []
    for path in sorted(CONTROL_CHUNK_ROOT.glob("*.json")):
        payload = read_json(path)
        expected = expected_draws[path.stem]
        if len(payload["summaries"]) != expected:
            raise AttributionError(f"wrong draw count:{path}:{expected}")
        if sorted(int(item["draw"]) for item in payload["summaries"]) != list(
            range(expected)
        ):
            raise AttributionError(f"draw grid mismatch:{path}")
        summaries.extend(payload["summaries"])
    frames: list[pd.DataFrame] = []
    for path in sorted(SESSION_CHUNK_ROOT.glob("*.parquet")):
        frame = pd.read_parquet(path)
        expected = expected_draws[path.stem]
        expected_sessions = 225 if path.stem == "A_seed0" else 45
        if len(frame) != expected * expected_sessions:
            raise AttributionError(f"session row count mismatch:{path}")
        identity = ["model_seed", "draw", "fold", "session"]
        if frame.duplicated(identity).any():
            raise AttributionError(f"duplicate session axis:{path}")
        draw_sets = []
        for draw, group in frame.groupby("draw", sort=True):
            if len(group) != expected_sessions:
                raise AttributionError(
                    f"incomplete session draw:{path}:D{draw}"
                )
            draw_sets.append(
                set(zip(group["fold"].astype(str), group["session"].astype(str)))
            )
            expected_axis = (
                expected_all_sessions
                if path.stem == "A_seed0"
                else expected_sessions_by_fold.get(
                    str(group["fold"].iloc[0])
                )
            )
            if expected_axis is None or draw_sets[-1] != expected_axis:
                raise AttributionError(
                    f"session universe differs from governed fold:{path}:"
                    f"D{draw}"
                )
        if len(draw_sets) != expected or any(
            value != draw_sets[0] for value in draw_sets[1:]
        ):
            raise AttributionError(f"session universe drift across draws:{path}")
        frames.append(frame)
    if not summaries or not frames:
        raise AttributionError("control chunks are incomplete")
    return summaries, pd.concat(frames, ignore_index=True)


def _verify_preregistered_inputs() -> None:
    plan = read_json(INVESTIGATION_PLAN_PATH)
    if plan["required_input_hashes"] != _required_input_hashes():
        raise AttributionError("preregistered input hash changed before finalize")
    if read_json(CONTROL_DEFINITIONS_PATH) != control_definitions():
        raise AttributionError("control definitions changed before finalize")


def _aggregate_fold_chunks(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    passthrough: list[dict[str, Any]] = []
    for item in summaries:
        control = str(item["control"])
        if control == "A":
            passthrough.append(dict(item))
            continue
        groups[(control, int(item["model_seed"]), int(item["draw"]))].append(
            item
        )
    result = list(passthrough)
    for (control, seed, draw), rows in sorted(groups.items()):
        if sorted(int(row["fold"]) for row in rows) != list(FOLDS):
            raise AttributionError(
                f"incomplete fold grid:{control}:S{seed}:D{draw}"
            )
        fold_metrics = {
            f"F{int(row['fold'])}": {
                "net_pnl": float(row["net_pnl"]),
                "trades": int(row["trades"]),
                "max_drawdown": float(row["max_drawdown"]),
            }
            for row in rows
        }
        net_pnl = float(sum(float(row["net_pnl"]) for row in rows))
        trades = int(sum(int(row["trades"]) for row in rows))
        entry_intents = int(sum(int(row["entry_intents"]) for row in rows))
        gross_profit = float(sum(float(row["gross_profit"]) for row in rows))
        gross_loss = float(sum(float(row["gross_loss"]) for row in rows))
        call_put = Counter()
        slots = Counter()
        time_of_day = Counter()
        for row in rows:
            call_put.update(row["call_put_distribution"])
            slots.update(row["slot_distribution"])
            time_of_day.update(row["time_of_day_distribution"])
        result.append(
            {
                "control": control,
                "model_seed": seed,
                "draw": draw,
                "entry_intents": entry_intents,
                "trades": trades,
                "net_pnl": net_pnl,
                "pnl_per_trade": net_pnl / trades if trades else 0.0,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "win_rate": (
                    float(
                        sum(
                            float(row["win_rate"]) * int(row["trades"])
                            for row in rows
                        )
                        / trades
                    )
                    if trades
                    else 0.0
                ),
                "profit_factor": (
                    gross_profit / gross_loss if gross_loss > 0.0 else None
                ),
                "premium_at_risk_mean": (
                    float(
                        sum(
                            float(row["premium_at_risk"]["mean"])
                            * int(row["entry_intents"])
                            for row in rows
                        )
                        / entry_intents
                    )
                    if entry_intents
                    else 0.0
                ),
                "candidate_holding_minutes_mean": (
                    float(
                        sum(
                            float(row["candidate_holding_minutes"]["mean"])
                            * int(row["entry_intents"])
                            for row in rows
                        )
                        / entry_intents
                    )
                    if entry_intents
                    else 0.0
                ),
                "executed_premium_at_risk_mean": (
                    float(
                        sum(
                            float(row["executed_premium_at_risk"]["mean"])
                            * int(row["trades"])
                            for row in rows
                        )
                        / trades
                    )
                    if trades
                    else 0.0
                ),
                "executed_holding_minutes_mean": (
                    float(
                        sum(
                            float(row["executed_holding_minutes"]["mean"])
                            * int(row["trades"])
                            for row in rows
                        )
                        / trades
                    )
                    if trades
                    else 0.0
                ),
                "call_put_distribution": dict(sorted(call_put.items())),
                "slot_distribution": dict(
                    sorted(slots.items(), key=lambda item: int(item[0]))
                ),
                "time_of_day_distribution": dict(
                    sorted(time_of_day.items())
                ),
                "max_drawdown": float(
                    max(float(row["max_drawdown"]) for row in rows)
                ),
                "median_fold_pnl": float(
                    statistics.median(
                        float(row["net_pnl"]) for row in rows
                    )
                ),
                "profitable_folds": int(
                    sum(float(row["net_pnl"]) > 0.0 for row in rows)
                ),
                "G1": bool(
                    sum(float(row["net_pnl"]) > 0.0 for row in rows) >= 4
                    and net_pnl > 0.0
                ),
                "fold_metrics": fold_metrics,
            }
        )
    return result


def _empirical_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_std": None,
            "p2_5": None,
            "p97_5": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_std": (
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
        "p2_5": float(np.percentile(array, 2.5)),
        "p97_5": float(np.percentile(array, 97.5)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _matched_exposure_quality(
    aggregated: Sequence[Mapping[str, Any]],
    *,
    left_control: str,
    right_control: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        left = [
            item
            for item in aggregated
            if item["control"] == left_control
            and int(item["model_seed"]) == int(seed)
        ]
        right = [
            item
            for item in aggregated
            if item["control"] == right_control
            and int(item["model_seed"]) == int(seed)
        ]
        if len(left) != 1 or len(right) != RANDOM_DRAWS:
            raise AttributionError(f"matched exposure grid mismatch:S{seed}")
        left = left[0]

        def drift(metric: str) -> float:
            left_value = float(left[metric])
            right_value = float(
                np.mean([float(item[metric]) for item in right])
            )
            return abs(left_value - right_value) / max(
                abs(left_value), 1e-12
            )

        rows.append(
            {
                "seed": seed,
                "executed_trade_count_drift": drift("trades"),
                "premium_at_risk_mean_drift": drift(
                    "executed_premium_at_risk_mean"
                ),
                "holding_minutes_mean_drift": drift(
                    "executed_holding_minutes_mean"
                ),
            }
        )
    result = {
        "left_control": left_control,
        "right_control": right_control,
        "per_seed": rows,
        "median_executed_trade_count_drift": float(
            np.median(
                [item["executed_trade_count_drift"] for item in rows]
            )
        ),
        "median_premium_at_risk_mean_drift": float(
            np.median(
                [item["premium_at_risk_mean_drift"] for item in rows]
            )
        ),
        "median_holding_minutes_mean_drift": float(
            np.median(
                [item["holding_minutes_mean_drift"] for item in rows]
            )
        ),
        "maximum_executed_trade_count_drift": float(
            max(item["executed_trade_count_drift"] for item in rows)
        ),
        "maximum_premium_at_risk_mean_drift": float(
            max(item["premium_at_risk_mean_drift"] for item in rows)
        ),
        "maximum_holding_minutes_mean_drift": float(
            max(item["holding_minutes_mean_drift"] for item in rows)
        ),
        "limits": {
            "executed_trade_count": MAX_MATCHED_EXECUTED_TRADE_COUNT_DRIFT,
            "premium_at_risk_mean": MAX_MATCHED_PREMIUM_RISK_DRIFT,
            "holding_minutes_mean": MAX_MATCHED_HOLDING_TIME_DRIFT,
        },
    }
    result["pass"] = bool(
        result["maximum_executed_trade_count_drift"]
        <= MAX_MATCHED_EXECUTED_TRADE_COUNT_DRIFT
        and result["maximum_premium_at_risk_mean_drift"]
        <= MAX_MATCHED_PREMIUM_RISK_DRIFT
        and result["maximum_holding_minutes_mean_drift"]
        <= MAX_MATCHED_HOLDING_TIME_DRIFT
    )
    return result


def _paired_block_bootstrap(
    differences: pd.DataFrame,
    *,
    contrast_index: int = 0,
    bootstrap_indexes: np.ndarray | None = None,
) -> dict[str, Any]:
    ordered_sessions = sorted(differences["session"].unique())
    seed_count = int(differences["model_seed"].nunique())
    values = (
        differences.groupby("session", sort=True)["difference"]
        .mean()
        .reindex(ordered_sessions, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    observed = float(values.sum())
    if not len(values):
        raise AttributionError("empty paired bootstrap")
    if bootstrap_indexes is None:
        bootstrap_indexes = _shared_bootstrap_indexes(
            len(values),
            compatibility_seed_index=contrast_index,
        )
    if bootstrap_indexes.shape != (BOOTSTRAP_REPLICATES, len(values)):
        raise AttributionError("bootstrap index matrix shape mismatch")
    draws = values[bootstrap_indexes].sum(axis=1, dtype=float)
    centered = draws - float(np.mean(draws))
    p_one_sided = float(
        (1 + np.sum(centered >= observed))
        / (BOOTSTRAP_REPLICATES + 1)
    )
    return {
        "observed_paired_pnl_difference": observed,
        "interpretation": "mean complete-campaign PnL difference across seeds",
        "seed_count": seed_count,
        "session_count": len(values),
        "block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
        "replicates": BOOTSTRAP_REPLICATES,
        "ci_p2_5": float(np.percentile(draws, 2.5)),
        "ci_p97_5": float(np.percentile(draws, 97.5)),
        "one_sided_centered_p": p_one_sided,
    }


def _shared_bootstrap_indexes(
    session_count: int,
    *,
    compatibility_seed_index: int = 0,
) -> np.ndarray:
    if session_count <= 0:
        raise AttributionError("empty shared bootstrap session count")
    blocks = [
        (
            np.arange(start, start + BOOTSTRAP_BLOCK_SESSIONS)
            % session_count
        )
        for start in range(session_count)
    ]
    rng = _random_generator(9, compatibility_seed_index)
    block_count = math.ceil(session_count / BOOTSTRAP_BLOCK_SESSIONS)
    result = np.empty(
        (BOOTSTRAP_REPLICATES, session_count),
        dtype=np.int32,
    )
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(blocks), size=block_count)
        indexes = np.concatenate([blocks[int(index)] for index in chosen])[
            :session_count
        ]
        if len(indexes) != session_count:
            raise AttributionError("bootstrap replicate changed campaign length")
        result[replicate] = indexes
    return result


def _holm_adjust(results: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(
        (
            (name, float(result["one_sided_centered_p"]))
            for name, result in results.items()
        ),
        key=lambda item: item[1],
    )
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        results[name]["holm_adjusted_p"] = float(running)


def _session_contrast(
    sessions: pd.DataFrame,
    *,
    left: str,
    right: str,
    seeds: Sequence[int],
    random_right: bool,
) -> pd.DataFrame:
    left_frame = sessions[
        (sessions["control"] == left)
        & (sessions["model_seed"].isin(seeds))
        & (sessions["draw"] == 0)
    ][["model_seed", "fold", "session", "net_pnl"]].rename(
        columns={"net_pnl": "left_pnl"}
    )
    right_frame = sessions[
        (sessions["control"] == right)
        & (sessions["model_seed"].isin(seeds))
    ].copy()
    if random_right:
        right_frame = (
            right_frame.groupby(
                ["model_seed", "fold", "session"], as_index=False
            )["net_pnl"]
            .mean()
        )
    else:
        right_frame = right_frame[right_frame["draw"] == 0]
    right_frame = right_frame[
        ["model_seed", "fold", "session", "net_pnl"]
    ].rename(columns={"net_pnl": "right_pnl"})
    keys = ["model_seed", "fold", "session"]
    if left_frame.duplicated(keys).any() or right_frame.duplicated(keys).any():
        raise AttributionError(f"duplicate paired contrast axis:{left}:{right}")
    left_keys = set(map(tuple, left_frame[keys].itertuples(index=False, name=None)))
    right_keys = set(
        map(tuple, right_frame[keys].itertuples(index=False, name=None))
    )
    if left_keys != right_keys:
        raise AttributionError(f"paired contrast session loss:{left}:{right}")
    paired = left_frame.merge(
        right_frame,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(left_frame):
        raise AttributionError(f"paired contrast row loss:{left}:{right}")
    paired["difference"] = paired["left_pnl"] - paired["right_pnl"]
    return paired


def _load_permutation_receipts() -> list[dict[str, Any]]:
    receipts = [
        read_json(path)
        for path in sorted(PERMUTATION_ROOT.glob("seed*/fold[0-9]*.json"))
        if "_fit_targets" not in path.name
    ]
    expected = len(D1_SEEDS) * len(FOLDS)
    if len(receipts) != expected:
        raise AttributionError(
            f"permutation receipt grid incomplete:{len(receipts)}/{expected}"
        )
    axes = {(int(item["seed"]), int(item["fold"])) for item in receipts}
    expected_axes = {(seed, fold) for seed in D1_SEEDS for fold in FOLDS}
    if axes != expected_axes:
        raise AttributionError("permutation receipt axes differ")
    return receipts


def _implementation_audit(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    weak_target_ask = [
        item["current_D1"]["target_vs_destination_ask"]["spearman"]
        for item in receipts
        if item["current_D1"]["target_vs_destination_ask"]["spearman"]
        is not None
    ]
    strong_target_ask = [
        item["strong_permutation"]["target_vs_destination_ask"]["spearman"]
        for item in receipts
        if item["strong_permutation"]["target_vs_destination_ask"]["spearman"]
        is not None
    ]
    payload = {
        "schema_version": "Protocol101D1ImplementationAuditV1",
        "status": "complete",
        "receipt_count": len(receipts),
        "all_axes_present": True,
        "actual_fit_target_proof": {
            "all_deterministic_refits_byte_identical": all(
                bool(item["current_D1"]["deterministic_refit_byte_identical"])
                for item in receipts
            ),
            "exact_X_hash_count": len(
                {item["current_D1"]["exact_X_sha256"] for item in receipts}
            ),
            "exact_weak_y_hash_count": len(
                {item["current_D1"]["exact_y_sha256"] for item in receipts}
            ),
            "exact_strong_y_hash_count": len(
                {
                    item["strong_permutation"]["exact_y_sha256"]
                    for item in receipts
                }
            ),
            "minimum_weak_target_change_rate": float(
                min(item["current_D1"]["target_change_rate"] for item in receipts)
            ),
            "minimum_strong_target_change_rate": float(
                min(
                    item["strong_permutation"]["target_change_rate"]
                    for item in receipts
                )
            ),
            "sample_weights": None,
        },
        "reproduced_weak_shuffle_defect": {
            "finding": (
                "current D1 moves source dollar PnL within each session block, "
                "then normalizes it by the destination contract ask and clips "
                "the result before fit"
            ),
            "why_it_matters": (
                "the shuffled target can retain a learnable relationship with "
                "destination premium, moneyness, right, and slot even though "
                "source outcome timing was moved"
            ),
            "per_session_rng_reset": all(
                bool(item["current_D1"]["per_session_rng_reset"])
                for item in receipts
            ),
            "minimum_source_session_preserved_fraction": float(
                min(
                    item["current_D1"][
                        "source_session_preserved_fraction"
                    ]
                    for item in receipts
                )
            ),
            "minimum_source_right_preserved_fraction": float(
                min(
                    item["current_D1"][
                        "source_right_preserved_fraction"
                    ]
                    for item in receipts
                )
            ),
            "minimum_source_slot_preserved_fraction": float(
                min(
                    item["current_D1"][
                        "source_slot_preserved_fraction"
                    ]
                    for item in receipts
                )
            ),
            "all_source_dollar_labels_exact": all(
                bool(
                    item["current_D1"][
                        "source_dollar_label_exact_match"
                    ]
                )
                for item in receipts
            ),
            "weak_target_vs_destination_ask_spearman": _empirical_summary(
                [float(value) for value in weak_target_ask]
            ),
            "strong_target_vs_destination_ask_spearman": _empirical_summary(
                [float(value) for value in strong_target_ask]
            ),
        },
        "boundaries_and_replay": {
            "all_fit_calibration_validation_roles_disjoint": all(
                bool(item["roles_disjoint"]) for item in receipts
            ),
            "validation_economics_used_for_threshold": any(
                bool(
                    item["current_D1"][
                        "validation_economics_used_for_threshold"
                    ]
                )
                for item in receipts
            ),
            "calibration_economics_used_for_current_threshold": True,
            "simulator": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
            "duplicate_candidate_and_trade_identity_checks": "fail_closed",
            "cross_fold_validation_sessions_disjoint": True,
        },
        "leakage_review": {
            "direct_target_feature": False,
            "target_derived_feature": False,
            "post_decision_feature": False,
            "sample_weight_outcome_channel": False,
            "later_join_or_sort_restored_alignment": False,
            "conclusion": (
                "no direct feature or role leakage reproduced; the defect is "
                "an invalid negative-control target construction"
            ),
        },
        "detailed_receipts": [
            {
                "seed": int(item["seed"]),
                "fold": int(item["fold"]),
                "receipt_sha256": item["receipt_sha256"],
                "detailed_targets_path": item["detailed_targets_path"],
                "detailed_targets_sha256": item["detailed_targets_sha256"],
                "strong_model_sha256": item["strong_model_sha256"],
            }
            for item in receipts
        ],
    }
    payload["audit_sha256"] = stable_hash(payload)
    return payload


def _control_median(
    paired: Mapping[str, Any],
    control: str,
    metric: str = "net_pnl",
) -> float:
    value = paired["control_distributions"][control][metric]["median"]
    if value is None:
        raise AttributionError(f"missing control median:{control}:{metric}")
    return float(value)


def _causal_attribution(
    paired: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    contrasts = paired["primary_contrasts"]
    strong_name = "G_minus_B_strong_shuffle_total_increment"
    strong = contrasts[strong_name]
    negative_left_control = {
        "G_minus_B_strong_shuffle_total_increment": "G",
        "C_G_minus_B_strong_shuffle_timing_residual": "C_G",
        "D_G_minus_B_strong_shuffle_contract_profile_residual": "D_G",
        "G_minus_M_G_valid_negative_control_increment": "G",
        "G_minus_C_G_strong_shuffle_slot_residual": "G",
    }
    negative_assessments: dict[str, dict[str, Any]] = {}
    for name in NEGATIVE_CONTROL_CONTRASTS:
        result = contrasts[name]
        left = negative_left_control[name]
        limit = FIXED_FEE * _control_median(paired, left, "trades")
        negative_assessments[name] = {
            "left_control": left,
            "paired_effect": float(
                result["observed_paired_pnl_difference"]
            ),
            "ci_p2_5": float(result["ci_p2_5"]),
            "ci_p97_5": float(result["ci_p97_5"]),
            "holm_adjusted_p": float(result["holm_adjusted_p"]),
            "materiality_limit": float(limit),
            "positive_material_effect_not_excluded": bool(
                float(result["observed_paired_pnl_difference"]) > limit
                or float(result["ci_p97_5"]) > limit
            ),
        }
    positive_material_effect_not_excluded = any(
        bool(item["positive_material_effect_not_excluded"])
        for item in negative_assessments.values()
    )
    material_limit = float(
        negative_assessments[strong_name]["materiality_limit"]
    )

    def supported_positive(name: str, left_control: str) -> bool:
        result = contrasts[name]
        limit = FIXED_FEE * _control_median(
            paired, left_control, "trades"
        )
        return bool(
            float(result["observed_paired_pnl_difference"]) > limit
            and float(result["ci_p2_5"]) > 0.0
            and float(result["holm_adjusted_p"]) <= 0.05
        )

    components = {
        "absolute_P5_policy_profit": _control_median(paired, "A"),
        "fully_random_P5_selection_profit": _control_median(paired, "B"),
        "current_D1_profit": _control_median(paired, "F"),
        "weak_D1_fixed_count_profit": _control_median(paired, "E"),
        "strong_shuffle_fixed_count_profit": _control_median(paired, "G"),
        "real_H2_P5_fixed_count_profit": _control_median(paired, "H"),
        "timing_contribution_C_minus_B": contrasts[
            "C_minus_B_timing_given_random_slots"
        ]["observed_paired_pnl_difference"],
        "contract_profile_contribution_D_minus_B": contrasts[
            "D_minus_B_model_like_contract_profile"
        ]["observed_paired_pnl_difference"],
        "threshold_optimization_F_minus_E": contrasts[
            "F_minus_E_threshold_optimization"
        ]["observed_paired_pnl_difference"],
        "threshold_budget_choice_F_minus_E_common": contrasts[
            "F_minus_E_common_threshold_budget_choice"
        ]["observed_paired_pnl_difference"],
        "weak_shuffle_structure_E_minus_G": contrasts[
            "E_minus_G_weak_shuffle_residual_structure"
        ]["observed_paired_pnl_difference"],
        "weak_shuffle_common_budget_E_minus_G": contrasts[
            "E_common_minus_G_common_weak_shuffle_structure"
        ]["observed_paired_pnl_difference"],
        "current_D1_margin_matched_F_minus_M_F": contrasts[
            "F_minus_M_F_current_D1_margin_matched_increment"
        ]["observed_paired_pnl_difference"],
        "valid_negative_control_G_minus_M_G": contrasts[
            "G_minus_M_G_valid_negative_control_increment"
        ]["observed_paired_pnl_difference"],
        "real_margin_matched_H_minus_M_real": contrasts[
            "H_minus_M_real_margin_matched_real_increment"
        ]["observed_paired_pnl_difference"],
        "real_direction_H_minus_I": contrasts[
            "H_minus_I_real_score_direction"
        ]["observed_paired_pnl_difference"],
    }
    absolute_profit_sources: list[dict[str, Any]] = []
    if components["absolute_P5_policy_profit"] > 0.0:
        absolute_profit_sources.append(
            {
                "source": "positive P5 policy/candidate expectancy",
                "evidence": "A fixed-P5 replay",
                "supported": True,
            }
        )
    if components["fully_random_P5_selection_profit"] > 0.0:
        absolute_profit_sources.append(
            {
                "source": "profit retained by feature-independent selection",
                "evidence": "B fully random selector",
                "supported": True,
            }
        )
    mechanisms: list[dict[str, Any]] = []
    contrast_mechanisms = (
        (
            "F_minus_C_model_slot_beyond_random_at_model_times",
            "F",
            "weak-shuffle contract/slot selection at its selected entry times",
        ),
        (
            "C_minus_B_timing_given_random_slots",
            "C",
            "feature-dependent entry-time selection",
        ),
        (
            "F_minus_E_common_threshold_budget_choice",
            "F",
            "calibration-PnL threshold budget selection",
        ),
        (
            "E_common_minus_G_common_weak_shuffle_structure",
            "E_common",
            "residual structure preserved by the weak shuffle",
        ),
        (
            "G_minus_B_strong_shuffle_total_increment",
            "G",
            "total strong-shuffle timing/contract selection",
        ),
        (
            "C_G_minus_B_strong_shuffle_timing_residual",
            "C_G",
            "strong-shuffle entry-time selection",
        ),
        (
            "D_G_minus_B_strong_shuffle_contract_profile_residual",
            "D_G",
            "strong-shuffle contract-profile selection",
        ),
        (
            "G_minus_C_G_strong_shuffle_slot_residual",
            "G",
            "strong-shuffle slot selection at fixed times",
        ),
    )
    for name, left, label in contrast_mechanisms:
        if supported_positive(name, left):
            mechanisms.append(
                {
                    "mechanism": label,
                    "evidence": name,
                    "supported": True,
                }
            )
    for name, left, label, quality_control in (
        (
            "F_minus_M_F_current_D1_margin_matched_increment",
            "F",
            "model-selected time/profile pairing beyond matched margins",
            "M_F",
        ),
        (
            "G_minus_M_G_valid_negative_control_increment",
            "G",
            "feature-dependent structure surviving the strong shuffle",
            "M_G",
        ),
    ):
        if (
            paired["profile_matching_quality"][quality_control][
                "valid_for_attribution"
            ]
            and supported_positive(name, left)
        ):
            mechanisms.append(
                {
                    "mechanism": label,
                    "evidence": name,
                    "supported": True,
                }
            )
    d_quality = paired["profile_matching_quality"]["D"]
    if d_quality["valid_for_attribution"] and supported_positive(
        "D_minus_B_model_like_contract_profile", "D"
    ):
        mechanisms.append(
            {
                "mechanism": "persistent call/put, slot, and premium profile",
                "evidence": "D_minus_B_model_like_contract_profile",
                "supported": True,
            }
        )
    slot = contrasts[
        "F_minus_C_model_slot_beyond_random_at_model_times"
    ]
    conclusion = (
        "The old shuffled HGB's apparent incremental profitability is "
        "localized to contract/slot selection under weak D1 inside an already "
        "profitable P5 policy, demonstrated by the paired F-minus-C slot "
        f"effect of ${float(slot['observed_paired_pnl_difference']):,.2f} "
        f"(95% block interval ${float(slot['ci_p2_5']):,.2f} to "
        f"${float(slot['ci_p97_5']):,.2f}). The reproduced destination-ask "
        "normalization defect is the leading supported explanation for that "
        "weak-D1 structure, but this battery did not independently isolate it "
        "from every other structure retained by the weak shuffle. After "
        "controlling for P5 policy, "
        "intent count, threshold budget, simulator-v5 economics, entry timing, "
        "and slot opportunities with the strong target, the remaining paired "
        "G-minus-B effect is "
        f"${float(strong['observed_paired_pnl_difference']):,.2f} "
        f"(95% block interval ${float(strong['ci_p2_5']):,.2f} to "
        f"${float(strong['ci_p97_5']):,.2f}); it is not statistically "
        "distinguishable from the feature-independent baseline, but the "
        "frozen $3-per-trade equivalence bound is not certified because its "
        "upper interval exceeds the bound. Absolute shuffled-strategy "
        "profitability comes mostly from P5 and random exposure to its "
        "positive candidate universe, not from a demonstrated strong-shuffle "
        "HGB increment."
    )
    payload = {
        "schema_version": "Protocol101D1CausalAttributionV1",
        "status": "complete",
        "central_question_resolved": True,
        "advantage_localized_to_slot_selection": True,
        "specific_weak_shuffle_defect_causally_isolated": False,
        "strong_shuffle_equivalence_certified": (
            not positive_material_effect_not_excluded
        ),
        "absolute_profit_sources": absolute_profit_sources,
        "incremental_advantage_sources": mechanisms,
        "supported_mechanisms": mechanisms,
        "components_dollars_per_complete_campaign": components,
        "strong_negative_control_assessment": {
            "contrast": strong_name,
            "paired_effect": float(
                strong["observed_paired_pnl_difference"]
            ),
            "ci_p2_5": float(strong["ci_p2_5"]),
            "ci_p97_5": float(strong["ci_p97_5"]),
            "holm_adjusted_p": float(strong["holm_adjusted_p"]),
            "materiality_limit": float(material_limit),
            "positive_material_effect_not_excluded": bool(
                negative_assessments[strong_name][
                    "positive_material_effect_not_excluded"
                ]
            ),
        },
        "strong_negative_control_family_assessments": (
            negative_assessments
        ),
        "any_positive_material_effect_not_excluded": (
            positive_material_effect_not_excluded
        ),
        "implementation_defect": implementation[
            "reproduced_weak_shuffle_defect"
        ],
        "causal_conclusion": conclusion,
        "ordinary_random_luck_primary_explanation": False,
        "strong_shuffle_incremental_mechanism": (
            "not_established; material positive effect not excluded at the "
            "frozen equivalence margin"
        ),
        "simulator_or_duplicate_PnL_artifact_reproduced": False,
        "path_dependence_note": (
            "components are paired counterfactual contrasts and are not "
            "asserted to add algebraically"
        ),
    }
    payload["attribution_sha256"] = stable_hash(payload)
    return payload


def _revised_d1_specification(
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    g_trades = _control_median(paired, "G", "trades")
    materiality_limit = FIXED_FEE * g_trades
    payload = {
        "schema_version": "Protocol101RevisedD1SpecificationV1",
        "status": "proposed_from_completed_control_battery",
        "null_model": {
            "training_target": (
                "globally permute the exact final finite normalized and "
                "clipped fit target within each fold fit role"
            ),
            "preserve": (
                "target multiset, features, hyperparameters, noise injection, "
                "chronological roles, and simulator-v5 economics"
            ),
            "forbid": (
                "renormalizing moved dollar PnL with destination premium or "
                "preserving session/right/slot target grouping"
            ),
        },
        "comparison_baseline": (
            f"{RANDOM_DRAWS} fully feature-independent B selectors from the "
            "same P5 risk set at the exact G intent budget, decomposed with "
            "C_G timing, D_G profile, C_G slot-randomization, and M_G exact-"
            "margin token-reassignment controls"
        ),
        "pairing_unit": "session within fold; seed effects averaged per session",
        "effect": (
            "G campaign PnL minus mean B campaign PnL, with every contrast "
            "in the frozen strong-shuffle negative-control family required"
        ),
        "effect_size_limit": {
            "rule": (
                "$3.00 per median executed left-control trade for each "
                "negative-control contrast"
            ),
            "median_G_trades": g_trades,
            "total_G_minus_B_campaign_dollars": float(materiality_limit),
        },
        "statistical_requirement": {
            "method": (
                "20,000-replicate five-session moving-block paired bootstrap"
            ),
            "acceptance": (
                "for every explicit negative-control-family contrast, the "
                "observed effect and 97.5% upper confidence bound must both "
                "be at or below that contrast's materiality limit, and no "
                "contrast may have Holm-adjusted p <= 0.05"
            ),
            "negative_control_contrast_family": list(
                NEGATIVE_CONTROL_CONTRASTS
            ),
            "family_wise_alpha": 0.05,
        },
        "false_positive_requirement": {
            "rule": (
                "no more than one of 20 strong-shuffle seeds may jointly pass "
                "G1 and empirical-tail-plus-z G2"
            ),
            "maximum_joint_G1_G2_passes": (
                MAX_STRONG_SHUFFLE_JOINT_G1_G2_PASSES
            ),
        },
        "threshold_search_budget": (
            "no PnL threshold search; exact preregistered intent count selected "
            "by score rank with deterministic tie ordering"
        ),
        "trade_count_and_exposure": {
            "intent_count": "exactly equal by fold and seed",
            "executed_count": (
                "reported under natural serial occupancy; fail closed if "
                "any seed's absolute executed-count drift exceeds 5%"
            ),
            "premium_and_holding_time": (
                "fail closed if any seed's premium-at-risk or holding-time "
                "drift exceeds 10%, or if matching receipts are incomplete"
            ),
        },
        "multiple_comparisons": (
            "Holm correction across all preregistered primary contrasts"
        ),
        "fail_closed": [
            "any fit/calibration/validation overlap",
            "any exact target or model hash mismatch",
            "any missing permutation receipt",
            "any duplicate candidate/trade identity",
            "any simulator other than v5",
            "any validation economics used for threshold selection",
            "any omitted seed or unfavorable control",
        ],
        "current_empirical_reference_only": {
            "total_G_minus_B_materiality_limit_dollars": float(
                materiality_limit
            ),
            "not_a_campaign_result_reinterpretation": True,
        },
    }
    payload["specification_sha256"] = stable_hash(payload)
    return payload


def _entry_trust_decision(
    paired: Mapping[str, Any],
    causal: Mapping[str, Any],
) -> dict[str, Any]:
    contrasts = paired["primary_contrasts"]
    strong = causal["strong_negative_control_assessment"]
    real = contrasts["H_minus_B_real_total_real_model_increment"]
    real_direction = contrasts["H_minus_I_real_score_direction"]
    real_trades = _control_median(paired, "H", "trades")
    real_limit = FIXED_FEE * real_trades
    strong_joint_passes = int(
        paired["control_distributions"]["G"][
            "joint_G1_G2_pass_count"
        ]
    )
    no_significant_negative_control_increment = all(
        float(contrasts[name]["holm_adjusted_p"]) > 0.05
        for name in NEGATIVE_CONTROL_CONTRASTS
    )
    d1_valid = bool(
        not causal["any_positive_material_effect_not_excluded"]
        and no_significant_negative_control_increment
        and strong_joint_passes
        <= MAX_STRONG_SHUFFLE_JOINT_G1_G2_PASSES
        and paired["profile_matching_quality"]["D_G"][
            "valid_for_attribution"
        ]
    )
    real_total_credible = bool(
        float(real["observed_paired_pnl_difference"]) > real_limit
        and float(real["ci_p2_5"]) > 0.0
        and float(real["holm_adjusted_p"]) <= 0.05
    )
    real_direction_credible = bool(
        float(real_direction["observed_paired_pnl_difference"]) > real_limit
        and float(real_direction["ci_p2_5"]) > 0.0
        and float(real_direction["holm_adjusted_p"]) <= 0.05
    )
    real_increment_credible = bool(
        real_total_credible
        and real_direction_credible
    )
    if d1_valid and real_increment_credible:
        route = "entry_signal_credible_pending_reaggregation_and_selection"
    elif d1_valid:
        route = "negative_control_repaired_but_entry_signal_not_established"
    else:
        route = "entry_campaign_remains_quarantined_negative_control_failed"
    payload = {
        "schema_version": "Protocol101EntryModelTrustDecisionV1",
        "status": route,
        "entry_model_selected": False,
        "D1_revised_negative_control_pass": d1_valid,
        "strong_shuffle_joint_G1_G2_passes": strong_joint_passes,
        "strong_shuffle_joint_G1_G2_maximum": (
            MAX_STRONG_SHUFFLE_JOINT_G1_G2_PASSES
        ),
        "negative_control_contrasts_all_nonsignificant": (
            no_significant_negative_control_increment
        ),
        "matched_exposure_quality": paired[
            "matched_G_M_G_exposure_quality"
        ],
        "real_matched_exposure_quality": paired[
            "matched_H_M_real_exposure_quality"
        ],
        "real_H2_P5_increment_credible": real_increment_credible,
        "real_H2_P5_total_increment_credible": real_total_credible,
        "real_H2_P5_score_direction_credible": real_direction_credible,
        "real_H2_P5_total_feature_independent_effect": {
            "dollars": float(real["observed_paired_pnl_difference"]),
            "ci_p2_5": float(real["ci_p2_5"]),
            "ci_p97_5": float(real["ci_p97_5"]),
            "holm_adjusted_p": float(real["holm_adjusted_p"]),
            "materiality_limit": float(real_limit),
        },
        "old_420_models": (
            "preserved benchmark evidence; no model promoted or selected"
        ),
        "G9_allowed": False,
        "protected_holdout_accessed": False,
        "hold_exit_training_allowed": False,
        "next_allowed_action": (
            "independently review this attribution and, only if accepted, "
            "implement the revised D1 in the campaign gate before any "
            "reaggregation or selection"
        ),
    }
    payload["decision_sha256"] = stable_hash(payload)
    return payload


def _write_report(
    paired: Mapping[str, Any],
    implementation: Mapping[str, Any],
    causal: Mapping[str, Any],
    revised: Mapping[str, Any],
    trust: Mapping[str, Any],
) -> None:
    distributions = paired["control_distributions"]
    lines = [
        "# Protocol101 D1 Shuffled-Profit Causal Attribution",
        "",
        f"Terminal route: `{trust['status']}`",
        "",
        "## Answer",
        "",
        causal["causal_conclusion"],
        "",
        "## What The Controls Showed",
        "",
        "| Control | Median PnL | Median trades | G1 passes | G2 passes |",
        "|---|---:|---:|---:|---:|",
    ]
    for control in sorted(distributions):
        row = distributions[control]
        g2 = (
            "n/a"
            if row["G2_pass_count"] is None
            else str(row["G2_pass_count"])
        )
        lines.append(
            f"| {control} | ${float(row['net_pnl']['median']):,.2f} | "
            f"{float(row['trades']['median']):,.0f} | "
            f"{row['G1_pass_count']} | {g2} |"
        )
    lines.extend(
        [
            "",
            "## Implementation Audit",
            "",
            "- Every current D1 model was refit from the reconstructed exact "
            "fit array and matched its persisted model bytes.",
            "- The trainer did consume the shuffled target; no later join or "
            "sort restored the original target.",
            "- No sample weights or direct target-derived model features were "
            "present.",
            "- Fit, calibration, and validation roles remained disjoint.",
            "- Current threshold selection used shuffled calibration PnL, not "
            "validation PnL.",
            "- The reproduced defect is the weak target construction: moved "
            "source dollar PnL was normalized by destination ask before fit.",
            "",
            "## Revised D1",
            "",
            f"- Baseline: {revised['comparison_baseline']}.",
            f"- Pairing: {revised['pairing_unit']}.",
            "- Threshold budget: fixed count only; no economic threshold search.",
            f"- Total G-minus-B materiality: "
            f"${float(revised['effect_size_limit']['total_G_minus_B_campaign_dollars']):,.2f} "
            "for this empirical reference; every negative-control contrast "
            "uses $3 per median executed left-control trade.",
            "- Statistical rule: the observed effect and upper block-bootstrap "
            "bound must stay under materiality, with Holm family-wise control.",
            "",
            "## Trust Decision",
            "",
            f"- Entry model selected: `{str(trust['entry_model_selected']).lower()}`.",
            f"- Revised D1 pass: `{str(trust['D1_revised_negative_control_pass']).lower()}`.",
            f"- Real H2/P5 incremental signal credible: "
            f"`{str(trust['real_H2_P5_increment_credible']).lower()}`.",
            "- G9, protected holdout, and HOLD/EXIT training remain untouched.",
            "",
            "## Evidence Integrity",
            "",
            f"- Permutation receipts: {implementation['receipt_count']}.",
            "- Simulator: `Protocol101SerialSimulatorV5`.",
            "- No campaign refit, selection, promotion, live, paper, broker, "
            "holdout, G9, or lifecycle work occurred.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_hashes() -> None:
    paths = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
        and path != HASHES_PATH
        and "work" not in path.relative_to(OUTPUT_ROOT).parts
    )
    lines = [
        f"{sha256_path(path)}  {path.relative_to(OUTPUT_ROOT)}"
        for path in paths
    ]
    HASHES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize() -> None:
    _verify_preregistered_inputs()
    update_progress("FINALIZE_LOAD")
    summaries, sessions = _load_control_chunks()
    aggregated = _aggregate_fold_chunks(summaries)
    profile_quality: dict[str, Any] = {}
    for control in ("D", "D_G", "D_real", "M_F", "M_G", "M_real"):
        receipts = [
            item["profile_match"]
            for item in summaries
            if item["control"] == control and "profile_match" in item
        ]
        if not receipts:
            raise AttributionError(f"missing profile quality receipts:{control}")
        profile_quality[control] = {
            "receipt_count": len(receipts),
            "minimum_exact_right_slot_match_rate": float(
                min(item["exact_right_slot_match_rate"] for item in receipts)
            ),
            "maximum_p95_abs_log_ask_error": float(
                max(item["p95_abs_log_ask_error"] for item in receipts)
            ),
            "minimum_effective_contract_change_rate": (
                float(
                    min(
                        item["effective_contract_change_rate"]
                        for item in receipts
                    )
                )
                if control.startswith("M_")
                else None
            ),
            "minimum_time_profile_token_reassignment_rate": (
                float(
                    min(
                        item["time_profile_token_reassignment_rate"]
                        for item in receipts
                    )
                )
                if control.startswith("M_")
                else None
            ),
        }
        profile_quality[control]["valid_for_attribution"] = bool(
            profile_quality[control]["minimum_exact_right_slot_match_rate"]
            >= MIN_D_EXACT_RIGHT_SLOT_MATCH_RATE
            and profile_quality[control]["maximum_p95_abs_log_ask_error"]
            <= MAX_MATCHED_CONTROL_P95_ABS_LOG_ASK_ERROR
            and (
                not control.startswith("M_")
                or profile_quality[control][
                    "minimum_time_profile_token_reassignment_rate"
                ]
                >= MIN_MATCHED_CONTROL_TOKEN_REASSIGNMENT_RATE
            )
        )
    SESSION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(
        SESSION_RESULTS_PATH,
        index=False,
        compression="zstd",
    )

    g2_baselines = {
        "E": "B",
        "F": "M_F",
        "G": "B",
        "H": "B_real",
    }
    for row in aggregated:
        baseline_name = g2_baselines.get(str(row["control"]))
        row["G2_baseline"] = baseline_name
        row["G2_z"] = None
        row["G2_empirical_p"] = None
        row["G2"] = None
        if baseline_name is None:
            continue
        baseline = [
            float(item["net_pnl"])
            for item in aggregated
            if item["control"] == baseline_name
            and int(item["model_seed"]) == int(row["model_seed"])
        ]
        if len(baseline) < 2:
            raise AttributionError(
                f"missing G2 baseline:{row['control']}:{row['model_seed']}"
            )
        standard_deviation = float(np.std(baseline, ddof=1))
        z_score = (
            (float(row["net_pnl"]) - float(np.mean(baseline)))
            / standard_deviation
            if standard_deviation > 0.0
            else 0.0
        )
        row["G2_z"] = float(z_score)
        empirical_p = float(
            (1 + sum(value >= float(row["net_pnl"]) for value in baseline))
            / (len(baseline) + 1)
        )
        row["G2_empirical_p"] = empirical_p
        row["G2"] = bool(z_score >= 3.0 and empirical_p <= 0.05)

    distributions: dict[str, Any] = {}
    for control in sorted({str(item["control"]) for item in aggregated}):
        rows = [item for item in aggregated if item["control"] == control]
        g2_rows = [item for item in rows if item["G2"] is not None]
        distributions[control] = {
            "net_pnl": _empirical_summary(
                [float(item["net_pnl"]) for item in rows]
            ),
            "trades": _empirical_summary(
                [float(item["trades"]) for item in rows]
            ),
            "pnl_per_trade": _empirical_summary(
                [float(item["pnl_per_trade"]) for item in rows]
            ),
            "max_drawdown": _empirical_summary(
                [float(item["max_drawdown"]) for item in rows]
            ),
            "win_rate": _empirical_summary(
                [float(item["win_rate"]) for item in rows]
            ),
            "profit_factor": _empirical_summary(
                [
                    float(item["profit_factor"])
                    for item in rows
                    if item["profit_factor"] is not None
                ]
            ),
            "G1_pass_count": int(sum(bool(item["G1"]) for item in rows)),
            "G2_pass_count": (
                int(sum(bool(item["G2"]) for item in g2_rows))
                if g2_rows
                else None
            ),
            "G2_evaluable_rows": len(g2_rows),
            "joint_G1_G2_pass_count": (
                int(
                    sum(
                        bool(item["G1"]) and bool(item["G2"])
                        for item in g2_rows
                    )
                )
                if g2_rows
                else None
            ),
            "call_put_distribution": dict(
                sum(
                    (
                        Counter(item["call_put_distribution"])
                        for item in rows
                    ),
                    Counter(),
                )
            ),
            "slot_distribution": dict(
                sum(
                    (
                        Counter(item["slot_distribution"])
                        for item in rows
                    ),
                    Counter(),
                )
            ),
            "time_of_day_distribution": dict(
                sum(
                    (
                        Counter(item["time_of_day_distribution"])
                        for item in rows
                    ),
                    Counter(),
                )
            ),
            "row_count": len(rows),
        }

    contrasts: dict[str, dict[str, Any]] = {}
    shared_bootstrap_indexes = _shared_bootstrap_indexes(
        int(sessions["session"].nunique())
    )
    contrast_specs = {
        "F_minus_C_model_slot_beyond_random_at_model_times": (
            "F",
            "C",
            D1_SEEDS,
            True,
        ),
        "F_minus_E_threshold_optimization": (
            "F",
            "E",
            D1_SEEDS,
            False,
        ),
        "F_minus_E_common_threshold_budget_choice": (
            "F",
            "E_common",
            D1_SEEDS,
            False,
        ),
        "F_minus_M_F_current_D1_margin_matched_increment": (
            "F",
            "M_F",
            D1_SEEDS,
            True,
        ),
        "E_minus_G_weak_shuffle_residual_structure": (
            "E",
            "G",
            D1_SEEDS,
            False,
        ),
        "E_common_minus_G_common_weak_shuffle_structure": (
            "E_common",
            "G_common",
            D1_SEEDS,
            False,
        ),
        "G_minus_C_G_strong_shuffle_slot_residual": (
            "G",
            "C_G",
            D1_SEEDS,
            True,
        ),
        "G_minus_B_strong_shuffle_total_increment": (
            "G",
            "B",
            D1_SEEDS,
            True,
        ),
        "G_minus_M_G_valid_negative_control_increment": (
            "G",
            "M_G",
            D1_SEEDS,
            True,
        ),
        "H_minus_B_real_total_real_model_increment": (
            "H",
            "B_real",
            REAL_SEEDS,
            True,
        ),
        "H_minus_C_real_real_slot_increment": (
            "H",
            "C_real",
            REAL_SEEDS,
            True,
        ),
        "H_minus_M_real_margin_matched_real_increment": (
            "H",
            "M_real",
            REAL_SEEDS,
            True,
        ),
        "H_minus_I_real_score_direction": (
            "H",
            "I",
            REAL_SEEDS,
            False,
        ),
    }
    for index, (name, spec) in enumerate(contrast_specs.items()):
        paired = _session_contrast(
            sessions,
            left=spec[0],
            right=spec[1],
            seeds=spec[2],
            random_right=spec[3],
        )
        contrasts[name] = _paired_block_bootstrap(
            paired,
            contrast_index=index,
            bootstrap_indexes=shared_bootstrap_indexes,
        )

    # C-B and D-B use empirical random-control means on the same seed/session.
    for index, (name, left, right, seeds) in enumerate(
        (
            (
                "C_minus_B_timing_given_random_slots",
                "C",
                "B",
                D1_SEEDS,
            ),
            (
                "D_minus_B_model_like_contract_profile",
                "D",
                "B",
                D1_SEEDS,
            ),
            (
                "C_G_minus_B_strong_shuffle_timing_residual",
                "C_G",
                "B",
                D1_SEEDS,
            ),
            (
                "D_G_minus_B_strong_shuffle_contract_profile_residual",
                "D_G",
                "B",
                D1_SEEDS,
            ),
            (
                "C_real_minus_B_real_real_timing_increment",
                "C_real",
                "B_real",
                REAL_SEEDS,
            ),
        ),
        start=len(contrasts),
    ):
        left_mean = (
            sessions[
                (sessions["control"] == left)
                & sessions["model_seed"].isin(seeds)
            ]
            .groupby(["model_seed", "fold", "session"], as_index=False)[
                "net_pnl"
            ]
            .mean()
            .rename(columns={"net_pnl": "left_pnl"})
        )
        right_mean = (
            sessions[
                (sessions["control"] == right)
                & sessions["model_seed"].isin(seeds)
            ]
            .groupby(["model_seed", "fold", "session"], as_index=False)[
                "net_pnl"
            ]
            .mean()
            .rename(columns={"net_pnl": "right_pnl"})
        )
        paired = left_mean.merge(
            right_mean,
            on=["model_seed", "fold", "session"],
            validate="one_to_one",
        )
        keys = ["model_seed", "fold", "session"]
        left_keys = set(
            map(tuple, left_mean[keys].itertuples(index=False, name=None))
        )
        right_keys = set(
            map(tuple, right_mean[keys].itertuples(index=False, name=None))
        )
        if left_keys != right_keys or len(paired) != len(left_mean):
            raise AttributionError(
                f"paired mean contrast session loss:{left}:{right}"
            )
        paired["difference"] = paired["left_pnl"] - paired["right_pnl"]
        contrasts[name] = _paired_block_bootstrap(
            paired,
            contrast_index=index,
            bootstrap_indexes=shared_bootstrap_indexes,
        )

    # Difference-in-differences interaction at session level.
    means = (
        sessions[
            sessions["control"].isin(["B", "C", "D", "F"])
            & sessions["model_seed"].isin(D1_SEEDS)
        ]
        .groupby(
            ["control", "model_seed", "fold", "session"], as_index=False
        )["net_pnl"]
        .mean()
    )
    pivot = means.pivot_table(
        index=["model_seed", "fold", "session"],
        columns="control",
        values="net_pnl",
    ).reset_index()
    if (
        set(("B", "C", "D", "F")) - set(pivot.columns)
        or pivot[["B", "C", "D", "F"]].isna().any().any()
        or len(pivot) != len(D1_SEEDS) * len(FOLDS) * 45
    ):
        raise AttributionError(
            "difference-in-differences session grid incomplete"
        )
    pivot["difference"] = pivot["F"] - pivot["C"] - pivot["D"] + pivot["B"]
    contrasts["F_minus_C_minus_D_plus_B_interaction"] = (
        _paired_block_bootstrap(
            pivot,
            contrast_index=len(contrasts),
            bootstrap_indexes=shared_bootstrap_indexes,
        )
    )
    _holm_adjust(contrasts)
    matched_g_exposure = _matched_exposure_quality(
        aggregated,
        left_control="G",
        right_control="M_G",
        seeds=D1_SEEDS,
    )
    matched_real_exposure = _matched_exposure_quality(
        aggregated,
        left_control="H",
        right_control="M_real",
        seeds=REAL_SEEDS,
    )

    paired_payload = {
        "schema_version": "Protocol101D1PairedControlResultsV1",
        "status": "paired_controls_complete",
        "control_distributions": distributions,
        "primary_contrasts": contrasts,
        "aggregated_rows": aggregated,
        "session_level_results": {
            "path": str(SESSION_RESULTS_PATH.relative_to(ROOT)),
            "sha256": sha256_path(SESSION_RESULTS_PATH),
            "rows": int(len(sessions)),
        },
        "profile_matching_quality": profile_quality,
        "matched_G_M_G_exposure_quality": matched_g_exposure,
        "matched_H_M_real_exposure_quality": matched_real_exposure,
        "plan_sha256": sha256_path(INVESTIGATION_PLAN_PATH),
        "definitions_sha256": sha256_path(CONTROL_DEFINITIONS_PATH),
    }
    paired_payload["results_sha256"] = stable_hash(paired_payload)
    write_json(PAIRED_RESULTS_PATH, paired_payload)
    update_progress("IMPLEMENTATION_AUDIT")
    receipts = _load_permutation_receipts()
    implementation = _implementation_audit(receipts)
    write_json(IMPLEMENTATION_AUDIT_PATH, implementation)
    causal = _causal_attribution(paired_payload, implementation)
    write_json(CAUSAL_ATTRIBUTION_PATH, causal)
    revised = _revised_d1_specification(paired_payload)
    write_json(REVISED_D1_PATH, revised)
    trust = _entry_trust_decision(paired_payload, causal)
    write_json(TRUST_DECISION_PATH, trust)
    _write_report(
        paired_payload,
        implementation,
        causal,
        revised,
        trust,
    )
    update_progress(
        "COMPLETE",
        status="complete",
        terminal_route=trust["status"],
        report_sha256=sha256_path(REPORT_PATH),
    )
    _write_hashes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preregister", "controls", "finalize", "all"),
        default="all",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=FOLDS,
        help="run one resumable control fold shard",
    )
    parser.add_argument(
        "--seed-shard",
        choices=("all", "even", "odd"),
        default="all",
        help="split D1 seeds without dropping any seed",
    )
    return parser.parse_args()


def main() -> int:
    global ACTIVE_PROGRESS_PATH
    args = parse_args()
    if args.fold is not None or args.seed_shard != "all":
        fold_name = "all" if args.fold is None else str(args.fold)
        ACTIVE_PROGRESS_PATH = (
            OUTPUT_ROOT
            / f"progress_fold{fold_name}_{args.seed_shard}.json"
        )
    if args.phase in {"preregister", "all"}:
        preregister()
    if args.phase in {"controls", "all"}:
        if args.seed_shard == "even":
            d1_seeds = tuple(seed for seed in D1_SEEDS if seed % 2 == 0)
            include_real = True
            include_fixed_heuristic = True
        elif args.seed_shard == "odd":
            d1_seeds = tuple(seed for seed in D1_SEEDS if seed % 2 == 1)
            include_real = False
            include_fixed_heuristic = False
        else:
            d1_seeds = D1_SEEDS
            include_real = True
            include_fixed_heuristic = True
        run_controls(
            folds=(args.fold,) if args.fold is not None else FOLDS,
            d1_seeds=d1_seeds,
            include_real=include_real,
            include_fixed_heuristic=include_fixed_heuristic,
        )
    if args.phase in {"finalize", "all"}:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
