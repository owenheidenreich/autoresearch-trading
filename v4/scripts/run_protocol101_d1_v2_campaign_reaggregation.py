"""Reaggregate the frozen Protocol101 Stage-1 campaign under D1 V2.

This runner never fits a model. It selects outcome-blind matched controls from
already-persisted random-control draws, recomputes paired inference, and
applies the superseding incremental-edge selection law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from v4.model.protocol101_d1_v2_contract import (
    ALPHA,
    AMENDMENT_PATH,
    CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
    CANDIDATE_MULTIPLICITY_FAMILY_SIZE,
    CANDIDATE_MULTIPLICITY_ROW_IDS,
    MATCH_LIMITS,
    aggregate_exposure,
    evaluate_candidate_incremental_edge,
    evaluate_d1_v2,
    exposure_match,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "v4/audit/autoresearch"
SOURCE_ROOT = (
    AUDIT_ROOT
    / "protocol101_d1_shuffled_profit_causal_attribution_attempt009"
)
CONTROL_ROOT = SOURCE_ROOT / "work/control_chunks"
SESSION_ROOT = SOURCE_ROOT / "work/session_chunks"
INDEPENDENT_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_"
    "independent_audit_selection_attempt001"
)
CAMPAIGN_ROOT = (
    AUDIT_ROOT / "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
ATTEMPT011_ROOT = (
    AUDIT_ROOT
    / "protocol101_d1_shuffled_profit_causal_attribution_"
    "inference_correction_attempt011"
)
DEFAULT_OUT = (
    AUDIT_ROOT / "protocol101_d1_v2_campaign_reaggregation_attempt005"
)

D1_SEEDS = tuple(range(8600, 8620))
REAL_SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BLOCK_SESSIONS = 5
BOOTSTRAP_SEED = 1_017_2026
CAMPAIGN_FAMILY_SIZE = CANDIDATE_MULTIPLICITY_FAMILY_SIZE
PRESERVED_SIGNAL_ROWS = ("H0/P5", "H0/P6", "H1/P5", "H1/P6", "H2/P5")


class ReaggregationError(RuntimeError):
    """Fail-closed reaggregation error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ReaggregationError(f"expected JSON object:{path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _required_inputs() -> dict[str, Path]:
    return {
        "reaggregation_runner": Path(__file__).resolve(),
        "d1_v2_contract": ROOT
        / "v4/model/protocol101_d1_v2_contract.py",
        "amendment": ROOT / AMENDMENT_PATH,
        "independent_audit": INDEPENDENT_ROOT / "independent_audit.json",
        "independent_selection": INDEPENDENT_ROOT / "selection.json",
        "authoritative_unit_integrity": INDEPENDENT_ROOT
        / "unit_replay_integrity.json",
        "attempt009_hashes": SOURCE_ROOT / "hashes.sha256",
        "attempt009_control_definitions": SOURCE_ROOT
        / "control_definitions.json",
        "attempt011_corrected_attribution": ATTEMPT011_ROOT
        / "causal_attribution_corrected.json",
        "attempt011_trust_decision": ATTEMPT011_ROOT
        / "entry_model_trust_decision_corrected.json",
    }


def _input_hashes() -> dict[str, str]:
    missing = [
        str(path) for path in _required_inputs().values() if not path.is_file()
    ]
    if missing:
        raise ReaggregationError(f"required_inputs_missing:{missing}")
    return {
        name: sha256_path(path)
        for name, path in sorted(_required_inputs().items())
    }


def _consumed_control_paths() -> list[Path]:
    paths = [
        CONTROL_ROOT / f"{control}_F{fold}_seed{seed}.json"
        for seed in D1_SEEDS
        for fold in FOLDS
        for control in ("G", "B")
    ]
    paths.extend(
        CONTROL_ROOT / f"{control}_F{fold}_seed{seed}.json"
        for seed in REAL_SEEDS
        for fold in FOLDS
        for control in ("H", "B_real")
    )
    return sorted(paths)


def _consumed_session_paths() -> list[Path]:
    paths = [
        SESSION_ROOT / f"{control}_F{fold}_seed{seed}.parquet"
        for seed in D1_SEEDS
        for fold in FOLDS
        for control in ("G", "B")
    ]
    paths.extend(
        SESSION_ROOT / f"{control}_F{fold}_seed{seed}.parquet"
        for seed in REAL_SEEDS
        for fold in FOLDS
        for control in ("H", "B_real")
    )
    return sorted(paths)


def _hashed_file_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ReaggregationError(f"consumed_inputs_missing:{missing}")
    files = {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in paths
    }
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def consumed_input_manifest() -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101D1V2ConsumedInputManifestV1",
        "control_chunks": _hashed_file_manifest(_consumed_control_paths()),
        "session_chunks": _hashed_file_manifest(_consumed_session_paths()),
    }
    payload["manifest_sha256"] = stable_hash(payload)
    return payload


def _model_hashes() -> dict[str, str]:
    paths = sorted(CAMPAIGN_ROOT.glob("H*/units/H*/policy*/seed*/expanding_fold_*/model.pkl"))
    if len(paths) != 420:
        raise ReaggregationError(f"campaign_model_grid_mismatch:{len(paths)}/420")
    return {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in paths
    }


def _authoritative_model_hashes() -> dict[str, str]:
    integrity = read_json(INDEPENDENT_ROOT / "unit_replay_integrity.json")
    units = integrity.get("units")
    if not isinstance(units, list) or len(units) != 420:
        raise ReaggregationError("authoritative_unit_integrity_grid_mismatch")
    result = {}
    for unit in units:
        hypothesis, policy, seed, fold = str(unit["unit_id"]).split("/")
        path = (
            CAMPAIGN_ROOT
            / hypothesis
            / "units"
            / hypothesis
            / f"policy{int(policy[1:])}"
            / f"seed{int(seed[1:])}"
            / f"expanding_fold_{int(fold[1:]):02d}"
            / "model.pkl"
        )
        result[str(path.relative_to(ROOT))] = str(unit["model_sha256"])
    return result


def preregistration(
    consumed_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101D1V2ReaggregationPreregistrationV1",
        "status": "owner_authorized_frozen_artifact_reaggregation",
        "amendment": AMENDMENT_PATH,
        "campaign_models": 420,
        "model_refit_allowed": False,
        "d1": {
            "left": "strong_global_target_permutation_HGB_G",
            "right": "fully_feature_independent_random_time_random_slot_B",
            "absolute_pnl_gate": False,
            "positive_artifact_rule": (
                "paired_lower_CI_gt_0_and_multiplicity_adjusted_p_lte_0.05"
            ),
        },
        "candidate_incremental_edge": {
            "candidate_family": list(CANDIDATE_MULTIPLICITY_ROW_IDS),
            "candidate_family_sha256": (
                CANDIDATE_MULTIPLICITY_FAMILY_SHA256
            ),
            "preserved_signal_rows": list(PRESERVED_SIGNAL_ROWS),
            "available_frozen_comparison": "H2/P5_H_minus_B_real",
            "missing_candidate_specific_evidence_fails_closed": True,
            "campaign_multiplicity_family_size": CAMPAIGN_FAMILY_SIZE,
            "pass": (
                "familywise_lower_CI_gt_0_and_"
                "multiplicity_adjusted_p_lte_0.05"
            ),
        },
        "matching": {
            "selection_uses_pnl": False,
            "fixed_preregistered_draw_zero_per_fold": True,
            "post_replay_metrics_are_validation_only": True,
            "limits": dict(MATCH_LIMITS),
            "moneyness_buckets": {
                "ATM": "abs(slot-10)<=1",
                "NEAR": "2<=abs(slot-10)<=5",
                "WING": "abs(slot-10)>5",
            },
            "all_dimensions_fail_closed": True,
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "moving_block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
            "seed": BOOTSTRAP_SEED,
            "paired_by": ["model_seed", "fold", "session"],
        },
        "equivalence": {
            "three_dollars_per_trade": "diagnostic_only",
            "can_change_gate": False,
        },
        "forbidden": {
            "model_fit_or_refit": True,
            "G9": True,
            "seed45": True,
            "protected_holdout": True,
            "learned_exit_training": True,
            "broker_or_paper": True,
            "runtime_or_launchd_change": True,
        },
        "input_hashes": _input_hashes(),
        "consumed_artifact_manifests": {
            name: {
                "file_count": consumed_manifest[name]["file_count"],
                "manifest_sha256": consumed_manifest[name]["manifest_sha256"],
            }
            for name in ("control_chunks", "session_chunks")
        },
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def _control_rows(control: str, fold: int, seed: int) -> list[dict[str, Any]]:
    path = CONTROL_ROOT / f"{control}_F{fold}_seed{seed}.json"
    payload = read_json(path)
    rows = payload.get("summaries")
    if not isinstance(rows, list) or not rows:
        raise ReaggregationError(f"control_rows_missing:{path}")
    return [dict(row) for row in rows]


def _match_draws(
    *,
    left_control: str,
    right_control: str,
    seed: int,
) -> dict[str, Any]:
    left_rows = {
        fold: _control_rows(left_control, fold, seed)[0] for fold in FOLDS
    }
    pools = {
        fold: _control_rows(right_control, fold, seed) for fold in FOLDS
    }
    selected_rows = []
    selected_draws = {}
    for fold in FOLDS:
        selected = [
            row for row in pools[fold] if int(row["draw"]) == 0
        ]
        if len(selected) != 1:
            raise ReaggregationError(
                f"fixed_draw_zero_missing:{right_control}:S{seed}:F{fold}"
            )
        selected_rows.append(selected[0])
        selected_draws[f"F{fold}"] = 0

    intent_budgets_equal = bool(
        right_control in {"B", "B_real"}
        and all(
            int(row["entry_intents"])
            == int(left_rows[int(row["fold"])]["entry_intents"])
            for row in selected_rows
        )
    )
    # The compact attempt009 summaries prove the randomized selector and
    # intent budget, but do not persist the complete governed risk-set
    # identity hash. Equality cannot be inferred from counts alone.
    random_selector_receipts_complete = False
    left_exposure = aggregate_exposure(list(left_rows.values()))
    right_exposure = aggregate_exposure(selected_rows)
    pooled_match = exposure_match(
        left_exposure,
        right_exposure,
        exact_opportunity_set_preserved=random_selector_receipts_complete,
        random_selector_receipts_complete=random_selector_receipts_complete,
    )
    per_fold_matches = {}
    for fold in FOLDS:
        per_fold_matches[f"F{fold}"] = exposure_match(
            aggregate_exposure([left_rows[fold]]),
            aggregate_exposure([selected_rows[fold - 1]]),
            exact_opportunity_set_preserved=(
                random_selector_receipts_complete
                and int(left_rows[fold]["entry_intents"])
                == int(selected_rows[fold - 1]["entry_intents"])
            ),
            random_selector_receipts_complete=(
                random_selector_receipts_complete
            ),
        )
    fold_blockers = [
        f"{fold}:{blocker}"
        for fold, fold_match in per_fold_matches.items()
        for blocker in fold_match["blockers"]
    ]
    match = dict(pooled_match)
    match["pooled_pass"] = pooled_match["pass"]
    match["per_fold"] = per_fold_matches
    match["pass"] = bool(
        pooled_match["pass"]
        and all(item["pass"] for item in per_fold_matches.values())
    )
    match["blockers"] = sorted(
        set(pooled_match["blockers"] + fold_blockers)
    )
    return {
        "seed": seed,
        "left_control": left_control,
        "right_control": right_control,
        "selection_rule": "preregistered_fixed_draw_zero_per_fold",
        "selection_used_pnl_or_post_replay_exposure": False,
        "intent_budgets_equal": intent_budgets_equal,
        "risk_set_identity_receipt": {
            "available": False,
            "reason": (
                "attempt009 compact controls omit the complete governed "
                "risk-set identity hash"
            ),
        },
        "selected_draws": selected_draws,
        "left_exposure": left_exposure,
        "right_exposure": right_exposure,
        "match": match,
        "selected_control_source_hashes": {
            f"F{fold}": sha256_path(
                CONTROL_ROOT / f"{right_control}_F{fold}_seed{seed}.json"
            )
            for fold in FOLDS
        },
    }


def _selected_session_frame(
    *,
    control: str,
    seed: int,
    draws: Mapping[str, int] | None,
    output_control: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in FOLDS:
        path = SESSION_ROOT / f"{control}_F{fold}_seed{seed}.parquet"
        frame = pd.read_parquet(path)
        draw = 0 if draws is None else int(draws[f"F{fold}"])
        frame = frame[frame["draw"].astype(int) == draw].copy()
        if len(frame) != 45:
            raise ReaggregationError(
                f"session_draw_incomplete:{control}:S{seed}:F{fold}:"
                f"{len(frame)}/45"
            )
        frame["control"] = output_control
        frame["draw"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _paired_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["model_seed", "fold", "session"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ReaggregationError("paired_contrast_duplicate_axis")
    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ReaggregationError("paired_contrast_session_axis_mismatch")
    paired = left[keys + ["net_pnl"]].rename(
        columns={"net_pnl": "left_pnl"}
    ).merge(
        right[keys + ["net_pnl"]].rename(columns={"net_pnl": "right_pnl"}),
        on=keys,
        validate="one_to_one",
    )
    paired["difference"] = paired["left_pnl"] - paired["right_pnl"]
    return paired


def _moving_block_indexes(session_count: int) -> np.ndarray:
    blocks = [
        np.arange(start, start + BOOTSTRAP_BLOCK_SESSIONS) % session_count
        for start in range(session_count)
    ]
    block_count = math.ceil(session_count / BOOTSTRAP_BLOCK_SESSIONS)
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    result = np.empty((BOOTSTRAP_REPLICATES, session_count), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(blocks), size=block_count)
        result[replicate] = np.concatenate(
            [blocks[int(index)] for index in chosen]
        )[:session_count]
    return result


def _paired_inference(
    paired: pd.DataFrame,
    *,
    multiplicity_family_size: int,
) -> dict[str, Any]:
    sessions = sorted(paired["session"].astype(str).unique())
    values = (
        paired.groupby("session", sort=True)["difference"]
        .mean()
        .reindex(sessions, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    observed = float(values.sum())
    indexes = _moving_block_indexes(len(values))
    draws = values[indexes].sum(axis=1, dtype=float)
    centered = draws - float(draws.mean())
    raw_p = float(
        (1 + int(np.sum(centered >= observed)))
        / (BOOTSTRAP_REPLICATES + 1)
    )
    adjusted_p = min(1.0, raw_p * multiplicity_family_size)
    tail = ALPHA / (2.0 * multiplicity_family_size)
    familywise_interval = [
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1.0 - tail)),
    ]
    return {
        "observed_paired_fee_adjusted_pnl_difference": observed,
        "session_count": len(values),
        "seed_count": int(paired["model_seed"].nunique()),
        "replicates": BOOTSTRAP_REPLICATES,
        "moving_block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
        "confidence_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "familywise_confidence_interval_95": familywise_interval,
        "multiplicity_family_size": multiplicity_family_size,
        "one_sided_centered_p": raw_p,
        "multiplicity_adjusted_p": adjusted_p,
    }


def _contrast_for_matches(
    *,
    left_control: str,
    right_control: str,
    matches: Sequence[Mapping[str, Any]],
    include_only_passing: bool,
    family_size: int,
) -> dict[str, Any] | None:
    left_frames: list[pd.DataFrame] = []
    right_frames: list[pd.DataFrame] = []
    for item in matches:
        if include_only_passing and item["match"]["pass"] is not True:
            continue
        seed = int(item["seed"])
        left_frames.append(
            _selected_session_frame(
                control=left_control,
                seed=seed,
                draws=None,
                output_control="LEFT",
            )
        )
        right_frames.append(
            _selected_session_frame(
                control=right_control,
                seed=seed,
                draws=item["selected_draws"],
                output_control="RIGHT",
            )
        )
    if not left_frames:
        return None
    paired = _paired_contrast(
        pd.concat(left_frames, ignore_index=True),
        pd.concat(right_frames, ignore_index=True),
    )
    return _paired_inference(paired, multiplicity_family_size=family_size)


def _reaggregate_rows(
    independent_rows: Sequence[Mapping[str, Any]],
    *,
    d1_result: Mapping[str, Any],
    incremental_results: Mapping[str, Mapping[str, Any]],
    D5_pass: bool,
    D6_pass: bool,
) -> list[dict[str, Any]]:
    output = []
    for source in independent_rows:
        row_id = str(source["row_id"])
        incremental = incremental_results[row_id]
        maxT_pass = bool(
            isinstance(source.get("maxT"), Mapping)
            and source["maxT"].get("hard_pass") is True
        )
        prior_hard_without_global_d1 = bool(
            all(source["gates"][f"G{index}"] for index in range(1, 8))
            and maxT_pass
        )
        current_eligible = bool(
            prior_hard_without_global_d1
            and d1_result["pass"] is True
            and D5_pass
            and D6_pass
            and incremental["pass"] is True
        )
        output.append(
            {
                "row_id": row_id,
                "hypothesis": source["hypothesis"],
                "policy": source["policy"],
                "gates": source["gates"],
                "maxT_pass": maxT_pass,
                "D5_pass": D5_pass,
                "D6_pass": D6_pass,
                "D1_v2_pass": d1_result["pass"],
                "candidate_incremental_edge": incremental,
                "hard_gate_eligible_v2": current_eligible,
                "historical_hard_gate_eligible_v1": source.get(
                    "hard_gate_eligible"
                ),
                "median_seed_fee_adjusted_continuous_strict_serial_net_pnl": (
                    source[
                        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
                    ]
                ),
            }
        )
    return output


def _write_hashes(out_dir: Path) -> None:
    paths = sorted(
        path for path in out_dir.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    )
    lines = [f"{sha256_path(path)}  {path.name}" for path in paths]
    (out_dir / "hashes.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReaggregationError(
            f"immutable_output_already_exists:{out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    consumed_manifest = consumed_input_manifest()
    prereg = preregistration(consumed_manifest)
    write_json(out_dir / "preregistration.json", prereg)
    write_json(
        out_dir / "consumed_input_manifest.json",
        consumed_manifest,
    )
    write_json(
        out_dir / "progress.json",
        {
            "schema_version": "Protocol101D1V2ReaggregationProgressV1",
            "status": "running",
            "node": "hash_frozen_models",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    model_hashes_before = _model_hashes()
    authoritative_model_hashes = _authoritative_model_hashes()
    if model_hashes_before != authoritative_model_hashes:
        raise ReaggregationError(
            "campaign_models_differ_from_independent_integrity_authority"
        )

    d1_matches = [
        _match_draws(
            left_control="G",
            right_control="B",
            seed=seed,
        )
        for seed in D1_SEEDS
    ]
    d1_inference = _contrast_for_matches(
        left_control="G",
        right_control="B",
        matches=d1_matches,
        include_only_passing=True,
        family_size=1,
    )
    d1_result = evaluate_d1_v2(
        exposure_matches=[item["match"] for item in d1_matches],
        effect_fee_adjusted_pnl=(
            d1_inference["observed_paired_fee_adjusted_pnl_difference"]
            if d1_inference is not None
            else None
        ),
        ci_lower=(
            d1_inference["confidence_interval_95"][0]
            if d1_inference is not None
            else None
        ),
        ci_upper=(
            d1_inference["confidence_interval_95"][1]
            if d1_inference is not None
            else None
        ),
        multiplicity_adjusted_p=(
            d1_inference["multiplicity_adjusted_p"]
            if d1_inference is not None
            else None
        ),
        absolute_shuffled_pnl=None,
        equivalence_bound_dollars=3.0
        * sum(
            int(item["left_exposure"]["trades"]) for item in d1_matches
        )
        / len(d1_matches),
    )
    d1_result["inference_on_passing_matches_only"] = d1_inference
    d1_result["matched_seed_count"] = sum(
        item["match"]["pass"] is True for item in d1_matches
    )
    d1_result["required_seed_count"] = len(D1_SEEDS)
    d1_result["matching_receipts"] = d1_matches
    write_json(out_dir / "d1_v2_control_result.json", d1_result)

    real_matches = [
        _match_draws(
            left_control="H",
            right_control="B_real",
            seed=seed,
        )
        for seed in REAL_SEEDS
    ]
    real_inference = _contrast_for_matches(
        left_control="H",
        right_control="B_real",
        matches=real_matches,
        include_only_passing=True,
        family_size=CAMPAIGN_FAMILY_SIZE,
    )
    incremental_results: dict[str, dict[str, Any]] = {}
    for row_id in [f"H{h}/P{p}" for h in range(4) for p in range(7)]:
        if row_id == "H2/P5":
            if real_inference is None:
                lower = upper = effect = adjusted_p = None
                evidence_complete = False
            else:
                lower, upper = real_inference[
                    "familywise_confidence_interval_95"
                ]
                effect = real_inference[
                    "observed_paired_fee_adjusted_pnl_difference"
                ]
                adjusted_p = real_inference["multiplicity_adjusted_p"]
                evidence_complete = True
            result = evaluate_candidate_incremental_edge(
                row_id=row_id,
                exposure_matches=[item["match"] for item in real_matches],
                effect_fee_adjusted_pnl=effect,
                ci_lower=lower,
                ci_upper=upper,
                multiplicity_adjusted_p=adjusted_p,
                multiplicity_family_size=(
                    real_inference["multiplicity_family_size"]
                    if real_inference is not None
                    else CAMPAIGN_FAMILY_SIZE
                ),
                multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
                multiplicity_family_sha256=(
                    CANDIDATE_MULTIPLICITY_FAMILY_SHA256
                ),
                multiplicity_receipt_complete=bool(
                    real_inference is not None
                    and real_inference["multiplicity_family_size"]
                    == CAMPAIGN_FAMILY_SIZE
                ),
                costs_included=True,
                evidence_complete=evidence_complete,
            )
            result["ordinary_confidence_interval_95"] = (
                real_inference["confidence_interval_95"]
                if real_inference is not None
                else [None, None]
            )
            result["inference"] = real_inference
            result["matching_receipts"] = real_matches
            result["prior_unmatched_total_increment_diagnostic"] = read_json(
                ATTEMPT011_ROOT / "paired_control_results_corrected.json"
            )["primary_contrasts"][
                "H_minus_B_real_total_real_model_increment"
            ]
        else:
            result = evaluate_candidate_incremental_edge(
                row_id=row_id,
                exposure_matches=[],
                effect_fee_adjusted_pnl=None,
                ci_lower=None,
                ci_upper=None,
                multiplicity_adjusted_p=None,
                multiplicity_family_size=CAMPAIGN_FAMILY_SIZE,
                multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
                multiplicity_family_sha256=(
                    CANDIDATE_MULTIPLICITY_FAMILY_SHA256
                ),
                multiplicity_receipt_complete=False,
                costs_included=True,
                evidence_complete=False,
            )
        incremental_results[row_id] = result
    write_json(
        out_dir / "candidate_incremental_results.json",
        {
            "schema_version": "Protocol101CandidateIncrementalResultsV1",
            "campaign_family_size": CAMPAIGN_FAMILY_SIZE,
            "results": incremental_results,
        },
    )

    independent = read_json(INDEPENDENT_ROOT / "independent_audit.json")
    if independent.get("accepted") is not True:
        raise ReaggregationError("independent_campaign_audit_not_accepted")
    global_controls = independent["independent_result"]["global_controls"]
    D5_pass = global_controls.get("D5") is True
    D6_pass = global_controls.get("D6") is True
    independent_rows = independent["independent_result"]["rows"]
    reaggregated_rows = _reaggregate_rows(
        independent_rows,
        d1_result=d1_result,
        incremental_results=incremental_results,
        D5_pass=D5_pass,
        D6_pass=D6_pass,
    )
    write_json(
        out_dir / "reaggregated_rows.json",
        {
            "schema_version": "Protocol101Stage1RowsD1V2V1",
            "row_count": len(reaggregated_rows),
            "rows": reaggregated_rows,
        },
    )
    eligible = [
        row for row in reaggregated_rows
        if row["hard_gate_eligible_v2"] is True
    ]
    if not d1_result["pass"]:
        route = (
            "corrected_D1_inconclusive_exposure_mismatch"
            if d1_result["status"] == "INSUFFICIENT_MATCHED_CONTROL"
            else "entry_campaign_invalid_negative_control_failed"
        )
    elif not eligible:
        route = "entry_campaign_rejected_no_incremental_model_edge"
    else:
        route = "entry_candidate_selected_pending_G9"
    selected = None
    if eligible:
        eligible.sort(
            key=lambda row: (
                -float(
                    row[
                        "candidate_incremental_edge"
                    ]["effect_fee_adjusted_pnl"]
                ),
                row["row_id"],
            )
        )
        selected = eligible[0]["row_id"]
    selection = {
        "schema_version": "Protocol101Stage1D1V2SelectionV1",
        "routing_decision": route,
        "selected_candidate": selected,
        "eligible_rows": [row["row_id"] for row in eligible],
        "D1_v2_status": d1_result["status"],
        "D1_v2_pass": d1_result["pass"],
        "candidate_rule": (
            "matched_random_fee_adjusted_familywise_lower_CI_gt_0_and_"
            "multiplicity_adjusted_p_lte_0.05"
        ),
        "G9_executed": False,
        "protected_holdout_accessed": False,
        "hold_exit_training_started": False,
    }
    write_json(out_dir / "selection.json", selection)

    model_hashes_after = _model_hashes()
    hashes_unchanged = model_hashes_before == model_hashes_after
    if not hashes_unchanged:
        raise ReaggregationError("frozen_campaign_model_hash_changed")
    write_json(
        out_dir / "model_hash_verification.json",
        {
            "schema_version": "Protocol101FrozenModelHashVerificationV1",
            "model_count": len(model_hashes_before),
            "all_unchanged": hashes_unchanged,
            "matches_independent_integrity_authority": (
                model_hashes_after == authoritative_model_hashes
            ),
            "independent_integrity_authority_sha256": sha256_path(
                INDEPENDENT_ROOT / "unit_replay_integrity.json"
            ),
            "model_hashes": model_hashes_after,
        },
    )
    side_effects = {
        "model_fit_or_refit": False,
        "model_scoring_or_replay": False,
        "G9_executed": False,
        "seed45_accessed": False,
        "protected_holdout_accessed": False,
        "learned_exit_training": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "paid_data_download": False,
        "real_money_path_changed": False,
    }
    write_json(out_dir / "side_effect_audit.json", side_effects)
    summary = {
        "schema_version": "Protocol101D1V2CampaignReaggregationSummaryV1",
        "status": "complete",
        "routing_decision": route,
        "D1_v2": {
            "status": d1_result["status"],
            "pass": d1_result["pass"],
            "matched_seeds": d1_result["matched_seed_count"],
            "required_seeds": d1_result["required_seed_count"],
            "absolute_pnl_gate": False,
            "three_dollars_per_trade_role": "diagnostic_only",
        },
        "H2_P5_incremental_edge": {
            "pass": incremental_results["H2/P5"]["pass"],
            "effect_fee_adjusted_pnl": incremental_results["H2/P5"].get(
                "effect_fee_adjusted_pnl"
            ),
            "familywise_confidence_interval_95": incremental_results["H2/P5"][
                "confidence_interval_95"
            ],
            "multiplicity_adjusted_p": incremental_results["H2/P5"][
                "multiplicity_adjusted_p"
            ],
        },
        "rows_reaggregated": len(reaggregated_rows),
        "consumed_input_files_hashed": (
            consumed_manifest["control_chunks"]["file_count"]
            + consumed_manifest["session_chunks"]["file_count"]
        ),
        "candidate_incremental_evidence_complete_rows": sum(
            item["evidence_complete"] is True
            for item in incremental_results.values()
        ),
        "models_refit": 0,
        "frozen_model_hashes_verified": len(model_hashes_after),
        "selected_candidate": selected,
        "G9_executed": False,
        "protected_holdout_accessed": False,
    }
    write_json(out_dir / "summary.json", summary)
    failed_matches = [
        item for item in d1_matches if item["match"]["pass"] is not True
    ]
    failed_lines = [
        f"- Seed {item['seed']}: {', '.join(item['match']['blockers'])}"
        for item in failed_matches
    ] or ["- None"]
    h2_effect = incremental_results["H2/P5"].get(
        "effect_fee_adjusted_pnl"
    )
    h2_interval = incremental_results["H2/P5"]["confidence_interval_95"]
    h2_effect_text = (
        f"${h2_effect:,.2f}" if h2_effect is not None else "not estimable"
    )
    h2_interval_text = (
        f"${h2_interval[0]:,.2f} to ${h2_interval[1]:,.2f}"
        if h2_interval[0] is not None
        else "not estimable because strict exposure matching failed"
    )
    h2_p = incremental_results["H2/P5"].get("multiplicity_adjusted_p")
    h2_p_text = f"{h2_p:.6f}" if h2_p is not None else "not estimable"
    report = f"""# Protocol101 D1 V2 Campaign Reaggregation

Terminal route: `{route}`

## What Changed

The official D1 decision now compares strong-shuffle HGB with an
outcome-blind, feature-independent exposure-matched random P5 selector.
Positive absolute shuffled PnL is not a failure. The former $3-per-trade
equivalence target is diagnostic only.

Real candidates now require a fee-adjusted improvement over their matched
random P5 control whose family-wise lower confidence bound is strictly above
zero and whose multiplicity-adjusted one-sided p-value is at most 0.05.

## Frozen Reaggregation Result

- Frozen models verified unchanged: **420/420**
- Models fitted or refitted: **0**
- Consumed control/session chunks hashed:
  **{summary['consumed_input_files_hashed']}**
- Rows reaggregated: **28/28**
- Rows with complete strict candidate-increment evidence: **0/28**
- D1 matched shuffled seeds: **{d1_result['matched_seed_count']}/20**
- D1 V2 status: **{d1_result['status']}**
- Selected entry candidate: **{selected or 'none'}**

The unmatched D1 seeds were:

{chr(10).join(failed_lines)}

This is insufficient matched-control evidence, not evidence that shuffled HGB
has a positive artifact.

## H2/P5 Incremental Evidence

- Fee-adjusted increment over strictly matched random P5:
  **{h2_effect_text}**
- Family-wise 95% interval:
  **{h2_interval_text}**
- Multiplicity-adjusted p:
  **{h2_p_text}**
- Hard incremental-edge pass:
  **{incremental_results['H2/P5']['pass']}**

The prior H2/P5 `H minus B_real` result remains a useful unmatched diagnostic:
the estimated increment was $14,078 and its ordinary confidence interval
crossed zero. It is not hard selection evidence because the existing random
draws do not satisfy every strict exposure balance requirement.

The other preserved signal rows also fail closed because this frozen control
packet does not contain candidate-specific matched-random evidence for them.

## Boundary

No G9, seed 45, protected holdout, learned-exit training, broker, paper,
promotion, runtime, launchd, paid-data, or real-money action occurred.
"""
    (out_dir / "report.md").write_text(report)
    write_json(
        out_dir / "progress.json",
        {
            "schema_version": "Protocol101D1V2ReaggregationProgressV1",
            "status": "complete",
            "node": "terminal_packet_written",
            "routing_decision": route,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_hashes(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
