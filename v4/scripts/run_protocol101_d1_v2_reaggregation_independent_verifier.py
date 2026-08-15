"""Independent verifier for the Protocol101 D1 V2 reaggregation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from v4.model.protocol101_d1_v2_contract import (
    AMENDMENT_PATH,
    CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
    CANDIDATE_MULTIPLICITY_FAMILY_SIZE,
    CANDIDATE_MULTIPLICITY_ROW_IDS,
    aggregate_exposure,
    evaluate_candidate_incremental_edge,
    evaluate_d1_v2,
    exposure_match,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "v4/audit/autoresearch"
DEFAULT_PACKET = (
    AUDIT_ROOT / "protocol101_d1_v2_campaign_reaggregation_attempt005"
)
DEFAULT_OUT = (
    AUDIT_ROOT
    / "protocol101_d1_v2_campaign_reaggregation_"
    "independent_verification_attempt005"
)
SOURCE_ROOT = (
    AUDIT_ROOT / "protocol101_d1_shuffled_profit_causal_attribution_attempt009"
)
CONTROL_ROOT = SOURCE_ROOT / "work/control_chunks"
SESSION_ROOT = SOURCE_ROOT / "work/session_chunks"
INDEPENDENT_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_"
    "independent_audit_selection_attempt001"
)
ATTEMPT011_ROOT = (
    AUDIT_ROOT
    / "protocol101_d1_shuffled_profit_causal_attribution_"
    "inference_correction_attempt011"
)
CAMPAIGN_ROOT = (
    AUDIT_ROOT / "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
RUNNER_PATH = (
    ROOT / "v4/scripts/run_protocol101_d1_v2_campaign_reaggregation.py"
)
CONTRACT_PATH = ROOT / "v4/model/protocol101_d1_v2_contract.py"
INPUT_PATHS = {
    "reaggregation_runner": RUNNER_PATH,
    "d1_v2_contract": CONTRACT_PATH,
    "amendment": ROOT / AMENDMENT_PATH,
    "independent_audit": INDEPENDENT_ROOT / "independent_audit.json",
    "independent_selection": INDEPENDENT_ROOT / "selection.json",
    "authoritative_unit_integrity": (
        INDEPENDENT_ROOT / "unit_replay_integrity.json"
    ),
    "attempt009_hashes": SOURCE_ROOT / "hashes.sha256",
    "attempt009_control_definitions": SOURCE_ROOT / "control_definitions.json",
    "attempt011_corrected_attribution": (
        ATTEMPT011_ROOT / "causal_attribution_corrected.json"
    ),
    "attempt011_trust_decision": (
        ATTEMPT011_ROOT / "entry_model_trust_decision_corrected.json"
    ),
}
D1_SEEDS = tuple(range(8600, 8620))
REAL_SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
ROW_IDS = CANDIDATE_MULTIPLICITY_ROW_IDS
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BLOCK_SESSIONS = 5
BOOTSTRAP_SEED = 1_017_2026


class IndependentVerificationError(RuntimeError):
    """Raised when the frozen D1 V2 packet cannot be independently accepted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise IndependentVerificationError(f"expected JSON object:{path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify_hash_manifest(packet_dir: Path) -> list[str]:
    blockers: list[str] = []
    manifest = packet_dir / "hashes.sha256"
    if not manifest.is_file():
        return ["producer_hash_manifest_missing"]
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = packet_dir / name
        if not path.is_file():
            blockers.append(f"producer_hashed_file_missing:{name}")
        elif sha256_path(path) != expected:
            blockers.append(f"producer_hashed_file_mismatch:{name}")
    return blockers


def _packet_fingerprint(packet_dir: Path) -> str:
    manifest = packet_dir / "hashes.sha256"
    files = {}
    for line in manifest.read_text().splitlines():
        _, name = line.split("  ", 1)
        path = packet_dir / name
        files[name] = sha256_path(path)
    return stable_hash(
        {
            "manifest_sha256": sha256_path(manifest),
            "files": files,
        }
    )


def _current_model_hashes() -> dict[str, str]:
    paths = sorted(
        CAMPAIGN_ROOT.glob(
            "H*/units/H*/policy*/seed*/expanding_fold_*/model.pkl"
        )
    )
    return {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in paths
    }


def _authoritative_model_hashes() -> dict[str, str]:
    integrity = read_json(INDEPENDENT_ROOT / "unit_replay_integrity.json")
    units = integrity.get("units")
    if not isinstance(units, list) or len(units) != 420:
        raise IndependentVerificationError(
            "authoritative_unit_integrity_grid_mismatch"
        )
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
    files = {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in paths
    }
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def _recomputed_consumed_manifest() -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101D1V2ConsumedInputManifestV1",
        "control_chunks": _hashed_file_manifest(_consumed_control_paths()),
        "session_chunks": _hashed_file_manifest(_consumed_session_paths()),
    }
    payload["manifest_sha256"] = stable_hash(payload)
    return payload


def _control_rows(control: str, fold: int, seed: int) -> list[dict[str, Any]]:
    payload = read_json(
        CONTROL_ROOT / f"{control}_F{fold}_seed{seed}.json"
    )
    rows = payload.get("summaries")
    if not isinstance(rows, list) or not rows:
        raise IndependentVerificationError(
            f"control_rows_missing:{control}:F{fold}:S{seed}"
        )
    return [dict(row) for row in rows]


def _recompute_matching_receipt(
    *,
    left_control: str,
    right_control: str,
    seed: int,
) -> dict[str, Any]:
    left_rows = {
        fold: _control_rows(left_control, fold, seed)[0] for fold in FOLDS
    }
    right_rows = []
    for fold in FOLDS:
        selected = [
            row
            for row in _control_rows(right_control, fold, seed)
            if int(row["draw"]) == 0
        ]
        if len(selected) != 1:
            raise IndependentVerificationError(
                f"fixed_draw_zero_missing:{right_control}:F{fold}:S{seed}"
            )
        right_rows.append(selected[0])

    left_exposure = aggregate_exposure(list(left_rows.values()))
    right_exposure = aggregate_exposure(right_rows)
    pooled = exposure_match(
        left_exposure,
        right_exposure,
        exact_opportunity_set_preserved=False,
        random_selector_receipts_complete=False,
    )
    per_fold = {
        f"F{fold}": exposure_match(
            aggregate_exposure([left_rows[fold]]),
            aggregate_exposure([right_rows[fold - 1]]),
            exact_opportunity_set_preserved=False,
            random_selector_receipts_complete=False,
        )
        for fold in FOLDS
    }
    match = dict(pooled)
    match["pooled_pass"] = pooled["pass"]
    match["per_fold"] = per_fold
    match["pass"] = bool(
        pooled["pass"] and all(item["pass"] for item in per_fold.values())
    )
    match["blockers"] = sorted(
        set(
            pooled["blockers"]
            + [
                f"{fold}:{blocker}"
                for fold, item in per_fold.items()
                for blocker in item["blockers"]
            ]
        )
    )
    return {
        "seed": seed,
        "left_control": left_control,
        "right_control": right_control,
        "selection_rule": "preregistered_fixed_draw_zero_per_fold",
        "selection_used_pnl_or_post_replay_exposure": False,
        "intent_budgets_equal": bool(
            right_control in {"B", "B_real"}
            and all(
                int(row["entry_intents"])
                == int(left_rows[int(row["fold"])]["entry_intents"])
                for row in right_rows
            )
        ),
        "risk_set_identity_receipt": {
            "available": False,
            "reason": (
                "attempt009 compact controls omit the complete governed "
                "risk-set identity hash"
            ),
        },
        "selected_draws": {f"F{fold}": 0 for fold in FOLDS},
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
) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        frame = pd.read_parquet(
            SESSION_ROOT / f"{control}_F{fold}_seed{seed}.parquet"
        )
        frame = frame[frame["draw"].astype(int) == 0].copy()
        if len(frame) != 45:
            raise IndependentVerificationError(
                f"session_draw_incomplete:{control}:F{fold}:S{seed}"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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


def _recompute_inference(
    *,
    left_control: str,
    right_control: str,
    passing_seeds: Sequence[int],
    family_size: int,
) -> dict[str, Any] | None:
    if not passing_seeds:
        return None
    left_frames = []
    right_frames = []
    for seed in passing_seeds:
        left_frames.append(_selected_session_frame(control=left_control, seed=seed))
        right_frames.append(
            _selected_session_frame(control=right_control, seed=seed)
        )
    left = pd.concat(left_frames, ignore_index=True)
    right = pd.concat(right_frames, ignore_index=True)
    keys = ["model_seed", "fold", "session"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise IndependentVerificationError("paired_contrast_duplicate_axis")
    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise IndependentVerificationError(
            "paired_contrast_session_axis_mismatch"
        )
    paired = left[keys + ["net_pnl"]].rename(
        columns={"net_pnl": "left_pnl"}
    ).merge(
        right[keys + ["net_pnl"]].rename(columns={"net_pnl": "right_pnl"}),
        on=keys,
        validate="one_to_one",
    )
    paired["difference"] = paired["left_pnl"] - paired["right_pnl"]
    sessions = sorted(paired["session"].astype(str).unique())
    values = (
        paired.groupby("session", sort=True)["difference"]
        .mean()
        .reindex(sessions, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    observed = float(values.sum())
    draws = values[_moving_block_indexes(len(values))].sum(axis=1, dtype=float)
    centered = draws - float(draws.mean())
    raw_p = float(
        (1 + int(np.sum(centered >= observed)))
        / (BOOTSTRAP_REPLICATES + 1)
    )
    tail = 0.05 / (2.0 * family_size)
    return {
        "observed_paired_fee_adjusted_pnl_difference": observed,
        "session_count": len(values),
        "seed_count": len(passing_seeds),
        "replicates": BOOTSTRAP_REPLICATES,
        "moving_block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
        "confidence_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "familywise_confidence_interval_95": [
            float(np.quantile(draws, tail)),
            float(np.quantile(draws, 1.0 - tail)),
        ],
        "multiplicity_family_size": family_size,
        "one_sided_centered_p": raw_p,
        "multiplicity_adjusted_p": min(1.0, raw_p * family_size),
    }


def verifier_preregistration(packet_dir: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101D1V2IndependentVerificationFreezeV1",
        "producer_packet": str(packet_dir),
        "producer_hash_manifest_sha256": sha256_path(
            packet_dir / "hashes.sha256"
        ),
        "producer_preregistration_sha256": sha256_path(
            packet_dir / "preregistration.json"
        ),
        "verifier_source_sha256": sha256_path(Path(__file__).resolve()),
        "verification_scope": [
            "producer_packet_hashes",
            "all_preregistered_inputs",
            "all_consumed_control_and_session_chunks",
            "420_frozen_model_hashes",
            "D1_matching_and_inference",
            "candidate_matching_and_inference",
            "28_row_gate_and_selection_grid",
            "forbidden_side_effects",
        ],
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def verify(packet_dir: Path) -> dict[str, Any]:
    blockers = _verify_hash_manifest(packet_dir)
    prereg = read_json(packet_dir / "preregistration.json")
    stored_prereg_hash = prereg.get("preregistration_sha256")
    recalculated_prereg_hash = stable_hash(
        {
            key: value
            for key, value in prereg.items()
            if key != "preregistration_sha256"
        }
    )
    if stored_prereg_hash != recalculated_prereg_hash:
        blockers.append("producer_preregistration_hash_mismatch")

    preregistered_inputs = prereg.get("input_hashes", {})
    if set(preregistered_inputs) != set(INPUT_PATHS):
        blockers.append("preregistered_input_set_mismatch")
    for name, source in INPUT_PATHS.items():
        expected = preregistered_inputs.get(name)
        if not source.is_file():
            blockers.append(f"frozen_input_missing:{name}")
        elif sha256_path(source) != expected:
            blockers.append(f"frozen_input_hash_mismatch:{name}")
    candidate_prereg = prereg.get("candidate_incremental_edge", {})
    if (
        tuple(candidate_prereg.get("candidate_family", ())) != ROW_IDS
        or candidate_prereg.get("candidate_family_sha256")
        != CANDIDATE_MULTIPLICITY_FAMILY_SHA256
    ):
        blockers.append("candidate_multiplicity_identity_prereg_mismatch")

    producer_consumed = read_json(
        packet_dir / "consumed_input_manifest.json"
    )
    recomputed_consumed = _recomputed_consumed_manifest()
    if producer_consumed != recomputed_consumed:
        blockers.append("consumed_input_manifest_mismatch")
    for name in ("control_chunks", "session_chunks"):
        prereg_entry = prereg.get("consumed_artifact_manifests", {}).get(name)
        expected = {
            "file_count": recomputed_consumed[name]["file_count"],
            "manifest_sha256": recomputed_consumed[name]["manifest_sha256"],
        }
        if prereg_entry != expected:
            blockers.append(f"consumed_input_preregistration_mismatch:{name}")

    model_verification = read_json(
        packet_dir / "model_hash_verification.json"
    )
    current_hashes = _current_model_hashes()
    authoritative_hashes = _authoritative_model_hashes()
    if len(current_hashes) != 420:
        blockers.append(f"current_model_grid_mismatch:{len(current_hashes)}/420")
    if current_hashes != model_verification.get("model_hashes"):
        blockers.append("frozen_model_hashes_no_longer_match")
    if current_hashes != authoritative_hashes:
        blockers.append("frozen_model_hashes_differ_from_authority")
    if model_verification.get("model_hashes") != authoritative_hashes:
        blockers.append("producer_model_hashes_differ_from_authority")
    if model_verification.get("all_unchanged") is not True:
        blockers.append("producer_model_hash_verification_failed")

    d1 = read_json(packet_dir / "d1_v2_control_result.json")
    producer_receipts = d1.get("matching_receipts", [])
    recomputed_receipts = [
        _recompute_matching_receipt(
            left_control="G",
            right_control="B",
            seed=seed,
        )
        for seed in D1_SEEDS
    ]
    if producer_receipts != recomputed_receipts:
        blockers.append("D1_matching_receipts_not_reproduced")
    passing_d1_seeds = [
        item["seed"]
        for item in recomputed_receipts
        if item["match"]["pass"] is True
    ]
    recomputed_d1_inference = _recompute_inference(
        left_control="G",
        right_control="B",
        passing_seeds=passing_d1_seeds,
        family_size=1,
    )
    if (
        d1.get("inference_on_passing_matches_only")
        != recomputed_d1_inference
    ):
        blockers.append("D1_inference_not_reproduced")
    recomputed_d1 = evaluate_d1_v2(
        exposure_matches=[item["match"] for item in recomputed_receipts],
        effect_fee_adjusted_pnl=(
            recomputed_d1_inference[
                "observed_paired_fee_adjusted_pnl_difference"
            ]
            if recomputed_d1_inference is not None
            else None
        ),
        ci_lower=(
            recomputed_d1_inference["confidence_interval_95"][0]
            if recomputed_d1_inference is not None
            else None
        ),
        ci_upper=(
            recomputed_d1_inference["confidence_interval_95"][1]
            if recomputed_d1_inference is not None
            else None
        ),
        multiplicity_adjusted_p=(
            recomputed_d1_inference["multiplicity_adjusted_p"]
            if recomputed_d1_inference is not None
            else None
        ),
        absolute_shuffled_pnl=d1["equivalence_diagnostic"][
            "absolute_shuffled_pnl"
        ],
        equivalence_bound_dollars=d1["equivalence_diagnostic"][
            "bound_dollars"
        ],
    )
    for key, value in recomputed_d1.items():
        if value != d1.get(key):
            blockers.append(f"independent_D1_mismatch:{key}")
    if (
        len(producer_receipts) != len(D1_SEEDS)
        or tuple(int(item["seed"]) for item in producer_receipts) != D1_SEEDS
        or d1.get("required_seed_count") != len(D1_SEEDS)
    ):
        blockers.append("D1_required_seed_grid_mismatch")
    if d1["equivalence_diagnostic"].get("changes_gate") is not False:
        blockers.append("three_dollar_diagnostic_changed_gate")

    candidates = read_json(
        packet_dir / "candidate_incremental_results.json"
    )["results"]
    if tuple(candidates) != ROW_IDS:
        blockers.append("candidate_result_grid_mismatch")
    h2 = candidates.get("H2/P5", {})
    producer_real_receipts = h2.get("matching_receipts", [])
    recomputed_real_receipts = [
        _recompute_matching_receipt(
            left_control="H",
            right_control="B_real",
            seed=seed,
        )
        for seed in REAL_SEEDS
    ]
    if producer_real_receipts != recomputed_real_receipts:
        blockers.append("candidate_matching_receipts_not_reproduced:H2/P5")
    passing_real_seeds = [
        item["seed"]
        for item in recomputed_real_receipts
        if item["match"]["pass"] is True
    ]
    recomputed_real_inference = _recompute_inference(
        left_control="H",
        right_control="B_real",
        passing_seeds=passing_real_seeds,
        family_size=CANDIDATE_MULTIPLICITY_FAMILY_SIZE,
    )
    if h2.get("inference") != recomputed_real_inference:
        blockers.append("candidate_inference_not_reproduced:H2/P5")

    for row_id, source in candidates.items():
        if row_id == "H2/P5" and recomputed_real_inference is not None:
            independent_effect = recomputed_real_inference[
                "observed_paired_fee_adjusted_pnl_difference"
            ]
            independent_ci = recomputed_real_inference[
                "familywise_confidence_interval_95"
            ]
            independent_p = recomputed_real_inference[
                "multiplicity_adjusted_p"
            ]
            independent_matches = [
                item["match"] for item in recomputed_real_receipts
            ]
            independent_evidence_complete = True
            independent_multiplicity_receipt = True
        else:
            independent_effect = None
            independent_ci = [None, None]
            independent_p = None
            independent_matches = (
                [item["match"] for item in recomputed_real_receipts]
                if row_id == "H2/P5"
                else []
            )
            independent_evidence_complete = False
            independent_multiplicity_receipt = False
        recomputed = evaluate_candidate_incremental_edge(
            row_id=row_id,
            exposure_matches=independent_matches,
            effect_fee_adjusted_pnl=independent_effect,
            ci_lower=independent_ci[0],
            ci_upper=independent_ci[1],
            multiplicity_adjusted_p=independent_p,
            multiplicity_family_size=CANDIDATE_MULTIPLICITY_FAMILY_SIZE,
            multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
            multiplicity_family_sha256=(
                CANDIDATE_MULTIPLICITY_FAMILY_SHA256
            ),
            multiplicity_receipt_complete=(
                independent_multiplicity_receipt
            ),
            costs_included=True,
            evidence_complete=independent_evidence_complete,
        )
        for key, value in recomputed.items():
            if source.get(key) != value:
                blockers.append(
                    f"candidate_incremental_gate_mismatch:{row_id}:{key}"
                )
        if (
            source["multiplicity_family_size"]
            != CANDIDATE_MULTIPLICITY_FAMILY_SIZE
        ):
            blockers.append(f"candidate_multiplicity_family_mismatch:{row_id}")

    rows = read_json(packet_dir / "reaggregated_rows.json")
    if (
        rows.get("row_count") != len(ROW_IDS)
        or tuple(row.get("row_id") for row in rows.get("rows", [])) != ROW_IDS
    ):
        blockers.append("reaggregated_28_row_grid_mismatch")
    for row in rows.get("rows", []):
        expected = bool(
            all(row["gates"][f"G{index}"] for index in range(1, 8))
            and row["maxT_pass"] is True
            and row["D1_v2_pass"] is True
            and row["D5_pass"] is True
            and row["D6_pass"] is True
            and row["candidate_incremental_edge"]["pass"] is True
        )
        if row["hard_gate_eligible_v2"] is not expected:
            blockers.append(f"row_eligibility_mismatch:{row['row_id']}")

    selection = read_json(packet_dir / "selection.json")
    eligible = [
        row["row_id"]
        for row in rows["rows"]
        if row["hard_gate_eligible_v2"] is True
    ]
    if selection.get("eligible_rows") != eligible:
        blockers.append("selection_eligible_rows_mismatch")
    if not eligible and selection.get("selected_candidate") is not None:
        blockers.append("selection_exists_without_eligible_row")
    if d1["status"] == "INSUFFICIENT_MATCHED_CONTROL" and selection.get(
        "routing_decision"
    ) != "corrected_D1_inconclusive_exposure_mismatch":
        blockers.append("terminal_route_mismatch")

    side_effects = read_json(packet_dir / "side_effect_audit.json")
    if any(value is not False for value in side_effects.values()):
        blockers.append("forbidden_side_effect_reported")

    return {
        "schema_version": "Protocol101D1V2IndependentVerificationV2",
        "accepted": not blockers,
        "routing_decision": (
            "d1_v2_reaggregation_independently_verified"
            if not blockers
            else "d1_v2_reaggregation_independent_verification_failed"
        ),
        "blockers": sorted(blockers),
        "checks": {
            "producer_packet_hashes_reproduced": not any(
                item.startswith("producer_hash")
                or item.startswith("producer_hashed")
                for item in blockers
            ),
            "all_preregistered_input_hashes_reproduced": not any(
                item == "preregistered_input_set_mismatch"
                or item.startswith("frozen_input_")
                for item in blockers
            ),
            "all_consumed_chunk_hashes_reproduced": not any(
                item.startswith("consumed_input_")
                for item in blockers
            ),
            "D1_matching_receipts_reproduced": (
                "D1_matching_receipts_not_reproduced" not in blockers
            ),
            "D1_inference_status": (
                "reproduced"
                if recomputed_d1_inference is not None
                else "not_applicable_no_passing_matches"
            ),
            "candidate_matching_receipts_reproduced": not any(
                item.startswith("candidate_matching_receipts_not_reproduced")
                for item in blockers
            ),
            "candidate_inference_status": (
                "reproduced"
                if recomputed_real_inference is not None
                else "not_applicable_no_passing_matches"
            ),
            "frozen_model_hashes_reproduced": (
                len(current_hashes) == 420
                and current_hashes == model_verification.get("model_hashes")
                and current_hashes == authoritative_hashes
            ),
            "models_refit": 0,
            "G9_executed": False,
            "protected_holdout_accessed": False,
        },
    }


def _write_hashes(out_dir: Path) -> None:
    paths = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    )
    (out_dir / "hashes.sha256").write_text(
        "\n".join(f"{sha256_path(path)}  {path.name}" for path in paths)
        + "\n"
    )


def main() -> None:
    args = parse_args()
    packet_dir = args.packet_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise IndependentVerificationError(
            f"immutable_output_already_exists:{out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    producer_packet_fingerprint_before = _packet_fingerprint(packet_dir)
    prereg = verifier_preregistration(packet_dir)
    write_json(out_dir / "preregistration.json", prereg)
    result = verify(packet_dir)
    producer_packet_fingerprint_after = _packet_fingerprint(packet_dir)
    producer_packet_unchanged = (
        producer_packet_fingerprint_before
        == producer_packet_fingerprint_after
    )
    result["checks"]["producer_packet_byte_immutable"] = (
        producer_packet_unchanged
    )
    if not producer_packet_unchanged:
        result["blockers"].append("producer_packet_changed_during_verification")
        result["blockers"] = sorted(set(result["blockers"]))
        result["accepted"] = False
        result["routing_decision"] = (
            "d1_v2_reaggregation_independent_verification_failed"
        )
    write_json(out_dir / "independent_verification.json", result)
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": (
                "Protocol101D1V2IndependentVerificationSummaryV1"
            ),
            "status": "complete",
            "accepted": result["accepted"],
            "routing_decision": result["routing_decision"],
            "producer_packet": str(packet_dir),
            "models_refit": 0,
            "G9_executed": False,
            "protected_holdout_accessed": False,
        },
    )
    (out_dir / "report.md").write_text(
        "# Protocol101 D1 V2 Independent Verification\n\n"
        f"Result: `{result['routing_decision']}`\n\n"
        f"- Accepted: **{result['accepted']}**\n"
        f"- Blockers: **{len(result['blockers'])}**\n"
        "- Producer packet remained byte-immutable during verification.\n"
        "- All consumed control/session chunks were independently rehashed.\n"
        "- Matching receipts and any available paired inference were "
        "recomputed from source chunks.\n"
        "- No models were fitted or refitted; G9 and the protected holdout "
        "remained untouched.\n"
    )
    _write_hashes(out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
