"""Freeze the owner-authorized Option-D walking-skeleton amendment.

This command performs specification and archival work only.  It never builds
a tensor, fits a model, contacts a broker, or downloads data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "v4/audit/autoresearch"
STAGE0 = AUDIT / "protocol101_walking_skeleton_stage0"
STAGE1 = AUDIT / "protocol101_walking_skeleton_stage1"
AUTHORITY = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_SHA256 = "edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3"
CENSUS = (
    AUDIT / "protocol101_ft2_04_path_label_freeze/census_sessions.json"
)
PROCESSED_MANIFEST = (
    AUDIT
    / "protocol101_live_v2_microstructure_masked_15mo_training_preflight/"
    "canonical_processed_session_manifest.json"
)
TENSOR_SCHEMA = (
    AUDIT
    / "protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json"
)
COMPOSER_SPEC = (
    AUDIT
    / "protocol101_ft2_10_entry_science_contract/composer_spec.json"
)
CALIBRATION_SPEC = COMPOSER_SPEC.with_name("calibration_spec.json")
FORECAST_HEADS = COMPOSER_SPEC.with_name("forecast_heads.json")
RLAC_SPEC = COMPOSER_SPEC.with_name("realized_label_audit_composer_spec.json")
SEALED_ASSIGNMENT = AUDIT / "protocol101_sealed_day_assignment/last_assignment_run.json"
PRE_ARCHIVE = STAGE0 / "superseded/option_d_pre_amendment_2026_07_31"
PRIOR_STAGE1 = STAGE1 / "superseded/quick_gbt_all_wait_2026_07_31"
QUARANTINE = ["walking_skeleton", "throwaway", "paper-only"]
PREFIX = "protocol101_walking_skeleton_throwaway_paper_only_"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_authority() -> None:
    observed = sha256_path(AUTHORITY)
    if observed != AUTHORITY_SHA256:
        raise RuntimeError(f"authority hash drift: {observed}")


def archive_prior() -> dict[str, Any]:
    PRE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    stage0_files = (
        "STAGE0_SPEC.md",
        "data_slice.json",
        "firewall_safety_proof.json",
        "receipt.json",
        "delta_scoped_review.json",
    )
    bindings: list[dict[str, Any]] = []
    for name in stage0_files:
        source = STAGE0 / name
        destination = PRE_ARCHIVE / name
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256_path(source) != sha256_path(destination):
            raise RuntimeError(f"Stage-0 archive mismatch: {name}")
        bindings.append(
            {
                "source": str(source.relative_to(ROOT)),
                "archive": str(destination.relative_to(ROOT)),
                "sha256": sha256_path(destination),
            }
        )
    if not PRIOR_STAGE1.exists():
        if not (STAGE1 / "receipt.json").is_file():
            raise RuntimeError("prior Stage-1 receipt missing before Option-D archive")
        prior = load_json(STAGE1 / "receipt.json")
        if prior.get("outcome") != "FAIL_STOP_BEFORE_STAGE2":
            raise RuntimeError("unexpected prior Stage-1 outcome")
        temporary = AUDIT / ".protocol101_walking_skeleton_stage1_option_d_archive"
        if temporary.exists():
            raise RuntimeError(f"temporary archive path exists: {temporary}")
        STAGE1.rename(temporary)
        (STAGE1 / "superseded").mkdir(parents=True, exist_ok=False)
        temporary.rename(PRIOR_STAGE1)
    prior_receipt = PRIOR_STAGE1 / "receipt.json"
    if not prior_receipt.is_file():
        raise RuntimeError("archived prior Stage-1 receipt missing")
    return {
        "stage0_bindings": bindings,
        "prior_stage1": {
            "path": str(PRIOR_STAGE1.relative_to(ROOT)),
            "receipt_sha256": sha256_path(prior_receipt),
            "outcome": load_json(prior_receipt)["outcome"],
        },
    }


def partition(sessions: list[str]) -> dict[str, list[str]]:
    if len(sessions) != 45:
        raise RuntimeError(f"expected 45 census sessions, got {len(sessions)}")
    return {
        "fit_first_20_sessions": sessions[:20],
        "embargo_next_1_session": sessions[20:21],
        "calibration_next_20_sessions": sessions[21:41],
        "plumbing_replay_last_4_sessions": sessions[41:],
    }


def firewall(sessions: list[str]) -> dict[str, Any]:
    census = load_json(CENSUS)
    outer = {
        value
        for values in census["outer_test_sessions_by_fold"].values()
        for value in values
    }
    protected = set(census["protected_holdout_sessions"])
    embargo = set(census["embargo_sessions"])
    sealed_run = load_json(SEALED_ASSIGNMENT)
    sealed_or_burned = {
        row["session"]
        for row in sealed_run["actions"]
        if row.get("class") in {"sealed", "burned"}
    }
    selected = set(sessions)
    rows = []
    for session in sessions:
        rows.append(
            {
                "session": session,
                "in_census": session in set(census["census_sessions"]),
                "outer_test_selected": session in outer,
                "protected_holdout_selected": session in protected,
                "embargo_selected": session in embargo,
                "sealed_or_burned_selected": session in sealed_or_burned,
                "firewall_safe": not bool(
                    {session} & (outer | protected | embargo | sealed_or_burned)
                ),
            }
        )
    intersections = {
        "outer_test": sorted(selected & outer),
        "protected_holdout": sorted(selected & protected),
        "embargo": sorted(selected & embargo),
        "sealed_or_burned": sorted(selected & sealed_or_burned),
    }
    gate = all(not values for values in intersections.values()) and all(
        row["in_census"] and row["firewall_safe"] for row in rows
    )
    return {
        "schema_version": "Protocol101WalkingSkeletonOptionDFirewallProofV1",
        "status": "PASS" if gate else "FAIL",
        "product_contract_hash": AUTHORITY_SHA256,
        "scope": "all 45 FT2-04 census development sessions",
        "session_count": len(sessions),
        "sessions": rows,
        "intersections": intersections,
        "summary": {
            "all_45_in_census": all(row["in_census"] for row in rows),
            "outer_test_intersection_count": len(intersections["outer_test"]),
            "protected_holdout_intersection_count": len(
                intersections["protected_holdout"]
            ),
            "embargo_intersection_count": len(intersections["embargo"]),
            "sealed_or_burned_intersection_count": len(
                intersections["sealed_or_burned"]
            ),
            "gate": "PASS" if gate else "FAIL",
        },
        "sources": {
            "census": {
                "path": str(CENSUS.relative_to(ROOT)),
                "sha256": sha256_path(CENSUS),
            },
            "sealed_assignment": {
                "path": str(SEALED_ASSIGNMENT.relative_to(ROOT)),
                "sha256": sha256_path(SEALED_ASSIGNMENT),
            },
        },
        "confirmation_disposition": (
            "Membership in the pinned D42 census is dispositive: frozen "
            "confirmation dates are subtracted from that census when assigned."
        ),
    }


def preregistration(sessions: list[str]) -> dict[str, Any]:
    parts = partition(sessions)
    primary_heads = [
        "8 quality-screen conservative heads",
        "28 horizon-specific conservative-upside heads",
        "WAIT probability",
        "expected normalized regret",
        "q90 normalized regret",
    ]
    return {
        "schema_version": "Protocol101WalkingSkeletonOptionDPreregistrationV1",
        "status": "FROZEN_BEFORE_FIT",
        "owner_authorized_date": "2026-07-30",
        "product_contract_hash": AUTHORITY_SHA256,
        "purpose": "faithful capped dry-run of real FT2-10 entry plumbing",
        "scientific_claim_allowed": False,
        "quarantine_labels": QUARANTINE,
        "artifact_prefix": PREFIX,
        "sessions": sessions,
        "partition": parts,
        "partition_rationale": (
            "20 fit and 20 disjoint calibration sessions satisfy the exact "
            "FT2-10 minima; one chronological embargo separates them and the "
            "last four sessions remain plumbing replay."
        ),
        "model": {
            "family": "sklearn HistGradientBoosting tabular only",
            "config_cap": 5,
            "preregistered_config_count": 1,
            "configs": [
                {
                    "config_id": "option_d_hgb_c1",
                    "seeds": [101, 102, 103],
                    "max_iter": 200,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "min_samples_leaf": 30,
                    "l2_regularization": 1.0,
                    "early_stopping": False,
                    "max_finite_examples_per_head": 1000000,
                    "selection_status": "sole preregistered configuration; no result-based selection",
                }
            ],
            "primary_head_count": 39,
            "primary_head_inventory": primary_heads,
            "continuous_calibration_companions": (
                "Every one of the 36 continuous primary path heads fits the "
                "opposite q10/q90 endpoint from the same features, rows, config, "
                "and seed. Companions exist only to form the signed 80% interval."
            ),
            "ensemble": (
                "Calibrate each seed-specific interval on the same disjoint "
                "calibration role, then take the median calibrated endpoint "
                "across seeds exactly as forecast_heads.json requires."
            ),
            "gpu_allowed": False,
            "neural_allowed": False,
        },
        "calibration": {
            "method": "same-final-model disjoint split conformalized quantile regression",
            "coverage": 0.8,
            "nonconformity": "max(predicted_q10-y,y-predicted_q90)",
            "finite_sample_order": "k=min(n,ceil((n+1)*0.8)); one-based kth sorted score",
            "calibrated_pair": "[predicted_q10-qhat,predicted_q90+qhat]",
            "phase_specific_minimum": {
                "distinct_sessions": 8,
                "valid_rows": 2000,
                "fallback": "global same-head disjoint-calibration qhat",
            },
            "post_calibration_refit": False,
        },
        "composer": {
            "implementation": "real FT2-10 order and frozen values; no threshold changes",
            "guardrail_anchor_grid": [0.0, 0.1, 0.2, 0.25, 0.3],
            "uncertainty_multiplier_grid": [1.0, 1.25, 1.5, 2.0],
            "headline_control": {"guardrail_alpha": 0.0, "k": 1.0},
            "other_grid_members": "report-only fixed-contract sensitivity; never used to chase BUY",
            "positive_upside_floor": "strictly greater than zero after fee on all four axes",
            "q90_regret_upper_bound_max": 0.10,
            "action_conditioned_gate_enforced": True,
        },
        "capability_budget": {
            "wall_clock_seconds": 2700,
            "config_cap": 5,
            "actual_preregistered_configs": 1,
            "stop_when_cap_reached": True,
            "no_additional_config_after_outcome": True,
        },
        "valid_outcomes": ["TRADES", "ABSTAINS"],
        "forbidden": [
            "composer threshold changes",
            "conservative-upside bar changes",
            "guardrail-anchor changes",
            "fabricated or injected BUY",
            "more than five configurations",
            "outer-test, protected-holdout, confirmation, or sealed evidence",
            "broker contact or paper submit",
            "download",
            "Stage 2 or later",
        ],
        "stop": "STOP_FOR_OWNER_AND_FABLE_REVIEW after Stage-1 TRADES or ABSTAINS receipt",
        "source_hashes": {
            "authority": sha256_path(AUTHORITY),
            "census": sha256_path(CENSUS),
            "processed_manifest": sha256_path(PROCESSED_MANIFEST),
            "tensor_schema": sha256_path(TENSOR_SCHEMA),
            "composer_spec": sha256_path(COMPOSER_SPEC),
            "calibration_spec": sha256_path(CALIBRATION_SPEC),
            "forecast_heads": sha256_path(FORECAST_HEADS),
            "rlac_spec": sha256_path(RLAC_SPEC),
        },
    }


def spec_markdown(prereg: dict[str, Any]) -> str:
    parts = prereg["partition"]
    return f"""# Protocol101 Walking-Skeleton — Stage-0 Option-D Amendment

**Status:** `OPTION_D_FROZEN_BEFORE_FIT`  
**Owner authorization:** `2026-07-30`  
**Authority:** `{AUTHORITY_SHA256}`

## Purpose

Option D supersedes only the original quick-GBT entry mock. The prior Stage-0
specification and the genuine all-WAIT Stage-1 run are preserved under
`superseded/`. This amendment authorizes one faithful, capped historical
dry-run of the real FT2-10 entry pipeline. It is plumbing evidence only—not a
scientific campaign, alpha claim, promotion candidate, or paper-readiness claim.

Binding artifact labels are `{QUARANTINE[0]}`, `{QUARANTINE[1]}`, and
`{QUARANTINE[2]}`. Every new artifact uses prefix `{PREFIX}`.

## Clean 45-session slice

The slice is exactly the full FT2-04 census development set. Its machine-
readable authority is `data_slice.json`; its independent set-intersection proof
is `option_d_firewall_proof.json`.

- fit: {len(parts['fit_first_20_sessions'])} sessions
- chronological embargo: {len(parts['embargo_next_1_session'])} session
- disjoint conformal calibration: {len(parts['calibration_next_20_sessions'])} sessions
- plumbing replay: {len(parts['plumbing_replay_last_4_sessions'])} sessions

No market data is downloaded. Outer-test, protected-holdout, confirmation,
embargo, sealed, and burned evidence remain closed.

## Faithful capped entry model

One HGB configuration is preregistered, below the hard cap of five. It uses
three independent seeds (101/102/103), 200 iterations, learning rate 0.05,
depth 4, minimum leaf 30, and L2 1.0. It fits the 39 composer-required primary
heads. Each of the 36 continuous path heads also fits its paired q10/q90
endpoint solely for signed 80% split-conformal calibration. The calibrated
seed endpoints are median-aggregated. There is no neural model, GPU, feature
search, result-based configuration selection, or post-calibration refit.

The wall-clock cap is 2,700 seconds. Reaching it stops the attempt; it never
authorizes another configuration.

## Real calibration and composer

Continuous conservative bounds follow `calibration_spec.json` exactly:

```text
score = max(predicted_q10 - y, y - predicted_q90)
k = min(n, ceil((n + 1) * 0.80))
qhat = kth sorted score
calibrated interval = [predicted_q10 - qhat, predicted_q90 + qhat]
```

Phase-specific calibration requires at least eight distinct sessions and 2,000
valid rows; otherwise it falls back to the global same-head disjoint-
calibration score. The real FT2-10 composer order and every frozen threshold
remain unchanged. The complete anchor grid and uncertainty-multiplier grid are
reported, while alpha=0/k=1 remains the preregistered headline control. BUY
counts never select a configuration or alter a threshold.

## Valid outcomes and stop

- `TRADES`: report exact-contract BUYs and serial legality; Stage 2 is merely
  unblocked, not started.
- `ABSTAINS`: report conservative-upside distributions, near misses, and the
  evidence-supported tentative cause; downstream remains pending owner choice.

Both end `STOP_FOR_OWNER_AND_FABLE_REVIEW`. Stage 2+, broker activity, paper
submit, downloads, runtime changes, launchd changes, and promotion changes are
forbidden.
"""


def run() -> dict[str, Any]:
    verify_authority()
    archive = archive_prior()
    census = load_json(CENSUS)
    sessions = list(census["census_sessions"])
    parts = partition(sessions)
    proof = firewall(sessions)
    if proof["status"] != "PASS":
        raise RuntimeError(f"Option-D firewall failed: {proof['intersections']}")
    prereg = preregistration(sessions)
    proof_path = STAGE0 / "option_d_firewall_proof.json"
    prereg_path = STAGE0 / "option_d_preregistration.json"
    write_json_atomic(proof_path, proof)
    write_json_atomic(prereg_path, prereg)
    previous_data = load_json(PRE_ARCHIVE / "data_slice.json")
    previous_data["schema_version"] = "Protocol101WalkingSkeletonStage0DataSliceV3OptionD"
    previous_data["status"] = "option_d_frozen_before_fit"
    previous_data["entry_minute_slice"] = {
        "selection_rule": "exact full 45-session FT2-04 census development set",
        "session_count": 45,
        "sessions": sessions,
        "frozen_partition": parts,
        "role": "walking_skeleton_throwaway_plumbing_only",
        "model_quality_claim_allowed": False,
        "outer_test_intersection_count": 0,
        "protected_holdout_intersection_count": 0,
        "embargo_intersection_count": 0,
        "sealed_or_burned_intersection_count": 0,
        "ownership_and_role_sources": [
            {"path": str(CENSUS.relative_to(ROOT)), "sha256": sha256_path(CENSUS)},
            {
                "path": str(PROCESSED_MANIFEST.relative_to(ROOT)),
                "sha256": sha256_path(PROCESSED_MANIFEST),
            },
        ],
    }
    previous_data["option_d"] = {
        "owner_authorized_date": "2026-07-30",
        "firewall_proof": str(proof_path.relative_to(ROOT)),
        "preregistration": str(prereg_path.relative_to(ROOT)),
        "prior_stage0_archive": str(PRE_ARCHIVE.relative_to(ROOT)),
        "prior_stage1_archive": str(PRIOR_STAGE1.relative_to(ROOT)),
    }
    previous_data["forbidden_resource_assertions"] = {
        "protected_holdout_selected": False,
        "fresh_confirmation_seed_selected": False,
        "sealed_recorder_evidence_selected": False,
        "broker_contacted": False,
        "paid_download_performed": False,
        "option_d_model_training_or_fitting_performed": False,
        "option_d_tensor_build_performed": False,
        "runtime_or_promotion_changed": False,
        "stage_2_or_later_started": False,
        "prior_stage1_all_wait_run_preserved_under_superseded": True,
    }
    write_json_atomic(STAGE0 / "data_slice.json", previous_data)
    write_text_atomic(STAGE0 / "STAGE0_SPEC.md", spec_markdown(prereg))
    receipt = {
        "schema_version": "Protocol101WalkingSkeletonStage0OptionDReceiptV1",
        "stage": 0,
        "outcome": "PASS_OPTION_D_FROZEN_STAGE1_RERUN_AUTHORIZED",
        "product_contract_hash": AUTHORITY_SHA256,
        "owner_authorized_date": "2026-07-30",
        "quarantine_labels": QUARANTINE,
        "artifact_prefix": PREFIX,
        "archive": archive,
        "entry_session_count": 45,
        "partition": parts,
        "firewall": {
            "path": str(proof_path.relative_to(ROOT)),
            "sha256": sha256_path(proof_path),
            "gate": proof["status"],
        },
        "preregistration": {
            "path": str(prereg_path.relative_to(ROOT)),
            "sha256": sha256_path(prereg_path),
            "config_cap": 5,
            "preregistered_config_count": 1,
            "wall_clock_seconds": 2700,
        },
        "stage0_spec_sha256": sha256_path(STAGE0 / "STAGE0_SPEC.md"),
        "data_slice_sha256": sha256_path(STAGE0 / "data_slice.json"),
        "side_effects": {
            "option_d_tensor_build": False,
            "option_d_model_fit": False,
            "broker_contacted": False,
            "paid_download": False,
            "protected_resource_read": False,
            "stage2_started": False,
            "stage3_started": False,
            "stage4_started": False,
            "runtime_or_promotion_changed": False,
        },
        "next": "run exactly the preregistered Option-D Stage-1 dry-run",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json_atomic(STAGE0 / "receipt.json", receipt)
    checks = {
        "authority_hash_matches": sha256_path(AUTHORITY) == AUTHORITY_SHA256,
        "prior_stage0_preserved": all(
            (PRE_ARCHIVE / name).is_file()
            for name in (
                "STAGE0_SPEC.md",
                "data_slice.json",
                "firewall_safety_proof.json",
                "receipt.json",
                "delta_scoped_review.json",
            )
        ),
        "prior_stage1_preserved": (PRIOR_STAGE1 / "receipt.json").is_file(),
        "exact_45_census_sessions": sessions == census["census_sessions"],
        "partition_20_1_20_4": [len(value) for value in parts.values()] == [20, 1, 20, 4],
        "firewall_pass": proof["status"] == "PASS",
        "config_count_within_cap": len(prereg["model"]["configs"]) <= 5,
        "one_config_preregistered": len(prereg["model"]["configs"]) == 1,
        "conformal_method_exact": prereg["calibration"]["nonconformity"]
        == "max(predicted_q10-y,y-predicted_q90)",
        "composer_values_unmodified": prereg["composer"]["q90_regret_upper_bound_max"] == 0.10,
        "no_option_d_fit_started": receipt["side_effects"]["option_d_model_fit"] is False,
        "stage2_plus_absent": not any(
            receipt["side_effects"][key]
            for key in ("stage2_started", "stage3_started", "stage4_started")
        ),
    }
    review = {
        "schema_version": "Protocol101WalkingSkeletonOptionDStage0ReviewV1",
        "stage": 0,
        "scope": "Option-D amendment only",
        "outcome": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pass_count": sum(checks.values()),
        "check_count": len(checks),
        "failed_checks": [key for key, value in checks.items() if not value],
        "stop_if_failed": True,
        "stage2_started": False,
        "stage4_started": False,
    }
    review["review_hash"] = stable_hash(review)
    write_json_atomic(STAGE0 / "delta_scoped_review.json", review)
    if review["outcome"] != "PASS":
        raise RuntimeError(f"Option-D amendment review failed: {review['failed_checks']}")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    run()
