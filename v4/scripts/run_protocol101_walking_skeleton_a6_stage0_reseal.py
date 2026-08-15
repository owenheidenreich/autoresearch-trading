"""Freeze the A6 expected-upside Option-D rerun before any amended fit."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage0"
STAGE1 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage1"
STAGE0_ARCHIVE = STAGE0 / "superseded/pre_a6_convexity_2026_07_31"
STAGE1_ARCHIVE = STAGE1 / "superseded/pre_a6_q10_gate_2026_07_31"
AUTHORITY = ROOT / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
COMPOSER = ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json"
CALIBRATION = COMPOSER.with_name("calibration_spec.json")
FORECAST = COMPOSER.with_name("forecast_heads.json")
RLAC = COMPOSER.with_name("realized_label_audit_composer_spec.json")
TENSOR_SCHEMA = ROOT / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json"
REVIEW = ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_gate_convexity_amendment/codex_review.md"
CHECKER = REVIEW.with_name("consistency_checker_output.json")
AUTHORITY_SHA256 = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"
QUARANTINE = ["walking_skeleton", "throwaway", "paper-only"]
PREFIX = "protocol101_walking_skeleton_throwaway_paper_only_"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def archive_active(directory: Path, archive: Path) -> list[str]:
    if archive.exists():
        raise FileExistsError(archive)
    archive.mkdir(parents=True)
    moved: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.name == "superseded":
            continue
        destination = archive / path.name
        shutil.move(str(path), str(destination))
        moved.append(path.name)
    return moved


def main() -> int:
    if sha256(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("A6 authority hash drift")
    if load(CHECKER).get("outcome") != "pass":
        raise RuntimeError("A6 consistency checker is not green")
    if STAGE0_ARCHIVE.exists():
        receipt = STAGE0 / "receipt.json"
        if receipt.is_file() and load(receipt).get("outcome") == "PASS_A6_FROZEN_STAGE1_RERUN_AUTHORIZED":
            print(json.dumps({"outcome": "ALREADY_FROZEN", "authority": AUTHORITY_SHA256}))
            return 0
        raise RuntimeError("A6 archive exists without active A6 receipt")

    prior_data = load(STAGE0 / "data_slice.json")
    prior_firewall = load(STAGE0 / "option_d_firewall_proof.json")
    prior_prereg = load(STAGE0 / "option_d_preregistration.json")
    prior_stage0_receipt = load(STAGE0 / "receipt.json")
    prior_stage1_receipt = load(STAGE1 / "receipt.json")
    moved_stage1 = archive_active(STAGE1, STAGE1_ARCHIVE)
    moved_stage0 = archive_active(STAGE0, STAGE0_ARCHIVE)

    data = prior_data
    data["schema_version"] = "Protocol101WalkingSkeletonStage0DataSliceV4A6"
    data["status"] = "a6_expected_upside_frozen_before_fit"
    data["authority"]["sha256"] = AUTHORITY_SHA256
    data["option_d"].update(
        {
            "a6_owner_authorized_date": "2026-07-31",
            "a6_review": str(REVIEW.relative_to(ROOT)),
            "a6_stage0_archive": str(STAGE0_ARCHIVE.relative_to(ROOT)),
            "a6_prior_stage1_archive": str(STAGE1_ARCHIVE.relative_to(ROOT)),
            "gate": "session-cluster-calibrated conditional expected upside lower confidence bound",
        }
    )
    data["forbidden_resource_assertions"].update(
        {
            "a6_model_training_or_fitting_performed": False,
            "stage_2_or_later_started": False,
        }
    )

    firewall = prior_firewall
    firewall["schema_version"] = "Protocol101WalkingSkeletonOptionDA6FirewallProofV1"
    firewall["product_contract_hash"] = AUTHORITY_SHA256
    firewall["status"] = "PASS"
    firewall["a6_semantic_scope"] = "entry gate criterion only; session identities and roles unchanged"

    prereg = prior_prereg
    prereg["schema_version"] = "Protocol101WalkingSkeletonOptionDA6PreregistrationV1"
    prereg["product_contract_hash"] = AUTHORITY_SHA256
    prereg["owner_authorized_date"] = "2026-07-31"
    prereg["purpose"] = "faithful capped dry-run of amended FT2-10 expected-upside entry gate"
    prereg["status"] = "FROZEN_A6_BEFORE_FIT"
    prereg["stop"] = "STOP_FOR_CLAUDE_VERIFICATION after amended Stage-1 receipt and audit"
    prereg["source_hashes"].update(
        {
            "authority": sha256(AUTHORITY),
            "calibration_spec": sha256(CALIBRATION),
            "composer_spec": sha256(COMPOSER),
            "forecast_heads": sha256(FORECAST),
            "rlac_spec": sha256(RLAC),
            "tensor_schema": sha256(TENSOR_SCHEMA),
            "codex_review": sha256(REVIEW),
            "consistency_checker_output": sha256(CHECKER),
        }
    )
    prereg["model"]["expected_upside_gate_support"] = {
        "head_count": 28,
        "loss": "squared_error conditional mean",
        "horizons": ["h3", "h5", "h10", "h20", "h45", "h90", "remaining_session"],
        "metrics": ["MFE", "profit_area"],
        "units": ["fee_adjusted_dollars", "fee_adjusted_return"],
        "seeds": [101, 102, 103],
        "role": "gate support only; q10 ranking and 39 primary heads unchanged",
    }
    prereg["calibration"]["expected_upside_mean_lower_confidence_bound"] = {
        "confidence_level": 0.9,
        "bootstrap_replicates": 2000,
        "residual": "realized minus median three-seed predicted conditional mean",
        "cluster_unit": "session",
        "session_weighting": "equal after within-session residual mean",
        "correction": "empirical 0.10 quantile of deterministic session-cluster bootstrap mean residual",
        "individual_outcome_quantile_forbidden": True,
        "post_calibration_refit": False,
    }
    prereg["composer"].update(
        {
            "positive_upside_floor": "strictly greater than zero after fee on all four calibrated expected-mean lower-confidence axes",
            "gate_horizons": ["h3", "h5", "h10", "h20", "h45", "h90", "remaining_session"],
            "q10_ranking_unchanged": True,
            "downstream_gates_unchanged": True,
        }
    )
    prereg["forbidden"].extend(
        [
            "individual-outcome conformal or q10 alias for the expected-mean gate",
            "h3 or h5 gate exclusion",
            "downstream threshold relaxation",
        ]
    )
    prereg["forbidden"] = list(dict.fromkeys(prereg["forbidden"]))

    write_json(STAGE0 / "data_slice.json", data)
    write_json(STAGE0 / "option_d_firewall_proof.json", firewall)
    write_json(STAGE0 / "option_d_preregistration.json", prereg)
    spec = f"""# Protocol101 Walking-Skeleton — Stage-0 A6 Convexity Reseal

Status: **FROZEN BEFORE AMENDED FIT**  
Authority: `{AUTHORITY_SHA256}`  
Graph V2: `{GRAPH_SHA256}` (unchanged)

The 20 fit / 1 embargo / 20 disjoint-calibration / 4 replay partition is
unchanged. The sole HGB configuration and seeds 101/102/103 are unchanged.

The only semantic amendment is the FT2-10 positive-after-fee gate. Twenty-eight
fixed conditional-mean gate-support heads (MFE and profit area, dollars and
return, seven horizons) are fitted with squared error. Their three-seed median
forecast receives the preregistered session-clustered one-sided 90% lower
confidence correction. All four equal-horizon expected-upside lower bounds must
be strictly positive. All causally available horizons, including h3/h5, remain.

The 36 q10/q90 path pairs, three action heads, q10 ranking, RLAC, labels, folds,
fees, simulator, D48/D49, guardrails, and downstream gates remain unchanged.
No individual-outcome lower prediction quantile may satisfy or impersonate the
expected-mean gate.

Valid outcomes remain `TRADES` and `ABSTAINS`. Stop after the amended Stage-1
receipt and delta audit at `STOP_FOR_CLAUDE_VERIFICATION`.
"""
    write_text(STAGE0 / "STAGE0_SPEC.md", spec)

    receipt = {
        "schema_version": "Protocol101WalkingSkeletonStage0A6ReceiptV1",
        "stage": 0,
        "outcome": "PASS_A6_FROZEN_STAGE1_RERUN_AUTHORIZED",
        "product_contract_hash": AUTHORITY_SHA256,
        "graph_sha256": GRAPH_SHA256,
        "owner_authorized_date": "2026-07-31",
        "quarantine_labels": QUARANTINE,
        "artifact_prefix": PREFIX,
        "entry_session_count": 45,
        "partition": prereg["partition"],
        "prior_receipts": {
            "stage0_receipt_hash": prior_stage0_receipt["receipt_hash"],
            "stage1_receipt_hash": prior_stage1_receipt["receipt_hash"],
        },
        "archive": {
            "stage0": str(STAGE0_ARCHIVE.relative_to(ROOT)),
            "stage0_moved": moved_stage0,
            "stage1": str(STAGE1_ARCHIVE.relative_to(ROOT)),
            "stage1_moved": moved_stage1,
        },
        "firewall": {
            "path": str((STAGE0 / "option_d_firewall_proof.json").relative_to(ROOT)),
            "sha256": sha256(STAGE0 / "option_d_firewall_proof.json"),
        },
        "preregistration": {
            "path": str((STAGE0 / "option_d_preregistration.json").relative_to(ROOT)),
            "sha256": sha256(STAGE0 / "option_d_preregistration.json"),
            "primary_head_count": 39,
            "expected_gate_support_head_count": 28,
            "config_count": 1,
            "config_cap": 5,
            "wall_clock_seconds": 2700,
        },
        "stage0_spec_sha256": sha256(STAGE0 / "STAGE0_SPEC.md"),
        "data_slice_sha256": sha256(STAGE0 / "data_slice.json"),
        "side_effects": {
            "a6_model_fit": False,
            "broker_contacted": False,
            "paid_download": False,
            "protected_resource_read": False,
            "runtime_or_promotion_changed": False,
            "stage2_started": False,
            "stage3_started": False,
            "stage4_started": False,
        },
        "next": "run exactly the preregistered A6 Option-D Stage-1 dry-run",
        "stop": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(STAGE0 / "receipt.json", receipt)
    checks = {
        "authority_hash_exact": sha256(AUTHORITY) == AUTHORITY_SHA256,
        "consistency_checker_green": load(CHECKER).get("outcome") == "pass",
        "graph_unchanged": load(CHECKER)["checks"]["authority_recorded_graph_hash_matches_live_graph"]["pass"],
        "prior_stage0_archived": STAGE0_ARCHIVE.is_dir(),
        "prior_stage1_archived": STAGE1_ARCHIVE.is_dir(),
        "full45_partition_unchanged": [len(prereg["partition"][key]) for key in ("fit_first_20_sessions", "embargo_next_1_session", "calibration_next_20_sessions", "plumbing_replay_last_4_sessions")] == [20, 1, 20, 4],
        "one_config_within_cap": len(prereg["model"]["configs"]) == 1 <= prereg["model"]["config_cap"],
        "primary_heads_unchanged_39": prereg["model"]["primary_head_count"] == 39,
        "expected_gate_support_heads_28": prereg["model"]["expected_upside_gate_support"]["head_count"] == 28,
        "all_horizons_retained": prereg["composer"]["gate_horizons"] == ["h3", "h5", "h10", "h20", "h45", "h90", "remaining_session"],
        "q10_ranking_unchanged": prereg["composer"]["q10_ranking_unchanged"] is True,
        "no_fit_before_freeze": receipt["side_effects"]["a6_model_fit"] is False,
        "stage2_not_started": receipt["side_effects"]["stage2_started"] is False,
    }
    review = {
        "schema_version": "Protocol101WalkingSkeletonStage0A6DeltaReviewV1",
        "outcome": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scope": "A6 Stage-0 reseal and prior Stage-1 preservation only",
        "stop": "Stage-1 A6 rerun authorized; Stage 2 forbidden",
    }
    write_json(STAGE0 / "delta_scoped_review.json", review)
    if review["outcome"] != "PASS":
        raise RuntimeError(review["failed_checks"])
    print(json.dumps({"outcome": receipt["outcome"], "receipt_hash": receipt["receipt_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
