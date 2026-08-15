"""Emit the Protocol101 fair-contract feature recovery ladder plan.

This is an offline governance artifact. It records that the v2 masked contract
is the fair baseline control, not the final feature ceiling, and defines the
evidence required before any masked feature group can become model-facing
alpha.

It does not train, tune thresholds, contact brokers/vendors, change defaults,
edit runtime flags, touch launchd, or authorize paper-submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "Protocol101FeatureRecoveryLadderPlanV1"
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_feature_recovery_ladder_plan"
)
DEFAULT_STEERING_DOC = Path(
    "v4/docs/protocol101/synchronization/history/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md"
)
DEFAULT_TRAINING_DOC = Path(
    "v4/docs/protocol101/synchronization/history/PROTOCOL101_FAIR_CONTRACT_TRAINING_PHASE_2026_07_08.md"
)
DEFAULT_STAGE1_GATES_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
)
DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/runner_plan.json"
)
DEFAULT_TRAINING_SCOPE = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance/summary.json"
)
DEFAULT_PARITY_CERT = Path(
    "v4/docs/protocol101/synchronization/contracts/PROTOCOL101_PARITY_CERTIFICATION_V2_MICROSTRUCTURE_MASKED_2026_07_07.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steering-doc", type=Path, default=DEFAULT_STEERING_DOC)
    parser.add_argument("--training-doc", type=Path, default=DEFAULT_TRAINING_DOC)
    parser.add_argument("--stage1-gates-doc", type=Path, default=DEFAULT_STAGE1_GATES_DOC)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--training-scope", type=Path, default=DEFAULT_TRAINING_SCOPE)
    parser.add_argument("--parity-cert", type=Path, default=DEFAULT_PARITY_CERT)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
    }


def ladder_groups() -> list[dict[str, Any]]:
    return [
        {
            "order": 0,
            "feature_group": "masked_v2_baseline_control",
            "current_role": "model_facing_alpha",
            "status": "active_control",
            "description": (
                "All model-facing features that remain after "
                "mask_vendor_sensitive_option_quote_greek_microstructure."
            ),
            "required_before_use": [
                "train Stage-1 masked baseline through governed five-fold CV",
                "record nulls, canaries, gates, and failure/success packet",
            ],
            "may_skip_parity_gate": True,
        },
        {
            "order": 1,
            "feature_group": "stable_index_context_refinements",
            "current_role": "mostly_model_facing_or_contract_semantics",
            "status": "eligible_for_repair_after_baseline",
            "examples": ["SPX/VIX timing", "VWAP", "momentum", "OMAR/range semantics"],
            "parity_focus": "same completed-minute timing and one-minute context lag",
            "uplift_focus": "improves CV without creating timing fragility",
        },
        {
            "order": 2,
            "feature_group": "candidate_geometry_and_moneyness",
            "current_role": "partly_model_facing",
            "status": "eligible_for_repair_after_baseline",
            "examples": ["right", "offset_points", "moneyness buckets", "local ladder geometry"],
            "parity_focus": "same contract universe, strike ladder, and candidate membership",
            "uplift_focus": "improves selected-contract quality without overfitting to one side/time",
        },
        {
            "order": 3,
            "feature_group": "internally_computed_greeks_and_iv",
            "current_role": "preserved_for_audit_masked_from_alpha",
            "status": "blocked_until_parity_plus_uplift",
            "examples": ["iv", "delta", "gamma", "theta", "breakeven_distance"],
            "parity_focus": "identical historical/live calculation code and source inputs",
            "uplift_focus": "adds convexity/risk signal after jitter and seed robustness",
        },
        {
            "order": 4,
            "feature_group": "normalized_liquidity_and_spread",
            "current_role": "raw_values_preserved_for_tradability_and_fills",
            "status": "blocked_until_parity_plus_uplift",
            "examples": ["spread_frac buckets", "spread-to-premium", "size imbalance buckets"],
            "parity_focus": "bounded drift across IBKR and historical quote views",
            "uplift_focus": "improves trade filtering without selecting quote noise",
        },
        {
            "order": 5,
            "feature_group": "volume_and_open_interest_semantics",
            "current_role": "preserved_if_available_masked_from_alpha",
            "status": "blocked_until_parity_plus_uplift",
            "examples": ["option_ohlcv_volume", "stat_open_interest"],
            "parity_focus": "causal timestamp and source semantics match live availability",
            "uplift_focus": "adds liquidity/participation signal beyond tradability guards",
        },
        {
            "order": 6,
            "feature_group": "raw_vendor_quote_microstructure",
            "current_role": "guards_fills_labels_pnl_audit_only",
            "status": "exceptional_addback_only",
            "examples": ["bid", "ask", "mid", "spread", "bid_size", "ask_size"],
            "parity_focus": "strict bounded drift and no non-threshold action flips",
            "uplift_focus": "proves robust net benefit after stress and feature jitter",
        },
    ]


def gate_policy() -> dict[str, Any]:
    return {
        "feature_addback_requires": [
            "preregistered_experiment_before_results",
            "paired_ibkr_vs_historical_parity_report",
            "same_decision_timestamp_contract",
            "bounded_feature_drift",
            "no_unclassified_non_threshold_action_flips",
            "out_of_sample_cv_uplift_after_fees_and_stress",
            "null_canary_comparison",
            "drawdown_concentration_seed_and_jitter_gates",
            "explicit_keep_or_reject_decision",
        ],
        "forbidden_shortcuts": [
            "add_feature_group_because_masked_baseline_pnl_is_weak",
            "use_vendor_sensitive_feature_as_alpha_without_live_reproducible_semantics",
            "tune_on_protected_holdout_recorder_or_parity_confirmation_sessions",
            "weaken_v2_mask_inside_stage1_baseline",
            "claim_paper_readiness_from_offline_training",
        ],
    }


def side_effect_policy() -> dict[str, bool]:
    return {
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_download_allowed": False,
        "promotion_or_default_change_allowed": False,
        "runtime_flag_edit_allowed": False,
        "launchd_change_allowed": False,
        "real_money_path_change_allowed": False,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    runner_plan = load_json_if_exists(args.runner_plan)
    training_scope = load_json_if_exists(args.training_scope)
    blockers: list[str] = []
    if runner_plan and runner_plan.get("selected_feature_contract") != CONTRACT:
        blockers.append("runner_plan_contract_not_v2_microstructure_masked")
    if runner_plan and runner_plan.get("model_scoring_feature_transform") != TRANSFORM:
        blockers.append("runner_plan_transform_not_required_mask")
    if runner_plan and runner_plan.get("paper_submit_allowed") is not False:
        blockers.append("runner_plan_paper_submit_not_false")
    if training_scope and training_scope.get("status") != "pass":
        blockers.append("training_scope_not_pass")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now_utc(),
        "status": "pass" if not blockers else "blocked",
        "decision": (
            "run_masked_baseline_first_then_feature_recovery_ladder"
            if not blockers
            else "repair_training_scope_before_feature_recovery_planning"
        ),
        "feature_contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "masked_baseline_is_control_not_ceiling": True,
        "raw_fields_preserved_for_market_mechanics": True,
        "feature_addback_policy": "parity_plus_uplift_required",
        "blockers": blockers,
        "artifacts": {
            "steering_doc": artifact(args.steering_doc),
            "training_doc": artifact(args.training_doc),
            "stage1_gates_doc": artifact(args.stage1_gates_doc),
            "runner_plan": artifact(args.runner_plan),
            "training_scope": artifact(args.training_scope),
            "parity_certification": artifact(args.parity_cert),
        },
        "current_training_scope": {
            "status": training_scope.get("status"),
            "session_count": training_scope.get("session_count"),
            "pass_count": training_scope.get("pass_count"),
            "registry_hash": training_scope.get("registry_hash"),
        },
        "ladder": ladder_groups(),
        "gate_policy": gate_policy(),
        "side_effect_policy": side_effect_policy(),
        "highest_allowed_claim_after_training": "offline candidate eligible for paper-readiness validation",
        "paper_readiness_claim_allowed": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Feature Recovery Ladder Plan",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Contract: `{payload['feature_contract']}`",
        f"- Transform: `{payload['model_facing_transform']}`",
        f"- Add-back policy: `{payload['feature_addback_policy']}`",
        f"- Masked baseline is control, not ceiling: `{payload['masked_baseline_is_control_not_ceiling']}`",
        "",
        "## Ladder",
        "",
    ]
    for group in payload["ladder"]:
        lines.append(
            f"{group['order']}. `{group['feature_group']}` - {group['status']}"
        )
    lines.extend(
        [
            "",
            "## Required Feature Add-Back Gates",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in payload["gate_policy"]["feature_addback_requires"])
    lines.extend(["", "## Forbidden Shortcuts", ""])
    lines.extend(f"- `{item}`" for item in payload["gate_policy"]["forbidden_shortcuts"])
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", payload)
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "summary": str(args.out_dir / "summary.json"),
                "report": str(args.out_dir / "report.md"),
                "blockers": payload["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
