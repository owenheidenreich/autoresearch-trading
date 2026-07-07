"""Create the guarded Protocol101 fair-contract training design packet.

This is a design artifact, not a training run. It requires the fair-contract
dataset preflight to pass, then writes the exact data manifest, split policy,
success gates, and forbidden actions for a future owner-approved retraining
attempt on the canonical live-reproducible game.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.protocol101_synchronization import Protocol101FairContractTrainingDesignV1


DEFAULT_PREFLIGHT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_preflight/summary.json"
)
DEFAULT_MANIFEST = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_preflight/"
    "canonical_processed_session_manifest.json"
)
DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_fair_contract_training_design")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def session_month(session: str) -> str:
    return session[:7]


def build_split_policy(sessions: list[str]) -> dict[str, Any]:
    mar = [session for session in sessions if session_month(session) == "2026-03"]
    train_sessions = [session for session in sessions if session < "2026-03-01"]
    raw_validation = mar[: max(len(mar) // 2, 1)]
    raw_diagnostic = mar[max(len(mar) // 2, 1) :]
    embargoed_sessions = []
    validation_sessions = list(raw_validation)
    diagnostic_sessions = list(raw_diagnostic)
    if validation_sessions:
        embargoed_sessions.append(
            {
                "session": validation_sessions.pop(0),
                "boundary": "train_to_validation",
                "reason": "one_trading_session_embargo",
            }
        )
    if diagnostic_sessions:
        embargoed_sessions.append(
            {
                "session": diagnostic_sessions.pop(0),
                "boundary": "validation_to_diagnostic_test",
                "reason": "one_trading_session_embargo",
            }
        )
    return {
        "schema_version": "Protocol101ChronologicalPurgedSplitPolicyV1",
        "scope": "expanded_historical_development_only_not_final_holdout",
        "unit": "trading_session",
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "diagnostic_test_sessions": diagnostic_sessions,
        "embargoed_sessions": embargoed_sessions,
        "purge_unit": "full_session",
        "embargo_sessions_between_splits": 1,
        "same_timestamp_split_allowed": False,
        "protected_holdout_status": "not_defined_in_this_packet",
        "warning": (
            "This expanded fair-contract training packet uses previously inspected "
            "development evidence, not an untouched final holdout. A promotion candidate "
            "must later use a newly reserved chronological holdout."
        ),
    }


def build_packet(
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> Protocol101FairContractTrainingDesignV1:
    if preflight.get("status") != "ready_for_owner_authorized_fair_contract_training_design":
        status = "blocked_preflight_not_ready"
        decision = "do_not_design_training_until_preflight_passes"
    else:
        status = "owner_approval_required_before_training"
        decision = "fair_contract_training_design_ready_no_training_executed"
    included = manifest.get("included_sessions") or []
    sessions = [str(row.get("session")) for row in included]
    excluded = [
        {
            "source": "recorder/parity captured sessions",
            "sessions": ["2026-06-30", "2026-07-01", "2026-07-02"],
            "reason": (
                "These sessions are synchronization/confirmation evidence and should not "
                "be used to train the first fair-contract replacement candidate."
            ),
        },
        *[
            {
                "source": "processed_duplicate",
                "file": row.get("file"),
                "reason": row.get("reason"),
            }
            for row in manifest.get("excluded_processed_files") or []
        ],
    ]
    return Protocol101FairContractTrainingDesignV1(
        status=status,
        decision=decision,
        selected_feature_contract=str(preflight.get("selected_feature_contract") or "protocol101-live-v1"),
        model_training_authorized=False,
        threshold_tuning_authorized=False,
        paper_submit_allowed=False,
        allowed_data={
            "canonical_manifest": str(manifest_path),
            "included_session_count": len(included),
            "included_first_session": sessions[0] if sessions else None,
            "included_last_session": sessions[-1] if sessions else None,
            "included_sessions": sessions,
            "require_manifest_loading": True,
            "glob_loading_allowed": False,
        },
        excluded_data=excluded,
        split_policy=build_split_policy(sessions),
        training_target={
            "purpose": "replace_or_challenge_frozen_protocol101_under_the_same_live_reproducible_game",
            "entry_model": "new_candidate_required_if_frozen_protocol101_cannot_pass_non_inferiority",
            "lifecycle": "keep frozen lifecycle initially unless separately preregistered",
            "labels": [
                "ask_to_bid_stop35_target60_hold10m",
                "ask_to_bid_stop50_target100_hold25m",
                "ask_to_bid_stop65_target150_hold45m",
            ],
            "primary_comparison": "same_runner_legacy_is_context_only_protocol101_live_v1_is_the_game_to_train_on",
        },
        success_gates=[
            {
                "gate": "causal_feature_contract",
                "required": "all model inputs from protocol101-live-v1 and no future/path labels in features",
            },
            {
                "gate": "chronological_validation",
                "required": "day-level chronological split with purge and embargo, no random timestamp split",
            },
            {
                "gate": "historical_non_inferiority_or_improvement",
                "required": (
                    "candidate must beat frozen protocol101-live-v1 under strict serial replay "
                    "and must not rely on same-minute completed-bar timing"
                ),
            },
            {
                "gate": "replay_reconstructability",
                "required": "every selected candidate, feature hash, score, action, lifecycle path, and block reason must be reconstructable",
            },
            {
                "gate": "shadow_confirmation",
                "required": "future recorder-first sessions must replay exactly before any paper-submit request",
            },
        ],
        forbidden_actions=[
            "Do not train without explicit owner approval.",
            "Do not tune thresholds against June/July parity days.",
            "Do not use glob-based dataset loading for this training scope.",
            "Do not restore same-minute completed-bar semantics.",
            "Do not change PAPER_TRADING_DEFAULT from this design packet.",
            "Do not enable paper-submit or broker execution from this design packet.",
        ],
        required_owner_approvals=[
            "approval_to_start_model_training_on_protocol101_live_v1_manifest",
            "approval_of_train_validation_diagnostic_test_split",
            "approval_of_primary_metric_and_promotion_gate_before_results",
            "separate_approval_for_any_paid_data_or_alternate_live_feed",
            "separate_approval_for_any_future_paper_submit",
        ],
    )


def render_report(packet: Protocol101FairContractTrainingDesignV1) -> str:
    split = packet.split_policy
    lines = [
        "# Protocol101 Fair Contract Training Design",
        "",
        "## Decision",
        "",
        f"- Status: `{packet.status}`",
        f"- Decision: `{packet.decision}`",
        f"- Selected feature contract: `{packet.selected_feature_contract}`",
        f"- Model training authorized: `{str(packet.model_training_authorized).lower()}`",
        f"- Threshold tuning authorized: `{str(packet.threshold_tuning_authorized).lower()}`",
        f"- Paper-submit allowed: `{str(packet.paper_submit_allowed).lower()}`",
        "",
        "## Allowed Data",
        "",
        f"- Canonical manifest: `{packet.allowed_data['canonical_manifest']}`",
        f"- Included sessions: `{packet.allowed_data['included_session_count']}`",
        f"- Range: `{packet.allowed_data['included_first_session']}` to `{packet.allowed_data['included_last_session']}`",
        f"- Manifest loading required: `{str(packet.allowed_data['require_manifest_loading']).lower()}`",
        f"- Glob loading allowed: `{str(packet.allowed_data['glob_loading_allowed']).lower()}`",
        "",
        "## Split Policy",
        "",
        f"- Scope: `{split['scope']}`",
        f"- Train sessions: `{len(split['train_sessions'])}`",
        f"- Validation sessions: `{len(split['validation_sessions'])}`",
        f"- Diagnostic test sessions: `{len(split['diagnostic_test_sessions'])}`",
        f"- Embargo sessions between splits: `{split['embargo_sessions_between_splits']}`",
        f"- Actual embargoed sessions: `{[row['session'] for row in split.get('embargoed_sessions', [])]}`",
        f"- Warning: {split['warning']}",
        "",
        "## Success Gates",
        "",
    ]
    lines.extend(f"- `{row['gate']}`: {row['required']}" for row in packet.success_gates)
    lines.extend(["", "## Forbidden Actions", ""])
    lines.extend(f"- {item}" for item in packet.forbidden_actions)
    lines.extend(["", "## Required Owner Approvals", ""])
    lines.extend(f"- `{item}`" for item in packet.required_owner_approvals)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight = load_json(args.preflight)
    manifest = load_json(args.manifest)
    packet = build_packet(preflight, manifest, manifest_path=args.manifest)
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(packet))
    print(
        json.dumps(
            {
                "status": packet.status,
                "decision": packet.decision,
                "model_training_authorized": packet.model_training_authorized,
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
