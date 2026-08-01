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

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.live.protocol101_synchronization import Protocol101FairContractTrainingDesignV1


DEFAULT_PREFLIGHT = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_training_preflight/summary.json"
)
DEFAULT_MANIFEST = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_training_preflight/"
    "canonical_processed_session_manifest.json"
)
DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_PROTECTED_HOLDOUT = Path("v4/audit/autoresearch/protocol101_protected_holdout/summary.json")
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_training_design"
)
REQUIRED_EXPANDING_FOLD_COUNT = 5
REQUIRED_EMBARGO_SESSIONS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--protected-holdout", type=Path, default=DEFAULT_PROTECTED_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def session_month(session: str) -> str:
    return session[:7]


def protected_holdout_sessions(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        return []
    return sorted(str(session) for session in payload.get("sessions") or [] if str(session).strip())


def contiguous_chunks(sessions: list[str], chunk_count: int) -> list[list[str]]:
    if chunk_count <= 0:
        return []
    total = len(sessions)
    base = total // chunk_count
    remainder = total % chunk_count
    chunks: list[list[str]] = []
    cursor = 0
    for idx in range(chunk_count):
        size = base + (1 if idx < remainder else 0)
        chunks.append(sessions[cursor : cursor + size])
        cursor += size
    return chunks


def build_expanding_window_folds(
    sessions: list[str],
    *,
    fold_count: int = REQUIRED_EXPANDING_FOLD_COUNT,
    embargo_sessions: int = REQUIRED_EMBARGO_SESSIONS,
) -> list[dict[str, Any]]:
    """Build chronological expanding-window folds with a one-session embargo."""
    ordered = sorted(sessions)
    chunks = contiguous_chunks(ordered, fold_count + 1)
    if len(chunks) != fold_count + 1 or not chunks[0]:
        return []
    folds: list[dict[str, Any]] = []
    for fold_idx in range(fold_count):
        validation_sessions = list(chunks[fold_idx + 1])
        train_pool = [session for chunk in chunks[: fold_idx + 1] for session in chunk]
        if not validation_sessions or len(train_pool) <= embargo_sessions:
            return []
        embargoed = train_pool[-embargo_sessions:]
        train_sessions = train_pool[:-embargo_sessions]
        folds.append(
            {
                "fold_index": fold_idx + 1,
                "fold_id": f"expanding_fold_{fold_idx + 1:02d}",
                "train_sessions": train_sessions,
                "validation_sessions": validation_sessions,
                "embargoed_sessions": [
                    {
                        "session": session,
                        "boundary": f"train_to_fold_{fold_idx + 1}_validation",
                        "reason": "one_trading_session_embargo",
                    }
                    for session in embargoed
                ],
                "train_first_session": train_sessions[0] if train_sessions else None,
                "train_last_session": train_sessions[-1] if train_sessions else None,
                "validation_first_session": validation_sessions[0],
                "validation_last_session": validation_sessions[-1],
            }
        )
    return folds


def build_split_policy(
    sessions: list[str],
    *,
    protected_sessions: list[str] | None = None,
) -> dict[str, Any]:
    protected = set(protected_sessions or [])
    split_eligible_sessions = [session for session in sessions if session not in protected]
    mar = [session for session in split_eligible_sessions if session_month(session) == "2026-03"]
    train_sessions = [session for session in split_eligible_sessions if session < "2026-03-01"]
    raw_validation = mar[: max(len(mar) // 2, 1)]
    raw_diagnostic = mar[max(len(mar) // 2, 1) :]
    folds = build_expanding_window_folds(split_eligible_sessions)
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
        "schema_version": "Protocol101ChronologicalExpandingWindowSplitPolicyV2",
        "scope": "expanded_historical_development_only_not_final_holdout",
        "unit": "trading_session",
        "required_expanding_window_cv": True,
        "required_fold_count": REQUIRED_EXPANDING_FOLD_COUNT,
        "fold_count": len(folds),
        "folds": folds,
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "diagnostic_test_sessions": diagnostic_sessions,
        "embargoed_sessions": embargoed_sessions,
        "excluded_from_splits": [
            {
                "session": session,
                "reason": "protected_holdout_artifact",
            }
            for session in sorted(protected & set(sessions))
        ],
        "purge_unit": "full_session",
        "embargo_sessions_between_splits": REQUIRED_EMBARGO_SESSIONS,
        "expanding_window_embargo_sessions": REQUIRED_EMBARGO_SESSIONS,
        "same_timestamp_split_allowed": False,
        "protected_holdout_status": (
            "excluded_from_train_validation_diagnostic_splits"
            if protected
            else "not_defined_in_this_packet"
        ),
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
    protected_holdout: dict[str, Any] | None = None,
) -> Protocol101FairContractTrainingDesignV1:
    if preflight.get("status") != "ready_for_owner_authorized_fair_contract_training_design":
        status = "blocked_preflight_not_ready"
        decision = "do_not_design_training_until_preflight_passes"
    else:
        status = "owner_approval_required_before_training"
        decision = "fair_contract_training_design_ready_no_training_executed"
    included = manifest.get("included_sessions") or []
    sessions = [str(row.get("session")) for row in included]
    protected_sessions = protected_holdout_sessions(protected_holdout)
    excluded = [
        {
            "source": "recorder/parity captured sessions",
            "sessions": ["2026-06-30", "2026-07-01", "2026-07-02"],
            "reason": (
                "These sessions are synchronization/confirmation evidence and should not "
                "be used to train the first fair-contract replacement candidate."
            ),
        },
        {
            "source": "protected_holdout_artifact",
            "artifact_status": (protected_holdout or {}).get("status"),
            "protected_holdout_hash": (protected_holdout or {}).get("protected_holdout_hash"),
            "sessions": [session for session in protected_sessions if session in set(sessions)],
            "reason": (
                "Protected holdout sessions may be built and accepted as data-plane evidence, "
                "but are excluded from train, validation, and diagnostic model-selection "
                "splits unless a separate owner-approved holdout evaluation is requested."
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
        *[
            {
                "source": "acceptance_registry_filter",
                "session": row.get("session"),
                "file": row.get("processed_file"),
                "reason": row.get("reason"),
            }
            for row in manifest.get("excluded_acceptance_sessions") or []
        ],
    ]
    return Protocol101FairContractTrainingDesignV1(
        status=status,
        decision=decision,
        selected_feature_contract=str(
            preflight.get("selected_feature_contract")
            or FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
        ),
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
        split_policy=build_split_policy(sessions, protected_sessions=protected_sessions),
        training_target={
            "purpose": "train_research_candidate_under_certified_live_reproducible_fair_contract",
            "entry_model": "new_candidate_required_if_frozen_protocol101_cannot_pass_non_inferiority",
            "lifecycle": "keep frozen lifecycle initially unless separately preregistered",
            "labels": [
                "ask_to_bid_stop35_target60_hold10m",
                "ask_to_bid_stop50_target100_hold25m",
                "ask_to_bid_stop65_target150_hold45m",
                "ask_to_bid_stop50_target200_hold90m",
                "ask_to_bid_stop100_target300_hold120m",
                "ask_to_bid_stop100_target999_hold384m",
                "ask_to_bid_stop100_target9900_hold384m",
            ],
            "label_menu": "protocol101_trade_shape_menu_v2_all_seven_shapes",
            "primary_comparison": (
                "candidate_vs_paper_default_protocol101_under_strict_one_account_serial_replay"
            ),
        },
        success_gates=[
            {
                "gate": "causal_feature_contract",
                "required": (
                    "all model inputs from protocol101-live-v2-microstructure-masked "
                    "and no future/path labels in features"
                ),
            },
            {
                "gate": "chronological_validation",
                "required": (
                    "five chronological expanding-window folds with a one-session "
                    "embargo, no random timestamp split"
                ),
            },
            {
                "gate": "historical_non_inferiority_or_improvement",
                "required": (
                    "candidate must beat PAPER_DEFAULT_PROTOCOL101 under strict serial replay "
                    "after training under protocol101-live-v2-microstructure-masked and must "
                    "not rely on same-minute completed-bar timing"
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
            "approval_to_start_model_training_on_protocol101_live_v2_microstructure_masked_manifest",
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
        f"- Expanding CV required: `{str(split.get('required_expanding_window_cv')).lower()}`",
        f"- Expanding CV folds: `{split.get('fold_count')}` / `{split.get('required_fold_count')}`",
        f"- Warning: {split['warning']}",
        "",
        "## Expanding Folds",
        "",
    ]
    for fold in split.get("folds") or []:
        lines.append(
            f"- `{fold['fold_id']}`: train=`{len(fold['train_sessions'])}`, "
            f"validation=`{len(fold['validation_sessions'])}`, "
            f"embargoed=`{[row['session'] for row in fold.get('embargoed_sessions', [])]}`."
        )
    lines.extend([
        "",
        "## Success Gates",
        "",
    ])
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
    protected_holdout = load_json(args.protected_holdout)
    packet = build_packet(
        preflight,
        manifest,
        manifest_path=args.manifest,
        protected_holdout=protected_holdout,
    )
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
