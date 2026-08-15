"""Run constrained offline model search for the Protocol101 v2 fair contract.

This script orchestrates owner-authorized offline training attempts only. It
does not contact brokers, download data, change paper defaults, promote models,
or use June/July recorder days for training. Each attempt is written to its own
artifact directory and registered in an experiment JSONL file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate import (
    STRICT_REPLAY_IMPLEMENTATION_VERSION,
)
from v4.scripts.run_protocol101_fair_contract_feature_jitter_gate import (
    IMPLEMENTATION_VERSION as FEATURE_JITTER_GATE_IMPLEMENTATION_VERSION,
)
from v4.model.supervised_pilot import (
    FEATURE_NOISE_AUGMENTATION_NONE,
    FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
    FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    SELECTION_MODE_STABLE_ABS_OFFSET_10,
    SELECTION_MODE_STABLE_ABS_OFFSET_15,
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
    SELECTION_MODE_TOP_SCORE,
)
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED


DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_model_search"
)
DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_design/summary.json"
)
EXPERIMENT_REGISTRY_ENTRY_VERSION = "Protocol101FairContractExperimentRegistryEntryV5"
REUSABLE_REGISTRY_ENTRY_VERSIONS = {
    EXPERIMENT_REGISTRY_ENTRY_VERSION,
    "Protocol101FairContractExperimentRegistryEntryV4",
    "Protocol101FairContractExperimentRegistryEntryV3",
    "Protocol101FairContractExperimentRegistryEntryV2",
}
MODEL_SEARCH_SUMMARY_VERSION = "Protocol101FairContractModelSearchSummaryV5"
MODEL_SEARCH_SELECTION_METRIC = "plain_fee_adjusted_net_pnl_after_hard_gates"
DEFAULT_ROUND_TRIP_FEE_DOLLARS = 3.0
FAILED_HARD_GATE_SCORE_FLOOR = -1_000_000_000_000.0


def current_implementation_versions() -> dict[str, str]:
    return {
        "selected_candidate_export": SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
        "strict_replay_gate": STRICT_REPLAY_IMPLEMENTATION_VERSION,
        "feature_jitter_gate": FEATURE_JITTER_GATE_IMPLEMENTATION_VERSION,
    }


@dataclass(frozen=True)
class AttemptConfig:
    attempt_id: str
    hypothesis: str
    policy_index: int = 1
    threshold_rule: str = "max_validation_stressed_pnl"
    fit_mode: str = "full_train"
    hidden_dim: int = 96
    epochs: int = 8
    max_train_examples: int = 350_000
    batch_size: int = 8192
    seed: int = 42
    threshold_stress_per_trade: float = 20.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_family: str = "mlp"
    target_mode: str = "regression"
    target_clip: float = 600.0
    positive_label_threshold: float = 20.0
    relative_target_weight: float = 0.5
    teacher_min_seed_count: int = 1
    ensemble_seeds: str = ""
    entry_filter: str = "none"
    min_score_margin: float = 0.0
    max_score_ceiling: float = 0.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    sample_weight_mode: str = "none"
    feature_transform: str = FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE
    feature_noise_augmentation: str = FEATURE_NOISE_AUGMENTATION_NONE
    selection_mode: str = SELECTION_MODE_TOP_SCORE
    run_feature_jitter_gate: bool = False


def default_attempts() -> list[AttemptConfig]:
    return [
        AttemptConfig(
            attempt_id=f"attempt_menuv2_policy{policy_index}_hgb_rop_s42",
            hypothesis=(
                "Bounded HGB menu-v2 baseline: score candidates on return-on-premium "
                f"for policy {policy_index} and select only by hard gates plus plain "
                "fee-adjusted strict-replay net PnL."
            ),
            policy_index=policy_index,
            threshold_rule="max_validation_stressed_pnl",
            fit_mode="train_tail20_calibration",
            model_family="sklearn_hist_gradient_boosting",
            target_mode="return_on_premium_regression",
            target_clip=5.0,
            learning_rate=0.05,
            weight_decay=0.01,
            max_trades_per_session=3,
            run_feature_jitter_gate=True,
        )
        for policy_index in range(7)
    ]

    # Legacy broad-search attempts were removed from the default menu. The
    # preregistered v2 objective allows only the seven bounded HGB ROP attempts
    # above; risk controls live in hard gates, not weighted utility coefficients.


def select_attempts(
    attempts: list[AttemptConfig],
    *,
    max_attempts: int,
    attempt_ids: str = "",
) -> list[AttemptConfig]:
    """Select attempts deterministically from the preregistered attempt list."""
    requested = [item.strip() for item in str(attempt_ids or "").split(",") if item.strip()]
    if not requested:
        return attempts[: max(int(max_attempts), 0)]
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    missing = [attempt_id for attempt_id in requested if attempt_id not in by_id]
    if missing:
        raise ValueError(f"unknown attempt id(s): {missing}")
    requested_set = set(requested)
    return [attempt for attempt in attempts if attempt.attempt_id in requested_set]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--max-attempts", type=int, default=7)
    parser.add_argument("--attempt-ids", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--append-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model-search orchestration without executing training attempts.",
    )
    parser.add_argument(
        "--owner-approved-model-training",
        action="store_true",
        help="Required for non-dry-run model-search execution.",
    )
    parser.add_argument(
        "--owner-approved-threshold-selection",
        action="store_true",
        help="Required for non-dry-run model-search execution.",
    )
    parser.add_argument(
        "--owner-approval-note",
        default="",
        help="Owner approval note required for non-dry-run model-search execution.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing registry rows by attempt id, keeping the last matching row."""
    if not path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            entries[attempt_id] = row
    return entries


def reusable_registry_entry(
    existing: dict[str, Any] | None,
    attempt: AttemptConfig,
) -> dict[str, Any] | None:
    """Return an existing registry entry only when config, versions, and artifacts match."""
    if not existing:
        return None
    if existing.get("schema_version") not in REUSABLE_REGISTRY_ENTRY_VERSIONS:
        return None
    if dict(existing.get("implementation_versions") or {}) != current_implementation_versions():
        return None
    expected = asdict(attempt)
    actual = dict(existing.get("config") or {})
    for key, value in expected.items():
        actual.setdefault(key, value)
    if actual != expected:
        return None
    artifacts = existing.get("artifacts") or {}
    if existing.get("fold_aware_training"):
        required = ("training_result", "model")
    else:
        required = (
            "training_result",
            "model",
            "selected_candidates",
            "strict_replay_trades",
            "candidate_validation_report",
            "strict_replay_report",
        )
    if any(not artifacts.get(key) for key in required):
        return None
    if any(not Path(str(artifacts[key])).exists() for key in required):
        return None
    refreshed = dict(existing)
    refreshed["schema_version"] = EXPERIMENT_REGISTRY_ENTRY_VERSION
    refreshed["implementation_versions"] = current_implementation_versions()
    feature_jitter_gate = refreshed.get("feature_jitter_gate") or None
    if isinstance(feature_jitter_gate, dict) and feature_jitter_gate.get("status") == "not_run":
        feature_jitter_gate = None
    refreshed["objective_score"] = attempt_score(
        refreshed.get("strict_replay_gate") or {},
        feature_jitter_gate,
        validation_gate=refreshed.get("candidate_validation_gate") or None,
    )
    refreshed["reason_accepted_or_rejected"] = reason_for_result(
        validation_gate=refreshed.get("candidate_validation_gate") or {},
        replay_summary=refreshed.get("strict_replay_gate") or {},
    )
    return refreshed


def run_command(args: list[str], *, cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def metric(payload: dict[str, Any], split: str, key: str) -> float:
    try:
        return float(((payload.get("metrics") or {}).get(split) or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def fee_adjusted_net_pnl(
    replay_summary: dict[str, Any],
    *,
    round_trip_fee_dollars: float = DEFAULT_ROUND_TRIP_FEE_DOLLARS,
) -> float:
    """Plain replay PnL after the preregistered fixed fee overlay."""
    total = 0.0
    for split in ("validation", "diagnostic_test"):
        total_pnl = metric(replay_summary, split, "total_pnl")
        trades = metric(replay_summary, split, "trades")
        total += total_pnl - float(round_trip_fee_dollars) * trades
    return float(total)


def hard_gate_passed(
    replay_summary: dict[str, Any],
    feature_jitter_summary: dict[str, Any] | None = None,
    *,
    validation_gate: dict[str, Any] | None = None,
) -> bool:
    if validation_gate is not None and validation_gate.get("status") != "pass":
        return False
    if replay_summary.get("status") != "pass":
        return False
    if feature_jitter_summary is not None and feature_jitter_summary.get("status") != "pass":
        return False
    return True


def attempt_score(
    replay_summary: dict[str, Any],
    feature_jitter_summary: dict[str, Any] | None = None,
    *,
    validation_gate: dict[str, Any] | None = None,
) -> float:
    """Rank accepted candidates by plain fee-adjusted replay PnL.

    Risk adjustment is intentionally not a weighted utility here. Drawdown,
    ruin, concentration, churn, frequency, and jitter stability belong in hard
    gates. This scalar is used only after those gates pass.
    """
    score = fee_adjusted_net_pnl(replay_summary)
    if hard_gate_passed(
        replay_summary,
        feature_jitter_summary,
        validation_gate=validation_gate,
    ):
        return score
    return FAILED_HARD_GATE_SCORE_FLOOR + score


def reason_for_result(
    *,
    validation_gate: dict[str, Any],
    replay_summary: dict[str, Any],
    feature_jitter_summary: dict[str, Any] | None = None,
) -> str:
    if (
        validation_gate.get("status") == "pass"
        and replay_summary.get("status") == "pass"
        and (not feature_jitter_summary or feature_jitter_summary.get("status") == "pass")
    ):
        return "accepted_for_next_shadow_replay_gate"
    jitter_blockers = (
        list(feature_jitter_summary.get("blockers") or [])
        if feature_jitter_summary
        else []
    )
    blockers = sorted(
        set(
            (validation_gate.get("blockers") or [])
            + (replay_summary.get("blockers") or [])
            + jitter_blockers
        )
    )
    if blockers:
        return "rejected:" + ",".join(str(item) for item in blockers)
    if feature_jitter_summary and feature_jitter_summary.get("status") != "pass":
        return f"rejected:feature_jitter_gate={feature_jitter_summary.get('status')}"
    return f"rejected:validation_gate={validation_gate.get('status')};strict_replay={replay_summary.get('status')}"


def fold_aware_registry_entry(
    *,
    attempt: AttemptConfig,
    design_payload: dict[str, Any],
    training: dict[str, Any],
    training_result: Path,
    train_dir: Path,
) -> dict[str, Any]:
    allowed_data = design_payload.get("allowed_data") or {}
    data_scope = (
        f"{allowed_data.get('included_session_count', 'unknown')}_sessions_"
        f"{allowed_data.get('included_first_session', 'unknown')}_to_"
        f"{allowed_data.get('included_last_session', 'unknown')}_5fold_expanding_cv"
    )
    aggregate_validation = (
        (training.get("fold_summary") or {}).get("aggregate_validation_metrics") or {}
    )
    validation_gate = {
        "status": "fail",
        "decision": "fold_cv_complete_downstream_strict_replay_not_yet_run",
        "blockers": ["fold_aware_selected_export_and_strict_replay_pending"],
        "metrics": {"validation": aggregate_validation},
    }
    replay = {
        "status": "fail",
        "decision": "fold_aware_cv_does_not_replace_selected_candidate_strict_replay",
        "blockers": ["fold_aware_selected_export_and_strict_replay_pending"],
        "metrics": {
            "validation": aggregate_validation,
            "diagnostic_test": {
                "trades": 0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            },
        },
    }
    return {
        "schema_version": EXPERIMENT_REGISTRY_ENTRY_VERSION,
        "implementation_versions": current_implementation_versions(),
        "attempt_id": attempt.attempt_id,
        "hypothesis": attempt.hypothesis,
        "config": asdict(attempt),
        "feature_contract": str(
            design_payload.get("selected_feature_contract")
            or FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
        ),
        "data_scope": data_scope,
        "forbidden_data_excluded": ["2026-06-30", "2026-07-01", "2026-07-02"],
        "model_training_executed": True,
        "threshold_selection_executed": True,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "chosen_threshold": training.get("chosen_threshold"),
        "training_metrics": training.get("neural") or {},
        "fold_aware_training": {
            "execution_mode": training.get("execution_mode"),
            "fold_count": training.get("fold_count"),
            "fold_summary": training.get("fold_summary") or {},
        },
        "candidate_validation_gate": {
            "status": validation_gate["status"],
            "decision": validation_gate["decision"],
            "blockers": validation_gate["blockers"],
        },
        "strict_replay_gate": {
            "status": replay["status"],
            "decision": replay["decision"],
            "blockers": replay["blockers"],
            "metrics": replay["metrics"],
        },
        "feature_jitter_gate": {
            "status": "not_run",
            "decision": "feature_jitter_gate_waiting_for_fold_aware_selected_export",
            "blockers": ["fold_aware_selected_export_and_strict_replay_pending"],
            "requirements": {},
        },
        "selection_metric": MODEL_SEARCH_SELECTION_METRIC,
        "round_trip_fee_dollars": DEFAULT_ROUND_TRIP_FEE_DOLLARS,
        "fee_adjusted_net_pnl": fee_adjusted_net_pnl(replay),
        "objective_score": attempt_score(replay, validation_gate=validation_gate),
        "reason_accepted_or_rejected": reason_for_result(
            validation_gate=validation_gate,
            replay_summary=replay,
        ),
        "artifacts": {
            "training_result": str(training_result),
            "model": str(train_dir / "model.pt"),
            "selected_candidates": "",
            "strict_replay_trades": "",
            "candidate_validation_report": "",
            "strict_replay_report": "",
            "feature_jitter_report": "",
        },
    }


def run_attempt(
    *,
    attempt: AttemptConfig,
    root: Path,
    design: Path,
    cwd: Path,
    force: bool,
    owner_approval_note: str = (
        "Active goal authorizes offline constrained Protocol101 "
        "protocol101-live-v2-microstructure-masked model search; no broker, "
        "paid data, paper-submit, promotion, or default change."
    ),
) -> dict[str, Any]:
    attempt_dir = root / "attempts" / attempt.attempt_id
    if force and attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    train_dir = attempt_dir / "training_runner"
    gate_dir = attempt_dir / "candidate_validation_gate"
    export_dir = attempt_dir / "selected_candidate_export"
    replay_dir = attempt_dir / "selected_candidate_replay_gate"
    jitter_dir = attempt_dir / "feature_jitter_gate"
    training_result = train_dir / "training_result.json"
    if not training_result.exists():
        train_command = [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_training_runner",
            "--mode",
            "train",
            "--design",
            str(design),
            "--out-dir",
            str(train_dir),
            "--model-out",
            str(train_dir / "model.pt"),
            "--policy-index",
            str(attempt.policy_index),
            "--fit-mode",
            str(attempt.fit_mode),
            "--threshold-rule",
            attempt.threshold_rule,
            "--threshold-stress-per-trade",
            str(attempt.threshold_stress_per_trade),
            "--hidden-dim",
            str(attempt.hidden_dim),
            "--model-family",
            str(attempt.model_family),
            "--learning-rate",
            str(attempt.learning_rate),
            "--weight-decay",
            str(attempt.weight_decay),
            "--target-mode",
            str(attempt.target_mode),
            "--target-clip",
            str(attempt.target_clip),
            "--positive-label-threshold",
            str(attempt.positive_label_threshold),
            "--relative-target-weight",
            str(attempt.relative_target_weight),
            "--teacher-min-seed-count",
            str(attempt.teacher_min_seed_count),
            "--entry-filter",
            str(attempt.entry_filter),
            "--min-score-margin",
            str(attempt.min_score_margin),
            "--max-score-ceiling",
            str(attempt.max_score_ceiling),
            "--max-trades-per-session",
            str(attempt.max_trades_per_session),
            "--max-daily-loss",
            str(attempt.max_daily_loss),
            "--sample-weight-mode",
            str(attempt.sample_weight_mode),
            "--feature-transform",
            str(attempt.feature_transform),
            "--feature-noise-augmentation",
            str(attempt.feature_noise_augmentation),
            "--selection-mode",
            str(attempt.selection_mode),
            "--epochs",
            str(attempt.epochs),
            "--max-train-examples",
            str(attempt.max_train_examples),
            "--batch-size",
            str(attempt.batch_size),
            "--seed",
            str(attempt.seed),
        ]
        if attempt.ensemble_seeds:
            train_command.extend(["--ensemble-seeds", str(attempt.ensemble_seeds)])
        train_command.extend(
            [
                "--owner-approved-model-training",
                "--owner-approved-threshold-selection",
                "--owner-approval-note",
                str(owner_approval_note),
            ]
        )
        run_command(train_command, cwd=cwd)
    training = load_json(training_result)
    design_payload = load_json(design)
    if training.get("execution_mode") == "fold_aware_expanding_cv":
        return fold_aware_registry_entry(
            attempt=attempt,
            design_payload=design_payload,
            training=training,
            training_result=training_result,
            train_dir=train_dir,
        )
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_candidate_validation_gate",
            "--runner-plan",
            str(train_dir / "runner_plan.json"),
            "--training-result",
            str(training_result),
            "--out-dir",
            str(gate_dir),
        ],
        cwd=cwd,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_selected_candidate_export",
            "--runner-plan",
            str(train_dir / "runner_plan.json"),
            "--training-result",
            str(training_result),
            "--out-dir",
            str(export_dir),
        ],
        cwd=cwd,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate",
            "--selected-export",
            str(export_dir / "summary.json"),
            "--out-dir",
            str(replay_dir),
            "--max-trades-per-session",
            str(attempt.max_trades_per_session),
            "--max-daily-loss",
            str(attempt.max_daily_loss),
        ],
        cwd=cwd,
    )
    if attempt.run_feature_jitter_gate:
        run_command(
            [
                sys.executable,
                "-m",
                "v4.scripts.run_protocol101_fair_contract_feature_jitter_gate",
                "--runner-plan",
                str(train_dir / "runner_plan.json"),
                "--training-result",
                str(training_result),
                "--out-dir",
                str(jitter_dir),
            ],
            cwd=cwd,
        )
    allowed_data = design_payload.get("allowed_data") or {}
    data_scope = (
        f"{allowed_data.get('included_session_count', 'unknown')}_sessions_"
        f"{allowed_data.get('included_first_session', 'unknown')}_to_"
        f"{allowed_data.get('included_last_session', 'unknown')}_manifest_only"
    )
    validation_gate = load_json(gate_dir / "summary.json")
    selected_export = load_json(export_dir / "summary.json")
    replay = load_json(replay_dir / "summary.json")
    feature_jitter_gate = (
        load_json(jitter_dir / "summary.json")
        if attempt.run_feature_jitter_gate
        else None
    )
    entry = {
        "schema_version": EXPERIMENT_REGISTRY_ENTRY_VERSION,
        "implementation_versions": {
            "selected_candidate_export": str(
                selected_export.get("implementation_version")
                or SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION
            ),
            "strict_replay_gate": str(
                replay.get("implementation_version") or STRICT_REPLAY_IMPLEMENTATION_VERSION
            ),
        },
        "attempt_id": attempt.attempt_id,
        "hypothesis": attempt.hypothesis,
        "config": asdict(attempt),
        "feature_contract": str(
            design_payload.get("selected_feature_contract")
            or FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
        ),
        "data_scope": data_scope,
        "forbidden_data_excluded": ["2026-06-30", "2026-07-01", "2026-07-02"],
        "model_training_executed": True,
        "threshold_selection_executed": True,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "chosen_threshold": training.get("chosen_threshold"),
        "training_metrics": (training.get("neural") or {}),
        "candidate_validation_gate": {
            "status": validation_gate.get("status"),
            "decision": validation_gate.get("decision"),
            "blockers": validation_gate.get("blockers") or [],
        },
        "strict_replay_gate": {
            "status": replay.get("status"),
            "decision": replay.get("decision"),
            "blockers": replay.get("blockers") or [],
            "metrics": replay.get("metrics") or {},
        },
        "feature_jitter_gate": (
            {
                "status": feature_jitter_gate.get("status"),
                "decision": feature_jitter_gate.get("decision"),
                "blockers": feature_jitter_gate.get("blockers") or [],
                "requirements": feature_jitter_gate.get("requirements") or {},
            }
            if feature_jitter_gate
            else {
                "status": "not_run",
                "decision": "feature_jitter_gate_not_requested_for_this_attempt",
                "blockers": [],
                "requirements": {},
            }
        ),
        "selection_metric": MODEL_SEARCH_SELECTION_METRIC,
        "round_trip_fee_dollars": DEFAULT_ROUND_TRIP_FEE_DOLLARS,
        "fee_adjusted_net_pnl": fee_adjusted_net_pnl(replay),
        "objective_score": attempt_score(
            replay,
            feature_jitter_gate,
            validation_gate=validation_gate,
        ),
        "reason_accepted_or_rejected": reason_for_result(
            validation_gate=validation_gate,
            replay_summary=replay,
            feature_jitter_summary=feature_jitter_gate,
        ),
        "artifacts": {
            "training_result": str(training_result),
            "model": str(train_dir / "model.pt"),
            "selected_candidates": str(export_dir / "selected_candidates.csv"),
            "strict_replay_trades": str(replay_dir / "strict_replay_trades.csv"),
            "candidate_validation_report": str(gate_dir / "report.md"),
            "strict_replay_report": str(replay_dir / "report.md"),
            "feature_jitter_report": str(jitter_dir / "report.md") if feature_jitter_gate else "",
        },
    }
    return entry


def render_report(entries: list[dict[str, Any]]) -> str:
    ranked = sorted(entries, key=lambda item: float(item.get("objective_score") or 0.0), reverse=True)
    lines = [
        "# Protocol101 Fair-Contract Model Search",
        "",
        "## Decision",
        "",
    ]
    best = ranked[0] if ranked else None
    if best:
        lines.extend(
            [
                f"- Best attempt: `{best['attempt_id']}`",
                f"- Selection metric: `{MODEL_SEARCH_SELECTION_METRIC}`",
                f"- Best fee-adjusted net PnL: `{best['objective_score']:.2f}`",
                f"- Reason: `{best['reason_accepted_or_rejected']}`",
                f"- Strict replay status: `{best['strict_replay_gate']['status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Attempt Summary",
            "",
            "| Attempt | Family | Fit mode | Weighting | Feature transform | Noise aug | Jitter gate | Policy | Entry filter | Selection | Margin | Score ceiling | Session cap | Daily loss | Threshold rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Status |",
            "|---|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for entry in ranked:
        metrics = entry["strict_replay_gate"].get("metrics") or {}
        val = metrics.get("validation") or {}
        diag = metrics.get("diagnostic_test") or {}
        cfg = entry["config"]
        jitter = (entry.get("feature_jitter_gate") or {}).get("status", "not_run")
        lines.append(
            f"| `{entry['attempt_id']}` | {cfg.get('model_family', 'mlp')} | {cfg.get('fit_mode', 'full_train')} | "
            f"{cfg.get('sample_weight_mode', 'none')} | {cfg.get('feature_transform', 'none')} | "
            f"{cfg.get('feature_noise_augmentation', 'none')} | "
            f"{jitter} | "
            f"{cfg['policy_index']} | {cfg.get('entry_filter', 'none')} | "
            f"{cfg.get('selection_mode', 'top_score')} | "
            f"{float(cfg.get('min_score_margin') or 0.0):.2f} | "
            f"{float(cfg.get('max_score_ceiling') or 0.0):.2f} | "
            f"{int(cfg.get('max_trades_per_session') or 0)} | {float(cfg.get('max_daily_loss') or 0.0):.0f} | "
            f"{cfg['threshold_rule']} | "
            f"{float(val.get('total_pnl') or 0.0):.2f} | {float(diag.get('total_pnl') or 0.0):.2f} | "
            f"{float(val.get('profit_factor') or 0.0):.3f} | {float(diag.get('profit_factor') or 0.0):.3f} | "
            f"`{entry['reason_accepted_or_rejected']}` |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- Selection metric is `{MODEL_SEARCH_SELECTION_METRIC}` with a frozen `${DEFAULT_ROUND_TRIP_FEE_DOLLARS:.2f}` round-trip fee overlay.",
            "- Risk controls are hard binary gates, not fitted penalty weights in the selection score.",
            "- June/July IBKR recorder days were excluded from training and threshold selection.",
            "- No broker endpoints, paper-submit, paid downloads, default changes, or promotions are performed by this search.",
            "- Registry reuse requires current selected-export and strict-replay implementation versions.",
            "- Failed attempts are still registry entries, because negative evidence is part of hill climbing safely.",
        ]
    )
    return "\n".join(lines) + "\n"


def preflight_blockers(
    *,
    args: argparse.Namespace,
    attempts: list[AttemptConfig],
    design_payload: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if design_payload.get("selected_feature_contract") != FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED:
        blockers.append("unexpected_feature_contract")
    if not attempts:
        blockers.append("no_attempts_selected")
    for attempt in attempts:
        if attempt.model_family not in {
            "sklearn_hist_gradient_boosting",
            "sklearn_hist_gradient_boosting_by_right",
        }:
            blockers.append(f"{attempt.attempt_id}:fold_aware_training_requires_hgb_tabular_model_family")
        if attempt.feature_transform != FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE:
            blockers.append(f"{attempt.attempt_id}:unexpected_model_scoring_feature_transform")
    if not args.dry_run:
        if not args.owner_approved_model_training:
            blockers.append("missing_owner_approved_model_training_flag")
        if not args.owner_approved_threshold_selection:
            blockers.append("missing_owner_approved_threshold_selection_flag")
        if not str(args.owner_approval_note).strip():
            blockers.append("missing_owner_approval_note")
    return sorted(set(blockers))


def dry_run_payload(
    *,
    args: argparse.Namespace,
    attempts: list[AttemptConfig],
    design_payload: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SEARCH_SUMMARY_VERSION,
        "status": "dry_run_ready" if not blockers else "blocked",
        "decision": (
            "fold_aware_model_search_ready_no_training_executed"
            if not blockers
            else "repair_model_search_inputs_before_training"
        ),
        "blockers": blockers,
        "attempts": len(attempts),
        "selected_attempts": [asdict(attempt) for attempt in attempts],
        "feature_contract": design_payload.get("selected_feature_contract"),
        "selection_metric": MODEL_SEARCH_SELECTION_METRIC,
        "round_trip_fee_dollars": DEFAULT_ROUND_TRIP_FEE_DOLLARS,
        "implementation_versions": current_implementation_versions(),
        "append_only": bool(args.append_only),
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }


def render_dry_run_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Model Search Dry-Run",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Attempts: `{payload['attempts']}`",
        f"- Feature contract: `{payload['feature_contract']}`",
        f"- Selection metric: `{payload['selection_metric']}`",
        f"- Model training executed: `{str(payload['model_training_executed']).lower()}`",
        f"- Threshold selection executed: `{str(payload['threshold_selection_executed']).lower()}`",
        f"- Broker endpoint called: `{str(payload['broker_endpoint_called']).lower()}`",
        f"- Paper-submit allowed: `{str(payload['paper_submit_allowed']).lower()}`",
        "",
        "## Attempts",
        "",
    ]
    for attempt in payload["selected_attempts"]:
        lines.append(
            f"- `{attempt['attempt_id']}`: policy=`{attempt['policy_index']}`, "
            f"family=`{attempt['model_family']}`, fit=`{attempt['fit_mode']}`, "
            f"transform=`{attempt['feature_transform']}`."
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in payload["blockers"]) if payload["blockers"] else lines.append("- None.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempts = select_attempts(
        default_attempts(),
        max_attempts=int(args.max_attempts),
        attempt_ids=str(args.attempt_ids),
    )
    design_payload = load_json(args.design)
    blockers = preflight_blockers(
        args=args,
        attempts=attempts,
        design_payload=design_payload,
    )
    if args.dry_run or blockers:
        summary = dry_run_payload(
            args=args,
            attempts=attempts,
            design_payload=design_payload,
            blockers=blockers,
        )
        args.out_dir.joinpath("summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        args.out_dir.joinpath("report.md").write_text(render_dry_run_report(summary))
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "decision": summary["decision"],
                    "attempts": summary["attempts"],
                    "blockers": summary["blockers"],
                    "model_training_executed": summary["model_training_executed"],
                    "report": str(args.out_dir / "report.md"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if args.dry_run and not blockers else 2
    entries: list[dict[str, Any]] = []
    registry = args.out_dir / "experiment_registry.jsonl"
    if args.force and registry.exists():
        registry.unlink()
    existing_by_id = load_registry(registry) if args.append_only and not args.force else {}
    reused_attempts: list[str] = []
    executed_attempts: list[str] = []
    for attempt in attempts:
        existing_entry = existing_by_id.get(attempt.attempt_id)
        entry = reusable_registry_entry(existing_entry, attempt)
        if entry is not None:
            reused_attempts.append(attempt.attempt_id)
        else:
            changed_existing_config = False
            if existing_entry:
                expected = asdict(attempt)
                actual = dict(existing_entry.get("config") or {})
                for key, value in expected.items():
                    actual.setdefault(key, value)
                changed_existing_config = actual != expected
            entry = run_attempt(
                attempt=attempt,
                root=args.out_dir,
                design=args.design,
                cwd=cwd,
                force=bool(args.force or changed_existing_config),
                owner_approval_note=str(args.owner_approval_note),
            )
            executed_attempts.append(attempt.attempt_id)
        entries.append(entry)
    registry.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    )
    ranked = sorted(entries, key=lambda item: float(item.get("objective_score") or 0.0), reverse=True)
    summary = {
        "schema_version": MODEL_SEARCH_SUMMARY_VERSION,
        "status": "pass" if ranked and ranked[0]["reason_accepted_or_rejected"].startswith("accepted") else "fail",
        "attempts": len(entries),
        "best_attempt": ranked[0] if ranked else None,
        "selection_metric": MODEL_SEARCH_SELECTION_METRIC,
        "round_trip_fee_dollars": DEFAULT_ROUND_TRIP_FEE_DOLLARS,
        "registry": str(registry),
        "implementation_versions": current_implementation_versions(),
        "append_only": bool(args.append_only),
        "reused_attempts": reused_attempts,
        "executed_attempts": executed_attempts,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(entries))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "attempts": len(entries),
                "best_attempt": ranked[0]["attempt_id"] if ranked else None,
                "reused_attempts": len(reused_attempts),
                "executed_attempts": len(executed_attempts),
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
