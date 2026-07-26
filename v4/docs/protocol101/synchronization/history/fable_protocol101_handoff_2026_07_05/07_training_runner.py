"""Guarded runner for a future Protocol101 fair-contract training attempt.

This runner is deliberately train-locked by default. Its dry-run mode verifies
that a future owner-approved run would consume only the preregistered manifest
and chronological split from the fair-contract design packet. Actual training
requires explicit acknowledgement flags and should not be launched from a
policy/design packet alone.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import (
    CLASSIFIER_TARGET_MODES,
    FEATURE_NOISE_AUGMENTATION_CHOICES,
    LISTWISE_TARGET_MODES,
    SELECTION_MODE_CHOICES,
    SELECTION_MODE_TOP_SCORE,
    TEACHER_TARGET_MODES,
    Trade,
    VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS,
    FEATURE_TRANSFORM_CHOICES,
    PilotConfig,
    augment_decision_features_with_noise,
    jitter_decision_features,
    choose_threshold,
    collect_examples,
    collect_training_examples,
    load_decisions,
    metrics_for_trades,
    model_state_dict,
    predict_decisions,
    simulate_baseline,
    simulate_model_policy,
    summarize_random_baseline,
    top_prediction,
    train_model,
    transform_decision_features,
    write_json,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_labels_design/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner"
)
DEFAULT_MODEL_OUT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/model.pt"
)
DEFAULT_TEACHER_EVENTS = Path(
    "v4/audit/autoresearch/unified_protocol101_baseline_attachment/"
    "protocol101_baseline_event_actions_training_scope.parquet"
)
POLICY_META = {
    0: ("ask_to_bid_stop35_target60_hold10m", 10),
    1: ("ask_to_bid_stop50_target100_hold25m", 25),
    2: ("ask_to_bid_stop65_target150_hold45m", 45),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--mode", choices=("dry-run", "train"), default="dry-run")
    parser.add_argument("--policy-index", type=int, choices=sorted(POLICY_META), default=1)
    parser.add_argument(
        "--fit-mode",
        choices=("full_train", "jan_fit_feb_calibration", "train_tail20_calibration"),
        default="full_train",
        help=(
            "Training/calibration chronology. jan_fit_feb_calibration fits on "
            "January train sessions and chooses thresholds on February train sessions, "
            "leaving March validation/diagnostic as forward checks. "
            "train_tail20_calibration fits on earlier train sessions and chooses "
            "thresholds on the last 20 chronological train sessions, which is useful "
            "when the train split includes prehistory before January/February."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=350_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument(
        "--model-family",
        choices=(
            "mlp",
            "sklearn_hist_gradient_boosting",
            "sklearn_hist_gradient_boosting_by_right",
        ),
        default="mlp",
        help="Offline model family. Sklearn mode is tabular-only and still uses the same fair-contract rows.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--target-mode",
        choices=(
            "regression",
            "profit_classifier",
            "decision_profit_presence_classifier",
            "decision_best_profit_regression",
            "decision_top_profit_classifier",
            "decision_top_profit_regression",
            "decision_top_profit_listwise",
            "decision_relative_regression",
            "blended_relative_regression",
            "protocol101_teacher_classifier",
            "protocol101_teacher_profitable_classifier",
            "protocol101_teacher_edge_regression",
        ),
        default="regression",
    )
    parser.add_argument(
        "--teacher-events",
        type=Path,
        default=DEFAULT_TEACHER_EVENTS,
        help=(
            "Protocol101 baseline event-action attachment used only for teacher-target "
            "training modes. Never a runtime feature."
        ),
    )
    parser.add_argument(
        "--teacher-min-seed-count",
        type=int,
        default=1,
        help="Minimum baseline seed count required to mark a fair candidate teacher-positive.",
    )
    parser.add_argument("--target-clip", type=float, default=600.0)
    parser.add_argument(
        "--positive-label-threshold",
        type=float,
        default=20.0,
        help="For profit_classifier, label candidates positive only above this dollar PnL threshold.",
    )
    parser.add_argument(
        "--relative-target-weight",
        type=float,
        default=0.5,
        help="For blended_relative_regression, weight on within-decision relative advantage.",
    )
    parser.add_argument(
        "--entry-filter",
        choices=(
            "none",
            "vwap_aligned",
            "premium_floor_3",
            "near_10_20_offset",
            "put_only",
            "put_near_10_20_offset",
            "put_near_after_0940",
            "put_near_after_0940_vwap_m2_10",
            "put_near_after_0940_vwap_m2_10_omar_neg",
            "put_near_after_0940_vwap_m2_10_range_20_45",
            "put_near_after_0940_vwap_m2_10_near_vwap",
            "put_near_after_0940_vwap_m2_10_premium_gte_7_5",
            "put_near_after_0940_vwap_m2_10_mom15_nonpos",
            "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
            "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos",
            "near_after_0940_vwap_m2_10_mom15_side",
            "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
            "morning_1000_1129",
            "morning_near_10_20_offset",
            "above_vwap_omar_pos_after_open",
        ),
        default="none",
        help="Optional live-causal first-stage candidate filter applied during threshold selection and replay.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=tuple(sorted(SELECTION_MODE_CHOICES)),
        default=SELECTION_MODE_TOP_SCORE,
        help="How to choose the exact contract among eligible candidates after scoring.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.0,
        help="Require top eligible score to exceed the runner-up eligible score by this amount before entering.",
    )
    parser.add_argument(
        "--max-score-ceiling",
        type=float,
        default=0.0,
        help=(
            "Optional live-causal overconfidence abstention guard. If positive, "
            "wait when the top eligible score is greater than or equal to this value."
        ),
    )
    parser.add_argument(
        "--max-trades-per-session",
        type=int,
        default=0,
        help="Optional live-reproducible session trade cap. Zero disables the cap.",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.0,
        help="Optional realized daily loss stop in dollars. Zero disables the stop.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=("none", "balanced_classifier", "decision_balanced_classifier"),
        default="none",
        help=(
            "Optional training sample weighting. Classifier weighting modes are "
            "allowed only for classifier targets and counteract rare-positive labels "
            "without changing runtime features. decision_balanced_classifier first "
            "gives each decision minute equal total weight, then balances positive "
            "and negative classes."
        ),
    )
    parser.add_argument(
        "--feature-transform",
        choices=tuple(sorted(FEATURE_TRANSFORM_CHOICES)),
        default="none",
        help=(
            "Named model-facing feature transform. This never changes labels, raw data, "
            "candidate masks, fills, or broker behavior; it only controls which fair-contract "
            "features a candidate model is allowed to score from."
        ),
    )
    parser.add_argument(
        "--feature-noise-augmentation",
        choices=tuple(sorted(FEATURE_NOISE_AUGMENTATION_CHOICES)),
        default="none",
        help=(
            "Fit-only deterministic feature-noise augmentation. Calibration, validation, "
            "diagnostic, and IBKR confirmation rows remain unaugmented."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-seeds",
        default="",
        help=(
            "Optional comma-separated seeds for an averaged ensemble. "
            "Empty keeps the single-model artifact format."
        ),
    )
    parser.add_argument(
        "--threshold-rule",
        choices=(
            "max_validation_pnl",
            "max_validation_stressed_pnl",
            "risk_adjusted_stressed",
            "frequency_sufficient_stressed",
            "daily_stability_stressed",
            "drawdown_guarded_stressed",
            "conservative_pnl_plateau_stressed",
            "jitter_stability_stressed",
        ),
        default="max_validation_pnl",
        help="Validation-only rule for choosing the entry threshold.",
    )
    parser.add_argument(
        "--threshold-stress-per-trade",
        type=float,
        default=20.0,
        help="Dollar stress subtracted from each validation trade for stressed threshold rules.",
    )
    parser.add_argument(
        "--owner-approved-model-training",
        action="store_true",
        help="Required for --mode train. Dry-run ignores this flag.",
    )
    parser.add_argument(
        "--owner-approved-threshold-selection",
        action="store_true",
        help="Required for --mode train because validation threshold selection is part of the run.",
    )
    parser.add_argument(
        "--owner-approval-note",
        default="",
        help="Short owner approval note or ticket id required for --mode train.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _manifest_by_session(manifest: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in manifest.get("included_sessions") or []:
        session = str(item.get("session") or "")
        processed_file = item.get("processed_file")
        if session and processed_file:
            out[session] = Path(str(processed_file))
    return out


def paths_by_split(design: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, list[Path]], list[str]]:
    """Resolve manifest-backed split paths without any glob fallback."""
    split_policy = design.get("split_policy") or {}
    by_session = _manifest_by_session(manifest)
    paths = {"train": [], "validation": [], "diagnostic_test": []}
    blockers: list[str] = []
    for split_name, key in (
        ("train", "train_sessions"),
        ("validation", "validation_sessions"),
        ("diagnostic_test", "diagnostic_test_sessions"),
    ):
        for session in split_policy.get(key) or []:
            path = by_session.get(str(session))
            if path is None:
                blockers.append(f"manifest_missing_session:{split_name}:{session}")
                continue
            paths[split_name].append(path)
    for split_name, split_paths in paths.items():
        if not split_paths:
            blockers.append(f"empty_split:{split_name}")
    return paths, blockers


def validate_plan(
    *,
    design: dict[str, Any],
    mode: str,
    owner_approved_model_training: bool,
    owner_approved_threshold_selection: bool,
    owner_approval_note: str,
) -> list[str]:
    blockers: list[str] = []
    allowed = design.get("allowed_data") or {}
    if allowed.get("require_manifest_loading") is not True:
        blockers.append("manifest_loading_not_required_by_design")
    if allowed.get("glob_loading_allowed") is not False:
        blockers.append("glob_loading_not_disabled_by_design")
    if design.get("paper_submit_allowed") is not False:
        blockers.append("design_allows_paper_submit")
    if design.get("selected_feature_contract") != "protocol101-live-v1":
        blockers.append("unexpected_feature_contract")
    if mode == "train":
        if not owner_approved_model_training:
            blockers.append("missing_owner_approved_model_training_flag")
        if not owner_approved_threshold_selection:
            blockers.append("missing_owner_approved_threshold_selection_flag")
        if not owner_approval_note.strip():
            blockers.append("missing_owner_approval_note")
    return blockers


def build_runner_plan(
    *,
    design: dict[str, Any],
    manifest: dict[str, Any],
    mode: str,
    policy_index: int,
    owner_approved_model_training: bool = False,
    owner_approved_threshold_selection: bool = False,
    owner_approval_note: str = "",
) -> dict[str, Any]:
    paths, split_blockers = paths_by_split(design, manifest)
    blockers = validate_plan(
        design=design,
        mode=mode,
        owner_approved_model_training=owner_approved_model_training,
        owner_approved_threshold_selection=owner_approved_threshold_selection,
        owner_approval_note=owner_approval_note,
    )
    blockers.extend(split_blockers)
    policy_name, cooldown_minutes = POLICY_META[int(policy_index)]
    status = "ready_to_train" if mode == "train" and not blockers else "dry_run_ready"
    if blockers:
        status = "blocked"
    decision = (
        "owner_approved_training_inputs_ready"
        if status == "ready_to_train"
        else "manifest_and_split_ready_no_training_executed"
        if status == "dry_run_ready"
        else "repair_runner_inputs_before_training"
    )
    return {
        "schema_version": "Protocol101FairContractTrainingRunnerPlanV1",
        "status": status,
        "decision": decision,
        "mode": mode,
        "selected_feature_contract": design.get("selected_feature_contract"),
        "policy_index": int(policy_index),
        "policy_name": policy_name,
        "cooldown_minutes": cooldown_minutes,
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "owner_approved_model_training": bool(owner_approved_model_training),
        "owner_approved_threshold_selection": bool(owner_approved_threshold_selection),
        "owner_approval_note_present": bool(owner_approval_note.strip()),
        "manifest_path": str((design.get("allowed_data") or {}).get("canonical_manifest") or ""),
        "split_sessions": {
            name: [path.name.removesuffix(".pkl") for path in split_paths]
            for name, split_paths in paths.items()
        },
        "split_files": {
            name: [str(path) for path in split_paths]
            for name, split_paths in paths.items()
        },
        "blockers": sorted(set(blockers)),
    }


def _load_split_decisions(plan: dict[str, Any]) -> dict[str, Any]:
    loaded = {}
    for split_name in ("train", "validation", "diagnostic_test"):
        paths = [Path(path) for path in plan["split_files"][split_name]]
        decisions = load_decisions(paths, policy_index=int(plan["policy_index"]))
        loaded[split_name] = {
            "paths": paths,
            "decisions": decisions,
            "decision_count": len(decisions),
            "candidate_count": int(sum(len(decision.labels) for decision in decisions)),
        }
    return loaded


def _load_teacher_positive_keys(
    teacher_events: Path,
    *,
    sessions: set[str],
    min_seed_count: int,
) -> set[tuple[str, str, str]]:
    """Load Protocol101 teacher enter keys for supervised labels only."""
    if not teacher_events.exists():
        raise FileNotFoundError(f"teacher events artifact missing: {teacher_events}")
    df = pd.read_parquet(teacher_events)
    df = df[
        (df["split"].astype(str) == "q1_2026")
        & (df["session"].astype(str).isin(sessions))
        & (df["protocol101_action"].astype(str) == "enter")
    ].copy()
    if df.empty:
        return set()
    df["decision_time"] = df["decision_time"].astype(str)
    df["contract_id"] = df["contract_id"].astype(str)
    grouped = (
        df.groupby(["session", "decision_time", "contract_id"], dropna=False)
        .agg(seed_count=("seed", "nunique"))
        .reset_index()
    )
    grouped = grouped[grouped["seed_count"] >= max(int(min_seed_count), 1)]
    return {
        (str(row.session), str(row.decision_time), str(row.contract_id))
        for row in grouped.itertuples(index=False)
    }


def _attach_teacher_labels(
    split_payload: dict[str, Any],
    *,
    teacher_events: Path,
    min_seed_count: int,
) -> dict[str, Any]:
    """Attach optional teacher labels to loaded decisions without changing features."""
    sessions = {
        str(decision.session)
        for payload in split_payload.values()
        for decision in payload["decisions"]
    }
    teacher_keys = _load_teacher_positive_keys(
        teacher_events,
        sessions=sessions,
        min_seed_count=min_seed_count,
    )
    summary: dict[str, Any] = {
        "teacher_events": str(teacher_events),
        "min_seed_count": int(min_seed_count),
        "teacher_positive_keys": len(teacher_keys),
        "positives_by_split": {},
    }
    for split_name, payload in split_payload.items():
        positives = 0
        candidates = 0
        present_decisions = 0
        for decision in payload["decisions"]:
            contract_ids = decision.contract_ids
            if contract_ids is None:
                labels = np.zeros(len(decision.labels), dtype=np.float32)
            else:
                labels = np.asarray(
                    [
                        1.0
                        if (
                            str(decision.session),
                            decision.decision_time.isoformat(),
                            str(contract_id),
                        )
                        in teacher_keys
                        else 0.0
                        for contract_id in contract_ids
                    ],
                    dtype=np.float32,
                )
            if bool(np.any(labels > 0.0)):
                present_decisions += 1
            positives += int(np.sum(labels > 0.0))
            candidates += len(labels)
            decision.teacher_labels = labels
        summary["positives_by_split"][split_name] = {
            "positive_candidates": positives,
            "candidate_count": candidates,
            "positive_decisions": present_decisions,
        }
    return summary


def _baseline_metrics(decisions, config: PilotConfig) -> dict[str, Any]:
    out = {
        "no_trade": metrics_for_trades([]),
        "random_valid": summarize_random_baseline(decisions, config=config),
    }
    for kind in ("atm_call", "atm_put", "vwap_omar"):
        trades = simulate_baseline(
            decisions,
            kind=kind,
            cooldown_minutes=config.cooldown_minutes,
            seed=config.seed,
        )
        out[kind] = metrics_for_trades(trades)
    return out


def _split_fit_and_calibration_decisions(
    train_decisions,
    *,
    fit_mode: str,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Split manifest train decisions into model-fit and threshold-calibration sets."""
    if fit_mode == "full_train":
        decisions = list(train_decisions)
        return decisions, decisions, {
            "fit_mode": fit_mode,
            "fit_sessions": sorted({decision.session for decision in decisions}),
            "calibration_sessions": sorted({decision.session for decision in decisions}),
            "fallback_used": False,
        }
    if fit_mode == "train_tail20_calibration":
        sessions = sorted({str(decision.session) for decision in train_decisions})
        fallback_used = False
        if len(sessions) < 2:
            decisions = list(train_decisions)
            return decisions, decisions, {
                "fit_mode": fit_mode,
                "fit_sessions": sessions,
                "calibration_sessions": sessions,
                "fallback_used": True,
            }
        calibration_count = min(20, max(1, len(sessions) // 5))
        calibration_sessions = set(sessions[-calibration_count:])
        fit_sessions = set(sessions[:-calibration_count])
        fit_decisions = [
            decision
            for decision in train_decisions
            if str(decision.session) in fit_sessions
        ]
        calibration_decisions = [
            decision
            for decision in train_decisions
            if str(decision.session) in calibration_sessions
        ]
        if not fit_decisions or not calibration_decisions:
            fit_decisions = list(train_decisions)
            calibration_decisions = list(train_decisions)
            fallback_used = True
        return fit_decisions, calibration_decisions, {
            "fit_mode": fit_mode,
            "fit_sessions": sorted({decision.session for decision in fit_decisions}),
            "calibration_sessions": sorted({decision.session for decision in calibration_decisions}),
            "fallback_used": fallback_used,
        }
    if fit_mode != "jan_fit_feb_calibration":
        raise ValueError(f"unknown fit_mode: {fit_mode}")
    fit_decisions = [
        decision
        for decision in train_decisions
        if str(decision.session)[5:7] == "01"
    ]
    calibration_decisions = [
        decision
        for decision in train_decisions
        if str(decision.session)[5:7] == "02"
    ]
    fallback_used = False
    if not fit_decisions:
        fit_decisions = list(train_decisions)
        fallback_used = True
    if not calibration_decisions:
        calibration_decisions = list(train_decisions)
        fallback_used = True
    return fit_decisions, calibration_decisions, {
        "fit_mode": fit_mode,
        "fit_sessions": sorted({decision.session for decision in fit_decisions}),
        "calibration_sessions": sorted({decision.session for decision in calibration_decisions}),
        "fallback_used": fallback_used,
    }


def parse_ensemble_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list for an explicit ensemble attempt."""
    seeds: list[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds


def _average_predictions(predictions_by_member: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Average per-decision candidate predictions from models with identical rows."""
    if not predictions_by_member:
        return []
    averaged: list[np.ndarray] = []
    for per_decision in zip(*predictions_by_member):
        if not per_decision:
            averaged.append(np.asarray([], dtype=np.float32))
            continue
        stacked = np.vstack([np.asarray(scores, dtype=np.float32) for scores in per_decision])
        averaged.append(stacked.mean(axis=0).astype(np.float32))
    return averaged


def _prediction_transform_for_target_mode(target_mode: str) -> str:
    if target_mode in CLASSIFIER_TARGET_MODES:
        return "sigmoid"
    if target_mode in LISTWISE_TARGET_MODES:
        return "identity_unit"
    return "identity"


def _balanced_classifier_sample_weight(
    y_fit: np.ndarray,
    *,
    mode: str,
    decision_ids: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return per-example weights for rare-positive classifier targets."""
    if str(mode) not in {"balanced_classifier", "decision_balanced_classifier"}:
        return None
    y = np.asarray(y_fit, dtype=np.int8)
    if y.size == 0:
        return None
    positives = int(np.sum(y > 0))
    negatives = int(y.size - positives)
    if positives <= 0 or negatives <= 0:
        return None
    if str(mode) == "decision_balanced_classifier" and decision_ids is not None:
        ids = np.asarray(decision_ids)
        if ids.shape[0] != y.shape[0]:
            raise ValueError("decision_ids length must match y_fit length")
        _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
        base = 1.0 / np.asarray(counts[inverse], dtype=np.float32)
    else:
        base = np.ones_like(y, dtype=np.float32)
    positive_weight = float(base[y > 0].sum())
    negative_weight = float(base[y <= 0].sum())
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return None
    total = positive_weight + negative_weight
    weights = np.where(
        y > 0,
        base * total / (2.0 * positive_weight),
        base * total / (2.0 * negative_weight),
    )
    weight_sum = float(weights.sum())
    if weight_sum > 0.0:
        weights = weights * (float(y.size) / weight_sum)
    return weights.astype(np.float32)


def _targets_for_decision(decision, *, config: PilotConfig) -> np.ndarray:
    """Build per-candidate training targets using the same fair labels everywhere."""
    labels = np.asarray(decision.labels, dtype=np.float32)
    if config.target_mode in {
        "protocol101_teacher_classifier",
        "protocol101_teacher_profitable_classifier",
    }:
        teacher = (
            np.asarray(decision.teacher_labels, dtype=np.float32)
            if decision.teacher_labels is not None
            else np.zeros(len(labels), dtype=np.float32)
        )
        targets = (teacher > 0.0).astype(np.float32)
        if config.target_mode == "protocol101_teacher_profitable_classifier":
            targets = (
                (targets > 0.0)
                & np.isfinite(labels)
                & (labels > float(config.positive_label_threshold))
            ).astype(np.float32)
    elif config.target_mode == "profit_classifier":
        targets = (labels > config.positive_label_threshold).astype(np.float32)
    elif config.target_mode in {
        "decision_profit_presence_classifier",
        "decision_best_profit_regression",
    }:
        targets = np.zeros(len(labels), dtype=np.float32)
        if np.isfinite(labels).any():
            max_label = float(np.nanmax(labels))
            if config.target_mode == "decision_profit_presence_classifier":
                target_value = 1.0 if max_label > float(config.positive_label_threshold) else 0.0
            else:
                target_value = (
                    np.clip(max_label, -config.target_clip, config.target_clip)
                    / config.target_scale
                )
            targets[:] = float(target_value)
    elif config.target_mode in {"decision_top_profit_classifier", "decision_top_profit_regression"}:
        targets = np.zeros(len(labels), dtype=np.float32)
        if np.isfinite(labels).any():
            max_label = float(np.nanmax(labels))
            if max_label > float(config.positive_label_threshold):
                best = np.asarray(labels == max_label, dtype=bool)
                if config.target_mode == "decision_top_profit_classifier":
                    targets[best] = 1.0
                else:
                    targets[best] = (
                        np.clip(max_label, -config.target_clip, config.target_clip)
                        / config.target_scale
                    )
    elif config.target_mode in {"decision_relative_regression", "blended_relative_regression"}:
        relative = (
            np.clip(labels - float(np.nanmedian(labels)), -config.target_clip, config.target_clip)
            / config.target_scale
        ).astype(np.float32)
        if config.target_mode == "blended_relative_regression":
            absolute = (
                np.clip(labels, -config.target_clip, config.target_clip)
                / config.target_scale
            ).astype(np.float32)
            weight = min(max(float(config.relative_target_weight), 0.0), 1.0)
            targets = ((1.0 - weight) * absolute + weight * relative).astype(np.float32)
        else:
            targets = relative
    else:
        targets = (
            np.clip(labels, -config.target_clip, config.target_clip) / config.target_scale
        ).astype(np.float32)
    return targets.astype(np.float32)


def _flatten_training_examples_with_decision_ids(
    decisions,
    *,
    config: PilotConfig,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect flat examples and retain a decision id for equal-minute weighting."""
    features_by_decision: list[np.ndarray] = []
    targets_by_decision: list[np.ndarray] = []
    decision_ids_by_decision: list[np.ndarray] = []
    for decision_idx, decision in enumerate(decisions):
        labels = np.asarray(decision.labels, dtype=np.float32)
        if len(labels) == 0:
            continue
        targets = _targets_for_decision(decision, config=config)
        features_by_decision.append(np.asarray(decision.features, dtype=np.float32))
        targets_by_decision.append(targets.astype(np.float32))
        decision_ids_by_decision.append(
            np.full(len(targets), int(decision_idx), dtype=np.int64)
        )
    features = np.vstack(features_by_decision).astype(np.float32)
    targets = np.concatenate(targets_by_decision).astype(np.float32)
    decision_ids = np.concatenate(decision_ids_by_decision).astype(np.int64)
    if max_examples is not None and len(targets) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(targets), size=max_examples, replace=False)
        features = features[idx]
        targets = targets[idx]
        decision_ids = decision_ids[idx]
    return features, targets, decision_ids


def _predict_sklearn_decisions(
    estimator: Any,
    decisions,
    *,
    config: PilotConfig,
) -> list[np.ndarray]:
    """Predict per-candidate scores from a sklearn tabular estimator."""
    if not decisions:
        return []
    lengths = [len(decision.labels) for decision in decisions]
    features = np.vstack([decision.features for decision in decisions]).astype(np.float32)
    if config.target_mode in CLASSIFIER_TARGET_MODES:
        proba = estimator.predict_proba(features)
        classes = list(getattr(estimator, "classes_", []))
        if 1 in classes:
            positive_idx = classes.index(1)
        elif 1.0 in classes:
            positive_idx = classes.index(1.0)
        else:
            positive_idx = proba.shape[1] - 1
        flat = np.asarray(proba[:, positive_idx], dtype=np.float32)
    else:
        flat = np.asarray(estimator.predict(features), dtype=np.float32) * float(config.target_scale)
    out: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        out.append(flat[cursor : cursor + length])
        cursor += length
    return out


def _flatten_training_examples_with_rights(
    decisions,
    *,
    config: PilotConfig,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect flat fair-contract examples with candidate right for specialists."""
    if config.target_mode in LISTWISE_TARGET_MODES:
        raise ValueError("right-specialized sklearn family does not support listwise target mode")
    features_by_decision: list[np.ndarray] = []
    targets_by_decision: list[np.ndarray] = []
    rights_by_decision: list[np.ndarray] = []
    for decision in decisions:
        labels = np.asarray(decision.labels, dtype=np.float32)
        if len(labels) == 0:
            continue
        targets = _targets_for_decision(decision, config=config)
        features_by_decision.append(np.asarray(decision.features, dtype=np.float32))
        targets_by_decision.append(targets.astype(np.float32))
        rights_by_decision.append(np.asarray(decision.rights, dtype=object))
    features = np.vstack(features_by_decision).astype(np.float32)
    targets = np.concatenate(targets_by_decision).astype(np.float32)
    rights = np.concatenate(rights_by_decision).astype(object)
    if max_examples is not None and len(targets) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(targets), size=max_examples, replace=False)
        features = features[idx]
        targets = targets[idx]
        rights = rights[idx]
    return features, targets, rights


def _sklearn_positive_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    proba = estimator.predict_proba(features)
    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        positive_idx = classes.index(1)
    elif 1.0 in classes:
        positive_idx = classes.index(1.0)
    else:
        positive_idx = proba.shape[1] - 1
    return np.asarray(proba[:, positive_idx], dtype=np.float32)


def _predict_sklearn_by_right_decisions(
    estimators_by_right: dict[str, Any],
    decisions,
    *,
    config: PilotConfig,
) -> list[np.ndarray]:
    """Predict scores from right-specialized estimators."""
    if not decisions:
        return []
    lengths = [len(decision.labels) for decision in decisions]
    features = np.vstack([decision.features for decision in decisions]).astype(np.float32)
    rights = np.concatenate([np.asarray(decision.rights, dtype=object) for decision in decisions])
    flat_scores = np.full(len(rights), -1e9, dtype=np.float32)
    for right, estimator in estimators_by_right.items():
        mask = np.asarray(rights == right, dtype=bool)
        if not mask.any():
            continue
        if config.target_mode in CLASSIFIER_TARGET_MODES:
            flat_scores[mask] = _sklearn_positive_scores(estimator, features[mask])
        else:
            flat_scores[mask] = (
                np.asarray(estimator.predict(features[mask]), dtype=np.float32)
                * float(config.target_scale)
            )
    predictions: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        predictions.append(flat_scores[cursor : cursor + length])
        cursor += length
    return predictions


def _train_sklearn_hist_gradient_boosting(
    train_decisions,
    validation_decisions,
    diagnostic_decisions,
    *,
    config: PilotConfig,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Train a tabular gradient-boosting candidate on flat fair-contract examples."""
    if config.target_mode in LISTWISE_TARGET_MODES:
        raise ValueError("sklearn_hist_gradient_boosting does not support listwise target mode")
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    x_train, y_train, decision_ids = _flatten_training_examples_with_decision_ids(
        train_decisions,
        config=config,
        max_examples=config.max_train_examples,
        seed=config.seed,
    )
    if config.target_mode in CLASSIFIER_TARGET_MODES:
        estimator = HistGradientBoostingClassifier(
            max_iter=max(int(config.epochs) * 40, 80),
            learning_rate=float(config.learning_rate),
            l2_regularization=float(config.weight_decay),
            max_leaf_nodes=31,
            random_state=int(config.seed),
        )
        y_fit = np.asarray(y_train > 0.5, dtype=np.int8)
        sample_weight = _balanced_classifier_sample_weight(
            y_fit,
            mode=config.sample_weight_mode,
            decision_ids=decision_ids,
        )
    else:
        estimator = HistGradientBoostingRegressor(
            max_iter=max(int(config.epochs) * 40, 80),
            learning_rate=float(config.learning_rate),
            l2_regularization=float(config.weight_decay),
            max_leaf_nodes=31,
            random_state=int(config.seed),
        )
        y_fit = y_train
        sample_weight = None
    if sample_weight is not None:
        estimator.fit(x_train, y_fit, sample_weight=sample_weight)
    else:
        estimator.fit(x_train, y_fit)
    predictions = {
        "train": _predict_sklearn_decisions(estimator, train_decisions, config=config),
        "validation": _predict_sklearn_decisions(estimator, validation_decisions, config=config),
        "diagnostic_test": _predict_sklearn_decisions(estimator, diagnostic_decisions, config=config),
    }
    state = {
        "feature_version": "candidate_v3_envpriors",
        "model_family": "sklearn_hist_gradient_boosting",
        "input_dim": int(x_train.shape[1]),
        "config": asdict(config),
        "sklearn_model": estimator,
        "history": [
            {
                "target_mode": config.target_mode,
                "train_examples": int(len(y_fit)),
                "positive_examples": int(np.sum(y_fit > 0)) if config.target_mode in CLASSIFIER_TARGET_MODES else None,
                "sample_weight_mode": str(config.sample_weight_mode),
                "sample_weight_positive_mean": (
                    float(sample_weight[y_fit > 0].mean())
                    if sample_weight is not None and bool(np.any(y_fit > 0))
                    else None
                ),
                "sample_weight_negative_mean": (
                    float(sample_weight[y_fit <= 0].mean())
                    if sample_weight is not None and bool(np.any(y_fit <= 0))
                    else None
                ),
                "max_iter": int(getattr(estimator, "max_iter", 0)),
            }
        ],
    }
    preview_extra = {
        "model_family": "sklearn_hist_gradient_boosting",
        "train_examples": int(len(y_fit)),
        "positive_examples": int(np.sum(y_fit > 0)) if config.target_mode in CLASSIFIER_TARGET_MODES else None,
        "sample_weight_mode": str(config.sample_weight_mode),
    }
    return state, predictions, preview_extra


def _train_sklearn_hist_gradient_boosting_by_right(
    train_decisions,
    validation_decisions,
    diagnostic_decisions,
    *,
    config: PilotConfig,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Train separate tabular boosted estimators for calls and puts."""
    if config.target_mode in LISTWISE_TARGET_MODES:
        raise ValueError("sklearn_hist_gradient_boosting_by_right does not support listwise target mode")
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    x_train, y_train, rights = _flatten_training_examples_with_rights(
        train_decisions,
        config=config,
        max_examples=config.max_train_examples,
        seed=config.seed,
    )
    estimators_by_right: dict[str, Any] = {}
    right_summary: dict[str, dict[str, Any]] = {}
    for right in ("C", "P"):
        mask = np.asarray(rights == right, dtype=bool)
        x_right = x_train[mask]
        y_right = y_train[mask]
        if len(y_right) == 0:
            continue
        if config.target_mode in CLASSIFIER_TARGET_MODES:
            y_fit = np.asarray(y_right > 0.5, dtype=np.int8)
            sample_weight = _balanced_classifier_sample_weight(
                y_fit,
                mode=config.sample_weight_mode,
            )
            if len(np.unique(y_fit)) < 2:
                estimator = DummyClassifier(strategy="constant", constant=int(y_fit[0]))
                sample_weight = None
            else:
                estimator = HistGradientBoostingClassifier(
                    max_iter=max(int(config.epochs) * 40, 80),
                    learning_rate=float(config.learning_rate),
                    l2_regularization=float(config.weight_decay),
                    max_leaf_nodes=31,
                    random_state=int(config.seed),
                )
        else:
            y_fit = y_right
            sample_weight = None
            if float(np.nanstd(y_fit)) < 1e-9:
                estimator = DummyRegressor(strategy="constant", constant=float(y_fit[0]))
            else:
                estimator = HistGradientBoostingRegressor(
                    max_iter=max(int(config.epochs) * 40, 80),
                    learning_rate=float(config.learning_rate),
                    l2_regularization=float(config.weight_decay),
                    max_leaf_nodes=31,
                    random_state=int(config.seed),
                )
        if sample_weight is not None:
            estimator.fit(x_right, y_fit, sample_weight=sample_weight)
        else:
            estimator.fit(x_right, y_fit)
        estimators_by_right[right] = estimator
        right_summary[right] = {
            "train_examples": int(len(y_fit)),
            "positive_examples": int(np.sum(y_fit > 0)) if config.target_mode in CLASSIFIER_TARGET_MODES else None,
            "sample_weight_mode": str(config.sample_weight_mode),
            "sample_weight_positive_mean": (
                float(sample_weight[y_fit > 0].mean())
                if sample_weight is not None and bool(np.any(y_fit > 0))
                else None
            ),
            "sample_weight_negative_mean": (
                float(sample_weight[y_fit <= 0].mean())
                if sample_weight is not None and bool(np.any(y_fit <= 0))
                else None
            ),
        }
    predictions = {
        "train": _predict_sklearn_by_right_decisions(estimators_by_right, train_decisions, config=config),
        "validation": _predict_sklearn_by_right_decisions(estimators_by_right, validation_decisions, config=config),
        "diagnostic_test": _predict_sklearn_by_right_decisions(estimators_by_right, diagnostic_decisions, config=config),
    }
    state = {
        "feature_version": "candidate_v3_envpriors",
        "model_family": "sklearn_hist_gradient_boosting_by_right",
        "input_dim": int(x_train.shape[1]),
        "config": asdict(config),
        "sklearn_models_by_right": estimators_by_right,
        "history": [
            {
                "target_mode": config.target_mode,
                "right_summary": right_summary,
            }
        ],
    }
    preview_extra = {
        "model_family": "sklearn_hist_gradient_boosting_by_right",
        "right_summary": right_summary,
    }
    return state, predictions, preview_extra


def _stress_trades(trades: list[Trade], *, stress_per_trade: float) -> list[Trade]:
    if stress_per_trade <= 0.0:
        return list(trades)
    return [
        replace(trade, pnl=float(trade.pnl) - float(stress_per_trade))
        for trade in trades
    ]


def _trade_action_presence_match_rate(
    baseline_trades: list[Trade],
    scenario_trades: list[Trade],
    *,
    total_decisions: int,
) -> float:
    if total_decisions <= 0:
        return 1.0
    baseline_keys = {
        (str(trade.session), str(trade.decision_time))
        for trade in baseline_trades
    }
    scenario_keys = {
        (str(trade.session), str(trade.decision_time))
        for trade in scenario_trades
    }
    mismatches = len(baseline_keys.symmetric_difference(scenario_keys))
    return max(float(total_decisions - mismatches), 0.0) / float(total_decisions)


def _enrich_threshold_row_with_jitter_metrics(
    row: dict[str, Any],
    *,
    decisions,
    jitter_predictions_by_scenario: dict[str, list[np.ndarray]],
    threshold: float,
    config: PilotConfig,
    stress_per_trade: float,
    baseline_trades: list[Trade],
) -> dict[str, Any]:
    if not jitter_predictions_by_scenario:
        return row
    scenario_rows: list[dict[str, Any]] = []
    baseline_stressed_pnl = float(row.get("stressed_total_pnl") or 0.0)
    for scenario_name, scenario_predictions in sorted(jitter_predictions_by_scenario.items()):
        trades = simulate_model_policy(
            decisions,
            scenario_predictions,
            threshold=float(threshold),
            cooldown_minutes=config.cooldown_minutes,
            strategy=f"neural_threshold_{scenario_name}",
            entry_filter=config.entry_filter,
            min_score_margin=config.min_score_margin,
            max_score_ceiling=config.max_score_ceiling,
            max_trades_per_session=config.max_trades_per_session,
            max_daily_loss=config.max_daily_loss,
            selection_mode=config.selection_mode,
            cash_pnl_adjustment=-float(stress_per_trade),
        )
        stressed = metrics_for_trades(
            _stress_trades(trades, stress_per_trade=stress_per_trade)
        )
        action_match = _trade_action_presence_match_rate(
            baseline_trades,
            trades,
            total_decisions=len(decisions),
        )
        pnl_drift_fraction = (
            abs(float(stressed["total_pnl"]) - baseline_stressed_pnl)
            / abs(baseline_stressed_pnl)
            if abs(baseline_stressed_pnl) > 1e-9
            else 0.0
        )
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "trades": int(stressed["trades"]),
                "stressed_total_pnl": float(stressed["total_pnl"]),
                "stressed_profit_factor": float(stressed["profit_factor"]),
                "stressed_max_drawdown": float(stressed["max_drawdown"]),
                "action_presence_match_rate": float(action_match),
                "pnl_drift_fraction": float(pnl_drift_fraction),
            }
        )
    if not scenario_rows:
        return row
    finite_pfs = [
        float(item["stressed_profit_factor"])
        if np.isfinite(float(item["stressed_profit_factor"]))
        else 10.0
        for item in scenario_rows
    ]
    enriched = dict(row)
    enriched["jitter_scenarios"] = scenario_rows
    enriched["jitter_scenario_count"] = len(scenario_rows)
    enriched["jitter_worst_total_pnl"] = min(float(item["stressed_total_pnl"]) for item in scenario_rows)
    enriched["jitter_worst_profit_factor"] = min(finite_pfs)
    enriched["jitter_min_trades"] = min(int(item["trades"]) for item in scenario_rows)
    enriched["jitter_worst_max_drawdown"] = min(float(item["stressed_max_drawdown"]) for item in scenario_rows)
    enriched["jitter_min_action_presence_match_rate"] = min(
        float(item["action_presence_match_rate"]) for item in scenario_rows
    )
    enriched["jitter_max_pnl_drift_fraction"] = max(
        float(item["pnl_drift_fraction"]) for item in scenario_rows
    )
    return enriched


def choose_threshold_with_rule(
    decisions,
    predictions,
    *,
    config: PilotConfig,
    threshold_rule: str,
    stress_per_trade: float,
    jitter_predictions_by_scenario: dict[str, list[np.ndarray]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose a threshold from validation predictions using a declared rule."""
    if threshold_rule == "max_validation_pnl" and stress_per_trade == 20.0:
        threshold, raw_sweep = choose_threshold(decisions, predictions, config=config)
        enriched = []
        for row in raw_sweep:
            trades = simulate_model_policy(
                decisions,
                predictions,
                threshold=float(row["threshold"]),
                cooldown_minutes=config.cooldown_minutes,
                strategy="neural_threshold",
                entry_filter=config.entry_filter,
                min_score_margin=config.min_score_margin,
                max_score_ceiling=config.max_score_ceiling,
                max_trades_per_session=config.max_trades_per_session,
                max_daily_loss=config.max_daily_loss,
                selection_mode=config.selection_mode,
                cash_pnl_adjustment=-float(stress_per_trade),
            )
            stressed = metrics_for_trades(
                _stress_trades(trades, stress_per_trade=stress_per_trade)
            )
            enriched.append(
                {
                    **row,
                    "raw_total_pnl": row["total_pnl"],
                    "raw_profit_factor": row["profit_factor"],
                    "stressed_total_pnl": stressed["total_pnl"],
                    "stressed_profit_factor": stressed["profit_factor"],
                    "stressed_max_drawdown": stressed["max_drawdown"],
                    "stressed_win_rate": stressed["win_rate"],
                    "stressed_positive_day_fraction": stressed["positive_day_fraction"],
                    "stressed_median_daily_pnl": stressed.get("median_daily_pnl", 0.0),
                }
            )
        return threshold, enriched

    top_scores = np.asarray(
        [
            float(top[1])
            for decision, pred in zip(decisions, predictions)
            if (
                top := top_prediction(
                    decision,
                    pred,
                    entry_filter=config.entry_filter,
                    selection_mode=config.selection_mode,
                )
            )
            is not None
            and (
                float(config.max_score_ceiling) <= 0.0
                or float(top[1]) < float(config.max_score_ceiling)
            )
        ],
        dtype=float,
    )
    if len(top_scores) == 0:
        return float("inf"), []
    quantiles = np.linspace(0.0, 0.98, 50)
    thresholds = sorted(set(np.quantile(top_scores, quantiles).round(4).tolist() + [0.0]))
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        trades = simulate_model_policy(
            decisions,
            predictions,
            threshold=float(threshold),
            cooldown_minutes=config.cooldown_minutes,
            strategy="neural_threshold",
            entry_filter=config.entry_filter,
            min_score_margin=config.min_score_margin,
            max_score_ceiling=config.max_score_ceiling,
            max_trades_per_session=config.max_trades_per_session,
            max_daily_loss=config.max_daily_loss,
            selection_mode=config.selection_mode,
            cash_pnl_adjustment=-float(stress_per_trade),
        )
        raw = metrics_for_trades(trades)
        stressed = metrics_for_trades(
            _stress_trades(trades, stress_per_trade=stress_per_trade)
        )
        row = {
                "threshold": float(threshold),
                **raw,
                "raw_total_pnl": raw["total_pnl"],
                "raw_profit_factor": raw["profit_factor"],
                "stressed_total_pnl": stressed["total_pnl"],
                "stressed_profit_factor": stressed["profit_factor"],
                "stressed_max_drawdown": stressed["max_drawdown"],
                "stressed_win_rate": stressed["win_rate"],
                "stressed_positive_day_fraction": stressed["positive_day_fraction"],
                "stressed_median_daily_pnl": stressed.get("median_daily_pnl", 0.0),
            }
        if threshold_rule == "jitter_stability_stressed":
            row = _enrich_threshold_row_with_jitter_metrics(
                row,
                decisions=decisions,
                jitter_predictions_by_scenario=jitter_predictions_by_scenario or {},
                threshold=float(threshold),
                config=config,
                stress_per_trade=float(stress_per_trade),
                baseline_trades=trades,
            )
        sweep.append(row)

    eligible = [row for row in sweep if row["trades"] >= config.min_validation_trades]
    pool = eligible if eligible else sweep
    if threshold_rule == "max_validation_stressed_pnl":
        best = max(
            pool,
            key=lambda row: (
                row["stressed_total_pnl"],
                row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 999.0,
                row["raw_total_pnl"],
                -abs(row["stressed_max_drawdown"]),
                row["trades"],
            ),
        )
    elif threshold_rule == "risk_adjusted_stressed":
        def risk_key(row: dict[str, Any]) -> tuple[float, ...]:
            trade_sufficiency = min(float(row["trades"]) / max(float(config.min_validation_trades), 1.0), 1.0)
            churn_penalty = max(float(row["trades"]) - 80.0, 0.0) * 10.0
            drawdown_penalty = abs(float(row["stressed_max_drawdown"])) * 0.35
            pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            score = (
                float(row["stressed_total_pnl"])
                + 450.0 * min(float(pf), 3.0)
                + 350.0 * float(row["positive_day_fraction"])
                + 250.0 * trade_sufficiency
                - churn_penalty
                - drawdown_penalty
            )
            return (
                score,
                float(row["stressed_total_pnl"]),
                float(pf),
                -abs(float(row["stressed_max_drawdown"])),
                -abs(float(row["trades"]) - 45.0),
            )

        best = max(pool, key=risk_key)
    elif threshold_rule == "frequency_sufficient_stressed":
        target_trades = max(float(config.min_validation_trades) * 2.0, 40.0)
        useful = [row for row in pool if float(row["trades"]) >= target_trades]
        candidate_pool = useful if useful else pool

        def frequency_key(row: dict[str, Any]) -> tuple[float, ...]:
            trades = float(row["trades"])
            pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            trade_score = min(trades / target_trades, 1.25) * 600.0
            churn_penalty = max(trades - 100.0, 0.0) * 12.0
            drawdown_penalty = abs(float(row["stressed_max_drawdown"])) * 0.25
            score = (
                float(row["stressed_total_pnl"])
                + trade_score
                + 300.0 * min(float(pf), 3.0)
                + 250.0 * float(row["positive_day_fraction"])
                - churn_penalty
                - drawdown_penalty
            )
            return (
                score,
                trades,
                float(row["stressed_total_pnl"]),
                float(pf),
                -abs(float(row["stressed_max_drawdown"])),
            )

        best = max(candidate_pool, key=frequency_key)
    elif threshold_rule == "daily_stability_stressed":
        useful = [
            row
            for row in pool
            if float(row["trades"]) >= max(float(config.min_validation_trades), 20.0)
        ]
        candidate_pool = useful if useful else pool

        def stability_key(row: dict[str, Any]) -> tuple[float, ...]:
            pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            score = (
                700.0 * float(row.get("stressed_positive_day_fraction") or 0.0)
                + 3.0 * float(row.get("stressed_median_daily_pnl") or 0.0)
                + 0.35 * float(row["stressed_total_pnl"])
                + 300.0 * min(float(pf), 3.0)
                - 0.20 * abs(float(row["stressed_max_drawdown"]))
                - max(float(row["trades"]) - 80.0, 0.0) * 8.0
            )
            return (
                score,
                float(row.get("stressed_positive_day_fraction") or 0.0),
                float(row.get("stressed_median_daily_pnl") or 0.0),
                float(row["stressed_total_pnl"]),
                float(pf),
                -abs(float(row["stressed_max_drawdown"])),
            )

        best = max(candidate_pool, key=stability_key)
    elif threshold_rule == "drawdown_guarded_stressed":
        max_allowed_drawdown = 6000.0
        min_trades = max(float(config.min_validation_trades), 20.0)
        guarded = [
            row
            for row in pool
            if float(row["trades"]) >= min_trades
            and float(row["stressed_total_pnl"]) > 0.0
            and abs(float(row["stressed_max_drawdown"])) <= max_allowed_drawdown
        ]
        if not guarded:
            guarded = [
                row
                for row in pool
                if float(row["trades"]) >= min_trades
                and float(row["stressed_total_pnl"]) > 0.0
            ]
        candidate_pool = guarded if guarded else pool

        def drawdown_key(row: dict[str, Any]) -> tuple[float, ...]:
            pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            drawdown = abs(float(row["stressed_max_drawdown"]))
            trades = float(row["trades"])
            trade_sufficiency = min(trades / min_trades, 1.0)
            score = (
                float(row["stressed_total_pnl"])
                + 550.0 * min(float(pf), 3.0)
                + 300.0 * float(row.get("stressed_positive_day_fraction") or 0.0)
                + 250.0 * trade_sufficiency
                - 0.80 * drawdown
                - max(trades - 70.0, 0.0) * 20.0
            )
            return (
                score,
                -drawdown,
                float(row["stressed_total_pnl"]),
                float(pf),
                -abs(trades - 35.0),
            )

        best = max(candidate_pool, key=drawdown_key)
    elif threshold_rule == "conservative_pnl_plateau_stressed":
        min_trades = max(float(config.min_validation_trades), 20.0)
        viable = [
            row
            for row in pool
            if float(row["trades"]) >= min_trades
            and float(row["stressed_total_pnl"]) > 0.0
            and float(row["stressed_profit_factor"]) >= 1.25
        ]
        if viable:
            max_pnl = max(float(row["stressed_total_pnl"]) for row in viable)
            pnl_floor = max_pnl * 0.60
            candidate_pool = [
                row
                for row in viable
                if float(row["stressed_total_pnl"]) >= pnl_floor
            ]
        else:
            candidate_pool = pool

        def plateau_key(row: dict[str, Any]) -> tuple[float, ...]:
            pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            return (
                float(row["threshold"]),
                min(float(pf), 3.0),
                float(row["stressed_total_pnl"]),
                -abs(float(row["stressed_max_drawdown"])),
            )

        best = max(candidate_pool, key=plateau_key)
    elif threshold_rule == "jitter_stability_stressed":
        min_trades = max(float(config.min_validation_trades), 20.0)
        viable = [
            row
            for row in pool
            if float(row["trades"]) >= min_trades
            and float(row.get("jitter_min_trades") or 0.0) >= min_trades
            and float(row.get("jitter_worst_total_pnl") or 0.0) > 0.0
            and float(row.get("jitter_worst_profit_factor") or 0.0) >= 1.10
            and float(row.get("jitter_min_action_presence_match_rate") or 0.0) >= 0.98
            and float(row.get("jitter_max_pnl_drift_fraction") or 0.0) <= 0.50
        ]
        candidate_pool = viable if viable else pool

        def jitter_key(row: dict[str, Any]) -> tuple[float, ...]:
            raw_pf = row["stressed_profit_factor"] if np.isfinite(row["stressed_profit_factor"]) else 10.0
            jitter_pf = float(row.get("jitter_worst_profit_factor") or 0.0)
            action_match = float(row.get("jitter_min_action_presence_match_rate") or 0.0)
            pnl_drift = float(row.get("jitter_max_pnl_drift_fraction") or 0.0)
            worst_pnl = float(row.get("jitter_worst_total_pnl") or -100_000.0)
            min_trades_seen = float(row.get("jitter_min_trades") or 0.0)
            trade_sufficiency = min(min_trades_seen / min_trades, 1.0)
            score = (
                worst_pnl
                + 650.0 * min(max(jitter_pf, 0.0), 3.0)
                + 450.0 * min(max(float(raw_pf), 0.0), 3.0)
                + 900.0 * action_match
                + 300.0 * trade_sufficiency
                - 1000.0 * max(pnl_drift - 0.35, 0.0)
                - 0.25 * abs(float(row.get("jitter_worst_max_drawdown") or 0.0))
                - max(float(row["trades"]) - 80.0, 0.0) * 10.0
            )
            return (
                score,
                worst_pnl,
                action_match,
                min(max(jitter_pf, 0.0), 3.0),
                float(row["stressed_total_pnl"]),
                -pnl_drift,
            )

        best = max(candidate_pool, key=jitter_key)
    else:
        best = max(
            pool,
            key=lambda row: (
                row["total_pnl"],
                row["profit_factor"] if np.isfinite(row["profit_factor"]) else 999.0,
                row["trades"],
            ),
        )
    return float(best["threshold"]), sweep


def run_training(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    policy_name, cooldown_minutes = POLICY_META[int(args.policy_index)]
    parsed_ensemble_seeds = parse_ensemble_seeds(str(args.ensemble_seeds))
    member_seeds = parsed_ensemble_seeds if parsed_ensemble_seeds else [int(args.seed)]
    is_ensemble = len(member_seeds) > 1
    config = replace(
        PilotConfig(),
        policy_index=int(args.policy_index),
        policy_name=policy_name,
        cooldown_minutes=cooldown_minutes,
        epochs=int(args.epochs),
        max_train_examples=int(args.max_train_examples),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        target_mode=str(args.target_mode),
        target_clip=float(args.target_clip),
        positive_label_threshold=float(args.positive_label_threshold),
        relative_target_weight=float(args.relative_target_weight),
        entry_filter=str(args.entry_filter),
        min_score_margin=float(args.min_score_margin),
        max_score_ceiling=max(float(args.max_score_ceiling), 0.0),
        max_trades_per_session=max(int(args.max_trades_per_session), 0),
        max_daily_loss=max(float(args.max_daily_loss), 0.0),
        sample_weight_mode=str(args.sample_weight_mode),
        feature_transform=str(args.feature_transform),
        feature_noise_augmentation=str(args.feature_noise_augmentation),
        selection_mode=str(args.selection_mode),
        seed=int(member_seeds[0]),
    )
    prediction_transform = _prediction_transform_for_target_mode(config.target_mode)
    loaded = _load_split_decisions(plan)
    teacher_label_summary: dict[str, Any] | None = None
    if config.target_mode in TEACHER_TARGET_MODES:
        teacher_label_summary = _attach_teacher_labels(
            loaded,
            teacher_events=Path(args.teacher_events),
            min_seed_count=int(args.teacher_min_seed_count),
        )
    if config.feature_transform != "none":
        for payload in loaded.values():
            payload["decisions"] = transform_decision_features(
                payload["decisions"],
                config.feature_transform,
            )
    train_decisions_all = loaded["train"]["decisions"]
    fit_decisions, calibration_decisions, fit_calibration_summary = _split_fit_and_calibration_decisions(
        train_decisions_all,
        fit_mode=str(args.fit_mode),
    )
    fit_decisions_unaugmented_count = len(fit_decisions)
    fit_decisions = augment_decision_features_with_noise(
        fit_decisions,
        config.feature_noise_augmentation,
    )
    validation_decisions = loaded["validation"]["decisions"]
    diagnostic_decisions = loaded["diagnostic_test"]["decisions"]
    preview_x, preview_y = collect_examples(fit_decisions, max_examples=20_000, seed=config.seed)
    preview = {
        "sample_examples": int(len(preview_y)),
        "input_dim": int(preview_x.shape[1]),
        "label_mean": float(preview_y.mean()) if len(preview_y) else 0.0,
        "label_median": float(np.median(preview_y)) if len(preview_y) else 0.0,
        "fit_mode": str(args.fit_mode),
        "fit_decisions": int(len(fit_decisions)),
        "fit_decisions_unaugmented": int(fit_decisions_unaugmented_count),
        "feature_noise_augmentation": str(args.feature_noise_augmentation),
        "calibration_decisions": int(len(calibration_decisions)),
        "teacher_label_summary": teacher_label_summary,
    }
    if config.target_mode in TEACHER_TARGET_MODES:
        _preview_features, preview_targets = collect_training_examples(
            fit_decisions,
            config=config,
            max_examples=20_000,
            seed=config.seed,
        )
        preview["target_mean"] = float(preview_targets.mean()) if len(preview_targets) else 0.0
        preview["target_positive_examples"] = int(np.sum(preview_targets > 0.5))
        preview["target_negative_examples"] = int(np.sum(preview_targets < -0.5))
        del _preview_features, preview_targets
    del preview_x, preview_y

    trained_members: list[dict[str, Any]] = []
    histories_by_seed: dict[str, list[dict[str, Any]]] = {}
    prediction_members_by_split: dict[str, list[list[np.ndarray]]] = {
        "train": [],
        "validation": [],
        "diagnostic_test": [],
    }
    sklearn_state: dict[str, Any] | None = None
    sklearn_preview: dict[str, Any] = {}
    mlp_runtime_members: list[tuple[Any, Any, float]] = []
    if str(args.model_family) in {
        "sklearn_hist_gradient_boosting",
        "sklearn_hist_gradient_boosting_by_right",
    }:
        if is_ensemble:
            raise ValueError(f"{args.model_family} does not support ensemble_seeds")
        if str(args.model_family) == "sklearn_hist_gradient_boosting_by_right":
            sklearn_state, sklearn_predictions, sklearn_preview = _train_sklearn_hist_gradient_boosting_by_right(
                fit_decisions,
                calibration_decisions,
                diagnostic_decisions,
                config=config,
            )
        else:
            sklearn_state, sklearn_predictions, sklearn_preview = _train_sklearn_hist_gradient_boosting(
                fit_decisions,
                calibration_decisions,
                diagnostic_decisions,
                config=config,
            )
        prediction_members_by_split["train"].append(
            _predict_sklearn_by_right_decisions(
                sklearn_state["sklearn_models_by_right"],
                train_decisions_all,
                config=config,
            )
            if str(args.model_family) == "sklearn_hist_gradient_boosting_by_right"
            else _predict_sklearn_decisions(
                sklearn_state["sklearn_model"],
                train_decisions_all,
                config=config,
            )
        )
        prediction_members_by_split["validation"].append(
            _predict_sklearn_by_right_decisions(
                sklearn_state["sklearn_models_by_right"],
                validation_decisions,
                config=config,
            )
            if str(args.model_family) == "sklearn_hist_gradient_boosting_by_right"
            else _predict_sklearn_decisions(
                sklearn_state["sklearn_model"],
                validation_decisions,
                config=config,
            )
        )
        prediction_members_by_split["diagnostic_test"].append(sklearn_predictions["diagnostic_test"])
        calibration_predictions = sklearn_predictions["validation"]
        histories_by_seed[str(member_seeds[0])] = list(sklearn_state.get("history") or [])
    else:
        for member_seed in member_seeds:
            member_config = replace(config, seed=int(member_seed))
            model, scaler, history = train_model(
                fit_decisions,
                calibration_decisions,
                config=member_config,
            )
            histories_by_seed[str(member_seed)] = history
            trained_members.append(
                model_state_dict(
                    model=model,
                    scaler=scaler,
                    config=member_config,
                    history=history,
                    input_dim=model.net[0].in_features,
                )
            )
            mlp_runtime_members.append((model, scaler, float(member_config.target_scale)))
            prediction_members_by_split["validation"].append(
                predict_decisions(
                    model,
                    scaler,
                    validation_decisions,
                    target_scale=member_config.target_scale,
                    prediction_transform=prediction_transform,
                )
            )
            prediction_members_by_split["train"].append(
                predict_decisions(
                    model,
                    scaler,
                    train_decisions_all,
                    target_scale=member_config.target_scale,
                    prediction_transform=prediction_transform,
                )
            )
            prediction_members_by_split.setdefault("calibration", []).append(
                predict_decisions(
                    model,
                    scaler,
                    calibration_decisions,
                    target_scale=member_config.target_scale,
                    prediction_transform=prediction_transform,
                )
            )
            prediction_members_by_split["diagnostic_test"].append(
                predict_decisions(
                    model,
                    scaler,
                    diagnostic_decisions,
                    target_scale=member_config.target_scale,
                    prediction_transform=prediction_transform,
                )
            )

    if str(args.model_family) != "mlp":
        prediction_members_by_split.setdefault("calibration", []).append(calibration_predictions)
    calibration_predictions_avg = _average_predictions(prediction_members_by_split.get("calibration") or [])
    validation_predictions = _average_predictions(prediction_members_by_split["validation"])
    threshold_decisions = calibration_decisions if str(args.fit_mode) != "full_train" else validation_decisions
    threshold_predictions = (
        calibration_predictions_avg
        if str(args.fit_mode) != "full_train"
        else validation_predictions
    )

    def _predict_current_candidate(decisions_for_prediction) -> list[np.ndarray]:
        if sklearn_state is not None:
            if str(args.model_family) == "sklearn_hist_gradient_boosting_by_right":
                return _predict_sklearn_by_right_decisions(
                    sklearn_state["sklearn_models_by_right"],
                    decisions_for_prediction,
                    config=config,
                )
            return _predict_sklearn_decisions(
                sklearn_state["sklearn_model"],
                decisions_for_prediction,
                config=config,
            )
        return _average_predictions(
            [
                predict_decisions(
                    model,
                    scaler,
                    decisions_for_prediction,
                    target_scale=target_scale,
                    prediction_transform=prediction_transform,
                )
                for model, scaler, target_scale in mlp_runtime_members
            ]
        )

    jitter_predictions_by_scenario: dict[str, list[np.ndarray]] = {}
    if str(args.threshold_rule) == "jitter_stability_stressed":
        for scenario_name, scenario in VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS:
            if scenario_name == "baseline":
                continue
            jittered_decisions = jitter_decision_features(threshold_decisions, **scenario)
            jitter_predictions_by_scenario[scenario_name] = _predict_current_candidate(jittered_decisions)

    threshold, threshold_sweep = choose_threshold_with_rule(
        threshold_decisions,
        threshold_predictions,
        config=config,
        threshold_rule=str(args.threshold_rule),
        stress_per_trade=float(args.threshold_stress_per_trade),
        jitter_predictions_by_scenario=jitter_predictions_by_scenario,
    )
    train_predictions = _average_predictions(prediction_members_by_split["train"])
    diagnostic_predictions = _average_predictions(prediction_members_by_split["diagnostic_test"])
    predictions_by_split = {
        "train": (train_decisions_all, train_predictions),
        "validation": (validation_decisions, validation_predictions),
        "diagnostic_test": (diagnostic_decisions, diagnostic_predictions),
    }
    neural = {}
    for split_name, (decisions, predictions) in predictions_by_split.items():
        trades = simulate_model_policy(
            decisions,
            predictions,
            threshold=threshold,
            cooldown_minutes=config.cooldown_minutes,
            strategy="protocol101_fair_contract_neural_threshold",
            entry_filter=config.entry_filter,
            min_score_margin=config.min_score_margin,
            max_score_ceiling=config.max_score_ceiling,
            max_trades_per_session=config.max_trades_per_session,
            max_daily_loss=config.max_daily_loss,
            selection_mode=config.selection_mode,
        )
        neural[split_name] = {
            "threshold": float(threshold),
            "metrics": metrics_for_trades(trades),
            "sample_trades": [asdict(trade) for trade in trades[:10]],
        }
    baselines = {
        "validation": _baseline_metrics(validation_decisions, config),
        "diagnostic_test": _baseline_metrics(diagnostic_decisions, config),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    if sklearn_state is not None:
        torch.save(sklearn_state, args.model_out)
    elif is_ensemble:
        torch.save(
            {
                "feature_version": trained_members[0].get("feature_version"),
                "input_dim": trained_members[0].get("input_dim"),
                "config": asdict(config),
                "ensemble_format_version": 1,
                "ensemble_seeds": [int(seed) for seed in member_seeds],
                "ensemble_members": trained_members,
                "history_by_seed": histories_by_seed,
            },
            args.model_out,
        )
    else:
        torch.save(trained_members[0], args.model_out)
    return {
        "config": asdict(config),
        "experiment": {
            "target_mode": str(args.target_mode),
            "model_family": str(args.model_family),
            "fit_mode": str(args.fit_mode),
            "threshold_selection_split": (
                "calibration_train_tail"
                if str(args.fit_mode) != "full_train"
                else "validation"
            ),
            "threshold_rule": str(args.threshold_rule),
            "threshold_stress_per_trade": float(args.threshold_stress_per_trade),
            "threshold_jitter_scenarios": sorted(jitter_predictions_by_scenario),
            "positive_label_threshold": float(args.positive_label_threshold),
            "relative_target_weight": float(args.relative_target_weight),
            "teacher_events": str(args.teacher_events),
            "teacher_min_seed_count": int(args.teacher_min_seed_count),
            "entry_filter": str(args.entry_filter),
            "selection_mode": str(args.selection_mode),
            "min_score_margin": float(args.min_score_margin),
            "max_score_ceiling": max(float(args.max_score_ceiling), 0.0),
            "max_trades_per_session": int(args.max_trades_per_session),
            "max_daily_loss": float(args.max_daily_loss),
            "sample_weight_mode": str(args.sample_weight_mode),
            "feature_transform": str(args.feature_transform),
            "feature_noise_augmentation": str(args.feature_noise_augmentation),
            "ensemble_seeds": [int(seed) for seed in member_seeds] if is_ensemble else [],
            "is_ensemble": bool(is_ensemble),
            "trained_at_utc": datetime.now(UTC).isoformat(),
        },
        "split_summary": {
            name: {
                "sessions": len(plan["split_sessions"][name]),
                "decisions": int(payload["decision_count"]),
                "candidates": int(payload["candidate_count"]),
            }
            for name, payload in loaded.items()
        },
        "fit_calibration_summary": fit_calibration_summary,
        "training_preview": preview,
        "training_history": histories_by_seed if is_ensemble else histories_by_seed[str(member_seeds[0])],
        "model_family": str(args.model_family),
        "model_family_preview": sklearn_preview,
        "chosen_threshold": float(threshold),
        "threshold_sweep_validation": threshold_sweep,
        "neural": neural,
        "baselines": baselines,
        "model_out": str(args.model_out),
    }


def render_report(plan: dict[str, Any], result: dict[str, Any] | None = None) -> str:
    lines = [
        "# Protocol101 Fair-Contract Training Runner",
        "",
        "## Decision",
        "",
        f"- Status: `{plan['status']}`",
        f"- Decision: `{plan['decision']}`",
        f"- Mode: `{plan['mode']}`",
        f"- Feature contract: `{plan['selected_feature_contract']}`",
        f"- Policy: `{plan['policy_name']}`",
        f"- Model training executed: `{str(plan['model_training_executed']).lower()}`",
        f"- Threshold selection executed: `{str(plan['threshold_selection_executed']).lower()}`",
        f"- Broker endpoint called: `{str(plan['broker_endpoint_called']).lower()}`",
        f"- Paper-submit allowed: `{str(plan['paper_submit_allowed']).lower()}`",
        "",
        "## Splits",
        "",
    ]
    for split_name in ("train", "validation", "diagnostic_test"):
        lines.append(
            f"- `{split_name}`: `{len(plan['split_sessions'][split_name])}` sessions."
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in plan["blockers"]) if plan["blockers"] else lines.append("- None.")
    if result:
        lines.extend(
            [
                "",
                "## Training Result",
                "",
                f"- Model artifact: `{result.get('model_out')}`",
                f"- Chosen threshold: `{result.get('chosen_threshold')}`",
            ]
        )
        for split_name, payload in (result.get("neural") or {}).items():
            metrics = payload.get("metrics") or {}
            lines.append(
                f"- `{split_name}`: trades=`{metrics.get('trades')}`, "
                f"total_pnl=`{metrics.get('total_pnl')}`, "
                f"profit_factor=`{metrics.get('profit_factor')}`."
            )
    else:
        lines.extend(
            [
                "",
                "## Meaning",
                "",
                "- Dry-run proves the runner will use the manifest and split, but it does not train.",
                "- Actual training still requires explicit owner approval flags and a note.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    design = load_json(args.design)
    manifest_path = Path(str((design.get("allowed_data") or {}).get("canonical_manifest") or ""))
    manifest = load_json(manifest_path)
    plan = build_runner_plan(
        design=design,
        manifest=manifest,
        mode=str(args.mode),
        policy_index=int(args.policy_index),
        owner_approved_model_training=bool(args.owner_approved_model_training),
        owner_approved_threshold_selection=bool(args.owner_approved_threshold_selection),
        owner_approval_note=str(args.owner_approval_note),
    )
    result: dict[str, Any] | None = None
    exit_code = 0
    if plan["status"] == "blocked":
        exit_code = 2
    elif args.mode == "train":
        result = run_training(plan, args)
        plan = {
            **plan,
            "model_training_executed": True,
            "threshold_selection_executed": True,
            "decision": "owner_approved_training_executed_no_promotion_or_paper_submit",
        }
    write_json(args.out_dir / "runner_plan.json", plan)
    if result:
        write_json(args.out_dir / "training_result.json", result)
    (args.out_dir / "report.md").write_text(render_report(plan, result))
    print(
        json.dumps(
            {
                "status": plan["status"],
                "decision": plan["decision"],
                "mode": plan["mode"],
                "model_training_executed": plan["model_training_executed"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
