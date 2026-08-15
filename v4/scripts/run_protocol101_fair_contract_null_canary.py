"""Recompute Protocol101 fair-contract nulls and canaries.

This is an offline plumbing diagnostic. It loads only the governed split files
already resolved by the fair-contract runner plan, verifies the v2 timing and
model-facing mask conventions, and recomputes simple null/canary summaries. It
does not train a model, tune a deployable threshold, contact brokers/vendors,
change defaults, or authorize paper-submit.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.model.supervised_pilot import (
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    PilotConfig,
    DecisionCandidates,
    _VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE_INDICES,
    load_decisions,
    metrics_for_trades,
    simulate_baseline,
    simulate_model_policy,
    summarize_random_baseline,
    transform_decision_features,
)


DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/runner_plan.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_null_canary_15mo_policy1"
)
POLICY_COOLDOWN_MINUTES = {
    0: 10,
    1: 25,
    2: 45,
    3: 90,
    4: 120,
    5: 384,
    6: 384,
}
RECORDER_CONFIRMATION_SESSIONS = {"2026-06-30", "2026-07-01", "2026-07-02"}
SCHEMA_VERSION = "Protocol101FairContractNullCanaryV1"
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-index", type=int, choices=tuple(range(7)), default=1)
    parser.add_argument("--random-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path-scope",
        choices=("split_files", "cv_unique"),
        default="split_files",
        help=(
            "split_files uses the runner's readiness split; cv_unique uses the unique "
            "train/validation sessions from the governed expanding-fold scaffold."
        ),
    )
    parser.add_argument("--expected-first-decision-et", default="09:32")
    parser.add_argument("--expected-last-decision-et", default="15:30")
    parser.add_argument("--expected-full-session-rows", type=int, default=359)
    parser.add_argument(
        "--feature-transform",
        default=FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"{path} expected list, got {type(rows).__name__}")
    return [row for row in rows if isinstance(row, dict)]


def local_hhmm(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    return ts.to_pydatetime().astimezone(NY).strftime("%H:%M")


def summarize_timing(
    split_files: dict[str, list[Path]],
    *,
    expected_first: str,
    expected_last: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows_out: list[dict[str, Any]] = []
    blockers: list[str] = []
    row_counts: list[int] = []
    first_values: list[str] = []
    last_values: list[str] = []
    context_lags: list[float] = []
    quote_lags: list[float] = []
    load_errors: dict[str, str] = {}
    for split, paths in sorted(split_files.items()):
        for path in paths:
            session = path.name.removesuffix(".pkl")
            try:
                rows = read_rows(path)
            except Exception as exc:
                load_errors[f"{split}:{session}"] = f"{type(exc).__name__}: {exc}"
                continue
            decision_times = [parse_ts(row.get("decision_time")) for row in rows]
            context_times = [parse_ts(row.get("source_context_time")) for row in rows]
            quote_times = [parse_ts(row.get("source_quote_time")) for row in rows]
            valid_decisions = [ts for ts in decision_times if ts is not None]
            first = valid_decisions[0] if valid_decisions else None
            last = valid_decisions[-1] if valid_decisions else None
            first_et = local_hhmm(first)
            last_et = local_hhmm(last)
            row_count = len(rows)
            row_counts.append(row_count)
            if first_et is not None:
                first_values.append(first_et)
            if last_et is not None:
                last_values.append(last_et)
            session_context_lags: list[float] = []
            session_quote_lags: list[float] = []
            for decision, context, quote in zip(decision_times, context_times, quote_times):
                if decision is not None and context is not None:
                    lag = (decision - context).total_seconds() / 60.0
                    context_lags.append(float(lag))
                    session_context_lags.append(float(lag))
                if decision is not None and quote is not None:
                    lag = (decision - quote).total_seconds() / 60.0
                    quote_lags.append(float(lag))
                    session_quote_lags.append(float(lag))
            if row_count != expected_rows:
                blockers.append(f"unexpected_row_count:{split}:{session}:{row_count}")
            if first_et != expected_first:
                blockers.append(f"unexpected_first_decision_et:{split}:{session}:{first_et}")
            if last_et != expected_last:
                blockers.append(f"unexpected_last_decision_et:{split}:{session}:{last_et}")
            if any(abs(lag - 1.0) > 1e-9 for lag in session_context_lags):
                blockers.append(f"context_lag_not_one_minute:{split}:{session}")
            rows_out.append(
                {
                    "split": split,
                    "session": session,
                    "file": str(path),
                    "rows": row_count,
                    "first_decision_et": first_et,
                    "last_decision_et": last_et,
                    "context_lag_min": min(session_context_lags) if session_context_lags else None,
                    "context_lag_max": max(session_context_lags) if session_context_lags else None,
                    "quote_lag_median": float(np.median(session_quote_lags)) if session_quote_lags else None,
                }
            )
    for key, message in load_errors.items():
        blockers.append(f"processed_load_error:{key}:{message}")
    summary = {
        "session_count": len(rows_out),
        "row_count_min": min(row_counts) if row_counts else 0,
        "row_count_max": max(row_counts) if row_counts else 0,
        "row_count_total": int(sum(row_counts)),
        "first_decision_et_values": sorted(set(first_values)),
        "last_decision_et_values": sorted(set(last_values)),
        "context_lag_min": min(context_lags) if context_lags else None,
        "context_lag_max": max(context_lags) if context_lags else None,
        "quote_lag_median": float(np.median(quote_lags)) if quote_lags else None,
        "load_errors": load_errors,
    }
    return rows_out, summary, blockers


def split_paths_from_plan(plan: dict[str, Any], *, path_scope: str = "split_files") -> dict[str, list[Path]]:
    if str(path_scope) == "cv_unique":
        unique_paths: dict[str, Path] = {}
        for fold in ((plan.get("expanding_folds") or {}).get("fold_files") or {}).values():
            for split_name in ("train", "validation"):
                for path in fold.get(split_name) or []:
                    path_obj = Path(str(path))
                    unique_paths[path_obj.name.removesuffix(".pkl")] = path_obj
        return {"cv_unique": [unique_paths[key] for key in sorted(unique_paths)]}
    split_files = plan.get("split_files") or {}
    return {
        split: [Path(str(path)) for path in paths]
        for split, paths in split_files.items()
        if split in {"train", "validation", "diagnostic_test"}
    }


def summarize_mask(decisions_by_split: dict[str, list[DecisionCandidates]]) -> dict[str, Any]:
    transformed = {
        split: transform_decision_features(
            decisions,
            FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
        )
        for split, decisions in decisions_by_split.items()
    }
    sensitive_idx = _VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE_INDICES
    split_summary: dict[str, Any] = {}
    max_sensitive_abs = 0.0
    non_sensitive_abs_sum = 0.0
    for split, decisions in transformed.items():
        if not decisions:
            split_summary[split] = {
                "decisions": 0,
                "candidate_count": 0,
                "sensitive_max_abs": 0.0,
                "non_sensitive_abs_sum": 0.0,
            }
            continue
        features = np.vstack([decision.features for decision in decisions]).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        sensitive = features[:, sensitive_idx]
        non_sensitive_mask = np.ones(features.shape[1], dtype=bool)
        non_sensitive_mask[sensitive_idx] = False
        non_sensitive = features[:, non_sensitive_mask]
        split_sensitive_max = float(np.max(np.abs(sensitive))) if sensitive.size else 0.0
        split_non_sensitive_sum = float(np.sum(np.abs(non_sensitive))) if non_sensitive.size else 0.0
        max_sensitive_abs = max(max_sensitive_abs, split_sensitive_max)
        non_sensitive_abs_sum += split_non_sensitive_sum
        split_summary[split] = {
            "decisions": len(decisions),
            "candidate_count": int(sum(len(decision.labels) for decision in decisions)),
            "sensitive_max_abs": split_sensitive_max,
            "non_sensitive_abs_sum": split_non_sensitive_sum,
        }
    return {
        "feature_transform": FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
        "sensitive_indices": [int(item) for item in sensitive_idx.tolist()],
        "sensitive_max_abs": max_sensitive_abs,
        "non_sensitive_abs_sum": non_sensitive_abs_sum,
        "sensitive_zeroed": max_sensitive_abs <= 1e-12,
        "non_sensitive_nonzero": non_sensitive_abs_sum > 0.0,
        "splits": split_summary,
    }


def label_summary(decisions: Sequence[DecisionCandidates]) -> dict[str, Any]:
    if not decisions:
        return {
            "decisions": 0,
            "candidate_count": 0,
            "finite_labels": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "label_mean": 0.0,
            "label_std": 0.0,
        }
    labels = np.concatenate([decision.labels for decision in decisions]).astype(np.float64)
    finite = labels[np.isfinite(labels)]
    return {
        "decisions": len(decisions),
        "candidate_count": int(len(labels)),
        "finite_labels": int(len(finite)),
        "positive_labels": int(np.sum(finite > 0.0)),
        "negative_labels": int(np.sum(finite < 0.0)),
        "label_mean": float(finite.mean()) if len(finite) else 0.0,
        "label_std": float(finite.std()) if len(finite) else 0.0,
        "label_p50": float(np.median(finite)) if len(finite) else 0.0,
        "label_p95": float(np.quantile(finite, 0.95)) if len(finite) else 0.0,
    }


def deterministic_baselines(
    decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {"no_trade": metrics_for_trades([])}
    for kind in ("atm_call", "atm_put", "vwap_omar"):
        out[kind] = metrics_for_trades(
            simulate_baseline(
                decisions,
                kind=kind,
                cooldown_minutes=config.cooldown_minutes,
                seed=config.seed,
            )
        )
    out["random_valid"] = summarize_random_baseline(decisions, config=config)
    return out


def label_canaries(
    decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    if not decisions:
        return {
            "label_oracle_positive": metrics_for_trades([]),
            "inverted_label_negative": metrics_for_trades([]),
        }
    label_predictions = [decision.labels.astype(np.float32) for decision in decisions]
    inverse_predictions = [(-decision.labels).astype(np.float32) for decision in decisions]
    oracle = simulate_model_policy(
        decisions,
        label_predictions,
        threshold=0.0,
        cooldown_minutes=config.cooldown_minutes,
        strategy="label_oracle_canary_not_live_reproducible",
    )
    inverted = simulate_model_policy(
        decisions,
        inverse_predictions,
        threshold=0.0,
        cooldown_minutes=config.cooldown_minutes,
        strategy="inverted_label_canary_not_live_reproducible",
    )
    return {
        "label_oracle_positive": metrics_for_trades(oracle),
        "inverted_label_negative": metrics_for_trades(inverted),
    }


def build_blockers(
    *,
    plan: dict[str, Any],
    timing_blockers: list[str],
    mask: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    nulls: dict[str, Any],
    canaries: dict[str, Any],
    feature_transform: str,
) -> list[str]:
    blockers: list[str] = list(timing_blockers)
    if plan.get("status") not in {"dry_run_ready", "ready_to_train"}:
        blockers.append(f"runner_plan_not_ready:{plan.get('status')}")
    if plan.get("blockers"):
        blockers.append("runner_plan_has_blockers")
    if plan.get("selected_feature_contract") != FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED:
        blockers.append("unexpected_feature_contract")
    if str(feature_transform) != FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE:
        blockers.append("unexpected_requested_feature_transform")
    if plan.get("model_scoring_feature_transform") != FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE:
        blockers.append("unexpected_plan_feature_transform")
    if bool(plan.get("model_training_executed")):
        blockers.append("model_training_executed_in_runner_plan")
    if bool(plan.get("threshold_selection_executed")):
        blockers.append("threshold_selection_executed_in_runner_plan")
    if bool(plan.get("broker_endpoint_called")):
        blockers.append("broker_endpoint_called")
    if bool(plan.get("paper_submit_allowed")):
        blockers.append("paper_submit_allowed")
    split_sessions = {
        session
        for sessions in (plan.get("split_sessions") or {}).values()
        for session in sessions
    }
    forbidden = sorted(split_sessions & RECORDER_CONFIRMATION_SESSIONS)
    if forbidden:
        blockers.append(f"recorder_confirmation_session_in_split:{','.join(forbidden)}")
    protected = set(((plan.get("governance") or {}).get("protected_holdout_sessions") or []))
    protected_conflicts = sorted(split_sessions & protected)
    if protected_conflicts:
        blockers.append(f"protected_holdout_session_in_split:{','.join(protected_conflicts)}")
    if not bool(mask.get("sensitive_zeroed")):
        blockers.append("masked_sensitive_features_not_zero")
    if not bool(mask.get("non_sensitive_nonzero")):
        blockers.append("masked_non_sensitive_features_all_zero")
    for split, summary in labels.items():
        if int(summary.get("candidate_count") or 0) <= 0:
            blockers.append(f"no_candidates:{split}")
        if float(summary.get("label_std") or 0.0) <= 0.0:
            blockers.append(f"degenerate_labels:{split}")
        if int(summary.get("positive_labels") or 0) <= 0:
            blockers.append(f"no_positive_labels:{split}")
        if int(summary.get("negative_labels") or 0) <= 0:
            blockers.append(f"no_negative_labels:{split}")
    for split, summary in nulls.items():
        random_valid = summary.get("random_valid") or {}
        if int(random_valid.get("runs") or 0) < 2:
            blockers.append(f"random_null_too_few_runs:{split}")
        if not np.isfinite(float(random_valid.get("total_pnl_mean") or 0.0)):
            blockers.append(f"random_null_non_finite:{split}")
    for split, summary in canaries.items():
        oracle = summary.get("label_oracle_positive") or {}
        inverted = summary.get("inverted_label_negative") or {}
        if int(oracle.get("trades") or 0) <= 0:
            blockers.append(f"oracle_canary_no_trades:{split}")
        if float(oracle.get("total_pnl") or 0.0) <= 0.0:
            blockers.append(f"oracle_canary_non_positive:{split}")
        if int(inverted.get("trades") or 0) <= 0:
            blockers.append(f"inverted_canary_no_trades:{split}")
    return sorted(set(blockers))


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Null/Canary Diagnostic",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Feature contract: `{payload['selected_feature_contract']}`",
        f"- Feature transform: `{payload['model_scoring_feature_transform']}`",
        f"- Model training executed: `{str(payload['model_training_executed']).lower()}`",
        f"- Threshold selection executed: `{str(payload['threshold_selection_executed']).lower()}`",
        f"- Broker endpoint called: `{str(payload['broker_endpoint_called']).lower()}`",
        f"- Paper-submit allowed: `{str(payload['paper_submit_allowed']).lower()}`",
        "",
        "## Timing",
        "",
        f"- Sessions: `{payload['timing_summary']['session_count']}`",
        f"- Row count min/max: `{payload['timing_summary']['row_count_min']}` / `{payload['timing_summary']['row_count_max']}`",
        f"- First decision ET values: `{payload['timing_summary']['first_decision_et_values']}`",
        f"- Last decision ET values: `{payload['timing_summary']['last_decision_et_values']}`",
        f"- Context lag min/max: `{payload['timing_summary']['context_lag_min']}` / `{payload['timing_summary']['context_lag_max']}`",
        "",
        "## Mask",
        "",
        f"- Sensitive feature max abs: `{payload['mask_summary']['sensitive_max_abs']}`",
        f"- Non-sensitive abs sum: `{payload['mask_summary']['non_sensitive_abs_sum']}`",
        "",
        "## Nulls And Canaries",
        "",
    ]
    for split in sorted(payload["label_summary"]):
        label = payload["label_summary"].get(split) or {}
        random_valid = (payload["null_baselines"].get(split) or {}).get("random_valid") or {}
        oracle = (payload["canaries"].get(split) or {}).get("label_oracle_positive") or {}
        lines.append(
            f"- `{split}`: decisions=`{label.get('decisions')}`, candidates=`{label.get('candidate_count')}`, "
            f"label_std=`{label.get('label_std')}`, random_mean_pnl=`{random_valid.get('total_pnl_mean')}`, "
            f"oracle_trades=`{oracle.get('trades')}`, oracle_pnl=`{oracle.get('total_pnl')}`."
        )
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_json(args.runner_plan)
    split_files = split_paths_from_plan(plan, path_scope=str(args.path_scope))
    timing_rows, timing_summary, timing_blockers = summarize_timing(
        split_files,
        expected_first=str(args.expected_first_decision_et),
        expected_last=str(args.expected_last_decision_et),
        expected_rows=int(args.expected_full_session_rows),
    )
    decisions_by_split = {
        split: load_decisions(paths, policy_index=int(args.policy_index))
        for split, paths in sorted(split_files.items())
    }
    transformed_decisions_by_split = {
        split: transform_decision_features(decisions, str(args.feature_transform))
        for split, decisions in decisions_by_split.items()
    }
    mask = summarize_mask(decisions_by_split)
    label_summaries = {
        split: label_summary(decisions)
        for split, decisions in transformed_decisions_by_split.items()
    }
    config = PilotConfig(
        policy_index=int(args.policy_index),
        cooldown_minutes=POLICY_COOLDOWN_MINUTES[int(args.policy_index)],
        random_seeds=max(int(args.random_runs), 1),
        seed=int(args.seed),
        feature_transform=str(args.feature_transform),
    )
    nulls = {
        split: deterministic_baselines(decisions, config=config)
        for split, decisions in transformed_decisions_by_split.items()
    }
    canaries = {
        split: label_canaries(decisions, config=config)
        for split, decisions in transformed_decisions_by_split.items()
    }
    blockers = build_blockers(
        plan=plan,
        timing_blockers=timing_blockers,
        mask=mask,
        labels=label_summaries,
        nulls=nulls,
        canaries=canaries,
        feature_transform=str(args.feature_transform),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if not blockers else "fail",
        "decision": (
            "null_canary_plumbing_passed_no_training"
            if not blockers
            else "repair_null_canary_plumbing_before_training"
        ),
        "runner_plan": str(args.runner_plan),
        "selected_feature_contract": plan.get("selected_feature_contract"),
        "model_scoring_feature_transform": str(args.feature_transform),
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "policy_index": int(args.policy_index),
        "path_scope": str(args.path_scope),
        "config": asdict(config),
        "split_sessions": plan.get("split_sessions") or {},
        "timing_summary": timing_summary,
        "timing_rows": timing_rows,
        "mask_summary": mask,
        "label_summary": label_summaries,
        "null_baselines": nulls,
        "canaries": canaries,
        "blockers": blockers,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
