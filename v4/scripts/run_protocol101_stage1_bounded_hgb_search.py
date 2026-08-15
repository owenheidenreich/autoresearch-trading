"""Stage-1 bounded HGB fair-contract search orchestration.

This script handles offline-only Protocol101 Stage-1 bookkeeping around the
governed v2 5-fold corpus. It never contacts brokers, submits paper orders,
downloads paid data, changes defaults, edits runtime flags, or touches launchd.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from v4.model.protocol101_governed_loader import file_sha256, stable_hash
from v4.scripts.run_protocol101_fair_contract_training_runner import POLICY_META


UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_stage1_attempt001"
)
DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_design/summary.json"
)
DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/runner_plan.json"
)
DEFAULT_TRAINING_SCOPE_REGISTRY = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance/summary.json"
)
DEFAULT_FULL_ACCEPTANCE = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_acceptance/summary.json"
)
DEFAULT_PROTECTED_HOLDOUT = Path(
    "v4/audit/autoresearch/protocol101_protected_holdout/summary.json"
)
DEFAULT_GATES_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
)
DEFAULT_FEATURE_RECOVERY_PLAN = Path(
    "v4/docs/protocol101/synchronization/history/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md"
)
ATTEMPT_ID = "protocol101_stage1_attempt001_bounded_hgb_menu_v2"
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
SELECTION_SEEDS = [42, 43, 44]
CONFIRMATION_SEED = 1042
ROUND_TRIP_FEE = 3.0
FEE_SENSITIVITIES = [2.0, 5.0]
ADVERSE_STRESS_PER_TRADE = 20.0
RECORDER_PARITY_SESSIONS = {"2026-06-30", "2026-07-01", "2026-07-02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "verify-inputs",
            "evaluate-primary",
            "evaluate-conservative",
            "write-final-packet",
        ),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--training-scope-registry", type=Path, default=DEFAULT_TRAINING_SCOPE_REGISTRY)
    parser.add_argument("--full-acceptance", type=Path, default=DEFAULT_FULL_ACCEPTANCE)
    parser.add_argument("--protected-holdout", type=Path, default=DEFAULT_PROTECTED_HOLDOUT)
    parser.add_argument("--gates-doc", type=Path, default=DEFAULT_GATES_DOC)
    parser.add_argument("--feature-recovery-plan", type=Path, default=DEFAULT_FEATURE_RECOVERY_PLAN)
    parser.add_argument("--expected-first-decision-et", default="09:32")
    parser.add_argument("--expected-last-decision-et", default="15:30")
    parser.add_argument("--expected-full-session-rows", type=int, default=359)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def artifact_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
    }


def cv_fold_hash(design: dict[str, Any]) -> str:
    split_policy = design.get("split_policy") or {}
    return stable_hash(
        {
            "schema_version": split_policy.get("schema_version"),
            "required_expanding_window_cv": split_policy.get("required_expanding_window_cv"),
            "fold_count": split_policy.get("fold_count"),
            "folds": split_policy.get("folds") or [],
        }
    )


def fold_session_sets(plan: dict[str, Any]) -> dict[str, set[str]]:
    out = {"train": set(), "validation": set(), "all": set()}
    folds = (plan.get("expanding_folds") or {}).get("fold_sessions") or {}
    for fold in folds.values():
        for split in ("train", "validation"):
            sessions = {str(session) for session in fold.get(split) or []}
            out[split].update(sessions)
            out["all"].update(sessions)
    return out


def split_session_set(plan: dict[str, Any]) -> set[str]:
    return {
        str(session)
        for sessions in (plan.get("split_sessions") or {}).values()
        for session in sessions
    }


def preregistration_payload(args: argparse.Namespace) -> dict[str, Any]:
    design = load_json(args.design)
    runner_plan = load_json(args.runner_plan)
    registry = load_json(args.training_scope_registry)
    return {
        "schema_version": "Protocol101Stage1BoundedHGBPreregistrationV1",
        "status": "preregistered",
        "attempt_id": ATTEMPT_ID,
        "created_at_utc": now_utc(),
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "scope": "offline_research_training_only",
        "side_effect_guardrails": {
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_download_allowed": False,
            "promotion_or_default_change_allowed": False,
            "runtime_flag_edit_allowed": False,
            "launchd_change_allowed": False,
            "real_money_path_change_allowed": False,
        },
        "artifacts": {
            "design": artifact_ref(args.design),
            "runner_plan": artifact_ref(args.runner_plan),
            "training_scope_registry": artifact_ref(args.training_scope_registry),
            "gates_source_of_truth": artifact_ref(args.gates_doc),
            "feature_recovery_plan": artifact_ref(args.feature_recovery_plan),
        },
        "corpus": {
            "training_scope_registry_hash": str(registry.get("registry_hash") or ""),
            "session_count": int(registry.get("session_count") or 0),
            "pass_count": int(registry.get("pass_count") or 0),
            "fail_count": int(registry.get("fail_count") or 0),
            "report_only_count": int(registry.get("report_only_count") or 0),
            "first_session": (design.get("allowed_data") or {}).get("included_first_session"),
            "last_session": (design.get("allowed_data") or {}).get("included_last_session"),
            "canonical_manifest": (design.get("allowed_data") or {}).get("canonical_manifest"),
        },
        "cv_scaffold": {
            "fold_hash": cv_fold_hash(design),
            "fold_count": len(((runner_plan.get("expanding_folds") or {}).get("fold_sessions") or {})),
            "required_embargo_sessions_per_fold": 1,
            "fold_session_counts": {
                fold_id: {
                    "train": len(splits.get("train") or []),
                    "validation": len(splits.get("validation") or []),
                }
                for fold_id, splits in sorted(
                    ((runner_plan.get("expanding_folds") or {}).get("fold_sessions") or {}).items()
                )
            },
        },
        "model": {
            "family": "sklearn_hist_gradient_boosting",
            "fit_mode": "train_tail20_calibration",
            "target_mode": "return_on_premium_regression",
            "target_clip": 5.0,
            "selection_mode": "top_score",
            "threshold_rule": "max_validation_stressed_pnl",
            "threshold_selection_split": "train_tail_calibration_only",
            "feature_transform": TRANSFORM,
            "max_trades_per_session": 3,
            "policies": {
                str(policy_index): {
                    "policy_name": name,
                    "cooldown_minutes": cooldown,
                }
                for policy_index, (name, cooldown) in sorted(POLICY_META.items())
            },
            "selection_seeds": SELECTION_SEEDS,
            "confirmation_seed": CONFIRMATION_SEED,
        },
        "fees_and_stress": {
            "round_trip_fee_dollars": ROUND_TRIP_FEE,
            "fee_sensitivity_round_trips": FEE_SENSITIVITIES,
            "adverse_bid_ask_stress_per_trade_dollars": ADVERSE_STRESS_PER_TRADE,
            "stress_values_preregistered": True,
        },
        "selection": {
            "eligible_only_if_all_hard_gates_pass": True,
            "rank_eligible_candidates_by": "plain_fee_adjusted_strict_serial_net_pnl",
            "risk_adjusted_utility_allowed": False,
            "tiebreaker": "fee_adjusted_net_pnl_then_lower_drawdown_then_lower_trade_count",
        },
        "feature_recovery_policy": {
            "masked_v2_baseline_is_control_not_ceiling": True,
            "feature_addback_policy": "parity_plus_uplift_required",
            "masked_feature_groups_may_not_be_restored_because_pnl_is_weak": True,
            "raw_fields_preserved_for_tradability_fills_labels_pnl_audit": True,
            "feature_groups_requiring_ladder": [
                "internally_computed_greeks_and_iv",
                "normalized_liquidity_and_spread",
                "volume_and_open_interest_semantics",
                "raw_vendor_quote_microstructure",
            ],
        },
        "hard_gates": {
            "G1_profitability": "fee_adjusted_net_pnl_gt_0_on_at_least_4_of_5_folds_and_pooled_gt_0",
            "G2_beats_no_skill": "pooled_top_selection_pnl_z_score_gte_3_vs_matched_random_null",
            "G3_beats_heuristic": "pooled_fee_adjusted_pnl_gt_fixed_vwap_heuristic_baseline_same_folds",
            "G4_drawdown": "strict_serial_max_drawdown_lte_25pct_peak_equity_each_fold",
            "G5_seed_robustness": "at_least_3_seeds_worst_seed_satisfies_G1_and_G2_z_gte_2",
            "G6_era_guard": "no_era_with_systematically_negative_test_folds",
            "G7_frequency": "0.3_to_6.0_trades_per_day_avg_per_fold_report_3_per_day_rail",
            "G8_calibration": "pooled_oof_isotonic_ece_lte_0.10_on_payoff_score_confidence",
            "G9_confirmation": "fresh_seed_1042_independently_satisfies_G1_G2_G4",
        },
        "required_reports": [
            "$3 fee overlay plus $2/$5 sensitivity",
            "adverse bid/ask stress",
            "no ruin / no cash exhaustion",
            "feature-jitter stability",
            "side/time exposure, concentration, churn, skipped opportunity, worst day, drawdown",
            "per-shape usage across all seven menu-v2 shapes",
            "learning curve at 50/75/100 percent training fraction",
            "no broker/paper/promotional side effects",
        ],
        "stopping_criteria": {
            "fixed_exit_drawdown_failure_with_real_entry_signal": "write_stage2_learned_exits_routing_packet_do_not_soften_G4",
            "no_real_signal_after_attempt001_and_one_nearby_conservative_batch": "write_clean_failure_packet_do_not_widen_search",
            "feature_unmasking": "forbidden_without_separate_preregistered_parity_gated_ladder",
            "highest_allowed_claim": "offline candidate eligible for paper-readiness validation",
            "paper_readiness_claim_allowed": False,
        },
        "nearby_conservative_batch_if_needed": {
            "preregistered": True,
            "trigger": "no_primary_policy_has_real_signal_after_attempt001",
            "family": "sklearn_hist_gradient_boosting",
            "fit_mode": "train_tail20_calibration",
            "target_mode": "return_on_premium_regression",
            "policies": list(range(7)),
            "selection_seeds": SELECTION_SEEDS,
            "max_trades_per_session": 1,
            "threshold_rule": "max_validation_stressed_pnl",
            "no_additional_feature_groups": True,
        },
        "results_inspected_before_preregistration": False,
    }


def read_processed_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"{path} expected list, got {type(rows).__name__}")
    return [row for row in rows if isinstance(row, dict)]


def parse_utc(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def local_hhmm(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    return ts.to_pydatetime().astimezone(NY).strftime("%H:%M")


def timing_summary(
    manifest_path: Path,
    *,
    expected_first: str,
    expected_last: str,
    expected_rows: int,
) -> tuple[dict[str, Any], list[str]]:
    manifest = load_json(manifest_path)
    blockers: list[str] = []
    rows_out: list[dict[str, Any]] = []
    row_counts: list[int] = []
    first_values: set[str] = set()
    last_values: set[str] = set()
    for item in manifest.get("included_sessions") or []:
        session = str(item.get("session") or "")
        path = Path(str(item.get("processed_file") or ""))
        try:
            rows = read_processed_rows(path)
        except Exception as exc:
            blockers.append(f"processed_load_error:{session}:{type(exc).__name__}:{exc}")
            continue
        decision_times = [parse_utc(row.get("decision_time")) for row in rows]
        valid_decisions = [ts for ts in decision_times if ts is not None]
        first = local_hhmm(valid_decisions[0] if valid_decisions else None)
        last = local_hhmm(valid_decisions[-1] if valid_decisions else None)
        row_count = len(rows)
        row_counts.append(row_count)
        if first:
            first_values.add(first)
        if last:
            last_values.add(last)
        if row_count != expected_rows:
            blockers.append(f"unexpected_row_count:{session}:{row_count}")
        if first != expected_first:
            blockers.append(f"unexpected_first_decision_et:{session}:{first}")
        if last != expected_last:
            blockers.append(f"unexpected_last_decision_et:{session}:{last}")
        rows_out.append(
            {
                "session": session,
                "processed_file": str(path),
                "rows": row_count,
                "first_decision_et": first,
                "last_decision_et": last,
            }
        )
    return (
        {
            "session_count": len(rows_out),
            "row_count_min": min(row_counts) if row_counts else 0,
            "row_count_max": max(row_counts) if row_counts else 0,
            "first_decision_et_values": sorted(first_values),
            "last_decision_et_values": sorted(last_values),
            "rows": rows_out,
        },
        sorted(set(blockers)),
    )


def verify_inputs_payload(args: argparse.Namespace) -> dict[str, Any]:
    design = load_json(args.design)
    runner_plan = load_json(args.runner_plan)
    training_scope = load_json(args.training_scope_registry)
    full_acceptance = load_json(args.full_acceptance)
    protected_holdout = load_json(args.protected_holdout)
    blockers: list[str] = []
    if runner_plan.get("status") != "dry_run_ready":
        blockers.append(f"runner_plan_not_dry_run_ready:{runner_plan.get('status')}")
    if runner_plan.get("blockers"):
        blockers.append("runner_plan_has_blockers")
    if runner_plan.get("selected_feature_contract") != CONTRACT:
        blockers.append("unexpected_contract")
    if runner_plan.get("model_scoring_feature_transform") != TRANSFORM:
        blockers.append("unexpected_model_facing_transform")
    if len(((runner_plan.get("expanding_folds") or {}).get("fold_sessions") or {})) != 5:
        blockers.append("unexpected_fold_count")
    if "fold_aware_training_execution_not_implemented" in json.dumps(runner_plan):
        blockers.append("fold_aware_training_execution_not_implemented")
    if training_scope.get("status") != "pass":
        blockers.append(f"training_scope_registry_not_pass:{training_scope.get('status')}")
    if int(training_scope.get("fail_count") or 0) != 0:
        blockers.append("training_scope_has_fail_sessions")
    if int(training_scope.get("report_only_count") or 0) != 0:
        blockers.append("training_scope_has_report_only_sessions")
    fold_sets = fold_session_sets(runner_plan)
    readiness_sessions = split_session_set(runner_plan)
    used_sessions = set(fold_sets["all"]) | readiness_sessions
    report_or_fail_sessions = {
        str(row.get("session"))
        for row in full_acceptance.get("sessions") or []
        if str(row.get("status") or "") != "pass"
    }
    report_fail_conflicts = sorted(used_sessions & report_or_fail_sessions)
    if report_fail_conflicts:
        blockers.append(f"report_or_fail_session_used:{','.join(report_fail_conflicts)}")
    protected_sessions = {str(session) for session in protected_holdout.get("sessions") or []}
    protected_conflicts = sorted(used_sessions & protected_sessions)
    if protected_conflicts:
        blockers.append(f"protected_holdout_session_used:{','.join(protected_conflicts)}")
    expected_protected_window = [
        session for session in sorted(protected_sessions)
        if "2025-05-16" <= session <= "2025-06-30"
    ]
    if len(expected_protected_window) != 30:
        blockers.append(f"unexpected_protected_holdout_window_count:{len(expected_protected_window)}")
    recorder_conflicts = sorted(used_sessions & RECORDER_PARITY_SESSIONS)
    if recorder_conflicts:
        blockers.append(f"recorder_parity_session_used:{','.join(recorder_conflicts)}")
    manifest_path = Path(str((design.get("allowed_data") or {}).get("canonical_manifest") or ""))
    timing, timing_blockers = timing_summary(
        manifest_path,
        expected_first=str(args.expected_first_decision_et),
        expected_last=str(args.expected_last_decision_et),
        expected_rows=int(args.expected_full_session_rows),
    )
    blockers.extend(timing_blockers)
    return {
        "schema_version": "Protocol101Stage1InputReadinessV1",
        "generated_at_utc": now_utc(),
        "status": "pass" if not blockers else "fail",
        "decision": "stage1_inputs_ready" if not blockers else "repair_stage1_inputs_before_training",
        "attempt_id": ATTEMPT_ID,
        "blockers": sorted(set(blockers)),
        "contract": runner_plan.get("selected_feature_contract"),
        "model_facing_transform": runner_plan.get("model_scoring_feature_transform"),
        "runner_plan_status": runner_plan.get("status"),
        "runner_plan_blockers": runner_plan.get("blockers") or [],
        "training_scope_registry": {
            "path": str(args.training_scope_registry),
            "status": training_scope.get("status"),
            "registry_hash": training_scope.get("registry_hash"),
            "session_count": training_scope.get("session_count"),
            "pass_count": training_scope.get("pass_count"),
            "fail_count": training_scope.get("fail_count"),
            "report_only_count": training_scope.get("report_only_count"),
        },
        "cv_scaffold": {
            "fold_hash": cv_fold_hash(design),
            "fold_count": len(((runner_plan.get("expanding_folds") or {}).get("fold_sessions") or {})),
            "fold_train_union_count": len(fold_sets["train"]),
            "fold_validation_union_count": len(fold_sets["validation"]),
            "fold_train_validation_union_count": len(fold_sets["all"]),
        },
        "excluded_session_checks": {
            "report_or_fail_sessions_in_used_scope": report_fail_conflicts,
            "protected_holdout_sessions_in_used_scope": protected_conflicts,
            "protected_holdout_window_count_2025_05_16_to_2025_06_30": len(expected_protected_window),
            "recorder_parity_sessions_in_used_scope": recorder_conflicts,
        },
        "timing_summary": {
            key: value for key, value in timing.items() if key != "rows"
        },
        "timing_rows_path": str(args.out_dir / "input_timing_rows.json"),
        "side_effect_flags": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
        },
    }, timing


def render_preregistration(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Protocol101 Stage-1 Attempt 001 Preregistration",
            "",
            f"- Attempt: `{payload['attempt_id']}`",
            f"- Status: `{payload['status']}`",
            f"- Contract: `{payload['contract']}`",
            f"- Transform: `{payload['model_facing_transform']}`",
            f"- Model family: `{payload['model']['family']}`",
            f"- Policies: `{sorted(payload['model']['policies'])}`",
            f"- Selection seeds: `{payload['model']['selection_seeds']}`",
            f"- Confirmation seed: `{payload['model']['confirmation_seed']}`",
            f"- Selection metric: `{payload['selection']['rank_eligible_candidates_by']}`",
            f"- Risk utility allowed: `{payload['selection']['risk_adjusted_utility_allowed']}`",
            f"- Feature add-back policy: `{payload['feature_recovery_policy']['feature_addback_policy']}`",
            "",
            "This artifact was written before Stage-1 model results were inspected.",
        ]
    ) + "\n"


def render_input_readiness(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Stage-1 Input Readiness",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Contract: `{payload['contract']}`",
        f"- Transform: `{payload['model_facing_transform']}`",
        f"- Fold count: `{payload['cv_scaffold']['fold_count']}`",
        f"- Protected holdout conflicts: `{payload['excluded_session_checks']['protected_holdout_sessions_in_used_scope']}`",
        f"- Report/fail conflicts: `{payload['excluded_session_checks']['report_or_fail_sessions_in_used_scope']}`",
        f"- Recorder/parity conflicts: `{payload['excluded_session_checks']['recorder_parity_sessions_in_used_scope']}`",
        f"- Row count min/max: `{payload['timing_summary']['row_count_min']}` / `{payload['timing_summary']['row_count_max']}`",
        f"- First decision ET values: `{payload['timing_summary']['first_decision_et_values']}`",
        f"- Last decision ET values: `{payload['timing_summary']['last_decision_et_values']}`",
    ]
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    return "\n".join(lines) + "\n"


def adjusted_trade_pnls(trades: list[dict[str, Any]], *, fee: float = ROUND_TRIP_FEE) -> list[float]:
    return [float(trade.get("pnl") or 0.0) - float(fee) for trade in trades]


def max_drawdown_pct_of_peak(pnls: list[float], *, starting_cash: float = 10_000.0) -> tuple[float, float, float]:
    equity = float(starting_cash)
    peak = float(starting_cash)
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    min_equity = float(starting_cash)
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        min_equity = min(min_equity, equity)
        dd_abs = max(peak - equity, 0.0)
        dd_pct = dd_abs / peak if peak > 0.0 else float("inf")
        max_dd_abs = max(max_dd_abs, dd_abs)
        max_dd_pct = max(max_dd_pct, dd_pct)
    return float(max_dd_abs), float(max_dd_pct), float(min_equity)


def fold_fee_metrics(
    trades: list[dict[str, Any]],
    *,
    validation_sessions: int,
    fee: float = ROUND_TRIP_FEE,
) -> dict[str, Any]:
    pnls = adjusted_trade_pnls(trades, fee=fee)
    total = float(sum(pnls))
    dd_abs, dd_pct, min_equity = max_drawdown_pct_of_peak(pnls)
    by_day: dict[str, float] = defaultdict(float)
    side_counts: Counter[str] = Counter()
    hour_counts: Counter[str] = Counter()
    offsets: list[float] = []
    wins = 0
    for trade, pnl in zip(trades, pnls):
        session = str(trade.get("session") or "")
        by_day[session] += float(pnl)
        side_counts[str(trade.get("right") or "UNKNOWN")] += 1
        try:
            hour_counts[str(pd.Timestamp(trade.get("decision_time")).tz_convert(NY).strftime("%H:%M")[:2])] += 1
        except Exception:
            hour_counts["UNKNOWN"] += 1
        try:
            offsets.append(float(trade.get("offset") or 0.0))
        except (TypeError, ValueError):
            pass
        wins += int(float(pnl) > 0.0)
    sessions = max(int(validation_sessions), 1)
    gross_win = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0.0))
    return {
        "trades": len(trades),
        "total_pnl": total,
        "avg_pnl": total / len(pnls) if pnls else 0.0,
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_win > 0 else 0.0),
        "max_drawdown": -float(dd_abs),
        "max_drawdown_pct_of_peak": float(dd_pct),
        "min_equity": float(min_equity),
        "no_ruin": bool(min_equity > 0.0),
        "trades_per_day": float(len(trades) / sessions),
        "sessions": int(validation_sessions),
        "worst_day_pnl": min(by_day.values()) if by_day else 0.0,
        "side_counts": dict(sorted(side_counts.items())),
        "hour_counts": dict(sorted(hour_counts.items())),
        "offset_mean": float(sum(offsets) / len(offsets)) if offsets else 0.0,
    }


def fee_adjusted_baseline_metric(metric: dict[str, Any], *, fee: float = ROUND_TRIP_FEE) -> float:
    return float(metric.get("total_pnl") or 0.0) - float(fee) * float(metric.get("trades") or 0.0)


def fee_adjusted_random_mean(metric: dict[str, Any], *, fee: float = ROUND_TRIP_FEE) -> float:
    return float(metric.get("total_pnl_mean") or 0.0) - float(fee) * float(metric.get("trades_mean") or 0.0)


def ece_from_trade_scores(trades: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    scored = [
        (float(trade.get("score")), 1.0 if float(trade.get("pnl") or 0.0) > 0.0 else 0.0)
        for trade in trades
        if trade.get("score") is not None and math.isfinite(float(trade.get("score")))
    ]
    if len(scored) < 10:
        return {"ece": 1.0, "samples": len(scored), "method": "insufficient_trade_scores"}
    scores = [item[0] for item in scored]
    wins = [item[1] for item in scored]
    try:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        probs = list(model.fit_transform(scores, wins))
        method = "pooled_oof_isotonic_trade_score_in_sample_map"
    except Exception:
        lo = min(scores)
        hi = max(scores)
        span = hi - lo if hi > lo else 1.0
        probs = [(score - lo) / span for score in scores]
        method = "minmax_score_fallback"
    order = sorted(range(len(probs)), key=lambda idx: probs[idx])
    bucket_size = max(len(order) // int(bins), 1)
    weighted_error = 0.0
    bucket_rows = []
    for start in range(0, len(order), bucket_size):
        idxs = order[start : start + bucket_size]
        if not idxs:
            continue
        avg_prob = sum(probs[idx] for idx in idxs) / len(idxs)
        avg_win = sum(wins[idx] for idx in idxs) / len(idxs)
        err = abs(avg_prob - avg_win)
        weighted_error += err * len(idxs) / len(order)
        bucket_rows.append({"count": len(idxs), "avg_probability": avg_prob, "win_rate": avg_win, "abs_error": err})
    return {"ece": float(weighted_error), "samples": len(scored), "method": method, "buckets": bucket_rows}


def attempt_identity(path: Path) -> tuple[str, int, int]:
    match = re.search(r"(primary|conservative)_policy(\d+)_seed(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot parse attempt identity from {path}")
    return str(match.group(1)), int(match.group(2)), int(match.group(3))


def era_map() -> dict[str, str]:
    path = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
    if not path.exists():
        return {}
    payload = load_json(path)
    return {str(row.get("session")): str(row.get("era") or "UNKNOWN") for row in payload.get("sessions") or []}


def evaluate_attempt(path: Path, *, eras: dict[str, str]) -> dict[str, Any]:
    batch, policy, seed = attempt_identity(path)
    result = load_json(path)
    runner_plan_path = path.parent / "runner_plan.json"
    runner_plan = load_json(runner_plan_path) if runner_plan_path.exists() else {}
    fold_rows = []
    pooled_trades: list[dict[str, Any]] = []
    pooled_fee_pnl = 0.0
    pooled_stress_pnl = 0.0
    pooled_random_mean = 0.0
    pooled_random_var = 0.0
    pooled_heuristic = 0.0
    era_fold_pnls: dict[str, list[float]] = defaultdict(list)
    fee_sensitivity = {str(fee): 0.0 for fee in [ROUND_TRIP_FEE, *FEE_SENSITIVITIES]}
    for fold_id, fold in sorted((result.get("fold_results") or {}).items()):
        validation = ((fold.get("neural") or {}).get("validation") or {})
        trades = list(validation.get("trades") or validation.get("sample_trades") or [])
        sessions = int((((fold.get("split_summary") or {}).get("validation") or {}).get("sessions") or 0))
        fee_metrics = fold_fee_metrics(trades, validation_sessions=sessions, fee=ROUND_TRIP_FEE)
        stress_metrics = fold_fee_metrics(trades, validation_sessions=sessions, fee=ROUND_TRIP_FEE + ADVERSE_STRESS_PER_TRADE)
        pooled_trades.extend(trades)
        pooled_fee_pnl += float(fee_metrics["total_pnl"])
        pooled_stress_pnl += float(stress_metrics["total_pnl"])
        for fee in fee_sensitivity:
            fee_sensitivity[fee] += float(fold_fee_metrics(trades, validation_sessions=sessions, fee=float(fee))["total_pnl"])
        baselines = ((fold.get("baselines") or {}).get("validation") or {})
        random_valid = baselines.get("random_valid") or {}
        pooled_random_mean += fee_adjusted_random_mean(random_valid)
        pooled_random_var += float(random_valid.get("total_pnl_std") or 0.0) ** 2
        heuristic_candidates = [
            fee_adjusted_baseline_metric(baselines.get(kind) or {})
            for kind in ("vwap_omar", "atm_call", "atm_put")
        ]
        pooled_heuristic += max(heuristic_candidates) if heuristic_candidates else 0.0
        for trade, pnl in zip(trades, adjusted_trade_pnls(trades, fee=ROUND_TRIP_FEE)):
            era = eras.get(str(trade.get("session") or ""), "UNKNOWN")
            era_fold_pnls[era].append(float(pnl))
        fold_rows.append(
            {
                "fold_id": fold_id,
                "threshold": fold.get("chosen_threshold"),
                "fee_metrics": fee_metrics,
                "stress_metrics": stress_metrics,
                "random_valid": random_valid,
                "heuristic_best_fee_adjusted_pnl": max(heuristic_candidates) if heuristic_candidates else 0.0,
            }
        )
    random_std = math.sqrt(pooled_random_var)
    z_score = (pooled_fee_pnl - pooled_random_mean) / random_std if random_std > 1e-9 else (float("inf") if pooled_fee_pnl > pooled_random_mean else 0.0)
    folds_positive = sum(1 for row in fold_rows if float(row["fee_metrics"]["total_pnl"]) > 0.0)
    drawdown_max = max((float(row["fee_metrics"]["max_drawdown_pct_of_peak"]) for row in fold_rows), default=0.0)
    frequency_values = [float(row["fee_metrics"]["trades_per_day"]) for row in fold_rows]
    ece = ece_from_trade_scores(pooled_trades)
    era_summary = {
        era: {
            "trade_count": len(values),
            "total_pnl": float(sum(values)),
            "median_trade_pnl": float(pd.Series(values).median()) if values else 0.0,
        }
        for era, values in sorted(era_fold_pnls.items())
    }
    negative_eras = [era for era, row in era_summary.items() if float(row["total_pnl"]) < 0.0]
    gates = {
        "G1_profitability": {
            "pass": bool(folds_positive >= 4 and pooled_fee_pnl > 0.0),
            "positive_folds": folds_positive,
            "pooled_fee_adjusted_pnl": pooled_fee_pnl,
        },
        "G2_beats_no_skill": {
            "pass": bool(z_score >= 3.0),
            "z_score": float(z_score),
            "random_fee_adjusted_mean": pooled_random_mean,
            "random_std": random_std,
        },
        "G3_beats_heuristic": {
            "pass": bool(pooled_fee_pnl > pooled_heuristic),
            "heuristic_fee_adjusted_pnl": pooled_heuristic,
        },
        "G4_drawdown": {
            "pass": bool(drawdown_max <= 0.25),
            "owner_review_stage2_zone": bool(0.25 < drawdown_max <= 0.35),
            "hard_fail": bool(drawdown_max > 0.35),
            "max_drawdown_pct_of_peak": drawdown_max,
        },
        "G6_era_guard": {
            "pass": not negative_eras,
            "negative_eras": negative_eras,
        },
        "G7_frequency": {
            "pass": bool(frequency_values and min(frequency_values) >= 0.3 and max(frequency_values) <= 6.0),
            "min_trades_per_day": min(frequency_values) if frequency_values else 0.0,
            "max_trades_per_day": max(frequency_values) if frequency_values else 0.0,
            "conservative_rail_3_per_day_all_folds": bool(frequency_values and max(frequency_values) <= 3.0),
        },
        "G8_calibration": {
            "pass": bool(float(ece.get("ece") or 1.0) <= 0.10),
            **ece,
        },
    }
    return {
        "batch": batch,
        "policy_index": policy,
        "policy_name": POLICY_META[policy][0],
        "seed": seed,
        "training_result": str(path),
        "runner_plan": str(runner_plan_path),
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "pooled": {
            "fee_adjusted_pnl": pooled_fee_pnl,
            "stress_fee_adjusted_pnl": pooled_stress_pnl,
            "fee_sensitivity_pnl": fee_sensitivity,
            "trades": len(pooled_trades),
            "random_z_score": float(z_score),
            "heuristic_fee_adjusted_pnl": pooled_heuristic,
            "no_ruin_all_folds": all(row["fee_metrics"]["no_ruin"] for row in fold_rows),
            "worst_day_pnl": min((row["fee_metrics"]["worst_day_pnl"] for row in fold_rows), default=0.0),
            "side_counts": dict(sum((Counter(row["fee_metrics"]["side_counts"]) for row in fold_rows), Counter())),
            "hour_counts": dict(sum((Counter(row["fee_metrics"]["hour_counts"]) for row in fold_rows), Counter())),
        },
        "era_summary": era_summary,
        "gates": gates,
        "side_effect_flags": {
            "model_training_executed": bool(runner_plan.get("model_training_executed")),
            "threshold_selection_executed": bool(runner_plan.get("threshold_selection_executed")),
            "broker_endpoint_called": bool(runner_plan.get("broker_endpoint_called")),
            "paper_submit_allowed": bool(runner_plan.get("paper_submit_allowed")),
        },
    }


def batch_from_mode(mode: str) -> str:
    if mode == "evaluate-conservative":
        return "conservative"
    return "primary"


def expected_attempt_ids(batch: str) -> list[str]:
    return [
        f"{batch}_policy{policy}_seed{seed}"
        for policy in range(7)
        for seed in SELECTION_SEEDS
    ]


def evaluate_batch_payload(args: argparse.Namespace, *, batch: str) -> dict[str, Any]:
    eras = era_map()
    attempt_paths = sorted(args.out_dir.glob(f"attempts/{batch}_policy*_seed*/training_runner/training_result.json"))
    found_attempt_ids = {
        Path(path).parts[-3]
        for path in attempt_paths
    }
    missing_attempt_ids = sorted(set(expected_attempt_ids(batch)) - found_attempt_ids)
    attempt_rows = [evaluate_attempt(path, eras=eras) for path in attempt_paths]
    by_policy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        by_policy[int(row["policy_index"])].append(row)
    policy_rows = []
    for policy, rows in sorted(by_policy.items()):
        rows_sorted = sorted(rows, key=lambda item: int(item["seed"]))
        seed_gate_pairs = [
            bool(item["gates"]["G1_profitability"]["pass"])
            and float(item["gates"]["G2_beats_no_skill"]["z_score"]) >= 2.0
            for item in rows_sorted
        ]
        all_hard_gates_except_g9 = all(
            all(
                bool(item["gates"][gate]["pass"])
                for gate in (
                    "G1_profitability",
                    "G2_beats_no_skill",
                    "G3_beats_heuristic",
                    "G4_drawdown",
                    "G6_era_guard",
                    "G7_frequency",
                    "G8_calibration",
                )
            )
            for item in rows_sorted
        )
        policy_rows.append(
            {
                "policy_index": policy,
                "policy_name": POLICY_META[policy][0],
                "seed_count": len(rows_sorted),
                "G5_seed_robustness": {
                    "pass": bool(len(rows_sorted) >= 3 and all(seed_gate_pairs)),
                    "seed_results": [
                        {
                            "seed": item["seed"],
                            "G1": item["gates"]["G1_profitability"]["pass"],
                            "G2_z": item["gates"]["G2_beats_no_skill"]["z_score"],
                            "G2_z_gte_2": item["gates"]["G2_beats_no_skill"]["z_score"] >= 2.0,
                        }
                        for item in rows_sorted
                    ],
                },
                "eligible_before_G9": bool(all_hard_gates_except_g9 and len(rows_sorted) >= 3 and all(seed_gate_pairs)),
                "selection_metric_fee_adjusted_pnl_mean": float(
                    sum(item["pooled"]["fee_adjusted_pnl"] for item in rows_sorted) / len(rows_sorted)
                )
                if rows_sorted
                else 0.0,
                "best_seed_fee_adjusted_pnl": max((item["pooled"]["fee_adjusted_pnl"] for item in rows_sorted), default=0.0),
                "worst_seed_fee_adjusted_pnl": min((item["pooled"]["fee_adjusted_pnl"] for item in rows_sorted), default=0.0),
            }
        )
    eligible = [row for row in policy_rows if row["eligible_before_G9"]]
    any_real_signal = any(
        bool(row["gates"]["G1_profitability"]["pass"]) and bool(row["gates"]["G2_beats_no_skill"]["pass"])
        for row in attempt_rows
    )
    any_drawdown_stage2_zone = any(
        bool(row["gates"]["G4_drawdown"]["owner_review_stage2_zone"])
        for row in attempt_rows
    )
    any_drawdown_hard_fail = any(
        bool(row["gates"]["G4_drawdown"]["hard_fail"])
        for row in attempt_rows
    )
    blockers: list[str] = []
    if missing_attempt_ids:
        blockers.append(f"missing_training_results:{','.join(missing_attempt_ids)}")
    if any(row["side_effect_flags"]["broker_endpoint_called"] for row in attempt_rows):
        blockers.append("broker_endpoint_called")
    if any(row["side_effect_flags"]["paper_submit_allowed"] for row in attempt_rows):
        blockers.append("paper_submit_allowed")
    routing = (
        "repair_or_run_missing_batch_attempts"
        if blockers
        else "run_G9_confirmation_for_best_policy"
        if eligible
        else (
            "run_preregistered_nearby_conservative_batch"
            if batch == "primary"
            else "write_clean_failure_packet"
        )
        if not any_real_signal
        else "write_stage2_learned_exits_routing_packet_or_owner_review"
    )
    best_policy = (
        max(eligible, key=lambda item: item["selection_metric_fee_adjusted_pnl_mean"])
        if eligible
        else None
    )
    payload = {
        "schema_version": "Protocol101Stage1BatchEvaluationV1",
        "generated_at_utc": now_utc(),
        "attempt_id": ATTEMPT_ID,
        "batch": batch,
        "status": "blocked" if blockers else "pass" if eligible else "fail",
        "routing_decision": routing,
        "blockers": blockers,
        "best_policy_before_G9": best_policy,
        "expected_attempt_count": len(expected_attempt_ids(batch)),
        "missing_attempt_ids": missing_attempt_ids,
        "attempt_count": len(attempt_rows),
        "policy_count": len(policy_rows),
        "selection_metric": "plain_fee_adjusted_strict_serial_net_pnl_after_hard_gates",
        "round_trip_fee_dollars": ROUND_TRIP_FEE,
        "fee_sensitivities": FEE_SENSITIVITIES,
        "adverse_stress_per_trade": ADVERSE_STRESS_PER_TRADE,
        "attempts": attempt_rows,
        "policies": policy_rows,
        "real_signal_observed": bool(any_real_signal),
        "drawdown_stage2_zone_observed": bool(any_drawdown_stage2_zone),
        "drawdown_hard_fail_observed": bool(any_drawdown_hard_fail),
        "gate_counts": {
            gate: sum(1 for row in attempt_rows if bool(row["gates"].get(gate, {}).get("pass")))
            for gate in (
                "G1_profitability",
                "G2_beats_no_skill",
                "G3_beats_heuristic",
                "G4_drawdown",
                "G6_era_guard",
                "G7_frequency",
                "G8_calibration",
            )
        },
        "side_effect_flags": {
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "model_training_artifacts_seen": any(
                row["side_effect_flags"]["model_training_executed"]
                for row in attempt_rows
            ),
            "threshold_selection_artifacts_seen": any(
                row["side_effect_flags"]["threshold_selection_executed"]
                for row in attempt_rows
            ),
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    return payload


def render_batch_evaluation(payload: dict[str, Any]) -> str:
    batch_title = str(payload.get("batch") or "primary").title()
    lines = [
        f"# Protocol101 Stage-1 {batch_title} Evaluation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Routing decision: `{payload['routing_decision']}`",
        f"- Blockers: `{payload.get('blockers') or []}`",
        f"- Attempts evaluated: `{payload['attempt_count']}`",
        f"- Expected attempts: `{payload.get('expected_attempt_count')}`",
        f"- Selection metric: `{payload['selection_metric']}`",
        f"- Real signal observed: `{payload.get('real_signal_observed')}`",
        "",
        "## Policy Summary",
        "",
        "| Policy | Eligible before G9 | Mean fee PnL | Worst seed fee PnL | G5 |",
        "|---:|---|---:|---:|---|",
    ]
    for row in payload["policies"]:
        lines.append(
            f"| {row['policy_index']} | `{row['eligible_before_G9']}` | "
            f"{row['selection_metric_fee_adjusted_pnl_mean']:.2f} | "
            f"{row['worst_seed_fee_adjusted_pnl']:.2f} | "
            f"`{row['G5_seed_robustness']['pass']}` |"
        )
    lines.extend(["", "## Seed Gate Snapshot", ""])
    for row in payload["attempts"]:
        gates = row["gates"]
        lines.append(
            f"- policy `{row['policy_index']}` seed `{row['seed']}`: "
            f"fee_pnl=`{row['pooled']['fee_adjusted_pnl']:.2f}`, "
            f"G1=`{gates['G1_profitability']['pass']}`, "
            f"G2_z=`{gates['G2_beats_no_skill']['z_score']:.3f}`, "
            f"G3=`{gates['G3_beats_heuristic']['pass']}`, "
            f"G4_dd_peak=`{gates['G4_drawdown']['max_drawdown_pct_of_peak']:.3f}`, "
            f"G7=`{gates['G7_frequency']['pass']}`, "
            f"G8_ece=`{gates['G8_calibration']['ece']:.3f}`."
        )
    return "\n".join(lines) + "\n"


def policy_usage_distribution(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    by_policy: dict[int, dict[str, Any]] = {}
    for attempt in evaluation.get("attempts") or []:
        policy = int(attempt.get("policy_index"))
        row = by_policy.setdefault(
            policy,
            {
                "policy_index": policy,
                "policy_name": attempt.get("policy_name"),
                "seed_count": 0,
                "total_trades": 0,
                "fee_adjusted_pnl": 0.0,
                "stress_fee_adjusted_pnl": 0.0,
                "side_counts": Counter(),
                "hour_counts": Counter(),
                "best_seed_fee_adjusted_pnl": None,
                "worst_seed_fee_adjusted_pnl": None,
            },
        )
        pooled = attempt.get("pooled") or {}
        fee_pnl = float(pooled.get("fee_adjusted_pnl") or 0.0)
        row["seed_count"] += 1
        row["total_trades"] += int(pooled.get("trades") or 0)
        row["fee_adjusted_pnl"] += fee_pnl
        row["stress_fee_adjusted_pnl"] += float(pooled.get("stress_fee_adjusted_pnl") or 0.0)
        row["side_counts"].update(Counter(pooled.get("side_counts") or {}))
        row["hour_counts"].update(Counter(pooled.get("hour_counts") or {}))
        row["best_seed_fee_adjusted_pnl"] = (
            fee_pnl
            if row["best_seed_fee_adjusted_pnl"] is None
            else max(float(row["best_seed_fee_adjusted_pnl"]), fee_pnl)
        )
        row["worst_seed_fee_adjusted_pnl"] = (
            fee_pnl
            if row["worst_seed_fee_adjusted_pnl"] is None
            else min(float(row["worst_seed_fee_adjusted_pnl"]), fee_pnl)
        )
    out = []
    for row in by_policy.values():
        seed_count = max(int(row["seed_count"]), 1)
        out.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"side_counts", "hour_counts"}
                },
                "mean_fee_adjusted_pnl": float(row["fee_adjusted_pnl"]) / seed_count,
                "mean_trades_per_seed": float(row["total_trades"]) / seed_count,
                "side_counts": dict(sorted(row["side_counts"].items())),
                "hour_counts": dict(sorted(row["hour_counts"].items())),
            }
        )
    return sorted(out, key=lambda item: int(item["policy_index"]))


def final_packet_payload(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = load_json(args.out_dir / "preregistration.json")
    input_readiness = load_json(args.out_dir / "input_readiness.json")
    primary = load_json(args.out_dir / "primary_evaluation.json")
    conservative = load_json(args.out_dir / "conservative_evaluation.json")
    no_real_signal = (
        primary.get("real_signal_observed") is False
        and conservative.get("real_signal_observed") is False
    )
    final_routing = (
        "failure_packet_route_feature_recovery_ladder"
        if no_real_signal
        else "stage2_or_eligibility_review_required"
    )
    side_effect_summary = {
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_download": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "real_money_path_changed": False,
    }
    return {
        "schema_version": "Protocol101Stage1FinalPacketV1",
        "generated_at_utc": now_utc(),
        "attempt_id": ATTEMPT_ID,
        "status": "fail" if no_real_signal else "review_required",
        "final_routing_decision": final_routing,
        "highest_allowed_claim": "no_stage1_candidate_eligible_for_paper_readiness_validation",
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "corpus": preregistration.get("corpus") or {},
        "cv_scaffold": preregistration.get("cv_scaffold") or {},
        "artifacts": {
            "preregistration": artifact_ref(args.out_dir / "preregistration.json"),
            "input_readiness": artifact_ref(args.out_dir / "input_readiness.json"),
            "primary_evaluation": artifact_ref(args.out_dir / "primary_evaluation.json"),
            "conservative_evaluation": artifact_ref(args.out_dir / "conservative_evaluation.json"),
        },
        "input_readiness": {
            "status": input_readiness.get("status"),
            "blockers": input_readiness.get("blockers") or [],
            "excluded_session_checks": input_readiness.get("excluded_session_checks") or {},
            "timing_summary": input_readiness.get("timing_summary") or {},
        },
        "batch_results": {
            "primary": {
                "status": primary.get("status"),
                "routing_decision": primary.get("routing_decision"),
                "attempt_count": primary.get("attempt_count"),
                "policy_count": primary.get("policy_count"),
                "real_signal_observed": primary.get("real_signal_observed"),
                "gate_counts": primary.get("gate_counts") or {},
                "per_shape_usage": policy_usage_distribution(primary),
            },
            "conservative": {
                "status": conservative.get("status"),
                "routing_decision": conservative.get("routing_decision"),
                "attempt_count": conservative.get("attempt_count"),
                "policy_count": conservative.get("policy_count"),
                "real_signal_observed": conservative.get("real_signal_observed"),
                "gate_counts": conservative.get("gate_counts") or {},
                "per_shape_usage": policy_usage_distribution(conservative),
            },
        },
        "gate_decisions": {
            "G1_to_G8": {
                "primary_pass_counts": primary.get("gate_counts") or {},
                "conservative_pass_counts": conservative.get("gate_counts") or {},
                "attempt_denominator_each_batch": 21,
            },
            "G5_seed_robustness": {
                "primary_policy_pass_count": sum(
                    1 for row in primary.get("policies") or []
                    if bool((row.get("G5_seed_robustness") or {}).get("pass"))
                ),
                "conservative_policy_pass_count": sum(
                    1 for row in conservative.get("policies") or []
                    if bool((row.get("G5_seed_robustness") or {}).get("pass"))
                ),
            },
            "G9_confirmation": {
                "status": "not_run_no_eligible_candidate",
                "reason": "No policy passed all pre-G9 hard gates in primary or conservative batch.",
            },
        },
        "required_report_gaps": {
            "feature_jitter_stability": "not_run_no_selected_candidate",
            "learning_curve_50_75_100": "not_implemented_in_stage1_batch_evaluator",
        },
        "stopping_rule": {
            "triggered": bool(no_real_signal),
            "rule": "no_real_signal_after_attempt001_and_one_preregistered_nearby_conservative_batch",
            "action": "write_clean_failure_packet_do_not_widen_search",
            "next_route": "feature_recovery_ladder",
        },
        "feature_policy": {
            "unmasking_performed": False,
            "unmasking_allowed_because_pnl_weak": False,
            "next_feature_work_requires": "separate_preregistered_parity_gated_feature_recovery_ladder",
        },
        "side_effect_summary": side_effect_summary,
    }


def render_final_packet(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Stage-1 Final Failure Packet",
        "",
        f"- Status: `{payload['status']}`",
        f"- Final routing: `{payload['final_routing_decision']}`",
        f"- Contract: `{payload['contract']}`",
        f"- Transform: `{payload['model_facing_transform']}`",
        f"- Training scope hash: `{payload['corpus'].get('training_scope_registry_hash')}`",
        f"- CV fold hash: `{payload['cv_scaffold'].get('fold_hash')}`",
        "",
        "## Batch Results",
        "",
    ]
    for batch_name, batch in payload["batch_results"].items():
        lines.extend(
            [
                f"### {batch_name.title()}",
                "",
                f"- Status: `{batch['status']}`",
                f"- Routing: `{batch['routing_decision']}`",
                f"- Attempts: `{batch['attempt_count']}`",
                f"- Real signal observed: `{batch['real_signal_observed']}`",
                f"- Gate pass counts: `{batch['gate_counts']}`",
                "",
                "| Policy | Mean Fee PnL | Worst Seed Fee PnL | Total Trades | Side Counts |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in batch["per_shape_usage"]:
            lines.append(
                f"| {row['policy_index']} | {float(row['mean_fee_adjusted_pnl']):.2f} | "
                f"{float(row['worst_seed_fee_adjusted_pnl']):.2f} | "
                f"{int(row['total_trades'])} | `{row['side_counts']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate Outcomes",
            "",
            f"- G1-G8 counts: `{payload['gate_decisions']['G1_to_G8']}`",
            f"- G5 seed robustness: `{payload['gate_decisions']['G5_seed_robustness']}`",
            f"- G9 confirmation: `{payload['gate_decisions']['G9_confirmation']}`",
            "",
            "## Required Diagnostics",
            "",
            f"- Feature-jitter stability: `{payload['required_report_gaps']['feature_jitter_stability']}`",
            f"- Learning curve 50/75/100: `{payload['required_report_gaps']['learning_curve_50_75_100']}`",
            "",
            "## Side Effects",
            "",
        ]
    )
    lines.extend(f"- {key}: `{str(value).lower()}`" for key, value in payload["side_effect_summary"].items())
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No Stage-1 masked-v2 HGB entry candidate is eligible for paper-readiness validation.",
            "The stopping rule routes next work to the preregistered feature recovery ladder.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preregister":
        payload = preregistration_payload(args)
        write_json(args.out_dir / "preregistration.json", payload)
        (args.out_dir / "preregistration.md").write_text(render_preregistration(payload))
        registry = args.out_dir / "experiment_registry.jsonl"
        with registry.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "attempt_id": payload["attempt_id"],
                    "preregistration": str(args.out_dir / "preregistration.json"),
                    "registry": str(registry),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode in {"evaluate-primary", "evaluate-conservative"}:
        batch = batch_from_mode(str(args.mode))
        payload = evaluate_batch_payload(args, batch=batch)
        stem = f"{batch}_evaluation"
        write_json(args.out_dir / f"{stem}.json", payload)
        (args.out_dir / f"{stem}.md").write_text(render_batch_evaluation(payload))
        registry = args.out_dir / "experiment_registry.jsonl"
        with registry.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "batch": payload["batch"],
                    "routing_decision": payload["routing_decision"],
                    "blockers": payload["blockers"],
                    "attempts": payload["attempt_count"],
                    "report": str(args.out_dir / f"{stem}.md"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["status"] != "blocked" else 2
    if args.mode == "write-final-packet":
        payload = final_packet_payload(args)
        write_json(args.out_dir / "failure_packet.json", payload)
        (args.out_dir / "failure_packet.md").write_text(render_final_packet(payload))
        registry = args.out_dir / "experiment_registry.jsonl"
        with registry.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "final_routing_decision": payload["final_routing_decision"],
                    "report": str(args.out_dir / "failure_packet.md"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload, timing = verify_inputs_payload(args)
    write_json(args.out_dir / "input_readiness.json", payload)
    write_json(args.out_dir / "input_timing_rows.json", {"rows": timing["rows"]})
    (args.out_dir / "input_readiness.md").write_text(render_input_readiness(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "report": str(args.out_dir / "input_readiness.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
