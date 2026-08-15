"""Research-only validation packet for the frozen learned-defer challenger.

This runner is deliberately diagnostic. It reads frozen artifacts, reproduces
the preregistered replay totals, and writes a packet that explains whether the
challenger is ready for future untouched scoring. It does not train models,
change thresholds, touch broker endpoints, or change PAPER_DEFAULT_PROTOCOL101.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.environment_diagnostics import time_bucket as environment_time_bucket
from v4.model.unified_conservative_policy import validate_no_future_feature_columns
from v4.model.unified_slot_opportunity_cost_estimator import (
    SLOT_COST_FORBIDDEN_FEATURE_COLUMNS,
    estimator_predictions,
    validate_slot_cost_feature_columns,
)
from v4.model.unified_slot_opportunity_defer import SlotOpportunityDeferConfig
from v4.scripts.run_unified_conservative_neural_policy_strict_replay import (
    attach_predictions,
    included_session_keys,
    load_baseline_actions,
    load_flat_dataset,
    load_holding_dataset,
    load_json,
    load_policy_bundle,
    write_json,
)


ROLE_LABEL = "VALIDATION_LEARNED_DEFER_CHALLENGER_RESEARCH_PACKET_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/learned_defer_challenger_research_packet_v1")
DEFAULT_DOC_PATH = Path("v4/docs/LEARNED_DEFER_CHALLENGER_RESEARCH_PACKET_V1.md")
DEFAULT_REPLAY_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1")
DEFAULT_MODEL_ARTIFACTS = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts")
DEFAULT_POLICY_SUMMARY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json")
DEFAULT_SLOT_ESTIMATOR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib")
DEFAULT_SLOT_ESTIMATOR_SUMMARY = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_SLOT_LABELS = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/slot_opportunity_cost_labels.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")

SLIPPAGES = (0.0, 0.10, 0.25)
EXPECTED_REPLAY_TOTALS = {
    0.0: {"challenger_entries": 60, "trades": 739, "delta_vs_protocol101_same_scope": 66_780.0},
    0.10: {"challenger_entries": 60, "trades": 739, "delta_vs_protocol101_same_scope": 67_580.0},
    0.25: {"challenger_entries": 60, "trades": 739, "delta_vs_protocol101_same_scope": 68_780.0},
}

FINAL_DECISIONS = {
    "ready": "learned_defer_packet_complete_holdout_ready_but_challenge_blocked",
    "label": "learned_defer_packet_complete_holdout_blocked_by_label_mismatch",
    "concentration": "learned_defer_packet_complete_holdout_blocked_by_concentration",
    "calibration": "learned_defer_packet_complete_holdout_blocked_by_calibration",
    "reproduction": "learned_defer_packet_complete_reproduction_failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--model-artifacts", type=Path, default=DEFAULT_MODEL_ARTIFACTS)
    parser.add_argument("--policy-summary", type=Path, default=DEFAULT_POLICY_SUMMARY)
    parser.add_argument("--slot-estimator", type=Path, default=DEFAULT_SLOT_ESTIMATOR)
    parser.add_argument("--slot-estimator-summary", type=Path, default=DEFAULT_SLOT_ESTIMATOR_SUMMARY)
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--slot-labels", type=Path, default=DEFAULT_SLOT_LABELS)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--skip-doc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required_paths = required_artifact_paths(args)
    path_status = check_required_paths(required_paths)
    replay_summary = load_json(args.replay_dir / "summary.json") if (args.replay_dir / "summary.json").exists() else {}
    reproduction = check_replay_reproduction(replay_summary)
    freeze_manifest = build_freeze_manifest(args, required_paths, path_status, replay_summary, reproduction)
    write_json(args.out_dir / "freeze_manifest.json", freeze_manifest)

    if path_status["missing"] or reproduction["status"] != "pass":
        payload = failure_payload(args, freeze_manifest, path_status, reproduction)
        write_packet_outputs(args, payload)
        print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
        return 0

    estimator_bundle = joblib.load(args.slot_estimator)
    policy_bundle = load_policy_bundle(args.model_artifacts)
    validate_no_future_feature_columns(policy_bundle["flat_feature_columns"])
    validate_no_future_feature_columns(policy_bundle["holding_feature_columns"])
    validate_slot_cost_feature_columns(list(estimator_bundle["feature_columns"]))

    session_manifest = pd.read_csv(args.session_manifest)
    included_sessions = included_session_keys(session_manifest)
    flat_feature_columns = list(
        dict.fromkeys([*policy_bundle["flat_feature_columns"], *list(estimator_bundle["feature_columns"])])
    )
    flat = load_flat_dataset(args.flat_dataset, flat_feature_columns, included_sessions)
    baseline = load_baseline_actions(args.baseline_actions, included_sessions, seed=1)

    flat = attach_predictions(flat, policy_bundle, head="flat")
    flat = attach_slot_estimates(flat, estimator_bundle)
    defer_config = frozen_defer_config(replay_summary)
    flat = attach_static_defer_fields(flat, policy_bundle["config"], defer_config)

    trades_by_slippage = load_replay_trades(args.replay_dir)
    decisions_by_slippage = load_replay_decisions(args.replay_dir)
    zero_trades = trades_by_slippage[0.0]
    zero_challengers = zero_trades[zero_trades["source"].astype(str).eq("challenger")].copy()

    blocked_rows = []
    for slippage, trades in trades_by_slippage.items():
        blocked_rows.extend(attribute_blocked_protocol101(trades, baseline, slippage_per_side=slippage))
    blocked_attribution = pd.DataFrame(blocked_rows)
    if blocked_attribution.empty:
        blocked_attribution = pd.DataFrame(columns=blocked_protocol101_columns())
    blocked_attribution.to_csv(args.out_dir / "blocked_protocol101_attribution.csv", index=False)

    holding_stats = challenger_holding_stats(args.holding_dataset, zero_challengers["candidate_uid"].astype(str).unique())
    flat_entry_anomaly = build_flat_entry_anomaly(flat, zero_challengers, blocked_attribution, holding_stats)
    flat_entry_anomaly.to_csv(args.out_dir / "flat_entry_anomaly.csv", index=False)
    anomaly_status = decide_anomaly_status(flat_entry_anomaly)

    concentration = build_concentration_fragility(
        trades_by_slippage,
        baseline,
        flat,
        blocked_attribution,
        holding_stats,
    )
    concentration["rows"].to_csv(args.out_dir / "concentration_fragility.csv", index=False)
    concentration_status = decide_concentration_status(replay_summary, concentration["summary"])

    slot_cost_calibration = build_slot_cost_calibration(flat, args.slot_labels)
    slot_cost_calibration.to_csv(args.out_dir / "slot_cost_calibration.csv", index=False)
    calibration_status = decide_calibration_status(slot_cost_calibration)

    validation_readiness = build_validation_readiness(args, zero_trades, baseline)
    packet_decision = decide_packet(anomaly_status, concentration_status, calibration_status)
    blockers = packet_blockers(packet_decision, anomaly_status, concentration_status, calibration_status, validation_readiness)

    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "research-only validation packet for the frozen learned-defer challenger",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "thresholds_retuned": False,
        "untouched_holdout_scored": False,
        "challenge_allowed": False,
        "decision": packet_decision,
        "blockers": blockers,
        "reproduction_status": reproduction["status"],
        "anomaly_status": anomaly_status,
        "concentration_status": concentration_status,
        "slot_cost_calibration_status": calibration_status,
        "validation_readiness_status": validation_readiness["status"],
        "defer_config": defer_config.to_dict(),
        "replay_reproduction": reproduction,
        "concentration_summary": concentration["summary"],
        "slot_cost_calibration_summary": summarize_calibration_status(slot_cost_calibration),
        "validation_readiness": validation_readiness,
        "live_no_order_parity_jsonl_schema": live_no_order_parity_schema(),
        "fill_observation_jsonl_schema": fill_observation_schema(),
        "artifacts": {
            "freeze_manifest": str(args.out_dir / "freeze_manifest.json"),
            "flat_entry_anomaly": str(args.out_dir / "flat_entry_anomaly.csv"),
            "concentration_fragility": str(args.out_dir / "concentration_fragility.csv"),
            "slot_cost_calibration": str(args.out_dir / "slot_cost_calibration.csv"),
            "blocked_protocol101_attribution": str(args.out_dir / "blocked_protocol101_attribution.csv"),
        },
        "source_artifacts": {name: str(path) for name, path in required_paths.items()},
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
        "next_required_evidence": next_required_evidence(packet_decision),
    }
    write_packet_outputs(args, payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def required_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "replay_summary": args.replay_dir / "summary.json",
        "policy_summary": args.policy_summary,
        "policy_manifest": args.model_artifacts / "manifest.json",
        "flat_entry_model": args.model_artifacts / "flat_entry_model.pt",
        "holding_lifecycle_model": args.model_artifacts / "holding_lifecycle_model.pt",
        "flat_scaler": args.model_artifacts / "flat_scaler.json",
        "holding_scaler": args.model_artifacts / "holding_scaler.json",
        "slot_estimator": args.slot_estimator,
        "slot_estimator_summary": args.slot_estimator_summary,
        "flat_dataset": args.flat_dataset,
        "holding_dataset": args.holding_dataset,
        "baseline_actions": args.baseline_actions,
        "slot_labels": args.slot_labels,
        "session_manifest": args.session_manifest,
    }
    for slippage in SLIPPAGES:
        suffix = slippage_suffix(slippage)
        paths[f"replay_trades_{suffix}"] = args.replay_dir / f"trades_slippage_{suffix}.csv"
        paths[f"replay_decisions_{suffix}"] = args.replay_dir / f"decisions_slippage_{suffix}.csv"
    if args.ledger.exists():
        paths["research_ledger"] = args.ledger
    return paths


def check_required_paths(paths: dict[str, Path]) -> dict[str, Any]:
    missing = [name for name, path in paths.items() if not path.exists()]
    return {"status": "fail" if missing else "pass", "missing": missing}


def check_replay_reproduction(summary: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    actual: dict[str, dict[str, Any]] = {}
    stress_results = summary.get("stress_results", []) if isinstance(summary, dict) else []
    by_slippage = {float(item.get("slippage_per_side", math.nan)): item for item in stress_results}
    for slippage, expected in EXPECTED_REPLAY_TOTALS.items():
        stress = by_slippage.get(float(slippage))
        if stress is None:
            mismatches.append({"slippage": slippage, "field": "stress_result", "expected": "present", "actual": "missing"})
            continue
        totals = stress.get("totals", {})
        actual[f"{slippage:.2f}"] = {
            "challenger_entries": int(totals.get("challenger_entries", -1)),
            "trades": int(totals.get("trades", -1)),
            "delta_vs_protocol101_same_scope": float(totals.get("delta_vs_protocol101_same_scope", math.nan)),
        }
        for field, expected_value in expected.items():
            observed = actual[f"{slippage:.2f}"][field]
            if isinstance(expected_value, float):
                ok = math.isfinite(float(observed)) and abs(float(observed) - expected_value) <= 1e-9
            else:
                ok = int(observed) == int(expected_value)
            if not ok:
                mismatches.append(
                    {
                        "slippage": slippage,
                        "field": field,
                        "expected": expected_value,
                        "actual": observed,
                    }
                )
    return {"status": "pass" if not mismatches else "fail", "expected": EXPECTED_REPLAY_TOTALS, "actual": actual, "mismatches": mismatches}


def build_freeze_manifest(
    args: argparse.Namespace,
    required_paths: dict[str, Path],
    path_status: dict[str, Any],
    replay_summary: dict[str, Any],
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    model_manifest = load_json(args.model_artifacts / "manifest.json") if (args.model_artifacts / "manifest.json").exists() else {}
    estimator_summary = load_json(args.slot_estimator_summary) if args.slot_estimator_summary.exists() else {}
    replay_defer_config = replay_summary.get("defer_config", {}) if isinstance(replay_summary, dict) else {}
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "frozen artifact manifest for the learned-defer research packet",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "challenge_allowed": False,
        "path_status": path_status,
        "reproduction": reproduction,
        "overlay_parameters": replay_defer_config,
        "model_config": model_manifest.get("config", {}),
        "flat_feature_columns": model_manifest.get("flat_feature_columns", []),
        "holding_feature_columns": model_manifest.get("holding_feature_columns", []),
        "estimator_config": estimator_summary.get("config", {}),
        "estimator_decision": estimator_summary.get("decision"),
        "splits": sorted({split for item in replay_summary.get("stress_results", []) for split in item.get("splits", {})})
        if isinstance(replay_summary, dict)
        else [],
        "fixed_pass_fail_gates": {
            "expected_replay_totals": EXPECTED_REPLAY_TOTALS,
            "top_day_positive_share_warn": 0.35,
            "top5_trade_positive_share_warn": 0.50,
            "top5_trade_positive_share_block": 0.70,
            "single_split_positive_delta_share_warn": 0.75,
            "protected_split_negative_delta_blocks": True,
            "calibration_coverage_min_q1_recent": 0.80,
        },
        "files": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in sorted(required_paths.items())
        },
    }


def failure_payload(
    args: argparse.Namespace,
    freeze_manifest: dict[str, Any],
    path_status: dict[str, Any],
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "research-only validation packet for the frozen learned-defer challenger",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "thresholds_retuned": False,
        "untouched_holdout_scored": False,
        "challenge_allowed": False,
        "decision": FINAL_DECISIONS["reproduction"],
        "blockers": ["frozen artifact missing or replay total drift"],
        "path_status": path_status,
        "replay_reproduction": reproduction,
        "freeze_manifest": str(args.out_dir / "freeze_manifest.json"),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
        "freeze_manifest_summary": {
            "missing": path_status.get("missing", []),
            "reproduction_status": reproduction.get("status"),
        },
    }


def frozen_defer_config(replay_summary: dict[str, Any]) -> SlotOpportunityDeferConfig:
    config = replay_summary.get("defer_config", {})
    return SlotOpportunityDeferConfig(
        min_net_advantage_margin=float(config.get("min_net_advantage_margin", 0.0)),
        blocked_cost_uncertainty_weight=float(config.get("blocked_cost_uncertainty_weight", 0.25)),
        max_blocked_protocol101_entries=int(config.get("max_blocked_protocol101_entries", 3)),
        require_nonnegative_q1_q3_stress=bool(config.get("require_nonnegative_q1_q3_stress", True)),
        baseline=str(config.get("baseline", "PAPER_DEFAULT_PROTOCOL101")),
    )


def attach_slot_estimates(frame: pd.DataFrame, estimator_bundle: dict[str, Any]) -> pd.DataFrame:
    estimates = estimator_predictions(estimator_bundle, frame)
    out = frame.copy()
    for column in estimates.columns:
        out[column] = estimates[column].to_numpy()
    return out


def attach_static_defer_fields(
    flat: pd.DataFrame,
    policy_config: Any,
    defer_config: SlotOpportunityDeferConfig,
) -> pd.DataFrame:
    out = flat.copy()
    out["adjusted_advantage"] = (
        pd.to_numeric(out["predicted_advantage"], errors="coerce").fillna(-math.inf)
        - np.maximum(0.0, pd.to_numeric(out["estimated_blocked_protocol101_cost"], errors="coerce").fillna(0.0))
        - float(defer_config.blocked_cost_uncertainty_weight)
        * np.maximum(0.0, pd.to_numeric(out["blocked_cost_uncertainty"], errors="coerce").fillna(0.0))
    )
    estimated_entries = pd.to_numeric(out["estimated_blocked_entries"], errors="coerce").fillna(math.inf).round().clip(lower=0)
    out["policy_allowed_static_10k"] = (
        pd.to_numeric(out["predicted_advantage"], errors="coerce").ge(float(policy_config.min_advantage_margin))
        & pd.to_numeric(out["positive_probability"], errors="coerce").ge(float(policy_config.positive_probability_min))
        & pd.to_numeric(out["tail_probability"], errors="coerce").le(float(policy_config.tail_probability_max))
        & pd.to_numeric(out["entry_premium"], errors="coerce").gt(0.0)
        & pd.to_numeric(out["entry_premium"], errors="coerce").le(10_000.0)
    )
    out["overlay_allowed_static_10k"] = (
        out["policy_allowed_static_10k"]
        & out["adjusted_advantage"].ge(float(defer_config.min_net_advantage_margin))
        & estimated_entries.le(int(defer_config.max_blocked_protocol101_entries))
    )
    return out


def load_replay_trades(replay_dir: Path) -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    for slippage in SLIPPAGES:
        frame = pd.read_csv(replay_dir / f"trades_slippage_{slippage_suffix(slippage)}.csv")
        frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        frame["exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
        frame["trade_key"] = trade_keys(frame)
        out[float(slippage)] = frame
    return out


def load_replay_decisions(replay_dir: Path) -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    for slippage in SLIPPAGES:
        frame = pd.read_csv(replay_dir / f"decisions_slippage_{slippage_suffix(slippage)}.csv")
        frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        out[float(slippage)] = frame
    return out


def trade_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["slippage_per_side"].astype(str)
        + "|"
        + frame["split"].astype(str)
        + "|"
        + frame["session"].astype(str)
        + "|"
        + frame["decision_time"].astype(str)
        + "|"
        + frame["candidate_uid"].astype(str)
        + "|"
        + frame["source"].astype(str)
    )


def attribute_blocked_protocol101(
    trades: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    slippage_per_side: float,
    contract_multiplier: float = 100.0,
) -> list[dict[str, Any]]:
    baseline_entries = baseline[baseline["protocol101_action"].astype(str).eq("enter")].copy()
    baseline_entries["decision_dt"] = pd.to_datetime(baseline_entries["decision_dt"], utc=True, errors="coerce")
    stress = 2.0 * float(slippage_per_side) * float(contract_multiplier)
    rows: list[dict[str, Any]] = []
    challengers = trades[trades["source"].astype(str).eq("challenger")].copy()
    for trade in challengers.to_dict("records"):
        start = pd.Timestamp(trade["decision_dt"])
        end = pd.Timestamp(trade["exit_dt"])
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        mask = (
            baseline_entries["split"].astype(str).eq(str(trade["split"]))
            & baseline_entries["session"].astype(str).eq(str(trade["session"]))
            & baseline_entries["decision_dt"].ge(start)
            & baseline_entries["decision_dt"].lt(end)
        )
        for _, base in baseline_entries.loc[mask].iterrows():
            raw_pnl = finite(base.get("baseline_trade_pnl"), 0.0)
            rows.append(
                {
                    "slippage_per_side": float(slippage_per_side),
                    "challenger_trade_key": str(trade["trade_key"]),
                    "split": str(trade["split"]),
                    "session": str(trade["session"]),
                    "challenger_candidate_uid": str(trade["candidate_uid"]),
                    "challenger_decision_time": pd.Timestamp(trade["decision_dt"]).isoformat(),
                    "challenger_exit_time": pd.Timestamp(trade["exit_dt"]).isoformat(),
                    "challenger_pnl": float(trade["pnl"]),
                    "baseline_decision_time": pd.Timestamp(base["decision_dt"]).isoformat(),
                    "baseline_candidate_uid": str(base.get("surface_candidate_uid", "")),
                    "baseline_contract_id": str(base.get("contract_id", "")),
                    "baseline_pnl_raw": raw_pnl,
                    "baseline_pnl_stressed": float(raw_pnl - stress),
                    "blocked_relation": "challenger_open_interval_contains_protocol101_entry",
                }
            )
    return rows


def blocked_protocol101_columns() -> list[str]:
    return [
        "slippage_per_side",
        "challenger_trade_key",
        "split",
        "session",
        "challenger_candidate_uid",
        "challenger_decision_time",
        "challenger_exit_time",
        "challenger_pnl",
        "baseline_decision_time",
        "baseline_candidate_uid",
        "baseline_contract_id",
        "baseline_pnl_raw",
        "baseline_pnl_stressed",
        "blocked_relation",
    ]


def challenger_holding_stats(holding_path: Path, challenger_candidate_uids: np.ndarray) -> pd.DataFrame:
    columns = ["candidate_uid", "state_time", "mfe_to_now", "mae_to_now", "giveback_from_mfe"]
    if len(challenger_candidate_uids) == 0:
        return pd.DataFrame(columns=["candidate_uid", "max_mfe", "min_mae", "max_giveback"])
    holding = pd.read_parquet(holding_path, columns=columns)
    holding = holding[holding["candidate_uid"].astype(str).isin(set(map(str, challenger_candidate_uids)))].copy()
    if holding.empty:
        return pd.DataFrame(columns=["candidate_uid", "max_mfe", "min_mae", "max_giveback"])
    grouped = holding.groupby("candidate_uid", sort=False)
    return grouped.agg(
        max_mfe=("mfe_to_now", "max"),
        min_mae=("mae_to_now", "min"),
        max_giveback=("giveback_from_mfe", "max"),
    ).reset_index()


def build_flat_entry_anomaly(
    flat: pd.DataFrame,
    zero_challengers: pd.DataFrame,
    blocked_attribution: pd.DataFrame,
    holding_stats: pd.DataFrame,
) -> pd.DataFrame:
    event_argmax = flat.loc[flat.groupby(["split", "session", "decision_dt"], sort=False)["predicted_advantage"].idxmax()].copy()
    overlay_allowed = flat[flat["overlay_allowed_static_10k"].astype(bool)].copy()
    actual = zero_challengers.merge(
        flat[
            [
                "candidate_uid",
                "a_enter",
                "adjusted_advantage",
            ]
        ],
        on="candidate_uid",
        how="left",
    ).merge(holding_stats, on="candidate_uid", how="left")
    blocked_zero = blocked_attribution[pd.to_numeric(blocked_attribution.get("slippage_per_side", -1), errors="coerce").eq(0.0)]
    blocked_by_trade = (
        blocked_zero.groupby("challenger_trade_key", sort=False)
        .agg(
            actual_blocked_protocol101_entries=("baseline_pnl_stressed", "size"),
            actual_blocked_protocol101_pnl=("baseline_pnl_stressed", "sum"),
        )
        .rename_axis("trade_key")
        .reset_index()
        if not blocked_zero.empty
        else pd.DataFrame(columns=["trade_key", "actual_blocked_protocol101_entries", "actual_blocked_protocol101_pnl"])
    )
    actual = actual.merge(blocked_by_trade, on="trade_key", how="left")
    actual["actual_blocked_protocol101_entries"] = pd.to_numeric(actual["actual_blocked_protocol101_entries"], errors="coerce").fillna(0)
    actual["actual_blocked_protocol101_pnl"] = pd.to_numeric(actual["actual_blocked_protocol101_pnl"], errors="coerce").fillna(0.0)
    actual["actual_slot_cost"] = np.maximum(0.0, actual["actual_blocked_protocol101_pnl"])
    actual["actual_serial_contribution"] = pd.to_numeric(actual["pnl"], errors="coerce").fillna(0.0) - actual["actual_slot_cost"]

    rows = []
    rows.extend(summarize_flat_view(flat, "all_flat_rows"))
    rows.extend(summarize_flat_view(event_argmax, "event_argmax_rows"))
    rows.extend(summarize_flat_view(overlay_allowed, "overlay_allowed_static_10k_rows"))
    rows.extend(summarize_actual_challengers(actual, "actual_replay_challenger_entries"))
    return pd.DataFrame(rows)


def summarize_flat_view(frame: pd.DataFrame, view: str) -> list[dict[str, Any]]:
    rows = [flat_view_row(frame, view, "all")]
    for split, group in frame.groupby("split", sort=True):
        rows.append(flat_view_row(group, view, str(split)))
    return rows


def flat_view_row(frame: pd.DataFrame, view: str, split: str) -> dict[str, Any]:
    return {
        "view": view,
        "split": split,
        "rows": int(len(frame)),
        "mean_a_enter": mean_or_none(frame.get("a_enter")),
        "median_a_enter": median_or_none(frame.get("a_enter")),
        "positive_a_enter_rate": mean_bool(frame.get("a_enter"), threshold=0.0),
        "mean_predicted_advantage": mean_or_none(frame.get("predicted_advantage")),
        "mean_adjusted_advantage": mean_or_none(frame.get("adjusted_advantage")),
        "mean_estimated_blocked_cost": mean_or_none(frame.get("estimated_blocked_protocol101_cost")),
        "mean_estimated_blocked_entries": mean_or_none(frame.get("estimated_blocked_entries")),
        "actual_pnl": None,
        "actual_blocked_protocol101_pnl": None,
        "actual_serial_contribution": None,
    }


def summarize_actual_challengers(frame: pd.DataFrame, view: str) -> list[dict[str, Any]]:
    rows = [actual_view_row(frame, view, "all")]
    for split, group in frame.groupby("split", sort=True):
        rows.append(actual_view_row(group, view, str(split)))
    return rows


def actual_view_row(frame: pd.DataFrame, view: str, split: str) -> dict[str, Any]:
    return {
        "view": view,
        "split": split,
        "rows": int(len(frame)),
        "mean_a_enter": mean_or_none(frame.get("a_enter")),
        "median_a_enter": median_or_none(frame.get("a_enter")),
        "positive_a_enter_rate": mean_bool(frame.get("a_enter"), threshold=0.0),
        "mean_predicted_advantage": mean_or_none(frame.get("predicted_advantage")),
        "mean_adjusted_advantage": mean_or_none(frame.get("adjusted_advantage")),
        "mean_estimated_blocked_cost": mean_or_none(frame.get("estimated_blocked_protocol101_cost")),
        "mean_estimated_blocked_entries": mean_or_none(frame.get("estimated_blocked_entries")),
        "actual_pnl": sum_or_none(frame.get("pnl")),
        "actual_blocked_protocol101_pnl": sum_or_none(frame.get("actual_blocked_protocol101_pnl")),
        "actual_serial_contribution": sum_or_none(frame.get("actual_serial_contribution")),
    }


def decide_anomaly_status(flat_entry_anomaly: pd.DataFrame) -> str:
    actual = flat_entry_anomaly[
        flat_entry_anomaly["view"].eq("actual_replay_challenger_entries") & flat_entry_anomaly["split"].eq("all")
    ]
    allowed = flat_entry_anomaly[
        flat_entry_anomaly["view"].eq("overlay_allowed_static_10k_rows") & flat_entry_anomaly["split"].eq("all")
    ]
    actual_net = finite(actual["actual_serial_contribution"].iloc[0] if not actual.empty else math.nan, -math.inf)
    allowed_mean = finite(allowed["mean_a_enter"].iloc[0] if not allowed.empty else math.nan, math.nan)
    if actual_net < -1e-9:
        return "blocked_unresolved_label_policy_mismatch"
    if math.isfinite(allowed_mean) and allowed_mean < 0.0:
        return "resolved_policy_weighting_mismatch"
    return "resolved_no_negative_allowed_anomaly"


def build_concentration_fragility(
    trades_by_slippage: dict[float, pd.DataFrame],
    baseline: pd.DataFrame,
    flat: pd.DataFrame,
    blocked_attribution: pd.DataFrame,
    holding_stats: pd.DataFrame,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    flat_join = flat[
        [
            "candidate_uid",
            "entry_spread",
            "entry_spread_frac",
            "entry_premium",
            "a_enter",
            "predicted_advantage",
            "adjusted_advantage",
            "estimated_blocked_protocol101_cost",
        ]
    ].drop_duplicates("candidate_uid")
    for slippage, trades in trades_by_slippage.items():
        session_delta = session_delta_frame(trades, baseline, slippage)
        rows.extend(group_metric_rows(session_delta, slippage, "session_delta", "split", value_col="delta"))
        rows.extend(group_metric_rows(session_delta, slippage, "session_delta", "session", value_col="delta"))

        challenger = trades[trades["source"].astype(str).eq("challenger")].copy()
        challenger = challenger.merge(flat_join, on="candidate_uid", how="left", suffixes=("", "_flat"))
        challenger = challenger.merge(holding_stats, on="candidate_uid", how="left")
        challenger = attach_trade_buckets(challenger)
        blocked = blocked_attribution[pd.to_numeric(blocked_attribution.get("slippage_per_side", -1), errors="coerce").eq(float(slippage))]
        blocked_by_trade = (
            blocked.groupby("challenger_trade_key", sort=False)
            .agg(actual_blocked_entries=("baseline_pnl_stressed", "size"), actual_blocked_pnl=("baseline_pnl_stressed", "sum"))
            .reset_index()
            if not blocked.empty
            else pd.DataFrame(columns=["challenger_trade_key", "actual_blocked_entries", "actual_blocked_pnl"])
        )
        challenger = challenger.merge(blocked_by_trade, left_on="trade_key", right_on="challenger_trade_key", how="left")
        challenger["actual_blocked_entries"] = pd.to_numeric(challenger["actual_blocked_entries"], errors="coerce").fillna(0)
        challenger["actual_blocked_pnl"] = pd.to_numeric(challenger["actual_blocked_pnl"], errors="coerce").fillna(0.0)
        challenger["net_contribution"] = pd.to_numeric(challenger["pnl"], errors="coerce").fillna(0.0) - np.maximum(0.0, challenger["actual_blocked_pnl"])

        for dimension in [
            "right",
            "time_bucket",
            "premium_bucket",
            "spread_bucket",
            "offset_bucket",
            "score_bucket",
            "slot_cost_bucket",
            "duration_bucket",
            "exit_reason",
            "forced_flat_flag",
            "mfe_bucket",
            "mae_bucket",
            "giveback_bucket",
        ]:
            rows.extend(group_metric_rows(challenger, slippage, "challenger_net_contribution", dimension, value_col="net_contribution"))

        summaries[f"{slippage:.2f}"] = concentration_summary_for_slippage(session_delta, challenger)
    return {"rows": pd.DataFrame(rows), "summary": summaries}


def session_delta_frame(trades: pd.DataFrame, baseline: pd.DataFrame, slippage: float) -> pd.DataFrame:
    replay = trades.groupby(["split", "session"], sort=False).agg(total_pnl=("pnl", "sum"), trades=("pnl", "size")).reset_index()
    baseline_entries = baseline[baseline["protocol101_action"].astype(str).eq("enter")].copy()
    stress = 2.0 * float(slippage) * 100.0
    baseline_entries["baseline_pnl_stressed"] = pd.to_numeric(baseline_entries["baseline_trade_pnl"], errors="coerce").fillna(0.0) - stress
    base = (
        baseline_entries.groupby(["split", "session"], sort=False)
        .agg(protocol101_same_scope_pnl=("baseline_pnl_stressed", "sum"), protocol101_entries=("baseline_pnl_stressed", "size"))
        .reset_index()
    )
    out = replay.merge(base, on=["split", "session"], how="outer").fillna({"total_pnl": 0.0, "trades": 0, "protocol101_same_scope_pnl": 0.0, "protocol101_entries": 0})
    out["delta"] = pd.to_numeric(out["total_pnl"], errors="coerce").fillna(0.0) - pd.to_numeric(out["protocol101_same_scope_pnl"], errors="coerce").fillna(0.0)
    return out


def attach_trade_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["time_bucket"] = [safe_time_bucket(value) for value in out["decision_dt"]]
    out["premium_bucket"] = pd.to_numeric(out.get("entry_premium", out.get("entry_ask")), errors="coerce").map(premium_bucket)
    out["spread_bucket"] = pd.to_numeric(out.get("entry_spread_frac"), errors="coerce").map(spread_bucket)
    out["offset_bucket"] = pd.to_numeric(out.get("offset"), errors="coerce").map(offset_bucket)
    out["score_bucket"] = pd.to_numeric(out.get("predicted_advantage"), errors="coerce").map(score_bucket)
    out["slot_cost_bucket"] = pd.to_numeric(out.get("estimated_blocked_protocol101_cost"), errors="coerce").map(slot_cost_bucket)
    out["duration_bucket"] = pd.to_numeric(out.get("duration_minutes"), errors="coerce").map(duration_bucket)
    out["forced_flat_flag"] = out.get("exit_reason", "").astype(str).eq("forced_flat_no_lifecycle_exit_signal").map({True: "forced_flat", False: "not_forced_flat"})
    out["mfe_bucket"] = pd.to_numeric(out.get("max_mfe"), errors="coerce").map(pnl_bucket)
    out["mae_bucket"] = pd.to_numeric(out.get("min_mae"), errors="coerce").map(pnl_bucket)
    out["giveback_bucket"] = pd.to_numeric(out.get("max_giveback"), errors="coerce").map(pnl_bucket)
    return out


def group_metric_rows(frame: pd.DataFrame, slippage: float, lens: str, dimension: str, *, value_col: str) -> list[dict[str, Any]]:
    if frame.empty or dimension not in frame.columns:
        return []
    rows = []
    for bucket, group in frame.groupby(dimension, dropna=False, sort=True):
        value = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0)
        rows.append(
            {
                "slippage_per_side": float(slippage),
                "lens": lens,
                "dimension": dimension,
                "bucket": "missing" if pd.isna(bucket) else str(bucket),
                "rows": int(len(group)),
                "pnl_or_delta": float(value.sum()),
                "mean": float(value.mean()) if len(value) else 0.0,
                "median": float(value.median()) if len(value) else 0.0,
                "positive_rows": int((value > 0).sum()),
                "negative_rows": int((value < 0).sum()),
                "challenger_entries": int(group["source"].astype(str).eq("challenger").sum()) if "source" in group.columns else 0,
            }
        )
    return rows


def concentration_summary_for_slippage(session_delta: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, Any]:
    positive_sessions = pd.to_numeric(session_delta["delta"], errors="coerce").fillna(0.0)
    positive_sessions = positive_sessions[positive_sessions > 0.0].sort_values(ascending=False)
    top_day_share = float(positive_sessions.iloc[0] / positive_sessions.sum()) if len(positive_sessions) and positive_sessions.sum() > 0 else 1.0

    trade_net = pd.to_numeric(challenger.get("net_contribution", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    positive_trades = trade_net[trade_net > 0.0].sort_values(ascending=False)
    top5_share = float(positive_trades.head(5).sum() / positive_trades.sum()) if len(positive_trades) and positive_trades.sum() > 0 else 1.0

    split_delta = session_delta.groupby("split", sort=False)["delta"].sum()
    positive_split = split_delta[split_delta > 0.0]
    top_split_share = float(positive_split.max() / positive_split.sum()) if len(positive_split) and positive_split.sum() > 0 else 1.0
    return {
        "total_delta": float(session_delta["delta"].sum()),
        "negative_split_count": int((split_delta < 0.0).sum()),
        "top_day_positive_share": top_day_share,
        "top5_trade_positive_share": top5_share,
        "top_split_positive_delta_share": top_split_share,
        "warnings": concentration_warnings(top_day_share, top5_share, top_split_share),
    }


def concentration_warnings(top_day_share: float, top5_share: float, top_split_share: float) -> list[str]:
    warnings = []
    if top_day_share > 0.35:
        warnings.append("top_day_positive_share_gt_0_35")
    if top5_share > 0.50:
        warnings.append("top5_trade_positive_share_gt_0_50")
    if top_split_share > 0.75:
        warnings.append("single_split_positive_delta_share_gt_0_75")
    return warnings


def decide_concentration_status(replay_summary: dict[str, Any], concentration_summary: dict[str, Any]) -> str:
    for stress in replay_summary.get("stress_results", []):
        totals = stress.get("totals", {})
        if finite(totals.get("delta_vs_protocol101_same_scope"), -math.inf) < -1e-9:
            return "blocked_negative_total_delta"
        for split, item in stress.get("splits", {}).items():
            if finite(item.get("delta_vs_protocol101_same_scope"), -math.inf) < -1e-9:
                return f"blocked_negative_diagnostic_split_{split}"
    for summary in concentration_summary.values():
        if finite(summary.get("top5_trade_positive_share"), 1.0) > 0.70:
            return "blocked_top5_trade_concentration_gt_0_70"
    return "pass_with_warnings" if any(item.get("warnings") for item in concentration_summary.values()) else "pass"


def build_slot_cost_calibration(flat: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(
        labels_path,
        columns=[
            "candidate_uid",
            "blocked_protocol101_entries",
            "blocked_protocol101_pnl_0_00",
            "has_blocked_protocol101_entry",
        ],
    )
    frame = flat.merge(labels, on="candidate_uid", how="inner")
    frame["actual_cost"] = np.maximum(0.0, pd.to_numeric(frame["blocked_protocol101_pnl_0_00"], errors="coerce").fillna(0.0))
    frame["actual_positive_cost"] = frame["actual_cost"].gt(0.0)
    frame["charge"] = pd.to_numeric(frame["estimated_blocked_protocol101_cost"], errors="coerce").fillna(0.0) + 0.25 * pd.to_numeric(
        frame["blocked_cost_uncertainty"], errors="coerce"
    ).fillna(0.0)
    frame["time_bucket"] = [safe_time_bucket(value) for value in frame["decision_dt"]]
    frame["premium_bucket"] = pd.to_numeric(frame["entry_premium"], errors="coerce").map(premium_bucket)
    frame["spread_bucket"] = pd.to_numeric(frame["entry_spread_frac"], errors="coerce").map(spread_bucket)
    rows: list[dict[str, Any]] = []
    rows.extend(calibration_rows(frame, "split", ["split"]))
    rows.extend(calibration_rows(frame, "split_time_bucket", ["split", "time_bucket"]))
    rows.extend(calibration_rows(frame, "split_side", ["split", "right"]))
    rows.extend(calibration_rows(frame, "split_premium_bucket", ["split", "premium_bucket"]))
    rows.extend(calibration_rows(frame, "split_spread_bucket", ["split", "spread_bucket"]))
    rows.extend(decile_reliability_rows(frame))
    return pd.DataFrame(rows)


def calibration_rows(frame: pd.DataFrame, regime: str, group_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append(calibration_metric_row(group, regime, "|".join("missing" if pd.isna(item) else str(item) for item in key_tuple)))
    return rows


def calibration_metric_row(group: pd.DataFrame, regime: str, bucket: str) -> dict[str, Any]:
    actual = pd.to_numeric(group["actual_cost"], errors="coerce").fillna(0.0)
    predicted = pd.to_numeric(group["estimated_blocked_protocol101_cost"], errors="coerce").fillna(0.0)
    probability = pd.to_numeric(group["blocked_cost_positive_probability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    positive = actual.gt(0.0)
    abs_error = (predicted - actual).abs()
    return {
        "regime": regime,
        "bucket": bucket,
        "rows": int(len(group)),
        "positive_cost_rate": float(positive.mean()) if len(group) else 0.0,
        "mean_actual_cost": float(actual.mean()) if len(group) else 0.0,
        "mean_predicted_cost": float(predicted.mean()) if len(group) else 0.0,
        "positive_auc": safe_auc(positive, probability),
        "positive_brier": safe_brier(positive, probability),
        "cost_mae": float(abs_error.mean()) if len(group) else 0.0,
        "cost_p90_abs_error": float(abs_error.quantile(0.90)) if len(group) else 0.0,
        "charge_coverage": float((pd.to_numeric(group["charge"], errors="coerce").fillna(0.0) >= actual).mean()) if len(group) else 0.0,
        "mean_charge": float(pd.to_numeric(group["charge"], errors="coerce").fillna(0.0).mean()) if len(group) else 0.0,
    }


def decile_reliability_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    working = frame.copy()
    probability = pd.to_numeric(working["blocked_cost_positive_probability"], errors="coerce").fillna(0.0)
    for split, group in working.groupby("split", sort=True):
        group = group.copy()
        try:
            group["probability_decile"] = pd.qcut(
                pd.to_numeric(group["blocked_cost_positive_probability"], errors="coerce").fillna(0.0),
                q=10,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            group["probability_decile"] = 0
        rows.extend(calibration_rows(group, "split_probability_decile", ["split", "probability_decile"]))
    if len(probability) == 0:
        return rows
    return rows


def decide_calibration_status(calibration: pd.DataFrame) -> str:
    split_rows = calibration[calibration["regime"].eq("split")].copy()
    by_bucket = {str(row["bucket"]): row for row in split_rows.to_dict("records")}
    weak = []
    for split in ("q1_2026", "recent_2026"):
        row = by_bucket.get(split)
        if not row:
            weak.append(f"{split}_missing")
            continue
        if finite(row.get("charge_coverage"), 0.0) < 0.80:
            weak.append(f"{split}_charge_coverage_lt_0_80")
    aucs = [finite(row.get("positive_auc"), math.nan) for row in by_bucket.values()]
    aucs = [value for value in aucs if math.isfinite(value)]
    if aucs and max(aucs) - min(aucs) > 0.20:
        weak.append("positive_auc_split_range_gt_0_20")
    return "blocked_undercoverage_or_split_instability" if weak else "pass"


def summarize_calibration_status(calibration: pd.DataFrame) -> list[dict[str, Any]]:
    split_rows = calibration[calibration["regime"].eq("split")].copy()
    return split_rows.sort_values("bucket").to_dict("records")


def build_validation_readiness(args: argparse.Namespace, zero_trades: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    summaries = sorted(Path("v4/audit/autoresearch").glob("*/summary.json"))
    comparable = 0
    for path in summaries:
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        if payload.get("stress_results") and payload.get("paper_default_baseline") == "PAPER_DEFAULT_PROTOCOL101":
            comparable += 1
    session_delta = session_delta_frame(zero_trades, baseline, 0.0)
    bootstrap = bootstrap_session_delta(session_delta["delta"].to_numpy(dtype=float))
    status = (
        "formal_overfit_control_scaffold_ready"
        if comparable >= 10
        else "formal_overfit_control_blocked_missing_comparable_strategy_matrix"
    )
    return {
        "status": status,
        "summary_json_files_scanned": int(len(summaries)),
        "comparable_protocol101_stress_result_files": int(comparable),
        "daily_block_bootstrap_delta": bootstrap,
        "pbo_cscv_status": "not_computed_missing_comparable_strategy_matrix" if comparable < 10 else "scaffold_only_not_promotion_grade",
    }


def bootstrap_session_delta(values: np.ndarray, *, samples: int = 1000, seed: int = 1) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"sessions": 0, "mean_delta": 0.0, "ci025_total_delta": 0.0, "ci975_total_delta": 0.0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True).sum(axis=1)
    return {
        "sessions": int(len(values)),
        "mean_delta": float(values.mean()),
        "total_delta": float(values.sum()),
        "ci025_total_delta": float(np.quantile(draws, 0.025)),
        "ci975_total_delta": float(np.quantile(draws, 0.975)),
    }


def decide_packet(anomaly_status: str, concentration_status: str, calibration_status: str) -> str:
    if anomaly_status.startswith("blocked"):
        return FINAL_DECISIONS["label"]
    if concentration_status.startswith("blocked"):
        return FINAL_DECISIONS["concentration"]
    if calibration_status.startswith("blocked"):
        return FINAL_DECISIONS["calibration"]
    return FINAL_DECISIONS["ready"]


def packet_blockers(
    decision: str,
    anomaly_status: str,
    concentration_status: str,
    calibration_status: str,
    validation_readiness: dict[str, Any],
) -> list[str]:
    blockers = [
        "calibrated stochastic fill model unavailable",
        "untouched holdout data pending",
        "live no-order full-action parity pending",
        "formal validation controls pending",
    ]
    if decision == FINAL_DECISIONS["label"]:
        blockers.insert(0, anomaly_status)
    elif decision == FINAL_DECISIONS["concentration"]:
        blockers.insert(0, concentration_status)
    elif decision == FINAL_DECISIONS["calibration"]:
        blockers.insert(0, calibration_status)
    if validation_readiness.get("status") != "formal_overfit_control_scaffold_ready":
        blockers.append(str(validation_readiness.get("status")))
    return blockers


def next_required_evidence(decision: str) -> list[str]:
    items = [
        "Keep Protocol101 as the paper default.",
        "Do not run additional neural experiments while this packet is unresolved.",
        "Build live no-order full-action parity for this exact frozen challenger.",
        "Collect stratified paper/no-order fill evidence before stochastic fill replay.",
        "Freeze and score an untouched holdout only after parity, fill, and formal validation gates pass.",
    ]
    if decision == FINAL_DECISIONS["label"]:
        items.insert(0, "Resolve the flat-entry label/policy mismatch before any untouched scoring.")
    elif decision == FINAL_DECISIONS["concentration"]:
        items.insert(0, "Investigate concentration and fragility before any untouched scoring.")
    elif decision == FINAL_DECISIONS["calibration"]:
        items.insert(0, "Replace or recalibrate the global slot-cost uncertainty charge before promotion-grade scoring.")
    elif decision == FINAL_DECISIONS["ready"]:
        items.insert(0, "Freeze this packet as research-only evidence before future untouched scoring.")
    return items


def live_no_order_parity_schema() -> list[str]:
    return [
        "timestamp",
        "session",
        "full_candidate_surface_count",
        "feature_hashes",
        "greeks_freshness",
        "quote_freshness",
        "action_masks",
        "account_affordability_state",
        "protocol101_action",
        "challenger_scores",
        "slot_cost_estimate",
        "slot_cost_uncertainty_charge",
        "final_defer_or_override_decision",
        "latency_ms",
        "missing_or_invalid_candidate_reasons",
        "live_orders_enabled_false",
        "broker_endpoint_called_false",
    ]


def fill_observation_schema() -> list[str]:
    return [
        "timestamp",
        "session",
        "intended_action",
        "side",
        "contract_id",
        "quote_age_ms",
        "bid",
        "ask",
        "spread",
        "limit_or_market_assumption",
        "submit_timestamp",
        "fill_timestamp",
        "fill_price",
        "cancel_or_reject_status",
        "realized_slippage",
    ]


def write_packet_outputs(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    write_json(args.out_dir / "summary.json", payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload.get('what_is_this', '')}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload.get('paper_default_baseline', 'PAPER_DEFAULT_PROTOCOL101')}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Challenge allowed: `{payload.get('challenge_allowed', False)}`",
        f"Decision: `{payload.get('decision', '')}`",
        "",
        "## Reproduction",
        "",
        f"- Status: `{payload.get('reproduction_status', payload.get('replay_reproduction', {}).get('status'))}`",
    ]
    reproduction = payload.get("replay_reproduction", {})
    for slip, totals in reproduction.get("actual", {}).items():
        lines.append(
            f"- Slippage `{slip}`: challenger `{totals.get('challenger_entries')}`, trades `{totals.get('trades')}`, "
            f"delta `{totals.get('delta_vs_protocol101_same_scope')}`"
        )
    if reproduction.get("mismatches"):
        lines.append(f"- Mismatches: `{reproduction['mismatches']}`")

    if "anomaly_status" in payload:
        lines.extend(
            [
                "",
                "## Diagnostic Status",
                "",
                f"- Flat-entry anomaly: `{payload['anomaly_status']}`",
                f"- Concentration: `{payload['concentration_status']}`",
                f"- Slot-cost calibration: `{payload['slot_cost_calibration_status']}`",
                f"- Formal validation readiness: `{payload['validation_readiness_status']}`",
                "",
                "## Concentration Summary",
                "",
                "| slippage | total delta | top day share | top 5 trade share | top split share | warnings |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for slip, item in payload.get("concentration_summary", {}).items():
            lines.append(
                f"| {slip} | {fmt(item.get('total_delta'))} | {fmt(item.get('top_day_positive_share'))} | "
                f"{fmt(item.get('top5_trade_positive_share'))} | {fmt(item.get('top_split_positive_delta_share'))} | "
                f"{', '.join(item.get('warnings', []))} |"
            )
        lines.extend(["", "## Slot-Cost Calibration By Split", ""])
        lines.extend(
            [
                "| split | rows | positive rate | AUC | Brier | MAE | p90 abs error | charge coverage |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload.get("slot_cost_calibration_summary", []):
            lines.append(
                f"| {row.get('bucket')} | {row.get('rows')} | {fmt(row.get('positive_cost_rate'))} | "
                f"{fmt(row.get('positive_auc'))} | {fmt(row.get('positive_brier'))} | {fmt(row.get('cost_mae'))} | "
                f"{fmt(row.get('cost_p90_abs_error'))} | {fmt(row.get('charge_coverage'))} |"
            )
        lines.extend(["", "## Validation Readiness", ""])
        readiness = payload.get("validation_readiness", {})
        lines.append(f"- Status: `{readiness.get('status')}`")
        lines.append(f"- Comparable strategy summaries: `{readiness.get('comparable_protocol101_stress_result_files')}`")
        lines.append(f"- PBO/CSCV status: `{readiness.get('pbo_cscv_status')}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in payload.get("blockers", []))
    lines.extend(["", "## Later Live Parity Schema", ""])
    lines.extend(f"- `{item}`" for item in payload.get("live_no_order_parity_jsonl_schema", live_no_order_parity_schema()))
    lines.extend(["", "## Later Fill Observation Schema", ""])
    lines.extend(f"- `{item}`" for item in payload.get("fill_observation_jsonl_schema", fill_observation_schema()))
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload.get("next_required_evidence", []), start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload.get("outputs", {}).items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Artifacts", ""])
    for name, path in payload.get("artifacts", {}).items():
        lines.append(f"- {name}: `{path}`")
    if "freeze_manifest" in payload:
        lines.append(f"- freeze_manifest: `{payload['freeze_manifest']}`")
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slippage_suffix(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_")


def safe_time_bucket(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return environment_time_bucket(ts.to_pydatetime())
    except Exception:
        return "unknown"


def premium_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value < 1_000:
        return "lt_1k"
    if value < 2_000:
        return "1k_2k"
    if value < 3_000:
        return "2k_3k"
    if value < 4_000:
        return "3k_4k"
    return "gte_4k"


def spread_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value < 0.01:
        return "lt_1pct"
    if value < 0.02:
        return "1_2pct"
    if value < 0.05:
        return "2_5pct"
    return "gte_5pct"


def offset_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    absolute = abs(value)
    if absolute <= 5:
        return "atm_5"
    if absolute <= 20:
        return "near_20"
    if absolute <= 50:
        return "wide_50"
    return "far_gt_50"


def score_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value < 250:
        return "lt_gate"
    if value < 500:
        return "250_500"
    if value < 1000:
        return "500_1000"
    return "gte_1000"


def slot_cost_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value <= 0:
        return "zero"
    if value < 250:
        return "lt_250"
    if value < 750:
        return "250_750"
    return "gte_750"


def duration_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value <= 5:
        return "lte_5m"
    if value <= 30:
        return "5_30m"
    if value <= 120:
        return "30_120m"
    return "gt_120m"


def pnl_bucket(value: Any) -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "unknown"
    if value < -2_000:
        return "lt_neg_2k"
    if value < 0:
        return "neg"
    if value < 1_000:
        return "0_1k"
    if value < 3_000:
        return "1k_3k"
    return "gte_3k"


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    y = pd.Series(y_true).astype(bool)
    if y.nunique(dropna=True) < 2:
        return None
    try:
        return float(roc_auc_score(y, y_score))
    except ValueError:
        return None


def safe_brier(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    if len(y_true) == 0:
        return None
    return float(brier_score_loss(pd.Series(y_true).astype(bool), pd.Series(y_prob).clip(0.0, 1.0)))


def mean_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    return None if series.dropna().empty else float(series.mean())


def median_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    return None if series.dropna().empty else float(series.median())


def sum_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    return None if series.dropna().empty else float(series.fillna(0.0).sum())


def mean_bool(values: Any, *, threshold: float) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    return None if series.empty else float(series.gt(float(threshold)).mean())


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    out = finite(value)
    return "" if not math.isfinite(out) else f"{out:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
