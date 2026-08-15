"""Foundation hardening audit before more model work.

This runner turns the recommendations in ``v4/docs/engineer-response.md`` into
a repeatable implementation audit. It does not train models, download paid
data, or call broker endpoints. Its main job is to explain why Protocol276
failed as the first unified entry/lifecycle replay, then preserve the current
foundation blockers before new experiments resume.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROLE_LABEL = "FOUNDATION_HARDENING_READINESS_PACKET_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/foundation_hardening_review")
DEFAULT_DOC_PATH = Path("v4/docs/FOUNDATION_HARDENING_AUDIT.md")
ENGINEER_RESPONSE = Path("v4/docs/engineer-response.md")
PROTOCOL266 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_266_protocol265_artifact_reproduction")
PROTOCOL269 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_269_protocol265_no_order_runtime_parity")
PROTOCOL270 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset")
PROTOCOL271 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy")
PROTOCOL272 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness")
PROTOCOL273 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk")
PROTOCOL274 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset")
PROTOCOL276 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_276_integrated_entry_lifecycle_serial_replay")
PROTOCOL276_ATTRIBUTION = Path("v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution")
UNIFIED_CONSERVATIVE_FOUNDATION = Path("v4/audit/autoresearch/unified_conservative_offline_policy_foundation")
UNIFIED_TRAJECTORY_FOUNDATION = Path("v4/audit/autoresearch/unified_policy_trajectory_foundation")
UNTOUCHED_HOLDOUT_RESERVATION = Path("v4/audit/autoresearch/unified_untouched_holdout_reservation")
UNIFIED_NEURAL_TRAINING_READINESS = Path("v4/audit/autoresearch/unified_neural_training_readiness")
UNIFIED_PROTOCOL101_BASELINE_ATTACHMENT = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment")
UNIFIED_SERIAL_DP_ORACLE = Path("v4/audit/autoresearch/unified_serial_dp_oracle")
UNIFIED_CONSERVATIVE_NEURAL_POLICY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_v1")
UNIFIED_CONSERVATIVE_STRICT_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1")
UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC = Path("v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic")
UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1")
UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC_FLAT_CALIBRATED = Path("v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic_flat_calibrated_v1")
UNIFIED_CONSERVATIVE_FLAT_CALIBRATED_STRICT_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1")
UNIFIED_CONSERVATIVE_FLAT_CALIBRATED_OVERRIDE_ATTRIBUTION = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1")
UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE = Path("v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1")
UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY = Path("v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation")
UNIFIED_SLOT_OPPORTUNITY_COST_LABELS = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset")
UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator")
UNIFIED_SLOT_OPPORTUNITY_LEARNED_OVERLAY_REPLAY = Path("v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3")
PREREGISTERED_LEARNED_DEFER_POLICY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1")
PREREGISTERED_LEARNED_DEFER_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1")
SPLIT_ORDER = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries()
    trades = load_csv(PROTOCOL276 / "integrated_entry_lifecycle_trades.csv")
    skips = load_csv(PROTOCOL276 / "integrated_entry_lifecycle_skips.csv")
    entry_only = load_csv(PROTOCOL271 / "unified_action_advantage_model_trades.csv")
    flat_labels = load_csv(PROTOCOL270 / "split_summary.csv")
    hold_labels = load_csv(PROTOCOL274 / "split_summary.csv")

    trades = enrich_trades(trades, entry_only)
    entry_only = enrich_entry_only(entry_only)
    skips = enrich_skips(skips)

    strict_comparison = strict_protocol101_comparison(summaries["protocol276"])
    protocol276_attribution = attribution_payload(trades, skips, entry_only)
    label_alignment = label_alignment_payload(trades, flat_labels, hold_labels)
    foundation_checks = build_foundation_checks(summaries, protocol276_attribution, label_alignment)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "implementation audit / foundation hardening packet",
        "source_recommendation_doc": str(ENGINEER_RESPONSE),
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(foundation_checks, summaries),
        "near_term_direction": near_term_direction(summaries),
        "strict_protocol101_comparison": strict_comparison,
        "protocol276_failure_attribution": protocol276_attribution,
        "label_alignment": label_alignment,
        "foundation_checks": foundation_checks,
        "recommendations_status": recommendation_status(summaries),
        "prioritized_checklist": prioritized_checklist(),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    write_csv_outputs(args.out_dir, protocol276_attribution)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_summaries() -> dict[str, dict[str, Any]]:
    return {
        "protocol266": load_json(PROTOCOL266 / "summary.json"),
        "protocol269": load_json(PROTOCOL269 / "summary.json"),
        "protocol270": load_json(PROTOCOL270 / "summary.json"),
        "protocol271": load_json(PROTOCOL271 / "summary.json"),
        "protocol272": load_json(PROTOCOL272 / "summary.json"),
        "protocol273": load_json(PROTOCOL273 / "summary.json"),
        "protocol274": load_json(PROTOCOL274 / "summary.json"),
        "protocol276": load_json(PROTOCOL276 / "summary.json"),
        "protocol276_attribution": load_json(PROTOCOL276_ATTRIBUTION / "summary.json"),
        "unified_conservative_foundation": load_json(UNIFIED_CONSERVATIVE_FOUNDATION / "summary.json"),
        "unified_trajectory_foundation": load_json(UNIFIED_TRAJECTORY_FOUNDATION / "summary.json"),
        "untouched_holdout_reservation": load_json(UNTOUCHED_HOLDOUT_RESERVATION / "summary.json"),
        "unified_neural_training_readiness": load_json(UNIFIED_NEURAL_TRAINING_READINESS / "summary.json"),
        "unified_protocol101_baseline_attachment": load_json(UNIFIED_PROTOCOL101_BASELINE_ATTACHMENT / "summary.json"),
        "unified_serial_dp_oracle": load_json(UNIFIED_SERIAL_DP_ORACLE / "summary.json"),
        "unified_conservative_neural_policy": load_json(UNIFIED_CONSERVATIVE_NEURAL_POLICY / "summary.json"),
        "unified_conservative_strict_replay": load_json(UNIFIED_CONSERVATIVE_STRICT_REPLAY / "summary.json"),
        "unified_conservative_flat_gate_diagnostic": load_json(UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC / "summary.json"),
        "unified_conservative_neural_policy_flat_calibrated": load_json(UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED / "summary.json"),
        "unified_conservative_flat_gate_diagnostic_flat_calibrated": load_json(UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC_FLAT_CALIBRATED / "summary.json"),
        "unified_conservative_flat_calibrated_strict_replay": load_json(UNIFIED_CONSERVATIVE_FLAT_CALIBRATED_STRICT_REPLAY / "summary.json"),
        "unified_conservative_flat_calibrated_override_attribution": load_json(UNIFIED_CONSERVATIVE_FLAT_CALIBRATED_OVERRIDE_ATTRIBUTION / "summary.json"),
        "unified_conservative_q1_q3_underperformance": load_json(UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE / "summary.json"),
        "unified_slot_opportunity_defer_overlay": load_json(UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY / "summary.json"),
        "unified_slot_opportunity_cost_labels": load_json(UNIFIED_SLOT_OPPORTUNITY_COST_LABELS / "summary.json"),
        "unified_slot_opportunity_cost_estimator": load_json(UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR / "summary.json"),
        "unified_slot_opportunity_learned_overlay_replay": load_json(UNIFIED_SLOT_OPPORTUNITY_LEARNED_OVERLAY_REPLAY / "summary.json"),
        "preregistered_learned_defer_policy": load_json(PREREGISTERED_LEARNED_DEFER_POLICY / "summary.json"),
        "preregistered_learned_defer_replay": load_json(PREREGISTERED_LEARNED_DEFER_REPLAY / "summary.json"),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def enrich_trades(trades: pd.DataFrame, entry_only: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    frame = trades.copy()
    for column in [
        "duration_minutes",
        "entry_premium",
        "pnl",
        "a_enter",
        "mfe_to_exit",
        "mae_to_exit",
        "giveback_from_mfe",
        "entry_model_margin",
        "entry_model_threshold",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["time_bucket"] = frame["decision_ts"].map(time_bucket)
    frame["premium_bucket"] = frame["entry_premium"].map(premium_bucket)
    frame["entry_advantage_bucket"] = frame["a_enter"].map(advantage_bucket)
    frame["pnl_bucket"] = np.where(pd.to_numeric(frame["pnl"], errors="coerce") >= 0, "winner", "loser")

    if not entry_only.empty and "candidate_uid" in entry_only.columns:
        base = entry_only[["candidate_uid", "exit_time", "pnl"]].copy()
        base = base.rename(columns={"exit_time": "entry_only_exit_time", "pnl": "entry_only_pnl"})
        base["entry_only_exit_ts"] = pd.to_datetime(base["entry_only_exit_time"], utc=True, errors="coerce")
        base["entry_only_pnl"] = pd.to_numeric(base["entry_only_pnl"], errors="coerce")
        frame = frame.merge(base, on="candidate_uid", how="left")
        frame["entry_only_duration_minutes"] = (
            frame["entry_only_exit_ts"] - frame["decision_ts"]
        ).dt.total_seconds().div(60.0)
        frame["duration_delta_vs_entry_only_minutes"] = frame["duration_minutes"] - frame["entry_only_duration_minutes"]
        frame["lifecycle_delta_vs_entry_only_pnl"] = frame["pnl"] - frame["entry_only_pnl"]
    else:
        frame["entry_only_pnl"] = np.nan
        frame["duration_delta_vs_entry_only_minutes"] = np.nan
        frame["lifecycle_delta_vs_entry_only_pnl"] = np.nan
    frame["lifecycle_failure_mode"] = frame.apply(lifecycle_failure_mode, axis=1)
    return frame


def enrich_entry_only(entry_only: pd.DataFrame) -> pd.DataFrame:
    if entry_only.empty:
        return entry_only
    frame = entry_only.copy()
    if "decision_time" in frame.columns:
        frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    else:
        frame["decision_ts"] = pd.NaT
    frame["time_bucket"] = frame["decision_ts"].map(time_bucket)
    frame["entry_premium"] = numeric_column(frame, "entry_premium")
    frame["pnl"] = numeric_column(frame, "pnl")
    frame["a_enter"] = numeric_column(frame, "a_enter")
    frame["premium_bucket"] = frame["entry_premium"].map(premium_bucket)
    frame["entry_advantage_bucket"] = frame["a_enter"].map(advantage_bucket)
    return frame


def enrich_skips(skips: pd.DataFrame) -> pd.DataFrame:
    if skips.empty:
        return skips
    frame = skips.copy()
    for column in ["entry_premium", "account_equity", "offset"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "entry_premium" in frame.columns:
        frame["premium_bucket"] = frame["entry_premium"].map(premium_bucket)
    else:
        frame["premium_bucket"] = "unknown"
    return frame


def time_bucket(value: Any) -> str:
    ts = pd.Timestamp(value) if not pd.isna(value) else pd.NaT
    if pd.isna(ts):
        return "unknown"
    if ts.tzinfo is None:
        local = ts.tz_localize("UTC").tz_convert(NY)
    else:
        local = ts.tz_convert(NY)
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60:
        return "first_30m"
    if minute < 11 * 60 + 30:
        return "post_open_morning"
    if minute < 13 * 60 + 30:
        return "midday"
    if minute < 15 * 60 + 30:
        return "late_afternoon"
    return "no_new_entry_window"


def premium_bucket(value: Any) -> str:
    x = finite(value)
    if not math.isfinite(x):
        return "unknown"
    if x < 500:
        return "lt_500"
    if x < 1000:
        return "500_1000"
    if x < 2000:
        return "1000_2000"
    if x < 3500:
        return "2000_3500"
    return "gte_3500"


def advantage_bucket(value: Any) -> str:
    x = finite(value)
    if not math.isfinite(x):
        return "unknown"
    if x <= -500:
        return "strong_negative"
    if x < 0:
        return "negative"
    if x < 500:
        return "positive_lt_500"
    return "positive_gte_500"


def lifecycle_failure_mode(row: pd.Series) -> str:
    reason = str(row.get("exit_reason", ""))
    pnl = finite(row.get("pnl"))
    giveback = finite(row.get("giveback_from_mfe"), 0.0)
    mfe = finite(row.get("mfe_to_exit"), 0.0)
    duration_delta = finite(row.get("duration_delta_vs_entry_only_minutes"))
    if reason == "forced_flat_no_lifecycle_exit_signal":
        return "overheld_to_forced_flat"
    if math.isfinite(duration_delta) and duration_delta > 30.0:
        return "overheld_vs_entry_only_exit"
    if math.isfinite(duration_delta) and duration_delta < -5.0:
        return "early_exit_vs_entry_only_exit"
    if pnl < 0.0 and mfe > 250.0 and giveback > 500.0:
        return "gave_back_prior_mfe_then_lost"
    if pnl < 0.0:
        return "losing_exit"
    return "profitable_or_neutral_exit"


def strict_protocol101_comparison(summary: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate = summary.get("aggregate", {})
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        item = aggregate.get(split, {})
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        rows.append(
            {
                "split": split,
                "protocol276_pnl": finite(item.get("median_total_pnl")),
                "protocol101_pnl": finite(item.get("frozen_protocol101_total_pnl")),
                "delta_vs_protocol101": finite(item.get("median_delta_vs_frozen_protocol101")),
                "trades": finite(item.get("median_trades")),
                "win_rate": finite(item.get("median_win_rate")),
                "profit_factor": finite(item.get("median_profit_factor")),
                "account_drawdown": finite(item.get("median_account_drawdown")),
                "beats_protocol101": bool(item.get("beats_frozen_protocol101")),
            }
        )
    return rows


def attribution_payload(trades: pd.DataFrame, skips: pd.DataFrame, entry_only: pd.DataFrame) -> dict[str, Any]:
    return {
        "trade_rows": int(len(trades)),
        "skip_rows": int(len(skips)),
        "integrated_by_split": summarize_numeric_by(trades, ["reported_split"], "pnl"),
        "skip_reasons_by_split": summarize_skips(skips),
        "lifecycle_failure_modes": summarize_lifecycle_modes(trades),
        "side_buckets": summarize_numeric_by(trades, ["reported_split", "right"], "pnl"),
        "premium_buckets": summarize_numeric_by(trades, ["reported_split", "premium_bucket"], "pnl"),
        "time_buckets": summarize_numeric_by(trades, ["reported_split", "time_bucket"], "pnl"),
        "entry_advantage_buckets": summarize_numeric_by(trades, ["reported_split", "entry_advantage_bucket"], "pnl"),
        "entry_policy_bridge": entry_policy_bridge(trades, entry_only),
        "top_loss_trades": top_trades(trades, n=12, ascending=True),
        "top_gain_trades": top_trades(trades, n=8, ascending=False),
        "diagnosis": diagnose_protocol276(trades, skips),
    }


def summarize_numeric_by(frame: pd.DataFrame, keys: list[str], value: str) -> list[dict[str, Any]]:
    if frame.empty or value not in frame.columns:
        return []
    working = frame.copy()
    working[value] = pd.to_numeric(working[value], errors="coerce").fillna(0.0)
    rows = []
    for group_key, group in working.groupby(keys, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {key: str(key_values[idx]) for idx, key in enumerate(keys)}
        row.update(
            {
                "rows": int(len(group)),
                "pnl": float(group[value].sum()),
                "avg_pnl": float(group[value].mean()) if len(group) else 0.0,
                "median_pnl": float(group[value].median()) if len(group) else 0.0,
                "win_rate": float((group[value] > 0.0).mean()) if len(group) else 0.0,
            }
        )
        if "lifecycle_delta_vs_entry_only_pnl" in group.columns:
            delta = pd.to_numeric(group["lifecycle_delta_vs_entry_only_pnl"], errors="coerce")
            if delta.notna().any():
                row["lifecycle_delta_vs_entry_only_pnl"] = float(delta.sum())
        if "duration_minutes" in group.columns:
            row["median_duration_minutes"] = median_numeric(group, "duration_minutes")
        if "giveback_from_mfe" in group.columns:
            row["median_giveback_from_mfe"] = median_numeric(group, "giveback_from_mfe")
        rows.append(row)
    return rows


def summarize_skips(skips: pd.DataFrame) -> list[dict[str, Any]]:
    if skips.empty:
        return []
    rows = []
    for keys, group in skips.groupby(["reported_split", "skip_reason"], dropna=False, sort=True):
        split, reason = keys
        rows.append(
            {
                "reported_split": str(split),
                "skip_reason": str(reason),
                "rows": int(len(group)),
                "median_entry_premium": median_numeric(group, "entry_premium"),
                "median_account_equity": median_numeric(group, "account_equity"),
            }
        )
    return rows


def summarize_lifecycle_modes(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows = summarize_numeric_by(trades, ["reported_split", "lifecycle_failure_mode"], "pnl")
    return sorted(rows, key=lambda row: (row["reported_split"], row["pnl"]))


def entry_policy_bridge(trades: pd.DataFrame, entry_only: pd.DataFrame) -> dict[str, Any]:
    if entry_only.empty:
        return {"entry_only_rows": 0, "note": "Protocol271 entry-only trade file missing or empty."}
    trade_uids = set(trades.get("candidate_uid", pd.Series(dtype=str)).astype(str))
    entry = entry_only.copy()
    entry["integrated_trade_present"] = entry["candidate_uid"].astype(str).isin(trade_uids)
    missed = entry[~entry["integrated_trade_present"]].copy()
    common = trades[trades["candidate_uid"].astype(str).isin(set(entry["candidate_uid"].astype(str)))].copy()
    return {
        "entry_only_rows": int(len(entry)),
        "integrated_rows_with_entry_only_match": int(len(common)),
        "entry_only_rows_not_integrated": int(len(missed)),
        "entry_only_pnl_not_integrated": float(pd.to_numeric(missed.get("pnl"), errors="coerce").fillna(0.0).sum()) if not missed.empty else 0.0,
        "common_lifecycle_delta_vs_entry_only_pnl": float(pd.to_numeric(common.get("lifecycle_delta_vs_entry_only_pnl"), errors="coerce").fillna(0.0).sum()) if not common.empty else 0.0,
        "missed_by_split": summarize_numeric_by(missed, ["reported_split"], "pnl") if not missed.empty else [],
    }


def top_trades(trades: pd.DataFrame, *, n: int, ascending: bool) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    ordered = trades.sort_values("pnl", ascending=ascending).head(n)
    keep = [
        "reported_split",
        "session",
        "decision_time",
        "exit_time",
        "contract_id",
        "right",
        "entry_premium",
        "pnl",
        "a_enter",
        "exit_reason",
        "lifecycle_failure_mode",
        "duration_minutes",
        "giveback_from_mfe",
    ]
    rows = []
    for _, row in ordered.iterrows():
        rows.append({key: scalar(row.get(key)) for key in keep})
    return rows


def diagnose_protocol276(trades: pd.DataFrame, skips: pd.DataFrame) -> list[str]:
    diagnosis = []
    if not skips.empty:
        counts = skips["skip_reason"].astype(str).value_counts().to_dict()
        if counts.get("unaffordable_current_equity", 0) > 0:
            diagnosis.append("Account-state/affordability skipped many candidate rows; this must be explained before blaming architecture.")
        if counts.get("missing_contract_quotes", 0) > 0:
            diagnosis.append("Quote-path coverage gaps remain in the integrated lifecycle replay.")
    if not trades.empty:
        modes = trades["lifecycle_failure_mode"].astype(str).value_counts().to_dict()
        if modes.get("overheld_to_forced_flat", 0) or modes.get("overheld_vs_entry_only_exit", 0):
            diagnosis.append("Lifecycle integration appears to overhold many positions versus the entry-only exit path.")
        negative_adv = trades[pd.to_numeric(trades.get("a_enter"), errors="coerce") < 0]
        if len(negative_adv):
            diagnosis.append("The entry policy still takes many candidates with negative oracle entry advantage.")
    if not diagnosis:
        diagnosis.append("No dominant failure mode detected from available artifacts; collect richer attribution before retraining.")
    return diagnosis


def label_alignment_payload(trades: pd.DataFrame, flat_labels: pd.DataFrame, hold_labels: pd.DataFrame) -> dict[str, Any]:
    return {
        "integrated_trade_label_mismatch": integrated_label_mismatch(trades),
        "flat_action_label_summary": records(flat_labels),
        "hold_exit_label_summary": records(hold_labels),
        "interpretation": [
            "Flat labels show most decision events are wait-dominant; entry recall must be calibrated carefully.",
            "Hold/exit labels are heavily hold-positive, so lifecycle clones can easily overhold without fill/slippage and live timing evidence.",
            "Protocol276 results prove the current label/model integration is not yet deployment-aligned.",
        ],
    }


def integrated_label_mismatch(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    a_enter = pd.to_numeric(trades.get("a_enter"), errors="coerce")
    pnl = pd.to_numeric(trades.get("pnl"), errors="coerce").fillna(0.0)
    positive_label_loss = trades[(a_enter > 0.0) & (pnl < 0.0)]
    negative_label_taken = trades[a_enter < 0.0]
    return {
        "trades": int(len(trades)),
        "positive_a_enter_losing_trades": int(len(positive_label_loss)),
        "positive_a_enter_losing_pnl": float(pnl.loc[positive_label_loss.index].sum()) if len(positive_label_loss) else 0.0,
        "negative_a_enter_trades_taken": int(len(negative_label_taken)),
        "negative_a_enter_trades_pnl": float(pnl.loc[negative_label_taken.index].sum()) if len(negative_label_taken) else 0.0,
        "median_a_enter": none_if_nan(a_enter.median()),
        "mean_a_enter": none_if_nan(a_enter.mean()),
    }


def build_foundation_checks(
    summaries: dict[str, dict[str, Any]],
    attribution: dict[str, Any],
    label_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    fill = summaries["protocol272"].get("readiness", {})
    overfit_decision = str(summaries["protocol273"].get("decision", ""))
    p269 = summaries["protocol269"]
    p276 = summaries["protocol276"]
    p276_attr = summaries.get("protocol276_attribution", {})
    holdout = summaries.get("untouched_holdout_reservation", {})
    readiness = summaries.get("unified_neural_training_readiness", {})
    trained_policy = summaries.get("unified_conservative_neural_policy", {})
    strict_replay = summaries.get("unified_conservative_strict_replay", {})
    flat_gate = summaries.get("unified_conservative_flat_gate_diagnostic", {})
    calibrated_gate = summaries.get("unified_conservative_flat_gate_diagnostic_flat_calibrated", {})
    calibrated_replay = summaries.get("unified_conservative_flat_calibrated_strict_replay", {})
    override_attribution = summaries.get("unified_conservative_flat_calibrated_override_attribution", {})
    q1_q3_underperformance = summaries.get("unified_conservative_q1_q3_underperformance", {})
    slot_overlay = summaries.get("unified_slot_opportunity_defer_overlay", {})
    slot_labels = summaries.get("unified_slot_opportunity_cost_labels", {})
    slot_estimator = summaries.get("unified_slot_opportunity_cost_estimator", {})
    slot_replay = summaries.get("unified_slot_opportunity_learned_overlay_replay", {})
    prereg_policy = summaries.get("preregistered_learned_defer_policy", {})
    prereg_replay = summaries.get("preregistered_learned_defer_replay", {})
    label_mismatch = label_alignment.get("integrated_trade_label_mismatch", {})
    attribution_complete = str(p276_attr.get("decision", "")).startswith("protocol276_failure_attribution_complete")
    holdout_reserved = str(holdout.get("decision", "")).startswith("untouched_holdout_reserved")
    slot_labels_ready = slot_labels.get("decision") == "slot_opportunity_cost_labels_ready_for_causal_estimator"
    slot_estimator_ready = slot_estimator.get("decision") == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay"
    slot_replay_ready = slot_replay.get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training"
    prereg_policy_ready = prereg_policy.get("decision") == "conservative_neural_policy_trained_replay_and_challenge_still_blocked"
    prereg_replay_ready = prereg_replay.get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training"
    return [
        check(
            "Protocol276 failure attribution",
            "pass" if attribution_complete else "partial",
            (
                "Attribution packet complete; fixes are still required before retraining."
                if attribution_complete
                else "Packet generated, but root-cause fix is not complete."
            ),
            (
                "; ".join(p276_attr.get("root_cause_summary", []))
                if attribution_complete
                else f"{attribution['trade_rows']} trades and {attribution['skip_rows']} skips audited."
            ),
        ),
        check(
            "Fill model calibration",
            "blocked" if fill.get("status") == "blocked_insufficient_fill_observations" else "partial",
            "No calibrated stochastic fill model may be used until enough observed fills exist.",
            f"Fill observations: {fill.get('fill_observations', 0)} / {fill.get('required_fill_observations', 30)}.",
        ),
        check(
            "Untouched holdout reservation",
            "pass" if holdout_reserved else "blocked" if "reserve_new_untouched_block" in overfit_decision else "partial",
            "Current repeated-research blocks are diagnostic only, not sacred holdouts.",
            str(holdout.get("decision") or summaries["protocol273"].get("decision", "missing")),
        ),
        check(
            "Unified neural training readiness",
            "pass"
            if readiness.get("next_training_decision")
            == "next_training_paused_preregistered_run_complete_protocol101_challenge_blocked"
            else "blocked"
            if readiness.get("training_decision") == "neural_training_not_ready_foundation_gates_blocked"
            else "pass"
            if readiness.get("training_decision")
            else "blocked",
            "Training readiness must either allow the one preregistered run or pause after that run is complete.",
            str(readiness.get("next_training_decision", readiness.get("training_decision", "missing"))),
        ),
        check(
            "Challenger runtime parity",
            "partial" if p269.get("historical_replay_proxy") else "blocked",
            "Protocol265 parity exists as historical proxy; live no-order parity remains required.",
            str(p269.get("decision", "missing")),
        ),
        check(
            "Unified label/policy alignment",
            "pass"
            if readiness.get("next_training_decision")
            == "next_training_paused_preregistered_run_complete_protocol101_challenge_blocked"
            or readiness.get("training_decision") == "neural_training_ready_for_preregistered_conservative_policy_run"
            else "blocked"
            if p276.get("decision") == "research_only_integrated_entry_lifecycle_does_not_surpass_protocol101"
            else "partial",
            "Protocol276 remains rejected; the new baseline-aligned serial DP oracle is the approved training formulation.",
            str(readiness.get("training_decision", f"Negative-A_enter trades taken: {label_mismatch.get('negative_a_enter_trades_taken', 'unknown')}.")),
        ),
        check(
            "Conservative neural policy V1 strict replay",
            "partial"
            if strict_replay.get("strict_replay_run")
            else "blocked"
            if trained_policy.get("model_training")
            else "blocked",
            "First trained policy was replayed strictly, but it produced no challenger overrides and cannot challenge Protocol101.",
            str(strict_replay.get("decision", trained_policy.get("decision", "missing"))),
        ),
        check(
            "Flat-entry gate calibration",
            "pass"
            if calibrated_gate.get("decision") == "flat_entry_gate_produces_model_overrides_needs_replay_attribution"
            else "blocked"
            if flat_gate.get("decision") == "flat_entry_gate_overconservative_zero_model_overrides"
            else "partial"
            if flat_gate.get("decision")
            else "blocked",
            "Flat calibration repair must create overrides without relying on ad hoc threshold loosening.",
            str(calibrated_gate.get("decision") or flat_gate.get("decision", "missing")),
        ),
        check(
            "Flat-calibrated strict replay",
            "partial" if calibrated_replay.get("strict_replay_run") else "blocked",
            "The calibrated repair produced overrides and positive same-scope total PnL, but it remains research-only.",
            str(calibrated_replay.get("decision", "missing")),
        ),
        check(
            "Override split stability",
            "blocked"
            if override_attribution.get("decision") == "override_attribution_mixed_split_research_only"
            else "partial"
            if override_attribution.get("decision")
            else "blocked",
            "Q4/recent gains must not mask Q1/Q3 underperformance; add stability/defer constraints before more training.",
            "; ".join(override_attribution.get("diagnosis", [])) if override_attribution.get("diagnosis") else str(override_attribution.get("decision", "missing")),
        ),
        check(
            "Q1/Q3 underperformance attribution",
            "pass"
            if q1_q3_underperformance.get("decision") == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost"
            else "blocked",
            "The mixed-split failure is explained by single-slot opportunity cost versus missed Protocol101 entries.",
            "; ".join(q1_q3_underperformance.get("q1_q3_diagnosis", [])) if q1_q3_underperformance.get("q1_q3_diagnosis") else str(q1_q3_underperformance.get("decision", "missing")),
        ),
        check(
            "Slot opportunity-cost defer overlay",
            "pass"
            if slot_replay_ready
            else "partial"
            if slot_overlay.get("decision") == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator"
            else "blocked",
            "Oracle target exists; learned replay status is tracked in the estimator and learned-overlay gates.",
            str(slot_overlay.get("decision", "missing")),
        ),
        check(
            "Causal opportunity-cost labels",
            "pass" if slot_labels_ready else "blocked",
            "Use candidate-level blocked-Protocol101 cost labels only as training targets, never as runtime features.",
            str(slot_labels.get("decision", "missing")),
        ),
        check(
            "Causal opportunity-cost estimator",
            "pass" if slot_estimator_ready else "blocked",
            "Train/calibrate the causal estimator; oracle realized blocked PnL is forbidden in live policy inputs.",
            (
                f"{slot_estimator.get('decision')}; q1_auc={slot_estimator.get('metrics', {}).get('q1_2026', {}).get('positive_auc')}"
                if slot_estimator_ready
                else f"Labels ready: {slot_labels.get('row_counts', {}).get('candidate_rows', 0)} candidate rows."
                if slot_labels_ready
                else str(slot_labels.get("decision", "slot opportunity-cost labels missing"))
            ),
        ),
        check(
            "Learned slot-opportunity overlay replay",
            "pass" if slot_replay_ready else "blocked",
            "Replay the learned estimator as a strict defer overlay before any further neural policy training.",
            str(slot_replay.get("decision", "missing")),
        ),
        check(
            "Preregistered learned-defer neural run",
            "pass" if prereg_policy_ready else "blocked",
            "Exactly one preregistered neural training run is allowed after the learned defer overlay is frozen.",
            str(prereg_policy.get("decision", "missing")),
        ),
        check(
            "Preregistered learned-defer replay",
            "pass" if prereg_replay_ready else "blocked",
            "The preregistered policy must pass strict one-account replay with Q1/Q3 nonnegative stress deltas.",
            preregistered_replay_evidence(prereg_replay),
        ),
        check(
            "Additional neural experiments",
            "blocked" if prereg_replay_ready else "pass",
            "Pause model search after the preregistered run; remaining work is simulator, validation, parity, and fill evidence.",
            "Single preregistered run completed." if prereg_replay_ready else "Preregistered run not complete.",
        ),
        check(
            "Paper default status",
            "pass",
            "Protocol101 remains the guarded paper default; challengers remain research-only.",
            "No paper default change is made by this packet.",
        ),
        check(
            "Broad paid data expansion",
            "blocked",
            "Do not buy broad historical data until simulator, label, parity, and holdout gates are stable.",
            "Current evidence supports staged data-value tests only after blockers close.",
        ),
    ]


def check(name: str, status: str, required_action: str, evidence: str) -> dict[str, str]:
    return {"name": name, "status": status, "evidence": evidence, "required_action": required_action}


def preregistered_replay_evidence(summary: dict[str, Any]) -> str:
    if not summary:
        return "missing"
    pieces = [str(summary.get("decision", "missing"))]
    for stress in summary.get("stress_results", []):
        slip = stress.get("slippage_per_side")
        totals = stress.get("totals", {})
        pieces.append(
            f"slippage={slip}: challenger={totals.get('challenger_entries', 0)}, "
            f"delta={money(totals.get('delta_vs_protocol101_same_scope', 0.0))}"
        )
    return "; ".join(pieces)


def recommendation_status(summaries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"recommendation": "Persist and reproduce Protocol265 artifacts", "status": "implemented", "evidence": str(summaries["protocol266"].get("decision", "missing"))},
        {"recommendation": "Attribute Protocol265 extension regimes", "status": "partially_implemented", "evidence": "Protocol268 exists, but many regime buckets are missing/constant."},
        {"recommendation": "Build Protocol265 no-order parity", "status": "partially_implemented", "evidence": str(summaries["protocol269"].get("decision", "missing"))},
        {"recommendation": "Build full-surface action-advantage labels", "status": "implemented", "evidence": str(summaries["protocol270"].get("decision", "missing"))},
        {"recommendation": "Train unified action-advantage policy", "status": "implemented_but_failed_replacement", "evidence": str(summaries["protocol271"].get("decision", "missing"))},
        {"recommendation": "Build hold/exit opportunity-cost labels", "status": "implemented", "evidence": str(summaries["protocol274"].get("decision", "missing"))},
        {"recommendation": "Integrate entry and lifecycle replay", "status": "implemented_but_failed_replacement", "evidence": str(summaries["protocol276"].get("decision", "missing"))},
        {
            "recommendation": "Attribute Protocol276 integrated replay failure",
            "status": "implemented" if summaries.get("protocol276_attribution", {}).get("decision") else "not_addressed",
            "evidence": str(summaries.get("protocol276_attribution", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Freeze unified conservative offline policy direction",
            "status": "implemented" if summaries.get("unified_conservative_foundation", {}).get("decision") else "not_addressed",
            "evidence": str(summaries.get("unified_conservative_foundation", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Build unified policy trajectory foundation",
            "status": "implemented_training_blocked"
            if summaries.get("unified_trajectory_foundation", {}).get("decision")
            else "not_addressed",
            "evidence": str(summaries.get("unified_trajectory_foundation", {}).get("decision", "missing")),
        },
        {"recommendation": "Calibrate stochastic fill model", "status": "blocked", "evidence": str(summaries["protocol272"].get("decision", "missing"))},
        {
            "recommendation": "Reserve new untouched holdout",
            "status": "implemented_pending_data"
            if summaries.get("untouched_holdout_reservation", {}).get("decision")
            else "blocked",
            "evidence": str(
                summaries.get("untouched_holdout_reservation", {}).get(
                    "decision",
                    summaries["protocol273"].get("decision", "missing"),
                )
            ),
        },
        {
            "recommendation": "Add unified neural training readiness gate",
            "status": "implemented_training_ready"
            if summaries.get("unified_neural_training_readiness", {}).get("training_decision")
            == "neural_training_ready_for_preregistered_conservative_policy_run"
            else "implemented_training_blocked"
            if summaries.get("unified_neural_training_readiness", {}).get("training_decision")
            else "not_addressed",
            "evidence": str(summaries.get("unified_neural_training_readiness", {}).get("training_decision", "missing")),
        },
        {
            "recommendation": "Attach Protocol101 baseline actions to unified trajectory",
            "status": "partially_implemented"
            if str(summaries.get("unified_protocol101_baseline_attachment", {}).get("decision", "")).startswith(
                "protocol101_baseline_attachment_partial"
            )
            else "implemented"
            if summaries.get("unified_protocol101_baseline_attachment", {}).get("decision")
            else "not_addressed",
            "evidence": str(summaries.get("unified_protocol101_baseline_attachment", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Materialize unified serial DP oracle training scope",
            "status": "implemented"
            if summaries.get("unified_serial_dp_oracle", {}).get("decision")
            == "unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope"
            else "not_addressed",
            "evidence": str(summaries.get("unified_serial_dp_oracle", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Train first conservative neural policy on frozen scope",
            "status": "implemented_but_abstention_policy",
            "evidence": str(summaries.get("unified_conservative_neural_policy", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Run strict replay for trained conservative neural policy",
            "status": "implemented_but_no_challenger_overrides"
            if summaries.get("unified_conservative_strict_replay", {}).get("strict_replay_run")
            else "not_addressed",
            "evidence": str(summaries.get("unified_conservative_strict_replay", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Diagnose conservative flat-entry abstention",
            "status": "implemented_retraining_formulation_fix_required"
            if summaries.get("unified_conservative_flat_gate_diagnostic", {}).get("decision")
            else "not_addressed",
            "evidence": str(summaries.get("unified_conservative_flat_gate_diagnostic", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Run flat-calibrated conservative neural repair",
            "status": "implemented_mixed_split_research_only",
            "evidence": str(summaries.get("unified_conservative_neural_policy_flat_calibrated", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Replay flat-calibrated conservative neural repair",
            "status": "implemented_mixed_split_research_only",
            "evidence": str(summaries.get("unified_conservative_flat_calibrated_strict_replay", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Attribute flat-calibrated challenger overrides",
            "status": "implemented_blocks_promotion"
            if summaries.get("unified_conservative_flat_calibrated_override_attribution", {}).get("decision")
            == "override_attribution_mixed_split_research_only"
            else "not_addressed",
            "evidence": str(summaries.get("unified_conservative_flat_calibrated_override_attribution", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Explain flat-calibrated Q1/Q3 underperformance",
            "status": "implemented_opportunity_cost_failure_identified"
            if summaries.get("unified_conservative_q1_q3_underperformance", {}).get("decision")
            == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost"
            else "not_addressed",
            "evidence": str(summaries.get("unified_conservative_q1_q3_underperformance", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Define slot-opportunity-cost defer overlay target",
            "status": "implemented_oracle_target_only"
            if summaries.get("unified_slot_opportunity_defer_overlay", {}).get("decision")
            == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator"
            else "not_addressed",
            "evidence": str(summaries.get("unified_slot_opportunity_defer_overlay", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Materialize causal blocked-Protocol101 opportunity-cost labels",
            "status": "implemented_ready_for_estimator"
            if summaries.get("unified_slot_opportunity_cost_labels", {}).get("decision")
            == "slot_opportunity_cost_labels_ready_for_causal_estimator"
            else "not_addressed",
            "evidence": str(summaries.get("unified_slot_opportunity_cost_labels", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Train causal blocked-Protocol101 opportunity-cost estimator",
            "status": "implemented_ready_for_overlay_replay"
            if summaries.get("unified_slot_opportunity_cost_estimator", {}).get("decision")
            == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay"
            else "not_addressed",
            "evidence": str(summaries.get("unified_slot_opportunity_cost_estimator", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Replay learned slot-opportunity-cost defer overlay",
            "status": "implemented_ready_for_next_preregistered_training"
            if summaries.get("unified_slot_opportunity_learned_overlay_replay", {}).get("decision")
            == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training"
            else "not_addressed",
            "evidence": str(summaries.get("unified_slot_opportunity_learned_overlay_replay", {}).get("decision", "missing")),
        },
        {
            "recommendation": "Run exactly one preregistered learned-defer neural policy",
            "status": "implemented_replay_passed_diagnostic_stress"
            if summaries.get("preregistered_learned_defer_replay", {}).get("decision")
            == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training"
            else "not_addressed",
            "evidence": str(summaries.get("preregistered_learned_defer_replay", {}).get("decision", "missing")),
        },
    ]


def prioritized_checklist() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "item": "Stop neural experiments after the completed preregistered run",
            "required_evidence": "The preregistered learned-defer replay passed diagnostic stress; additional model work now waits on promotion-grade foundation blockers.",
        },
        {
            "priority": "2",
            "item": "Collect fill evidence before stochastic fill replay",
            "required_evidence": "Paper/no-order observations sufficient for calibration; until then use deterministic ask/bid plus `$0.10`/`$0.25` stress only.",
        },
        {
            "priority": "3",
            "item": "Upgrade challenger runtime parity to live no-order",
            "required_evidence": "Full candidate breadth, freshness, Greeks, masks, account state, action schema, and latency logged live without broker orders.",
        },
        {
            "priority": "4",
            "item": "Collect and freeze the untouched evaluation block",
            "required_evidence": "A named block that has not influenced feature, threshold, objective, architecture, sizing, exit, or model choices.",
        },
        {
            "priority": "5",
            "item": "Add formal validation controls",
            "required_evidence": "PBO/CSCV-style false-discovery controls or an equivalent strategy-matrix audit before any better-than-Protocol101 claim.",
        },
        {
            "priority": "6",
            "item": "Freeze a promotion packet before scoring new data",
            "required_evidence": "Fixed artifacts, fixed learned defer overlay, fixed metrics, and Protocol101 as the strict serial baseline before touching the reserved block.",
        },
        {
            "priority": "7",
            "item": "Keep deterministic stress as the approved offline replay assumption",
            "required_evidence": "No stochastic fill replay or paper-default discussion until fill observations support calibration.",
        },
        {
            "priority": "8",
            "item": "Defer broad historical data purchase",
            "required_evidence": "A staged data-value test only after simulator, labels, live parity, and untouched validation are stable.",
        },
    ]


def near_term_direction(summaries: dict[str, dict[str, Any]]) -> str:
    prereg_replay = summaries.get("preregistered_learned_defer_replay", {})
    if prereg_replay.get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return (
            "The single preregistered learned-defer neural run is complete and passed diagnostic strict replay against "
            "same-scope Protocol101 under all deterministic stress levels. Stop neural experiments here: the remaining "
            "blockers are calibrated fill evidence, untouched holdout data, live no-order parity, and formal validation controls."
        )
    slot_replay = summaries.get("unified_slot_opportunity_learned_overlay_replay", {})
    if slot_replay.get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return (
            "The foundation is ready for one preregistered next neural policy run with the learned slot-opportunity defer "
            "overlay frozen in advance. This is not a Protocol101 challenge claim: recent 2026 still underperforms in the "
            "overlay replay, and fill calibration, untouched holdout data, live no-order parity, and formal validation remain blocked."
        )
    slot_estimator = summaries.get("unified_slot_opportunity_cost_estimator", {})
    if slot_estimator.get("decision") == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay":
        q1_auc = slot_estimator.get("metrics", {}).get("q1_2026", {}).get("positive_auc")
        return (
            "Stay in learned slot-opportunity overlay replay work. The causal estimator is trained and ready for strict "
            f"defer-overlay replay (Q1 positive-cost AUC {fmt(q1_auc)}), so the next allowed work is to charge estimated "
            "Protocol101 slot cost plus uncertainty before any challenger override can consume the single slot."
        )
    slot_labels = summaries.get("unified_slot_opportunity_cost_labels", {})
    if slot_labels.get("decision") == "slot_opportunity_cost_labels_ready_for_causal_estimator":
        rows = slot_labels.get("row_counts", {}).get("candidate_rows", 0)
        return (
            "Stay in causal slot-opportunity-cost estimator work. Candidate-level blocked-Protocol101 labels now exist "
            f"over {rows} rows, so the next allowed work is to train/calibrate the defer estimator using only current-state "
            "features, then replay it as a strict overlay before any broader neural policy run."
        )
    slot_overlay = summaries.get("unified_slot_opportunity_defer_overlay", {})
    if slot_overlay.get("decision") == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator":
        return (
            "Stay in causal slot-opportunity-cost estimator work. The oracle overlay target repairs Q1/Q3, "
            "but it uses realized blocked Protocol101 PnL and is diagnostic only. The next allowed work is to "
            "materialize causal blocked-Protocol101 opportunity-cost labels and train/calibrate a small defer estimator."
        )
    q1_q3 = summaries.get("unified_conservative_q1_q3_underperformance", {})
    if q1_q3.get("decision") == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost":
        return (
            "Stay in slot-opportunity-cost repair mode. The flat-calibrated policy's Q1/Q3 losses are now explained: "
            "challenger overrides consumed the single slot and missed stronger Protocol101 entries. The next allowed "
            "work is a preregistered defer overlay that prices missed Protocol101 opportunity cost before any challenger "
            "override can replace or block the baseline."
        )
    override_attr = summaries.get("unified_conservative_flat_calibrated_override_attribution", {})
    calibrated_replay = summaries.get("unified_conservative_flat_calibrated_strict_replay", {})
    if override_attr.get("decision") == "override_attribution_mixed_split_research_only":
        return (
            "Stay in split-stability and conservative-defer repair mode. The flat-calibrated policy produced "
            "78 challenger overrides and positive same-scope total PnL under slippage stress, but attribution "
            "is mixed: Q4/recent win while Q1/Q3 lose to same-scope Protocol101. This is research-only evidence, "
            "not a challenge packet; the next allowed work is to explain the unstable older-block losses and "
            "pre-register stricter defer/stability constraints."
        )
    if calibrated_replay.get("strict_replay_run"):
        return (
            "Attribute the flat-calibrated overrides by split, side, timing, premium, moneyness, and exit reason "
            "before considering more training. Protocol101 remains the paper default."
        )
    flat_gate = summaries.get("unified_conservative_flat_gate_diagnostic", {})
    strict_replay = summaries.get("unified_conservative_strict_replay", {})
    if flat_gate.get("decision") == "flat_entry_gate_overconservative_zero_model_overrides":
        return (
            "Stay in targeted formulation-repair mode. The first conservative neural policy trained and replayed "
            "strictly, but it deferred to Protocol101 everywhere because the flat advantage head never crossed "
            "the override margin. The next allowed model work is a preregistered flat-entry calibration/loss-balance "
            "fix with gate-component diagnostics before replay; Protocol101 remains the paper default."
        )
    if strict_replay.get("strict_replay_run"):
        return (
            "Use the strict replay packet to attribute any challenger overrides before further training. "
            "Protocol101 remains the paper default, and no challenge claim is allowed until fill, holdout, "
            "live parity, and formal validation gates close."
        )
    readiness = summaries.get("unified_neural_training_readiness", {})
    if readiness.get("training_decision") == "neural_training_ready_for_preregistered_conservative_policy_run":
        return (
            "Proceed only to a preregistered conservative neural training run on the frozen baseline-aligned "
            "serial DP oracle scope. Protocol101 remains the paper default, and no better-than-Protocol101 "
            "claim is allowed until fill, live no-order parity, untouched holdout data, and formal validation close."
        )
    return (
        "Stay in foundation-hardening mode. Complete Protocol276 attribution, fill evidence, "
        "untouched holdout reservation, live no-order parity, and label-alignment checks before "
        "running more open-ended model experiments."
    )


def decide(checks: list[dict[str, str]], summaries: dict[str, dict[str, Any]] | None = None) -> str:
    if summaries and summaries.get("preregistered_learned_defer_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return "foundation_preregistered_neural_run_complete_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_slot_opportunity_learned_overlay_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return "foundation_ready_for_next_preregistered_neural_training_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_slot_opportunity_cost_estimator", {}).get("decision") == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay":
        return "foundation_slot_opportunity_estimator_ready_learned_overlay_replay_required_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_slot_opportunity_cost_labels", {}).get("decision") == "slot_opportunity_cost_labels_ready_for_causal_estimator":
        return "foundation_slot_opportunity_labels_ready_causal_estimator_required_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_slot_opportunity_defer_overlay", {}).get("decision") == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator":
        return "foundation_slot_opportunity_oracle_target_ready_causal_estimator_required_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_conservative_q1_q3_underperformance", {}).get("decision") == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost":
        return "foundation_q1_q3_loss_explained_slot_opportunity_cost_overlay_required_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_conservative_flat_calibrated_override_attribution", {}).get("decision") == "override_attribution_mixed_split_research_only":
        return "foundation_flat_calibrated_replay_mixed_split_research_only_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_conservative_flat_gate_diagnostic", {}).get("decision") == "flat_entry_gate_overconservative_zero_model_overrides":
        return "foundation_training_v1_replay_complete_flat_gate_retraining_required_protocol101_challenge_blocked"
    if summaries and summaries.get("unified_neural_training_readiness", {}).get("training_decision") == "neural_training_ready_for_preregistered_conservative_policy_run":
        if summaries.get("unified_neural_training_readiness", {}).get("protocol101_challenge_decision") != "protocol101_challenge_ready_for_untouched_strict_serial_replay":
            return "foundation_ready_for_preregistered_neural_training_protocol101_challenge_blocked"
        return "foundation_ready_for_protocol101_challenge"
    if any(item["status"] == "blocked" for item in checks):
        return "foundation_hardening_required_before_model_experiments"
    if any(item["status"] == "partial" for item in checks):
        return "foundation_hardening_partially_ready_no_open_ended_experiments"
    return "foundation_ready_for_targeted_preregistered_experiment"


def write_csv_outputs(out_dir: Path, attribution: dict[str, Any]) -> None:
    for name in [
        "skip_reasons_by_split",
        "lifecycle_failure_modes",
        "side_buckets",
        "premium_buckets",
        "time_buckets",
        "entry_advantage_buckets",
    ]:
        pd.DataFrame(attribution.get(name, [])).to_csv(out_dir / f"{name}.csv", index=False)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Bottom Line",
        "",
        payload["near_term_direction"],
        "",
        "## Protocol276 Versus Protocol101",
        "",
        "| split | Protocol276 PnL | Protocol101 PnL | delta | trades | win rate | PF | acct DD | beats? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["strict_protocol101_comparison"]:
        lines.append(
            f"| {row['split']} | {money(row['protocol276_pnl'])} | {money(row['protocol101_pnl'])} | "
            f"{money(row['delta_vs_protocol101'])} | {row['trades']:.0f} | {row['win_rate']:.3f} | "
            f"{row['profit_factor']:.3f} | {money(row['account_drawdown'])} | {bool_word(row['beats_protocol101'])} |"
        )
    attr = payload["protocol276_failure_attribution"]
    lines.extend(
        [
            "",
            "## Failure Attribution",
            "",
            f"- Integrated trade rows: `{attr['trade_rows']}`",
            f"- Skip rows: `{attr['skip_rows']}`",
            "- Diagnosis:",
        ]
    )
    lines.extend(f"  - {item}" for item in attr["diagnosis"])
    lines.extend(["", "### Skip Reasons", "", table(attr["skip_reasons_by_split"], ["reported_split", "skip_reason", "rows", "median_entry_premium", "median_account_equity"])])
    lines.extend(["", "### Lifecycle Modes", "", table(attr["lifecycle_failure_modes"], ["reported_split", "lifecycle_failure_mode", "rows", "pnl", "lifecycle_delta_vs_entry_only_pnl", "median_duration_minutes", "median_giveback_from_mfe"])])
    lines.extend(["", "### Entry Advantage Buckets", "", table(attr["entry_advantage_buckets"], ["reported_split", "entry_advantage_bucket", "rows", "pnl", "win_rate", "median_duration_minutes"])])
    bridge = attr["entry_policy_bridge"]
    lines.extend(
        [
            "",
            "## Entry/Lifecycle Bridge",
            "",
            f"- Protocol271 entry-only rows: `{bridge.get('entry_only_rows', 0)}`",
            f"- Integrated rows with entry-only match: `{bridge.get('integrated_rows_with_entry_only_match', 0)}`",
            f"- Entry-only rows not integrated: `{bridge.get('entry_only_rows_not_integrated', 0)}`",
            f"- Entry-only PnL not integrated: `{money(bridge.get('entry_only_pnl_not_integrated', 0.0))}`",
            f"- Common lifecycle delta versus entry-only exits: `{money(bridge.get('common_lifecycle_delta_vs_entry_only_pnl', 0.0))}`",
        ]
    )
    mismatch = payload["label_alignment"]["integrated_trade_label_mismatch"]
    lines.extend(
        [
            "",
            "## Label Alignment",
            "",
            f"- Negative `A_enter` trades taken: `{mismatch.get('negative_a_enter_trades_taken', 0)}` for `{money(mismatch.get('negative_a_enter_trades_pnl', 0.0))}` PnL.",
            f"- Positive `A_enter` losing trades: `{mismatch.get('positive_a_enter_losing_trades', 0)}` for `{money(mismatch.get('positive_a_enter_losing_pnl', 0.0))}` PnL.",
            f"- Median `A_enter`: `{fmt(mismatch.get('median_a_enter'))}`.",
            "",
            "Flat action labels and hold/exit labels exist, but Protocol276 shows the current integrated policy is not deployment-aligned yet.",
            "",
            "## Recommendation Audit",
            "",
            "| recommendation | status | evidence |",
            "|---|---|---|",
        ]
    )
    for item in payload["recommendations_status"]:
        lines.append(f"| {item['recommendation']} | `{item['status']}` | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Foundation Checklist",
            "",
            "| gate | status | evidence | required action |",
            "|---|---|---|---|",
        ]
    )
    for item in payload["foundation_checks"]:
        lines.append(f"| {item['name']} | `{item['status']}` | {item['evidence']} | {item['required_action']} |")
    lines.extend(
        [
            "",
            "## Prioritized Checklist",
            "",
        ]
    )
    for item in payload["prioritized_checklist"]:
        lines.append(f"{item['priority']}. **{item['item']}**: {item['required_evidence']}")
    lines.extend(
        [
            "",
            "## Allowed Next Work",
            "",
            "- Stop neural/model experiments after the completed preregistered learned-defer run.",
            "- Work the promotion-grade blockers: fill evidence, untouched holdout data, live no-order parity, and formal validation controls.",
            "- Keep Protocol101 as paper default until the challenger passes promotion-grade fill, holdout, parity, and validation gates.",
            "- Keep deterministic ask/bid plus `$0.10`/`$0.25` stress as the approved offline replay assumption until fill observations support calibration.",
            "- Do not run open-ended model experiments, architecture searches, ad hoc threshold loosening, or broad paid-data acquisition while these challenge blockers remain open.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
            f"- Docs copy: `{payload['outputs']['doc']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        overflow = ["..."] + [f"{len(rows) - max_rows} more rows"] + [""] * max(0, len(columns) - 2)
        lines.append("| " + " | ".join(overflow[: len(columns)]) + " |")
    return "\n".join(lines)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: scalar(value) for key, value in row.items()} for row in frame.to_dict("records")]


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    "- Result: Open-ended model experiments remain paused until the blocked foundation gates close.",
                ]
            )
            + "\n"
        )


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return none_if_nan(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def median_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def none_if_nan(value: Any) -> float | None:
    x = finite(value)
    return None if not math.isfinite(x) else float(x)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    x = finite(value)
    return "" if not math.isfinite(x) else f"{x:.2f}"


def money(value: Any) -> str:
    x = finite(value)
    if not math.isfinite(x):
        return ""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
