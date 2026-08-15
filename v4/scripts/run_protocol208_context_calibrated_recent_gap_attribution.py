"""AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1.

Historically Protocol208. This is a diagnostic/audit, not a model.

It explains why CHALLENGER_LIFECYCLE_CONTEXT_CALIBRATED_V1 beat the paper
default but still missed the stronger lifecycle baseline on recent_2026.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money


ROLE_LABEL = "AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1"
HISTORICAL_ID = "Protocol208"
CANDIDATE_LABEL = "CHALLENGER_LIFECYCLE_CONTEXT_CALIBRATED_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
OTHER_BASELINE_LABEL = "CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1_WITH_FROZEN_PROTOCOL081_EXITS"
LOOP_ID = "v4_aplus_hypothesis_208_context_calibrated_recent_gap_attribution"
DEFAULT_SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_207_lifecycle_context_calibration")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
RECENT_SPLIT = "recent_2026"
NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = normalize(pd.read_csv(args.source_dir / "challenger_lifecycle_context_calibrated_trades.csv"), has_model_seed=True)
    baseline = normalize(pd.read_csv(args.source_dir / "lifecycle_baseline_trades.csv"), has_model_seed=False)
    model_recent = model[model["reported_split"].eq(RECENT_SPLIT)].copy()
    baseline_recent = baseline[baseline["reported_split"].eq(RECENT_SPLIT)].copy()
    matched = build_matched(model_recent, baseline_recent)
    baseline_only = build_baseline_only(model_recent, baseline_recent)
    model_only = build_model_only(model_recent, baseline_recent)
    combo_summary = build_combo_summary(model_recent, baseline_recent)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / audit",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "other_baseline_label": OTHER_BASELINE_LABEL,
        "data_used": "existing Protocol207 challenger trades and lifecycle baseline trades",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "source_dir": str(args.source_dir),
        "row_counts": {
            "model_recent_rows": int(len(model_recent)),
            "baseline_recent_rows": int(len(baseline_recent)),
            "matched_rows": int(len(matched)),
            "baseline_only_rows": int(len(baseline_only)),
            "model_only_rows": int(len(model_only)),
        },
        "combo_summary": combo_summary,
        "contribution_summary": contribution_summary(matched, baseline_only, model_only),
        "side_summary": build_dimension_summary(matched, baseline_only, model_only, "right"),
        "time_bucket_summary": build_dimension_summary(matched, baseline_only, model_only, "time_bucket"),
        "exit_reason_summary": build_dimension_summary(matched, baseline_only, model_only, "exit_reason_model"),
        "day_summary": build_day_summary(matched, baseline_only, model_only),
        "top_gap_rows": top_gap_rows(matched, baseline_only, model_only),
        "decision": decide(matched, baseline_only, model_only, combo_summary),
        "next_experiment": next_experiment(matched, baseline_only, model_only),
    }
    matched.to_csv(args.out_dir / "matched_exit_changes.csv", index=False)
    baseline_only.to_csv(args.out_dir / "baseline_only_blocked_or_missed.csv", index=False)
    model_only.to_csv(args.out_dir / "challenger_only_extra_trades.csv", index=False)
    pd.DataFrame(payload["combo_summary"]).to_csv(args.out_dir / "combo_seed_summary.csv", index=False)
    pd.DataFrame(payload["side_summary"]).to_csv(args.out_dir / "side_summary.csv", index=False)
    pd.DataFrame(payload["time_bucket_summary"]).to_csv(args.out_dir / "time_bucket_summary.csv", index=False)
    pd.DataFrame(payload["exit_reason_summary"]).to_csv(args.out_dir / "exit_reason_summary.csv", index=False)
    pd.DataFrame(payload["day_summary"]).to_csv(args.out_dir / "day_summary.csv", index=False)
    pd.DataFrame(payload["top_gap_rows"]).to_csv(args.out_dir / "top_gap_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def normalize(frame: pd.DataFrame, *, has_model_seed: bool) -> pd.DataFrame:
    out = frame.copy()
    out["decision_dt"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_dt"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["hold_minutes"] = (out["exit_dt"] - out["decision_dt"]).dt.total_seconds() / 60.0
    out["session"] = out["session"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["right"] = out["right"].astype(str)
    out["entry_seed"] = pd.to_numeric(out["entry_seed"], errors="coerce").fillna(0).astype(int)
    if has_model_seed:
        out["model_seed"] = pd.to_numeric(out["model_seed"], errors="coerce").fillna(0).astype(int)
        out["combo_seed"] = pd.to_numeric(out["combo_seed"], errors="coerce").fillna(0).astype(int)
    for column in ["pnl", "score", "offset", "entry_ask", "entry_premium"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["time_bucket"] = [time_bucket(ts) for ts in out["decision_dt"]]
    out["trade_key"] = trade_key(out)
    return out


def trade_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["entry_seed"].astype(str)
        + "|"
        + frame["session"].astype(str)
        + "|"
        + frame["decision_time"].astype(str)
        + "|"
        + frame["contract_id"].astype(str)
    )


def time_bucket(timestamp: pd.Timestamp) -> str:
    local = pd.Timestamp(timestamp).tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    if minutes < 10 * 60:
        return "first_30"
    if minutes < 11 * 60 + 30:
        return "post_open_morning"
    if minutes < 13 * 60 + 30:
        return "midday"
    return "late_afternoon"


def build_matched(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "trade_key",
        "exit_time",
        "exit_dt",
        "hold_minutes",
        "pnl",
        "exit_reason",
        "account_equity_before",
        "account_equity_after",
    ]
    matched = model.merge(baseline[base_cols], on="trade_key", how="inner", suffixes=("_model", "_baseline"))
    matched["exit_delta_minutes"] = (
        pd.to_datetime(matched["exit_dt_model"], utc=True) - pd.to_datetime(matched["exit_dt_baseline"], utc=True)
    ).dt.total_seconds() / 60.0
    matched["pnl_delta"] = matched["pnl_model"] - matched["pnl_baseline"]
    matched["contribution"] = matched["pnl_delta"]
    matched["category"] = "same_entry_exit_change"
    return matched


def build_baseline_only(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_seed, model_seed_frame in model.groupby("model_seed", sort=True):
        keys = set(model_seed_frame["trade_key"].astype(str))
        for _, base in baseline.iterrows():
            if str(base["trade_key"]) in keys:
                continue
            blocker = active_trade_at(model_seed_frame, int(base["entry_seed"]), str(base["session"]), pd.Timestamp(base["decision_dt"]))
            reason = "blocked_by_challenger_open_position" if blocker is not None else "not_taken_without_open_position"
            rows.append(
                {
                    "category": "baseline_only",
                    "model_seed": int(model_seed),
                    "combo_seed": int(model_seed) * 100 + int(base["entry_seed"]),
                    "entry_seed": int(base["entry_seed"]),
                    "session": str(base["session"]),
                    "decision_time": base["decision_time"],
                    "exit_time": base["exit_time"],
                    "contract_id": str(base["contract_id"]),
                    "right": str(base["right"]),
                    "time_bucket": str(base["time_bucket"]),
                    "exit_reason": str(base.get("exit_reason", "")),
                    "pnl": float(base["pnl"]),
                    "contribution": -float(base["pnl"]),
                    "reason": reason,
                    "blocker_contract_id": None if blocker is None else str(blocker["contract_id"]),
                    "blocker_right": None if blocker is None else str(blocker["right"]),
                    "blocker_decision_time": None if blocker is None else str(blocker["decision_time"]),
                    "blocker_exit_time": None if blocker is None else str(blocker["exit_time"]),
                    "blocker_pnl": None if blocker is None else float(blocker["pnl"]),
                    "blocker_same_side": None if blocker is None else bool(str(blocker["right"]) == str(base["right"])),
                }
            )
    return pd.DataFrame(rows)


def build_model_only(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_keys = set(baseline["trade_key"].astype(str))
    for _, model_row in model.iterrows():
        if str(model_row["trade_key"]) in baseline_keys:
            continue
        blocker = active_trade_at(baseline, int(model_row["entry_seed"]), str(model_row["session"]), pd.Timestamp(model_row["decision_dt"]))
        reason = "baseline_was_holding" if blocker is not None else "challenger_extra_free_slot"
        rows.append(
            {
                "category": "challenger_only",
                "model_seed": int(model_row["model_seed"]),
                "combo_seed": int(model_row["combo_seed"]),
                "entry_seed": int(model_row["entry_seed"]),
                "session": str(model_row["session"]),
                "decision_time": model_row["decision_time"],
                "exit_time": model_row["exit_time"],
                "contract_id": str(model_row["contract_id"]),
                "right": str(model_row["right"]),
                "time_bucket": str(model_row["time_bucket"]),
                "exit_reason": str(model_row.get("exit_reason", "")),
                "pnl": float(model_row["pnl"]),
                "contribution": float(model_row["pnl"]),
                "reason": reason,
                "baseline_blocker_contract_id": None if blocker is None else str(blocker["contract_id"]),
                "baseline_blocker_right": None if blocker is None else str(blocker["right"]),
                "baseline_blocker_decision_time": None if blocker is None else str(blocker["decision_time"]),
                "baseline_blocker_exit_time": None if blocker is None else str(blocker["exit_time"]),
                "baseline_blocker_pnl": None if blocker is None else float(blocker["pnl"]),
                "blocker_same_side": None if blocker is None else bool(str(blocker["right"]) == str(model_row["right"])),
            }
        )
    return pd.DataFrame(rows)


def active_trade_at(frame: pd.DataFrame, entry_seed: int, session: str, timestamp: pd.Timestamp) -> pd.Series | None:
    candidates = frame[
        frame["entry_seed"].eq(entry_seed)
        & frame["session"].eq(session)
        & (frame["decision_dt"] <= timestamp)
        & (frame["exit_dt"] > timestamp)
    ].copy()
    if candidates.empty:
        return None
    return candidates.sort_values("decision_dt").iloc[-1]


def build_combo_summary(model: pd.DataFrame, baseline: pd.DataFrame) -> list[dict[str, Any]]:
    baseline_by_seed = baseline.groupby("entry_seed")["pnl"].sum().to_dict()
    rows = []
    for combo_seed, group in model.groupby("combo_seed", sort=True):
        model_seed = int(group["model_seed"].iloc[0])
        entry_seed = int(group["entry_seed"].iloc[0])
        model_pnl = float(group["pnl"].sum())
        baseline_pnl = float(baseline_by_seed.get(entry_seed, 0.0))
        rows.append(
            {
                "combo_seed": int(combo_seed),
                "model_seed": model_seed,
                "entry_seed": entry_seed,
                "model_pnl": model_pnl,
                "baseline_pnl": baseline_pnl,
                "delta": model_pnl - baseline_pnl,
                "model_trades": int(len(group)),
                "baseline_trades": int((baseline["entry_seed"] == entry_seed).sum()),
            }
        )
    return sorted(rows, key=lambda row: row["delta"])


def contribution_summary(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> dict[str, Any]:
    same_entry = finite_sum(matched, "contribution")
    baseline_contribution = finite_sum(baseline_only, "contribution")
    model_contribution = finite_sum(model_only, "contribution")
    return {
        "matched_exit_change_contribution": same_entry,
        "baseline_only_contribution": baseline_contribution,
        "challenger_only_contribution": model_contribution,
        "net_contribution": same_entry + baseline_contribution + model_contribution,
        "baseline_only_blocked_rows": int((baseline_only.get("reason", pd.Series(dtype=str)) == "blocked_by_challenger_open_position").sum())
        if not baseline_only.empty
        else 0,
        "challenger_only_baseline_blocked_rows": int((model_only.get("reason", pd.Series(dtype=str)) == "baseline_was_holding").sum())
        if not model_only.empty
        else 0,
    }


def build_dimension_summary(
    matched: pd.DataFrame,
    baseline_only: pd.DataFrame,
    model_only: pd.DataFrame,
    dimension: str,
) -> list[dict[str, Any]]:
    rows = []
    values = set()
    for frame in [matched, baseline_only, model_only]:
        if not frame.empty and dimension in frame.columns:
            values.update(str(v) for v in frame[dimension].dropna().unique())
    # Matched uses suffixes for exit reason.
    if dimension == "exit_reason_model" and not matched.empty:
        values.update(str(v) for v in matched["exit_reason_model"].dropna().unique())
    for value in sorted(values):
        matched_mask = pd.Series(False, index=matched.index)
        if not matched.empty and dimension in matched.columns:
            matched_mask = matched[dimension].astype(str) == value
        elif dimension == "exit_reason_model" and not matched.empty:
            matched_mask = matched["exit_reason_model"].astype(str) == value
        base_mask = pd.Series(False, index=baseline_only.index)
        if not baseline_only.empty:
            base_col = "exit_reason" if dimension == "exit_reason_model" else dimension
            if base_col in baseline_only.columns:
                base_mask = baseline_only[base_col].astype(str) == value
        model_mask = pd.Series(False, index=model_only.index)
        if not model_only.empty:
            model_col = "exit_reason" if dimension == "exit_reason_model" else dimension
            if model_col in model_only.columns:
                model_mask = model_only[model_col].astype(str) == value
        matched_contribution = finite_sum(matched.loc[matched_mask], "contribution")
        baseline_contribution = finite_sum(baseline_only.loc[base_mask], "contribution")
        model_contribution = finite_sum(model_only.loc[model_mask], "contribution")
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "matched_rows": int(matched_mask.sum()),
                "baseline_only_rows": int(base_mask.sum()),
                "challenger_only_rows": int(model_mask.sum()),
                "matched_contribution": matched_contribution,
                "baseline_only_contribution": baseline_contribution,
                "challenger_only_contribution": model_contribution,
                "net_contribution": matched_contribution + baseline_contribution + model_contribution,
            }
        )
    return sorted(rows, key=lambda row: row["net_contribution"])


def build_day_summary(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> list[dict[str, Any]]:
    return build_dimension_summary(matched, baseline_only, model_only, "session")


def top_gap_rows(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for _, row in matched.sort_values("contribution").head(limit).iterrows():
        rows.append(
            {
                "category": "same_entry_exit_change",
                "session": str(row["session"]),
                "decision_time": str(row["decision_time"]),
                "right": str(row["right"]),
                "time_bucket": str(row["time_bucket"]),
                "contract_id": str(row["contract_id"]),
                "contribution": float(row["contribution"]),
                "detail": (
                    f"challenger {money(row['pnl_model'])} vs baseline {money(row['pnl_baseline'])}; "
                    f"exit delta {float(row['exit_delta_minutes']):.1f}m"
                ),
            }
        )
    if not baseline_only.empty:
        for _, row in baseline_only.sort_values("contribution").head(limit).iterrows():
            rows.append(
                {
                    "category": "baseline_only",
                    "session": str(row["session"]),
                    "decision_time": str(row["decision_time"]),
                    "right": str(row["right"]),
                    "time_bucket": str(row["time_bucket"]),
                    "contract_id": str(row["contract_id"]),
                    "contribution": float(row["contribution"]),
                    "detail": f"missed baseline {money(row['pnl'])}; {row['reason']}; blocker={row.get('blocker_contract_id')}",
                }
            )
    if not model_only.empty:
        for _, row in model_only.sort_values("contribution").head(limit).iterrows():
            rows.append(
                {
                    "category": "challenger_only",
                    "session": str(row["session"]),
                    "decision_time": str(row["decision_time"]),
                    "right": str(row["right"]),
                    "time_bucket": str(row["time_bucket"]),
                    "contract_id": str(row["contract_id"]),
                    "contribution": float(row["contribution"]),
                    "detail": f"extra challenger {money(row['pnl'])}; {row['reason']}",
                }
            )
    return sorted(rows, key=lambda row: row["contribution"])[:limit]


def decide(
    matched: pd.DataFrame,
    baseline_only: pd.DataFrame,
    model_only: pd.DataFrame,
    combo_summary: list[dict[str, Any]],
) -> str:
    summary = contribution_summary(matched, baseline_only, model_only)
    median_delta = float(np.median([row["delta"] for row in combo_summary])) if combo_summary else 0.0
    if median_delta >= 0:
        return "no_recent_gap_against_lifecycle_baseline"
    parts = {
        "same_entry": abs(float(summary["matched_exit_change_contribution"])),
        "baseline_only": abs(float(summary["baseline_only_contribution"])),
        "challenger_only": abs(float(summary["challenger_only_contribution"])),
    }
    dominant = max(parts, key=parts.get)
    if dominant == "same_entry":
        return "recent_gap_dominated_by_same_entry_lifecycle_exit_timing"
    if dominant == "baseline_only":
        return "recent_gap_dominated_by_slot_blocked_baseline_winners"
    return "recent_gap_dominated_by_extra_bad_challenger_trades"


def next_experiment(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> str:
    summary = contribution_summary(matched, baseline_only, model_only)
    same_entry = float(summary["matched_exit_change_contribution"])
    baseline_only_contribution = float(summary["baseline_only_contribution"])
    challenger_only = float(summary["challenger_only_contribution"])
    if same_entry < 0 and abs(same_entry) >= abs(baseline_only_contribution) and abs(same_entry) >= abs(challenger_only):
        return (
            "Build CHALLENGER_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1. The repeated failure is not candidate "
            "selection alone; the model needs one sequence objective that learns entry, hold, and exit together."
        )
    if baseline_only_contribution < 0:
        return (
            "Build a slot-value entry objective that penalizes entering positions that block better later "
            "opportunities, then pair it with the frozen lifecycle baseline before changing exits again."
        )
    return "Attribute extra challenger-only losers by side/time/premium before adding any new model knob."


def finite_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def write_report(path: Path, payload: dict[str, Any]) -> None:
    c = payload["contribution_summary"]
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        f"Does it change the paper-trading default: {'yes' if payload['changes_paper_default'] else 'no'}",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Contribution Summary",
        "",
        f"- Same-entry exit changes: {money(c['matched_exit_change_contribution'])}",
        f"- Baseline-only missed/blocked trades: {money(c['baseline_only_contribution'])}",
        f"- Challenger-only extra trades: {money(c['challenger_only_contribution'])}",
        f"- Net contribution: {money(c['net_contribution'])}",
        f"- Baseline-only rows blocked by challenger open position: `{c['baseline_only_blocked_rows']}`",
        f"- Challenger-only rows where baseline was holding: `{c['challenger_only_baseline_blocked_rows']}`",
        "",
        "## Combo Seeds",
        "",
        "| combo | model seed | entry seed | challenger | lifecycle baseline | delta | challenger trades | baseline trades |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["combo_summary"]:
        lines.append(
            f"| {row['combo_seed']} | {row['model_seed']} | {row['entry_seed']} | "
            f"{money(row['model_pnl'])} | {money(row['baseline_pnl'])} | {money(row['delta'])} | "
            f"{row['model_trades']} | {row['baseline_trades']} |"
        )
    for title, key in [("Side Summary", "side_summary"), ("Time Bucket Summary", "time_bucket_summary"), ("Exit Reason Summary", "exit_reason_summary")]:
        lines.extend(["", f"## {title}", "", "| value | net | matched | baseline-only | challenger-only |", "|---|---:|---:|---:|---:|"])
        for row in payload[key]:
            lines.append(
                f"| {row['value']} | {money(row['net_contribution'])} | {money(row['matched_contribution'])} | "
                f"{money(row['baseline_only_contribution'])} | {money(row['challenger_only_contribution'])} |"
            )
    lines.extend(["", "## Largest Negative Drivers", "", "| category | session | side | time | contribution | detail |", "|---|---|---|---|---:|---|"])
    for row in payload["top_gap_rows"]:
        lines.append(
            f"| {row['category']} | {row['session']} | {row['right']} | {row['time_bucket']} | "
            f"{money(row['contribution'])} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Matched exit changes: `{path.parent / 'matched_exit_changes.csv'}`",
            f"- Baseline-only rows: `{path.parent / 'baseline_only_blocked_or_missed.csv'}`",
            f"- Challenger-only rows: `{path.parent / 'challenger_only_extra_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

