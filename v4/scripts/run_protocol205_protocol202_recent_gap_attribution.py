"""Protocol205: narrow recent-gap attribution for Protocol202.

Protocol202 is the strongest lifecycle challenger so far, but it missed the
recent_2026 protected block by a small median amount. This runner does not
train or change a model. It explains the recent gap mechanically:

* same entry, different exit timing
* baseline trade skipped by Protocol202 because the model was already holding
* extra Protocol202 trade unavailable to the baseline because the baseline was
  still holding
* side/time/day concentration of the remaining gap

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

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct


LOOP_ID = "v4_aplus_hypothesis_205_protocol202_recent_gap_attribution"
DEFAULT_PROTOCOL202_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
RECENT_SPLIT = "recent_2026"
NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol202-dir", type=Path, default=DEFAULT_PROTOCOL202_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.protocol202_dir / "protocol202_model_serial_trades.csv")
    baseline = load_baseline(args.protocol202_dir / "protocol194_baseline_serial_trades.csv")
    model_recent = model[model["reported_split"].eq(RECENT_SPLIT)].copy()
    baseline_recent = baseline[baseline["reported_split"].eq(RECENT_SPLIT)].copy()
    matched = build_matched(model_recent, baseline_recent)
    baseline_only = build_baseline_only(model_recent, baseline_recent)
    model_only = build_model_only(model_recent, baseline_recent)
    combo_summary = build_combo_summary(model_recent, baseline_recent)
    day_summary = build_day_summary(matched, baseline_only, model_only)
    side_summary = build_dimension_summary(matched, baseline_only, model_only, "right")
    time_summary = build_dimension_summary(matched, baseline_only, model_only, "time_bucket")
    top_rows = top_gap_rows(matched, baseline_only, model_only)
    payload = {
        "protocol": "205_protocol202_recent_gap_attribution",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_protocol202_dir": str(args.protocol202_dir),
        "row_counts": {
            "model_recent_rows": int(len(model_recent)),
            "baseline_recent_rows": int(len(baseline_recent)),
            "matched_rows": int(len(matched)),
            "baseline_only_rows": int(len(baseline_only)),
            "model_only_rows": int(len(model_only)),
        },
        "combo_summary": combo_summary,
        "contribution_summary": contribution_summary(matched, baseline_only, model_only),
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "day_summary": day_summary,
        "top_gap_rows": top_rows,
        "decision": decide(combo_summary, matched, baseline_only, model_only),
        "next_hypothesis": next_hypothesis(combo_summary, side_summary, time_summary),
    }
    matched.to_csv(args.out_dir / "matched_exit_changes.csv", index=False)
    baseline_only.to_csv(args.out_dir / "baseline_only_blocked_or_missed.csv", index=False)
    model_only.to_csv(args.out_dir / "protocol202_only_extra_trades.csv", index=False)
    pd.DataFrame(combo_summary).to_csv(args.out_dir / "combo_seed_summary.csv", index=False)
    pd.DataFrame(day_summary).to_csv(args.out_dir / "day_summary.csv", index=False)
    pd.DataFrame(side_summary).to_csv(args.out_dir / "side_summary.csv", index=False)
    pd.DataFrame(time_summary).to_csv(args.out_dir / "time_bucket_summary.csv", index=False)
    pd.DataFrame(top_rows).to_csv(args.out_dir / "top_gap_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_model(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize(frame, has_model_seed=True)


def load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize(frame, has_model_seed=False)


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
    matched = model.merge(
        baseline[base_cols],
        on="trade_key",
        how="inner",
        suffixes=("_model", "_baseline"),
    )
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
            reason = "blocked_by_protocol202_open_position" if blocker is not None else "not_taken_without_open_position"
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
        reason = "baseline_was_holding" if blocker is not None else "model_extra_free_slot"
        rows.append(
            {
                "category": "protocol202_only",
                "model_seed": int(model_row["model_seed"]),
                "combo_seed": int(model_row["combo_seed"]),
                "entry_seed": int(model_row["entry_seed"]),
                "session": str(model_row["session"]),
                "decision_time": model_row["decision_time"],
                "exit_time": model_row["exit_time"],
                "contract_id": str(model_row["contract_id"]),
                "right": str(model_row["right"]),
                "time_bucket": str(model_row["time_bucket"]),
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
    rows: list[dict[str, Any]] = []
    baseline_by_seed = baseline.groupby("entry_seed")["pnl"].sum().to_dict()
    for combo_seed, group in model.groupby("combo_seed", sort=True):
        entry_seed = int(group["entry_seed"].iloc[0])
        model_pnl = float(group["pnl"].sum())
        baseline_pnl = float(baseline_by_seed.get(entry_seed, 0.0))
        rows.append(
            {
                "combo_seed": int(combo_seed),
                "model_seed": int(group["model_seed"].iloc[0]),
                "entry_seed": entry_seed,
                "model_pnl": model_pnl,
                "baseline_pnl": baseline_pnl,
                "delta": model_pnl - baseline_pnl,
                "model_trades": int(len(group)),
                "baseline_trades": int((baseline["entry_seed"] == entry_seed).sum()),
            }
        )
    return rows


def contribution_summary(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> dict[str, Any]:
    matched_contribution = finite_sum(matched, "contribution")
    baseline_only_contribution = finite_sum(baseline_only, "contribution")
    model_only_contribution = finite_sum(model_only, "contribution")
    return {
        "matched_exit_change_contribution": matched_contribution,
        "baseline_only_contribution": baseline_only_contribution,
        "protocol202_only_contribution": model_only_contribution,
        "net_contribution": matched_contribution + baseline_only_contribution + model_only_contribution,
        "baseline_only_blocked_rows": int((baseline_only.get("reason", pd.Series(dtype=str)) == "blocked_by_protocol202_open_position").sum()) if not baseline_only.empty else 0,
        "protocol202_only_baseline_blocked_rows": int((model_only.get("reason", pd.Series(dtype=str)) == "baseline_was_holding").sum()) if not model_only.empty else 0,
    }


def build_dimension_summary(
    matched: pd.DataFrame,
    baseline_only: pd.DataFrame,
    model_only: pd.DataFrame,
    dimension: str,
) -> list[dict[str, Any]]:
    rows = []
    values = sorted(
        set(matched.get(dimension, pd.Series(dtype=str)).dropna().astype(str))
        | set(baseline_only.get(dimension, pd.Series(dtype=str)).dropna().astype(str))
        | set(model_only.get(dimension, pd.Series(dtype=str)).dropna().astype(str))
    )
    for value in values:
        m = matched[matched[dimension].astype(str).eq(value)] if not matched.empty and dimension in matched else matched.iloc[0:0]
        b = baseline_only[baseline_only[dimension].astype(str).eq(value)] if not baseline_only.empty and dimension in baseline_only else baseline_only.iloc[0:0]
        x = model_only[model_only[dimension].astype(str).eq(value)] if not model_only.empty and dimension in model_only else model_only.iloc[0:0]
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "matched_rows": int(len(m)),
                "baseline_only_rows": int(len(b)),
                "protocol202_only_rows": int(len(x)),
                "matched_contribution": finite_sum(m, "contribution"),
                "baseline_only_contribution": finite_sum(b, "contribution"),
                "protocol202_only_contribution": finite_sum(x, "contribution"),
                "net_contribution": finite_sum(m, "contribution") + finite_sum(b, "contribution") + finite_sum(x, "contribution"),
            }
        )
    return rows


def build_day_summary(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> list[dict[str, Any]]:
    return build_dimension_summary(matched, baseline_only, model_only, "session")


def top_gap_rows(matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if not matched.empty:
        for _, row in matched.nsmallest(12, "contribution").iterrows():
            rows.append(
                {
                    "category": "same_entry_exit_change",
                    "session": row["session"],
                    "time_bucket": row["time_bucket"],
                    "right": row["right"],
                    "contract_id": row["contract_id"],
                    "decision_time": row["decision_time"],
                    "contribution": float(row["contribution"]),
                    "detail": f"model {money(row['pnl_model'])} vs baseline {money(row['pnl_baseline'])}; exit delta {row['exit_delta_minutes']:.1f}m",
                }
            )
    if not baseline_only.empty:
        for _, row in baseline_only.nsmallest(12, "contribution").iterrows():
            rows.append(
                {
                    "category": "baseline_only",
                    "session": row["session"],
                    "time_bucket": row["time_bucket"],
                    "right": row["right"],
                    "contract_id": row["contract_id"],
                    "decision_time": row["decision_time"],
                    "contribution": float(row["contribution"]),
                    "detail": f"missed baseline {money(row['pnl'])}; {row['reason']}; blocker={row.get('blocker_contract_id')}",
                }
            )
    if not model_only.empty:
        for _, row in model_only.nsmallest(12, "contribution").iterrows():
            rows.append(
                {
                    "category": "protocol202_only_loss",
                    "session": row["session"],
                    "time_bucket": row["time_bucket"],
                    "right": row["right"],
                    "contract_id": row["contract_id"],
                    "decision_time": row["decision_time"],
                    "contribution": float(row["contribution"]),
                    "detail": f"extra Protocol202 trade {money(row['pnl'])}; {row['reason']}",
                }
            )
    return sorted(rows, key=lambda item: float(item["contribution"]))[:24]


def decide(combo_summary: list[dict[str, Any]], matched: pd.DataFrame, baseline_only: pd.DataFrame, model_only: pd.DataFrame) -> str:
    del matched, model_only
    deltas = [float(row["delta"]) for row in combo_summary]
    negative_fraction = float(np.mean([delta < 0 for delta in deltas])) if deltas else 0.0
    baseline_only_loss = -finite_sum(baseline_only, "contribution")
    median_delta = float(np.median(deltas)) if deltas else 0.0
    if median_delta < 0 and baseline_only_loss > abs(median_delta):
        return "recent_gap_driven_by_missed_baseline_slot_opportunities"
    if negative_fraction >= 0.50:
        return "recent_gap_broad_across_combo_seeds"
    return "recent_gap_small_and_seed_specific"


def next_hypothesis(combo_summary: list[dict[str, Any]], side_summary: list[dict[str, Any]], time_summary: list[dict[str, Any]]) -> str:
    median_delta = float(np.median([row["delta"] for row in combo_summary])) if combo_summary else 0.0
    worst_side = min(side_summary, key=lambda row: float(row["net_contribution"])) if side_summary else {}
    worst_time = min(time_summary, key=lambda row: float(row["net_contribution"])) if time_summary else {}
    if abs(median_delta) < 5_000:
        return (
            "Do not add an architecture knob for this small recent miss. Keep Protocol202 as lifecycle challenger "
            f"and test a narrow calibration only if the same gap repeats; worst side={worst_side.get('value')}, "
            f"worst time={worst_time.get('value')}."
        )
    return (
        "If repairing Protocol202, target the specific recent gap bucket rather than changing the whole model: "
        f"worst side={worst_side.get('value')}, worst time={worst_time.get('value')}."
    )


def finite_sum(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def write_report(path: Path, payload: dict[str, Any]) -> None:
    c = payload["contribution_summary"]
    combo = payload["combo_summary"]
    median_delta = float(np.median([row["delta"] for row in combo])) if combo else 0.0
    lines = [
        "# Protocol205 Protocol202 Recent Gap Attribution",
        "",
        "No paid data was downloaded. No broker endpoint was called. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Median combo-seed delta vs matching baseline seed: {money(median_delta)}",
        f"- Next hypothesis: {payload['next_hypothesis']}",
        "",
        "## Contribution Summary",
        "",
        f"- Same-entry exit changes: {money(c['matched_exit_change_contribution'])}",
        f"- Baseline-only missed/blocked trades: {money(c['baseline_only_contribution'])}",
        f"- Protocol202-only extra trades: {money(c['protocol202_only_contribution'])}",
        f"- Net contribution: {money(c['net_contribution'])}",
        f"- Baseline-only rows blocked by Protocol202 open position: `{c['baseline_only_blocked_rows']}`",
        f"- Protocol202-only rows where baseline was holding: `{c['protocol202_only_baseline_blocked_rows']}`",
        "",
        "## Combo Seeds",
        "",
        "| combo | model seed | entry seed | model | baseline | delta | model trades | baseline trades |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["combo_summary"], key=lambda item: float(item["delta"])):
        lines.append(
            f"| {row['combo_seed']} | {row['model_seed']} | {row['entry_seed']} | "
            f"{money(row['model_pnl'])} | {money(row['baseline_pnl'])} | {money(row['delta'])} | "
            f"{row['model_trades']} | {row['baseline_trades']} |"
        )
    lines.extend(["", "## Side Summary", "", "| side | net | matched | baseline-only | p202-only |", "|---|---:|---:|---:|---:|"])
    for row in payload["side_summary"]:
        lines.append(
            f"| {row['value']} | {money(row['net_contribution'])} | {money(row['matched_contribution'])} | "
            f"{money(row['baseline_only_contribution'])} | {money(row['protocol202_only_contribution'])} |"
        )
    lines.extend(["", "## Time Bucket Summary", "", "| bucket | net | matched | baseline-only | p202-only |", "|---|---:|---:|---:|---:|"])
    for row in payload["time_bucket_summary"]:
        lines.append(
            f"| {row['value']} | {money(row['net_contribution'])} | {money(row['matched_contribution'])} | "
            f"{money(row['baseline_only_contribution'])} | {money(row['protocol202_only_contribution'])} |"
        )
    lines.extend(["", "## Largest Negative Drivers", "", "| category | session | side | time | contribution | detail |", "|---|---|---|---|---:|---|"])
    for row in payload["top_gap_rows"][:12]:
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
            f"- Protocol202-only rows: `{path.parent / 'protocol202_only_extra_trades.csv'}`",
            f"- Combo seed summary: `{path.parent / 'combo_seed_summary.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
