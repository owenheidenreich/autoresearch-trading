"""Attribute challenger overrides from conservative neural strict replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "ATTRIBUTION_UNIFIED_CONSERVATIVE_NEURAL_OVERRIDES_V1"
DEFAULT_REPLAY_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED_OVERRIDE_ATTRIBUTION_V1.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_summary = load_json(args.replay_dir / "summary.json")
    trades = load_trades(args.replay_dir)
    trades = enrich(trades)
    challenger = trades[trades["source"].astype(str).eq("challenger")].copy()
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "attribution / calibrated conservative neural challenger overrides",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(replay_summary),
        "replay_decision": replay_summary.get("decision", "missing"),
        "replay_dir": str(args.replay_dir),
        "stress_totals": stress_totals(replay_summary),
        "challenger_rows": int(len(challenger)),
        "by_split_source": summarize_numeric_by(trades, ["slippage_per_side", "split", "source"], "pnl"),
        "challenger_by_split": summarize_numeric_by(challenger, ["slippage_per_side", "split"], "pnl"),
        "challenger_by_right": summarize_numeric_by(challenger, ["slippage_per_side", "right"], "pnl"),
        "challenger_by_time_bucket": summarize_numeric_by(challenger, ["slippage_per_side", "time_bucket"], "pnl"),
        "challenger_by_premium_bucket": summarize_numeric_by(challenger, ["slippage_per_side", "premium_bucket"], "pnl"),
        "challenger_by_offset_bucket": summarize_numeric_by(challenger, ["slippage_per_side", "offset_bucket"], "pnl"),
        "challenger_by_exit_reason": summarize_numeric_by(challenger, ["slippage_per_side", "exit_reason"], "pnl"),
        "top_challenger_losses": top_trades(challenger, n=12, ascending=True),
        "top_challenger_gains": top_trades(challenger, n=12, ascending=False),
        "diagnosis": diagnose(replay_summary, challenger),
        "next_required_evidence": next_required_evidence(replay_summary),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    write_tables(args.out_dir, payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(replay_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(replay_dir.glob("trades_slippage_*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    out["entry_ask"] = pd.to_numeric(out["entry_ask"], errors="coerce")
    out["entry_premium"] = out["entry_ask"] * 100.0
    out["offset"] = pd.to_numeric(out["offset"], errors="coerce")
    out["duration_minutes"] = pd.to_numeric(out["duration_minutes"], errors="coerce")
    out["predicted_advantage"] = pd.to_numeric(out["predicted_advantage"], errors="coerce")
    out["positive_probability"] = pd.to_numeric(out["positive_probability"], errors="coerce")
    out["tail_probability"] = pd.to_numeric(out["tail_probability"], errors="coerce")
    out["time_bucket"] = out["decision_ts"].map(time_bucket)
    out["premium_bucket"] = out["entry_premium"].map(premium_bucket)
    out["offset_bucket"] = out["offset"].map(offset_bucket)
    return out


def stress_totals(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stress in summary.get("stress_results", []):
        total = dict(stress.get("totals", {}))
        total["slippage_per_side"] = stress.get("slippage_per_side")
        rows.append(total)
    return rows


def summarize_numeric_by(frame: pd.DataFrame, keys: list[str], value: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {key: scalar(values[idx]) for idx, key in enumerate(keys)}
        pnl = pd.to_numeric(group[value], errors="coerce").fillna(0.0)
        row.update(
            {
                "rows": int(len(group)),
                "pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()) if len(group) else 0.0,
                "win_rate": float((pnl > 0.0).mean()) if len(group) else 0.0,
                "median_duration_minutes": median(group, "duration_minutes"),
                "median_predicted_advantage": median(group, "predicted_advantage"),
            }
        )
        rows.append(row)
    return rows


def decide(summary: dict[str, Any]) -> str:
    deltas = []
    for stress in summary.get("stress_results", []):
        for split, item in stress.get("splits", {}).items():
            if split == "total":
                continue
            deltas.append(float(item.get("delta_vs_protocol101_same_scope", 0.0)))
    if deltas and all(value > 0.0 for value in deltas):
        return "override_attribution_positive_all_available_splits_challenge_still_blocked"
    if deltas and any(value > 0.0 for value in deltas):
        return "override_attribution_mixed_split_research_only"
    return "override_attribution_no_replay_improvement_research_only"


def diagnose(summary: dict[str, Any], challenger: pd.DataFrame) -> list[str]:
    out = []
    for stress in summary.get("stress_results", []):
        slip = stress.get("slippage_per_side")
        split_deltas = {
            split: item.get("delta_vs_protocol101_same_scope", 0.0)
            for split, item in stress.get("splits", {}).items()
        }
        losers = [split for split, delta in split_deltas.items() if float(delta) < 0.0]
        winners = [split for split, delta in split_deltas.items() if float(delta) > 0.0]
        out.append(f"At slippage {float(slip):.2f}, split deltas are mixed: winners={winners}, losers={losers}.")
    if not challenger.empty:
        base = challenger[challenger["slippage_per_side"].astype(float).eq(0.0)]
        if not base.empty:
            side = base.groupby("right")["pnl"].sum().sort_values()
            out.append(f"Zero-slippage challenger side PnL: {side.to_dict()}.")
            exits = base.groupby("exit_reason")["pnl"].sum().sort_values()
            out.append(f"Zero-slippage challenger exit-reason PnL: {exits.to_dict()}.")
    return out


def next_required_evidence(summary: dict[str, Any]) -> list[str]:
    decision = decide(summary)
    if decision == "override_attribution_mixed_split_research_only":
        return [
            "Do not promote: explain Q1/Q3 underperformance before any additional training.",
            "Add split-stability constraints or defer logic so Q4/recent gains cannot mask older-block losses.",
            "Keep fill, holdout, live parity, and formal validation gates blocking Protocol101 challenge.",
        ]
    return [
        "Keep Protocol101 as paper default.",
        "Close fill, holdout, live parity, and formal validation gates before challenge claims.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        f"Replay decision: `{payload['replay_decision']}`",
        "",
        "## Stress Totals",
        "",
        "| slippage | PnL | same-scope Protocol101 | delta | trades | challenger entries |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["stress_totals"]:
        lines.append(
            f"| {float(row.get('slippage_per_side', 0.0)):.2f} | {float(row.get('total_pnl', 0.0)):.2f} | "
            f"{float(row.get('protocol101_same_scope_pnl', 0.0)):.2f} | {float(row.get('delta_vs_protocol101_same_scope', 0.0)):.2f} | "
            f"{int(row.get('trades', 0))} | {int(row.get('challenger_entries', 0))} |"
        )
    lines.extend(["", "## Diagnosis", ""])
    lines.extend(f"- {item}" for item in payload["diagnosis"])
    lines.extend(["", "## Challenger By Split", "", table(payload["challenger_by_split"], ["slippage_per_side", "split", "rows", "pnl", "win_rate", "median_duration_minutes"])])
    lines.extend(["", "## Challenger By Exit Reason", "", table(payload["challenger_by_exit_reason"], ["slippage_per_side", "exit_reason", "rows", "pnl", "win_rate", "median_duration_minutes"])])
    lines.extend(["", "## Challenger By Time Bucket", "", table(payload["challenger_by_time_bucket"], ["slippage_per_side", "time_bucket", "rows", "pnl", "win_rate", "median_duration_minutes"])])
    lines.extend(["", "## Top Losses", "", table(payload["top_challenger_losses"], ["slippage_per_side", "split", "session", "decision_time", "right", "offset", "entry_premium", "pnl", "exit_reason", "duration_minutes"])])
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def write_tables(out_dir: Path, payload: dict[str, Any]) -> None:
    for name in [
        "by_split_source",
        "challenger_by_split",
        "challenger_by_right",
        "challenger_by_time_bucket",
        "challenger_by_premium_bucket",
        "challenger_by_offset_bucket",
        "challenger_by_exit_reason",
        "top_challenger_losses",
        "top_challenger_gains",
    ]:
        pd.DataFrame(payload.get(name, [])).to_csv(out_dir / f"{name}.csv", index=False)


def top_trades(frame: pd.DataFrame, *, n: int, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    keep = [
        "slippage_per_side",
        "split",
        "session",
        "decision_time",
        "exit_time",
        "right",
        "offset",
        "entry_premium",
        "pnl",
        "predicted_advantage",
        "positive_probability",
        "tail_probability",
        "exit_reason",
        "duration_minutes",
    ]
    rows = []
    ordered = frame.sort_values("pnl", ascending=ascending).head(n)
    for _, row in ordered.iterrows():
        rows.append({key: scalar(row.get(key)) for key in keep})
    return rows


def table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 16) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append("| " + " | ".join(["...", f"{len(rows) - max_rows} more rows", *[""] * max(0, len(columns) - 2)]) + " |")
    return "\n".join(lines)


def time_bucket(value: Any) -> str:
    ts = pd.Timestamp(value) if not pd.isna(value) else pd.NaT
    if pd.isna(ts):
        return "unknown"
    minute = ts.hour * 60 + ts.minute
    if minute < 15 * 60:
        return "pre_1500_utc"
    if minute < 16 * 60:
        return "1500_1559_utc"
    if minute < 18 * 60:
        return "1600_1759_utc"
    if minute < 20 * 60:
        return "1800_1959_utc"
    return "after_2000_utc"


def premium_bucket(value: Any) -> str:
    x = finite(value)
    if x < 1000:
        return "lt_1000"
    if x < 2000:
        return "1000_2000"
    if x < 3000:
        return "2000_3000"
    return "gte_3000"


def offset_bucket(value: Any) -> str:
    x = abs(finite(value))
    if x <= 10:
        return "atm_10"
    if x <= 25:
        return "otm_25"
    return "outer_50"


def median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else 0.0


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return none_if_nan(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    return None if isinstance(missing, (bool, np.bool_)) and missing else value


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def none_if_nan(value: Any) -> float | None:
    x = finite(value, math.nan)
    return None if not math.isfinite(x) else float(x)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    if marker in ledger.read_text():
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
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
