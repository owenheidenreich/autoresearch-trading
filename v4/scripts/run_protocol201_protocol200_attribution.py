"""Protocol201: attribution for Protocol200 lifecycle continuation failure.

Protocol200 was profitable but failed to beat the frozen Protocol194/081 serial
baseline. This audit explains the failure before another objective is added.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct


LOOP_ID = "v4_aplus_hypothesis_201_protocol200_failure_attribution"
DEFAULT_PROTOCOL200_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_200_lifecycle_continuation_policy")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol200-dir", type=Path, default=DEFAULT_PROTOCOL200_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trades(args.protocol200_dir / "protocol200_model_serial_trades.csv")
    baseline = load_trades(args.protocol200_dir / "protocol194_baseline_serial_trades.csv")
    matched = matched_rows(model, baseline)
    missed = missed_baseline_rows(model, baseline)
    split_summary = summarize_split(matched, missed)
    side_summary = summarize_group(matched, ["reported_split", "right"])
    exit_summary = summarize_group(matched, ["reported_split", "exit_reason_model"])
    missed_summary = summarize_missed(missed)
    payload = {
        "protocol": "201_protocol200_failure_attribution",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_protocol200_dir": str(args.protocol200_dir),
        "row_counts": {
            "model_rows": int(len(model)),
            "baseline_rows": int(len(baseline)),
            "matched_rows": int(len(matched)),
            "missed_baseline_rows": int(len(missed)),
        },
        "split_summary": split_summary,
        "side_summary": side_summary,
        "exit_reason_summary": exit_summary,
        "missed_baseline_summary": missed_summary,
        "decision": decide(split_summary),
        "next_hypothesis": (
            "Protocol202 should train lifecycle continuation against a slot-aware target: hold only when "
            "continuing the current contract beats exiting now and preserving the single position slot "
            "for future candidate entries."
        ),
    }
    matched.to_csv(args.out_dir / "matched_trade_attribution.csv", index=False)
    missed.to_csv(args.out_dir / "missed_baseline_trades.csv", index=False)
    pd.DataFrame(split_summary).to_csv(args.out_dir / "split_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["hold_minutes"] = (frame["exit_dt"] - frame["decision_dt"]).dt.total_seconds() / 60.0
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    frame["entry_seed"] = pd.to_numeric(frame["entry_seed"], errors="coerce").fillna(0).astype(int)
    return frame


def matched_rows(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["reported_split", "entry_seed", "session", "decision_time", "contract_id"]
    model_cols = keys + ["model_seed", "combo_seed", "exit_time", "hold_minutes", "pnl", "exit_reason", "right"]
    base_cols = keys + ["exit_time", "hold_minutes", "pnl", "exit_reason"]
    matched = model[model_cols].merge(
        baseline[base_cols],
        on=keys,
        how="inner",
        suffixes=("_model", "_baseline"),
    )
    matched["exit_delta_minutes"] = (
        pd.to_datetime(matched["exit_time_model"], utc=True) - pd.to_datetime(matched["exit_time_baseline"], utc=True)
    ).dt.total_seconds() / 60.0
    matched["pnl_delta"] = matched["pnl_model"] - matched["pnl_baseline"]
    return matched


def missed_baseline_rows(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["reported_split", "entry_seed", "session", "decision_time", "contract_id"]
    rows = []
    for model_seed, seed_model in model.groupby("model_seed", sort=True):
        model_keys = set(map(tuple, seed_model[keys].astype(str).to_numpy().tolist()))
        for _, row in baseline.iterrows():
            key = tuple(str(row[column]) for column in keys)
            if key not in model_keys:
                rows.append({**row.to_dict(), "model_seed": int(model_seed)})
    return pd.DataFrame(rows)


def summarize_split(matched: pd.DataFrame, missed: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split, group in matched.groupby("reported_split", sort=True):
        split_missed = missed[missed["reported_split"].eq(split)] if not missed.empty else missed
        rows.append(
            {
                "reported_split": str(split),
                "matched_rows": int(len(group)),
                "matched_pnl_delta": float(group["pnl_delta"].sum()),
                "matched_median_pnl_delta": float(group["pnl_delta"].median()),
                "mean_exit_delta_minutes": float(group["exit_delta_minutes"].mean()),
                "median_exit_delta_minutes": float(group["exit_delta_minutes"].median()),
                "model_longer_exit_fraction": float((group["exit_delta_minutes"] > 0).mean()),
                "missed_baseline_rows": int(len(split_missed)),
                "missed_baseline_pnl": float(split_missed["pnl"].sum()) if not split_missed.empty else 0.0,
            }
        )
    return rows


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: str(value) for column, value in zip(group_cols, key)}
        row.update(
            {
                "rows": int(len(group)),
                "pnl_delta": float(group["pnl_delta"].sum()),
                "median_exit_delta_minutes": float(group["exit_delta_minutes"].median()),
                "model_longer_exit_fraction": float((group["exit_delta_minutes"] > 0).mean()),
            }
        )
        rows.append(row)
    return rows


def summarize_missed(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for (split, model_seed), group in frame.groupby(["reported_split", "model_seed"], sort=True):
        rows.append(
            {
                "reported_split": str(split),
                "model_seed": int(model_seed),
                "missed_rows": int(len(group)),
                "missed_pnl": float(group["pnl"].sum()),
            }
        )
    return rows


def decide(split_summary: list[dict[str, Any]]) -> str:
    if not split_summary:
        return "blocked_no_matched_rows"
    missed_positive = [row["missed_baseline_pnl"] > 0 for row in split_summary]
    fewer_due_longer = [row["mean_exit_delta_minutes"] > 0 for row in split_summary]
    if all(missed_positive) and any(fewer_due_longer):
        return "protocol200_overheld_and_blocked_profitable_later_entries"
    return "protocol200_failure_mixed_requires_manual_review"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol201 Protocol200 Failure Attribution",
        "",
        "No paid data was downloaded. No broker endpoint was called. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next hypothesis: {payload['next_hypothesis']}",
        "",
        "## Split Attribution",
        "",
        "| split | matched rows | matched pnl delta | missed baseline pnl | missed rows | mean exit delta | longer-exit share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['matched_rows']} | {money(row['matched_pnl_delta'])} | "
            f"{money(row['missed_baseline_pnl'])} | {row['missed_baseline_rows']} | "
            f"{row['mean_exit_delta_minutes']:.1f}m | {pct(row['model_longer_exit_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Matched attribution: `{path.parent / 'matched_trade_attribution.csv'}`",
            f"- Missed baseline trades: `{path.parent / 'missed_baseline_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
