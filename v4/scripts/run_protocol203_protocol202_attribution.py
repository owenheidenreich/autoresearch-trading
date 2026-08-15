"""Protocol203: attribution for Protocol202 mixed slot-aware result.

Protocol202 fixed the Protocol200 over-holding problem on Q1/March but missed
the recent 2026 baseline by a small amount. This audit explains whether the
remaining gap is side-specific, caused by skipped baseline trades, or caused by
matched trade lifecycle changes.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol201_protocol200_attribution import (
    load_trades,
    matched_rows,
    missed_baseline_rows,
    summarize_group,
    summarize_missed,
)


LOOP_ID = "v4_aplus_hypothesis_203_protocol202_mixed_result_attribution"
DEFAULT_PROTOCOL202_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol202-dir", type=Path, default=DEFAULT_PROTOCOL202_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trades(args.protocol202_dir / "protocol202_model_serial_trades.csv")
    baseline = load_trades(args.protocol202_dir / "protocol194_baseline_serial_trades.csv")
    matched = matched_rows(model, baseline)
    missed = missed_baseline_rows(model, baseline)
    split_summary = summarize_split(matched, missed)
    side_summary = summarize_group(matched, ["reported_split", "right"])
    exit_summary = summarize_group(matched, ["reported_split", "exit_reason_model"])
    missed_summary = summarize_missed(missed)
    payload = {
        "protocol": "203_protocol202_mixed_result_attribution",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_protocol202_dir": str(args.protocol202_dir),
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
        "next_hypothesis": next_hypothesis(split_summary, side_summary),
    }
    matched.to_csv(args.out_dir / "matched_trade_attribution.csv", index=False)
    missed.to_csv(args.out_dir / "missed_baseline_trades.csv", index=False)
    pd.DataFrame(split_summary).to_csv(args.out_dir / "split_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


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
                "net_vs_baseline_proxy": float(group["pnl_delta"].sum()) - (float(split_missed["pnl"].sum()) if not split_missed.empty else 0.0),
            }
        )
    return rows


def decide(split_summary: list[dict[str, Any]]) -> str:
    recent = next((row for row in split_summary if row["reported_split"] == "recent_2026"), None)
    q1 = next((row for row in split_summary if row["reported_split"] == "q1_2026"), None)
    if q1 and q1["matched_pnl_delta"] > 0 and recent and recent["net_vs_baseline_proxy"] < 0:
        return "protocol202_improves_q1_but_recent_gap_is_missed_slot_opportunity"
    return "protocol202_mixed_result_requires_manual_review"


def next_hypothesis(split_summary: list[dict[str, Any]], side_summary: list[dict[str, Any]]) -> str:
    recent_sides = [row for row in side_summary if row.get("reported_split") == "recent_2026"]
    put_delta = sum(float(row["pnl_delta"]) for row in recent_sides if row.get("right") == "P")
    call_delta = sum(float(row["pnl_delta"]) for row in recent_sides if row.get("right") == "C")
    if put_delta < 0 and call_delta >= 0:
        return "Next test should add side-aware slot-aware lifecycle calibration, focused on recent put exits, without changing entry-side knobs."
    return "Next test should use a recurrent holding-state model; the slot-aware target helped, but the MLP still misses recent slot timing."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol203 Protocol202 Mixed Result Attribution",
        "",
        "No paid data was downloaded. No broker endpoint was called. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next hypothesis: {payload['next_hypothesis']}",
        "",
        "## Split Attribution",
        "",
        "| split | matched rows | matched pnl delta | missed baseline pnl | net proxy | mean exit delta | longer-exit share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['matched_rows']} | {money(row['matched_pnl_delta'])} | "
            f"{money(row['missed_baseline_pnl'])} | {money(row['net_vs_baseline_proxy'])} | "
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
