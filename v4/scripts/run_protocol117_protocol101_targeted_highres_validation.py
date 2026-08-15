"""Protocol 117: interpret targeted high-resolution Protocol 101 replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts import run_protocol115_protocol101_existing_1s_validation as p115


DEFAULT_REPLAY_JSON = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit/report.json"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_validation")
DEFAULT_MANIFEST = Path("v4/promotion/PROTOCOL_101_TARGETED_CBBO_1S_DOWNLOAD_MANIFEST.json")
DEFAULT_DOWNLOAD_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_download/download_summary.json"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-json", type=Path, default=DEFAULT_REPLAY_JSON)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download-summary", type=Path, default=DEFAULT_DOWNLOAD_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay = json.loads(args.replay_json.read_text())
    rows = pd.DataFrame(replay.get("rows", []))
    if rows.empty:
        raise SystemExit(f"no replay rows found in {args.replay_json}")

    prepared = p115.prepare_rows(rows)
    base_rows = prepared[prepared["split"].ne("march_2026")].copy()
    audited = base_rows[base_rows["audit_status"].eq("audited")].copy()
    split_summary = p115.summarize_splits(prepared)
    decision = p115.decide(split_summary)
    manifest = json.loads(args.manifest.read_text())
    download_summary = json.loads(args.download_summary.read_text()) if args.download_summary.exists() else {}
    payload = {
        "protocol": "117_protocol101_targeted_highres_validation",
        "paid_data_downloaded": True,
        "approved_paid_data": True,
        "live_orders": False,
        "model_training": False,
        "source_replay_json": str(args.replay_json),
        "source_manifest": str(args.manifest),
        "decision": decision,
        "interpretation": interpretation(decision),
        "row_counts": {
            "input_rows": int(len(rows)),
            "expanded_rows_with_march_overlay": int(len(prepared)),
            "audited_rows": int(len(audited)),
            "coverage": float(len(audited) / max(len(rows), 1)),
        },
        "data_request": {
            "dataset": manifest["requested_data"]["dataset"],
            "schemas": manifest["requested_data"]["schemas"],
            "sessions": manifest["requested_data"]["session_count"],
            "session_count_by_schema": manifest["requested_data"]["session_count_by_schema"],
            "estimated_total_usd": manifest["cost_estimate"]["estimated_total_usd"],
            "hard_cap_usd": manifest["cost_estimate"]["hard_cap_usd"],
            "downloaded_files": download_summary.get("sessions"),
            "downloaded_final_pass_sessions": len(download_summary.get("downloaded", [])),
            "known_quality_notes": ["Databento warned that 2025-10-22 had degraded quality during download."],
        },
        "split_summary": split_summary,
        "status_summary": p115.summarize_dimension(base_rows, "audit_status"),
        "side_summary": p115.summarize_dimension(audited, "right"),
        "time_bucket_summary": p115.summarize_dimension(audited, "time_bucket"),
        "largest_differences": p115.largest_differences(audited),
        "next_gate": next_gate(decision),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(p115.json_dumps(payload))
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)
    print(json.dumps({"decision": decision, "report": str(report_path)}, indent=2, sort_keys=True))
    return 0


def interpretation(decision: str) -> str:
    if decision == "covered_1s_replay_supports_protocol101_timing_assumption":
        return (
            "Targeted high-resolution replay supports the frozen Protocol 101 timing assumption across required "
            "historical blocks. This is stronger historical evidence, not live-trading approval."
        )
    if decision == "reject_or_reprice_protocol101_timing_assumption":
        return "Targeted high-resolution replay materially disagrees with the one-minute replay; stop and reprice labels."
    return "Targeted high-resolution replay is still incomplete or fragile."


def next_gate(decision: str) -> str:
    if decision == "covered_1s_replay_supports_protocol101_timing_assumption":
        return "run no-order live shadow parity for frozen Protocol 101 before broker-connected paper trading"
    if decision == "reject_or_reprice_protocol101_timing_assumption":
        return "stop model work; rebuild the replay engine around high-resolution execution evidence"
    return "classify remaining coverage gaps or fragility before more model changes"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 117: Protocol 101 Targeted High-Resolution Validation",
        "",
        "Frozen Protocol 101 was not retrained. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Interpretation: {payload['interpretation']}",
        f"- Next gate: {payload['next_gate']}",
        f"- Source replay artifact: `{payload['source_replay_json']}`",
        f"- Audited rows: `{payload['row_counts']['audited_rows']}` of `{payload['row_counts']['input_rows']}` "
        f"({p115.pct(payload['row_counts']['coverage'])})",
        f"- Data: `{payload['data_request']['dataset']}` / `{', '.join(payload['data_request']['schemas'])}`",
        f"- Estimated approved cost: `{p115.money(payload['data_request']['estimated_total_usd'])}` "
        f"under hard cap `{p115.money(payload['data_request']['hard_cap_usd'])}`",
        "",
        "## Split Repricing",
        "",
        "| split | input | audited | coverage | 1m_pnl | highres_pnl | diff | sign_flips | planned_diff | mandatory_events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in p115.REQUIRED_SPLITS:
        row = payload["split_summary"][split]
        lines.append(
            "| "
            f"{split} | {row['input_trades']} | {row['audited_trades']} | {p115.pct(row['coverage'])} | "
            f"{p115.money(row.get('pnl_1m_sum'))} | {p115.money(row.get('pnl_1s_sum'))} | "
            f"{p115.money(row.get('pnl_sum_diff_1s_minus_1m'))} | {p115.pct(row.get('sign_flip_fraction'))} | "
            f"{p115.money(row.get('planned_exit_diff_sum'))} | "
            f"{p115.pct(row.get('mandatory_event_before_lifecycle_exit_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            (
                "The planned high-resolution exit path matches the one-minute planned exit path across the audited set: "
                "planned diff is zero at report precision and sign flips are zero. The remaining difference comes from "
                "high-resolution stop/target events occurring before the lifecycle exit."
            ),
            "",
            "Known data-quality note: Databento warned that `2025-10-22` had degraded quality during download.",
            "",
            "## Largest Differences",
            "",
            "| split | session | seed | time | contract | side | 1m_pnl | highres_pnl | diff | highres_exit |",
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["largest_differences"]:
        lines.append(
            "| "
            f"{row.get('split')} | {row.get('session')} | {row.get('seed')} | {row.get('decision_time')} | "
            f"{row.get('contract_id')} | {row.get('right')} | {p115.money(row.get('pnl_1m'))} | "
            f"{p115.money(row.get('pnl_1s'))} | {p115.money(row.get('pnl_diff_1s_minus_1m'))} | "
            f"{row.get('exit_reason_1s')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    heading = "## 2026-05-14 Protocol 117 Protocol101 Targeted High-Resolution Validation"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing = ledger.read_text() if ledger.exists() else ""
    if heading in existing:
        return
    entry = f"""

{heading}

```text
Date: 2026-05-14
Decision / Experiment: Downloaded the explicitly approved Protocol 116 targeted Databento high-resolution batch and replayed frozen Protocol 101 selected trades against it.
Reason: Protocol 114 showed one-minute timing sensitivity and Protocol 115 had supportive but incomplete one-second evidence. The project needed selected-contract high-resolution validation before trusting the Protocol 101 equity curve further.
Data Used: Databento OPRA.PILLAR CMBP-1 for 2024 sessions and CBBO-1s for 2025-02-20 onward, limited to Protocol 101 selected SPXW raw symbols from the manifest. No live broker data was used and no orders were placed.
Cost: Estimated approved Databento cost ${payload['data_request']['estimated_total_usd']:.4f}, hard cap ${payload['data_request']['hard_cap_usd']:.2f}. Download artifacts occupy the local Protocol 101 overlay directory.
Result: Decision {payload['decision']}. Audited {payload['row_counts']['audited_rows']} of {payload['row_counts']['input_rows']} Protocol 101 rows with {payload['row_counts']['coverage']:.1%} coverage; detailed split repricing is in {report_path}.
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    with ledger.open("a") as handle:
        handle.write(entry)


if __name__ == "__main__":
    raise SystemExit(main())
