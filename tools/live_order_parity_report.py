#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any


def _safe_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def build_report(audit_path: str) -> dict[str, Any]:
    entry_intents: dict[str, dict[str, Any]] = {}
    entry_applied: set[str] = set()
    execution_seen: set[str] = set()
    status_seen: set[str] = set()
    fill_seen: set[str] = set()
    ib_errors: list[dict[str, Any]] = []
    orphan_execution_events: list[dict[str, Any]] = []

    with open(audit_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(row.get("event", ""))
            payload = _safe_payload(row)

            if event == "entry_intent":
                intent_id = payload.get("intent_id")
                if intent_id:
                    entry_intents[str(intent_id)] = {
                        "decision_id": payload.get("decision_id"),
                        "action": payload.get("action"),
                        "qty": payload.get("qty"),
                    }
            elif event == "entry_intent_applied":
                intent_id = payload.get("intent_id")
                if intent_id:
                    entry_applied.add(str(intent_id))
            elif event in {"entry_live", "entry_dry_run"}:
                intent_id = payload.get("intent_id") or _safe_payload(payload.get("intent", {})).get("intent_id")
                if intent_id:
                    execution_seen.add(str(intent_id))
            elif event == "ib_order_status":
                intent_id = payload.get("intent_id")
                if intent_id:
                    status_seen.add(str(intent_id))
                elif payload:
                    orphan_execution_events.append({"event": event, "payload": payload})
            elif event == "ib_exec_details":
                intent_id = payload.get("intent_id")
                if intent_id:
                    fill_seen.add(str(intent_id))
                elif payload:
                    orphan_execution_events.append({"event": event, "payload": payload})
            elif event == "ib_error_event":
                ib_errors.append(payload)

    mismatches: list[str] = []
    for intent_id in sorted(entry_intents.keys()):
        if intent_id not in entry_applied:
            mismatches.append(f"missing_entry_intent_applied:{intent_id}")
        if intent_id not in execution_seen:
            mismatches.append(f"missing_entry_execution:{intent_id}")

    # Live sessions should eventually emit status/fill evidence. For dry-run this may be absent.
    live_intents = sorted(intent_id for intent_id in entry_intents.keys() if intent_id in execution_seen)
    for intent_id in live_intents:
        if intent_id not in status_seen:
            mismatches.append(f"missing_ib_order_status:{intent_id}")

    report = {
        "schema_version": "live_order_parity_v1",
        "generated_at": dt.datetime.utcnow().isoformat(),
        "audit_path": os.path.abspath(audit_path),
        "counts": {
            "entry_intents": len(entry_intents),
            "entry_intents_applied": len(entry_applied),
            "entry_execution_events": len(execution_seen),
            "status_events_linked": len(status_seen),
            "fill_events_linked": len(fill_seen),
            "ib_error_events": len(ib_errors),
            "orphan_execution_events": len(orphan_execution_events),
        },
        "mismatches": mismatches,
        "ib_errors": ib_errors[:50],
        "orphan_execution_events": orphan_execution_events[:50],
    }
    report["pass"] = (
        report["counts"]["entry_intents"] > 0
        and len(mismatches) == 0
        and report["counts"]["ib_error_events"] == 0
        and report["counts"]["orphan_execution_events"] == 0
    )
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Order Parity Report",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- audit_path: `{report.get('audit_path')}`",
        f"- pass: `{report.get('pass')}`",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k, v in sorted((report.get("counts") or {}).items()):
        lines.append(f"| {k} | {v} |")

    lines.extend(["", "## Mismatches", ""])
    mismatches = report.get("mismatches") or []
    if mismatches:
        for m in mismatches:
            lines.append(f"- {m}")
    else:
        lines.append("- none")

    if report.get("ib_errors"):
        lines.extend(["", "## IB Errors", ""])
        for err in report["ib_errors"][:10]:
            code = err.get("error_code")
            text = err.get("error_string")
            lines.append(f"- code={code} msg={text}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate live signal-to-execution parity from audit.jsonl")
    parser.add_argument("--audit-path", default=os.path.join("results", "live", "audit.jsonl"))
    parser.add_argument("--out-json", default=os.path.join("results", "live", "order_parity_report.json"))
    parser.add_argument("--out-md", default=os.path.join("results", "live", "order_parity_report.md"))
    args = parser.parse_args()

    report = build_report(args.audit_path)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(args.out_md, "w") as f:
        f.write(_to_markdown(report))

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"pass={report.get('pass')}")


if __name__ == "__main__":
    main()
