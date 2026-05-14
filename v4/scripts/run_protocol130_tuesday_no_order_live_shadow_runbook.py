"""Protocol 130: Tuesday no-order live shadow runbook.

This script prepares the operational checklist for the next IBKR session. It is
explicitly no-order: no broker endpoint is called and no paper order is enabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.protocol101_shadow_schema import SHADOW_SCHEMA_VERSION, schema_contract
from v4.live.protocol101_risk_gate import Protocol101RiskConfig


DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_130_protocol101_tuesday_no_order_live_shadow_runbook"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_PROMOTION_CHECKLIST = Path("v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md")
SUMMARY_INPUTS = {
    "protocol126": Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/summary.json"
    ),
    "protocol127": Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening/summary.json"
    ),
    "protocol128": Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate/summary.json"
    ),
    "protocol129": Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_129_protocol101_offline_position_sizing/summary.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--promotion-checklist", type=Path, default=DEFAULT_PROMOTION_CHECKLIST)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_inputs(SUMMARY_INPUTS)
    config = Protocol101RiskConfig()
    readiness = readiness_check(summaries)
    decision = "ready_for_tuesday_no_order_live_shadow_only" if readiness["ready"] else "blocked_missing_hardening_inputs"
    payload = {
        "protocol": "130_protocol101_tuesday_no_order_live_shadow_runbook",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "schema_version": SHADOW_SCHEMA_VERSION,
        "risk_config": config.__dict__,
        "input_summaries": summaries,
        "readiness": readiness,
        "run_sequence": run_sequence(),
        "stop_conditions": stop_conditions(),
        "explicit_approval_required_before_orders": True,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "runbook": str(args.out_dir / "tuesday_no_order_live_shadow_runbook.md"),
            "preflight": str(args.out_dir / "protocol101_live_shadow_preflight.json"),
            "promotion_checklist": str(args.promotion_checklist),
        },
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    (args.out_dir / "protocol101_live_shadow_preflight.json").write_text(
        json_dumps(
            {
                "schema": schema_contract(),
                "risk_config": config.__dict__,
                "live_orders_enabled": False,
                "broker_endpoint_called": False,
                "approval_required_before_orders": True,
                "stop_conditions": stop_conditions(),
            }
        )
    )
    write_runbook(args.out_dir / "tuesday_no_order_live_shadow_runbook.md", payload)
    write_runbook(args.promotion_checklist, payload)
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_inputs(inputs: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, path in inputs.items():
        if path.exists():
            payload = json.loads(path.read_text())
            out[key] = {
                "path": str(path),
                "present": True,
                "decision": payload.get("decision"),
                "protocol": payload.get("protocol"),
            }
        else:
            out[key] = {"path": str(path), "present": False, "decision": None, "protocol": None}
    return out


def readiness_check(summaries: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key, value in summaries.items() if not value.get("present")]
    hard_required = {"protocol127", "protocol128"}
    blocked = [
        key
        for key, value in summaries.items()
        if key in hard_required
        and value.get("present")
        and str(value.get("decision", "")).startswith(("blocked_", "fail_"))
    ]
    cautions = [
        key
        for key, value in summaries.items()
        if key not in hard_required
        and value.get("present")
        and str(value.get("decision", "")).startswith(("blocked_", "fail_", "reject_"))
    ]
    return {
        "ready": not missing and not blocked,
        "missing_inputs": missing,
        "blocked_inputs": blocked,
        "caution_inputs": cautions,
        "required_before_any_order_endpoint": [
            "valid protocol101_shadow_v2 JSONL stream",
            "fresh SPX/VIX context and SPXW NBBO",
            "risk_gate event for every enter/wait/blocked decision",
            "paper_account_state event after every decision and exit",
            "order-state rehearsal from captured live shadow rows",
            "explicit user approval after no-order parity passes",
        ],
    }


def run_sequence() -> list[str]:
    return [
        "Connect IB Gateway.",
        "Confirm SPX, VIX, and SPXW NBBO are live and fresh.",
        "Run no-order Protocol101 shadow capture with live_orders_enabled=false.",
        "Validate protocol101_shadow_v2 schema, feature parity, quote freshness, SPXW root, PM settlement, and account state.",
        "Run order-state rehearsal on captured live shadow rows.",
        "Stop and ask for explicit approval before any paper-order endpoint is enabled.",
    ]


def stop_conditions() -> list[str]:
    return [
        "IBKR connection fails or reconnects repeatedly.",
        "SPX or VIX context is missing, stale, or not timestamped.",
        "SPXW option NBBO is missing, stale, locked, crossed, zero, or wrong root/settlement.",
        "Live feature values cannot be mapped to the historical Protocol101 schema.",
        "Any event has live_orders_enabled=true or broker_endpoint_called=true.",
        "Paper account risk gate blocks for a hard reason.",
        "Order-state rehearsal shows overlap, unaffordable trade, or non-flat session.",
    ]


def next_gate(decision: str) -> str:
    if decision.startswith("ready_"):
        return (
            "Tuesday is a no-order live data parity experiment. Paper-order rehearsal happens only after schema, "
            "freshness, risk-gate, and order-state checks pass on captured live rows."
        )
    return "Run or fix the missing hardening protocols before Tuesday's no-order live shadow session."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 130: Protocol101 Tuesday No-Order Live Shadow Runbook",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Schema: `{payload['schema_version']}`",
        f"- Explicit approval required before any paper-order endpoint: `{payload['explicit_approval_required_before_orders']}`",
        "",
        "## Readiness Inputs",
        "",
        "| protocol | present | decision |",
        "| --- | ---: | --- |",
    ]
    for key, value in payload["input_summaries"].items():
        lines.append(f"| {key} | `{value['present']}` | `{value['decision']}` |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Runbook: `{payload['outputs']['runbook']}`",
            f"- Preflight contract: `{payload['outputs']['preflight']}`",
            f"- Promotion checklist: `{payload['outputs']['promotion_checklist']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_runbook(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Protocol 101 Tuesday No-Order Live Shadow Checklist",
        "",
        "Purpose: data parity and operational rehearsal only. This is not a profit experiment.",
        "",
        "## Non-Negotiables",
        "",
        "- `live_orders_enabled = false` for the whole run.",
        "- `broker_endpoint_called = false` for the whole run.",
        "- Protocol 101 remains frozen.",
        "- Paper account starts at `$10,000`; the real `$500` IBKR reserve is not trading capital.",
        "- Max initial paper size remains `1` contract and max concurrent positions remains `1`.",
        "- Stop and ask for explicit approval before enabling any paper-order endpoint.",
        "",
        "## Run Sequence",
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(payload["run_sequence"], start=1))
    lines.extend(["", "## Stop Conditions", ""])
    lines.extend(f"- {item}" for item in payload["stop_conditions"])
    lines.extend(
        [
            "",
            "## Required JSONL Events",
            "",
            "- `market_snapshot`",
            "- `candidate_set`",
            "- `model_decision`",
            "- `risk_gate`",
            "- `paper_account_state`",
            "- `exit_decision`",
            "",
            "## Current Decision",
            "",
            f"`{payload['decision']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 130 Protocol101 Tuesday No-Order Live Shadow Runbook"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Prepared the Tuesday no-order live-shadow runbook and preflight contract for frozen Protocol101.
Reason: The next live session must test data parity, schema, quote freshness, and order-state accounting before any paper order is considered.
Data Used: Existing hardening summaries only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Runbook {payload['outputs']['runbook']}.
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    existing = path.read_text() if path.exists() else ""
    if marker not in existing:
        path.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
