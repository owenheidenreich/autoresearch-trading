"""Freeze the unified conservative offline policy foundation.

This is not a model experiment. It records the next ML direction as a frozen
state/action/execution/label contract, keeps Protocol101 as the paper default,
and marks Protocol276 as abandoned as a candidate while preserving its
attribution as evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_conservative_policy import (
    ACTION_ADVANTAGE_LABEL_CONTRACT,
    EXECUTION_MODEL_CONTRACT,
    PAPER_DEFAULT_BASELINE,
    ROLE_LABEL,
    UNIFIED_DECISION_STATE_CONTRACT,
    unified_conservative_policy_contract,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_offline_policy_foundation")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md")
PROTOCOL276_ATTRIBUTION = Path("v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/summary.json")
FOUNDATION_AUDIT = Path("v4/audit/autoresearch/foundation_hardening_review/summary.json")


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
    contract = unified_conservative_policy_contract()
    p276 = load_json(PROTOCOL276_ATTRIBUTION)
    foundation = load_json(FOUNDATION_AUDIT)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation spec / unified conservative offline policy direction",
        "changes_paper_default": False,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "abandoned_candidate": "CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1 / Protocol276",
        "abandonment_reason": p276.get(
            "root_cause_summary",
            [
                "Protocol276 failed Protocol101 outside recent 2026 and showed entry/lifecycle distribution mismatch.",
            ],
        ),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": "unified_conservative_offline_policy_foundation_frozen_no_model_training",
        "foundation_audit_decision": foundation.get("decision", "missing"),
        "contracts": contract,
        "implementation_status": [
            {
                "area": "state/action contract",
                "status": "frozen",
                "evidence": UNIFIED_DECISION_STATE_CONTRACT,
            },
            {
                "area": "execution model",
                "status": "deterministic_ready_fill_calibration_blocked",
                "evidence": EXECUTION_MODEL_CONTRACT,
            },
            {
                "area": "action-advantage labels",
                "status": "primitive_contract_ready",
                "evidence": ACTION_ADVANTAGE_LABEL_CONTRACT,
            },
            {
                "area": "neural policy training",
                "status": "blocked_until_foundation_gates_close",
                "evidence": "No training run is authorized by this foundation packet.",
            },
            {
                "area": "paper default",
                "status": "unchanged",
                "evidence": PAPER_DEFAULT_BASELINE,
            },
        ],
        "next_allowed_work": [
            "Build trajectory-dataset extraction against the frozen UnifiedDecisionStateV1 contract.",
            "Expand the DP oracle from flat and holding primitives into full wait/enter/hold/exit trajectories.",
            "Collect paper/no-order fill observations before enabling stochastic fill replay.",
            "Reserve a new untouched evaluation block before any future promotion claim.",
            "Build live no-order parity for the exact full candidate surface and feature contract.",
        ],
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "state_contract": str(args.out_dir / "unified_decision_state_v1_contract.json"),
            "execution_contract": str(args.out_dir / "execution_model_v1_contract.json"),
            "label_contract": str(args.out_dir / "action_advantage_label_v1_contract.json"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    write_json(args.out_dir / "unified_decision_state_v1_contract.json", contract["state_contract"])
    write_json(args.out_dir / "execution_model_v1_contract.json", contract["execution_model"])
    write_json(args.out_dir / "action_advantage_label_v1_contract.json", contract["label_contract"])
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        f"Abandoned candidate: `{payload['abandoned_candidate']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Summary",
        "",
        "The next model direction is a unified conservative offline policy, not another Protocol276 tweak. The frozen target game is `wait / enter candidate / hold / exit` under the same one-account, one-contract, ask-entry, bid-exit, flat-by-close constraints required for live/paper operation.",
        "",
        "## Why Protocol276 Is Abandoned",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["abandonment_reason"])
    lines.extend(
        [
            "",
            "## Frozen Contracts",
            "",
            f"- State/action: `{UNIFIED_DECISION_STATE_CONTRACT}`",
            f"- Execution: `{EXECUTION_MODEL_CONTRACT}`",
            f"- Labels: `{ACTION_ADVANTAGE_LABEL_CONTRACT}`",
            "- Conservative gate: challenger actions defer to Protocol101 unless advantage clears uncertainty, OOD, and minimum-margin penalties.",
            "",
            "## Implementation Status",
            "",
            "| area | status | evidence |",
            "|---|---|---|",
        ]
    )
    for row in payload["implementation_status"]:
        lines.append(f"| {row['area']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## Next Allowed Work", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_allowed_work"], start=1))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
            f"- State contract: `{payload['outputs']['state_contract']}`",
            f"- Execution contract: `{payload['outputs']['execution_contract']}`",
            f"- Label contract: `{payload['outputs']['label_contract']}`",
            f"- Docs copy: `{payload['outputs']['doc']}`",
        ]
    )
    return "\n".join(lines) + "\n"


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
                    f"- Paper default baseline: `{payload['paper_default_baseline']}`",
                    f"- Abandoned candidate: `{payload['abandoned_candidate']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
