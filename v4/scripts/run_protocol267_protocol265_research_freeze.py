"""DECISION_FREEZE_PROTOCOL265_RESEARCH_ONLY_V1.

Freeze Protocol265 as a reproducible research challenger while explicitly
leaving PAPER_DEFAULT_PROTOCOL101 unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_LABEL = "DECISION_FREEZE_PROTOCOL265_RESEARCH_ONLY_V1"
HISTORICAL_ID = "Protocol267"
DEFAULT_PROTOCOL265_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation")
DEFAULT_PROTOCOL266_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_266_protocol265_artifact_reproduction")
DEFAULT_OUT_JSON = Path("v4/promotion/PROTOCOL_265_RESEARCH_FREEZE.json")
DEFAULT_OUT_MD = Path("v4/promotion/PROTOCOL_265_RESEARCH_FREEZE.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol265-dir", type=Path, default=DEFAULT_PROTOCOL265_DIR)
    parser.add_argument("--protocol266-dir", type=Path, default=DEFAULT_PROTOCOL266_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p265 = _read_json(args.protocol265_dir / "summary.json")
    p266 = _read_json(args.protocol266_dir / "summary.json")
    artifact_manifest = _read_json(args.protocol265_dir / "artifact_manifest.json")
    reproduction = p266.get("comparison", {})
    artifact_files = artifact_manifest.get("files", [])
    packet = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "freeze / research-only decision packet",
        "changes_paper_default": False,
        "paper_default": "PAPER_DEFAULT_PROTOCOL101",
        "candidate": "CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1",
        "source_protocol": "Protocol265",
        "reproduction_protocol": "Protocol266",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "evidence": {
            "protocol265_decision": p265.get("decision"),
            "protocol266_decision": p266.get("decision"),
            "artifact_file_count": len(artifact_files),
            "artifact_manifest": str(args.protocol265_dir / "artifact_manifest.json"),
            "reproduction_rows_match": bool(reproduction.get("rows_match")),
            "reproduction_reference_rows": int(reproduction.get("reference_rows", 0)),
            "reproduction_reproduced_rows": int(reproduction.get("reproduced_rows", 0)),
            "reproduction_missing_keys": int(reproduction.get("missing_keys", 0)),
            "reproduction_extra_keys": int(reproduction.get("extra_keys", 0)),
            "reproduction_exit_time_mismatches": int(reproduction.get("exit_time_mismatches", 0)),
            "reproduction_max_abs_pnl_diff": reproduction.get("max_abs_pnl_diff"),
        },
        "status": "research_only",
        "paper_default_change": "none",
        "not_paper_ready_reasons": [
            "requires Protocol265 no-order runtime parity",
            "candidate remains anchored to Protocol261 selected entries",
            "not a full-action unified wait/enter/hold/exit policy",
            "extension attribution is mixed across recent, March, and Q1",
        ],
        "next_required_gates": [
            "AUDIT_PROTOCOL265_EXTENSION_REGIME_ATTRIBUTION_V1",
            "RUNTIME_PROTOCOL265_NO_ORDER_PARITY_V1",
            "DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1",
        ],
        "decision": _decision(p265, p266, artifact_files),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(packet, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_md, packet)
    if not args.skip_ledger:
        append_ledger(packet, args.out_md)
    print(json.dumps({"decision": packet["decision"], "packet": str(args.out_md)}, indent=2, sort_keys=True))
    return 0


def _decision(p265: dict[str, Any], p266: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        return "reject_freeze_missing_protocol265_artifacts"
    if p266.get("decision") != "pass_protocol265_artifact_reproduction_exact":
        return "reject_freeze_protocol265_reproduction_not_exact"
    if not str(p265.get("decision", "")).startswith("research_only"):
        return "freeze_protocol265_research_only_with_nonstandard_source_decision"
    return "freeze_protocol265_research_only_protocol101_unchanged"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_report(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: freeze / research-only decision packet",
        "Does it change the paper-trading default: no",
        f"Candidate: `{packet['candidate']}`",
        f"Paper default baseline: `{packet['paper_default']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{packet['decision']}`",
        "",
        "## Evidence",
        "",
        f"- Protocol265 decision: `{packet['evidence']['protocol265_decision']}`",
        f"- Protocol266 decision: `{packet['evidence']['protocol266_decision']}`",
        f"- Artifact files with hashes: `{packet['evidence']['artifact_file_count']}`",
        f"- Reproduction rows: `{packet['evidence']['reproduction_reproduced_rows']}`",
        f"- Missing keys: `{packet['evidence']['reproduction_missing_keys']}`",
        f"- Extra keys: `{packet['evidence']['reproduction_extra_keys']}`",
        f"- Exit-time mismatches: `{packet['evidence']['reproduction_exit_time_mismatches']}`",
        "",
        "## Status",
        "",
        "Protocol265 is frozen as research-only. Protocol101 remains the paper default.",
        "",
        "## Next Gates",
        "",
        *[f"- `{gate}`" for gate in packet["next_required_gates"]],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(packet: dict[str, Any], report: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
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
                    "- What is this: freeze / research-only decision packet",
                    "- Changes paper default: no",
                    f"- Candidate: `{packet['candidate']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{packet['decision']}`",
                    f"- Report: `{report}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
