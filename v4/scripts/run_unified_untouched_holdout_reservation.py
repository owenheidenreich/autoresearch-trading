"""Reserve the untouched evaluation block for future Protocol101 challenges.

This runner does not collect data, train models, or score a challenger. It
turns the repeated-research holdout warning into a concrete reservation: the
current exposed splits are diagnostics, and the next final claim needs a newly
collected/frozen block.
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

from v4.model.neural_training_readiness import (
    EXPOSED_DIAGNOSTIC_SPLITS,
    HOLDOUT_ROLE_LABEL,
    PAPER_DEFAULT_BASELINE,
    make_default_holdout_reservation,
    validate_untouched_holdout_reservation,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_untouched_holdout_reservation")
DEFAULT_DOC_PATH = Path("v4/docs/UNTOUCHED_HOLDOUT_RESERVATION.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--reserved-on", default="2026-05-24")
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reservation = make_default_holdout_reservation(reserved_on=str(args.reserved_on))
    reservation_payload = reservation.to_dict()
    validation = validate_untouched_holdout_reservation(reservation)
    decision = (
        "untouched_holdout_reserved_pending_new_data_collection"
        if validation["status"] == "pass"
        else "untouched_holdout_reservation_invalid"
    )
    payload = {
        "role_label": HOLDOUT_ROLE_LABEL,
        "what_is_this": "foundation gate / untouched evaluation block reservation",
        "changes_paper_default": False,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decision,
        "reservation": reservation_payload,
        "validation": validation,
        "exposed_diagnostic_splits": list(EXPOSED_DIAGNOSTIC_SPLITS),
        "final_claim_available": bool(validation["data_available"]),
        "next_required_evidence": [
            "Collect or purchase the named new block only after the policy, labels, metrics, and thresholds are frozen.",
            "Freeze the raw/processed data manifest before scoring the challenger.",
            "Score the final challenger once against Protocol101 under strict one-account replay and stress assumptions.",
        ],
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "reservation": str(args.out_dir / "untouched_holdout_reservation_v1.json"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    write_json(args.out_dir / "untouched_holdout_reservation_v1.json", reservation_payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def render_report(payload: dict[str, Any]) -> str:
    reservation = payload["reservation"]
    validation = payload["validation"]
    lines = [
        f"# {HOLDOUT_ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Reserved Block",
        "",
        f"- Name: `{reservation['name']}`",
        f"- Status: `{reservation['status']}`",
        f"- Reserved on: `{reservation['reserved_on']}`",
        f"- Start after: `{reservation['start_after']}`",
        f"- Data status: `{reservation['data_status']}`",
        f"- Intended use: `{reservation['intended_use']}`",
        f"- Split label: `{reservation['split_label']}`",
        f"- Final claim available now: `{payload['final_claim_available']}`",
        "",
        "## Exposed Diagnostics",
        "",
        "These blocks are no longer sacred final holdouts for a new challenger:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["exposed_diagnostic_splits"])
    lines.extend(
        [
            "",
            "## Forbidden Uses",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in reservation["forbidden_uses"])
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Status: `{validation['status']}`",
            f"- Data available: `{validation['data_available']}`",
            f"- Errors: `{validation['errors']}`",
            f"- Warnings: `{validation['warnings']}`",
            "",
            "## Next Required Evidence",
            "",
        ]
    )
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
            f"- Reservation: `{payload['outputs']['reservation']}`",
            f"- Docs copy: `{payload['outputs']['doc']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HOLDOUT_ROLE_LABEL}"
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
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    "- Result: Existing Q3/Q4/Q1/March/recent blocks are diagnostics; a future unseen block is reserved for final claims.",
                ]
            )
            + "\n"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
