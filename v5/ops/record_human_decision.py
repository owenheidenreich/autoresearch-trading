"""Safely create, append, inspect, or verify a local human-decision JSONL.

The command has no default action and no default path.  It cannot edit or
delete an event, connect to a broker/vendor, read market data, or record an
outcome.  ``append`` verifies the complete chain before adding one record.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

from v5.research.human_decision_log import (
    CORRECTION,
    CORRECTION_ANNOTATION_VOCABULARY,
    DECISION,
    DECISIONS,
    EVENT_KINDS,
    HumanDecisionLogError,
    INFORMATION_SOURCE_VOCABULARY,
    OWNER_INTENT_VOCABULARY,
    PROGRAM_CONTRACT_SHA256,
    REASON_CODE_VOCABULARY,
    RISK_CONTRACT_SHA256,
    _append_event_with_clocks,
    append_event,
    initialize_log,
    log_status,
    utc_now,
    verify_log,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "initialize", help="exclusively create one empty local JSONL journal"
    )
    initialize.add_argument("--log", type=Path, required=True)
    initialize.add_argument("--log-id", required=True)
    initialize.add_argument(
        "--created-at",
        help="timezone-aware ISO timestamp; defaults to the actual UTC append time",
    )

    for name in ("status", "verify"):
        command = subparsers.add_parser(name, help=f"read-only {name} of a journal")
        command.add_argument("--log", type=Path, required=True)
        command.add_argument(
            "--as-of",
            help="timezone-aware ISO timestamp used only to classify prompt response status",
        )
        command.add_argument(
            "--expected-head",
            help="out-of-band terminal SHA-256 that must match exactly",
        )
        command.add_argument(
            "--expected-min-sequence",
            type=int,
            help="out-of-band minimum terminal sequence; regression is refused",
        )

    append = subparsers.add_parser(
        "append", help="verify the chain and append exactly one sealed event"
    )
    append.add_argument("--log", type=Path, required=True)
    append.add_argument("--kind", choices=sorted(EVENT_KINDS), required=True)
    append.add_argument("--event-id", required=True)
    append.add_argument("--session", required=True)
    append.add_argument(
        "--occurred-at",
        required=True,
        help="timezone-aware causal event time; DECISION must be appended within 30s",
    )
    append.add_argument("--decision-id")
    append.add_argument("--monitoring-state", choices=("ON", "OFF"))
    append.add_argument("--position-state", choices=("FLAT", "OPEN"))
    append.add_argument("--action", choices=sorted(DECISIONS))
    append.add_argument("--contract-osi")
    append.add_argument("--market-state-sha256")
    append.add_argument("--universe-sha256")
    append.add_argument("--prompt-event-id")
    append.add_argument(
        "--spontaneous", action=argparse.BooleanOptionalAction, default=None
    )
    append.add_argument(
        "--information-source",
        action="append",
        choices=sorted(INFORMATION_SOURCE_VOCABULARY),
    )
    append.add_argument("--confidence", type=float)
    append.add_argument(
        "--reason-code", action="append", choices=sorted(REASON_CODE_VOCABULARY)
    )
    append.add_argument("--quantity", type=int)
    append.add_argument("--intended-order-type", choices=("LIMIT", "MARKETABLE_LIMIT"))
    append.add_argument("--intended-limit-price", type=float)
    append.add_argument("--estimated-entry-debit-usd", type=float)
    append.add_argument("--declared-stop-fraction", type=float)
    append.add_argument("--owner-intent", choices=sorted(OWNER_INTENT_VOCABULARY))
    append.add_argument("--corrects-event-id")
    append.add_argument(
        "--note",
        choices=sorted(CORRECTION_ANNOTATION_VOCABULARY),
        help="audit-only CORRECTION annotation; refused on every other event kind",
    )
    return parser


def _run(
    argv: Sequence[str] | None = None,
    *,
    test_now: datetime | None = None,
    test_monotonic_ns: int | None = None,
) -> int:
    """Private runner; non-null test clocks are reserved for offline tests."""

    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    instant = test_now or utc_now()
    try:
        if args.command == "initialize":
            header = initialize_log(
                args.log,
                log_id=args.log_id,
                created_at=args.created_at or instant,
            )
            result = {
                "verdict": "INITIALIZED_LOCAL_ONLY",
                "log_id": header["log_id"],
                "created_at": header["created_at"],
                "head": header["record_hash"],
            }
        elif args.command == "append":
            event_arguments = {
                "path": args.log,
                "kind": args.kind,
                "event_id": args.event_id,
                "decision_id": args.decision_id,
                "session": args.session,
                "occurred_at": args.occurred_at,
                "monitoring_state": args.monitoring_state,
                "position_state": args.position_state,
                "action": args.action,
                "contract_osi": args.contract_osi,
                "market_state_sha256": args.market_state_sha256,
                "universe_sha256": args.universe_sha256,
                "prompt_event_id": args.prompt_event_id,
                "spontaneous": args.spontaneous,
                "information_sources": args.information_source,
                "confidence": args.confidence,
                "reason_codes": args.reason_code,
                "program_contract_sha256": (
                    PROGRAM_CONTRACT_SHA256 if args.kind == DECISION else None
                ),
                "risk_contract_sha256": (
                    RISK_CONTRACT_SHA256 if args.kind == DECISION else None
                ),
                "quantity": args.quantity,
                "intended_order_type": args.intended_order_type,
                "intended_limit_price": args.intended_limit_price,
                "estimated_entry_debit_usd": args.estimated_entry_debit_usd,
                "declared_stop_fraction": args.declared_stop_fraction,
                "owner_intent": args.owner_intent,
                "corrects_event_id": args.corrects_event_id,
                "note": args.note,
            }
            if test_now is None:
                event = append_event(**event_arguments)
            else:
                event = _append_event_with_clocks(
                    **event_arguments,
                    appended_at=instant,
                    local_monotonic_ns=(
                        test_monotonic_ns
                        if test_monotonic_ns is not None
                        else time.monotonic_ns()
                    ),
                )
            result = {
                "verdict": "APPENDED_LOCAL_ONLY",
                "sequence": event["sequence"],
                "event_id": event["event_id"],
                "kind": event["kind"],
                "head": event["record_hash"],
            }
        elif args.command == "status":
            result = log_status(
                args.log,
                now=args.as_of or instant,
                expected_head=args.expected_head,
                expected_min_sequence=args.expected_min_sequence,
            )
        else:
            result = verify_log(
                args.log,
                now=args.as_of or instant,
                expected_head=args.expected_head,
                expected_min_sequence=args.expected_min_sequence,
            ).payload()
    except HumanDecisionLogError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _main_for_test(
    argv: Sequence[str],
    *,
    now: datetime,
    monotonic_ns: int | None = None,
) -> int:
    """Private deterministic entry point for local tests and fixtures."""

    return _run(
        argv,
        test_now=now,
        test_monotonic_ns=monotonic_ns,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Production CLI entry point with writer-owned append clocks."""

    return _run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
