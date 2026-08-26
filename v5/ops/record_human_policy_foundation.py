#!/usr/bin/env python3
"""Build or verify Job 49's repaired V2 local-only foundation receipt.

The command reads only declared local artifacts.  It has no network, vendor,
broker, market-data, model-fitting, or order dependency and refuses overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from v5.research.human_policy_foundation_receipt import (
    FoundationReceiptError,
    build_local_foundation_receipt,
    strict_json,
    validate_local_foundation_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = REPO_ROOT / "v5/work/human-policy-foundation"


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-report",
        type=Path,
        default=WORK_ROOT / "TEST_RESULTS_V2.xml",
    )
    parser.add_argument(
        "--scope-manifest",
        type=Path,
        default=WORK_ROOT / "CMBP_SCOPE_MANIFEST_V1.json",
    )
    parser.add_argument(
        "--catalogue-declaration",
        type=Path,
        default=WORK_ROOT / "CMBP_CATALOGUE_DECLARATION_V1.json",
    )
    parser.add_argument(
        "--synthetic-journal",
        type=Path,
        default=WORK_ROOT / "SYNTHETIC_HUMAN_JOURNAL_V2.jsonl",
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build one exclusive local pass receipt")
    _shared_arguments(build)
    build.add_argument(
        "--output",
        type=Path,
        default=WORK_ROOT / "LOCAL_FOUNDATION_RECEIPT_V2.json",
    )
    verify = sub.add_parser("verify", help="verify a receipt and every current binding")
    _shared_arguments(verify)
    verify.add_argument(
        "--receipt",
        type=Path,
        default=WORK_ROOT / "LOCAL_FOUNDATION_RECEIPT_V2.json",
    )
    return value


def _write_exclusive(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FoundationReceiptError(f"refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise FoundationReceiptError(f"receipt parent does not exist: {path.parent}")
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_local_foundation_receipt(
                repo_root=REPO_ROOT,
                test_report_path=args.test_report,
                scope_manifest_path=args.scope_manifest,
                catalogue_declaration_path=args.catalogue_declaration,
                synthetic_journal_path=args.synthetic_journal,
            )
            _write_exclusive(args.output, receipt)
            validate_local_foundation_receipt(
                receipt,
                repo_root=REPO_ROOT,
                test_report_path=args.test_report,
                scope_manifest_path=args.scope_manifest,
                catalogue_declaration_path=args.catalogue_declaration,
                synthetic_journal_path=args.synthetic_journal,
            )
        else:
            receipt = strict_json(args.receipt)
            validate_local_foundation_receipt(
                receipt,
                repo_root=REPO_ROOT,
                test_report_path=args.test_report,
                scope_manifest_path=args.scope_manifest,
                catalogue_declaration_path=args.catalogue_declaration,
                synthetic_journal_path=args.synthetic_journal,
            )
    except (FoundationReceiptError, OSError, ValueError) as exc:
        print(f"STOP_LOCAL_RECEIPT_INVALID: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "next_state": receipt["next_state"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
