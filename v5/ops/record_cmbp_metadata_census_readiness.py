#!/usr/bin/env python3
"""Build or verify Job 50's synthetic, no-network local-readiness receipt."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from v5.research.cmbp_metadata_census_receipt import (
    MetadataCensusReceiptError,
    build_local_readiness_receipt,
    canonical_repository_artifact_path,
    strict_json,
    validate_local_readiness_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = REPO_ROOT / "v5/work/cmbp-metadata-census"


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-report",
        type=Path,
        default=WORK_ROOT / "TEST_RESULTS_V1.xml",
    )
    parser.add_argument(
        "--synthetic-journal",
        type=Path,
        default=WORK_ROOT / "SYNTHETIC_CALL_JOURNAL_V1.jsonl",
    )
    parser.add_argument(
        "--synthetic-response",
        type=Path,
        default=WORK_ROOT / "SYNTHETIC_METADATA_RESPONSES_V1.json",
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write one exclusive local-readiness receipt")
    _shared_arguments(build)
    build.add_argument(
        "--output",
        type=Path,
        default=WORK_ROOT / "LOCAL_READINESS_RECEIPT_V1.json",
    )
    verify = sub.add_parser("verify", help="rebuild and verify every local binding")
    _shared_arguments(verify)
    verify.add_argument(
        "--receipt",
        type=Path,
        default=WORK_ROOT / "LOCAL_READINESS_RECEIPT_V1.json",
    )
    return value


def _write_exclusive(path: Path, value: dict[str, object]) -> None:
    target = Path(path)
    if not target.parent.is_dir():
        raise MetadataCensusReceiptError(f"receipt parent does not exist: {target.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise MetadataCensusReceiptError(f"refusing to overwrite receipt {target}: {exc}") from exc
    try:
        raw = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise MetadataCensusReceiptError(
                    f"short write while sealing local receipt: {target}"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        reread = target.read_bytes()
        if reread != raw:
            raise MetadataCensusReceiptError(
                f"local receipt on-disk bytes differ after fsync: {target}"
            )
        if target.stat().st_mode & 0o777 != 0o600:
            raise MetadataCensusReceiptError(
                f"local receipt mode differs from 0600 after fsync: {target}"
            )
    except Exception:
        # An exclusive incomplete path remains evidence of a failed write and
        # is never overwritten or silently repaired.
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_canonical_cli_path(
    path: Path,
    expected: Path,
    name: str,
    *,
    must_exist: bool = True,
) -> None:
    canonical_repository_artifact_path(
        REPO_ROOT,
        path,
        expected.relative_to(REPO_ROOT),
        name,
        must_exist=must_exist,
    )


def _validate(
    receipt: dict[str, object],
    *,
    test_report: Path,
    synthetic_journal: Path,
    synthetic_response: Path,
) -> None:
    validate_local_readiness_receipt(
        receipt,
        repo_root=REPO_ROOT,
        test_report_path=test_report,
        synthetic_journal_path=synthetic_journal,
        synthetic_response_path=synthetic_response,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _require_canonical_cli_path(
            args.test_report,
            WORK_ROOT / "TEST_RESULTS_V1.xml",
            "JUnit report",
        )
        _require_canonical_cli_path(
            args.synthetic_journal,
            WORK_ROOT / "SYNTHETIC_CALL_JOURNAL_V1.jsonl",
            "synthetic journal",
        )
        _require_canonical_cli_path(
            args.synthetic_response,
            WORK_ROOT / "SYNTHETIC_METADATA_RESPONSES_V1.json",
            "synthetic response",
        )
        if args.command == "build":
            _require_canonical_cli_path(
                args.output,
                WORK_ROOT / "LOCAL_READINESS_RECEIPT_V1.json",
                "local receipt output",
                must_exist=False,
            )
            receipt = build_local_readiness_receipt(
                repo_root=REPO_ROOT,
                test_report_path=args.test_report,
                synthetic_journal_path=args.synthetic_journal,
                synthetic_response_path=args.synthetic_response,
            )
            _write_exclusive(args.output, receipt)
            written = strict_json(args.output)
            _validate(
                written,
                test_report=args.test_report,
                synthetic_journal=args.synthetic_journal,
                synthetic_response=args.synthetic_response,
            )
            receipt = written
        else:
            _require_canonical_cli_path(
                args.receipt,
                WORK_ROOT / "LOCAL_READINESS_RECEIPT_V1.json",
                "local receipt",
            )
            receipt = strict_json(args.receipt)
            _validate(
                receipt,
                test_report=args.test_report,
                synthetic_journal=args.synthetic_journal,
                synthetic_response=args.synthetic_response,
            )
    except MetadataCensusReceiptError as exc:
        print(f"{exc.status}: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"STOP_RESPONSE_OR_RECEIPT_INVALID: {exc}", file=sys.stderr)
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
