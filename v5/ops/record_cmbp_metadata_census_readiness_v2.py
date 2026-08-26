#!/usr/bin/env python3
"""Build or verify Job 50's audit-superseding V2 local receipt."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5.research import cmbp_metadata_census_receipt as legacy
from v5.research.cmbp_metadata_census_receipt_v2 import (
    MetadataCensusReceiptV2Error,
    build_local_readiness_receipt_v2,
    validate_local_readiness_receipt_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = REPO_ROOT / "v5/work/cmbp-metadata-census"
TEST_REPORT = WORK_ROOT / "TEST_RESULTS_V2.xml"
SYNTHETIC_JOURNAL = WORK_ROOT / "SYNTHETIC_CALL_JOURNAL_V2.jsonl"
SYNTHETIC_RESPONSE = WORK_ROOT / "SYNTHETIC_METADATA_RESPONSES_V2.json"
READINESS_RECEIPT = WORK_ROOT / "LOCAL_READINESS_RECEIPT_V2.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="exclusively seal the canonical V2 receipt")
    commands.add_parser("verify", help="natively rebuild every V2 receipt binding")
    return value


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise MetadataCensusReceiptV2Error(
                f"short V2 receipt write at {offset} of {len(view)} bytes"
            )
        offset += written


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    legacy.canonical_repository_artifact_path(
        REPO_ROOT,
        path,
        Path("v5/work/cmbp-metadata-census/LOCAL_READINESS_RECEIPT_V2.json"),
        "V2 readiness receipt output",
        must_exist=False,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MetadataCensusReceiptV2Error(
            f"refusing to overwrite V2 readiness receipt {path}: {exc}"
        ) from exc
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
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if path.read_bytes() != raw or path.stat().st_mode & 0o777 != 0o600:
            raise MetadataCensusReceiptV2Error(
                "V2 readiness receipt on-disk bytes or mode drifted after fsync"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate(receipt: Mapping[str, Any]) -> None:
    validate_local_readiness_receipt_v2(
        receipt,
        repo_root=REPO_ROOT,
        test_report_path=TEST_REPORT,
        synthetic_journal_path=SYNTHETIC_JOURNAL,
        synthetic_response_path=SYNTHETIC_RESPONSE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_local_readiness_receipt_v2(
                repo_root=REPO_ROOT,
                test_report_path=TEST_REPORT,
                synthetic_journal_path=SYNTHETIC_JOURNAL,
                synthetic_response_path=SYNTHETIC_RESPONSE,
            )
            _write_exclusive(READINESS_RECEIPT, receipt)
            receipt = legacy.strict_json(READINESS_RECEIPT)
            _validate(receipt)
        else:
            legacy.canonical_repository_artifact_path(
                REPO_ROOT,
                READINESS_RECEIPT,
                Path("v5/work/cmbp-metadata-census/LOCAL_READINESS_RECEIPT_V2.json"),
                "V2 readiness receipt",
            )
            receipt = legacy.strict_json(READINESS_RECEIPT)
            _validate(receipt)
    except legacy.MetadataCensusReceiptError as exc:
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
