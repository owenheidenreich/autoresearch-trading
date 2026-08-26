#!/usr/bin/env python3
"""Acquire and stream-QC the exact owner-authorized Job-51 Tier-0 scope."""
from __future__ import annotations

import importlib
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
READINESS_PATH = REPO / "v5/work/cmbp-tier0-acquisition/LOCAL_READINESS_RECEIPT_V1.json"


def _progress(message: str) -> None:
    print(message, flush=True)


def _read_api_key_after_gates() -> tuple[str, str]:
    value = os.environ.get("DATABENTO_API_KEY")
    if isinstance(value, str) and value.strip():
        return value.strip(), "process_environment"
    env_path = REPO / ".env"
    metadata = env_path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError("repository-root .env owner/mode/type is unsafe")
    found: list[str] = []
    with env_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise RuntimeError("repository-root .env contains a malformed assignment")
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                raise RuntimeError("repository-root .env contains an invalid key name")
            if key == "DATABENTO_API_KEY":
                found.append(raw_value.strip().strip('"').strip("'"))
    if len(found) != 1 or not found[0]:
        raise RuntimeError("repository-root .env lacks one usable Databento key")
    return found[0], "repository_root_dotenv"


def _construct_exact_client(api_key: str) -> Any:
    client_module = importlib.import_module("databento.historical.client")
    historical = getattr(client_module, "Historical", None)
    if not isinstance(historical, type) or historical.__module__ != "databento.historical.client":
        raise RuntimeError("pinned Databento Historical class identity drifted")
    client = historical(key=api_key)
    if type(client) is not historical:
        raise RuntimeError("constructed Databento client exact type drifted")
    return client


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this runner accepts no arguments", file=sys.stderr, flush=True)
        return 2

    from v5.research import cmbp_tier0 as tier0

    attempt_id = str(uuid.uuid4())
    attempt_dir: Path | None = None
    journal: tier0.AttemptJournal | None = None
    try:
        bundle = tier0.load_scope_bundle(REPO)
        readiness = tier0.validate_readiness_receipt(REPO, READINESS_PATH)
        readiness_file_sha = tier0.file_sha256(READINESS_PATH)
        volume = tier0.inspect_destination_volume()
        job_root = tier0.initialize_destination_tree(volume)
        with tier0.RunLock(
            job_root,
            volume=volume,
            scope_sha256=tier0.EXPECTED_SCOPE_SHA256,
            readiness_sha256=str(readiness["receipt_sha256"]),
        ):
            # Revalidate after the lock and before any credential access.
            readiness = tier0.validate_readiness_receipt(REPO, READINESS_PATH)
            current_volume = tier0.inspect_destination_volume(
                unpublished_records=tier0.EXPECTED_RECORD_COUNT,
                expected_device_identifier=volume.device_identifier,
            )
            if current_volume.st_dev != volume.st_dev:
                raise tier0.Tier0Error("external volume device changed before attempt", status="STOP_EXTERNAL_VOLUME")
            receipt_path = job_root / "receipts" / "JOB51_ACQUISITION_QC_RECEIPT_V1.json"
            if receipt_path.exists() or receipt_path.is_symlink():
                raise tier0.Tier0Error(
                    "terminal aggregate receipt path already exists",
                    status="STOP_JOB51_ALREADY_TERMINAL",
                )
            # Prior frozen-scope attempts are a permanent part of the money boundary.
            # Validate them before credential access or any new quote, including across
            # older readiness seals. A corrupt journal or any nonzero/invalid quote is
            # terminal until the owner makes a new hash-bound decision.
            prior_attempts = tier0.summarize_attempts(job_root, bundle.sessions)
            if prior_attempts["nonzero_or_invalid_quote_observed"]:
                raise tier0.Tier0Error(
                    "a prior Job-51 attempt observed a nonzero or invalid quote",
                    status="STOP_NONZERO_OR_INVALID_COST",
                )
            prior_source_records = {
                (str(record["attempt_id"]), int(record["sequence"])): record
                for record in prior_attempts["timeseries_result_records"]
            }
            tier0.validate_existing_session_population(
                job_root,
                bundle=bundle,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                volume=volume,
                source_record_lookup=prior_source_records,
            )
            attempt_dir, journal = tier0.create_attempt(job_root, attempt_id=attempt_id, volume=volume)
            journal.append(
                "ATTEMPT_START",
                payload={
                    "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
                    "scope_file_sha256": tier0.EXPECTED_SCOPE_FILE_SHA256,
                    "program_contract_sha256": readiness["program_contract_sha256"],
                    "readiness_receipt_sha256": readiness["receipt_sha256"],
                    "readiness_receipt_file_sha256": readiness_file_sha,
                    "volume_identity": tier0.asdict(volume),
                    "authority": "OWNER_CURRENT_CONVERSATION_EXACT_ZERO_COST_TIER0_ACQUISITION",
                    "actual_vendor_invoice_cost_usd": "UNKNOWN",
                },
            )

            # Credential access happens only after scope, seal, mount, lock and durable attempt header.
            try:
                api_key, credential_source = _read_api_key_after_gates()
                client = _construct_exact_client(api_key)
            except Exception as exc:  # noqa: BLE001 - never print secret-derived exception text
                raise tier0.Tier0Error(
                    f"credential/client construction failed ({type(exc).__name__})",
                    status="STOP_VENDOR_AUTH_OR_ENTITLEMENT",
                ) from exc
            journal.append(
                "CLIENT_CONSTRUCTED",
                payload={"credential_source": credential_source, "sdk_identity_sha256": tier0.json_sha256(tier0.sdk_identity())},
            )
            api_key = ""  # minimize the live local reference; never log or hash the secret

            sessions_root = job_root / "sessions"
            staging_root = attempt_dir / "sessions"
            published_records = 0
            for ordinal, request in enumerate(bundle.sessions, start=1):
                final_dir = sessions_root / request.session
                if final_dir.exists() or final_dir.is_symlink():
                    qc = tier0.validate_session_bundle(
                        final_dir,
                        request=request,
                        readiness=readiness,
                        readiness_file_sha256=readiness_file_sha,
                        volume=volume,
                        source_record_lookup=prior_source_records,
                    )
                    published_records += request.expected_record_count
                    journal.append(
                        "SESSION_REUSED",
                        session=request.session,
                        request_sha256=request.market_request_sha256,
                        payload={
                            "ordinal": ordinal,
                            "session_qc_sha256": qc["session_qc_sha256"],
                            "source_attempt_id": qc["source_attempt"]["attempt_id"],
                        },
                    )
                    _progress(f"{ordinal:02d}/21 {request.session}: verified existing current-seal bundle")
                    continue

                staging_dir = staging_root / f"{request.session}.bundle.part"
                if staging_dir.exists() or staging_dir.is_symlink():
                    raise tier0.Tier0Error("attempt staging bundle already exists", status="STOP_ATTEMPT_PATH_EXISTS")
                os.mkdir(staging_dir, 0o700)
                tier0.fsync_directory(staging_root)
                tier0.ensure_nofollow_directory(staging_dir, volume=volume)
                data_path = staging_dir / "data.cmbp-1.dbn.zst"

                def pre_pair_gate() -> None:
                    tier0.validate_readiness_receipt(REPO, READINESS_PATH)
                    remaining = tier0.EXPECTED_RECORD_COUNT - published_records
                    observed = tier0.inspect_destination_volume(
                        unpublished_records=remaining,
                        expected_device_identifier=volume.device_identifier,
                    )
                    if observed.st_dev != volume.st_dev:
                        raise tier0.Tier0Error("external volume changed before vendor pair", status="STOP_EXTERNAL_VOLUME")

                quote = tier0.acquire_session_bytes(
                    client,
                    request,
                    output_path=data_path,
                    journal=journal,
                    pre_pair_gate=pre_pair_gate,
                    progress=_progress,
                )
                source_record = journal.last_record
                if source_record is None or source_record.get("event") != "TIMESERIES_CALL_RESULT":
                    raise tier0.Tier0Error("time-series result journal binding is absent", status="STOP_JOURNAL_INVALID")
                _progress(
                    f"{ordinal:02d}/21 {request.session}: {data_path.stat().st_size / 1_000_000_000:.2f} GB downloaded; streaming QC"
                )
                metadata_summary, decoder_summary = tier0.stream_dbn_qc(
                    data_path,
                    request,
                    progress=lambda count, s=request.session: _progress(
                        f"{s}: QC {count:,}/{request.expected_record_count:,} records"
                    ),
                )
                session_qc = tier0.build_session_qc(
                    request=request,
                    data_path=data_path,
                    quote_usd=quote,
                    metadata_summary=metadata_summary,
                    decoder_summary=decoder_summary,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    attempt_id=attempt_id,
                    journal_record=source_record,
                )
                qc_path = staging_dir / "SESSION_QC_V1.json"
                tier0.write_canonical_exclusive(qc_path, session_qc)
                tier0.validate_session_bundle(
                    staging_dir,
                    request=request,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=volume,
                    source_record_lookup={
                        (attempt_id, int(source_record["sequence"])): source_record,
                    },
                )
                tier0.publish_session_bundle(staging_dir, final_dir, volume=volume)
                published_records += request.expected_record_count
                journal.append(
                    "SESSION_PUBLISHED",
                    session=request.session,
                    request_sha256=request.market_request_sha256,
                    payload={
                        "ordinal": ordinal,
                        "session_qc_sha256": session_qc["session_qc_sha256"],
                        "decoded_records": request.expected_record_count,
                        "compressed_bytes": data_path.stat().st_size if data_path.exists() else session_qc["raw_dbn"]["compressed_bytes"],
                    },
                )
                _progress(f"{ordinal:02d}/21 {request.session}: published PASS bundle")

            journal.append(
                "ATTEMPT_SEALED_FOR_AGGREGATE",
                payload={"published_or_reused_sessions": len(bundle.sessions), "decoded_records": tier0.EXPECTED_RECORD_COUNT},
            )
            aggregate = tier0.build_aggregate_receipt(
                job_root,
                bundle=bundle,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                volume=tier0.inspect_destination_volume(
                    unpublished_records=0,
                    expected_device_identifier=volume.device_identifier,
                ),
            )
            if receipt_path.exists() or receipt_path.is_symlink():
                raise tier0.Tier0Error("aggregate V1 receipt path already exists", status="STOP_AGGREGATE_QC")
            tier0.write_canonical_exclusive(receipt_path, aggregate)
            _progress(f"COMPLETE {aggregate['receipt_sha256']} {receipt_path}")
            return 0
    except Exception as exc:  # noqa: BLE001 - terminal error is sanitized below
        from v5.research import cmbp_tier0 as tier0

        status = exc.status if isinstance(exc, tier0.Tier0Error) else "STOP_UNEXPECTED_LOCAL_ERROR"
        if journal is not None:
            try:
                journal.append("ATTEMPT_STOP", payload={"status": status, "error_class": type(exc).__name__})
                if attempt_dir is not None:
                    stop = {
                        "artifact_type": "JOB51_ATTEMPT_STOP_V1",
                        "schema_version": "v5.job51-attempt-stop.v1",
                        "attempt_id": attempt_id,
                        "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
                        "status": status,
                        "error_class": type(exc).__name__,
                        "actual_vendor_invoice_cost_usd": "UNKNOWN",
                        "journal_terminal_sequence": journal.sequence,
                        "journal_terminal_head": journal.head,
                    }
                    stop["stop_sha256"] = tier0.self_hash(stop, "stop_sha256")
                    tier0.write_canonical_exclusive(attempt_dir / "ATTEMPT_STOP_V1.json", stop)
            except Exception:
                pass
        print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
