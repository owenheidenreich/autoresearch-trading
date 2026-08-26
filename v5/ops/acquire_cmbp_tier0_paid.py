#!/usr/bin/env python3
"""Resume Job-51's exact Tier-0 scope under the owner's Job-52 paid caps."""
from __future__ import annotations

import importlib
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
READINESS_PATH = REPO / "v5/work/cmbp-tier0-paid-resume/LOCAL_READINESS_RECEIPT_V1.json"
FORBIDDEN_REQUEST_ENVIRONMENT = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
        "NETRC",
    }
)


def _progress(message: str) -> None:
    print(message, flush=True)


def _reject_request_environment_overrides() -> None:
    present = sorted(
        key
        for key in os.environ
        if key.upper() in FORBIDDEN_REQUEST_ENVIRONMENT
    )
    if present:
        raise RuntimeError("HTTP/TLS environment overrides are present")
    # Requests' module-level API creates a trust_env=True Session for each
    # call, which also discovers default netrc files without a NETRC variable.
    # Refuse their presence by path only; never read their contents.
    for name in (".netrc", "_netrc"):
        path = Path(os.path.expanduser(f"~/{name}"))
        if path.exists() or path.is_symlink():
            raise RuntimeError("a default netrc credential file is present")


def _read_api_key_after_gates() -> tuple[str, str]:
    value = os.environ.get("DATABENTO_API_KEY")
    if isinstance(value, str) and value.strip():
        return value.strip(), "process_environment"
    env_path = REPO / ".env"
    found: list[str] = []
    descriptor = os.open(env_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RuntimeError("repository-root .env owner/mode/type is unsafe")
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
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


def _source_record_lookup(summary: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in summary["timeseries_start_records"] + summary["timeseries_result_records"]:
        key = (str(record["attempt_id"]), int(record["sequence"]))
        if key in result:
            raise RuntimeError("duplicate source record locator")
        result[key] = record
    return result


def _source_pair_for_result(
    result_record: Mapping[str, Any],
    source_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    start_hash = result_record.get("payload", {}).get("timeseries_start_record_hash")
    starts = [
        record
        for record in source_lookup.values()
        if record.get("event") == "TIMESERIES_CALL_START"
        and record.get("record_hash") == start_hash
    ]
    if len(starts) != 1:
        raise RuntimeError("result-bound paid start is absent or duplicated")
    return starts[0], result_record


def _publication_payload(
    *,
    event: str,
    ordinal: int,
    qc: Mapping[str, Any],
) -> dict[str, Any]:
    source = qc["source_attempt"]
    return {
        "ordinal": ordinal,
        "publication_mode": event,
        "session_qc_sha256": qc["session_qc_sha256"],
        "decoded_records": qc["request"]["expected_record_count"],
        "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
        "source_attempt_id": source["attempt_id"],
        "source_timeseries_result_sequence": source["timeseries_result_sequence"],
        "source_timeseries_result_record_hash": source["timeseries_result_record_hash"],
    }


def _write_paid_stop(
    *,
    paid: Any,
    journal: Any,
    attempt_dir: Path,
    attempt_id: str,
    status: str,
    error_class: str,
    readiness_sha256: str,
) -> None:
    if journal.last_record is not None and journal.last_record.get("event") == "ATTEMPT_STOP":
        return
    journal.append("ATTEMPT_STOP", payload={"status": status, "error_class": error_class})
    receipt = {
        "artifact_type": paid.ATTEMPT_STOP_ARTIFACT,
        "schema_version": "v5.job52-cmbp-tier0-paid-attempt-stop.v1",
        "job_id": paid.JOB_ID,
        "target_job_id": paid.TARGET_JOB_ID,
        "attempt_id": attempt_id,
        "scope_sha256": paid.base.EXPECTED_SCOPE_SHA256,
        "paid_program_contract_sha256": paid.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_readiness_receipt_sha256": readiness_sha256,
        "status": status,
        "error_class": error_class,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "journal_terminal_sequence": journal.sequence,
        "journal_terminal_head": journal.head,
    }
    receipt["stop_sha256"] = paid.base.self_hash(receipt, "stop_sha256")
    paid.base.write_canonical_exclusive(attempt_dir / paid.PAID_ATTEMPT_STOP_NAME, receipt)


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this paid runner accepts no arguments", file=sys.stderr, flush=True)
        return 2

    from v5.research import cmbp_tier0 as base
    from v5.research import cmbp_tier0_paid as paid

    attempt_id = str(uuid.uuid4())
    attempt_dir: Path | None = None
    journal: Any | None = None
    header_registered_in_anchor = False
    aggregate_publication_started = False
    readiness_sha256 = ""
    try:
        bundle = paid.load_paid_scope_bundle(REPO)
        readiness = paid.validate_paid_readiness_receipt(REPO, READINESS_PATH)
        readiness_sha256 = str(readiness["receipt_sha256"])
        readiness_file_sha = base.file_sha256(READINESS_PATH)
        expected_sdk_identity_sha = base.json_sha256(readiness["sdk_identity"])
        volume = base.inspect_destination_volume()
        job_root = Path(volume.mount_point) / "cmbp-tier0" / "job51"
        base.ensure_nofollow_directory(job_root, volume=volume, create=False)

        with paid.PaidRunLock(job_root, volume=volume):
            readiness = paid.validate_paid_readiness_receipt(REPO, READINESS_PATH)
            current_volume = base.inspect_destination_volume(
                unpublished_records=base.EXPECTED_RECORD_COUNT,
                expected_device_identifier=volume.device_identifier,
            )
            if current_volume.st_dev != volume.st_dev:
                raise paid.Tier0Error("external volume device changed before paid preflight", status="STOP_EXTERNAL_VOLUME")
            paid.validate_paid_job_tree(
                job_root,
                volume=current_volume,
                allowed_requests=bundle.sessions,
                allow_missing_paid_controls=True,
            )
            paid.ensure_paid_lock_binding(
                job_root,
                volume=current_volume,
                readiness_receipt_sha256=readiness_sha256,
                readiness_receipt_file_sha256=readiness_file_sha,
            )
            anchor = paid.validate_paid_attempt_anchor(
                job_root,
                volume=current_volume,
                readiness_receipt_sha256=readiness_sha256,
                readiness_receipt_file_sha256=readiness_file_sha,
                allow_initialize=True,
                repair_header_only=True,
            )
            paid.validate_paid_job_tree(
                job_root,
                volume=current_volume,
                allowed_requests=bundle.sessions,
                allow_missing_paid_controls=False,
            )

            receipt_path = job_root / "receipts" / paid.PAID_AGGREGATE_NAME
            if receipt_path.exists() or receipt_path.is_symlink():
                receipt = paid.validate_paid_aggregate_receipt(
                    job_root,
                    bundle=bundle,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=current_volume,
                )
                _progress(f"COMPLETE {receipt['receipt_sha256']} {receipt_path}")
                return 0

            summary = paid.summarize_paid_attempts(
                job_root,
                bundle.sessions,
                require_legacy_stop=True,
                expected_readiness_sha256=readiness_sha256,
                expected_readiness_file_sha256=readiness_file_sha,
                require_client_constructed=True,
                expected_sdk_identity_sha256=expected_sdk_identity_sha,
            )
            if summary["terminal_authority_failure_observed"]:
                raise paid.Tier0Error(
                    "a prior cap or malformed quote failure is permanently terminal",
                    status="STOP_PAID_AUTHORITY_TERMINAL",
                )
            budget_state = paid.PaidBudgetState.from_summary(summary)
            source_lookup = _source_record_lookup(summary)
            finals = paid.validate_paid_existing_session_population(
                job_root,
                bundle=bundle,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                volume=current_volume,
                source_record_lookup=source_lookup,
                published_sessions_in_journals=summary["published_sessions_in_journals"],
            )
            recoverable_result_hashes = {
                str(item["timeseries_terminal_record_hash"])
                for item in summary["recoverable_staging"]
            }
            final_result_hashes = {
                str(qc["source_attempt"]["timeseries_result_record_hash"])
                for qc in finals.values()
            }
            lost_results = [
                record
                for record in summary["timeseries_result_records"]
                if record["record_hash"] not in recoverable_result_hashes | final_result_hashes
            ]
            if lost_results:
                raise paid.Tier0Error(
                    "a successful paid stream lost both staging and final evidence",
                    status="STOP_PAID_JOURNAL_INVALID",
                )

            # A prior seal with all finals is a local receipt-recovery path. It
            # must not read a credential or create another paid attempt.
            if summary["aggregate_seal_count"] == 1:
                aggregate = paid.build_paid_aggregate_receipt(
                    job_root,
                    bundle=bundle,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=current_volume,
                )
                aggregate_publication_started = True
                base.write_canonical_exclusive(receipt_path, aggregate)
                _progress(f"COMPLETE {aggregate['receipt_sha256']} {receipt_path}")
                return 0

            previous = anchor["paid_attempts"][-1] if anchor["paid_attempts"] else None
            next_attempt_ordinal = int(anchor["paid_attempt_count"]) + 1
            attempt_dir, journal = base.create_attempt(job_root, attempt_id=attempt_id, volume=current_volume)
            journal.append(
                "ATTEMPT_START",
                payload=paid.build_paid_attempt_header_payload(
                    volume=current_volume,
                    readiness_receipt_sha256=readiness_sha256,
                    readiness_receipt_file_sha256=readiness_file_sha,
                    paid_attempt_ordinal=next_attempt_ordinal,
                    previous_paid_attempt_id=None if previous is None else str(previous["attempt_id"]),
                    previous_paid_attempt_header_record_hash=(
                        None if previous is None else str(previous["header_record_hash"])
                    ),
                ),
            )
            paid.validate_paid_attempt_anchor(
                job_root,
                volume=current_volume,
                readiness_receipt_sha256=readiness_sha256,
                readiness_receipt_file_sha256=readiness_file_sha,
                repair_header_only=True,
            )
            header_registered_in_anchor = True

            recoverable_by_session: dict[str, list[Mapping[str, Any]]] = {}
            for item in summary["recoverable_staging"]:
                recoverable_by_session.setdefault(str(item["session"]), []).append(item)
            for session, items in recoverable_by_session.items():
                if session not in finals and len(items) != 1:
                    raise paid.Tier0Error(
                        "a missing final has ambiguous recoverable staging",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )

            sessions_root = job_root / "sessions"
            completed_records = sum(request.expected_record_count for request in bundle.sessions if request.session in finals)
            request_by_session = {request.session: request for request in bundle.sessions}
            ordinal_by_session = {
                request.session: ordinal for ordinal, request in enumerate(bundle.sessions, start=1)
            }
            current_mode_sessions: set[str] = set()

            def append_local_mode(request: Any, qc: Mapping[str, Any], event: str) -> None:
                ordinal = ordinal_by_session[request.session]
                journal.append(
                    event,
                    session=request.session,
                    request_sha256=request.market_request_sha256,
                    payload=_publication_payload(event=event, ordinal=ordinal, qc=qc),
                )
                current_mode_sessions.add(request.session)

            def recover_staged_session(request: Any, staged: Mapping[str, Any]) -> None:
                nonlocal completed_records
                ordinal = ordinal_by_session[request.session]
                final_dir = sessions_root / request.session
                data_path = Path(str(staged["data_path"]))
                staging_dir = data_path.parent
                result_key = (str(staged["attempt_id"]), int(staged["timeseries_terminal_sequence"]))
                result_record = source_lookup.get(result_key)
                if result_record is None or result_record.get("record_hash") != staged["timeseries_terminal_record_hash"]:
                    raise paid.Tier0Error("recoverable result locator drifted", status="STOP_PAID_JOURNAL_INVALID")
                start_record, result_record = _source_pair_for_result(result_record, source_lookup)
                qc_path = staging_dir / paid.PAID_SESSION_QC_NAME
                if not qc_path.exists() and not qc_path.is_symlink():
                    _progress(f"{ordinal:02d}/21 {request.session}: locally recovering streamed DBN QC")
                    metadata_summary, decoder_summary = base.stream_dbn_qc(
                        data_path,
                        request,
                        progress=lambda count, s=request.session: _progress(
                            f"{s}: QC {count:,}/{request.expected_record_count:,} records"
                        ),
                    )
                    session_qc = paid.build_paid_session_qc(
                        request=request,
                        data_path=data_path,
                        metadata_summary=metadata_summary,
                        decoder_summary=decoder_summary,
                        readiness=readiness,
                        readiness_file_sha256=readiness_file_sha,
                        start_record=start_record,
                        result_record=result_record,
                    )
                    base.write_canonical_exclusive(qc_path, session_qc)
                qc = paid.validate_paid_session_bundle(
                    staging_dir,
                    request=request,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=current_volume,
                    source_record_lookup=source_lookup,
                )
                base.publish_session_bundle(staging_dir, final_dir, volume=current_volume)
                completed_records += request.expected_record_count
                finals[request.session] = qc
                append_local_mode(request, qc, "SESSION_RECOVERED")
                _progress(f"{ordinal:02d}/21 {request.session}: locally recovered and published")

            barrier_record = summary["successful_stream_recovery_record"]
            if summary["successful_stream_recovery_required"]:
                if not isinstance(barrier_record, dict):
                    raise paid.Tier0Error("successful-stream recovery barrier is malformed", status="STOP_PAID_JOURNAL_INVALID")
                barrier_session = str(barrier_record["session"])
                request = request_by_session[barrier_session]
                if barrier_session in finals:
                    qc = finals[barrier_session]
                    if (
                        qc["source_attempt"]["timeseries_result_record_hash"]
                        != barrier_record["record_hash"]
                    ):
                        raise paid.Tier0Error("recovery barrier/final source mismatch", status="STOP_PAID_JOURNAL_INVALID")
                    append_local_mode(request, qc, "SESSION_RECOVERED")
                    _progress(
                        f"{ordinal_by_session[barrier_session]:02d}/21 {barrier_session}: recovered renamed final before credential"
                    )
                else:
                    candidates = recoverable_by_session.get(barrier_session, [])
                    if len(candidates) != 1 or candidates[0]["timeseries_terminal_record_hash"] != barrier_record["record_hash"]:
                        raise paid.Tier0Error("recovery barrier lacks its exact staged result", status="STOP_PAID_JOURNAL_INVALID")
                    recover_staged_session(request, candidates[0])

            # Any remaining result-bound staging is also completed locally
            # before a credential is read. In a valid lineage this population
            # is normally empty after discharging the single global barrier.
            for session, candidates in sorted(recoverable_by_session.items()):
                if session in finals:
                    continue
                recover_staged_session(request_by_session[session], candidates[0])

            # The sealing attempt carries exactly one current-state mode for
            # every final: first recovery above, then all remaining reuses.
            for ordinal, request in enumerate(bundle.sessions, start=1):
                if request.session not in finals or request.session in current_mode_sessions:
                    continue
                qc = finals[request.session]
                event = (
                    "SESSION_REUSED"
                    if request.session in summary["published_sessions_in_journals"]
                    else "SESSION_RECOVERED"
                )
                append_local_mode(request, qc, event)
                _progress(f"{ordinal:02d}/21 {request.session}: verified existing paid bundle")

            def revalidate_local_state(*, required_missing_session: str | None = None) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
                nonlocal current_volume
                current_readiness = paid.validate_paid_readiness_receipt(REPO, READINESS_PATH)
                if (
                    current_readiness["receipt_sha256"] != readiness_sha256
                    or base.file_sha256(READINESS_PATH) != readiness_file_sha
                ):
                    raise paid.Tier0Error("paid readiness changed during execution", status="STOP_PAID_READINESS")
                current_volume = base.inspect_destination_volume(
                    unpublished_records=base.EXPECTED_RECORD_COUNT - completed_records,
                    expected_device_identifier=volume.device_identifier,
                )
                if current_volume.st_dev != volume.st_dev:
                    raise paid.Tier0Error("external volume changed during execution", status="STOP_EXTERNAL_VOLUME")
                paid.validate_paid_job_tree(
                    job_root,
                    volume=current_volume,
                    allowed_requests=bundle.sessions,
                    allow_missing_paid_controls=False,
                )
                paid.ensure_paid_lock_binding(
                    job_root,
                    volume=current_volume,
                    readiness_receipt_sha256=readiness_sha256,
                    readiness_receipt_file_sha256=readiness_file_sha,
                )
                paid.validate_paid_attempt_anchor(
                    job_root,
                    volume=current_volume,
                    readiness_receipt_sha256=readiness_sha256,
                    readiness_receipt_file_sha256=readiness_file_sha,
                )
                if receipt_path.exists() or receipt_path.is_symlink():
                    raise paid.Tier0Error("paid aggregate appeared during execution", status="STOP_PAID_ALREADY_TERMINAL")
                current_summary = paid.summarize_paid_attempts(
                    job_root,
                    bundle.sessions,
                    require_legacy_stop=True,
                    expected_readiness_sha256=readiness_sha256,
                    expected_readiness_file_sha256=readiness_file_sha,
                    require_client_constructed=True,
                    expected_sdk_identity_sha256=expected_sdk_identity_sha,
                )
                if current_summary["terminal_authority_failure_observed"]:
                    raise paid.Tier0Error(
                        "terminal paid authority failure exists",
                        status="STOP_PAID_AUTHORITY_TERMINAL",
                    )
                if current_summary["aggregate_seal_count"] != 0:
                    raise paid.Tier0Error("paid aggregate seal appeared before final gate", status="STOP_PAID_AGGREGATE_QC")
                if current_summary["successful_stream_recovery_required"]:
                    raise paid.Tier0Error(
                        "a successful paid stream still requires local recovery",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )
                disk_state = paid.PaidBudgetState.from_summary(current_summary)
                if (
                    disk_state.commitment_count != budget_state.commitment_count
                    or disk_state.committed_total_usd != budget_state.committed_total_usd
                    or disk_state.committed_by_session_usd != budget_state.committed_by_session_usd
                ):
                    raise paid.Tier0Error("paid budget state changed", status="STOP_PAID_BUDGET_INVALID")
                current_source_lookup = _source_record_lookup(current_summary)
                current_finals = paid.validate_paid_existing_session_population(
                    job_root,
                    bundle=bundle,
                    readiness=current_readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=current_volume,
                    source_record_lookup=current_source_lookup,
                    published_sessions_in_journals=current_summary["published_sessions_in_journals"],
                )
                if set(current_finals) != set(finals) or any(
                    current_finals[session]["session_qc_sha256"] != finals[session]["session_qc_sha256"]
                    for session in finals
                ):
                    raise paid.Tier0Error("paid final population changed", status="STOP_PAID_SESSION_QC")
                if completed_records != sum(
                    request.expected_record_count
                    for request in bundle.sessions
                    if request.session in current_finals
                ):
                    raise paid.Tier0Error("paid completed-record count drifted", status="STOP_PAID_SESSION_QC")
                current_recoverable_hashes = {
                    str(item["timeseries_terminal_record_hash"])
                    for item in current_summary["recoverable_staging"]
                }
                current_final_hashes = {
                    str(qc["source_attempt"]["timeseries_result_record_hash"])
                    for qc in current_finals.values()
                }
                if any(
                    record["record_hash"] not in current_recoverable_hashes | current_final_hashes
                    for record in current_summary["timeseries_result_records"]
                ):
                    raise paid.Tier0Error(
                        "successful paid result evidence disappeared",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )
                if required_missing_session is not None and (
                    required_missing_session in current_finals
                    or (sessions_root / required_missing_session).exists()
                    or (sessions_root / required_missing_session).is_symlink()
                ):
                    raise paid.Tier0Error(
                        "requested paid session is no longer missing",
                        status="STOP_PAID_SESSION_QC",
                    )
                return current_readiness, current_summary

            missing_requests = [request for request in bundle.sessions if request.session not in finals]
            client: Any | None = None
            if missing_requests:
                _reject_request_environment_overrides()
                readiness, _credential_summary = revalidate_local_state()
                _reject_request_environment_overrides()
                try:
                    api_key, credential_source = _read_api_key_after_gates()
                    client = _construct_exact_client(api_key)
                except Exception as exc:  # noqa: BLE001 - never expose secret-derived exception text
                    raise paid.Tier0Error(
                        f"credential/client construction failed ({type(exc).__name__})",
                        status="STOP_VENDOR_AUTH_OR_ENTITLEMENT",
                    ) from exc
                journal.append(
                    "CLIENT_CONSTRUCTED",
                    payload={
                        "credential_source": credential_source,
                        "sdk_identity_sha256": expected_sdk_identity_sha,
                    },
                )
                if credential_source == "process_environment":
                    os.environ.pop("DATABENTO_API_KEY", None)
                api_key = ""

            staging_root = attempt_dir / "sessions"
            for ordinal, request in enumerate(bundle.sessions, start=1):
                if request.session in finals:
                    continue
                staging_dir = staging_root / f"{request.session}.bundle.part"
                if staging_dir.exists() or staging_dir.is_symlink():
                    raise paid.Tier0Error("paid attempt staging bundle already exists", status="STOP_ATTEMPT_PATH_EXISTS")
                os.mkdir(staging_dir, 0o700)
                base.fsync_directory(staging_root)
                base.ensure_nofollow_directory(staging_dir, volume=current_volume)
                data_path = staging_dir / "data.cmbp-1.dbn.zst"

                def pre_pair_gate() -> None:
                    _reject_request_environment_overrides()
                    revalidate_local_state(required_missing_session=request.session)

                if client is None:
                    raise paid.Tier0Error("paid client is absent after gated construction", status="STOP_VENDOR_AUTH_OR_ENTITLEMENT")
                paid.acquire_paid_session_bytes(
                    client,
                    request,
                    output_path=data_path,
                    journal=journal,
                    pre_pair_gate=pre_pair_gate,
                    budget_state=budget_state,
                    readiness_receipt_sha256=readiness_sha256,
                    readiness_receipt_file_sha256=readiness_file_sha,
                    ordinal=ordinal,
                    progress=_progress,
                )
                verified = base.verify_attempt_journal(attempt_dir)
                result_record = verified["records"][-1]
                if result_record.get("event") != "TIMESERIES_CALL_RESULT":
                    raise paid.Tier0Error("paid time-series result is absent", status="STOP_PAID_JOURNAL_INVALID")
                current_lookup = {
                    (str(record["attempt_id"]), int(record["sequence"])): record
                    for record in verified["records"]
                    if record.get("event") in {"TIMESERIES_CALL_START", "TIMESERIES_CALL_RESULT"}
                }
                start_record, result_record = _source_pair_for_result(result_record, current_lookup)
                _progress(
                    f"{ordinal:02d}/21 {request.session}: {data_path.stat().st_size / 1_000_000_000:.2f} GB downloaded; streaming QC"
                )
                metadata_summary, decoder_summary = base.stream_dbn_qc(
                    data_path,
                    request,
                    progress=lambda count, s=request.session: _progress(
                        f"{s}: QC {count:,}/{request.expected_record_count:,} records"
                    ),
                )
                session_qc = paid.build_paid_session_qc(
                    request=request,
                    data_path=data_path,
                    metadata_summary=metadata_summary,
                    decoder_summary=decoder_summary,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    start_record=start_record,
                    result_record=result_record,
                )
                qc_path = staging_dir / paid.PAID_SESSION_QC_NAME
                base.write_canonical_exclusive(qc_path, session_qc)
                paid.validate_paid_session_bundle(
                    staging_dir,
                    request=request,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    volume=current_volume,
                    source_record_lookup=current_lookup,
                )
                final_dir = sessions_root / request.session
                base.publish_session_bundle(staging_dir, final_dir, volume=current_volume)
                completed_records += request.expected_record_count
                finals[request.session] = session_qc
                event = "SESSION_PUBLISHED"
                journal.append(
                    event,
                    session=request.session,
                    request_sha256=request.market_request_sha256,
                    payload=_publication_payload(event=event, ordinal=ordinal, qc=session_qc),
                )
                _progress(f"{ordinal:02d}/21 {request.session}: published paid PASS bundle")

            if len(finals) != base.EXPECTED_SESSION_COUNT:
                raise paid.Tier0Error("paid final population is incomplete", status="STOP_PAID_AGGREGATE_QC")
            readiness, _final_summary = revalidate_local_state()
            journal.append(
                "ATTEMPT_SEALED_FOR_AGGREGATE",
                payload={
                    "published_or_reused_sessions": base.EXPECTED_SESSION_COUNT,
                    "decoded_records": base.EXPECTED_RECORD_COUNT,
                    "commitment_count": budget_state.commitment_count,
                    "committed_quote_total_usd": format(budget_state.committed_total_usd, "f"),
                    "paid_readiness_receipt_sha256": readiness_sha256,
                },
            )
            current_volume = base.inspect_destination_volume(
                unpublished_records=0,
                expected_device_identifier=volume.device_identifier,
            )
            aggregate = paid.build_paid_aggregate_receipt(
                job_root,
                bundle=bundle,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                volume=current_volume,
            )
            aggregate_publication_started = True
            base.write_canonical_exclusive(receipt_path, aggregate)
            _progress(f"COMPLETE {aggregate['receipt_sha256']} {receipt_path}")
            return 0
    except Exception as exc:  # noqa: BLE001 - never print vendor or secret-derived exception text
        from v5.research import cmbp_tier0_paid as paid

        status = exc.status if isinstance(exc, paid.Tier0Error) else "STOP_UNEXPECTED_LOCAL_ERROR"
        if (
            journal is not None
            and attempt_dir is not None
            and header_registered_in_anchor
            and not aggregate_publication_started
        ):
            try:
                _write_paid_stop(
                    paid=paid,
                    journal=journal,
                    attempt_dir=attempt_dir,
                    attempt_id=attempt_id,
                    status=status,
                    error_class=type(exc).__name__,
                    readiness_sha256=readiness_sha256,
                )
            except Exception:
                pass
        print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
