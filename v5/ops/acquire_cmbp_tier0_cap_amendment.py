#!/usr/bin/env python3
"""Execute one fail-closed Job-55 session pair or local recovery per process."""
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
    if any(key.upper() in FORBIDDEN_REQUEST_ENVIRONMENT for key in os.environ):
        raise RuntimeError("HTTP/TLS environment override is present")
    for name in (".netrc", "_netrc"):
        path = Path(os.path.expanduser(f"~/{name}"))
        if path.exists() or path.is_symlink():
            raise RuntimeError("a default netrc credential file is present")


def _read_api_key_after_gates() -> tuple[str, str]:
    value = os.environ.get("DATABENTO_API_KEY")
    if isinstance(value, str) and value.strip():
        return value.strip(), "process_environment"
    env_path = REPO / ".env"
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
    found: list[str] = []
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
                raise RuntimeError("repository-root .env contains an invalid key")
            if key == "DATABENTO_API_KEY":
                found.append(raw_value.strip().strip('"').strip("'"))
    if len(found) != 1 or not found[0]:
        raise RuntimeError("repository-root .env lacks one usable Databento key")
    return found[0], "repository_root_dotenv"


def _construct_exact_client(api_key: str) -> Any:
    module = importlib.import_module("databento.historical.client")
    historical = getattr(module, "Historical", None)
    if not isinstance(historical, type) or historical.__module__ != "databento.historical.client":
        raise RuntimeError("pinned Databento Historical class drifted")
    client = historical(key=api_key)
    if type(client) is not historical:
        raise RuntimeError("constructed Databento client type drifted")
    return client


def _source_pair_for_result(
    result_record: Mapping[str, Any],
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    start_hash = result_record.get("payload", {}).get("timeseries_start_record_hash")
    starts = [
        record
        for record in lookup.values()
        if record.get("event") == "TIMESERIES_CALL_START" and record.get("record_hash") == start_hash
    ]
    if len(starts) != 1:
        raise RuntimeError("Job-55 source start is absent or duplicated")
    return starts[0], result_record


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP_UNEXPECTED_ARGUMENTS: this Job-55 runner accepts no arguments", file=sys.stderr, flush=True)
        return 2
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from v5.research import cmbp_tier0 as base
    from v5.research import cmbp_tier0_cap_amendment as job55

    attempt_id = str(uuid.uuid4())
    attempt_dir: Path | None = None
    journal: Any | None = None
    header_registered = False
    aggregate_publication_started = False
    readiness_sha = ""
    adoption_sha = ""
    volume: Any | None = None

    def load_state(
        initial_volume: Any,
        *,
        allow_recovery: bool,
    ) -> dict[str, Any]:
        current_volume = base.inspect_destination_volume(
            unpublished_records=base.EXPECTED_RECORD_COUNT,
            expected_device_identifier=initial_volume.device_identifier,
        )
        bundle = job55.load_cap_amendment_scope_bundle(REPO)
        readiness = job55.validate_job55_readiness_receipt(REPO)
        readiness_file_sha = base.file_sha256(job55._paths(REPO)["readiness"])
        opening = job55.validate_job52_opening_evidence(
            Path(current_volume.mount_point) / job55.JOB51_ROOT_RELATIVE,
            volume=current_volume,
        )
        root = Path(current_volume.mount_point) / job55.JOB55_ROOT_RELATIVE
        adoption = job55.ensure_job52_terminal_adoption(root, opening, volume=current_volume, allow_initialize=False)
        job55.ensure_job55_lock_binding(
            root,
            volume=current_volume,
            readiness_receipt_sha256=readiness["receipt_sha256"],
            readiness_receipt_file_sha256=readiness_file_sha,
            adoption_receipt_sha256=adoption["adoption_sha256"],
            allow_initialize=False,
        )
        anchor = job55.validate_job55_attempt_anchor(
            root,
            volume=current_volume,
            readiness_receipt_sha256=readiness["receipt_sha256"],
            readiness_receipt_file_sha256=readiness_file_sha,
            adoption_receipt_sha256=adoption["adoption_sha256"],
            allow_initialize=False,
            repair_header_only=False,
        )
        job55.validate_job55_job_tree(root, volume=current_volume)
        sdk_sha = base.json_sha256(job55.job55_sdk_identity())
        summary = job55.summarize_job55_attempts(
            root,
            bundle.sessions,
            opening=opening,
            volume=current_volume,
            readiness_receipt_sha256=readiness["receipt_sha256"],
            readiness_receipt_file_sha256=readiness_file_sha,
            adoption_receipt_sha256=adoption["adoption_sha256"],
            require_client_constructed=True,
            expected_sdk_identity_sha256=sdk_sha,
        )
        if summary["terminal_authority_failure_observed"]:
            raise job55.Tier0Error("Job-55 authority is terminal", status="STOP_JOB55_AUTHORITY_TERMINAL")
        source_lookup = job55.job55_source_record_lookup(summary)
        finals = job55.validate_job55_existing_session_population(
            root,
            bundle=bundle,
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha,
            adoption=adoption,
            volume=current_volume,
            source_record_lookup=source_lookup,
            published_sessions_in_journals=summary["published_sessions_in_journals"],
        )
        recoverable_hashes = {
            str(item["timeseries_result_record_hash"]) for item in summary["recoverable_staging"]
        }
        final_hashes = {
            str(qc["source_attempt"]["timeseries_result_record_hash"]) for qc in finals.values()
        }
        lost = [
            record
            for record in summary["result_records"]
            if record["record_hash"] not in recoverable_hashes | final_hashes
        ]
        if lost:
            raise job55.Tier0Error(
                "a successful Job-55 stream lost both staging and final evidence",
                status="STOP_JOB55_RECOVERY_REQUIRED",
            )
        if summary["successful_stream_recovery_required"] and not allow_recovery:
            raise job55.Tier0Error("Job-55 local recovery is required", status="STOP_JOB55_RECOVERY_REQUIRED")
        return {
            "volume": current_volume,
            "bundle": bundle,
            "readiness": readiness,
            "readiness_file_sha256": readiness_file_sha,
            "opening": opening,
            "root": root,
            "adoption": adoption,
            "anchor": anchor,
            "sdk_sha": sdk_sha,
            "summary": summary,
            "source_lookup": source_lookup,
            "finals": finals,
        }

    try:
        bundle = job55.load_cap_amendment_scope_bundle(REPO)
        readiness = job55.validate_job55_readiness_receipt(REPO)
        readiness_sha = str(readiness["receipt_sha256"])
        readiness_file_sha = base.file_sha256(job55._paths(REPO)["readiness"])
        volume = base.inspect_destination_volume()
        job51_root = Path(volume.mount_point) / job55.JOB51_ROOT_RELATIVE
        with job55.Job55RunLock(job51_root, volume=volume):
            opening = job55.validate_job52_opening_evidence(job51_root, volume=volume)
            # This is both initialization and a no-follow check of every
            # Job-55 namespace ancestor.  It creates only missing Job-55
            # directories and never touches the immutable Job-51/52 tree.
            job55_root = job55.initialize_job55_destination_tree(volume)

            # Recover only the narrowly safe initialization windows while the
            # shared flock is held.  Each helper refuses to recreate a missing
            # control after an attempt has advanced; the anchor repair accepts
            # only a fully verified header-only attempt.
            adoption = job55.ensure_job52_terminal_adoption(
                job55_root, opening, volume=volume, allow_initialize=True
            )
            adoption_sha = str(adoption["adoption_sha256"])
            job55.ensure_job55_lock_binding(
                job55_root,
                volume=volume,
                readiness_receipt_sha256=readiness_sha,
                readiness_receipt_file_sha256=readiness_file_sha,
                adoption_receipt_sha256=adoption_sha,
                allow_initialize=True,
            )
            job55.validate_job55_attempt_anchor(
                job55_root,
                volume=volume,
                readiness_receipt_sha256=readiness_sha,
                readiness_receipt_file_sha256=readiness_file_sha,
                adoption_receipt_sha256=adoption_sha,
                allow_initialize=True,
                repair_header_only=True,
            )

            state = load_state(volume, allow_recovery=True)
            volume = state["volume"]
            bundle = state["bundle"]
            readiness = state["readiness"]
            readiness_sha = readiness["receipt_sha256"]
            readiness_file_sha = state["readiness_file_sha256"]
            adoption = state["adoption"]
            adoption_sha = adoption["adoption_sha256"]
            job55_root = state["root"]
            receipt_path = job55_root / "receipts" / job55.JOB55_AGGREGATE_NAME
            seal_path = job55_root / job55.JOB55_AGGREGATE_SEAL_NAME

            # A seal is terminal execution state.  It may be used only to
            # finish/revalidate the receipt after all 21 finals exist; an
            # early or stranded seal must stop before a new attempt, secret,
            # client, quote, or download.
            if (
                seal_path.exists() or seal_path.is_symlink()
            ) and len(state["finals"]) != base.EXPECTED_SESSION_COUNT:
                raise job55.Tier0Error(
                    "Job-55 aggregate seal exists before all finals",
                    status="STOP_JOB55_AGGREGATE_QC",
                )
            if seal_path.exists() or seal_path.is_symlink():
                job55.validate_job55_aggregate_seal(
                    seal_path,
                    job55_root=job55_root,
                    summary=state["summary"],
                    final_qc=state["finals"],
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )

            if receipt_path.exists() or receipt_path.is_symlink():
                seal = job55.validate_job55_aggregate_seal(
                    seal_path,
                    job55_root=job55_root,
                    summary=state["summary"],
                    final_qc=state["finals"],
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )
                receipt = job55.validate_job55_aggregate_receipt(
                    receipt_path,
                    bundle=bundle,
                    opening=state["opening"],
                    summary=state["summary"],
                    final_qc=state["finals"],
                    seal=seal,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )
                _progress(f"COMPLETE {receipt['receipt_sha256']} {receipt_path}")
                return 0

            def create_process_attempt(current: Mapping[str, Any]) -> tuple[Path, Any]:
                nonlocal attempt_id, header_registered
                anchor = current["anchor"]
                previous = anchor["job55_attempts"][-1] if anchor["job55_attempts"] else None
                attempt_path, attempt_journal = base.create_attempt(
                    current["root"], attempt_id=attempt_id, volume=current["volume"]
                )
                attempt_journal.append(
                    "ATTEMPT_START",
                    payload=job55.build_job55_attempt_header_payload(
                        attempt_id=attempt_id,
                        volume=current["volume"],
                        readiness_receipt_sha256=current["readiness"]["receipt_sha256"],
                        readiness_receipt_file_sha256=current["readiness_file_sha256"],
                        adoption_receipt_sha256=current["adoption"]["adoption_sha256"],
                        job55_attempt_ordinal=anchor["job55_attempt_count"] + 1,
                        previous_job55_attempt_id=None if previous is None else previous["attempt_id"],
                        previous_job55_attempt_header_record_hash=(
                            None if previous is None else previous["header_record_hash"]
                        ),
                    ),
                )
                job55.validate_job55_attempt_anchor(
                    current["root"],
                    volume=current["volume"],
                    readiness_receipt_sha256=current["readiness"]["receipt_sha256"],
                    readiness_receipt_file_sha256=current["readiness_file_sha256"],
                    adoption_receipt_sha256=current["adoption"]["adoption_sha256"],
                    repair_header_only=True,
                )
                header_registered = True
                return attempt_path, attempt_journal

            # Discharge a prior successful stream locally, with no credential.
            if state["summary"]["successful_stream_recovery_required"]:
                attempt_dir, journal = create_process_attempt(state)
                barrier = state["summary"]["successful_stream_recovery_record"]
                if not isinstance(barrier, dict):
                    raise job55.Tier0Error("Job-55 recovery barrier is malformed", status="STOP_JOB55_RECOVERY_REQUIRED")
                request = next(item for item in bundle.sessions if item.session == barrier["session"])
                ordinal = next(index for index, item in enumerate(bundle.sessions, start=1) if item.session == request.session)
                qc = state["finals"].get(request.session)
                if qc is None:
                    candidates = [
                        item
                        for item in state["summary"]["recoverable_staging"]
                        if item["timeseries_result_record_hash"] == barrier["record_hash"]
                    ]
                    if len(candidates) != 1:
                        raise job55.Tier0Error("Job-55 recovery bytes are absent or ambiguous", status="STOP_JOB55_RECOVERY_REQUIRED")
                    staged = candidates[0]
                    staging_dir = Path(staged["data_path"]).parent
                    data_path = Path(staged["data_path"])
                    result = state["source_lookup"].get(
                        (str(staged["attempt_id"]), int(staged["timeseries_result_sequence"]))
                    )
                    if result is None:
                        raise job55.Tier0Error("Job-55 recovery result is absent", status="STOP_JOB55_RECOVERY_REQUIRED")
                    start, result = _source_pair_for_result(result, state["source_lookup"])
                    qc_path = staging_dir / job55.JOB55_SESSION_QC_NAME
                    if not qc_path.exists() and not qc_path.is_symlink():
                        metadata_summary, decoder_summary = base.stream_dbn_qc(
                            data_path,
                            request,
                            progress=lambda count: _progress(
                                f"{request.session}: recovery QC {count:,}/{request.expected_record_count:,}"
                            ),
                        )
                        qc = job55.build_job55_session_qc(
                            request=request,
                            data_path=data_path,
                            metadata_summary=metadata_summary,
                            decoder_summary=decoder_summary,
                            readiness=readiness,
                            readiness_file_sha256=readiness_file_sha,
                            adoption=adoption,
                            start_record=start,
                            result_record=result,
                        )
                        base.write_canonical_exclusive(qc_path, qc)
                    qc = job55.validate_job55_session_bundle(
                        staging_dir,
                        request=request,
                        readiness=readiness,
                        readiness_file_sha256=readiness_file_sha,
                        adoption=adoption,
                        volume=volume,
                        source_record_lookup=state["source_lookup"],
                    )
                    base.publish_session_bundle(staging_dir, job55_root / "sessions" / request.session, volume=volume)
                if qc["source_attempt"]["timeseries_result_record_hash"] != barrier["record_hash"]:
                    raise job55.Tier0Error("Job-55 recovery final cites another result", status="STOP_JOB55_RECOVERY_REQUIRED")
                journal.append(
                    "SESSION_RECOVERED",
                    session=request.session,
                    request_sha256=request.market_request_sha256,
                    payload=job55.build_job55_publication_payload(
                        request=request, qc=qc, mode="SESSION_RECOVERED", ordinal=ordinal
                    ),
                )
                _progress(f"RECOVERED {ordinal:02d}/21 {request.session}; rerun the bare Job-55 command")
                return 0

            # If all finals exist, seal and publish the aggregate locally.
            if len(state["finals"]) == base.EXPECTED_SESSION_COUNT:
                aggregate_publication_started = True
                if not seal_path.exists() and not seal_path.is_symlink():
                    seal = job55.build_job55_aggregate_seal(
                        job55_root=job55_root,
                        summary=state["summary"],
                        final_qc=state["finals"],
                        readiness=readiness,
                        readiness_file_sha256=readiness_file_sha,
                        adoption=adoption,
                        volume=volume,
                    )
                    base.write_canonical_exclusive(seal_path, seal)
                seal = job55.validate_job55_aggregate_seal(
                    seal_path,
                    job55_root=job55_root,
                    summary=state["summary"],
                    final_qc=state["finals"],
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )
                aggregate = job55.build_job55_aggregate_receipt(
                    bundle=bundle,
                    opening=state["opening"],
                    summary=state["summary"],
                    final_qc=state["finals"],
                    seal=seal,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )
                base.write_canonical_exclusive(receipt_path, aggregate)
                receipt = job55.validate_job55_aggregate_receipt(
                    receipt_path,
                    bundle=bundle,
                    opening=state["opening"],
                    summary=state["summary"],
                    final_qc=state["finals"],
                    seal=seal,
                    readiness=readiness,
                    readiness_file_sha256=readiness_file_sha,
                    adoption=adoption,
                    volume=volume,
                )
                _progress(f"COMPLETE {receipt['receipt_sha256']} {receipt_path}")
                return 0

            missing = [request for request in bundle.sessions if request.session not in state["finals"]]
            request = missing[0]
            ordinal = next(index for index, item in enumerate(bundle.sessions, start=1) if item.session == request.session)
            attempt_dir, journal = create_process_attempt(state)
            staging_dir = attempt_dir / "sessions" / f"{request.session}.bundle.part"
            os.mkdir(staging_dir, 0o700)
            base.fsync_directory(staging_dir.parent)
            base.ensure_nofollow_directory(staging_dir, volume=volume)
            data_path = staging_dir / "data.cmbp-1.dbn.zst"

            _reject_request_environment_overrides()
            before_secret = load_state(volume, allow_recovery=False)
            if request.session in before_secret["finals"]:
                raise job55.Tier0Error("Job-55 target stopped being missing", status="STOP_JOB55_SESSION_QC")
            _reject_request_environment_overrides()
            try:
                api_key, credential_source = _read_api_key_after_gates()
                client = _construct_exact_client(api_key)
            except Exception as exc:  # noqa: BLE001
                raise job55.Tier0Error(
                    f"credential/client construction failed ({type(exc).__name__})",
                    status="STOP_JOB55_VENDOR_AUTH_OR_ENTITLEMENT",
                ) from exc
            journal.append(
                "CLIENT_CONSTRUCTED",
                payload={"credential_source": credential_source, "sdk_identity_sha256": state["sdk_sha"]},
            )
            if credential_source == "process_environment":
                os.environ.pop("DATABENTO_API_KEY", None)
            api_key = ""
            budget = job55.amended_budget_state_from_summary(before_secret["summary"])

            def pre_pair_gate() -> None:
                _reject_request_environment_overrides()

            job55.acquire_amended_session_bytes(
                client,
                request,
                output_path=data_path,
                journal=journal,
                pre_pair_gate=pre_pair_gate,
                budget_state=budget,
                volume=volume,
                readiness_receipt_sha256=readiness_sha,
                readiness_receipt_file_sha256=readiness_file_sha,
                adoption_receipt_sha256=adoption_sha,
                ordinal=ordinal,
                progress=_progress,
            )
            verified = base.verify_attempt_journal(attempt_dir)
            result = verified["records"][-1]
            if result.get("event") != "TIMESERIES_CALL_RESULT":
                raise job55.Tier0Error("Job-55 result is absent", status="STOP_JOB55_JOURNAL_INVALID")
            local_lookup = {
                (record["attempt_id"], record["sequence"]): record
                for record in verified["records"]
                if record.get("event") in {"TIMESERIES_CALL_START", "TIMESERIES_CALL_RESULT"}
            }
            start, result = _source_pair_for_result(result, local_lookup)
            _progress(f"{ordinal:02d}/21 {request.session}: streaming bounded QC")
            metadata_summary, decoder_summary = base.stream_dbn_qc(
                data_path,
                request,
                progress=lambda count: _progress(
                    f"{request.session}: QC {count:,}/{request.expected_record_count:,} records"
                ),
            )
            qc = job55.build_job55_session_qc(
                request=request,
                data_path=data_path,
                metadata_summary=metadata_summary,
                decoder_summary=decoder_summary,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                adoption=adoption,
                start_record=start,
                result_record=result,
            )
            base.write_canonical_exclusive(staging_dir / job55.JOB55_SESSION_QC_NAME, qc)
            job55.validate_job55_session_bundle(
                staging_dir,
                request=request,
                readiness=readiness,
                readiness_file_sha256=readiness_file_sha,
                adoption=adoption,
                volume=volume,
                source_record_lookup=local_lookup,
            )
            base.publish_session_bundle(staging_dir, job55_root / "sessions" / request.session, volume=volume)
            journal.append(
                "SESSION_PUBLISHED",
                session=request.session,
                request_sha256=request.market_request_sha256,
                payload=job55.build_job55_publication_payload(
                    request=request, qc=qc, mode="SESSION_PUBLISHED", ordinal=ordinal
                ),
            )
            _progress(f"PUBLISHED {ordinal:02d}/21 {request.session}; rerun the bare Job-55 command")
            return 0
    except Exception as exc:  # noqa: BLE001 - never print vendor or secret-derived text
        from v5.research import cmbp_tier0_cap_amendment as job55

        status = exc.status if isinstance(exc, job55.Tier0Error) else "STOP_JOB55_UNEXPECTED_LOCAL_ERROR"
        if journal is not None and attempt_dir is not None and header_registered and not aggregate_publication_started:
            try:
                terminal = journal.append(
                    "ATTEMPT_STOP",
                    payload={"status": status, "error_class": type(exc).__name__},
                )
                if volume is not None:
                    job55.write_job55_attempt_stop_receipt(
                        attempt_dir,
                        terminal_record=terminal,
                        volume=volume,
                        readiness_receipt_sha256=readiness_sha,
                        adoption_receipt_sha256=adoption_sha,
                    )
            except Exception:
                pass
        print(f"{status}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
