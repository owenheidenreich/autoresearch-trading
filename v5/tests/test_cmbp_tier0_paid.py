"""Offline money-boundary and orchestration tests for Job 52 paid resume."""
from __future__ import annotations

import builtins
import copy
import os
import shutil
import socket
import sys
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from v5.research import cmbp_tier0 as tier0
from v5.research import cmbp_tier0_paid as paid


ATTEMPT_ID = "33333333-3333-4333-8333-333333333333"
SECOND_ATTEMPT_ID = "44444444-4444-4444-8444-444444444444"
THIRD_ATTEMPT_ID = "55555555-5555-4555-8555-555555555555"
LEXICALLY_FIRST_ATTEMPT_ID = "11111111-1111-4111-8111-111111111111"
LEXICALLY_LAST_ATTEMPT_ID = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
REPO = Path(__file__).resolve().parents[2]
PAID_WORK = REPO / "v5/work/cmbp-tier0-paid-resume"
PAID_PLAN_FILE_SHA256 = "7eed19e91e6e094778ba7d596ff24dc716557742cf9bda17cbbea7351254b0ba"
PAID_CONTRACT_SHA256 = "094733cb8e6145214161cf4fd3765b3667cc8fabac53eea9abc08796d00600cf"
PAID_CONTRACT_FILE_SHA256 = "befa3f3fa2e5d7f5d4b667da78061a2a5def7c1b4802a8c6b7a2cf801af277ac"
PAID_READINESS_SHA256 = "6" * 64
PAID_READINESS_FILE_SHA256 = "7" * 64


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paid-resume tests must fail before any accidental real socket use."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Job-52 focused tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def _volume(root: Path) -> tier0.VolumeIdentity:
    root.mkdir(parents=True, exist_ok=True)
    return tier0.VolumeIdentity(
        mount_point=str(root),
        volume_uuid=tier0.EXPECTED_VOLUME_UUID,
        device_identifier="synthetic-device",
        filesystem=tier0.EXPECTED_FILESYSTEM,
        bus_protocol="Synthetic",
        st_dev=root.stat().st_dev,
        free_bytes=10**12,
        total_bytes=2 * 10**12,
    )


def _request() -> tier0.SessionRequest:
    return tier0.SessionRequest(
        session="2026-07-02",
        start="2026-07-02T13:30:00Z",
        end="2026-07-02T17:00:00Z",
        symbols=("SPXW  260702C06000000",),
        expected_mappings=((12345, "SPXW  260702C06000000"),),
        expected_record_count=7,
        expected_cost_usd="0",
        cost_request_sha256="a" * 64,
        record_count_request_sha256="b" * 64,
        symbology_request_sha256="c" * 64,
    )


def _journal(
    tmp_path: Path,
    *,
    attempt_id: str = ATTEMPT_ID,
) -> tuple[tier0.AttemptJournal, Path]:
    attempt_dir = tmp_path / attempt_id
    attempt_dir.mkdir(parents=True)
    journal = tier0.AttemptJournal(
        attempt_dir,
        attempt_id=attempt_id,
        volume=_volume(tmp_path),
    )
    journal.append(
        "ATTEMPT_START",
        payload={
            "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
            "authority": paid.PAID_AUTHORITY,
        },
    )
    return journal, attempt_dir


def _job_root(tmp_path: Path) -> Path:
    job_root = tmp_path / "job51"
    for directory in (
        job_root / "attempts",
        job_root / "sessions",
        job_root / "receipts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return job_root


def _legacy_summary() -> dict[str, Any]:
    return {
        "attempt_id": paid.EXPECTED_LEGACY_ATTEMPT_ID,
        "observed_sdk_quote_usd": "0.950392448902",
        "committed_quote_usd": "0",
        "cost_call_starts": 1,
        "timeseries_call_starts": 0,
        "journal_file_sha256": "1" * 64,
        "watermark_file_sha256": "2" * 64,
        "marker_files": [],
        "attempt_stop_file_sha256": paid.EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256,
    }


def _write_exact_job51_run_lock(job_root: Path) -> bytes:
    payload = {
        "artifact_type": "JOB51_RUN_LOCK_V1",
        "readiness_sha256": "851ae1de70308a19fad4160bf33464a3e0a2c79bba15085bde80aea5ea99ff03",
        "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
    }
    raw = tier0.canonical_json_bytes(payload) + b"\n"
    (job_root / "RUN_LOCK_V1").write_bytes(raw)
    return raw


def _patch_paid_runner_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_root: Path | None = None,
) -> SimpleNamespace:
    from v5.ops import acquire_cmbp_tier0_paid as runner

    root = (
        _job_root(tmp_path / "synthetic-volume" / "cmbp-tier0")
        if job_root is None
        else job_root
    )
    assert root.name == "job51" and root.parent.name == "cmbp-tier0"
    legacy_dir = root / "attempts" / paid.EXPECTED_LEGACY_ATTEMPT_ID
    legacy_dir.mkdir(exist_ok=True)
    run_lock_bytes = _write_exact_job51_run_lock(root)
    readiness_path = tmp_path / "LOCAL_READINESS_RECEIPT_V1.json"
    readiness_path.write_bytes(b"synthetic paid readiness fixture\n")
    readiness = {
        "receipt_sha256": PAID_READINESS_SHA256,
        "sdk_identity": {"synthetic_paid_sdk": "offline-test-only"},
    }
    local_volume = _volume(root)
    volume = tier0.VolumeIdentity(
        mount_point=str(root.parents[1]),
        volume_uuid=local_volume.volume_uuid,
        device_identifier=local_volume.device_identifier,
        filesystem=local_volume.filesystem,
        bus_protocol=local_volume.bus_protocol,
        st_dev=local_volume.st_dev,
        free_bytes=local_volume.free_bytes,
        total_bytes=local_volume.total_bytes,
    )
    real_file_sha256 = tier0.file_sha256

    def file_sha256(path: Path) -> str:
        return (
            PAID_READINESS_FILE_SHA256
            if Path(path) == readiness_path
            else real_file_sha256(Path(path))
        )

    monkeypatch.setattr(runner, "READINESS_PATH", readiness_path)
    monkeypatch.setattr(runner.uuid, "uuid4", lambda: ATTEMPT_ID)
    monkeypatch.setattr(
        paid,
        "validate_paid_readiness_receipt",
        lambda _repo, path: (
            copy.deepcopy(readiness)
            if Path(path) == readiness_path
            else (_ for _ in ()).throw(AssertionError(f"unexpected readiness path: {path}"))
        ),
    )
    monkeypatch.setattr(tier0, "file_sha256", file_sha256)
    monkeypatch.setattr(tier0, "inspect_destination_volume", lambda **_kwargs: volume)
    monkeypatch.setattr(
        paid,
        "_validate_legacy_attempt",
        lambda actual_root, _contract: (
            copy.deepcopy(_legacy_summary())
            if Path(actual_root) == root
            else (_ for _ in ()).throw(AssertionError(f"unexpected job root: {actual_root}"))
        ),
    )
    home = tmp_path / "empty-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    for key in list(os.environ):
        if key.upper() in runner.FORBIDDEN_REQUEST_ENVIRONMENT:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", [str(Path(runner.__file__))])
    return SimpleNamespace(
        runner=runner,
        job_root=root,
        readiness=readiness,
        readiness_path=readiness_path,
        home=home,
        volume=volume,
        run_lock_bytes=run_lock_bytes,
        bundle=paid.load_paid_scope_bundle(REPO),
    )


def _initialize_paid_runner_controls(context: SimpleNamespace) -> None:
    paid.ensure_paid_lock_binding(
        context.job_root,
        volume=context.volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    )
    paid.validate_paid_attempt_anchor(
        context.job_root,
        volume=context.volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        allow_initialize=True,
    )


def _paid_job_attempt(
    job_root: Path,
    *,
    attempt_id: str = ATTEMPT_ID,
) -> tuple[tier0.AttemptJournal, Path]:
    volume = _volume(job_root)
    prior_paid_headers: list[dict[str, Any]] = []
    for prior_dir in (job_root / "attempts").iterdir():
        if not prior_dir.is_dir() or prior_dir.name == paid.EXPECTED_LEGACY_ATTEMPT_ID:
            continue
        prior_records = tier0.verify_attempt_journal(prior_dir)["records"]
        prior_payload = prior_records[0].get("payload", {})
        if prior_payload.get("authority") != paid.PAID_AUTHORITY:
            continue
        prior_paid_headers.append(
            {
                "attempt_id": prior_dir.name,
                "header_record_hash": prior_records[0]["record_hash"],
                "paid_attempt_ordinal": prior_payload["paid_attempt_ordinal"],
            }
        )
    previous = (
        max(prior_paid_headers, key=lambda item: item["paid_attempt_ordinal"])
        if prior_paid_headers
        else None
    )
    paid_attempt_ordinal = 1 if previous is None else int(previous["paid_attempt_ordinal"]) + 1
    attempt_dir, journal = tier0.create_attempt(job_root, attempt_id=attempt_id, volume=volume)
    contract = tier0.strict_json(PAID_WORK / "PROGRAM_CONTRACT_V1.json")
    legacy = contract["legacy_stop_evidence"]
    journal.append(
        "ATTEMPT_START",
        payload={
            "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
            "scope_file_sha256": tier0.EXPECTED_SCOPE_FILE_SHA256,
            "job51_program_contract_sha256": tier0.EXPECTED_PROGRAM_CONTRACT_SHA256,
            "paid_program_contract_sha256": PAID_CONTRACT_SHA256,
            "paid_program_contract_file_sha256": PAID_CONTRACT_FILE_SHA256,
            "paid_readiness_receipt_sha256": PAID_READINESS_SHA256,
            "paid_readiness_receipt_file_sha256": PAID_READINESS_FILE_SHA256,
            "legacy_stop_receipt_sha256": legacy["stop_receipt_sha256"],
            "legacy_stop_receipt_file_sha256": legacy["stop_receipt_file_sha256"],
            "volume_identity": tier0.asdict(volume),
            "authority": paid.PAID_AUTHORITY,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "paid_attempt_ordinal": paid_attempt_ordinal,
            "previous_paid_attempt_id": None if previous is None else previous["attempt_id"],
            "previous_paid_attempt_header_record_hash": (
                None if previous is None else previous["header_record_hash"]
            ),
        },
    )
    return journal, attempt_dir


def _validate_paid_anchor(
    job_root: Path,
    *,
    allow_initialize: bool = False,
    repair_header_only: bool = False,
) -> dict[str, Any]:
    return paid.validate_paid_attempt_anchor(
        job_root,
        volume=_volume(job_root),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        allow_initialize=allow_initialize,
        repair_header_only=repair_header_only,
    )


def _staging_data_path(attempt_dir: Path, session: str) -> Path:
    staging = attempt_dir / "sessions" / f"{session}.bundle.part"
    staging.mkdir()
    return staging / "data.cmbp-1.dbn.zst"


class _FakeClient:
    def __init__(
        self,
        quote: Any,
        *,
        actions: list[str] | None = None,
        timeseries_error: BaseException | None = None,
        partial_bytes_before_error: bytes | None = None,
    ) -> None:
        self.quote = quote
        self.actions = actions if actions is not None else []
        self.timeseries_error = timeseries_error
        self.partial_bytes_before_error = partial_bytes_before_error
        self.cost_kwargs: list[dict[str, Any]] = []
        self.timeseries_kwargs: list[dict[str, Any]] = []
        self.metadata = SimpleNamespace(get_cost=self._get_cost)
        self.timeseries = SimpleNamespace(get_range=self._get_range)

    def _get_cost(self, **kwargs: Any) -> Any:
        self.actions.append("metadata.get_cost")
        self.cost_kwargs.append(copy.deepcopy(kwargs))
        if isinstance(self.quote, BaseException):
            raise self.quote
        return self.quote

    def _get_range(self, **kwargs: Any) -> None:
        self.actions.append("timeseries.get_range")
        self.timeseries_kwargs.append(copy.deepcopy(kwargs))
        if self.partial_bytes_before_error is not None:
            Path(kwargs["path"]).write_bytes(self.partial_bytes_before_error)
        if self.timeseries_error is not None:
            raise self.timeseries_error
        Path(kwargs["path"]).write_bytes(b"synthetic-paid-dbn-stream")


class _FakeDBNStore:
    """Tiny iterable carrying the exact DBN metadata surface used by QC."""

    def __init__(self, request: tier0.SessionRequest, records: list[Any] | None = None) -> None:
        next_date = date.fromisoformat(request.session) + timedelta(days=1)
        self.dataset = tier0.EXPECTED_DATASET
        self.schema = tier0.EXPECTED_SCHEMA
        self.stype_in = tier0.EXPECTED_STYPE_IN
        self.stype_out = tier0.EXPECTED_STYPE_OUT
        self.limit = None
        self.symbols = list(request.symbols)
        self.mappings = {
            symbol: [
                {
                    "start_date": request.session,
                    "end_date": next_date.isoformat(),
                    "symbol": str(instrument_id),
                }
            ]
            for instrument_id, symbol in request.expected_mappings
        }
        self.metadata = SimpleNamespace(
            version=3,
            ts_out=False,
            start=tier0._timestamp_ns(request.start),
            end=tier0._timestamp_ns(request.end),
            partial=[],
            not_found=[],
        )
        self._records = list(records or [])

    def __iter__(self):
        return iter(self._records)


def _patch_dbn_store(
    monkeypatch: pytest.MonkeyPatch,
    requests: Mapping[str, tier0.SessionRequest],
) -> None:
    import databento

    class Factory:
        @staticmethod
        def from_file(path: Path) -> _FakeDBNStore:
            parent = Path(path).parent.name
            session = parent.removesuffix(".bundle.part")
            if session not in requests:
                session = Path(path).parent.name
            try:
                request = requests[session]
            except KeyError as exc:
                raise AssertionError(f"unexpected synthetic paid DBN path: {path}") from exc
            return _FakeDBNStore(request)

    monkeypatch.setattr(databento, "DBNStore", Factory)


def _decoder_summary(request: tier0.SessionRequest) -> dict[str, Any]:
    mappings = [
        {"instrument_id": instrument_id, "raw_symbol": symbol}
        for instrument_id, symbol in request.expected_mappings
    ]
    first_recv = tier0._timestamp_ns(request.start) + 1
    return {
        "artifact_type": "JOB51_CMBP_STREAM_SUMMARY_V1",
        "session": request.session,
        "source_kind": "historical",
        "window_start_ns": tier0._timestamp_ns(request.start),
        "window_end_ns": tier0._timestamp_ns(request.end),
        "expected_mappings": mappings,
        "active_mappings": mappings,
        "seen_instrument_ids": [instrument_id for instrument_id, _symbol in request.expected_mappings],
        "mapping_records": 0,
        "mapping_reconciled": True,
        "all_expected_instruments_seen": True,
        "total_records_accepted": request.expected_record_count,
        "cmbp1_records": request.expected_record_count,
        "expected_cmbp1_records": request.expected_record_count,
        "record_count_reconciled": True,
        "trade_records": 0,
        "strict_prior_trades": 0,
        "tied_prior_trades_excluded": 0,
        "no_prior_trades": 0,
        "signed_trades": 0,
        "at_bid_trades": 0,
        "at_ask_trades": 0,
        "inside_trades": 0,
        "outside_trades": 0,
        "ambiguous_trades": 0,
        "undefined_trade_price_excluded": 0,
        "missing_prior_book_excluded": 0,
        "undefined_prior_book_excluded": 0,
        "locked_prior_book_excluded": 0,
        "crossed_prior_book_excluded": 0,
        "trade_bad_ts_recv_excluded": 0,
        "prior_bad_ts_recv_excluded": 0,
        "prior_maybe_bad_book_excluded": 0,
        "all_causal_priors_strict": True,
        "tied_receive_priors_excluded": True,
        "global_receive_ties": 0,
        "global_receive_regressions": 0,
        "global_event_regressions": 0,
        "instrument_event_regressions": 0,
        "first_ts_recv": first_recv,
        "last_ts_recv": first_recv,
        "max_stream_silence_ns": 0,
        "max_instrument_silence_ns": 0,
        "instrument_silences": [
            {"instrument_id": instrument_id, "max_silence_ns": 0}
            for instrument_id, _symbol in request.expected_mappings
        ],
        "disconnect_count": 0,
        "reconnect_count": 0,
        "gap_count": 0,
        "book_state_clear_count": 0,
        "gaps_with_known_bounds": 0,
        "total_known_gap_ns": 0,
        "max_known_gap_ns": None,
        "explicit_connection_telemetry": "UNKNOWN",
        "system_records": 0,
        "heartbeat_records": 0,
        "system_code_counts": [],
        "rows_with_flags": 0,
        "unknown_flag_rows": 0,
        "flag_counts": [
            {"name": name, "count": 0}
            for name in (
                "LAST",
                "TOB",
                "SNAPSHOT",
                "MBP",
                "BAD_TS_RECV",
                "MAYBE_BAD_BOOK",
                "PUBLISHER_SPECIFIC",
            )
        ],
        "flag_value_counts": [{"name": "0", "count": request.expected_record_count}],
        "action_counts": [{"name": "A", "count": request.expected_record_count}],
        "side_counts": [{"name": "B", "count": request.expected_record_count}],
        "classification_reconciled": True,
    }


def _write_paid_junit(path: Path, *, tests: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "job52-paid-focused",
            "tests": str(tests),
            "failures": "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    for index in range(tests):
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "v5.tests.test_cmbp_tier0_paid",
                "name": f"test_paid_population_{index:02d}",
            },
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _stage_paid_session_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    paid_bundle = paid.load_paid_scope_bundle(REPO)
    request = paid_bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    bundle_dir = job_root / "sessions" / request.session
    bundle_dir.mkdir()
    data_path = bundle_dir / "data.cmbp-1.dbn.zst"
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.25")),
        request,
        output_path=data_path,
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    start = next(record for record in records if record["event"] == "TIMESERIES_CALL_START")
    result = next(record for record in records if record["event"] == "TIMESERIES_CALL_RESULT")
    metadata = tier0.validate_dbn_metadata(_FakeDBNStore(request), request)
    readiness = {"receipt_sha256": PAID_READINESS_SHA256}
    receipt = paid.build_paid_session_qc(
        request=request,
        data_path=data_path,
        metadata_summary=metadata,
        decoder_summary=_decoder_summary(request),
        readiness=readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        start_record=start,
        result_record=result,
    )
    qc_path = bundle_dir / paid.PAID_SESSION_QC_NAME
    qc_path.write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")
    _patch_dbn_store(monkeypatch, {request.session: request})
    source_lookup = {
        (record["attempt_id"], record["sequence"]): record
        for record in records
    }
    return SimpleNamespace(
        paid_bundle=paid_bundle,
        request=request,
        job_root=job_root,
        bundle_dir=bundle_dir,
        data_path=data_path,
        qc_path=qc_path,
        receipt=receipt,
        readiness=readiness,
        source_lookup=source_lookup,
        volume=_volume(job_root),
    )


def _append_paid_publication(
    journal: tier0.AttemptJournal,
    *,
    event: str,
    ordinal: int,
    request: tier0.SessionRequest,
    result_record: Mapping[str, Any],
    session_qc_sha256: str | None = None,
) -> dict[str, Any]:
    return journal.append(
        event,
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload={
            "ordinal": ordinal,
            "publication_mode": event,
            "session_qc_sha256": session_qc_sha256
            or tier0.json_sha256({"synthetic_paid_session_qc": request.session}),
            "decoded_records": request.expected_record_count,
            "compressed_bytes": result_record["payload"]["compressed_bytes"],
            "source_attempt_id": result_record["attempt_id"],
            "source_timeseries_result_sequence": result_record["sequence"],
            "source_timeseries_result_record_hash": result_record["record_hash"],
        },
    )


def _stage_full_paid_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    bundle = paid.load_paid_scope_bundle(REPO)
    job_root = _job_root(tmp_path)
    volume = _volume(job_root)
    (job_root / "RUN_LOCK_V1").write_bytes(b"immutable Job-51 lock fixture\n")
    paid.ensure_paid_lock_binding(
        job_root,
        volume=volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    )
    (job_root / "attempts" / paid.EXPECTED_LEGACY_ATTEMPT_ID).mkdir()
    legacy_summary = {
        "attempt_id": paid.EXPECTED_LEGACY_ATTEMPT_ID,
        "observed_sdk_quote_usd": "0.950392448902",
        "committed_quote_usd": "0",
        "cost_call_starts": 1,
        "timeseries_call_starts": 0,
        "journal_file_sha256": "1" * 64,
        "watermark_file_sha256": "2" * 64,
        "marker_files": [],
        "attempt_stop_file_sha256": paid.EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256,
    }
    monkeypatch.setattr(
        paid,
        "_validate_legacy_attempt",
        lambda _job_root, _contract: copy.deepcopy(legacy_summary),
    )
    journal, attempt_dir = _paid_job_attempt(job_root)
    _validate_paid_anchor(job_root, allow_initialize=True)
    readiness = {
        "receipt_sha256": PAID_READINESS_SHA256,
        "sdk_identity": {"synthetic_paid_sdk": "offline-test-only"},
    }
    sdk_identity_sha256 = tier0.json_sha256(readiness["sdk_identity"])
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "process_environment",
            "sdk_identity_sha256": sdk_identity_sha256,
        },
    )
    _patch_dbn_store(
        monkeypatch,
        {request.session: request for request in bundle.sessions},
    )
    state = paid.PaidBudgetState()
    for ordinal, request in enumerate(bundle.sessions, start=1):
        final_dir = job_root / "sessions" / request.session
        final_dir.mkdir()
        data_path = final_dir / "data.cmbp-1.dbn.zst"
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.10")),
            request,
            output_path=data_path,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=ordinal,
        )
        records = tier0.verify_attempt_journal(attempt_dir)["records"]
        result = records[-1]
        assert result["event"] == "TIMESERIES_CALL_RESULT"
        start = next(
            record
            for record in reversed(records)
            if record["record_hash"] == result["payload"]["timeseries_start_record_hash"]
        )
        qc = paid.build_paid_session_qc(
            request=request,
            data_path=data_path,
            metadata_summary=tier0.validate_dbn_metadata(_FakeDBNStore(request), request),
            decoder_summary=_decoder_summary(request),
            readiness=readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            start_record=start,
            result_record=result,
        )
        (final_dir / paid.PAID_SESSION_QC_NAME).write_bytes(
            tier0.canonical_json_bytes(qc) + b"\n"
        )
        _append_paid_publication(
            journal,
            event="SESSION_PUBLISHED",
            ordinal=ordinal,
            request=request,
            result_record=result,
            session_qc_sha256=qc["session_qc_sha256"],
        )
    journal.append(
        "ATTEMPT_SEALED_FOR_AGGREGATE",
        payload={
            "published_or_reused_sessions": 21,
            "decoded_records": tier0.EXPECTED_RECORD_COUNT,
            "commitment_count": state.commitment_count,
            "committed_quote_total_usd": "2.10",
            "paid_readiness_receipt_sha256": PAID_READINESS_SHA256,
        },
    )
    aggregate = paid.build_paid_aggregate_receipt(
        job_root,
        bundle=bundle,
        readiness=readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        volume=volume,
    )
    aggregate_path = job_root / "receipts" / paid.PAID_AGGREGATE_NAME
    aggregate_path.write_bytes(tier0.canonical_json_bytes(aggregate) + b"\n")
    return SimpleNamespace(
        bundle=bundle,
        job_root=job_root,
        volume=volume,
        readiness=readiness,
        aggregate=aggregate,
        aggregate_path=aggregate_path,
        journal=journal,
        attempt_dir=attempt_dir,
    )


def test_paid_contract_preserves_exact_job51_scope_v1_evidence_and_lifetime_caps() -> None:
    plan_path = PAID_WORK / "PLAN.md"
    contract_path = PAID_WORK / "PROGRAM_CONTRACT_V1.json"
    contract = tier0.strict_json(contract_path)
    bundle = tier0.load_scope_bundle(REPO)
    assert tier0.file_sha256(plan_path) == PAID_PLAN_FILE_SHA256
    assert contract["contract_sha256"] == PAID_CONTRACT_SHA256
    assert tier0.self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert tier0.file_sha256(contract_path) == PAID_CONTRACT_FILE_SHA256
    assert contract["plan_file_sha256"] == PAID_PLAN_FILE_SHA256
    assert contract["job_number_resolution"] == {
        "canonical_source": "v5/STATUS.md",
        "job49_later_owner_gated_numbers_reserved_authority": False,
        "job49_numbers_meaning": "NON_AUTHORIZING_SEQUENCE_FORECASTS_SUPERSEDED_BY_THE_CANONICAL_REGISTER",
        "unopened_economic_candidate_formerly_labelled_52": "MUST_BE_RENUMBERED_IF_SEPARATELY_AUTHORIZED",
    }
    assert contract["scope"] == {
        "dataset": tier0.EXPECTED_DATASET,
        "destination_root": str(tier0.EXPECTED_VOLUME_ROOT),
        "path": "v5/work/cmbp-metadata-census/TIER0_ACQUISITION_SCOPE_V1.json",
        "raw_file_sha256": tier0.EXPECTED_SCOPE_FILE_SHA256,
        "schema": tier0.EXPECTED_SCHEMA,
        "semantic_sha256": tier0.EXPECTED_SCOPE_SHA256,
        "session_count": 21,
        "stype_in": tier0.EXPECTED_STYPE_IN,
        "total_record_count": 2_373_877_845,
        "total_session_symbols": 1_001,
    }
    assert len(bundle.sessions) == 21
    assert contract["authority"]["per_session_lifetime_committed_quote_cap_usd"] == "1.50"
    assert contract["authority"]["total_committed_quote_cap_usd"] == "32.00"
    assert contract["legacy_stop_evidence"]["observed_sdk_quote_usd"] == "0.950392448902"
    assert contract["legacy_stop_evidence"]["paid_commitment_usd"] == "0"
    assert contract["legacy_stop_evidence"]["time_series_call_starts"] == 0
    assert contract["v1_immutable_evidence"] == {
        "job51_local_readiness_file_sha256": tier0.file_sha256(
            REPO / "v5/work/cmbp-tier0-acquisition/LOCAL_READINESS_RECEIPT_V1.json"
        ),
        "job51_local_readiness_sha256": "851ae1de70308a19fad4160bf33464a3e0a2c79bba15085bde80aea5ea99ff03",
        "job51_plan_file_sha256": tier0.file_sha256(
            REPO / "v5/work/cmbp-tier0-acquisition/PLAN.md"
        ),
        "job51_program_contract_file_sha256": tier0.file_sha256(
            REPO / "v5/work/cmbp-tier0-acquisition/PROGRAM_CONTRACT_V1.json"
        ),
        "job51_program_contract_sha256": tier0.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "job51_test_report_file_sha256": tier0.file_sha256(
            REPO / "v5/work/cmbp-tier0-acquisition/TEST_RESULTS_V1.xml"
        ),
    }
    assert contract["legacy_stop_evidence"]["stop_receipt_file_sha256"] == tier0.file_sha256(
        REPO / "v5/work/cmbp-tier0-acquisition/JOB51_ZERO_COST_GATE_STOP_RECEIPT_V1.json"
    )
    paid_bundle = paid.load_paid_scope_bundle(REPO)
    assert paid_bundle.contract["contract_sha256"] == PAID_CONTRACT_SHA256
    assert paid_bundle.sessions == bundle.sessions


def test_paid_readiness_junit_requires_exact_pinned_population_not_clean_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = tmp_path / "complete.xml"
    _write_paid_junit(complete, tests=13)
    population = paid._paid_junit_counts(complete, enforce_frozen_population=False)
    monkeypatch.setattr(paid, "EXPECTED_FOCUSED_TEST_COUNT", population["tests"])
    monkeypatch.setattr(
        paid,
        "EXPECTED_FOCUSED_TEST_IDENTITY_SHA256",
        population["test_case_identity_sha256"],
    )
    assert paid._paid_junit_counts(complete) == population

    selected_subset = tmp_path / "selected-subset.xml"
    _write_paid_junit(selected_subset, tests=12)
    with pytest.raises(tier0.Tier0Error) as raised:
        paid._paid_junit_counts(selected_subset)
    assert raised.value.status == "STOP_PAID_LOCAL_TESTS"


def test_paid_readiness_build_validate_binds_inputs_sdk_and_local_only_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    work = root / "v5/work/cmbp-tier0-paid-resume"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness_path = work / "LOCAL_READINESS_RECEIPT_V1.json"
    legacy_path = root / "legacy-stop.json"
    bound_relative = "v5/research/synthetic_paid_bound.py"
    bound_path = root / bound_relative
    bound_path.parent.mkdir(parents=True)
    bound_path.write_text("synthetic paid readiness input\n", encoding="utf-8")
    _write_paid_junit(report, tests=13)
    population = paid._paid_junit_counts(report, enforce_frozen_population=False)
    monkeypatch.setattr(paid, "EXPECTED_FOCUSED_TEST_COUNT", population["tests"])
    monkeypatch.setattr(
        paid,
        "EXPECTED_FOCUSED_TEST_IDENTITY_SHA256",
        population["test_case_identity_sha256"],
    )
    legacy_path.write_bytes(
        tier0.canonical_json_bytes({"receipt_sha256": "a" * 64}) + b"\n"
    )
    paths = {
        "readiness": readiness_path,
        "test_report": report,
        "legacy_stop": legacy_path,
    }
    monkeypatch.setattr(paid, "_paths", lambda _root: paths)
    monkeypatch.setattr(
        paid,
        "load_paid_scope_bundle",
        lambda _root: SimpleNamespace(contract={"required_bound_files": [bound_relative]}),
    )
    synthetic_sdk_identity = {
        "job51_sdk_identity": {"synthetic_sdk_identity": "offline-only"},
        "job52_compression_identity": {
            "active_backend": "cext",
            "version": paid.EXPECTED_ZSTANDARD_VERSION,
            "files": {"zstandard.backend_c": {"file_sha256": "b" * 64}},
            "distribution": {
                "file_count": 1,
                "manifest_sha256": "c" * 64,
                "files": {"zstandard/backend_c.so": "d" * 64},
            },
        },
    }
    monkeypatch.setattr(
        paid,
        "paid_sdk_identity",
        lambda: copy.deepcopy(synthetic_sdk_identity),
    )

    receipt = paid.build_paid_readiness_receipt(root, test_report_path=report)
    readiness_path.write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")
    assert paid.validate_paid_readiness_receipt(root, readiness_path) == receipt
    assert receipt["bound_files"] == {bound_relative: tier0.file_sha256(bound_path)}
    assert receipt["sdk_identity"] == synthetic_sdk_identity
    assert receipt["sdk_identity"]["job52_compression_identity"]["version"] == "0.25.0"
    assert receipt["integrity"]["credential_read"] is False
    assert receipt["integrity"]["external_calls"] == 0
    assert receipt["integrity"]["timeseries_calls"] == 0
    assert receipt["integrity"]["data_downloaded"] is False
    assert receipt["money_boundary"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"

    bound_path.write_text("changed after readiness seal\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_readiness_receipt(root, readiness_path)
    assert raised.value.status == "STOP_PAID_READINESS"
    with pytest.raises(tier0.Tier0Error) as widened:
        paid.validate_paid_readiness_receipt(root, tmp_path / "widened-readiness.json")
    assert widened.value.status == "STOP_PAID_READINESS"


def test_paid_sdk_identity_binds_compression_default_ca_and_http_dependencies() -> None:
    identity = paid.paid_sdk_identity()
    assert set(identity) == {
        "job51_sdk_identity",
        "job52_compression_identity",
        "job52_transport_trust_identity",
        "job52_http_dependency_identity",
    }
    compression = identity["job52_compression_identity"]
    assert compression["active_backend"] == "cext"
    assert compression["version"] == paid.EXPECTED_ZSTANDARD_VERSION
    assert set(compression["files"]) == {"zstandard.__init__", "zstandard.backend_c"}
    assert compression["distribution"]["file_count"] == len(
        compression["distribution"]["files"]
    ) > 0

    trust = identity["job52_transport_trust_identity"]
    assert trust["version"] == paid.EXPECTED_CERTIFI_VERSION
    assert Path(trust["default_ca_bundle"]["path"]).is_file()
    assert tier0.file_sha256(Path(trust["default_ca_bundle"]["path"])) == trust[
        "default_ca_bundle"
    ]["file_sha256"]
    assert trust["distribution"]["file_count"] == len(trust["distribution"]["files"]) > 0

    http = identity["job52_http_dependency_identity"]
    assert set(http["distributions"]) == {"idna", "charset-normalizer"}
    assert http["distributions"]["idna"]["version"] == paid.EXPECTED_IDNA_VERSION
    assert (
        http["distributions"]["charset-normalizer"]["version"]
        == paid.EXPECTED_CHARSET_NORMALIZER_VERSION
    )
    assert set(http["modules"]) >= {
        "idna.__init__",
        "idna.core",
        "charset_normalizer.__init__",
        "charset_normalizer.api",
        "charset_normalizer.md",
    }
    assert all(
        distribution["file_count"] == len(distribution["files"]) > 0
        for distribution in http["distributions"].values()
    )


def test_paid_session_qc_round_trips_against_dbn_and_exact_journal_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_paid_session_qc(tmp_path, monkeypatch)
    validated = paid.validate_paid_session_bundle(
        staged.bundle_dir,
        request=staged.request,
        readiness=staged.readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        volume=staged.volume,
        source_record_lookup=staged.source_lookup,
    )
    assert validated == staged.receipt
    assert validated["cost_commitment"] == {
        "fresh_observed_sdk_quote_usd": "0.25",
        "commitment_index": 1,
        "committed_quote_session_before_usd": "0",
        "committed_quote_session_after_usd": "0.25",
        "committed_quote_total_before_usd": "0",
        "committed_quote_total_after_usd": "0.25",
        "per_session_lifetime_cap_usd": "1.50",
        "total_cap_usd": "32.00",
        "acquisition_initiated": True,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "quote_is_atomic_invoice_lock": False,
    }
    population = paid.validate_paid_existing_session_population(
        staged.job_root,
        bundle=staged.paid_bundle,
        readiness=staged.readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        volume=staged.volume,
        source_record_lookup=staged.source_lookup,
        published_sessions_in_journals=[staged.request.session],
    )
    assert population == {staged.request.session: staged.receipt}


@pytest.mark.parametrize(
    "tamper",
    (
        "paid_readiness",
        "self_consistent_cost_quote",
        "commitment_index",
        "source_sequence",
        "source_hash",
        "raw_dbn",
        "dbn_metadata",
        "missing_max_stream_silence",
        "negative_max_stream_silence",
        "missing_instrument_silences",
        "inconsistent_instrument_silence",
    ),
)
def test_paid_session_qc_refuses_self_rehashed_source_cost_readiness_or_dbn_tamper(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_paid_session_qc(tmp_path, monkeypatch)
    receipt = copy.deepcopy(staged.receipt)
    if tamper == "paid_readiness":
        receipt["paid_readiness_receipt_sha256"] = "8" * 64
    elif tamper == "self_consistent_cost_quote":
        receipt["cost_commitment"].update(
            {
                "fresh_observed_sdk_quote_usd": "0.20",
                "committed_quote_session_after_usd": "0.20",
                "committed_quote_total_after_usd": "0.20",
            }
        )
    elif tamper == "commitment_index":
        receipt["cost_commitment"]["commitment_index"] = 2
    elif tamper == "source_sequence":
        receipt["source_attempt"]["timeseries_result_sequence"] += 10
    elif tamper == "source_hash":
        receipt["source_attempt"]["timeseries_result_record_hash"] = "9" * 64
    elif tamper == "raw_dbn":
        receipt["raw_dbn"]["compressed_bytes"] += 1
        receipt["raw_dbn"]["file_sha256"] = "9" * 64
    elif tamper == "dbn_metadata":
        receipt["dbn_metadata"]["dbn_version"] = 2
    elif tamper == "missing_max_stream_silence":
        receipt["decoder"].pop("max_stream_silence_ns")
    elif tamper == "negative_max_stream_silence":
        receipt["decoder"]["max_stream_silence_ns"] = -1
    elif tamper == "missing_instrument_silences":
        receipt["decoder"].pop("instrument_silences")
    elif tamper == "inconsistent_instrument_silence":
        receipt["decoder"]["max_instrument_silence_ns"] = 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)
    receipt["session_qc_sha256"] = tier0.self_hash(receipt, "session_qc_sha256")
    staged.qc_path.write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_session_bundle(
            staged.bundle_dir,
            request=staged.request,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
            source_record_lookup=staged.source_lookup,
        )
    assert raised.value.status == "STOP_PAID_SESSION_QC"


def test_paid_session_qc_refuses_source_result_absent_from_all_attempt_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_paid_session_qc(tmp_path, monkeypatch)
    lookup = dict(staged.source_lookup)
    source = staged.receipt["source_attempt"]
    lookup.pop((source["attempt_id"], source["timeseries_result_sequence"]))
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_session_bundle(
            staged.bundle_dir,
            request=staged.request,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
            source_record_lookup=lookup,
        )
    assert raised.value.status == "STOP_PAID_SESSION_QC"


def test_paid_population_refuses_disappeared_durably_published_final_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_paid_session_qc(tmp_path, monkeypatch)
    shutil.rmtree(staged.bundle_dir)
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_existing_session_population(
            staged.job_root,
            bundle=staged.paid_bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
            source_record_lookup=staged.source_lookup,
            published_sessions_in_journals=[staged.request.session],
        )
    assert raised.value.status == "STOP_PAID_SESSION_QC"


def test_full_21_session_paid_aggregate_round_trips_and_refuses_self_rehashed_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_full_paid_aggregate(tmp_path, monkeypatch)
    aggregate = staged.aggregate
    assert paid.validate_paid_aggregate_receipt(
        staged.job_root,
        bundle=staged.bundle,
        readiness=staged.readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        volume=staged.volume,
    ) == aggregate
    assert aggregate["completeness"]["published_sessions"] == 21
    assert aggregate["completeness"]["decoded_records"] == tier0.EXPECTED_RECORD_COUNT
    assert aggregate["cost_boundary"]["committed_quote_total_usd"] == "2.10"
    assert aggregate["cost_boundary"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    assert len(aggregate["cost_boundary"]["commitments"]) == 21
    assert len(aggregate["cost_boundary"]["all_quote_observations"]) == 22
    assert aggregate["call_accounting"]["paid_cost_call_starts"] == 21
    assert aggregate["call_accounting"]["paid_cost_call_results"] == 21
    assert aggregate["call_accounting"]["paid_timeseries_call_starts"] == 21
    assert aggregate["call_accounting"]["paid_timeseries_call_results"] == 21
    assert len(aggregate["files"]) == 21

    tampered = copy.deepcopy(aggregate)
    tampered["cost_boundary"]["actual_vendor_invoice_cost_usd"] = "0"
    tampered["receipt_sha256"] = tier0.self_hash(tampered, "receipt_sha256")
    staged.aggregate_path.write_bytes(tier0.canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_aggregate_receipt(
            staged.job_root,
            bundle=staged.bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
        )
    assert raised.value.status == "STOP_PAID_ALREADY_TERMINAL"

    malformed_volume_receipt = copy.deepcopy(aggregate)
    malformed_volume_receipt["volume_identity"]["free_bytes"] = True
    malformed_volume_receipt["receipt_sha256"] = tier0.self_hash(
        malformed_volume_receipt,
        "receipt_sha256",
    )
    staged.aggregate_path.write_bytes(
        tier0.canonical_json_bytes(malformed_volume_receipt) + b"\n"
    )
    malformed_current_volume = tier0.VolumeIdentity(
        mount_point=staged.volume.mount_point,
        volume_uuid=staged.volume.volume_uuid,
        device_identifier=staged.volume.device_identifier,
        filesystem=staged.volume.filesystem,
        bus_protocol=staged.volume.bus_protocol,
        st_dev=staged.volume.st_dev,
        free_bytes=True,
        total_bytes=staged.volume.total_bytes,
    )
    with pytest.raises(tier0.Tier0Error) as raised_volume:
        paid.validate_paid_aggregate_receipt(
            staged.job_root,
            bundle=staged.bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=malformed_current_volume,
        )
    assert raised_volume.value.status == "STOP_PAID_ALREADY_TERMINAL"


def test_paid_aggregate_restart_tolerates_free_space_change_but_not_stable_volume_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_full_paid_aggregate(tmp_path, monkeypatch)
    current = staged.volume
    changed_free_space = tier0.VolumeIdentity(
        mount_point=current.mount_point,
        volume_uuid=current.volume_uuid,
        device_identifier=current.device_identifier,
        filesystem=current.filesystem,
        bus_protocol=current.bus_protocol,
        st_dev=current.st_dev,
        free_bytes=current.free_bytes - 123_456,
        total_bytes=current.total_bytes,
    )
    assert paid.validate_paid_aggregate_receipt(
        staged.job_root,
        bundle=staged.bundle,
        readiness=staged.readiness,
        readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        volume=changed_free_space,
    ) == staged.aggregate

    stable_identity_drift = tier0.VolumeIdentity(
        mount_point=current.mount_point,
        volume_uuid="DIFFERENT-SYNTHETIC-UUID",
        device_identifier=current.device_identifier,
        filesystem=current.filesystem,
        bus_protocol=current.bus_protocol,
        st_dev=current.st_dev,
        free_bytes=current.free_bytes,
        total_bytes=current.total_bytes,
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_aggregate_receipt(
            staged.job_root,
            bundle=staged.bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=stable_identity_drift,
        )
    assert raised.value.status == "STOP_PAID_ALREADY_TERMINAL"


def test_paid_aggregate_refuses_equal_size_publication_bound_to_different_result_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_full_paid_aggregate(tmp_path, monkeypatch)
    summary = paid.summarize_paid_attempts(
        staged.job_root,
        staged.bundle.sessions,
        require_legacy_stop=True,
        expected_readiness_sha256=PAID_READINESS_SHA256,
        expected_readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        require_client_constructed=True,
        expected_sdk_identity_sha256=tier0.json_sha256(staged.readiness["sdk_identity"]),
    )
    target = staged.bundle.sessions[0]
    publication = next(
        record for record in summary["publication_records"] if record["session"] == target.session
    )
    historical_matching_publication = copy.deepcopy(publication)
    source_a = next(
        record
        for record in summary["timeseries_result_records"]
        if record["record_hash"]
        == publication["payload"]["source_timeseries_result_record_hash"]
    )
    source_b = copy.deepcopy(source_a)
    source_b["sequence"] = 10_000
    source_b["record_hash"] = "8" * 64
    source_b["payload"]["dbn_file_sha256"] = "9" * 64
    assert source_a["payload"]["compressed_bytes"] == source_b["payload"]["compressed_bytes"]
    assert source_a["payload"]["dbn_file_sha256"] != source_b["payload"]["dbn_file_sha256"]
    summary["timeseries_result_records"].append(source_b)
    publication["payload"]["source_timeseries_result_sequence"] = source_b["sequence"]
    publication["payload"]["source_timeseries_result_record_hash"] = source_b["record_hash"]
    monkeypatch.setattr(
        paid,
        "summarize_paid_attempts",
        lambda *_args, **_kwargs: copy.deepcopy(summary),
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.build_paid_aggregate_receipt(
            staged.job_root,
            bundle=staged.bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
        )
    assert raised.value.status == "STOP_PAID_AGGREGATE_QC"

    historical_matching_publication["attempt_id"] = SECOND_ATTEMPT_ID
    summary["publication_records"].insert(0, historical_matching_publication)
    with pytest.raises(tier0.Tier0Error) as raised_historical_mask:
        paid.build_paid_aggregate_receipt(
            staged.job_root,
            bundle=staged.bundle,
            readiness=staged.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            volume=staged.volume,
        )
    assert raised_historical_mask.value.status == "STOP_PAID_AGGREGATE_QC"


def test_paid_job_tree_refuses_fabricated_job51_aggregate_after_exact_legacy_stop(
    tmp_path: Path,
) -> None:
    job_root = _job_root(tmp_path)
    volume = _volume(job_root)
    (job_root / "RUN_LOCK_V1").write_bytes(b"immutable shared lock\n")
    paid.ensure_paid_lock_binding(
        job_root,
        volume=volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    )
    _paid_job_attempt(job_root)
    _validate_paid_anchor(job_root, allow_initialize=True)
    (job_root / "receipts" / "JOB51_ACQUISITION_QC_RECEIPT_V1.json").write_bytes(b"{}\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.validate_paid_job_tree(
            job_root,
            volume=volume,
            allowed_requests=paid.load_paid_scope_bundle(REPO).sessions,
        )
    assert raised.value.status == "STOP_PAID_EXTERNAL_TREE"


@pytest.mark.parametrize("tamper", ("scope_widening", "different_legacy_quote"))
def test_paid_scope_loader_refuses_self_rehashed_scope_or_legacy_authority_widening(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_bundle = tier0.load_scope_bundle(REPO)
    relative_paths = (
        "v5/work/cmbp-tier0-paid-resume/PLAN.md",
        "v5/work/cmbp-tier0-paid-resume/PROGRAM_CONTRACT_V1.json",
        "v5/work/cmbp-tier0-acquisition/JOB51_ZERO_COST_GATE_STOP_RECEIPT_V1.json",
    )
    for relative in relative_paths:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(paid.base, "load_scope_bundle", lambda _root: real_bundle)

    if tamper == "scope_widening":
        path = tmp_path / "v5/work/cmbp-tier0-paid-resume/PROGRAM_CONTRACT_V1.json"
        value = tier0.strict_json(path)
        value["scope"]["dataset"] = "OUTSIDE.FROZEN.SCOPE"
        value["contract_sha256"] = tier0.self_hash(value, "contract_sha256")
    else:
        path = tmp_path / "v5/work/cmbp-tier0-acquisition/JOB51_ZERO_COST_GATE_STOP_RECEIPT_V1.json"
        value = tier0.strict_json(path)
        value["cost_boundary"]["observed_sdk_quote_usd"] = "0.950392448903"
        value["receipt_sha256"] = tier0.self_hash(value, "receipt_sha256")
    path.write_bytes(tier0.canonical_json_bytes(value) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.load_paid_scope_bundle(tmp_path)
    assert raised.value.status == "STOP_PAID_CONTRACT_DRIFT"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0, "0"),
        (Decimal("0.950392448902"), "0.950392448902"),
        (1, "1"),
        (1.5, "1.5"),
        (Decimal("1.50"), "1.50"),
    ),
)
def test_paid_quote_normalizer_accepts_unsigned_finite_numeric(value: Any, expected: str) -> None:
    assert paid.normalize_paid_quote(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        "0.95",
        None,
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        float("nan"),
        float("inf"),
        -1,
        Decimal("-0"),
        -0.0,
    ),
)
def test_paid_quote_normalizer_rejects_bool_string_nonfinite_negative_and_signed_zero(value: Any) -> None:
    with pytest.raises(tier0.Tier0Error):
        paid.normalize_paid_quote(value)


def test_per_session_lifetime_cap_allows_exact_150_and_refuses_smallest_excess_before_timeseries(
    tmp_path: Path,
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()

    first = _FakeClient(Decimal("0.75"), timeseries_error=OSError("synthetic transport failure"))
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            first,
            request,
            output_path=attempt_dir / "first.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.committed_total_usd == Decimal("0.75")
    assert state.committed_by_session_usd == {request.session: Decimal("0.75")}

    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(Decimal("0.75"))
    assert paid.acquire_paid_session_bytes(
        second,
        request,
        output_path=second_attempt_dir / "second.dbn.zst",
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
    ) == "0.75"
    assert state.committed_total_usd == Decimal("1.50")
    assert state.committed_by_session_usd == {request.session: Decimal("1.50")}

    third_journal, third_attempt_dir = _journal(tmp_path, attempt_id=THIRD_ATTEMPT_ID)
    excess = _FakeClient(Decimal("0.000000000001"))
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            excess,
            request,
            output_path=third_attempt_dir / "excess.dbn.zst",
            journal=third_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert excess.cost_kwargs == [request.market_parameters]
    assert excess.timeseries_kwargs == []
    assert state.committed_total_usd == Decimal("1.50")
    assert not (third_attempt_dir / "excess.dbn.zst").exists()


def test_single_quote_above_150_stops_before_timeseries_or_commitment(tmp_path: Path) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    client = _FakeClient(Decimal("1.500000000001"))

    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=attempt_dir / "data.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.cost_kwargs == [request.market_parameters]
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == Decimal("0")
    assert state.committed_by_session_usd == {}
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    result = next(record for record in records if record["event"] == "COST_CALL_RESULT")
    assert result["payload"] == {
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "commitment_count_before": 0,
        "committed_quote_session_before_usd": "0",
        "committed_quote_session_projected_usd": "1.500000000001",
        "committed_quote_total_before_usd": "0",
        "committed_quote_total_projected_usd": "1.500000000001",
        "observed_sdk_quote_usd": "1.500000000001",
        "per_session_lifetime_cap_pass": False,
        "per_session_lifetime_cap_usd": "1.50",
        "quote_valid": True,
        "time_series_start_permitted": False,
        "total_cap_pass": True,
        "total_cap_usd": "32.00",
    }


def test_exact_zero_paid_quote_still_creates_one_disclosed_start_commitment(tmp_path: Path) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    client = _FakeClient(Decimal("0"))
    assert paid.acquire_paid_session_bytes(
        client,
        request,
        output_path=attempt_dir / "data.dbn.zst",
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
    ) == "0"
    assert state.committed_total_usd == Decimal("0")
    assert state.committed_by_session_usd == {request.session: Decimal("0")}
    assert state.commitment_count == 1
    assert len(client.timeseries_kwargs) == 1


@pytest.mark.parametrize(
    "value",
    (True, "0.50", Decimal("NaN"), Decimal("Infinity"), -1, Decimal("-0"), -0.0),
)
def test_malformed_paid_quote_is_durably_refused_before_timeseries(
    value: Any,
    tmp_path: Path,
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    client = _FakeClient(value)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=attempt_dir / "data.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == Decimal("0")
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    result = next(record for record in records if record["event"] == "COST_CALL_RESULT")
    assert result["payload"]["quote_valid"] is False
    assert result["payload"]["commitment_count_before"] == 0
    assert result["payload"]["time_series_start_permitted"] is False
    assert result["payload"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    assert "committed_quote_total_projected_usd" not in result["payload"]


def test_cost_transport_error_has_zero_commitment_no_timeseries_and_redacted_journal(
    tmp_path: Path,
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    secret_text = "synthetic-secret-that-must-not-be-journaled"
    client = _FakeClient(RuntimeError(secret_text))
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=attempt_dir / "data.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == Decimal("0")
    raw_journal = (attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl").read_text(encoding="utf-8")
    assert secret_text not in raw_journal
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    error = next(record for record in records if record["event"] == "COST_CALL_ERROR")
    assert error["payload"] == {
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "error_class": "RuntimeError",
    }


def test_high_scale_decimal_addition_cannot_round_a_session_cap_breach_down_to_150(
    tmp_path: Path,
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    quote = Decimal("0.7500000000000000000000000000000000000001")
    first = _FakeClient(quote, timeseries_error=OSError("synthetic transport failure"))
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            first,
            request,
            output_path=attempt_dir / "first.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.committed_total_usd == quote

    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(quote)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            second,
            request,
            output_path=second_attempt_dir / "second.dbn.zst",
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert second.cost_kwargs == [request.market_parameters]
    assert second.timeseries_kwargs == []
    assert state.committed_total_usd == quote


def test_pure_projection_allows_exact_3200_and_refuses_smallest_total_excess() -> None:
    """Exercise the redundant total guard without inventing a reachable 21-session ledger."""

    at_cap = paid.project_paid_commitment(
        committed_total_usd=Decimal("31.50"),
        committed_session_usd=Decimal("0"),
        fresh_quote_usd=Decimal("0.50"),
    )
    assert at_cap == {
        "committed_quote_session_after_usd": Decimal("0.50"),
        "committed_quote_total_after_usd": Decimal("32.00"),
        "per_session_lifetime_cap_pass": True,
        "total_cap_pass": True,
    }

    above = paid.project_paid_commitment(
        committed_total_usd=Decimal("31.50"),
        committed_session_usd=Decimal("0"),
        fresh_quote_usd=Decimal("0.500000000001"),
    )
    assert above["committed_quote_total_after_usd"] == Decimal("32.000000000001")
    assert above["per_session_lifetime_cap_pass"] is True
    assert above["total_cap_pass"] is False


@pytest.mark.parametrize(
    "kwargs",
    (
        {"committed_total_usd": Decimal("-0")},
        {"committed_total_usd": Decimal("NaN")},
        {"committed_total_usd": Decimal("0.10")},
        {
            "committed_total_usd": Decimal("1.500000000001"),
            "committed_by_session_usd": {"2026-07-02": Decimal("1.500000000001")},
        },
        {
            "committed_total_usd": Decimal("-0.01"),
            "committed_by_session_usd": {"2026-07-02": Decimal("-0.01")},
        },
        {"commitment_count": True},
    ),
)
def test_paid_budget_state_refuses_signed_nonfinite_inconsistent_or_over_cap_inputs(
    kwargs: Mapping[str, Any],
) -> None:
    with pytest.raises(tier0.Tier0Error):
        paid.PaidBudgetState(**kwargs)


def test_paid_run_lock_reuses_job51_lock_without_mutating_any_v1_byte(tmp_path: Path) -> None:
    job_root = tmp_path / "job51"
    volume = _volume(job_root)
    lock_path = job_root / "RUN_LOCK_V1"
    original = tier0.canonical_json_bytes(
        {
            "artifact_type": "JOB51_RUN_LOCK_V1",
            "readiness_sha256": "851ae1de70308a19fad4160bf33464a3e0a2c79bba15085bde80aea5ea99ff03",
            "scope_sha256": tier0.EXPECTED_SCOPE_SHA256,
        }
    ) + b"\n"
    lock_path.write_bytes(original)

    with paid.PaidRunLock(job_root, volume=volume):
        with pytest.raises(tier0.Tier0Error) as raised:
            paid.PaidRunLock(job_root, volume=volume)
        assert raised.value.status == "STOP_CONCURRENT_RUNNER"
        assert lock_path.read_bytes() == original
    with paid.PaidRunLock(job_root, volume=volume):
        assert lock_path.read_bytes() == original
    assert lock_path.read_bytes() == original


def test_paid_lock_binding_is_separate_idempotent_and_refuses_self_rehashed_drift(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "job51"
    volume = _volume(job_root)
    lock_path = job_root / "RUN_LOCK_V1"
    original = b"immutable Job-51 flock bytes\n"
    lock_path.write_bytes(original)

    binding = paid.ensure_paid_lock_binding(
        job_root,
        volume=volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    )
    binding_path = job_root / paid.PAID_LOCK_BINDING_NAME
    assert binding["shared_flock_file"] == "RUN_LOCK_V1"
    assert binding["shared_flock_bytes_mutated"] is False
    assert binding["paid_program_contract_sha256"] == PAID_CONTRACT_SHA256
    assert lock_path.read_bytes() == original
    assert paid.ensure_paid_lock_binding(
        job_root,
        volume=volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    ) == binding
    assert lock_path.read_bytes() == original

    tampered = copy.deepcopy(binding)
    tampered["paid_readiness_receipt_sha256"] = "8" * 64
    tampered["binding_sha256"] = tier0.self_hash(tampered, "binding_sha256")
    binding_path.write_bytes(tier0.canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.ensure_paid_lock_binding(
            job_root,
            volume=volume,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        )
    assert raised.value.status == "STOP_PAID_LOCK_BINDING"
    assert lock_path.read_bytes() == original


def test_paid_attempt_anchor_detects_deleted_latest_committed_attempt(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt_dir = _paid_job_attempt(job_root)
    state = paid.PaidBudgetState()

    initial = _validate_paid_anchor(job_root, allow_initialize=True)
    assert initial["paid_attempt_count"] == 1
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        request,
        output_path=_staging_data_path(first_attempt_dir, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )

    second_journal, second_attempt_dir = _paid_job_attempt(
        job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    repaired = _validate_paid_anchor(job_root, repair_header_only=True)
    assert repaired["paid_attempt_count"] == 2
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.50")),
        request,
        output_path=_staging_data_path(second_attempt_dir, request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    assert state.committed_total_usd == Decimal("0.90")

    shutil.rmtree(second_attempt_dir)
    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_paid_anchor(job_root)
    assert raised.value.status == "STOP_PAID_ATTEMPT_SET"


def test_paid_attempt_anchor_refuses_self_rehashed_rollback_omitting_advanced_attempt(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    _paid_job_attempt(job_root)
    _validate_paid_anchor(job_root, allow_initialize=True)
    second_journal, second_attempt_dir = _paid_job_attempt(
        job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    anchor = _validate_paid_anchor(job_root, repair_header_only=True)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.25")),
        request,
        output_path=_staging_data_path(second_attempt_dir, request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )

    rolled_back = copy.deepcopy(anchor)
    rolled_back["paid_attempts"] = [
        item for item in rolled_back["paid_attempts"] if item["attempt_id"] == ATTEMPT_ID
    ]
    rolled_back["paid_attempt_count"] = 1
    rolled_back["anchor_sha256"] = tier0.self_hash(rolled_back, "anchor_sha256")
    anchor_path = job_root / paid.PAID_ATTEMPT_ANCHOR_NAME
    anchor_path.write_bytes(tier0.canonical_json_bytes(rolled_back) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_paid_anchor(job_root, repair_header_only=True)
    assert raised.value.status == "STOP_PAID_ATTEMPT_SET"


def test_paid_attempt_anchor_cannot_be_reinitialized_after_advanced_attempt(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    _validate_paid_anchor(job_root, allow_initialize=True)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(attempt_dir, request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    (job_root / paid.PAID_ATTEMPT_ANCHOR_NAME).unlink()

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_paid_anchor(job_root, allow_initialize=True)
    assert raised.value.status == "STOP_PAID_ATTEMPT_SET"


def test_fresh_quote_and_identical_request_are_adjacent_and_commit_before_vendor_start(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    actions: list[str] = []

    class InspectingClient(_FakeClient):
        def _get_range(self, **kwargs: Any) -> None:
            assert state.committed_total_usd == Decimal("1.25")
            super()._get_range(**kwargs)

    class TrackingJournal:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            record = self.delegate.append(event, **kwargs)
            if event == "TIMESERIES_CALL_START":
                actions.append("journal.timeseries_start.durable")
            return record

    client = InspectingClient(Decimal("1.25"), actions=actions)
    gates: list[str] = []
    quote = paid.acquire_paid_session_bytes(
        client,
        request,
        output_path=attempt_dir / "data.dbn.zst",
        journal=TrackingJournal(journal),
        pre_pair_gate=lambda: gates.append("pre_pair_gate"),
        budget_state=state,
        progress=lambda message: actions.append(f"progress:{message}"),
    )
    assert quote == "1.25"
    assert gates == ["pre_pair_gate"]
    quote_index = actions.index("metadata.get_cost")
    assert actions[quote_index + 1:quote_index + 3] == [
        "journal.timeseries_start.durable",
        "timeseries.get_range",
    ]
    assert not any(action.startswith("progress:") for action in actions[quote_index + 1:quote_index + 3])
    assert client.cost_kwargs == [request.market_parameters]
    assert {
        key: client.timeseries_kwargs[0][key] for key in request.market_parameters
    } == request.market_parameters
    assert client.timeseries_kwargs[0]["stype_out"] == tier0.EXPECTED_STYPE_OUT
    assert client.timeseries_kwargs[0]["limit"] is None

    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    result_index = next(index for index, record in enumerate(records) if record["event"] == "COST_CALL_RESULT")
    result = records[result_index]
    start = records[result_index + 1]
    assert result["payload"]["commitment_count_before"] == 0
    assert start["event"] == "TIMESERIES_CALL_START"
    assert start["sequence"] == result["sequence"] + 1
    assert start["previous_hash"] == result["record_hash"]
    assert start["session"] == result["session"] == request.session
    assert start["request_sha256"] == result["request_sha256"] == request.market_request_sha256
    assert start["payload"]["fresh_quote_usd"] == "1.25"
    assert start["payload"]["committed_quote_session_before_usd"] == "0"
    assert start["payload"]["committed_quote_total_after_usd"] == "1.25"
    assert start["payload"]["committed_quote_session_after_usd"] == "1.25"
    assert start["payload"]["committed_quote_total_before_usd"] == "0"
    assert start["payload"]["per_session_lifetime_cap_usd"] == "1.50"
    assert start["payload"]["total_cap_usd"] == "32.00"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quote_only_crash_commits_zero_and_restart_must_get_a_fresh_quote(tmp_path: Path) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    actions: list[str] = []

    class SyntheticCrash(BaseException):
        pass

    class CrashBeforeDurableStart:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            if event == "TIMESERIES_CALL_START":
                raise SyntheticCrash()
            return self.delegate.append(event, **kwargs)

    first = _FakeClient(Decimal("0.40"), actions=actions)
    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            first,
            request,
            output_path=attempt_dir / "first.dbn.zst",
            journal=CrashBeforeDurableStart(journal),
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.committed_total_usd == Decimal("0")
    assert first.timeseries_kwargs == []

    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(Decimal("0.40"), actions=actions)
    assert paid.acquire_paid_session_bytes(
        second,
        request,
        output_path=second_attempt_dir / "second.dbn.zst",
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
    ) == "0.40"
    assert actions == [
        "metadata.get_cost",
        "metadata.get_cost",
        "timeseries.get_range",
    ]
    assert state.committed_total_usd == Decimal("0.40")


def test_crash_after_durable_start_before_sdk_call_leaves_full_commitment_on_disk(
    tmp_path: Path,
) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()

    class SyntheticCrash(BaseException):
        pass

    class CrashAfterDurableStart:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            record = self.delegate.append(event, **kwargs)
            if event == "TIMESERIES_CALL_START":
                raise SyntheticCrash()
            return record

    client = _FakeClient(Decimal("0.60"))
    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=attempt_dir / "data.dbn.zst",
            journal=CrashAfterDurableStart(journal),
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    assert client.timeseries_kwargs == []
    # The process-local mutation never happened, so only reconstruction may
    # recover the irrevocable commitment after this crash window.
    assert state.committed_total_usd == Decimal("0")
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    assert records[-1]["event"] == "TIMESERIES_CALL_START"
    assert records[-1]["payload"]["fresh_quote_usd"] == "0.60"
    assert records[-1]["payload"]["committed_quote_total_after_usd"] == "0.60"


def test_disk_reconstruction_counts_durable_pending_start_even_when_process_state_never_mutated(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    state = paid.PaidBudgetState()

    class SyntheticCrash(BaseException):
        pass

    class CrashAfterDurableStart:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            record = self.delegate.append(event, **kwargs)
            if event == "TIMESERIES_CALL_START":
                raise SyntheticCrash()
            return record

    client = _FakeClient(Decimal("0.60"))
    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=CrashAfterDurableStart(journal),
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    assert state.committed_total_usd == Decimal("0")

    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["committed_quote_total_usd"] == "0.60"
    assert summary["committed_quote_by_session_usd"] == {request.session: "0.60"}
    assert summary["cost_call_starts"] == 1
    assert summary["cost_call_results"] == 1
    assert summary["timeseries_call_starts"] == 1
    assert summary["timeseries_call_results"] == 0
    assert summary["pending_timeseries_starts"] == 1
    assert summary["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    reconstructed = paid.PaidBudgetState.from_summary(summary)
    assert reconstructed.committed_total_usd == Decimal("0.60")
    assert reconstructed.committed_by_session_usd == {request.session: Decimal("0.60")}


def test_disk_reconstruction_excludes_stale_quote_then_fresh_retry_commits_once(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    first_state = paid.PaidBudgetState()

    class SyntheticCrash(BaseException):
        pass

    class CrashBeforeDurableStart:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            if event == "TIMESERIES_CALL_START":
                raise SyntheticCrash()
            return self.delegate.append(event, **kwargs)

    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.40")),
            request,
            output_path=_staging_data_path(first_attempt, request.session),
            journal=CrashBeforeDurableStart(first_journal),
            pre_pair_gate=lambda: None,
            budget_state=first_state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert first_summary["committed_quote_total_usd"] == "0"
    assert first_summary["uncommitted_passing_quote_count"] == 1
    assert first_summary["timeseries_call_starts"] == 0

    retry_state = paid.PaidBudgetState.from_summary(first_summary)
    retry_journal, retry_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)
    retry_client = _FakeClient(Decimal("0.40"))
    assert paid.acquire_paid_session_bytes(
        retry_client,
        request,
        output_path=_staging_data_path(retry_attempt, request.session),
        journal=retry_journal,
        pre_pair_gate=lambda: None,
        budget_state=retry_state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    ) == "0.40"
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["cost_call_starts"] == 2
    assert summary["cost_call_results"] == 2
    assert summary["timeseries_call_starts"] == 1
    assert summary["committed_quote_total_usd"] == "0.40"
    assert summary["committed_quote_by_session_usd"] == {request.session: "0.40"}
    assert summary["uncommitted_passing_quote_count"] == 1


def test_disk_reconstruction_adds_failed_start_and_fresh_retry_without_refund(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    first_state = paid.PaidBudgetState()
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.90"), timeseries_error=OSError("synthetic failure")),
            request,
            output_path=_staging_data_path(first_attempt, request.session),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=first_state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert first_summary["committed_quote_total_usd"] == "0.90"
    assert first_summary["failed_timeseries_starts"] == 1

    retry_state = paid.PaidBudgetState.from_summary(first_summary)
    retry_journal, retry_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)
    assert paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.60")),
        request,
        output_path=_staging_data_path(retry_attempt, request.session),
        journal=retry_journal,
        pre_pair_gate=lambda: None,
        budget_state=retry_state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    ) == "0.60"
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["timeseries_call_starts"] == 2
    assert summary["timeseries_call_results"] == 1
    assert summary["failed_timeseries_starts"] == 1
    assert summary["session_timeseries_attempt_counts"] == {request.session: 2}
    assert summary["committed_quote_total_usd"] == "1.50"
    assert summary["committed_quote_by_session_usd"] == {request.session: "1.50"}


def test_disk_reconstruction_cost_error_is_zero_commitment_and_new_attempt_requotes(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(OSError("synthetic cost transport")),
            request,
            output_path=_staging_data_path(first_attempt, request.session),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert first_summary["cost_call_starts"] == 1
    assert first_summary["cost_call_errors"] == 1
    assert first_summary["timeseries_call_starts"] == 0
    assert first_summary["committed_quote_total_usd"] == "0"
    assert first_summary["terminal_authority_failure_observed"] is False

    retry_journal, retry_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)
    retry_client = _FakeClient(Decimal("0.40"))
    assert paid.acquire_paid_session_bytes(
        retry_client,
        request,
        output_path=_staging_data_path(retry_attempt, request.session),
        journal=retry_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState.from_summary(first_summary),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    ) == "0.40"
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["cost_call_starts"] == 2
    assert summary["cost_call_errors"] == 1
    assert summary["timeseries_call_starts"] == 1
    assert summary["committed_quote_total_usd"] == "0.40"


def test_disk_reconstruction_uses_commitment_index_not_random_uuid_lexical_order(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    first_request, second_request = bundle.sessions[:2]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_LAST_ATTEMPT_ID)
    first_state = paid.PaidBudgetState()
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        first_request,
        output_path=_staging_data_path(first_attempt, first_request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=first_state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    first_result = first_journal.last_record
    assert first_result is not None and first_result["event"] == "TIMESERIES_CALL_RESULT"
    (job_root / "sessions" / first_request.session).mkdir()
    _append_paid_publication(
        first_journal,
        event="SESSION_PUBLISHED",
        ordinal=1,
        request=first_request,
        result_record=first_result,
    )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_FIRST_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        second_request,
        output_path=_staging_data_path(second_attempt, second_request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState.from_summary(first_summary),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=2,
    )
    assert sorted(path.name for path in (job_root / "attempts").iterdir()) == [
        LEXICALLY_FIRST_ATTEMPT_ID,
        LEXICALLY_LAST_ATTEMPT_ID,
    ]
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["timeseries_call_starts"] == 2
    assert summary["committed_quote_total_usd"] == "0.80"
    assert summary["committed_quote_by_session_usd"] == {
        first_request.session: "0.40",
        second_request.session: "0.40",
    }


def test_commitment_indices_must_increase_in_physical_attempt_and_journal_order(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    first_request, second_request = bundle.sessions[:2]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    forged_later_state = paid.PaidBudgetState(
        committed_total_usd=Decimal("0.40"),
        committed_by_session_usd={first_request.session: Decimal("0.40")},
        commitment_count=1,
    )
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        second_request,
        output_path=_staging_data_path(attempt_dir, second_request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=forged_later_state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=2,
    )
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        first_request,
        output_path=_staging_data_path(attempt_dir, first_request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    assert [
        record["payload"]["commitment_index"]
        for record in records
        if record["event"] == "TIMESERIES_CALL_START"
    ] == [2, 1]

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_later_uncommitted_quote_cannot_reset_count_below_prior_physical_start(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        request,
        output_path=_staging_data_path(first_attempt, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)

    class SyntheticCrash(BaseException):
        pass

    class CrashBeforeDurableStart:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            if event == "TIMESERIES_CALL_START":
                raise SyntheticCrash()
            return self.delegate.append(event, **kwargs)

    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.40")),
            request,
            output_path=_staging_data_path(second_attempt, request.session),
            journal=CrashBeforeDurableStart(second_journal),
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    assert second_journal.last_record is not None
    assert second_journal.last_record["payload"]["commitment_count_before"] == 0

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_later_attempt_cannot_resume_through_prior_terminal_cap_failure(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    failed_journal, failed_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_LAST_ATTEMPT_ID)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("1.500000000001")),
            request,
            output_path=_staging_data_path(failed_attempt, request.session),
            journal=failed_journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    later_journal, later_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_FIRST_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        request,
        output_path=_staging_data_path(later_attempt, request.session),
        journal=later_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


@pytest.mark.parametrize("later_event", ("passing_quote", "cost_error", "client_constructed"))
def test_terminal_quote_failure_forbids_any_later_attempt_vendor_activity_at_same_count(
    later_event: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    failed_journal, failed_attempt = _paid_job_attempt(
        job_root,
        attempt_id=LEXICALLY_LAST_ATTEMPT_ID,
    )
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("1.500000000001")),
            request,
            output_path=_staging_data_path(failed_attempt, request.session),
            journal=failed_journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    later_journal, later_attempt = _paid_job_attempt(
        job_root,
        attempt_id=LEXICALLY_FIRST_ATTEMPT_ID,
    )
    if later_event == "passing_quote":
        class SyntheticCrash(BaseException):
            pass

        class CrashBeforeDurableStart:
            def __init__(self, delegate: tier0.AttemptJournal) -> None:
                self.delegate = delegate

            def __getattr__(self, name: str) -> Any:
                return getattr(self.delegate, name)

            def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
                if event == "TIMESERIES_CALL_START":
                    raise SyntheticCrash()
                return self.delegate.append(event, **kwargs)

        with pytest.raises(SyntheticCrash):
            paid.acquire_paid_session_bytes(
                _FakeClient(Decimal("0.40")),
                request,
                output_path=_staging_data_path(later_attempt, request.session),
                journal=CrashBeforeDurableStart(later_journal),
                pre_pair_gate=lambda: None,
                budget_state=paid.PaidBudgetState(),
                readiness_receipt_sha256=PAID_READINESS_SHA256,
                readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
                ordinal=1,
            )
    elif later_event == "cost_error":
        with pytest.raises(tier0.Tier0Error):
            paid.acquire_paid_session_bytes(
                _FakeClient(OSError("cost unavailable")),
                request,
                output_path=_staging_data_path(later_attempt, request.session),
                journal=later_journal,
                pre_pair_gate=lambda: None,
                budget_state=paid.PaidBudgetState(),
                readiness_receipt_sha256=PAID_READINESS_SHA256,
                readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
                ordinal=1,
            )
    else:
        later_journal.append(
            "CLIENT_CONSTRUCTED",
            payload={
                "credential_source": "process_environment",
                "sdk_identity_sha256": "a" * 64,
            },
        )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_AUTHORITY_TERMINAL"


def test_prior_start_then_later_terminal_cap_is_reported_without_erasing_commitment(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    first_request, second_request = bundle.sessions[:2]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_LAST_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0")),
        first_request,
        output_path=_staging_data_path(first_attempt, first_request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    first_result = first_journal.last_record
    assert first_result is not None and first_result["event"] == "TIMESERIES_CALL_RESULT"
    (job_root / "sessions" / first_request.session).mkdir()
    _append_paid_publication(
        first_journal,
        event="SESSION_PUBLISHED",
        ordinal=1,
        request=first_request,
        result_record=first_result,
    )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_FIRST_ATTEMPT_ID)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("1.500000000001")),
            second_request,
            output_path=_staging_data_path(second_attempt, second_request.session),
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState.from_summary(first_summary),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=2,
        )
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["terminal_authority_failure_observed"] is True
    assert summary["terminal_authority_failures"][0]["commitment_count_before"] == 1
    assert summary["timeseries_call_starts"] == 1
    assert summary["committed_quote_total_usd"] == "0"


def test_paid_scanner_binds_client_sdk_identity_to_readiness_identity(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _job_root(tmp_path)
    journal, _attempt_dir = _paid_job_attempt(job_root)
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "repository_root_dotenv",
            "sdk_identity_sha256": "a" * 64,
        },
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(
            job_root,
            bundle.sessions,
            require_client_constructed=True,
            expected_sdk_identity_sha256="b" * 64,
        )
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_scanner_refuses_duplicate_commitment_index_or_reset_running_total_across_attempts(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_LAST_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        request,
        output_path=_staging_data_path(first_attempt, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=LEXICALLY_FIRST_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.40")),
        request,
        output_path=_staging_data_path(second_attempt, request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        # Simulate a concurrent/forged process that ignored the first durable start.
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


def test_semantic_reconstruction_refuses_validly_hash_chained_forged_running_total(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)

    class SyntheticCrash(BaseException):
        pass

    class ForgeTotalThenCrash:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            if event == "TIMESERIES_CALL_START":
                kwargs = dict(kwargs)
                payload = dict(kwargs["payload"])
                payload["committed_quote_total_after_usd"] = "0.61"
                kwargs["payload"] = payload
                self.delegate.append(event, **kwargs)
                raise SyntheticCrash()
            return self.delegate.append(event, **kwargs)

    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.60")),
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=ForgeTotalThenCrash(journal),
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    # The generic verifier proves the forgery is structurally hash-bound; only
    # paid semantic reconstruction can reject the false arithmetic claim.
    assert tier0.verify_attempt_journal(attempt_dir)["records"][-1]["event"] == "TIMESERIES_CALL_START"
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


@pytest.mark.parametrize("marker_tamper", ("payload", "recorded_at_utc", "unexpected_field"))
def test_paid_scanner_refuses_self_rehashed_marker_semantics_that_differ_from_journal(
    marker_tamper: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.60")),
        request,
        output_path=_staging_data_path(attempt_dir, request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    marker_path = next((attempt_dir / "call-markers").glob("*-timeseries_call_start.json"))
    marker = tier0.strict_json(marker_path)
    if marker_tamper == "payload":
        marker["payload"]["committed_quote_total_after_usd"] = "0.61"
    elif marker_tamper == "recorded_at_utc":
        marker["recorded_at_utc"] = "2099-01-01T00:00:00.000000Z"
    else:
        marker["unexpected"] = "self-rehashed widening"
    marker["marker_sha256"] = tier0.self_hash(marker, "marker_sha256")
    marker_path.write_bytes(tier0.canonical_json_bytes(marker) + b"\n")
    # V1 structural verification binds the marker to the source record hash but
    # does not compare the duplicated payload; the paid semantic layer must.
    assert tier0.verify_attempt_journal(attempt_dir)["records"][-1]["event"] == "TIMESERIES_CALL_RESULT"
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


def test_paid_scanner_refuses_unexpected_attempt_tree_child(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _job_root(tmp_path)
    _journal, attempt_dir = _paid_job_attempt(job_root)
    (attempt_dir / "operator-note.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


@pytest.mark.parametrize(
    "tamper",
    ("quote", "cost_result_hash", "request_sha", "session", "parameters"),
)
def test_paid_scanner_refuses_cost_result_start_adjacency_or_identity_tamper(
    tamper: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)

    class SyntheticCrash(BaseException):
        pass

    class TamperStartThenCrash:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def append(self, event: str, **kwargs: Any) -> dict[str, Any]:
            if event != "TIMESERIES_CALL_START":
                return self.delegate.append(event, **kwargs)
            kwargs = dict(kwargs)
            payload = copy.deepcopy(kwargs["payload"])
            if tamper == "quote":
                payload["fresh_quote_usd"] = "0.61"
            elif tamper == "cost_result_hash":
                payload["cost_result_record_hash"] = "0" * 64
            elif tamper == "request_sha":
                kwargs["request_sha256"] = "f" * 64
            elif tamper == "session":
                kwargs["session"] = bundle.sessions[1].session
            else:
                payload["parameters"]["symbols"] = ["OUTSIDE_FROZEN_SCOPE"]
            kwargs["payload"] = payload
            self.delegate.append(event, **kwargs)
            raise SyntheticCrash()

    with pytest.raises(SyntheticCrash):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.60")),
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=TamperStartThenCrash(journal),
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    tier0.verify_attempt_journal(attempt_dir)
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


def test_semantic_reconstruction_refuses_validly_journaled_out_of_scope_call_and_staging(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    outside = tier0.SessionRequest(
        session="2099-01-01",
        start="2099-01-01T14:30:00Z",
        end="2099-01-01T21:00:00Z",
        symbols=("SPXW  990101C06000000",),
        expected_mappings=((99999, "SPXW  990101C06000000"),),
        expected_record_count=1,
        expected_cost_usd="0",
        cost_request_sha256="d" * 64,
        record_count_request_sha256="e" * 64,
        symbology_request_sha256="f" * 64,
    )
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        outside,
        output_path=_staging_data_path(attempt_dir, outside.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


@pytest.mark.parametrize("quote", (Decimal("1.500000000001"), "0.10", Decimal("-0")))
def test_disk_reconstruction_makes_cap_or_malformed_quote_terminal_across_resume(
    quote: Any,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(quote),
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["terminal_authority_failure_observed"] is True
    assert summary["committed_quote_total_usd"] == "0"
    assert summary["timeseries_call_starts"] == 0
    assert summary["terminal_authority_failures"][0]["session"] == request.session
    assert summary["terminal_authority_failures"][0]["request_sha256"] == request.market_request_sha256


def test_paid_scanner_refuses_timeseries_start_after_failed_cap_decision(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    state = paid.PaidBudgetState()
    output_path = _staging_data_path(attempt_dir, request.session)
    quote = Decimal("1.500000000001")
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(quote),
            request,
            output_path=output_path,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    cost_result = journal.last_record
    assert cost_result is not None and cost_result["event"] == "COST_CALL_RESULT"
    projection = state.projection(request.session, quote)
    journal.append(
        "TIMESERIES_CALL_START",
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload=paid._paid_start_payload(
            request=request,
            output_path=output_path,
            quote_text=str(quote),
            cost_result_record_hash=cost_result["record_hash"],
            state=state,
            projection=projection,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        ),
    )
    with pytest.raises(tier0.Tier0Error):
        paid.summarize_paid_attempts(job_root, bundle.sessions)


def test_time_series_start_commitment_is_not_refunded_after_transport_failure(tmp_path: Path) -> None:
    request = _request()
    journal, attempt_dir = _journal(tmp_path)
    state = paid.PaidBudgetState()
    first = _FakeClient(
        Decimal("0.90"),
        timeseries_error=OSError("partial stream"),
        partial_bytes_before_error=b"preserved-partial-stream",
    )

    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            first,
            request,
            output_path=attempt_dir / "first.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.committed_total_usd == Decimal("0.90")
    assert state.committed_by_session_usd == {request.session: Decimal("0.90")}
    assert (attempt_dir / "first.dbn.zst").read_bytes() == b"preserved-partial-stream"
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    error = next(record for record in records if record["event"] == "TIMESERIES_CALL_ERROR")
    assert error["payload"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"

    retry_journal, retry_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    retry = _FakeClient(Decimal("0.60"))
    assert paid.acquire_paid_session_bytes(
        retry,
        request,
        output_path=retry_attempt_dir / "retry.dbn.zst",
        journal=retry_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
    ) == "0.60"
    assert state.committed_total_usd == Decimal("1.50")
    assert len(first.cost_kwargs) == len(first.timeseries_kwargs) == 1
    assert len(retry.cost_kwargs) == len(retry.timeseries_kwargs) == 1


@pytest.mark.parametrize("failed_call", ("cost", "timeseries"))
def test_paid_scanner_requires_new_process_attempt_after_vendor_call_error(
    failed_call: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    journal, attempt_dir = _paid_job_attempt(job_root)
    output_path = _staging_data_path(attempt_dir, request.session)
    state = paid.PaidBudgetState()
    failed_client = (
        _FakeClient(OSError("metadata unavailable"))
        if failed_call == "cost"
        else _FakeClient(Decimal("0.40"), timeseries_error=OSError("stream interrupted"))
    )
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            failed_client,
            request,
            output_path=output_path,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.50")),
        request,
        output_path=output_path,
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


@pytest.mark.parametrize(
    "scenario",
    ("first_reused", "recovered_after_publication", "duplicate_same_attempt"),
)
def test_paid_publication_modes_follow_global_state_machine(
    scenario: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    (job_root / "sessions" / request.session).mkdir()
    journal, attempt_dir = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(attempt_dir, request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    result = journal.last_record
    assert result is not None and result["event"] == "TIMESERIES_CALL_RESULT"
    if scenario == "first_reused":
        _append_paid_publication(
            journal,
            event="SESSION_REUSED",
            ordinal=1,
            request=request,
            result_record=result,
        )
    else:
        _append_paid_publication(
            journal,
            event="SESSION_PUBLISHED",
            ordinal=1,
            request=request,
            result_record=result,
        )
        if scenario == "recovered_after_publication":
            later_journal, _later_attempt = _paid_job_attempt(
                job_root,
                attempt_id=SECOND_ATTEMPT_ID,
            )
            target_journal = later_journal
            event = "SESSION_RECOVERED"
        else:
            target_journal = journal
            event = "SESSION_REUSED"
        _append_paid_publication(
            target_journal,
            event=event,
            ordinal=1,
            request=request,
            result_record=result,
        )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_paid_publication_cannot_cite_result_from_future_attempt_lineage(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    (job_root / "sessions" / request.session).mkdir()
    first_journal, _first_attempt = _paid_job_attempt(job_root)
    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(second_attempt, request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    future_result = second_journal.last_record
    assert future_result is not None and future_result["event"] == "TIMESERIES_CALL_RESULT"
    _append_paid_publication(
        first_journal,
        event="SESSION_RECOVERED",
        ordinal=1,
        request=request,
        result_record=future_result,
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_global_successful_result_barrier_forbids_next_attempt_client_before_recovery(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(first_attempt, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    result = first_journal.last_record
    assert result is not None and result["event"] == "TIMESERIES_CALL_RESULT"

    second_journal, _second_attempt = _paid_job_attempt(
        job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    second_journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "repository_root_dotenv",
            "sdk_identity_sha256": "a" * 64,
        },
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_global_successful_result_barrier_rejects_duplicate_successful_stream(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    first_journal, first_attempt = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(first_attempt, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    first_summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert first_summary["successful_stream_recovery_required"] is True

    second_journal, second_attempt = _paid_job_attempt(
        job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(second_attempt, request.session),
        journal=second_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState.from_summary(first_summary),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_paid_result_must_be_published_before_next_vendor_pair_in_same_attempt(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    first_request, second_request = bundle.sessions[:2]
    job_root = _job_root(tmp_path)
    for request in (first_request, second_request):
        (job_root / "sessions" / request.session).mkdir()
    journal, attempt_dir = _paid_job_attempt(job_root)
    state = paid.PaidBudgetState()
    results: dict[str, Mapping[str, Any]] = {}
    for ordinal, request in enumerate((first_request, second_request), start=1):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.10")),
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=ordinal,
        )
        result = journal.last_record
        assert result is not None and result["event"] == "TIMESERIES_CALL_RESULT"
        results[request.session] = result
    for ordinal, request in enumerate((first_request, second_request), start=1):
        _append_paid_publication(
            journal,
            event="SESSION_PUBLISHED",
            ordinal=ordinal,
            request=request,
            result_record=results[request.session],
        )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


@pytest.mark.parametrize("publication_mode", ("SESSION_PUBLISHED", "SESSION_RECOVERED", "SESSION_REUSED"))
def test_paid_publication_forbids_any_later_vendor_call_for_same_session(
    publication_mode: str,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    request = bundle.sessions[0]
    job_root = _job_root(tmp_path)
    (job_root / "sessions" / request.session).mkdir()
    state = paid.PaidBudgetState()
    first_journal, first_attempt = _paid_job_attempt(job_root)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(first_attempt, request.session),
        journal=first_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    source = first_journal.last_record
    assert source is not None and source["event"] == "TIMESERIES_CALL_RESULT"
    if publication_mode == "SESSION_PUBLISHED":
        _append_paid_publication(
            first_journal,
            event=publication_mode,
            ordinal=1,
            request=request,
            result_record=source,
        )
    else:
        publication_journal, _publication_attempt = _paid_job_attempt(
            job_root,
            attempt_id=SECOND_ATTEMPT_ID,
        )
        if publication_mode == "SESSION_REUSED":
            _append_paid_publication(
                first_journal,
                event="SESSION_PUBLISHED",
                ordinal=1,
                request=request,
                result_record=source,
            )
        _append_paid_publication(
            publication_journal,
            event=publication_mode,
            ordinal=1,
            request=request,
            result_record=source,
        )
    call_attempt_id = (
        THIRD_ATTEMPT_ID
        if publication_mode in {"SESSION_RECOVERED", "SESSION_REUSED"}
        else SECOND_ATTEMPT_ID
    )
    call_journal, call_attempt = _paid_job_attempt(job_root, attempt_id=call_attempt_id)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=_staging_data_path(call_attempt, request.session),
        journal=call_journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert raised.value.status == "STOP_PAID_JOURNAL_INVALID"


def test_missing_paid_lock_binding_cannot_be_recreated_after_paid_state_exists(
    tmp_path: Path,
) -> None:
    job_root = _job_root(tmp_path)
    volume = _volume(job_root)
    (job_root / "RUN_LOCK_V1").write_bytes(b"immutable Job-51 lock\n")
    paid.ensure_paid_lock_binding(
        job_root,
        volume=volume,
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
    )
    _paid_job_attempt(job_root)
    _validate_paid_anchor(job_root, allow_initialize=True)
    (job_root / paid.PAID_LOCK_BINDING_NAME).unlink()

    with pytest.raises(tier0.Tier0Error) as raised:
        paid.ensure_paid_lock_binding(
            job_root,
            volume=volume,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        )
    assert raised.value.status == "STOP_PAID_LOCK_BINDING"


def test_partial_resume_seal_accepts_prior_publications_and_exact_current_epoch_reuse(
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _job_root(tmp_path)
    state = paid.PaidBudgetState()
    for request in bundle.sessions:
        (job_root / "sessions" / request.session).mkdir()
    first_journal, first_attempt = _paid_job_attempt(job_root)
    prior_results: dict[str, Mapping[str, Any]] = {}
    completed_before_crash = 5
    for ordinal, request in enumerate(bundle.sessions[:completed_before_crash], start=1):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.10")),
            request,
            output_path=_staging_data_path(first_attempt, request.session),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=ordinal,
        )
        result = first_journal.last_record
        assert result is not None and result["event"] == "TIMESERIES_CALL_RESULT"
        prior_results[request.session] = result
        _append_paid_publication(
            first_journal,
            event="SESSION_PUBLISHED",
            ordinal=ordinal,
            request=request,
            result_record=result,
        )

    second_journal, second_attempt = _paid_job_attempt(job_root, attempt_id=SECOND_ATTEMPT_ID)
    for ordinal, request in enumerate(bundle.sessions[:completed_before_crash], start=1):
        _append_paid_publication(
            second_journal,
            event="SESSION_REUSED",
            ordinal=ordinal,
            request=request,
            result_record=prior_results[request.session],
        )
    for ordinal, request in enumerate(
        bundle.sessions[completed_before_crash:],
        start=completed_before_crash + 1,
    ):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("0.10")),
            request,
            output_path=_staging_data_path(second_attempt, request.session),
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=ordinal,
        )
        result = second_journal.last_record
        assert result is not None and result["event"] == "TIMESERIES_CALL_RESULT"
        _append_paid_publication(
            second_journal,
            event="SESSION_PUBLISHED",
            ordinal=ordinal,
            request=request,
            result_record=result,
        )
    second_journal.append(
        "ATTEMPT_SEALED_FOR_AGGREGATE",
        payload={
            "published_or_reused_sessions": 21,
            "decoded_records": tier0.EXPECTED_RECORD_COUNT,
            "commitment_count": state.commitment_count,
            "committed_quote_total_usd": "2.10",
            "paid_readiness_receipt_sha256": PAID_READINESS_SHA256,
        },
    )

    summary = paid.summarize_paid_attempts(job_root, bundle.sessions)
    assert summary["aggregate_seal_count"] == 1
    assert summary["timeseries_call_starts"] == 21
    assert summary["committed_quote_total_usd"] == "2.10"
    assert summary["published_sessions_in_journals"] == [
        request.session for request in bundle.sessions
    ]
    assert len(summary["publication_records"]) == 21 + completed_before_crash
    second_records = tier0.verify_attempt_journal(second_attempt)["records"]
    assert sum(
        record["event"] in {"SESSION_PUBLISHED", "SESSION_REUSED", "SESSION_RECOVERED"}
        for record in second_records
    ) == 21


def test_all_21_session_lifetime_caps_reach_exact_reachable_maximum_3150(tmp_path: Path) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    journal, attempt_dir = _journal(tmp_path)
    output_root = tmp_path / "paid-output"
    output_root.mkdir()
    state = paid.PaidBudgetState()
    actions: list[str] = []

    for request in bundle.sessions:
        client = _FakeClient(Decimal("1.50"), actions=actions)
        assert paid.acquire_paid_session_bytes(
            client,
            request,
            output_path=output_root / f"{request.session}.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        ) == "1.50"

    assert state.committed_total_usd == Decimal("31.50")
    assert state.committed_by_session_usd == {
        request.session: Decimal("1.50") for request in bundle.sessions
    }
    assert actions == [
        action
        for _request_item in bundle.sessions
        for action in ("metadata.get_cost", "timeseries.get_range")
    ]
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    starts = [record for record in records if record["event"] == "TIMESERIES_CALL_START"]
    assert len(starts) == 21
    assert starts[-1]["payload"]["committed_quote_total_after_usd"] == "31.50"


@pytest.mark.parametrize(
    "module_name, expected_message",
    (
        ("acquire_cmbp_tier0_paid", "STOP_UNEXPECTED_ARGUMENTS: this paid runner accepts no arguments"),
        ("seal_cmbp_tier0_paid_readiness", "STOP_UNEXPECTED_ARGUMENTS: this paid sealer accepts no arguments"),
    ),
)
def test_paid_ops_argv_refusal_precedes_research_imports(
    module_name: str,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if module_name == "acquire_cmbp_tier0_paid":
        from v5.ops import acquire_cmbp_tier0_paid as operation
    else:
        from v5.ops import seal_cmbp_tier0_paid_readiness as operation

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("v5.research"):
            raise AssertionError(f"argument refusal imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", [str(Path(operation.__file__)), "unexpected"])
    assert operation.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected_message in captured.err


def test_paid_runner_rejects_proxy_tls_ca_netrc_environment_before_credential_and_vendor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    secret = "synthetic-secret-must-not-appear"
    for key in (
        "HTTPS_PROXY",
        "all_proxy",
        "Requests_CA_Bundle",
        "curl_ca_bundle",
        "SSL_CERT_FILE",
        "ssl_cert_dir",
        "SSLKEYLOGFILE",
        "netrc",
    ):
        monkeypatch.setenv(key, secret)
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or (secret, "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key) or _FakeClient(Decimal("0.10")),
    )

    assert context.runner.main() == 2
    captured = capsys.readouterr()
    assert credential_calls == []
    assert client_calls == []
    assert secret not in captured.out + captured.err
    assert "STOP_" in captured.err
    assert (context.job_root / "RUN_LOCK_V1").read_bytes() == context.run_lock_bytes

    attempt_dir = context.job_root / "attempts" / ATTEMPT_ID
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    assert [record["event"] for record in records] == ["ATTEMPT_START", "ATTEMPT_STOP"]
    anchor = _validate_paid_anchor(context.job_root)
    assert anchor["paid_attempt_count"] == 1
    assert anchor["paid_attempts"][0]["attempt_id"] == ATTEMPT_ID
    assert anchor["paid_attempts"][0]["header_record_hash"] == records[0]["record_hash"]


@pytest.mark.parametrize("netrc_name, dangling", ((".netrc", False), ("_netrc", True)))
def test_paid_runner_rejects_default_netrc_path_without_reading_it_or_accessing_credential(
    netrc_name: str,
    dangling: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    netrc = context.home / netrc_name
    if dangling:
        netrc.symlink_to(context.home / "missing-secret-target")
    else:
        netrc.write_text("this secret file must never be read\n", encoding="utf-8")
        netrc.chmod(0)
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    try:
        assert context.runner.main() == 2
    finally:
        if not dangling:
            netrc.chmod(0o600)
    assert credential_calls == []
    assert client_calls == []
    records = tier0.verify_attempt_journal(context.job_root / "attempts" / ATTEMPT_ID)["records"]
    assert [record["event"] for record in records] == ["ATTEMPT_START", "ATTEMPT_STOP"]


def test_paid_runner_rechecks_request_environment_immediately_before_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    real_guard = context.runner._reject_request_environment_overrides
    guard_calls: list[int] = []
    credential_calls: list[str] = []
    client_calls: list[str] = []

    def changing_guard() -> None:
        guard_calls.append(len(guard_calls) + 1)
        if len(guard_calls) == 2:
            monkeypatch.setenv("https_proxy", "https://late-hostile.invalid")
        real_guard()

    monkeypatch.setattr(context.runner, "_reject_request_environment_overrides", changing_guard)
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 2
    assert guard_calls == [1, 2]
    assert credential_calls == []
    assert client_calls == []


def test_paid_runner_prior_terminal_quote_failure_stops_before_new_attempt_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    _initialize_paid_runner_controls(context)
    request = context.bundle.sessions[0]
    journal, attempt_dir = _paid_job_attempt(
        context.job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    _validate_paid_anchor(context.job_root, repair_header_only=True)
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "process_environment",
            "sdk_identity_sha256": tier0.json_sha256(context.readiness["sdk_identity"]),
        },
    )
    with pytest.raises(tier0.Tier0Error):
        paid.acquire_paid_session_bytes(
            _FakeClient(Decimal("1.500000000001")),
            request,
            output_path=_staging_data_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=paid.PaidBudgetState(),
            readiness_receipt_sha256=PAID_READINESS_SHA256,
            readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
            ordinal=1,
        )
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 2
    assert credential_calls == []
    assert client_calls == []
    assert not (context.job_root / "attempts" / ATTEMPT_ID).exists()
    assert tier0.verify_attempt_journal(attempt_dir)["records"][-1]["event"] == "COST_CALL_RESULT"


def test_paid_runner_invalid_existing_aggregate_stops_before_attempt_credential_or_vendor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    aggregate_path = context.job_root / "receipts" / paid.PAID_AGGREGATE_NAME
    aggregate_path.write_bytes(b'{"artifact_type":"forged"}\n')
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 2
    assert credential_calls == []
    assert client_calls == []
    assert not (context.job_root / "attempts" / ATTEMPT_ID).exists()
    assert aggregate_path.read_bytes() == b'{"artifact_type":"forged"}\n'


def test_paid_runner_lost_successful_stream_stops_before_new_attempt_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    _initialize_paid_runner_controls(context)
    request = context.bundle.sessions[0]
    journal, attempt_dir = _paid_job_attempt(
        context.job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    _validate_paid_anchor(context.job_root, repair_header_only=True)
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "process_environment",
            "sdk_identity_sha256": tier0.json_sha256(context.readiness["sdk_identity"]),
        },
    )
    data_path = _staging_data_path(attempt_dir, request.session)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=data_path,
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    shutil.rmtree(data_path.parent)
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 2
    assert credential_calls == []
    assert client_calls == []
    assert not (context.job_root / "attempts" / ATTEMPT_ID).exists()


@pytest.mark.parametrize("recovery_kind", ("staging", "renamed_final"))
def test_paid_runner_recovers_successful_stream_locally_before_credential(
    recovery_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    _initialize_paid_runner_controls(context)
    request = context.bundle.sessions[0]
    prior_journal, prior_attempt = _paid_job_attempt(
        context.job_root,
        attempt_id=SECOND_ATTEMPT_ID,
    )
    _validate_paid_anchor(context.job_root, repair_header_only=True)
    prior_journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "process_environment",
            "sdk_identity_sha256": tier0.json_sha256(context.readiness["sdk_identity"]),
        },
    )
    data_path = _staging_data_path(prior_attempt, request.session)
    paid.acquire_paid_session_bytes(
        _FakeClient(Decimal("0.10")),
        request,
        output_path=data_path,
        journal=prior_journal,
        pre_pair_gate=lambda: None,
        budget_state=paid.PaidBudgetState(),
        readiness_receipt_sha256=PAID_READINESS_SHA256,
        readiness_receipt_file_sha256=PAID_READINESS_FILE_SHA256,
        ordinal=1,
    )
    records = tier0.verify_attempt_journal(prior_attempt)["records"]
    result = records[-1]
    start = next(
        record
        for record in records
        if record["record_hash"] == result["payload"]["timeseries_start_record_hash"]
    )
    requests = {item.session: item for item in context.bundle.sessions}
    _patch_dbn_store(monkeypatch, requests)
    monkeypatch.setattr(
        tier0,
        "stream_dbn_qc",
        lambda _path, actual_request, *, progress: (
            tier0.validate_dbn_metadata(_FakeDBNStore(actual_request), actual_request),
            _decoder_summary(actual_request),
        ),
    )
    if recovery_kind == "renamed_final":
        qc = paid.build_paid_session_qc(
            request=request,
            data_path=data_path,
            metadata_summary=tier0.validate_dbn_metadata(_FakeDBNStore(request), request),
            decoder_summary=_decoder_summary(request),
            readiness=context.readiness,
            readiness_file_sha256=PAID_READINESS_FILE_SHA256,
            start_record=start,
            result_record=result,
        )
        (data_path.parent / paid.PAID_SESSION_QC_NAME).write_bytes(
            tier0.canonical_json_bytes(qc) + b"\n"
        )
        tier0.publish_session_bundle(
            data_path.parent,
            context.job_root / "sessions" / request.session,
            volume=context.volume,
        )
    preflight_summary = paid.summarize_paid_attempts(
        context.job_root,
        context.bundle.sessions,
        require_legacy_stop=True,
        expected_readiness_sha256=PAID_READINESS_SHA256,
        expected_readiness_file_sha256=PAID_READINESS_FILE_SHA256,
        require_client_constructed=True,
        expected_sdk_identity_sha256=tier0.json_sha256(context.readiness["sdk_identity"]),
    )
    assert preflight_summary["successful_stream_recovery_required"] is True
    assert preflight_summary["successful_stream_recovery_record"]["record_hash"] == result["record_hash"]
    if recovery_kind == "staging":
        assert len(preflight_summary["recoverable_staging"]) == 1
    credential_checks: list[str] = []
    client_calls: list[str] = []

    def credential_after_recovery() -> tuple[str, str]:
        credential_checks.append("checked")
        final_dir = context.job_root / "sessions" / request.session
        assert final_dir.is_dir()
        assert (final_dir / "data.cmbp-1.dbn.zst").is_file()
        assert (final_dir / paid.PAID_SESSION_QC_NAME).is_file()
        assert not data_path.parent.exists()
        recovery_records = tier0.verify_attempt_journal(
            context.job_root / "attempts" / ATTEMPT_ID
        )["records"]
        assert any(
            record["event"] == "SESSION_RECOVERED" and record["session"] == request.session
            for record in recovery_records
        )
        raise RuntimeError("synthetic stop after proving precredential recovery")

    monkeypatch.setattr(context.runner, "_read_api_key_after_gates", credential_after_recovery)
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 2
    recovery_records = tier0.verify_attempt_journal(
        context.job_root / "attempts" / ATTEMPT_ID
    )["records"]
    assert credential_checks == ["checked"], [
        (record["event"], record.get("session")) for record in recovery_records
    ]
    assert client_calls == []


def test_paid_runner_anchor_update_failure_leaves_header_only_without_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    real_validate_anchor = paid.validate_paid_attempt_anchor
    calls: list[dict[str, Any]] = []

    def fail_header_registration(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls.append(dict(kwargs))
        if len(calls) == 2:
            raise tier0.Tier0Error(
                "synthetic anchor update crash",
                status="STOP_PAID_ATTEMPT_ANCHOR",
            )
        return real_validate_anchor(*args, **kwargs)

    monkeypatch.setattr(paid, "validate_paid_attempt_anchor", fail_header_registration)
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: (_ for _ in ()).throw(AssertionError("credential read after anchor failure")),
    )

    assert context.runner.main() == 2
    assert len(calls) == 2
    records = tier0.verify_attempt_journal(context.job_root / "attempts" / ATTEMPT_ID)["records"]
    assert [record["event"] for record in records] == ["ATTEMPT_START"]
    monkeypatch.setattr(paid, "validate_paid_attempt_anchor", real_validate_anchor)
    anchor = _validate_paid_anchor(context.job_root, repair_header_only=True)
    assert anchor["paid_attempt_count"] == 1


def test_paid_runner_does_not_append_stop_after_aggregate_bytes_become_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_full_paid_aggregate(
        tmp_path / "synthetic-volume" / "cmbp-tier0",
        monkeypatch,
    )
    staged.aggregate_path.unlink()
    context = _patch_paid_runner_context(
        tmp_path,
        monkeypatch,
        job_root=staged.job_root,
    )
    journal_path = staged.attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl"
    journal_before = journal_path.read_bytes()

    def durable_write_then_crash(path: Path, payload: Mapping[str, Any]) -> None:
        assert Path(path) == staged.aggregate_path
        Path(path).write_bytes(tier0.canonical_json_bytes(payload) + b"\n")
        raise OSError("synthetic crash after aggregate durability")

    monkeypatch.setattr(tier0, "write_canonical_exclusive", durable_write_then_crash)
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: (_ for _ in ()).throw(AssertionError("credential read during local aggregate recovery")),
    )

    assert context.runner.main() == 2
    assert staged.aggregate_path.is_file()
    assert journal_path.read_bytes() == journal_before
    assert tier0.verify_attempt_journal(staged.attempt_dir)["records"][-1]["event"] == "ATTEMPT_SEALED_FOR_AGGREGATE"


def test_paid_runner_mocked_21_session_success_preserves_exact_pair_order_and_publishes_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_paid_runner_context(tmp_path, monkeypatch)
    actions: list[str] = []
    client = _FakeClient(Decimal("0.10"), actions=actions)
    credential_calls: list[str] = []
    client_keys: list[str] = []
    secret = "synthetic-paid-key-never-print"
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("read") or (secret, "process_environment"),
    )

    def construct_client(key: str) -> _FakeClient:
        client_keys.append(key)
        assert key == secret
        return client

    monkeypatch.setattr(context.runner, "_construct_exact_client", construct_client)
    requests = {request.session: request for request in context.bundle.sessions}
    _patch_dbn_store(monkeypatch, requests)
    monkeypatch.setattr(
        tier0,
        "stream_dbn_qc",
        lambda _path, request, *, progress: (
            tier0.validate_dbn_metadata(_FakeDBNStore(request), request),
            _decoder_summary(request),
        ),
    )

    assert context.runner.main() == 0
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert credential_calls == ["read"]
    assert client_keys == [secret]
    assert actions == [
        action
        for _request in context.bundle.sessions
        for action in ("metadata.get_cost", "timeseries.get_range")
    ]
    assert len(client.cost_kwargs) == 21
    assert len(client.timeseries_kwargs) == 21
    for request, cost_kwargs, stream_kwargs in zip(
        context.bundle.sessions,
        client.cost_kwargs,
        client.timeseries_kwargs,
        strict=True,
    ):
        assert cost_kwargs == request.market_parameters
        assert {key: value for key, value in stream_kwargs.items() if key != "path"} == {
            **request.market_parameters,
            "stype_out": tier0.EXPECTED_STYPE_OUT,
            "limit": None,
        }
        assert Path(stream_kwargs["path"]).name == "data.cmbp-1.dbn.zst"

    assert not list((context.job_root / "attempts" / ATTEMPT_ID / "sessions").glob("*.bundle.part"))
    assert sorted(path.name for path in (context.job_root / "sessions").iterdir()) == sorted(requests)
    for request in context.bundle.sessions:
        final_dir = context.job_root / "sessions" / request.session
        assert (final_dir / "data.cmbp-1.dbn.zst").read_bytes() == b"synthetic-paid-dbn-stream"
        assert (final_dir / paid.PAID_SESSION_QC_NAME).is_file()

    aggregate_path = context.job_root / "receipts" / paid.PAID_AGGREGATE_NAME
    aggregate = tier0.strict_json(aggregate_path)
    assert aggregate["status"] == "JOB52_CMBP_TIER0_PAID_ACQUISITION_AND_QC_PASS"
    assert aggregate["cost_boundary"]["committed_quote_total_usd"] == "2.10"
    assert aggregate["cost_boundary"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    assert aggregate["call_accounting"]["paid_cost_call_starts"] == 21
    assert aggregate["call_accounting"]["paid_timeseries_call_starts"] == 21
    assert aggregate["call_accounting"]["paid_timeseries_call_results"] == 21
    assert aggregate["completeness"]["published_sessions"] == 21
    assert (context.job_root / "RUN_LOCK_V1").read_bytes() == context.run_lock_bytes

    records = tier0.verify_attempt_journal(context.job_root / "attempts" / ATTEMPT_ID)["records"]
    assert records[-1]["event"] == "ATTEMPT_SEALED_FOR_AGGREGATE"
    assert sum(record["event"] == "COST_CALL_RESULT" for record in records) == 21
    assert sum(record["event"] == "TIMESERIES_CALL_START" for record in records) == 21
    assert sum(record["event"] == "SESSION_PUBLISHED" for record in records) == 21
    for index, record in enumerate(records):
        if record["event"] == "COST_CALL_RESULT":
            assert records[index + 1]["event"] == "TIMESERIES_CALL_START"


def test_paid_runner_reconstructs_missing_aggregate_from_sealed_finals_without_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_full_paid_aggregate(
        tmp_path / "synthetic-volume" / "cmbp-tier0",
        monkeypatch,
    )
    staged.aggregate_path.unlink()
    journal_before = (staged.attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl").read_bytes()
    context = _patch_paid_runner_context(
        tmp_path,
        monkeypatch,
        job_root=staged.job_root,
    )
    credential_calls: list[str] = []
    client_calls: list[str] = []
    monkeypatch.setattr(
        context.runner,
        "_read_api_key_after_gates",
        lambda: credential_calls.append("credential") or ("forbidden", "process_environment"),
    )
    monkeypatch.setattr(
        context.runner,
        "_construct_exact_client",
        lambda key: client_calls.append(key),
    )

    assert context.runner.main() == 0
    assert credential_calls == []
    assert client_calls == []
    assert staged.aggregate_path.is_file()
    assert (staged.attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl").read_bytes() == journal_before
    assert not (staged.job_root / "attempts" / SECOND_ATTEMPT_ID).exists()


def test_paid_sealer_runs_exact_sterile_command_and_atomically_publishes_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_paid_readiness as sealer

    repo = tmp_path / "repo"
    work = repo / "v5/work/cmbp-tier0-paid-resume"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    temporary = work / f".TEST_RESULTS_V1.{os.getpid()}.xml.tmp"
    junit_bytes = b"<testsuites tests='117' failures='0' errors='0' skipped='0'/>\n"
    readiness_bytes = b'{"synthetic_paid_readiness":"sealed"}\n'
    run_calls: list[dict[str, Any]] = []
    seal_calls: list[Path] = []
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sealer, "WORK", work)
    monkeypatch.setattr(sealer, "REPORT", report)
    monkeypatch.setattr(sealer, "READINESS", readiness)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])
    for key, value in {
        "DATABENTO_API_KEY": "must-not-reach-child",
        "PYTEST_ADDOPTS": "--collect-only",
        "PYTEST_PLUGINS": "hostile_plugin",
        "HTTPS_PROXY": "https://hostile.invalid",
        "SSLKEYLOGFILE": str(tmp_path / "tls.keys"),
    }.items():
        monkeypatch.setenv(key, value)

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        check: bool,
    ) -> SimpleNamespace:
        run_calls.append(
            {
                "command": list(command),
                "cwd": Path(cwd),
                "env": dict(env),
                "check": check,
            }
        )
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        Path(junit_arg.removeprefix("--junitxml=")).write_bytes(junit_bytes)
        return SimpleNamespace(returncode=0)

    def fake_seal(actual_repo: Path) -> Path:
        seal_calls.append(Path(actual_repo))
        assert report.read_bytes() == junit_bytes
        assert report.stat().st_nlink == 1
        assert not temporary.exists()
        readiness.write_bytes(readiness_bytes)
        return readiness

    monkeypatch.setattr(sealer.subprocess, "run", fake_run)
    monkeypatch.setattr(tier0, "fsync_directory", lambda path: None)
    monkeypatch.setattr(paid, "seal_paid_readiness", fake_seal)

    assert sealer.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == str(readiness)
    assert seal_calls == [repo]
    assert report.read_bytes() == junit_bytes
    assert readiness.read_bytes() == readiness_bytes
    assert not temporary.exists()
    assert run_calls == [
        {
            "command": [
                str(repo / ".venv/bin/python"),
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                f"--rootdir={repo}",
                "-p",
                "no:cacheprovider",
                "-q",
                *sealer.TESTS,
                f"--junitxml={temporary}",
            ],
            "cwd": repo,
            "env": {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
            "check": False,
        }
    ]


def test_paid_sealer_failed_tests_retain_only_private_diagnostic_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_paid_readiness as sealer

    repo = tmp_path / "repo"
    work = repo / "v5/work/cmbp-tier0-paid-resume"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    temporary = work / f".TEST_RESULTS_V1.{os.getpid()}.xml.tmp"
    seal_calls: list[Path] = []
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sealer, "WORK", work)
    monkeypatch.setattr(sealer, "REPORT", report)
    monkeypatch.setattr(sealer, "READINESS", readiness)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])

    def failing_run(
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        check: bool,
    ) -> SimpleNamespace:
        assert cwd == repo and check is False and env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        Path(junit_arg.removeprefix("--junitxml=")).write_bytes(b"synthetic failure diagnostics\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(sealer.subprocess, "run", failing_run)
    monkeypatch.setattr(
        paid,
        "seal_paid_readiness",
        lambda root: seal_calls.append(Path(root)),
    )

    assert sealer.main() == 7
    captured = capsys.readouterr()
    assert "STOP_PAID_LOCAL_TESTS" in captured.err
    assert temporary.read_bytes() == b"synthetic failure diagnostics\n"
    assert not report.exists()
    assert not readiness.exists()
    assert seal_calls == []
