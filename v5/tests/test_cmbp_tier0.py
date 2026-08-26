"""Offline refusal, durability, and scope tests for Job 51 Tier-0 acquisition."""
from __future__ import annotations

import builtins
import copy
import json
import shutil
import socket
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from v5.research import cmbp_tier0 as tier0


REPO = Path(__file__).resolve().parents[2]
EXPECTED_DATES = (
    "2025-08-22",
    "2025-09-10",
    "2025-09-29",
    "2025-10-15",
    "2025-11-03",
    "2025-11-19",
    "2025-12-09",
    "2025-12-30",
    "2026-01-16",
    "2026-02-04",
    "2026-02-20",
    "2026-03-10",
    "2026-03-27",
    "2026-04-15",
    "2026-05-01",
    "2026-05-19",
    "2026-06-05",
    "2026-06-25",
    "2026-07-02",
    "2026-07-14",
    "2026-07-30",
)
ATTEMPT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental real network path fail at its first socket operation."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Job-51 focused tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tier0.canonical_json_bytes(value) + b"\n")


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


def _journal(tmp_path: Path) -> tuple[tier0.AttemptJournal, Path]:
    attempt_dir = tmp_path / ATTEMPT_ID
    attempt_dir.mkdir(parents=True)
    journal = tier0.AttemptJournal(
        attempt_dir,
        attempt_id=ATTEMPT_ID,
        volume=_volume(tmp_path),
    )
    return journal, attempt_dir


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


class _FakeClient:
    def __init__(
        self,
        quote: Any,
        *,
        actions: list[str] | None = None,
        timeseries_error: BaseException | None = None,
    ) -> None:
        self.quote = quote
        self.actions = actions if actions is not None else []
        self.cost_kwargs: list[dict[str, Any]] = []
        self.timeseries_kwargs: list[dict[str, Any]] = []
        self.timeseries_error = timeseries_error
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
        if self.timeseries_error is not None:
            raise self.timeseries_error
        Path(kwargs["path"]).write_bytes(b"synthetic-dbn-stream")


class _FakeDBNStore:
    """Small iterable carrying only the pinned header surface used by Job 51."""

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
            try:
                request = requests[session]
            except KeyError as exc:
                raise AssertionError(f"unexpected synthetic DBN path: {path}") from exc
            return _FakeDBNStore(request)

    monkeypatch.setattr(databento, "DBNStore", Factory)


def _readiness() -> dict[str, Any]:
    return {
        "receipt_sha256": "1" * 64,
        "program_contract_sha256": tier0.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": tier0.EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
    }


READINESS_FILE_SHA256 = "2" * 64


def _decoder_summary(request: tier0.SessionRequest) -> dict[str, Any]:
    mappings = [
        {"instrument_id": instrument_id, "raw_symbol": symbol}
        for instrument_id, symbol in request.expected_mappings
    ]
    zero_flags = [
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
        "flag_counts": zero_flags,
        "flag_value_counts": [{"name": "0", "count": request.expected_record_count}],
        "action_counts": [{"name": "A", "count": request.expected_record_count}],
        "side_counts": [{"name": "B", "count": request.expected_record_count}],
        "classification_reconciled": True,
    }


def _append_zero_gated_result(
    journal: tier0.AttemptJournal,
    request: tier0.SessionRequest,
    data_path: Path,
) -> dict[str, Any]:
    request_sha = request.market_request_sha256
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    journal.append(
        "COST_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={"observed_sdk_quote_usd": "0", "zero_gate_pass": True},
    )
    journal.append(
        "TIMESERIES_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={
            "method": "timeseries.get_range",
            "parameters": {
                **request.market_parameters,
                "stype_out": tier0.EXPECTED_STYPE_OUT,
                "limit": None,
            },
            "output_relative": "data.cmbp-1.dbn.zst",
            "immediately_preceding_quote_usd": "0",
        },
    )
    return journal.append(
        "TIMESERIES_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={
            "observed_sdk_quote_usd": "0",
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "compressed_bytes": data_path.stat().st_size,
            "dbn_file_sha256": tier0.file_sha256(data_path),
        },
    )


def _write_session_bundle(
    directory: Path,
    request: tier0.SessionRequest,
    *,
    attempt_id: str,
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "data.cmbp-1.dbn.zst"
    data_path.write_bytes(f"synthetic-dbn-{request.session}\n".encode())
    qc = tier0.build_session_qc(
        request=request,
        data_path=data_path,
        quote_usd="0",
        metadata_summary=tier0.validate_dbn_metadata(_FakeDBNStore(request), request),
        decoder_summary=_decoder_summary(request),
        readiness=_readiness(),
        readiness_file_sha256=READINESS_FILE_SHA256,
        attempt_id=attempt_id,
        journal_record=source_record,
    )
    _write_json(directory / "SESSION_QC_V1.json", qc)
    return qc


def _stage_frozen_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for name, source in tier0._repo_paths(REPO).items():
        if name in {"readiness", "test_report"}:
            continue
        target = root / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return root


def _write_junit(
    path: Path,
    *,
    tests: int = 12,
    failures: int = 0,
    errors: int = 0,
    skipped: int = 0,
    classname: str = "v5.tests.test_cmbp_tier0",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "job51-focused",
            "tests": str(tests),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
        },
    )
    for index in range(tests):
        ET.SubElement(
            suite,
            "testcase",
            {"classname": classname, "name": f"test_synthetic_{index:02d}"},
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _stage_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    required_bound_files: tuple[str, ...] = ("v5/research/synthetic_bound.py",),
) -> tuple[Path, Path, Path, tier0.ScopeBundle]:
    root = tmp_path / "repo"
    report = root / "v5/work/cmbp-tier0-acquisition/TEST_RESULTS_V1.xml"
    _write_junit(report)
    synthetic_population = tier0._junit_counts(report, enforce_frozen_population=False)
    monkeypatch.setattr(tier0, "EXPECTED_FOCUSED_TEST_COUNT", synthetic_population["tests"])
    monkeypatch.setattr(
        tier0,
        "EXPECTED_FOCUSED_TEST_IDENTITY_SHA256",
        synthetic_population["test_case_identity_sha256"],
    )
    for relative in required_bound_files:
        if relative.startswith("/") or ".." in Path(relative).parts:
            continue
        bound_path = root / relative
        bound_path.parent.mkdir(parents=True, exist_ok=True)
        bound_path.write_text("synthetic readiness input\n", encoding="utf-8")
    contract = {
        "contract_sha256": "d" * 64,
        "required_bound_files": list(required_bound_files),
    }
    bundle = tier0.ScopeBundle(
        contract=contract,
        scope={},
        job50_receipt={},
        job50_response={},
        job49_declaration={},
        sessions=(),
        contract_file_sha256="e" * 64,
        plan_file_sha256="f" * 64,
    )
    monkeypatch.setattr(tier0, "load_scope_bundle", lambda _root: bundle)
    monkeypatch.setattr(tier0, "sdk_identity", lambda: {"synthetic_sdk_identity": True})
    readiness = root / "v5/work/cmbp-tier0-acquisition/LOCAL_READINESS_RECEIPT_V1.json"
    return root, report, readiness, bundle


@pytest.fixture(scope="module")
def aggregate_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Construct a complete tiny-byte tree with the real frozen 21-session accounting."""

    bundle = tier0.load_scope_bundle(REPO)
    job_root = tmp_path_factory.mktemp("job51-aggregate-template") / "job51"
    attempts_root = job_root / "attempts"
    sessions_root = job_root / "sessions"
    attempts_root.mkdir(parents=True)
    sessions_root.mkdir()
    attempt_dir = attempts_root / ATTEMPT_ID
    attempt_dir.mkdir()
    (attempt_dir / "sessions").mkdir()
    journal = tier0.AttemptJournal(
        attempt_dir,
        attempt_id=ATTEMPT_ID,
        volume=_volume(job_root),
    )
    journal.append("ATTEMPT_START", payload={"scope_sha256": tier0.EXPECTED_SCOPE_SHA256})
    for ordinal, request in enumerate(bundle.sessions, start=1):
        final_dir = sessions_root / request.session
        final_dir.mkdir()
        data_path = final_dir / "data.cmbp-1.dbn.zst"
        data_path.write_bytes(f"synthetic-dbn-{request.session}\n".encode())
        source = _append_zero_gated_result(journal, request, data_path)
        qc = _write_session_bundle(
            final_dir,
            request,
            attempt_id=ATTEMPT_ID,
            source_record=source,
        )
        journal.append(
            "SESSION_PUBLISHED",
            session=request.session,
            request_sha256=request.market_request_sha256,
            payload={"ordinal": ordinal, "session_qc_sha256": qc["session_qc_sha256"]},
        )
    journal.append(
        "ATTEMPT_SEALED_FOR_AGGREGATE",
        payload={"published_or_reused_sessions": 21, "decoded_records": tier0.EXPECTED_RECORD_COUNT},
    )
    return job_root


def _copy_aggregate_template(template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "job51"
    shutil.copytree(template, target)
    return target


class _NoopRunLock:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoopRunLock":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _patch_runner_local_preflight(
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    *,
    job_root: Path,
    bundle: tier0.ScopeBundle,
) -> tier0.VolumeIdentity:
    volume = _volume(job_root)
    readiness_path = job_root.parent / f"{job_root.name}-synthetic-readiness.json"
    readiness_path.write_text("local-only\n", encoding="utf-8")
    real_file_sha256 = tier0.file_sha256

    def staged_file_sha256(path: Path) -> str:
        if Path(path) == readiness_path:
            return READINESS_FILE_SHA256
        return real_file_sha256(path)

    monkeypatch.setattr(sys, "argv", [str(runner.__file__)])
    monkeypatch.setattr(runner, "READINESS_PATH", readiness_path)
    monkeypatch.setattr(tier0, "load_scope_bundle", lambda _root: bundle)
    monkeypatch.setattr(tier0, "validate_readiness_receipt", lambda *_args, **_kwargs: _readiness())
    monkeypatch.setattr(tier0, "file_sha256", staged_file_sha256)
    monkeypatch.setattr(tier0, "inspect_destination_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(tier0, "initialize_destination_tree", lambda _volume_value: job_root)
    monkeypatch.setattr(tier0, "RunLock", _NoopRunLock)
    return volume


def test_frozen_scope_reconstructs_exact_21_sessions_and_totals() -> None:
    bundle = tier0.load_scope_bundle(REPO)
    assert tuple(request.session for request in bundle.sessions) == EXPECTED_DATES
    assert len(bundle.sessions) == tier0.EXPECTED_SESSION_COUNT == 21
    assert sum(request.expected_record_count for request in bundle.sessions) == 2_373_877_845
    assert sum(len(request.symbols) for request in bundle.sessions) == 1_001
    assert all(request.expected_cost_usd == "0" for request in bundle.sessions)
    assert all(request.market_parameters["dataset"] == "OPRA.PILLAR" for request in bundle.sessions)
    assert all(request.market_parameters["schema"] == "cmbp-1" for request in bundle.sessions)
    assert all(request.market_parameters["stype_in"] == "raw_symbol" for request in bundle.sessions)
    early_close = next(request for request in bundle.sessions if request.session == "2026-07-02")
    assert early_close.start == "2026-07-02T13:30:00Z"
    assert early_close.end == "2026-07-02T17:00:00Z"


def test_outcome_blind_spacing_reconstructs_from_all_228_zero_cost_sessions() -> None:
    bundle = tier0.load_scope_bundle(REPO)
    response_sessions = bundle.job50_response["sessions"]
    zero_dates = sorted(
        str(item["session"])
        for item in response_sessions
        if item.get("status") == "OK" and item.get("cost_usd") == "0"
    )
    assert len(zero_dates) == 228
    spaced = [zero_dates[round(index * 227 / 19)] for index in range(20)]
    if "2026-07-02" not in spaced:
        spaced.append("2026-07-02")
    assert tuple(sorted(spaced)) == EXPECTED_DATES
    assert tuple(tier0._selection_dates(response_sessions)) == EXPECTED_DATES


def test_selection_refuses_a_changed_zero_cost_population() -> None:
    bundle = tier0.load_scope_bundle(REPO)
    sessions = copy.deepcopy(bundle.job50_response["sessions"])
    first_zero = next(item for item in sessions if item.get("cost_usd") == "0")
    first_zero["cost_usd"] = "0.01"
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0._selection_dates(sessions)
    assert raised.value.status == "STOP_SCOPE_OR_SEAL_DRIFT"


def test_scope_destination_widening_is_refused_even_when_self_rehashed(tmp_path: Path) -> None:
    root = _stage_frozen_inputs(tmp_path)
    scope_path = tier0._repo_paths(root)["scope"]
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["destination_root"] = str(tmp_path / "not-the-authorized-volume")
    scope["scope_sha256"] = tier0.self_hash(scope, "scope_sha256")
    _write_json(scope_path, scope)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.load_scope_bundle(root)
    assert raised.value.status == "STOP_SCOPE_OR_SEAL_DRIFT"


def test_program_contract_cannot_be_rehashed_to_allow_scope_widening(tmp_path: Path) -> None:
    root = _stage_frozen_inputs(tmp_path)
    contract_path = tier0._repo_paths(root)["contract"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["call_law"]["scope_widening"] = True
    contract["contract_sha256"] = tier0.self_hash(contract, "contract_sha256")
    _write_json(contract_path, contract)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.load_scope_bundle(root)
    assert raised.value.status == "STOP_SCOPE_OR_SEAL_DRIFT"


@pytest.mark.parametrize(
    "value",
    [0, 0.0, Decimal("0"), Decimal("0E-18")],
    ids=["integer", "float", "decimal", "decimal-exponent"],
)
def test_exact_zero_quote_accepts_only_unsigned_numeric_zero(value: Any) -> None:
    assert tier0.exact_zero_quote(value) == "0"


@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        "0",
        b"0",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        -0.0,
        Decimal("-0"),
        1,
        0.000001,
        Decimal("0.000001"),
        -1,
    ],
    ids=[
        "false",
        "true",
        "string",
        "bytes",
        "none",
        "float-nan",
        "float-inf",
        "float-minus-inf",
        "decimal-nan",
        "decimal-inf",
        "negative-float-zero",
        "negative-decimal-zero",
        "positive-int",
        "positive-float",
        "positive-decimal",
        "negative-int",
    ],
)
def test_exact_zero_quote_rejects_malformed_nonfinite_signed_or_nonzero(value: Any) -> None:
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.exact_zero_quote(value)
    assert raised.value.status == "STOP_NONZERO_OR_INVALID_COST"


def test_acquisition_calls_gate_then_cost_then_identical_timeseries_request(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path / "journal")
    request = _request()
    actions: list[str] = []
    client = _FakeClient(Decimal("0"), actions=actions)
    output = attempt_dir / "session.dbn.zst"

    quote = tier0.acquire_session_bytes(
        client,
        request,
        output_path=output,
        journal=journal,
        pre_pair_gate=lambda: actions.append("pre_pair_gate"),
    )

    assert quote == "0"
    assert actions == ["pre_pair_gate", "metadata.get_cost", "timeseries.get_range"]
    assert client.cost_kwargs == [request.market_parameters]
    timeseries = client.timeseries_kwargs[0]
    assert {key: timeseries[key] for key in request.market_parameters} == request.market_parameters
    assert timeseries["stype_out"] == "instrument_id"
    assert timeseries["limit"] is None
    assert timeseries["path"] == output
    assert output.read_bytes() == b"synthetic-dbn-stream"

    verified = tier0.verify_attempt_journal(attempt_dir)
    assert [record["event"] for record in verified["records"]] == [
        "COST_CALL_START",
        "COST_CALL_RESULT",
        "TIMESERIES_CALL_START",
        "TIMESERIES_CALL_RESULT",
    ]
    assert {record["request_sha256"] for record in verified["records"]} == {
        request.market_request_sha256
    }
    assert verified["marker_count"] == 4


def test_positive_quote_stops_before_timeseries_and_records_failed_gate(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path / "journal")
    actions: list[str] = []
    client = _FakeClient(Decimal("0.01"), actions=actions)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.acquire_session_bytes(
            client,
            _request(),
            output_path=attempt_dir / "must-not-exist.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: actions.append("pre_pair_gate"),
        )
    assert raised.value.status == "STOP_NONZERO_OR_INVALID_COST"
    assert actions == ["pre_pair_gate", "metadata.get_cost"]
    verified = tier0.verify_attempt_journal(attempt_dir)
    assert [record["event"] for record in verified["records"]] == [
        "COST_CALL_START",
        "COST_CALL_RESULT",
    ]
    assert verified["records"][-1]["payload"]["zero_gate_pass"] is False
    assert verified["marker_count"] == 2


def test_cost_transport_error_stops_before_timeseries_and_redacts_message(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path / "journal")
    actions: list[str] = []
    client = _FakeClient(RuntimeError("secret-bearing vendor message"), actions=actions)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.acquire_session_bytes(
            client,
            _request(),
            output_path=attempt_dir / "must-not-exist.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: actions.append("pre_pair_gate"),
        )
    assert raised.value.status == "STOP_VENDOR_COST_CALL"
    assert actions == ["pre_pair_gate", "metadata.get_cost"]
    verified = tier0.verify_attempt_journal(attempt_dir)
    assert [record["event"] for record in verified["records"]] == [
        "COST_CALL_START",
        "COST_CALL_ERROR",
    ]
    assert verified["records"][-1]["payload"] == {"error_class": "RuntimeError"}
    assert "secret-bearing" not in (attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl").read_text()


def test_existing_output_path_stops_before_gate_or_vendor_call(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path / "journal")
    output = attempt_dir / "already-there.dbn.zst"
    output.write_bytes(b"preexisting")
    actions: list[str] = []
    client = _FakeClient(0, actions=actions)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.acquire_session_bytes(
            client,
            _request(),
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: actions.append("pre_pair_gate"),
        )
    assert raised.value.status == "STOP_EXTERNAL_PATH"
    assert actions == []
    assert output.read_bytes() == b"preexisting"


def test_timeseries_failure_is_journaled_without_automatic_retry(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path / "journal")
    actions: list[str] = []
    client = _FakeClient(0, actions=actions, timeseries_error=OSError("synthetic failure"))
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.acquire_session_bytes(
            client,
            _request(),
            output_path=attempt_dir / "failed.dbn.zst",
            journal=journal,
            pre_pair_gate=lambda: actions.append("pre_pair_gate"),
        )
    assert raised.value.status == "STOP_VENDOR_TIMESERIES_CALL"
    assert actions == ["pre_pair_gate", "metadata.get_cost", "timeseries.get_range"]
    assert len(client.cost_kwargs) == len(client.timeseries_kwargs) == 1
    verified = tier0.verify_attempt_journal(attempt_dir)
    assert [record["event"] for record in verified["records"]][-1] == "TIMESERIES_CALL_ERROR"


def test_journal_watermark_advances_and_call_markers_bind_every_call_event(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path)
    first = journal.append("ATTEMPT_START", payload={"scope_sha256": tier0.EXPECTED_SCOPE_SHA256})
    first_watermark = tier0.strict_json(attempt_dir / "JOURNAL_WATERMARK_V1.json")
    second = journal.append(
        "COST_CALL_START",
        session="2026-07-02",
        request_sha256="a" * 64,
        payload={"method": "metadata.get_cost"},
    )
    second_watermark = tier0.strict_json(attempt_dir / "JOURNAL_WATERMARK_V1.json")

    assert first_watermark["terminal_sequence"] == 0
    assert first_watermark["terminal_head"] == first["record_hash"]
    assert second_watermark["terminal_sequence"] == 1
    assert second_watermark["terminal_head"] == second["record_hash"]
    assert second_watermark["journal_file_sha256"] != first_watermark["journal_file_sha256"]
    verified = tier0.verify_attempt_journal(attempt_dir)
    assert verified["marker_count"] == 1
    assert list(verified["marker_files"]) == ["000001-cost_call_start.json"]


def test_valid_prefix_journal_truncation_is_detected_by_retained_watermark(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path)
    journal.append("ATTEMPT_START")
    journal.append("SESSION_START", session="2026-07-02")
    journal.append("SESSION_STOP", session="2026-07-02")
    path = attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl"
    lines = path.read_bytes().splitlines()
    path.write_bytes(b"\n".join(lines[:-1]) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.verify_attempt_journal(attempt_dir)
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_missing_exclusive_call_marker_is_detected(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path)
    journal.append("ATTEMPT_START")
    journal.append(
        "COST_CALL_START",
        session="2026-07-02",
        request_sha256="a" * 64,
    )
    marker = next((attempt_dir / "call-markers").glob("*.json"))
    marker.unlink()
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.verify_attempt_journal(attempt_dir)
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_marker_directory_refuses_unexpected_filename_or_file_type(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path)
    journal.append("ATTEMPT_START")
    (attempt_dir / "call-markers" / ".DS_Store").write_bytes(b"unexpected")
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.verify_attempt_journal(attempt_dir)
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_watermark_identity_cannot_be_rehashed_for_another_attempt(tmp_path: Path) -> None:
    journal, attempt_dir = _journal(tmp_path)
    journal.append("ATTEMPT_START")
    watermark_path = attempt_dir / "JOURNAL_WATERMARK_V1.json"
    watermark = tier0.strict_json(watermark_path)
    watermark["attempt_id"] = str(uuid.uuid4())
    watermark["watermark_sha256"] = tier0.self_hash(watermark, "watermark_sha256")
    _write_json(watermark_path, watermark)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.verify_attempt_journal(attempt_dir)
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_run_lock_is_nonblocking_bound_and_reacquirable_after_close(tmp_path: Path) -> None:
    volume = _volume(tmp_path)
    scope_sha = "a" * 64
    readiness_sha = "b" * 64
    first = tier0.RunLock(
        tmp_path,
        volume=volume,
        scope_sha256=scope_sha,
        readiness_sha256=readiness_sha,
    )
    try:
        binding = json.loads((tmp_path / "RUN_LOCK_V1").read_text(encoding="utf-8"))
        assert binding == {
            "artifact_type": "JOB51_RUN_LOCK_V1",
            "scope_sha256": scope_sha,
            "readiness_sha256": readiness_sha,
        }
        with pytest.raises(tier0.Tier0Error) as raised:
            tier0.RunLock(
                tmp_path,
                volume=volume,
                scope_sha256=scope_sha,
                readiness_sha256=readiness_sha,
            )
        assert raised.value.status == "STOP_CONCURRENT_RUNNER"
    finally:
        first.close()

    with tier0.RunLock(
        tmp_path,
        volume=volume,
        scope_sha256=scope_sha,
        readiness_sha256=readiness_sha,
    ):
        pass


def test_noncanonical_destination_path_stops_before_diskutil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_diskutil(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("diskutil must not run for a widened path")

    monkeypatch.setattr(tier0.subprocess, "run", forbidden_diskutil)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.inspect_destination_volume(tmp_path)
    assert raised.value.status == "STOP_EXTERNAL_VOLUME"


def test_readiness_junit_helper_accepts_only_clean_focused_job51_report(tmp_path: Path) -> None:
    clean = tmp_path / "clean.xml"
    _write_junit(clean)
    counts = tier0._junit_counts(clean, enforce_frozen_population=False)
    assert counts["tests"] == 12
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0

    failed = tmp_path / "failed.xml"
    _write_junit(failed, failures=1)
    with pytest.raises(tier0.Tier0Error) as raised_failure:
        tier0._junit_counts(failed)
    assert raised_failure.value.status == "STOP_LOCAL_TESTS"

    foreign = tmp_path / "foreign.xml"
    _write_junit(foreign, classname="v5.tests.test_unrelated_job")
    with pytest.raises(tier0.Tier0Error) as raised_foreign:
        tier0._junit_counts(foreign)
    assert raised_foreign.value.status == "STOP_LOCAL_TESTS"


def test_readiness_build_and_validation_bind_files_report_sdk_and_local_only_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, report, readiness_path, _bundle = _stage_readiness(tmp_path, monkeypatch)
    receipt = tier0.build_readiness_receipt(root, test_report_path=report)
    _write_json(readiness_path, receipt)
    assert tier0.validate_readiness_receipt(root, readiness_path) == receipt
    assert receipt["integrity"]["credential_read"] is False
    assert receipt["integrity"]["external_calls"] == 0
    assert receipt["integrity"]["timeseries_calls"] == 0
    assert receipt["sdk_identity"] == {"synthetic_sdk_identity": True}

    tampered = copy.deepcopy(receipt)
    tampered["integrity"]["external_calls"] = 1
    tampered["receipt_sha256"] = tier0.self_hash(tampered, "receipt_sha256")
    _write_json(readiness_path, tampered)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.validate_readiness_receipt(root, readiness_path)
    assert raised.value.status == "STOP_READINESS_SEAL"


def test_readiness_validation_refuses_bound_file_drift_and_noncanonical_receipt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, report, readiness_path, bundle = _stage_readiness(tmp_path, monkeypatch)
    receipt = tier0.build_readiness_receipt(root, test_report_path=report)
    _write_json(readiness_path, receipt)

    alternate = tmp_path / "copied-readiness.json"
    _write_json(alternate, receipt)
    with pytest.raises(tier0.Tier0Error) as raised_path:
        tier0.validate_readiness_receipt(root, alternate)
    assert raised_path.value.status == "STOP_READINESS_SEAL"

    bound_path = root / bundle.contract["required_bound_files"][0]
    bound_path.write_text("drifted after seal\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error) as raised_drift:
        tier0.validate_readiness_receipt(root, readiness_path)
    assert raised_drift.value.status == "STOP_READINESS_SEAL"


@pytest.mark.parametrize("relative", ["../outside.py", "/tmp/outside.py"])
def test_readiness_build_refuses_contract_path_widening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root, report, _readiness_path, _bundle = _stage_readiness(
        tmp_path,
        monkeypatch,
        required_bound_files=(relative,),
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_readiness_receipt(root, test_report_path=report)
    assert raised.value.status == "STOP_SCOPE_OR_SEAL_DRIFT"


def test_dbn_metadata_accepts_exact_header_and_refuses_partial_or_mapping_drift() -> None:
    request = _request()
    store = _FakeDBNStore(request)
    summary = tier0.validate_dbn_metadata(store, request)
    assert summary["dataset"] == "OPRA.PILLAR"
    assert summary["schema"] == "cmbp-1"
    assert summary["symbols"] == list(request.symbols)
    assert summary["mapping_sha256"] == tier0.json_sha256(summary["mappings"])

    partial = _FakeDBNStore(request)
    partial.metadata.partial = [request.symbols[0]]
    with pytest.raises(tier0.Tier0Error) as raised_partial:
        tier0.validate_dbn_metadata(partial, request)
    assert raised_partial.value.status == "STOP_DBN_METADATA"

    remapped = _FakeDBNStore(request)
    remapped.mappings[request.symbols[0]][0]["symbol"] = "99999"
    with pytest.raises(tier0.Tier0Error) as raised_mapping:
        tier0.validate_dbn_metadata(remapped, request)
    assert raised_mapping.value.status == "STOP_DBN_METADATA"

    unsupported = _FakeDBNStore(request)
    unsupported.metadata.version = 4
    with pytest.raises(tier0.Tier0Error) as raised_version:
        tier0.validate_dbn_metadata(unsupported, request)
    assert raised_version.value.status == "STOP_DBN_METADATA"

    timestamp_appended = _FakeDBNStore(request)
    timestamp_appended.metadata.ts_out = True
    with pytest.raises(tier0.Tier0Error) as raised_ts_out:
        tier0.validate_dbn_metadata(timestamp_appended, request)
    assert raised_ts_out.value.status == "STOP_DBN_METADATA"


def test_stream_dbn_qc_consumes_fake_store_incrementally_and_reconciles_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v5.research.cmbp_stream import CmbpRecord

    request = _request()
    start = tier0._timestamp_ns(request.start)
    instrument_id = request.expected_mappings[0][0]
    records = [
        CmbpRecord(instrument_id, start + 1, start + 1, "A", "B", 100, 1, bid_px=100, ask_px=110),
        CmbpRecord(instrument_id, start + 2, start + 2, "T", "N", 110, 1, bid_px=100, ask_px=110),
        CmbpRecord(instrument_id, start + 3, start + 3, "A", "B", 101, 1, bid_px=101, ask_px=111),
        CmbpRecord(instrument_id, start + 3, start + 3, "T", "N", 111, 1, bid_px=101, ask_px=111),
        CmbpRecord(instrument_id, start + 4, start + 4, "A", "B", 102, 1, bid_px=102, ask_px=112),
        CmbpRecord(instrument_id, start + 5, start + 5, "T", "N", 112, 1, bid_px=102, ask_px=112),
        CmbpRecord(instrument_id, start + 6, start + 6, "A", "B", 103, 1, bid_px=103, ask_px=113),
    ]
    store = _FakeDBNStore(request, records)
    import databento

    monkeypatch.setattr(
        databento,
        "DBNStore",
        SimpleNamespace(from_file=lambda _path: store),
    )
    path = tmp_path / "tiny.dbn.zst"
    path.write_bytes(b"synthetic")
    progress: list[int] = []
    metadata, decoder = tier0.stream_dbn_qc(
        path,
        request,
        progress=progress.append,
        progress_every=2,
    )
    assert metadata == tier0.validate_dbn_metadata(store, request)
    assert progress == [2, 4, 6]
    assert decoder["cmbp1_records"] == decoder["expected_cmbp1_records"] == 7
    assert decoder["record_count_reconciled"] is True
    assert decoder["trade_records"] == 3
    assert decoder["strict_prior_trades"] == 2
    assert decoder["tied_prior_trades_excluded"] == 1
    assert decoder["all_causal_priors_strict"] is True
    assert decoder["tied_receive_priors_excluded"] is True


def test_staged_part_validates_publishes_as_one_directory_and_is_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    job_root = tmp_path / "job51"
    staging_parent = job_root / "attempts" / ATTEMPT_ID / "sessions"
    sessions_root = job_root / "sessions"
    staging_parent.mkdir(parents=True)
    sessions_root.mkdir(parents=True)
    staging = staging_parent / f"{request.session}.bundle.part"
    source = {
        "sequence": 4,
        "record_hash": "a" * 64,
        "event": "TIMESERIES_CALL_RESULT",
    }
    qc = _write_session_bundle(
        staging,
        request,
        attempt_id=ATTEMPT_ID,
        source_record=source,
    )
    _patch_dbn_store(monkeypatch, {request.session: request})
    volume = _volume(job_root)

    unexpected = staging / "not-part-of-bundle.txt"
    unexpected.write_text("must refuse\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error) as raised_extra:
        tier0.validate_session_bundle(
            staging,
            request=request,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=volume,
        )
    assert raised_extra.value.status == "STOP_SESSION_BUNDLE"
    unexpected.unlink()

    validated = tier0.validate_session_bundle(
        staging,
        request=request,
        readiness=_readiness(),
        readiness_file_sha256=READINESS_FILE_SHA256,
        volume=volume,
    )
    assert validated == qc
    staging_inode = staging.stat().st_ino
    final = sessions_root / request.session
    tier0.publish_session_bundle(staging, final, volume=volume)
    assert not staging.exists()
    assert final.is_dir()
    assert final.stat().st_ino == staging_inode
    assert tier0.validate_session_bundle(
        final,
        request=request,
        readiness=_readiness(),
        readiness_file_sha256=READINESS_FILE_SHA256,
        volume=volume,
    ) == qc

    replacement_part = staging_parent / f"{request.session}.second.bundle.part"
    replacement_part.mkdir()
    with pytest.raises(tier0.Tier0Error) as raised_existing:
        tier0.publish_session_bundle(replacement_part, final, volume=volume)
    assert raised_existing.value.status == "STOP_SESSION_BUNDLE"
    assert replacement_part.is_dir()


def test_aggregate_reconstructs_all_21_tiny_bound_bundles(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    receipt = tier0.build_aggregate_receipt(
        job_root,
        bundle=bundle,
        readiness=_readiness(),
        readiness_file_sha256=READINESS_FILE_SHA256,
        volume=_volume(job_root),
    )
    assert receipt["status"] == "JOB51_TIER0_ACQUISITION_AND_QC_PASS"
    assert receipt["completeness"]["published_sessions"] == 21
    assert receipt["completeness"]["decoded_records"] == 2_373_877_845
    assert receipt["completeness"]["mapped_session_symbol_memberships"] == 1_001
    assert receipt["attempt_accounting"]["cost_call_starts"] == 21
    assert receipt["attempt_accounting"]["timeseries_call_starts"] == 21
    assert sum(
        attempt["cost_starts"] for attempt in receipt["attempt_accounting"]["attempts"]
    ) == receipt["attempt_accounting"]["cost_call_starts"]
    assert sum(
        attempt["timeseries_starts"] for attempt in receipt["attempt_accounting"]["attempts"]
    ) == receipt["attempt_accounting"]["timeseries_call_starts"]
    assert sum(
        attempt["timeseries_results"] for attempt in receipt["attempt_accounting"]["attempts"]
    ) == receipt["attempt_accounting"]["timeseries_call_results"]


def test_existing_final_preflight_reconstructs_sources_and_refuses_source_drift(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    attempts = tier0.summarize_attempts(job_root, bundle.sessions)
    source_lookup = {
        (str(record["attempt_id"]), int(record["sequence"])): record
        for record in attempts["timeseries_result_records"]
    }
    validated = tier0.validate_existing_session_population(
        job_root,
        bundle=bundle,
        readiness=_readiness(),
        readiness_file_sha256=READINESS_FILE_SHA256,
        volume=_volume(job_root),
        source_record_lookup=source_lookup,
    )
    assert list(validated) == list(EXPECTED_DATES)

    request = bundle.sessions[0]
    qc_path = job_root / "sessions" / request.session / "SESSION_QC_V1.json"
    qc = tier0.strict_json(qc_path)
    qc["source_attempt"]["journal_record_hash"] = "0" * 64
    qc["session_qc_sha256"] = tier0.self_hash(qc, "session_qc_sha256")
    _write_json(qc_path, qc)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.validate_existing_session_population(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
            source_record_lookup=source_lookup,
        )
    assert raised.value.status == "STOP_SESSION_BUNDLE"


def test_aggregate_refuses_validly_journaled_out_of_scope_vendor_call(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    attempt_id = "22222222-2222-4222-8222-222222222222"
    attempt_dir = job_root / "attempts" / attempt_id
    attempt_dir.mkdir()
    (attempt_dir / "sessions").mkdir()
    journal = tier0.AttemptJournal(attempt_dir, attempt_id=attempt_id, volume=_volume(job_root))
    journal.append("ATTEMPT_START", payload={"scope_sha256": tier0.EXPECTED_SCOPE_SHA256})
    journal.append(
        "COST_CALL_START",
        session="2099-01-01",
        request_sha256="f" * 64,
        payload={"method": "metadata.get_cost", "parameters": {}},
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_aggregate_refuses_out_of_scope_part_bundle_in_attempt_staging(
    aggregate_template: Path,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    staging = job_root / "attempts" / ATTEMPT_ID / "sessions" / "2099-01-01.bundle.part"
    staging.mkdir()
    (staging / "data.cmbp-1.dbn.zst").write_bytes(b"out-of-scope")
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_aggregate_refuses_self_rehashed_final_with_widened_request_projection(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    request = bundle.sessions[0]
    qc_path = job_root / "sessions" / request.session / "SESSION_QC_V1.json"
    qc = tier0.strict_json(qc_path)
    qc["request"]["market_parameters"]["symbols"].append("OUTSIDE_FROZEN_SCOPE")
    qc["session_qc_sha256"] = tier0.self_hash(qc, "session_qc_sha256")
    _write_json(qc_path, qc)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status in {"STOP_SESSION_BUNDLE", "STOP_AGGREGATE_QC"}


def test_aggregate_refuses_session_qc_source_attempt_hash_mismatch(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    request = bundle.sessions[0]
    qc_path = job_root / "sessions" / request.session / "SESSION_QC_V1.json"
    qc = tier0.strict_json(qc_path)
    qc["source_attempt"]["journal_record_hash"] = "0" * 64
    qc["session_qc_sha256"] = tier0.self_hash(qc, "session_qc_sha256")
    _write_json(qc_path, qc)
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status == "STOP_SESSION_BUNDLE"


def test_aggregate_refuses_unexpected_attempts_tree_entry(
    aggregate_template: Path,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    (job_root / "attempts" / "operator-note.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status == "STOP_JOURNAL_INVALID"


def test_aggregate_refuses_unexpected_sessions_tree_entry(
    aggregate_template: Path,
    tmp_path: Path,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    (job_root / "sessions" / "2099-01-01").mkdir()
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status == "STOP_AGGREGATE_QC"


def test_reuse_only_post_completion_attempt_cannot_recreate_pass(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    shutil.rmtree(job_root / "attempts")
    attempts_root = job_root / "attempts"
    attempts_root.mkdir()
    attempt_dir = attempts_root / ATTEMPT_ID
    attempt_dir.mkdir()
    (attempt_dir / "sessions").mkdir()
    journal = tier0.AttemptJournal(attempt_dir, attempt_id=ATTEMPT_ID, volume=_volume(job_root))
    journal.append("ATTEMPT_START", payload={"scope_sha256": tier0.EXPECTED_SCOPE_SHA256})
    for ordinal, request in enumerate(bundle.sessions, start=1):
        journal.append(
            "SESSION_REUSED",
            session=request.session,
            request_sha256=request.market_request_sha256,
            payload={"ordinal": ordinal},
        )
    journal.append("ATTEMPT_SEALED_FOR_AGGREGATE")
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    with pytest.raises(tier0.Tier0Error) as raised:
        tier0.build_aggregate_receipt(
            job_root,
            bundle=bundle,
            readiness=_readiness(),
            readiness_file_sha256=READINESS_FILE_SHA256,
            volume=_volume(job_root),
        )
    assert raised.value.status in {"STOP_SESSION_BUNDLE", "STOP_AGGREGATE_QC"}


def test_runner_main_happy_path_orchestrates_all_21_fake_sessions_to_aggregate_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import acquire_cmbp_tier0 as runner

    bundle = tier0.load_scope_bundle(REPO)
    job_root = tmp_path / "job51"
    for directory in (
        job_root / "attempts",
        job_root / "sessions",
        job_root / "receipts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _patch_dbn_store(monkeypatch, {request.session: request for request in bundle.sessions})
    _patch_runner_local_preflight(
        monkeypatch,
        runner,
        job_root=job_root,
        bundle=bundle,
    )

    actions: list[str] = []
    client = _FakeClient(Decimal("0"), actions=actions)
    constructed_with: list[str] = []
    streamed: list[tuple[str, Path]] = []

    def construct_client(api_key: str) -> _FakeClient:
        constructed_with.append(api_key)
        return client

    def synthetic_stream_qc(
        path: Path,
        request: tier0.SessionRequest,
        *,
        progress: Any = None,
        progress_every: int = 10_000_000,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert progress_every == 10_000_000
        assert path.is_file()
        assert path.name == "data.cmbp-1.dbn.zst"
        assert path.parent.name == f"{request.session}.bundle.part"
        streamed.append((request.session, path))
        if progress is not None:
            progress(request.expected_record_count)
        return (
            tier0.validate_dbn_metadata(_FakeDBNStore(request), request),
            _decoder_summary(request),
        )

    monkeypatch.setattr(runner, "_read_api_key_after_gates", lambda: ("synthetic-api-key", "test-only"))
    monkeypatch.setattr(runner, "_construct_exact_client", construct_client)
    monkeypatch.setattr(tier0, "sdk_identity", lambda: {"synthetic_sdk_identity": True})
    monkeypatch.setattr(tier0, "stream_dbn_qc", synthetic_stream_qc)

    assert runner.main() == 0
    assert constructed_with == ["synthetic-api-key"]
    assert actions == [event for _request_item in bundle.sessions for event in (
        "metadata.get_cost",
        "timeseries.get_range",
    )]
    assert len(client.cost_kwargs) == len(client.timeseries_kwargs) == 21
    for request, cost_kwargs, timeseries_kwargs in zip(
        bundle.sessions,
        client.cost_kwargs,
        client.timeseries_kwargs,
        strict=True,
    ):
        assert cost_kwargs == request.market_parameters
        assert {
            key: timeseries_kwargs[key] for key in request.market_parameters
        } == request.market_parameters
        assert timeseries_kwargs["stype_out"] == "instrument_id"
        assert timeseries_kwargs["limit"] is None
    assert [session for session, _path in streamed] == list(EXPECTED_DATES)

    published = sorted((job_root / "sessions").iterdir())
    assert [path.name for path in published] == list(EXPECTED_DATES)
    assert all(sorted(child.name for child in path.iterdir()) == [
        "SESSION_QC_V1.json",
        "data.cmbp-1.dbn.zst",
    ] for path in published)
    attempt_dirs = list((job_root / "attempts").iterdir())
    assert len(attempt_dirs) == 1
    assert list((attempt_dirs[0] / "sessions").iterdir()) == []
    verified = tier0.verify_attempt_journal(attempt_dirs[0])
    assert sum(record["event"] == "COST_CALL_START" for record in verified["records"]) == 21
    assert sum(record["event"] == "TIMESERIES_CALL_START" for record in verified["records"]) == 21

    receipt_path = job_root / "receipts" / "JOB51_ACQUISITION_QC_RECEIPT_V1.json"
    receipt = tier0.strict_json(receipt_path)
    assert receipt["status"] == "JOB51_TIER0_ACQUISITION_AND_QC_PASS"
    assert receipt["completeness"]["published_sessions"] == 21
    assert receipt["attempt_accounting"]["cost_call_starts"] == 21
    assert "COMPLETE" in capsys.readouterr().out


def test_runner_prior_nonzero_quote_stops_before_credential_client_or_new_attempt(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import acquire_cmbp_tier0 as runner

    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    _patch_runner_local_preflight(
        monkeypatch,
        runner,
        job_root=job_root,
        bundle=bundle,
    )
    reached: list[str] = []

    def should_not_reach(*_args: Any, **_kwargs: Any) -> None:
        reached.append("credential_client_or_attempt")
        raise AssertionError("prior paid quote guard ran too late")

    monkeypatch.setattr(
        tier0,
        "summarize_attempts",
        lambda *_args, **_kwargs: {"nonzero_or_invalid_quote_observed": True},
    )
    monkeypatch.setattr(tier0, "validate_existing_session_population", should_not_reach)
    monkeypatch.setattr(tier0, "create_attempt", should_not_reach)
    monkeypatch.setattr(runner, "_read_api_key_after_gates", should_not_reach)
    monkeypatch.setattr(runner, "_construct_exact_client", should_not_reach)
    assert runner.main() == 2
    assert "STOP_NONZERO_OR_INVALID_COST" in capsys.readouterr().err
    assert reached == []


def test_runner_existing_final_source_mismatch_stops_before_credential_or_new_attempt(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import acquire_cmbp_tier0 as runner

    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    request = bundle.sessions[0]
    qc_path = job_root / "sessions" / request.session / "SESSION_QC_V1.json"
    qc = tier0.strict_json(qc_path)
    qc["source_attempt"]["journal_record_hash"] = "0" * 64
    qc["session_qc_sha256"] = tier0.self_hash(qc, "session_qc_sha256")
    _write_json(qc_path, qc)
    _patch_dbn_store(monkeypatch, {item.session: item for item in bundle.sessions})
    _patch_runner_local_preflight(
        monkeypatch,
        runner,
        job_root=job_root,
        bundle=bundle,
    )
    reached: list[str] = []

    def should_not_reach(*_args: Any, **_kwargs: Any) -> None:
        reached.append("credential_client_or_attempt")
        raise AssertionError("existing-final source guard ran too late")

    monkeypatch.setattr(tier0, "create_attempt", should_not_reach)
    monkeypatch.setattr(runner, "_read_api_key_after_gates", should_not_reach)
    monkeypatch.setattr(runner, "_construct_exact_client", should_not_reach)
    assert runner.main() == 2
    assert "STOP_SESSION_BUNDLE" in capsys.readouterr().err
    assert reached == []


def test_runner_terminal_aggregate_receipt_stops_before_reuse_or_credential(
    aggregate_template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import acquire_cmbp_tier0 as runner

    bundle = tier0.load_scope_bundle(REPO)
    job_root = _copy_aggregate_template(aggregate_template, tmp_path)
    receipt = job_root / "receipts" / "JOB51_ACQUISITION_QC_RECEIPT_V1.json"
    receipt.parent.mkdir()
    receipt.write_text("terminal\n", encoding="utf-8")
    _patch_runner_local_preflight(
        monkeypatch,
        runner,
        job_root=job_root,
        bundle=bundle,
    )
    reached: list[str] = []

    def should_not_reach(*_args: Any, **_kwargs: Any) -> None:
        reached.append("reuse_or_credential")
        raise AssertionError("terminal receipt guard ran too late")

    monkeypatch.setattr(tier0, "summarize_attempts", should_not_reach)
    monkeypatch.setattr(tier0, "create_attempt", should_not_reach)
    monkeypatch.setattr(runner, "_read_api_key_after_gates", should_not_reach)
    assert runner.main() == 2
    assert "STOP_JOB51_ALREADY_TERMINAL" in capsys.readouterr().err
    assert reached == []


def test_runner_refuses_any_argv_before_scope_or_credential_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import acquire_cmbp_tier0 as runner

    reached: list[str] = []

    def should_not_reach(*_args: Any, **_kwargs: Any) -> None:
        reached.append("scope_or_credential")
        raise AssertionError("argv guard ran too late")

    monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--destination", "/tmp/widened"])
    monkeypatch.setattr(tier0, "load_scope_bundle", should_not_reach)
    monkeypatch.setattr(runner, "_read_api_key_after_gates", should_not_reach)
    assert runner.main() == 2
    captured = capsys.readouterr()
    assert "STOP_UNEXPECTED_ARGUMENTS" in captured.err
    assert reached == []


def test_readiness_sealer_runs_exact_tests_in_credential_free_plugin_isolated_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_readiness as sealer

    work = tmp_path / "repo" / "v5/work/cmbp-tier0-acquisition"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    captured_runs: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        check: bool,
    ) -> SimpleNamespace:
        captured_runs.append(
            {"command": list(command), "cwd": Path(cwd), "env": dict(env), "check": check}
        )
        junit_arg = next(item for item in command if item.startswith("--junitxml="))
        Path(junit_arg.split("=", 1)[1]).write_text("<testsuite tests='1'/>\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sys, "argv", [str(sealer.__file__)])
    monkeypatch.setattr(sealer, "REPO", tmp_path / "repo")
    monkeypatch.setattr(sealer, "WORK", work)
    monkeypatch.setattr(sealer, "REPORT", report)
    monkeypatch.setattr(sealer, "READINESS", readiness)
    monkeypatch.setattr(sealer.subprocess, "run", fake_run)
    monkeypatch.setattr(tier0, "fsync_directory", lambda _path: None)
    monkeypatch.setattr(tier0, "seal_readiness", lambda _root: readiness)
    monkeypatch.setenv("DATABENTO_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lf -k attacker_selected_subset")
    monkeypatch.setenv("PYTEST_PLUGINS", "attacker_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")

    assert sealer.main() == 0
    assert len(captured_runs) == 1
    launched = captured_runs[0]
    assert launched["cwd"] == tmp_path / "repo"
    assert launched["check"] is False
    assert launched["command"][:3] == [
        str(tmp_path / "repo/.venv/bin/python"),
        "-m",
        "pytest",
    ]
    assert launched["command"][3:5] == ["-c", "/dev/null"]
    assert f"--rootdir={tmp_path / 'repo'}" in launched["command"]
    assert ["-p", "no:cacheprovider"] == launched["command"][
        launched["command"].index("-p") : launched["command"].index("-p") + 2
    ]
    assert "-q" in launched["command"]
    assert all(launched["command"].count(test) == 1 for test in sealer.TESTS)
    assert launched["command"][-1].startswith("--junitxml=")
    assert "DATABENTO_API_KEY" not in launched["env"]
    assert "PYTEST_ADDOPTS" not in launched["env"]
    assert "PYTEST_PLUGINS" not in launched["env"]
    assert launched["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert report.is_file()
    assert str(readiness) in capsys.readouterr().out


def test_readiness_sealer_refuses_argv_before_importing_tier0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_readiness as sealer

    sentinel_report = tmp_path / "must-not-be-inspected.xml"
    sentinel_report.write_text("sentinel\n", encoding="utf-8")
    imported: list[str] = []
    real_import = builtins.__import__

    def tracked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "v5.research.cmbp_tier0":
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", [str(sealer.__file__), "--junitxml=/tmp/widened.xml"])
    monkeypatch.setattr(sealer, "REPORT", sentinel_report)
    monkeypatch.setattr(builtins, "__import__", tracked_import)
    assert sealer.main() == 2
    assert "STOP_UNEXPECTED_ARGUMENTS" in capsys.readouterr().err
    assert imported == []
