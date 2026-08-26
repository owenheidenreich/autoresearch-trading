"""Offline money-boundary tests for the Job-55 USD-2.00 cap amendment."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import socket
import sys
import uuid
import builtins
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from v5.research import cmbp_tier0 as tier0
from v5.research import cmbp_tier0_cap_amendment as amendment
from v5.research import cmbp_tier0_paid as paid


REPO = Path(__file__).resolve().parents[2]
JOB55_WORK = REPO / "v5/work/cmbp-tier0-cap-amendment"
JOB55_PLAN_FILE_SHA256 = "b59da7b859f1b6175ccfda111dee904ece2ee14aa514e7ffc26a9bcfa53c4aa7"
JOB55_CONTRACT_SHA256 = "60011bd6a2eba526f3690fe01326fb1074dcba14c00f0b72fc0ed2ff3d8ada78"
JOB55_CONTRACT_FILE_SHA256 = "11b7576dfa47f3e951cf5717ab26a4d53355bfb61d4305cb15786e24d4c2a1de"
ATTEMPT_ID = "66666666-6666-4666-8666-666666666666"
SECOND_ATTEMPT_ID = "77777777-7777-4777-8777-777777777777"
FAILED_SESSION = "2025-08-22"
OPENING_COMMITMENT = Decimal("0.950392448902")
SAME_PRICE_RETRY_TOTAL = Decimal("1.900784897804")
SESSION_HEADROOM = Decimal("1.049607551098")
JOB55_READINESS_SHA256 = "6" * 64
JOB55_READINESS_FILE_SHA256 = "7" * 64


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Focused Job-55 tests must never reach a real socket."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Job-55 focused tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture(autouse=True)
def _permit_only_this_test_synthetic_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use Job-55's private volume seam without weakening production APIs."""

    test_root = tmp_path.resolve()

    def validate(
        volume: tier0.VolumeIdentity,
        *,
        unpublished_records: int = tier0.EXPECTED_RECORD_COUNT,
    ) -> tier0.VolumeIdentity:
        del unpublished_records
        try:
            Path(volume.mount_point).resolve().relative_to(test_root)
        except ValueError as exc:
            raise AssertionError(
                "Job-55 tests may inspect only their own synthetic tmp volume"
            ) from exc
        return volume

    monkeypatch.setattr(amendment, "_assert_current_production_volume", validate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, tuple[str, int, str | None]]:
    """Return a byte-sensitive snapshot without following fixture symlinks."""

    snapshot: dict[str, tuple[str, int, str | None]] = {}
    if not root.exists():
        return snapshot
    for path in sorted((root, *root.rglob("*"))):
        relative = "." if path == root else str(path.relative_to(root))
        stat_result = path.lstat()
        if path.is_symlink():
            snapshot[relative] = ("symlink", stat_result.st_size, None)
        elif path.is_dir():
            snapshot[relative] = ("directory", stat_result.st_size, None)
        elif path.is_file():
            snapshot[relative] = ("file", stat_result.st_size, _sha256(path))
        else:
            snapshot[relative] = ("other", stat_result.st_size, None)
    return snapshot


def _volume(root: Path) -> tier0.VolumeIdentity:
    root.mkdir(parents=True, exist_ok=True)
    return tier0.VolumeIdentity(
        mount_point=str(root),
        volume_uuid=tier0.EXPECTED_VOLUME_UUID,
        device_identifier="synthetic-job55-device",
        filesystem=tier0.EXPECTED_FILESYSTEM,
        bus_protocol="Synthetic",
        st_dev=root.stat().st_dev,
        free_bytes=10**12,
        total_bytes=2 * 10**12,
    )


def _failed_request() -> tier0.SessionRequest:
    matches = [
        request
        for request in tier0.load_scope_bundle(REPO).sessions
        if request.session == FAILED_SESSION
    ]
    assert len(matches) == 1
    return matches[0]


def _opening_summary() -> dict[str, Any]:
    """Minimal validated-ledger surface accepted by AmendedBudgetState."""

    return {
        "artifact_type": "JOB55_JOB52_OPENING_EVIDENCE_V1",
        "opening_commitment_total_usd": format(OPENING_COMMITMENT, "f"),
        "opening_commitment_by_session_usd": {
            FAILED_SESSION: format(OPENING_COMMITMENT, "f")
        },
        "opening_commitment_count": 1,
    }


def _opening_state() -> amendment.AmendedBudgetState:
    return amendment.AmendedBudgetState.from_opening_evidence(_opening_summary())


def _next_process_state(state: amendment.AmendedBudgetState) -> amendment.AmendedBudgetState:
    """Mirror disk reconstruction while clearing only process-local pair counters."""

    return amendment.AmendedBudgetState(
        job55_committed_total_usd=state.job55_committed_total_usd,
        job55_committed_by_session_usd=dict(state.job55_committed_by_session_usd),
        job55_start_count_by_session=dict(state.job55_start_count_by_session),
        job55_commitment_count=state.job55_commitment_count,
        job55_cost_call_count_by_session=dict(state.job55_cost_call_count_by_session),
        published_sessions=set(state.published_sessions),
    )


def _journal(
    tmp_path: Path,
    *,
    attempt_id: str = ATTEMPT_ID,
) -> tuple[tier0.AttemptJournal, Path]:
    job55_root = tmp_path / "cmbp-tier0" / "job55"
    for directory in (
        job55_root / "attempts",
        job55_root / "sessions",
        job55_root / "receipts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    adoption_path = job55_root / amendment.JOB55_ADOPTION_NAME
    if not adoption_path.exists():
        adoption = {"artifact_type": "SYNTHETIC_JOB55_ADOPTION_TEST_ONLY"}
        adoption["adoption_sha256"] = tier0.self_hash(adoption, "adoption_sha256")
        adoption_path.write_bytes(tier0.canonical_json_bytes(adoption) + b"\n")
    adoption_sha256 = tier0.strict_json(adoption_path)["adoption_sha256"]

    prior: list[tuple[int, str, str]] = []
    for prior_dir in (job55_root / "attempts").iterdir():
        if not prior_dir.is_dir() or prior_dir.name == attempt_id:
            continue
        records = tier0.verify_attempt_journal(prior_dir)["records"]
        payload = records[0]["payload"]
        prior.append(
            (
                payload["job55_attempt_ordinal"],
                prior_dir.name,
                records[0]["record_hash"],
            )
        )
    previous = max(prior) if prior else None
    ordinal = 1 if previous is None else previous[0] + 1
    attempt_dir = job55_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    volume = _volume(tmp_path)
    journal = tier0.AttemptJournal(
        attempt_dir,
        attempt_id=attempt_id,
        volume=volume,
    )
    journal.append(
        "ATTEMPT_START",
        payload=amendment.build_job55_attempt_header_payload(
            attempt_id=attempt_id,
            volume=volume,
            readiness_receipt_sha256=JOB55_READINESS_SHA256,
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption_receipt_sha256=adoption_sha256,
            job55_attempt_ordinal=ordinal,
            previous_job55_attempt_id=None if previous is None else previous[1],
            previous_job55_attempt_header_record_hash=None if previous is None else previous[2],
        ),
    )
    (attempt_dir / "sessions").mkdir()
    amendment.validate_job55_attempt_anchor(
        job55_root,
        volume=volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption_receipt_sha256=adoption_sha256,
        allow_initialize=previous is None,
        repair_header_only=previous is not None,
    )
    journal._job55_test_volume = volume  # type: ignore[attr-defined]
    journal._job55_test_adoption_sha256 = adoption_sha256  # type: ignore[attr-defined]
    return journal, attempt_dir


def _staging_path(attempt_dir: Path, session: str) -> Path:
    bundle = attempt_dir / "sessions" / f"{session}.bundle.part"
    bundle.mkdir(parents=True)
    return bundle / "data.cmbp-1.dbn.zst"


def _request_ordinal(request: tier0.SessionRequest) -> int:
    return next(
        index
        for index, frozen in enumerate(
            amendment.load_cap_amendment_scope_bundle(REPO).sessions,
            start=1,
        )
        if frozen == request
    )


def _acquire_amended_session_bytes(
    client: Any,
    request: tier0.SessionRequest,
    *,
    output_path: Path,
    journal: tier0.AttemptJournal,
    pre_pair_gate: Any,
    budget_state: amendment.AmendedBudgetState,
    volume: tier0.VolumeIdentity | None = None,
    readiness_receipt_sha256: str = JOB55_READINESS_SHA256,
    readiness_receipt_file_sha256: str = JOB55_READINESS_FILE_SHA256,
    adoption_receipt_sha256: str | None = None,
    ordinal: int | None = None,
    progress: Any = None,
) -> str:
    mount = journal.attempt_dir.parents[3]
    actual_volume = (
        getattr(journal, "_job55_test_volume", _volume(mount))
        if volume is None
        else volume
    )
    actual_adoption = (
        getattr(journal, "_job55_test_adoption_sha256", None)
        if adoption_receipt_sha256 is None
        else adoption_receipt_sha256
    )
    frozen_bundle = amendment.load_cap_amendment_scope_bundle(REPO)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(amendment.base, "EXPECTED_VOLUME_ROOT", Path(actual_volume.mount_point))
        patcher.setattr(
            amendment.base,
            "inspect_destination_volume",
            lambda *_args, **_kwargs: actual_volume,
        )
        patcher.setattr(
            amendment,
            "load_cap_amendment_scope_bundle",
            lambda _repo: frozen_bundle,
        )
        patcher.setattr(
            amendment,
            "reconstruct_job55_local_state",
            lambda **_kwargs: {
                "readiness": {"receipt_sha256": readiness_receipt_sha256},
                "readiness_file_sha256": readiness_receipt_file_sha256,
                "adoption": {"adoption_sha256": str(actual_adoption)},
                "budget_state": budget_state,
                "final_qc": {},
            },
        )
        return amendment.acquire_amended_session_bytes(
            client,
            request,
            output_path=output_path,
            journal=journal,
            pre_pair_gate=pre_pair_gate,
            budget_state=budget_state,
            volume=actual_volume,
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            adoption_receipt_sha256=str(actual_adoption),
            ordinal=_request_ordinal(request) if ordinal is None else ordinal,
            progress=progress,
        )


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
        Path(kwargs["path"]).write_bytes(b"synthetic-job55-dbn-stream")


class _FakeDBNStore:
    """Minimal exact DBN metadata surface; iteration remains bounded and empty."""

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
            session = Path(path).parent.name.removesuffix(".bundle.part")
            if session not in requests:
                session = Path(path).parent.name
            if session not in requests:
                raise AssertionError(f"unexpected synthetic Job55 DBN path: {path}")
            return _FakeDBNStore(requests[session])

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
        "seen_instrument_ids": [
            instrument_id for instrument_id, _symbol in request.expected_mappings
        ],
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


def _synthetic_job52_summary(*, partial_bytes: int, partial_sha256: str) -> dict[str, Any]:
    evidence = amendment.load_cap_amendment_scope_bundle(REPO).contract[
        "opening_job52_evidence"
    ]
    start_hash = evidence["job52_attempt_1_time_series_start_record_hash"]
    return {
        "paid_attempt_count": 2,
        "cost_call_starts": 2,
        "cost_call_results": 2,
        "cost_call_errors": 0,
        "timeseries_call_starts": 1,
        "timeseries_call_results": 0,
        "failed_timeseries_starts": 1,
        "pending_timeseries_starts": 0,
        "committed_quote_total_usd": "0.950392448902",
        "committed_quote_by_session_usd": {FAILED_SESSION: "0.950392448902"},
        "terminal_authority_failure_observed": True,
        "successful_stream_recovery_required": False,
        "published_sessions_in_journals": [],
        "aggregate_seal_count": 0,
        "commitments": [
            {
                "commitment_index": 1,
                "attempt_id": evidence["job52_attempt_1_id"],
                "timeseries_start_record_hash": start_hash,
                "fresh_quote_usd": "0.950392448902",
            }
        ],
        "terminal_authority_failures": [
            {
                "attempt_id": evidence["old_cap_stop_superseded_only_for_attempt_id"],
                "cost_result_sequence": 3,
                "session": FAILED_SESSION,
                "observed_sdk_quote": "0.950392448902",
            }
        ],
        "quote_observations": [
            {
                "attempt_id": evidence["old_cap_stop_superseded_only_for_attempt_id"],
                "cost_result_sequence": 3,
                "cost_result_record_hash": evidence[
                    "old_cap_stop_superseded_only_for_cost_result_record_hash"
                ],
            }
        ],
        "partial_or_failed_staging": [
            {
                "compressed_bytes": partial_bytes,
                "file_sha256": partial_sha256,
                "timeseries_start_record_hash": start_hash,
            }
        ],
    }


def _synthetic_job52_tree(tmp_path: Path) -> SimpleNamespace:
    job51_root = tmp_path / "synthetic-volume" / "cmbp-tier0" / "job51"
    attempt_sessions = job51_root / "attempts" / "synthetic-job52-attempt" / "sessions"
    attempt_sessions.mkdir(parents=True)
    partial_path = attempt_sessions / "2025-08-22.bundle.part.dbn.zst"
    partial_path.write_bytes(b"synthetic immutable Job52 partial bytes")

    anchor = {
        "artifact_type": "SYNTHETIC_JOB52_ATTEMPT_SET_ANCHOR_V1",
        "attempt_ids": ["synthetic-job52-attempt"],
    }
    anchor["anchor_sha256"] = tier0.self_hash(anchor, "anchor_sha256")
    anchor_path = job51_root / paid.PAID_ATTEMPT_ANCHOR_NAME
    anchor_path.write_bytes(tier0.canonical_json_bytes(anchor) + b"\n")

    directories = tuple(
        sorted(
            (
                ".",
                "attempts",
                "attempts/synthetic-job52-attempt",
                "attempts/synthetic-job52-attempt/sessions",
            )
        )
    )
    files = {
        anchor_path.relative_to(job51_root).as_posix(): tier0.file_sha256(anchor_path),
        partial_path.relative_to(job51_root).as_posix(): tier0.file_sha256(partial_path),
    }
    pins = amendment.Job52EvidencePins(
        directories=directories,
        file_sha256=dict(sorted(files.items())),
        anchor_sha256=anchor["anchor_sha256"],
        failed_partial_bytes=partial_path.stat().st_size,
        failed_partial_sha256=tier0.file_sha256(partial_path),
    )
    return SimpleNamespace(
        job51_root=job51_root,
        volume=_volume(tmp_path / "synthetic-volume"),
        pins=pins,
        anchor_path=anchor_path,
        partial_path=partial_path,
        summary=_synthetic_job52_summary(
            partial_bytes=partial_path.stat().st_size,
            partial_sha256=tier0.file_sha256(partial_path),
        ),
    )


def _patch_job52_semantic_validation(
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    sdk_identity = {"synthetic_job52_sdk": "offline-only"}

    monkeypatch.setattr(
        paid,
        "validate_paid_readiness_receipt",
        lambda *_args, **_kwargs: {
            "receipt_sha256": amendment.EXPECTED_JOB52_READINESS_SHA256
        },
    )
    monkeypatch.setattr(paid, "paid_sdk_identity", lambda: copy.deepcopy(sdk_identity))

    def summarize(job_root: Path, sessions: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(
            {
                "job_root": Path(job_root),
                "sessions": tuple(sessions),
                "kwargs": copy.deepcopy(kwargs),
            }
        )
        return copy.deepcopy(summary)

    monkeypatch.setattr(paid, "summarize_paid_attempts", summarize)
    return calls


def _opening_evidence_for_controls(job51_root: Path) -> amendment.Job52OpeningEvidence:
    contract_evidence = amendment.load_cap_amendment_scope_bundle(REPO).contract[
        "opening_job52_evidence"
    ]
    return amendment.Job52OpeningEvidence(
        job51_root=str(job51_root),
        manifest_sha256="1" * 64,
        anchor_sha256="2" * 64,
        opening_commitment_total_usd=OPENING_COMMITMENT,
        opening_commitment_by_session_usd={FAILED_SESSION: OPENING_COMMITMENT},
        opening_commitment_count=1,
        cost_call_starts=2,
        cost_call_results=2,
        time_series_call_starts=1,
        time_series_call_results=0,
        failed_partial_bytes=89_745,
        failed_partial_sha256="3" * 64,
        superseded_terminal_attempt_id=contract_evidence[
            "old_cap_stop_superseded_only_for_attempt_id"
        ],
        superseded_terminal_cost_result_record_hash=contract_evidence[
            "old_cap_stop_superseded_only_for_cost_result_record_hash"
        ],
        predecessor_summary={},
    )


def _job55_control_context(tmp_path: Path) -> SimpleNamespace:
    mount = tmp_path / "synthetic-job55-volume"
    (mount / "cmbp-tier0").mkdir(parents=True)
    job51_root = mount / amendment.JOB51_ROOT_RELATIVE
    job51_root.mkdir()
    shared_lock = job51_root / "RUN_LOCK_V1"
    shared_lock.write_bytes(b"synthetic immutable shared Job51 lock\n")
    volume = _volume(mount)
    opening = _opening_evidence_for_controls(job51_root)
    job51_before = _tree_snapshot(job51_root)
    job55_root = amendment.initialize_job55_destination_tree(volume)
    adoption = amendment.ensure_job52_terminal_adoption(
        job55_root,
        opening,
        volume=volume,
        allow_initialize=True,
    )
    amendment.ensure_job55_lock_binding(
        job55_root,
        volume=volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption_receipt_sha256=adoption["adoption_sha256"],
        allow_initialize=True,
    )
    assert _tree_snapshot(job51_root) == job51_before
    return SimpleNamespace(
        mount=mount,
        volume=volume,
        job51_root=job51_root,
        job51_before=job51_before,
        job55_root=job55_root,
        opening=opening,
        adoption=adoption,
    )


def _job55_attempt(
    context: SimpleNamespace,
    *,
    attempt_id: str = ATTEMPT_ID,
    ordinal: int = 1,
    previous_attempt_id: str | None = None,
    previous_header_hash: str | None = None,
) -> tuple[tier0.AttemptJournal, Path, dict[str, Any]]:
    attempt_dir = context.job55_root / "attempts" / attempt_id
    attempt_dir.mkdir()
    journal = tier0.AttemptJournal(
        attempt_dir,
        attempt_id=attempt_id,
        volume=context.volume,
    )
    header = journal.append(
        "ATTEMPT_START",
        payload=amendment.build_job55_attempt_header_payload(
            attempt_id=attempt_id,
            volume=context.volume,
            readiness_receipt_sha256=JOB55_READINESS_SHA256,
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
            job55_attempt_ordinal=ordinal,
            previous_job55_attempt_id=previous_attempt_id,
            previous_job55_attempt_header_record_hash=previous_header_hash,
        ),
    )
    (attempt_dir / "sessions").mkdir()
    return journal, attempt_dir, header


def _validate_job55_anchor(
    context: SimpleNamespace,
    *,
    allow_initialize: bool = False,
    repair_header_only: bool = False,
) -> dict[str, Any]:
    return amendment.validate_job55_attempt_anchor(
        context.job55_root,
        volume=context.volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption_receipt_sha256=context.adoption["adoption_sha256"],
        allow_initialize=allow_initialize,
        repair_header_only=repair_header_only,
    )


def _summarize_job55(
    context: SimpleNamespace,
    *,
    require_client_constructed: bool = False,
    expected_sdk_identity_sha256: str | None = None,
) -> dict[str, Any]:
    return amendment.summarize_job55_attempts(
        context.job55_root,
        amendment.load_cap_amendment_scope_bundle(REPO).sessions,
        opening=context.opening,
        volume=context.volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption_receipt_sha256=context.adoption["adoption_sha256"],
        require_client_constructed=require_client_constructed,
        expected_sdk_identity_sha256=expected_sdk_identity_sha256,
    )


def _append_client_constructed(
    journal: tier0.AttemptJournal,
    *,
    sdk_identity_sha256: str = "4" * 64,
) -> dict[str, Any]:
    return journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "process_environment",
            "sdk_identity_sha256": sdk_identity_sha256,
        },
    )


def _acquire_in_context(
    context: SimpleNamespace,
    journal: tier0.AttemptJournal,
    attempt_dir: Path,
    request: tier0.SessionRequest,
    *,
    quote: Any,
    state: amendment.AmendedBudgetState | None = None,
    timeseries_error: BaseException | None = None,
    partial_bytes_before_error: bytes | None = None,
) -> tuple[_FakeClient, amendment.AmendedBudgetState]:
    budget = _opening_state() if state is None else state
    anchor_path = context.job55_root / amendment.JOB55_ATTEMPT_ANCHOR_NAME
    _validate_job55_anchor(
        context,
        allow_initialize=not anchor_path.exists(),
        repair_header_only=anchor_path.exists(),
    )
    client = _FakeClient(
        quote,
        timeseries_error=timeseries_error,
        partial_bytes_before_error=partial_bytes_before_error,
    )
    _acquire_amended_session_bytes(
        client,
        request,
        output_path=_staging_path(attempt_dir, request.session),
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=budget,
        volume=context.volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption_receipt_sha256=context.adoption["adoption_sha256"],
        ordinal=_request_ordinal(request),
    )
    return client, budget


def _publication_payload(
    request: tier0.SessionRequest,
    source_result: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "ordinal": _request_ordinal(request),
        "publication_mode": mode,
        "session_qc_sha256": "9" * 64,
        "decoded_records": request.expected_record_count,
        "compressed_bytes": source_result["payload"]["compressed_bytes"],
        "dbn_file_sha256": source_result["payload"]["dbn_file_sha256"],
        "source_attempt_id": source_result["attempt_id"],
        "source_timeseries_result_sequence": source_result["sequence"],
        "source_timeseries_result_record_hash": source_result["record_hash"],
    }


def _append_publication(
    journal: tier0.AttemptJournal,
    request: tier0.SessionRequest,
    source_result: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    return journal.append(
        mode,
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload=_publication_payload(request, source_result, mode=mode),
    )


def _job55_qc_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build one real journal/result and an offline source-bound QC V3 bundle."""

    context = _job55_control_context(tmp_path)
    request = _failed_request()
    journal, attempt_dir, _header = _job55_attempt(context)
    _acquire_in_context(
        context,
        journal,
        attempt_dir,
        request,
        quote=OPENING_COMMITMENT,
    )
    summary = _summarize_job55(context)
    assert summary["successful_stream_recovery_required"] is True
    assert len(summary["start_records"]) == 1
    assert len(summary["result_records"]) == 1
    start_record = summary["start_records"][0]
    result_record = summary["result_records"][0]
    bundle_dir = attempt_dir / "sessions" / f"{request.session}.bundle.part"
    data_path = bundle_dir / "data.cmbp-1.dbn.zst"
    metadata = tier0.validate_dbn_metadata(_FakeDBNStore(request), request)
    decoder = _decoder_summary(request)
    readiness = {"receipt_sha256": JOB55_READINESS_SHA256}
    qc = amendment.build_job55_session_qc(
        request=request,
        data_path=data_path,
        metadata_summary=metadata,
        decoder_summary=decoder,
        readiness=readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=context.adoption,
        start_record=start_record,
        result_record=result_record,
    )
    qc_path = bundle_dir / amendment.JOB55_SESSION_QC_NAME
    qc_path.write_bytes(tier0.canonical_json_bytes(qc) + b"\n")
    _patch_dbn_store(monkeypatch, {request.session: request})
    return SimpleNamespace(
        context=context,
        request=request,
        journal=journal,
        attempt_dir=attempt_dir,
        bundle_dir=bundle_dir,
        data_path=data_path,
        qc_path=qc_path,
        metadata=metadata,
        decoder=decoder,
        readiness=readiness,
        qc=qc,
        summary=summary,
        start_record=start_record,
        result_record=result_record,
        source_lookup=amendment.job55_source_record_lookup(summary),
    )


def _validate_job55_qc(fixture: SimpleNamespace) -> dict[str, Any]:
    return amendment.validate_job55_session_bundle(
        fixture.bundle_dir,
        request=fixture.request,
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=fixture.context.volume,
        source_record_lookup=fixture.source_lookup,
    )


def _publish_job55_qc(fixture: SimpleNamespace) -> tuple[Path, dict[str, Any]]:
    final_dir = fixture.context.job55_root / "sessions" / fixture.request.session
    fixture.bundle_dir.rename(final_dir)
    fixture.bundle_dir = final_dir
    payload = amendment.build_job55_publication_payload(
        request=fixture.request,
        qc=fixture.qc,
        mode="SESSION_PUBLISHED",
        ordinal=_request_ordinal(fixture.request),
    )
    publication = fixture.journal.append(
        "SESSION_PUBLISHED",
        session=fixture.request.session,
        request_sha256=fixture.request.market_request_sha256,
        payload=payload,
    )
    return final_dir, publication


def _full_job55_aggregate_fixture(tmp_path: Path) -> SimpleNamespace:
    """Construct a complete exact-scope aggregate without vendor or DBN access."""

    context = _job55_control_context(tmp_path)
    _validate_job55_anchor(context, allow_initialize=True)
    bundle = amendment.load_cap_amendment_scope_bundle(REPO)
    final_qc: dict[str, dict[str, Any]] = {}
    publications: list[dict[str, Any]] = []
    commitments: list[dict[str, Any]] = []
    quote_observations: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    job55_by_session: dict[str, str] = {}
    combined_by_session: dict[str, str] = {}
    job55_total = Decimal("0")
    for ordinal, request in enumerate(bundle.sessions, start=1):
        attempt_id = f"{ordinal:08d}-0000-4000-8000-000000000000"
        if request.session == FAILED_SESSION:
            quote = OPENING_COMMITMENT
        elif ordinal == len(bundle.sessions):
            quote = Decimal("1.599215102196")
        else:
            quote = Decimal("1.50")
        combined = quote + (OPENING_COMMITMENT if request.session == FAILED_SESSION else Decimal("0"))
        job55_total += quote
        job55_by_session[request.session] = format(quote, "f")
        combined_by_session[request.session] = format(combined, "f")
        result_hash = tier0.json_sha256(
            {"kind": "synthetic Job55 result", "session": request.session}
        )
        start_hash = tier0.json_sha256(
            {"kind": "synthetic Job55 start", "session": request.session}
        )
        raw_hash = tier0.json_sha256(
            {"kind": "synthetic Job55 DBN", "session": request.session}
        )
        source = {
            "attempt_id": attempt_id,
            "timeseries_start_sequence": 5,
            "timeseries_start_record_hash": start_hash,
            "timeseries_result_sequence": 6,
            "timeseries_result_record_hash": result_hash,
        }
        qc: dict[str, Any] = {
            "session": request.session,
            "request": amendment._request_projection(request),
            "raw_dbn": {
                "file_name": "data.cmbp-1.dbn.zst",
                "file_sha256": raw_hash,
                "compressed_bytes": 1000 + ordinal,
            },
            "decoder": _decoder_summary(request),
            "source_attempt": source,
        }
        qc["session_qc_sha256"] = tier0.json_sha256(qc)
        final_qc[request.session] = qc
        publications.append(
            {
                "attempt_id": attempt_id,
                "sequence": 7,
                "event": "SESSION_PUBLISHED",
                "session": request.session,
                "request_sha256": request.market_request_sha256,
                "payload": {
                    "ordinal": ordinal,
                    "publication_mode": "SESSION_PUBLISHED",
                    "session_qc_sha256": qc["session_qc_sha256"],
                    "decoded_records": request.expected_record_count,
                    "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
                    "dbn_file_sha256": raw_hash,
                    "source_attempt_id": attempt_id,
                    "source_timeseries_result_sequence": 6,
                    "source_timeseries_result_record_hash": result_hash,
                },
            }
        )
        commitments.append(
            {
                "job55_commitment_index": ordinal,
                "combined_commitment_index": ordinal + 1,
                "session": request.session,
                "fresh_quote_usd": format(quote, "f"),
                "timeseries_start_record_hash": start_hash,
            }
        )
        quote_observations.append(
            {
                "attempt_id": attempt_id,
                "session": request.session,
                "observed_sdk_quote": format(quote, "f"),
                "time_series_start_permitted": True,
            }
        )
        attempts.append(
            {
                "attempt_id": attempt_id,
                "job55_attempt_ordinal": ordinal,
            }
        )
    assert job55_total == Decimal("31.049607551098")
    assert job55_total + OPENING_COMMITMENT == Decimal("32.000000000000")
    summary: dict[str, Any] = {
        "job55_attempt_count": 21,
        "job55_cost_call_starts": 21,
        "job55_cost_call_results": 21,
        "job55_cost_call_errors": 0,
        "job55_timeseries_call_starts": 21,
        "job55_timeseries_call_results": 21,
        "job55_timeseries_call_errors": 0,
        "job55_commitment_count": 21,
        "combined_commitment_count": 22,
        "job55_committed_quote_total_usd": format(job55_total, "f"),
        "job55_committed_quote_by_session_usd": job55_by_session,
        "combined_committed_quote_total_usd": "32.000000000000",
        "combined_committed_quote_by_session_usd": combined_by_session,
        "terminal_authority_failure_observed": False,
        "successful_stream_recovery_required": False,
        "publication_records": publications,
        "published_sessions_in_journals": sorted(final_qc),
        "commitments": commitments,
        "quote_observations": quote_observations,
        "attempts": attempts,
    }
    readiness = {"receipt_sha256": JOB55_READINESS_SHA256}
    seal = amendment.build_job55_aggregate_seal(
        job55_root=context.job55_root,
        summary=summary,
        final_qc=final_qc,
        readiness=readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=context.adoption,
        volume=context.volume,
    )
    seal_path = context.job55_root / amendment.JOB55_AGGREGATE_SEAL_NAME
    seal_path.write_bytes(tier0.canonical_json_bytes(seal) + b"\n")
    receipt = amendment.build_job55_aggregate_receipt(
        bundle=bundle,
        opening=context.opening,
        summary=summary,
        final_qc=final_qc,
        seal=seal,
        readiness=readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=context.adoption,
        volume=context.volume,
    )
    receipt_path = context.job55_root / "receipts" / amendment.JOB55_AGGREGATE_NAME
    receipt_path.write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")
    return SimpleNamespace(
        context=context,
        bundle=bundle,
        readiness=readiness,
        final_qc=final_qc,
        summary=summary,
        seal=seal,
        seal_path=seal_path,
        receipt=receipt,
        receipt_path=receipt_path,
    )


def _job55_readiness_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Create a complete sterile readiness tree with an exact synthetic JUnit set."""

    bundle = amendment.load_cap_amendment_scope_bundle(REPO)
    repo = tmp_path / "synthetic-job55-readiness-repo"
    for relative in bundle.contract["required_bound_files"]:
        source = REPO / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    paths = amendment._paths(repo)
    cases = "".join(
        f'<testcase classname="v5.tests.test_cmbp_tier0_cap_amendment" '
        f'name="test_synthetic_job55_readiness_{index:02d}" />'
        for index in range(12)
    )
    paths["test_report"].write_text(
        f'<testsuite tests="12" failures="0" errors="0" skipped="0">{cases}</testsuite>\n',
        encoding="utf-8",
    )
    predecessor_readiness = {"receipt_sha256": "a" * 64}
    sdk_identity = {
        "artifact_type": "SYNTHETIC_JOB55_SDK_IDENTITY_TEST_ONLY",
        "databento": {"version": "test-only"},
        "compression": {"zstandard": "test-only"},
        "http_dependencies": {"requests": "test-only"},
        "trust_store": {"certifi": "test-only"},
    }
    monkeypatch.setattr(
        amendment,
        "load_cap_amendment_scope_bundle",
        lambda _root: bundle,
    )
    monkeypatch.setattr(
        paid,
        "validate_paid_readiness_receipt",
        lambda _root: predecessor_readiness,
    )
    monkeypatch.setattr(
        amendment,
        "job55_sdk_identity",
        lambda: copy.deepcopy(sdk_identity),
    )
    junit = amendment._job55_junit_counts(
        paths["test_report"], enforce_frozen_population=False
    )
    monkeypatch.setattr(amendment, "EXPECTED_FOCUSED_TEST_COUNT", junit["tests"])
    monkeypatch.setattr(
        amendment,
        "EXPECTED_FOCUSED_TEST_IDENTITY_SHA256",
        junit["test_case_identity_sha256"],
    )
    receipt = amendment.build_job55_readiness_receipt(
        repo,
        test_report_path=paths["test_report"],
    )
    paths["readiness"].write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")
    return SimpleNamespace(
        repo=repo,
        bundle=bundle,
        paths=paths,
        receipt=receipt,
        predecessor_readiness=predecessor_readiness,
        sdk_identity=sdk_identity,
        junit=junit,
    )


def _patch_job55_runner_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bypass_environment_gate: bool = True,
) -> SimpleNamespace:
    """Patch the runner around real temp journals/publication, never network."""

    from v5.ops import acquire_cmbp_tier0_cap_amendment as runner

    bundle = amendment.load_cap_amendment_scope_bundle(REPO)
    repo = tmp_path / "synthetic-job55-runner-repo"
    readiness_path = repo / "v5/work/cmbp-tier0-cap-amendment/LOCAL_READINESS_RECEIPT_V1.json"
    readiness_path.parent.mkdir(parents=True)
    readiness_path.write_bytes(b"synthetic immutable Job55 readiness\n")
    mount = tmp_path / "synthetic-job55-runner-volume"
    job51_root = mount / amendment.JOB51_ROOT_RELATIVE
    job55_root = mount / amendment.JOB55_ROOT_RELATIVE
    job51_root.mkdir(parents=True)
    (job51_root / "RUN_LOCK_V1").write_bytes(b"synthetic shared Job51 lock\n")
    for directory in (
        job55_root / "attempts",
        job55_root / "sessions",
        job55_root / "receipts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    volume = _volume(mount)
    opening = _opening_evidence_for_controls(job51_root)
    adoption = {"adoption_sha256": "b" * 64}
    readiness = {"receipt_sha256": "c" * 64}
    sdk_identity = {"synthetic_job55_runner_sdk": "offline-only"}
    actions: list[tuple[str, str]] = []
    lock_trace: list[tuple[str, Path]] = []
    anchor_calls: list[dict[str, Any]] = []
    anchored_attempt_ids: set[str] = set()
    stop_calls: list[Path] = []
    next_uuid = {"value": 100}

    def attempt_records() -> list[list[dict[str, Any]]]:
        records: list[list[dict[str, Any]]] = []
        for child in sorted((job55_root / "attempts").iterdir()):
            if child.is_dir():
                records.append(tier0.verify_attempt_journal(child)["records"])
        return records

    def anchor(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        anchor_calls.append(copy.deepcopy(kwargs))
        entries: list[dict[str, Any]] = []
        for records in attempt_records():
            header = records[0]
            if header["attempt_id"] not in anchored_attempt_ids:
                if (
                    len(records) != 1
                    or not (
                        kwargs.get("allow_initialize") is True
                        or kwargs.get("repair_header_only") is True
                    )
                ):
                    raise tier0.Tier0Error(
                        "synthetic unanchored attempt is not header-only",
                        status="STOP_JOB55_ATTEMPT_SET",
                    )
                anchored_attempt_ids.add(header["attempt_id"])
            payload = header["payload"]
            entries.append(
                {
                    "attempt_id": header["attempt_id"],
                    "header_record_hash": header["record_hash"],
                    "job55_attempt_ordinal": payload["job55_attempt_ordinal"],
                    "previous_job55_attempt_id": payload["previous_job55_attempt_id"],
                    "previous_job55_attempt_header_record_hash": payload[
                        "previous_job55_attempt_header_record_hash"
                    ],
                }
            )
        entries.sort(key=lambda item: item["job55_attempt_ordinal"])
        if anchored_attempt_ids != {item["attempt_id"] for item in entries}:
            raise tier0.Tier0Error(
                "synthetic anchored attempt disappeared",
                status="STOP_JOB55_ATTEMPT_SET",
            )
        return {
            "anchor_sha256": "d" * 64,
            "job55_attempt_count": len(entries),
            "job55_attempts": entries,
        }

    def summary(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        records = [record for group in attempt_records() for record in group]
        starts = [record for record in records if record["event"] == "TIMESERIES_CALL_START"]
        results = [record for record in records if record["event"] == "TIMESERIES_CALL_RESULT"]
        costs = [record for record in records if record["event"] == "COST_CALL_START"]
        cost_results = [record for record in records if record["event"] == "COST_CALL_RESULT"]
        publications = [record for record in records if record["event"] == "SESSION_PUBLISHED"]
        published = {str(record["session"]) for record in publications}
        job55_by_session = {session: Decimal("1.00") for session in published}
        combined_by_session = {
            session: value + (OPENING_COMMITMENT if session == FAILED_SESSION else Decimal("0"))
            for session, value in job55_by_session.items()
        }
        return {
            "job55_attempt_count": len(attempt_records()),
            "job55_cost_call_starts": len(costs),
            "job55_cost_call_results": len(cost_results),
            "job55_cost_call_errors": 0,
            "job55_timeseries_call_starts": len(starts),
            "job55_timeseries_call_results": len(results),
            "job55_timeseries_call_errors": 0,
            "job55_commitment_count": len(starts),
            "combined_commitment_count": len(starts) + 1,
            "job55_committed_quote_total_usd": format(Decimal(len(starts)), "f"),
            "job55_committed_quote_by_session_usd": {
                key: format(value, "f") for key, value in job55_by_session.items()
            },
            "combined_committed_quote_total_usd": format(
                OPENING_COMMITMENT + Decimal(len(starts)), "f"
            ),
            "combined_committed_quote_by_session_usd": {
                key: format(value, "f") for key, value in combined_by_session.items()
            },
            "job55_cost_call_count_by_session": {str(record["session"]): 1 for record in costs},
            "job55_start_count_by_session": {str(record["session"]): 1 for record in starts},
            "start_records": starts,
            "result_records": results,
            "publication_records": publications,
            "published_sessions_in_journals": sorted(published),
            "recoverable_staging": [],
            "partial_or_failed_staging": [],
            "successful_stream_recovery_required": False,
            "successful_stream_recovery_record": None,
            "terminal_authority_failure_observed": False,
            "terminal_authority_failure": None,
            "commitments": [],
            "quote_observations": [],
            "attempts": [],
        }

    def finals(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for child in sorted((job55_root / "sessions").iterdir()):
            if child.is_dir():
                result[child.name] = tier0.strict_json(
                    child / amendment.JOB55_SESSION_QC_NAME
                )
        return result

    class Lock:
        def __init__(self, root: Path, *, volume: tier0.VolumeIdentity) -> None:
            assert volume == volume_identity
            self.root = Path(root)

        def __enter__(self) -> "Lock":
            lock_trace.append(("enter", self.root))
            return self

        def __exit__(self, *_args: Any) -> None:
            lock_trace.append(("exit", self.root))

    volume_identity = volume

    class Client:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(get_cost=self.get_cost)
            self.timeseries = SimpleNamespace(get_range=self.get_range)

        def get_cost(self, **kwargs: Any) -> Decimal:
            session = str(kwargs["start"])[:10]
            actions.append(("metadata.get_cost", session))
            return Decimal("1.00")

        def get_range(self, **kwargs: Any) -> None:
            session = str(kwargs["start"])[:10]
            actions.append(("timeseries.get_range", session))
            Path(kwargs["path"]).write_bytes(f"synthetic DBN {session}".encode())

    def acquire(
        client: Client,
        request: tier0.SessionRequest,
        *,
        output_path: Path,
        journal: tier0.AttemptJournal,
        **_kwargs: Any,
    ) -> str:
        request_sha = request.market_request_sha256
        journal.append(
            "COST_CALL_START",
            session=request.session,
            request_sha256=request_sha,
            payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
        )
        quote = client.metadata.get_cost(**request.market_parameters)
        cost_result = journal.append(
            "COST_CALL_RESULT",
            session=request.session,
            request_sha256=request_sha,
            payload={"observed_sdk_quote": format(quote, "f")},
        )
        start = journal.append(
            "TIMESERIES_CALL_START",
            session=request.session,
            request_sha256=request_sha,
            payload={
                "method": "timeseries.get_range",
                "parameters": request.market_parameters,
                "cost_result_record_hash": cost_result["record_hash"],
            },
        )
        client.timeseries.get_range(**request.market_parameters, path=str(output_path))
        journal.append(
            "TIMESERIES_CALL_RESULT",
            session=request.session,
            request_sha256=request_sha,
            payload={
                "timeseries_start_record_hash": start["record_hash"],
                "compressed_bytes": output_path.stat().st_size,
                "dbn_file_sha256": tier0.file_sha256(output_path),
            },
        )
        return format(quote, "f")

    def build_qc(**kwargs: Any) -> dict[str, Any]:
        request = kwargs["request"]
        data_path = Path(kwargs["data_path"])
        start = kwargs["start_record"]
        result = kwargs["result_record"]
        qc: dict[str, Any] = {
            "session": request.session,
            "request": amendment._request_projection(request),
            "raw_dbn": {
                "file_sha256": tier0.file_sha256(data_path),
                "compressed_bytes": data_path.stat().st_size,
            },
            "decoder": kwargs["decoder_summary"],
            "source_attempt": {
                "attempt_id": start["attempt_id"],
                "timeseries_start_sequence": start["sequence"],
                "timeseries_start_record_hash": start["record_hash"],
                "timeseries_result_sequence": result["sequence"],
                "timeseries_result_record_hash": result["record_hash"],
            },
        }
        qc["session_qc_sha256"] = tier0.json_sha256(qc)
        return qc

    def build_seal(**_kwargs: Any) -> dict[str, Any]:
        seal = {"artifact_type": "SYNTHETIC_JOB55_AGGREGATE_SEAL_TEST_ONLY"}
        seal["seal_sha256"] = tier0.self_hash(seal, "seal_sha256")
        return seal

    def validate_seal(path: Path, **_kwargs: Any) -> dict[str, Any]:
        return tier0.strict_json(path)

    def build_receipt(**_kwargs: Any) -> dict[str, Any]:
        receipt = {
            "artifact_type": "SYNTHETIC_JOB55_AGGREGATE_TEST_ONLY",
            "status": "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_PASS",
        }
        receipt["receipt_sha256"] = tier0.self_hash(receipt, "receipt_sha256")
        return receipt

    def validate_receipt(path: Path, **_kwargs: Any) -> dict[str, Any]:
        return tier0.strict_json(path)

    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(sys, "argv", [str(Path(runner.__file__))])
    monkeypatch.setattr(runner.uuid, "uuid4", lambda: uuid.UUID(int=next_uuid.__setitem__("value", next_uuid["value"] + 1) or next_uuid["value"], version=4))
    monkeypatch.setattr(tier0, "inspect_destination_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(amendment, "load_cap_amendment_scope_bundle", lambda _root: bundle)
    monkeypatch.setattr(amendment, "validate_job55_readiness_receipt", lambda _root: readiness)
    monkeypatch.setattr(amendment, "_paths", lambda _root: {"readiness": readiness_path})
    monkeypatch.setattr(amendment, "validate_job52_opening_evidence", lambda *_args, **_kwargs: opening)
    monkeypatch.setattr(amendment, "initialize_job55_destination_tree", lambda _volume: job55_root)
    monkeypatch.setattr(amendment, "ensure_job52_terminal_adoption", lambda *_args, **_kwargs: adoption)
    monkeypatch.setattr(amendment, "ensure_job55_lock_binding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(amendment, "validate_job55_attempt_anchor", anchor)
    monkeypatch.setattr(amendment, "validate_job55_job_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(amendment, "job55_sdk_identity", lambda: sdk_identity)
    monkeypatch.setattr(amendment, "summarize_job55_attempts", summary)
    monkeypatch.setattr(amendment, "validate_job55_existing_session_population", finals)
    monkeypatch.setattr(amendment, "Job55RunLock", Lock)
    monkeypatch.setattr(
        runner,
        "_read_api_key_after_gates",
        lambda: actions.append(("credential", "read")) or ("synthetic-key", "synthetic_fixture"),
    )
    monkeypatch.setattr(
        runner,
        "_construct_exact_client",
        lambda _key: actions.append(("client", "constructed")) or Client(),
    )
    if bypass_environment_gate:
        monkeypatch.setattr(runner, "_reject_request_environment_overrides", lambda: None)
    monkeypatch.setattr(amendment, "acquire_amended_session_bytes", acquire)
    monkeypatch.setattr(
        tier0,
        "stream_dbn_qc",
        lambda _path, request, **_kwargs: ({}, _decoder_summary(request)),
    )
    monkeypatch.setattr(amendment, "build_job55_session_qc", build_qc)
    monkeypatch.setattr(
        amendment,
        "validate_job55_session_bundle",
        lambda path, **_kwargs: tier0.strict_json(Path(path) / amendment.JOB55_SESSION_QC_NAME),
    )
    monkeypatch.setattr(amendment, "build_job55_aggregate_seal", build_seal)
    monkeypatch.setattr(amendment, "validate_job55_aggregate_seal", validate_seal)
    monkeypatch.setattr(amendment, "build_job55_aggregate_receipt", build_receipt)
    monkeypatch.setattr(amendment, "validate_job55_aggregate_receipt", validate_receipt)
    monkeypatch.setattr(
        amendment,
        "write_job55_attempt_stop_receipt",
        lambda path, **_kwargs: stop_calls.append(Path(path)),
    )
    return SimpleNamespace(
        runner=runner,
        repo=repo,
        bundle=bundle,
        volume=volume,
        job51_root=job51_root,
        job55_root=job55_root,
        readiness_path=readiness_path,
        actions=actions,
        lock_trace=lock_trace,
        anchor_calls=anchor_calls,
        stop_calls=stop_calls,
        adoption=adoption,
        readiness=readiness,
        validate_receipt=validate_receipt,
    )


def test_job55_contract_and_scope_are_exactly_cap_only() -> None:
    bundle = amendment.load_cap_amendment_scope_bundle(REPO)

    assert amendment.EXPECTED_JOB55_PLAN_FILE_SHA256 == JOB55_PLAN_FILE_SHA256
    assert amendment.EXPECTED_JOB55_PROGRAM_CONTRACT_SHA256 == JOB55_CONTRACT_SHA256
    assert amendment.EXPECTED_JOB55_PROGRAM_CONTRACT_FILE_SHA256 == JOB55_CONTRACT_FILE_SHA256
    assert bundle.plan_file_sha256 == JOB55_PLAN_FILE_SHA256
    assert bundle.contract["contract_sha256"] == JOB55_CONTRACT_SHA256
    assert bundle.contract_file_sha256 == JOB55_CONTRACT_FILE_SHA256
    assert bundle.base_bundle == tier0.load_scope_bundle(REPO)
    assert amendment.JOB55_PER_SESSION_LIFETIME_CAP_USD == Decimal("2.00")
    assert amendment.JOB55_TOTAL_COMMITTED_CAP_USD == Decimal("32.00")
    assert amendment.JOB55_NAMESPACE == "cmbp-tier0/job55"
    assert amendment.JOB55_SESSION_QC_NAME == "SESSION_QC_V3.json"
    assert amendment.JOB55_AGGREGATE_NAME == (
        "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_RECEIPT_V1.json"
    )


def test_job55_scope_reconstructs_the_frozen_21_requests_without_widening() -> None:
    job55 = amendment.load_cap_amendment_scope_bundle(REPO).base_bundle
    frozen = tier0.load_scope_bundle(REPO)

    assert len(job55.sessions) == 21
    assert sum(request.expected_record_count for request in job55.sessions) == 2_373_877_845
    assert job55.sessions == frozen.sessions
    assert job55.scope == frozen.scope
    assert all(
        request.market_parameters["dataset"] == tier0.EXPECTED_DATASET
        for request in job55.sessions
    )
    assert all(
        request.market_parameters["schema"] == tier0.EXPECTED_SCHEMA
        for request in job55.sessions
    )
    assert all(
        request.market_parameters["stype_in"] == tier0.EXPECTED_STYPE_IN
        for request in job55.sessions
    )


def test_job55_contract_loading_does_not_mutate_repo_job51_or_job52_evidence() -> None:
    roots = (
        REPO / "v5/work/cmbp-tier0-acquisition",
        REPO / "v5/work/cmbp-tier0-paid-resume",
    )
    before = {str(root): _tree_snapshot(root) for root in roots}
    amendment.load_cap_amendment_scope_bundle(REPO)
    after = {str(root): _tree_snapshot(root) for root in roots}
    assert after == before


def test_job52_production_opening_pins_bind_the_exact_stopped_evidence() -> None:
    pins = amendment.EXPECTED_JOB52_EVIDENCE_PINS
    files = dict(pins.file_sha256)

    assert pins.anchor_sha256 == "4fccdedd4e26d7a2dd2fe7fe7fc346be777b40367513be5ecc66cede2a518068"
    assert pins.failed_partial_bytes == 89_745
    assert pins.failed_partial_sha256 == (
        "a0049b77c2129d6302bbaa3b03a2bcade3213fc9f172bf7961d13082d71c3da8"
    )
    assert files[paid.PAID_ATTEMPT_ANCHOR_NAME] == (
        "464618482b4ac43086175d299943be560821a9135bb39cefe610892168a0361b"
    )
    assert files["RUN_LOCK_V1"] == (
        "af91deb42e8fe0d798af38763c4edee44e51da443ac4bbda1e6aec92014ee2d1"
    )
    assert files[paid.PAID_LOCK_BINDING_NAME] == (
        "d41f6651eaee3b2916346ada4708e3e6e1206d991d333093b0f572313be01505"
    )
    assert all(not path.startswith("job55") for path in (*pins.directories, *files))


def test_job52_opening_evidence_reconstructs_exactly_and_does_not_mutate_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_job52_tree(tmp_path)
    before = _tree_snapshot(fixture.job51_root)
    calls = _patch_job52_semantic_validation(monkeypatch, fixture.summary)

    opening = amendment._validate_job52_opening_evidence_with_pins(
        fixture.job51_root,
        volume=fixture.volume,
        pins=fixture.pins,
    )

    assert opening.job51_root == str(fixture.job51_root)
    assert opening.opening_commitment_total_usd == OPENING_COMMITMENT
    assert dict(opening.opening_commitment_by_session_usd) == {
        FAILED_SESSION: OPENING_COMMITMENT
    }
    assert opening.opening_commitment_count == 1
    assert opening.cost_call_starts == 2
    assert opening.cost_call_results == 2
    assert opening.time_series_call_starts == 1
    assert opening.time_series_call_results == 0
    assert opening.failed_partial_bytes == fixture.partial_path.stat().st_size
    assert opening.failed_partial_sha256 == tier0.file_sha256(fixture.partial_path)
    assert opening.anchor_sha256 == fixture.pins.anchor_sha256
    assert tier0.SHA256_RE.fullmatch(opening.manifest_sha256)
    assert _tree_snapshot(fixture.job51_root) == before
    assert len(calls) == 1
    assert calls[0]["job_root"] == fixture.job51_root
    assert calls[0]["sessions"] == amendment.load_cap_amendment_scope_bundle(REPO).sessions
    assert calls[0]["kwargs"]["require_legacy_stop"] is True
    assert calls[0]["kwargs"]["require_client_constructed"] is True
    assert calls[0]["kwargs"]["expected_readiness_sha256"] == (
        amendment.EXPECTED_JOB52_READINESS_SHA256
    )


@pytest.mark.parametrize(
    "tamper",
    ("extra_file", "extra_directory", "partial_bytes", "missing_anchor", "symlink"),
)
def test_job52_opening_evidence_refuses_population_or_byte_drift_before_semantic_parser(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_job52_tree(tmp_path)
    calls = _patch_job52_semantic_validation(monkeypatch, fixture.summary)
    if tamper == "extra_file":
        (fixture.job51_root / "unexpected.txt").write_text("out of scope", encoding="utf-8")
    elif tamper == "extra_directory":
        (fixture.job51_root / "unexpected-directory").mkdir()
    elif tamper == "partial_bytes":
        fixture.partial_path.write_bytes(b"self-consistent but different partial")
    elif tamper == "missing_anchor":
        fixture.anchor_path.unlink()
    else:
        (fixture.job51_root / "unexpected-link").symlink_to(fixture.partial_path)

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment._validate_job52_opening_evidence_with_pins(
            fixture.job51_root,
            volume=fixture.volume,
            pins=fixture.pins,
        )
    assert raised.value.status == "STOP_JOB55_PREDECESSOR_DRIFT"
    assert calls == []


def test_job52_opening_evidence_refuses_self_rehashed_anchor_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_job52_tree(tmp_path)
    calls = _patch_job52_semantic_validation(monkeypatch, fixture.summary)
    anchor = tier0.strict_json(fixture.anchor_path)
    anchor["attempt_ids"] = ["different-self-consistent-attempt"]
    anchor["anchor_sha256"] = tier0.self_hash(anchor, "anchor_sha256")
    fixture.anchor_path.write_bytes(tier0.canonical_json_bytes(anchor) + b"\n")
    changed_files = dict(fixture.pins.file_sha256)
    changed_files[paid.PAID_ATTEMPT_ANCHOR_NAME] = tier0.file_sha256(fixture.anchor_path)
    changed_pins = amendment.Job52EvidencePins(
        directories=fixture.pins.directories,
        file_sha256=changed_files,
        anchor_sha256=fixture.pins.anchor_sha256,
        failed_partial_bytes=fixture.pins.failed_partial_bytes,
        failed_partial_sha256=fixture.pins.failed_partial_sha256,
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment._validate_job52_opening_evidence_with_pins(
            fixture.job51_root,
            volume=fixture.volume,
            pins=changed_pins,
        )
    assert raised.value.status == "STOP_JOB55_PREDECESSOR_DRIFT"
    assert calls == []


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("cost_call_starts",), 1),
        (("cost_call_starts",), 3),
        (("timeseries_call_starts",), 0),
        (("timeseries_call_starts",), 2),
        (("committed_quote_total_usd",), "0.950392448901"),
        (("published_sessions_in_journals",), [FAILED_SESSION]),
        (("commitments", 0, "fresh_quote_usd"), "0.950392448903"),
        (("quote_observations", 0, "cost_result_record_hash"), "f" * 64),
        (("partial_or_failed_staging", 0, "compressed_bytes"), 1),
        (("partial_or_failed_staging", 0, "file_sha256"), "e" * 64),
    ),
)
def test_job52_opening_evidence_refuses_semantic_ledger_or_source_drift(
    path: tuple[Any, ...],
    value: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_job52_tree(tmp_path)
    summary = copy.deepcopy(fixture.summary)
    target: Any = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    calls = _patch_job52_semantic_validation(monkeypatch, summary)

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment._validate_job52_opening_evidence_with_pins(
            fixture.job51_root,
            volume=fixture.volume,
            pins=fixture.pins,
        )
    assert raised.value.status == "STOP_JOB55_PREDECESSOR_DRIFT"
    assert len(calls) == 1


def test_opening_ledger_and_same_price_retry_project_exactly_without_rounding() -> None:
    projection = amendment.project_amended_commitment(
        committed_total_usd=OPENING_COMMITMENT,
        committed_session_usd=OPENING_COMMITMENT,
        fresh_quote_usd=OPENING_COMMITMENT,
    )
    assert projection == {
        "committed_quote_session_after_usd": SAME_PRICE_RETRY_TOTAL,
        "committed_quote_total_after_usd": SAME_PRICE_RETRY_TOTAL,
        "per_session_lifetime_cap_pass": True,
        "total_cap_pass": True,
        "job55_retry_start_allowance_pass": True,
        "time_series_start_permitted": True,
    }


def test_amended_session_cap_allows_exact_200_and_refuses_smallest_excess() -> None:
    exact = amendment.project_amended_commitment(
        committed_total_usd=OPENING_COMMITMENT,
        committed_session_usd=OPENING_COMMITMENT,
        fresh_quote_usd=SESSION_HEADROOM,
    )
    excess = amendment.project_amended_commitment(
        committed_total_usd=OPENING_COMMITMENT,
        committed_session_usd=OPENING_COMMITMENT,
        fresh_quote_usd=SESSION_HEADROOM + Decimal("0.000000000001"),
    )
    assert exact["committed_quote_session_after_usd"] == Decimal("2.000000000000")
    assert exact["per_session_lifetime_cap_pass"] is True
    assert excess["committed_quote_session_after_usd"] == Decimal("2.000000000001")
    assert excess["per_session_lifetime_cap_pass"] is False
    assert excess["total_cap_pass"] is True


def test_amended_total_cap_allows_exact_3200_and_refuses_smallest_excess() -> None:
    exact = amendment.project_amended_commitment(
        committed_total_usd=Decimal("31.00"),
        committed_session_usd=Decimal("1.00"),
        fresh_quote_usd=Decimal("1.00"),
    )
    excess = amendment.project_amended_commitment(
        committed_total_usd=Decimal("31.00"),
        committed_session_usd=Decimal("1.00"),
        fresh_quote_usd=Decimal("1.000000000001"),
    )
    assert exact["committed_quote_total_after_usd"] == Decimal("32.00")
    assert exact["total_cap_pass"] is True
    assert excess["committed_quote_total_after_usd"] == Decimal("32.000000000001")
    assert excess["total_cap_pass"] is False


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        0,
        1.0,
        "0.95",
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("-1"),
        Decimal("-0"),
    ),
)
def test_amended_projection_refuses_non_decimal_nonfinite_negative_and_signed_zero(
    value: Any,
) -> None:
    with pytest.raises(tier0.Tier0Error):
        amendment.project_amended_commitment(
            committed_total_usd=OPENING_COMMITMENT,
            committed_session_usd=OPENING_COMMITMENT,
            fresh_quote_usd=value,
        )


def test_amended_projection_is_independent_of_decimal_context_precision() -> None:
    tiny_excess = Decimal("1.0496075510980000000000000000000000000001")
    projection = amendment.project_amended_commitment(
        committed_total_usd=OPENING_COMMITMENT,
        committed_session_usd=OPENING_COMMITMENT,
        fresh_quote_usd=tiny_excess,
    )
    assert projection["committed_quote_session_after_usd"] > Decimal("2.00")
    assert projection["per_session_lifetime_cap_pass"] is False


def test_amended_budget_state_opens_only_from_the_exact_job52_commitment() -> None:
    state = _opening_state()
    assert state.committed_total_usd == OPENING_COMMITMENT
    assert state.committed_by_session_usd == {FAILED_SESSION: OPENING_COMMITMENT}
    assert state.commitment_count == 1
    assert state.job55_committed_total_usd == Decimal("0")
    assert state.job55_committed_by_session_usd == {}
    assert state.job55_commitment_count == 0
    assert state.job55_start_count_by_session == {}
    assert state.job55_cost_call_count == 0
    assert state.job55_cost_call_count_by_session == {}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("opening_commitment_total_usd", "0.950392448901"),
        ("opening_commitment_total_usd", "0.950392448903"),
        ("opening_commitment_by_session_usd", {}),
        ("opening_commitment_count", 0),
        ("opening_commitment_count", 2),
        ("opening_commitment_count", True),
    ),
)
def test_amended_budget_state_refuses_forged_or_nonopening_evidence(
    field: str,
    value: Any,
) -> None:
    opening = _opening_summary()
    opening[field] = value
    with pytest.raises(tier0.Tier0Error):
        amendment.AmendedBudgetState.from_opening_evidence(opening)


def test_same_price_retry_is_quote_start_adjacent_and_carries_both_ledgers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    output_path = _staging_path(attempt_dir, request.session)
    state = _opening_state()
    actions: list[str] = []

    durable_append = journal.append

    def tracking_append(event: str, **kwargs: Any) -> dict[str, Any]:
        record = durable_append(event, **kwargs)
        if event == "TIMESERIES_CALL_START":
            actions.append("journal.timeseries_start.durable")
        return record

    monkeypatch.setattr(journal, "append", tracking_append)

    client = _FakeClient(OPENING_COMMITMENT, actions=actions)
    quote = _acquire_amended_session_bytes(
        client,
        request,
        output_path=output_path,
        journal=journal,
        pre_pair_gate=lambda: None,
        budget_state=state,
        readiness_receipt_sha256="6" * 64,
        readiness_receipt_file_sha256="7" * 64,
        ordinal=1,
        progress=lambda message: actions.append(f"progress:{message}"),
    )

    assert quote == format(OPENING_COMMITMENT, "f")
    quote_index = actions.index("metadata.get_cost")
    assert actions[quote_index + 1:quote_index + 3] == [
        "journal.timeseries_start.durable",
        "timeseries.get_range",
    ]
    assert not any(
        action.startswith("progress:")
        for action in actions[quote_index + 1:quote_index + 3]
    )
    assert client.cost_kwargs == [request.market_parameters]
    assert {
        key: client.timeseries_kwargs[0][key]
        for key in request.market_parameters
    } == request.market_parameters
    assert client.timeseries_kwargs[0]["stype_out"] == tier0.EXPECTED_STYPE_OUT
    assert client.timeseries_kwargs[0]["limit"] is None

    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    result_index = next(
        index for index, record in enumerate(records) if record["event"] == "COST_CALL_RESULT"
    )
    result = records[result_index]
    start = records[result_index + 1]
    assert start["event"] == "TIMESERIES_CALL_START"
    assert start["sequence"] == result["sequence"] + 1
    assert start["previous_hash"] == result["record_hash"]
    assert start["request_sha256"] == result["request_sha256"] == request.market_request_sha256
    assert start["payload"]["fresh_quote_usd"] == "0.950392448902"
    assert start["payload"]["combined_committed_quote_session_before_usd"] == "0.950392448902"
    assert start["payload"]["combined_committed_quote_session_after_usd"] == "1.900784897804"
    assert start["payload"]["combined_committed_quote_total_before_usd"] == "0.950392448902"
    assert start["payload"]["combined_committed_quote_total_after_usd"] == "1.900784897804"
    assert start["payload"]["job55_committed_quote_session_before_usd"] == "0"
    assert start["payload"]["job55_committed_quote_session_after_usd"] == "0.950392448902"
    assert start["payload"]["job55_committed_quote_total_before_usd"] == "0"
    assert start["payload"]["job55_committed_quote_total_after_usd"] == "0.950392448902"
    assert start["payload"]["combined_commitment_index"] == 2
    assert start["payload"]["job55_commitment_index"] == 1
    assert start["payload"]["per_session_lifetime_cap_usd"] == "2.00"
    assert start["payload"]["total_cap_usd"] == "32.00"
    assert state.committed_total_usd == SAME_PRICE_RETRY_TOTAL
    assert state.job55_committed_total_usd == OPENING_COMMITMENT
    assert state.job55_start_count_by_session == {FAILED_SESSION: 1}
    assert state.job55_cost_call_count_by_session == {FAILED_SESSION: 1}
    assert capsys.readouterr() == ("", "")


def test_exact_session_headroom_passes_but_smallest_excess_stops_before_timeseries(
    tmp_path: Path,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    state = _opening_state()
    client = _FakeClient(SESSION_HEADROOM + Decimal("0.000000000001"))

    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=_staging_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert raised.value.status == "STOP_JOB55_SESSION_CAP"
    assert client.cost_kwargs == [request.market_parameters]
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == OPENING_COMMITMENT
    assert state.job55_commitment_count == 0
    assert state.job55_cost_call_count_by_session == {FAILED_SESSION: 1}


@pytest.mark.parametrize(
    "value",
    (True, "0.50", Decimal("NaN"), Decimal("Infinity"), -1, Decimal("-0"), -0.0),
)
def test_malformed_job55_quote_is_durably_terminal_before_timeseries(
    value: Any,
    tmp_path: Path,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    state = _opening_state()
    client = _FakeClient(value)

    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=_staging_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == OPENING_COMMITMENT
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    result = next(record for record in records if record["event"] == "COST_CALL_RESULT")
    assert result["payload"]["quote_valid"] is False
    assert result["payload"]["time_series_start_permitted"] is False
    assert result["payload"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    assert state.job55_cost_call_count_by_session == {FAILED_SESSION: 1}


def test_cost_error_commits_nothing_and_redacts_vendor_text(tmp_path: Path) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    state = _opening_state()
    secret = "synthetic-secret-never-journal"
    client = _FakeClient(RuntimeError(secret))

    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=_staging_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.timeseries_kwargs == []
    assert state.committed_total_usd == OPENING_COMMITMENT
    assert state.job55_commitment_count == 0
    assert state.job55_cost_call_count_by_session == {FAILED_SESSION: 1}
    raw = (attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl").read_text(encoding="utf-8")
    assert secret not in raw

    state = _next_process_state(state)
    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(Decimal("0"))
    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            second,
            request,
            output_path=_staging_path(second_attempt_dir, request.session),
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert raised.value.status == "STOP_JOB55_RETRY_EXHAUSTED"
    assert second.cost_kwargs == []


@pytest.mark.parametrize("first_quote", (Decimal("0"), Decimal("0.000000000001"), OPENING_COMMITMENT))
def test_first_job55_start_exhausts_failed_session_retry_even_when_transport_fails(
    first_quote: Decimal,
    tmp_path: Path,
) -> None:
    request = _failed_request()
    first_journal, first_attempt_dir = _journal(tmp_path)
    state = _opening_state()
    first = _FakeClient(
        first_quote,
        timeseries_error=OSError("synthetic transport failure"),
        partial_bytes_before_error=b"immutable-job55-partial",
    )

    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            first,
            request,
            output_path=_staging_path(first_attempt_dir, request.session),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.job55_start_count_by_session == {FAILED_SESSION: 1}
    expected_total = OPENING_COMMITMENT + first_quote
    assert state.committed_total_usd == expected_total

    state = _next_process_state(state)
    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(Decimal("0"))
    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            second,
            request,
            output_path=_staging_path(second_attempt_dir, request.session),
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert raised.value.status == "STOP_JOB55_RETRY_EXHAUSTED"
    assert second.cost_kwargs == []
    assert second.timeseries_kwargs == []


def test_quote_only_crash_commits_zero_but_exhausts_the_single_cost_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _failed_request()
    first_journal, first_attempt_dir = _journal(tmp_path)
    state = _opening_state()

    class SyntheticCrash(BaseException):
        pass

    durable_append = first_journal.append

    def crash_before_start(event: str, **kwargs: Any) -> dict[str, Any]:
        if event == "TIMESERIES_CALL_START":
            raise SyntheticCrash()
        return durable_append(event, **kwargs)

    monkeypatch.setattr(first_journal, "append", crash_before_start)

    first = _FakeClient(OPENING_COMMITMENT)
    with pytest.raises(SyntheticCrash):
        _acquire_amended_session_bytes(
            first,
            request,
            output_path=_staging_path(first_attempt_dir, request.session),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert state.committed_total_usd == OPENING_COMMITMENT
    assert state.job55_start_count_by_session == {}
    assert state.job55_cost_call_count_by_session == {FAILED_SESSION: 1}
    assert first.timeseries_kwargs == []

    state = _next_process_state(state)
    second_journal, second_attempt_dir = _journal(tmp_path, attempt_id=SECOND_ATTEMPT_ID)
    second = _FakeClient(OPENING_COMMITMENT)
    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            second,
            request,
            output_path=_staging_path(second_attempt_dir, request.session),
            journal=second_journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert raised.value.status == "STOP_JOB55_RETRY_EXHAUSTED"
    assert first.cost_kwargs == [request.market_parameters]
    assert second.cost_kwargs == []
    assert second.timeseries_kwargs == []


def test_durable_start_crash_is_not_refunded_in_the_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    state = _opening_state()

    class SyntheticCrash(BaseException):
        pass

    durable_append = journal.append

    def crash_after_start(event: str, **kwargs: Any) -> dict[str, Any]:
        record = durable_append(event, **kwargs)
        if event == "TIMESERIES_CALL_START":
            raise SyntheticCrash()
        return record

    monkeypatch.setattr(journal, "append", crash_after_start)

    client = _FakeClient(OPENING_COMMITMENT)
    with pytest.raises(SyntheticCrash):
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=_staging_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert client.timeseries_kwargs == []
    records = tier0.verify_attempt_journal(attempt_dir)["records"]
    assert records[-1]["event"] == "TIMESERIES_CALL_START"
    assert records[-1]["payload"]["fresh_quote_usd"] == "0.950392448902"
    assert records[-1]["payload"]["job55_commitment_index"] == 1
    assert records[-1]["payload"]["combined_committed_quote_total_after_usd"] == "1.900784897804"


@pytest.mark.parametrize("variant", ("outside_attempt", "wrong_filename", "job51_namespace"))
def test_acquire_refuses_any_noncanonical_job55_staging_path_before_gate_or_vendor(
    variant: str,
    tmp_path: Path,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    if variant == "outside_attempt":
        output = tmp_path / "outside.dbn.zst"
    elif variant == "wrong_filename":
        staging = attempt_dir / "sessions" / f"{request.session}.bundle.part"
        staging.mkdir(parents=True)
        output = staging / "widened-name.dbn.zst"
    else:
        output = (
            tmp_path
            / "cmbp-tier0"
            / "job51"
            / "attempts"
            / ATTEMPT_ID
            / "sessions"
            / f"{request.session}.bundle.part"
            / "data.cmbp-1.dbn.zst"
        )
    client = _FakeClient(Decimal("0"))
    gates: list[str] = []

    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: gates.append("gate"),
            budget_state=_opening_state(),
        )
    assert raised.value.status == "STOP_JOB55_EXTERNAL_PATH"
    assert gates == []
    assert client.cost_kwargs == []
    assert client.timeseries_kwargs == []


def test_failed_session_is_mandatory_first_job55_vendor_target(tmp_path: Path) -> None:
    request = next(
        item
        for item in tier0.load_scope_bundle(REPO).sessions
        if item.session != FAILED_SESSION
    )
    journal, attempt_dir = _journal(tmp_path)
    client = _FakeClient(Decimal("0"))
    gates: list[str] = []

    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            client,
            request,
            output_path=_staging_path(attempt_dir, request.session),
            journal=journal,
            pre_pair_gate=lambda: gates.append("gate"),
            budget_state=_opening_state(),
        )
    assert raised.value.status == "STOP_JOB55_FIRST_TARGET"
    assert gates == []
    assert client.cost_kwargs == []


def test_later_session_requires_failed_session_publication_even_after_its_cost_call() -> None:
    later_session = next(
        request.session
        for request in tier0.load_scope_bundle(REPO).sessions
        if request.session != FAILED_SESSION
    )
    state = amendment.AmendedBudgetState(
        job55_cost_call_count_by_session={FAILED_SESSION: 1},
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        state.assert_cost_call_permitted(later_session)
    assert raised.value.status == "STOP_JOB55_FIRST_TARGET"

    resumed = amendment.AmendedBudgetState(
        job55_cost_call_count_by_session={FAILED_SESSION: 1},
        published_sessions={FAILED_SESSION},
    )
    resumed.assert_cost_call_permitted(later_session)


def test_one_process_attempt_cannot_make_a_second_vendor_pair_after_cap_stop(
    tmp_path: Path,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    output = _staging_path(attempt_dir, request.session)
    state = _opening_state()
    first = _FakeClient(SESSION_HEADROOM + Decimal("0.000000000001"))
    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            first,
            request,
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )

    second = _FakeClient(Decimal("0"))
    with pytest.raises(tier0.Tier0Error) as raised:
        _acquire_amended_session_bytes(
            second,
            request,
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=state,
        )
    assert raised.value.status == "STOP_JOB55_ATTEMPT_PAIR_LIMIT"
    assert second.cost_kwargs == []


def test_job55_destination_initialization_creates_only_the_sibling_namespace(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "synthetic-volume"
    parent = mount / "cmbp-tier0"
    job51_root = parent / "job51"
    job51_root.mkdir(parents=True)
    sentinel = job51_root / "immutable-evidence.bin"
    sentinel.write_bytes(b"preserve these exact Job51/52 bytes")
    before = _tree_snapshot(job51_root)
    volume = _volume(mount)

    job55_root = amendment.initialize_job55_destination_tree(volume)

    assert job55_root == mount / "cmbp-tier0" / "job55"
    assert sorted(path.name for path in job55_root.iterdir()) == [
        "attempts",
        "receipts",
        "sessions",
    ]
    assert _tree_snapshot(job51_root) == before


def test_job55_destination_initialization_refuses_to_create_missing_parent(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "synthetic-volume"
    volume = _volume(mount)
    with pytest.raises(tier0.Tier0Error):
        amendment.initialize_job55_destination_tree(volume)
    assert not (mount / "cmbp-tier0").exists()


def test_job52_adoption_and_job55_lock_binding_preserve_job51_bytes_and_reconstruct(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)

    assert _tree_snapshot(context.job51_root) == context.job51_before
    assert context.adoption["opening_commitment_total_usd"] == "0.950392448902"
    assert context.adoption["opening_commitment_by_session_usd"] == {
        FAILED_SESSION: "0.950392448902"
    }
    assert context.adoption["prior_partial_reuse_permitted"] is False
    binding = tier0.strict_json(
        context.job55_root / amendment.JOB55_LOCK_BINDING_NAME
    )
    assert binding["shared_flock_relative"] == "cmbp-tier0/job51/RUN_LOCK_V1"
    assert binding["shared_flock_bytes_mutated"] is False

    changed_free = dataclasses.replace(
        context.volume,
        free_bytes=context.volume.free_bytes - 1,
    )
    reconstructed = amendment.ensure_job52_terminal_adoption(
        context.job55_root,
        context.opening,
        volume=changed_free,
    )
    assert reconstructed == context.adoption
    assert _tree_snapshot(context.job51_root) == context.job51_before


def test_self_rehashed_adoption_tamper_is_refused(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    path = context.job55_root / amendment.JOB55_ADOPTION_NAME
    adoption = tier0.strict_json(path)
    adoption["prior_partial_reuse_permitted"] = True
    adoption["adoption_sha256"] = tier0.self_hash(adoption, "adoption_sha256")
    path.write_bytes(tier0.canonical_json_bytes(adoption) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.ensure_job52_terminal_adoption(
            context.job55_root,
            context.opening,
            volume=context.volume,
        )
    assert raised.value.status == "STOP_JOB55_ADOPTION_DRIFT"
    assert _tree_snapshot(context.job51_root) == context.job51_before


def test_missing_adoption_or_lock_binding_cannot_be_recreated_after_job55_state(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    _job55_attempt(context)
    adoption_path = context.job55_root / amendment.JOB55_ADOPTION_NAME
    binding_path = context.job55_root / amendment.JOB55_LOCK_BINDING_NAME
    adoption_path.unlink()

    with pytest.raises(tier0.Tier0Error) as adoption_error:
        amendment.ensure_job52_terminal_adoption(
            context.job55_root,
            context.opening,
            volume=context.volume,
            allow_initialize=True,
        )
    assert adoption_error.value.status == "STOP_JOB55_ADOPTION_DRIFT"

    adoption_path.write_bytes(tier0.canonical_json_bytes(context.adoption) + b"\n")
    binding_path.unlink()
    with pytest.raises(tier0.Tier0Error) as binding_error:
        amendment.ensure_job55_lock_binding(
            context.job55_root,
            volume=context.volume,
            readiness_receipt_sha256=JOB55_READINESS_SHA256,
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
            allow_initialize=True,
        )
    assert binding_error.value.status == "STOP_JOB55_LOCK_BINDING"
    assert _tree_snapshot(context.job51_root) == context.job51_before


def test_job55_attempt_header_anchor_and_header_only_repair_bind_creation_order(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    _first_journal, _first_dir, first_header = _job55_attempt(context)
    first_anchor = _validate_job55_anchor(context, allow_initialize=True)
    assert first_anchor["job55_attempt_count"] == 1

    _second_journal, _second_dir, second_header = _job55_attempt(
        context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=first_header["record_hash"],
    )
    with pytest.raises(tier0.Tier0Error) as unanchored:
        _validate_job55_anchor(context)
    assert unanchored.value.status == "STOP_JOB55_ATTEMPT_SET"

    repaired = _validate_job55_anchor(context, repair_header_only=True)
    assert repaired["job55_attempt_count"] == 2
    assert [item["attempt_id"] for item in repaired["job55_attempts"]] == [
        ATTEMPT_ID,
        SECOND_ATTEMPT_ID,
    ]
    assert repaired["job55_attempts"][1]["previous_job55_attempt_header_record_hash"] == (
        first_header["record_hash"]
    )
    assert second_header["event"] == "ATTEMPT_START"
    assert _tree_snapshot(context.job51_root) == context.job51_before


def test_advanced_unanchored_attempt_cannot_use_header_repair(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    _first_journal, _first_dir, first_header = _job55_attempt(context)
    _validate_job55_anchor(context, allow_initialize=True)
    second_journal, _second_dir, _second_header = _job55_attempt(
        context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=first_header["record_hash"],
    )
    second_journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "DATABENTO_API_KEY",
            "sdk_identity_sha256": "4" * 64,
        },
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_job55_anchor(context, repair_header_only=True)
    assert raised.value.status == "STOP_JOB55_ATTEMPT_SET"


def test_job55_anchor_refuses_vanished_or_rolled_back_attempt(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    _journal_value, attempt_dir, _header = _job55_attempt(context)
    _validate_job55_anchor(context, allow_initialize=True)
    attempt_dir.rename(tmp_path / "recoverable-moved-attempt")

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_job55_anchor(context)
    assert raised.value.status == "STOP_JOB55_ATTEMPT_SET"


def test_job55_anchor_cannot_initialize_after_an_attempt_advanced(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    journal, _attempt_dir, _header = _job55_attempt(context)
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "DATABENTO_API_KEY",
            "sdk_identity_sha256": "4" * 64,
        },
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_job55_anchor(context, allow_initialize=True)
    assert raised.value.status == "STOP_JOB55_ATTEMPT_SET"


@pytest.mark.parametrize("child_kind", ("file", "symlink", "malformed_directory"))
def test_job55_anchor_refuses_unexpected_attempt_children(
    child_kind: str,
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    attempts = context.job55_root / "attempts"
    if child_kind == "file":
        (attempts / "unexpected").write_text("not an attempt", encoding="utf-8")
    elif child_kind == "symlink":
        (attempts / ATTEMPT_ID).symlink_to(context.job51_root, target_is_directory=True)
    else:
        (attempts / "not-a-uuid").mkdir()

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_job55_anchor(context, allow_initialize=True)
    assert raised.value.status == "STOP_JOB55_ATTEMPT_SET"


def test_public_opening_validator_has_no_test_pin_override(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_job52_tree(tmp_path)
    with pytest.raises(TypeError):
        amendment.validate_job52_opening_evidence(
            fixture.job51_root,
            volume=fixture.volume,
            pins=fixture.pins,  # type: ignore[call-arg]
        )


def test_acquire_requires_real_journal_exact_request_ordinal_and_evidence_hashes(
    tmp_path: Path,
) -> None:
    request = _failed_request()
    journal, attempt_dir = _journal(tmp_path)
    output = _staging_path(attempt_dir, request.session)

    class JournalProxy:
        def __init__(self, delegate: tier0.AttemptJournal) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

    with pytest.raises(tier0.Tier0Error) as proxy_error:
        _acquire_amended_session_bytes(
            _FakeClient(Decimal("0")),
            request,
            output_path=output,
            journal=JournalProxy(journal),
            pre_pair_gate=lambda: None,
            budget_state=_opening_state(),
            readiness_receipt_sha256=JOB55_READINESS_SHA256,
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            ordinal=_request_ordinal(request),
        )
    assert proxy_error.value.status == "STOP_JOB55_JOURNAL_INVALID"

    widened = dataclasses.replace(request, symbols=(*request.symbols, "OUTSIDE SCOPE"))
    with pytest.raises(tier0.Tier0Error) as scope_error:
        _acquire_amended_session_bytes(
            _FakeClient(Decimal("0")),
            widened,
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=_opening_state(),
            readiness_receipt_sha256=JOB55_READINESS_SHA256,
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            ordinal=_request_ordinal(request),
        )
    assert scope_error.value.status == "STOP_JOB55_SCOPE_WIDENING"

    with pytest.raises(tier0.Tier0Error) as identity_error:
        _acquire_amended_session_bytes(
            _FakeClient(Decimal("0")),
            request,
            output_path=output,
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=_opening_state(),
            readiness_receipt_sha256="UNSEALED",
            readiness_receipt_file_sha256=JOB55_READINESS_FILE_SHA256,
            ordinal=_request_ordinal(request),
        )
    assert identity_error.value.status == "STOP_JOB55_READINESS_DRIFT"
    assert tier0.verify_attempt_journal(attempt_dir)["records"][-1]["event"] == "ATTEMPT_START"


def test_job55_scanner_empty_state_is_exact_job52_opening_ledger(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    summary = _summarize_job55(context)

    assert summary["job55_attempt_count"] == 0
    assert summary["job55_cost_call_starts"] == 0
    assert summary["job55_timeseries_call_starts"] == 0
    assert summary["job55_commitment_count"] == 0
    assert summary["combined_commitment_count"] == 1
    assert summary["job55_committed_quote_total_usd"] == "0"
    assert summary["combined_committed_quote_total_usd"] == "0.950392448902"
    assert summary["combined_committed_quote_by_session_usd"] == {
        FAILED_SESSION: "0.950392448902"
    }
    assert summary["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    state = amendment.amended_budget_state_from_summary(summary)
    assert state.committed_total_usd == OPENING_COMMITMENT


def test_job55_scanner_reconstructs_successful_retry_and_recovery_barrier(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _client, state = _acquire_in_context(
        context,
        journal,
        attempt_dir,
        _failed_request(),
        quote=OPENING_COMMITMENT,
    )

    summary = _summarize_job55(context)
    assert summary["job55_attempt_count"] == 1
    assert summary["job55_cost_call_starts"] == 1
    assert summary["job55_cost_call_results"] == 1
    assert summary["job55_timeseries_call_starts"] == 1
    assert summary["job55_timeseries_call_results"] == 1
    assert summary["job55_commitment_count"] == 1
    assert summary["combined_commitment_count"] == 2
    assert summary["job55_committed_quote_total_usd"] == "0.950392448902"
    assert summary["combined_committed_quote_total_usd"] == "1.900784897804"
    assert summary["job55_cost_call_count_by_session"] == {FAILED_SESSION: 1}
    assert summary["job55_start_count_by_session"] == {FAILED_SESSION: 1}
    assert summary["successful_stream_recovery_required"] is True
    assert len(summary["recoverable_staging"]) == 1
    reconstructed = amendment.amended_budget_state_from_summary(summary)
    assert reconstructed.committed_total_usd == state.committed_total_usd
    assert reconstructed.job55_start_count_by_session == {FAILED_SESSION: 1}


def test_job55_scanner_requires_exact_client_identity_before_vendor_events(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _validate_job55_anchor(context, allow_initialize=True)
    sdk_sha = "4" * 64
    _append_client_constructed(journal, sdk_identity_sha256=sdk_sha)
    _acquire_in_context(
        context,
        journal,
        attempt_dir,
        _failed_request(),
        quote=OPENING_COMMITMENT,
    )
    assert _summarize_job55(
        context,
        require_client_constructed=True,
        expected_sdk_identity_sha256=sdk_sha,
    )["job55_cost_call_starts"] == 1

    with pytest.raises(tier0.Tier0Error):
        _summarize_job55(
            context,
            require_client_constructed=True,
            expected_sdk_identity_sha256="5" * 64,
        )


def test_job55_scanner_refuses_vendor_call_without_required_client_record(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _acquire_in_context(
        context,
        journal,
        attempt_dir,
        _failed_request(),
        quote=OPENING_COMMITMENT,
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(
            context,
            require_client_constructed=True,
            expected_sdk_identity_sha256="4" * 64,
        )
    assert raised.value.status == "STOP_JOB55_JOURNAL_INVALID"


def test_failed_session_cost_error_is_terminal_across_later_attempts(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    first_journal, first_dir, first_header = _job55_attempt(context)
    _validate_job55_anchor(context, allow_initialize=True)
    client = _FakeClient(RuntimeError("redacted synthetic cost failure"))
    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            client,
            _failed_request(),
            output_path=_staging_path(first_dir, FAILED_SESSION),
            journal=first_journal,
            pre_pair_gate=lambda: None,
            budget_state=_opening_state(),
            volume=context.volume,
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
        )
    first_only = _summarize_job55(context)
    assert first_only["terminal_authority_failure_observed"] is True
    assert first_only["terminal_authority_failure"]["status"] == (
        "STOP_JOB55_SINGULAR_RETRY_COST_ERROR"
    )

    _job55_attempt(
        context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=first_header["record_hash"],
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(context)
    assert raised.value.status == "STOP_JOB55_AUTHORITY_TERMINAL"


@pytest.mark.parametrize(
    "quote",
    (True, SESSION_HEADROOM + Decimal("0.000000000001")),
)
def test_malformed_or_above_cap_failed_session_quote_is_disk_terminal(
    quote: Any,
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _validate_job55_anchor(context, allow_initialize=True)
    with pytest.raises(tier0.Tier0Error):
        _acquire_amended_session_bytes(
            _FakeClient(quote),
            _failed_request(),
            output_path=_staging_path(attempt_dir, FAILED_SESSION),
            journal=journal,
            pre_pair_gate=lambda: None,
            budget_state=_opening_state(),
            volume=context.volume,
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
        )
    summary = _summarize_job55(context)
    assert summary["terminal_authority_failure_observed"] is True
    assert summary["terminal_authority_failure"]["status"] == (
        "STOP_JOB55_SINGULAR_RETRY_QUOTE_WITHOUT_START"
    )
    assert summary["quote_observations"][0]["time_series_start_permitted"] is False
    assert summary["job55_commitment_count"] == 0


def test_pending_failed_session_cost_start_is_terminal_before_any_later_attempt(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    request = _failed_request()
    journal, _attempt_dir, header = _job55_attempt(context)
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    summary = _summarize_job55(context)
    assert summary["terminal_authority_failure"]["status"] == (
        "STOP_JOB55_SINGULAR_RETRY_COST_INDETERMINATE"
    )

    _job55_attempt(
        context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=header["record_hash"],
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(context)
    assert raised.value.status == "STOP_JOB55_AUTHORITY_TERMINAL"


def test_job55_scanner_refuses_self_hashed_forged_commitment_arithmetic(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    request = _failed_request()
    journal, _attempt_dir, _header = _job55_attempt(context)
    state = _opening_state()
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    state.note_cost_call(request.session)
    payload = amendment._amended_cost_payload(state, request.session, "0.950392448902")
    payload["combined_committed_quote_total_projected_usd"] = "1.900784897805"
    journal.append(
        "COST_CALL_RESULT",
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload=payload,
    )
    assert tier0.verify_attempt_journal(journal.attempt_dir)["records"][-1]["event"] == (
        "COST_CALL_RESULT"
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(context)
    assert raised.value.status == "STOP_JOB55_JOURNAL_INVALID"


def test_job55_scanner_refuses_out_of_scope_call_even_when_structurally_hash_bound(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    request = _failed_request()
    journal, _attempt_dir, _header = _job55_attempt(context)
    widened = {**request.market_parameters, "dataset": "OUTSIDE.SCOPE"}
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request.market_request_sha256,
        payload={"method": "metadata.get_cost", "parameters": widened},
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(context)
    assert raised.value.status == "STOP_JOB55_SCOPE_WIDENING"


@pytest.mark.parametrize("marker_tamper", ("payload", "recorded_at_utc", "unexpected_field"))
def test_job55_scanner_refuses_self_rehashed_marker_semantic_drift(
    marker_tamper: str,
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _acquire_in_context(
        context,
        journal,
        attempt_dir,
        _failed_request(),
        quote=OPENING_COMMITMENT,
    )
    marker_path = next((attempt_dir / "call-markers").glob("*-timeseries_call_start.json"))
    marker = tier0.strict_json(marker_path)
    if marker_tamper == "payload":
        marker["payload"]["combined_committed_quote_total_after_usd"] = "1.900784897805"
    elif marker_tamper == "recorded_at_utc":
        marker["recorded_at_utc"] = "2099-01-01T00:00:00.000000Z"
    else:
        marker["unexpected"] = "self-rehashed widening"
    marker["marker_sha256"] = tier0.self_hash(marker, "marker_sha256")
    marker_path.write_bytes(tier0.canonical_json_bytes(marker) + b"\n")
    assert tier0.verify_attempt_journal(attempt_dir)["records"][-1]["event"] == (
        "TIMESERIES_CALL_RESULT"
    )
    with pytest.raises(tier0.Tier0Error):
        _summarize_job55(context)


def test_job55_successful_result_can_be_recovered_exactly_in_next_attempt(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    request = _failed_request()
    first_journal, first_dir, first_header = _job55_attempt(context)
    _acquire_in_context(
        context,
        first_journal,
        first_dir,
        request,
        quote=OPENING_COMMITMENT,
    )
    source = tier0.verify_attempt_journal(first_dir)["records"][-1]
    staged = first_dir / "sessions" / f"{request.session}.bundle.part"
    final = context.job55_root / "sessions" / request.session
    staged.rename(final)

    second_journal, _second_dir, _second_header = _job55_attempt(
        context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=first_header["record_hash"],
    )
    _append_publication(second_journal, request, source, mode="SESSION_RECOVERED")

    summary = _summarize_job55(context)
    assert summary["successful_stream_recovery_required"] is False
    assert summary["published_sessions_in_journals"] == [FAILED_SESSION]
    assert len(summary["publication_records"]) == 1
    assert summary["publication_records"][0]["event"] == "SESSION_RECOVERED"


def test_job55_result_bound_staging_byte_drift_is_terminal_recovery_failure(
    tmp_path: Path,
) -> None:
    context = _job55_control_context(tmp_path)
    journal, attempt_dir, _header = _job55_attempt(context)
    _acquire_in_context(
        context,
        journal,
        attempt_dir,
        _failed_request(),
        quote=OPENING_COMMITMENT,
    )
    data_path = (
        attempt_dir
        / "sessions"
        / f"{FAILED_SESSION}.bundle.part"
        / "data.cmbp-1.dbn.zst"
    )
    data_path.write_bytes(b"tampered after durable result")

    with pytest.raises(tier0.Tier0Error) as raised:
        _summarize_job55(context)
    assert raised.value.status == "STOP_JOB55_RECOVERY_REQUIRED"


def test_job55_terminal_stop_requires_exact_side_receipt(tmp_path: Path) -> None:
    context = _job55_control_context(tmp_path)
    journal, _attempt_dir, _header = _job55_attempt(context)
    terminal = journal.append(
        "ATTEMPT_STOP",
        payload={"status": "STOP_SYNTHETIC", "error_class": "SyntheticError"},
    )
    with pytest.raises(tier0.Tier0Error):
        _summarize_job55(context)

    amendment.write_job55_attempt_stop_receipt(
        journal.attempt_dir,
        terminal_record=terminal,
        volume=context.volume,
        readiness_receipt_sha256=JOB55_READINESS_SHA256,
        adoption_receipt_sha256=context.adoption["adoption_sha256"],
    )
    summary = _summarize_job55(context)
    assert summary["attempts"][0]["attempt_stop_file_sha256"] is not None


def test_job55_session_qc_v3_round_trips_exact_source_cost_decoder_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)

    rebuilt = _validate_job55_qc(fixture)

    assert rebuilt == fixture.qc
    assert rebuilt["artifact_type"] == amendment.SESSION_QC_ARTIFACT
    assert rebuilt["status"] == "JOB55_CMBP_TIER0_CAP_AMENDMENT_SESSION_QC_PASS"
    assert rebuilt["cost_commitment"] == {
        "fresh_observed_sdk_quote_usd": "0.950392448902",
        "job52_carried_commitment_usd": "0.950392448902",
        "combined_commitment_index": 2,
        "job55_commitment_index": 1,
        "combined_committed_quote_session_before_usd": "0.950392448902",
        "combined_committed_quote_session_after_usd": "1.900784897804",
        "combined_committed_quote_total_before_usd": "0.950392448902",
        "combined_committed_quote_total_after_usd": "1.900784897804",
        "job55_committed_quote_session_before_usd": "0",
        "job55_committed_quote_session_after_usd": "0.950392448902",
        "job55_committed_quote_total_before_usd": "0",
        "job55_committed_quote_total_after_usd": "0.950392448902",
        "job55_session_start_count_before": 0,
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "acquisition_initiated": True,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "quote_is_atomic_invoice_lock": False,
    }
    assert rebuilt["raw_dbn"] == {
        "file_name": "data.cmbp-1.dbn.zst",
        "file_sha256": tier0.file_sha256(fixture.data_path),
        "compressed_bytes": fixture.data_path.stat().st_size,
    }
    assert rebuilt["dbn_metadata"] == fixture.metadata
    assert rebuilt["decoder"] == fixture.decoder
    assert rebuilt["decoder"]["cmbp1_records"] == fixture.request.expected_record_count
    assert rebuilt["decoder"]["global_receive_regressions"] == 0
    assert rebuilt["decoder"]["all_causal_priors_strict"] is True
    assert rebuilt["decoder"]["explicit_connection_telemetry"] == "UNKNOWN"
    assert rebuilt["source_attempt"] == {
        "attempt_id": ATTEMPT_ID,
        "timeseries_start_sequence": fixture.start_record["sequence"],
        "timeseries_start_record_hash": fixture.start_record["record_hash"],
        "timeseries_result_sequence": fixture.result_record["sequence"],
        "timeseries_result_record_hash": fixture.result_record["record_hash"],
    }
    assert amendment.build_job55_publication_payload(
        request=fixture.request,
        qc=rebuilt,
        mode="SESSION_PUBLISHED",
        ordinal=1,
    ) == {
        "ordinal": 1,
        "publication_mode": "SESSION_PUBLISHED",
        "session_qc_sha256": rebuilt["session_qc_sha256"],
        "decoded_records": fixture.request.expected_record_count,
        "compressed_bytes": fixture.data_path.stat().st_size,
        "dbn_file_sha256": tier0.file_sha256(fixture.data_path),
        "source_attempt_id": ATTEMPT_ID,
        "source_timeseries_result_sequence": fixture.result_record["sequence"],
        "source_timeseries_result_record_hash": fixture.result_record["record_hash"],
    }


@pytest.mark.parametrize(
    "tamper",
    (
        "unexpected_top_level",
        "request",
        "quote",
        "combined_total",
        "raw_hash",
        "metadata",
        "source_hash",
        "decoder_records",
        "decoder_causal",
        "decoder_receive_regression",
        "decoder_silence",
        "qc_law",
    ),
)
def test_job55_session_qc_refuses_self_rehashed_semantic_tamper(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    qc = copy.deepcopy(fixture.qc)
    if tamper == "unexpected_top_level":
        qc["unexpected"] = "scope widening"
    elif tamper == "request":
        qc["request"]["dataset"] = "OUTSIDE.SCOPE"
    elif tamper == "quote":
        qc["cost_commitment"]["fresh_observed_sdk_quote_usd"] = "0.950392448903"
    elif tamper == "combined_total":
        qc["cost_commitment"]["combined_committed_quote_total_after_usd"] = (
            "1.900784897805"
        )
    elif tamper == "raw_hash":
        qc["raw_dbn"]["file_sha256"] = "a" * 64
    elif tamper == "metadata":
        qc["dbn_metadata"]["dataset"] = "OUTSIDE.SCOPE"
    elif tamper == "source_hash":
        qc["source_attempt"]["timeseries_result_record_hash"] = "a" * 64
    elif tamper == "decoder_records":
        qc["decoder"]["cmbp1_records"] -= 1
    elif tamper == "decoder_causal":
        qc["decoder"]["all_causal_priors_strict"] = False
    elif tamper == "decoder_receive_regression":
        qc["decoder"]["global_receive_regressions"] = 1
    elif tamper == "decoder_silence":
        qc["decoder"]["max_stream_silence_ns"] = 1
    else:
        qc["qc_law"]["strict_prior_inequality"] = "prior.ts_recv <= trade.ts_recv"
    qc["session_qc_sha256"] = tier0.self_hash(qc, "session_qc_sha256")
    fixture.qc_path.write_bytes(tier0.canonical_json_bytes(qc) + b"\n")

    with pytest.raises(tier0.Tier0Error):
        _validate_job55_qc(fixture)


@pytest.mark.parametrize("drift", ("version", "schema", "symbols", "mapping"))
def test_job55_session_qc_reopens_dbn_and_refuses_runtime_header_or_mapping_drift(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    import databento

    class DriftedFactory:
        @staticmethod
        def from_file(_path: Path) -> _FakeDBNStore:
            store = _FakeDBNStore(fixture.request)
            if drift == "version":
                store.metadata.version = 4
            elif drift == "schema":
                store.schema = "trades"
            elif drift == "symbols":
                store.symbols = [*store.symbols, "OUTSIDE SCOPE"]
            else:
                symbol = fixture.request.symbols[0]
                store.mappings[symbol][0]["symbol"] = "999999999"
            return store

    monkeypatch.setattr(databento, "DBNStore", DriftedFactory)
    with pytest.raises(tier0.Tier0Error):
        _validate_job55_qc(fixture)


@pytest.mark.parametrize("tamper", ("missing_start", "start_hash", "result_hash"))
def test_job55_session_qc_requires_exact_source_records_from_all_attempts(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    source_lookup = dict(fixture.source_lookup)
    start_key = (ATTEMPT_ID, fixture.start_record["sequence"])
    result_key = (ATTEMPT_ID, fixture.result_record["sequence"])
    if tamper == "missing_start":
        del source_lookup[start_key]
    elif tamper == "start_hash":
        source_lookup[start_key] = {
            **source_lookup[start_key],
            "record_hash": "a" * 64,
        }
    else:
        source_lookup[result_key] = {
            **source_lookup[result_key],
            "record_hash": "a" * 64,
        }
    fixture.source_lookup = source_lookup

    with pytest.raises(tier0.Tier0Error) as raised:
        _validate_job55_qc(fixture)
    assert raised.value.status == "STOP_JOB55_SESSION_QC"


def test_job55_published_qc_population_reconciles_journal_and_preserves_job51(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    final_dir, publication = _publish_job55_qc(fixture)
    summary = _summarize_job55(fixture.context)
    source_lookup = amendment.job55_source_record_lookup(summary)

    population = amendment.validate_job55_existing_session_population(
        fixture.context.job55_root,
        bundle=amendment.load_cap_amendment_scope_bundle(REPO),
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=fixture.context.volume,
        source_record_lookup=source_lookup,
        published_sessions_in_journals=summary["published_sessions_in_journals"],
    )

    assert population == {fixture.request.session: fixture.qc}
    assert publication["payload"] == amendment.build_job55_publication_payload(
        request=fixture.request,
        qc=fixture.qc,
        mode="SESSION_PUBLISHED",
        ordinal=1,
    )
    assert final_dir == fixture.context.job55_root / "sessions" / FAILED_SESSION
    assert _tree_snapshot(fixture.context.job51_root) == fixture.context.job51_before


def test_job55_published_session_disappearance_is_refused_by_population_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    final_dir, _publication = _publish_job55_qc(fixture)
    summary = _summarize_job55(fixture.context)
    moved = tmp_path / "reversibly-moved-published-bundle"
    final_dir.rename(moved)

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.validate_job55_existing_session_population(
            fixture.context.job55_root,
            bundle=amendment.load_cap_amendment_scope_bundle(REPO),
            readiness=fixture.readiness,
            readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption=fixture.context.adoption,
            volume=fixture.context.volume,
            source_record_lookup=amendment.job55_source_record_lookup(summary),
            published_sessions_in_journals=summary["published_sessions_in_journals"],
        )
    assert raised.value.status == "STOP_JOB55_SESSION_QC"


def test_job55_qc_bundle_can_be_source_bound_to_exact_next_attempt_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_qc_fixture(tmp_path, monkeypatch)
    final_dir = fixture.context.job55_root / "sessions" / fixture.request.session
    fixture.bundle_dir.rename(final_dir)
    first_header = tier0.verify_attempt_journal(fixture.attempt_dir)["records"][0]
    second_journal, _second_dir, _second_header = _job55_attempt(
        fixture.context,
        attempt_id=SECOND_ATTEMPT_ID,
        ordinal=2,
        previous_attempt_id=ATTEMPT_ID,
        previous_header_hash=first_header["record_hash"],
    )
    payload = amendment.build_job55_publication_payload(
        request=fixture.request,
        qc=fixture.qc,
        mode="SESSION_RECOVERED",
        ordinal=1,
    )
    second_journal.append(
        "SESSION_RECOVERED",
        session=fixture.request.session,
        request_sha256=fixture.request.market_request_sha256,
        payload=payload,
    )

    summary = _summarize_job55(fixture.context)
    assert summary["successful_stream_recovery_required"] is False
    population = amendment.validate_job55_existing_session_population(
        fixture.context.job55_root,
        bundle=amendment.load_cap_amendment_scope_bundle(REPO),
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=fixture.context.volume,
        source_record_lookup=amendment.job55_source_record_lookup(summary),
        published_sessions_in_journals=summary["published_sessions_in_journals"],
    )
    assert population == {FAILED_SESSION: fixture.qc}


@pytest.mark.parametrize("bad_sequence", (True, -1))
def test_job55_source_record_lookup_refuses_bool_or_negative_sequence(
    bad_sequence: Any,
) -> None:
    record = {
        "attempt_id": ATTEMPT_ID,
        "sequence": bad_sequence,
        "event": "TIMESERIES_CALL_START",
    }
    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.job55_source_record_lookup(
            {"start_records": [record], "result_records": []}
        )
    assert raised.value.status == "STOP_JOB55_JOURNAL_INVALID"


def test_job55_source_record_lookup_refuses_duplicate_attempt_sequence() -> None:
    record = {
        "attempt_id": ATTEMPT_ID,
        "sequence": 5,
        "event": "TIMESERIES_CALL_START",
    }
    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.job55_source_record_lookup(
            {"start_records": [record], "result_records": [dict(record)]}
        )
    assert raised.value.status == "STOP_JOB55_JOURNAL_INVALID"


def test_job55_full_aggregate_round_trips_exact_21_scope_and_32_commitment_boundary(
    tmp_path: Path,
) -> None:
    fixture = _full_job55_aggregate_fixture(tmp_path)

    seal = amendment.validate_job55_aggregate_seal(
        fixture.seal_path,
        job55_root=fixture.context.job55_root,
        summary=fixture.summary,
        final_qc=fixture.final_qc,
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=fixture.context.volume,
    )
    receipt = amendment.validate_job55_aggregate_receipt(
        fixture.receipt_path,
        bundle=fixture.bundle,
        opening=fixture.context.opening,
        summary=fixture.summary,
        final_qc=fixture.final_qc,
        seal=seal,
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=fixture.context.volume,
    )

    assert receipt == fixture.receipt
    assert receipt["status"] == (
        "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_PASS"
    )
    assert receipt["scope_reconciliation"] == {
        "dataset": tier0.EXPECTED_DATASET,
        "schema": tier0.EXPECTED_SCHEMA,
        "stype_in": tier0.EXPECTED_STYPE_IN,
        "stype_out": tier0.EXPECTED_STYPE_OUT,
        "session_count": 21,
        "decoded_records": 2_373_877_845,
        "expected_records": 2_373_877_845,
        "session_symbol_memberships": tier0.EXPECTED_SESSION_SYMBOLS,
        "expected_session_symbol_memberships": tier0.EXPECTED_SESSION_SYMBOLS,
        "out_of_scope_calls": 0,
    }
    assert receipt["call_accounting"]["job55_cost_call_starts"] == 21
    assert receipt["call_accounting"]["job55_timeseries_call_starts"] == 21
    assert receipt["call_accounting"]["job55_timeseries_call_results"] == 21
    assert receipt["commitment_reconciliation"] == {
        "job52_opening_commitment_count": 1,
        "job52_opening_total_usd": "0.950392448902",
        "job55_commitment_count": 21,
        "job55_total_usd": "31.049607551098",
        "combined_commitment_count": 22,
        "combined_total_usd": "32.000000000000",
        "combined_by_session_usd": fixture.summary[
            "combined_committed_quote_by_session_usd"
        ],
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "all_caps_pass": True,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    assert len(receipt["final_sessions"]) == 21
    assert len(receipt["job55_commitments"]) == 21
    assert receipt["decoder_aggregate"]["cmbp1_records"] == 2_373_877_845
    assert receipt["decoder_aggregate"]["global_receive_regressions"] == 0
    assert receipt["qc_law"]["all_causal_priors_strict"] is True
    assert receipt["qc_law"]["actual_vendor_invoice_cost_usd"] == "UNKNOWN"
    assert _tree_snapshot(fixture.context.job51_root) == fixture.context.job51_before


def test_job55_aggregate_receipt_allows_dynamic_free_space_change_but_not_identity_drift(
    tmp_path: Path,
) -> None:
    fixture = _full_job55_aggregate_fixture(tmp_path)
    changed_free = dataclasses.replace(
        fixture.context.volume,
        free_bytes=fixture.context.volume.free_bytes - 123_456,
    )
    assert amendment.validate_job55_aggregate_receipt(
        fixture.receipt_path,
        bundle=fixture.bundle,
        opening=fixture.context.opening,
        summary=fixture.summary,
        final_qc=fixture.final_qc,
        seal=fixture.seal,
        readiness=fixture.readiness,
        readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
        adoption=fixture.context.adoption,
        volume=changed_free,
    ) == fixture.receipt

    changed_identity = dataclasses.replace(
        fixture.context.volume,
        device_identifier="different-synthetic-device",
    )
    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.validate_job55_aggregate_receipt(
            fixture.receipt_path,
            bundle=fixture.bundle,
            opening=fixture.context.opening,
            summary=fixture.summary,
            final_qc=fixture.final_qc,
            seal=fixture.seal,
            readiness=fixture.readiness,
            readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption=fixture.context.adoption,
            volume=changed_identity,
        )
    assert raised.value.status == "STOP_JOB55_AGGREGATE_QC"


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_session",
        "terminal",
        "recovery",
        "missing_result",
        "total_cap",
        "session_cap",
        "record_count",
    ),
)
def test_job55_aggregate_builder_refuses_incomplete_or_over_cap_reconciliation(
    tamper: str,
    tmp_path: Path,
) -> None:
    fixture = _full_job55_aggregate_fixture(tmp_path)
    summary = copy.deepcopy(fixture.summary)
    final_qc = copy.deepcopy(fixture.final_qc)
    if tamper == "missing_session":
        final_qc.pop(next(iter(final_qc)))
    elif tamper == "terminal":
        summary["terminal_authority_failure_observed"] = True
    elif tamper == "recovery":
        summary["successful_stream_recovery_required"] = True
    elif tamper == "missing_result":
        summary["job55_timeseries_call_results"] = 20
    elif tamper == "total_cap":
        summary["combined_committed_quote_total_usd"] = "32.000000000001"
    elif tamper == "session_cap":
        summary["combined_committed_quote_by_session_usd"][FAILED_SESSION] = (
            "2.000000000001"
        )
    else:
        first = next(iter(final_qc.values()))
        first["decoder"]["cmbp1_records"] -= 1

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.build_job55_aggregate_receipt(
            bundle=fixture.bundle,
            opening=fixture.context.opening,
            summary=summary,
            final_qc=final_qc,
            seal=fixture.seal,
            readiness=fixture.readiness,
            readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption=fixture.context.adoption,
            volume=fixture.context.volume,
        )
    assert raised.value.status == "STOP_JOB55_AGGREGATE_QC"


@pytest.mark.parametrize(
    "tamper",
    ("qc_hash", "raw_hash", "source_attempt", "source_sequence", "source_hash"),
)
def test_job55_aggregate_seal_refuses_publication_final_source_swap(
    tamper: str,
    tmp_path: Path,
) -> None:
    fixture = _full_job55_aggregate_fixture(tmp_path)
    summary = copy.deepcopy(fixture.summary)
    payload = summary["publication_records"][0]["payload"]
    if tamper == "qc_hash":
        payload["session_qc_sha256"] = "a" * 64
    elif tamper == "raw_hash":
        payload["dbn_file_sha256"] = "a" * 64
    elif tamper == "source_attempt":
        payload["source_attempt_id"] = SECOND_ATTEMPT_ID
    elif tamper == "source_sequence":
        payload["source_timeseries_result_sequence"] += 1
    else:
        payload["source_timeseries_result_record_hash"] = "a" * 64

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.build_job55_aggregate_seal(
            job55_root=fixture.context.job55_root,
            summary=summary,
            final_qc=fixture.final_qc,
            readiness=fixture.readiness,
            readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption=fixture.context.adoption,
            volume=fixture.context.volume,
        )
    assert raised.value.status == "STOP_JOB55_AGGREGATE_QC"


@pytest.mark.parametrize(
    "tamper",
    ("invoice", "combined_total", "decoded_records", "source_hash", "free_bytes_type"),
)
def test_job55_aggregate_receipt_refuses_self_rehashed_semantic_tamper(
    tamper: str,
    tmp_path: Path,
) -> None:
    fixture = _full_job55_aggregate_fixture(tmp_path)
    receipt = copy.deepcopy(fixture.receipt)
    if tamper == "invoice":
        receipt["commitment_reconciliation"]["actual_vendor_invoice_cost_usd"] = "31.99"
    elif tamper == "combined_total":
        receipt["commitment_reconciliation"]["combined_total_usd"] = "31.99"
    elif tamper == "decoded_records":
        receipt["scope_reconciliation"]["decoded_records"] -= 1
    elif tamper == "source_hash":
        receipt["final_sessions"][0]["source_attempt"][
            "timeseries_result_record_hash"
        ] = "a" * 64
    else:
        receipt["volume_snapshot"]["free_bytes"] = "1000000000000"
    receipt["receipt_sha256"] = tier0.self_hash(receipt, "receipt_sha256")
    fixture.receipt_path.write_bytes(tier0.canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.validate_job55_aggregate_receipt(
            fixture.receipt_path,
            bundle=fixture.bundle,
            opening=fixture.context.opening,
            summary=fixture.summary,
            final_qc=fixture.final_qc,
            seal=fixture.seal,
            readiness=fixture.readiness,
            readiness_file_sha256=JOB55_READINESS_FILE_SHA256,
            adoption=fixture.context.adoption,
            volume=fixture.context.volume,
        )
    assert raised.value.status == "STOP_JOB55_AGGREGATE_QC"


def test_job55_readiness_round_trips_exact_authority_money_boundary_and_sterile_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_readiness_fixture(tmp_path, monkeypatch)

    rebuilt = amendment.validate_job55_readiness_receipt(
        fixture.repo,
        fixture.paths["readiness"],
    )

    assert rebuilt == fixture.receipt
    assert rebuilt["status"] == amendment.READINESS_STATUS
    assert rebuilt["status_meaning"] == (
        "LOCAL_BUILD_AND_TEST_SEAL_ONLY; NO_VENDOR_OR_ACQUISITION_RESULT"
    )
    assert rebuilt["money_boundary"] == {
        "job52_opening_commitment_usd": "0.950392448902",
        "per_session_lifetime_committed_quote_cap_usd": "2.00",
        "total_committed_quote_cap_usd": "32.00",
        "failed_session_job55_cost_call_allowance": 1,
        "failed_session_job55_time_series_start_allowance": 1,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    assert rebuilt["integrity"] == {
        "credential_read": False,
        "authenticated_client_constructed": False,
        "external_calls": 0,
        "metadata_calls": 0,
        "timeseries_calls": 0,
        "data_downloaded": False,
        "acquisition_initiated": False,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "models_fit": 0,
        "broker_calls": 0,
        "orders_submitted": 0,
    }
    assert rebuilt["focused_test_report"]["tests"] == 12
    assert rebuilt["focused_test_report"]["failures"] == 0
    assert rebuilt["focused_test_report"]["errors"] == 0
    assert rebuilt["focused_test_report"]["skipped"] == 0
    assert rebuilt["focused_test_report"]["pytest_plugin_autoload_disabled"] is True
    assert rebuilt["focused_test_report"][
        "repository_and_environment_pytest_addopts_ignored"
    ] is True
    assert rebuilt["sdk_identity"] == fixture.sdk_identity
    assert set(rebuilt["bound_files"]) == set(
        fixture.bundle.contract["required_bound_files"]
    )


@pytest.mark.parametrize(
    "tamper",
    ("money_cap", "invoice", "bound_file", "sdk", "test_identity", "external_call"),
)
def test_job55_readiness_refuses_self_rehashed_semantic_drift(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_readiness_fixture(tmp_path, monkeypatch)
    receipt = copy.deepcopy(fixture.receipt)
    if tamper == "money_cap":
        receipt["money_boundary"]["per_session_lifetime_committed_quote_cap_usd"] = (
            "2.01"
        )
    elif tamper == "invoice":
        receipt["money_boundary"]["actual_vendor_invoice_cost_usd"] = "1.00"
    elif tamper == "bound_file":
        first = next(iter(receipt["bound_files"]))
        receipt["bound_files"][first] = "b" * 64
    elif tamper == "sdk":
        receipt["sdk_identity"]["compression"] = {"zstandard": "drifted"}
    elif tamper == "test_identity":
        receipt["focused_test_report"]["test_case_identity_sha256"] = "b" * 64
    else:
        receipt["integrity"]["external_calls"] = 1
    receipt["receipt_sha256"] = tier0.self_hash(receipt, "receipt_sha256")
    fixture.paths["readiness"].write_bytes(
        tier0.canonical_json_bytes(receipt) + b"\n"
    )

    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.validate_job55_readiness_receipt(
            fixture.repo,
            fixture.paths["readiness"],
        )
    assert raised.value.status == "STOP_JOB55_READINESS"


@pytest.mark.parametrize(
    "junit",
    (
        '<testsuite tests="12" failures="1" errors="0" skipped="0" />',
        '<testsuite tests="12" failures="0" errors="0" skipped="1" />',
        '<testsuite tests="12" failures="0" errors="0" skipped="0">'
        '<testcase classname="hostile.plugin" name="test_injected" />'
        "</testsuite>",
    ),
)
def test_job55_readiness_junit_parser_refuses_failure_skip_or_injected_population(
    junit: str,
    tmp_path: Path,
) -> None:
    report = tmp_path / "synthetic-junit.xml"
    report.write_text(junit + "\n", encoding="utf-8")
    with pytest.raises(tier0.Tier0Error) as raised:
        amendment._job55_junit_counts(report, enforce_frozen_population=False)
    assert raised.value.status == "STOP_JOB55_LOCAL_TESTS"


def test_job55_readiness_path_must_be_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _job55_readiness_fixture(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate-readiness.json"
    alternate.write_bytes(fixture.paths["readiness"].read_bytes())
    with pytest.raises(tier0.Tier0Error) as raised:
        amendment.validate_job55_readiness_receipt(fixture.repo, alternate)
    assert raised.value.status == "STOP_JOB55_READINESS"


@pytest.mark.parametrize(
    "module_name, expected",
    (
        (
            "acquire_cmbp_tier0_cap_amendment",
            "STOP_UNEXPECTED_ARGUMENTS: this Job-55 runner accepts no arguments",
        ),
        (
            "seal_cmbp_tier0_cap_amendment_readiness",
            "STOP_UNEXPECTED_ARGUMENTS: this Job-55 sealer accepts no arguments",
        ),
    ),
)
def test_job55_ops_refuse_argv_before_any_v5_import(
    module_name: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if module_name == "acquire_cmbp_tier0_cap_amendment":
        from v5.ops import acquire_cmbp_tier0_cap_amendment as operation
    else:
        from v5.ops import seal_cmbp_tier0_cap_amendment_readiness as operation

    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("v5"):
            imported.append(name)
            raise AssertionError("argv refusal attempted a protected import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", [str(Path(operation.__file__)), "unexpected"])
    assert operation.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == expected
    assert imported == []


def test_job55_runner_21_one_pair_processes_then_local_aggregate_without_post_durable_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_job55_runner_context(tmp_path, monkeypatch)

    for ordinal, request in enumerate(context.bundle.sessions, start=1):
        assert context.runner.main() == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.strip().splitlines()[-1] == (
            f"PUBLISHED {ordinal:02d}/21 {request.session}; rerun the bare Job-55 command"
        )
        assert len(list((context.job55_root / "attempts").iterdir())) == ordinal
        assert len(list((context.job55_root / "sessions").iterdir())) == ordinal

    assert [name for name, _value in context.actions] == [
        event
        for _request in context.bundle.sessions
        for event in (
            "credential",
            "client",
            "metadata.get_cost",
            "timeseries.get_range",
        )
    ]
    assert [session for name, session in context.actions if name == "metadata.get_cost"] == [
        request.session for request in context.bundle.sessions
    ]
    assert [session for name, session in context.actions if name == "timeseries.get_range"] == [
        request.session for request in context.bundle.sessions
    ]
    for attempt_dir in (context.job55_root / "attempts").iterdir():
        events = [
            record["event"]
            for record in tier0.verify_attempt_journal(attempt_dir)["records"]
        ]
        assert events == [
            "ATTEMPT_START",
            "CLIENT_CONSTRUCTED",
            "COST_CALL_START",
            "COST_CALL_RESULT",
            "TIMESERIES_CALL_START",
            "TIMESERIES_CALL_RESULT",
            "SESSION_PUBLISHED",
        ]

    attempts_before_aggregate = _tree_snapshot(context.job55_root / "attempts")
    stop_count = len(context.stop_calls)

    def fail_after_receipt(path: Path, **_kwargs: Any) -> dict[str, Any]:
        assert Path(path).is_file()
        raise OSError("synthetic crash after durable aggregate receipt")

    monkeypatch.setattr(amendment, "validate_job55_aggregate_receipt", fail_after_receipt)
    assert context.runner.main() == 2
    captured = capsys.readouterr()
    assert "STOP_JOB55_UNEXPECTED_LOCAL_ERROR" in captured.err
    seal_path = context.job55_root / amendment.JOB55_AGGREGATE_SEAL_NAME
    receipt_path = context.job55_root / "receipts" / amendment.JOB55_AGGREGATE_NAME
    assert seal_path.is_file()
    assert receipt_path.is_file()
    assert _tree_snapshot(context.job55_root / "attempts") == attempts_before_aggregate
    assert len(context.stop_calls) == stop_count

    monkeypatch.setattr(
        amendment,
        "validate_job55_aggregate_receipt",
        context.validate_receipt,
    )
    actions_before_local_completion = list(context.actions)
    assert context.runner.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip().startswith("COMPLETE ")
    assert context.actions == actions_before_local_completion
    assert _tree_snapshot(context.job55_root / "attempts") == attempts_before_aggregate
    assert len(context.stop_calls) == stop_count
    assert all(
        root == context.job51_root and root != context.job55_root
        for _mode, root in context.lock_trace
    )
    assert [mode for mode, _root in context.lock_trace] == [
        mode for _invocation in range(23) for mode in ("enter", "exit")
    ]
    assert any(
        call.get("allow_initialize") is True
        and call.get("repair_header_only") is True
        for call in context.anchor_calls
    )


def test_job55_runner_repairs_exact_stranded_header_before_credential_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_job55_runner_context(tmp_path, monkeypatch)
    attempt_dir, journal = tier0.create_attempt(
        context.job55_root,
        attempt_id=ATTEMPT_ID,
        volume=context.volume,
    )
    journal.append(
        "ATTEMPT_START",
        payload=amendment.build_job55_attempt_header_payload(
            attempt_id=ATTEMPT_ID,
            volume=context.volume,
            readiness_receipt_sha256=context.readiness["receipt_sha256"],
            readiness_receipt_file_sha256=tier0.file_sha256(context.readiness_path),
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
            job55_attempt_ordinal=1,
            previous_job55_attempt_id=None,
            previous_job55_attempt_header_record_hash=None,
        ),
    )
    before = _tree_snapshot(attempt_dir)

    assert context.runner.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip().splitlines()[-1].startswith("PUBLISHED 01/21 ")
    assert _tree_snapshot(attempt_dir) == before
    attempts = sorted((context.job55_root / "attempts").iterdir())
    assert len(attempts) == 2
    assert [
        record["event"] for record in tier0.verify_attempt_journal(attempt_dir)["records"]
    ] == ["ATTEMPT_START"]
    repaired_calls = [
        call
        for call in context.anchor_calls
        if call.get("allow_initialize") is True
        and call.get("repair_header_only") is True
    ]
    assert repaired_calls
    assert [name for name, _value in context.actions[:2]] == ["credential", "client"]


def test_job55_runner_refuses_advanced_unanchored_attempt_before_credential_or_vendor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_job55_runner_context(tmp_path, monkeypatch)
    attempt_dir, journal = tier0.create_attempt(
        context.job55_root,
        attempt_id=ATTEMPT_ID,
        volume=context.volume,
    )
    journal.append(
        "ATTEMPT_START",
        payload=amendment.build_job55_attempt_header_payload(
            attempt_id=ATTEMPT_ID,
            volume=context.volume,
            readiness_receipt_sha256=context.readiness["receipt_sha256"],
            readiness_receipt_file_sha256=tier0.file_sha256(context.readiness_path),
            adoption_receipt_sha256=context.adoption["adoption_sha256"],
            job55_attempt_ordinal=1,
            previous_job55_attempt_id=None,
            previous_job55_attempt_header_record_hash=None,
        ),
    )
    journal.append(
        "CLIENT_CONSTRUCTED",
        payload={
            "credential_source": "synthetic_fixture",
            "sdk_identity_sha256": "4" * 64,
        },
    )
    before = _tree_snapshot(attempt_dir)

    assert context.runner.main() == 2
    captured = capsys.readouterr()
    assert "STOP_JOB55_ATTEMPT_SET" in captured.err
    assert context.actions == []
    assert context.stop_calls == []
    assert _tree_snapshot(attempt_dir) == before
    assert len(list((context.job55_root / "attempts").iterdir())) == 1


def test_job55_runner_refuses_premature_aggregate_seal_before_attempt_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_job55_runner_context(tmp_path, monkeypatch)
    seal_path = context.job55_root / amendment.JOB55_AGGREGATE_SEAL_NAME
    seal_bytes = b'{"artifact_type":"premature synthetic Job55 aggregate seal"}\n'
    seal_path.write_bytes(seal_bytes)
    attempts_before = _tree_snapshot(context.job55_root / "attempts")

    assert context.runner.main() == 2
    captured = capsys.readouterr()
    assert "STOP_JOB55_AGGREGATE_QC" in captured.err
    assert captured.out == ""
    assert context.actions == []
    assert context.stop_calls == []
    assert _tree_snapshot(context.job55_root / "attempts") == attempts_before
    assert seal_path.read_bytes() == seal_bytes


@pytest.mark.parametrize(
    "hostile_key",
    ("HTTPS_PROXY", "https_proxy", "SSLKEYLOGFILE", "NETRC"),
)
def test_job55_runner_hostile_transport_environment_stops_before_secret_or_client(
    hostile_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _patch_job55_runner_context(
        tmp_path,
        monkeypatch,
        bypass_environment_gate=False,
    )
    for key in list(os.environ):
        if key.upper() in context.runner.FORBIDDEN_REQUEST_ENVIRONMENT:
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(hostile_key, "synthetic-hostile-value")

    assert context.runner.main() == 2
    captured = capsys.readouterr()
    assert "STOP_JOB55_UNEXPECTED_LOCAL_ERROR" in captured.err
    assert context.actions == []
    attempts = list((context.job55_root / "attempts").iterdir())
    assert len(attempts) == 1
    assert [
        record["event"] for record in tier0.verify_attempt_journal(attempts[0])["records"]
    ] == ["ATTEMPT_START", "ATTEMPT_STOP"]
    assert context.stop_calls == [attempts[0]]


@pytest.mark.parametrize("name", (".netrc", "_netrc"))
def test_job55_runner_default_netrc_is_rejected_even_without_environment_override(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v5.ops import acquire_cmbp_tier0_cap_amendment as runner

    for key in list(os.environ):
        if key.upper() in runner.FORBIDDEN_REQUEST_ENVIRONMENT:
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home-with-netrc"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / name).write_bytes(b"synthetic netrc must never be read\n")
    with pytest.raises(RuntimeError):
        runner._reject_request_environment_overrides()


def test_job55_sealer_runs_exact_sterile_command_and_atomically_publishes_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_cap_amendment_readiness as sealer

    repo = tmp_path / "synthetic-job55-sealer-repo"
    work = repo / "v5/work/cmbp-tier0-cap-amendment"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    junit_bytes = b"<testsuites tests='158' failures='0' errors='0' skipped='0'/>\n"
    readiness_receipt = {"receipt_sha256": "e" * 64}
    run_calls: list[dict[str, Any]] = []
    seal_calls: list[Path] = []
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])
    monkeypatch.setattr(
        amendment,
        "_paths",
        lambda _root: {"test_report": report, "readiness": readiness},
    )
    for key, value in {
        "DATABENTO_API_KEY": "must-not-reach-child",
        "PYTEST_ADDOPTS": "--collect-only",
        "PYTEST_PLUGINS": "hostile_plugin",
        "HTTPS_PROXY": "https://hostile.invalid",
        "SSLKEYLOGFILE": str(tmp_path / "tls.keys"),
        "PYTHONPATH": str(tmp_path / "hostile-pythonpath"),
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

    def seal(actual_repo: Path) -> Path:
        seal_calls.append(Path(actual_repo))
        assert report.read_bytes() == junit_bytes
        assert report.stat().st_nlink == 1
        assert not temporary.exists()
        readiness.write_bytes(tier0.canonical_json_bytes(readiness_receipt) + b"\n")
        return readiness

    monkeypatch.setattr(sealer.subprocess, "run", fake_run)
    monkeypatch.setattr(tier0, "fsync_directory", lambda _path: None)
    monkeypatch.setattr(amendment, "seal_job55_readiness", seal)
    monkeypatch.setattr(
        amendment,
        "validate_job55_readiness_receipt",
        lambda _root, path=None: readiness_receipt,
    )

    assert sealer.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == f"SEALED {'e' * 64} {readiness}"
    assert seal_calls == [repo]
    assert report.read_bytes() == junit_bytes
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
                "v5/tests/test_cmbp_stream.py",
                "v5/tests/test_cmbp_tier0.py",
                "v5/tests/test_cmbp_tier0_paid.py",
                "v5/tests/test_cmbp_tier0_cap_amendment.py",
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


def test_job55_sealer_recovers_identical_stranded_junit_with_fresh_sterile_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_cap_amendment_readiness as sealer

    repo = tmp_path / "synthetic-job55-sealer-recovery-repo"
    work = repo / "v5/work/cmbp-tier0-cap-amendment"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    junit_bytes = b"<synthetic exact frozen Job55 junit/>\n"
    report.write_bytes(junit_bytes)
    report_before = report.read_bytes()
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])
    monkeypatch.setattr(
        amendment,
        "_paths",
        lambda _root: {"test_report": report, "readiness": readiness},
    )
    monkeypatch.setattr(
        amendment,
        "_job55_junit_counts",
        lambda path: {"content_sha256": tier0.file_sha256(Path(path))},
    )

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        target = Path(next(item for item in command if item.startswith("--junitxml=")).split("=", 1)[1])
        target.write_bytes(junit_bytes)
        return SimpleNamespace(returncode=0)

    def seal(_root: Path) -> Path:
        assert report.read_bytes() == report_before
        assert not temporary.exists()
        readiness.write_bytes(b'{"synthetic":"sealed"}\n')
        return readiness

    monkeypatch.setattr(sealer.subprocess, "run", run)
    monkeypatch.setattr(tier0, "fsync_directory", lambda _path: None)
    monkeypatch.setattr(amendment, "seal_job55_readiness", seal)
    monkeypatch.setattr(
        amendment,
        "validate_job55_readiness_receipt",
        lambda _root, path=None: {"receipt_sha256": "f" * 64},
    )

    assert sealer.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip().startswith("SEALED ")
    assert report.read_bytes() == report_before
    assert not temporary.exists()


def test_job55_sealer_refuses_corrupt_stranded_junit_without_subprocess_or_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_cap_amendment_readiness as sealer

    repo = tmp_path / "synthetic-job55-sealer-corruption-repo"
    work = repo / "v5/work/cmbp-tier0-cap-amendment"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    report.write_bytes(b"corrupt stranded JUnit\n")
    subprocess_calls: list[str] = []
    seal_calls: list[str] = []
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])
    monkeypatch.setattr(
        amendment,
        "_paths",
        lambda _root: {"test_report": report, "readiness": readiness},
    )
    monkeypatch.setattr(
        amendment,
        "_job55_junit_counts",
        lambda _path: (_ for _ in ()).throw(
            tier0.Tier0Error("synthetic corrupt JUnit", status="STOP_JOB55_LOCAL_TESTS")
        ),
    )
    monkeypatch.setattr(
        sealer.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess_calls.append("run"),
    )
    monkeypatch.setattr(
        amendment,
        "seal_job55_readiness",
        lambda _root: seal_calls.append("seal"),
    )

    assert sealer.main() == 2
    captured = capsys.readouterr()
    assert "STOP_JOB55_LOCAL_TESTS" in captured.err
    assert subprocess_calls == []
    assert seal_calls == []
    assert report.read_bytes() == b"corrupt stranded JUnit\n"
    assert not readiness.exists()


def test_job55_sealer_existing_valid_readiness_is_idempotent_and_runs_no_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from v5.ops import seal_cmbp_tier0_cap_amendment_readiness as sealer

    repo = tmp_path / "synthetic-job55-sealer-idempotent-repo"
    work = repo / "v5/work/cmbp-tier0-cap-amendment"
    work.mkdir(parents=True)
    report = work / "TEST_RESULTS_V1.xml"
    readiness = work / "LOCAL_READINESS_RECEIPT_V1.json"
    readiness.write_bytes(b'{"synthetic":"existing readiness"}\n')
    calls: list[str] = []
    monkeypatch.setattr(sealer, "REPO", repo)
    monkeypatch.setattr(sys, "argv", [str(Path(sealer.__file__))])
    monkeypatch.setattr(
        amendment,
        "_paths",
        lambda _root: {"test_report": report, "readiness": readiness},
    )
    monkeypatch.setattr(
        amendment,
        "validate_job55_readiness_receipt",
        lambda _root, path=None: {"receipt_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        sealer.subprocess,
        "run",
        lambda *_args, **_kwargs: calls.append("subprocess"),
    )
    monkeypatch.setattr(
        amendment,
        "seal_job55_readiness",
        lambda _root: calls.append("seal"),
    )

    assert sealer.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == f"SEALED {'1' * 64} {readiness}"
    assert calls == []
