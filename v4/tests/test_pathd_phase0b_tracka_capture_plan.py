from __future__ import annotations

from datetime import date
import json

import pytest

from v4.research.pathd_phase0b_tracka_capture_plan import (
    DECLARED_SESSIONS,
    EXPECTED_SYMBOL_COUNT,
    SCHEMAS,
    declaration_payload,
    verify_declaration,
    write_declaration,
)
from v4.research.pathd_phase0b_tracka_receipts import (
    ARRIVAL_CLOCK_KIND,
    build_multisession_arrival_receipt,
)
from v4.scripts.capture_databento_live_opra_training_twin import (
    LOCAL_RECEIPT_SCHEMA,
    _sha256_path,
    _stable_hash,
)


def test_tracka_window_is_frozen_before_capture_and_consecutive() -> None:
    """Updated 2026-08-04: owner reduced the window to Wed-Fri (Codex out of usage
    until 08-08). Three CONSECUTIVE sessions, and sampling did not shrink -- the
    original declaration was 5 sessions x 1 midday window = 5 windows; this is
    3 x 2 = 6 windows, and unlike the original it covers the open."""
    assert DECLARED_SESSIONS == (
        date(2026, 8, 5),
        date(2026, 8, 6),
        date(2026, 8, 7),
    )
    # consecutive weekdays, no substitution
    assert [d.weekday() for d in DECLARED_SESSIONS] == [2, 3, 4]  # Wed, Thu, Fri
    assert SCHEMAS == ("cbbo-1s", "cbbo-1m", "ohlcv-1m", "trades")
    assert EXPECTED_SYMBOL_COUNT == 510


def test_capture_windows_span_the_open_and_respect_the_recorder_cap() -> None:
    """The single 09:10 PT (12:10 ET) window measured only quiet midday latency.

    Arrival latency is stressed by message volume, which peaks at the open, so a
    midday-only sample reports an optimistically low p99 -- a model trained on it
    would decide earlier than is live-possible during exactly the period it trades
    most. The recorder also enforces 1 <= duration <= 300 s, so the first attempt
    at a 900 s open window was unrunnable.
    """
    from v4.research.pathd_phase0b_tracka_capture_plan import (
        CAPTURE_WINDOWS,
        CLOCK_SELECTION_LAW,
    )

    names = {w["name"] for w in CAPTURE_WINDOWS}
    assert names == {"open", "midday"}
    for w in CAPTURE_WINDOWS:
        assert 1.0 <= w["duration_seconds"] <= 300.0, "recorder caps duration at 300 s"
    # the admitted clock must be the worst case, never the quiet window alone
    assert "max" in CLOCK_SELECTION_LAW and "never the mean" in CLOCK_SELECTION_LAW


def test_declaration_is_signed_and_explicitly_disallows_connection(tmp_path) -> None:
    path = tmp_path / "declaration.json"
    written = write_declaration(path)
    verified = verify_declaration(path)
    assert verified == written
    assert verified["authorization_gate"]["connection_allowed"] is False
    assert verified["hard_stops"]["holdout_open_count"] == 0
    assert verified["measurements"]["per_message_local_receipt_sidecar_required"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        write_declaration(path)


def test_tampered_declaration_fails_signature(tmp_path) -> None:
    path = tmp_path / "declaration.json"
    write_declaration(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["subscription"]["expected_symbol_count"] = 509
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="signature mismatch"):
        verify_declaration(path)


def test_declaration_does_not_claim_historical_api_authorization() -> None:
    payload = declaration_payload()
    assert payload["historical_api_scope"]["new_request_authorized"] is False
    assert payload["status"] == "DECLARED_AWAITING_EXPLICIT_OWNER_AUTHORIZATION"


def _synthetic_capture(tmp_path, session: date, ordinal: int):
    capture_dir = tmp_path / session.isoformat()
    capture_dir.mkdir()
    raw_path = capture_dir / "opra_live_mixed.dbn.zst"
    raw_path.write_bytes(f"raw-{session}".encode())
    sidecar_path = capture_dir / "local_receipts.jsonl"
    rows = []
    sequence = 0
    for rtype, base_lag_ns in ((192, 100_000_000), (193, 200_000_000), (33, 300_000_000)):
        for repeat in range(2):
            interval_end = 1_000_000_000 + repeat
            rows.append(
                {
                    "schema_version": LOCAL_RECEIPT_SCHEMA,
                    "sequence_index": sequence,
                    "local_receipt_unix_ns": interval_end + base_lag_ns + ordinal,
                    "record_class": "Synthetic",
                    "rtype": rtype,
                    "publisher_id": 1,
                    "instrument_id": 2,
                    "ts_event": interval_end,
                    "ts_recv": interval_end,
                    "ts_out": interval_end,
                    "interval_end_unix_ns": interval_end,
                }
            )
            sequence += 1
    sidecar_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "autoresearch.databento-live-opra-training-twin-capture.v1",
        "status": "CAPTURED_NO_ORDER_LIVE_SAMPLE",
        "plan": {
            "session_date": session.isoformat(),
            "symbol_count": EXPECTED_SYMBOL_COUNT,
            "schemas": list(SCHEMAS),
            "network": True,
            "broker_or_order_path": False,
        },
        "records_total": len(rows),
        "raw_dbn": {"path": str(raw_path), "sha256": _sha256_path(raw_path)},
        "local_receipts": {
            "schema_version": LOCAL_RECEIPT_SCHEMA,
            "path": str(sidecar_path),
            "rows": len(rows),
            "sha256": _sha256_path(sidecar_path),
        },
        "hard_stops": {
            "broker_accessed": False,
            "order_path_accessed": False,
            "paper_runtime_accessed": False,
            "model_loaded_or_fit": False,
            "holdout_open_count": 0,
            "promotion_or_default_changed": False,
        },
    }
    summary["summary_sha256"] = _stable_hash(summary)
    summary_path = capture_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_multisession_receipt_uses_each_familys_own_arrival_source(tmp_path) -> None:
    declaration = tmp_path / "declaration.json"
    write_declaration(declaration)
    summaries = [
        _synthetic_capture(tmp_path, session, ordinal)
        for ordinal, session in enumerate(DECLARED_SESSIONS)
    ]
    receipt = build_multisession_arrival_receipt(
        summaries, declaration_path=declaration
    )
    families = receipt["family_distributions"]
    assert families["entry.opra_cbbo1s_rolling.v1"]["availability_clock_ms"] == pytest.approx(100.000004)
    assert families["entry.opra_cbbo1m_native.v1"]["availability_clock_ms"] == pytest.approx(200.000004)
    assert families["entry.opra_ohlcv1m_sparse.v1"]["availability_clock_ms"] == pytest.approx(300.000004)
    assert all(item["availability_clock_kind"] == ARRIVAL_CLOCK_KIND for item in families.values())
    assert receipt["prior_319_5ms_used_as_measurement"] is False


def test_multisession_receipt_rejects_missing_declared_session(tmp_path) -> None:
    declaration = tmp_path / "declaration.json"
    write_declaration(declaration)
    summaries = [
        _synthetic_capture(tmp_path, session, ordinal)
        for ordinal, session in enumerate(DECLARED_SESSIONS[:-1])
    ]
    with pytest.raises(RuntimeError, match="declared order"):
        build_multisession_arrival_receipt(summaries, declaration_path=declaration)
