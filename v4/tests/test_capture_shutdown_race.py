"""The capture must survive a record delivered after its window closes.

On 2026-08-10 the midday Track-A window ran its full 180.5 seconds, collected
40,708 rows with every hard stop clean, and was still voided: a record arrived
after the receipt file had been closed, raising ``ValueError: write to closed
file``, which the error callback recorded as a failure reason.

This exercises the REAL capture code against a fake client that reproduces the
race deterministically, contacting nothing. The fake delivers a straggler from
inside ``stop()``, which the repaired code calls *after* setting the guard and
*before* closing the file. So the receipt row count is the discriminator:
without the guard the straggler is written (31 rows), with it the straggler is
counted and dropped (30 rows).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = (
    REPO / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
)
REAL_DEFS = CAPTURE_ROOT / "2026-08-10/open/definitions/opra_live_definitions.dbn.zst"
APPROVAL = CAPTURE_ROOT / "authorization_v2.json"
DELIVERED = 30


class _FakeRecord:
    def __init__(self, instrument_id: int, ts_event: int, ts_recv: int, ts_out: int):
        self.rtype = 193
        self.instrument_id = instrument_id
        self.publisher_id = 1
        self.ts_event = ts_event
        self.ts_recv = ts_recv
        self.ts_out = ts_out


def _interval_end() -> int:
    minute = 60_000_000_000
    return (time.time_ns() // minute) * minute - minute


class _FakeLive:
    """A client that delivers a straggler from inside stop()."""

    stop_called = False
    late_delivered = threading.Event()

    def __init__(self, **kwargs):
        self._callbacks: list[tuple] = []
        _FakeLive.stop_called = False
        _FakeLive.late_delivered.clear()

    def subscribe(self, **kwargs) -> None:
        pass

    def add_stream(self, path, exception_callback=None) -> None:
        # The capture refuses a zero-byte raw file.
        Path(path).write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 512)

    def add_callback(self, record_callback, exception_callback=None) -> None:
        self._callbacks.append((record_callback, exception_callback))

    def _deliver(self, record) -> None:
        for callback, on_error in self._callbacks:
            try:
                callback(record)
            except Exception as exc:  # mirrors the client's own error routing
                if on_error is not None:
                    on_error(exc)

    def start(self) -> None:
        end = _interval_end()
        for index in range(DELIVERED):
            self._deliver(_FakeRecord(1000 + index, end - 60_000_000_000, end, end + 1))

    def block_for_close(self, timeout=None) -> None:
        time.sleep(0.01)

    def stop(self) -> None:
        _FakeLive.stop_called = True
        end = _interval_end()
        self._deliver(_FakeRecord(9999, end - 60_000_000_000, end, end + 1))
        _FakeLive.late_delivered.set()


@pytest.mark.skipif(
    not REAL_DEFS.is_file() or not APPROVAL.is_file(),
    reason="capture evidence tree not present",
)
def test_a_record_after_the_window_is_counted_not_fatal(tmp_path, monkeypatch) -> None:
    import databento as real_databento

    fake = types.ModuleType("databento")
    for name in dir(real_databento):
        if not name.startswith("__"):
            setattr(fake, name, getattr(real_databento, name))
    fake.Live = _FakeLive
    monkeypatch.setitem(sys.modules, "databento", fake)

    approval = json.loads(APPROVAL.read_text())["approval_required"][
        "exact_approval_text"
    ]
    monkeypatch.setenv("V4_PAID_DATA_APPROVAL_TEXT", approval)

    out_dir = tmp_path / "market"
    monkeypatch.setattr(
        sys, "argv",
        [
            "capture",
            "--session-date", "2026-08-10",
            "--definition-path", str(REAL_DEFS),
            "--duration-seconds", "2",
            "--schemas", "cbbo-1m",
            "--output-dir", str(out_dir),
            "--env-file", str(REPO / "v4/.env"),
            "--approval-manifest", str(APPROVAL),
        ],
    )

    from v4.scripts import capture_databento_live_opra_training_twin as capture

    assert capture.main() == 0
    assert _FakeLive.stop_called, "the client must be stopped before the file closes"
    assert _FakeLive.late_delivered.is_set(), "the fake never exercised the race"

    summary = json.loads((out_dir / "capture_summary.json").read_text())
    assert summary["status"] == "CAPTURED_NO_ORDER_LIVE_SAMPLE"
    assert not summary.get("failure_reasons")
    assert summary["records_total"] == DELIVERED
    # The guard fired...
    assert summary["late_records_after_window"] == 1
    # ...and the straggler was dropped rather than written. 31 here would mean
    # the guard is gone and the window is one timing slip from being voided.
    rows = [
        line
        for line in (out_dir / "local_receipts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == DELIVERED
