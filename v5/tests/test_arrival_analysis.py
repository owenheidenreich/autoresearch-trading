"""Adversarial tests for the Track-A arrival analysis and ledger re-issue.

The synthetic fixtures are built with the capture script's own statistic
(``np.quantile(..., method="nearest")``) so the cross-check tests exercise the
refusal logic, not a method mismatch.  Two integration tests read the real
2026-08-05 midday infrastructure run and the real legacy ledger read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from v5.research import arrival_analysis as aa
from v5.research import feature_admission
from v5.research.training_twin import make_latency_receipt


FAKE_DECLARATION = {
    "capture_window": {
        "sessions": ["2026-08-06", "2026-08-07"],
        "windows": [{"name": "open"}, {"name": "midday"}],
    }
}


def _write_window(
    root: Path,
    session: str,
    window: str,
    *,
    lags_ns: list[int],
    status: str = "CAPTURED_NO_ORDER_LIVE_SAMPLE",
    hard_stop: bool = False,
    tamper_p99: bool = False,
    drop_row_from_summary: bool = False,
) -> None:
    market = root / session / window / "market"
    market.mkdir(parents=True)
    base_end = 1_785_946_380_000_000_000
    rows = []
    for index, lag in enumerate(lags_ns):
        interval_end = base_end + (index % 3) * 60_000_000_000
        rows.append(
            {
                "record_class": "CBBOMsg",
                "rtype": 193,
                "instrument_id": 100 + index,
                "interval_end_unix_ns": interval_end,
                "local_receipt_unix_ns": interval_end + lag,
            }
        )
    rows.append(
        {
            "record_class": "SystemMsg",
            "rtype": 23,
            "instrument_id": 0,
            "interval_end_unix_ns": None,
            "local_receipt_unix_ns": base_end,
        }
    )
    with (market / "local_receipts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    array = np.asarray(sorted(lags_ns), dtype=np.int64)
    stats = {
        "min": int(array.min()),
        "p50": int(np.quantile(array, 0.50, method="nearest")),
        "p90": int(np.quantile(array, 0.90, method="nearest")),
        "p99": int(np.quantile(array, 0.99, method="nearest")),
        "max": int(array.max()),
    }
    if tamper_p99:
        stats["p99"] += 1
    summary = {
        "status": status,
        "hard_stops": {"broker_accessed": hard_stop},
        "records_total": len(rows) - (1 if drop_row_from_summary else 0),
        "plan": {"symbol_count": 510},
        "local_receipt_minus_interval_end_ns": {"CBBOMsg:rtype=193": stats},
    }
    (market / "capture_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _analyze(root: Path, session: str = "2026-08-06", window: str = "open"):
    return aa.analyze_window(
        session, window, capture_root=root, declaration=FAKE_DECLARATION
    )


def test_analyze_window_recomputes_and_classifies(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[10_000_000 * k for k in range(1, 60)])
    result = _analyze(tmp_path)
    assert result.is_evidence
    assert result.records_total == 60
    assert result.symbol_count == 510
    stats = result.lag_ns_by_class["CBBOMsg:rtype=193"]
    assert stats["min"] == 10_000_000 and stats["max"] == 590_000_000
    coverage = result.coverage_by_class["CBBOMsg:rtype=193"]
    assert coverage["instruments"] == 59
    assert coverage["interval_ends"] == 3


def test_undeclared_session_is_never_evidence(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-05", "midday", lags_ns=[5_000_000, 6_000_000])
    result = _analyze(tmp_path, "2026-08-05", "midday")
    assert not result.is_evidence


def test_stat_drift_is_refused(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[1_000_000] * 10, tamper_p99=True)
    with pytest.raises(aa.ArrivalAnalysisError, match="lag_stat_mismatch"):
        _analyze(tmp_path)


def test_row_count_mismatch_is_refused(tmp_path) -> None:
    _write_window(
        tmp_path, "2026-08-06", "open", lags_ns=[1_000_000] * 5, drop_row_from_summary=True
    )
    with pytest.raises(aa.ArrivalAnalysisError, match="row_count_mismatch"):
        _analyze(tmp_path)


def test_hard_stop_and_bad_status_are_refused(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[1_000_000], hard_stop=True)
    with pytest.raises(aa.ArrivalAnalysisError, match="hard_stop_recorded"):
        _analyze(tmp_path)
    _write_window(tmp_path, "2026-08-07", "open", lags_ns=[1_000_000], status="PARTIAL")
    with pytest.raises(aa.ArrivalAnalysisError, match="window_unhealthy"):
        _analyze(tmp_path, "2026-08-07", "open")


def test_envelope_takes_the_worst_and_needs_evidence(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[100_000_000, 200_000_000])
    _write_window(tmp_path, "2026-08-07", "open", lags_ns=[50_000_000, 500_000_000])
    _write_window(tmp_path, "2026-08-05", "midday", lags_ns=[999_000_000])
    windows = [
        _analyze(tmp_path, "2026-08-06", "open"),
        _analyze(tmp_path, "2026-08-07", "open"),
        _analyze(tmp_path, "2026-08-05", "midday"),  # infra run: excluded
    ]
    envelope = aa.evidence_envelope(windows)
    assert envelope["session_count"] == 2
    assert envelope["worst_ns"]["max"] == 500_000_000  # not 999 — infra excluded
    # p50 is the capture's nearest-quantile, not a mean: 100M and 50M per window.
    assert envelope["worst_ns"]["p50"] == 100_000_000
    with pytest.raises(aa.ArrivalAnalysisError, match="no_evidence_windows"):
        aa.evidence_envelope([windows[2]])


def test_guard_clock_floor_and_multiplier() -> None:
    assert aa.guard_clock_ms(527.622) == 10_000.0
    assert aa.guard_clock_ms(3_000.0) == 12_000.0
    with pytest.raises(aa.ArrivalAnalysisError):
        aa.guard_clock_ms(0.0)


def test_receipt_from_envelope(tmp_path) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[100_000_000, 527_622_000])
    envelope = aa.evidence_envelope([_analyze(tmp_path)])
    receipt = aa.build_cbbo1m_latency_receipt(
        envelope,
        measured_on="2026-08-07",
        valid_until="2026-09-07",
        evidence_path="v4/audit/somewhere",
    )
    assert receipt.source_family == aa.OPRA_CBBO_1M_FAMILY
    assert receipt.p99_ms == pytest.approx(527.622)
    assert receipt.session_count == 1
    receipt.assert_usable(source_family=aa.OPRA_CBBO_1M_FAMILY, as_of="2026-08-08")


def test_certification_wording_never_says_worst_case() -> None:
    wording = aa.certification_wording(2)
    assert "2-session envelope" in wording
    assert "not a population worst case" in wording


# --- ledger re-issue ---------------------------------------------------------


def _legacy_ledger(tmp_path: Path) -> tuple[Path, Path]:
    old_receipt = tmp_path / "old_receipt.json"
    old_receipt.write_text('{"old": true}', encoding="utf-8")
    entry = {"path": str(old_receipt), "sha256": aa.sha256_file(old_receipt)}
    rows = [
        {
            "name": "clock_minute",
            "family": "entry.contract_clock.v1",
            "contract_id": "entry.contract_clock.v1",
            "status": "ADMITTED",
            "availability_clock_ms": 0.0,
            "receipts": [entry],
        },
        {
            "name": "native_a",
            "family": aa.NATIVE_FAMILY,
            "contract_id": aa.NATIVE_FAMILY,
            "status": "BARRED",
            "barred_reason": aa.NATIVE_BLOCKER,
            "receipts": [],
        },
        {
            "name": "native_b",
            "family": aa.NATIVE_FAMILY,
            "contract_id": aa.NATIVE_FAMILY,
            "status": "BARRED",
            "barred_reason": aa.NATIVE_BLOCKER,
            "receipts": [],
        },
        {
            "name": "cross_a",
            "family": "entry.opra_cbbo1m_cross_section.v1",
            "contract_id": "entry.opra_cbbo1m_cross_section.v1",
            "status": "BARRED",
            "barred_reason": f"{aa.PARENT_BLOCKER_PREFIX}{aa.NATIVE_FAMILY}",
            "receipts": [],
        },
        {
            "name": "spot_a",
            "family": "entry.opra_implied_spot.v1",
            "contract_id": "entry.opra_implied_spot.v1",
            "status": "BARRED",
            "barred_reason": f"{aa.PARENT_BLOCKER_PREFIX}{aa.NATIVE_FAMILY}",
            "receipts": [],
        },
        {
            "name": "vol_a",
            "family": "entry.opra_implied_volatility.v1",
            "contract_id": "entry.opra_implied_volatility.v1",
            "status": "BARRED",
            "barred_reason": f"{aa.PARENT_BLOCKER_PREFIX}{aa.NATIVE_FAMILY}",
            "receipts": [],
        },
        {
            "name": "no_twin",
            "family": "entry.barred_no_live_twin.v1",
            "contract_id": "entry.barred_no_live_twin.v1",
            "status": "BARRED",
            "barred_reason": "permanently_barred_no_matching_live_twin",
            "receipts": [],
        },
    ]
    payload = {"schema_version": "pathd.feature-admission-ledger.v1", "features": rows}
    payload["ledger_sha256"] = feature_admission.ledger_sha256(payload)
    path = tmp_path / "legacy_ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, old_receipt


def _receipt_and_files(tmp_path: Path, valid_until: str = "2026-09-07"):
    latency_file = tmp_path / "latency_receipt.json"
    sparse_file = tmp_path / "sparse_report.json"
    receipt = make_latency_receipt(
        source_family=aa.OPRA_CBBO_1M_FAMILY,
        p50_ms=369.619,
        p99_ms=527.622,
        max_ms=527.864,
        session_count=2,
        measured_on="2026-08-07",
        valid_until=valid_until,
        evidence_path="v4/audit/somewhere",
    )
    latency_file.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    sparse_file.write_text('{"coverage": "report"}', encoding="utf-8")
    return receipt, [latency_file, sparse_file]


def test_reissue_admits_the_dependency_chain(tmp_path) -> None:
    legacy_path, _ = _legacy_ledger(tmp_path)
    receipt, files = _receipt_and_files(tmp_path)
    payload = aa.reissue_ledger(
        legacy_path=legacy_path,
        latency_receipt=receipt,
        receipt_files=files,
        availability_clock_ms=aa.guard_clock_ms(receipt.p99_ms),
        valid_until="2026-09-07",
        issued_on="2026-08-08",
    )
    by_name = {row["name"]: row for row in payload["features"]}
    for name in ("native_a", "native_b", "cross_a", "spot_a", "vol_a"):
        assert by_name[name]["status"] == "ADMITTED", name
        assert by_name[name]["valid_until"] == "2026-09-07T23:59:59+00:00"
        assert by_name[name]["availability_clock_ms"] == 10_000.0
        assert by_name[name]["superseded_barred_reason"]
        assert len(by_name[name]["receipts"]) == 2
    assert by_name["no_twin"]["status"] == "BARRED"
    assert "superseded_barred_reason" not in by_name["no_twin"]
    assert by_name["clock_minute"]["status"] == "ADMITTED"
    assert by_name["clock_minute"]["valid_until"] == "2026-09-07T23:59:59+00:00"

    written = aa.write_reissued_ledger(payload, tmp_path / "v5_ledger.json")
    assert written["ledger_sha256"] == payload["ledger_sha256"]
    feature_admission.assert_features_admitted(
        ["native_a", "vol_a"],
        ledger_path=tmp_path / "v5_ledger.json",
        as_of=__import__("datetime").datetime(
            2026, 8, 9, tzinfo=__import__("datetime").timezone.utc
        ),
    )
    with pytest.raises(feature_admission.FeatureAdmissionError, match="stale"):
        feature_admission.assert_features_admitted(
            ["native_a"],
            ledger_path=tmp_path / "v5_ledger.json",
            as_of=__import__("datetime").datetime(
                2026, 10, 1, tzinfo=__import__("datetime").timezone.utc
            ),
        )


def test_reissue_refuses_weak_inputs(tmp_path) -> None:
    legacy_path, _ = _legacy_ledger(tmp_path)
    receipt, files = _receipt_and_files(tmp_path)
    with pytest.raises(aa.ArrivalAnalysisError, match="guard_floor"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            receipt_files=files,
            availability_clock_ms=527.622,
            valid_until="2026-09-07",
            issued_on="2026-08-08",
        )
    with pytest.raises(aa.ArrivalAnalysisError, match="validity_windows_differ"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            receipt_files=files,
            availability_clock_ms=10_000.0,
            valid_until="2026-12-31",
            issued_on="2026-08-08",
        )


def test_write_refuses_overwrite(tmp_path) -> None:
    target = tmp_path / "ledger.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(aa.ArrivalAnalysisError, match="refusing_to_overwrite"):
        aa.write_reissued_ledger({}, target)


# --- integration against the real banked artifacts ---------------------------


REAL_MIDDAY = aa.CAPTURE_ROOT / "2026-08-05/midday/market/capture_summary.json"


@pytest.mark.skipif(not REAL_MIDDAY.is_file(), reason="real capture not present")
def test_real_midday_infra_run_recomputes_and_is_not_evidence() -> None:
    result = aa.analyze_window("2026-08-05", "midday")
    assert not result.is_evidence  # infrastructure verification, never evidence
    assert result.records_total == 51_232
    assert result.lag_ns_by_class["CBBOMsg:rtype=193"]["p99"] == 527_622_000
    assert result.symbol_count == 510


@pytest.mark.skipif(
    not feature_admission.LEGACY_LEDGER_PATH.is_file(),
    reason="legacy ledger not present",
)
def test_reissue_on_the_real_legacy_ledger(tmp_path) -> None:
    receipt, files = _receipt_and_files(tmp_path)
    payload = aa.reissue_ledger(
        latency_receipt=receipt,
        receipt_files=files,
        availability_clock_ms=aa.guard_clock_ms(receipt.p99_ms),
        valid_until="2026-09-07",
        issued_on="2026-08-08",
    )
    rows = payload["features"]
    admitted = [row for row in rows if row["status"] == "ADMITTED"]
    newly = [row for row in admitted if "superseded_barred_reason" in row]
    assert len(rows) == 83
    assert len(newly) == 43
    assert len(admitted) == 8 + 43
    families = {row["family"] for row in newly}
    assert families == {
        aa.NATIVE_FAMILY,
        "entry.opra_cbbo1m_cross_section.v1",
        "entry.opra_implied_spot.v1",
        "entry.opra_implied_volatility.v1",
        "entry.self_computed_greeks.v1",
    }
    written = aa.write_reissued_ledger(payload, tmp_path / "v5_ledger.json")
    assert written["ledger_sha256"] == payload["ledger_sha256"]
