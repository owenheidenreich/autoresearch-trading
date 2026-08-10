"""Adversarial tests for the Track-A arrival analysis and ledger re-issue.

The synthetic fixtures are built with the capture script's own statistic
(``np.quantile(..., method="nearest")``) so the cross-check tests exercise the
refusal logic, not a method mismatch.  Two integration tests read the real
2026-08-05 midday infrastructure run and the real legacy ledger read-only.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np
import pytest

from v5.research import arrival_analysis as aa
from v5.research import feature_admission
from v5.research.training_twin import (
    TrainingTwinError,
    make_freshness_receipt,
    make_latency_receipt,
)
from v5.ops import certify_tracka_arrival as driver


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
    duration_seconds: float = 180.0,
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
        "plan": {"symbol_count": 510, "duration_seconds": duration_seconds},
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
    assert coverage["expected_interval_ends"] == 3
    assert coverage["expected_instrument_intervals"] == 1_530
    assert coverage["missing_instrument_intervals"] == 1_471


def test_module_declaration_matches_all_capture_runners() -> None:
    """The Phase-2 evidence law and every active runner name the same file."""

    runner_dir = aa.REPO_ROOT / "v4/ops/tracka"
    runner_paths = [
        runner_dir / "check_tracka.sh",
        runner_dir / "run_tracka_attended.sh",
        runner_dir / "run_tracka_window.sh",
        runner_dir / "tracka_launcher.sh",
    ]
    expected_relative = str(aa.DECLARATION_PATH.relative_to(aa.REPO_ROOT))
    expected_root_relative = str(aa.CAPTURE_ROOT.relative_to(aa.REPO_ROOT))
    for path in runner_paths:
        active_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#") or "capture_declaration_v" not in line:
                continue
            active_lines.append(line)
        assert len(active_lines) == 1, path
        line = active_lines[0]
        match = re.search(r"capture_declaration_v[0-9]+\.json", line)
        assert match is not None, path
        assert match.group(0) == aa.DECLARATION_PATH.name
        if "$ROOT/" in line:
            root_lines = [
                candidate
                for candidate in path.read_text(encoding="utf-8").splitlines()
                if candidate.startswith('ROOT="$REPO/')
            ]
            assert len(root_lines) == 1, path
            assert expected_root_relative in root_lines[0]
        else:
            assert f"$REPO/{expected_relative}" in line


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


def test_actual_n_probability_arithmetic_matches_status_section_13() -> None:
    probabilities = driver._sample_max_probabilities(3)
    assert probabilities["exceeds_daily_p95"] == pytest.approx(0.142625)
    assert probabilities["exceeds_daily_p99"] == pytest.approx(0.029701)


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


def test_freshness_receipt_signs_missing_minutes_without_cross_window_product(
    tmp_path,
) -> None:
    _write_window(tmp_path, "2026-08-06", "open", lags_ns=[100_000_000] * 2)
    _write_window(tmp_path, "2026-08-07", "open", lags_ns=[200_000_000] * 3)
    windows = [
        _analyze(tmp_path, "2026-08-06", "open"),
        _analyze(tmp_path, "2026-08-07", "open"),
    ]
    coverage = aa.evidence_coverage(windows)
    assert coverage["instrument_intervals"] == 5
    assert coverage["expected_instrument_intervals"] == 3_060
    assert coverage["missing_instrument_intervals"] == 3_055
    receipt = aa.build_cbbo1m_freshness_receipt(
        coverage,
        measured_on="2026-08-07",
        valid_until="2026-11-10",
        evidence_path="v4/audit/somewhere",
    )
    assert receipt.expected_instrument_intervals == 3_060
    assert receipt.missing_instrument_intervals == 3_055
    assert receipt.coverage_sha256 == aa.canonical_mapping_sha256(coverage)
    receipt.assert_usable(source_family=aa.OPRA_CBBO_1M_FAMILY, as_of="2026-08-08")
    with pytest.raises(TrainingTwinError, match="self-hash mismatch"):
        replace(receipt, coverage_sha256="f" * 64).assert_usable(
            source_family=aa.OPRA_CBBO_1M_FAMILY, as_of="2026-08-08"
        )


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
    """Both receipts the re-issue requires, plus their on-disk evidence files."""

    latency_file = tmp_path / "latency_receipt.json"
    sparse_file = tmp_path / "freshness_receipt.json"
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
    freshness = make_freshness_receipt(
        source_family=aa.OPRA_CBBO_1M_FAMILY,
        record_class=aa.CBBO_1M_CLASS,
        session_count=2,
        instruments=510,
        interval_ends=6,
        expected_interval_ends=6,
        instrument_intervals=1_361,
        expected_instrument_intervals=1_530,
        worst_window_coverage_ratio=1_361 / 1_530,
        early_rows=0,
        coverage_sha256="a" * 64,
        measured_on="2026-08-07",
        valid_until=valid_until,
        evidence_path="v4/audit/somewhere",
    )
    latency_file.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    sparse_file.write_text(json.dumps(freshness.to_dict()), encoding="utf-8")
    return receipt, freshness, [latency_file, sparse_file]


def test_reissue_admits_the_dependency_chain(tmp_path) -> None:
    legacy_path, _ = _legacy_ledger(tmp_path)
    receipt, freshness, files = _receipt_and_files(tmp_path)
    payload = aa.reissue_ledger(
        legacy_path=legacy_path,
        latency_receipt=receipt,
        freshness_receipt=freshness,
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
    assert payload["availability_guard"]["latency_receipt_sha256"] == (
        receipt.receipt_sha256
    )
    assert payload["availability_guard"]["freshness_receipt_sha256"] == (
        freshness.receipt_sha256
    )
    assert payload["availability_guard"]["freshness_coverage_sha256"] == (
        freshness.coverage_sha256
    )

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
    receipt, freshness, files = _receipt_and_files(tmp_path)
    with pytest.raises(aa.ArrivalAnalysisError, match="preregistered_law"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            freshness_receipt=freshness,
            receipt_files=files,
            availability_clock_ms=527.622,
            valid_until="2026-09-07",
            issued_on="2026-08-08",
        )
    with pytest.raises(aa.ArrivalAnalysisError, match="validity_windows_differ"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            freshness_receipt=freshness,
            receipt_files=files,
            availability_clock_ms=10_000.0,
            valid_until="2026-12-31",
            issued_on="2026-08-08",
        )


def test_reissue_needs_both_receipts_not_just_latency(tmp_path) -> None:
    """The eleven native rows record two blockers; one receipt clears one half.

    Supplying a freshness receipt measured over a different number of sessions
    is the realistic version of this mistake — two receipts that do not describe
    the same sample.
    """

    legacy_path, _ = _legacy_ledger(tmp_path)
    receipt, freshness, files = _receipt_and_files(tmp_path)

    with pytest.raises(TypeError):
        aa.reissue_ledger(  # type: ignore[call-arg]
            legacy_path=legacy_path,
            latency_receipt=receipt,
            receipt_files=files,
            availability_clock_ms=10_000.0,
            valid_until="2026-09-07",
            issued_on="2026-08-08",
        )

    forged_latency = tmp_path / "forged_latency.json"
    forged_latency.write_text(
        json.dumps({"receipt_sha256": receipt.receipt_sha256}), encoding="utf-8"
    )
    with pytest.raises(aa.ArrivalAnalysisError, match="exact_persisted"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            freshness_receipt=freshness,
            receipt_files=[forged_latency, files[1]],
            availability_clock_ms=10_000.0,
            valid_until="2026-09-07",
            issued_on="2026-08-08",
        )

    mismatched = make_freshness_receipt(
        source_family=aa.OPRA_CBBO_1M_FAMILY,
        record_class=aa.CBBO_1M_CLASS,
        session_count=1,
        instruments=510,
        interval_ends=3,
        expected_interval_ends=3,
        instrument_intervals=1_361,
        expected_instrument_intervals=1_530,
        worst_window_coverage_ratio=1_361 / 1_530,
        early_rows=0,
        coverage_sha256="b" * 64,
        measured_on="2026-08-07",
        valid_until="2026-09-07",
        evidence_path="v4/audit/somewhere",
    )
    with pytest.raises(aa.ArrivalAnalysisError, match="different_samples"):
        aa.reissue_ledger(
            legacy_path=legacy_path,
            latency_receipt=receipt,
            freshness_receipt=mismatched,
            receipt_files=files,
            availability_clock_ms=10_000.0,
            valid_until="2026-09-07",
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
    receipt, freshness, files = _receipt_and_files(tmp_path)
    payload = aa.reissue_ledger(
        latency_receipt=receipt,
        freshness_receipt=freshness,
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


def _sealed_declaration(path: Path, *, session: str, window: str) -> Path:
    return _sealed_declaration_matrix(path, sessions=[session], windows=[window])


def _sealed_declaration_matrix(
    path: Path, *, sessions: list[str], windows: list[str]
) -> Path:
    payload = {
        "capture_window": {
            "sessions": sessions,
            "windows": [{"name": window} for window in windows],
        }
    }
    payload["declaration_sha256"] = driver._canonical_hash(
        payload, "declaration_sha256"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2_driver_runs_end_to_end_and_writes_a_new_dated_tree(
    tmp_path, capsys
) -> None:
    # Must be an open window: a quiet-window-only envelope is refused by the
    # declaration's envelope_law.
    session, window = "2026-08-10", "open"
    _write_window(
        tmp_path,
        session,
        window,
        lags_ns=[100_000_000, 527_622_000, 200_000_000],
    )
    declaration = _sealed_declaration(
        tmp_path / "capture_declaration_v8.json", session=session, window=window
    )
    out_dir = tmp_path / "phase2_issuance_2026-08-10"
    result = driver.main(
        [
            "--capture-root",
            str(tmp_path),
            "--declaration",
            str(declaration),
            "--measured-on",
            session,
            "--out-dir",
            str(out_dir),
        ]
    )
    assert result == 0, capsys.readouterr()
    assert (out_dir / driver.LEDGER_NAME).is_file()
    summary = json.loads((out_dir / "issuance_summary.json").read_text())
    assert summary["valid_until"] == "2026-11-10"
    assert summary["counts"]["admitted"] == 51
    assert summary["counts"]["scoped_admitted"] == 51
    assert summary["counts"]["scoped_features"] == 73
    assert summary["counts"]["barred"] == 32
    assert len(summary["counts"]["remaining_barred_reasons"]) == 4
    verified = feature_admission.verify_ledger(out_dir / driver.LEDGER_NAME)
    assert len(verified["features"]) == 83
    freshness = json.loads((out_dir / "freshness_receipt.json").read_text())
    coverage = json.loads((out_dir / "freshness_envelope.json").read_text())
    assert freshness["coverage_sha256"] == aa.canonical_mapping_sha256(coverage)
    # The bounding sentence must survive into the artifact a reader opens
    # first, not only into the ledger.
    assert summary["certification_wording"] == aa.certification_wording(1)
    assert "not a population worst case" in summary["certification_wording"]
    assert summary["sample_max_exceedance_probabilities"] == pytest.approx(
        {"exceeds_daily_p95": 0.05, "exceeds_daily_p99": 0.01}
    )


def test_phase2_driver_composes_the_full_three_session_six_window_shape(
    tmp_path, capsys
) -> None:
    sessions = ["2026-08-10", "2026-08-11", "2026-08-12"]
    windows = ["open", "midday"]
    for session_index, session in enumerate(sessions):
        for window in windows:
            worst_lag = 3_000_000_000 if (session, window) == (
                "2026-08-12",
                "open",
            ) else 400_000_000 + session_index * 100_000_000
            _write_window(
                tmp_path,
                session,
                window,
                lags_ns=[100_000_000, 200_000_000, worst_lag],
                duration_seconds=300.0 if window == "open" else 180.0,
            )
    declaration = _sealed_declaration_matrix(
        tmp_path / "capture_declaration_v8.json",
        sessions=sessions,
        windows=windows,
    )
    out_dir = tmp_path / "phase2_issuance_2026-08-12"
    result = driver.main(
        [
            "--capture-root",
            str(tmp_path),
            "--declaration",
            str(declaration),
            "--measured-on",
            "2026-08-12",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert result == 0, capsys.readouterr()
    summary = json.loads((out_dir / "issuance_summary.json").read_text())
    assert summary["session_count"] == 3
    assert summary["window_count"] == 6
    assert summary["guard_clock_ms"] == 12_000.0
    assert summary["sample_max_exceedance_probabilities"] == pytest.approx(
        {"exceeds_daily_p95": 0.142625, "exceeds_daily_p99": 0.029701}
    )
    assert summary["counts"]["scoped_admitted"] == 51
    assert summary["counts"]["scoped_features"] == 73


def test_phase2_driver_refuses_a_partial_declared_window(tmp_path, capsys) -> None:
    session, window = "2026-08-10", "midday"
    partial = tmp_path / session / window / "definitions"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("incomplete", encoding="utf-8")
    declaration = _sealed_declaration(
        tmp_path / "capture_declaration_v8.json", session=session, window=window
    )
    result = driver.main(
        [
            "--capture-root",
            str(tmp_path),
            "--declaration",
            str(declaration),
            "--measured-on",
            session,
            "--dry-run",
        ]
    )
    assert result == 2
    assert f"declared_window_incomplete:{session}/{window}" in capsys.readouterr().err


def test_phase2_driver_refuses_when_zero_windows_banked(tmp_path, capsys) -> None:
    declaration = _sealed_declaration(
        tmp_path / "capture_declaration_v8.json",
        session="2026-08-10",
        window="midday",
    )
    result = driver.main(
        [
            "--capture-root",
            str(tmp_path),
            "--declaration",
            str(declaration),
            "--measured-on",
            "2026-08-10",
            "--dry-run",
        ]
    )
    assert result == 2
    assert "nothing_banked_yet" in capsys.readouterr().err


def test_phase2_driver_rehearses_real_infrastructure_without_issuing(capsys) -> None:
    result = driver.main(
        [
            "--measured-on",
            "2026-08-08",
            "--dry-run-window",
            "2026-08-05/midday",
        ]
    )
    assert result == 0, capsys.readouterr()
    output = capsys.readouterr().out
    assert "evidence=False" in output
    assert "no receipt or ledger was issued" in output


def test_a_failed_window_is_excluded_not_fatal(tmp_path, capsys) -> None:
    """One dead window must not permanently poison every good one.

    2026-08-10 midday failed on a shutdown race and can never be re-run. An
    abort here would have blocked certification forever.
    """

    sessions, windows = ["2026-08-10", "2026-08-11"], ["open", "midday"]
    for session in sessions:
        for window in windows:
            _write_window(
                tmp_path, session, window,
                lags_ns=[100_000_000, 200_000_000, 400_000_000],
                duration_seconds=300.0 if window == "open" else 180.0,
            )
    # Break one window the way the real one broke.
    broken = tmp_path / "2026-08-10/midday/market/capture_summary.json"
    payload = json.loads(broken.read_text())
    payload["status"] = "FAILED_LIVE_CAPTURE_RECORDED_NO_ORDER"
    payload["failure_reasons"] = ["ValueError:write to closed file"]
    broken.write_text(json.dumps(payload))

    declaration = _sealed_declaration_matrix(
        tmp_path / "capture_declaration_v9.json", sessions=sessions, windows=windows
    )
    out_dir = tmp_path / "phase2_issuance_2026-08-11"
    assert driver.main([
        "--capture-root", str(tmp_path), "--declaration", str(declaration),
        "--measured-on", "2026-08-11", "--out-dir", str(out_dir),
    ]) == 0, capsys.readouterr()

    summary = json.loads((out_dir / "issuance_summary.json").read_text())
    assert summary["window_count"] == 3
    assert [e["window"] for e in summary["excluded_windows"]] == ["2026-08-10/midday"]
    assert "FAILED_LIVE_CAPTURE" in summary["excluded_windows"][0]["reason"]


def test_a_midday_only_envelope_is_refused(tmp_path, capsys) -> None:
    """The declaration's envelope_law: a quiet window is a floor, not the p99."""

    sessions, windows = ["2026-08-10", "2026-08-11"], ["open", "midday"]
    for session in sessions:
        for window in windows:
            _write_window(
                tmp_path, session, window,
                lags_ns=[100_000_000, 200_000_000, 400_000_000],
                duration_seconds=300.0 if window == "open" else 180.0,
            )
    for session in sessions:  # break every open, leaving only quiet windows
        path = tmp_path / session / "open/market/capture_summary.json"
        payload = json.loads(path.read_text())
        payload["status"] = "FAILED_LIVE_CAPTURE_RECORDED_NO_ORDER"
        path.write_text(json.dumps(payload))

    declaration = _sealed_declaration_matrix(
        tmp_path / "capture_declaration_v9.json", sessions=sessions, windows=windows
    )
    assert driver.main([
        "--capture-root", str(tmp_path), "--declaration", str(declaration),
        "--measured-on", "2026-08-11", "--out-dir", str(tmp_path / "out"),
    ]) != 0
    assert "no_open_window_survived" in capsys.readouterr().err
