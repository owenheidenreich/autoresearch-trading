#!/usr/bin/env python3
"""Certify banked Track-A arrival evidence and re-issue the feature ledger."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from v5.research import arrival_analysis as aa
from v5.research import feature_admission
from v5.research.training_twin import TrainingTwinError


# Owner decision 2026-08-06, before any declared evidence landed. These are
# constants rather than command-line choices so a banking-day result cannot
# change the certification law.
VALID_UNTIL = "2026-11-10"
OUTPUT_PREFIX = "phase2_issuance_"
LEDGER_NAME = "feature_admission_ledger.json"
EXPECTED_FEATURES = 83
EXPECTED_SCOPED_FEATURES = 73
EXPECTED_NEW_ADMISSIONS = 43
EXPECTED_ADMITTED = 51
EXPECTED_BARRED = 32
NO_LIVE_TWIN_FAMILY = "entry.barred_no_live_twin.v1"
EXPECTED_REMAINING_BY_FAMILY = {
    NO_LIVE_TWIN_FAMILY: 10,
    "entry.causal_account_state.v1": 6,
    "entry.opra_cbbo1s_rolling.v1": 12,
    "entry.opra_ohlcv1m_sparse.v1": 4,
}


class Phase2DriverError(RuntimeError):
    """The one-command certification driver refused an incomplete run."""


def _canonical_hash(payload: Mapping[str, Any], field: str) -> str:
    material = dict(payload)
    material.pop(field, None)
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_declaration(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2DriverError(f"declaration_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise Phase2DriverError(f"declaration_not_an_object:{path}")
    expected = payload.get("declaration_sha256")
    if not isinstance(expected, str) or _canonical_hash(
        payload, "declaration_sha256"
    ) != expected:
        raise Phase2DriverError(f"declaration_sha256_mismatch:{path}")
    return payload


def _banked_declared_windows(
    capture_root: Path, declaration: Mapping[str, Any]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return complete banked windows and untouched declared windows.

    An absent window is allowed because fewer than three sessions may bank. A
    window directory with any bytes but no sealed summary is a partial attempt,
    not an absence, and fails closed with its exact session/window name.
    """

    banked: list[tuple[str, str]] = []
    absent: list[tuple[str, str]] = []
    for session, window in aa.declared_evidence_windows(declaration):
        window_root = capture_root / session / window
        summary = window_root / "market" / "capture_summary.json"
        if summary.is_file():
            banked.append((session, window))
            continue
        if window_root.exists() and any(window_root.iterdir()):
            raise Phase2DriverError(
                f"declared_window_incomplete:{session}/{window}:missing_capture_summary"
            )
        absent.append((session, window))
    return banked, absent


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise Phase2DriverError(f"refusing_to_overwrite:{path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(aa.REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _ledger_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = {}
    remaining_reasons: Counter[str] = Counter()
    for row in payload["features"]:
        family = str(row.get("contract_id", row.get("family", "")))
        status = str(row["status"])
        by_family.setdefault(family, Counter())[status] += 1
        if status == feature_admission.BARRED:
            remaining_reasons[str(row["barred_reason"])] += 1
    admitted = sum(
        1 for row in payload["features"] if row["status"] == feature_admission.ADMITTED
    )
    newly_admitted = sum(
        1
        for row in payload["features"]
        if row["status"] == feature_admission.ADMITTED
        and "superseded_barred_reason" in row
    )
    scoped = [
        row
        for row in payload["features"]
        if str(row.get("contract_id", row.get("family", "")))
        != NO_LIVE_TWIN_FAMILY
    ]
    scoped_admitted = sum(
        1 for row in scoped if row["status"] == feature_admission.ADMITTED
    )
    return {
        "features": len(payload["features"]),
        "scoped_features": len(scoped),
        "admitted": admitted,
        "scoped_admitted": scoped_admitted,
        "newly_admitted": newly_admitted,
        "barred": len(payload["features"]) - admitted,
        "scoped_barred": len(scoped) - scoped_admitted,
        "by_family": {
            family: {
                "admitted": counts[feature_admission.ADMITTED],
                "barred": counts[feature_admission.BARRED],
            }
            for family, counts in sorted(by_family.items())
        },
        "remaining_barred_reasons": dict(sorted(remaining_reasons.items())),
    }


def _assert_expected_counts(counts: Mapping[str, Any]) -> None:
    observed = (
        counts["features"],
        counts["scoped_features"],
        counts["newly_admitted"],
        counts["admitted"],
        counts["barred"],
    )
    expected = (
        EXPECTED_FEATURES,
        EXPECTED_SCOPED_FEATURES,
        EXPECTED_NEW_ADMISSIONS,
        EXPECTED_ADMITTED,
        EXPECTED_BARRED,
    )
    if observed != expected:
        raise Phase2DriverError(f"unexpected_ledger_counts:{observed}!={expected}")
    remaining_by_family = {
        family: values["barred"]
        for family, values in counts["by_family"].items()
        if values["barred"]
    }
    if remaining_by_family != EXPECTED_REMAINING_BY_FAMILY:
        raise Phase2DriverError(
            "unexpected_remaining_barred_families:"
            f"{remaining_by_family}!={EXPECTED_REMAINING_BY_FAMILY}"
        )
    if len(counts["remaining_barred_reasons"]) != 4:
        raise Phase2DriverError(
            "unexpected_remaining_barred_reason_groups:"
            f"{len(counts['remaining_barred_reasons'])}!=4"
        )


def _print_counts(counts: Mapping[str, Any]) -> None:
    print(f"admitted      {counts['admitted']} of {counts['features']}")
    print(
        f"scoped        {counts['scoped_admitted']} of "
        f"{counts['scoped_features']} admitted"
    )
    print(f"still barred  {counts['barred']}")
    print("by family")
    for family, values in counts["by_family"].items():
        print(
            f"  {family:42s} admitted={values['admitted']:>2d} "
            f"barred={values['barred']:>2d}"
        )


def _sample_max_probabilities(session_count: int) -> dict[str, float]:
    if session_count < 1:
        raise Phase2DriverError("sample_max_probability_needs_a_session")
    return {
        "exceeds_daily_p95": 1.0 - 0.95 ** session_count,
        "exceeds_daily_p99": 1.0 - 0.99 ** session_count,
    }


def _derive_certification(
    *,
    capture_root: Path,
    declaration: Mapping[str, Any],
) -> tuple[list[aa.WindowAnalysis], dict[str, Any], dict[str, Any], float]:
    banked, absent = _banked_declared_windows(capture_root, declaration)
    declared = aa.declared_evidence_windows(declaration)
    print(
        f"declared      {len(declared)} windows over "
        f"{len({session for session, _ in declared})} sessions"
    )
    print(f"banked        {len(banked)} of {len(declared)}")
    if absent:
        print("absent        " + ", ".join(f"{s}/{w}" for s, w in absent))
    if not banked:
        raise Phase2DriverError("nothing_banked_yet:no_evidence_to_certify")

    windows: list[aa.WindowAnalysis] = []
    for session, window in banked:
        try:
            analysis = aa.analyze_window(
                session,
                window,
                capture_root=capture_root,
                declaration=declaration,
            )
        except aa.ArrivalAnalysisError as exc:
            raise Phase2DriverError(
                f"declared_window_failed:{session}/{window}:{exc}"
            ) from exc
        if not analysis.is_evidence:
            raise Phase2DriverError(
                f"declared_window_classified_non_evidence:{session}/{window}"
            )
        windows.append(analysis)
        print(
            f"  {session} {window:7s} rows={analysis.records_total:>7d} "
            f"symbols={analysis.symbol_count:>4d} evidence={analysis.is_evidence}"
        )

    envelope = aa.evidence_envelope(windows)
    coverage = aa.evidence_coverage(windows)
    guard_ms = aa.guard_clock_ms(envelope["worst_ms"]["p99"])
    n = int(envelope["session_count"])
    probabilities = _sample_max_probabilities(n)
    print(f"sessions      {n}  {envelope['sessions']}")
    print(
        "worst p50/p99/max ms   "
        f"{envelope['worst_ms']['p50']:.3f} / "
        f"{envelope['worst_ms']['p99']:.3f} / "
        f"{envelope['worst_ms']['max']:.3f}"
    )
    print(
        f"guard clock   {guard_ms:.1f} ms "
        f"({'floor binds' if guard_ms == aa.GUARD_FLOOR_MS else '4x p99 binds'})"
    )
    print(
        f"coverage      worst={coverage['worst_coverage_ratio']:.6f} "
        f"missing={coverage['missing_instrument_intervals']} instrument-minutes"
    )
    print(
        "sample max   exceeds daily p95 with "
        f"{100 * probabilities['exceeds_daily_p95']:.1f}% probability; p99 with "
        f"{100 * probabilities['exceeds_daily_p99']:.1f}%"
    )
    print(f"wording       {aa.certification_wording(n)}")
    print(f"valid until   {VALID_UNTIL} (pre-registered 2026-08-06)")
    return windows, envelope, coverage, guard_ms


def _build_ledger(
    *,
    envelope: Mapping[str, Any],
    coverage: Mapping[str, Any],
    guard_ms: float,
    measured_on: str,
    evidence_path: str,
    receipt_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    latency = aa.build_cbbo1m_latency_receipt(
        envelope,
        measured_on=measured_on,
        valid_until=VALID_UNTIL,
        evidence_path=evidence_path,
    )
    freshness = aa.build_cbbo1m_freshness_receipt(
        coverage,
        measured_on=measured_on,
        valid_until=VALID_UNTIL,
        evidence_path=evidence_path,
    )
    latency_path = receipt_dir / "latency_receipt.json"
    freshness_path = receipt_dir / "freshness_receipt.json"
    _write_json(latency_path, latency.to_dict())
    _write_json(freshness_path, freshness.to_dict())
    ledger = aa.reissue_ledger(
        latency_receipt=latency,
        freshness_receipt=freshness,
        receipt_files=[latency_path, freshness_path],
        availability_clock_ms=guard_ms,
        valid_until=VALID_UNTIL,
        issued_on=measured_on,
    )
    return latency.to_dict(), freshness.to_dict(), ledger


def _dry_run_infrastructure_window(
    raw: str, *, capture_root: Path, declaration: Mapping[str, Any]
) -> int:
    parts = raw.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise Phase2DriverError("dry_run_window_must_be_SESSION/WINDOW")
    session, window = parts
    analysis = aa.analyze_window(
        session, window, capture_root=capture_root, declaration=declaration
    )
    print(json.dumps(analysis.to_dict(), indent=2, sort_keys=True))
    print(
        "DRY RUN: recomputed cleanly and classified "
        f"evidence={analysis.is_evidence}; no receipt or ledger was issued."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=aa.CAPTURE_ROOT)
    parser.add_argument("--declaration", type=Path, default=aa.DECLARATION_PATH)
    parser.add_argument("--measured-on", default=date.today().isoformat())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="dated output directory under the capture audit tree",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="run end to end, retain nothing"
    )
    parser.add_argument(
        "--dry-run-window",
        metavar="SESSION/WINDOW",
        help="recompute one infrastructure window without issuing evidence",
    )
    args = parser.parse_args(argv)

    try:
        date.fromisoformat(args.measured_on)
        declaration = _load_declaration(args.declaration)
        print(f"declaration   {args.declaration.name}")
        if args.dry_run_window:
            if args.dry_run:
                raise Phase2DriverError("choose_only_one_dry_run_mode")
            return _dry_run_infrastructure_window(
                args.dry_run_window,
                capture_root=args.capture_root,
                declaration=declaration,
            )

        windows, envelope, coverage, guard_ms = _derive_certification(
            capture_root=args.capture_root,
            declaration=declaration,
        )
        evidence_path = _repo_path(args.capture_root)

        if args.dry_run:
            with TemporaryDirectory(prefix="tracka-phase2-dry-run-") as raw_tmp:
                tmp = Path(raw_tmp)
                _, _, ledger = _build_ledger(
                    envelope=envelope,
                    coverage=coverage,
                    guard_ms=guard_ms,
                    measured_on=args.measured_on,
                    evidence_path=evidence_path,
                    receipt_dir=tmp,
                )
                aa.write_reissued_ledger(ledger, tmp / LEDGER_NAME)
                counts = _ledger_counts(ledger)
                _assert_expected_counts(counts)
            _print_counts(counts)
            print("DRY RUN: end-to-end verification passed; nothing retained.")
            return 0

        out_dir = args.out_dir or (
            args.capture_root / f"{OUTPUT_PREFIX}{args.measured_on}"
        )
        expected_out_name = f"{OUTPUT_PREFIX}{args.measured_on}"
        if out_dir.name != expected_out_name:
            raise Phase2DriverError(
                f"output_directory_must_be_dated:{out_dir.name}!={expected_out_name}"
            )
        capture_root_resolved = args.capture_root.resolve()
        out_resolved = out_dir.resolve()
        try:
            out_resolved.relative_to(capture_root_resolved)
        except ValueError as exc:
            raise Phase2DriverError(
                f"output_must_be_inside_capture_audit_tree:{out_dir}"
            ) from exc
        if out_dir.exists():
            raise Phase2DriverError(
                f"refusing_to_overwrite_output_directory:{out_dir}"
            )

        # Exercise the complete composition against temporary bytes before the
        # protected audit tree gains a dated directory. Logical or ledger-drift
        # failures therefore leave no partial issuance behind.
        with TemporaryDirectory(prefix="tracka-phase2-preflight-") as raw_tmp:
            tmp = Path(raw_tmp)
            _, _, preflight_ledger = _build_ledger(
                envelope=envelope,
                coverage=coverage,
                guard_ms=guard_ms,
                measured_on=args.measured_on,
                evidence_path=evidence_path,
                receipt_dir=tmp,
            )
            aa.write_reissued_ledger(preflight_ledger, tmp / LEDGER_NAME)
            _assert_expected_counts(_ledger_counts(preflight_ledger))

        out_dir.mkdir(parents=True, exist_ok=False)

        _write_json(
            out_dir / "window_analyses.json", [w.to_dict() for w in windows]
        )
        _write_json(out_dir / "latency_envelope.json", envelope)
        _write_json(out_dir / "freshness_envelope.json", coverage)
        latency, freshness, ledger = _build_ledger(
            envelope=envelope,
            coverage=coverage,
            guard_ms=guard_ms,
            measured_on=args.measured_on,
            evidence_path=evidence_path,
            receipt_dir=out_dir,
        )
        verified = aa.write_reissued_ledger(ledger, out_dir / LEDGER_NAME)
        counts = _ledger_counts(verified)
        _assert_expected_counts(counts)
        persisted_coverage = json.loads(
            (out_dir / "freshness_envelope.json").read_text(encoding="utf-8")
        )
        persisted_coverage_sha256 = aa.canonical_mapping_sha256(persisted_coverage)
        if persisted_coverage_sha256 != freshness["coverage_sha256"]:
            raise Phase2DriverError(
                "persisted_freshness_envelope_sha256_mismatch:"
                f"{persisted_coverage_sha256}!={freshness['coverage_sha256']}"
            )
        probabilities = _sample_max_probabilities(int(envelope["session_count"]))
        summary = {
            "schema_version": "v5.tracka-phase2-issuance.v1",
            "status": "ISSUED",
            "declaration": _repo_path(args.declaration),
            "declaration_sha256": declaration["declaration_sha256"],
            "issued_on": args.measured_on,
            "valid_until": VALID_UNTIL,
            "record_class": aa.CBBO_1M_CLASS,
            "session_count": envelope["session_count"],
            "window_count": envelope["window_count"],
            # The sentence that bounds the claim. It is already inside the
            # re-issued ledger, but the summary is what a reader opens first,
            # and a limitation only a second file records is a limitation that
            # gets dropped when the result is quoted.
            "certification_wording": aa.certification_wording(
                int(envelope["session_count"])
            ),
            "sample_max_exceedance_probabilities": probabilities,
            "guard_clock_ms": guard_ms,
            "latency_receipt_sha256": latency["receipt_sha256"],
            "freshness_receipt_sha256": freshness["receipt_sha256"],
            "freshness_coverage_sha256": persisted_coverage_sha256,
            "ledger_sha256": verified["ledger_sha256"],
            "counts": counts,
        }
        _write_json(out_dir / "issuance_summary.json", summary)
        _print_counts(counts)
        print(f"ledger written {out_dir / LEDGER_NAME}")
        return 0
    except (
        aa.ArrivalAnalysisError,
        feature_admission.FeatureAdmissionError,
        TrainingTwinError,
        Phase2DriverError,
        OSError,
        ValueError,
    ) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
