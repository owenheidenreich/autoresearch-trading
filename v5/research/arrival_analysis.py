"""Track-A arrival analysis: from banked capture windows to a signed latency
receipt and a re-issued v5 feature-admission ledger.

This is the Phase-2 machinery for STATUS job 4 (Track-A option-feature arrival
capture).  It consumes the per-record local receipts banked by the attended
capture, recomputes every statistic from the raw rows, and refuses to proceed
when the recomputation disagrees with the capture's own sealed summary — the
receipt is derived from evidence, never copied from it.

Scope rules encoded here rather than remembered:

- A window counts as **evidence** only when its session and window name appear
  in the frozen capture declaration and its summary is healthy.  The 2026-08-05
  midday run is structurally non-evidence (infrastructure verification), which
  this module derives from the declaration rather than special-casing.
- The envelope is the **worst observed** statistic across evidence windows,
  never a mean (declaration ``clock_selection_law``).
- The availability guard is ``L = max(10,000 ms, 4 x worst observed live
  p99)`` — the conservative engineering clock from the gate-chain audit §5.
  It exceeds the declaration's minimum law (max p99) by construction.
- Certification wording is limited to "causal under this fixed guard and the
  observed N-session envelope".  It may not say "worst case": two sessions
  cannot support an extreme-value claim (audit §2.8).
- This module deliberately does **not** call ``reservation.assert_development_only``:
  arrival latency and feed parity evaluate no policy, and the signed forward
  confirmation reservation explicitly permits them on 08-06/08-07.

Model-free and network-free; it only reads files it is given.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v5.research import feature_admission
from v5.research.training_twin import LatencyReceipt, make_latency_receipt


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = (
    REPO_ROOT / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
)
DECLARATION_PATH = CAPTURE_ROOT / "capture_declaration_v6.json"

OPRA_CBBO_1M_FAMILY = "DATABENTO_OPRA_CBBO_1M"
CBBO_1M_CLASS = "CBBOMsg:rtype=193"
HEALTHY_STATUS = "CAPTURED_NO_ORDER_LIVE_SAMPLE"

# Gate-chain audit §5: an explicit conservative engineering choice, not an
# extreme-value estimate.  The floor covers the largest documented example
# (2,335 ms, ThetaData) more than four times; the multiplier scales upward if
# the OPRA result is worse.
GUARD_FLOOR_MS = 10_000.0
GUARD_MULTIPLIER = 4.0

# The exact recorded blocker that the new latency + sparse-minute evidence
# answers, and the prefix of the dependency blocker it unlocks downstream.
NATIVE_FAMILY = "entry.opra_cbbo1m_native.v1"
NATIVE_BLOCKER = (
    "missing_required_receipts:multi-session local receipt-latency distribution"
    "|sparse-minute and freshness receipt"
)
PARENT_BLOCKER_PREFIX = "parent_family_not_admitted:"

CERTIFICATION_WORDING = (
    "causal under this fixed guard and the observed {sessions}-session envelope; "
    "this is an engineering envelope, not a population worst case"
)


class ArrivalAnalysisError(RuntimeError):
    """The capture evidence is missing, unhealthy, or does not reproduce."""


def _quantiles_ns(values: Sequence[int]) -> dict[str, int]:
    """The capture script's own statistic, reproduced method-for-method."""

    array = np.asarray(sorted(values), dtype=np.int64)
    if array.size == 0:
        raise ArrivalAnalysisError("no values to summarize")
    return {
        "min": int(array.min()),
        "p50": int(np.quantile(array, 0.50, method="nearest")),
        "p90": int(np.quantile(array, 0.90, method="nearest")),
        "p99": int(np.quantile(array, 0.99, method="nearest")),
        "max": int(array.max()),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArrivalAnalysisError(f"{label}_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise ArrivalAnalysisError(f"{label}_not_an_object:{path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowAnalysis:
    """One capture window, recomputed from raw rows and cross-checked."""

    session: str
    window: str
    is_evidence: bool
    status: str
    records_total: int
    symbol_count: int
    lag_ns_by_class: Mapping[str, Mapping[str, int]]
    coverage_by_class: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def declared_evidence_windows(
    declaration: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """(session, window) pairs the frozen declaration names as evidence."""

    payload = declaration or _load_json(DECLARATION_PATH, "capture_declaration")
    capture_window = payload.get("capture_window", {})
    sessions = capture_window.get("sessions", [])
    windows = [w.get("name") for w in capture_window.get("windows", [])]
    if not sessions or not windows:
        raise ArrivalAnalysisError("capture_declaration_names_no_windows")
    return tuple((str(s), str(w)) for s in sessions for w in windows)


def analyze_window(
    session: str,
    window: str,
    *,
    capture_root: Path = CAPTURE_ROOT,
    declaration: Mapping[str, Any] | None = None,
) -> WindowAnalysis:
    """Recompute one window's arrival statistics from its raw receipt rows.

    Every recomputed statistic must equal the capture's sealed summary exactly;
    a mismatch means the rows and the summary are not the same evidence, and
    the window is refused rather than partially trusted.
    """

    market_dir = capture_root / session / window / "market"
    summary = _load_json(market_dir / "capture_summary.json", "capture_summary")

    status = str(summary.get("status", ""))
    if status != HEALTHY_STATUS:
        raise ArrivalAnalysisError(f"window_unhealthy:{session}/{window}:{status}")
    hard_stops = summary.get("hard_stops", {})
    if not isinstance(hard_stops, dict) or any(bool(v) for v in hard_stops.values()):
        raise ArrivalAnalysisError(f"hard_stop_recorded:{session}/{window}")

    rows_path = market_dir / "local_receipts.jsonl"
    lags: dict[str, list[int]] = {}
    coverage: dict[str, dict[str, set]] = {}
    total = 0
    try:
        with rows_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                name = f"{row['record_class']}:rtype={row['rtype']}"
                interval_end = row.get("interval_end_unix_ns")
                if interval_end is not None:
                    local_ns = int(row["local_receipt_unix_ns"])
                    # The capture's sealed statistic includes only rows whose
                    # receipt is at or after the interval end (its line
                    # `0 < interval_end_ns <= local_ns`); early rows are
                    # counted separately rather than silently averaged in.
                    slot = coverage.setdefault(
                        name,
                        {
                            "instruments": set(),
                            "instrument_intervals": set(),
                            "early_rows": 0,
                        },
                    )
                    if 0 < int(interval_end) <= local_ns:
                        lags.setdefault(name, []).append(local_ns - int(interval_end))
                    else:
                        slot["early_rows"] += 1
                    slot["instruments"].add(int(row["instrument_id"]))
                    slot["instrument_intervals"].add(
                        (int(row["instrument_id"]), int(interval_end))
                    )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArrivalAnalysisError(f"receipt_rows_unreadable:{rows_path}") from exc

    if total != int(summary.get("records_total", -1)):
        raise ArrivalAnalysisError(
            f"row_count_mismatch:{session}/{window}: rows={total} "
            f"summary={summary.get('records_total')}"
        )

    recomputed = {name: _quantiles_ns(values) for name, values in lags.items()}
    sealed = summary.get("local_receipt_minus_interval_end_ns", {})
    sealed_intervals = {
        name: stats for name, stats in sealed.items() if isinstance(stats, dict)
    }
    if set(recomputed) != set(sealed_intervals):
        raise ArrivalAnalysisError(
            f"lag_class_mismatch:{session}/{window}: "
            f"rows={sorted(recomputed)} summary={sorted(sealed_intervals)}"
        )
    for name, stats in recomputed.items():
        for key, value in stats.items():
            if int(sealed_intervals[name].get(key, value + 1)) != value:
                raise ArrivalAnalysisError(
                    f"lag_stat_mismatch:{session}/{window}:{name}:{key}: "
                    f"recomputed={value} sealed={sealed_intervals[name].get(key)}"
                )

    plan = summary.get("plan", {})
    coverage_out = {
        name: {
            "instruments": len(slot["instruments"]),
            "instrument_intervals": len(slot["instrument_intervals"]),
            "interval_ends": len({iv for _, iv in slot["instrument_intervals"]}),
            "early_rows": int(slot["early_rows"]),
        }
        for name, slot in coverage.items()
    }
    evidence_pairs = declared_evidence_windows(declaration)
    return WindowAnalysis(
        session=str(session),
        window=str(window),
        is_evidence=(str(session), str(window)) in evidence_pairs,
        status=status,
        records_total=total,
        symbol_count=int(plan.get("symbol_count", 0)),
        lag_ns_by_class=recomputed,
        coverage_by_class=coverage_out,
    )


def evidence_envelope(
    windows: Iterable[WindowAnalysis], *, record_class: str = CBBO_1M_CLASS
) -> dict[str, Any]:
    """Worst observed statistics for one record class across evidence windows.

    The declaration's ``clock_selection_law`` is explicit: the admitted clock
    derives from the max across all declared windows and sessions, never the
    mean and never the quiet window alone.
    """

    evidence = [w for w in windows if w.is_evidence]
    if not evidence:
        raise ArrivalAnalysisError("no_evidence_windows: nothing banked yet")
    missing = [
        f"{w.session}/{w.window}" for w in evidence if record_class not in w.lag_ns_by_class
    ]
    if missing:
        raise ArrivalAnalysisError(
            f"record_class_absent:{record_class}: " + ",".join(missing)
        )
    sessions = sorted({w.session for w in evidence})
    stats = [w.lag_ns_by_class[record_class] for w in evidence]
    worst = {key: max(int(s[key]) for s in stats) for key in ("p50", "p90", "p99", "max")}
    return {
        "record_class": record_class,
        "sessions": sessions,
        "session_count": len(sessions),
        "window_count": len(evidence),
        "worst_ns": worst,
        "worst_ms": {key: value / 1e6 for key, value in worst.items()},
        "per_window": {
            f"{w.session}/{w.window}": dict(w.lag_ns_by_class[record_class])
            for w in evidence
        },
    }


def guard_clock_ms(worst_p99_ms: float) -> float:
    """``L = max(10,000 ms, 4 x worst observed live p99)``, audit §5."""

    if worst_p99_ms <= 0.0:
        raise ArrivalAnalysisError("guard_needs_a_positive_p99")
    return max(GUARD_FLOOR_MS, GUARD_MULTIPLIER * float(worst_p99_ms))


def build_cbbo1m_latency_receipt(
    envelope: Mapping[str, Any],
    *,
    measured_on: str,
    valid_until: str,
    evidence_path: str,
) -> LatencyReceipt:
    """Sign the OPRA CBBO-1m arrival envelope as a v5 latency receipt.

    ``valid_until`` is a deliberate caller decision recorded at issuance —
    there is no hidden default lifetime for a two-session measurement.
    """

    if envelope.get("record_class") != CBBO_1M_CLASS:
        raise ArrivalAnalysisError(
            f"envelope_is_not_cbbo1m:{envelope.get('record_class')}"
        )
    session_count = int(envelope.get("session_count", 0))
    if session_count < 1:
        raise ArrivalAnalysisError("latency_receipt_needs_at_least_one_session")
    worst_ms = envelope["worst_ms"]
    return make_latency_receipt(
        source_family=OPRA_CBBO_1M_FAMILY,
        p50_ms=float(worst_ms["p50"]),
        p99_ms=float(worst_ms["p99"]),
        max_ms=float(worst_ms["max"]),
        session_count=session_count,
        measured_on=measured_on,
        valid_until=valid_until,
        evidence_path=evidence_path,
    )


def certification_wording(session_count: int) -> str:
    return CERTIFICATION_WORDING.format(sessions=int(session_count))


def reissue_ledger(
    *,
    legacy_path: Path = feature_admission.LEGACY_LEDGER_PATH,
    latency_receipt: LatencyReceipt,
    receipt_files: Sequence[Path],
    availability_clock_ms: float,
    valid_until: str,
    issued_on: str,
) -> dict[str, Any]:
    """Transform the verified legacy ledger into a v5 ledger with validity.

    Row law, applied mechanically:

    1. The eleven ``entry.opra_cbbo1m_native.v1`` rows barred exactly for the
       missing receipt-latency / sparse-minute evidence admit, carrying the new
       receipt files and the guard clock.
    2. Rows barred only as ``parent_family_not_admitted:<family>`` admit once
       the named family — and every dependency-law parent — is fully admitted,
       to a fixpoint.  Nothing else changes state.
    3. Every admitted row (including the eight legacy admissions) gains the
       declared ``valid_until``; every state change preserves its old blocker
       as ``superseded_barred_reason``.

    The caller must persist with :func:`write_reissued_ledger`, which re-runs
    the full fail-closed verifier on the written bytes.
    """

    if availability_clock_ms < GUARD_FLOOR_MS:
        raise ArrivalAnalysisError(
            f"availability_clock_below_guard_floor:{availability_clock_ms}"
        )
    if latency_receipt.assert_usable(
        source_family=OPRA_CBBO_1M_FAMILY, as_of=issued_on
    ).valid_until != valid_until:
        raise ArrivalAnalysisError("ledger_and_receipt_validity_windows_differ")

    # The receipt compares plain ``YYYY-MM-DD`` strings; the admission verifier
    # requires a timezone-aware instant.  End-of-day UTC makes the two agree
    # exactly, so a feature never outlives its own latency evidence by a day.
    row_valid_until = f"{valid_until}T23:59:59+00:00"

    legacy = feature_admission.verify_ledger(legacy_path)
    new_receipt_entries = []
    for path in receipt_files:
        resolved = Path(path)
        if not resolved.is_file():
            raise ArrivalAnalysisError(f"receipt_file_missing:{resolved}")
        try:
            relative = str(resolved.relative_to(REPO_ROOT))
        except ValueError:
            relative = str(resolved)
        new_receipt_entries.append({"path": relative, "sha256": sha256_file(resolved)})
    if not new_receipt_entries:
        raise ArrivalAnalysisError("reissue_requires_at_least_one_receipt_file")

    rows = [dict(row) for row in legacy["features"]]

    def family_of(row: Mapping[str, Any]) -> str:
        return str(row.get("contract_id", row.get("family", "")))

    def admit(row: dict[str, Any]) -> None:
        row["superseded_barred_reason"] = row.pop("barred_reason")
        row["status"] = feature_admission.ADMITTED
        row["availability_clock_ms"] = float(availability_clock_ms)
        row["receipts"] = list(row.get("receipts", [])) + new_receipt_entries

    for row in rows:
        if (
            row["status"] == feature_admission.BARRED
            and family_of(row) == NATIVE_FAMILY
            and row.get("barred_reason") == NATIVE_BLOCKER
        ):
            admit(row)

    def fully_admitted(family: str) -> bool:
        members = [row for row in rows if family_of(row) == family]
        return bool(members) and all(
            row["status"] == feature_admission.ADMITTED for row in members
        )

    changed = True
    while changed:
        changed = False
        for row in rows:
            reason = str(row.get("barred_reason") or "")
            if row["status"] != feature_admission.BARRED or not reason.startswith(
                PARENT_BLOCKER_PREFIX
            ):
                continue
            named = reason[len(PARENT_BLOCKER_PREFIX):]
            law_parents = feature_admission.DEFAULT_PARENT_FAMILIES.get(
                family_of(row), ()
            )
            if all(fully_admitted(parent) for parent in {named, *law_parents}):
                admit(row)
                changed = True

    for row in rows:
        if row["status"] == feature_admission.ADMITTED:
            row["valid_until"] = row_valid_until

    session_count = int(latency_receipt.session_count)
    payload: dict[str, Any] = {
        "schema_version": "v5.feature-admission-ledger.v1",
        "issued_on": issued_on,
        "reissued_from": {
            "path": str(legacy_path),
            "ledger_sha256": legacy["ledger_sha256"],
        },
        "certification_wording": certification_wording(session_count),
        "availability_guard": {
            "floor_ms": GUARD_FLOOR_MS,
            "multiplier": GUARD_MULTIPLIER,
            "clock_ms": float(availability_clock_ms),
            "latency_receipt_sha256": latency_receipt.receipt_sha256,
        },
        "features": rows,
    }
    payload["ledger_sha256"] = feature_admission.ledger_sha256(payload)
    return payload


def write_reissued_ledger(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Persist the re-issued ledger and prove the bytes pass the verifier."""

    target = Path(path)
    if target.exists():
        raise ArrivalAnalysisError(f"refusing_to_overwrite_ledger:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        return feature_admission.verify_ledger(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
