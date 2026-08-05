"""Verify a Track-A capture window actually produced its receipt.

Run after a window fires. Exits non-zero if the capture did not produce the one
thing Track A exists for: a per-record local-receipt arrival distribution for
the OPRA families, measured against each family's own interval-end clock.

Silence is not success. A launchd job that never fired, a job that fired and
refused, and a job that connected but captured nothing all leave an output
directory that looks broadly similar, so each is checked separately and named.

No network, no broker, no paid data. Reads local artifacts only.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
# Summary keys are "ClassName:rtype=N", where ClassName is a databento library
# detail that has already changed once between captures. rtype is the stable
# semantic identity, so match on that.
#
# These two families are the blocking receipt: rtype 193 (cbbo-1m) gates 65
# entry features, and rtype 192 (cbbo-1s) additionally gates 29 exit features
# through entry.opra_cbbo1s_rolling.v1.
REQUIRED_INTERVAL_END_RTYPES = {193: "cbbo-1m", 192: "cbbo-1s"}


def _family_for_rtype(interval: dict[str, Any], rtype: int) -> tuple[str, Any] | None:
    suffix = f"rtype={rtype}"
    for key, value in interval.items():
        if str(key).endswith(suffix):
            return str(key), value
    return None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_window(session: str, window: str) -> dict[str, Any]:
    out = ROOT / session / window
    log = ROOT / "run_logs" / f"{session}_{window}.log"
    problems: list[str] = []
    notes: list[str] = []

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if not log_text:
        problems.append(f"NO_RUN_LOG: {log} -- the launchd job did not fire at all")
    elif "REFUSED" in log_text:
        line = next(
            (l for l in log_text.splitlines() if "REFUSED" in l), "REFUSED"
        )
        problems.append(f"WRAPPER_REFUSED: {line.strip()}")

    definitions = _load(out / "definitions" / "definition_capture_summary.json")
    if definitions is None:
        problems.append("NO_DEFINITION_SUMMARY: current-session universe was never captured")
    else:
        if definitions.get("status") != "CAPTURED_CURRENT_SESSION_DEFINITIONS":
            problems.append(f"DEFINITION_STATUS: {definitions.get('status')}")
        count = definitions.get("current_session_definition_count")
        notes.append(f"current-session definitions: {count}")
        if not count:
            problems.append("EMPTY_DEFINITION_UNIVERSE")

    quotes = _load(out / "market" / "capture_summary.json")
    if quotes is None:
        problems.append("NO_CAPTURE_SUMMARY: the market window produced no summary")
        return _result(session, window, problems, notes)

    plan = quotes.get("plan") or {}
    symbol_count = plan.get("symbol_count")
    expected = plan.get("expected_symbol_count")
    notes.append(f"symbols: {symbol_count} (expected {expected})")
    if expected is not None and symbol_count != expected:
        problems.append(f"UNIVERSE_DRIFT: {symbol_count} != {expected}")

    interval = quotes.get("local_receipt_minus_interval_end_ns") or {}
    if not interval:
        problems.append(
            "NO_INTERVAL_END_LATENCY: capture ran but produced no arrival "
            "distribution -- this is the whole purpose of Track A"
        )
    for rtype, label in REQUIRED_INTERVAL_END_RTYPES.items():
        found = _family_for_rtype(interval, rtype)
        if found is None:
            problems.append(
                f"MISSING_ARRIVAL_FAMILY: {label} (rtype={rtype}); "
                f"observed families: {sorted(interval)}"
            )
            continue
        key, quantiles = found
        p99 = (quantiles or {}).get("p99")
        if not isinstance(p99, int) or p99 <= 0:
            problems.append(f"INVALID_P99: {key}:{p99}")
        else:
            notes.append(f"{label} ({key}) p99: {p99 / 1e6:.1f} ms")

    receipts = out / "market" / "local_receipts.jsonl"
    if not receipts.is_file() or receipts.stat().st_size == 0:
        problems.append("NO_LOCAL_RECEIPTS: per-record sidecar is missing or empty")
    else:
        rows = sum(1 for _ in receipts.open("r", encoding="utf-8"))
        notes.append(f"local receipt rows: {rows:,}")
        if rows < 100:
            problems.append(f"THIN_CAPTURE: only {rows} records")

    stops = quotes.get("hard_stops") or {}
    if not stops:
        problems.append("NO_HARD_STOPS_BLOCK: cannot confirm the capture stayed no-order")
    for flag, value in sorted(stops.items()):
        if value not in (False, 0):
            problems.append(f"SAFETY: hard_stops.{flag} = {value!r}")

    return _result(session, window, problems, notes)


def _result(session: str, window: str, problems: list[str], notes: list[str]) -> dict[str, Any]:
    return {
        "session": session,
        "window": window,
        "status": "CAPTURE_RECEIPT_PRESENT" if not problems else "CAPTURE_INCOMPLETE",
        "problems": problems,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=date.today().isoformat())
    parser.add_argument("--window", action="append", default=None)
    args = parser.parse_args()
    windows = args.window or ["open", "midday"]
    results = [verify_window(args.session, window) for window in windows]
    print(json.dumps({"results": results}, indent=2))
    return 0 if all(r["status"] == "CAPTURE_RECEIPT_PRESENT" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
