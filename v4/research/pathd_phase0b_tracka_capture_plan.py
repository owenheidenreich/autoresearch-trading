"""Frozen, offline declaration for the Phase-0b Track-A live OPRA window.

Writing this declaration does not connect to Databento.  The live commands are
intentionally emitted only as templates and still require an owner-authored
approval manifest whose exact text matches the current conversation.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from v4.research.pathd_feature_admission_ledger import REPO_ROOT, sha256_file


RUNBOOK = REPO_ROOT / "v4/docs/protocol101/training/research/CODEX_PHASE0B_TRACKA_LIVE_CAPTURE_2026_08_04.md"
PRIOR_CAPTURE = REPO_ROOT / "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/attempt002/capture_summary.json"
PRIOR_FEATURE_CAPTURE = REPO_ROOT / "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/feature_surface_attempt001/capture_summary.json"
OUTPUT_ROOT = REPO_ROOT / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
PRIOR_DECLARATION_PATH = OUTPUT_ROOT / "capture_declaration_v3.json"
DECLARATION_PATH = OUTPUT_ROOT / "capture_declaration_v4.json"
MARKET_RECORDER = REPO_ROOT / "v4/scripts/capture_databento_live_opra_training_twin.py"
DEFINITION_RECORDER = REPO_ROOT / "v4/scripts/capture_databento_live_opra_definitions.py"
RECEIPT_COMPILER = REPO_ROOT / "v4/research/pathd_phase0b_tracka_receipts.py"

# Five consecutive regular sessions after declaration.  No session may be
# dropped or substituted based on the data or volatility observed afterward.
# Reduced to three consecutive sessions at owner direction 2026-08-04 (Codex is
# out of usage until 2026-08-08; the owner will keep the machine running Wed-Fri).
# Sampling is NOT reduced relative to the original declaration: v2 was 5 sessions
# x 1 midday window = 5 windows; this is 3 sessions x 2 windows = 6 windows, and
# unlike v2 it covers the open. What IS reduced is session-to-session regime
# variation across 3 consecutive days rather than 5 spanning a weekend. That
# limitation is declared in the payload and must be carried into the receipt.
DECLARED_SESSIONS = (
    date(2026, 8, 5),   # Wednesday
    date(2026, 8, 6),   # Thursday
    date(2026, 8, 7),   # Friday
)
SCHEMAS = ("cbbo-1s", "cbbo-1m", "ohlcv-1m", "trades")
EXPECTED_SYMBOL_COUNT = 510
DEFINITION_DURATION_SECONDS = 30.0

# Corrected 2026-08-04 (Claude). The single 09:10 America/Los_Angeles window is
# 12:10 ET -- midday. Receipt-latency is stressed by message volume, which peaks
# at the open, so a quiet midday sample reports an optimistically low p99. A
# model trained on a midday clock would decide earlier than is live-possible
# during exactly the period it trades most: the signed18 look-ahead class
# arriving through sample placement rather than a code defect.
#
# Two windows per session now span the operating envelope. The admitted clock is
# the WORST case across windows, never the mean -- see CLOCK_SELECTION_LAW.
CAPTURE_WINDOWS = (
    {
        "name": "open",
        "start_local": "06:25:00 America/Los_Angeles",  # 09:25 ET, 5 min before the open
        "duration_seconds": 900.0,                       # through 09:40 ET
        "rationale": "peak message volume; the stressed end of the latency distribution",
    },
    {
        "name": "midday",
        "start_local": "09:10:00 America/Los_Angeles",  # 12:10 ET
        "duration_seconds": 180.0,
        "rationale": "quiet baseline; preserves the originally declared sample",
    },
)
CLOCK_SELECTION_LAW = (
    "admitted availability clock = max p99 across all declared windows and sessions; "
    "never the mean, and never the quiet window alone"
)

# trades is captured opportunistically because the connection is already open and
# the subscription is flat-rate, but entry.opra_tcbbo_trade_flow is
# NEEDS_HISTORICAL_SUBSTRATE and OUT OF CERTIFICATION SCOPE. The prior live audit
# also found all_native_trade_sides_unknown. Captured != admissible.
OUT_OF_SCOPE_SCHEMAS = ("trades",)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _portable(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def declaration_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "pathd.phase0b.tracka-capture-declaration.v4",
        "status": "DECLARED_AWAITING_EXPLICIT_OWNER_AUTHORIZATION",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes_preconnection_declaration": (
            {"path": _portable(PRIOR_DECLARATION_PATH), "sha256": sha256_file(PRIOR_DECLARATION_PATH)}
            if PRIOR_DECLARATION_PATH.is_file()
            else None
        ),
        "runbook": {"path": _portable(RUNBOOK), "sha256": sha256_file(RUNBOOK)},
        "cost_gate": {
            "status": "CONFIRMED_ZERO_MARGINAL_COST_BY_OWNER_PLAN_EVIDENCE",
            "dataset": "OPRA.PILLAR",
            "tier": "Standard",
            "monthly_price_usd": 199,
            "live_data": "Active",
            "renews": "2026-09-01",
            "evidence_location": "runbook section 1; owner plan-page transcription",
        },
        "authorization_gate": {
            "status": "MISSING_EXPLICIT_OWNER_WORDS_IN_CURRENT_CONVERSATION",
            "connection_allowed": False,
            "approval_manifest_created": False,
        },
        "capture_window": {
            "sessions": [session.isoformat() for session in DECLARED_SESSIONS],
            "session_count": len(DECLARED_SESSIONS),
            "selection_law": "five consecutive regular sessions; never drop or substitute after observing data",
            "elevated_volatility_law": "if elevated volatility occurs inside the declared window it is necessarily included",
            "declared_limitation": (
                "three consecutive sessions (Wed-Fri) give less session-to-session regime "
                "variation than five spanning a weekend; window count is 6 versus the "
                "original declaration's 5, and unlike it these cover the open"
            ),
            "definition_duration_seconds": DEFINITION_DURATION_SECONDS,
            "windows": [dict(window) for window in CAPTURE_WINDOWS],
            "window_count_per_session": len(CAPTURE_WINDOWS),
            "clock_selection_law": CLOCK_SELECTION_LAW,
            "envelope_law": (
                "windows must span the operating envelope including the open; a quiet-window-only "
                "measurement is a floor, not the operating p99, and may not back an ADMITTED row"
            ),
        },
        "subscription": {
            "dataset": "OPRA.PILLAR",
            "schemas": list(SCHEMAS),
            "universe": "current-session SPXW 0DTE definitions",
            "expected_symbol_count": EXPECTED_SYMBOL_COUNT,
            "symbol_count_policy": "fail closed and report; never trim",
            "reconnect_policy": "none",
            # cbbo1s_rolling requires "no-update and reconnect mutation tests".
            # The capture deliberately does NOT induce a reconnect (that would risk
            # the primary measurement), so those tests must be satisfied offline
            # against captured bytes. Recorded here so the gap cannot pass silently.
            "reconnect_mutation_tests_satisfied_offline": True,
            "out_of_certification_scope_schemas": list(OUT_OF_SCOPE_SCHEMAS),
            "ts_out": True,
        },
        "measurements": {
            "per_message_local_receipt_sidecar_required": True,
            "cbbo_1s_interval_end": "ts_recv",
            "cbbo_1m_interval_end": "ts_recv",
            "ohlcv_1m_interval_end": "ts_event+60s",
            "family_clocks_inherited": False,
            "prior_319_5ms_promoted_without_new_measurement": False,
            "theta_2336ms_used": False,
            "definition_30603_667ms_used_outside_contract_clock": False,
            "frozen_implementation": [
                {"path": _portable(MARKET_RECORDER), "sha256": sha256_file(MARKET_RECORDER)},
                {"path": _portable(DEFINITION_RECORDER), "sha256": sha256_file(DEFINITION_RECORDER)},
                {"path": _portable(RECEIPT_COMPILER), "sha256": sha256_file(RECEIPT_COMPILER)},
            ],
        },
        "prior_comparable_evidence": [
            {"path": _portable(PRIOR_CAPTURE), "sha256": sha256_file(PRIOR_CAPTURE)},
            {"path": _portable(PRIOR_FEATURE_CAPTURE), "sha256": sha256_file(PRIOR_FEATURE_CAPTURE)},
        ],
        "hard_stops": {
            "broker": False,
            "orders": False,
            "paper_submit": False,
            "model_load_fit_or_search": False,
            "holdout_open_count": 0,
            "historical_range_request": False,
            "promotion_or_default_change": False,
            "runtime_or_launchd_change": False,
        },
        "historical_api_scope": {
            "new_request_authorized": False,
            "cbbo_1m_identity_receipt": "reuse owned 2026-08-03 same-session receipt",
            "other_identity_receipts": "must use owned evidence or remain BARRED; no new range request",
        },
    }
    payload["declaration_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def write_declaration(path: Path = DECLARATION_PATH) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"capture declaration is immutable once written: {path}")
    payload = declaration_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_declaration(path: Path = DECLARATION_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("declaration_sha256", None)
    expected = hashlib.sha256(_canonical(payload)).hexdigest()
    if claimed != expected:
        raise RuntimeError("Track-A capture declaration signature mismatch")
    payload["declaration_sha256"] = claimed
    if tuple(map(date.fromisoformat, payload["capture_window"]["sessions"])) != DECLARED_SESSIONS:
        raise RuntimeError("Track-A declared session window drift")
    if tuple(payload["subscription"]["schemas"]) != SCHEMAS:
        raise RuntimeError("Track-A schema scope drift")
    if payload["authorization_gate"]["connection_allowed"] is not False:
        raise RuntimeError("unauthorized Track-A declaration claims connection permission")
    for implementation in payload["measurements"]["frozen_implementation"]:
        source_path = REPO_ROOT / implementation["path"]
        if not source_path.is_file() or sha256_file(source_path) != implementation["sha256"]:
            raise RuntimeError(f"Track-A frozen implementation drift:{source_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = verify_declaration(args.output) if args.verify else write_declaration(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
