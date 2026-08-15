"""Audit Protocol101 VIX warm-up trace infrastructure.

This is infrastructure-only. It checks whether the local burned-day IBKR and
historical sources can support 5m/15m VIX lookback features at the 09:32 ET
decision without changing the frozen v1.4 contract or admitting any feature.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_capture_replay import build_replay_inputs


SCHEMA_VERSION = "Protocol101VixWarmupTraceInfrastructureAuditV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_vix_warmup_trace_infra_audit")
SESSIONS = ("2026-06-30", "2026-07-01", "2026-07-02")
NY = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--capture-root", type=Path, default=Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture")
    parser.add_argument("--decision-start-et", default="09:12")
    parser.add_argument("--decision-end-et", default="09:35")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def local_hhmm(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        stamp = pd.Timestamp(ts)
    except Exception:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(NY).strftime("%H:%M")


def index_source_summary(session: str, symbol: str) -> dict[str, Any]:
    path = Path(f"data/raw/index/{symbol.lower()}_1m/{session}.official_{symbol.lower()}.parquet")
    if not path.exists():
        return {"symbol": symbol, "path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    times = pd.to_datetime(frame["event_time"], utc=True, errors="coerce").dropna().sort_values()
    first = times.iloc[0] if len(times) else None
    last = times.iloc[-1] if len(times) else None
    first_et = local_hhmm(first)
    # A decision at t can use a 5m/15m exact lookback only if t-lookback exists.
    return {
        "symbol": symbol,
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "first_event_time_utc": first.isoformat() if first is not None else None,
        "last_event_time_utc": last.isoformat() if last is not None else None,
        "first_event_time_et": first_et,
        "last_event_time_et": local_hhmm(last),
        "first_decision_with_5m_exact_lookup_et": "09:35" if first_et == "09:30" else None,
        "first_decision_with_15m_exact_lookup_et": "09:45" if first_et == "09:30" else None,
        "supports_09_32_5m_lookup": False if first_et == "09:30" else None,
        "supports_09_32_15m_lookup": False if first_et == "09:30" else None,
    }


def ibkr_source_summary(capture_root: Path, session: str, start_et: str, end_et: str) -> dict[str, Any]:
    events = capture_root / session / f"protocol101-recorder-{session}" / "market_events.jsonl"
    if not events.exists():
        return {"session": session, "events": str(events), "exists": False}
    checkpoints, index_state = build_replay_inputs(
        events,
        session=session,
        decision_start_et=start_et,
        decision_end_et=end_et,
    )
    first = checkpoints[0].get("decision_time_et") if checkpoints else None
    last = checkpoints[-1].get("decision_time_et") if checkpoints else None
    return {
        "session": session,
        "events": str(events),
        "exists": True,
        "requested_start_et": start_et,
        "requested_end_et": end_et,
        "checkpoint_count": len(checkpoints),
        "first_checkpoint_decision_et": local_hhmm(first),
        "last_checkpoint_decision_et": local_hhmm(last),
        "compact_index_rows": len(index_state.rows),
        "supports_pre_09_31_checkpoints": bool(checkpoints and local_hhmm(first) < "09:31"),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    blockers = []
    for session in SESSIONS:
        historical = {
            "spx": index_source_summary(session, "SPX"),
            "vix": index_source_summary(session, "VIX"),
        }
        ibkr = ibkr_source_summary(args.capture_root, session, args.decision_start_et, args.decision_end_et)
        if historical["vix"].get("supports_09_32_15m_lookup") is False:
            blockers.append(f"{session}:historical_vix_starts_at_09_30_no_09_32_15m_lookup")
        if ibkr.get("supports_pre_09_31_checkpoints") is False:
            blockers.append(f"{session}:ibkr_replay_no_pre_09_31_checkpoints")
        sessions.append({"session": session, "historical": historical, "ibkr": ibkr})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "blocked" if blockers else "pass",
        "decision": "vix_warmup_infra_added_but_current_sources_do_not_close_initial_15m_gap"
        if blockers
        else "vix_warmup_sources_ready",
        "blockers": blockers,
        "sessions": sessions,
        "implemented_infrastructure": {
            "build_replay_inputs_decision_start_et": True,
            "run_protocol101_capture_replay_decision_start_et_arg": True,
            "default_behavior_preserved": "09:31 start remains default",
            "new_prefix_traces_built": False,
            "reason_new_prefix_not_built": "current burned-day sources cannot provide pre-09:31 paired checkpoints / historical pre-09:30 VIX context",
        },
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
            "sealed_market_data_read": False,
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    report = [
        "# Protocol101 VIX Warm-Up Trace Infrastructure Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Blockers: `{len(blockers)}`",
        "",
        "Infrastructure was added so replay inputs can request a `decision_start_et` earlier than 09:31 while preserving the default 09:31 behavior.",
        "",
        "The current local burned-day sources still do not close the initial VIX-change gap: historical official SPX/VIX files start at 09:30 ET, and the IBKR replay input builder still first produces usable checkpoints at 09:31 ET on these captures.",
        "",
        "No VIX feature was admitted and no L0/L1/L3 audit was rerun.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
