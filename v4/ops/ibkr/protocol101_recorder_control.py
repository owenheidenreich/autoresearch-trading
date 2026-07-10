"""Health, finalization, and integrity controls for recorder-first evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import sys
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.ibkr_market_capture import (
    CapturePaths,
    MANIFEST_VERSION,
    QUALITY_VERSION,
    apply_option_capture_event,
    atomic_write_json,
    clean_json,
    file_sha256,
    iso_utc,
    iter_capture_rows,
)


NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("health", "finalize", "audit", "recover-checkpoints"):
        item = sub.add_parser(name)
        item.add_argument("--session", required=True)
        item.add_argument("--capture-id", default=None)
        item.add_argument("--capture-root", type=Path, default=Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture")
        item.add_argument("--max-heartbeat-age-seconds", type=float, default=10.0)
        item.add_argument("--require-live", action="store_true")
        item.add_argument("--require-ladder", action="store_true")
        item.add_argument("--require-complete-session", action="store_true")
        item.add_argument("--port", type=int, default=4002)
    return parser.parse_args()


def paths_for(args: argparse.Namespace) -> CapturePaths:
    capture_id = args.capture_id or f"protocol101-recorder-{args.session}"
    return CapturePaths.for_capture(args.capture_root, args.session, capture_id)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def expected_minutes(session: str) -> list[str]:
    day = datetime.strptime(session, "%Y-%m-%d").date()
    current = datetime.combine(day, datetime.min.time(), tzinfo=NY).replace(hour=9, minute=30)
    return [(current + timedelta(minutes=index)).isoformat() for index in range(390)]


def checkpoint_overlay_path(paths: CapturePaths) -> Path:
    return paths.root / "derived_checkpoint_overlays.jsonl"


def checkpoint_quality(payload: dict[str, Any]) -> dict[str, Any]:
    contracts = payload.get("contracts") if isinstance(payload.get("contracts"), list) else []
    required = ("contract_id", "strike", "right", "market_data_type_name")
    tradable = sum(
        1
        for contract in contracts
        if isinstance(contract, dict)
        and contract.get("bid") is not None
        and contract.get("ask") is not None
        and contract.get("last_received_timestamp_utc") is not None
    )
    incomplete = len(contracts) < 42 or any(
        not isinstance(contract, dict) or any(contract.get(field) is None for field in required)
        for contract in contracts
    )
    return {
        "contract_count": len(contracts),
        "tradable_count": tradable,
        "incomplete": incomplete,
        "passing": not incomplete and tradable >= 40,
    }


def recover_missing_checkpoints(paths: CapturePaths, session: str) -> dict[str, Any]:
    targets = [(minute, datetime.fromisoformat(minute).astimezone(timezone.utc) + timedelta(minutes=1)) for minute in expected_minutes(session)]
    target_index = 0
    raw_minutes: set[str] = set()
    raw_quality: dict[str, list[dict[str, Any]]] = {}
    derived: dict[str, dict[str, Any]] = {}
    option_state: dict[str, dict[str, Any]] = {}
    active_contract_ids: list[str] = []
    latest_index: dict[str, dict[str, Any]] = {}
    last_received: datetime | None = None

    def snapshot_until(cutoff: datetime, *, inclusive: bool) -> None:
        nonlocal target_index
        while target_index < len(targets):
            minute, decision = targets[target_index]
            if decision > cutoff or (decision == cutoff and not inclusive):
                break
            spx = latest_index.get("SPX")
            vix = latest_index.get("VIX")
            contracts = [dict(option_state[key]) for key in active_contract_ids if key in option_state]
            if spx and vix and contracts:
                try:
                    atm = int(round(float(spx.get("price")) / 5.0) * 5)
                except (TypeError, ValueError):
                    atm = None
                derived[minute] = {
                    "schema_version": "Protocol101DerivedCheckpointV1",
                    "session": session,
                    "event_type": "ladder_checkpoint",
                    "received_timestamp_utc": decision.isoformat().replace("+00:00", "Z"),
                    "payload": {
                        "completed_minute_et": minute,
                        "decision_time_et": (datetime.fromisoformat(minute) + timedelta(minutes=1)).isoformat(),
                        "spx": dict(spx),
                        "vix": dict(vix),
                        "atm_strike": atm,
                        "contracts": contracts,
                        "contract_count": len(contracts),
                        "opening_context_policy": "09:30_completed_candle_feeds_09:31_decision",
                        "recovery_provenance": {
                            "schema_version": "Protocol101CheckpointRecoveryV1",
                            "source": str(paths.events),
                            "policy": "latest_raw_ibkr_observation_received_at_or_before_decision_time",
                        },
                    },
                }
            target_index += 1

    for _, row in iter_capture_rows(paths.events):
        if row is None:
            continue
        received = parse_time(row.get("received_timestamp_utc"))
        if received is not None:
            snapshot_until(received, inclusive=False)
            last_received = received
        event_type = row.get("event_type")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if event_type == "ladder_checkpoint" and payload.get("completed_minute_et"):
            minute = str(payload["completed_minute_et"])
            raw_minutes.add(minute)
            raw_quality.setdefault(minute, []).append(checkpoint_quality(payload))
        elif event_type == "ladder_definition":
            definitions = payload.get("contracts") if isinstance(payload.get("contracts"), list) else []
            active_contract_ids = []
            for definition in definitions:
                if not isinstance(definition, dict) or not definition.get("contract_id"):
                    continue
                key = str(definition["contract_id"])
                active_contract_ids.append(key)
                option_state.setdefault(key, {}).update(definition)
        elif event_type in {"option_update", "option_delta"}:
            apply_option_capture_event(option_state, row)
            contract_id = str(payload.get("contract_id") or "")
            if contract_id and contract_id in option_state:
                option_state[contract_id]["last_received_timestamp_utc"] = row.get("received_timestamp_utc")
        elif event_type == "index_update" and payload.get("symbol") in {"SPX", "VIX"}:
            latest_index[str(payload["symbol"])] = dict(payload)
    if last_received is not None:
        snapshot_until(last_received, inclusive=True)

    missing = sorted(set(expected_minutes(session)) - raw_minutes)
    bad = sorted(
        minute
        for minute, rows in raw_quality.items()
        if minute in set(expected_minutes(session)) and not any(bool(row.get("passing")) for row in rows)
    )
    targets_to_recover = sorted(set(missing) | set(bad))
    recovered = []
    for minute in targets_to_recover:
        if minute not in derived:
            continue
        row = derived[minute]
        row["payload"]["replacement_for_raw_checkpoint"] = minute in bad
        row["payload"]["recovery_provenance"]["reason"] = (
            "replace_incomplete_or_untradable_raw_checkpoint" if minute in bad else "missing_raw_checkpoint"
        )
        recovered.append(row)
    overlay = checkpoint_overlay_path(paths)
    temporary = overlay.with_suffix(overlay.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in recovered:
            handle.write(json.dumps(clean_json(row), sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(overlay)
    summary = {
        "schema_version": "Protocol101CheckpointRecoverySummaryV1",
        "status": "pass" if len(recovered) == len(targets_to_recover) else "fail",
        "session": session,
        "raw_checkpoint_count": len(raw_minutes),
        "missing_raw_checkpoint_minutes": missing,
        "replaced_raw_checkpoint_minutes": bad,
        "recovered_checkpoint_minutes": [row["payload"]["completed_minute_et"] for row in recovered],
        "overlay_path": str(overlay),
        "raw_source_modified": False,
    }
    atomic_write_json(paths.root / "checkpoint_recovery_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def scan(paths: CapturePaths, session: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    parse_errors = 0
    schema_errors = 0
    sequence_errors = 0
    previous_sequence = 0
    checkpoint_minutes: list[str] = []
    producer_ids: set[str] = set()
    connection_epochs: set[tuple[str, int]] = set()
    delayed_rows = 0
    order_rows = 0
    raw_checkpoint_quality: dict[str, list[dict[str, Any]]] = {}
    first_received = None
    last_received = None
    for _, row in iter_capture_rows(paths.events):
        if row is None:
            parse_errors += 1
            continue
        if row.get("schema_version") != "IBKRMarketCaptureV1" or row.get("session") != session:
            schema_errors += 1
        try:
            sequence = int(row.get("sequence", 0))
        except (TypeError, ValueError):
            sequence = 0
        if sequence <= previous_sequence:
            sequence_errors += 1
        previous_sequence = max(previous_sequence, sequence)
        event_type = str(row.get("event_type") or "missing")
        counts[event_type] += 1
        producer = str(row.get("producer_instance_id") or "")
        if producer:
            producer_ids.add(producer)
            try:
                connection_epochs.add((producer, int(row.get("connection_epoch", 0))))
            except (TypeError, ValueError):
                pass
        received = row.get("received_timestamp_utc")
        first_received = first_received or received
        last_received = received or last_received
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if event_type == "ladder_checkpoint":
            minute = payload.get("completed_minute_et")
            if minute:
                checkpoint_minutes.append(str(minute))
                raw_checkpoint_quality.setdefault(str(minute), []).append(checkpoint_quality(payload))
            contracts = payload.get("contracts") if isinstance(payload.get("contracts"), list) else []
            for contract in contracts:
                if isinstance(contract, dict) and "delayed" in str(contract.get("market_data_type_name", "")):
                    delayed_rows += 1
        if "delayed" in str(payload.get("market_data_type_name", "")):
            delayed_rows += 1
        if event_type.startswith("order") or event_type.startswith("paper_order"):
            order_rows += 1
    raw_checkpoint_minutes = list(checkpoint_minutes)
    recovered_checkpoint_minutes: list[str] = []
    effective_quality: dict[str, dict[str, Any]] = {}
    for minute, rows in raw_checkpoint_quality.items():
        effective_quality[minute] = max(
            rows,
            key=lambda row: (
                int(bool(row.get("passing"))),
                int(row.get("tradable_count") or 0),
                int(row.get("contract_count") or 0),
            ),
        )
    overlay = checkpoint_overlay_path(paths)
    if overlay.exists():
        for line in overlay.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            minute = payload.get("completed_minute_et")
            replacement = bool(payload.get("replacement_for_raw_checkpoint"))
            if not minute or (str(minute) in effective_quality and not replacement):
                continue
            quality = checkpoint_quality(payload)
            effective_quality[str(minute)] = quality
            recovered_checkpoint_minutes.append(str(minute))
    checkpoint_contract_counts = [int(row.get("contract_count") or 0) for row in effective_quality.values()]
    checkpoint_tradable_counts = [int(row.get("tradable_count") or 0) for row in effective_quality.values()]
    incomplete_checkpoint_rows = sum(bool(row.get("incomplete")) for row in effective_quality.values())
    low_tradable_checkpoint_rows = sum(int(row.get("tradable_count") or 0) < 40 for row in effective_quality.values())
    expected = expected_minutes(session)
    unique_checkpoints = sorted(effective_quality)
    missing = sorted(set(expected) - set(unique_checkpoints))
    extra = sorted(set(unique_checkpoints) - set(expected))
    return {
        "events_path": str(paths.events),
        "events_exist": paths.events.exists(),
        "events_bytes": paths.events.stat().st_size if paths.events.exists() else 0,
        "event_counts": dict(sorted(counts.items())),
        "valid_rows": sum(counts.values()),
        "parse_errors": parse_errors,
        "schema_errors": schema_errors,
        "sequence_errors": sequence_errors,
        "last_sequence": previous_sequence,
        "producer_instance_count": len(producer_ids),
        "connection_epoch_count": len(connection_epochs),
        "first_received_at_utc": first_received,
        "last_received_at_utc": last_received,
        "checkpoint_count": len(unique_checkpoints),
        "raw_checkpoint_count": len(set(raw_checkpoint_minutes)),
        "derived_checkpoint_count": len(recovered_checkpoint_minutes),
        "recovered_checkpoint_minutes": recovered_checkpoint_minutes,
        "checkpoint_duplicate_rows": len(raw_checkpoint_minutes) - len(set(raw_checkpoint_minutes)),
        "minimum_checkpoint_contract_count": min(checkpoint_contract_counts) if checkpoint_contract_counts else 0,
        "minimum_tradable_contract_count": min(checkpoint_tradable_counts) if checkpoint_tradable_counts else 0,
        "incomplete_checkpoint_rows": incomplete_checkpoint_rows,
        "low_tradable_checkpoint_rows": low_tradable_checkpoint_rows,
        "trace_required_quote_fields_present": bool(checkpoint_contract_counts) and incomplete_checkpoint_rows == 0 and low_tradable_checkpoint_rows == 0,
        "expected_checkpoint_count": 390,
        "missing_checkpoint_minutes": missing,
        "extra_checkpoint_minutes": extra,
        "opening_minute_present": bool(expected and expected[0] in unique_checkpoints),
        "closing_minute_present": bool(expected and expected[-1] in unique_checkpoints),
        "delayed_market_data_rows": delayed_rows,
        "order_event_rows": order_rows,
        "broker_order_endpoint_called": False,
    }


def health(args: argparse.Namespace) -> int:
    paths = paths_for(args)
    state = load_json(paths.state)
    last_heartbeat = parse_time(state.get("last_heartbeat_at_utc"))
    age = (datetime.now(timezone.utc) - last_heartbeat).total_seconds() if last_heartbeat else None
    checks = {
        "state_exists": paths.state.exists(),
        "events_exists": paths.events.exists(),
        "events_nonempty": paths.events.exists() and paths.events.stat().st_size > 0,
        "heartbeat_fresh": age is not None and age <= float(args.max_heartbeat_age_seconds),
        "api_port_open": port_open(args.port),
        "recorder_connected": bool(state.get("connected")),
        "live_feed_confirmed": bool(state.get("live_feed_confirmed")) if args.require_live else True,
        "ladder_ready": int(state.get("subscribed_contracts") or 0) >= 42 if args.require_ladder else True,
        "no_write_errors": int(state.get("write_errors") or 0) == 0,
        "no_subscription_errors": int(state.get("subscription_errors") or 0) == 0,
        "no_order_endpoint": state.get("broker_order_endpoint_called") is not True,
    }
    payload = {
        "schema_version": "Protocol101RecorderHealthV1",
        "status": "pass" if all(checks.values()) else "fail",
        "checked_at_utc": iso_utc(),
        "session": args.session,
        "capture_id": paths.root.name,
        "heartbeat_age_seconds": age,
        "checks": checks,
        "state": state,
    }
    out = paths.root / f"health_{datetime.now(NY).strftime('%H%M%S')}.json"
    atomic_write_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


def finalize(args: argparse.Namespace) -> int:
    paths = paths_for(args)
    evidence = scan(paths, args.session)
    state = load_json(paths.state)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "session": args.session,
        "capture_id": paths.root.name,
        "finalized_at_utc": iso_utc(),
        "immutable_source": str(paths.events),
        "events_sha256": file_sha256(paths.events) if paths.events.exists() else None,
        "events_bytes": evidence["events_bytes"],
        "last_sequence": evidence["last_sequence"],
        "event_counts": evidence["event_counts"],
        "producer_instance_count": evidence["producer_instance_count"],
        "connection_epoch_count": evidence["connection_epoch_count"],
        "requested_market_data_type": "live",
        "broker_order_endpoint_called": False,
        "real_money_trading": False,
        "state_at_finalize": state,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(paths.root / "ibkr_capture_manifest.json", manifest)
    paths.checksums.write_text(f"{manifest['events_sha256']}  market_events.jsonl\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["events_sha256"] else 2


def audit(args: argparse.Namespace) -> int:
    paths = paths_for(args)
    evidence = scan(paths, args.session)
    state = load_json(paths.state)
    checks = {
        "events_present": bool(evidence["events_exist"] and evidence["events_bytes"] > 0),
        "raw_json_valid": evidence["parse_errors"] == 0,
        "schema_valid": evidence["schema_errors"] == 0,
        "sequence_monotonic": evidence["sequence_errors"] == 0,
        "opening_context": evidence["opening_minute_present"],
        "complete_regular_session": evidence["checkpoint_count"] == 390 and not evidence["missing_checkpoint_minutes"],
        "live_market_data": bool(state.get("live_feed_confirmed")) and evidence["delayed_market_data_rows"] == 0,
        "ladder_definitions": int(evidence["event_counts"].get("ladder_definition", 0)) > 0,
        "option_updates": int(evidence["event_counts"].get("option_update", 0)) > 0,
        "trace_required_quote_fields": bool(evidence["trace_required_quote_fields_present"]),
        "tradable_ladder_coverage": int(evidence.get("minimum_tradable_contract_count") or 0) >= 40,
        "no_orders": evidence["order_event_rows"] == 0 and state.get("broker_order_endpoint_called") is not True,
        "no_write_errors": int(state.get("write_errors") or 0) == 0,
    }
    if not args.require_complete_session:
        checks["complete_regular_session"] = True
    quality = {
        "schema_version": QUALITY_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
        "session": args.session,
        "capture_id": paths.root.name,
        "generated_at_utc": iso_utc(),
        "opening_context_ready": evidence["opening_minute_present"],
        "missing_opening_minutes": 0 if evidence["opening_minute_present"] else 1,
        "checks": checks,
        "evidence": evidence,
    }
    atomic_write_json(paths.root / "ibkr_capture_quality.json", quality)
    print(json.dumps(quality, indent=2, sort_keys=True))
    return 0 if quality["status"] == "pass" else 2


def main() -> int:
    args = parse_args()
    if args.command == "health":
        return health(args)
    if args.command == "finalize":
        return finalize(args)
    if args.command == "audit":
        return audit(args)
    if args.command == "recover-checkpoints":
        return 0 if recover_missing_checkpoints(paths_for(args), args.session)["status"] == "pass" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
