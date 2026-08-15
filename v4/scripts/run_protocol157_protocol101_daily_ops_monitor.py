"""Protocol 157: daily Protocol101 live/paper operations monitor.

This is the operator-facing status page for the morning automation. It does
not download market data and does not call any broker order endpoint. It reads
local logs/artifacts, optionally reads an IBKR paper account summary, and
summarizes startup, live capture, order activity, paper PnL, contracts, and
failure reasons in one place.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import errno
import html
import json
import os
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, flatten_trade_event, load_trade_log
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import analyze_session_rows, resolve_trade_log
from v4.scripts.run_protocol156_ibkr_autostart_observability import (
    DEFAULT_ENTITLEMENT_SUMMARY,
    LABELS,
    collect_entitlement_summary,
    launchd_status,
)
from v4.ops.ibkr.ibkr_account_snapshot import fetch_ibkr_account_snapshot


DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor")
DEFAULT_PROTOCOL147_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session")
DEFAULT_RUNTIME_FLAG = Path("v4/runtime/protocol101_paper_order_enablement.json")
PACIFIC = ZoneInfo("America/Los_Angeles")
CONTRACT_MULTIPLIER = 100.0
DAILY_LAUNCHD_LABELS = tuple(LABELS)
FIVEDAY_LAUNCHD_LABELS = (
    "com.autoresearch.protocol101.fiveday.ibgateway.paper",
    "com.autoresearch.protocol101.fiveday.paper-preflight",
    "com.autoresearch.protocol101.fiveday.paper-session",
    "com.autoresearch.protocol101.fiveday.daily-monitor",
    "com.autoresearch.protocol101.fiveday.recovery-watchdog",
    "com.autoresearch.protocol101.fiveday.paper-shutdown",
    "com.autoresearch.protocol101.fiveday.evidence-postsession",
)
LAUNCHD_PROFILES = {
    "daily": DAILY_LAUNCHD_LABELS,
    "fiveday": FIVEDAY_LAUNCHD_LABELS,
}
BENIGN_RISK_REASONS = {
    "candidate_set_built",
    "candidate_set_blocked_insufficient_live_index_context",
    "candidate_set_blocked_outside_time_bucket",
    "entry_bridge_blocked_paper_submit_no_order_submitted",
    "entry_bridge_pass_live_entry_intent_shadow_logged",
    "entry_bridge_pass_no_entry_intents_to_dry_run",
    "entry_bridge_pass_no_entry_intents_to_paper_submit",
    "insufficient_live_index_context",
    "live_capture_pass",
    "no_candidates",
    "no_entry_intent",
    "outside_time_bucket",
    "paper_account_state",
    "ready_for_protocol101_no_order_live_capture",
}
RESOLVED_BY_CURRENT_LIVE_DATA = {
    "blocked_live_subscriptions_delayed_plumbing_passed",
}
TIMELINE_NOISE_EVENTS = {
    "heartbeat",
    "market_snapshot",
    "candidate_set",
    "paper_account_state",
}
TIMELINE_NOISE_REASONS = BENIGN_RISK_REASONS | RESOLVED_BY_CURRENT_LIVE_DATA | {
    "live_snapshot",
    "no_candidates",
}
TRANSIENT_FILE_WRITE_ERRNOS = {
    errno.EAGAIN,
    getattr(errno, "EDEADLK", 11),
}


def safe_write_text(path: Path, text: str, *, attempts: int = 6, sleep_seconds: float = 0.25) -> None:
    """Atomically write monitor artifacts, retrying transient macOS file-provider deadlocks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = max(1, int(attempts))
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for attempt in range(1, attempts + 1):
        try:
            tmp_path.write_text(text)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            if exc.errno in TRANSIENT_FILE_WRITE_ERRNOS and attempt < attempts:
                time.sleep(max(0.0, float(sleep_seconds)) * attempt)
                continue
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--protocol147-root", type=Path, default=DEFAULT_PROTOCOL147_ROOT)
    parser.add_argument("--runtime-flag", type=Path, default=DEFAULT_RUNTIME_FLAG)
    parser.add_argument("--runtime-state", type=Path, default=Path("v4/runtime/protocol101_live_paper_state.json"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    parser.add_argument("--event-limit", type=int, default=250)
    parser.add_argument("--skip-launchd", action="store_true")
    parser.add_argument("--skip-ibkr-account-snapshot", action="store_true")
    parser.add_argument("--ibkr-account-host", default="127.0.0.1")
    parser.add_argument("--ibkr-account-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-account-client-id", type=int, default=257)
    parser.add_argument("--ibkr-account-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--ibkr-account-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trade_log = resolve_trade_log(
        trade_log=args.trade_log,
        root=args.trade_log_root,
        session=args.session,
        run_id=args.run_id,
    )
    rows = load_trade_log(trade_log)
    if not rows:
        raise SystemExit(f"no rows found in {trade_log}")
    analysis = analyze_session_rows(rows, trade_log=trade_log)
    session = str(analysis["session"])
    run_id = str(analysis["run_id"])
    protocol147_dir = resolve_protocol147_dir(args.protocol147_root, session=session, run_id=run_id)
    launchd = {} if args.skip_launchd else collect_launchd()
    entitlement = collect_entitlement_summary(DEFAULT_ENTITLEMENT_SUMMARY)
    ibkr_account_snapshot = (
        {}
        if args.skip_ibkr_account_snapshot
        else fetch_ibkr_account_snapshot(
            host=args.ibkr_account_host,
            ports=args.ibkr_account_ports,
            client_id=int(args.ibkr_account_client_id),
            timeout_seconds=float(args.ibkr_account_timeout_seconds),
            account_id=args.ibkr_account_id,
        )
    )
    payload = build_monitor_payload(
        trade_log=trade_log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=protocol147_dir,
        launchd=launchd,
        entitlement=entitlement,
        ibkr_account_snapshot=ibkr_account_snapshot,
        runtime_flag=load_json(args.runtime_flag),
        runtime_state=load_json(args.runtime_state),
        event_limit=max(25, int(args.event_limit)),
        refresh_seconds=max(5, int(args.refresh_seconds)),
    )
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_html = args.out_root / "latest_daily_monitor.html"
    latest_summary = args.out_root / "latest_summary.json"
    session_alias_html = out_dir / "daily_monitor.html"
    payload["outputs"] = {
        "summary": str(out_dir / "summary.json"),
        "report": str(out_dir / "report.md"),
        "monitor_html": str(latest_html),
        "dated_monitor_alias": str(session_alias_html),
        "latest_summary": str(latest_summary),
    }
    summary_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    safe_write_text(out_dir / "summary.json", summary_text)
    safe_write_text(latest_summary, summary_text)
    write_report(out_dir / "report.md", payload)
    write_html(latest_html, payload, [flatten_trade_event(row) for row in rows])
    write_monitor_alias(session_alias_html, latest_html, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "monitor_html": str(latest_html),
                "dated_monitor_alias": str(session_alias_html),
                "report": str(out_dir / "report.md"),
                "summary": str(out_dir / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not payload["decision"].startswith("blocked_monitor_") else 1


def build_monitor_payload(
    *,
    trade_log: Path,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    protocol147_dir: Path | None,
    launchd: dict[str, Any],
    entitlement: dict[str, Any],
    runtime_flag: dict[str, Any],
    ibkr_account_snapshot: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
    event_limit: int = 250,
    refresh_seconds: int = 30,
) -> dict[str, Any]:
    protocol147_summary = load_json(protocol147_dir / "summary.json") if protocol147_dir else {}
    capture_summaries = collect_capture_summaries(protocol147_dir)
    shadow_logs = sorted({str(path) for path in analysis.get("linked_shadow_logs", [])})
    shadow_logs.extend(str(item["shadow_log"]) for item in capture_summaries if item.get("shadow_log"))
    shadow = summarize_shadow_logs(shadow_logs)
    live_evidence = current_live_trade_log_evidence(rows)
    account_snapshot = _object(ibkr_account_snapshot)
    paper = summarize_paper_activity(rows, analysis=analysis, ibkr_account_snapshot=account_snapshot)
    position = summarize_current_position(runtime_state or {}, paper=paper)
    startup = summarize_startup(
        rows=rows,
        launchd=launchd,
        entitlement=entitlement,
        protocol147_summary=protocol147_summary,
        capture_summaries=capture_summaries,
        analysis=analysis,
        live_evidence=live_evidence,
    )
    failures = summarize_failures(
        rows=rows,
        launchd=launchd,
        entitlement=entitlement,
        analysis=analysis,
        protocol147_summary=protocol147_summary,
        shadow=shadow,
        startup=startup,
        live_evidence=live_evidence,
    )
    decision = decide_monitor(
        analysis=analysis,
        startup=startup,
        shadow=shadow,
        paper=paper,
        failures=failures,
    )
    trader_status = summarize_trader_status(
        rows=rows,
        decision=decision,
        startup=startup,
        shadow=shadow,
        paper=paper,
        position=position,
        failures=failures,
        ibkr_account_snapshot=account_snapshot,
    )
    return {
        "protocol": "157_protocol101_daily_ops_monitor",
        "decision": decision,
        "generated_at_pacific": datetime.now(PACIFIC).isoformat(),
        "paid_data_downloaded": False,
        "real_money_trading": False,
        "live_orders": False,
        "broker_order_endpoint_called": paper["broker_order_endpoint_called_rows"] > 0,
        "paper_orders_submitted": paper["paper_orders_submitted"],
        "session": analysis["session"],
        "run_id": analysis["run_id"],
        "source_trade_log": str(trade_log),
        "protocol147_dir": str(protocol147_dir) if protocol147_dir else None,
        "protocol147_summary": protocol147_summary,
        "current_live_trade_log_evidence": live_evidence,
        "launchd": launchd,
        "entitlement": entitlement,
        "ibkr_account_snapshot": account_snapshot,
        "runtime_flag": summarize_runtime_flag(runtime_flag),
        "runtime_state": summarize_runtime_state(runtime_state or {}),
        "monitor_config": {
            "scope": "single_session_today_only",
            "event_limit": int(event_limit),
            "refresh_seconds": int(refresh_seconds),
        },
        "startup": startup,
        "paper": paper,
        "current_position": position,
        "trader_status": trader_status,
        "shadow": shadow,
        "failures": failures,
        "analysis": analysis,
        "next_action": next_action(decision, startup=startup, paper=paper),
    }


def collect_launchd() -> dict[str, Any]:
    import os

    uid = os.getuid()
    labels = sorted(set(DAILY_LAUNCHD_LABELS) | set(FIVEDAY_LAUNCHD_LABELS))
    return {label: launchd_status(label, uid=uid) for label in labels}


def resolve_protocol147_dir(root: Path, *, session: str, run_id: str) -> Path | None:
    direct = root / session / run_id
    if direct.exists():
        return direct
    return None


def collect_capture_summaries(protocol147_dir: Path | None) -> list[dict[str, Any]]:
    if protocol147_dir is None or not protocol147_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(protocol147_dir.rglob("*summary.json")):
        if "capture" not in str(path.parent) and path.name != "ibkr-live-capture_summary.json":
            continue
        payload = load_json(path)
        if not payload:
            continue
        rows.append(
            {
                "path": str(path),
                "decision": payload.get("decision"),
                "captured_rows": int(payload.get("captured_rows") or 0),
                "shadow_log": payload.get("shadow_log"),
                "market_time": payload.get("market_time"),
                "blocked_reason": payload.get("blocked_reason"),
                "broker_order_endpoint_called": (payload.get("no_order_guarantee") or {}).get("broker_order_endpoint_called"),
                "order_intent_non_null_rows": (payload.get("no_order_guarantee") or {}).get("order_intent_non_null_rows"),
                "feed_probe": payload.get("feed_probe") or {},
                "shadow_parity": payload.get("shadow_parity") or {},
            }
        )
    return rows


def summarize_shadow_logs(paths: list[str]) -> dict[str, Any]:
    seen: set[str] = set()
    logs: list[dict[str, Any]] = []
    total_rows = 0
    live_rows = 0
    delayed_rows = 0
    action_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    latest_contract_counts: Counter[str] = Counter()
    latest_observation_time = None
    latest_spx = None
    order_intent_non_null = 0
    first_time = None
    last_time = None
    for raw in paths:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = Path(raw)
        counts = Counter()
        row_count = 0
        if path.exists():
            for row in read_jsonl(path):
                row_count += 1
                total_rows += 1
                decision_time = row.get("decision_time")
                if decision_time:
                    first_time = min(first_time, decision_time) if first_time else decision_time
                    last_time = max(last_time, decision_time) if last_time else decision_time
                router = _object(row.get("router"))
                market_data_type = str(router.get("market_data_type") or "")
                live_rows += int(market_data_type == "live")
                delayed_rows += int("delayed" in market_data_type)
                action = str(_object(row.get("decision")).get("action") or "missing")
                state = str(row.get("position_state") or "missing")
                contract_id = str(row.get("contract_id") or "missing")
                action_counts[action] += 1
                state_counts[state] += 1
                contract_counts[contract_id] += 1
                if decision_time and (latest_observation_time is None or str(decision_time) > str(latest_observation_time)):
                    latest_observation_time = decision_time
                    latest_spx = _object(row.get("context")).get("spx")
                    latest_contract_counts = Counter({contract_id: 1})
                elif decision_time and str(decision_time) == str(latest_observation_time):
                    latest_contract_counts[contract_id] += 1
                counts[action] += 1
                order_intent_non_null += int(row.get("order_intent") is not None)
        logs.append({"path": raw, "exists": path.exists(), "rows": row_count, "action_counts": dict(sorted(counts.items()))})
    return {
        "logs": logs,
        "total_rows": total_rows,
        "live_rows": live_rows,
        "delayed_rows": delayed_rows,
        "order_intent_non_null_rows": order_intent_non_null,
        "action_counts": dict(sorted(action_counts.items())),
        "position_state_counts": dict(sorted(state_counts.items())),
        "top_contracts_observed": [{"contract_id": key, "rows": value} for key, value in contract_counts.most_common(20)],
        "latest_contracts_observed": [{"contract_id": key, "rows": value} for key, value in latest_contract_counts.most_common(30)],
        "latest_spx": latest_spx,
        "latest_observation_time": latest_observation_time,
        "first_decision_time": first_time,
        "last_decision_time": last_time,
    }


def summarize_paper_activity(
    rows: list[dict[str, Any]],
    *,
    analysis: dict[str, Any],
    ibkr_account_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_counts = Counter(str(row.get("event_type") or "missing") for row in rows)
    latest_account = latest_account_state(rows)
    contracts = Counter()
    quantity_by_contract: Counter[str] = Counter()
    submitted = []
    blocked = Counter()
    broker_rows = 0
    for row in rows:
        event_type = str(row.get("event_type") or "")
        broker_rows += int(bool(row.get("broker_order_endpoint_called")))
        if event_type in {"paper_order_blocked", "risk_gate", "paper_error"}:
            reason = str((_object(row.get("risk_gate")).get("reason") or row.get("blocked_reason") or "missing"))
            if reason:
                blocked[reason] += 1
        if event_type in {"paper_order_submitted", "paper_entry_fill", "paper_exit_fill", "paper_exit_submitted"}:
            key = contract_key(row)
            contracts[key] += 1
            qty = _float(_object(row.get("order")).get("quantity") or _object(row.get("order")).get("filled")) or 0.0
            quantity_by_contract[key] += int(qty)
            submitted.append(flatten_trade_event(row))
    pnl = analysis.get("paper_fill_pnl") or {}
    realized = _float(latest_account.get("realized_daily_pnl"))
    return {
        "event_counts": dict(sorted(event_counts.items())),
        "market_snapshot_rows": int(event_counts.get("market_snapshot", 0)),
        "model_decision_rows": int(event_counts.get("model_decision", 0)),
        "candidate_set_rows": int(event_counts.get("candidate_set", 0)),
        "paper_orders_submitted": int(event_counts.get("paper_order_submitted", 0)),
        "paper_entry_fills": int(event_counts.get("paper_entry_fill", 0)),
        "paper_exit_fills": int(event_counts.get("paper_exit_fill", 0)),
        "broker_order_endpoint_called_rows": broker_rows,
        "closed_trades": int(pnl.get("closed_trades") or 0),
        "open_trades": int(pnl.get("open_trades") or 0),
        "reconstructed_closed_pnl": float(pnl.get("total_pnl") or 0.0),
        "realized_daily_pnl": realized,
        "latest_account": latest_account,
        "account_source": account_source(latest_account=latest_account, ibkr_account_snapshot=ibkr_account_snapshot or {}),
        "blocked_reason_counts": dict(blocked.most_common()),
        "contracts_traded": [{"contract": key, "events": value, "quantity_sum": quantity_by_contract[key]} for key, value in contracts.most_common(30)],
        "recent_order_events": submitted[-30:],
    }


def current_live_trade_log_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_market = latest_market_snapshot_summary(rows)
    latest_candidate = latest_candidate_set_summary(rows)
    latest_decision = latest_model_decision_summary(rows)
    latest_account = latest_account_state(rows)
    live_market_ready = bool(latest_market.get("is_live"))
    context = _object(latest_candidate.get("live_index_context"))
    context_span = _float(context.get("span_minutes"))
    context_rows = int(_float(context.get("row_count")) or 0)
    opening_context_ready = context.get("opening_context_ready")
    context_ready = bool(
        context_span is not None
        and context_span >= 30.0
        and context_rows > 0
        and opening_context_ready is not False
    )
    candidate_count = _float(latest_candidate.get("candidate_count"))
    return {
        "live_market_ready": live_market_ready,
        "live_capture_pass": live_market_ready,
        "live_context_ready": context_ready,
        "live_context_span_minutes": context_span,
        "live_context_rows": context_rows,
        "opening_context_ready": opening_context_ready,
        "missing_opening_minutes": context.get("missing_opening_minutes"),
        "latest_market_timestamp": latest_market.get("timestamp"),
        "latest_decision_timestamp": latest_decision.get("timestamp"),
        "latest_candidate_timestamp": latest_candidate.get("timestamp"),
        "latest_model_action": latest_decision.get("action"),
        "latest_model_reason": latest_decision.get("reason"),
        "candidate_count": int(candidate_count or 0) if candidate_count is not None else None,
        "option_quote_count": int(_float(latest_market.get("option_quote_count")) or 0),
        "spx_market_data_type": latest_market.get("spx_market_data_type"),
        "vix_market_data_type": latest_market.get("vix_market_data_type"),
        "account_snapshot_ready": _float(latest_account.get("equity")) is not None,
        "account_id_redacted": latest_account.get("account_id_redacted"),
    }


def summarize_startup(
    *,
    rows: list[dict[str, Any]],
    launchd: dict[str, Any],
    entitlement: dict[str, Any],
    protocol147_summary: dict[str, Any],
    capture_summaries: list[dict[str, Any]],
    analysis: dict[str, Any],
    live_evidence: dict[str, Any],
) -> dict[str, Any]:
    startup_data = analysis.get("startup_and_data") or {}
    launchd_profile, expected_labels = active_launchd_profile(launchd)
    loaded = [label for label in expected_labels if _object(launchd.get(label)).get("loaded")]
    launchd_all_loaded = bool(expected_labels) and len(loaded) == len(expected_labels)
    live_market_ready = bool(live_evidence.get("live_market_ready"))
    entitlement_ready = bool(entitlement.get("decision") == "pass") or live_market_ready
    effective_entitlement_decision = entitlement.get("decision")
    if entitlement.get("decision") not in (None, "pass") and live_market_ready:
        effective_entitlement_decision = "pass_current_live_paper_log"
    risk_reasons = analysis.get("risk_reason_counts") or {}
    parity_ready_from_trade_log = int(risk_reasons.get("ready_for_protocol101_no_order_live_capture") or 0) > 0
    capture_passes = sum(
        1
        for item in capture_summaries
        if item.get("decision") == "pass" and int(item.get("captured_rows") or 0) > 0
    )
    live_capture_pass = (
        bool(startup_data.get("live_capture_pass"))
        or int(protocol147_summary.get("live_capture_passes") or 0) > 0
        or capture_passes > 0
        or live_market_ready
    )
    return {
        "launchd_all_loaded": launchd_all_loaded,
        "launchd_profile": launchd_profile,
        "launchd_loaded_count": len(loaded),
        "launchd_expected_count": len(expected_labels),
        "launchd_loaded_labels": loaded,
        "launchd_expected_labels": list(expected_labels),
        "entitlement_decision": entitlement.get("decision"),
        "effective_entitlement_decision": effective_entitlement_decision,
        "entitlement_resolved_by_live_trade_log": bool(entitlement.get("decision") not in (None, "pass") and live_market_ready),
        "entitlement_ready": entitlement_ready,
        "ibkr_connected": entitlement.get("ibkr_connected") or startup_data.get("ibkr_connected") or live_market_ready,
        "ibkr_port": entitlement.get("ibkr_port"),
        "live_parity_ready": bool(startup_data.get("live_parity_ready")) or parity_ready_from_trade_log or live_market_ready,
        "live_capture_pass": live_capture_pass,
        "observed_capture_passes": capture_passes,
        "protocol147_decision": protocol147_summary.get("decision"),
        "protocol147_cycles_run": int(protocol147_summary.get("cycles_run") or 0),
        "protocol147_live_capture_passes": int(protocol147_summary.get("live_capture_passes") or 0),
        "protocol147_live_capture_blocks": int(protocol147_summary.get("live_capture_blocks") or 0),
        "trade_log_valid": (analysis.get("validation") or {}).get("status") == "pass",
    }


def summarize_failures(
    *,
    rows: list[dict[str, Any]],
    launchd: dict[str, Any],
    entitlement: dict[str, Any],
    analysis: dict[str, Any],
    protocol147_summary: dict[str, Any],
    shadow: dict[str, Any],
    startup: dict[str, Any],
    live_evidence: dict[str, Any],
) -> dict[str, Any]:
    reasons = Counter()
    resolved = Counter()
    expected_launchd_labels = set(startup.get("launchd_expected_labels") or ())
    live_market_ready = bool(live_evidence.get("live_market_ready"))
    for label, row in launchd.items():
        if label not in expected_launchd_labels:
            continue
        if launchd and not row.get("loaded"):
            key = f"launchd_not_loaded:{label}"
            if live_market_ready:
                resolved[key] += 1
            else:
                reasons[key] += 1
        if row.get("last_exit_code") not in (None, 0):
            key = f"launchd_last_exit_nonzero:{label}"
            active_now = bool(row.get("loaded")) and str(row.get("state") or "").lower() in {"active", "running"}
            resolved_by_live_log = live_market_ready and any(
                part in label for part in {"preflight", "ibgateway", "paper-session", "daily-monitor"}
            )
            if active_now or resolved_by_live_log:
                resolved[key] += 1
            else:
                reasons[key] += 1
    if entitlement.get("decision") not in (None, "pass"):
        key = f"entitlement:{entitlement.get('decision')}"
        if live_market_ready:
            resolved[key] += 1
        else:
            reasons[key] += 1
    if entitlement.get("blocked_reason"):
        key = f"entitlement_blocked:{entitlement.get('blocked_reason')}"
        if live_market_ready:
            resolved[key] += 1
        else:
            reasons[key] += 1
    for key, value in (analysis.get("risk_reason_counts") or {}).items():
        if is_resolved_by_later_live_session(str(key), rows=rows):
            resolved[f"trade_log:{key}"] += int(value)
            continue
        if is_resolved_by_later_entry_bridge(str(key), rows=rows):
            resolved[f"trade_log:{key}"] += int(value)
            continue
        if is_resolved_by_current_live_data(str(key), startup=startup, shadow=shadow):
            resolved[f"trade_log:{key}"] += int(value)
            continue
        if is_benign_risk_reason(str(key)):
            continue
        reasons[f"trade_log:{key}"] += int(value)
    if protocol147_summary.get("decision", "").startswith("blocked_"):
        reasons[f"protocol147:{protocol147_summary.get('decision')}"] += 1
    if int(shadow.get("order_intent_non_null_rows") or 0) > 0:
        reasons["unexpected_order_intent_in_shadow"] += int(shadow["order_intent_non_null_rows"])
    return {
        "reason_counts": dict(reasons.most_common()),
        "resolved_reason_counts": dict(resolved.most_common()),
        "has_failures": bool(reasons),
    }


def active_launchd_profile(launchd: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    if not launchd:
        return "none", ()
    scored: list[tuple[int, int, str, tuple[str, ...]]] = []
    for profile, labels in LAUNCHD_PROFILES.items():
        loaded_count = sum(1 for label in labels if _object(launchd.get(label)).get("loaded"))
        running_count = sum(1 for label in labels if str(_object(launchd.get(label)).get("state") or "").lower() == "running")
        profile_preference = 1 if profile == "fiveday" else 0
        scored.append((loaded_count, running_count + profile_preference, profile, labels))
    loaded_count, _running_count, profile, labels = max(scored, key=lambda item: (item[0], item[1]))
    if loaded_count == 0:
        return "none", ()
    return profile, labels


def is_benign_risk_reason(reason: str) -> bool:
    if reason in BENIGN_RISK_REASONS:
        return True
    return reason.startswith("pass_") or reason.startswith("entry_bridge_pass_")


def is_resolved_by_later_live_session(reason: str, *, rows: list[dict[str, Any]]) -> bool:
    if reason not in {"no_valid_spxw_nbbo_quotes"}:
        return False
    last_block_idx = None
    for idx, row in enumerate(rows):
        row_reason = str(_object(row.get("risk_gate")).get("reason") or row.get("blocked_reason") or "")
        model_reason = str(_object(row.get("model_decision")).get("reason") or "")
        if row_reason == reason or model_reason == reason:
            last_block_idx = idx
    if last_block_idx is None:
        return False
    for row in rows[last_block_idx + 1 :]:
        event_type = str(row.get("event_type") or "")
        if event_type == "market_snapshot":
            snapshot = _object(row.get("market_snapshot"))
            option_nbbo = _object(snapshot.get("option_nbbo"))
            if int(_float(option_nbbo.get("quote_count")) or 0) > 0:
                return True
        if event_type == "candidate_set":
            extra_count = row.get("candidate_count")
            if extra_count is not None:
                return True
            if row.get("candidate_gate_diagnostics"):
                return True
    return False


def is_resolved_by_later_entry_bridge(reason: str, *, rows: list[dict[str, Any]]) -> bool:
    if reason not in {
        "protocol158_exception",
        "entry_bridge_blocked_protocol158_exception",
        "blocked_ibkr_connection",
        "preflight_failed",
        "persistent_ibkr_connection_failed",
    }:
        return False
    last_block_idx = None
    for idx, row in enumerate(rows):
        row_reason = str(_object(row.get("risk_gate")).get("reason") or row.get("blocked_reason") or "")
        model_reason = str(_object(row.get("model_decision")).get("reason") or "")
        if row_reason == reason or model_reason == reason:
            last_block_idx = idx
    if last_block_idx is None:
        return False
    for row in rows[last_block_idx + 1 :]:
        if row.get("mode") != "paper-submit":
            continue
        if row.get("event_type") == "market_snapshot":
            snapshot = _object(row.get("market_snapshot"))
            context = _object(snapshot.get("context"))
            if context.get("source") == "ibkr_live":
                return True
        if row.get("event_type") == "candidate_set" and row.get("candidate_gate_diagnostics"):
            return True
        if row.get("event_type") == "model_decision":
            model = _object(row.get("model_decision"))
            if model.get("reason") != "protocol158_exception":
                return True
    return False


def is_resolved_by_current_live_data(reason: str, *, startup: dict[str, Any], shadow: dict[str, Any]) -> bool:
    if reason not in RESOLVED_BY_CURRENT_LIVE_DATA:
        return False
    return bool(startup.get("live_capture_pass")) and int(shadow.get("live_rows") or 0) > 0


def decide_monitor(
    *,
    analysis: dict[str, Any],
    startup: dict[str, Any],
    shadow: dict[str, Any],
    paper: dict[str, Any],
    failures: dict[str, Any],
) -> str:
    if (analysis.get("validation") or {}).get("status") != "pass":
        return "blocked_monitor_trade_log_validation_failed"
    if paper["broker_order_endpoint_called_rows"] > 0:
        return "observe_paper_broker_activity_logged"
    if paper["market_snapshot_rows"] > 0 and paper["model_decision_rows"] > 0:
        return "pass_live_entry_monitor_ready"
    if startup["live_capture_pass"] and shadow["live_rows"] > 0:
        return "pass_live_shadow_monitor_ready"
    if startup["entitlement_ready"] and not startup["live_capture_pass"]:
        return "blocked_monitor_entitlements_ready_but_no_live_capture"
    if failures["has_failures"]:
        return "blocked_monitor_startup_or_data_failures"
    return "observe_monitor_waiting_for_session_activity"


def next_action(decision: str, *, startup: dict[str, Any], paper: dict[str, Any]) -> str:
    if decision == "observe_paper_broker_activity_logged":
        return "Review fills, PnL, open positions, and contract list before the next session."
    if decision == "pass_live_entry_monitor_ready":
        return "Review live entry decisions, risk gates, and paper-submit state; no-entry rows are normal when the model chooses to wait."
    if decision == "pass_live_shadow_monitor_ready":
        return "Keep the scheduled session running; use this monitor after each session and enable paper order submission only after the explicit order gate is cleared."
    if decision == "blocked_monitor_entitlements_ready_but_no_live_capture":
        return "Rerun Protocol147 after the stale Protocol124 wiring fix; the data feed is ready but capture did not start."
    if decision == "blocked_monitor_trade_log_validation_failed":
        return "Fix the invalid JSONL event row before using the run for any paper-order decision."
    if paper["open_trades"] > 0:
        return "Confirm all paper positions are flat before close and reconcile any broker fills."
    return "Wait for the next scheduled run or inspect the failure counts shown in this monitor."


def summarize_trader_status(
    *,
    rows: list[dict[str, Any]],
    decision: str,
    startup: dict[str, Any],
    shadow: dict[str, Any],
    paper: dict[str, Any],
    position: dict[str, Any],
    failures: dict[str, Any],
    ibkr_account_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest_market = latest_market_snapshot_summary(rows)
    latest_decision = latest_model_decision_summary(rows)
    latest_candidate = latest_candidate_set_summary(rows)
    ladder = summarize_latest_ladder(shadow, rows=rows)
    account = effective_account_state(paper=paper, ibkr_account_snapshot=ibkr_account_snapshot or {})
    freshness = summarize_freshness(
        latest_market=latest_market,
        latest_decision=latest_decision,
        latest_candidate=latest_candidate,
    )
    active_blockers = failures.get("reason_counts") or {}
    current_monitor_blocked = decision.startswith("blocked_monitor_")
    status = "In position" if position.get("status") == "holding" else "Flat and watching"
    if current_monitor_blocked and active_blockers:
        status = "Blocked"
    elif freshness.get("state") == "stale":
        status = "Stale data"
    elif freshness.get("state") == "lagging":
        status = "Lagging data"
    elif paper.get("paper_orders_submitted"):
        status = "Paper order activity logged"
    feed = "Live feed" if latest_market.get("is_live") else "Feed not current"
    if current_monitor_blocked and active_blockers:
        feed = "Blocked"
    elif latest_market.get("is_live") and freshness.get("market_state") == "stale":
        feed = "Stale live feed"
    elif latest_market.get("is_live") and freshness.get("market_state") == "lagging":
        feed = "Lagging live feed"
    last_reason = latest_decision.get("reason") or latest_candidate.get("reason")
    last_action = latest_decision.get("action") or "wait"
    if last_action == "wait" and last_reason in {"no_candidates", "no_entry_intent"}:
        gate = _object(latest_candidate.get("gate_diagnostics"))
        if gate.get("filter_reason") == "below_min_edge" and _float(gate.get("max_edge")) is not None:
            last_readable = (
                f"Waiting: best edge {float(gate['max_edge']):.1f} is below the live entry gate "
                f"{float(gate.get('min_edge') or 0.0):.1f}"
            )
        elif gate.get("filter_reason") == "outside_time_bucket":
            last_readable = f"Waiting: {gate.get('time_bucket')} is outside the allowed entry buckets"
        else:
            last_readable = "Waiting: no contract currently clears the model/risk gate"
    elif last_action == "wait":
        last_readable = f"Waiting: {last_reason or 'no trade signal'}"
    elif last_action == "enter":
        last_readable = "Entry signal emitted"
    elif last_action == "exit":
        last_readable = "Exit signal emitted"
    else:
        last_readable = f"{last_action}: {last_reason or 'latest model action'}"
    return {
        "status": status,
        "feed": feed,
        "decision": decision,
        "latest_market": latest_market,
        "latest_model_decision": latest_decision,
        "latest_candidate_set": latest_candidate,
        "latest_ladder": ladder,
        "freshness": freshness,
        "last_readable": last_readable,
        "account": {
            "starting_cash": _float(account.get("starting_cash")),
            "cash": _float(account.get("cash")),
            "equity": _float(account.get("equity")),
            "realized_daily_pnl": _float(account.get("realized_daily_pnl")),
            "unrealized_pnl": _float(account.get("unrealized_pnl")),
            "available_funds": _float(account.get("available_funds")),
            "buying_power": _float(account.get("buying_power")),
            "open_positions": int(_float(account.get("open_positions")) or 0),
            "account_id_redacted": account.get("account_id_redacted"),
            "source": account.get("source"),
            "pnl_source": account.get("pnl_source"),
            "snapshot_status": account.get("snapshot_status"),
            "snapshot_checked_at_utc": account.get("snapshot_checked_at_utc"),
        },
        "market_open_plumbing": {
            "launchd": f"{startup.get('launchd_loaded_count')}/{startup.get('launchd_expected_count')}",
            "entitlements": startup.get("effective_entitlement_decision") or startup.get("entitlement_decision"),
            "live_capture_pass": startup.get("live_capture_pass"),
            "live_rows": shadow.get("live_rows"),
            "delayed_rows": shadow.get("delayed_rows"),
        },
    }


def summarize_freshness(
    *,
    latest_market: dict[str, Any],
    latest_decision: dict[str, Any],
    latest_candidate: dict[str, Any],
) -> dict[str, Any]:
    market_age = age_seconds(latest_market.get("timestamp"))
    decision_age = age_seconds(latest_decision.get("timestamp") or latest_candidate.get("timestamp"))
    candidate_age = age_seconds(latest_candidate.get("timestamp"))
    market_state = freshness_state(market_age, warn_seconds=90, stale_seconds=180)
    decision_state = freshness_state(decision_age, warn_seconds=150, stale_seconds=300)
    state = "fresh"
    if "stale" in {market_state, decision_state}:
        state = "stale"
    elif "lagging" in {market_state, decision_state}:
        state = "lagging"
    return {
        "state": state,
        "market_state": market_state,
        "decision_state": decision_state,
        "market_age_seconds": market_age,
        "decision_age_seconds": decision_age,
        "candidate_age_seconds": candidate_age,
        "market_age": age_label(market_age),
        "decision_age": age_label(decision_age),
        "candidate_age": age_label(candidate_age),
    }


def freshness_state(value: float | None, *, warn_seconds: int, stale_seconds: int) -> str:
    if value is None:
        return "missing"
    if value >= stale_seconds:
        return "stale"
    if value >= warn_seconds:
        return "lagging"
    return "fresh"


def age_seconds(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(parsed.tzinfo or PACIFIC)
    return max(0.0, (now - parsed).total_seconds())


def age_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    seconds = int(max(0, value))
    if seconds < 90:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h {minutes % 60}m ago"


def latest_market_snapshot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("event_type") != "market_snapshot":
            continue
        snapshot = _object(row.get("market_snapshot"))
        underlying = _object(snapshot.get("underlying"))
        context = _object(snapshot.get("context"))
        option_nbbo = _object(snapshot.get("option_nbbo"))
        spx_market_data_type = context.get("spx_market_data_type") or underlying.get("spx_market_data_type")
        vix_market_data_type = context.get("vix_market_data_type") or underlying.get("vix_market_data_type")
        is_live = (
            context.get("source") == "ibkr_live"
            and spx_market_data_type == "live"
            and vix_market_data_type == "live"
            and int(_float(option_nbbo.get("quote_count")) or 0) > 0
        )
        return {
            "timestamp": row.get("timestamp"),
            "spx": _float(underlying.get("spx")),
            "vix": _float(underlying.get("vix")),
            "context_age_ms": _float(context.get("context_age_ms")),
            "source": context.get("source"),
            "spx_market_data_type": spx_market_data_type,
            "vix_market_data_type": vix_market_data_type,
            "option_quote_count": int(_float(option_nbbo.get("quote_count")) or 0),
            "is_live": is_live,
        }
    return {"is_live": False}


def latest_model_decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("event_type") != "model_decision":
            continue
        if row.get("mode") != "paper-submit":
            continue
        decision = _object(row.get("model_decision"))
        contract = _object(row.get("selected_contract"))
        return {
            "timestamp": row.get("timestamp"),
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "score": _float(decision.get("score")),
            "threshold": _float(decision.get("threshold")),
            "contract": display_contract(contract),
        }
    return {}


def latest_candidate_set_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("event_type") != "candidate_set":
            continue
        gate = _object(row.get("candidate_gate_diagnostics"))
        return {
            "timestamp": row.get("timestamp"),
            "candidate_count": int(_float(row.get("candidate_count")) or 0),
            "threshold": _float(row.get("model_threshold")),
            "reason": _object(row.get("risk_gate")).get("reason"),
            "gate_diagnostics": gate,
            "live_index_context": _object(row.get("live_index_context")),
        }
    return {}


def summarize_latest_ladder(shadow: dict[str, Any], *, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    contracts = [str(row.get("contract_id") or "") for row in shadow.get("latest_contracts_observed") or []]
    if not contracts and rows:
        for row in reversed(rows):
            if row.get("event_type") != "candidate_set":
                continue
            gate = _object(row.get("candidate_gate_diagnostics"))
            candidates = gate.get("top_rejected_contracts") or gate.get("top_surface_tokens") or row.get("candidate_contracts") or []
            contracts = [str(_object(item).get("contract_id") or "") for item in candidates if _object(item).get("contract_id")]
            if contracts:
                break
    strikes = []
    rights = set()
    for contract in contracts:
        parsed = parse_shadow_contract(contract)
        if parsed:
            strikes.append(parsed["strike"])
            rights.add(parsed["right"])
    if not strikes:
        return {
            "timestamp": shadow.get("latest_observation_time"),
            "spx": shadow.get("latest_spx"),
            "summary": "No current option ladder observed",
            "contracts": [],
        }
    min_strike = min(strikes)
    max_strike = max(strikes)
    right_text = "/".join(sorted(rights)) if rights else "unknown"
    spx_value = _float(shadow.get("latest_spx"))
    spx_text = f"{spx_value:,.2f}" if spx_value is not None else "unknown"
    return {
        "timestamp": shadow.get("latest_observation_time"),
        "spx": shadow.get("latest_spx"),
        "min_strike": min_strike,
        "max_strike": max_strike,
        "right_text": right_text,
        "contract_count": len(contracts),
        "summary": f"{min_strike:.0f}-{max_strike:.0f} {right_text} around SPX {spx_text}",
        "contracts": contracts,
    }


def parse_shadow_contract(contract_id: str) -> dict[str, Any] | None:
    # Expected shape: SPXW-YYYYMMDD-07385.000-C
    parts = contract_id.split("-")
    if len(parts) < 4:
        return None
    try:
        strike = float(parts[-2])
    except ValueError:
        return None
    return {"strike": strike, "right": parts[-1]}


def display_contract(contract: dict[str, Any]) -> str:
    if not contract:
        return ""
    root = contract.get("trading_class") or contract.get("root") or "SPXW"
    expiry = contract.get("expiry") or "?"
    strike = contract.get("strike") or "?"
    right = contract.get("right") or "?"
    return f"{root} {expiry} {strike} {right}"


def summarize_runtime_flag(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": bool(payload),
        "paper_orders_enabled": bool(payload.get("paper_orders_enabled")),
        "enabled_at": payload.get("enabled_at"),
        "session": payload.get("session"),
        "run_id": payload.get("run_id"),
        "account_id_redacted": payload.get("account_id_redacted"),
    }


def summarize_runtime_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"exists": False, "state": "unknown"}
    intent = _object(payload.get("intent"))
    return {
        "exists": True,
        "state": payload.get("state") or "unknown",
        "entry_time": payload.get("entry_time"),
        "entry_fill_price": _float(payload.get("entry_fill_price")),
        "mfe_to_now": _float(payload.get("mfe_to_now")),
        "mae_to_now": _float(payload.get("mae_to_now")),
        "symbol": intent.get("symbol"),
        "expiry": intent.get("expiry"),
        "strike": intent.get("strike"),
        "right": intent.get("right"),
        "quantity": intent.get("quantity"),
    }


def summarize_current_position(runtime_state: dict[str, Any], *, paper: dict[str, Any]) -> dict[str, Any]:
    state = str(runtime_state.get("state") or "").lower()
    intent = _object(runtime_state.get("intent"))
    latest_account = _object(paper.get("latest_account"))
    if state == "holding" or int(_float(latest_account.get("open_positions")) or 0) > 0:
        contract = "-".join(
            str(part)
            for part in [
                intent.get("trading_class") or "SPXW",
                intent.get("expiry") or "?",
                intent.get("strike") or "?",
                intent.get("right") or "?",
            ]
        )
        return {
            "status": "holding",
            "contract": contract,
            "quantity": intent.get("quantity") or 1,
            "entry_time": runtime_state.get("entry_time"),
            "entry_fill_price": _float(runtime_state.get("entry_fill_price")),
            "mfe_to_now": _float(runtime_state.get("mfe_to_now")),
            "mae_to_now": _float(runtime_state.get("mae_to_now")),
        }
    return {"status": "flat", "contract": "", "quantity": 0, "entry_time": None, "entry_fill_price": None}


def account_source(*, latest_account: dict[str, Any], ibkr_account_snapshot: dict[str, Any]) -> str:
    if ibkr_account_ready(ibkr_account_snapshot):
        return "ibkr_account_summary"
    if latest_account:
        return "trade_log_account"
    return "unknown"


def effective_account_state(*, paper: dict[str, Any], ibkr_account_snapshot: dict[str, Any]) -> dict[str, Any]:
    local_account = _object(paper.get("latest_account"))
    reconstructed_pnl = _float(paper.get("reconstructed_closed_pnl"))
    if ibkr_account_ready(ibkr_account_snapshot):
        values = _object(ibkr_account_snapshot.get("values"))
        realized = _float(values.get("realized_pnl"))
        return {
            "starting_cash": _float(local_account.get("starting_cash")),
            "cash": _first_float(values.get("cash"), local_account.get("cash")),
            "equity": _first_float(values.get("net_liquidation"), local_account.get("equity")),
            "realized_daily_pnl": _first_float(realized, reconstructed_pnl, local_account.get("realized_daily_pnl")),
            "unrealized_pnl": _float(values.get("unrealized_pnl")),
            "available_funds": _float(values.get("available_funds")),
            "buying_power": _float(values.get("buying_power")),
            "open_positions": int(_float(local_account.get("open_positions")) or 0),
            "account_id_redacted": ibkr_account_snapshot.get("account_id_redacted")
            or local_account.get("account_id_redacted"),
            "source": "ibkr_account_summary",
            "pnl_source": "ibkr_account_summary" if realized is not None else "reconstructed_fills",
            "snapshot_status": ibkr_account_snapshot.get("status"),
            "snapshot_checked_at_utc": ibkr_account_snapshot.get("checked_at_utc"),
        }
    return {
        "starting_cash": _float(local_account.get("starting_cash")),
        "cash": _float(local_account.get("cash")),
        "equity": _float(local_account.get("equity")),
        "realized_daily_pnl": _first_float(local_account.get("realized_daily_pnl"), reconstructed_pnl),
        "unrealized_pnl": None,
        "available_funds": None,
        "buying_power": None,
        "open_positions": int(_float(local_account.get("open_positions")) or 0),
        "account_id_redacted": local_account.get("account_id_redacted"),
        "source": "trade_log_account" if local_account else "unknown",
        "pnl_source": "trade_log_account" if local_account else "reconstructed_fills",
        "snapshot_status": ibkr_account_snapshot.get("status") or "not_checked",
        "snapshot_checked_at_utc": ibkr_account_snapshot.get("checked_at_utc"),
    }


def ibkr_account_ready(snapshot: dict[str, Any]) -> bool:
    values = _object(snapshot.get("values"))
    return (
        snapshot.get("status") == "pass"
        and snapshot.get("broker_order_endpoint_called") is False
        and snapshot.get("real_money_trading") is False
        and _float(values.get("net_liquidation")) is not None
    )


def latest_account_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        account = _object(row.get("account"))
        if account:
            return {
                "cash": _float(account.get("cash")),
                "equity": _float(account.get("equity")),
                "realized_daily_pnl": _float(account.get("realized_daily_pnl")),
                "open_positions": int(_float(account.get("open_positions")) or 0),
                "starting_cash": _float(account.get("starting_cash")),
                "account_id_redacted": account.get("account_id_redacted"),
            }
    return {}


def contract_key(row: dict[str, Any]) -> str:
    contract = _object(row.get("selected_contract"))
    parts = [
        contract.get("trading_class") or contract.get("root") or "SPXW",
        contract.get("expiry") or "?",
        contract.get("strike") or "?",
        contract.get("right") or "?",
    ]
    return "-".join(str(part) for part in parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    paper = payload["paper"]
    account_snapshot = payload.get("ibkr_account_snapshot") or {}
    snapshot_values = _object(account_snapshot.get("values"))
    startup = payload["startup"]
    shadow = payload["shadow"]
    failures = payload["failures"]
    lines = [
        "# Protocol 157: Protocol101 Daily Ops Monitor",
        "",
        "No paid data was downloaded. This monitor only reads local live/paper logs and artifacts.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Generated: `{payload['generated_at_pacific']}`",
        f"- Session: `{payload['session']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Trade log: `{payload['source_trade_log']}`",
        f"- Protocol147 directory: `{payload['protocol147_dir']}`",
        f"- Next action: {payload['next_action']}",
        "",
        "## Startup",
        "",
        f"- LaunchAgents loaded: `{startup['launchd_loaded_count']}/{startup['launchd_expected_count']}`",
        f"- Entitlement decision: `{startup['entitlement_decision']}`",
        f"- Effective entitlement decision: `{startup.get('effective_entitlement_decision')}`",
        f"- Entitlement resolved by current live paper log: `{startup.get('entitlement_resolved_by_live_trade_log')}`",
        f"- IBKR connected: `{startup['ibkr_connected']}`",
        f"- Live parity ready: `{startup['live_parity_ready']}`",
        f"- Live capture pass: `{startup['live_capture_pass']}`",
        f"- Observed capture passes: `{startup['observed_capture_passes']}`",
        f"- Protocol147 cycles: `{startup['protocol147_cycles_run']}`",
        f"- Protocol147 live capture passes/blocks: `{startup['protocol147_live_capture_passes']}` / `{startup['protocol147_live_capture_blocks']}`",
        "",
        "## Paper Trading",
        "",
        f"- Paper orders submitted: `{paper['paper_orders_submitted']}`",
        f"- Market snapshots / model decisions: `{paper['market_snapshot_rows']}` / `{paper['model_decision_rows']}`",
        f"- Broker endpoint rows: `{paper['broker_order_endpoint_called_rows']}`",
        f"- Entry fills / exit fills: `{paper['paper_entry_fills']}` / `{paper['paper_exit_fills']}`",
        f"- Closed trades / open trades: `{paper['closed_trades']}` / `{paper['open_trades']}`",
        f"- Reconstructed closed PnL: `${paper['reconstructed_closed_pnl']:.2f}`",
        f"- Latest account: `{json.dumps(paper['latest_account'], sort_keys=True)}`",
        f"- Dashboard account source: `{paper.get('account_source', 'unknown')}`",
        f"- IBKR account snapshot: status=`{account_snapshot.get('status', 'not_checked')}` "
        f"account=`{account_snapshot.get('account_id_redacted')}` "
        f"net_liquidation=`{snapshot_values.get('net_liquidation')}` "
        f"realized_pnl=`{snapshot_values.get('realized_pnl')}`",
        "",
        "## Live Shadow",
        "",
        f"- Shadow rows: `{shadow['total_rows']}`",
        f"- Live rows / delayed rows: `{shadow['live_rows']}` / `{shadow['delayed_rows']}`",
        f"- Order-intent rows inside shadow: `{shadow['order_intent_non_null_rows']}`",
        f"- Action counts: `{json.dumps(shadow['action_counts'], sort_keys=True)}`",
        f"- Position-state counts: `{json.dumps(shadow['position_state_counts'], sort_keys=True)}`",
        "",
        "## Contracts",
        "",
    ]
    if paper["contracts_traded"]:
        for item in paper["contracts_traded"]:
            lines.append(f"- `{item['contract']}` events=`{item['events']}` quantity_sum=`{item['quantity_sum']}`")
    else:
        lines.append("- No paper order/fill contracts logged yet.")
    lines.extend(
        [
            "",
            "## Failures And Blockers",
            "",
            "```json",
            json.dumps(failures["reason_counts"], indent=2, sort_keys=True),
            "```",
        ]
    )
    safe_write_text(path, "\n".join(lines) + "\n")


def write_html(path: Path, payload: dict[str, Any], flat_rows: list[dict[str, Any]]) -> None:
    paper = payload["paper"]
    startup = payload["startup"]
    shadow = payload["shadow"]
    config = payload.get("monitor_config") or {}
    position = payload.get("current_position") or {}
    trader = payload.get("trader_status") or {}
    latest_market = trader.get("latest_market") or {}
    latest_decision = trader.get("latest_model_decision") or {}
    latest_candidate = trader.get("latest_candidate_set") or {}
    ladder = trader.get("latest_ladder") or {}
    account = trader.get("account") or {}
    freshness = trader.get("freshness") or {}
    refresh_seconds = int(config.get("refresh_seconds") or 30)
    event_limit = int(config.get("event_limit") or 250)
    generated_at = str(payload.get("generated_at_pacific") or "")
    active_timeline = important_timeline_rows(
        flat_rows,
        limit=event_limit,
        resolved_reasons=(payload.get("failures") or {}).get("resolved_reason_counts") or {},
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <meta http-equiv="Cache-Control" content="no-store">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Protocol101 Live Monitor</title>
  <style>
    :root {{
      color-scheme: light;
      --bg:#f6f7f8; --ink:#17202c; --muted:#657286; --panel:#fff; --soft:#eef2f5;
      --line:#d8e1ea; --ok:#11754f; --warn:#987000; --bad:#b33636; --accent:#2f6f97;
      --ok-bg:#edf8f2; --warn-bg:#fff8e6; --bad-bg:#fff0f0;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:#263241; color:#fff; padding:16px 24px; border-bottom:4px solid #8bb7c9; }}
    h1 {{ margin:0 0 2px; font-size:22px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:16px; }}
    h3 {{ margin:12px 0 8px; font-size:13px; color:var(--muted); text-transform:uppercase; }}
    main {{ max-width:1460px; margin:auto; padding:16px 22px 36px; }}
    section {{ margin-top:14px; }}
    .subline {{ color:#dbe5ec; }}
    .dashboard-grid {{ display:grid; grid-template-columns:1.15fr .9fr 1.05fr; gap:14px; align-items:stretch; }}
    .operator-grid {{ display:grid; grid-template-columns:1.15fr 1fr 1fr; gap:14px; align-items:start; }}
    .two-col {{ display:grid; grid-template-columns:minmax(320px, 1fr) minmax(320px, 1fr); gap:14px; align-items:start; }}
    .panel, .hero {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .hero {{ border-top:5px solid var(--ok); }}
    .hero.warn {{ border-top-color:var(--warn); }}
    .hero.bad {{ border-top-color:var(--bad); }}
    .state {{ margin:2px 0 4px; font-size:28px; font-weight:850; line-height:1.1; }}
    .explain {{ color:var(--muted); font-size:15px; margin-top:6px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:9px; }}
    .metric {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:10px; min-height:62px; }}
    .metric.big .v {{ font-size:22px; }}
    .k {{ color:var(--muted); text-transform:uppercase; font-size:11px; font-weight:750; }}
    .v {{ margin-top:4px; font-weight:760; font-size:16px; overflow-wrap:anywhere; }}
    .ok {{ color:var(--ok); }}
    .warn {{ color:var(--warn); }}
    .bad {{ color:var(--bad); }}
    .muted {{ color:var(--muted); }}
    .pill-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }}
    .pill {{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:var(--soft); border-radius:999px; padding:5px 9px; font-size:13px; }}
    .pill.ok {{ border-color:#b8ddca; background:var(--ok-bg); }}
    .pill.warn {{ border-color:#eadb9b; background:var(--warn-bg); }}
    .pill.bad {{ border-color:#edbebe; background:var(--bad-bg); }}
    .pill strong {{ font-variant-numeric:tabular-nums; }}
    .badge {{ display:inline-flex; align-items:center; border-radius:999px; padding:5px 10px; border:1px solid var(--line); background:var(--soft); font-weight:700; }}
    .status-line {{ border-left:4px solid var(--ok); background:#edf8f2; padding:10px 12px; border-radius:6px; }}
    .status-line.warn {{ border-left-color:var(--warn); background:#fff8e8; }}
    .status-line.bad {{ border-left-color:var(--bad); background:#fff0f0; }}
    .notice {{ margin-top:12px; }}
    table {{ border-collapse:collapse; width:100%; background:var(--panel); }}
    th, td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#eef3f7; z-index:1; }}
    .table-wrap {{ border:1px solid var(--line); border-radius:8px; overflow:auto; max-height:330px; background:var(--panel); }}
    .empty {{ color:var(--muted); background:var(--panel); border:1px dashed var(--line); border-radius:8px; padding:14px; }}
    .primary-table th {{ width:42%; color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .why {{ font-size:18px; font-weight:800; margin:3px 0 8px; }}
    details {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
    summary {{ cursor:pointer; font-weight:800; }}
    .details-body {{ margin-top:12px; }}
    @media (max-width:1100px) {{ .dashboard-grid, .operator-grid, .two-col {{ grid-template-columns:1fr; }} th, td {{ white-space:normal; }} }}
  </style>
  <script>
    function monitorAgeLabel(seconds) {{
      if (seconds < 90) return seconds + 's ago';
      var minutes = Math.floor(seconds / 60);
      if (minutes < 90) return minutes + 'm ago';
      var hours = Math.floor(minutes / 60);
      return hours + 'h ' + (minutes % 60) + 'm ago';
    }}
    function updateMonitorAge() {{
      var el = document.getElementById('page-age');
      if (!el) return;
      var builtAt = Date.parse(el.dataset.builtAt || '');
      if (!Number.isFinite(builtAt)) return;
      var seconds = Math.max(0, Math.floor((Date.now() - builtAt) / 1000));
      el.textContent = monitorAgeLabel(seconds);
      el.className = seconds > {max(90, refresh_seconds * 4)} ? 'bad' : seconds > {max(60, refresh_seconds * 2)} ? 'warn' : 'ok';
    }}
    window.setInterval(updateMonitorAge, 1000);
    window.addEventListener('load', updateMonitorAge);
    window.setTimeout(function () {{
      var base = window.location.href.split('?')[0];
      window.location.replace(base + '?monitor_refresh=' + Date.now());
    }}, {max(5, refresh_seconds) * 1000});
  </script>
</head>
<body>
  <header>
    <h1>Protocol101 Paper Trader</h1>
    <div class="subline">Live Monitor · {esc(payload['session'])} · {esc(payload['run_id'])} · refresh {refresh_seconds}s · built {esc(local_time(generated_at))} · page age <span id="page-age" data-built-at="{esc(generated_at)}">0s ago</span></div>
  </header>
  <main>
    <section class="dashboard-grid">
      <div class="hero {health_class(freshness.get('state'))}">
        <div class="k">Trader State</div>
        <div class="state {state_class(trader.get('status'))}">{esc(trader.get('status', 'Unknown'))}</div>
        <div class="explain">{esc(trader.get('last_readable', 'Waiting for model activity'))}</div>
        <div class="pill-row">
          {pill("Feed", trader.get("feed", "unknown"), state_class(trader.get("feed")))}
          {pill("Market", freshness.get("market_age", "unknown"), health_class(freshness.get("market_state")))}
          {pill("Decision", freshness.get("decision_age", "unknown"), health_class(freshness.get("decision_state")))}
          {pill("Blockers", len((payload.get("failures") or {}).get("reason_counts") or {}), "bad" if (payload.get("failures") or {}).get("reason_counts") else "ok")}
        </div>
      </div>
      <div class="panel">
        <h2>Is It Working?</h2>
        {working_panel(payload, latest_market, latest_candidate, paper, freshness)}
      </div>
      <div class="panel">
        <h2>Decision Gates</h2>
        {decision_gates_panel(latest_decision, latest_candidate)}
      </div>
    </section>
    {account_reconciliation_notice(trader, paper)}
    <section class="operator-grid">
      <div class="panel">
        <h2>Why No Trade?</h2>
        {why_no_trade_panel(latest_decision, latest_candidate)}
      </div>
      <div class="panel">
        <h2>Live Tape</h2>
        {market_panel(latest_market, ladder)}
      </div>
      <div class="panel">
        <h2>Position And Orders</h2>
        {operator_position_panel(position, paper, account)}
      </div>
    </section>
    <section class="two-col">
      <div class="panel">
        <h2>Session Counts</h2>
        {session_counts_panel(paper)}
      </div>
      <div class="panel">
        <h2>Full Model Read</h2>
        {last_decision_panel(latest_decision, latest_candidate)}
      </div>
    </section>
    <section class="panel">
      <h2>Recent Meaningful Events</h2>
      <div class="table-wrap">{event_table(active_timeline, limit=event_limit)}</div>
    </section>
    <section>
      <details>
        <summary>Diagnostics</summary>
        <div class="details-body">
          <div class="two-col">
            <div class="panel">
              <h2>Startup And Shutdown</h2>
              <div class="metric-grid">
                {''.join(metric(label.split('.')[-1], launchd_value(row)) for label, row in payload['launchd'].items())}
              </div>
            </div>
            <div class="panel">
              <h2>Live Shadow Diagnostics</h2>
              {shadow_panel(shadow)}
            </div>
          </div>
          <div class="two-col">
            <div class="panel">
              <h2>Contracts Bought/Sold</h2>
              {contracts_table(paper['contracts_traded'])}
            </div>
            <div class="panel">
              <h2>Failures And Blockers</h2>
              {failures_panel(payload['failures'])}
            </div>
          </div>
        </div>
      </details>
    </section>
  </main>
</body>
</html>
"""
    safe_write_text(path, body)


def write_monitor_alias(path: Path, canonical_path: Path, payload: dict[str, Any]) -> None:
    relative = os.path.relpath(canonical_path, start=path.parent)
    generated_at = str(payload.get("generated_at_pacific") or "")
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={esc(relative)}?monitor_refresh=live">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Protocol101 Live Monitor Redirect</title>
  <style>
    body {{ margin:0; background:#f4f6f8; color:#17202c; font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:760px; margin:12vh auto; padding:24px; background:white; border:1px solid #d8e1ea; border-radius:8px; }}
    a {{ color:#23689b; font-weight:700; }}
  </style>
</head>
<body>
  <main>
    <h1>Protocol101 Live Monitor Moved</h1>
    <p>This dated monitor is only an alias now. The single current monitor is
      <a href="{esc(relative)}?monitor_refresh=live">latest_daily_monitor.html</a>.
    </p>
    <p>Session {esc(payload.get("session"))} · run {esc(payload.get("run_id"))} · last built {esc(local_time(generated_at))}</p>
  </main>
</body>
</html>
"""
    safe_write_text(path, body)


def pill(label: str, value: Any, klass: str = "") -> str:
    return f'<span class="pill {esc(klass)}">{esc(label)} <strong>{esc(value)}</strong></span>'


def working_panel(
    payload: dict[str, Any],
    market: dict[str, Any],
    candidate: dict[str, Any],
    paper: dict[str, Any],
    freshness: dict[str, Any],
) -> str:
    evidence = payload.get("current_live_trade_log_evidence") or {}
    failures = payload.get("failures") or {}
    generated_at = payload.get("generated_at_pacific")
    rows = [
        (
            "Monitor refresh",
            snapshot_age_text(generated_at),
            health_class("fresh" if (age_seconds(generated_at) or 0) < 90 else "lagging"),
        ),
        ("Market log", freshness.get("market_age", "unknown"), health_class(freshness.get("market_state"))),
        ("Decision log", freshness.get("decision_age", "unknown"), health_class(freshness.get("decision_state"))),
        ("SPX/VIX feed", f"{market.get('spx_market_data_type') or '?'} / {market.get('vix_market_data_type') or '?'}", state_class(market.get("is_live"))),
        ("Option quotes", market.get("option_quote_count", 0), "ok" if int(_float(market.get("option_quote_count")) or 0) > 0 else "bad"),
        ("Context", live_context_summary(_object(candidate.get("live_index_context"))) or "missing", "ok" if evidence.get("live_context_ready") else "bad"),
        ("Model reads", paper.get("model_decision_rows", 0), "ok" if int(paper.get("model_decision_rows") or 0) > 0 else "bad"),
        ("Active failures", len(failures.get("reason_counts") or {}), "bad" if failures.get("reason_counts") else "ok"),
    ]
    return '<div class="pill-row">' + "".join(pill(label, value, klass) for label, value, klass in rows) + "</div>"


def decision_gates_panel(decision: dict[str, Any], candidate: dict[str, Any]) -> str:
    gate = _object(candidate.get("gate_diagnostics"))
    threshold = _first_float(decision.get("threshold"), candidate.get("threshold"))
    score = _float(decision.get("score"))
    distance = (score - threshold) if score is not None and threshold is not None else None
    allowed = bool(gate.get("allowed_time_bucket", True))
    context_ready = bool(gate.get("context_ready", True))
    opening_context_ready = gate.get("opening_context_ready")
    missing_opening = gate.get("missing_opening_minutes")
    rows = [
        ("Protocol101 threshold", fmt_number(threshold, digits=4), "ok" if threshold is not None else "bad"),
        ("Current model score", fmt_number(score, digits=4) if score is not None else "not scored", "warn" if score is None else "ok"),
        ("Distance to threshold", fmt_signed(distance, digits=4), threshold_distance_class(distance)),
        ("Min edge gate", fmt_number(gate.get("min_edge"), digits=2), "ok"),
        ("Max observed edge", fmt_number(gate.get("max_edge"), digits=2), "ok" if _float(gate.get("max_edge")) is not None else "warn"),
        ("Best call / put edge", edge_pair(gate) or "unknown", "ok" if edge_pair(gate) else "warn"),
        ("Time bucket", f"{gate.get('time_bucket') or 'unknown'}", "ok" if allowed else "warn"),
        ("Allowed buckets", ", ".join(str(x) for x in gate.get("allowed_buckets") or []) or "unknown", "ok" if allowed else "warn"),
        ("Context gate", "ready" if context_ready else "not ready", "ok" if context_ready else "bad"),
        (
            "Opening context",
            "ready" if opening_context_ready is not False else f"missing {missing_opening or '?'} min",
            "ok" if opening_context_ready is not False else "bad",
        ),
        ("Candidates", candidate.get("candidate_count", 0), "ok" if int(_float(candidate.get("candidate_count")) or 0) > 0 else "warn"),
        ("Valid surface scores", gate.get("valid_score_count", 0), "ok" if int(_float(gate.get("valid_score_count")) or 0) > 0 else "warn"),
        ("Above min edge", gate.get("above_min_edge_count", 0), "ok" if int(_float(gate.get("above_min_edge_count")) or 0) > 0 else "warn"),
    ]
    return '<div class="metric-grid">' + "".join(gate_metric(label, value, klass) for label, value, klass in rows) + "</div>"


def why_no_trade_panel(decision: dict[str, Any], candidate: dict[str, Any]) -> str:
    gate = _object(candidate.get("gate_diagnostics"))
    reason = str(decision.get("reason") or candidate.get("reason") or "unknown")
    action = str(decision.get("action") or "wait")
    readable = explain_no_trade(reason, gate=gate, candidate=candidate, decision=decision)
    rows = [
        ("Action", action),
        ("Reason", reason),
        ("Gate diagnosis", gate.get("filter_reason") or ""),
        ("Time bucket", gate.get("time_bucket") or ""),
        ("Candidate count", candidate.get("candidate_count", "")),
        ("Top rejected", top_rejected_summary(gate)),
    ]
    return f'<div class="why">{esc(readable)}</div>' + key_value_table(rows)


def explain_no_trade(reason: str, *, gate: dict[str, Any], candidate: dict[str, Any], decision: dict[str, Any]) -> str:
    if reason == "outside_time_bucket" or gate.get("filter_reason") == "outside_time_bucket":
        allowed = ", ".join(str(x) for x in gate.get("allowed_buckets") or [])
        bucket = gate.get("time_bucket") or "current bucket"
        return f"Waiting because {bucket} is not an entry bucket. Allowed buckets: {allowed or 'unknown'}."
    if reason == "below_protocol101_threshold":
        score = fmt_number(decision.get("score"), digits=4)
        threshold = fmt_number(decision.get("threshold") or candidate.get("threshold"), digits=4)
        return f"Waiting because Protocol101 score {score} is below threshold {threshold}."
    if gate.get("filter_reason") == "below_min_edge":
        return (
            f"Waiting because best edge {fmt_number(gate.get('max_edge'), digits=2)} "
            f"is below min edge {fmt_number(gate.get('min_edge'), digits=2)}."
        )
    if reason in {"no_candidates", "no_entry_intent"}:
        return "Waiting because no candidate currently clears the live candidate and risk filters."
    if reason in {"insufficient_live_index_context", "candidate_set_blocked_insufficient_live_index_context"}:
        return "Waiting for enough live SPX/VIX context rows."
    if reason.startswith("candidate_set_blocked_"):
        return f"Candidate gate blocked: {reason.replace('candidate_set_blocked_', '')}."
    if action == "enter":
        return "Entry signal emitted. Check order/fill rows below."
    return f"Waiting: {reason}."


def operator_position_panel(position: dict[str, Any], paper: dict[str, Any], account: dict[str, Any]) -> str:
    rows = [
        ("Position", position.get("status", "unknown"), "ok" if position.get("status") == "flat" else "warn"),
        ("Orders", paper.get("paper_orders_submitted", 0), "ok" if int(paper.get("paper_orders_submitted") or 0) == 0 else "warn"),
        ("Entry fills", paper.get("paper_entry_fills", 0), ""),
        ("Exit fills", paper.get("paper_exit_fills", 0), ""),
        ("Logged P&L", money(paper.get("reconstructed_closed_pnl")), ""),
        ("Realized P&L", money(account.get("realized_daily_pnl")), ""),
        ("Equity", money(account.get("equity")), ""),
        ("Account source", account.get("source") or paper.get("account_source") or "unknown", ""),
    ]
    return '<div class="pill-row">' + "".join(pill(label, value, klass) for label, value, klass in rows) + "</div>"


def session_counts_panel(paper: dict[str, Any]) -> str:
    event_counts = paper.get("event_counts") or {}
    rows = [
        ("Snapshots", paper.get("market_snapshot_rows", 0), "ok"),
        ("Candidate sets", paper.get("candidate_set_rows", 0), "ok"),
        ("Model decisions", paper.get("model_decision_rows", 0), "ok"),
        ("Risk gates", event_counts.get("risk_gate", 0), "ok"),
        ("Order blocks", event_counts.get("paper_order_blocked", 0), "warn" if event_counts.get("paper_order_blocked") else ""),
        ("Paper errors", event_counts.get("paper_error", 0), "warn" if event_counts.get("paper_error") else "ok"),
        ("Broker endpoint calls", paper.get("broker_order_endpoint_called_rows", 0), "warn" if paper.get("broker_order_endpoint_called_rows") else "ok"),
        ("Submitted/fills", f"{paper.get('paper_orders_submitted', 0)} / {paper.get('paper_entry_fills', 0) + paper.get('paper_exit_fills', 0)}", ""),
    ]
    return '<div class="metric-grid">' + "".join(gate_metric(label, value, klass) for label, value, klass in rows) + "</div>"


def gate_metric(label: str, value: Any, klass: str = "") -> str:
    return f'<div class="metric"><div class="k">{esc(label)}</div><div class="v {esc(klass)}">{esc(value)}</div></div>'


def fmt_signed(value: Any, *, digits: int = 2) -> str:
    number = _float(value)
    if number is None:
        return "not available"
    return f"{number:+,.{digits}f}"


def threshold_distance_class(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "warn"
    if number >= 0:
        return "ok"
    return "bad"


def metric(key: str, value: Any) -> str:
    text = "" if value is None else str(value)
    klass = (
        "ok"
        if text in {"pass", "ready", "True", "flat"} or text.startswith("pass_")
        else "bad"
        if text.startswith("blocked_") or text in {"False", "fail"}
        else ""
    )
    return f'<div class="metric"><div class="k">{esc(key)}</div><div class="v {klass}">{esc(text)}</div></div>'


def state_class(value: Any) -> str:
    if isinstance(value, bool):
        return "ok" if value else "bad"
    text = str(value or "").lower()
    if "blocked" in text or "not current" in text or "stale" in text or "missing" in text:
        return "bad"
    if "lagging" in text or "warn" in text:
        return "warn"
    if "live" in text or "flat" in text or "paper order" in text or "position" in text or "fresh" in text or "pass" in text:
        return "ok"
    return ""


def health_class(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"fresh", "ok", "pass", "ready", "live"}:
        return "ok"
    if text in {"lagging", "warn", "warning"}:
        return "warn"
    if text in {"stale", "missing", "bad", "fail", "blocked"}:
        return "bad"
    return state_class(value)


def money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "unknown"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}"


def account_equity_delta(account: dict[str, Any]) -> float | None:
    equity = _float(account.get("equity"))
    starting_cash = _float(account.get("starting_cash"))
    if equity is None or starting_cash is None:
        return None
    return equity - starting_cash


def account_reconciliation_notice(trader: dict[str, Any], paper: dict[str, Any]) -> str:
    account = _object(trader.get("account"))
    delta = account_equity_delta(account)
    if delta is None or abs(delta) < 0.005:
        return ""
    logged_pnl = _float(paper.get("reconstructed_closed_pnl")) or 0.0
    realized_pnl = _float(account.get("realized_daily_pnl")) or 0.0
    entry_fills = int(_float(paper.get("paper_entry_fills")) or 0)
    exit_fills = int(_float(paper.get("paper_exit_fills")) or 0)
    submitted = int(_float(paper.get("paper_orders_submitted")) or 0)
    if submitted or entry_fills or exit_fills or abs(logged_pnl) >= 0.005:
        return ""
    if abs(realized_pnl) >= 0.005:
        return ""
    return (
        '<div class="notice status-line warn">'
        "<strong>Broker equity moved without logged Protocol101 trades.</strong><br>"
        f"The IBKR account snapshot is {esc(money(delta))} from starting cash, "
        "but this session log has 0 submitted orders, 0 fills, and $0.00 reconstructed trade P&L. "
        "Treat this as an account-level broker value until an order/fill row appears in the paper log."
        "</div>"
    )


def snapshot_age_text(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        generated = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    now = datetime.now(generated.tzinfo or PACIFIC)
    seconds = max(0, int((now - generated).total_seconds()))
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h {minutes % 60}m"


def fmt_number(value: Any, *, digits: int = 2) -> str:
    number = _float(value)
    if number is None:
        return "unknown"
    return f"{number:,.{digits}f}"


def local_time(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return parsed.astimezone(PACIFIC).strftime("%H:%M:%S PT")


def status_panel(payload: dict[str, Any]) -> str:
    failures = payload.get("failures") or {}
    if failures.get("has_failures"):
        klass = "status-line bad"
        headline = "Active blocker present"
    else:
        klass = "status-line"
        headline = "Live paper monitor is healthy"
    resolved = failures.get("resolved_reason_counts") or {}
    resolved_note = ""
    if resolved:
        resolved_note = (
            '<div class="pill-row">'
            + "".join(f'<span class="pill warn">resolved <strong>{esc(key)}</strong> x{esc(value)}</span>' for key, value in resolved.items())
            + "</div>"
        )
    return (
        f'<div class="{klass}"><strong>{esc(headline)}</strong><br>{esc(payload.get("next_action", ""))}{resolved_note}</div>'
    )


def position_table(row: dict[str, Any]) -> str:
    if not row or row.get("status") != "holding":
        return '<div class="status-line"><strong>Flat</strong><br>No open paper position recorded.</div>'
    fields = ["status", "contract", "quantity", "entry_time", "entry_fill_price", "mfe_to_now", "mae_to_now"]
    body = "<tr>" + "".join(f"<td>{esc(row.get(field, ''))}</td>" for field in fields) + "</tr>"
    return "<table><thead><tr>" + "".join(f"<th>{esc(field)}</th>" for field in fields) + "</tr></thead><tbody>" + body + "</tbody></table>"


def orders_panel(paper: dict[str, Any]) -> str:
    event_counts = paper.get("event_counts") or {}
    rows = [
        ("Paper orders", paper.get("paper_orders_submitted", 0)),
        ("Entry fills", paper.get("paper_entry_fills", 0)),
        ("Exit fills", paper.get("paper_exit_fills", 0)),
        ("Closed PnL", f"${float(paper.get('reconstructed_closed_pnl') or 0.0):.2f}"),
        ("Model decisions", paper.get("model_decision_rows", 0)),
        ("No-entry checks", event_counts.get("risk_gate", 0)),
    ]
    return '<div class="pill-row">' + "".join(f'<span class="pill">{esc(label)} <strong>{esc(value)}</strong></span>' for label, value in rows) + "</div>"


def market_panel(market: dict[str, Any], ladder: dict[str, Any]) -> str:
    rows = [
        ("SPX", fmt_number(market.get("spx"), digits=2)),
        ("VIX", fmt_number(market.get("vix"), digits=2)),
        ("Source", market.get("source") or "unknown"),
        ("SPX/VIX Type", f"{market.get('spx_market_data_type') or '?'} / {market.get('vix_market_data_type') or '?'}"),
        ("Context Age", f"{fmt_number(market.get('context_age_ms'), digits=0)} ms"),
        ("Option Quotes", market.get("option_quote_count", 0)),
        ("Latest Ladder", ladder_summary(ladder, market=market)),
        ("Market Time", local_time(market.get("timestamp"))),
        ("Ladder Time", local_time(ladder.get("timestamp"))),
    ]
    return key_value_table(rows)


def ladder_summary(ladder: dict[str, Any], *, market: dict[str, Any]) -> str:
    summary = str(ladder.get("summary") or "")
    spx = _first_float(ladder.get("spx"), market.get("spx"))
    min_strike = _float(ladder.get("min_strike"))
    max_strike = _float(ladder.get("max_strike"))
    right_text = ladder.get("right_text") or "unknown"
    if min_strike is not None and max_strike is not None:
        spx_text = fmt_number(spx, digits=2) if spx is not None else "unknown"
        return f"{min_strike:.0f}-{max_strike:.0f} {right_text} around SPX {spx_text}"
    return summary or "unknown"


def last_decision_panel(decision: dict[str, Any], candidate: dict[str, Any]) -> str:
    gate = _object(candidate.get("gate_diagnostics"))
    context = _object(candidate.get("live_index_context"))
    rows = [
        ("Time", local_time(decision.get("timestamp") or candidate.get("timestamp"))),
        ("Action", decision.get("action") or "wait"),
        ("Reason", decision.get("reason") or candidate.get("reason") or ""),
        ("Candidates", candidate.get("candidate_count", "")),
        ("Live Context", live_context_summary(context)),
        ("Gate Diagnosis", gate.get("filter_reason", "")),
        ("Time Bucket", gate.get("time_bucket", "")),
        ("Max Edge", fmt_number(gate.get("max_edge"), digits=2)),
        ("Best Call / Put", edge_pair(gate)),
        ("Edge Gate", fmt_number(gate.get("min_edge"), digits=2)),
        ("Top Rejected", top_rejected_summary(gate)),
        ("Score", fmt_number(decision.get("score"), digits=4) if decision.get("score") is not None else ""),
        ("Threshold", fmt_number(decision.get("threshold") if decision.get("threshold") is not None else candidate.get("threshold"), digits=4)),
        ("Contract", decision.get("contract") or ""),
    ]
    return key_value_table(rows)


def live_context_summary(context: dict[str, Any]) -> str:
    rows = int(_float(context.get("row_count")) or 0)
    span = _float(context.get("span_minutes"))
    if not rows:
        return ""
    return f"{rows} SPX/VIX rows over {fmt_number(span, digits=1)} min"


def edge_pair(gate: dict[str, Any]) -> str:
    call = fmt_number(gate.get("best_call_edge"), digits=2)
    put = fmt_number(gate.get("best_put_edge"), digits=2)
    if call == "unknown" and put == "unknown":
        return ""
    return f"{call} / {put}"


def top_rejected_summary(gate: dict[str, Any]) -> str:
    rows = gate.get("top_rejected_contracts") or gate.get("top_surface_tokens") or []
    if not rows:
        return ""
    top = _object(rows[0])
    contract = str(top.get("contract_id") or "")
    if not contract:
        return ""
    return f"{contract} edge {fmt_number(top.get('edge'), digits=2)} ask {fmt_number(top.get('ask'), digits=2)}"


def key_value_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in rows)
    return "<table><tbody>" + body + "</tbody></table>"


def shadow_panel(shadow: dict[str, Any]) -> str:
    rows = [
        ("Live rows", shadow.get("live_rows", 0), "ok"),
        ("Delayed rows", shadow.get("delayed_rows", 0), "warn" if int(shadow.get("delayed_rows") or 0) else "ok"),
        ("Shadow rows", shadow.get("total_rows", 0), ""),
        ("Order intents", shadow.get("order_intent_non_null_rows", 0), "bad" if int(shadow.get("order_intent_non_null_rows") or 0) else "ok"),
        ("Latest SPX", shadow.get("latest_spx", ""), ""),
    ]
    summary = '<div class="pill-row">' + "".join(
        f'<span class="pill {klass}">{esc(label)} <strong>{esc(value)}</strong></span>' for label, value, klass in rows
    ) + "</div>"
    actions = counts_pills(shadow.get("action_counts") or {}, label="Action")
    states = counts_pills(shadow.get("position_state_counts") or {}, label="State")
    top_contracts = shadow.get("top_contracts_observed") or []
    contract_rows = "".join(
        f"<tr><td>{esc(row.get('contract_id'))}</td><td>{esc(row.get('rows'))}</td></tr>" for row in top_contracts[:8]
    )
    top = (
        "<p class=\"muted\">No session contract counts yet.</p>"
        if not contract_rows
        else "<table><thead><tr><th>Session Top Contract</th><th>Rows</th></tr></thead><tbody>" + contract_rows + "</tbody></table>"
    )
    return (
        summary
        + f'<p class="muted">Latest ladder time: {esc(local_time(shadow.get("latest_observation_time")))}</p>'
        + "<h3>Actions</h3>"
        + actions
        + "<h3>Position States</h3>"
        + states
        + "<h3>Historical Session Contract Frequency</h3>"
        + top
    )


def counts_pills(counts: dict[str, Any], *, label: str) -> str:
    if not counts:
        return f'<p class="muted">No {esc(label.lower())} counts yet.</p>'
    return '<div class="pill-row">' + "".join(
        f'<span class="pill">{esc(label)} {esc(key)} <strong>{esc(value)}</strong></span>' for key, value in sorted(counts.items())
    ) + "</div>"


def failures_panel(failures: dict[str, Any]) -> str:
    active = failures.get("reason_counts") or {}
    resolved = failures.get("resolved_reason_counts") or {}
    if not active and not resolved:
        return '<div class="status-line"><strong>No active failures.</strong><br>Current live data and paper-monitor plumbing are passing.</div>'
    sections = []
    if active:
        sections.append('<div class="status-line bad"><strong>Active blockers</strong></div>')
        sections.append(reason_table(active))
    else:
        sections.append('<div class="status-line"><strong>No active failures.</strong><br>Current live data and paper-monitor plumbing are passing.</div>')
    if resolved:
        sections.append('<div class="status-line warn"><strong>Resolved earlier warnings</strong><br>These appeared earlier in the log, but current live capture/parity is passing.</div>')
        sections.append(reason_table(resolved))
    return "".join(sections)


def active_failures_panel(failures: dict[str, Any]) -> str:
    active = failures.get("reason_counts") or {}
    if not active:
        return '<div class="status-line"><strong>None</strong><br>No active blocker is preventing live paper monitoring.</div>'
    return '<div class="status-line bad"><strong>Needs attention</strong></div>' + reason_table(active)


def reason_table(counts: dict[str, Any]) -> str:
    body = "".join(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>" for key, value in counts.items())
    return "<table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>" + body + "</tbody></table>"


def contracts_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No paper order/fill contracts logged yet.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(row.get('contract'))}</td>"
            f"<td>{esc(row.get('events'))}</td>"
            f"<td>{esc(row.get('quantity_sum'))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Contract</th><th>Events</th><th>Quantity Sum</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def important_timeline_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    resolved_reasons: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    important = [row for row in rows if not is_timeline_noise(row, resolved_reasons=resolved_reasons or {})]
    return important[-max(1, int(limit)):]


def is_timeline_noise(row: dict[str, Any], *, resolved_reasons: dict[str, Any] | None = None) -> bool:
    event_type = str(row.get("event_type") or "")
    action = str(row.get("model_action") or "")
    reason = str(row.get("risk_reason") or row.get("blocked_reason") or "")
    if f"trade_log:{reason}" in (resolved_reasons or {}):
        return True
    if bool(row.get("broker_order_endpoint_called")):
        return False
    if event_type.startswith("paper_entry") or event_type.startswith("paper_exit") or event_type == "paper_order_submitted":
        return False
    if event_type in TIMELINE_NOISE_EVENTS:
        return True
    if reason in TIMELINE_NOISE_REASONS or reason.startswith("entry_bridge_pass_"):
        return True
    if action in {"enter", "exit", "stop", "forced_flat", "blocked"}:
        return False
    return False


def event_table(rows: list[dict[str, Any]], *, limit: int) -> str:
    if not rows:
        return '<div class="empty">No trade, order, fill, exit, or active blocker yet. Startup retries, resolved blockers, and wait/no-entry heartbeats are hidden.</div>'
    columns = [
        ("time_pt", "Time (PT)"),
        ("event_type", "Event"),
        ("model_action", "Model"),
        ("risk_reason", "Reason"),
        ("contract_id", "Contract"),
        ("action", "Order"),
        ("quantity", "Qty"),
        ("limit_price", "Limit"),
        ("avg_fill_price", "Avg Fill"),
        ("cash", "Cash"),
        ("equity", "Equity"),
        ("realized_daily_pnl", "Day P&L"),
        ("open_positions", "Open"),
        ("bid", "Bid"),
        ("ask", "Ask"),
    ]
    body = []
    for row in rows[-max(1, int(limit)):]:
        body.append("<tr>" + "".join(f"<td>{esc(event_cell(row, key))}</td>" for key, _ in columns) + "</tr>")
    return "<table><thead><tr>" + "".join(f"<th>{esc(label)}</th>" for _, label in columns) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def event_cell(row: dict[str, Any], column: str) -> Any:
    if column == "time_pt":
        return local_time(row.get("timestamp"))
    return row.get(column, "")


def launchd_value(row: dict[str, Any]) -> str:
    if not row:
        return "not checked"
    loaded = "loaded" if row.get("loaded") else "not loaded"
    state = row.get("state") or "unknown"
    runs = row.get("runs")
    exit_code = row.get("last_exit_code")
    return f"{loaded}; {state}; runs={runs}; last_exit={exit_code}"


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _float(value)
        if number is not None:
            return number
    return None


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


if __name__ == "__main__":
    raise SystemExit(main())
