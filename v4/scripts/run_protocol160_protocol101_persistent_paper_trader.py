"""Protocol 160: persistent Protocol101 IBKR paper trader.

Protocol147/158 proved the live data and guarded paper-order path, but they
run as repeated short capture windows. This runner keeps one IBKR connection
and one option ladder subscription alive for the full session, while preserving
the frozen Protocol101 entry model, Protocol066/081 lifecycle exit model, and
paper-order guardrails.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from v4.live.ibkr_paper_executor import PaperExecutionConfig, execute_guarded_paper_order
from v4.live.paper_trade_log import (
    DEFAULT_TRADE_LOG_ROOT,
    export_trade_log_csv,
    load_trade_log,
    stable_json_hash,
    trade_log_path,
    validate_observability_contract,
    validate_trade_log,
)
from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol066_inference import load_protocol066_artifact
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_frame_from_surface,
    protocol101_candidate_gate_diagnostics,
)
from v4.live.protocol101_clean_window_guard import Protocol101CleanWindowGuard
from v4.live.protocol101_live_entry import (
    LiveIndexState,
    build_live_surface_row,
    live_surface_decision,
    order_intent_from_prediction,
    selected_contract_payload,
)
from v4.ops.ibkr.ibkr_account_snapshot import parse_account_summary_rows
from v4.scripts.run_protocol081_live_shadow_router import (
    DEFAULT_PROTOCOL081_MANIFEST,
    IbkrErrorLog,
    _connect_ibkr,
    _discover_spxw_0dte_contracts,
    _is_regular_market_hours,
    _market_data_type_name,
    _request_index_ticker,
    _round_to_5,
    _ticker_market_data_type,
    _ticker_price,
    _wait_for_price,
)
from v4.scripts.run_protocol119_protocol101_live_readiness import (
    DEFAULT_PROTOCOL101_MANIFEST,
    DEFAULT_PROTOCOL101_SUMMARY,
)
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST
from v4.scripts.run_protocol150_protocol101_paper_order_enablement_gate import DEFAULT_RUNTIME_FLAG
from v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge import (
    DEFAULT_PRE_LIVE_SANITY_GATE_SUMMARY,
    append_event,
    append_live_index_context,
    append_time_bucket_block,
    candidate_contracts_payload,
    candidate_feature_hash,
    candidate_sample,
    clean_json,
    first_managed_account,
    handle_open_position,
    intent_order_payload,
    live_context_ready,
    live_option_quotes,
    load_json,
    load_live_index_context,
    log_fill_or_status,
    market_snapshot_payload,
    model_reconstruction_payload,
    pre_live_sanity_gate_status,
    protocol101_blocked_decision_trace_extra,
    protocol101_decision_trace_extra,
    redact_account,
    risk_gate_payload,
    runtime_flag_summary,
    spxw_open_positions,
    state_from_entry_result,
    time_bucket,
    validate_intent,
    variant_for,
    write_runtime_state,
)


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("intent-shadow", "paper-dry-run", "paper-submit"), default="paper-submit")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--runtime-flag", type=Path, default=DEFAULT_RUNTIME_FLAG)
    parser.add_argument("--runtime-state", type=Path, default=Path("v4/runtime/protocol101_live_paper_state.json"))
    parser.add_argument("--live-index-context-log", type=Path, default=Path("v4/runtime/protocol101_live_index_context.jsonl"))
    parser.add_argument("--pre-live-sanity-gate-summary", type=Path, default=DEFAULT_PRE_LIVE_SANITY_GATE_SUMMARY)
    parser.add_argument("--skip-pre-live-sanity-gate", action="store_true")
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=160)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--open-positions", type=int, default=0)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--order-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--market-close-time", default="16:00")
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--live-strikes-around-atm", type=int, default=10)
    parser.add_argument("--quote-loop-seconds", type=float, default=1.0)
    parser.add_argument("--entry-decision-interval-seconds", type=float, default=60.0)
    parser.add_argument("--entry-decision-mode", choices=("minute", "interval"), default="minute")
    parser.add_argument("--decision-interval-seconds", type=float, default=None, help="Deprecated alias for --entry-decision-interval-seconds.")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--contract-refresh-seconds", type=float, default=60.0)
    parser.add_argument("--refresh-contracts-drift-points", type=float, default=15.0)
    parser.add_argument("--min-live-context-minutes", type=float, default=30.0)
    parser.add_argument("--clean-window-warmup-minutes", type=int, default=15)
    parser.add_argument(
        "--decision-shadow-log",
        type=Path,
        default=None,
        help="Append-only canonical decision shadow; defaults inside the run output.",
    )
    parser.add_argument("--reconnect-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-reconnects", type=int, default=200)
    parser.add_argument("--max-decisions", type=int, default=100_000)
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--skip-market-clock", action="store_true")
    parser.add_argument(
        "--wait-for-market-open",
        action="store_true",
        help="If launched before RTH, keep the loaded process alive until 09:30 ET instead of exiting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(tz=NY)
    session = args.session_date or now.date().isoformat()
    run_id = args.run_id or f"protocol101_persistent-paper_{session}"
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.decision_shadow_log is None:
        args.decision_shadow_log = out_dir / "decision_shadow.jsonl"
    trade_log = trade_log_path(root=args.trade_log_root, session=session, run_id=run_id)
    csv_log = trade_log.with_suffix(".csv")

    if not args.skip_market_clock and not _is_regular_market_hours(now):
        if args.wait_for_market_open and now < session_open(now):
            wait_for_market_open(now)
        else:
            payload = blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="outside_regular_market_hours")
            return finish(out_dir, payload, trade_log, csv_log)

    try:
        payload = run_persistent_trader(args, session=session, run_id=run_id, trade_log=trade_log)
    except Exception as exc:
        payload = blocked_payload(
            args,
            session=session,
            run_id=run_id,
            trade_log=trade_log,
            reason="protocol160_exception",
            detail=str(exc),
        )
    return finish(out_dir, payload, trade_log, csv_log)


def run_persistent_trader(args: argparse.Namespace, *, session: str, run_id: str, trade_log: Path) -> dict[str, Any]:
    try:
        from ib_insync import IB, Index, LimitOrder, Option  # type: ignore
    except ImportError:
        return blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="missing_ib_insync")

    surface_artifact = load_surface_edge_artifact(args.surface_manifest)
    protocol101 = load_protocol101_entry_artifact(args.protocol101_manifest, args.protocol101_summary)
    lifecycle = load_protocol066_artifact(DEFAULT_PROTOCOL081_MANIFEST)
    variant = variant_for(surface_artifact.variant_name)
    artifact_ids = {
        "surface_manifest": str(args.surface_manifest),
        "surface_variant": str(surface_artifact.variant_name),
        "protocol101_manifest": str(args.protocol101_manifest),
        "protocol101_threshold": protocol101.threshold,
        "lifecycle_manifest": str(DEFAULT_PROTOCOL081_MANIFEST),
    }
    history = Protocol101HistoryState()
    index_state = LiveIndexState()
    clean_window_guard = Protocol101CleanWindowGuard(
        warmup_minutes=int(args.clean_window_warmup_minutes)
    )
    context_rows_loaded = load_live_index_context(args.live_index_context_log, index_state, session=session, include_prior_session_close=True)
    runtime_flag = load_json(args.runtime_flag)
    sanity_gate = pre_live_sanity_gate_status(args)
    if args.mode == "paper-submit" and not sanity_gate["passed"]:
        return blocked_payload(
            args,
            session=session,
            run_id=run_id,
            trade_log=trade_log,
            reason="pre_live_historical_sanity_gate_not_passed",
            detail=sanity_gate["reason"],
            extra={"pre_live_historical_sanity_gate": sanity_gate},
        )
    close_deadline = session_deadline(datetime.now(tz=NY), args.market_close_time)

    append_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="protocol160_persistent_trader_started",
        extra={
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "artifact_ids": artifact_ids,
            "live_orders_enabled": args.mode == "paper-submit",
            "quote_loop_seconds": float(args.quote_loop_seconds),
            "entry_decision_interval_seconds": entry_decision_interval_seconds(args),
            "entry_decision_mode": str(args.entry_decision_mode),
            "contract_refresh_seconds": float(args.contract_refresh_seconds),
            "refresh_contracts_drift_points": float(args.refresh_contracts_drift_points),
            "context_rows_loaded": int(context_rows_loaded),
            "clean_window_warmup_minutes": int(args.clean_window_warmup_minutes),
            "decision_shadow_log": str(args.decision_shadow_log),
        },
    )

    ib = None
    error_log = IbkrErrorLog()
    subscribed_index_contracts: list[Any] = []
    option_subscriptions: list[tuple[Any, Any]] = []
    chain_meta: dict[str, Any] = {}
    ladder_atm: int | None = None
    last_contract_refresh = 0.0
    last_heartbeat = 0.0
    next_decision_monotonic = 0.0
    last_entry_decision_minute: str | None = None
    decisions: list[dict[str, Any]] = []
    executor_results: list[dict[str, Any]] = []
    reconnects = 0
    paper_submit_blocked = 0
    broker_order_endpoint_called = False
    account_id = args.account_id
    connected_port = None
    connection_attempts: list[dict[str, Any]] = []
    spx_ticker = None
    vix_ticker = None
    spx_contract = None

    while keep_running(args, close_deadline=close_deadline) and len(decisions) < int(args.max_decisions):
        try:
            if ib is None or not ib.isConnected():
                if reconnects >= int(args.max_reconnects):
                    append_event(
                        trade_log,
                        event_type="paper_error",
                        session=session,
                        run_id=run_id,
                        mode=args.mode,
                        paper_cash=args.paper_cash,
                        reason="max_reconnects_exhausted",
                        extra={"max_reconnects": int(args.max_reconnects)},
                    )
                    break
                cleanup_subscriptions(ib, subscribed_index_contracts, option_subscriptions)
                ib, connected_port, attempts = _connect_ibkr(args, IB)
                connection_attempts.extend(attempts)
                reconnects += 1
                if ib is None:
                    clean_window_guard.interrupt("ibkr_connection_failed")
                    append_event(
                        trade_log,
                        event_type="paper_error",
                        session=session,
                        run_id=run_id,
                        mode=args.mode,
                        paper_cash=args.paper_cash,
                        reason="persistent_ibkr_connection_failed",
                        extra={"connection_attempts": attempts[-5:]},
                    )
                    time.sleep(max(1.0, float(args.reconnect_sleep_seconds)))
                    continue
                clean_window_guard.interrupt(
                    "ibkr_connection_established"
                    if reconnects == 1
                    else "ibkr_reconnected"
                )
                error_log = IbkrErrorLog()
                ib.errorEvent += error_log.handler
                account_id = account_id or first_managed_account(ib)
                ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)
                spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
                vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
                subscribed_index_contracts = [spx_contract, vix_contract]
                option_subscriptions = []
                chain_meta = {}
                ladder_atm = None
                last_contract_refresh = 0.0
                append_event(
                    trade_log,
                    event_type="heartbeat",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="persistent_ibkr_connected",
                    extra={"ibkr_port": connected_port, "account_id_redacted": redact_account(account_id), "reconnect_count": reconnects - 1},
                )

            assert ib is not None
            ib.sleep(max(0.25, min(1.0, float(args.quote_loop_seconds))))
            now_utc = datetime.now(tz=UTC)
            now_local = now_utc.astimezone(NY)

            if not args.skip_market_clock and not _is_regular_market_hours(now_local):
                break

            current_spx = _ticker_price(spx_ticker)
            current_vix = _ticker_price(vix_ticker)
            if current_spx is None or current_vix is None:
                clean_window_guard.interrupt("missing_live_context_snapshot")
                append_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="missing_live_context_snapshot",
                    extra={"ibkr_errors": error_log.events[-10:]},
                )
                continue

            index_state.add(timestamp=now_utc, spx=float(current_spx), vix=float(current_vix))
            append_live_index_context(
                args.live_index_context_log,
                session=session,
                timestamp=now_utc,
                spx=float(current_spx),
                vix=float(current_vix),
            )

            if should_emit_heartbeat(last_heartbeat, heartbeat_seconds=float(args.heartbeat_seconds)):
                last_heartbeat = time.monotonic()
                append_event(
                    trade_log,
                    event_type="heartbeat",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="persistent_connection_healthy",
                    market_snapshot={
                        "underlying": {"spx": float(current_spx), "vix": float(current_vix)},
                        "option_nbbo": {"subscribed_contracts": len(option_subscriptions)},
                        "context": {"source": "ibkr_live", "context_age_ms": 0},
                    },
                    extra={"ibkr_port": connected_port, "ladder_atm": ladder_atm, "reconnect_count": reconnects - 1},
                )

            refresh_reason = contract_refresh_reason(
                spx=float(current_spx),
                ladder_atm=ladder_atm,
                last_refresh_monotonic=last_contract_refresh,
                refresh_seconds=float(args.contract_refresh_seconds),
                drift_points=float(args.refresh_contracts_drift_points),
            )
            if refresh_reason:
                cleanup_option_subscriptions(ib, option_subscriptions)
                option_subscriptions, chain_meta, ladder_atm = refresh_option_ladder(
                    ib=ib,
                    option_cls=Option,
                    spx_contract=spx_contract,
                    spx=float(current_spx),
                    strikes_around_atm=int(args.live_strikes_around_atm),
                    now=now_local,
                )
                last_contract_refresh = time.monotonic()
                append_event(
                    trade_log,
                    event_type="heartbeat",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="persistent_contract_ladder_refreshed",
                    market_snapshot={
                        "underlying": {"spx": float(current_spx), "vix": float(current_vix)},
                        "option_nbbo": {"subscribed_contracts": len(option_subscriptions)},
                        "context": {"source": "ibkr_live", "context_age_ms": 0},
                    },
                    extra={"refresh_reason": refresh_reason, "chain_meta": chain_meta},
                )
                if not option_subscriptions:
                    clean_window_guard.interrupt("empty_option_subscription_ladder")
                    continue

            live_positions = spxw_open_positions(ib)
            if live_positions:
                should_evaluate = True
            else:
                should_evaluate, next_decision_monotonic, last_entry_decision_minute = should_evaluate_entry(
                    args=args,
                    now_utc=now_utc,
                    next_decision_monotonic=next_decision_monotonic,
                    last_entry_decision_minute=last_entry_decision_minute,
                )
            if not should_evaluate:
                continue
            result = evaluate_once(
                args=args,
                ib=ib,
                option_cls=Option,
                order_cls=LimitOrder,
                lifecycle=lifecycle,
                surface_artifact=surface_artifact,
                protocol101=protocol101,
                variant=variant,
                history=history,
                index_state=index_state,
                option_subscriptions=option_subscriptions,
                spx_ticker=spx_ticker,
                vix_ticker=vix_ticker,
                account_id=account_id,
                trade_log=trade_log,
                session=session,
                run_id=run_id,
                runtime_flag=runtime_flag,
                artifact_ids=artifact_ids,
                live_positions=live_positions,
                ibkr_errors=error_log.events[-20:],
                clean_window_guard=clean_window_guard,
                decision_shadow_log=args.decision_shadow_log,
            )
            decisions.extend(result["decisions"])
            executor_results.extend(result["executor_results"])
            paper_submit_blocked += int(result.get("paper_submit_blocked", 0))
            broker_order_endpoint_called = broker_order_endpoint_called or bool(result.get("broker_order_endpoint_called"))
            runtime_flag = load_json(args.runtime_flag)
        except KeyboardInterrupt:
            append_event(
                trade_log,
                event_type="paper_error",
                session=session,
                run_id=run_id,
                mode=args.mode,
                paper_cash=args.paper_cash,
                reason="persistent_trader_interrupted",
            )
            break
        except Exception as exc:
            append_event(
                trade_log,
                event_type="paper_error",
                session=session,
                run_id=run_id,
                mode=args.mode,
                paper_cash=args.paper_cash,
                reason="persistent_loop_exception",
                extra={"detail": str(exc), "ibkr_errors": error_log.events[-20:]},
            )
            cleanup_subscriptions(ib, subscribed_index_contracts, option_subscriptions)
            clean_window_guard.interrupt("persistent_loop_exception")
            ib = None
            option_subscriptions = []
            time.sleep(max(1.0, float(args.reconnect_sleep_seconds)))

    cleanup_subscriptions(ib, subscribed_index_contracts, option_subscriptions)
    if ib is not None and ib.isConnected():
        ib.disconnect()

    return {
        "protocol": "160_protocol101_persistent_paper_trader",
        "decision": decide(
            mode=args.mode,
            decisions=decisions,
            executor_results=executor_results,
            paper_submit_blocked=paper_submit_blocked,
            broker_order_endpoint_called=broker_order_endpoint_called,
        ),
        "mode": args.mode,
        "paid_data_downloaded": False,
        "real_money_trading": False,
        "live_orders": False,
        "paper_orders_submitted": sum(1 for row in executor_results if row.get("paper_order_submitted")),
        "broker_order_endpoint_called": bool(broker_order_endpoint_called),
        "ibkr_connected": connected_port is not None,
        "ibkr_port": connected_port,
        "connection_attempts": connection_attempts[-20:],
        "reconnect_count": max(0, reconnects - 1),
        "account_id_redacted": redact_account(account_id),
        "runtime_flag": runtime_flag_summary(runtime_flag),
        "live_index_context": {
            "path": str(args.live_index_context_log),
            "rows_loaded": int(context_rows_loaded),
            "session_summary": clean_json(index_state.session_context_summary(datetime.now(tz=UTC))),
        },
        "clean_window_guard": {
            "warmup_minutes": int(args.clean_window_warmup_minutes),
            "healthy_streak_minutes": int(
                clean_window_guard.healthy_streak_minutes
            ),
            "interruption_count": int(clean_window_guard.interruption_count),
            "decision_shadow_log": str(args.decision_shadow_log),
        },
        "chain_meta": chain_meta,
        "decision_count": len(decisions),
        "enter_intents": sum(1 for row in decisions if row.get("action") == "enter"),
        "executor_results": summarize_executor_results(executor_results),
        "paper_submit_blocked": paper_submit_blocked,
        "ibkr_errors": error_log.events[-30:],
        "trade_log": str(trade_log),
        "next_gate": "Use the persistent log as the source of truth for end-of-day paper/live parity and timing analysis.",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "market_close_time": args.market_close_time,
        "forced_flat_time": args.forced_flat_time,
    }


def clean_window_health_reasons(
    *,
    option_quotes: list[dict[str, Any]],
    spx: float,
    spx_ticker: Any,
    vix_ticker: Any,
    context_summary: dict[str, Any],
    min_context_minutes: float,
) -> tuple[str, ...]:
    """Return causal reasons that make a minute ineligible for new entries."""
    reasons: list[str] = []
    atm = int(_round_to_5(spx))
    expected_slots = {
        (float(atm + offset), right)
        for offset in range(-50, 51, 5)
        for right in ("C", "P")
    }
    observed_slots = {
        (float(row.get("strike", 0.0)), str(row.get("right", "")).upper())
        for row in option_quotes
    }
    if observed_slots != expected_slots:
        reasons.append("incomplete_or_shifted_static_ladder")

    quote_ages = [row.get("quote_age_ms") for row in option_quotes]
    if any(age is None or not math.isfinite(float(age)) for age in quote_ages):
        reasons.append("missing_quote_timestamp")
    elif any(float(age) > 90_000.0 for age in quote_ages):
        reasons.append("stale_quote")

    if _market_data_type_name(_ticker_market_data_type(spx_ticker)) != "live":
        reasons.append("spx_not_live")
    if _market_data_type_name(_ticker_market_data_type(vix_ticker)) != "live":
        reasons.append("vix_not_live")
    if not live_context_ready(
        context_summary, min_context_minutes=float(min_context_minutes)
    ):
        reasons.append("insufficient_live_index_context")
    return tuple(sorted(set(reasons)))


def append_decision_shadow(
    path: Path | None,
    *,
    session: str,
    run_id: str,
    mode: str,
    timestamp: datetime,
    action: str,
    reason: str,
    guard_status: Any = None,
    market_snapshot: dict[str, Any] | None = None,
    candidate_contracts: list[dict[str, Any]] | None = None,
    selected_contract: dict[str, Any] | None = None,
    trace_extra: dict[str, Any] | None = None,
    canonical_input: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    payload = {
        "schema_version": "Protocol101DecisionShadowV1",
        "session_date": session,
        "run_id": run_id,
        "mode": mode,
        "decision_timestamp_utc": timestamp.astimezone(UTC).isoformat(),
        "action": action,
        "reason": reason,
        "clean_window_guard": (
            asdict(guard_status) if guard_status is not None else None
        ),
        "market_snapshot": market_snapshot or {},
        "candidate_contracts": candidate_contracts or [],
        "selected_contract": selected_contract or {},
        "decision_trace": trace_extra or {},
        "canonical_input": canonical_input or {},
        "broker_order_endpoint_called": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(shadow_json(payload), sort_keys=True, separators=(",", ":"))
            + "\n"
        )


def shadow_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): shadow_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shadow_json(item) for item in value]
    if hasattr(value, "tolist"):
        return shadow_json(value.tolist())
    if hasattr(value, "item"):
        try:
            return shadow_json(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def evaluate_once(
    *,
    args: argparse.Namespace,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    lifecycle: Any,
    surface_artifact: Any,
    protocol101: Any,
    variant: Any,
    history: Protocol101HistoryState,
    index_state: LiveIndexState,
    option_subscriptions: list[tuple[Any, Any]],
    spx_ticker: Any,
    vix_ticker: Any,
    account_id: str | None,
    trade_log: Path,
    session: str,
    run_id: str,
    runtime_flag: dict[str, Any],
    artifact_ids: dict[str, Any] | None = None,
    live_positions: list[Any] | None = None,
    ibkr_errors: list[dict[str, Any]] | None = None,
    clean_window_guard: Protocol101CleanWindowGuard | None = None,
    decision_shadow_log: Path | None = None,
) -> dict[str, Any]:
    artifact_ids = artifact_ids or {}
    now = datetime.now(tz=UTC)
    current_spx = _ticker_price(spx_ticker)
    current_vix = _ticker_price(vix_ticker)
    if current_spx is None or current_vix is None:
        if clean_window_guard is not None:
            clean_window_guard.observe(
                decision_time=now,
                healthy=False,
                reasons=("missing_live_context_snapshot",),
            )
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="missing_live_context_snapshot",
        )
        return {"decisions": [], "executor_results": [], "paper_submit_blocked": 0, "broker_order_endpoint_called": False}

    if live_positions is None:
        live_positions = spxw_open_positions(ib)
    if live_positions:
        position_result = handle_open_position(
            args=args,
            ib=ib,
            option_cls=option_cls,
            order_cls=order_cls,
            lifecycle=lifecycle,
            position=live_positions[0],
            spx=float(current_spx),
            vix=float(current_vix),
            now=now,
            account_id=account_id,
            trade_log=trade_log,
            session=session,
            run_id=run_id,
        )
        return {
            "decisions": [position_result["decision_record"]],
            "executor_results": position_result.get("executor_results", []),
            "paper_submit_blocked": 0,
            "broker_order_endpoint_called": bool(position_result.get("broker_order_endpoint_called")),
        }

    option_quotes = live_option_quotes(option_subscriptions, spx=float(current_spx), now=now)
    append_event(
        trade_log,
        event_type="market_snapshot",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="live_snapshot",
        market_snapshot=market_snapshot_payload(
            spx=float(current_spx),
            vix=float(current_vix),
            spx_ticker=spx_ticker,
            vix_ticker=vix_ticker,
            observed_at=now,
            option_quotes=option_quotes,
            context_summary=None,
        ),
    )
    if not option_quotes:
        guard_status = None
        if clean_window_guard is not None:
            guard_status = clean_window_guard.observe(
                decision_time=now,
                healthy=False,
                reasons=("no_valid_spxw_nbbo_quotes",),
            )
        append_decision_shadow(
            decision_shadow_log,
            session=session,
            run_id=run_id,
            mode=args.mode,
            timestamp=now,
            action="wait",
            reason="no_valid_spxw_nbbo_quotes",
            guard_status=guard_status,
        )
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="no_valid_spxw_nbbo_quotes",
            extra={
                "ibkr_errors": ibkr_errors or [],
                "runtime_flag": runtime_flag_summary(runtime_flag),
                "artifact_ids": artifact_ids,
            },
        )
        return {"decisions": [], "executor_results": [], "paper_submit_blocked": 0, "broker_order_endpoint_called": False}

    normalized_row, lookup = build_live_surface_row(
        decision_time=now,
        spx=float(current_spx),
        vix=float(current_vix),
        option_quotes=option_quotes,
        index_state=index_state,
    )
    append_live_index_context(
        args.live_index_context_log,
        session=session,
        timestamp=now,
        spx=float(current_spx),
        vix=float(current_vix),
    )
    context_summary = index_state.session_context_summary(now)
    decision_market = market_snapshot_payload(
        spx=float(current_spx),
        vix=float(current_vix),
        spx_ticker=spx_ticker,
        vix_ticker=vix_ticker,
        observed_at=now,
        option_quotes=option_quotes,
        context_summary=context_summary,
    )
    health_reasons = clean_window_health_reasons(
        option_quotes=option_quotes,
        spx=float(current_spx),
        spx_ticker=spx_ticker,
        vix_ticker=vix_ticker,
        context_summary=context_summary,
        min_context_minutes=float(args.min_live_context_minutes),
    )
    guard_status = None
    if clean_window_guard is not None:
        guard_status = clean_window_guard.observe(
            decision_time=now,
            healthy=not health_reasons,
            reasons=health_reasons,
        )
    if guard_status is not None and not guard_status.eligible:
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="clean_window_entry_abstention",
            market_snapshot=decision_market,
            extra={
                "clean_window_guard": asdict(guard_status),
                "runtime_flag": runtime_flag_summary(runtime_flag),
                "artifact_ids": artifact_ids,
            },
        )
        append_decision_shadow(
            decision_shadow_log,
            session=session,
            run_id=run_id,
            mode=args.mode,
            timestamp=now,
            action="wait",
            reason="clean_window_entry_abstention",
            guard_status=guard_status,
            market_snapshot=decision_market,
            canonical_input=normalized_row,
        )
        return {
            "decisions": [
                {
                    "timestamp": now.isoformat(),
                    "action": "wait",
                    "reason": "clean_window_entry_abstention",
                    "candidate_count": 0,
                    "selected_contract": {},
                    "order_intent": None,
                    "validation": {
                        "passed": True,
                        "reason": "clean_window_entry_abstention",
                    },
                }
            ],
            "executor_results": [],
            "paper_submit_blocked": 0,
            "broker_order_endpoint_called": False,
        }
    bucket = time_bucket(now)
    if bucket not in {"post_open_morning", "late_afternoon"}:
        append_decision_shadow(
            decision_shadow_log,
            session=session,
            run_id=run_id,
            mode=args.mode,
            timestamp=now,
            action="wait",
            reason="outside_time_bucket",
            guard_status=guard_status,
            market_snapshot=decision_market,
            canonical_input=normalized_row,
        )
        append_time_bucket_block(
            args,
            trade_log,
            session,
            run_id,
            protocol101.threshold,
            now,
            context_summary,
            decision_market=decision_market,
            option_quotes=option_quotes,
            runtime_flag=runtime_flag,
            artifact_ids=artifact_ids,
        )
        return {
            "decisions": [
                {
                    "timestamp": now.isoformat(),
                    "action": "wait",
                    "reason": "outside_time_bucket",
                    "candidate_count": 0,
                    "selected_contract": {},
                    "order_intent": None,
                    "validation": {"passed": True, "reason": "outside_time_bucket"},
                }
            ],
            "executor_results": [],
            "paper_submit_blocked": 0,
            "broker_order_endpoint_called": False,
        }
    if not live_context_ready(context_summary, min_context_minutes=float(args.min_live_context_minutes)):
        append_decision_shadow(
            decision_shadow_log,
            session=session,
            run_id=run_id,
            mode=args.mode,
            timestamp=now,
            action="wait",
            reason="insufficient_live_index_context",
            guard_status=guard_status,
            market_snapshot=decision_market,
            canonical_input=normalized_row,
        )
        append_context_block(
            args,
            trade_log,
            session,
            run_id,
            protocol101.threshold,
            context_summary,
            decision_market=decision_market,
            option_quotes=option_quotes,
            runtime_flag=runtime_flag,
            artifact_ids=artifact_ids,
        )
        return {
            "decisions": [
                {
                    "timestamp": now.isoformat(),
                    "action": "wait",
                    "reason": "insufficient_live_index_context",
                    "candidate_count": 0,
                    "selected_contract": {},
                    "order_intent": None,
                    "validation": {"passed": True, "reason": "insufficient_live_index_context"},
                }
            ],
            "executor_results": [],
            "paper_submit_blocked": 0,
            "broker_order_endpoint_called": False,
        }

    surface_decision = live_surface_decision(
        session=session,
        row=normalized_row,
        variant=variant,
        policy_index=surface_artifact.policy_index,
        index_state=index_state,
    )
    surface_scores = score_surface_decisions(surface_artifact, [surface_decision])[0]
    diagnostics = protocol101_candidate_gate_diagnostics(surface_decision, surface_scores, min_edge=float(args.min_edge), max_rows=1000)
    candidates = protocol101_candidate_frame_from_surface(surface_decision, surface_scores, history, min_edge=float(args.min_edge))
    prediction = predict_protocol101_entry(protocol101, candidates)
    if not candidates.empty:
        history.update(candidates, pd.Timestamp(now))
    intent = order_intent_from_prediction(prediction, lookup, quantity=int(args.quantity))
    selected_contract = selected_contract_payload(intent, lookup)
    validation = validate_intent(intent=intent, selected_contract=selected_contract, args=args)
    model_action = "enter" if intent is not None else "wait"
    candidate_contracts = candidate_contracts_payload(candidates, lookup)
    candidate_set_hash = stable_json_hash(candidate_contracts)
    feature_vector_hash = candidate_feature_hash(candidates)
    trace_extra = protocol101_decision_trace_extra(
        surface_decision=surface_decision,
        surface_scores=surface_scores,
        lookup=lookup,
        prediction=prediction,
    )
    append_decision_shadow(
        decision_shadow_log,
        session=session,
        run_id=run_id,
        mode=args.mode,
        timestamp=now,
        action=model_action,
        reason=str(prediction.get("reason") or model_action),
        guard_status=guard_status,
        market_snapshot=decision_market,
        candidate_contracts=candidate_contracts,
        selected_contract=selected_contract,
        trace_extra=trace_extra,
        canonical_input=normalized_row,
    )
    decision_market = market_snapshot_payload(
        spx=float(current_spx),
        vix=float(current_vix),
        spx_ticker=spx_ticker,
        vix_ticker=vix_ticker,
        observed_at=now,
        option_quotes=option_quotes,
        selected_contract=selected_contract,
        context_summary=context_summary,
    )
    decision_record = {
        "timestamp": now.isoformat(),
        "action": model_action,
        "reason": prediction.get("reason"),
        "margin": prediction.get("margin"),
        "threshold": prediction.get("threshold"),
        "candidate_count": int(len(candidates)),
        "selected_contract": selected_contract,
        "order_intent": intent.__dict__ if intent else None,
        "validation": validation,
    }
    append_event(
        trade_log,
        event_type="candidate_set",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="candidate_set_built",
        market_snapshot=decision_market,
        extra={
            "candidate_set_hash": candidate_set_hash,
            "feature_vector_hash": feature_vector_hash,
            **trace_extra,
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "artifact_ids": artifact_ids,
            "candidate_count": int(len(candidates)),
            "candidate_sample": candidate_sample(candidates),
            "candidate_contracts": candidate_contracts,
            "candidate_gate_diagnostics": clean_json(diagnostics),
            "live_index_context": clean_json(context_summary),
            "model_threshold": protocol101.threshold,
        },
    )
    append_event(
        trade_log,
        event_type="model_decision",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason=str(prediction.get("reason") or model_action),
        selected_contract=selected_contract,
        order=intent_order_payload(intent),
        market_snapshot=decision_market,
        model_decision=model_reconstruction_payload(
            prediction=prediction,
            model_action=model_action,
            selected_contract=selected_contract,
            candidate_set_hash=candidate_set_hash,
            feature_vector_hash=feature_vector_hash,
        ),
        extra={
            "candidate_set_hash": candidate_set_hash,
            "feature_vector_hash": feature_vector_hash,
            "candidate_universe_hash": trace_extra["candidate_universe_hash"],
            "feature_hash": trace_extra["feature_hash"],
            "score_hash": trace_extra["score_hash"],
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "artifact_ids": artifact_ids,
        },
    )
    append_event(
        trade_log,
        event_type="risk_gate",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason=str(validation.get("reason") or "no_intent"),
        selected_contract=selected_contract,
        order=intent_order_payload(intent),
        market_snapshot=decision_market,
        risk_gate=risk_gate_payload(validation),
        extra={
            "candidate_set_hash": candidate_set_hash,
            "feature_vector_hash": feature_vector_hash,
            "candidate_universe_hash": trace_extra["candidate_universe_hash"],
            "feature_hash": trace_extra["feature_hash"],
            "score_hash": trace_extra["score_hash"],
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "artifact_ids": artifact_ids,
        },
    )
    append_event(
        trade_log,
        event_type="paper_account_state",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="paper_account_state",
        account=paper_account_payload(
            ib,
            account_id=account_id,
            open_positions=len(live_positions),
            fallback_cash=float(args.paper_cash),
        ),
    )
    if intent is None:
        return {"decisions": [decision_record], "executor_results": [], "paper_submit_blocked": 0, "broker_order_endpoint_called": False}
    if args.mode == "intent-shadow":
        return {"decisions": [decision_record], "executor_results": [], "paper_submit_blocked": 0, "broker_order_endpoint_called": False}
    if args.mode == "paper-submit" and not bool(runtime_flag.get("paper_orders_enabled")):
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="runtime_flag_not_enabled",
            selected_contract=selected_contract,
            order=intent_order_payload(intent),
            market_snapshot=decision_market,
            risk_gate=risk_gate_payload({"passed": False, "reason": "runtime_flag_not_enabled", "reasons": ["runtime_flag_not_enabled"]}),
            extra={
                "candidate_set_hash": candidate_set_hash,
                "feature_vector_hash": feature_vector_hash,
                "candidate_universe_hash": trace_extra["candidate_universe_hash"],
                "feature_hash": trace_extra["feature_hash"],
                "score_hash": trace_extra["score_hash"],
                "runtime_flag": runtime_flag_summary(runtime_flag),
                "artifact_ids": artifact_ids,
            },
        )
        return {"decisions": [decision_record], "executor_results": [], "paper_submit_blocked": 1, "broker_order_endpoint_called": False}

    selected_quote = lookup[selected_contract["contract_id"]]
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=int(args.open_positions),
        quote={
            "bid": float(selected_quote["bid"]),
            "ask": float(selected_quote["ask"]),
            "reference_ask": float(selected_quote["ask"]),
            "quote_age_ms": selected_quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": selected_quote.get("raw_quote_timestamp_utc") or selected_quote.get("quote_timestamp"),
            "quote_timestamp": selected_quote.get("quote_timestamp"),
            "received_timestamp_utc": selected_quote.get("received_timestamp_utc") or selected_quote.get("received_timestamp"),
            "decision_timestamp_utc": selected_quote.get("decision_timestamp_utc") or selected_quote.get("decision_timestamp"),
        },
        context={"context_age_ms": 0, "underlying": decision_market["underlying"]},
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=args.mode != "paper-submit",
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=f"protocol160_{len(load_trade_log(trade_log)):06d}",
        artifact_ids=artifact_ids,
        runtime_flag_digest=stable_json_hash(runtime_flag_summary(runtime_flag)),
        wait_for_fill_seconds=float(args.order_timeout_seconds),
        cancel_unfilled=True,
    )
    log_fill_or_status(
        trade_log=trade_log,
        result=result,
        event_prefix="paper_entry",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        trade_uid=f"protocol160_{len(load_trade_log(trade_log)):06d}",
    )
    if result.get("fill_summary", {}).get("filled"):
        write_runtime_state(args.runtime_state, state_from_entry_result(result, fallback_intent=intent))
    return {
        "decisions": [decision_record],
        "executor_results": [result],
        "paper_submit_blocked": 0,
        "broker_order_endpoint_called": bool(result.get("broker_order_endpoint_called")),
    }


def append_context_block(
    args: argparse.Namespace,
    trade_log: Path,
    session: str,
    run_id: str,
    model_threshold: float,
    context_summary: dict[str, Any],
    *,
    decision_market: dict[str, Any] | None = None,
    option_quotes: list[dict[str, Any]] | None = None,
    runtime_flag: dict[str, Any] | None = None,
    artifact_ids: dict[str, Any] | None = None,
) -> None:
    opening_context_ready = context_summary.get("opening_context_ready")
    filter_reason = (
        "missing_opening_live_index_context"
        if opening_context_ready is False
        else "insufficient_live_index_context"
    )
    diagnostics = {
        "filter_reason": filter_reason,
        "canonical_filter_reason": "insufficient_index_context",
        "required_minutes": float(args.min_live_context_minutes),
        "minute_row_count": context_summary.get("minute_row_count"),
        "span_minutes": context_summary.get("span_minutes"),
        "expected_first_timestamp": context_summary.get("expected_first_timestamp"),
        "first_timestamp": context_summary.get("first_timestamp"),
        "last_timestamp": context_summary.get("last_timestamp"),
        "opening_context_ready": opening_context_ready,
        "missing_opening_minutes": context_summary.get("missing_opening_minutes"),
    }
    decision_time = datetime.now(tz=UTC)
    trace_extra = protocol101_blocked_decision_trace_extra(
        reason="insufficient_live_index_context",
        decision_time=decision_time,
        option_quotes=option_quotes,
        context_summary=context_summary,
        model_threshold=model_threshold,
    )
    append_event(
        trade_log,
        event_type="candidate_set",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="candidate_set_blocked_insufficient_live_index_context",
        market_snapshot=decision_market,
        extra={
            "candidate_set_hash": stable_json_hash({"reason": "insufficient_live_index_context", "option_quotes": option_quotes or []}),
            "feature_vector_hash": stable_json_hash([]),
            **trace_extra,
            "runtime_flag": runtime_flag_summary(runtime_flag or {}),
            "artifact_ids": artifact_ids or {},
            "candidate_count": 0,
            "candidate_sample": [],
            "candidate_gate_diagnostics": clean_json(diagnostics),
            "live_index_context": clean_json(context_summary),
            "model_threshold": model_threshold,
        },
    )
    append_event(
        trade_log,
        event_type="model_decision",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="insufficient_live_index_context",
        market_snapshot=decision_market,
        model_decision={
            "action": "wait",
            "selected_action": "wait",
            "score": None,
            "selected_margin": None,
            "threshold": model_threshold,
            "reason": "insufficient_live_index_context",
            "no_entry_reason": "insufficient_live_index_context",
            "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
            "raw_logits": [],
            "wait_logit": None,
            "candidate_logits": [],
        },
        extra={
            "candidate_set_hash": stable_json_hash({"reason": "insufficient_live_index_context", "option_quotes": option_quotes or []}),
            "feature_vector_hash": stable_json_hash([]),
            **trace_extra,
            "runtime_flag": runtime_flag_summary(runtime_flag or {}),
            "artifact_ids": artifact_ids or {},
        },
    )
    append_event(
        trade_log,
        event_type="risk_gate",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="insufficient_live_index_context",
        market_snapshot=decision_market,
        risk_gate=risk_gate_payload({"passed": True, "reason": "insufficient_live_index_context", "reasons": []}),
        extra={"runtime_flag": runtime_flag_summary(runtime_flag or {}), "artifact_ids": artifact_ids or {}, **trace_extra},
    )
    append_event(
        trade_log,
        event_type="paper_account_state",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="paper_account_state",
    )


def refresh_option_ladder(
    *,
    ib: Any,
    option_cls: Any,
    spx_contract: Any,
    spx: float,
    strikes_around_atm: int,
    now: datetime,
) -> tuple[list[tuple[Any, Any]], dict[str, Any], int | None]:
    contracts, chain_meta = _discover_spxw_0dte_contracts(
        ib=ib,
        option_cls=option_cls,
        spx_contract=spx_contract,
        spx_price=float(spx),
        now=now,
        strikes_around_atm=int(strikes_around_atm),
    )
    subscriptions = [(contract, ib.reqMktData(contract, "", False, False)) for contract in contracts]
    return subscriptions, chain_meta, int(chain_meta.get("atm_strike")) if chain_meta.get("atm_strike") is not None else None


def contract_refresh_reason(
    *,
    spx: float,
    ladder_atm: int | None,
    last_refresh_monotonic: float,
    refresh_seconds: float,
    drift_points: float,
) -> str | None:
    if ladder_atm is None or last_refresh_monotonic <= 0.0:
        return "initial_ladder"
    atm = _round_to_5(spx)
    if abs(float(atm) - float(ladder_atm)) >= max(5.0, float(drift_points)):
        return "atm_drift"
    if time.monotonic() - float(last_refresh_monotonic) >= max(1.0, float(refresh_seconds)):
        return "scheduled_refresh"
    return None


def should_emit_heartbeat(last_heartbeat_monotonic: float, *, heartbeat_seconds: float) -> bool:
    return last_heartbeat_monotonic <= 0.0 or time.monotonic() - float(last_heartbeat_monotonic) >= max(1.0, float(heartbeat_seconds))


def should_evaluate_entry(
    *,
    args: argparse.Namespace,
    now_utc: datetime,
    next_decision_monotonic: float,
    last_entry_decision_minute: str | None,
) -> tuple[bool, float, str | None]:
    """Gate entry decisions to the historical decision cadence.

    The broker loop can run every second, but Protocol101 entry was trained as a
    minute-level decision policy. In minute mode, the first quote loop observed
    in each UTC minute may emit one entry decision. In interval mode, tests and
    manual diagnostics can force a wall-clock interval without changing the
    model.
    """

    if str(args.entry_decision_mode) == "interval":
        now_monotonic = time.monotonic()
        if now_monotonic < float(next_decision_monotonic):
            return False, next_decision_monotonic, last_entry_decision_minute
        return True, now_monotonic + entry_decision_interval_seconds(args), last_entry_decision_minute

    minute_key = now_utc.astimezone(UTC).replace(second=0, microsecond=0).isoformat()
    if minute_key == last_entry_decision_minute:
        return False, next_decision_monotonic, last_entry_decision_minute
    return True, next_decision_monotonic, minute_key


def entry_decision_interval_seconds(args: argparse.Namespace) -> float:
    value = getattr(args, "entry_decision_interval_seconds", None)
    if value is None:
        value = getattr(args, "decision_interval_seconds", None)
    if value is None:
        value = 60.0
    return max(1.0, float(value))


def paper_account_payload(
    ib: Any,
    *,
    account_id: str | None,
    open_positions: int,
    fallback_cash: float,
) -> dict[str, Any]:
    """Build a redacted paper-account snapshot for reconstructing live sessions."""

    base = {
        "account_id_redacted": redact_account(account_id),
        "cash": round(float(fallback_cash), 6),
        "open_positions": int(open_positions),
        "real_money_trading": False,
    }
    try:
        rows = []
        if hasattr(ib, "accountSummary"):
            rows.extend(list(ib.accountSummary(account=account_id or "") or []))
        if hasattr(ib, "accountValues"):
            rows.extend(list(ib.accountValues(account=account_id or "") or []))
        snapshot = parse_account_summary_rows(rows, account_id=account_id)
    except Exception as exc:
        return {
            **base,
            "snapshot_status": "blocked",
            "blocked_reason": "paper_account_snapshot_failed",
            "snapshot_error": str(exc),
        }

    values = snapshot.get("values") or {}
    cash = values.get("cash")
    return clean_json(
        {
            **base,
            "snapshot_status": snapshot.get("status"),
            "checked_at_utc": snapshot.get("checked_at_utc"),
            "paper_account_confirmed": snapshot.get("paper_account_confirmed"),
            "cash": round(float(cash), 6) if cash is not None else base["cash"],
            "equity": values.get("net_liquidation"),
            "available_funds": values.get("available_funds"),
            "buying_power": values.get("buying_power"),
            "realized_daily_pnl": values.get("realized_pnl"),
            "unrealized_pnl": values.get("unrealized_pnl"),
            "gross_position_value": values.get("gross_position_value"),
            "currency": values.get("currency"),
        }
    )


def cleanup_subscriptions(ib: Any, index_contracts: list[Any], option_subscriptions: list[tuple[Any, Any]]) -> None:
    cleanup_option_subscriptions(ib, option_subscriptions)
    if ib is None:
        return
    for contract in index_contracts:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass


def cleanup_option_subscriptions(ib: Any, option_subscriptions: list[tuple[Any, Any]]) -> None:
    if ib is None:
        return
    for contract, _ticker in option_subscriptions:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass
    option_subscriptions.clear()


def keep_running(args: argparse.Namespace, *, close_deadline: datetime) -> bool:
    if args.skip_market_clock:
        return True
    return datetime.now(tz=NY) <= close_deadline


def session_open(now: datetime) -> datetime:
    local = now.astimezone(NY)
    return local.replace(hour=9, minute=30, second=0, microsecond=0)


def wait_for_market_open(now: datetime) -> None:
    open_time = session_open(now)
    while datetime.now(tz=NY) < open_time:
        remaining = (open_time - datetime.now(tz=NY)).total_seconds()
        time.sleep(max(0.25, min(30.0, remaining)))


def session_deadline(now: datetime, wall: str) -> datetime:
    hour, minute = [int(part) for part in str(wall).split(":", 1)]
    local = now.astimezone(NY)
    return local.replace(hour=hour, minute=minute, second=0, microsecond=0)


def blocked_payload(
    args: argparse.Namespace,
    *,
    session: str,
    run_id: str,
    trade_log: Path,
    reason: str,
    detail: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    append_event(
        trade_log,
        event_type="paper_order_blocked",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason=reason,
        extra={"detail": detail, **(extra or {})},
    )
    return {
        "protocol": "160_protocol101_persistent_paper_trader",
        "decision": f"blocked_{reason}",
        "mode": args.mode,
        "paid_data_downloaded": False,
        "real_money_trading": False,
        "live_orders": False,
        "paper_orders_submitted": 0,
        "broker_order_endpoint_called": False,
        "decision_count": 0,
        "enter_intents": 0,
        "blocked_reason": reason,
        "detail": detail,
        "trade_log": str(trade_log),
        **(extra or {}),
    }


def decide(
    *,
    mode: str,
    decisions: list[dict[str, Any]],
    executor_results: list[dict[str, Any]],
    paper_submit_blocked: int,
    broker_order_endpoint_called: bool,
) -> str:
    if not decisions:
        return "blocked_no_persistent_live_decisions_emitted"
    if mode == "intent-shadow":
        return "pass_persistent_live_intent_shadow_logged"
    if mode == "paper-dry-run":
        if any(row.get("status") == "dry_run_pass" for row in executor_results):
            return "pass_persistent_guarded_paper_order_dry_run_logged"
        return "pass_persistent_no_entry_intents_to_dry_run"
    if any(row.get("paper_order_submitted") for row in executor_results):
        return "paper_order_submitted_persistent_trader"
    if paper_submit_blocked:
        return "blocked_persistent_paper_submit_runtime_flag_missing"
    if broker_order_endpoint_called:
        return "pass_persistent_broker_endpoint_called_no_fill"
    return "pass_persistent_no_entry_intents_to_paper_submit"


def finish(out_dir: Path, payload: dict[str, Any], trade_log: Path, csv_log: Path) -> int:
    rows = load_trade_log(trade_log)
    validation = validate_trade_log(rows)
    observability = validate_observability_contract(rows)
    csv_summary = export_trade_log_csv(trade_log, csv_log) if trade_log.exists() else {"rows": 0}
    payload["trade_log_validation"] = validation
    payload["observability_validation"] = observability
    payload["trade_log_csv"] = str(csv_log)
    payload["trade_log_csv_summary"] = csv_summary
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=clean_json) + "\n")
    write_report(out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md"), "trade_log": str(trade_log)}, indent=2))
    return 0 if validation["status"] == "pass" and not str(payload["decision"]).startswith("blocked_protocol160_exception") else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 160: Protocol101 Persistent Paper Trader",
        "",
        "No paid data was downloaded. Real-money trading is disabled.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Decisions emitted: `{payload.get('decision_count', 0)}`",
        f"- Enter intents: `{payload.get('enter_intents', 0)}`",
        f"- Paper orders submitted: `{payload.get('paper_orders_submitted', 0)}`",
        f"- Broker order endpoint called: `{payload.get('broker_order_endpoint_called', False)}`",
        f"- Reconnect count: `{payload.get('reconnect_count', 0)}`",
        f"- Trade log: `{payload.get('trade_log')}`",
        f"- Trade log CSV: `{payload.get('trade_log_csv')}`",
        "",
        "## Next Gate",
        "",
        payload.get("next_gate", "Review persistent live paper log."),
    ]
    if payload.get("blocked_reason"):
        lines.extend(["", "## Blocker", "", f"- `{payload['blocked_reason']}`", f"- {payload.get('detail')}"])
    path.write_text("\n".join(lines) + "\n")


def summarize_executor_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows[-10:]:
        out.append(
            {
                "status": row.get("status"),
                "reason": row.get("reason"),
                "paper_order_submitted": row.get("paper_order_submitted"),
                "broker_order_endpoint_called": row.get("broker_order_endpoint_called"),
                "validation": row.get("validation"),
                "permission": row.get("permission"),
            }
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
