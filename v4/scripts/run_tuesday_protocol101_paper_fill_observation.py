"""Tuesday Protocol101 paper fill observation runner.

This is not a strategy promotion and not a live-money path. It submits bounded
IBKR paper-only one-contract probes for Protocol101-selected or near-selected
SPXW 0DTE candidates so the project can collect fill/cancel/timeout evidence
without waiting for the frozen model to naturally trade enough times in one day.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from v4.live.ibkr_paper_executor import PaperExecutionConfig, execute_guarded_paper_order
from v4.live.ibkr_paper_guard import PaperOrderIntent
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
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_frame_from_surface,
    protocol101_candidate_gate_diagnostics,
)
from v4.live.protocol101_live_entry import LiveIndexState, build_live_surface_row, live_surface_decision
from v4.model.hypothesis_protocol import time_bucket
from v4.scripts.run_protocol081_live_shadow_router import (
    IbkrErrorLog,
    _connect_ibkr,
    _is_regular_market_hours,
    _market_data_type_name,
    _request_index_ticker,
    _ticker_market_data_type,
    _ticker_price,
)
from v4.scripts.run_protocol119_protocol101_live_readiness import DEFAULT_PROTOCOL101_MANIFEST, DEFAULT_PROTOCOL101_SUMMARY
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST
from v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge import (
    append_event,
    append_live_index_context,
    candidate_contracts_payload,
    candidate_feature_hash,
    clean_json,
    first_managed_account,
    intent_order_payload,
    live_context_ready,
    live_option_quotes,
    load_live_index_context,
    log_fill_or_status,
    market_snapshot_payload,
    model_reconstruction_payload,
    quote_freshness_summary,
    redact_account,
    risk_gate_payload,
    spxw_open_positions,
    variant_for,
)
from v4.scripts.run_protocol160_protocol101_persistent_paper_trader import cleanup_subscriptions, refresh_option_ladder


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation")
APPROVAL_ENV = "TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--live-index-context-log", type=Path, default=Path("v4/runtime/protocol101_live_index_context.jsonl"))
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=286)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--live-strikes-around-atm", type=int, default=10)
    parser.add_argument("--quote-warmup-seconds", type=float, default=3.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=60.0)
    parser.add_argument("--contract-refresh-seconds", type=float, default=60.0)
    parser.add_argument("--refresh-contracts-drift-points", type=float, default=15.0)
    parser.add_argument("--min-live-context-minutes", type=float, default=30.0)
    parser.add_argument("--entry-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--exit-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--exit-force-offset", type=float, default=0.50)
    parser.add_argument("--max-probes", type=int, default=8)
    parser.add_argument("--max-filled-round-trips", type=int, default=3)
    parser.add_argument("--max-observations-per-contract", type=int, default=2)
    parser.add_argument("--market-close-time", default="16:00")
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--skip-market-clock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(tz=NY)
    session = args.session_date or now.date().isoformat()
    run_id = args.run_id or f"tuesday_protocol101_paper_fill_observation_{session}"
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log = trade_log_path(root=args.trade_log_root, session=session, run_id=run_id)
    observation_log = out_dir / "execution_observations.jsonl"
    csv_log = trade_log.with_suffix(".csv")

    if not args.skip_market_clock and not _is_regular_market_hours(now):
        payload = blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="outside_regular_market_hours")
        return finish(out_dir, payload, trade_log, csv_log)
    approval = local_observation_approval(args)
    if not approval["passed"]:
        payload = blocked_payload(
            args,
            session=session,
            run_id=run_id,
            trade_log=trade_log,
            reason="paper_fill_observation_approval_missing",
            extra={"approval": approval},
        )
        return finish(out_dir, payload, trade_log, csv_log)
    try:
        payload = run_observation_session(args, session=session, run_id=run_id, trade_log=trade_log, observation_log=observation_log)
    except Exception as exc:
        payload = blocked_payload(
            args,
            session=session,
            run_id=run_id,
            trade_log=trade_log,
            reason="paper_fill_observation_exception",
            detail=str(exc),
        )
    return finish(out_dir, payload, trade_log, csv_log)


def run_observation_session(args: argparse.Namespace, *, session: str, run_id: str, trade_log: Path, observation_log: Path) -> dict[str, Any]:
    try:
        from ib_insync import IB, Index, LimitOrder, Option  # type: ignore
    except ImportError:
        return blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="missing_ib_insync")

    surface_artifact = load_surface_edge_artifact(args.surface_manifest)
    protocol101 = load_protocol101_entry_artifact(args.protocol101_manifest, args.protocol101_summary)
    variant = variant_for(surface_artifact.variant_name)
    artifact_ids = {
        "surface_manifest": str(args.surface_manifest),
        "surface_variant": str(surface_artifact.variant_name),
        "protocol101_manifest": str(args.protocol101_manifest),
        "protocol101_threshold": protocol101.threshold,
    }
    approval = local_observation_approval(args)
    history = Protocol101HistoryState()
    index_state = LiveIndexState()
    context_rows_loaded = load_live_index_context(args.live_index_context_log, index_state, session=session)
    error_log = IbkrErrorLog()
    ib = None
    subscribed_index_contracts: list[Any] = []
    option_subscriptions: list[tuple[Any, Any]] = []
    chain_meta: dict[str, Any] = {}
    ladder_atm: int | None = None
    last_contract_refresh = 0.0
    next_decision = 0.0
    connected_port = None
    connection_attempts: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    observed_contract_counts: dict[str, int] = {}
    decision_count = 0
    probe_count = 0
    filled_round_trips = 0
    broker_order_endpoint_called = False
    account_id = args.account_id
    close_deadline = session_deadline(datetime.now(tz=NY), args.market_close_time)

    append_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        mode="paper-fill-observation",
        paper_cash=float(args.paper_cash),
        reason="paper_fill_observation_started",
        extra={
            "artifact_ids": artifact_ids,
            "runtime_flag": approval,
            "approval_env": APPROVAL_ENV,
            "max_probes": int(args.max_probes),
            "max_filled_round_trips": int(args.max_filled_round_trips),
            "context_rows_loaded": int(context_rows_loaded),
        },
    )

    try:
        ib, connected_port, attempts = _connect_ibkr(args, IB)
        connection_attempts.extend(attempts)
        if ib is None:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason="ibkr_connection_failed",
                extra={"connection_attempts": attempts},
            )
        ib.errorEvent += error_log.handler
        account_id = account_id or first_managed_account(ib)
        ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)
        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
        subscribed_index_contracts = [contract for contract in (spx_contract, vix_contract) if contract is not None]
        ib.sleep(max(0.5, float(args.quote_warmup_seconds)))

        while keep_running(args, close_deadline=close_deadline) and probe_count < int(args.max_probes) and filled_round_trips < int(args.max_filled_round_trips):
            ib.sleep(0.25)
            now_utc = datetime.now(tz=UTC)
            if time.monotonic() < next_decision:
                continue
            next_decision = time.monotonic() + max(1.0, float(args.decision_interval_seconds))
            current_spx = _ticker_price(spx_ticker)
            current_vix = _ticker_price(vix_ticker)
            if current_spx is None or current_vix is None:
                append_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode="paper-fill-observation",
                    paper_cash=float(args.paper_cash),
                    reason="missing_live_spx_or_vix",
                    extra={"ibkr_errors": error_log.events[-10:]},
                )
                continue
            current_atm = round(float(current_spx) / 5.0) * 5
            refresh_due = (
                not option_subscriptions
                or time.monotonic() - last_contract_refresh >= float(args.contract_refresh_seconds)
                or ladder_atm is None
                or abs(float(current_atm) - float(ladder_atm)) >= float(args.refresh_contracts_drift_points)
            )
            if refresh_due:
                cleanup_subscriptions(ib, [], option_subscriptions)
                option_subscriptions, chain_meta, ladder_atm = refresh_option_ladder(
                    ib=ib,
                    option_cls=Option,
                    spx_contract=spx_contract,
                    spx=float(current_spx),
                    strikes_around_atm=int(args.live_strikes_around_atm),
                    now=now_utc.astimezone(NY),
                )
                last_contract_refresh = time.monotonic()
                ib.sleep(max(0.5, float(args.quote_warmup_seconds)))
                now_utc = datetime.now(tz=UTC)
                current_spx = _ticker_price(spx_ticker)
                current_vix = _ticker_price(vix_ticker)
                if current_spx is None or current_vix is None:
                    append_event(
                        trade_log,
                        event_type="paper_order_blocked",
                        session=session,
                        run_id=run_id,
                        mode="paper-fill-observation",
                        paper_cash=float(args.paper_cash),
                        reason="missing_live_spx_or_vix_after_ladder_refresh",
                        extra={"ibkr_errors": error_log.events[-10:]},
                    )
                    continue
            if not option_subscriptions:
                continue
            open_positions = spxw_open_positions(ib)
            if open_positions:
                append_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode="paper-fill-observation",
                    paper_cash=float(args.paper_cash),
                    reason="nonflat_paper_account_blocks_new_probe",
                    extra={"open_position_count": len(open_positions)},
                )
                break
            option_quotes = live_option_quotes(option_subscriptions, spx=float(current_spx), now=now_utc)
            if not option_quotes:
                continue
            index_state.add(timestamp=now_utc, spx=float(current_spx), vix=float(current_vix))
            append_live_index_context(args.live_index_context_log, session=session, timestamp=now_utc, spx=float(current_spx), vix=float(current_vix))
            context_summary = index_state.session_context_summary(now_utc)
            snapshot = market_snapshot_payload(
                spx=float(current_spx),
                vix=float(current_vix),
                spx_ticker=spx_ticker,
                vix_ticker=vix_ticker,
                observed_at=now_utc,
                option_quotes=option_quotes,
                context_summary=context_summary,
            )
            append_event(
                trade_log,
                event_type="market_snapshot",
                session=session,
                run_id=run_id,
                mode="paper-fill-observation",
                paper_cash=float(args.paper_cash),
                reason="paper_fill_observation_live_snapshot",
                market_snapshot=snapshot,
                extra={"artifact_ids": artifact_ids, "runtime_flag": approval},
            )
            normalized_row, lookup = build_live_surface_row(
                decision_time=now_utc,
                spx=float(current_spx),
                vix=float(current_vix),
                option_quotes=option_quotes,
                index_state=index_state,
            )
            decision_market = snapshot
            if not live_context_ready(context_summary, min_context_minutes=float(args.min_live_context_minutes)):
                append_event(
                    trade_log,
                    event_type="candidate_set",
                    session=session,
                    run_id=run_id,
                    mode="paper-fill-observation",
                    paper_cash=float(args.paper_cash),
                    reason="candidate_set_blocked_insufficient_live_index_context",
                    market_snapshot=decision_market,
                    extra={
                        "artifact_ids": artifact_ids,
                        "runtime_flag": approval,
                        "candidate_set_hash": stable_json_hash({"reason": "insufficient_live_index_context", "option_quotes": option_quotes}),
                        "feature_vector_hash": stable_json_hash([]),
                        "candidate_count": 0,
                        "candidate_gate_diagnostics": {"filter_reason": "insufficient_live_index_context"},
                        "live_index_context": clean_json(context_summary),
                        "quote_freshness": quote_freshness_summary(option_quotes),
                    },
                )
                continue
            surface_decision = live_surface_decision(
                session=session,
                row=normalized_row,
                variant=variant,
                policy_index=surface_artifact.policy_index,
                index_state=index_state,
            )
            surface_scores = score_surface_decisions(surface_artifact, [surface_decision])[0]
            diagnostics = protocol101_candidate_gate_diagnostics(surface_decision, surface_scores, min_edge=float(args.min_edge))
            candidates = protocol101_candidate_frame_from_surface(surface_decision, surface_scores, history, min_edge=float(args.min_edge))
            prediction = predict_protocol101_entry(protocol101, candidates)
            if not candidates.empty:
                history.update(candidates, pd.Timestamp(now_utc))
            decision_count += 1
            candidate_contracts = candidate_contracts_payload(candidates, lookup)
            candidate_set_hash = stable_json_hash(candidate_contracts)
            feature_vector_hash = candidate_feature_hash(candidates)
            append_event(
                trade_log,
                event_type="candidate_set",
                session=session,
                run_id=run_id,
                mode="paper-fill-observation",
                paper_cash=float(args.paper_cash),
                reason="paper_fill_observation_candidate_set_built",
                market_snapshot=decision_market,
                extra={
                    "artifact_ids": artifact_ids,
                    "runtime_flag": approval,
                    "candidate_set_hash": candidate_set_hash,
                    "feature_vector_hash": feature_vector_hash,
                    "candidate_count": int(len(candidates)),
                    "candidate_contracts": candidate_contracts,
                    "candidate_gate_diagnostics": clean_json(diagnostics),
                    "quote_freshness": quote_freshness_summary(option_quotes),
                    "live_index_context": clean_json(context_summary),
                },
            )
            chosen = choose_observation_candidate(
                candidates=candidates,
                prediction=prediction,
                lookup=lookup,
                observed_contract_counts=observed_contract_counts,
                paper_cash=float(args.paper_cash),
                max_observations_per_contract=int(args.max_observations_per_contract),
            )
            if chosen is None:
                append_event(
                    trade_log,
                    event_type="model_decision",
                    session=session,
                    run_id=run_id,
                    mode="paper-fill-observation",
                    paper_cash=float(args.paper_cash),
                    reason="no_probe_candidate_available",
                    market_snapshot=decision_market,
                    model_decision=model_reconstruction_payload(
                        prediction=prediction,
                        model_action="wait",
                        selected_contract={},
                        candidate_set_hash=candidate_set_hash,
                        feature_vector_hash=feature_vector_hash,
                    ),
                    extra={"artifact_ids": artifact_ids, "runtime_flag": approval, "candidate_set_hash": candidate_set_hash, "feature_vector_hash": feature_vector_hash},
                )
                continue
            candidate, quote, selection_reason = chosen
            observation = execute_probe(
                args=args,
                ib=ib,
                option_cls=Option,
                order_cls=LimitOrder,
                account_id=account_id,
                trade_log=trade_log,
                observation_log=observation_log,
                session=session,
                run_id=run_id,
                candidate=candidate,
                quote=quote,
                selection_reason=selection_reason,
                spx=float(current_spx),
                vix=float(current_vix),
                decision_time=now_utc,
                probe_index=probe_count,
                option_subscriptions=option_subscriptions,
                artifact_ids=artifact_ids,
                runtime_flag=approval,
                candidate_set_hash=candidate_set_hash,
                feature_vector_hash=feature_vector_hash,
            )
            observations.append(observation)
            observed_contract_counts[observation["contract_id"]] = observed_contract_counts.get(observation["contract_id"], 0) + 1
            probe_count += 1
            broker_order_endpoint_called = broker_order_endpoint_called or bool(observation.get("broker_order_endpoint_called"))
            if observation.get("fill_status") == "filled" and observation.get("exit_fill_status") == "filled":
                filled_round_trips += 1
            if observation.get("open_position_risk"):
                break
    finally:
        cleanup_subscriptions(ib, subscribed_index_contracts, option_subscriptions)
        if ib is not None and ib.isConnected():
            ib.disconnect()

    payload = {
        "protocol": "tuesday_protocol101_paper_fill_observation",
        "decision": decide(observations, broker_order_endpoint_called=broker_order_endpoint_called),
        "mode": "paper-fill-observation",
        "paid_data_downloaded": False,
        "real_money_trading": False,
        "live_orders": False,
        "paper_orders_submitted": int(sum(1 for row in observations if row.get("paper_order_submitted"))),
        "broker_order_endpoint_called": bool(broker_order_endpoint_called),
        "paper_order_permission": "user_approved_for_tuesday_observation_only",
        "ibkr_connected": connected_port is not None,
        "ibkr_port": connected_port,
        "connection_attempts": connection_attempts,
        "account_id_redacted": redact_account(account_id),
        "decision_count": int(decision_count),
        "probe_count": int(probe_count),
        "filled_round_trips": int(filled_round_trips),
        "observation_status_counts": status_counts(observations),
        "chain_meta": chain_meta,
        "ibkr_errors": error_log.events[-50:],
        "subscription_errors": error_log.subscription_errors[-50:],
        "trade_log": str(trade_log),
        "observation_log": str(observation_log),
        "next_gate": "Review execution observations and rerun fill readiness; do not promote or train from this packet alone.",
    }
    return payload


def execute_probe(
    *,
    args: argparse.Namespace,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    account_id: str | None,
    trade_log: Path,
    observation_log: Path,
    session: str,
    run_id: str,
    candidate: pd.Series,
    quote: dict[str, Any],
    selection_reason: str,
    spx: float,
    vix: float,
    decision_time: datetime,
    probe_index: int,
    option_subscriptions: list[tuple[Any, Any]],
    artifact_ids: dict[str, Any],
    runtime_flag: dict[str, Any],
    candidate_set_hash: str,
    feature_vector_hash: str,
) -> dict[str, Any]:
    contract_id = str(candidate.get("contract_id"))
    ask = finite(quote.get("ask"), 0.0)
    bid = finite(quote.get("bid"), 0.0)
    intent = PaperOrderIntent(
        action="BUY",
        symbol=str(quote.get("symbol") or "SPX"),
        expiry=str(quote.get("expiry") or decision_time.strftime("%Y%m%d")),
        strike=float(quote.get("strike")),
        right=str(quote.get("right")).upper(),
        quantity=int(args.quantity),
        limit_price=float(ask),
        trading_class=str(quote.get("trading_class") or "SPXW"),
        exchange=str(quote.get("exchange") or "SMART"),
        currency=str(quote.get("currency") or "USD"),
    )
    trade_uid = f"tuesday_fill_probe_{probe_index:03d}_{contract_id}"
    append_event(
        trade_log,
        event_type="model_decision",
        session=session,
        run_id=run_id,
        mode="paper-fill-observation",
        paper_cash=float(args.paper_cash),
        reason=selection_reason,
        selected_contract=selected_contract_from_intent(intent, contract_id=contract_id, quote=quote),
        order=intent_order_payload(intent),
        market_snapshot=market_snapshot(spx=spx, vix=vix, quote=quote),
        model_decision={
            "action": "paper_probe",
            "selected_action": "paper_probe",
            "reason": selection_reason,
            "score": finite(candidate.get("score"), None),
            "selected_margin": finite(candidate.get("score"), None),
            "threshold": finite(candidate.get("threshold"), 0.0),
            "threshold_source": "paper_fill_observation_probe",
            "no_entry_reason": "not_applicable_paper_probe",
            "action_mask": {"paper_probe": True, "candidate_count": 1},
            "raw_logits": [],
            "wait_logit": None,
            "candidate_logits": [],
            "selected_contract": selected_contract_from_intent(intent, contract_id=contract_id, quote=quote),
            "candidate_set_hash": candidate_set_hash,
            "feature_vector_hash": feature_vector_hash,
        },
        trade_uid=trade_uid,
        extra={"artifact_ids": artifact_ids, "runtime_flag": runtime_flag, "candidate_set_hash": candidate_set_hash, "feature_vector_hash": feature_vector_hash},
    )
    guard_quote = {
        "bid": bid,
        "ask": ask,
        "reference_ask": ask,
        "quote_age_ms": quote.get("quote_age_ms"),
        "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
        "quote_timestamp": quote.get("quote_timestamp"),
        "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
        "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
    }
    guard_context = {"context_age_ms": 0, "underlying": {"spx": spx, "vix": vix}}
    dry_run = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=0,
        quote=guard_quote,
        context=guard_context,
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=True,
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=trade_uid,
        artifact_ids=artifact_ids,
        runtime_flag_digest=stable_json_hash(runtime_flag),
        wait_for_fill_seconds=0.0,
        cancel_unfilled=True,
    )
    if dry_run.get("status") != "dry_run_pass":
        observation = observation_row(
            candidate=candidate,
            quote=quote,
            exit_quote=quote,
            entry=dry_run,
            exit_result=None,
            selection_reason=f"{selection_reason}_dry_run_blocked",
            decision_time=decision_time,
            fill_status="dry_run_blocked",
            exit_fill_status="not_attempted",
            latency_ms=0.0,
            post_fill_pnl=0.0,
            open_position_risk=False,
            spx=spx,
            vix=vix,
            trade_uid=trade_uid,
        )
        append_observation(observation_log, observation)
        return observation
    start = time.perf_counter()
    entry = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=0,
        quote=guard_quote,
        context=guard_context,
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=False,
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=trade_uid,
        artifact_ids=artifact_ids,
        runtime_flag_digest=stable_json_hash(runtime_flag),
        wait_for_fill_seconds=float(args.entry_timeout_seconds),
        cancel_unfilled=True,
    )
    entry_latency_ms = (time.perf_counter() - start) * 1000.0
    log_fill_or_status(
        trade_log=trade_log,
        result=entry,
        event_prefix="paper_entry",
        session=session,
        run_id=run_id,
        mode="paper-fill-observation",
        paper_cash=float(args.paper_cash),
        trade_uid=trade_uid,
    )
    entry_fill = entry.get("fill_summary") or {}
    fill_status = observation_fill_status(entry_fill)
    exit_result: dict[str, Any] | None = None
    exit_quote = quote
    exit_fill_status = "not_attempted"
    post_fill_pnl = 0.0
    open_position_risk = False
    if entry_fill.get("filled"):
        ib.sleep(1.0)
        refreshed_quotes = live_option_quotes(option_subscriptions, spx=spx, now=datetime.now(tz=UTC))
        exit_quote = next((row for row in refreshed_quotes if str(row.get("contract_id")) == contract_id), quote)
        exit_result = submit_exit(
            args=args,
            ib=ib,
            option_cls=option_cls,
            order_cls=order_cls,
            account_id=account_id,
            quote=exit_quote,
            entry_intent=intent,
            entry_fill=entry_fill,
            trade_log=trade_log,
            session=session,
            run_id=run_id,
            trade_uid=trade_uid,
            force_offset=0.0,
        )
        exit_fill = exit_result.get("fill_summary") or {}
        exit_fill_status = observation_fill_status(exit_fill)
        if not exit_fill.get("filled"):
            ib.sleep(0.5)
            refreshed_quotes = live_option_quotes(option_subscriptions, spx=spx, now=datetime.now(tz=UTC))
            exit_quote = next((row for row in refreshed_quotes if str(row.get("contract_id")) == contract_id), exit_quote)
            exit_result = submit_exit(
                args=args,
                ib=ib,
                option_cls=option_cls,
                order_cls=order_cls,
                account_id=account_id,
                quote=exit_quote,
                entry_intent=intent,
                entry_fill=entry_fill,
                trade_log=trade_log,
                session=session,
                run_id=run_id,
                trade_uid=f"{trade_uid}_force_exit",
                force_offset=float(args.exit_force_offset),
            )
            exit_fill = exit_result.get("fill_summary") or {}
            exit_fill_status = observation_fill_status(exit_fill)
        entry_price = finite(entry_fill.get("avg_fill_price"), ask)
        exit_price = finite(exit_fill.get("avg_fill_price"), finite(exit_quote.get("bid"), bid))
        post_fill_pnl = (exit_price - entry_price) * 100.0 * int(args.quantity)
        open_position_risk = not bool(exit_fill.get("filled"))
    observation = observation_row(
        candidate=candidate,
        quote=quote,
        exit_quote=exit_quote,
        entry=entry,
        exit_result=exit_result,
        selection_reason=selection_reason,
        decision_time=decision_time,
        fill_status=fill_status,
        exit_fill_status=exit_fill_status,
        latency_ms=entry_latency_ms,
        post_fill_pnl=post_fill_pnl,
        open_position_risk=open_position_risk,
        spx=spx,
        vix=vix,
        trade_uid=trade_uid,
    )
    append_observation(observation_log, observation)
    return observation


def submit_exit(
    *,
    args: argparse.Namespace,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    account_id: str | None,
    quote: dict[str, Any],
    entry_intent: PaperOrderIntent,
    entry_fill: dict[str, Any],
    trade_log: Path,
    session: str,
    run_id: str,
    trade_uid: str,
    force_offset: float,
) -> dict[str, Any]:
    bid = finite(quote.get("bid"), 0.0)
    ask = finite(quote.get("ask"), max(bid, 0.01))
    qty = max(1, min(int(entry_fill.get("filled_quantity") or entry_intent.quantity), int(args.quantity)))
    limit_price = max(float(bid) - max(0.0, float(force_offset)), 0.01)
    intent = PaperOrderIntent(
        action="SELL",
        symbol=entry_intent.symbol,
        expiry=entry_intent.expiry,
        strike=entry_intent.strike,
        right=entry_intent.right,
        quantity=qty,
        limit_price=limit_price,
        trading_class=entry_intent.trading_class,
        exchange=entry_intent.exchange,
        currency=entry_intent.currency,
    )
    exit_artifact_ids = {"protocol": "tuesday_protocol101_paper_fill_observation"}
    dry_run = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=1,
        quote={
            "bid": bid,
            "ask": ask,
            "reference_ask": ask,
            "quote_age_ms": quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
            "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
        },
        context={"context_age_ms": 0},
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=True,
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=trade_uid,
        artifact_ids=exit_artifact_ids,
        runtime_flag_digest=stable_json_hash(local_observation_approval(args)),
        wait_for_fill_seconds=0.0,
        cancel_unfilled=True,
    )
    if dry_run.get("status") != "dry_run_pass":
        log_fill_or_status(
            trade_log=trade_log,
            result=dry_run,
            event_prefix="paper_exit",
            session=session,
            run_id=run_id,
            mode="paper-fill-observation",
            paper_cash=float(args.paper_cash),
            trade_uid=trade_uid,
        )
        return dry_run
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=1,
        quote={
            "bid": bid,
            "ask": ask,
            "reference_ask": ask,
            "quote_age_ms": quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
            "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
        },
        context={"context_age_ms": 0},
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=False,
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=trade_uid,
        artifact_ids=exit_artifact_ids,
        runtime_flag_digest=stable_json_hash(local_observation_approval(args)),
        wait_for_fill_seconds=float(args.exit_timeout_seconds),
        cancel_unfilled=True,
    )
    log_fill_or_status(
        trade_log=trade_log,
        result=result,
        event_prefix="paper_exit",
        session=session,
        run_id=run_id,
        mode="paper-fill-observation",
        paper_cash=float(args.paper_cash),
        trade_uid=trade_uid,
    )
    return result


def choose_observation_candidate(
    *,
    candidates: pd.DataFrame,
    prediction: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    observed_contract_counts: dict[str, int],
    paper_cash: float,
    max_observations_per_contract: int,
) -> tuple[pd.Series, dict[str, Any], str] | None:
    if candidates.empty:
        return None
    selected_id = str((prediction.get("selected") or {}).get("contract_id") or "")
    ordered: list[tuple[int, pd.Series, str]] = []
    for idx, row in candidates.iterrows():
        reason = "protocol101_selected" if str(row.get("contract_id")) == selected_id and prediction.get("action") == "enter" else "near_selected_top_protocol051_edge"
        priority = 0 if reason == "protocol101_selected" else 1
        ordered.append((priority, row, reason))
    ordered.sort(key=lambda item: (item[0], -finite(item[1].get("score"), -1e9), str(item[1].get("contract_id"))))
    for _priority, row, reason in ordered:
        contract_id = str(row.get("contract_id") or "")
        quote = lookup.get(contract_id)
        if not quote:
            continue
        if observed_contract_counts.get(contract_id, 0) >= int(max_observations_per_contract):
            continue
        ask = finite(quote.get("ask"), math.nan)
        bid = finite(quote.get("bid"), math.nan)
        quote_age = finite(quote.get("quote_age_ms"), math.nan)
        if not math.isfinite(ask) or not math.isfinite(bid) or ask <= 0 or bid <= 0 or ask < bid:
            continue
        if math.isfinite(quote_age) and quote_age > 1500.0:
            continue
        if ask * 100.0 > float(paper_cash):
            continue
        return row, quote, reason
    return None


def observation_row(
    *,
    candidate: pd.Series,
    quote: dict[str, Any],
    exit_quote: dict[str, Any],
    entry: dict[str, Any],
    exit_result: dict[str, Any] | None,
    selection_reason: str,
    decision_time: datetime,
    fill_status: str,
    exit_fill_status: str,
    latency_ms: float,
    post_fill_pnl: float,
    open_position_risk: bool,
    spx: float,
    vix: float,
    trade_uid: str,
) -> dict[str, Any]:
    entry_fill = entry.get("fill_summary") or {}
    exit_fill = (exit_result or {}).get("fill_summary") or {}
    ask = finite(quote.get("ask"), 0.0)
    bid = finite(quote.get("bid"), 0.0)
    spread = finite(quote.get("spread"), ask - bid)
    return clean_json(
        {
            "event_type": "paper_execution_observation",
            "trade_uid": trade_uid,
            "decision_timestamp": decision_time.isoformat(),
            "raw_quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp": quote.get("received_timestamp"),
            "quote_age_ms": quote.get("quote_age_ms"),
            "bid": bid,
            "ask": ask,
            "bid_size": quote.get("bid_size"),
            "ask_size": quote.get("ask_size"),
            "spread": spread,
            "premium": ask * 100.0,
            "side": str(quote.get("right") or candidate.get("right")),
            "moneyness": finite(candidate.get("offset_points"), finite(quote.get("distance_points"), 0.0)),
            "time_bucket": time_bucket(decision_time),
            "intended_ask_entry": ask,
            "submitted_limit": finite((entry.get("intent") or {}).get("limit_price"), ask),
            "fill_status": fill_status,
            "cancel_status": "cancel_requested" if entry_fill.get("cancel_requested") else "not_cancelled",
            "timeout_status": "timeout" if fill_status in {"timeout", "cancelled"} else "not_timeout",
            "latency_ms": float(latency_ms),
            "exit_bid": finite(exit_quote.get("bid"), bid),
            "post_fill_pnl": float(post_fill_pnl),
            "entry_fill_price": entry_fill.get("avg_fill_price"),
            "exit_fill_price": exit_fill.get("avg_fill_price"),
            "exit_fill_status": exit_fill_status,
            "entry_order_status": entry_fill.get("status") or entry.get("status"),
            "exit_order_status": exit_fill.get("status") if exit_result else None,
            "paper_order_submitted": bool(entry.get("paper_order_submitted")),
            "broker_order_endpoint_called": bool(entry.get("broker_order_endpoint_called")) or bool((exit_result or {}).get("broker_order_endpoint_called")),
            "selection_reason": selection_reason,
            "contract_id": str(candidate.get("contract_id")),
            "strike": quote.get("strike"),
            "expiry": quote.get("expiry"),
            "spx": spx,
            "vix": vix,
            "open_position_risk": bool(open_position_risk),
        }
    )


def observation_fill_status(fill: dict[str, Any]) -> str:
    if fill.get("filled"):
        return "filled"
    if fill.get("cancel_requested"):
        return "cancelled"
    status = str(fill.get("status") or "").lower()
    if "cancel" in status:
        return "cancelled"
    if status in {"inactive", "apicancelled"}:
        return "cancelled"
    return "timeout"


def selected_contract_from_intent(intent: PaperOrderIntent, *, contract_id: str, quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": contract_id,
        "symbol": intent.symbol,
        "root": "SPXW",
        "trading_class": intent.trading_class,
        "settlement": str(quote.get("settlement") or "PM"),
        "expiry": intent.expiry,
        "strike": intent.strike,
        "right": intent.right,
        "exchange": intent.exchange,
        "currency": intent.currency,
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "bid_size": quote.get("bid_size"),
        "ask_size": quote.get("ask_size"),
        "quote_age_ms": quote.get("quote_age_ms"),
        "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
        "quote_timestamp": quote.get("quote_timestamp"),
        "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
        "received_timestamp": quote.get("received_timestamp"),
    }


def market_snapshot(*, spx: float, vix: float, quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "underlying": {"spx": float(spx), "vix": float(vix)},
        "option_nbbo": {
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "bid_size": quote.get("bid_size"),
            "ask_size": quote.get("ask_size"),
            "quote_age_ms": quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
            "received_timestamp": quote.get("received_timestamp"),
        },
        "context": {"source": "ibkr_paper_observation"},
    }


def local_observation_approval(args: argparse.Namespace, environ: dict[str, str] | None = None) -> dict[str, Any]:
    import os

    env = os.environ if environ is None else environ
    reasons: list[str] = []
    if not args.enable_paper_orders:
        reasons.append("enable_paper_orders_flag_missing")
    if not args.acknowledge_paper_loss:
        reasons.append("acknowledge_paper_loss_flag_missing")
    if str(env.get("V4_ALLOW_IBKR_PAPER_ORDERS", "")).upper() != "YES":
        reasons.append("paper_order_env_not_set")
    if str(env.get(APPROVAL_ENV, "")).upper() != "YES":
        reasons.append("paper_fill_observation_approval_env_not_set")
    return {"passed": not reasons, "reasons": reasons, "required_env": [f"{APPROVAL_ENV}=YES", "V4_ALLOW_IBKR_PAPER_ORDERS=YES"]}


def append_observation(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def keep_running(args: argparse.Namespace, *, close_deadline: datetime) -> bool:
    if args.skip_market_clock:
        return True
    return datetime.now(tz=NY) <= close_deadline


def session_deadline(now: datetime, wall: str) -> datetime:
    hour, minute = [int(part) for part in str(wall).split(":", 1)]
    local = now.astimezone(NY)
    return local.replace(hour=hour, minute=minute, second=0, microsecond=0)


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("fill_status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def decide(observations: list[dict[str, Any]], *, broker_order_endpoint_called: bool) -> str:
    if not observations:
        return "blocked_no_paper_fill_observations_collected"
    if any(row.get("open_position_risk") for row in observations):
        return "blocked_paper_fill_observation_left_open_position_risk"
    if broker_order_endpoint_called:
        return "paper_fill_observations_collected_for_execution_truth_review"
    return "blocked_no_broker_order_endpoint_called"


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
        mode="paper-fill-observation",
        paper_cash=float(args.paper_cash),
        reason=reason,
        extra={"detail": detail, **(extra or {})},
    )
    return {
        "protocol": "tuesday_protocol101_paper_fill_observation",
        "decision": f"blocked_{reason}",
        "mode": "paper-fill-observation",
        "paid_data_downloaded": False,
        "real_money_trading": False,
        "live_orders": False,
        "paper_orders_submitted": 0,
        "broker_order_endpoint_called": False,
        "decision_count": 0,
        "probe_count": 0,
        "filled_round_trips": 0,
        "blocked_reason": reason,
        "detail": detail,
        "trade_log": str(trade_log),
        **(extra or {}),
    }


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
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md"), "trade_log": str(trade_log)}, indent=2, sort_keys=True))
    return 0 if validation["status"] == "pass" and "exception" not in str(payload["decision"]) else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Tuesday Protocol101 Paper Fill Observation",
        "",
        "This is a bounded IBKR paper-only execution observation packet. It does not change the operational default.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Paper orders submitted: `{payload.get('paper_orders_submitted', 0)}`",
        f"- Broker order endpoint called: `{payload.get('broker_order_endpoint_called', False)}`",
        f"- Probe count: `{payload.get('probe_count', 0)}`",
        f"- Filled round trips: `{payload.get('filled_round_trips', 0)}`",
        f"- Observation log: `{payload.get('observation_log')}`",
        f"- Trade log: `{payload.get('trade_log')}`",
        "",
        "## Next Gate",
        "",
        str(payload.get("next_gate", "Review the paper fill observations before training or promotion work.")),
    ]
    if payload.get("blocked_reason"):
        lines.extend(["", "## Blocker", "", f"- `{payload['blocked_reason']}`", f"- {payload.get('detail')}"])
    path.write_text("\n".join(lines) + "\n")


def finite(value: Any, default: Any = math.nan) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
