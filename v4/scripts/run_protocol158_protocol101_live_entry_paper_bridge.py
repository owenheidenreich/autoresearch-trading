"""Protocol 158: live Protocol101 entry-to-paper-order bridge.

This is the missing bridge between live SPX/VIX/SPXW quotes and guarded IBKR
paper order intents. Default mode emits decisions and risk gates only; broker
order submission remains disabled unless paper-submit mode, the runtime flag,
the environment variable, and explicit acknowledgement flags all pass.
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
from v4.live.ibkr_paper_guard import PaperOrderIntent, validate_order_intent
from v4.live.paper_trade_log import (
    DEFAULT_TRADE_LOG_ROOT,
    append_trade_event,
    export_trade_log_csv,
    make_trade_log_event,
    retry_io,
    stable_json_hash,
    trade_log_path,
    validate_observability_contract,
    validate_trade_log,
    load_trade_log,
)
from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol066_inference import load_protocol066_artifact, predict_protocol066_sequence, prediction_for_step
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_gate_diagnostics,
    protocol101_candidate_frame_from_surface,
)
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION
from v4.live.protocol101_live_entry import (
    LiveIndexState,
    build_live_surface_row,
    live_surface_decision,
    order_intent_from_prediction,
    scalar_feature_payload,
    scored_token_universe_payload,
    selected_contract_payload,
)
from v4.model.hypothesis_protocol import SurfaceVariant, registered_aplus_surface_variants, time_bucket
from v4.scripts.run_protocol081_live_shadow_router import (
    DEFAULT_PROTOCOL081_MANIFEST,
    _compute_live_greeks,
    _connect_ibkr,
    _contract_id,
    _discover_spxw_0dte_contracts,
    _is_regular_market_hours,
    _market_data_type_name,
    _option_quote,
    _live_feature_row,
    _request_index_ticker,
    _round_to_5,
    _ticker_market_data_type,
    _ticker_price,
    _wait_for_price,
    IbkrErrorLog,
)
from v4.scripts.run_protocol119_protocol101_live_readiness import DEFAULT_PROTOCOL101_MANIFEST, DEFAULT_PROTOCOL101_SUMMARY
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST
from v4.scripts.run_protocol150_protocol101_paper_order_enablement_gate import DEFAULT_RUNTIME_FLAG


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_158_protocol101_live_entry_paper_bridge")
DEFAULT_PRE_LIVE_SANITY_GATE_SUMMARY = Path("v4/audit/autoresearch/protocol101_pre_live_historical_sanity_gate/summary.json")
QUOTE_TIME_FIELDS = ("time", "rtTime", "timestamp", "quote_time", "quoteTimestamp")
ALLOWED_ENTRY_TIME_BUCKETS = ("post_open_morning", "late_afternoon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("intent-shadow", "paper-dry-run", "paper-submit"), default="intent-shadow")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--runtime-flag", type=Path, default=DEFAULT_RUNTIME_FLAG)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=158)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--open-positions", type=int, default=0)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--order-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--runtime-state", type=Path, default=Path("v4/runtime/protocol101_live_paper_state.json"))
    parser.add_argument("--live-index-context-log", type=Path, default=Path("v4/runtime/protocol101_live_index_context.jsonl"))
    parser.add_argument("--pre-live-sanity-gate-summary", type=Path, default=DEFAULT_PRE_LIVE_SANITY_GATE_SUMMARY)
    parser.add_argument("--skip-pre-live-sanity-gate", action="store_true")
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--live-strikes-around-atm", type=int, default=10)
    parser.add_argument("--live-capture-seconds", type=float, default=45.0)
    parser.add_argument("--live-sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--min-live-context-minutes", type=float, default=30.0)
    parser.add_argument("--max-decisions", type=int, default=20)
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--skip-market-clock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(tz=NY)
    session = args.session_date or now.date().isoformat()
    run_id = args.run_id or f"protocol101_{args.mode}_{session}"
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log = trade_log_path(root=args.trade_log_root, session=session, run_id=run_id)
    csv_log = trade_log.with_suffix(".csv")

    if not args.skip_market_clock and not _is_regular_market_hours(now):
        payload = blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="outside_regular_market_hours")
        return finish(out_dir, payload, trade_log, csv_log)

    try:
        payload = run_live_bridge(args, session=session, run_id=run_id, trade_log=trade_log)
    except Exception as exc:
        payload = blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="protocol158_exception", detail=str(exc))
    return finish(out_dir, payload, trade_log, csv_log)


def run_live_bridge(args: argparse.Namespace, *, session: str, run_id: str, trade_log: Path) -> dict[str, Any]:
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
    context_rows_loaded = load_live_index_context(args.live_index_context_log, index_state, session=session, include_prior_session_close=True)
    error_log = IbkrErrorLog()
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
    ib = None
    subscribed: list[Any] = []
    decisions: list[dict[str, Any]] = []
    executor_results: list[dict[str, Any]] = []
    paper_submit_blocked = 0
    broker_rows_before = 0

    append_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="protocol158_started",
        extra={"runtime_flag": runtime_flag_summary(runtime_flag), "artifact_ids": artifact_ids, "live_orders_enabled": args.mode == "paper-submit"},
    )

    try:
        ib, port, attempts = _connect_ibkr(args, IB)
        if ib is None:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason="ibkr_connection_failed",
                detail="No configured IBKR API port accepted a connection.",
                extra={"connection_attempts": attempts},
            )
        ib.errorEvent += error_log.handler
        account_id = args.account_id or first_managed_account(ib)
        ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)
        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
        subscribed.extend([spx_contract, vix_contract])
        spx_price = _wait_for_price(ib, spx_ticker, seconds=8.0)
        vix_value = _wait_for_price(ib, vix_ticker, seconds=6.0)
        if spx_price is None or vix_value is None:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason="missing_live_spx_or_vix",
                detail="SPX and VIX must both produce positive live values before entry decisions.",
                extra={"connection_attempts": attempts, "ibkr_port": port, "ibkr_errors": error_log.events[-20:]},
            )
        contracts, chain_meta = _discover_spxw_0dte_contracts(
            ib=ib,
            option_cls=Option,
            spx_contract=spx_contract,
            spx_price=float(spx_price),
            now=datetime.now(tz=NY),
            strikes_around_atm=int(args.live_strikes_around_atm),
        )
        if not contracts:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason=chain_meta.get("blocked_reason", "no_spxw_0dte_contracts"),
                detail="No SPXW 0DTE contracts could be qualified.",
                extra={"chain_meta": chain_meta, "ibkr_port": port},
            )
        option_tickers = []
        for contract in contracts:
            ticker = ib.reqMktData(contract, "", False, False)
            option_tickers.append((contract, ticker))
            subscribed.append(contract)

        deadline = time.monotonic() + max(float(args.live_capture_seconds), float(args.live_sample_interval_seconds))
        while time.monotonic() < deadline and len(decisions) < int(args.max_decisions):
            ib.sleep(max(0.5, float(args.live_sample_interval_seconds)))
            decision_time = datetime.now(tz=UTC)
            current_spx = _ticker_price(spx_ticker)
            current_vix = _ticker_price(vix_ticker)
            if current_spx is None or current_vix is None:
                append_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="missing_live_context_snapshot",
                )
                continue
            live_positions = spxw_open_positions(ib)
            if live_positions:
                position_result = handle_open_position(
                    args=args,
                    ib=ib,
                    option_cls=Option,
                    order_cls=LimitOrder,
                    lifecycle=lifecycle,
                    position=live_positions[0],
                    spx=float(current_spx),
                    vix=float(current_vix),
                    now=decision_time,
                    account_id=account_id,
                    trade_log=trade_log,
                    session=session,
                    run_id=run_id,
                )
                decisions.append(position_result["decision_record"])
                executor_results.extend(position_result.get("executor_results", []))
                if position_result.get("broker_order_endpoint_called"):
                    break
                continue
            option_quotes = live_option_quotes(option_tickers, spx=float(current_spx), now=decision_time)
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
                    observed_at=decision_time,
                    option_quotes=option_quotes,
                    context_summary=None,
                ),
            )
            if not option_quotes:
                append_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="no_valid_spxw_nbbo_quotes",
                )
                continue

            normalized_row, lookup = build_live_surface_row(
                decision_time=decision_time,
                spx=float(current_spx),
                vix=float(current_vix),
                option_quotes=option_quotes,
                index_state=index_state,
            )
            append_live_index_context(
                args.live_index_context_log,
                session=session,
                timestamp=decision_time,
                spx=float(current_spx),
                vix=float(current_vix),
            )
            context_summary = index_state.session_context_summary(decision_time)
            decision_market = market_snapshot_payload(
                spx=float(current_spx),
                vix=float(current_vix),
                spx_ticker=spx_ticker,
                vix_ticker=vix_ticker,
                observed_at=decision_time,
                option_quotes=option_quotes,
                context_summary=context_summary,
            )
            bucket = time_bucket(decision_time)
            if bucket not in set(ALLOWED_ENTRY_TIME_BUCKETS):
                decision_record = {
                    "timestamp": decision_time.isoformat(),
                    "action": "wait",
                    "reason": "outside_time_bucket",
                    "candidate_count": 0,
                    "selected_contract": {},
                    "order_intent": None,
                    "validation": {"passed": True, "reason": "outside_time_bucket"},
                }
                decisions.append(decision_record)
                append_time_bucket_block(
                    args,
                    trade_log,
                    session,
                    run_id,
                    protocol101.threshold,
                    decision_time,
                    context_summary,
                    decision_market=decision_market,
                    option_quotes=option_quotes,
                    runtime_flag=runtime_flag,
                    artifact_ids=artifact_ids,
                )
                continue
            if not live_context_ready(context_summary, min_context_minutes=float(args.min_live_context_minutes)):
                trace_extra = protocol101_blocked_decision_trace_extra(
                    reason="insufficient_live_index_context",
                    decision_time=decision_time,
                    option_quotes=option_quotes,
                    context_summary=context_summary,
                    model_threshold=protocol101.threshold,
                )
                decision_record = {
                    "timestamp": decision_time.isoformat(),
                    "action": "wait",
                    "reason": "insufficient_live_index_context",
                    "candidate_count": 0,
                    "selected_contract": {},
                    "order_intent": None,
                    "validation": {"passed": True, "reason": "insufficient_live_index_context"},
                }
                decisions.append(decision_record)
                diagnostics = {
                    "filter_reason": "insufficient_live_index_context",
                    "required_minutes": float(args.min_live_context_minutes),
                    "minute_row_count": context_summary.get("minute_row_count"),
                    "span_minutes": context_summary.get("span_minutes"),
                }
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
                        "candidate_set_hash": stable_json_hash({"reason": "insufficient_live_index_context", "option_quotes": option_quotes}),
                        "feature_vector_hash": stable_json_hash([]),
                        **trace_extra,
                        "runtime_flag": runtime_flag_summary(runtime_flag),
                        "artifact_ids": artifact_ids,
                        "candidate_count": 0,
                        "candidate_sample": [],
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
                    reason="insufficient_live_index_context",
                    market_snapshot=decision_market,
                    model_decision={
                        "action": "wait",
                        "selected_action": "wait",
                        "score": None,
                        "selected_margin": None,
                        "threshold": protocol101.threshold,
                        "reason": "insufficient_live_index_context",
                        "no_entry_reason": "insufficient_live_index_context",
                        "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
                        "raw_logits": [],
                        "wait_logit": None,
                        "candidate_logits": [],
                    },
                    extra={
                        "candidate_set_hash": stable_json_hash({"reason": "insufficient_live_index_context", "option_quotes": option_quotes}),
                        "feature_vector_hash": stable_json_hash([]),
                        **trace_extra,
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
                    reason="insufficient_live_index_context",
                    market_snapshot=decision_market,
                    risk_gate=risk_gate_payload({"passed": True, "reason": "insufficient_live_index_context", "reasons": []}),
                    extra={"runtime_flag": runtime_flag_summary(runtime_flag), "artifact_ids": artifact_ids, **trace_extra},
                )
                append_event(
                    trade_log,
                    event_type="paper_account_state",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason="paper_account_state",
                    extra={"account_id_redacted": redact_account(account_id)},
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
            candidate_gate_diagnostics = protocol101_candidate_gate_diagnostics(
                surface_decision,
                surface_scores,
                min_edge=float(args.min_edge),
                max_rows=1000,
            )
            candidates = protocol101_candidate_frame_from_surface(
                surface_decision,
                surface_scores,
                history,
                min_edge=float(args.min_edge),
            )
            prediction = predict_protocol101_entry(protocol101, candidates)
            if not candidates.empty:
                history.update(candidates, pd.Timestamp(decision_time))
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
            decision_market = market_snapshot_payload(
                spx=float(current_spx),
                vix=float(current_vix),
                spx_ticker=spx_ticker,
                vix_ticker=vix_ticker,
                observed_at=decision_time,
                option_quotes=option_quotes,
                selected_contract=selected_contract,
                context_summary=context_summary,
            )
            decision_record = {
                "timestamp": decision_time.isoformat(),
                "action": model_action,
                "reason": prediction.get("reason"),
                "margin": prediction.get("margin"),
                "threshold": prediction.get("threshold"),
                "candidate_count": int(len(candidates)),
                "selected_contract": selected_contract,
                "order_intent": intent.__dict__ if intent else None,
                "validation": validation,
            }
            decisions.append(decision_record)
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
                    "candidate_gate_diagnostics": clean_json(candidate_gate_diagnostics),
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
                extra={"account_id_redacted": redact_account(account_id)},
            )
            if intent is None:
                continue
            if args.mode == "intent-shadow":
                continue
            if args.mode == "paper-submit" and not bool(runtime_flag.get("paper_orders_enabled")):
                paper_submit_blocked += 1
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
                continue
            selected_quote = lookup[selected_contract["contract_id"]]
            result = execute_guarded_paper_order(
                ib=ib,
                option_cls=Option,
                order_cls=LimitOrder,
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
                trade_uid=f"protocol158_{len(decisions):04d}",
                artifact_ids=artifact_ids,
                runtime_flag_digest=stable_json_hash(runtime_flag_summary(runtime_flag)),
                wait_for_fill_seconds=float(args.order_timeout_seconds),
                cancel_unfilled=True,
            )
            executor_results.append(result)
            log_fill_or_status(
                trade_log=trade_log,
                result=result,
                event_prefix="paper_entry",
                session=session,
                run_id=run_id,
                mode=args.mode,
                paper_cash=args.paper_cash,
                trade_uid=f"protocol158_{len(decisions):04d}",
            )
            if result.get("fill_summary", {}).get("filled"):
                write_runtime_state(args.runtime_state, state_from_entry_result(result, fallback_intent=intent))
            broker_rows_before += int(bool(result.get("broker_order_endpoint_called")))
            if args.mode == "paper-submit" and result.get("paper_order_submitted"):
                break

        return {
            "protocol": "158_protocol101_live_entry_paper_bridge",
            "decision": decide(args=args, decisions=decisions, executor_results=executor_results, paper_submit_blocked=paper_submit_blocked),
            "mode": args.mode,
            "paid_data_downloaded": False,
            "real_money_trading": False,
            "live_orders": False,
            "paper_orders_submitted": sum(1 for row in executor_results if row.get("paper_order_submitted")),
            "broker_order_endpoint_called": any(row.get("broker_order_endpoint_called") for row in executor_results),
            "ibkr_connected": True,
            "ibkr_port": port,
            "connection_attempts": attempts,
            "account_id_redacted": redact_account(account_id),
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "live_index_context": {
                "path": str(args.live_index_context_log),
                "rows_loaded": int(context_rows_loaded),
                "session_summary": clean_json(index_state.session_context_summary(datetime.now(tz=UTC))),
            },
            "chain_meta": chain_meta,
            "decision_count": len(decisions),
            "enter_intents": sum(1 for row in decisions if row["action"] == "enter"),
            "executor_results": summarize_executor_results(executor_results),
            "paper_submit_blocked": paper_submit_blocked,
            "ibkr_errors": error_log.events[-30:],
            "trade_log": str(trade_log),
            "next_gate": next_gate(args.mode),
        }
    finally:
        if ib is not None and ib.isConnected():
            for contract in subscribed:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
            ib.disconnect()


def _coerce_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        numeric = float(value)
        abs_numeric = abs(numeric)
        if abs_numeric > 1e14:
            ts = pd.Timestamp(numeric, unit="ns", tz="UTC")
        elif abs_numeric > 1e11:
            ts = pd.Timestamp(numeric, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(numeric, unit="s", tz="UTC")
    elif isinstance(value, str) and value.strip():
        ts = pd.Timestamp(value.strip())
    else:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _epoch_ms(ts: datetime) -> int:
    return int(round(pd.Timestamp(ts).timestamp() * 1000))


def quote_freshness_from_ticker(ticker: Any, *, observed_at: datetime) -> dict[str, Any]:
    observed = _coerce_utc_datetime(observed_at) or datetime.now(tz=UTC)
    raw_timestamp = None
    raw_field = None
    for field in QUOTE_TIME_FIELDS:
        raw_timestamp = getattr(ticker, field, None)
        if raw_timestamp is not None:
            raw_field = field
            break
    quote_ts = _coerce_utc_datetime(raw_timestamp)
    payload: dict[str, Any] = {
        "received_timestamp": observed.isoformat(),
        "received_timestamp_ms": _epoch_ms(observed),
        "decision_timestamp": observed.isoformat(),
        "decision_timestamp_ms": _epoch_ms(observed),
    }
    if quote_ts is None:
        payload.update(
            {
                "quote_timestamp": None,
                "quote_timestamp_ms": None,
                "quote_age_ms": None,
                "quote_age_source": "missing_ticker_time",
            }
        )
        return payload
    age_ms = int(round((observed - quote_ts).total_seconds() * 1000))
    payload.update(
        {
            "quote_timestamp": quote_ts.isoformat(),
            "quote_timestamp_ms": _epoch_ms(quote_ts),
            "quote_age_ms": max(age_ms, 0) if age_ms >= -1000 else None,
            "quote_age_source": raw_field if age_ms >= -1000 else "quote_timestamp_after_decision",
        }
    )
    return payload


def quote_freshness_summary(option_quotes: list[dict[str, Any]]) -> dict[str, Any]:
    ages = [float(row["quote_age_ms"]) for row in option_quotes if row.get("quote_age_ms") is not None]
    missing = sum(1 for row in option_quotes if row.get("quote_age_ms") is None)
    if not ages:
        return {"count": len(option_quotes), "known_count": 0, "missing_count": missing}
    return {
        "count": len(option_quotes),
        "known_count": len(ages),
        "missing_count": missing,
        "min_ms": min(ages),
        "median_ms": float(pd.Series(ages).median()),
        "max_ms": max(ages),
    }


def index_freshness_payload(ticker: Any, *, observed_at: datetime, prefix: str) -> dict[str, Any]:
    freshness = quote_freshness_from_ticker(ticker, observed_at=observed_at)
    return {
        f"{prefix}_raw_quote_timestamp_utc": freshness.get("quote_timestamp"),
        f"{prefix}_quote_timestamp_ms": freshness.get("quote_timestamp_ms"),
        f"{prefix}_quote_age_ms": freshness.get("quote_age_ms"),
        f"{prefix}_quote_age_source": freshness.get("quote_age_source"),
        f"{prefix}_received_timestamp_utc": freshness.get("received_timestamp"),
        f"{prefix}_received_timestamp_ms": freshness.get("received_timestamp_ms"),
    }


def market_snapshot_payload(
    *,
    spx: float,
    vix: float,
    spx_ticker: Any | None,
    vix_ticker: Any | None,
    observed_at: datetime,
    option_quotes: list[dict[str, Any]] | None = None,
    selected_contract: dict[str, Any] | None = None,
    context_summary: dict[str, Any] | None = None,
    context_age_ms: int | float | None = 0,
) -> dict[str, Any]:
    underlying = {
        "spx": float(spx),
        "vix": float(vix),
        "spx_market_data_type": _market_data_type_name(_ticker_market_data_type(spx_ticker)) if spx_ticker is not None else None,
        "vix_market_data_type": _market_data_type_name(_ticker_market_data_type(vix_ticker)) if vix_ticker is not None else None,
    }
    if spx_ticker is not None:
        underlying.update(index_freshness_payload(spx_ticker, observed_at=observed_at, prefix="spx"))
    if vix_ticker is not None:
        underlying.update(index_freshness_payload(vix_ticker, observed_at=observed_at, prefix="vix"))
    quotes = option_quotes or []
    option_nbbo = dict(selected_contract or {})
    if not option_nbbo:
        option_nbbo = {
            "quote_count": len(quotes),
            "freshness": quote_freshness_summary(quotes),
            "option_quotes_digest": stable_json_hash(quotes),
        }
    option_nbbo.setdefault("quote_count", len(quotes))
    context_summary_obj = context_summary if isinstance(context_summary, dict) else {}
    context = {
        "source": "ibkr_live",
        "context_age_ms": context_age_ms,
        "context_ready": bool(context_summary_obj.get("context_ready", bool(context_summary))),
    }
    if context_summary:
        context.update(
            {
                "context_rows": context_summary.get("row_count"),
                "context_minute_rows": context_summary.get("minute_row_count"),
                "context_start_timestamp_utc": context_summary.get("first_timestamp"),
                "context_last_timestamp_utc": context_summary.get("last_timestamp"),
                "context_span_minutes": context_summary.get("span_minutes"),
                "expected_first_timestamp_utc": context_summary.get("expected_first_timestamp"),
                "opening_context_ready": context_summary.get("opening_context_ready"),
                "missing_opening_minutes": context_summary.get("missing_opening_minutes"),
            }
        )
    return {"underlying": underlying, "option_nbbo": option_nbbo, "context": context}


def candidate_contracts_payload(candidates: Any, lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if getattr(candidates, "empty", True):
        return []
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        contract_id = str(row.get("contract_id") or "")
        quote = dict(lookup.get(contract_id) or {})
        rows.append(
            clean_json(
                {
                    "contract_id": contract_id,
                    "right": row.get("right"),
                    "offset_points": row.get("offset_points"),
                    "edge": row.get("score") if "score" in row else row.get("edge"),
                    "surface_action_score": row.get("surface_action_score"),
                    "surface_flat_score": row.get("surface_flat_score"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "bid_size": quote.get("bid_size"),
                    "ask_size": quote.get("ask_size"),
                    "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
                    "quote_age_ms": quote.get("quote_age_ms"),
                    "quote_age_source": quote.get("quote_age_source"),
                }
            )
        )
    return rows


def protocol101_decision_trace_extra(
    *,
    surface_decision: Any,
    surface_scores: Any,
    lookup: dict[str, dict[str, Any]],
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full model-facing parity trace fields for live-vs-historical diffs."""

    scored_tokens = scored_token_universe_payload(surface_decision, surface_scores, lookup)
    scalar_payload = scalar_feature_payload(surface_decision)
    token_feature_payload = [
        {
            "contract_id": row.get("contract_id"),
            "token_idx": row.get("token_idx"),
            "token_feature_hash": row.get("token_feature_hash"),
            "token_features": row.get("token_features"),
        }
        for row in scored_tokens
    ]
    features = {
        **scalar_payload,
        "token_features": token_feature_payload,
    }
    def score_value(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    surface_score_list = [score_value(value) for value in list(surface_scores)]
    model_scores = {
        "surface_scores": surface_score_list,
        "protocol101_raw_logits": [] if prediction is None else prediction.get("raw_logits", []),
        "protocol101_wait_logit": None if prediction is None else prediction.get("wait_logit"),
        "protocol101_candidate_logits": [] if prediction is None else prediction.get("candidate_logits", []),
    }
    feature_contract_versions = sorted(
        {
            str(row.get("feature_contract_version"))
            for row in scored_tokens
            if row.get("feature_contract_version")
        }
    )
    source_quote_ts = max_iso_timestamp(row.get("source_quote_time") for row in scored_tokens)
    source_context_ts = max_iso_timestamp(row.get("source_context_time") for row in scored_tokens)
    quote_ages = [age for age in (finite_float(row.get("quote_age_ms")) for row in scored_tokens) if age is not None]
    return clean_json(
        {
            "decision_trace_schema_version": "Protocol101DecisionTraceV1",
            "feature_contract_version": feature_contract_versions[0] if len(feature_contract_versions) == 1 else None,
            "source_quote_ts": source_quote_ts,
            "source_context_ts": source_context_ts,
            "quote_freshness_ms": max(quote_ages) if quote_ages else None,
            "candidate_universe": scored_tokens,
            "candidate_universe_hash": stable_json_hash(scored_tokens),
            "features": features,
            "feature_hash": stable_json_hash(features),
            "full_feature_hash": stable_json_hash(features),
            "model_scores": model_scores,
            "score_hash": stable_json_hash(model_scores),
        }
    )


def protocol101_blocked_decision_trace_extra(
    *,
    reason: str,
    decision_time: datetime,
    option_quotes: list[dict[str, Any]] | None,
    context_summary: dict[str, Any] | None,
    model_threshold: float | None,
) -> dict[str, Any]:
    candidate_universe = clean_json(option_quotes or [])
    source_quote_ts = max_iso_timestamp(
        (row.get("source_quote_time") or row.get("raw_quote_timestamp_utc") or row.get("quote_timestamp"))
        for row in candidate_universe
        if isinstance(row, dict)
    )
    source_context_ts = None
    if context_summary:
        source_context_ts = context_summary.get("last_timestamp") or context_summary.get("context_last_timestamp_utc")
    features = {"blocked_reason": reason, "candidate_features": []}
    scores = {"surface_scores": [], "protocol101_raw_logits": [], "threshold": model_threshold}
    return clean_json(
        {
            "decision_trace_schema_version": "Protocol101DecisionTraceV1",
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "source_quote_ts": source_quote_ts,
            "source_context_ts": source_context_ts,
            "quote_freshness_ms": max(
                [age for age in (finite_float(row.get("quote_age_ms")) for row in candidate_universe if isinstance(row, dict)) if age is not None],
                default=None,
            ),
            "candidate_universe": candidate_universe,
            "candidate_universe_hash": stable_json_hash(candidate_universe),
            "features": features,
            "feature_hash": stable_json_hash(features),
            "model_scores": scores,
            "score_hash": stable_json_hash(scores),
            "selected_action": "wait",
            "decision_threshold": model_threshold,
            "block_reasons": [reason],
            "decision_ts": decision_time.astimezone(UTC).isoformat(),
        }
    )


def max_iso_timestamp(values: Any) -> str | None:
    timestamps: list[pd.Timestamp] = []
    for value in values:
        if value is None:
            continue
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        timestamps.append(ts)
    if not timestamps:
        return None
    return max(timestamps).isoformat()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def candidate_feature_hash(candidates: Any) -> str:
    if getattr(candidates, "empty", True):
        return stable_json_hash([])
    return stable_json_hash(clean_json(candidates.to_dict(orient="records")))


def model_reconstruction_payload(
    *,
    prediction: dict[str, Any],
    model_action: str,
    selected_contract: dict[str, Any],
    candidate_set_hash: str,
    feature_vector_hash: str,
) -> dict[str, Any]:
    reason = prediction.get("reason")
    return {
        "action": model_action,
        "selected_action": prediction.get("selected_action") or model_action,
        "score": prediction.get("margin"),
        "selected_margin": prediction.get("selected_margin", prediction.get("margin")),
        "threshold": prediction.get("threshold"),
        "threshold_source": "protocol101_artifact",
        "reason": reason,
        "no_entry_reason": prediction.get("no_entry_reason") or (reason if model_action == "wait" else None),
        "action_mask": prediction.get("action_mask") or {"wait": True, "candidate_count": 0, "enter": False},
        "raw_logits": prediction.get("raw_logits", []),
        "wait_logit": prediction.get("wait_logit"),
        "candidate_logits": prediction.get("candidate_logits", []),
        "selected_contract": selected_contract,
        "candidate_set_hash": candidate_set_hash,
        "feature_vector_hash": feature_vector_hash,
    }


def risk_gate_payload(validation: dict[str, Any]) -> dict[str, Any]:
    reasons = validation.get("reasons")
    if reasons is None:
        reason = validation.get("reason")
        reasons = [] if reason in {None, "pass", "no_entry_intent", "insufficient_live_index_context", "outside_time_bucket"} else [reason]
    return {
        **validation,
        "passed": bool(validation.get("passed")),
        "guard_passed": bool(validation.get("passed")),
        "guard_block_reasons": list(reasons or []),
        "reason": validation.get("reason"),
    }


def live_option_quotes(option_tickers: list[tuple[Any, Any]], *, spx: float, now: datetime) -> list[dict[str, Any]]:
    rows = []
    atm = _round_to_5(spx)
    for contract, ticker in option_tickers:
        quote = _option_quote(ticker)
        if quote is None:
            continue
        strike = float(getattr(contract, "strike", 0.0))
        right = str(getattr(contract, "right", ""))
        greek_row = _compute_live_greeks(
            spx=spx,
            strike=strike,
            right=right,
            mid=float(quote["mid"]),
            ask=float(quote["ask"]),
            bid=float(quote["bid"]),
            now=now,
        )
        if greek_row is None:
            continue
        freshness = quote_freshness_from_ticker(ticker, observed_at=now)
        rows.append(
            {
                "contract_id": _contract_id(contract),
                "symbol": "SPX",
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", now.strftime("%Y%m%d"))),
                "strike": strike,
                "right": right,
                "trading_class": str(getattr(contract, "tradingClass", "SPXW") or "SPXW"),
                "settlement": "PM",
                "exchange": str(getattr(contract, "exchange", "SMART") or "SMART"),
                "currency": str(getattr(contract, "currency", "USD") or "USD"),
                "distance_points": strike - atm,
                **quote,
                **freshness,
                "raw_quote_timestamp_utc": freshness.get("quote_timestamp"),
                "received_timestamp_utc": freshness.get("received_timestamp"),
                "decision_timestamp_utc": freshness.get("decision_timestamp"),
                **greek_row,
            }
        )
    return rows


def spxw_open_positions(ib: Any) -> list[Any]:
    try:
        positions = list(ib.positions() or [])
    except Exception:
        return []
    out = []
    for position in positions:
        contract = getattr(position, "contract", None)
        qty = abs(float(getattr(position, "position", 0.0) or 0.0))
        trading_class = str(getattr(contract, "tradingClass", "") or getattr(contract, "trading_class", ""))
        symbol = str(getattr(contract, "symbol", "") or "")
        if qty > 0 and symbol == "SPX" and trading_class == "SPXW":
            out.append(position)
    return out


def handle_open_position(
    *,
    args: argparse.Namespace,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    lifecycle: Any,
    position: Any,
    spx: float,
    vix: float,
    now: datetime,
    account_id: str | None,
    trade_log: Path,
    session: str,
    run_id: str,
) -> dict[str, Any]:
    contract = getattr(position, "contract", None)
    qty = int(abs(float(getattr(position, "position", 0.0) or 0.0)))
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(1.0)
    quote = _option_quote(ticker)
    if quote is None:
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="holding_position_missing_exit_nbbo",
        )
        return {
            "decision_record": {"timestamp": now.isoformat(), "action": "hold", "reason": "holding_position_missing_exit_nbbo"},
            "executor_results": [],
            "broker_order_endpoint_called": False,
        }
    quote = {**quote, **quote_freshness_from_ticker(ticker, observed_at=now)}
    row = lifecycle_row_for_position(
        contract=contract,
        position=position,
        quote=quote,
        spx=spx,
        now=now,
        lifecycle=lifecycle,
        state=load_json(args.runtime_state),
        forced_flat_time=str(args.forced_flat_time),
    )
    frame = pd.DataFrame([row])
    value, recovery, decay = predict_protocol066_sequence(lifecycle, frame)
    prediction = prediction_for_step(
        step_index=0,
        value=value,
        recovery=recovery,
        decay=decay,
        override_threshold=lifecycle.selected_override_threshold,
        step_row=row,
    )
    selected = {
        **selected_from_contract(contract),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "bid_size": quote.get("bid_size"),
        "ask_size": quote.get("ask_size"),
        "quote_age_ms": quote.get("quote_age_ms"),
        "quote_age_source": quote.get("quote_age_source"),
        "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
        "quote_timestamp": quote.get("quote_timestamp"),
        "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
        "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
    }
    runtime_flag = load_json(args.runtime_flag)
    artifact_ids = {
        "lifecycle_manifest": str(DEFAULT_PROTOCOL081_MANIFEST),
        "runtime_state": str(args.runtime_state),
    }
    intent = None
    if prediction.action in {"exit", "stop", "forced_flat"}:
        intent = PaperOrderIntent(
            action="SELL",
            symbol="SPX",
            expiry=str(getattr(contract, "lastTradeDateOrContractMonth")),
            strike=float(getattr(contract, "strike")),
            right=str(getattr(contract, "right")),
            quantity=max(1, min(qty, 1)),
            limit_price=float(quote["bid"]),
            trading_class="SPXW",
            exchange="SMART",
            currency="USD",
        )
    append_event(
        trade_log,
        event_type="paper_exit_intent",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason=prediction.reason,
        selected_contract=selected,
        order=intent_order_payload(intent),
        model_decision={
            "action": prediction.action,
            "selected_action": prediction.action,
            "score": prediction.predicted_continuation_value,
            "selected_margin": prediction.predicted_continuation_value,
            "threshold": prediction.override_threshold,
            "reason": prediction.reason,
            "no_entry_reason": None if prediction.action in {"exit", "stop", "forced_flat"} else prediction.reason,
            "action_mask": {"hold": True, "exit": True, "forced_flat": minutes_to_forced_flat(now, str(args.forced_flat_time)) <= 0},
            "raw_logits": [
                prediction.predicted_continuation_value,
                prediction.predicted_recovery_probability,
                prediction.predicted_decay_probability,
            ],
            "wait_logit": prediction.predicted_continuation_value,
            "candidate_logits": [prediction.predicted_recovery_probability, prediction.predicted_decay_probability],
            "predicted_recovery_probability": prediction.predicted_recovery_probability,
            "predicted_decay_probability": prediction.predicted_decay_probability,
        },
        market_snapshot={
            "underlying": {"spx": spx, "vix": vix},
            "option_nbbo": {
                "bid": quote["bid"],
                "ask": quote["ask"],
                "bid_size": quote["bid_size"],
                "ask_size": quote["ask_size"],
                "quote_age_ms": quote.get("quote_age_ms"),
                "quote_age_source": quote.get("quote_age_source"),
                "quote_timestamp": quote.get("quote_timestamp"),
                "quote_timestamp_ms": quote.get("quote_timestamp_ms"),
                "received_timestamp": quote.get("received_timestamp"),
                "received_timestamp_ms": quote.get("received_timestamp_ms"),
                "decision_timestamp": quote.get("decision_timestamp"),
                "decision_timestamp_ms": quote.get("decision_timestamp_ms"),
            },
            "context": {"context_age_ms": 0, "source": "ibkr_live"},
        },
        extra={
            "runtime_flag": runtime_flag_summary(runtime_flag),
            "artifact_ids": artifact_ids,
            "lifecycle": {
                "position_detected": True,
                "position_contract_payload": selected,
                "position_quantity": qty,
                "position_avg_cost": _number(getattr(position, "avgCost", None), 0.0),
                "position_source": "ibkr_positions",
                "runtime_state_hash": stable_json_hash(load_json(args.runtime_state)),
                "lifecycle_feature_vector_hash": stable_json_hash(clean_json(row.to_dict())),
                "lifecycle_raw_scores": {
                    "value": prediction.predicted_continuation_value,
                    "recovery": prediction.predicted_recovery_probability,
                    "decay": prediction.predicted_decay_probability,
                },
                "lifecycle_action": prediction.action,
                "exit_intent_id": "protocol158_exit" if intent is not None else "",
                "forced_flat_due": minutes_to_forced_flat(now, str(args.forced_flat_time)) <= 0,
                "forced_flat_triggered": prediction.action == "forced_flat",
                "disconnect_while_holding": False,
                "final_position_state": "exit_intent" if intent is not None else "holding",
            }
        },
    )
    if intent is None:
        return {
            "decision_record": {"timestamp": now.isoformat(), "action": prediction.action, "reason": prediction.reason, "selected_contract": selected},
            "executor_results": [],
            "broker_order_endpoint_called": False,
        }
    if args.mode == "paper-submit" and not bool(runtime_flag.get("paper_orders_enabled")):
        append_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="runtime_flag_not_enabled_for_exit",
            selected_contract=selected,
            order=intent_order_payload(intent),
            risk_gate=risk_gate_payload({"passed": False, "reason": "runtime_flag_not_enabled_for_exit", "reasons": ["runtime_flag_not_enabled_for_exit"]}),
            extra={"runtime_flag": runtime_flag_summary(runtime_flag), "artifact_ids": artifact_ids},
        )
        return {
            "decision_record": {"timestamp": now.isoformat(), "action": "blocked", "reason": "runtime_flag_not_enabled_for_exit"},
            "executor_results": [],
            "broker_order_endpoint_called": False,
        }
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=1,
        quote={
            "bid": quote["bid"],
            "ask": quote["ask"],
            "reference_ask": quote["ask"],
            "quote_age_ms": quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
            "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
        },
        context={"context_age_ms": 0},
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=args.mode != "paper-submit",
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid="protocol158_exit",
        artifact_ids=artifact_ids,
        runtime_flag_digest=stable_json_hash(runtime_flag_summary(runtime_flag)),
        wait_for_fill_seconds=float(args.order_timeout_seconds),
        cancel_unfilled=True,
    )
    log_fill_or_status(
        trade_log=trade_log,
        result=result,
        event_prefix="paper_exit",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        trade_uid="protocol158_exit",
    )
    if result.get("fill_summary", {}).get("filled"):
        clear_runtime_state(args.runtime_state)
    return {
        "decision_record": {"timestamp": now.isoformat(), "action": prediction.action, "reason": prediction.reason, "selected_contract": selected},
        "executor_results": [result],
        "broker_order_endpoint_called": bool(result.get("broker_order_endpoint_called")),
    }


def lifecycle_row_for_position(
    *,
    contract: Any,
    position: Any,
    quote: dict[str, float],
    spx: float,
    now: datetime,
    lifecycle: Any,
    state: dict[str, Any],
    forced_flat_time: str,
) -> pd.Series:
    row = _live_feature_row(
        contract=contract,
        quote=quote,
        spx_price=spx,
        now=now,
        atm_strike=_round_to_5(spx),
        feature_columns=lifecycle.feature_columns,
    )
    if row is None:
        raise ValueError("could not build lifecycle row for open position")
    entry_price = entry_price_from_state_or_position(state, position)
    entry_time = pd.Timestamp(state.get("entry_time") or now.isoformat())
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    minutes_since_entry = max(0.0, (pd.Timestamp(now).tz_convert("UTC") - entry_time).total_seconds() / 60.0)
    current_pnl = (float(quote["bid"]) - entry_price) * 100.0
    previous_mfe = _number(state.get("mfe_to_now"), current_pnl)
    previous_mae = _number(state.get("mae_to_now"), current_pnl)
    mfe = max(previous_mfe, current_pnl)
    mae = min(previous_mae, current_pnl)
    row["minutes_since_entry"] = minutes_since_entry
    row["current_pnl"] = current_pnl
    row["mfe_to_now"] = mfe
    row["mae_to_now"] = mae
    row["giveback_from_mfe"] = max(0.0, mfe - current_pnl)
    row["giveback_fraction"] = max(0.0, (mfe - current_pnl) / max(abs(mfe), 1.0))
    row["bid_over_entry_ask"] = float(quote["bid"]) / max(entry_price, 1e-6)
    row["mid_over_entry_ask"] = float(quote["mid"]) / max(entry_price, 1e-6)
    row["entry_edge"] = _number(state.get("entry_edge"), 0.0)
    row["entry_offset"] = float(getattr(contract, "strike", 0.0)) - _round_to_5(spx)
    row["entry_is_call"] = 1.0 if str(getattr(contract, "right", "")) == "C" else 0.0
    row["entry_is_put"] = 1.0 if str(getattr(contract, "right", "")) == "P" else 0.0
    row["is_baseline_exit_step"] = False
    row["baseline_exit_reason"] = ""
    if minutes_to_forced_flat(now, forced_flat_time) <= 0:
        row["minutes_to_forced_flat"] = 0.0
    for column in lifecycle.feature_columns:
        row[column] = _number(row.get(column), 0.0)
    return row


def minutes_to_forced_flat(now: datetime, forced_flat_time: str) -> float:
    local = now.astimezone(NY)
    hour, minute = [int(part) for part in str(forced_flat_time).split(":", 1)]
    forced = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return (forced - local).total_seconds() / 60.0


def entry_price_from_state_or_position(state: dict[str, Any], position: Any) -> float:
    stored = _number(state.get("entry_fill_price"), 0.0)
    if stored > 0:
        return stored
    avg_cost = _number(getattr(position, "avgCost", None), 0.0)
    if avg_cost > 100.0:
        return avg_cost / 100.0
    if avg_cost > 0:
        return avg_cost
    return 0.01


def selected_from_contract(contract: Any) -> dict[str, Any]:
    return {
        "contract_id": _contract_id(contract),
        "symbol": "SPX",
        "root": "SPXW",
        "trading_class": "SPXW",
        "settlement": "PM",
        "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
        "strike": float(getattr(contract, "strike", 0.0)),
        "right": str(getattr(contract, "right", "")),
        "exchange": str(getattr(contract, "exchange", "SMART") or "SMART"),
        "currency": str(getattr(contract, "currency", "USD") or "USD"),
    }


def log_fill_or_status(
    *,
    trade_log: Path,
    result: dict[str, Any],
    event_prefix: str,
    session: str,
    run_id: str,
    mode: str,
    paper_cash: float,
    trade_uid: str,
) -> None:
    fill = result.get("fill_summary") or {}
    intent = result.get("intent") or {}
    contract = result.get("contract_preview") or {}
    order = result.get("order_preview") or {}
    event_type = "paper_order_status"
    if fill.get("filled"):
        event_type = f"{event_prefix}_fill"
    selected = {
        "symbol": contract.get("symbol") or intent.get("symbol"),
        "root": contract.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "trading_class": contract.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "settlement": "PM",
        "expiry": contract.get("lastTradeDateOrContractMonth") or intent.get("expiry"),
        "strike": contract.get("strike") or intent.get("strike"),
        "right": contract.get("right") or intent.get("right"),
        "exchange": contract.get("exchange") or intent.get("exchange"),
        "currency": contract.get("currency") or intent.get("currency"),
    }
    quote = result.get("quote") or {}
    selected.update(
        {
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "quote_age_ms": quote.get("quote_age_ms"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_timestamp": quote.get("quote_timestamp"),
            "received_timestamp_utc": quote.get("received_timestamp_utc") or quote.get("received_timestamp"),
            "decision_timestamp_utc": quote.get("decision_timestamp_utc") or quote.get("decision_timestamp"),
        }
    )
    order_row = {
        "action": order.get("action") or intent.get("action"),
        "quantity": order.get("totalQuantity") or intent.get("quantity"),
        "limit_price": order.get("lmtPrice") or intent.get("limit_price"),
        "contract_payload": selected,
        "order_payload": order or intent,
        "status": fill.get("status") or result.get("status"),
        "final_status": fill.get("status") or result.get("status"),
        "broker_order_id": result.get("broker_order_id"),
        "filled": fill.get("filled_quantity"),
        "remaining": fill.get("remaining_quantity"),
        "avg_fill_price": fill.get("avg_fill_price"),
        "cancel_requested": fill.get("cancel_requested"),
    }
    lifecycle_payload = {
        "position_detected": bool(fill.get("filled") or event_prefix == "paper_exit"),
        "runtime_state_hash": result.get("runtime_state_hash") or "",
        "lifecycle_action": "entry_fill" if event_prefix == "paper_entry" and fill.get("filled") else "broker_status",
        "exit_intent_id": trade_uid if event_prefix == "paper_exit" else "",
        "forced_flat_triggered": False,
        "final_position_state": "holding" if event_prefix == "paper_entry" and fill.get("filled") else "flat" if event_prefix == "paper_exit" and fill.get("filled") else "unknown",
    }
    append_event(
        trade_log,
        event_type=event_type,
        session=session,
        run_id=run_id,
        mode=mode,
        paper_cash=paper_cash,
        reason=str(fill.get("status") or result.get("reason") or result.get("status")),
        selected_contract=selected,
        order=order_row,
        risk_gate=risk_gate_payload(
            {
                "passed": bool(result.get("permission", {}).get("passed")) and bool(result.get("validation", {}).get("passed")),
                "reason": result.get("reason"),
                "reasons": [*result.get("permission", {}).get("reasons", []), *result.get("validation", {}).get("reasons", [])],
            }
        ),
        trade_uid=trade_uid,
        broker_order_endpoint_called=bool(result.get("broker_order_endpoint_called")),
        extra={
            "intent_id": result.get("intent_id"),
            "artifact_ids": result.get("artifact_ids"),
            "runtime_flag_digest": result.get("runtime_flag_digest"),
            "broker_status": fill,
            "lifecycle": lifecycle_payload,
        },
    )


def state_from_entry_result(result: dict[str, Any], *, fallback_intent: PaperOrderIntent) -> dict[str, Any]:
    fill = result.get("fill_summary") or {}
    intent = result.get("intent") or fallback_intent.__dict__
    fill_price = _number(fill.get("avg_fill_price"), float(intent.get("limit_price") or fallback_intent.limit_price))
    return {
        "state": "holding",
        "entry_time": datetime.now(tz=UTC).isoformat(),
        "entry_fill_price": fill_price,
        "entry_edge": 0.0,
        "mfe_to_now": 0.0,
        "mae_to_now": 0.0,
        "intent": intent,
    }


def write_runtime_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n")


def clear_runtime_state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": "flat", "cleared_at": datetime.now(tz=UTC).isoformat()}, indent=2, sort_keys=True) + "\n")


def validate_intent(*, intent: PaperOrderIntent | None, selected_contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if intent is None:
        return {
            "passed": True,
            "guard_passed": True,
            "reason": "no_entry_intent",
            "reasons": [],
            "guard_block_reasons": [],
            "quantity_ok": True,
            "one_open_position_ok": True,
            "quote_freshness_ok": True,
            "context_freshness_ok": True,
            "affordability_ok": True,
            "real_money_false_confirmed": True,
        }
    return validate_order_intent(
        intent,
        account_cash=float(args.paper_cash),
        open_positions=int(args.open_positions),
        quote={
            "bid": float(selected_contract.get("bid") or max(0.01, intent.limit_price - 0.10)),
            "ask": float(selected_contract.get("ask") or intent.limit_price),
            "reference_ask": float(selected_contract.get("ask") or intent.limit_price),
            "quote_age_ms": selected_contract.get("quote_age_ms"),
        },
        context={"context_age_ms": 0},
    )


def variant_for(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise ValueError(f"no registered A+ surface variant named {name!r}")


def candidate_sample(candidates: Any) -> list[dict[str, Any]]:
    if getattr(candidates, "empty", True):
        return []
    keep = ["contract_id", "right", "offset_points", "edge", "entry_ask", "entry_spread_frac", "entry_gamma", "entry_theta"]
    rows = []
    for _, row in candidates.head(8).iterrows():
        rows.append({key: clean_json(row.get(key)) for key in keep if key in row})
    return rows


def load_live_index_context(path: Path, state: LiveIndexState, *, session: str, include_prior_session_close: bool = False) -> int:
    """Load same-session SPX/VIX context so live features persist across cycles."""

    if not path.exists():
        return 0
    loaded = 0
    prior_row: dict[str, Any] | None = None
    same_session_rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row_session = str(row.get("session") or "")
        if row_session == str(session):
            same_session_rows.append(row)
        elif include_prior_session_close and row_session < str(session):
            prior_row = row
    rows_to_load = ([prior_row] if prior_row is not None else []) + same_session_rows
    for row in rows_to_load:
        if row is None:
            continue
        try:
            timestamp = pd.Timestamp(row["timestamp"])
            spx = float(row["spx"])
            vix = float(row["vix"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(spx) or not math.isfinite(vix):
            continue
        state.add(timestamp=timestamp, spx=spx, vix=vix)
        loaded += 1
    return loaded


def append_live_index_context(path: Path, *, session: str, timestamp: datetime, spx: float, vix: float) -> None:
    """Append one causal SPX/VIX point for later bridge cycles."""

    if not math.isfinite(float(spx)) or not math.isfinite(float(vix)):
        return
    row = {
        "session": session,
        "timestamp": timestamp.astimezone(UTC).isoformat(),
        "spx": float(spx),
        "vix": float(vix),
        "source": "ibkr_live",
    }

    def write_once() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    retry_io(write_once)


def live_context_ready(summary: dict[str, Any], *, min_context_minutes: float) -> bool:
    required = max(float(min_context_minutes), 0.0)
    if required <= 0:
        return True
    if summary.get("opening_context_ready") is False:
        return False
    try:
        span = float(summary.get("span_minutes") or 0.0)
        minute_rows = int(float(summary.get("minute_row_count") or 0.0))
    except (TypeError, ValueError):
        return False
    return span >= required and minute_rows >= int(math.ceil(required))


def append_time_bucket_block(
    args: argparse.Namespace,
    trade_log: Path,
    session: str,
    run_id: str,
    model_threshold: float,
    decision_time: datetime,
    context_summary: dict[str, Any],
    *,
    decision_market: dict[str, Any] | None = None,
    option_quotes: list[dict[str, Any]] | None = None,
    runtime_flag: dict[str, Any] | None = None,
    artifact_ids: dict[str, Any] | None = None,
) -> None:
    bucket = time_bucket(decision_time)
    diagnostics = {
        "filter_reason": "outside_time_bucket",
        "time_bucket": bucket,
        "allowed_time_bucket": False,
        "allowed_buckets": list(ALLOWED_ENTRY_TIME_BUCKETS),
        "min_edge": float(args.min_edge),
        "token_count": int(len(option_quotes or [])),
        "eligible_token_count": 0,
        "valid_score_count": 0,
        "above_min_edge_count": 0,
        "context_ready": live_context_ready(context_summary, min_context_minutes=float(args.min_live_context_minutes)),
        "required_minutes": float(args.min_live_context_minutes),
        "minute_row_count": context_summary.get("minute_row_count"),
        "span_minutes": context_summary.get("span_minutes"),
    }
    hash_payload = {"reason": "outside_time_bucket", "time_bucket": bucket, "option_quotes": option_quotes or []}
    trace_extra = protocol101_blocked_decision_trace_extra(
        reason="outside_time_bucket",
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
        reason="candidate_set_blocked_outside_time_bucket",
        market_snapshot=decision_market,
        extra={
            "candidate_set_hash": stable_json_hash(hash_payload),
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
        reason="outside_time_bucket",
        market_snapshot=decision_market,
        model_decision={
            "action": "wait",
            "selected_action": "wait",
            "score": None,
            "selected_margin": None,
            "threshold": model_threshold,
            "reason": "outside_time_bucket",
            "no_entry_reason": "outside_time_bucket",
            "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
            "raw_logits": [],
            "wait_logit": None,
            "candidate_logits": [],
        },
        extra={
            "candidate_set_hash": stable_json_hash(hash_payload),
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
        reason="outside_time_bucket",
        market_snapshot=decision_market,
        risk_gate=risk_gate_payload({"passed": True, "reason": "outside_time_bucket", "reasons": []}),
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


def intent_order_payload(intent: PaperOrderIntent | None) -> dict[str, Any]:
    if intent is None:
        return {}
    return {
        "action": intent.action,
        "quantity": intent.quantity,
        "limit_price": intent.limit_price,
        "premium_required": intent.premium_required,
    }


def append_event(
    path: Path,
    *,
    event_type: str,
    session: str,
    run_id: str,
    mode: str,
    paper_cash: float,
    reason: str,
    selected_contract: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
    model_decision: dict[str, Any] | None = None,
    risk_gate: dict[str, Any] | None = None,
    market_snapshot: dict[str, Any] | None = None,
    trade_uid: str | None = None,
    broker_order_endpoint_called: bool = False,
    account: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    extra_row = dict(extra or {})
    runtime_flag = extra_row.pop("runtime_flag", None)
    runtime_flag_digest = extra_row.pop("runtime_flag_digest", None)
    artifact_ids = extra_row.pop("artifact_ids", None)
    live_orders_enabled = extra_row.pop("live_orders_enabled", None)
    account_row = {
        "account_id_redacted": None,
        "starting_cash": float(paper_cash),
        "cash": float(paper_cash),
        "equity": float(paper_cash),
        "realized_daily_pnl": 0.0,
        "open_positions": 0,
    }
    if account:
        account_row.update(account)
    default_model = {
        "action": "blocked" if "blocked" in reason else "wait",
        "selected_action": "blocked" if "blocked" in reason else "wait",
        "reason": reason,
        "no_entry_reason": reason if "blocked" not in reason else None,
        "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
        "raw_logits": [],
        "candidate_logits": [],
        "wait_logit": None,
        "threshold": None,
    }
    default_risk = risk_gate_payload({"passed": not is_blocking_reason(reason), "reason": reason, "reasons": [] if not is_blocking_reason(reason) else [reason]})
    row = make_trade_log_event(
        event_type=event_type,
        session=session,
        run_id=run_id,
        mode=mode,
        trade_uid=trade_uid,
        selected_contract=selected_contract or {},
        order=order or {},
        account=account_row,
        market_snapshot=market_snapshot or {"underlying": {}, "option_nbbo": {}, "context": {}},
        model_decision=model_decision or default_model,
        risk_gate=risk_gate_payload(risk_gate) if risk_gate else default_risk,
        broker_order_endpoint_called=bool(broker_order_endpoint_called),
        paper_trading=True,
        real_money_trading=False,
        runtime_flag_digest=runtime_flag_digest or (stable_json_hash(runtime_flag) if runtime_flag is not None else None),
        artifact_ids=artifact_ids if isinstance(artifact_ids, dict) else None,
        live_orders_enabled=live_orders_enabled,
        extra=extra_row,
    )
    append_trade_event(path, row)


def is_blocking_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    if not text:
        return False
    blocking_prefixes = (
        "blocked",
        "entry_bridge_blocked",
        "missing_",
        "outside_",
        "no_valid_",
        "ibkr_connection_failed",
    )
    return text.startswith(blocking_prefixes) or "exception" in text or "failed" in text


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
        "protocol": "158_protocol101_live_entry_paper_bridge",
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


def decide(*, args: argparse.Namespace, decisions: list[dict[str, Any]], executor_results: list[dict[str, Any]], paper_submit_blocked: int) -> str:
    if not decisions:
        return "blocked_no_live_entry_decisions_emitted"
    if args.mode == "intent-shadow":
        return "pass_live_entry_intent_shadow_logged"
    if args.mode == "paper-dry-run":
        if any(row.get("status") == "dry_run_pass" for row in executor_results):
            return "pass_guarded_paper_order_dry_run_logged"
        if sum(1 for row in decisions if row["action"] == "enter") <= 0:
            return "pass_no_entry_intents_to_dry_run"
        return "blocked_paper_dry_run_permission_or_validation_failed"
    if any(row.get("paper_order_submitted") for row in executor_results):
        return "paper_order_submitted"
    if sum(1 for row in decisions if row["action"] == "enter") <= 0:
        return "pass_no_entry_intents_to_paper_submit"
    if paper_submit_blocked:
        return "blocked_paper_submit_runtime_flag_missing"
    return "blocked_paper_submit_no_order_submitted"


def next_gate(mode: str) -> str:
    if mode == "intent-shadow":
        return "Review model decisions and risk gates. Then run paper-dry-run before paper-submit."
    if mode == "paper-dry-run":
        return "If dry-run emits a valid order for a real Protocol101 entry, enable paper-submit only with explicit flags/env/runtime gate."
    return "Immediately inspect broker order state, fills, and account logs before allowing another paper order."


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
    return 0 if validation["status"] == "pass" and not str(payload["decision"]).startswith("blocked_protocol158_exception") else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 158: Protocol101 Live Entry Paper Bridge",
        "",
        "No paid data was downloaded. Real-money trading is disabled.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Decisions emitted: `{payload.get('decision_count', 0)}`",
        f"- Enter intents: `{payload.get('enter_intents', 0)}`",
        f"- Paper orders submitted: `{payload.get('paper_orders_submitted', 0)}`",
        f"- Broker order endpoint called: `{payload.get('broker_order_endpoint_called', False)}`",
        f"- Trade log: `{payload.get('trade_log')}`",
        f"- Trade log CSV: `{payload.get('trade_log_csv')}`",
        "",
        "## Next Gate",
        "",
        payload.get("next_gate", "Review blocker and retry."),
    ]
    if payload.get("blocked_reason"):
        lines.extend(["", "## Blocker", "", f"- `{payload['blocked_reason']}`", f"- {payload.get('detail')}"])
    path.write_text("\n".join(lines) + "\n")


def first_managed_account(ib: Any) -> str | None:
    try:
        accounts = list(ib.managedAccounts() or [])
    except Exception:
        return None
    return str(accounts[0]) if accounts else None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def pre_live_sanity_gate_status(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "skip_pre_live_sanity_gate", False)):
        return {"passed": True, "reason": "skipped_by_explicit_cli_flag", "path": None}
    path = Path(getattr(args, "pre_live_sanity_gate_summary", DEFAULT_PRE_LIVE_SANITY_GATE_SUMMARY))
    if not path.exists():
        return {"passed": False, "reason": "missing_pre_live_historical_sanity_gate_summary", "path": str(path)}
    payload = load_json(path)
    passed = bool(payload.get("paper_submit_allowed_by_gate")) and str(payload.get("status")) == "pass"
    return {
        "passed": passed,
        "reason": "pass" if passed else str(payload.get("status") or "pre_live_historical_sanity_gate_failed"),
        "path": str(path),
        "summary_status": payload.get("status"),
        "errors": payload.get("errors", []),
        "warnings": payload.get("warnings", []),
    }


def runtime_flag_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": bool(payload),
        "paper_orders_enabled": bool(payload.get("paper_orders_enabled")),
        "enabled_at": payload.get("enabled_at"),
        "session": payload.get("session"),
        "run_id": payload.get("run_id"),
        "account_id_redacted": payload.get("account_id_redacted"),
    }


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


def redact_account(account_id: str | None) -> str | None:
    if not account_id:
        return None
    text = str(account_id)
    return f"{text[:2]}***{text[-2:]}" if len(text) > 4 else "***"


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
