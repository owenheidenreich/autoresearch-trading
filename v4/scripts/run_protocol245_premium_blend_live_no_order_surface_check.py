"""RUNTIME_PREMIUM_BLEND_LIVE_SURFACE_AUTOTEST_V1.

Historically Protocol245. This is a broker-connected, no-order runtime check for
CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1. It verifies that the live IBKR
surface can supply the same ATM +/- $50 SPXW 0DTE action space that the
challenger expects, then scores the challenger without submitting orders.

PAPER_DEFAULT_PROTOCOL101 remains unchanged. This runner does not place paper
orders and does not call any broker order endpoint.
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

import numpy as np
import pandas as pd

from v4.live.full_action_challenger_adapter import FullActionHistoryState, build_full_action_candidates_from_quotes
from v4.live.paper_trade_log import (
    DEFAULT_TRADE_LOG_ROOT,
    append_trade_event,
    export_trade_log_csv,
    load_trade_log,
    make_trade_log_event,
    trade_log_path,
    validate_trade_log,
)
from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol101_live_entry import LiveIndexState, build_live_surface_row, live_surface_decision
from v4.scripts.run_protocol081_live_shadow_router import (
    IbkrErrorLog,
    _connect_ibkr,
    _discover_spxw_0dte_contracts,
    _is_regular_market_hours,
    _market_data_type_name,
    _request_index_ticker,
    _ticker_market_data_type,
    _ticker_price,
)
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST
from v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge import clean_json, live_option_quotes, variant_for
from v4.scripts.run_protocol217_full_action_history_runtime_parity import (
    build_feature_tensor,
    choose_candidate,
    live_safe_candidate_mask,
    load_artifact,
    run_model,
)
from v4.scripts.run_protocol241_premium_blend_runtime_parity import DEFAULT_ARTIFACT_MANIFEST


ROLE_LABEL = "RUNTIME_PREMIUM_BLEND_LIVE_SURFACE_AUTOTEST_V1"
HISTORICAL_ID = "Protocol245"
PROTOCOL_ID = "challenger_premium_leaning_blended_utility_v1"
CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest")
NY = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=245)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--live-strikes-around-atm", type=int, default=10)
    parser.add_argument("--min-valid-candidates", type=int, default=30)
    parser.add_argument("--capture-seconds", type=float, default=300.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=60.0)
    parser.add_argument("--quote-warmup-seconds", type=float, default=3.0)
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    parser.add_argument("--skip-market-clock", action="store_true")
    parser.add_argument("--config-check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(tz=NY)
    session = args.session_date or now.date().isoformat()
    run_id = args.run_id or f"premium_blend_live_surface_autotest_{session}"
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log = trade_log_path(root=args.trade_log_root, session=session, run_id=run_id)
    csv_log = trade_log.with_suffix(".csv")

    if args.config_check_only:
        payload = config_payload(args, session=session, run_id=run_id, trade_log=trade_log)
        return finish(out_dir, payload, trade_log, csv_log)

    if not args.skip_market_clock and not _is_regular_market_hours(now):
        payload = blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="outside_regular_market_hours")
        return finish(out_dir, payload, trade_log, csv_log)

    try:
        payload = run_live_surface_check(args, session=session, run_id=run_id, trade_log=trade_log)
    except Exception as exc:
        payload = blocked_payload(
            args,
            session=session,
            run_id=run_id,
            trade_log=trade_log,
            reason="protocol245_exception",
            detail=str(exc),
        )
    return finish(out_dir, payload, trade_log, csv_log)


def run_live_surface_check(args: argparse.Namespace, *, session: str, run_id: str, trade_log: Path) -> dict[str, Any]:
    try:
        from ib_insync import IB, Index, Option  # type: ignore
    except ImportError:
        return blocked_payload(args, session=session, run_id=run_id, trade_log=trade_log, reason="missing_ib_insync")

    surface_artifact = load_surface_edge_artifact(args.surface_manifest)
    surface_variant = variant_for(surface_artifact.variant_name)
    challenger = load_artifact(args.artifact_manifest)
    history = FullActionHistoryState()
    index_state = LiveIndexState()
    error_log = IbkrErrorLog()
    ib = None
    index_contracts: list[Any] = []
    option_subscriptions: list[tuple[Any, Any]] = []
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    chain_meta: dict[str, Any] = {}
    connected_port: int | None = None
    attempts: list[dict[str, Any]] = []
    broker_order_endpoint_called = False

    append_runtime_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        paper_cash=float(args.paper_cash),
        reason="premium_blend_live_surface_autotest_started",
        model_decision={"action": "startup", "candidate": CHALLENGER_LABEL},
        extra={
            "role_label": ROLE_LABEL,
            "expected_raw_contracts": expected_contract_count(int(args.live_strikes_around_atm)),
            "live_orders_enabled": False,
        },
    )

    try:
        ib, connected_port, attempts = _connect_ibkr(args, IB)
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
        ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)
        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
        index_contracts = [contract for contract in (spx_contract, vix_contract) if contract is not None]
        ib.sleep(max(0.5, float(args.quote_warmup_seconds)))
        current_spx = _ticker_price(spx_ticker)
        current_vix = _ticker_price(vix_ticker)
        if current_spx is None or current_vix is None:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason="missing_live_spx_or_vix",
                extra={"ibkr_errors": error_log.events[-20:]},
            )
        option_subscriptions, chain_meta = refresh_live_surface(
            ib=ib,
            option_cls=Option,
            spx_contract=spx_contract,
            spx=float(current_spx),
            strikes_around_atm=int(args.live_strikes_around_atm),
            now=datetime.now(tz=NY),
        )
        if not option_subscriptions:
            return blocked_payload(
                args,
                session=session,
                run_id=run_id,
                trade_log=trade_log,
                reason="no_spxw_0dte_option_ladder",
                extra={"chain_meta": chain_meta, "ibkr_errors": error_log.events[-20:]},
            )
        ib.sleep(max(0.5, float(args.quote_warmup_seconds)))

        deadline = time.monotonic() + max(1.0, float(args.capture_seconds))
        next_decision = 0.0
        while time.monotonic() <= deadline:
            ib.sleep(0.25)
            if time.monotonic() < next_decision:
                continue
            next_decision = time.monotonic() + max(1.0, float(args.decision_interval_seconds))
            decision_time = datetime.now(tz=UTC)
            current_spx = _ticker_price(spx_ticker)
            current_vix = _ticker_price(vix_ticker)
            if current_spx is None or current_vix is None:
                append_runtime_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    paper_cash=float(args.paper_cash),
                    reason="missing_live_context_snapshot",
                    risk_gate={"passed": False, "reason": "missing_live_context_snapshot"},
                    extra={"ibkr_errors": error_log.events[-10:]},
                )
                continue
            index_state.add(timestamp=decision_time, spx=float(current_spx), vix=float(current_vix))
            option_quotes = live_option_quotes(option_subscriptions, spx=float(current_spx), now=decision_time)
            option_quotes = add_surface_edges(
                surface_artifact=surface_artifact,
                surface_variant=surface_variant,
                session=session,
                decision_time=decision_time,
                spx=float(current_spx),
                vix=float(current_vix),
                option_quotes=option_quotes,
                index_state=index_state,
            )
            market_features = live_market_features(index_state, decision_time=decision_time)
            start = time.perf_counter()
            candidates = build_full_action_candidates_from_quotes(
                decision_time=decision_time,
                spx=float(current_spx),
                vix=float(current_vix),
                option_quotes=option_quotes,
                history=history,
                account_cash=float(args.paper_cash),
                starting_cash=float(args.paper_cash),
                market_features=market_features,
                session=session,
            )
            feature_build_ms = (time.perf_counter() - start) * 1000.0
            account_state = {
                "starting_cash": float(args.paper_cash),
                "account_equity": float(args.paper_cash),
                "cash_available": float(args.paper_cash),
                "open_position_count": 0,
                "max_concurrent_positions": 1,
                "max_contracts": 1,
            }
            validation_start = time.perf_counter()
            mask = live_safe_candidate_mask(candidates, account_state=account_state) if not candidates.empty else np.zeros(0, dtype=bool)
            validation_ms = (time.perf_counter() - validation_start) * 1000.0
            inference_start = time.perf_counter()
            if candidates.empty:
                candidate_idx, score = None, float("-inf")
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
            else:
                tensor = build_feature_tensor(candidates, mask, challenger["feature_columns"], challenger["scaler"])
                output = run_model(challenger["model"], tensor)
                candidate_idx, score = choose_candidate(output, mask)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
            total_ms = feature_build_ms + validation_ms + inference_ms
            selected_action = "wait"
            selected_contract: dict[str, Any] = {}
            order: dict[str, Any] = {}
            if bool(mask.any()) and candidate_idx is not None and float(score) >= float(challenger["threshold"]):
                selected = candidates.iloc[int(candidate_idx)]
                selected_action = "enter"
                selected_contract = selected_contract_payload(selected, quantity=int(args.quantity))
                order = order_payload(selected, quantity=int(args.quantity))
            candidate_summary = candidate_set_summary(
                candidates,
                option_quotes=option_quotes,
                mask=mask,
                chain_meta=chain_meta,
                strikes_around_atm=int(args.live_strikes_around_atm),
            )
            risk_gate = {
                "passed": bool(candidate_summary["valid_candidate_count"] >= int(args.min_valid_candidates)),
                "reason": "pass" if candidate_summary["valid_candidate_count"] >= int(args.min_valid_candidates) else "insufficient_valid_live_candidates",
            }
            market_snapshot = {
                "underlying": {"spx": float(current_spx), "vix": float(current_vix)},
                "option_nbbo": candidate_summary,
                "context": {
                    "source": "ibkr_live",
                    "context_age_ms": 0,
                    "spx_market_data_type": _market_data_type_name(_ticker_market_data_type(spx_ticker)),
                    "vix_market_data_type": _market_data_type_name(_ticker_market_data_type(vix_ticker)),
                },
            }
            append_runtime_event(
                trade_log,
                event_type="market_snapshot",
                session=session,
                run_id=run_id,
                paper_cash=float(args.paper_cash),
                reason="live_full_action_surface_snapshot",
                market_snapshot=market_snapshot,
            )
            append_runtime_event(
                trade_log,
                event_type="candidate_set",
                session=session,
                run_id=run_id,
                paper_cash=float(args.paper_cash),
                reason="premium_blend_candidate_set_built",
                market_snapshot=market_snapshot,
                model_decision={
                    "action": "candidate_set",
                    "candidate_count": int(candidate_summary["candidate_count"]),
                    "valid_candidate_count": int(candidate_summary["valid_candidate_count"]),
                },
                extra={"candidate_sample": candidate_sample(candidates, mask), "chain_meta": chain_meta},
            )
            append_runtime_event(
                trade_log,
                event_type="model_decision",
                session=session,
                run_id=run_id,
                paper_cash=float(args.paper_cash),
                reason=selected_action,
                selected_contract=selected_contract,
                order=order,
                market_snapshot=market_snapshot,
                model_decision={
                    "action": selected_action,
                    "score": float(score) if math.isfinite(float(score)) else None,
                    "threshold": float(challenger["threshold"]),
                    "candidate_index": None if candidate_idx is None else int(candidate_idx),
                    "candidate_label": CHALLENGER_LABEL,
                    "paper_default_unchanged": PAPER_DEFAULT_LABEL,
                },
            )
            append_runtime_event(
                trade_log,
                event_type="risk_gate",
                session=session,
                run_id=run_id,
                paper_cash=float(args.paper_cash),
                reason=str(risk_gate["reason"]),
                selected_contract=selected_contract,
                order=order,
                market_snapshot=market_snapshot,
                model_decision={"action": selected_action, "score": finite(score, None), "threshold": float(challenger["threshold"])},
                risk_gate=risk_gate,
            )
            latency_row = {
                "timestamp": decision_time.isoformat(),
                "selected_action": selected_action,
                "candidate_count": int(candidate_summary["candidate_count"]),
                "valid_candidate_count": int(candidate_summary["valid_candidate_count"]),
                "raw_quote_count": int(candidate_summary["raw_quote_count"]),
                "expected_raw_contracts": expected_contract_count(int(args.live_strikes_around_atm)),
                "score": finite(score, None),
                "threshold": float(challenger["threshold"]),
                "feature_build_ms": float(feature_build_ms),
                "candidate_validation_ms": float(validation_ms),
                "model_inference_ms": float(inference_ms),
                "total_decision_ms": float(total_ms),
                "budget_passed": bool(total_ms <= 1000.0),
            }
            latency_rows.append(latency_row)
            rows.append(
                {
                    "timestamp": decision_time.isoformat(),
                    "selected_action": selected_action,
                    "selected_contract": selected_contract,
                    "candidate_set": candidate_summary,
                    "risk_gate": risk_gate,
                    "latency": latency_row,
                }
            )
            if not candidates.empty:
                history.update(candidates, decision_time=decision_time)
    finally:
        cleanup_subscriptions(ib, index_contracts, option_subscriptions)
        if ib is not None and ib.isConnected():
            ib.disconnect()

    latency_summary = summarize_latency(latency_rows)
    candidate_breadth = summarize_candidate_breadth(rows, expected_raw=expected_contract_count(int(args.live_strikes_around_atm)))
    validation = validate_live_surface_summary(
        rows=rows,
        trade_rows=load_trade_log(trade_log),
        candidate_breadth=candidate_breadth,
        min_valid_candidates=int(args.min_valid_candidates),
        broker_order_endpoint_called=broker_order_endpoint_called,
    )
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / broker-connected no-order live surface breadth and challenger scoring check",
        "changes_paper_default": False,
        "candidate_label": CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "data_used": "IBKR live market data only; no paid historical download",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "paper_orders_submitted": 0,
        "model_training": False,
        "session": session,
        "run_id": run_id,
        "ibkr_connected": connected_port is not None,
        "ibkr_port": connected_port,
        "connection_attempts": attempts,
        "chain_meta": chain_meta,
        "expected_raw_contracts": expected_contract_count(int(args.live_strikes_around_atm)),
        "capture_seconds": float(args.capture_seconds),
        "decision_count": int(len(rows)),
        "enter_intents": int(sum(1 for row in rows if row.get("selected_action") == "enter")),
        "latency_summary": latency_summary,
        "candidate_breadth": candidate_breadth,
        "live_surface_validation": validation,
        "ibkr_errors": error_log.events[-50:],
        "subscription_errors": error_log.subscription_errors[-50:],
        "trade_log": str(trade_log),
        "decision": decision_from_validation(validation),
        "next_experiment": (
            "If this passes during market hours, compare the challenger no-order decisions with PAPER_DEFAULT_PROTOCOL101 "
            "for the same live session before any paper-default replacement."
        ),
    }
    pd.DataFrame(latency_rows).to_csv((trade_log.parent / f"{trade_log.stem}_latency.csv"), index=False)
    return payload


def add_surface_edges(
    *,
    surface_artifact: Any,
    surface_variant: Any,
    session: str,
    decision_time: datetime,
    spx: float,
    vix: float,
    option_quotes: list[dict[str, Any]],
    index_state: LiveIndexState,
) -> list[dict[str, Any]]:
    if not option_quotes:
        return []
    row, _lookup = build_live_surface_row(
        decision_time=decision_time,
        spx=float(spx),
        vix=float(vix),
        option_quotes=option_quotes,
        index_state=index_state,
    )
    decision = live_surface_decision(
        session=session,
        row=row,
        variant=surface_variant,
        policy_index=surface_artifact.policy_index,
        index_state=index_state,
    )
    scores = np.asarray(score_surface_decisions(surface_artifact, [decision])[0], dtype=float)
    flat = finite(scores[0], 0.0)
    edges: dict[str, float] = {}
    for contract_id, token_score in zip(decision.contract_ids, scores[1:]):
        if str(contract_id):
            edges[str(contract_id)] = finite(token_score, flat) - flat
    enriched = []
    for quote in option_quotes:
        item = dict(quote)
        edge = edges.get(str(item.get("contract_id")), 0.0)
        item["surface_edge"] = float(edge)
        item["edge"] = float(edge)
        item["root"] = "SPXW"
        item["settlement_style"] = "PM"
        enriched.append(item)
    return enriched


def refresh_live_surface(
    *,
    ib: Any,
    option_cls: Any,
    spx_contract: Any,
    spx: float,
    strikes_around_atm: int,
    now: datetime,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    contracts, chain_meta = _discover_spxw_0dte_contracts(
        ib=ib,
        option_cls=option_cls,
        spx_contract=spx_contract,
        spx_price=float(spx),
        now=now,
        strikes_around_atm=int(strikes_around_atm),
    )
    subscriptions = [(contract, ib.reqMktData(contract, "", False, False)) for contract in contracts]
    return subscriptions, chain_meta


def live_market_features(index_state: LiveIndexState, *, decision_time: datetime) -> dict[str, float]:
    frame = index_state.frame(decision_time)
    if frame.empty:
        return {}
    spx = pd.to_numeric(frame["spx"], errors="coerce").dropna()
    vix = pd.to_numeric(frame["vix"], errors="coerce").dropna()
    current = float(spx.iloc[-1]) if len(spx) else 0.0
    high = float(spx.max()) if len(spx) else current
    low = float(spx.min()) if len(spx) else current
    session_range = max(high - low, 0.0)
    mid = (high + low) / 2.0
    return {
        "market_spx_close": current,
        "market_vix_close": float(vix.iloc[-1]) if len(vix) else 0.0,
        "market_spx_vwap": float(spx.mean()) if len(spx) else current,
        "market_omar": min(abs(current - high), abs(current - low), abs(current - mid)) if len(spx) else 0.0,
        "market_session_range": session_range,
        "market_momentum_5m": current - float(spx.iloc[-6]) if len(spx) >= 6 else 0.0,
        "market_momentum_15m": current - float(spx.iloc[-16]) if len(spx) >= 16 else 0.0,
    }


def expected_contract_count(strikes_around_atm: int) -> int:
    return (int(strikes_around_atm) * 2 + 1) * 2


def candidate_set_summary(
    candidates: pd.DataFrame,
    *,
    option_quotes: list[dict[str, Any]],
    mask: np.ndarray,
    chain_meta: dict[str, Any],
    strikes_around_atm: int,
) -> dict[str, Any]:
    if candidates.empty:
        return {
            "candidate_count": 0,
            "valid_candidate_count": 0,
            "raw_quote_count": int(len(option_quotes)),
            "expected_raw_contracts": expected_contract_count(strikes_around_atm),
            "requested_contracts": int(chain_meta.get("requested_contracts") or 0),
            "qualified_contracts": int(chain_meta.get("qualified_contracts") or 0),
            "root": "SPXW",
            "settlement_style": "PM",
            "call_count": 0,
            "put_count": 0,
            "max_abs_offset": None,
            "protocol101_min_edge_gate_applied": False,
            "protocol101_time_bucket_gate_applied": False,
        }
    quote_age_values = (
        pd.to_numeric(candidates["quote_age_ms"], errors="coerce").dropna()
        if "quote_age_ms" in candidates
        else pd.Series(dtype=float)
    )
    return {
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(mask.sum()),
        "raw_quote_count": int(len(option_quotes)),
        "expected_raw_contracts": expected_contract_count(strikes_around_atm),
        "requested_contracts": int(chain_meta.get("requested_contracts") or 0),
        "qualified_contracts": int(chain_meta.get("qualified_contracts") or 0),
        "root": "SPXW" if candidates["root"].astype(str).eq("SPXW").all() else "mixed",
        "settlement_style": "PM" if candidates["settlement_style"].astype(str).eq("PM").all() else "mixed",
        "call_count": int(candidates["right"].astype(str).eq("C").sum()),
        "put_count": int(candidates["right"].astype(str).eq("P").sum()),
        "max_abs_offset": finite(pd.to_numeric(candidates["offset"], errors="coerce").abs().max(), None),
        "max_option_quote_age_ms": finite(quote_age_values.max(), 0.0) if len(quote_age_values) else 0.0,
        "protocol101_min_edge_gate_applied": False,
        "protocol101_time_bucket_gate_applied": False,
    }


def candidate_sample(candidates: pd.DataFrame, mask: np.ndarray, *, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(candidates.head(limit).iterrows()):
        out.append(
            {
                "contract_id": str(row.get("contract_id")),
                "right": str(row.get("right")),
                "offset": finite(row.get("offset"), None),
                "entry_ask": finite(row.get("entry_ask"), None),
                "entry_premium": finite(row.get("entry_premium"), None),
                "surface_edge": finite(row.get("surface_edge"), None),
                "valid": bool(mask[idx]) if idx < len(mask) else False,
            }
        )
    return out


def selected_contract_payload(row: pd.Series, *, quantity: int) -> dict[str, Any]:
    return {
        "contract_id": str(row["contract_id"]),
        "symbol": "SPX",
        "root": "SPXW",
        "trading_class": "SPXW",
        "settlement_style": "PM",
        "expiry": expiry_from_contract_id(str(row["contract_id"])),
        "strike": strike_from_contract_id(str(row["contract_id"]), fallback=finite(row.get("entry_underlying_price"), 0.0) + finite(row.get("offset"), 0.0)),
        "right": str(row["right"]),
        "quantity": int(quantity),
        "exchange": "SMART",
        "currency": "USD",
    }


def order_payload(row: pd.Series, *, quantity: int) -> dict[str, Any]:
    ask = finite(row.get("entry_ask"), 0.0)
    return {
        "action": "BUY",
        "quantity": int(quantity),
        "limit_price": ask,
        "premium_required": ask * CONTRACT_MULTIPLIER * int(quantity),
        "mode": "no_order_intent_only",
    }


def append_runtime_event(
    path: Path,
    *,
    event_type: str,
    session: str,
    run_id: str,
    paper_cash: float,
    reason: str,
    selected_contract: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
    account: dict[str, Any] | None = None,
    market_snapshot: dict[str, Any] | None = None,
    model_decision: dict[str, Any] | None = None,
    risk_gate: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    row = make_trade_log_event(
        event_type=event_type,
        session=session,
        run_id=run_id,
        mode="challenger-no-order-live-surface",
        selected_contract=selected_contract or {},
        order=order or {},
        account=account or {
            "account_id_redacted": None,
            "starting_cash": float(paper_cash),
            "cash": float(paper_cash),
            "equity": float(paper_cash),
            "realized_daily_pnl": 0.0,
            "open_positions": 0,
        },
        market_snapshot=market_snapshot or {"underlying": {}, "option_nbbo": {}, "context": {}},
        model_decision=model_decision or {"action": "wait", "reason": reason},
        risk_gate=risk_gate or {"passed": not str(reason).startswith("blocked"), "reason": reason},
        broker_order_endpoint_called=False,
        paper_trading=True,
        real_money_trading=False,
        protocol_id=PROTOCOL_ID,
        extra={"reason": reason, "live_orders_enabled": False, **(extra or {})},
    )
    append_trade_event(path, row)


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "rows": int(len(frame)),
        "action_counts": frame["selected_action"].value_counts().to_dict(),
        "budget_pass_fraction": float(frame["budget_passed"].mean()) if len(frame) else 0.0,
    }
    for column in ["feature_build_ms", "candidate_validation_ms", "model_inference_ms", "total_decision_ms"]:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        out[column] = {
            "p50": float(values.quantile(0.50)) if len(values) else None,
            "p95": float(values.quantile(0.95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    return out


def summarize_candidate_breadth(rows: list[dict[str, Any]], *, expected_raw: int) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "expected_raw_contracts": int(expected_raw)}
    frame = pd.DataFrame([row["candidate_set"] for row in rows])
    return {
        "rows": int(len(frame)),
        "expected_raw_contracts": int(expected_raw),
        "max_requested_contracts": int(column_max(frame, "requested_contracts", 0)),
        "max_qualified_contracts": int(column_max(frame, "qualified_contracts", 0)),
        "max_raw_quote_count": int(column_max(frame, "raw_quote_count", 0)),
        "max_candidate_count": int(column_max(frame, "candidate_count", 0)),
        "max_valid_candidate_count": int(column_max(frame, "valid_candidate_count", 0)),
        "min_valid_candidate_count": int(column_min(frame, "valid_candidate_count", 0)),
        "max_abs_offset": finite(column_max(frame, "max_abs_offset", 0), None),
        "call_count_max": int(column_max(frame, "call_count", 0)),
        "put_count_max": int(column_max(frame, "put_count", 0)),
    }


def column_max(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if len(values) else float(default)


def column_min(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if len(values) else float(default)


def validate_live_surface_summary(
    *,
    rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    candidate_breadth: dict[str, Any],
    min_valid_candidates: int,
    broker_order_endpoint_called: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    trade_validation = validate_trade_log(trade_rows)
    if trade_validation.get("status") != "pass":
        errors.extend(str(item) for item in trade_validation.get("errors", []))
    if broker_order_endpoint_called or int(trade_validation.get("broker_order_endpoint_called_rows", 0)) != 0:
        errors.append("broker_order_endpoint_called")
    if not rows:
        errors.append("no_live_surface_decisions_emitted")
    if int(candidate_breadth.get("max_requested_contracts") or 0) < int(candidate_breadth.get("expected_raw_contracts") or 0):
        errors.append("full_ladder_not_requested")
    if int(candidate_breadth.get("max_valid_candidate_count") or 0) < int(min_valid_candidates):
        errors.append("insufficient_valid_live_candidates")
    if int(candidate_breadth.get("call_count_max") or 0) <= 0 or int(candidate_breadth.get("put_count_max") or 0) <= 0:
        errors.append("missing_call_or_put_side")
    if any(row.get("risk_gate", {}).get("passed") is False for row in rows):
        warnings.append("one_or_more_decision_rows_failed_candidate_breadth_gate")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "trade_log_validation": trade_validation,
    }


def decision_from_validation(validation: dict[str, Any]) -> str:
    if validation.get("status") != "pass":
        return "blocked_live_surface_autotest_failed_protocol101_default_unchanged"
    return "pass_live_surface_autotest_ready_for_challenger_shadow_protocol101_default_unchanged"


def config_payload(args: argparse.Namespace, *, session: str, run_id: str, trade_log: Path) -> dict[str, Any]:
    surface = load_surface_edge_artifact(args.surface_manifest)
    challenger = load_artifact(args.artifact_manifest)
    return {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / config-only check for Tuesday no-order live surface autotest",
        "changes_paper_default": False,
        "candidate_label": CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "paper_orders_submitted": 0,
        "model_training": False,
        "session": session,
        "run_id": run_id,
        "surface_manifest": str(args.surface_manifest),
        "surface_variant": surface.variant_name,
        "artifact_manifest": str(args.artifact_manifest),
        "challenger_feature_count": int(len(challenger["feature_columns"])),
        "expected_raw_contracts": expected_contract_count(int(args.live_strikes_around_atm)),
        "min_valid_candidates": int(args.min_valid_candidates),
        "capture_seconds": float(args.capture_seconds),
        "decision": "ready_config_only_tuesday_live_surface_autotest_protocol101_default_unchanged",
        "trade_log": str(trade_log),
        "next_experiment": "Install launchd assets and let this no-order live surface check run on Tuesday during market hours.",
    }


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
    append_runtime_event(
        trade_log,
        event_type="paper_order_blocked",
        session=session,
        run_id=run_id,
        paper_cash=float(args.paper_cash),
        reason=reason,
        risk_gate={"passed": False, "reason": reason},
        extra={"detail": detail, **(extra or {})},
    )
    return {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / broker-connected no-order live surface breadth and challenger scoring check",
        "changes_paper_default": False,
        "candidate_label": CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "paper_orders_submitted": 0,
        "model_training": False,
        "session": session,
        "run_id": run_id,
        "decision": f"blocked_{reason}_protocol101_default_unchanged",
        "blocked_reason": reason,
        "detail": detail,
        "trade_log": str(trade_log),
        **(extra or {}),
    }


def finish(out_dir: Path, payload: dict[str, Any], trade_log: Path, csv_log: Path) -> int:
    rows = load_trade_log(trade_log)
    validation = validate_trade_log(rows)
    csv_summary = export_trade_log_csv(trade_log, csv_log) if trade_log.exists() else {"rows": 0}
    payload["trade_log_validation"] = validation
    payload["trade_log_csv"] = str(csv_log)
    payload["trade_log_csv_summary"] = csv_summary
    (out_dir / "summary.json").write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True, default=str) + "\n")
    write_report(out_dir / "report.md", payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "report": str(out_dir / "report.md"),
                "summary": str(out_dir / "summary.json"),
                "trade_log": str(trade_log),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["status"] == "pass" and not str(payload["decision"]).startswith("blocked_protocol245_exception") else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload.get('next_experiment', 'Review the live no-order surface check output.')}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Session: `{payload['session']}`",
        f"Run ID: `{payload['run_id']}`",
    ]
    if payload.get("blocked_reason"):
        lines.extend(["", "## Blocker", "", f"- Reason: `{payload['blocked_reason']}`", f"- Detail: `{payload.get('detail')}`"])
    if payload.get("candidate_breadth"):
        breadth = payload["candidate_breadth"]
        lines.extend(
            [
                "",
                "## Surface Breadth",
                "",
                f"- Expected raw contracts: `{breadth.get('expected_raw_contracts')}`",
                f"- Max requested contracts: `{breadth.get('max_requested_contracts')}`",
                f"- Max qualified contracts: `{breadth.get('max_qualified_contracts')}`",
                f"- Max raw quote count: `{breadth.get('max_raw_quote_count')}`",
                f"- Max candidate count: `{breadth.get('max_candidate_count')}`",
                f"- Max valid candidate count: `{breadth.get('max_valid_candidate_count')}`",
                f"- Call/put max counts: `{breadth.get('call_count_max')}` / `{breadth.get('put_count_max')}`",
            ]
        )
    if payload.get("latency_summary"):
        latency = payload["latency_summary"]
        lines.extend(
            [
                "",
                "## Latency",
                "",
                f"- Rows: `{latency.get('rows')}`",
                f"- Action counts: `{latency.get('action_counts')}`",
                f"- Budget pass fraction: `{latency.get('budget_pass_fraction')}`",
                f"- Total decision p95 ms: `{(latency.get('total_decision_ms') or {}).get('p95')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Report: `{path}`",
            f"- Paper-style JSONL: `{payload.get('trade_log')}`",
            f"- Paper-style CSV: `{payload.get('trade_log_csv')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def cleanup_subscriptions(ib: Any, index_contracts: list[Any], option_subscriptions: list[tuple[Any, Any]]) -> None:
    if ib is None:
        return
    for contract, _ticker in option_subscriptions:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass
    for contract in index_contracts:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass


def expiry_from_contract_id(contract_id: str) -> str:
    parts = str(contract_id).split("-")
    return parts[1] if len(parts) >= 2 else ""


def strike_from_contract_id(contract_id: str, *, fallback: float) -> float:
    parts = str(contract_id).split("-")
    return finite(parts[2], fallback) if len(parts) >= 3 else float(fallback)


def finite(value: Any, default: Any = 0.0) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
