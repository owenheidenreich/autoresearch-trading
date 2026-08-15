"""Capture a bounded IBKR paper long-position stream for lifecycle parity.

This runner is deliberately separate from Protocol101's persistent runtime. It
places one explicitly specified SPXW paper BUY, records the broker-visible
in-position state for a bounded interval, then submits a guarded SELL to flatten.
It does not train a model or change the paper-default registry/runtime flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.ibkr_paper_executor import PaperExecutionConfig, execute_guarded_paper_order
from v4.live.ibkr_paper_guard import PaperOrderIntent, paper_order_permission, validate_order_intent
from v4.live.paper_trade_log import append_trade_event, make_intent_id, make_trade_log_event, trade_log_path


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/ibkr_long_position_stream_observation")
DEFAULT_TRADE_LOG_ROOT = Path("v4/logs/paper_trading")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expiry", required=True, help="YYYYMMDD")
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--right", choices=("C", "P"), required=True)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--capture-seconds", type=float, default=180.0)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--entry-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--exit-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--force-exit-offset", type=float, default=0.50)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=608)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def ticker_age_ms(ticker: Any, now: datetime) -> float | None:
    observed = getattr(ticker, "time", None)
    if not isinstance(observed, datetime):
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return max(0.0, (now - observed.astimezone(UTC)).total_seconds() * 1000.0)


def ticker_price(ticker: Any) -> float | None:
    for name in ("last", "close"):
        value = finite(getattr(ticker, name, None))
        if value is not None and value > 0:
            return value
    bid = finite(getattr(ticker, "bid", None))
    ask = finite(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return None


def greek_payload(greeks: Any) -> dict[str, Any] | None:
    if greeks is None:
        return None
    return {
        "implied_vol": finite(getattr(greeks, "impliedVol", None)),
        "delta": finite(getattr(greeks, "delta", None)),
        "gamma": finite(getattr(greeks, "gamma", None)),
        "vega": finite(getattr(greeks, "vega", None)),
        "theta": finite(getattr(greeks, "theta", None)),
        "option_price": finite(getattr(greeks, "optPrice", None)),
        "underlying_price": finite(getattr(greeks, "undPrice", None)),
    }


def quote_payload(ticker: Any, now: datetime) -> dict[str, Any]:
    return {
        "bid": finite(getattr(ticker, "bid", None)),
        "ask": finite(getattr(ticker, "ask", None)),
        "bid_size": finite(getattr(ticker, "bidSize", None)),
        "ask_size": finite(getattr(ticker, "askSize", None)),
        "last": finite(getattr(ticker, "last", None)),
        "last_size": finite(getattr(ticker, "lastSize", None)),
        "volume": finite(getattr(ticker, "volume", None)),
        "call_open_interest": finite(getattr(ticker, "callOpenInterest", None)),
        "put_open_interest": finite(getattr(ticker, "putOpenInterest", None)),
        "quote_age_ms": ticker_age_ms(ticker, now),
        "source_timestamp_utc": getattr(ticker, "time", None),
        "received_timestamp_utc": now,
        "market_data_type": getattr(ticker, "marketDataType", None),
        "bid_greeks": greek_payload(getattr(ticker, "bidGreeks", None)),
        "ask_greeks": greek_payload(getattr(ticker, "askGreeks", None)),
        "last_greeks": greek_payload(getattr(ticker, "lastGreeks", None)),
        "model_greeks": greek_payload(getattr(ticker, "modelGreeks", None)),
    }


def contract_payload(contract: Any) -> dict[str, Any]:
    return {
        "conid": int(getattr(contract, "conId", 0) or 0),
        "symbol": str(getattr(contract, "symbol", "")),
        "local_symbol": str(getattr(contract, "localSymbol", "")),
        "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
        "strike": finite(getattr(contract, "strike", None)),
        "right": str(getattr(contract, "right", "")),
        "trading_class": str(getattr(contract, "tradingClass", "")),
        "multiplier": str(getattr(contract, "multiplier", "")),
        "exchange": str(getattr(contract, "exchange", "")),
        "currency": str(getattr(contract, "currency", "")),
    }


def is_target(contract: Any, *, expiry: str, strike: float, right: str) -> bool:
    return (
        str(getattr(contract, "symbol", "")).upper() == "SPX"
        and str(getattr(contract, "tradingClass", "")).upper() == "SPXW"
        and str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8] == expiry
        and finite(getattr(contract, "strike", None)) == float(strike)
        and str(getattr(contract, "right", "")).upper() == right
    )


def target_positions(ib: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in ib.positions():
        if is_target(position.contract, expiry=args.expiry, strike=args.strike, right=args.right):
            rows.append(
                {
                    "contract": contract_payload(position.contract),
                    "quantity": finite(position.position),
                    "average_cost": finite(position.avgCost),
                }
            )
    return rows


def all_spxw_positions(ib: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in ib.positions():
        contract = position.contract
        if str(getattr(contract, "tradingClass", "")).upper() == "SPXW":
            rows.append(
                {
                    "contract": contract_payload(contract),
                    "quantity": finite(position.position),
                    "average_cost": finite(position.avgCost),
                }
            )
    return rows


def portfolio_payload(ib: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ib.portfolio():
        if is_target(item.contract, expiry=args.expiry, strike=args.strike, right=args.right):
            rows.append(
                {
                    "contract": contract_payload(item.contract),
                    "position": finite(item.position),
                    "market_price": finite(item.marketPrice),
                    "market_value": finite(item.marketValue),
                    "average_cost": finite(item.averageCost),
                    "unrealized_pnl": finite(item.unrealizedPNL),
                    "realized_pnl": finite(item.realizedPNL),
                }
            )
    return rows


def account_payload(ib: Any, account_id: str) -> dict[str, Any]:
    wanted = {
        "AvailableFunds",
        "BuyingPower",
        "NetLiquidation",
        "RealizedPnL",
        "UnrealizedPnL",
        "GrossPositionValue",
    }
    values: dict[str, Any] = {}
    for row in ib.accountValues(account_id):
        if row.tag in wanted and row.currency in ("USD", "BASE", ""):
            values[row.tag] = finite(row.value)
    return values


def fresh_guard_quote(ticker: Any, ib: Any, *, timeout_seconds: float = 12.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        now = datetime.now(UTC)
        quote = quote_payload(ticker, now)
        bid = finite(quote.get("bid"))
        ask = finite(quote.get("ask"))
        age = finite(quote.get("quote_age_ms"))
        if bid is not None and ask is not None and bid > 0 and ask >= bid and age is not None and age <= 1500:
            return quote
        ib.sleep(0.25)
    raise RuntimeError("fresh_exact_contract_nbbo_unavailable")


def execute(
    *,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    intent: PaperOrderIntent,
    account_id: str,
    quote: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    trade_uid: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    open_positions = len(all_spxw_positions(ib))
    guard_quote = {
        "bid": quote["bid"],
        "ask": quote["ask"],
        "reference_ask": quote["ask"],
        "quote_age_ms": quote["quote_age_ms"],
        "raw_quote_timestamp_utc": quote.get("source_timestamp_utc"),
        "received_timestamp_utc": quote.get("received_timestamp_utc"),
    }
    guard_context = {"context_age_ms": 0}
    permission = paper_order_permission(
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        account_id=account_id,
    )
    validation = validate_order_intent(
        intent,
        account_cash=float(args.paper_cash),
        open_positions=open_positions,
        quote=guard_quote,
        context=guard_context,
    )
    intent_id = make_intent_id(
        {
            "run_id": run_id,
            "trade_uid": trade_uid,
            "selected_contract": intent.__dict__,
            "order": intent.__dict__,
        }
    )
    block_reasons = [*permission.get("reasons", []), *validation.get("reasons", [])]
    risk_event = make_trade_log_event(
        event_type="risk_gate",
        run_id=run_id,
        mode="paper-submit",
        trade_uid=trade_uid,
        intent_id=intent_id,
        selected_contract={
            **intent.__dict__,
            "root": "SPXW",
            "settlement": "PM",
            **guard_quote,
        },
        order={"action": intent.action, "quantity": intent.quantity, "limit_price": intent.limit_price},
        account={
            "account_id_redacted": f"{account_id[:2]}***{account_id[-2:]}",
            "cash": float(args.paper_cash),
            "open_positions": open_positions,
        },
        market_snapshot={"option_nbbo": guard_quote, "context": guard_context},
        risk_gate={
            "passed": bool(permission.get("passed") and validation.get("passed")),
            "reason": "pass" if not block_reasons else ",".join(block_reasons),
            "guard_passed": bool(permission.get("passed") and validation.get("passed")),
            "guard_block_reasons": block_reasons,
            "permission": permission,
            "validation": validation,
        },
        artifact_ids={"protocol": "ibkr_long_position_stream_observation"},
        runtime_flag_digest="one_off_owner_authorized_observation_runtime_flag_unchanged",
        live_orders_enabled=True,
        source_script="run_ibkr_long_position_stream_observation",
    )
    append_trade_event(
        trade_log_path(root=args.trade_log_root, session=datetime.now(UTC).date().isoformat(), run_id=run_id),
        risk_event,
    )
    return execute_guarded_paper_order(
        ib=ib,
        option_cls=option_cls,
        order_cls=order_cls,
        intent=intent,
        account_id=account_id,
        account_cash=float(args.paper_cash),
        open_positions=open_positions,
        quote=guard_quote,
        context=guard_context,
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        dry_run=False,
        config=PaperExecutionConfig(),
        trade_log_root=args.trade_log_root,
        trade_log_run_id=run_id,
        trade_uid=trade_uid,
        artifact_ids={"protocol": "ibkr_long_position_stream_observation"},
        runtime_flag_digest="one_off_owner_authorized_observation_runtime_flag_unchanged",
        wait_for_fill_seconds=float(timeout_seconds),
        cancel_unfilled=True,
    )


def main() -> int:
    args = parse_args()
    now = datetime.now(NY)
    session = now.date().isoformat()
    expected_expiry = session.replace("-", "")
    if args.expiry != expected_expiry:
        raise SystemExit(f"expiry_must_match_today:{expected_expiry}")
    if args.quantity != 1:
        raise SystemExit("quantity_must_equal_one")
    if not (9 <= now.hour < 16):
        raise SystemExit("outside_regular_market_hours")

    from ib_insync import IB, Index, LimitOrder, Option

    run_id = args.run_id or f"ibkr_long_position_stream_{session}_{now.strftime('%H%M%S')}_et"
    out_dir = args.out_root / session / run_id
    stream_path = out_dir / "position_stream.jsonl"
    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    ib = IB()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "session": session,
        "status": "started",
        "real_money_trading": False,
        "paper_default_runtime_flag_changed": False,
        "model_training_performed": False,
        "stream_path": str(stream_path),
    }
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=8)
        accounts = list(ib.managedAccounts() or [])
        account_id = accounts[0] if accounts else ""
        if not account_id.startswith("DU"):
            raise RuntimeError("account_not_recognized_as_paper")
        if all_spxw_positions(ib):
            raise RuntimeError("paper_account_not_flat_at_start")

        requested = Option("SPX", args.expiry, args.strike, args.right, "SMART", currency="USD", tradingClass="SPXW")
        qualified = list(ib.qualifyContracts(requested) or [])
        if len(qualified) != 1:
            raise RuntimeError(f"exact_contract_resolution_count:{len(qualified)}")
        contract = qualified[0]
        if not is_target(contract, expiry=args.expiry, strike=args.strike, right=args.right):
            raise RuntimeError("qualified_contract_identity_mismatch")

        option_ticker = ib.reqMktData(contract, "100,101,104,106", False, False)
        spx_contract = (ib.qualifyContracts(Index("SPX", "CBOE", "USD")) or [Index("SPX", "CBOE", "USD")])[0]
        vix_contract = (ib.qualifyContracts(Index("VIX", "CBOE", "USD")) or [Index("VIX", "CBOE", "USD")])[0]
        spx_ticker = ib.reqMktData(spx_contract, "", False, False)
        vix_ticker = ib.reqMktData(vix_contract, "", False, False)
        ib.sleep(3.0)

        entry_quote = fresh_guard_quote(option_ticker, ib)
        if getattr(option_ticker, "marketDataType", None) != 1:
            raise RuntimeError("exact_contract_market_data_not_live")
        entry_intent = PaperOrderIntent(
            action="BUY",
            symbol="SPX",
            expiry=args.expiry,
            strike=args.strike,
            right=args.right,
            quantity=1,
            limit_price=float(entry_quote["ask"]),
        )
        summary.update(
            {
                "account_id_redacted": f"{account_id[:2]}***{account_id[-2:]}",
                "contract": contract_payload(contract),
                "entry_quote": entry_quote,
                "context_at_entry": {"spx": ticker_price(spx_ticker), "vix": ticker_price(vix_ticker)},
            }
        )
        print(json.dumps({"phase": "entry_submit", "contract": contract_payload(contract), "ask": entry_quote["ask"]}), flush=True)
        entry_result = execute(
            ib=ib,
            option_cls=Option,
            order_cls=LimitOrder,
            intent=entry_intent,
            account_id=account_id,
            quote=entry_quote,
            args=args,
            run_id=run_id,
            trade_uid=f"{run_id}_entry",
            timeout_seconds=args.entry_timeout_seconds,
        )
        summary["entry_result"] = entry_result
        if not (entry_result.get("fill_summary") or {}).get("filled"):
            summary["status"] = "entry_not_filled"
            write_json(summary_path, summary)
            print(json.dumps({"phase": "entry_not_filled", "summary": str(summary_path)}), flush=True)
            return 1

        print(json.dumps({"phase": "position_filled", "capture_seconds": args.capture_seconds}), flush=True)
        start = time.monotonic()
        sequence = 0
        while time.monotonic() - start < float(args.capture_seconds):
            ib.sleep(max(0.1, float(args.sample_seconds)))
            sequence += 1
            sample_now = datetime.now(UTC)
            append_jsonl(
                stream_path,
                {
                    "schema_version": "IBKRLongPositionStreamObservationV1",
                    "sequence": sequence,
                    "received_timestamp_utc": sample_now,
                    "elapsed_seconds": time.monotonic() - start,
                    "contract": contract_payload(contract),
                    "option_quote": quote_payload(option_ticker, sample_now),
                    "underlying": {"spx": ticker_price(spx_ticker), "vix": ticker_price(vix_ticker)},
                    "position": target_positions(ib, args),
                    "portfolio": portfolio_payload(ib, args),
                    "account": account_payload(ib, account_id),
                },
            )
            if sequence % max(1, int(30.0 / max(0.1, float(args.sample_seconds)))) == 0:
                print(json.dumps({"phase": "capturing", "samples": sequence}), flush=True)

        exit_results: list[dict[str, Any]] = []
        for attempt in range(2):
            exit_quote = fresh_guard_quote(option_ticker, ib)
            limit_price = max(0.01, float(exit_quote["bid"]) - (float(args.force_exit_offset) if attempt else 0.0))
            exit_intent = PaperOrderIntent(
                action="SELL",
                symbol="SPX",
                expiry=args.expiry,
                strike=args.strike,
                right=args.right,
                quantity=1,
                limit_price=limit_price,
            )
            print(json.dumps({"phase": "exit_submit", "attempt": attempt + 1, "limit": limit_price}), flush=True)
            result = execute(
                ib=ib,
                option_cls=Option,
                order_cls=LimitOrder,
                intent=exit_intent,
                account_id=account_id,
                quote=exit_quote,
                args=args,
                run_id=run_id,
                trade_uid=f"{run_id}_exit_{attempt + 1}",
                timeout_seconds=args.exit_timeout_seconds,
            )
            exit_results.append(result)
            if (result.get("fill_summary") or {}).get("filled"):
                break
            ib.sleep(1.0)

        ib.sleep(1.0)
        remaining = all_spxw_positions(ib)
        summary.update(
            {
                "status": "complete_flat" if not remaining else "blocked_open_position_remains",
                "samples": sequence,
                "capture_seconds": float(args.capture_seconds),
                "exit_results": exit_results,
                "remaining_spxw_positions": remaining,
                "account_after": account_payload(ib, account_id),
                "completed_at_utc": datetime.now(UTC),
            }
        )
        write_json(summary_path, summary)
        print(json.dumps({"phase": summary["status"], "samples": sequence, "summary": str(summary_path)}), flush=True)
        return 0 if not remaining else 2
    except Exception as exc:
        summary.update({"status": "blocked_exception", "error_type": type(exc).__name__, "error": str(exc), "completed_at_utc": datetime.now(UTC)})
        write_json(summary_path, summary)
        print(json.dumps({"phase": "blocked_exception", "error": str(exc), "summary": str(summary_path)}), flush=True)
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
