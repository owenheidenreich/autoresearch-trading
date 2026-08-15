"""RUNTIME_PREMIUM_BLEND_PAPER_RUNTIME_SHELL_V1.

Historically Protocol244. This runner exercises the premium-leaning challenger
through a paper-trading style event/log contract without using IBKR. It converts
historical rows into live-style quote/context objects, builds challenger
features through the same adapter intended for live runtime, emits monitorable
paper-log events, and validates hypothetical order intents. Broker endpoints
remain disabled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.full_action_challenger_adapter import FullActionHistoryState, build_full_action_candidates_from_quotes
from v4.live.ibkr_paper_guard import PaperOrderGuardConfig, PaperOrderIntent, validate_order_intent
from v4.live.paper_trade_log import (
    append_trade_event,
    export_trade_log_csv,
    load_trade_log,
    make_trade_log_event,
    trade_log_path,
    validate_trade_log,
)
from v4.scripts.run_protocol217_full_action_history_runtime_parity import (
    DEFAULT_DATASET,
    build_feature_tensor,
    choose_candidate,
    live_safe_candidate_mask,
    load_artifact,
    run_model,
)
from v4.scripts.run_protocol241_premium_blend_runtime_parity import DEFAULT_ARTIFACT_MANIFEST


ROLE_LABEL = "RUNTIME_PREMIUM_BLEND_PAPER_RUNTIME_SHELL_V1"
HISTORICAL_ID = "Protocol244"
PROTOCOL_ID = "challenger_premium_leaning_blended_utility_v1"
CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_244_premium_blend_paper_runtime_shell")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_OUT_DIR / "paper_logs")
    parser.add_argument("--session", default="2026-05-20")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-events", type=int, default=240)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--order-quantity", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"premium_blend_runtime_shell_{args.session}"
    trade_log = trade_log_path(root=args.trade_log_root, session=str(args.session), run_id=run_id)
    csv_log = trade_log.with_suffix(".csv")

    artifact = load_artifact(args.artifact_manifest)
    frame = load_session_frame(args.dataset, session=str(args.session), feature_columns=artifact["feature_columns"])
    result = run_runtime_shell(
        frame,
        artifact=artifact,
        trade_log=trade_log,
        session=str(args.session),
        run_id=run_id,
        max_events=int(args.max_events),
        paper_cash=float(args.paper_cash),
        order_quantity=int(args.order_quantity),
        max_open_positions=int(args.max_open_positions),
    )
    rows = load_trade_log(trade_log)
    validation = validate_trade_log(rows)
    csv_summary = export_trade_log_csv(trade_log, csv_log)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / paper-style no-broker shell",
        "changes_paper_default": False,
        "candidate_label": CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "other_baseline_label": "Protocol101 persistent paper-trader log/order contract",
        "data_used": {
            "dataset": str(args.dataset),
            "artifact_manifest": str(args.artifact_manifest),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "session": str(args.session),
        "run_id": run_id,
        "paper_cash": float(args.paper_cash),
        "events_replayed": result["events_replayed"],
        "decisions": result["decisions"],
        "enter_intents": result["enter_intents"],
        "risk_gate_passed": result["risk_gate_passed"],
        "risk_gate_blocked": result["risk_gate_blocked"],
        "selected_trade_rows": result["selected_trade_rows"],
        "event_counts": validation["event_counts"],
        "trade_log_validation": validation,
        "trade_log": str(trade_log),
        "trade_log_csv": str(csv_log),
        "trade_log_csv_summary": csv_summary,
        "decision": decide(result, validation),
        "next_experiment": (
            "Use this same shell shape in a broker-connected challenger no-order runner. "
            "Only switch paper default after live surface breadth/freshness and log validation pass automatically."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(result["selected_trade_rows"]).to_csv(args.out_dir / "selected_intents.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md"), "trade_log": str(trade_log)}, indent=2, sort_keys=True))
    return 0


def load_session_frame(path: Path, *, session: str, feature_columns: list[str]) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                "session",
                "decision_dt",
                "candidate_uid",
                "contract_id",
                "root",
                "settlement_style",
                "right",
                "offset",
                "entry_bid",
                "entry_ask",
                "entry_mid",
                "entry_spread",
                "entry_bid_size",
                "entry_ask_size",
                "entry_underlying_price",
                "entry_iv",
                "entry_delta",
                "entry_gamma",
                "entry_theta",
                "entry_premium",
                "candidate_exit_dt",
                "candidate_pnl",
                "surface_edge",
                "edge",
                *feature_columns,
            ]
        )
    )
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["session"].astype(str).eq(str(session))].copy()
    if frame.empty:
        raise ValueError(f"no rows found for session {session}")
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    return frame.sort_values(["decision_dt", "contract_id"]).reset_index(drop=True)


def run_runtime_shell(
    frame: pd.DataFrame,
    *,
    artifact: dict[str, Any],
    trade_log: Path,
    session: str,
    run_id: str,
    max_events: int,
    paper_cash: float,
    order_quantity: int,
    max_open_positions: int,
) -> dict[str, Any]:
    history = FullActionHistoryState()
    equity = float(paper_cash)
    open_until: pd.Timestamp | None = None
    decisions = 0
    enter_intents = 0
    risk_passed = 0
    risk_blocked = 0
    selected_rows: list[dict[str, Any]] = []
    append_runtime_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        paper_cash=paper_cash,
        reason="premium_blend_runtime_shell_started",
        model_decision={"action": "startup", "candidate": CHALLENGER_LABEL},
    )
    for event_idx, (_, event) in enumerate(frame.groupby("decision_dt", sort=True)):
        if event_idx >= int(max_events):
            break
        decision_dt = pd.Timestamp(event["decision_dt"].iloc[0])
        if open_until is not None and decision_dt < open_until:
            continue
        candidates = live_style_candidates(event, history=history)
        if candidates.empty:
            continue
        merged = candidates.merge(
            event[["contract_id", "candidate_exit_dt", "candidate_pnl"]],
            on="contract_id",
            how="left",
            validate="one_to_one",
        )
        account = {
            "starting_cash": float(paper_cash),
            "cash_available": float(equity),
            "account_equity": float(equity),
            "open_position_count": 0,
            "max_concurrent_positions": int(max_open_positions),
            "max_contracts": int(order_quantity),
        }
        mask = live_safe_candidate_mask(merged, account_state=account)
        tensors = build_feature_tensor(merged, mask, artifact["feature_columns"], artifact["scaler"])
        output = run_model(artifact["model"], tensors)
        candidate_idx, score = choose_candidate(output, mask)
        selected_action = "wait"
        selected_contract: dict[str, Any] = {}
        order: dict[str, Any] = {}
        risk_gate = {"passed": True, "reason": "no_entry_intent"}
        selected_row = None
        if bool(mask.any()) and candidate_idx is not None and float(score) >= float(artifact["threshold"]):
            selected_row = merged.iloc[int(candidate_idx)]
            selected_action = "enter"
            enter_intents += 1
            selected_contract = selected_contract_payload(selected_row, quantity=order_quantity)
            order = order_payload(selected_row, quantity=order_quantity)
            risk_gate = validate_shell_order_intent(selected_row, paper_cash=equity, quantity=order_quantity)
            risk_passed += int(bool(risk_gate["passed"]))
            risk_blocked += int(not bool(risk_gate["passed"]))
            if risk_gate["passed"]:
                pnl = _finite_float(selected_row.get("candidate_pnl"), 0.0)
                equity += pnl
                open_until = pd.Timestamp(selected_row["candidate_exit_dt"])
                selected_rows.append(
                    {
                        "session": session,
                        "decision_time": decision_dt.isoformat(),
                        "exit_time": pd.Timestamp(selected_row["candidate_exit_dt"]).isoformat(),
                        "contract_id": str(selected_row["contract_id"]),
                        "right": str(selected_row["right"]),
                        "offset": _finite_float(selected_row.get("offset"), 0.0),
                        "entry_ask": _finite_float(selected_row.get("entry_ask"), 0.0),
                        "entry_premium": _finite_float(selected_row.get("entry_premium"), 0.0),
                        "score": float(score),
                        "threshold": float(artifact["threshold"]),
                        "pnl": pnl,
                        "equity_after": float(equity),
                    }
                )
        append_runtime_event(
            trade_log,
            event_type="market_snapshot",
            session=session,
            run_id=run_id,
            paper_cash=paper_cash,
            reason="historical_replay_as_live_style_snapshot",
            timestamp=decision_dt,
            market_snapshot=market_snapshot_payload(merged),
            account=account_payload(equity, paper_cash),
        )
        append_runtime_event(
            trade_log,
            event_type="candidate_set",
            session=session,
            run_id=run_id,
            paper_cash=paper_cash,
            reason="challenger_candidate_set_built",
            timestamp=decision_dt,
            market_snapshot=market_snapshot_payload(merged),
            model_decision={"action": "candidate_set", "candidate_count": int(len(merged)), "valid_candidate_count": int(mask.sum())},
            account=account_payload(equity, paper_cash),
            extra={"candidate_sample": candidate_sample(merged, mask), "historical_replay_proxy": True},
        )
        append_runtime_event(
            trade_log,
            event_type="model_decision",
            session=session,
            run_id=run_id,
            paper_cash=paper_cash,
            reason=selected_action,
            timestamp=decision_dt,
            selected_contract=selected_contract,
            order=order,
            market_snapshot=market_snapshot_payload(merged, selected_row),
            model_decision={
                "action": selected_action,
                "score": float(score),
                "threshold": float(artifact["threshold"]),
                "candidate_index": None if candidate_idx is None else int(candidate_idx),
                "candidate_label": CHALLENGER_LABEL,
            },
            account=account_payload(equity, paper_cash),
        )
        append_runtime_event(
            trade_log,
            event_type="risk_gate",
            session=session,
            run_id=run_id,
            paper_cash=paper_cash,
            reason=str(risk_gate.get("reason") or "pass"),
            timestamp=decision_dt,
            selected_contract=selected_contract,
            order=order,
            market_snapshot=market_snapshot_payload(merged, selected_row),
            model_decision={"action": selected_action, "score": float(score), "threshold": float(artifact["threshold"])},
            risk_gate=risk_gate,
            account=account_payload(equity, paper_cash),
        )
        append_runtime_event(
            trade_log,
            event_type="paper_account_state",
            session=session,
            run_id=run_id,
            paper_cash=paper_cash,
            reason="paper_account_state",
            timestamp=decision_dt,
            account=account_payload(equity, paper_cash),
        )
        history.update(candidates, decision_time=decision_dt)
        decisions += 1
    return {
        "events_replayed": int(decisions),
        "decisions": int(decisions),
        "enter_intents": int(enter_intents),
        "risk_gate_passed": int(risk_passed),
        "risk_gate_blocked": int(risk_blocked),
        "selected_trade_rows": selected_rows,
    }


def live_style_candidates(event: pd.DataFrame, *, history: FullActionHistoryState) -> pd.DataFrame:
    first = event.iloc[0]
    market = {column: _finite_float(first.get(column), 0.0) for column in event.columns if str(column).startswith("market_")}
    quotes = [
        {
            "contract_id": str(row["contract_id"]),
            "root": str(row.get("root") or "SPXW"),
            "settlement_style": str(row.get("settlement_style") or "PM"),
            "strike": _strike_from_contract_id(str(row["contract_id"]), fallback=_finite_float(row["entry_underlying_price"], 0.0) + _finite_float(row.get("offset"), 0.0)),
            "right": str(row["right"]),
            "bid": _finite_float(row["entry_bid"], 0.0),
            "ask": _finite_float(row["entry_ask"], 0.0),
            "mid": _finite_float(row["entry_mid"], 0.0),
            "spread": _finite_float(row["entry_spread"], 0.0),
            "bid_size": _finite_float(row["entry_bid_size"], 0.0),
            "ask_size": _finite_float(row["entry_ask_size"], 0.0),
            "underlying_price": _finite_float(row["entry_underlying_price"], 0.0),
            "iv": _finite_float(row["entry_iv"], 0.0),
            "delta": _finite_float(row["entry_delta"], 0.0),
            "gamma": _finite_float(row["entry_gamma"], 0.0),
            "theta": _finite_float(row["entry_theta"], 0.0),
            "surface_edge": _finite_float(row.get("surface_edge", row.get("edge", 0.0)), 0.0),
            "edge": _finite_float(row.get("edge", row.get("surface_edge", 0.0)), 0.0),
        }
        for _, row in event.iterrows()
    ]
    return build_full_action_candidates_from_quotes(
        decision_time=first["decision_dt"],
        spx=_finite_float(first["entry_underlying_price"], 0.0),
        vix=_finite_float(first.get("market_vix_close"), 0.0),
        option_quotes=quotes,
        history=history,
        market_features=market,
        session=str(first["session"]),
    )


def validate_shell_order_intent(row: pd.Series, *, paper_cash: float, quantity: int) -> dict[str, Any]:
    intent = PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry=str(pd.Timestamp(row["decision_dt"]).strftime("%Y%m%d")),
        strike=_strike_from_contract_id(str(row["contract_id"]), fallback=_finite_float(row["entry_underlying_price"], 0.0) + _finite_float(row.get("offset"), 0.0)),
        right=str(row["right"]),
        quantity=int(quantity),
        limit_price=_finite_float(row["entry_ask"], 0.0),
        trading_class="SPXW",
    )
    result = validate_order_intent(
        intent,
        account_cash=float(paper_cash),
        open_positions=0,
        quote={
            "bid": _finite_float(row["entry_bid"], 0.0),
            "ask": _finite_float(row["entry_ask"], 0.0),
            "reference_ask": _finite_float(row["entry_ask"], 0.0),
            "quote_age_ms": 0.0,
        },
        context={"context_age_ms": 0.0},
        config=PaperOrderGuardConfig(),
    )
    return {
        "passed": bool(result["passed"]),
        "reason": result["reason"],
        "validation": result,
        "broker_endpoint_called": False,
    }


def append_runtime_event(
    path: Path,
    *,
    event_type: str,
    session: str,
    run_id: str,
    paper_cash: float,
    reason: str,
    timestamp: Any | None = None,
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
        timestamp=timestamp,
        session=session,
        run_id=run_id,
        mode="paper-runtime-shell",
        selected_contract=selected_contract or {},
        order=order or {},
        account=account or account_payload(paper_cash, paper_cash),
        market_snapshot=market_snapshot or {"underlying": {}, "option_nbbo": {}, "context": {}},
        model_decision=model_decision or {"action": "wait", "reason": reason},
        risk_gate=risk_gate or {"passed": True, "reason": reason},
        broker_order_endpoint_called=False,
        paper_trading=True,
        real_money_trading=False,
        protocol_id=PROTOCOL_ID,
        extra={"reason": reason, "live_orders_enabled": False, **(extra or {})},
    )
    append_trade_event(path, row)


def selected_contract_payload(row: pd.Series, *, quantity: int) -> dict[str, Any]:
    return {
        "contract_id": str(row["contract_id"]),
        "symbol": "SPX",
        "root": "SPXW",
        "trading_class": "SPXW",
        "settlement_style": "PM",
        "expiry": pd.Timestamp(row["decision_dt"]).strftime("%Y%m%d"),
        "strike": _strike_from_contract_id(str(row["contract_id"]), fallback=_finite_float(row["entry_underlying_price"], 0.0) + _finite_float(row.get("offset"), 0.0)),
        "right": str(row["right"]),
        "quantity": int(quantity),
        "exchange": "SMART",
        "currency": "USD",
    }


def order_payload(row: pd.Series, *, quantity: int) -> dict[str, Any]:
    ask = _finite_float(row["entry_ask"], 0.0)
    return {"action": "BUY", "quantity": int(quantity), "limit_price": ask, "premium_required": ask * CONTRACT_MULTIPLIER * int(quantity)}


def market_snapshot_payload(frame: pd.DataFrame, selected_row: pd.Series | None = None) -> dict[str, Any]:
    first = frame.iloc[0]
    row = selected_row if selected_row is not None else first
    return {
        "underlying": {"spx": _finite_float(first.get("entry_underlying_price"), 0.0), "vix": _finite_float(first.get("market_vix_close"), 0.0)},
        "option_nbbo": {
            "candidate_count": int(len(frame)),
            "bid": _finite_float(row.get("entry_bid"), 0.0),
            "ask": _finite_float(row.get("entry_ask"), 0.0),
            "bid_size": _finite_float(row.get("entry_bid_size"), 0.0),
            "ask_size": _finite_float(row.get("entry_ask_size"), 0.0),
            "quote_age_ms": 0.0,
        },
        "context": {"source": "historical_replay_as_live_style", "context_age_ms": 0.0},
    }


def account_payload(equity: float, starting_cash: float) -> dict[str, Any]:
    return {
        "account_id_redacted": None,
        "starting_cash": float(starting_cash),
        "cash": float(equity),
        "equity": float(equity),
        "realized_daily_pnl": float(equity - starting_cash),
        "open_positions": 0,
    }


def candidate_sample(frame: pd.DataFrame, mask: np.ndarray, *, limit: int = 5) -> list[dict[str, Any]]:
    out = []
    for idx, (_, row) in enumerate(frame.head(limit).iterrows()):
        out.append(
            {
                "contract_id": str(row["contract_id"]),
                "right": str(row["right"]),
                "offset": _finite_float(row.get("offset"), 0.0),
                "entry_ask": _finite_float(row.get("entry_ask"), 0.0),
                "entry_premium": _finite_float(row.get("entry_premium"), 0.0),
                "valid": bool(mask[idx]) if idx < len(mask) else False,
            }
        )
    return out


def decide(result: dict[str, Any], validation: dict[str, Any]) -> str:
    if validation.get("status") != "pass":
        return "blocked_paper_runtime_log_validation_failed"
    if int(validation.get("broker_order_endpoint_called_rows", 0)) != 0:
        return "blocked_broker_endpoint_was_called"
    if int(result.get("decisions", 0)) <= 0:
        return "blocked_no_runtime_decisions_emitted"
    return "paper_runtime_shell_ready_no_broker_protocol101_default_unchanged"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']['dataset']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Runtime Shell Summary",
        "",
        f"- Session: `{payload['session']}`",
        f"- Events replayed: `{payload['events_replayed']}`",
        f"- Enter intents: `{payload['enter_intents']}`",
        f"- Risk-gate passed intents: `{payload['risk_gate_passed']}`",
        f"- Risk-gate blocked intents: `{payload['risk_gate_blocked']}`",
        f"- Trade-log validation: `{payload['trade_log_validation']['status']}`",
        f"- Event counts: `{payload['event_counts']}`",
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Report: `{path}`",
        f"- Paper-style JSONL: `{payload['trade_log']}`",
        f"- Paper-style CSV: `{payload['trade_log_csv']}`",
        f"- Selected intents: `{path.parent / 'selected_intents.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def _strike_from_contract_id(contract_id: str, *, fallback: float) -> float:
    parts = contract_id.split("-")
    if len(parts) >= 3:
        return _finite_float(parts[2], fallback)
    return fallback


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


if __name__ == "__main__":
    raise SystemExit(main())
