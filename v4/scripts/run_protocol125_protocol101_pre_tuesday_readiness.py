"""Protocol 125: pre-Tuesday Protocol101 replay and shadow readiness pack.

This script does not train, download data, call IBKR, or place orders. It
packages the current frozen Protocol101 paper-account replay into the exact
inspection surfaces needed before the next live no-order shadow session:

* replay realism summary around the $10,000 paper account
* execution-skepticism links back to Protocol114/118/123/124
* trade visual-inspection targets for trades.html
* no-order shadow JSONL event shape for Tuesday
* Tuesday promotion checklist
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CHARTS_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts")
DEFAULT_FALSIFICATION_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification"
)
DEFAULT_SHADOW_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_118_protocol101_shadow_rehearsal")
DEFAULT_ORDER_STATE_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_123_protocol101_order_state_rehearsal"
)
DEFAULT_LIVE_PARITY_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_124_protocol101_live_data_parity_checkpoint"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness")
DEFAULT_CHECKLIST = Path("v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
STARTING_CASH = 10_000.0
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS_DIR)
    parser.add_argument("--falsification-dir", type=Path, default=DEFAULT_FALSIFICATION_DIR)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--order-state-dir", type=Path, default=DEFAULT_ORDER_STATE_DIR)
    parser.add_argument("--live-parity-dir", type=Path, default=DEFAULT_LIVE_PARITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checklist.parent.mkdir(parents=True, exist_ok=True)

    trades = load_trades(args.charts_dir / "trades.csv", paper_seed=int(args.paper_seed))
    skipped = load_optional_csv(args.charts_dir / "skipped_trades.csv")
    paper_summary = load_json(args.charts_dir / "paper_account_summary.json", default=[])
    falsification = load_json(args.falsification_dir / "summary.json", default={})
    shadow = load_json(args.shadow_dir / "summary.json", default={})
    order_state = load_json(args.order_state_dir / "summary.json", default={})
    live_parity = load_json(args.live_parity_dir / "summary.json", default={})

    daily = daily_pnl_rows(trades, starting_cash=float(args.starting_cash))
    targets = visual_inspection_targets(trades, daily)
    events = build_shadow_event_rows(trades, starting_cash=float(args.starting_cash))
    schema = shadow_event_schema()
    verification = verify_shadow_events(events, starting_cash=float(args.starting_cash))

    write_csv(args.out_dir / "daily_pnl.csv", daily)
    write_csv(args.out_dir / "visual_inspection_targets.csv", targets)
    (args.out_dir / "protocol101_shadow_event_schema.json").write_text(json_dumps(schema))
    (args.out_dir / "protocol101_shadow_event_examples.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in events) + "\n"
    )
    write_checklist(args.checklist)

    payload = {
        "protocol": "125_protocol101_pre_tuesday_readiness",
        "decision": decide(verification, falsification, shadow, order_state, live_parity),
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "paper_seed": int(args.paper_seed),
        "starting_cash": float(args.starting_cash),
        "source_of_truth": str(args.charts_dir / "equity.html"),
        "trade_overlay": str(args.charts_dir / "trades.html"),
        "replay_realism": replay_realism_summary(trades, skipped, paper_summary),
        "execution_skepticism": execution_skepticism_summary(falsification, shadow, order_state, live_parity),
        "shadow_event_verification": verification,
        "visual_inspection": visual_summary(targets, daily),
        "paper_concentration": paper_concentration_summary(trades, daily),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "daily_pnl": str(args.out_dir / "daily_pnl.csv"),
            "visual_inspection_targets": str(args.out_dir / "visual_inspection_targets.csv"),
            "shadow_event_schema": str(args.out_dir / "protocol101_shadow_event_schema.json"),
            "shadow_event_examples": str(args.out_dir / "protocol101_shadow_event_examples.jsonl"),
            "tuesday_checklist": str(args.checklist),
        },
        "next_gate": (
            "On Tuesday, run Protocol101 no-order live shadow capture first. Compare emitted JSONL to "
            "protocol101_shadow_event_schema.json and keep paper orders disabled until live feature/quote parity passes."
        ),
    }
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload, targets, daily)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2))
    return 0


def load_trades(path: Path, *, paper_seed: int) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if "seed" in frame.columns:
        frame = frame[pd.to_numeric(frame["seed"], errors="coerce").astype("Int64") == int(paper_seed)].copy()
    if frame.empty:
        raise SystemExit(f"no trades for paper seed {paper_seed} in {path}")
    frame = frame.sort_values(["decision_time", "exit_time", "candidate_uid"]).reset_index(drop=True)
    return [clean_row(row) for row in frame.to_dict("records")]


def load_optional_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return [clean_row(row) for row in frame.to_dict("records")]


def load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def daily_pnl_rows(trades: list[dict[str, Any]], *, starting_cash: float) -> list[dict[str, Any]]:
    frame = pd.DataFrame(trades)
    if frame.empty:
        return []
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    cash = float(starting_cash)
    for session, group in frame.groupby("session", sort=True):
        pnl = float(group["pnl"].sum())
        cash += pnl
        wins = int((group["pnl"] >= 0).sum())
        rows.append(
            {
                "session": str(session),
                "trades": int(len(group)),
                "wins": wins,
                "win_rate": round(wins / len(group), 6) if len(group) else 0.0,
                "daily_pnl": round(pnl, 2),
                "ending_equity": round(cash, 2),
            }
        )
    return rows


def visual_inspection_targets(trades: list[dict[str, Any]], daily: list[dict[str, Any]], *, top_n: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame(trades)
    for column in ("pnl", "premium_paid", "paper_buying_power_pct_cash", "path_mae"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    targets: list[dict[str, Any]] = []
    targets.extend(target_rows(frame.sort_values("pnl", ascending=False).head(top_n), "top_winner"))
    targets.extend(target_rows(frame.sort_values("pnl", ascending=True).head(top_n), "worst_loser"))
    if "premium_paid" in frame.columns:
        targets.extend(target_rows(frame.sort_values("premium_paid", ascending=False).head(top_n), "highest_premium"))
    if "paper_buying_power_pct_cash" in frame.columns:
        targets.extend(
            target_rows(
                frame.sort_values("paper_buying_power_pct_cash", ascending=False).head(top_n),
                "highest_starting_cash_usage",
            )
        )
    if "path_mae" in frame.columns:
        targets.extend(target_rows(frame.sort_values("path_mae", ascending=True).head(top_n), "worst_intratrade_mae"))

    daily_frame = pd.DataFrame(daily)
    if not daily_frame.empty:
        daily_frame["daily_pnl"] = pd.to_numeric(daily_frame["daily_pnl"], errors="coerce")
        for category, sessions in (
            ("best_day_context", daily_frame.sort_values("daily_pnl", ascending=False)["session"].head(5).tolist()),
            ("worst_day_context", daily_frame.sort_values("daily_pnl", ascending=True)["session"].head(5).tolist()),
        ):
            day_trades = frame[frame["session"].isin(sessions)].copy()
            day_trades = day_trades.sort_values(["session", "decision_time", "trade_number"])
            targets.extend(target_rows(day_trades, category, max_rows=50))
    return targets


def target_rows(frame: pd.DataFrame, category: str, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    if max_rows is not None:
        frame = frame.head(max_rows)
    rows = []
    for rank, row in enumerate(frame.to_dict("records"), start=1):
        rows.append(
            {
                "category": category,
                "rank": rank,
                "trade_number": int(float(row.get("trade_number", 0) or 0)),
                "session": str(row.get("session", "")),
                "decision_time": str(row.get("decision_time", "")),
                "exit_time": str(row.get("exit_time", "")),
                "side": str(row.get("side", "")),
                "offset": finite_or_none(row.get("offset")),
                "contract_id": str(row.get("contract_id", "")),
                "premium_paid": finite_or_none(row.get("premium_paid")),
                "cash_before": finite_or_none(row.get("paper_cash_before")),
                "buying_power_pct_cash": finite_or_none(row.get("paper_buying_power_pct_cash")),
                "pnl": finite_or_none(row.get("pnl")),
                "path_mae": finite_or_none(row.get("path_mae")),
                "exit_reason": str(row.get("exit_reason", "")),
            }
        )
    return rows


def build_shadow_event_rows(trades: list[dict[str, Any]], *, starting_cash: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda row: (str(row["decision_time"]), str(row["candidate_uid"]))):
        rows.append(shadow_event(trade, event_type="entry_decision", starting_cash=starting_cash))
        rows.append(shadow_event(trade, event_type="exit_decision", starting_cash=starting_cash))
    return rows


def shadow_event(trade: dict[str, Any], *, event_type: str, starting_cash: float) -> dict[str, Any]:
    is_entry = event_type == "entry_decision"
    timestamp = str(trade["decision_time"] if is_entry else trade["exit_time"])
    bid = finite_or_none(trade.get("entry_bid" if is_entry else "exit_bid"))
    ask = finite_or_none(trade.get("entry_ask" if is_entry else "exit_ask"))
    quote_time = trade.get("entry_quote_time" if is_entry else "exit_quote_time") or timestamp
    premium = finite_or_none(trade.get("paper_premium")) or finite_or_none(trade.get("premium_paid")) or 0.0
    cash_before = finite_or_none(trade.get("paper_cash_before")) or starting_cash
    cash_after = finite_or_none(trade.get("paper_cash_after"))
    open_positions_after = 1 if is_entry else 0
    reference_price = ask if is_entry else bid
    side = "BUY" if is_entry else "SELL"
    action = "enter_long_option" if is_entry else "exit_or_flat"
    return {
        "schema_version": "protocol101_shadow_v1",
        "protocol_id": "protocol101",
        "event_type": event_type,
        "timestamp": timestamp,
        "session": str(trade["session"]),
        "paper_seed": int(float(trade.get("seed", 1) or 1)),
        "replay_mode": "historical_replay_no_order",
        "broker_endpoint_called": False,
        "live_orders_enabled": False,
        "market_snapshot": {
            "underlying": {
                "symbol": "SPX",
                "price": finite_or_none(trade.get("entry_spx" if is_entry else "exit_spx")),
                "timestamp": timestamp,
                "source": "historical_spx_1m",
            },
            "option_nbbo": {
                "contract_id": str(trade["contract_id"]),
                "bid": bid,
                "ask": ask,
                "bid_size": finite_or_none(trade.get("entry_bid_size")) if is_entry else None,
                "ask_size": finite_or_none(trade.get("entry_ask_size")) if is_entry else None,
                "timestamp": str(quote_time),
                "quote_gap_seconds": finite_or_none(trade.get("quote_gap_seconds")) if is_entry else 0.0,
                "source": "historical_replay_quote",
            },
            "context": {
                "stage": str(trade.get("stage", "")),
                "segment": str(trade.get("segment", "")),
                "source_protocol": str(trade.get("source_protocol", "")),
            },
        },
        "model_decision": {
            "action": action,
            "score": finite_or_none(trade.get("score")),
            "threshold": finite_or_none(trade.get("threshold")),
            "entry_reason": str(trade.get("exit_reason", "")),
        },
        "selected_contract": {
            "contract_id": str(trade["contract_id"]),
            "root": "SPXW",
            "settlement_style": "PM",
            "right": str(trade.get("right", "")),
            "side": str(trade.get("side", "")),
            "offset": finite_or_none(trade.get("offset")),
            "quantity": 1,
            "multiplier": CONTRACT_MULTIPLIER,
        },
        "intended_order": {
            "mode": "no_order_shadow",
            "will_submit_to_broker": False,
            "side": side,
            "quantity": 1,
            "reference_price": reference_price,
            "order_type_if_enabled_later": "marketable_limit_reference_only",
        },
        "fill_assumption": {
            "historical_fill_only": True,
            "entry_fill_price": finite_or_none(trade.get("entry_ask")),
            "exit_fill_price": finite_or_none(trade.get("exit_bid")),
            "pricing_rule": "ask_entry_bid_exit",
            "real_broker_fill": False,
            "reported_trade_pnl": finite_or_none(trade.get("pnl")),
        },
        "exit_plan": {
            "exit_reason": str(trade.get("exit_reason", "")),
            "exit_time": str(trade.get("exit_time", "")),
            "mandatory_flat_before_close": True,
        },
        "account_state": {
            "starting_cash": float(starting_cash),
            "cash_before_trade": cash_before,
            "cash_reserved_after_entry": cash_before - premium if is_entry else 0.0,
            "cash_after_trade": cash_after if not is_entry else None,
            "premium_required": premium,
            "buying_power_used": finite_or_none(trade.get("paper_buying_power_used")) or premium,
            "buying_power_pct_cash": finite_or_none(trade.get("paper_buying_power_pct_cash")),
            "affordable": premium <= cash_before + 1e-9,
            "open_positions_after_event": open_positions_after,
            "equity_after_event": cash_before if is_entry else cash_after,
        },
        "audit": {
            "trade_number": int(float(trade.get("trade_number", 0) or 0)),
            "candidate_uid": str(trade.get("candidate_uid", "")),
            "historical_replay_only": True,
            "paid_data_downloaded": False,
            "live_order_placed": False,
        },
    }


def shadow_event_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "protocol_id",
        "event_type",
        "timestamp",
        "session",
        "paper_seed",
        "replay_mode",
        "broker_endpoint_called",
        "live_orders_enabled",
        "market_snapshot",
        "model_decision",
        "selected_contract",
        "intended_order",
        "fill_assumption",
        "exit_plan",
        "account_state",
        "audit",
    ]
    return {
        "schema": "protocol101_shadow_v1",
        "description": (
            "Tuesday live shadow rows must use these top-level fields. Live capture changes data sources to "
            "fresh broker/feed sources, but order mode remains no_order_shadow until explicit paper-order approval."
        ),
        "required_top_level_fields": required,
        "required_nested_fields": {
            "market_snapshot": ["underlying", "option_nbbo", "context"],
            "model_decision": ["action", "score", "threshold", "entry_reason"],
            "selected_contract": ["contract_id", "root", "settlement_style", "right", "side", "offset", "quantity", "multiplier"],
            "intended_order": ["mode", "will_submit_to_broker", "side", "quantity", "reference_price"],
            "fill_assumption": ["historical_fill_only", "pricing_rule", "real_broker_fill", "reported_trade_pnl"],
            "exit_plan": ["exit_reason", "exit_time", "mandatory_flat_before_close"],
            "account_state": [
                "starting_cash",
                "cash_before_trade",
                "premium_required",
                "buying_power_used",
                "affordable",
                "open_positions_after_event",
                "equity_after_event",
            ],
            "audit": ["trade_number", "candidate_uid", "historical_replay_only", "paid_data_downloaded", "live_order_placed"],
        },
        "hard_rules": [
            "broker_endpoint_called must be false",
            "live_orders_enabled must be false",
            "intended_order.mode must be no_order_shadow",
            "selected_contract.root must be SPXW",
            "selected_contract.quantity must be 1",
            "account_state.starting_cash must be 10000.0",
        ],
    }


def verify_shadow_events(events: list[dict[str, Any]], *, starting_cash: float) -> dict[str, Any]:
    schema = shadow_event_schema()
    required = set(schema["required_top_level_fields"])
    errors: list[str] = []
    open_positions = 0
    max_open = 0
    for index, event in enumerate(sorted(events, key=lambda row: (row["timestamp"], 0 if row["event_type"] == "exit_decision" else 1))):
        missing = sorted(required - set(event))
        if missing:
            errors.append(f"event {index} missing top-level fields: {missing}")
        if event.get("broker_endpoint_called") is not False or event.get("live_orders_enabled") is not False:
            errors.append(f"event {index} is not no-order")
        if event.get("intended_order", {}).get("mode") != "no_order_shadow":
            errors.append(f"event {index} has non-shadow order mode")
        if event.get("selected_contract", {}).get("root") != "SPXW":
            errors.append(f"event {index} selected non-SPXW root")
        if int(event.get("selected_contract", {}).get("quantity", 0)) != 1:
            errors.append(f"event {index} selected quantity other than one")
        if abs(float(event.get("account_state", {}).get("starting_cash", 0.0)) - float(starting_cash)) > 1e-9:
            errors.append(f"event {index} has wrong starting cash")
        if not bool(event.get("account_state", {}).get("affordable", False)):
            errors.append(f"event {index} is unaffordable")
        if event.get("event_type") == "entry_decision":
            open_positions += 1
        elif event.get("event_type") == "exit_decision":
            open_positions -= 1
        if open_positions < 0:
            errors.append(f"event {index} exits without an open position")
            open_positions = 0
        max_open = max(max_open, open_positions)
    if open_positions != 0:
        errors.append("stream ends with an open position")
    return {
        "events": len(events),
        "entry_events": sum(1 for event in events if event.get("event_type") == "entry_decision"),
        "exit_events": sum(1 for event in events if event.get("event_type") == "exit_decision"),
        "max_open_positions": max_open,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }


def replay_realism_summary(
    trades: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    paper_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    pnl = sum(float(row.get("pnl") or 0.0) for row in trades)
    premiums = [float(row["premium_paid"]) for row in trades if finite_or_none(row.get("premium_paid")) is not None]
    maes = [float(row["path_mae"]) for row in trades if finite_or_none(row.get("path_mae")) is not None]
    summary = paper_summary[0] if paper_summary else {}
    return {
        "trades": len(trades),
        "skipped_trades": len(skipped),
        "unaffordable_skips": sum(1 for row in skipped if row.get("paper_skip_reason") == "insufficient_cash"),
        "starting_cash": float(summary.get("starting_equity", STARTING_CASH)),
        "ending_equity": float(summary.get("ending_equity", STARTING_CASH + pnl)),
        "total_pnl": round(pnl, 2),
        "max_drawdown": summary.get("max_drawdown"),
        "worst_day_pnl": summary.get("worst_day_pnl"),
        "max_known_buying_power": max(premiums) if premiums else None,
        "premium_coverage": len(premiums) / len(trades) if trades else 0.0,
        "worst_intratrade_mae": min(maes) if maes else None,
        "path_coverage": len(maes) / len(trades) if trades else 0.0,
    }


def execution_skepticism_summary(
    falsification: dict[str, Any],
    shadow: dict[str, Any],
    order_state: dict[str, Any],
    live_parity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol114_decision": falsification.get("decision"),
        "protocol114_next_hypothesis": falsification.get("next_hypothesis"),
        "protocol114_concentration": falsification.get("concentration", {}),
        "protocol114_delay_stress": falsification.get("delay_stress_summary", {}),
        "protocol118_decision": shadow.get("decision"),
        "protocol118_shadow_parity_status": shadow.get("shadow_parity", {}).get("status"),
        "protocol118_shadow_paper_status": shadow.get("shadow_paper", {}).get("status"),
        "protocol123_decision": order_state.get("decision"),
        "protocol123_stress_results": order_state.get("stress_results", []),
        "protocol124_decision": live_parity.get("decision"),
        "live_data_blocker": live_parity.get("next_gate") or live_parity.get("blocker"),
    }


def visual_summary(targets: list[dict[str, Any]], daily: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for row in targets:
        category = str(row["category"])
        by_category[category] = by_category.get(category, 0) + 1
    daily_sorted = sorted(daily, key=lambda row: float(row["daily_pnl"]), reverse=True)
    return {
        "target_rows": len(targets),
        "target_categories": by_category,
        "best_day": daily_sorted[0] if daily_sorted else None,
        "worst_day": daily_sorted[-1] if daily_sorted else None,
    }


def paper_concentration_summary(trades: list[dict[str, Any]], daily: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = sorted((float(row.get("pnl") or 0.0) for row in trades), reverse=True)
    total = sum(pnls)
    gross_profit = sum(value for value in pnls if value > 0)
    daily_pnls = sorted((float(row["daily_pnl"]) for row in daily), reverse=True)
    return {
        "total_pnl": round(total, 2),
        "gross_profit": round(gross_profit, 2),
        "top_5_trades_pnl": round(sum(pnls[:5]), 2),
        "top_10_trades_pnl": round(sum(pnls[:10]), 2),
        "top_20_trades_pnl": round(sum(pnls[:20]), 2),
        "top_5_trades_share_net": sum(pnls[:5]) / total if total else None,
        "top_10_trades_share_net": sum(pnls[:10]) / total if total else None,
        "top_20_trades_share_net": sum(pnls[:20]) / total if total else None,
        "top_20_trades_share_gross": sum(pnls[:20]) / gross_profit if gross_profit else None,
        "top_day_share_net": daily_pnls[0] / total if daily_pnls and total else None,
        "top_5_days_share_net": sum(daily_pnls[:5]) / total if daily_pnls and total else None,
        "single_trade_majority": bool(pnls and total > 0 and pnls[0] > total * 0.5),
        "top20_majority": bool(total > 0 and sum(pnls[:20]) > total * 0.5),
    }


def decide(
    verification: dict[str, Any],
    falsification: dict[str, Any],
    shadow: dict[str, Any],
    order_state: dict[str, Any],
    live_parity: dict[str, Any],
) -> str:
    if verification.get("status") != "pass":
        return "blocked_shadow_event_format_failed"
    if order_state.get("decision") != "pass_10000_order_state_rehearsal_live_data_pending":
        return "blocked_order_state_rehearsal_failed"
    if shadow.get("decision") != "pass_historical_no_order_shadow_rehearsal_live_capture_next":
        return "blocked_historical_shadow_rehearsal_failed"
    if falsification.get("decision") == "reject_edge_hypothesis":
        return "blocked_protocol101_rejected_by_falsification"
    if live_parity.get("decision") == "pass_live_data_parity":
        return "ready_for_no_order_live_shadow_recheck"
    return "ready_for_tuesday_no_order_live_shadow_only"


def write_report(path: Path, payload: dict[str, Any], targets: list[dict[str, Any]], daily: list[dict[str, Any]]) -> None:
    realism = payload["replay_realism"]
    skepticism = payload["execution_skepticism"]
    visual = payload["visual_inspection"]
    concentration = payload["paper_concentration"]
    lines = [
        "# Protocol 125: Protocol101 Pre-Tuesday Readiness Pack",
        "",
        "No paid market data was downloaded. No live broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source-of-truth equity: `{payload['source_of_truth']}`",
        f"- Trade overlay: `{payload['trade_overlay']}`",
        f"- Tuesday checklist: `{payload['outputs']['tuesday_checklist']}`",
        "",
        "## Replay Realism",
        "",
        f"- Paper starting cash: `${realism['starting_cash']:,.0f}`",
        f"- Trades: `{realism['trades']}`",
        f"- Skipped trades: `{realism['skipped_trades']}`",
        f"- Unaffordable skips: `{realism['unaffordable_skips']}`",
        f"- Ending equity: `${realism['ending_equity']:,.0f}`",
        f"- Replay PnL: `${realism['total_pnl']:,.0f}`",
        f"- Max drawdown: `${float(realism['max_drawdown'] or 0):,.0f}`",
        f"- Worst day PnL: `${float(realism['worst_day_pnl'] or 0):,.0f}`",
        f"- Max known buying power: `${float(realism['max_known_buying_power'] or 0):,.0f}`",
        f"- Premium coverage: `{float(realism['premium_coverage']) * 100:.1f}%`",
        f"- Worst intratrade MAE: `${float(realism['worst_intratrade_mae'] or 0):,.0f}`",
        f"- Path coverage: `{float(realism['path_coverage']) * 100:.1f}%`",
        f"- Top 20 trades share of net PnL: `{pct(concentration['top_20_trades_share_net'])}`",
        f"- Top day share of net PnL: `{pct(concentration['top_day_share_net'])}`",
        f"- Single-trade majority: `{concentration['single_trade_majority']}`",
        f"- Top-20 majority: `{concentration['top20_majority']}`",
        "",
        "## Execution Skepticism",
        "",
        f"- Protocol 114 decision: `{skepticism['protocol114_decision']}`",
        f"- Protocol 114 next hypothesis: `{skepticism['protocol114_next_hypothesis']}`",
        f"- Protocol 118 historical shadow decision: `{skepticism['protocol118_decision']}`",
        f"- Protocol 123 order-state decision: `{skepticism['protocol123_decision']}`",
        f"- Protocol 124 live-data parity decision: `{skepticism['protocol124_decision']}`",
        "",
        "## Shadow Event Contract",
        "",
        f"- Event schema: `{payload['outputs']['shadow_event_schema']}`",
        f"- JSONL examples: `{payload['outputs']['shadow_event_examples']}`",
        f"- Events: `{payload['shadow_event_verification']['events']}`",
        f"- Max open positions: `{payload['shadow_event_verification']['max_open_positions']}`",
        f"- Verification status: `{payload['shadow_event_verification']['status']}`",
        "",
        "## Visual Inspection Targets",
        "",
        f"- Target file: `{payload['outputs']['visual_inspection_targets']}`",
        f"- Daily PnL file: `{payload['outputs']['daily_pnl']}`",
        f"- Best day: `{visual['best_day']['session'] if visual['best_day'] else 'n/a'}` `${visual['best_day']['daily_pnl'] if visual['best_day'] else 0:,.0f}`",
        f"- Worst day: `{visual['worst_day']['session'] if visual['worst_day'] else 'n/a'}` `${visual['worst_day']['daily_pnl'] if visual['worst_day'] else 0:,.0f}`",
        "",
        "Use `trades.html` Trade # jump with these rows first:",
        "",
        "| category | rank | trade # | session | side | premium | pnl | exit |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in targets[:40]:
        lines.append(
            "| "
            f"{row['category']} | {row['rank']} | {row['trade_number']} | {row['session']} | {row['side']} | "
            f"${float(row['premium_paid'] or 0):,.0f} | ${float(row['pnl'] or 0):,.0f} | {row['exit_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_checklist(path: Path) -> None:
    lines = [
        "# Protocol 101 Tuesday Live-Shadow Checklist",
        "",
        "This checklist is for no-order live shadow only. The $500 in IBKR is the access/data reserve. The paper account baseline is $10,000. Do not place paper orders until the no-order live-data parity gate passes and the user explicitly approves paper order testing.",
        "",
        "## 1. Pre-Open Guardrails",
        "",
        "- Confirm `live_orders_enabled = false` and `broker_endpoint_called = false` in the router config.",
        "- Confirm no paid historical download job is running.",
        "- Confirm the frozen Protocol101 artifacts and Protocol051/054/081 fallback artifacts are unchanged.",
        "- Confirm the JSONL output path is new for the session and will not overwrite historical artifacts.",
        "",
        "## 2. Market-Data Connection",
        "",
        "- Connect to IB Gateway/TWS.",
        "- Verify fresh SPX context, VIX context, and SPXW option NBBO are available.",
        "- Reject AM-settled `SPX` contracts; accept PM-settled `SPXW` only.",
        "- Record quote/context freshness and root/settlement metadata in every row.",
        "",
        "## 3. No-Order Shadow Capture",
        "",
        "- Emit JSONL rows matching `protocol101_shadow_v1`.",
        "- Include market snapshot, model decision, selected contract, intended no-order order reference, fill assumption, exit plan, and account state.",
        "- Keep `intended_order.mode = no_order_shadow` and `will_submit_to_broker = false`.",
        "- Run long enough to capture both entry opportunities and flat/no-entry periods.",
        "",
        "## 4. Live Parity Review",
        "",
        "- Validate feature columns against the frozen Protocol101 feature schema.",
        "- Validate live SPX/VIX/option quote timestamps are fresh.",
        "- Validate one contract max, no overlaps, flat-before-close behavior, and $10,000 affordability checks.",
        "- Compare live feature distributions against historical replay distributions before trusting decisions.",
        "",
        "## 5. Paper-Order Rehearsal Gate",
        "",
        "- Only after no-order live parity passes, rerun order-state rehearsal on captured live shadow rows.",
        "- Require max concurrent positions = 1, all order intents affordable, and no stale quotes.",
        "- Ask for explicit user approval before enabling any paper order endpoint.",
        "",
        "## Blockers",
        "",
        "- Missing SPX/VIX context.",
        "- Missing OPRA/SPXW NBBO.",
        "- Stale quote/context rows.",
        "- Any row with `live_orders_enabled = true` before approval.",
        "- Any selected contract with root `SPX` instead of `SPXW`.",
        "- Any unaffordable one-contract trade under the $10,000 paper-account baseline.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 125 Protocol101 Pre-Tuesday Readiness Pack"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Packaged frozen Protocol101 replay realism, execution-skepticism status, no-order shadow event contract, visual inspection targets, and the Tuesday live-shadow checklist.
Reason: IBKR cash settlement blocks live paper trading, but the project still needed the Tuesday no-order shadow run to be precise instead of improvised.
Data Used: Existing Protocol101/107/113/114/118/123/124 artifacts only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Shadow event verification {payload['shadow_event_verification']['status']} with {payload['shadow_event_verification']['events']} events and max_open_positions={payload['shadow_event_verification']['max_open_positions']}. Report: {payload['outputs']['report']}
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    existing = path.read_text() if path.exists() else ""
    if marker not in existing:
        path.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def pct(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.1f}%"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n"


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
