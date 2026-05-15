"""Protocol 153: live-stack compatibility for the multi-contract challenger.

This is not a new model and not live/paper approval. It checks whether the
current account-aware sizing candidate can be represented by the same risk-gate
and no-order shadow schema that will protect the future paper bot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_risk_gate import AccountState, Protocol101RiskConfig, evaluate_entry_risk_gate
from v4.live.protocol101_shadow_schema import validate_shadow_event


DEFAULT_CANDIDATE_ROWS = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/candidate_trade_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_153_protocol101_multi_contract_live_stack_compatibility"
)
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = load_candidate_rows(args.candidate_rows)
    config = compatibility_risk_config(candidate_rows)
    risk_rows, schema_rows = evaluate_compatibility(candidate_rows, config=config)
    summary = summarize(risk_rows, schema_rows, config)
    decision = decide(summary)
    payload = {
        "protocol": "153_protocol101_multi_contract_live_stack_compatibility",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "paper_orders": False,
        "model_training": False,
        "protocol101_frozen": True,
        "challenger": "account_aware_sizer_v1",
        "source_candidate_rows": str(args.candidate_rows),
        "risk_config": config.__dict__,
        "summary": summary,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "risk_gate_rows": str(args.out_dir / "risk_gate_rows.csv"),
            "schema_rows": str(args.out_dir / "schema_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(risk_rows).to_csv(args.out_dir / "risk_gate_rows.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(args.out_dir / "schema_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"no candidate rows found in {path}")
    frame = frame.sort_values(["decision_time", "trade_number"]).reset_index(drop=True)
    return [clean(row) for row in frame.to_dict("records")]


def compatibility_risk_config(rows: list[dict[str, Any]]) -> Protocol101RiskConfig:
    max_qty = max([int(number(row.get("quantity")) or 0) for row in rows] or [1])
    return Protocol101RiskConfig(
        max_contracts_initial=max(1, max_qty),
        max_premium_dollars=8_000.0,
        max_premium_fraction_of_equity=0.40,
        daily_new_entry_stop_loss=-1_500.0,
        daily_new_entry_stop_fraction_of_equity=0.005,
        max_option_quote_age_ms=1_500,
        max_context_age_ms=5_000,
        max_entry_ask_move=0.25,
    )


def evaluate_compatibility(
    rows: list[dict[str, Any]],
    *,
    config: Protocol101RiskConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    risk_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    max_contracts = int(config.max_contracts_initial)
    for row in rows:
        quantity = int(number(row.get("quantity")) or 0)
        if quantity <= 0:
            continue
        contract = contract_payload(row, quantity)
        quote = quote_payload(row)
        account = AccountState(
            cash=float(number(row.get("cash_before")) or 0.0),
            equity=float(number(row.get("cash_before")) or 0.0),
            realized_daily_pnl=float(number(row.get("daily_pnl_before")) or 0.0),
            open_positions=0,
        )
        risk = evaluate_entry_risk_gate(
            contract=contract,
            quote=quote,
            context={"context_age_ms": 0},
            account=account,
            config=config,
        )
        event = shadow_event(row, contract=contract, quote=quote, risk=risk, max_contracts=max_contracts)
        schema = validate_shadow_event(event, max_contracts_per_position=max_contracts)
        risk_rows.append(
            {
                "trade_number": int(row["trade_number"]),
                "session": row["session"],
                "segment": row.get("segment"),
                "decision_time": row["decision_time"],
                "contract_id": row["contract_id"],
                "side": row.get("side"),
                "quantity": quantity,
                "cash_before": row.get("cash_before"),
                "daily_pnl_before": row.get("daily_pnl_before"),
                "premium_exposure": row.get("premium_exposure"),
                "risk_passed": bool(risk["passed"]),
                "risk_reason": risk["reason"],
                "premium_required": risk["premium_required"],
                "premium_cap": risk["premium_cap"],
                "daily_stop_loss": risk["daily_stop_loss"],
            }
        )
        schema_rows.append(
            {
                "trade_number": int(row["trade_number"]),
                "session": row["session"],
                "segment": row.get("segment"),
                "decision_time": row["decision_time"],
                "contract_id": row["contract_id"],
                "quantity": quantity,
                "schema_status": schema.status,
                "schema_errors": "; ".join(schema.errors),
                "schema_warnings": "; ".join(schema.warnings),
            }
        )
    return risk_rows, schema_rows


def contract_payload(row: dict[str, Any], quantity: int) -> dict[str, Any]:
    contract_id = str(row["contract_id"])
    return {
        "contract_id": contract_id,
        "root": contract_id.split("-", 1)[0],
        "settlement_style": "PM",
        "quantity": quantity,
        "multiplier": CONTRACT_MULTIPLIER,
    }


def quote_payload(row: dict[str, Any]) -> dict[str, Any]:
    ask = (number(row.get("one_contract_premium")) or 0.0) / CONTRACT_MULTIPLIER
    bid = max(0.01, ask - min(0.10, ask * 0.50))
    return {
        "bid": round(bid, 6),
        "ask": round(ask, 6),
        "bid_size": 10,
        "ask_size": 10,
        "quote_age_ms": 0,
        "reference_ask": round(ask, 6),
        "timestamp": str(row["decision_time"]),
    }


def shadow_event(
    row: dict[str, Any],
    *,
    contract: dict[str, Any],
    quote: dict[str, Any],
    risk: dict[str, Any],
    max_contracts: int,
) -> dict[str, Any]:
    decision_time = str(row["decision_time"])
    strike = strike_from_contract_id(row.get("contract_id"))
    return {
        "schema_version": "protocol101_shadow_v2",
        "protocol_id": "protocol101",
        "event_type": "model_decision",
        "timestamp": decision_time,
        "session": str(row["session"]),
        "live_orders_enabled": False,
        "broker_endpoint_called": False,
        "market_snapshot": {
            "underlying": {
                "spx": strike or 1.0,
                "vix": 20.0,
                "spx_timestamp": decision_time,
                "vix_timestamp": decision_time,
                "source": "historical_replay_compatibility_fixture",
            },
            "option_nbbo": quote,
        },
        "model_decision": {
            "score": number(row.get("score")),
            "threshold": number(row.get("threshold")),
            "features": {
                "score_margin": number(row.get("score_margin")),
                "side": row.get("side"),
                "one_contract_premium": number(row.get("one_contract_premium")),
            },
        },
        "selected_action": "enter",
        "selected_contract": {
            **contract,
            "right": right_from_contract_id(row.get("contract_id")),
        },
        "risk_gate": {
            "passed": bool(risk["passed"]),
            "reason": risk["reason"],
            "reasons": list(risk["reasons"]),
            "premium_required": risk["premium_required"],
            "premium_cap": risk["premium_cap"],
        },
        "paper_account_state": {
            "starting_cash": 10_000.0,
            "cash": number(row.get("cash_before")),
            "equity": number(row.get("cash_before")),
            "daily_pnl": number(row.get("daily_pnl_before")),
            "open_positions": 0,
            "max_concurrent_positions": 1,
            "max_contracts_per_position": max_contracts,
        },
    }


def summarize(
    risk_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    config: Protocol101RiskConfig,
) -> dict[str, Any]:
    risk_failed = [row for row in risk_rows if not bool(row["risk_passed"])]
    schema_failed = [row for row in schema_rows if row["schema_status"] != "pass"]
    quantity_counts: dict[str, int] = {}
    for row in risk_rows:
        qty = str(int(row["quantity"]))
        quantity_counts[qty] = quantity_counts.get(qty, 0) + 1
    risk_reason_counts: dict[str, int] = {}
    for row in risk_failed:
        for reason in str(row.get("risk_reason") or "").split(","):
            if reason:
                risk_reason_counts[reason] = risk_reason_counts.get(reason, 0) + 1
    schema_error_counts: dict[str, int] = {}
    for row in schema_failed:
        for error in str(row.get("schema_errors") or "").split("; "):
            if error:
                schema_error_counts[error] = schema_error_counts.get(error, 0) + 1
    default_guards = default_one_contract_guards()
    return {
        "rows": len(risk_rows),
        "max_contracts_configured": int(config.max_contracts_initial),
        "quantity_counts": dict(sorted(quantity_counts.items())),
        "risk_passed_rows": len(risk_rows) - len(risk_failed),
        "risk_failed_rows": len(risk_failed),
        "risk_reason_counts": dict(sorted(risk_reason_counts.items())),
        "schema_passed_rows": len(schema_rows) - len(schema_failed),
        "schema_failed_rows": len(schema_failed),
        "schema_error_counts": dict(sorted(schema_error_counts.items())),
        "default_one_contract_schema_still_rejects_qty2": default_guards[
            "default_one_contract_schema_still_rejects_qty2"
        ],
        "default_one_contract_risk_gate_still_rejects_qty2": default_guards[
            "default_one_contract_risk_gate_still_rejects_qty2"
        ],
    }


def default_one_contract_guards() -> dict[str, bool]:
    row = {
        "trade_number": 1,
        "session": "2026-03-06",
        "decision_time": "2026-03-06T15:00:00+00:00",
        "contract_id": "SPXW-20260306-06700.000-C",
        "side": "CALL",
        "score": 1.0,
        "threshold": 0.0,
        "score_margin": 1.0,
        "one_contract_premium": 1000.0,
        "cash_before": 50_000.0,
        "daily_pnl_before": 0.0,
    }
    contract = contract_payload(row, 2)
    quote = quote_payload(row)
    risk = evaluate_entry_risk_gate(
        contract=contract,
        quote=quote,
        context={"context_age_ms": 0},
        account=AccountState(cash=50_000.0, equity=50_000.0),
        config=Protocol101RiskConfig(),
    )
    event = shadow_event(row, contract=contract, quote=quote, risk=risk, max_contracts=1)
    schema = validate_shadow_event(event)
    return {
        "default_one_contract_schema_still_rejects_qty2": schema.status == "fail",
        "default_one_contract_risk_gate_still_rejects_qty2": (
            risk["passed"] is False and "position_size_exceeds_max_contracts" in risk["reasons"]
        ),
    }


def decide(summary: dict[str, Any]) -> str:
    if int(summary.get("rows", 0)) <= 0:
        return "blocked_no_multi_contract_candidate_rows"
    if not bool(summary.get("default_one_contract_schema_still_rejects_qty2")):
        return "reject_default_schema_no_longer_protects_one_contract_mode"
    if not bool(summary.get("default_one_contract_risk_gate_still_rejects_qty2")):
        return "reject_default_risk_gate_no_longer_protects_one_contract_mode"
    if int(summary.get("risk_failed_rows", 0)) > 0:
        return "reject_multi_contract_live_stack_risk_gate_failure"
    if int(summary.get("schema_failed_rows", 0)) > 0:
        return "reject_multi_contract_live_stack_schema_failure"
    return "pass_multi_contract_live_stack_compatible_research_only"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Keep one-contract as the operational paper default. The multi-contract challenger can now be replayed "
            "through live shadow logs once one-contract paper trading is stable."
        )
    if decision.startswith("blocked_"):
        return "Generate candidate sizing rows before judging live-stack compatibility."
    return "Count this as a failed infrastructure hypothesis and inspect the failure rows before changing sizing logic."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Protocol 153: Multi-Contract Live-Stack Compatibility",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries and exits remain frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Candidate: `{payload['challenger']}`",
        f"- Rows checked: `{summary['rows']}`",
        f"- Configured max contracts: `{summary['max_contracts_configured']}`",
        f"- Quantity counts: `{summary['quantity_counts']}`",
        f"- Risk failures: `{summary['risk_failed_rows']}`",
        f"- Schema failures: `{summary['schema_failed_rows']}`",
        f"- Default one-contract schema still rejects qty=2: `{summary['default_one_contract_schema_still_rejects_qty2']}`",
        f"- Default one-contract risk gate still rejects qty=2: `{summary['default_one_contract_risk_gate_still_rejects_qty2']}`",
        "",
        "## Interpretation",
        "",
        (
            "This protocol only proves the research sizing candidate can be represented by the protected live/paper "
            "interfaces. It does not promote multi-contract paper trading. The one-contract path remains the operational baseline."
        ),
        "",
        "## Outputs",
        "",
        f"- Risk rows: `{payload['outputs']['risk_gate_rows']}`",
        f"- Schema rows: `{payload['outputs']['schema_rows']}`",
        f"- Summary: `{payload['outputs']['summary']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
        "",
    ]
    path.write_text("\n".join(lines))


def clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def strike_from_contract_id(value: Any) -> float | None:
    parts = str(value or "").split("-")
    if len(parts) < 3:
        return None
    return number(parts[2])


def right_from_contract_id(value: Any) -> str:
    parts = str(value or "").split("-")
    return parts[-1] if parts else ""


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
