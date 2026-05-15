from __future__ import annotations

from v4.live.protocol101_risk_gate import Protocol101RiskConfig
from v4.scripts.run_protocol153_multi_contract_live_stack_compatibility import (
    compatibility_risk_config,
    decide,
    default_one_contract_guards,
    evaluate_compatibility,
    shadow_event,
    summarize,
)


def _row(quantity: int = 2) -> dict:
    return {
        "policy": "account_aware_sizer_v1",
        "trade_number": 1,
        "session": "2026-03-06",
        "segment": "q1_2026",
        "decision_time": "2026-03-06T15:00:00+00:00",
        "exit_time": "2026-03-06T15:05:00+00:00",
        "contract_id": "SPXW-20260306-06700.000-C",
        "side": "CALL",
        "score": 1.5,
        "threshold": 0.5,
        "score_margin": 1.0,
        "one_contract_premium": 1_000.0,
        "one_contract_pnl": 200.0,
        "quantity": quantity,
        "premium_exposure": 1_000.0 * quantity,
        "cash_before": 50_000.0,
        "cash_after": 50_400.0,
        "daily_pnl_before": 0.0,
        "realized_pnl": 200.0 * quantity,
        "skip_reason": "",
    }


def test_protocol153_config_uses_candidate_max_quantity() -> None:
    config = compatibility_risk_config([_row(1), _row(3)])

    assert isinstance(config, Protocol101RiskConfig)
    assert config.max_contracts_initial == 3
    assert config.daily_new_entry_stop_fraction_of_equity == 0.005


def test_protocol153_compatibility_passes_configured_multi_contract_rows() -> None:
    rows = [_row(1), _row(2), _row(3)]
    config = compatibility_risk_config(rows)
    risk_rows, schema_rows = evaluate_compatibility(rows, config=config)
    summary = summarize(risk_rows, schema_rows, config)

    assert summary["risk_failed_rows"] == 0
    assert summary["schema_failed_rows"] == 0
    assert summary["quantity_counts"] == {"1": 1, "2": 1, "3": 1}
    assert decide(summary) == "pass_multi_contract_live_stack_compatible_research_only"


def test_protocol153_rejects_schema_or_risk_failures() -> None:
    summary = {
        "rows": 1,
        "default_one_contract_schema_still_rejects_qty2": True,
        "default_one_contract_risk_gate_still_rejects_qty2": True,
        "risk_failed_rows": 1,
        "schema_failed_rows": 0,
    }

    assert decide(summary) == "reject_multi_contract_live_stack_risk_gate_failure"

    summary["risk_failed_rows"] = 0
    summary["schema_failed_rows"] = 1
    assert decide(summary) == "reject_multi_contract_live_stack_schema_failure"


def test_protocol153_default_guards_keep_one_contract_mode_protected() -> None:
    guards = default_one_contract_guards()

    assert guards["default_one_contract_schema_still_rejects_qty2"] is True
    assert guards["default_one_contract_risk_gate_still_rejects_qty2"] is True


def test_protocol153_shadow_event_has_no_order_intent() -> None:
    row = _row(2)
    event = shadow_event(
        row,
        contract={
            "contract_id": row["contract_id"],
            "root": "SPXW",
            "settlement_style": "PM",
            "quantity": 2,
            "multiplier": 100.0,
        },
        quote={
            "bid": 9.9,
            "ask": 10.0,
            "quote_age_ms": 0,
            "reference_ask": 10.0,
            "timestamp": row["decision_time"],
        },
        risk={"passed": True, "reason": "pass", "reasons": [], "premium_required": 2_000.0, "premium_cap": 8_000.0},
        max_contracts=3,
    )

    assert event["live_orders_enabled"] is False
    assert event["broker_endpoint_called"] is False
    assert "order_intent" not in event
    assert event["selected_contract"]["quantity"] == 2
