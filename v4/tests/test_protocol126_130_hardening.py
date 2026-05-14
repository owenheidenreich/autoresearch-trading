from __future__ import annotations

from v4.live.protocol101_risk_gate import AccountState, Protocol101RiskConfig, evaluate_entry_risk_gate
from v4.live.protocol101_shadow_schema import validate_shadow_event, validate_shadow_stream
from v4.scripts.run_protocol126_timing_fragility_hardening import decide as decide_protocol126
from v4.scripts.run_protocol127_live_shadow_schema_hardening import build_examples
from v4.scripts.run_protocol128_paper_risk_gate import (
    decide as decide_protocol128,
    evaluate_replay_risk_gates,
    max_concurrent_positions_from_times,
    replay_invariants,
)
from v4.scripts.run_protocol129_offline_position_sizing import (
    max_contracts_for_state,
    simulate_sizing,
)
from v4.scripts.run_protocol130_tuesday_no_order_live_shadow_runbook import readiness_check


def _trade(
    *,
    trade_number: int = 1,
    contract_id: str = "SPXW-20260306-06700.000-C",
    decision_time: str = "2026-03-06T15:00:00+00:00",
    exit_time: str = "2026-03-06T15:05:00+00:00",
    premium: float = 1_000.0,
    cash_before: float = 10_000.0,
    pnl: float = 100.0,
) -> dict:
    entry_ask = premium / 100.0
    return {
        "seed": 1,
        "trade_number": trade_number,
        "candidate_uid": f"t{trade_number}",
        "session": decision_time[:10],
        "decision_time": decision_time,
        "exit_time": exit_time,
        "contract_id": contract_id,
        "side": "CALL",
        "right": "C",
        "offset": 0.0,
        "entry_spx": 6700.0,
        "exit_spx": 6705.0,
        "entry_bid": entry_ask - 0.10,
        "entry_ask": entry_ask,
        "exit_bid": entry_ask + pnl / 100.0,
        "exit_ask": entry_ask + pnl / 100.0 + 0.10,
        "entry_bid_size": 10,
        "entry_ask_size": 10,
        "premium_paid": premium,
        "quote_gap_seconds": 0.0,
        "paper_cash_before": cash_before,
        "paper_cash_after": cash_before + pnl,
        "paper_premium": premium,
        "pnl": pnl,
        "score": 1.0,
        "threshold": 0.5,
        "exit_reason": "test_exit",
    }


def test_protocol126_blocks_incomplete_timing_coverage() -> None:
    decision = decide_protocol126(
        [
            {"split": "q3_2025", "delay_seconds": 1, "coverage": 0.90, "delayed_pnl": 100.0},
            {"split": "q3_2025", "delay_seconds": 5, "coverage": 1.00, "delayed_pnl": 100.0},
        ]
    )

    assert decision == "blocked_incomplete_timing_coverage"


def test_protocol127_shadow_schema_rejects_order_intent_and_bad_quotes() -> None:
    event = build_examples(_trade())[2]
    assert validate_shadow_event(event).status == "pass"

    event["order_intent"] = {"action": "BUY"}
    event["market_snapshot"]["option_nbbo"]["ask"] = event["market_snapshot"]["option_nbbo"]["bid"] - 0.01
    result = validate_shadow_event(event)

    assert result.status == "fail"
    assert any("order" in error for error in result.errors)
    assert any("ask must be >= bid" in error for error in result.errors)


def test_protocol127_shadow_stream_requires_block_reason() -> None:
    event = build_examples(_trade())[3]
    event["risk_gate"]["reason"] = ""
    event["risk_gate"]["reasons"] = []

    result = validate_shadow_stream([event])

    assert result["status"] == "fail"
    assert any("blocked action requires risk_gate.reason" in error for error in result["errors"])


def test_protocol128_risk_gate_rejects_wrong_root_stale_quote_and_unaffordable_trade() -> None:
    result = evaluate_entry_risk_gate(
        contract={"contract_id": "SPX-20260306-06700.000-C", "root": "SPX", "settlement_style": "AM", "quantity": 1},
        quote={"bid": 10.0, "ask": 20.0, "quote_age_ms": 2_000, "reference_ask": 10.0},
        context={"context_age_ms": 0},
        account=AccountState(cash=1_000.0, equity=1_000.0),
        config=Protocol101RiskConfig(),
    )

    assert result["passed"] is False
    assert "wrong_root" in result["reasons"]
    assert "wrong_settlement" in result["reasons"]
    assert "stale_option_quote" in result["reasons"]
    assert "insufficient_cash" in result["reasons"]


def test_protocol128_replay_invariants_pass_clean_one_contract_path() -> None:
    trades = [
        _trade(trade_number=1, decision_time="2026-03-06T15:00:00+00:00", exit_time="2026-03-06T15:05:00+00:00"),
        _trade(trade_number=2, decision_time="2026-03-06T15:06:00+00:00", exit_time="2026-03-06T15:10:00+00:00", cash_before=10_100.0),
    ]
    config = Protocol101RiskConfig()
    rows, summary = evaluate_replay_risk_gates(trades, config=config)
    invariants = replay_invariants(trades, config=config)

    assert len(rows) == 2
    assert summary["hard_block_rows"] == 0
    assert invariants["zero_unaffordable_trades"] is True
    assert invariants["max_concurrent_positions"] == 1
    assert decide_protocol128(invariants, summary) == "pass_paper_risk_gate_overlay_ready_for_live_shadow"


def test_protocol128_detects_overlap_from_timestamps() -> None:
    assert max_concurrent_positions_from_times(
        [
            _trade(trade_number=1, decision_time="2026-03-06T15:00:00+00:00", exit_time="2026-03-06T15:10:00+00:00"),
            _trade(trade_number=2, decision_time="2026-03-06T15:05:00+00:00", exit_time="2026-03-06T15:15:00+00:00"),
        ]
    ) == 2


def test_protocol129_contract_sizing_requires_drawdown_and_recent_positive() -> None:
    assert max_contracts_for_state(equity=19_999.0, drawdown_pct=0.0, recent_pnls=[100.0]) == 1
    assert max_contracts_for_state(equity=25_000.0, drawdown_pct=0.04, recent_pnls=[100.0]) == 2
    assert max_contracts_for_state(equity=25_000.0, drawdown_pct=0.06, recent_pnls=[100.0]) == 1
    assert max_contracts_for_state(equity=60_000.0, drawdown_pct=0.04, recent_pnls=[100.0]) == 3
    assert max_contracts_for_state(equity=60_000.0, drawdown_pct=0.04, recent_pnls=[-100.0]) == 2


def test_protocol129_sizing_cap_skips_when_premium_exceeds_20pct_equity() -> None:
    trades = [
        _trade(trade_number=1, premium=3_000.0, pnl=500.0),
        _trade(trade_number=2, premium=1_000.0, pnl=100.0, decision_time="2026-03-06T15:10:00+00:00", exit_time="2026-03-06T15:20:00+00:00"),
    ]

    result = simulate_sizing(trades, mode="one_contract_20pct_cap")

    assert result["summary"]["taken_trades"] == 1
    assert result["summary"]["skip_counts"]["premium_exposure_cap"] == 1


def test_protocol130_readiness_allows_timing_and_scaling_cautions_for_no_order_shadow() -> None:
    summaries = {
        "protocol126": {"present": True, "decision": "blocked_incomplete_timing_coverage"},
        "protocol127": {"present": True, "decision": "pass_schema_hardening_ready_for_live_capture"},
        "protocol128": {"present": True, "decision": "pass_paper_risk_gate_overlay_ready_for_live_shadow"},
        "protocol129": {"present": True, "decision": "reject_scaling_loss_clustering_worse"},
    }

    result = readiness_check(summaries)

    assert result["ready"] is True
    assert result["blocked_inputs"] == []
    assert set(result["caution_inputs"]) == {"protocol126", "protocol129"}
