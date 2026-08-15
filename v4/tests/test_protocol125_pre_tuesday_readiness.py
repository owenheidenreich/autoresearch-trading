from __future__ import annotations

from v4.scripts.run_protocol125_protocol101_pre_tuesday_readiness import (
    build_shadow_event_rows,
    daily_pnl_rows,
    verify_shadow_events,
    visual_inspection_targets,
)


def _trade(uid: str, trade_number: int, decision: str, exit_time: str, pnl: float, premium: float) -> dict:
    cash_before = 10_000.0 + (trade_number - 1) * 100.0
    return {
        "seed": 1,
        "trade_number": trade_number,
        "candidate_uid": uid,
        "session": decision[:10],
        "decision_time": decision,
        "exit_time": exit_time,
        "decision_ms": trade_number * 60_000,
        "exit_ms": trade_number * 60_000 + 30_000,
        "stage": "test",
        "segment": "q1_2026",
        "source_protocol": "test",
        "right": "C",
        "side": "CALL",
        "offset": -25.0,
        "contract_id": "SPXW-20260306-06700.000-C",
        "entry_spx": 6700.0,
        "exit_spx": 6705.0,
        "entry_quote_time": decision,
        "exit_quote_time": exit_time,
        "entry_bid": 10.0,
        "entry_ask": premium / 100.0,
        "exit_bid": premium / 100.0 + pnl / 100.0,
        "exit_ask": premium / 100.0 + pnl / 100.0 + 0.2,
        "entry_bid_size": 10,
        "entry_ask_size": 8,
        "premium_paid": premium,
        "paper_premium": premium,
        "paper_cash_before": cash_before,
        "paper_cash_after": cash_before + pnl,
        "paper_buying_power_used": premium,
        "paper_buying_power_pct_cash": premium / cash_before,
        "quote_gap_seconds": 0.0,
        "path_mae": min(0.0, pnl),
        "pnl": pnl,
        "score": 1.2,
        "threshold": 0.5,
        "exit_reason": "target" if pnl >= 0 else "hard_stop",
    }


def test_daily_pnl_rows_tracks_ending_equity() -> None:
    trades = [
        _trade("a", 1, "2026-03-06T15:00:00+00:00", "2026-03-06T15:05:00+00:00", 100.0, 1_000.0),
        _trade("b", 2, "2026-03-06T15:10:00+00:00", "2026-03-06T15:15:00+00:00", -50.0, 1_000.0),
        _trade("c", 3, "2026-03-09T15:00:00+00:00", "2026-03-09T15:05:00+00:00", 200.0, 1_000.0),
    ]

    rows = daily_pnl_rows(trades, starting_cash=10_000.0)

    assert rows[0]["session"] == "2026-03-06"
    assert rows[0]["daily_pnl"] == 50.0
    assert rows[0]["ending_equity"] == 10_050.0
    assert rows[1]["ending_equity"] == 10_250.0


def test_shadow_event_rows_are_no_order_and_verified() -> None:
    trades = [
        _trade("a", 1, "2026-03-06T15:00:00+00:00", "2026-03-06T15:05:00+00:00", 100.0, 1_000.0),
    ]

    events = build_shadow_event_rows(trades, starting_cash=10_000.0)
    verification = verify_shadow_events(events, starting_cash=10_000.0)

    assert len(events) == 2
    assert events[0]["market_snapshot"]["option_nbbo"]["contract_id"].startswith("SPXW-")
    assert events[0]["intended_order"]["mode"] == "no_order_shadow"
    assert events[0]["broker_endpoint_called"] is False
    assert events[0]["account_state"]["affordable"] is True
    assert verification["status"] == "pass"
    assert verification["max_open_positions"] == 1


def test_visual_inspection_targets_include_risk_categories() -> None:
    trades = [
        _trade("winner", 1, "2026-03-06T15:00:00+00:00", "2026-03-06T15:05:00+00:00", 500.0, 1_000.0),
        _trade("loser", 2, "2026-03-07T15:00:00+00:00", "2026-03-07T15:05:00+00:00", -400.0, 2_000.0),
    ]
    daily = daily_pnl_rows(trades, starting_cash=10_000.0)

    targets = visual_inspection_targets(trades, daily, top_n=1)
    categories = {row["category"] for row in targets}

    assert "top_winner" in categories
    assert "worst_loser" in categories
    assert "highest_premium" in categories
    assert "highest_starting_cash_usage" in categories
