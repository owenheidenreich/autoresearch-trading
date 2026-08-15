from __future__ import annotations

from v4.scripts.run_protocol122_protocol101_capital_realism import (
    _premium_summary,
    _single_summary,
    decide,
)


def _trade(uid: str, premium: float, pnl: float, decision_ms: int = 0, exit_ms: int = 60_000) -> dict:
    return {
        "seed": 1,
        "candidate_uid": uid,
        "decision_ms": decision_ms,
        "exit_ms": exit_ms,
        "decision_time": "2026-03-06T15:00:00+00:00",
        "exit_time": "2026-03-06T15:01:00+00:00",
        "session": "2026-03-06",
        "entry_ask": premium / 100.0,
        "premium_paid": premium,
        "pnl": pnl,
    }


def test_protocol122_premium_summary_reports_affordable_fractions() -> None:
    summary = _premium_summary(
        [
            _trade("cheap", premium=400.0, pnl=10.0),
            _trade("mid", premium=1_200.0, pnl=20.0),
            _trade("large", premium=3_000.0, pnl=30.0),
        ]
    )

    assert summary["count"] == 3
    assert summary["median"] == 1_200.0
    assert abs(summary["pct_at_or_below_500"] - 1 / 3) < 1e-5
    assert abs(summary["pct_at_or_below_2500"] - 2 / 3) < 1e-5


def test_protocol122_single_summary_handles_no_taken_trades() -> None:
    skipped = [_trade("expensive", premium=2_000.0, pnl=100.0)]
    skipped[0]["paper_skip_reason"] = "insufficient_cash"

    summary = _single_summary([], skipped, starting_equity=500.0, paper_seed=1)

    assert summary["trades"] == 0
    assert summary["ending_equity"] == 500.0
    assert summary["insufficient_cash_skips"] == 1
    assert summary["trade_capture_rate"] == 0.0


def test_protocol122_decision_uses_10000_as_trading_capital_baseline() -> None:
    decision = decide(
        [
            {"starting_equity": 500.0, "trades": 0, "trade_capture_rate": 0.0},
            {
                "starting_equity": 10_000.0,
                "trades": 10,
                "trade_capture_rate": 1.0,
                "insufficient_cash_skips": 0,
            },
        ]
    )

    assert decision == "pass_10000_paper_capital_baseline"
