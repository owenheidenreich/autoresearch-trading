from __future__ import annotations

from v4.scripts.run_protocol134_sizing_attribution import (
    build_attribution,
    concentration_checks,
    decide,
)


def _row(trade_number: int, *, quantity: int, pnl: float, session: str = "2026-03-06") -> dict:
    return {
        "trade_number": trade_number,
        "session": session,
        "segment": "q1_2026",
        "stage": "test",
        "decision_time": f"{session}T15:00:00+00:00",
        "exit_time": f"{session}T15:05:00+00:00",
        "contract_id": "SPXW-20260306-06700.000-C",
        "side": "CALL",
        "score_margin": 2.5,
        "one_contract_premium": 1_000.0,
        "quantity": quantity,
        "realized_pnl": pnl,
        "skip_reason": "",
    }


def test_build_attribution_computes_incremental_pnl_and_scaled_flag() -> None:
    baseline = [_row(1, quantity=1, pnl=100.0)]
    candidate = [_row(1, quantity=2, pnl=200.0)]

    rows = build_attribution(baseline, candidate)

    assert rows[0]["incremental_pnl"] == 100.0
    assert rows[0]["is_scaled"] is True
    assert rows[0]["quantity_delta"] == 1


def test_concentration_tracks_top_day_and_skipped_pnl_given_up() -> None:
    rows = build_attribution(
        [_row(1, quantity=1, pnl=100.0), _row(2, quantity=1, pnl=-50.0, session="2026-03-07")],
        [_row(1, quantity=2, pnl=200.0), _row(2, quantity=0, pnl=0.0, session="2026-03-07")],
    )

    result = concentration_checks(rows)

    assert result["incremental_pnl"] == 150.0
    assert result["scaled_trades"] == 1
    assert result["skipped_trades"] == 1
    assert result["skipped_pnl_given_up"] == -50.0


def test_decide_rejects_top_day_concentration() -> None:
    concentration = {
        "top_day_share_of_positive": 0.50,
        "top_month_share_of_positive": 0.20,
        "scaled_positive_fraction": 0.80,
    }
    candidate = {"total_pnl": 200.0}
    baseline = {"total_pnl": 100.0}

    assert decide(concentration, candidate, baseline) == "fragile_sizing_top_day_concentrated"

