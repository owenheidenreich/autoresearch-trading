from __future__ import annotations

from v4.scripts.export_protocol101_trade_charts import (
    MEASURED_COMMISSION_PER_SIDE_USD,
    _normalize_trade_chart_bars,
    build_paper_account_trades,
    commission_cost,
    equity_after_commission,
    equity_after_stress,
    net_trade_pnl,
    slippage_cost,
    stressed_trade_pnl,
)


def _trade(uid: str, decision_ms: int, exit_ms: int, premium: float, pnl: float) -> dict:
    return {
        "seed": 1,
        "candidate_uid": uid,
        "decision_ms": decision_ms,
        "exit_ms": exit_ms,
        "decision_time": f"2026-03-06T15:{decision_ms // 60_000:02d}:00+00:00",
        "exit_time": f"2026-03-06T15:{exit_ms // 60_000:02d}:00+00:00",
        "session": "2026-03-06",
        "entry_ask": premium / 100.0,
        "premium_paid": premium,
        "path_mae": -min(premium, 100.0),
        "path_mfe": 200.0,
        "pnl": pnl,
    }


def test_paper_account_skips_unaffordable_contracts() -> None:
    trades = [
        _trade("too_expensive", 0, 60_000, premium=12_000.0, pnl=500.0),
        _trade("affordable", 120_000, 180_000, premium=2_000.0, pnl=300.0),
    ]

    taken, skipped = build_paper_account_trades(trades, starting_equity=10_000.0, paper_seed=1)

    assert [row["candidate_uid"] for row in taken] == ["affordable"]
    assert skipped[0]["candidate_uid"] == "too_expensive"
    assert skipped[0]["paper_skip_reason"] == "insufficient_cash"
    assert taken[0]["paper_cash_before"] == 10_000.0
    assert taken[0]["paper_cash_after"] == 10_300.0


def test_paper_account_enforces_one_open_position() -> None:
    trades = [
        _trade("first", 0, 180_000, premium=1_000.0, pnl=100.0),
        _trade("overlap", 60_000, 120_000, premium=1_000.0, pnl=100.0),
        _trade("after_exit", 240_000, 300_000, premium=1_000.0, pnl=100.0),
    ]

    taken, skipped = build_paper_account_trades(trades, starting_equity=10_000.0, paper_seed=1)

    assert [row["candidate_uid"] for row in taken] == ["first", "after_exit"]
    assert skipped[0]["candidate_uid"] == "overlap"
    assert skipped[0]["paper_skip_reason"] == "overlap_open_position"


def test_trade_chart_reindexes_filtered_nonzero_spx_bars_with_markers() -> None:
    spx = [
        {"bar": 41_091, "t": 1, "session": "2025-03-06", "close": 5_700.0},
        {"bar": 41_092, "t": 2, "session": "2025-03-06", "close": 5_701.0},
        {"bar": 42_010, "t": 3, "session": "2025-03-10", "close": 5_600.0},
    ]
    trades = [{"entry_bar": 41_091.0, "exit_bar": 42_010.0, "seed": 101}]

    normalized_trades, normalized_spx = _normalize_trade_chart_bars(trades, spx)

    assert [row["bar"] for row in normalized_spx] == [0, 1, 2]
    assert normalized_trades[0]["entry_bar"] == 0.0
    assert normalized_trades[0]["exit_bar"] == 2.0
    assert spx[0]["bar"] == 41_091
    assert trades[0]["entry_bar"] == 41_091.0


def test_commission_is_dollars_and_slippage_is_price_points() -> None:
    """The two frictions carry different units; mixing them is a 100x error.

    Commission is quoted in dollars per side and must not take the contract
    multiplier. Slippage is quoted in option price points and must. Scaling
    commission by 100 would turn a $3.08 round trip into $308.
    """

    assert commission_cost(1.54) == 3.08
    assert slippage_cost(0.10) == 20.0


def test_base_equity_curve_now_charges_commission() -> None:
    """Gross pnl crosses the spread but paid no commission at all.

    Every equity curve this exporter has produced was overstated by roughly
    $3.08 per round trip because commission was absent from both the base line
    and the stress line.
    """

    row = {"pnl": 100.0}
    assert net_trade_pnl(row, 1.54) == 96.92
    assert equity_after_commission([row, row], 10_000.0, 1.54) == 10_193.84


def test_stress_line_is_commission_plus_slippage_not_slippage_alone() -> None:
    row = {"pnl": 100.0}
    assert stressed_trade_pnl(row, 0.10, 1.54) == 76.92
    assert equity_after_stress([row], 10_000.0, 0.10, 1.54) == 10_076.92


def test_measured_commission_default_matches_the_ibkr_observation() -> None:
    """1.54/side was measured on IBKR paper 2026-08-04 (Track C).

    It corrected a $0.65/side figure that was wrong by 2.4x across four prior
    documents, and it nearly confirms the frozen FILL_LAW at $1.50/side.
    """

    assert MEASURED_COMMISSION_PER_SIDE_USD == 1.54
    row = {"pnl": 0.0}
    assert stressed_trade_pnl(row, 0.0) == -3.08
