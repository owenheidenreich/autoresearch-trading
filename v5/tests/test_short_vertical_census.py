from __future__ import annotations

import pandas as pd
import pytest

from v5.research.short_vertical_census import (
    FAMILY,
    FAMILY_SIZE,
    MAX_LOSS_USD,
    VerticalCensusError,
    evaluate_session,
)


def _snapshot(minute: str, spot: float, quotes: dict[tuple[float, str], tuple[float, float]]):
    rows = []
    for (strike, right), (bid, ask) in quotes.items():
        rows.append(
            {
                "minute": minute,
                "contract_id": f"{right}{int(strike)}",
                "strike": strike,
                "right": right,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "bid_size": 5.0,
                "ask_size": 5.0,
                "quote_age_ms": 0.0,
                "underlying_price": spot,
            }
        )
    return rows


def _session_frame(spot: float = 6000.0) -> pd.DataFrame:
    quotes = {}
    for strike in (5970.0, 5980.0, 5990.0, 5995.0, 6000.0, 6005.0, 6010.0, 6020.0, 6030.0):
        for right in ("C", "P"):
            # Richer credits near the money; every pair is live and two-sided.
            distance = abs(strike - spot)
            mid = max(0.5, 30.0 - distance)
            quotes[(strike, right)] = (mid - 0.4, mid + 0.4)
    rows = []
    for minute in ("13:00", "14:00", "15:00"):
        rows.extend(_snapshot(minute, spot, quotes))
    return pd.DataFrame(rows)


def test_family_is_36_cells_and_all_rows_present() -> None:
    assert FAMILY_SIZE == 36
    result = evaluate_session(_session_frame(), session="2026-01-02", settlement_spx=6000.0)
    assert len(result.rows) == 36
    assert {row["cell"] for row in result.rows} == {cell["cell"] for cell in FAMILY}


def test_call_credit_selects_short_at_target_and_long_wider() -> None:
    result = evaluate_session(_session_frame(), session="s", settlement_spx=6000.0)
    cell = next(
        row
        for row in result.rows
        if row["cell"] == "call_credit_w5_d0_1500" and row["traded"]
    )
    assert cell["short_strike"] == 6000.0
    assert cell["long_strike"] == 6005.0
    # Touch credit is short bid minus long ask: (30-0.4) - (25+0.4) = 4.2.
    assert cell["entry_touch_credit_usd"] == pytest.approx(420.0)
    # Fee-only credit is mid minus mid: 30 - 25 = 5.0.
    assert cell["entry_mid_credit_usd"] == pytest.approx(500.0)


def test_settlement_debit_is_capped_by_the_wing() -> None:
    result = evaluate_session(_session_frame(), session="s", settlement_spx=6050.0)
    cell = next(
        row
        for row in result.rows
        if row["cell"] == "call_credit_w5_d0_1500" and row["traded"]
    )
    # Settlement 6050: short pays 50, long recovers 45, debit capped at width 5.
    assert cell["settlement_debit_usd"] == pytest.approx(500.0)
    assert cell["net_touch_usd"] == pytest.approx(420.0 - 500.0 - cell["fees_usd"])
    # The loss can never exceed the entry-declared maximum.
    assert cell["net_touch_usd"] >= -cell["declared_max_loss_usd"] - 1e-6


def test_put_credit_expires_worthless_above_short_strike() -> None:
    result = evaluate_session(_session_frame(), session="s", settlement_spx=6100.0)
    cell = next(
        row
        for row in result.rows
        if row["cell"] == "put_credit_w5_d0_1500" and row["traded"]
    )
    assert cell["settlement_debit_usd"] == pytest.approx(0.0)
    assert cell["net_touch_usd"] == pytest.approx(
        cell["entry_touch_credit_usd"] - cell["fees_usd"]
    )


def test_risk_law_abstains_when_max_loss_exceeds_the_breaker() -> None:
    # Far-OTM quotes decay to pennies: the wide wing's credit cannot cover the
    # $1,000 maximum loss inside the $500 breaker, so the cell must abstain.
    quotes = {}
    spot = 6000.0
    for strike in (6000.0, 6005.0, 6010.0, 6020.0, 6030.0):
        for right in ("C", "P"):
            distance = abs(strike - spot)
            mid = max(0.6, 30.0 - distance) if distance < 20 else (1.0 if distance == 20 else 0.6)
            quotes[(strike, right)] = (mid - 0.4, mid + 0.4)
    frame = pd.DataFrame(
        _snapshot("15:00", spot, quotes)
        + _snapshot("13:00", spot, quotes)
        + _snapshot("14:00", spot, quotes)
    )
    result = evaluate_session(frame, session="s", settlement_spx=6000.0)
    cell = next(
        row for row in result.rows if row["cell"] == "call_credit_w10_d20_1500"
    )
    assert not cell["traded"]
    assert cell["status"] == "abstain_credit_or_risk"
    assert cell["net_touch_usd"] == 0.0
    assert cell["declared_max_loss_usd"] > MAX_LOSS_USD


def test_missing_columns_raise() -> None:
    frame = _session_frame().drop(columns=["quote_age_ms"])
    with pytest.raises(VerticalCensusError):
        evaluate_session(frame, session="s", settlement_spx=6000.0)


def test_summarize_cells_aggregates_every_cell_with_folds_and_bounds() -> None:
    # The first census run died in aggregation, after evaluation and before
    # any outcome was read; this covers that path end to end on synthetic rows.
    from v5.ops.run_short_vertical_census import summarize_cells

    frames = []
    for session in ("2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"):
        result = evaluate_session(
            _session_frame(), session=session, settlement_spx=6000.0
        )
        frames.append(pd.DataFrame(result.rows))
    combined = pd.concat(frames, ignore_index=True)
    cells = summarize_cells(combined)
    assert len(cells) == 36
    for cell in cells:
        assert cell["sessions"] == 5
        assert 0 <= cell["net_fee_only_positive_folds"] <= 5
        assert cell["net_touch_corrected_lcb_usd"] <= cell["net_touch_mean_per_session_usd"]
        assert isinstance(cell["clears_fee_only"], bool)


def test_empty_session_abstains_everywhere() -> None:
    frame = _session_frame().iloc[0:0]
    result = evaluate_session(frame, session="s", settlement_spx=6000.0)
    assert len(result.rows) == 36
    assert all(not row["traded"] for row in result.rows)
    assert all(row["net_touch_usd"] == 0.0 for row in result.rows)
