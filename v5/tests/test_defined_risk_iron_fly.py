from __future__ import annotations

import pandas as pd
import pytest

from v5.research.defined_risk_iron_fly import evaluate_session


def _quotes(*, exit_debit_too_wide: bool = False) -> pd.DataFrame:
    rows = []
    ids = {
        (100.0, "C"): "c100",
        (100.0, "P"): "p100",
        (105.0, "C"): "c105",
        (95.0, "P"): "p95",
    }
    entry = {
        (100.0, "C"): (3.0, 3.2),
        (100.0, "P"): (3.0, 3.2),
        (105.0, "C"): (0.8, 1.0),
        (95.0, "P"): (0.8, 1.0),
    }
    exit_values = {
        (100.0, "C"): (2.0, 2.2),
        (100.0, "P"): (2.0, 2.2),
        (105.0, "C"): (0.7, 0.9),
        (95.0, "P"): (0.7, 0.9),
    }
    if exit_debit_too_wide:
        exit_values[(100.0, "C")] = (6.0, 6.2)
    for minute, values in (("15:00", entry), ("15:15", exit_values)):
        for (strike, right), (bid, ask) in values.items():
            rows.append(
                {
                    "minute": minute,
                    "contract_id": ids[(strike, right)],
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                    "quote_age_ms": 0.0,
                    "underlying_price": 100.2,
                }
            )
    return pd.DataFrame(rows)


def test_four_leg_touch_and_fee_arithmetic() -> None:
    got = evaluate_session(_quotes(), session="2025-08-01", settlement_spx=100.0).row
    assert got["traded"] is True
    # Entry credit 4.0 points; exit debit 3.0; four x $3.08 fees.
    assert got["entry_touch_credit_usd"] == pytest.approx(400.0)
    assert got["exit_touch_debit_usd"] == pytest.approx(300.0)
    assert got["net_touch_usd"] == pytest.approx(87.68)
    assert got["declared_max_loss_usd"] == pytest.approx(112.32)


def test_combo_refuses_debit_above_width_and_uses_settlement() -> None:
    got = evaluate_session(
        _quotes(exit_debit_too_wide=True), session="2025-08-01", settlement_spx=103.0
    ).row
    assert got["exit_type"] == "validated_cash_settlement"
    assert got["exit_touch_debit_usd"] == pytest.approx(300.0)
    assert got["net_touch_usd"] == pytest.approx(87.68)


def test_entry_strike_does_not_change_for_future_exit_availability() -> None:
    quotes = _quotes()
    # A farther complete structure appears only later. It cannot change K=100.
    extra = quotes[quotes["minute"].eq("15:15")].copy()
    extra["strike"] += 10.0
    extra["contract_id"] = "later-" + extra["contract_id"]
    got = evaluate_session(
        pd.concat([quotes, extra], ignore_index=True),
        session="2025-08-01",
        settlement_spx=100.0,
    ).row
    assert got["center_strike"] == 100.0


def test_missing_entry_structure_abstains_and_keeps_zero_session() -> None:
    quotes = _quotes()
    quotes = quotes[~quotes["contract_id"].eq("c105")]
    got = evaluate_session(quotes, session="2025-08-01", settlement_spx=100.0).row
    assert got["traded"] is False
    assert got["net_touch_usd"] == 0.0
