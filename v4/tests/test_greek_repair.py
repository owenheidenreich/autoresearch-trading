from __future__ import annotations

from v4.greeks.repair import compute_repaired_greeks


def test_compute_repaired_greeks_uses_ask_when_mid_is_below_no_arb_floor() -> None:
    estimate = compute_repaired_greeks(
        S=6500.0,
        K=6480.0,
        T=1.0 / (365.0 * 24.0),
        is_call=True,
        mid=20.0,
        ask=20.30,
        bid=19.70,
    )

    assert estimate is not None
    assert estimate.source == "black_scholes_ask_repair"
    assert estimate.iv > 0.0
    assert estimate.gamma >= 0.0


def test_compute_repaired_greeks_returns_none_when_observable_prices_are_impossible() -> None:
    estimate = compute_repaired_greeks(
        S=6500.0,
        K=6480.0,
        T=1.0 / (365.0 * 24.0),
        is_call=True,
        mid=19.0,
        ask=19.20,
        bid=18.80,
    )

    assert estimate is None
