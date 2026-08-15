from __future__ import annotations

import pytest

from v5.research.mes_strategic_pivot import (
    scaled_data_cost,
    target_stop_economics,
    translate_es_spread_to_mes,
)


def test_measured_es_tick_proxy_translates_to_mes_dollars() -> None:
    got = translate_es_spread_to_mes(1.0734099412079101)
    assert got.spread_points == pytest.approx(0.2683524853)
    assert got.spread_usd == pytest.approx(1.3417624265)
    assert got.stated_fees_usd == pytest.approx(1.20)
    assert got.translated_round_trip_points == pytest.approx(0.5083524853)
    assert got.translated_round_trip_usd == pytest.approx(2.5417624265)
    assert got.conservative_round_trip_points >= got.translated_round_trip_points


def test_ten_over_five_move_has_defined_net_arithmetic() -> None:
    got = target_stop_economics(10, 5)
    assert got["net_win_usd"] == pytest.approx(47.25)
    assert got["net_loss_usd"] == pytest.approx(-27.75)
    assert got["reward_to_risk"] == pytest.approx(47.25 / 27.75)
    assert got["breakeven_target_hit_rate"] == pytest.approx(0.37)


def test_data_cost_is_scaled_from_receipts_not_asserted() -> None:
    got = scaled_data_cost(
        1830,
        ohlcv_cost_usd=2.592221274972,
        ohlcv_sessions=311,
        bbo_cost_usd=1.478327661753,
        bbo_sessions=20,
    )
    assert got["scaled_total_usd"] == pytest.approx(150.52, abs=0.02)
    assert got["scaled_total_usd"] < 200


@pytest.mark.parametrize("ticks", [0, -1, 10, float("inf")])
def test_spread_translation_refuses_nonsense(ticks: float) -> None:
    with pytest.raises(ValueError):
        translate_es_spread_to_mes(ticks)

