"""The shared statistics, and the invariant the 2026-08 measurement work found."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from v5.research import statistics as st
from v5.research.direction import campaign


RECEIPT = Path(__file__).resolve().parents[1] / "work/occupancy-feasibility/feasibility_receipt.json"


# Acklam's approximation is accurate to ~1.15e-9 *relative*, so absolute error
# near 1.64 is ~1.9e-9. Anything tighter than 1e-8 tests the approximation
# rather than the code.
QUANTILE_TOLERANCE = 1e-8


def test_normal_quantile_matches_known_values() -> None:
    assert st.normal_quantile(0.95) == pytest.approx(
        1.6448536269514722, abs=QUANTILE_TOLERANCE
    )
    assert st.normal_quantile(0.80) == pytest.approx(
        0.8416212335729143, abs=QUANTILE_TOLERANCE
    )
    assert st.normal_quantile(0.975) == pytest.approx(
        1.959963984540054, abs=QUANTILE_TOLERANCE
    )
    assert st.normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)


def test_normal_quantile_is_symmetric_and_monotone() -> None:
    for p in (0.01, 0.1, 0.25, 0.4, 0.6, 0.9, 0.99):
        assert st.normal_quantile(p) == pytest.approx(-st.normal_quantile(1 - p), abs=1e-8)
    grid = [st.normal_quantile(p / 100) for p in range(1, 100)]
    assert all(later > earlier for earlier, later in zip(grid, grid[1:]))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_normal_quantile_refuses_a_value_outside_the_open_unit_interval(bad) -> None:
    with pytest.raises(st.StatisticsError):
        st.normal_quantile(bad)


def test_the_shared_constant_pins_the_copy_inside_the_campaign_module() -> None:
    """Two implementations of the same constant are one drift away from a bug."""

    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert repr(st.Z_95) in source


def test_bonferroni_quantile_widens_with_family_size() -> None:
    single = st.bonferroni_quantile(1)
    assert single == pytest.approx(st.Z_95, abs=QUANTILE_TOLERANCE)
    assert st.bonferroni_quantile(18) > st.bonferroni_quantile(2) > single
    # The factor the feasibility module divides G1's measured 2-4x conjunction
    # penalty by, so that Bonferroni is not charged twice.
    factor = (st.bonferroni_quantile(18) + st.Z_POWER_80) / (st.Z_95 + st.Z_POWER_80)
    assert factor == pytest.approx(1.453682, abs=1e-6)


@pytest.mark.parametrize("bad", [0, -1])
def test_bonferroni_quantile_refuses_an_empty_family(bad) -> None:
    with pytest.raises(st.StatisticsError):
        st.bonferroni_quantile(bad)


def test_detectable_sharpe_falls_as_the_root_of_the_session_count() -> None:
    at_247 = st.detectable_sharpe(247, z_alpha=st.Z_95)
    at_988 = st.detectable_sharpe(988, z_alpha=st.Z_95)
    assert at_247 / at_988 == pytest.approx(2.0, abs=1e-6)


def test_sessions_for_sharpe_inverts_detectable_sharpe() -> None:
    for target in (0.5, 1.0, 1.5, 2.0, 3.0):
        n = st.sessions_for_sharpe(target, z_alpha=st.Z_95)
        assert st.detectable_sharpe(n, z_alpha=st.Z_95) <= target + 1e-9
        assert st.detectable_sharpe(n - 1, z_alpha=st.Z_95) > target


@pytest.mark.parametrize(
    "call",
    [
        lambda: st.detectable_sharpe(0, z_alpha=st.Z_95),
        lambda: st.detectable_sharpe(-5, z_alpha=st.Z_95),
        lambda: st.detectable_sharpe(100, z_alpha=st.Z_95, penalty=0.0),
        lambda: st.sessions_for_sharpe(0.0, z_alpha=st.Z_95),
        lambda: st.sessions_for_sharpe(-1.0, z_alpha=st.Z_95),
    ],
)
def test_the_invariant_refuses_nonsense_rather_than_returning_nan(call) -> None:
    with pytest.raises(st.StatisticsError):
        call()


def test_the_invariant_is_actually_invariant() -> None:
    """The finding itself, encoded as a test.

    Vary the move dispersion, the horizon, the trades per session, the tradeable
    minutes and the friction bar over wide ranges, compute the detectable Sharpe
    the long way round each time, and it must not move. This is why neither a
    higher-occupancy design nor a different instrument can rescue a screen: they
    change every input below and none of the output.
    """

    reference = st.detectable_sharpe(247, z_alpha=st.Z_95)
    z_total = st.Z_95 + st.Z_POWER_80

    for sigma_per_root_minute in (0.5, 1.3, 4.0):
        for tradeable_minutes in (210, 390, 1380):
            for horizon in (5, 15, 60, 120):
                for friction in (0.05, 0.358, 2.5):
                    trades = tradeable_minutes / horizon
                    sigma_h = sigma_per_root_minute * math.sqrt(horizon)
                    # Sessions at which the detection floor equals the friction bar.
                    n = (z_total * sigma_h / friction) ** 2 / trades
                    # Express that same requirement as an annualised Sharpe.
                    long_way = (friction / sigma_h) * math.sqrt(trades * 252)
                    assert long_way == pytest.approx(
                        st.detectable_sharpe(n, z_alpha=st.Z_95), rel=1e-9
                    )

    # And the value at the owned session count is the number the plan quotes.
    assert reference == pytest.approx(2.51, abs=0.01)


def test_the_invariant_reproduces_the_committed_occupancy_surface() -> None:
    """Cross-check against evidence produced by a completely different route.

    The job-15 surface computed a per-trade detection floor from measured ES bar
    dispersion at six horizons. Converting each of those cells to an annualised
    Sharpe must land on the invariant, which knows nothing about ES.
    """

    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    penalty = receipt["conjunction_penalty_bracket"][1]
    expected = st.detectable_sharpe(
        receipt["sessions"], z_alpha=st.Z_95, penalty=penalty
    )

    cells = [row for row in receipt["rows"] if row["family_size"] == 1]
    assert len(cells) == 6
    for row in cells:
        floor = row["analytic_mde_points_per_trade"] * penalty
        from_surface = (floor / row["move_sd_points"]) * math.sqrt(
            row["trades_per_session"] * 252
        )
        assert from_surface == pytest.approx(expected, rel=0.001)


def test_the_receipt_carries_the_ladder_the_plan_is_built_on() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    standards = receipt["detectable_sharpe_invariant"]["standards"]
    assert set(standards) == {
        "single_hypothesis",
        "single_hypothesis_conservative",
        "g1_eighteen_member_gate",
    }
    owned = standards["single_hypothesis"]["detectable_sharpe_by_sessions"]["247"]
    assert owned == pytest.approx(2.51, abs=0.01)
    # Ten years of history is the Phase 1 purchase; this is what it buys.
    ten_years = standards["single_hypothesis"]["detectable_sharpe_by_sessions"]["2520"]
    assert ten_years == pytest.approx(0.79, abs=0.01)


# --- the option-layer translation, which sets G1's real threshold -------------


def _option_payoffs() -> tuple[float, float]:
    from v5.research import knobs

    return (
        knobs.frozen_value("option_correct_call_dollars"),
        knobs.frozen_value("option_wrong_call_dollars"),
    )


def test_breakeven_accuracy_reproduces_the_audited_figure() -> None:
    win, loss = _option_payoffs()
    assert st.breakeven_accuracy(win=win, loss=loss) == pytest.approx(0.5799, abs=1e-4)
    assert st.accuracy_payoff(
        st.breakeven_accuracy(win=win, loss=loss), win=win, loss=loss
    ) == pytest.approx(0.0, abs=0.05)


def test_the_option_layer_needs_more_accuracy_to_be_seen_than_to_be_profitable() -> None:
    """The trap one level up: measurable and profitable are different bars."""

    win, loss = _option_payoffs()
    breakeven = st.breakeven_accuracy(win=win, loss=loss)
    detectable = st.detectable_accuracy(251, win=win, loss=loss, z_alpha=st.Z_95)

    assert detectable == pytest.approx(0.6573, abs=1e-4)
    assert detectable > breakeven
    assert (detectable - breakeven) == pytest.approx(0.0774, abs=1e-3)


def test_detectable_accuracy_improves_with_more_option_sessions() -> None:
    win, loss = _option_payoffs()
    at_251 = st.detectable_accuracy(251, win=win, loss=loss, z_alpha=st.Z_95)
    at_2520 = st.detectable_accuracy(2520, win=win, loss=loss, z_alpha=st.Z_95)
    assert at_2520 < at_251
    assert at_2520 == pytest.approx(0.6044, abs=1e-3)


def test_the_frozen_g1_threshold_is_6_23_es_points() -> None:
    """The number Phase 2 must freeze, recomputed from its inputs.

    G1's old bar was 0.358 ES points, which is ES friction and answers the wrong
    question. The right bar is what the option layer needs to survive
    translation. See G1_THRESHOLD_FROM_THE_OPTION_LAYER_2026_08_12.md.
    """

    win, loss = _option_payoffs()
    detectable = st.detectable_accuracy(251, win=win, loss=loss, z_alpha=st.Z_95)
    points = st.accuracy_to_underlying_points(detectable, move_sd=24.807)

    assert points == pytest.approx(6.23, abs=0.01)
    # 17.4x the ES friction bar it replaces.
    from v5.research import knobs

    friction = knobs.frozen_value("es_round_trip_friction_points")
    assert points / friction == pytest.approx(17.4, abs=0.1)


def test_sessions_for_accuracy_edge_inverts_detectable_accuracy() -> None:
    """The inverse must land exactly where the forward function says it can see."""

    win, loss = 2_460.18, 522.28
    target = 0.30
    n = st.sessions_for_accuracy_edge(target, win=win, loss=loss, z_alpha=st.Z_95)
    # At the returned trade count the forward function is at or just inside the
    # target; one trade fewer and it cannot see it.
    assert st.detectable_accuracy(n, win=win, loss=loss, z_alpha=st.Z_95) <= target
    assert st.detectable_accuracy(n - 1, win=win, loss=loss, z_alpha=st.Z_95) > target


def test_halving_the_precision_gap_quadruples_the_sessions() -> None:
    """The 1/n**2 law in the gap, which is what makes a marginal edge unprovable."""

    win, loss = 2_460.18, 522.28
    breakeven = st.breakeven_accuracy(win=win, loss=loss)
    near = st.sessions_for_accuracy_edge(
        breakeven + 0.04, win=win, loss=loss, z_alpha=st.Z_95
    )
    far = st.sessions_for_accuracy_edge(
        breakeven + 0.08, win=win, loss=loss, z_alpha=st.Z_95
    )
    assert near / far == pytest.approx(4.0, rel=0.02)


def test_trades_per_session_converts_trades_into_calendar_days() -> None:
    win, loss = 2_460.18, 522.28
    one = st.sessions_for_accuracy_edge(0.30, win=win, loss=loss, z_alpha=st.Z_95)
    three = st.sessions_for_accuracy_edge(
        0.30, win=win, loss=loss, z_alpha=st.Z_95, trades_per_session=3.0
    )
    assert three == math.ceil(one / 3)


def test_the_deep_target_asks_less_accuracy_than_the_shallow_one() -> None:
    """The owner's instinct, checked: a bigger move is an easier bar, not a harder one.

    Both targets are bought for the same $519.20 mean ask. The 30-point target
    pays 4.7 to 1 and breaks even at 17.5%; the 10-point target pays about 0.9
    to 1 and needs 53.2%. Holding for the larger move lowers the accuracy
    required, which is why the sparse tail is worth measuring at all.
    """

    entry, spread_and_fees = 519.1962187299046, 17.54 + 3.08
    bars = {}
    for depth in (10, 30):
        win = depth * 100.0 - entry - spread_and_fees
        bars[depth] = st.breakeven_accuracy(win=win, loss=entry + 3.08)

    assert bars[10] == pytest.approx(0.5316, abs=1e-3)
    assert bars[30] == pytest.approx(0.1751, abs=1e-3)
    assert bars[30] < bars[10]


def test_a_target_at_or_below_breakeven_is_refused() -> None:
    """There is no edge to prove, and returning a session count would imply there is."""

    win, loss = 2_460.18, 522.28
    breakeven = st.breakeven_accuracy(win=win, loss=loss)
    with pytest.raises(st.StatisticsError, match="no edge to prove"):
        st.sessions_for_accuracy_edge(breakeven, win=win, loss=loss, z_alpha=st.Z_95)


def test_accuracy_to_underlying_points_is_zero_at_a_coin_flip() -> None:
    assert st.accuracy_to_underlying_points(0.5, move_sd=24.807) == pytest.approx(0.0)
    assert st.accuracy_to_underlying_points(1.0, move_sd=24.807) == pytest.approx(
        24.807 * math.sqrt(2 / math.pi)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: st.breakeven_accuracy(win=0.0, loss=1.0),
        lambda: st.breakeven_accuracy(win=1.0, loss=-1.0),
        lambda: st.accuracy_payoff(1.5, win=1.0, loss=1.0),
        lambda: st.detectable_accuracy(0, win=1.0, loss=1.0, z_alpha=st.Z_95),
        lambda: st.accuracy_to_underlying_points(0.6, move_sd=0.0),
        lambda: st.accuracy_to_underlying_points(-0.1, move_sd=1.0),
    ],
)
def test_the_translation_refuses_nonsense(call) -> None:
    with pytest.raises(st.StatisticsError):
        call()
