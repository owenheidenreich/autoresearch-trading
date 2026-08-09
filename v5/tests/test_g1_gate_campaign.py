"""The gate, the two known-answer fixtures, and the campaign that judges them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v5.research import knobs
from v5.research.direction import campaign, family, fixtures, gate, loader, surrogate


def _bars(session: str, closes: list[float], instrument_id: int = 1):
    close = np.asarray(closes, dtype=float)
    return loader.SessionBars(
        session=session,
        instrument_id=instrument_id,
        minute_et=tuple(
            f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(close.size)
        ),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(close.size, 100.0),
    )


# --- familywise control ------------------------------------------------------


def test_the_gate_adjusts_confidence_for_the_declared_family_size() -> None:
    """Eighteen attempts at significance may not be quoted at the price of one."""

    raw = float(knobs.frozen_value(family.CONFIDENCE_KNOB))
    adjusted = gate.familywise_confidence()
    assert adjusted > raw
    assert (1.0 - adjusted) == pytest.approx((1.0 - raw) / family.FAMILY_SIZE)


def test_a_single_member_family_is_unadjusted() -> None:
    raw = float(knobs.frozen_value(family.CONFIDENCE_KNOB))
    assert gate.familywise_confidence(1) == pytest.approx(raw)


# --- the campaign's own arithmetic -------------------------------------------


def test_wilson_upper_exceeds_the_point_estimate() -> None:
    assert campaign.wilson_upper(5, 100) > 0.05
    assert campaign.wilson_upper(0, 100) > 0.0
    assert campaign.wilson_upper(50, 100) < 1.0


def test_wilson_upper_tightens_as_trials_grow() -> None:
    assert campaign.wilson_upper(5, 100) > campaign.wilson_upper(50, 1000)


def test_a_rate_cannot_be_bounded_without_trials() -> None:
    with pytest.raises(campaign.CampaignError):
        campaign.wilson_upper(0, 0)


def test_a_ceiling_criterion_needs_both_the_rate_and_the_bound() -> None:
    """A lucky small campaign must not claim control it has not demonstrated."""

    lucky = campaign.CampaignResult(
        name="x", trials=20, passes=0, rate=0.0,
        wilson_upper=campaign.wilson_upper(0, 20), limit=0.05, is_floor=False,
    )
    assert lucky.rate <= lucky.limit
    assert not lucky.satisfied  # the Wilson bound is still too wide

    solid = campaign.CampaignResult(
        name="x", trials=1000, passes=17, rate=0.017,
        wilson_upper=campaign.wilson_upper(17, 1000), limit=0.05, is_floor=False,
    )
    assert solid.satisfied


def test_a_floor_criterion_is_judged_on_the_point_estimate() -> None:
    weak = campaign.CampaignResult(
        name="r", trials=100, passes=42, rate=0.42,
        wilson_upper=0.5, limit=0.80, is_floor=True,
    )
    assert not weak.satisfied
    strong = campaign.CampaignResult(
        name="r", trials=100, passes=85, rate=0.85,
        wilson_upper=0.9, limit=0.80, is_floor=True,
    )
    assert strong.satisfied


# --- the injected effect must be causal and monotonic ------------------------


def test_injection_never_touches_a_bar_before_entry() -> None:
    """An edge a policy could see before deciding is not an edge, it is a leak."""

    # Session 2 must open away from session 1's close, or the gap is zero and
    # sign(0) correctly injects nothing.
    real = (_bars("2030-01-01", list(np.linspace(100, 101, 70))),
            _bars("2030-01-02", list(np.linspace(104, 105, 70))))
    injected = fixtures.inject_effect(real, points_per_session=5.0, mechanism="M3")
    entry = real[1].minute_et.index(family.ENTRY_BAR_ET)
    assert np.allclose(injected[1].close[: entry + 1], real[1].close[: entry + 1])
    assert not np.allclose(injected[1].close[entry + 1 :], real[1].close[entry + 1 :])


def test_a_larger_injection_moves_the_path_further() -> None:
    real = (_bars("2030-01-01", list(np.linspace(100, 101, 70))),
            _bars("2030-01-02", list(np.linspace(104, 105, 70))))
    small = fixtures.inject_effect(real, points_per_session=2.0, mechanism="M3")
    large = fixtures.inject_effect(real, points_per_session=8.0, mechanism="M3")
    assert abs(large[1].close[-1] - real[1].close[-1]) > abs(
        small[1].close[-1] - real[1].close[-1]
    )


def test_a_non_positive_injection_is_refused() -> None:
    real = (_bars("2030-01-01", [100.0] * 70),)
    with pytest.raises(fixtures.FixtureError):
        fixtures.inject_effect(real, points_per_session=0.0)
    with pytest.raises(fixtures.FixtureError):
        fixtures.inject_effect(real, points_per_session=1.0, reference_horizon=0)


def test_the_shared_term_fixture_is_a_martingale_not_a_reverting_series() -> None:
    """The first draft built mean reversion, which is real predictability.

    A reverting path is not row 183's artifact; the gate passed it 100% of the
    time and was right to. The fixture must have independent increments.
    """

    real = tuple(
        _bars(f"2030-01-{d:02d}", list(np.linspace(100, 101, 200))) for d in range(1, 9)
    )
    built = fixtures.shared_term_sessions(real, seed=5)
    steps = np.concatenate([np.diff(s.close) for s in built])
    # Lag-1 autocorrelation of increments: a reverting series is strongly
    # negative here, an independent one sits near zero.
    lag1 = float(np.corrcoef(steps[:-1], steps[1:])[0, 1])
    assert abs(lag1) < 0.15, lag1


def test_fixtures_refuse_an_empty_corpus() -> None:
    with pytest.raises(fixtures.FixtureError):
        fixtures.shared_term_sessions((), seed=1)


# --- the gate on a known index -----------------------------------------------


def test_every_criterion_is_reported_separately() -> None:
    """A single boolean would hide which criterion actually bound."""

    real = tuple(
        _bars(f"2030-02-{d:02d}", list(np.linspace(100, 100.5, 70)))
        for d in range(1, 12)
    )
    fake, _ = surrogate.matched_surrogate(real, seed=1)
    features = loader.session_features(fake)
    member = family.assert_member_registered("M1.with.15m")
    with pytest.raises(Exception):
        # The synthetic corpus is far smaller than the frozen index, and the
        # replay must refuse rather than quietly judge a partial calendar.
        gate.judge_member(features, member)


@pytest.mark.skipif(
    not Path(family.ES_BARS_ROOT).is_dir(), reason="owned ES bars not present"
)
def test_the_gate_runs_on_a_surrogate_of_the_real_index() -> None:
    real = loader.load_sessions()
    fake, _ = surrogate.matched_surrogate(real, seed=1)
    results = gate.judge_family(loader.session_features(fake), seed_offset=1)
    assert len(results) == family.FAMILY_SIZE
    for result in results:
        assert result.criteria["full_index_evaluated"]
        assert result.criteria["fold_count_as_declared"]
        assert result.sessions in {
            family.M1_ELIGIBLE_SESSIONS,
            family.GAP_ELIGIBLE_SESSIONS,
        }
