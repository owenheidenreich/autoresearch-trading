"""An experiment must be declared before it is scored, and always spend alpha."""
from __future__ import annotations

import numpy as np
import pytest

from v5.research.autoresearch import budget as b, experiment as ex


def _decl(i=0):
    return ex.Declaration(
        experiment_id=f"exp-{i:04d}",
        hypothesis="overnight gap continues over 60 minutes",
        features_used=("overnight_gap",),
        horizon_minutes=60,
        declared_on="2026-08-13",
    )


def _ledger(tmp_path, sessions=2520):
    return b.AlphaLedger(tmp_path / "alpha.json", option_sessions=sessions, universe="phase1_otm")


def _calls(n, accuracy, seed=0):
    rng = np.random.default_rng(seed)
    calls = rng.choice([-1, 1], n)
    ok = rng.random(n) < accuracy
    move = np.where(ok, calls, -calls) * np.abs(rng.normal(0, 8, n))
    return calls, move


def test_a_declaration_hash_is_stable_and_content_addressed() -> None:
    assert _decl().sha256() == _decl().sha256()
    other = ex.Declaration(
        experiment_id="exp-0000",
        hypothesis="overnight gap FADES over 60 minutes",
        features_used=("overnight_gap",),
        horizon_minutes=60,
        declared_on="2026-08-13",
    )
    assert other.sha256() != _decl().sha256()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hypothesis": ""},
        {"features_used": ()},
        {"horizon_minutes": 0},
        {"experiment_id": ""},
    ],
)
def test_an_underspecified_experiment_is_refused(kwargs) -> None:
    base = dict(
        experiment_id="x",
        hypothesis="h",
        features_used=("f",),
        horizon_minutes=60,
        declared_on="2026-08-13",
    )
    with pytest.raises(ex.ExperimentError):
        ex.Declaration(**{**base, **kwargs})


def test_a_strong_experiment_passes_and_a_coin_flip_does_not(tmp_path) -> None:
    led = _ledger(tmp_path)
    calls, move = _calls(2520, 0.50, seed=1)
    assert ex.run_and_record(_decl(0), calls, move, ledger=led).passed is False

    calls, move = _calls(2520, 0.72, seed=2)
    assert ex.run_and_record(_decl(1), calls, move, ledger=led).passed is True


def test_every_scored_experiment_spends_alpha(tmp_path) -> None:
    led = _ledger(tmp_path)
    before = led.required_accuracy()
    calls, move = _calls(2520, 0.50, seed=3)
    ex.run_and_record(_decl(0), calls, move, ledger=led)
    assert led.experiments_run == 1
    assert led.required_accuracy() > before


def test_an_abandoned_experiment_still_spends_alpha(tmp_path) -> None:
    """Starting a look costs, even when the look is abandoned."""

    led = _ledger(tmp_path)
    with pytest.raises(ex.ExperimentError):
        ex.run_and_record(_decl(0), [0, 0, 0], [1.0, -1.0, 1.0], ledger=led)
    assert led.experiments_run == 1
    assert next(iter(led)).outcome == "REFUSED"


def test_standing_down_is_excluded_not_counted_wrong(tmp_path) -> None:
    led = _ledger(tmp_path)
    calls = [1, 0, 1, 0]
    move = [5.0, -99.0, 5.0, -99.0]
    score = ex.score_calls(_decl(), calls, move, ledger=led)
    assert score.called == 2 and score.correct == 2
    assert score.accuracy == pytest.approx(1.0)


def test_a_zero_move_under_a_live_call_counts_against(tmp_path) -> None:
    led = _ledger(tmp_path)
    score = ex.score_calls(_decl(), [1, 1], [0.0, 4.0], ledger=led)
    assert score.called == 2 and score.correct == 1


def test_the_loop_refuses_to_run_once_the_budget_is_exhausted(tmp_path) -> None:
    led = b.AlphaLedger(tmp_path / "a.json", option_sessions=251, universe="phase1_otm", ceiling=0.60)
    assert led.exhausted
    calls, move = _calls(251, 0.9, seed=4)
    with pytest.raises(ex.ExperimentError, match="alpha budget exhausted"):
        ex.run_and_record(_decl(), calls, move, ledger=led)
    assert led.experiments_run == 0, "a refused-to-start run must not spend alpha"


def test_the_bar_an_experiment_faces_is_the_one_recorded(tmp_path) -> None:
    led = _ledger(tmp_path)
    calls, move = _calls(2520, 0.55, seed=5)
    score = ex.run_and_record(_decl(0), calls, move, ledger=led)
    assert score.bar == pytest.approx(next(iter(led)).bar_at_time_of_run, abs=1e-8)


def test_option_pnl_tracks_accuracy_through_the_frozen_outcomes(tmp_path) -> None:
    led = _ledger(tmp_path)
    calls, move = _calls(2520, 0.58, seed=6)
    score = ex.score_calls(_decl(), calls, move, ledger=led)
    # Break-even accuracy is 57.99%, so a 58% experiment sits near zero.
    assert abs(score.expected_option_pnl_per_trade) < 20.0


@pytest.mark.parametrize("bad", [[2, 1], [1, 5]])
def test_calls_must_be_minus_one_zero_or_one(tmp_path, bad) -> None:
    led = _ledger(tmp_path)
    with pytest.raises(ex.ExperimentError, match="calls must be"):
        ex.score_calls(_decl(), bad, [1.0, 1.0], ledger=led)


def test_misaligned_inputs_are_refused(tmp_path) -> None:
    led = _ledger(tmp_path)
    with pytest.raises(ex.ExperimentError, match="not aligned"):
        ex.score_calls(_decl(), [1, 1], [1.0], ledger=led)
