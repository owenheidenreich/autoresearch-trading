"""The matched surrogate must preserve exactly what the declaration names."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v5.research.direction import family, loader, surrogate


def _bars(session: str, closes: list[float], *, opens: list[float] | None = None):
    close = np.asarray(closes, dtype=float)
    open_ = np.asarray(opens if opens is not None else closes, dtype=float)
    return loader.SessionBars(
        session=session,
        instrument_id=1,
        minute_et=tuple(
            f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(close.size)
        ),
        open=open_,
        high=np.maximum(open_, close) + 1.0,
        low=np.minimum(open_, close) - 1.0,
        close=close,
        volume=np.arange(1.0, close.size + 1.0),
    )


# --- what must be preserved --------------------------------------------------


def test_volume_and_calendar_are_preserved_exactly() -> None:
    real = (_bars("2026-01-05", [10, 11, 9, 12]), _bars("2026-01-06", [12, 13, 11]))
    fake, _ = surrogate.matched_surrogate(real, seed=7)
    assert [s.session for s in fake] == [s.session for s in real]
    for got, want in zip(fake, real):
        assert got.minute_et == want.minute_et
        assert np.array_equal(got.volume, want.volume)
        assert got.close.size == want.close.size
        assert got.instrument_id == want.instrument_id


def test_close_to_close_magnitudes_are_preserved() -> None:
    real = (_bars("2026-01-05", [100, 103, 101, 108, 104]),)
    fake, _ = surrogate.matched_surrogate(real, seed=3)
    want = np.abs(np.diff(real[0].close))
    got = np.abs(np.diff(fake[0].close))
    assert np.allclose(got, want)


def test_the_overnight_gap_magnitude_is_preserved() -> None:
    real = (
        _bars("2026-01-05", [100.0, 101.0]),
        _bars("2026-01-06", [104.0, 105.0], opens=[104.0, 104.5]),
    )
    fake, _ = surrogate.matched_surrogate(real, seed=11)
    want = abs(real[1].open[0] - real[0].close[-1])
    got = abs(fake[1].open[0] - fake[0].close[-1])
    assert got == pytest.approx(want)


def test_the_range_magnitude_is_preserved_when_geometry_allows() -> None:
    real = (_bars("2026-01-05", [100, 101, 100.5, 102]),)
    fake, audit = surrogate.matched_surrogate(real, seed=5)
    if audit.range_violations == 0:
        assert np.allclose(fake[0].high - fake[0].low, real[0].high - real[0].low)
    assert (fake[0].high >= fake[0].low).all()


def test_highs_and_lows_bracket_the_body() -> None:
    real = (_bars("2026-01-05", [100, 104, 99, 107, 103, 101]),)
    fake, _ = surrogate.matched_surrogate(real, seed=9)
    top = np.maximum(fake[0].open, fake[0].close)
    bottom = np.minimum(fake[0].open, fake[0].close)
    assert (fake[0].high >= top - 1e-9).all()
    assert (fake[0].low <= bottom + 1e-9).all()


# --- what must be randomized -------------------------------------------------


def test_different_seeds_give_different_paths() -> None:
    real = (_bars("2026-01-05", list(np.linspace(100, 130, 40))),)
    a, _ = surrogate.matched_surrogate(real, seed=1)
    b, _ = surrogate.matched_surrogate(real, seed=2)
    assert not np.allclose(a[0].close, b[0].close)


def test_the_same_seed_is_reproducible() -> None:
    real = (_bars("2026-01-05", [100, 102, 99, 105]),)
    a, _ = surrogate.matched_surrogate(real, seed=42)
    b, _ = surrogate.matched_surrogate(real, seed=42)
    assert np.array_equal(a[0].close, b[0].close)


def test_a_flat_session_stays_flat() -> None:
    """A zero move has no sign to flip; inventing one adds motion."""

    real = (_bars("2026-01-05", [100.0, 100.0, 100.0]),)
    fake, _ = surrogate.matched_surrogate(real, seed=4)
    assert np.allclose(np.diff(fake[0].close), 0.0)


def test_the_surrogate_corpus_is_internally_chained() -> None:
    """Each session's gap hangs off the previous *surrogate* close, not the real one."""

    real = (
        _bars("2026-01-05", [100.0, 101.0]),
        _bars("2026-01-06", [104.0, 105.0], opens=[104.0, 104.5]),
        _bars("2026-01-07", [108.0, 109.0], opens=[108.0, 108.5]),
    )
    fake, _ = surrogate.matched_surrogate(real, seed=13)
    for i in range(1, len(fake)):
        want = abs(real[i].open[0] - real[i - 1].close[-1])
        got = abs(fake[i].open[0] - fake[i - 1].close[-1])
        assert got == pytest.approx(want), i


# --- forbidden nulls ---------------------------------------------------------


@pytest.mark.parametrize("name", list(family.FORBIDDEN_NULLS))
def test_forbidden_nulls_are_refused(name: str) -> None:
    with pytest.raises(surrogate.SurrogateError, match="forbidden null"):
        surrogate.assert_null_is_permitted(name)


def test_a_permitted_null_passes() -> None:
    surrogate.assert_null_is_permitted("matched_surrogate")


def test_an_empty_corpus_is_refused() -> None:
    with pytest.raises(surrogate.SurrogateError):
        surrogate.matched_surrogate((), seed=1)


# --- the whole feature path must survive a surrogate -------------------------


@pytest.mark.skipif(
    not Path(family.ES_BARS_ROOT).is_dir(), reason="owned ES bars not present"
)
def test_features_rebuild_from_a_real_surrogate_corpus() -> None:
    """The identical feature code must run on rebuilt paths.

    This is the property the campaign depends on: features, selections and
    targets are re-derived, never reshuffled.
    """

    real = loader.load_sessions()
    fake, audit = surrogate.matched_surrogate(real, seed=2026)
    assert audit.sessions == len(real)
    # Reported rather than absorbed; the real corpus should be well behaved.
    assert audit.range_violation_rate < 0.01, audit

    features = loader.session_features(fake)
    assert len(features) == len(real)
    assert loader.eligible_sessions(features, "M1").sum() == family.M1_ELIGIBLE_SESSIONS
    assert (
        loader.eligible_sessions(features, "M3").sum() == family.GAP_ELIGIBLE_SESSIONS
    )
