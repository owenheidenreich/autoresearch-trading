"""The effective-sample-size measurement that collapsed section 4.

These are known-answer tests on synthetic series. They do not read the corpus,
so they pin the estimator's behaviour rather than the measured result.
"""
from __future__ import annotations

import numpy as np
import pytest

from v5.ops.measure_effective_sample_size import integrated_autocorrelation


def test_independent_noise_has_an_autocorrelation_time_of_about_one() -> None:
    rng = np.random.default_rng(0)
    tau, _ = integrated_autocorrelation(rng.normal(size=4_000), 180)
    assert tau == pytest.approx(1.0, abs=0.35)


def test_a_persistent_series_reports_a_long_autocorrelation_time() -> None:
    """An AR(1) with rho=0.9 has a theoretical tau of (1+rho)/(1-rho) = 19."""

    rng = np.random.default_rng(1)
    rho, n = 0.9, 20_000
    series = np.empty(n)
    series[0] = rng.normal()
    for i in range(1, n):
        series[i] = rho * series[i - 1] + rng.normal()
    tau, _ = integrated_autocorrelation(series, 400)
    assert tau == pytest.approx(19.0, rel=0.25)


def test_a_constant_series_cannot_divide_by_zero() -> None:
    tau, lags = integrated_autocorrelation(np.ones(500), 180)
    assert tau == 1.0
    assert lags == 0


def test_tau_never_returns_below_one() -> None:
    """A tau below 1 would inflate the effective count above the raw count."""

    rng = np.random.default_rng(2)
    # Strong negative lag-1 correlation, which naively sums to less than one.
    series = rng.normal(size=2_000)
    alternating = series * np.where(np.arange(2_000) % 2 == 0, 1.0, -1.0)
    tau, _ = integrated_autocorrelation(alternating, 180)
    assert tau >= 1.0


def test_the_measured_states_are_worth_far_fewer_observations() -> None:
    """The finding's load-bearing claim, as an arithmetic guard.

    A 60-minute forward label sampled every minute cannot carry one independent
    observation per minute. If a future change made tau collapse toward 1 on a
    strongly persistent series, this estimator would be broken.
    """

    rng = np.random.default_rng(3)
    # A binary label held true across a ~60-minute window, sampled per minute.
    # The event rate is kept low enough that the series does not saturate to a
    # constant, which would trip the zero-variance guard instead.
    base = rng.random(4_000) < 0.005
    held = np.convolve(base.astype(float), np.ones(60), mode="same") > 0
    assert 0.05 < held.mean() < 0.95, "synthetic series must not be near-constant"
    tau, _ = integrated_autocorrelation(held.astype(float), 400)
    assert tau > 5.0
