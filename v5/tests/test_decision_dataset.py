"""Could the bot have known this in the minute it decided?

The 2026-08-13 leak was one minute wide, correlated +0.19 with the outcome, and
lifted a headline to +$119/trade with an interval that cleared zero. The
shuffled-label null did not catch it and structurally cannot. The control that
works is this one: mutate everything after the decision minute and assert that
no feature moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import build_decision_dataset as bdd

FEATURE_COLUMNS = [
    "moneyness",
    "moneyness_rel",
    "entry_premium",
    "entry_premium_rel",
    "minutes_to_expiry",
    "contract_volume_5m",
    "contract_volume_15m",
    "contract_share_of_chain",
    "chain_volume_15m",
    "chain_call_share_15m",
    "round_trip_usd",
    "range_position",
    "session_range_rel",
    "realised_vol_30m",
    "minutes_since_open",
    "minutes_to_close",
    "iv",
    "delta",
    "gamma_dollars",
    "theta_share",
    "vega_rel",
] + [f"move_{b}m_rel" for b in bdd.LOOKBACKS] + [f"range_{b}m_rel" for b in bdd.LOOKBACKS]


def _synthetic_session(path, *, seed: int, after_minute: str | None = None,
                       scale: float = 1.0):
    """A whole session of a small chain, optionally deformed after a minute."""

    rng = np.random.default_rng(seed)
    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("16:00") + 1)]
    strikes = np.arange(5000.0, 5105.0, 5.0)
    spot = 5050.0 + np.cumsum(rng.normal(0.0, 0.6, len(minutes)))
    rows = []
    for i, minute in enumerate(minutes):
        bump = scale if after_minute and minute > after_minute else 1.0
        left = max(bdd._index("16:00") - bdd._index(minute), 1)
        for strike in strikes:
            for right in ("C", "P"):
                money = (spot[i] - strike) if right == "C" else (strike - spot[i])
                value = max(money, 0.0) + 6.0 * np.sqrt(left / 390.0)
                rows.append(
                    {
                        "ts_event": pd.Timestamp(f"2024-06-03 {minute}", tz="America/New_York"),
                        "close": float(value * bump),
                        "high": float(value * bump),
                        "low": float(value * bump),
                        "volume": float(rng.integers(1, 400) * bump),
                        "strike": float(strike),
                        "right": right,
                    }
                )
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_no_feature_reads_a_minute_after_the_decision(tmp_path) -> None:
    """The regression that matters. Deform the future; features must not move."""

    clean = _synthetic_session(tmp_path / "clean.parquet", seed=1)
    dirty = _synthetic_session(
        tmp_path / "dirty.parquet", seed=1, after_minute="12:00", scale=4.0
    )

    a = bdd.session_rows(clean)
    b = bdd.session_rows(dirty)
    assert a is not None and b is not None

    early = a["entry_minute"] <= "12:00"
    assert early.sum() > 0
    for column in FEATURE_COLUMNS:
        left = a.loc[early, column].to_numpy(float)
        right = b.loc[early, column].to_numpy(float)
        assert np.allclose(left, right, equal_nan=True), f"{column} reads the future"


def test_the_outcome_does_move_when_the_future_moves(tmp_path) -> None:
    """The other half: if nothing moved, the test above would prove nothing."""

    clean = _synthetic_session(tmp_path / "clean.parquet", seed=2)
    dirty = _synthetic_session(
        tmp_path / "dirty.parquet", seed=2, after_minute="12:00", scale=4.0
    )
    a, b = bdd.session_rows(clean), bdd.session_rows(dirty)
    early = a["entry_minute"] <= "12:00"
    assert not np.allclose(
        a.loc[early, f"gross_{bdd.LABEL_HORIZON}m"].to_numpy(float),
        b.loc[early, f"gross_{bdd.LABEL_HORIZON}m"].to_numpy(float),
        equal_nan=True,
    )


def test_cost_scales_with_the_contract_and_matches_the_measured_points() -> None:
    got = bdd.round_trip_cost(np.array(bdd.COST_PREMIUM_USD))
    assert np.allclose(got, bdd.COST_ROUND_TRIP_USD)
    # A flat cost is what lets a model dodge the toll by buying cheap paper.
    assert bdd.round_trip_cost(np.array([35.0]))[0] < bdd.round_trip_cost(np.array([1945.0]))[0]
    assert bdd.round_trip_cost(np.array([1945.0]))[0] < bdd.round_trip_cost(np.array([6778.0]))[0]


def test_cost_never_goes_below_the_measured_fee_floor() -> None:
    tiny = bdd.round_trip_cost(np.array([0.01, 1.0, 10.0]))
    assert (tiny >= 9.0 - 1e-9).all(), "interpolation must not extrapolate below the floor"


def test_the_chain_is_wider_than_the_charter_band(tmp_path) -> None:
    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=3))
    assert got["moneyness"].abs().max() > 25.0
    assert got["moneyness"].abs().max() <= bdd.MONEYNESS_BAND_POINTS


def test_both_sides_of_the_chain_are_offered(tmp_path) -> None:
    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=4))
    assert got["is_call"].any() and (~got["is_call"]).any()


def test_decision_minutes_respect_the_declared_grid(tmp_path) -> None:
    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=5))
    offsets = {
        (bdd._index(m) - bdd.FIRST_DECISION) % bdd.DECISION_EVERY
        for m in got["entry_minute"].unique()
    }
    assert offsets == {0}
    assert bdd._index(got["entry_minute"].min()) >= bdd.FIRST_DECISION
    assert bdd._index(got["entry_minute"].max()) <= bdd.LAST_DECISION


def test_path_extremes_bracket_the_horizon_outcome(tmp_path) -> None:
    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=6))
    inside = got.dropna(subset=[f"gross_{bdd.LABEL_HORIZON}m", "path_max", "path_min"])
    assert (inside["path_max"] >= inside[f"gross_{bdd.LABEL_HORIZON}m"] - 1e-6).all()
    assert (inside["path_min"] <= inside[f"gross_{bdd.LABEL_HORIZON}m"] + 1e-6).all()


def test_a_contract_that_did_not_print_is_not_offered(tmp_path) -> None:
    """Tradeability at the decision minute is knowable; the outcome is not."""

    path = _synthetic_session(tmp_path / "s.parquet", seed=7)
    frame = pd.read_parquet(path)
    gone = (frame["strike"] == 5050.0) & (frame["right"] == "C")
    frame.loc[gone & (frame["ts_event"].dt.strftime("%H:%M") == "10:00"), "close"] = np.nan
    frame.to_parquet(path)

    got = bdd.session_rows(path)
    at_ten = got[got["entry_minute"] == "10:00"]
    assert not ((at_ten["strike"] == 5050.0) & (at_ten["is_call"])).any()


def test_path_features_refuse_rather_than_guess_without_history() -> None:
    spot = pd.Series(
        {bdd._label(i): 5000.0 for i in range(bdd._index("09:30"), bdd._index("09:40"))}
    )
    assert bdd.path_features(spot, "09:35") is None


def test_parity_price_inverts_the_call_put_relation() -> None:
    columns = pd.MultiIndex.from_tuples(
        [(5000.0, "C"), (5000.0, "P")], names=["strike", "right"]
    )
    prices = pd.DataFrame([[30.0, 10.0]], index=["10:00"], columns=columns)
    spot = pd.Series({"10:00": 5020.0})
    got = bdd.parity_prices(prices, spot, np.array([5000.0, 5000.0]),
                            np.array([True, False]))
    # C_fair = P + S - K = 10 + 20 = 30 ; P_fair = C - S + K = 30 - 20 = 10
    assert got.iloc[0, 0] == pytest.approx(30.0)
    assert got.iloc[0, 1] == pytest.approx(10.0)


def test_parity_price_is_nan_without_a_twin() -> None:
    columns = pd.MultiIndex.from_tuples([(5000.0, "C")], names=["strike", "right"])
    prices = pd.DataFrame([[30.0]], index=["10:00"], columns=columns)
    got = bdd.parity_prices(prices, pd.Series({"10:00": 5020.0}),
                            np.array([5000.0]), np.array([True]))
    assert np.isnan(got.iloc[0, 0])


def test_parity_residual_is_zero_when_prints_agree_with_parity(tmp_path) -> None:
    """The synthetic chain is built to satisfy parity, so residuals must vanish."""

    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=11))
    assert got["parity_residual"].abs().median() < 1e-6


def test_fair_scoring_is_present_for_every_horizon(tmp_path) -> None:
    got = bdd.session_rows(_synthetic_session(tmp_path / "s.parquet", seed=12))
    for horizon in bdd.HORIZONS:
        assert f"gross_fair_{horizon}m" in got.columns


def test_fair_columns_do_not_read_the_future(tmp_path) -> None:
    clean = _synthetic_session(tmp_path / "clean.parquet", seed=13)
    dirty = _synthetic_session(
        tmp_path / "dirty.parquet", seed=13, after_minute="12:00", scale=4.0
    )
    a, b = bdd.session_rows(clean), bdd.session_rows(dirty)
    early = a["entry_minute"] <= "12:00"
    for column in ("entry_fair_premium", "parity_residual"):
        assert np.allclose(
            a.loc[early, column].to_numpy(float),
            b.loc[early, column].to_numpy(float),
            equal_nan=True,
        ), f"{column} reads the future"


def test_fixed_lookbacks_cannot_score_the_first_hour() -> None:
    """The blind spot this flag exists to remove, asserted so it cannot return."""

    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("11:00"))]
    spot = pd.Series({m: 5000.0 + i * 0.1 for i, m in enumerate(minutes)})
    # The magic minute the hypothesis is about cannot be scored at all.
    assert bdd.path_features(spot, "10:00") is None
    assert bdd.path_features(spot, "09:45") is None
    # With a perfectly dense series 10:30 has exactly 61 points and just clears;
    # in the real corpus sparse minutes push the true first entry to 10:35.
    assert bdd.path_features(spot, "10:30") is not None


def test_adaptive_lookbacks_score_the_magic_minute() -> None:
    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("11:00"))]
    spot = pd.Series({m: 5000.0 + i * 0.1 for i, m in enumerate(minutes)})
    for minute in ("09:35", "09:45", "10:00", "10:30"):
        got = bdd.path_features(spot, minute, adaptive=True)
        assert got is not None, minute
        assert np.isfinite(got["range_60m_rel"])
        assert np.isfinite(got["move_5m_rel"])


def test_adaptive_still_refuses_before_the_minimum_history() -> None:
    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("09:34"))]
    spot = pd.Series({m: 5000.0 for m in minutes})
    assert bdd.path_features(spot, "09:33", adaptive=True) is None


def test_history_minutes_records_the_truncation() -> None:
    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("12:00"))]
    spot = pd.Series({m: 5000.0 + i * 0.1 for i, m in enumerate(minutes)})
    early = bdd.path_features(spot, "09:45", adaptive=True)
    late = bdd.path_features(spot, "11:30", adaptive=True)
    assert early["history_minutes"] < late["history_minutes"]
    assert late["history_minutes"] >= max(bdd.LOOKBACKS) + 1


def test_adaptive_reads_only_prior_minutes() -> None:
    """Adaptive windows must stay causal; that is the whole constraint."""

    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("12:00"))]
    base = {m: 5000.0 + i * 0.1 for i, m in enumerate(minutes)}
    clean = bdd.path_features(pd.Series(base), "10:00", adaptive=True)
    deformed = dict(base)
    for m in minutes:
        if m > "10:00":
            deformed[m] = 9999.0
    dirty = bdd.path_features(pd.Series(deformed), "10:00", adaptive=True)
    assert clean == dirty


def test_adaptive_matches_fixed_once_history_is_complete() -> None:
    minutes = [bdd._label(i) for i in range(bdd._index("09:30"), bdd._index("13:00"))]
    spot = pd.Series({m: 5000.0 + np.sin(i / 7.0) for i, m in enumerate(minutes)})
    fixed = bdd.path_features(spot, "12:00")
    adaptive = bdd.path_features(spot, "12:00", adaptive=True)
    for key in fixed:
        assert fixed[key] == pytest.approx(adaptive[key]), key
