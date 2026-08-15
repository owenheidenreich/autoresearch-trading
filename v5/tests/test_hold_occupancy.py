"""The rebuilt hold-length and occupancy table.

Synthetic sessions only, so the suite stays green with the evidence drive
unmounted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import measure_hold_occupancy as ho


STRIKES = [5875.0, 5880.0, 5885.0, 5890.0, 5895.0, 5900.0, 5905.0, 5910.0, 5915.0]


def _chain(minute: str, spot: float, *, skip: set[tuple[float, str]] = frozenset()) -> list[dict]:
    """A chain whose put/call parity gap is smallest exactly at ``spot``.

    ``C - P`` equals ``spot - strike`` by construction, so ``idxmin`` of the
    absolute gap lands on ``spot`` and the fixture never depends on a numerical
    accident.
    """

    rows = []
    for strike in STRIKES:
        for right in ("C", "P"):
            if (strike, right) in skip:
                continue
            intrinsic = max(0.0, spot - strike) if right == "C" else max(0.0, strike - spot)
            rows.append(
                {
                    "ts_event": pd.Timestamp(
                        f"2025-01-02 {minute}", tz="America/New_York"
                    ).tz_convert("UTC"),
                    "close": intrinsic + 10.0,
                    "strike": strike,
                    "right": right,
                }
            )
    return rows


def _session(tmp_path, rows: list[dict], name: str = "2025-01-02"):
    path = tmp_path / f"{name}.spxw_0dte.ohlcv-1m.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------


def test_the_tradeable_window_is_the_declared_one() -> None:
    assert ho.FIRST_ENTRY_MINUTE == "09:35"
    assert ho.LAST_EXIT_MINUTE == "16:00"
    assert ho.TRADEABLE_MINUTES == 385


@pytest.mark.parametrize("minute", ["09:35", "10:00", "16:00", "00:00"])
def test_minute_labels_round_trip(minute: str) -> None:
    assert ho._label(ho._index(minute)) == minute


def test_the_clock_offers_the_theoretical_maximum_number_of_slots(tmp_path) -> None:
    """385 minutes of runway, one position at a time: 385 // hold slots."""

    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5905.0))
    for hold, expected in ((5, 77), (15, 25), (60, 6)):
        _, offered, _ = ho.session_trades(path, hold)
        assert offered == expected, hold


# --------------------------------------------------------------------------
# pricing a slot
# --------------------------------------------------------------------------


def test_session_pivot_locates_the_spot_from_parity(tmp_path) -> None:
    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5905.0))
    _, spot = ho.session_pivot(path)
    assert spot["09:35"] == pytest.approx(5900.0, abs=1e-6)
    assert spot["09:40"] == pytest.approx(5905.0, abs=1e-6)


def test_a_chain_too_thin_to_place_a_spot_is_refused(tmp_path) -> None:
    thin = [row for row in _chain("09:35", 5900.0) if row["strike"] >= 5905.0]
    assert ho.session_pivot(_session(tmp_path, thin)) is None


def test_a_minute_where_no_strike_printed_both_sides_leaves_the_spot_missing(
    tmp_path,
) -> None:
    """``idxmin`` raises on an all-missing row; a real session contains them."""

    calls_only = [row for row in _chain("09:40", 5905.0) if row["right"] == "C"]
    _, spot = ho.session_pivot(_session(tmp_path, _chain("09:35", 5900.0) + calls_only))
    assert spot["09:35"] == pytest.approx(5900.0)
    assert pd.isna(spot["09:40"])


def test_a_slot_whose_minutes_are_missing_is_skipped_and_counted(tmp_path) -> None:
    # Only the 09:35 and 09:40 minutes exist, so exactly one 5-minute slot prices.
    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5905.0))
    trades, offered, skipped = ho.session_trades(path, 5)
    assert offered == 77
    assert skipped["minute_absent"] == 76
    assert sum(skipped.values()) == 76
    assert set(trades["entry_minute"]) == {"09:35"}


def test_a_small_move_is_kept(tmp_path) -> None:
    """Regression: skipping small moves conditioned on the outcome.

    The earlier estimator quantised the underlying to the five-point strike
    grid and skipped any slot whose spot did not change. Measured on the real
    corpus that discarded 18.7% of sixty-minute slots averaging -$186.90 each —
    exactly the slots where the underlying went nowhere and long premium decays.
    A half-point move must survive.
    """

    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5900.5))
    trades, _, skipped = ho.session_trades(path, 5)
    assert trades is not None
    assert skipped["exact_tie"] == 0
    assert set(trades["entry_minute"]) == {"09:35"}


def test_a_genuinely_unchanged_spot_is_the_only_tie_skipped(tmp_path) -> None:
    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5900.0))
    trades, _, skipped = ho.session_trades(path, 5)
    assert trades is None
    assert skipped["exact_tie"] == 1
    assert skipped["minute_absent"] == 76


def test_the_spot_estimate_is_continuous_not_snapped_to_the_grid(tmp_path) -> None:
    _, spot = ho.session_pivot(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5903.7))
    )
    assert spot["09:35"] == pytest.approx(5900.0, abs=1e-6)
    assert spot["09:40"] == pytest.approx(5903.7, abs=1e-6)


# --------------------------------------------------------------------------
# the settled exit-price convention
# --------------------------------------------------------------------------


def test_a_contract_that_stops_printing_is_valued_at_its_last_print(tmp_path) -> None:
    """The Phase-0 settlement, wired into the occupancy table."""

    rows = (
        _chain("09:35", 5900.0)
        + _chain("09:37", 5903.0)
        # The 5895 call goes quiet for the exit minute after printing at 09:37.
        + _chain("09:40", 5905.0, skip={(5895.0, "C")})
    )
    trades, _, _ = ho.session_trades(_session(tmp_path, rows), 5)
    quiet = trades[
        (trades["entry_minute"] == "09:35") & trades["vanished"]
    ]
    assert len(quiet) == 1
    # Its 09:37 print: intrinsic 5903 - 5895 = 8, plus the fixture's flat 10.
    assert quiet.iloc[0]["exit_close"] == pytest.approx(18.0)
    # Not the exit-minute value it never printed, which would have been 20.
    assert quiet.iloc[0]["correct"]


def test_the_slot_picks_the_nearest_contract_on_each_side(tmp_path) -> None:
    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5905.0))
    trades, _, _ = ho.session_trades(path, 5)
    chosen = trades[trades["chosen"]]
    assert len(chosen) == 2
    # One right-side and one wrong-side contract, which is the conditional pair
    # a break-even needs.
    assert set(chosen["correct"]) == {True, False}


def test_only_the_near_atm_band_is_eligible(tmp_path) -> None:
    path = _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5905.0))
    trades, _, _ = ho.session_trades(path, 5)
    # |moneyness| <= 25 keeps 5875..5915 against a 5900 spot, so every fixture
    # strike qualifies on both sides.
    assert len(trades) == 2 * len(STRIKES)


# --------------------------------------------------------------------------
# the declared stop
# --------------------------------------------------------------------------


def test_the_stop_is_the_level_the_signed_amendment_names() -> None:
    assert ho.STOP_LEVEL == -0.30


def test_a_path_that_never_breaches_rides_to_the_horizon() -> None:
    path = np.array([10.0, 9.5, 8.0, 7.5])
    assert ho._stopped_exit(path, entry=10.0, held=7.5) == pytest.approx(7.5)


def test_the_stop_takes_the_fill_available_not_the_level_declared() -> None:
    """A 0DTE contract can fall through a level between one minute and the next."""

    path = np.array([10.0, 9.0, 5.5, 8.0])
    # -45% is the first print at or below -30%, and it is what the exit gets.
    assert ho._stopped_exit(path, entry=10.0, held=8.0) == pytest.approx(5.5)


def test_the_stop_cannot_fire_on_the_price_it_bought_at() -> None:
    """The entry bar is not an opportunity to exit."""

    path = np.array([10.0, 12.0])
    assert ho._stopped_exit(path, entry=20.0, held=12.0) == pytest.approx(12.0)


def test_a_single_minute_path_rides_to_the_horizon() -> None:
    assert ho._stopped_exit(np.array([10.0]), entry=10.0, held=3.0) == pytest.approx(3.0)


def test_session_trades_records_the_stopped_exit_only_for_the_traded_contracts(
    tmp_path,
) -> None:
    rows = (
        _chain("09:35", 5900.0)
        # A hard move down mid-hold, then a partial recovery by the exit.
        + _chain("09:37", 5860.0)
        + _chain("09:40", 5905.0)
    )
    trades, _, _ = ho.session_trades(_session(tmp_path, rows), 5)
    assert trades[~trades["chosen"]]["stopped_close"].isna().all()
    chosen = trades[trades["chosen"]]
    assert chosen["stopped_close"].notna().all()
    # The nearest put to a 5900 spot is worth 10 at entry and 55 at 09:37, so it
    # never breaches and rides to its 09:40 value; the nearest call is worth 10
    # at entry and 10 at 09:37, so it does not breach either.
    assert (chosen["stopped_close"] == chosen["exit_close"]).all()


def test_a_contract_that_collapses_mid_hold_exits_at_the_stop(tmp_path) -> None:
    mid = _chain("09:37", 5900.0)
    for row in mid:
        if row["strike"] == 5900.0 and row["right"] == "C":
            row["close"] = 5.0  # half its $10 entry, well through -30%
    trades, _, _ = ho.session_trades(
        _session(tmp_path, _chain("09:35", 5900.0) + mid + _chain("09:40", 5905.0)), 5
    )
    call = trades[trades["chosen"] & trades["correct"]].iloc[0]
    # It rides to $15 without a stop and exits at the $5 print with one.
    assert call["exit_close"] == pytest.approx(15.0)
    assert call["stopped_close"] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# how much a session's trades really tell you
# --------------------------------------------------------------------------


def test_perfectly_persistent_direction_collapses_to_one_observation() -> None:
    groups = np.repeat(np.arange(50), 8)
    values = np.repeat(np.arange(50) % 2, 8).astype(float)
    icc, k0 = ho.intraclass_correlation(values, groups)
    assert icc == pytest.approx(1.0)
    assert k0 == pytest.approx(8.0)
    assert 1.0 + (k0 - 1.0) * icc == pytest.approx(8.0)


def test_direction_that_is_a_fresh_coin_flip_costs_nothing() -> None:
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(400), 8)
    values = rng.integers(0, 2, size=groups.size).astype(float)
    icc, _ = ho.intraclass_correlation(values, groups)
    assert icc < 0.02


def test_a_negative_correlation_is_reported_as_no_saving() -> None:
    """Anti-correlated groups cannot buy back more than independence."""

    groups = np.repeat(np.arange(60), 2)
    values = np.tile([0.0, 1.0], 60)
    icc, _ = ho.intraclass_correlation(values, groups)
    assert icc == 0.0


def test_too_little_data_for_a_correlation_returns_no_penalty() -> None:
    assert ho.intraclass_correlation(np.array([1.0]), np.array(["a"])) == (0.0, 1.0)


# --------------------------------------------------------------------------
# the assembled row
# --------------------------------------------------------------------------


def _trade_rows(sessions: int, slots: int, win: float, loss: float) -> pd.DataFrame:
    rows = []
    for s in range(sessions):
        for slot in range(slots):
            up = (s + slot) % 2 == 0
            for correct, exit_close in ((True, 10.0 + win), (False, 10.0 - loss)):
                rows.append(
                    {
                        "session": f"2025-01-{s + 1:02d}",
                        "entry_minute": f"09:{35 + slot:02d}",
                        "entry_close": 10.0,
                        "exit_close": exit_close,
                        "stopped_close": exit_close,
                        "correct": correct,
                        "chosen": True,
                        "up": up,
                        "vanished": False,
                    }
                )
    return pd.DataFrame(rows)


def test_assess_counts_one_trade_per_slot_not_one_per_contract() -> None:
    trades = _trade_rows(sessions=30, slots=4, win=6.0, loss=3.0)
    got = ho.assess(trades, offered=30 * 5, skipped={"minute_absent": 30}, sessions=30)
    assert got["trades_total"] == 120
    assert got["contract_observations"] == 240
    assert got["trades_per_session_measured"] == pytest.approx(4.0)
    assert got["slots_offered_per_session"] == pytest.approx(5.0)
    assert got["slots_skipped_per_session"] == pytest.approx(1.0)


def test_assess_reports_the_break_even_the_payoffs_imply() -> None:
    trades = _trade_rows(sessions=30, slots=4, win=6.0, loss=3.0)
    got = ho.assess(trades, offered=120, skipped={}, sessions=30)
    # +$600 and -$300 gross, each less the $25 round trip.
    assert got["trade_level"]["mean_net_when_correct_usd"] == pytest.approx(575.0)
    assert got["trade_level"]["mean_net_when_wrong_usd"] == pytest.approx(-325.0)
    assert got["detectable"]["breakeven_accuracy"] == pytest.approx(
        325.0 / 900.0, abs=1e-6
    )


def test_a_shorter_hold_is_only_worth_it_through_the_effective_sample() -> None:
    """More trades lower the provable bar, but only after the design effect."""

    few = ho.assess(_trade_rows(30, 2, 6.0, 3.0), offered=60, skipped={}, sessions=30)
    many = ho.assess(_trade_rows(30, 8, 6.0, 3.0), offered=240, skipped={}, sessions=30)
    assert (
        many["detectable"]["accuracy_at_effective_n"]
        < few["detectable"]["accuracy_at_effective_n"]
    )
    for row in (few, many):
        det = row["detectable"]
        assert det["effective_trades"] <= row["trades_total"]
        assert (
            det["accuracy_if_trades_were_independent"]
            <= det["accuracy_at_effective_n"]
            <= det["accuracy_at_effective_n_with_conjunction_penalty"]
        )


def test_a_payoff_that_never_wins_is_reported_as_not_measurable() -> None:
    trades = _trade_rows(sessions=30, slots=4, win=0.1, loss=3.0)
    got = ho.assess(trades, offered=120, skipped={}, sessions=30)
    assert got["detectable"] is None


# --------------------------------------------------------------------------
# summarising an outcome distribution for a downstream risk check
# --------------------------------------------------------------------------


def test_strata_means_reproduce_the_true_mean_exactly() -> None:
    """The property the whole risk chain depends on."""

    rng = np.random.default_rng(4)
    values = rng.lognormal(0.0, 1.5, size=10_000) - 1.0
    strata = np.array(ho.strata_means(values))
    assert strata.mean() == pytest.approx(values.mean(), rel=1e-3)


def test_strata_keep_the_tail_a_quantile_grid_throws_away() -> None:
    """The defect this replaced: a heavy right tail vanishes from a quantile grid."""

    rng = np.random.default_rng(9)
    values = rng.lognormal(0.0, 2.0, size=20_000)
    quantile_grid = np.quantile(values, ho.OUTCOME_QUANTILES)
    strata = np.array(ho.strata_means(values))
    assert quantile_grid.mean() < 0.9 * values.mean()
    assert strata.mean() == pytest.approx(values.mean(), rel=1e-3)
    assert strata.max() > quantile_grid.max()


def test_strata_handle_a_sample_smaller_than_the_stratum_count() -> None:
    assert ho.strata_means(np.array([1.0, 3.0])) == [1.0, 3.0]


def test_strata_of_an_empty_sample_are_empty() -> None:
    assert ho.strata_means(np.array([])) == []


def test_strata_are_ordered_because_the_sample_is_sorted_first() -> None:
    got = ho.strata_means(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), strata=5)
    assert got == sorted(got)
