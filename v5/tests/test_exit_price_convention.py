"""The Phase-0 measurement that settles which exit price a vanished contract gets.

These run on synthetic frames, never on the external corpora, so the suite stays
green with the evidence drive unmounted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import resolve_exit_price_convention as rx


def _entries(rows: list[dict]) -> pd.DataFrame:
    """A table shaped like ``session_entries`` output, with sane defaults."""

    base = {
        "session": "2025-01-02",
        "symbol": "SPXW  250102C05900000",
        "strike": 5900.0,
        "right": "C",
        "moneyness": 0.0,
        "premium": 1000.0,
        "correct": True,
        "entry_close": 10.0,
        "exit_close": np.nan,
        "last_close": 10.0,
        "last_minute": rx.EXIT_MINUTE,
        "intrinsic_exit": 0.0,
        "spot_entry": 5900.0,
        "spot_exit": 5910.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


CORRECT_SYMBOL = "SPXW  250102C05900000"
WRONG_SYMBOL = "SPXW  250102P05900000"
VANISHED_SYMBOL = "SPXW  250102P05905000"


def _balanced(correct_exit: float, wrong_exit: float, n: int = 60) -> pd.DataFrame:
    """``n`` right-side and ``n`` wrong-side contracts, all present at the exit."""

    return _entries(
        [
            {
                "correct": bool(i < n),
                "symbol": CORRECT_SYMBOL if i < n else WRONG_SYMBOL,
                "exit_close": correct_exit if i < n else wrong_exit,
            }
            for i in range(2 * n)
        ]
    )


# --------------------------------------------------------------------------
# break-even arithmetic
# --------------------------------------------------------------------------


def test_breakeven_is_the_loss_share_of_the_two_outcomes() -> None:
    table = _balanced(correct_exit=20.0, wrong_exit=5.0)
    got = rx.breakeven(
        table["entry_close"].to_numpy(float),
        table["exit_close"].to_numpy(float),
        table["correct"].to_numpy(bool),
    )
    # +$1000 gross less $25, and -$500 gross less $25.
    assert got["mean_net_when_correct_usd"] == pytest.approx(975.0)
    assert got["mean_net_when_wrong_usd"] == pytest.approx(-525.0)
    assert got["breakeven"] == pytest.approx(525.0 / 1500.0)


def test_breakeven_refuses_a_sample_too_small_to_mean_anything() -> None:
    assert rx.breakeven(*[np.ones(10)] * 2, np.ones(10, bool))["breakeven"] is None


def test_breakeven_reports_nothing_when_the_right_side_still_loses_money() -> None:
    """A convention that makes even correct calls unprofitable has no break-even."""

    table = _balanced(correct_exit=10.1, wrong_exit=0.0)
    got = rx.breakeven(
        table["entry_close"].to_numpy(float),
        table["exit_close"].to_numpy(float),
        table["correct"].to_numpy(bool),
    )
    assert got["mean_net_when_correct_usd"] < 0
    assert got["breakeven"] is None


# --------------------------------------------------------------------------
# the four conventions, and that they really do differ
# --------------------------------------------------------------------------


def test_the_conventions_separate_exactly_on_the_vanished_contracts() -> None:
    present = _balanced(correct_exit=20.0, wrong_exit=5.0)
    # Twenty wrong-side contracts that stop printing on the way down, last seen
    # at $2 against a $10 entry and worth nothing at expiry: the dying contract
    # the whole question is about.
    gone = _entries(
        [
            {
                "correct": False,
                "symbol": VANISHED_SYMBOL,
                "exit_close": np.nan,
                "last_close": 2.0,
                "last_minute": "10:02",
                "intrinsic_exit": 0.0,
            }
        ]
        * 20
    )
    conv = rx.conventions(pd.concat([present, gone], ignore_index=True))

    assert conv["drop_vanished"]["observations"] == 120
    for name in ("last_print", "mark_to_zero", "mark_to_intrinsic"):
        assert conv[name]["observations"] == 140

    # Dropping them hides the worst losses entirely, so it is the most
    # flattering; believing the stale price is next; assuming they died is worst.
    assert conv["drop_vanished"]["breakeven"] == pytest.approx(525.0 / 1500.0)
    assert conv["last_print"]["breakeven"] == pytest.approx(600.0 / 1575.0)
    assert conv["mark_to_zero"]["breakeven"] == pytest.approx(650.0 / 1625.0)
    # Zero and intrinsic coincide when the contracts expired worthless anyway.
    assert conv["mark_to_intrinsic"]["breakeven"] == pytest.approx(
        conv["mark_to_zero"]["breakeven"]
    )


def test_every_convention_states_what_it_assumes() -> None:
    conv = rx.conventions(_balanced(correct_exit=20.0, wrong_exit=5.0))
    for row in conv.values():
        assert row["meaning"].strip()


def test_the_conventions_agree_when_no_contract_vanishes() -> None:
    conv = rx.conventions(_balanced(correct_exit=20.0, wrong_exit=5.0))
    values = {row["breakeven"] for row in conv.values()}
    assert len(values) == 1


# --------------------------------------------------------------------------
# who the vanished contracts are
# --------------------------------------------------------------------------


def test_vanished_profile_separates_a_stale_price_from_a_dead_contract() -> None:
    table = pd.concat(
        [
            _balanced(correct_exit=20.0, wrong_exit=5.0),
            _entries(
                [
                    {"correct": False, "exit_close": np.nan, "last_close": 5.0,
                     "last_minute": rx.ENTRY_MINUTE},
                    {"correct": False, "exit_close": np.nan, "last_close": 2.0,
                     "last_minute": "10:11"},
                ]
            ),
        ],
        ignore_index=True,
    )
    got = rx.vanished_profile(table)
    assert got["vanished_n"] == 2
    assert got["share_of_vanished_that_never_traded_again"] == pytest.approx(0.5)
    assert got["share_of_vanished_on_the_correct_side"] == 0.0
    # $10 entry, last prints of $5 and $2.
    assert got["mean_last_print_vs_entry_pct"] == pytest.approx(-65.0)


def test_vanished_profile_says_so_when_nothing_vanished() -> None:
    assert rx.vanished_profile(_balanced(20.0, 5.0)) == {"vanished_n": 0}


def test_by_year_splits_on_the_session_date_and_keeps_every_entry() -> None:
    early = _balanced(correct_exit=20.0, wrong_exit=5.0)
    early["session"] = "2022-06-01"
    late = _balanced(correct_exit=18.0, wrong_exit=6.0)
    late["session"] = "2025-01-02"
    got = rx.by_year(pd.concat([early, late], ignore_index=True))

    assert set(got) == {"2022", "2025"}
    assert sum(row["entries"] for row in got.values()) == 240
    assert got["2022"]["sessions"] == 1
    # No contract vanished in either year, so the two conventions coincide.
    assert got["2022"]["breakeven_drop_vanished"] == got["2022"]["breakeven_last_print"]
    assert got["2022"]["breakeven_last_print"] != got["2025"]["breakeven_last_print"]
    assert got["2025"]["share_of_vanished_on_the_correct_side"] is None


# --------------------------------------------------------------------------
# reading the entry population out of a trade corpus file
# --------------------------------------------------------------------------


def _bar(minute: str, strike: float, right: str, close: float) -> dict:
    return {
        "ts_event": pd.Timestamp(f"2025-01-02 {minute}", tz="America/New_York").tz_convert(
            "UTC"
        ),
        "close": close,
        "symbol": f"SPXW  250102{right}0{int(strike * 1000):08d}",
        "strike": strike,
        "right": right,
    }


def _corpus_file(tmp_path, bars: list[dict]):
    path = tmp_path / "2025-01-02.spxw_0dte.ohlcv-1m.parquet"
    pd.DataFrame(bars).to_parquet(path)
    return path


def test_session_entries_marks_a_contract_with_no_exit_print_as_vanished(tmp_path) -> None:
    # Parity puts spot at 5900 on entry and 5910 on exit, so calls are correct.
    bars = [
        _bar(rx.ENTRY_MINUTE, 5900.0, "C", 10.0),
        _bar(rx.ENTRY_MINUTE, 5900.0, "P", 10.0),
        _bar(rx.ENTRY_MINUTE, 5905.0, "P", 4.0),
        _bar("10:02", 5905.0, "P", 1.5),
        _bar(rx.EXIT_MINUTE, 5910.0, "C", 12.0),
        _bar(rx.EXIT_MINUTE, 5910.0, "P", 12.0),
        _bar(rx.EXIT_MINUTE, 5900.0, "C", 18.0),
    ]
    got = rx.session_entries(_corpus_file(tmp_path, bars)).set_index(["strike", "right"])

    call = got.loc[(5900.0, "C")]
    assert call["correct"] and call["exit_close"] == pytest.approx(18.0)

    put = got.loc[(5905.0, "P")]
    assert not put["correct"]
    assert np.isnan(put["exit_close"])
    # The last print is the 10:02 one, not the entry bar.
    assert put["last_close"] == pytest.approx(1.5)
    assert put["last_minute"] == "10:02"
    # 5905 put against a 5910 exit spot is out of the money.
    assert put["intrinsic_exit"] == pytest.approx(0.0)


def test_session_entries_falls_back_to_the_entry_bar_when_nothing_traded_after(
    tmp_path,
) -> None:
    bars = [
        _bar(rx.ENTRY_MINUTE, 5900.0, "C", 10.0),
        _bar(rx.ENTRY_MINUTE, 5900.0, "P", 10.0),
        _bar(rx.ENTRY_MINUTE, 5890.0, "C", 16.0),
        _bar(rx.EXIT_MINUTE, 5910.0, "C", 12.0),
        _bar(rx.EXIT_MINUTE, 5910.0, "P", 12.0),
    ]
    got = rx.session_entries(_corpus_file(tmp_path, bars)).set_index(["strike", "right"])
    row = got.loc[(5890.0, "C")]
    assert row["last_minute"] == rx.ENTRY_MINUTE
    assert row["last_close"] == pytest.approx(16.0)
    # 5890 call against a 5910 exit spot holds $20 of exercise value.
    assert row["intrinsic_exit"] == pytest.approx(20.0)


def test_session_entries_keeps_only_the_near_atm_band(tmp_path) -> None:
    bars = [
        _bar(rx.ENTRY_MINUTE, 5900.0, "C", 10.0),
        _bar(rx.ENTRY_MINUTE, 5900.0, "P", 10.0),
        _bar(rx.ENTRY_MINUTE, 5700.0, "C", 205.0),
        _bar(rx.EXIT_MINUTE, 5910.0, "C", 12.0),
        _bar(rx.EXIT_MINUTE, 5910.0, "P", 12.0),
    ]
    got = rx.session_entries(_corpus_file(tmp_path, bars))
    assert set(got["strike"]) == {5900.0}


def test_session_entries_declines_a_session_that_cannot_be_priced(tmp_path) -> None:
    # No exit minute at all.
    assert rx.session_entries(
        _corpus_file(tmp_path, [_bar(rx.ENTRY_MINUTE, 5900.0, "C", 10.0)])
    ) is None


def test_session_entries_survives_a_non_trading_day_file(tmp_path) -> None:
    path = tmp_path / "2025-01-02.spxw_0dte.ohlcv-1m.parquet"
    pd.DataFrame().to_parquet(path)
    assert rx.session_entries(path) is None


# --------------------------------------------------------------------------
# the settlement against real quotes
# --------------------------------------------------------------------------


def test_settlement_measures_how_wrong_the_last_print_was() -> None:
    table = pd.concat(
        [
            _balanced(correct_exit=20.0, wrong_exit=5.0),
            _entries(
                [
                    {
                        "symbol": VANISHED_SYMBOL,
                        "correct": False,
                        "exit_close": np.nan,
                        "last_close": 8.0,
                        "last_minute": "10:02",
                    }
                ]
                * 20
            ),
        ],
        ignore_index=True,
    )
    quotes = pd.DataFrame(
        [
            {"raw_symbol": CORRECT_SYMBOL, "quote_bid": 19.5,
             "quote_ask": 20.5, "quote_mid": 20.0},
            {"raw_symbol": WRONG_SYMBOL, "quote_bid": 4.8,
             "quote_ask": 5.2, "quote_mid": 5.0},
            # The vanished contract was really worth $1, not the $8 it last printed.
            {"raw_symbol": VANISHED_SYMBOL, "quote_bid": 0.8,
             "quote_ask": 1.2, "quote_mid": 1.0},
        ]
    )
    got = rx.settle(table, quotes)
    priced = got["vanished_contracts_priced_by_quotes"]
    assert got["vanished_and_quoted_n"] == 20
    assert priced["mean_assumed_last_print_usd"] == pytest.approx(800.0)
    assert priced["mean_actual_mid_usd"] == pytest.approx(100.0)
    assert priced["mean_error_usd"] == pytest.approx(-700.0)
    assert priced["share_where_last_print_overstates"] == pytest.approx(1.0)
    # The truth must be harsher than the convention that believed the stale price.
    assert got["truth_quote_mid"]["breakeven"] > got["same_population_last_print"]["breakeven"]
    # Correcting only the vanished contracts leaves every other price alone, so
    # here it lands exactly on the all-quotes truth.
    # Break-evens are stored rounded to six places, so compare at that scale.
    assert got["corrected_quote_for_vanished_only"]["breakeven"] == pytest.approx(
        625.0 / 1600.0, abs=1e-6
    )
    assert got["same_population_last_print"]["breakeven"] == pytest.approx(
        450.0 / 1425.0, abs=1e-6
    )
    assert got["same_population_drop_vanished"]["breakeven"] == pytest.approx(
        525.0 / 1500.0, abs=1e-6
    )


def test_settlement_drops_contracts_the_quote_corpus_cannot_price() -> None:
    table = _balanced(correct_exit=20.0, wrong_exit=5.0)
    quotes = pd.DataFrame(
        [
            {"raw_symbol": "SPXW  250102C05900000", "quote_bid": np.nan,
             "quote_ask": np.nan, "quote_mid": np.nan}
        ]
    )
    got = rx.settle(table, quotes)
    assert got["observations"] == 0
    assert got["unquoted_dropped"] == 120


# --------------------------------------------------------------------------
# the declared constants are declared, not discovered
# --------------------------------------------------------------------------


def test_the_measurement_carries_the_numbers_that_produced_the_disagreement() -> None:
    assert (rx.ENTRY_MINUTE, rx.EXIT_MINUTE) == ("09:35", "10:35")
    assert rx.NEAR_ATM_POINTS == 25.0
    assert rx.ROUND_TRIP_USD == 25.0
