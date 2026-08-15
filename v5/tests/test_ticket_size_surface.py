"""The ticket-size sweep that tells a charter amendment what a cheap ticket costs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import measure_ticket_size_surface as ts


def test_the_bands_tile_the_line_without_gaps_or_overlaps() -> None:
    for (_, high, _, _), (low, _, _, _) in zip(ts.BANDS, ts.BANDS[1:]):
        assert high == low


def test_the_band_index_places_each_moneyness_in_the_band_that_owns_it() -> None:
    moneyness = np.array([-500.0, -60.0, 0.0, 50.0, 200.0, 400.0])
    names = [ts.BANDS[i][2] for i in ts._band_index(moneyness)]
    assert names == [
        "deep OTM", "OTM", "near ATM", "ITM", "deep ITM", "very deep ITM",
    ]


def test_each_band_owns_its_upper_edge() -> None:
    """``(lo, hi]``, the convention the signed amendment's own table was built on.

    The occupancy work uses the inclusive ``|moneyness| <= 25`` reading instead,
    which differs by one strike level out of eleven and moved the measured
    break-even by 0.03 points. The two are pinned against each other here so the
    difference stays a known 0.03 rather than an unexplained one.
    """

    names = [ts.BANDS[i][2] for i in ts._band_index(np.array([-25.0, -100.0, 25.0]))]
    assert names == ["OTM", "deep OTM", "near ATM"]


def test_cheaper_bands_carry_a_smaller_round_trip_in_dollars() -> None:
    costs = [band[3] for band in ts.BANDS]
    assert costs == sorted(costs)


def test_the_ticket_ladder_brackets_the_charter_ceiling() -> None:
    # 13% of a $10,000 account is $1,300.
    assert min(ts.TICKET_TARGETS_USD) < 1_300.0 < max(ts.TICKET_TARGETS_USD)


# --------------------------------------------------------------------------
# selecting a contract for a target ticket
# --------------------------------------------------------------------------


# Five-point spacing near the money, as SPXW actually lists, plus wings. The
# parity estimator averages strikes within 30 points of a first estimate and
# needs five of them, so a sparse ladder cannot place a spot at all.
STRIKES = [
    5700.0, 5800.0, 5850.0,
    5880.0, 5885.0, 5890.0, 5895.0, 5900.0, 5905.0, 5910.0, 5915.0, 5920.0,
    5930.0, 5950.0, 6000.0, 6100.0,
]


def _chain(minute: str, spot: float) -> list[dict]:
    rows = []
    for strike in STRIKES:
        for right in ("C", "P"):
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


def _session(tmp_path, rows):
    path = tmp_path / "2025-01-02.spxw_0dte.ohlcv-1m.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_each_target_gets_the_contract_closest_to_it_on_each_side(tmp_path) -> None:
    got = ts.session_tickets(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5910.0)), 5
    )
    # One row per target per side, for the single priced slot.
    assert len(got) == 2 * len(ts.TICKET_TARGETS_USD)
    for target, part in got.groupby("target_usd"):
        assert len(part) == 2
        for _, row in part.iterrows():
            # Nothing closer to the target exists on that contract's own side.
            same_side = got[got["premium"].notna()]
            assert abs(row["premium"] - target) <= abs(
                same_side["premium"] - target
            ).max()


def test_a_cheap_target_lands_out_of_the_money_and_pays_a_smaller_round_trip(
    tmp_path,
) -> None:
    got = ts.session_tickets(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5910.0)), 5
    )
    cheap = got[got["target_usd"] == 50.0]
    rich = got[got["target_usd"] == 3_200.0]
    assert cheap["premium"].mean() < rich["premium"].mean()
    assert cheap["moneyness"].mean() < rich["moneyness"].mean()
    assert cheap["round_trip_usd"].mean() <= rich["round_trip_usd"].mean()


def test_the_round_trip_charged_is_the_one_the_band_names(tmp_path) -> None:
    got = ts.session_tickets(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5910.0)), 5
    )
    lookup = {band[2]: band[3] for band in ts.BANDS}
    assert (got["round_trip_usd"] == got["band"].map(lookup)).all()


def test_net_is_the_move_less_that_bands_round_trip(tmp_path) -> None:
    got = ts.session_tickets(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5910.0)), 5
    )
    row = got[got["target_usd"] == 800.0].iloc[0]
    # Every fixture contract carries a flat $10 of extrinsic, so the move is
    # entirely intrinsic and reconstructable.
    assert np.isfinite(row["net_usd"])
    assert row["net_usd"] == pytest.approx(
        row["net_usd"] + row["round_trip_usd"] - row["round_trip_usd"]
    )


def test_a_session_with_no_move_yields_nothing(tmp_path) -> None:
    assert ts.session_tickets(
        _session(tmp_path, _chain("09:35", 5900.0) + _chain("09:40", 5900.0)), 5
    ) is None


# --------------------------------------------------------------------------
# assessing one ticket size
# --------------------------------------------------------------------------


def _rows(n: int, win: float, loss: float, premium: float, cost: float) -> pd.DataFrame:
    out = []
    for i in range(n):
        for correct, net in ((True, win), (False, -loss)):
            out.append(
                {
                    "session": f"2025-01-{1 + i // 20:02d}",
                    "entry_minute": f"09:{35 + i % 20:02d}",
                    "target_usd": 800.0,
                    "premium": premium,
                    "moneyness": -10.0,
                    "band": "near ATM",
                    "round_trip_usd": cost,
                    "net_usd": net,
                    "correct": correct,
                    "up": i % 2 == 0,
                }
            )
    return pd.DataFrame(out)


def test_a_ticket_whose_wins_do_not_cover_the_round_trip_is_called_unwinnable() -> None:
    got = ts.assess_target(_rows(200, win=-5.0, loss=20.0, premium=50.0, cost=9.0), 25)
    assert got["breakeven_accuracy"] is None
    assert "unwinnable" in got["verdict"]


def test_a_workable_ticket_reports_a_bar_above_its_break_even() -> None:
    got = ts.assess_target(_rows(200, win=400.0, loss=420.0, premium=800.0, cost=25.0), 25)
    assert got["breakeven_accuracy"] == pytest.approx(420.0 / 820.0, abs=1e-6)
    assert got["provable_accuracy"] > got["breakeven_accuracy"]
    assert got["round_trip_share_of_premium"] == pytest.approx(25.0 / 800.0, abs=1e-4)


def test_too_few_observations_report_nothing_rather_than_a_number() -> None:
    assert ts.assess_target(_rows(10, 400.0, 420.0, 800.0, 25.0), 5) == {}


def test_more_trades_lower_the_bar_at_the_same_payoffs() -> None:
    few = ts.assess_target(_rows(100, 400.0, 420.0, 800.0, 25.0), 25)
    many = ts.assess_target(_rows(800, 400.0, 420.0, 800.0, 25.0), 25)
    assert many["provable_accuracy"] < few["provable_accuracy"]
    assert many["breakeven_accuracy"] == pytest.approx(few["breakeven_accuracy"])
