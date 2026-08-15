"""Break-even by entry hour, and at more than one execution assumption."""
from __future__ import annotations

import pandas as pd
import pytest

from v5.ops import measure_time_of_day as tod


def test_the_three_costs_are_the_declared_execution_assumptions() -> None:
    # Fees only, half the spread, and the full aggressive round trip.
    assert tod.COSTS_USD == (3.08, 14.0, 25.0)


def _rows(n: int, right: float, wrong: float, hour: str = "09") -> pd.DataFrame:
    out = []
    for i in range(n):
        for correct, gross in ((True, right), (False, -wrong)):
            out.append(
                {
                    "session": f"2025-01-{1 + i % 20:02d}",
                    "entry_minute": f"{hour}:35",
                    "hour": hour,
                    "minutes_to_close": 385,
                    "premium": 1_000.0,
                    "gross_usd": gross,
                    "correct": correct,
                }
            )
    return pd.DataFrame(out)


def test_a_cheaper_round_trip_always_lowers_the_break_even() -> None:
    got = tod.assess(_rows(200, right=600.0, wrong=560.0))
    be = got["breakeven_by_cost"]
    assert be["$25.0"] > be["$14.0"] > be["$3.08"]


def test_break_even_is_computed_from_gross_and_the_named_cost() -> None:
    got = tod.assess(_rows(200, right=600.0, wrong=560.0))
    # win = 600 - 25, loss = 560 + 25.
    assert got["breakeven_by_cost"]["$25.0"] == pytest.approx(585.0 / 1160.0, abs=1e-6)
    # The cost is charged on both sides, so the spread is unchanged at 1,160.
    assert got["breakeven_by_cost"]["$3.08"] == pytest.approx(
        563.08 / 1160.0, abs=1e-6
    )


def test_both_sides_reports_what_buying_the_pair_earned_before_cost() -> None:
    """Positive means realised movement outran what the options charged."""

    got = tod.assess(_rows(200, right=600.0, wrong=560.0))
    assert got["both_sides_gross_usd"] == pytest.approx(40.0)
    assert got["mean_gross_when_correct_usd"] == pytest.approx(600.0)
    assert got["mean_gross_when_wrong_usd"] == pytest.approx(-560.0)


def test_a_pair_that_cannot_pay_its_cost_reports_no_break_even() -> None:
    got = tod.assess(_rows(200, right=20.0, wrong=560.0))
    assert got["breakeven_by_cost"]["$25.0"] is None
    assert got["breakeven_by_cost"]["$3.08"] is not None


def test_too_few_observations_report_nothing() -> None:
    assert tod.assess(_rows(10, 600.0, 560.0)) == {}


def test_the_hour_tag_is_the_entry_hour() -> None:
    late = tod.assess(_rows(200, right=600.0, wrong=560.0, hour="15"))
    assert late["trades"] == 400
    assert late["sessions"] == 20
