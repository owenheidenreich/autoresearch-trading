"""The divergence register must refuse, and must not be parkable."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v5.research import divergence as dv
from v5.research import knobs
from v5.research.direction import loader


# --- the register's own shape ------------------------------------------------


def test_every_axis_states_an_exit_and_a_gate() -> None:
    """An axis with no exit condition is a parked problem, not a record."""

    for axis in dv.REGISTRY.values():
        assert axis.settles_when, axis.name
        assert axis.binds_at_gate, axis.name
        assert axis.summary, axis.name
        assert axis.evidence, axis.name


def test_status_and_group_are_closed_vocabularies() -> None:
    with pytest.raises(dv.DivergenceError, match="unknown status"):
        dv.Axis(
            name="x", group=dv.TIMING, status="PROBABLY_FINE", summary="s",
            evidence="e", settles_when="w", binds_at_gate="G4",
        )
    with pytest.raises(dv.DivergenceError, match="unknown group"):
        dv.Axis(
            name="x", group="vibes", status=dv.UNKNOWN, summary="s",
            evidence="e", settles_when="w", binds_at_gate="G4",
        )


def test_an_axis_cannot_be_parked_without_an_exit() -> None:
    with pytest.raises(dv.DivergenceError, match="must state what would settle it"):
        dv.Axis(
            name="x", group=dv.TIMING, status=dv.UNKNOWN, summary="s",
            evidence="e", settles_when="", binds_at_gate="G4",
        )


def test_a_repair_only_means_something_for_a_measured_difference() -> None:
    with pytest.raises(dv.DivergenceError, match="names a repair but is"):
        dv.Axis(
            name="x", group=dv.TIMING, status=dv.UNKNOWN, summary="s",
            evidence="e", settles_when="w", binds_at_gate="G4", repair="r",
        )


# --- what it refuses ---------------------------------------------------------


def test_unknown_axis_refuses_a_fit() -> None:
    with pytest.raises(dv.DivergenceError, match="emission_lag"):
        dv.assert_no_unknown_on_path(
            ["arrival_latency", "emission_lag"], purpose="a fit"
        )


def test_measured_difference_without_a_repair_also_refuses() -> None:
    """Knowing the size of a gap you have not closed is not safety."""

    unrepaired = dv.Axis(
        name="x", group=dv.TIMING, status=dv.MEASURED_DIFFERENT, summary="s",
        evidence="e", settles_when="w", binds_at_gate="G4",
    )
    assert unrepaired.blocks_a_fit
    repaired = dv.Axis(
        name="y", group=dv.TIMING, status=dv.MEASURED_DIFFERENT, summary="s",
        evidence="e", settles_when="w", binds_at_gate="G4", repair="declared",
    )
    assert not repaired.blocks_a_fit


def test_a_repaired_measured_difference_is_permitted() -> None:
    axes = dv.assert_no_unknown_on_path(
        ["arrival_latency", "sparse_minutes"], purpose="a repaired fit"
    )
    assert {a.name for a in axes} == {"arrival_latency", "sparse_minutes"}


def test_an_undeclared_axis_is_refused_rather_than_ignored() -> None:
    """The failure this module exists to prevent is a divergence nobody wrote down."""

    with pytest.raises(dv.DivergenceError, match="not in the divergence register"):
        dv.assert_no_unknown_on_path(["some_axis_nobody_declared"], purpose="a fit")


def test_settling_must_name_a_real_axis() -> None:
    with pytest.raises(dv.DivergenceError, match="settled names not in the register"):
        dv.assert_no_unknown_on_path(
            ["emission_lag"], purpose="a fit", settled=["not_an_axis"]
        )


def test_settled_axes_are_allowed_through() -> None:
    axes = dv.assert_no_unknown_on_path(
        ["emission_lag"], purpose="a fit", settled=["emission_lag"]
    )
    assert axes[0].name == "emission_lag"


# --- the register must agree with the knob registry --------------------------


def test_every_uncertified_knob_is_explained_by_an_axis() -> None:
    """The two registries must not disagree about what is unknown.

    A knob may be UNCERTIFIED only if some axis says *why* it diverges. The
    link is a declared reference, not a string match, so renaming either side
    breaks the test instead of silently passing it.
    """

    uncertified = {knob.name for knob in knobs.by_class(knobs.UNCERTIFIED)}
    claimed = {n for axis in dv.REGISTRY.values() for n in axis.knob_names}
    assert uncertified <= claimed, (
        "UNCERTIFIED knobs with no divergence axis explaining them: "
        f"{sorted(uncertified - claimed)}"
    )


def test_an_axis_cannot_reference_a_knob_that_does_not_exist() -> None:
    with pytest.raises(dv.DivergenceError, match="not in the knob registry"):
        dv.Axis(
            name="x", group=dv.TIMING, status=dv.UNKNOWN, summary="s",
            evidence="e", settles_when="w", binds_at_gate="G4",
            knob_names=("no_such_knob",),
        )


def test_blocking_axes_are_reported() -> None:
    assert dv.summary()["UNKNOWN"] == len(dv.unchecked())
    assert "emission_lag" in {a.name for a in dv.unchecked()}


# --- the bar-labelling guard the register demanded ---------------------------


def _frame(minutes: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(
        [f"2026-01-05T{m}:00-05:00" for m in minutes], utc=True
    )
    return pd.DataFrame({"close": range(len(minutes))}, index=index)


def test_end_labelled_bars_are_refused() -> None:
    """The convention is load-bearing: a flip shifts every feature one minute."""

    minutes = [f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(1, 6)]
    assert minutes[0] == "09:31"
    with pytest.raises(loader.LoaderError, match="first regular-hours bar is 09:31"):
        loader._assert_start_labelled("2026-01-05", _frame(minutes).index.tz_convert(
            "America/New_York"
        ).strftime("%H:%M").tolist())


def test_start_labelled_bars_pass_including_a_short_session() -> None:
    """A short session is a different axis and must not trip the label guard."""

    full = [f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(0, 5)]
    assert full[0] == "09:30"
    loader._assert_start_labelled("2026-01-05", full)
    # 2025-11-27 style: opens 09:30, closes early at 12:59.
    loader._assert_start_labelled("2025-11-27", ["09:30", "09:31", "12:59"])


@pytest.mark.skipif(
    not Path(loader.family.ES_BARS_ROOT).is_dir(), reason="owned ES bars not present"
)
def test_the_real_corpus_is_start_labelled_on_every_session() -> None:
    sessions = loader.load_sessions()
    assert len(sessions) == loader.family.M1_ELIGIBLE_SESSIONS
    assert all(s.minute_et[0] == "09:30" for s in sessions)
