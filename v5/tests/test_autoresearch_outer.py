"""The outer loop searches constraints; the knob registry decides what it may touch."""
from __future__ import annotations

import numpy as np
import pytest

from v5.research import knobs
from v5.research.autoresearch import budget as b, experiment as ex, outer


def _setting(sid="near_atm_60m", **params):
    return outer.ConstraintSetting(
        setting_id=sid,
        params=params or {"moneyness_band": "near_atm"},
        rationale="the measured optimum of the strike ladder",
        declared_on="2026-08-13",
    )


def _ledger(tmp_path, sessions=2520, ceiling=0.75):
    return b.AlphaLedger(
        tmp_path / "alpha.json",
        option_sessions=sessions,
        universe="phase1_otm",
        ceiling=ceiling,
    )


def _inner(n_experiments=3, accuracy=0.55, sessions=2520):
    def make(setting):
        for i in range(n_experiments):
            rng = np.random.default_rng(i)
            calls = rng.choice([-1, 1], sessions)
            ok = rng.random(sessions) < accuracy
            move = np.where(ok, calls, -calls) * np.abs(rng.normal(0, 8, sessions))
            yield (
                ex.Declaration(
                    experiment_id=f"{setting.setting_id}-{i}",
                    hypothesis="a candidate under this setting",
                    features_used=("overnight_gap",),
                    horizon_minutes=60,
                    declared_on="2026-08-13",
                ),
                calls,
                move,
            )

    return make


# --- the referee -------------------------------------------------------------


def test_the_registry_refuses_a_frozen_constant() -> None:
    """The outer loop must not be able to make a result better by lowering costs."""

    cheat = _setting("cheat", es_round_trip_friction_points=0.05)
    with pytest.raises(knobs.KnobError, match="FROZEN"):
        outer.check(cheat, released_gates={"G1", "G2", "G4"})


def test_the_registry_refuses_an_uncertified_constant() -> None:
    setting = _setting("uncert", emission_lag_ms=900)
    with pytest.raises(knobs.KnobError, match="UNCERTIFIED"):
        outer.check(setting, released_gates={"G1", "G2", "G4"})


def test_the_registry_refuses_an_undeclared_knob() -> None:
    """A search cannot smuggle in a new degree of freedom."""

    with pytest.raises(knobs.KnobError, match="not in the knob registry"):
        outer.check(_setting("new", some_new_dial=1.0), released_gates={"G2"})


def test_the_registry_refuses_a_value_outside_the_declared_space() -> None:
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        outer.check(_setting("bad", moneyness_band="whatever"), released_gates={"G2"})


def test_a_searchable_knob_with_no_amendment_waits_for_its_gate() -> None:
    """The project's actual state: horizon_minutes is still gated on G1.

    moneyness_band is deliberately not used here -- a signed amendment releases
    it, which is the behaviour tested further down.
    """

    with pytest.raises(knobs.KnobError, match="gate G1 has not passed"):
        outer.check(_setting("h", horizon_minutes=60), released_gates=())
    outer.check(_setting("h", horizon_minutes=60), released_gates={"G1"})


def test_occupiable_reports_the_whole_refusal_surface_at_once() -> None:
    settings = [
        _setting("good", moneyness_band="near_atm"),
        _setting("cheat", es_round_trip_friction_points=0.05),
        _setting("uncert", emission_lag_ms=900),
    ]
    allowed, refused = outer.occupiable(settings, released_gates={"G2"})
    assert [s.setting_id for s in allowed] == ["good"]
    assert set(refused) == {"cheat", "uncert"}
    assert "FROZEN" in refused["cheat"]
    assert "UNCERTIFIED" in refused["uncert"]


# --- declaration hygiene -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"setting_id": ""}, {"params": {}}, {"rationale": ""}]
)
def test_an_underspecified_setting_is_refused(kwargs) -> None:
    base = dict(
        setting_id="s",
        params={"moneyness_band": "near_atm"},
        rationale="because",
        declared_on="2026-08-13",
    )
    with pytest.raises(outer.OuterLoopError):
        outer.ConstraintSetting(**{**base, **kwargs})


def test_a_setting_is_content_addressed() -> None:
    a = _setting("s", moneyness_band="near_atm")
    b_ = _setting("s", moneyness_band="otm")
    assert a.sha256() == _setting("s", moneyness_band="near_atm").sha256()
    assert a.sha256() != b_.sha256()


# --- alpha accounting --------------------------------------------------------


def test_occupying_a_setting_spends_alpha_before_any_inner_run(tmp_path) -> None:
    """Trying a configuration is a look at the data, not a free choice."""

    led = _ledger(tmp_path)
    before = led.required_accuracy()
    outer.run_setting(
        _setting(moneyness_band="near_atm"),
        ledger=led,
        inner=lambda s: iter(()),
        released_gates={"G2"},
    )
    assert led.experiments_run == 1
    assert led.required_accuracy() > before
    assert next(iter(led)).experiment_id == "setting:near_atm_60m"


def test_inner_experiments_spend_alpha_on_top_of_the_setting(tmp_path) -> None:
    led = _ledger(tmp_path)
    result = outer.run_setting(
        _setting(moneyness_band="near_atm"),
        ledger=led,
        inner=_inner(3),
        released_gates={"G2"},
    )
    assert result.inner_experiments == 3
    assert led.experiments_run == 4  # one setting plus three candidates


def test_a_widened_constraint_search_raises_the_bar_like_any_other(tmp_path) -> None:
    """Searching the constraint space is not cheaper than searching the model space."""

    led = _ledger(tmp_path)
    start = led.required_accuracy()
    for band in ("near_atm", "otm", "itm"):
        outer.run_setting(
            _setting(f"band_{band}", moneyness_band=band),
            ledger=led,
            inner=_inner(2),
            released_gates={"G2"},
        )
    assert led.experiments_run == 9  # 3 settings + 6 candidates
    assert led.required_accuracy() > start


def test_the_outer_loop_halts_when_the_budget_is_gone(tmp_path) -> None:
    led = _ledger(tmp_path, sessions=251, ceiling=0.55)
    assert led.exhausted
    with pytest.raises(outer.OuterLoopError, match="alpha budget exhausted"):
        outer.run_setting(
            _setting(moneyness_band="near_atm"),
            ledger=led,
            inner=_inner(1),
            released_gates={"G2"},
        )


def test_a_blocked_setting_never_reaches_the_ledger(tmp_path) -> None:
    """Refusal by the referee must not cost alpha; it is not a look at the data."""

    led = _ledger(tmp_path)
    with pytest.raises(knobs.KnobError):
        outer.run_setting(
            _setting("cheat", es_round_trip_friction_points=0.05),
            ledger=led,
            inner=_inner(1),
            released_gates={"G2"},
        )
    assert led.experiments_run == 0


# --- release by signed amendment ---------------------------------------------


def test_a_signed_amendment_releases_a_searchable_knob() -> None:
    """The charter can open a degree of freedom a gate has not opened.

    moneyness_band is released by G2, which is blocked behind a G1 pass. The
    2026-08-13 amendment releases it directly, because which contracts the bot
    may buy is a charter question rather than a measurement question.
    """

    assert "moneyness_band" in outer.SIGNED_AMENDMENTS
    outer.check(_setting(moneyness_band="near_atm"), released_gates=())


def test_an_amendment_cannot_release_a_frozen_or_uncertified_constant() -> None:
    """An amendment changes what may be TRIED, never what the evidence says."""

    for params, pattern in (
        ({"es_round_trip_friction_points": 0.05}, "FROZEN"),
        ({"emission_lag_ms": 900}, "UNCERTIFIED"),
    ):
        with pytest.raises(knobs.KnobError, match=pattern):
            knobs.assert_search_space(
                params,
                released_by_amendment={**outer.SIGNED_AMENDMENTS, **{k: "x" for k in params}},
            )


def test_an_amendment_cannot_widen_a_declared_space() -> None:
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        outer.check(_setting("bad", moneyness_band="whatever"), released_gates=())


def test_an_amendment_naming_an_unregistered_knob_is_refused() -> None:
    with pytest.raises(knobs.KnobError, match="not registered"):
        knobs.assert_search_space(
            {"moneyness_band": "near_atm"},
            released_by_amendment={"no_such_knob": "somewhere.md"},
        )


def test_the_amendment_file_named_by_the_release_exists_and_is_signed() -> None:
    """A release that points at a missing or unsigned document is not a release."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for knob_name, relative in outer.SIGNED_AMENDMENTS.items():
        doc = root / relative
        assert doc.is_file(), f"{knob_name} names a missing amendment: {relative}"
        text = doc.read_text(encoding="utf-8")
        assert "SIGNED AND IN FORCE" in text, f"{relative} is not signed"
        assert knob_name in text, f"{relative} does not mention {knob_name}"


def test_the_entry_premium_band_is_declared_but_not_yet_released() -> None:
    """The 2026-08-13 decay finding, available to search only once a gate opens.

    The signed charter amendment released moneyness_band because which contracts
    the bot may buy is a charter question. It says nothing about entry premium,
    so that knob correctly still waits on G2 rather than being released by
    association.
    """

    assert "entry_premium_band" not in outer.SIGNED_AMENDMENTS
    with pytest.raises(knobs.KnobError, match="gate G2 has not passed"):
        outer.check(
            _setting("prem", entry_premium_band="exclude_richest_quartile"),
            released_gates=(),
        )
    outer.check(
        _setting("prem", entry_premium_band="exclude_richest_quartile"),
        released_gates={"G2"},
    )


def test_a_constraint_pair_composes_and_is_content_addressed() -> None:
    """The outer loop searches combinations, not one knob at a time."""

    pair = _setting(
        "atm_cheap",
        moneyness_band="near_atm",
        entry_premium_band="exclude_richest_quartile",
    )
    outer.check(pair, released_gates={"G2"})
    other = _setting(
        "atm_any", moneyness_band="near_atm", entry_premium_band="any"
    )
    assert pair.sha256() != other.sha256()
