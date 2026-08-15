"""The second G1 attempt's declaration must be frozen, honest, and tamper-evident."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v5.research import knobs, statistics as st
from v5.research.direction import family, hypothesis_2026_08 as h


FROZEN_SHA256 = "e940a38833de7b54183e5c9dab29f907d67d9a3d109537220035d352ff133f37"


def test_the_declaration_hash_is_pinned() -> None:
    """If this fails, the search space moved. That is the whole point of it."""

    assert h.freeze_sha256() == FROZEN_SHA256


def test_the_declaration_is_deterministic() -> None:
    assert h.freeze_sha256() == h.freeze_sha256()
    assert json.dumps(h.declaration(), sort_keys=True)  # serialises cleanly


def test_the_first_attempt_is_untouched() -> None:
    """family.py is the record of what was frozen for the closed attempt."""

    assert family.FAMILY_SIZE == 18
    assert family.M1_ELIGIBLE_SESSIONS == 247
    assert h.declaration()["supersedes"]["module"] == "v5/research/direction/family.py"


def test_exactly_two_members_one_per_direction() -> None:
    assert h.FAMILY_SIZE == 2
    assert set(h.MEMBERS) == {"M3.with.60m", "M3.against.60m"}
    assert set(h.DIRECTIONS) == {"with", "against"}


def test_the_threshold_is_accuracy_and_matches_the_option_derivation() -> None:
    """The bar comes from the option layer, not from ES friction."""

    win = knobs.frozen_value("option_correct_call_dollars")
    loss = knobs.frozen_value("option_wrong_call_dollars")

    assert h.REQUIRED_ACCURACY == pytest.approx(
        st.detectable_accuracy(251, win=win, loss=loss, z_alpha=st.Z_95), abs=1e-4
    )
    assert h.OPTION_BREAKEVEN_ACCURACY == pytest.approx(
        st.breakeven_accuracy(win=win, loss=loss), abs=1e-4
    )
    # And it is emphatically not the ES friction bar it replaces.
    assert h.REQUIRED_ACCURACY > h.OPTION_BREAKEVEN_ACCURACY
    assert h.declaration()["threshold"]["units"].startswith("directional accuracy")


def test_the_power_statement_shows_the_edge_is_visible() -> None:
    """The purchase only mattered if the required edge clears the floor."""

    p = h.power()
    assert p.eligible_sessions == 2435
    assert p.detectable_accuracy < p.required_accuracy, "required edge is invisible"
    assert p.margin_points == pytest.approx(7.92, abs=0.05)
    # A detectable Sharpe near 0.9 is inside the realistic range; the old corpus
    # sat at 2.5-6.9, which was not.
    assert p.detectable_sharpe < 1.0


def test_the_owned_corpus_could_not_have_run_this() -> None:
    """The counterfactual that justifies Phase 1, asserted rather than asserted at."""

    import math

    z = st.bonferroni_quantile(h.FAMILY_SIZE)
    penalty = 4.0 / (
        (st.bonferroni_quantile(18) + st.Z_POWER_80) / (st.Z_95 + st.Z_POWER_80)
    )
    old = 0.5 + (z + st.Z_POWER_80) * penalty * 0.5 / math.sqrt(247)
    assert old > h.REQUIRED_ACCURACY, (
        "on 247 sessions the required accuracy must be INVISIBLE, otherwise the "
        "history purchase bought nothing"
    )


def test_the_fold_rule_is_a_diagnostic_and_says_why_that_is_risky() -> None:
    assert h.declaration()["folds"]["status"] == "DIAGNOSTIC"
    assert "7.08x" in h.FOLD_DIAGNOSTIC
    assert "volatility artifact" in h.FOLD_DIAGNOSTIC
    # The fold report is a pass criterion even though the 4-of-5 rule is not.
    assert any("per-fold" in c for c in h.PASS_CRITERIA)


def test_the_known_answer_campaign_must_precede_any_economics() -> None:
    first = h.PASS_CRITERIA[0]
    assert "known-answer campaign" in first
    assert "BEFORE any real economics" in first
    assert h.KNOWN_ANSWER_CRITERIA["recovery_of_a_detectable_effect_min"] == 0.80
    assert h.KNOWN_ANSWER_CRITERIA["matched_surrogate_false_pass_max"] == 0.05


def test_every_exclusion_is_structural_and_counted() -> None:
    total = sum(v["count"] for v in h.EXCLUSIONS.values())
    assert total == 106  # 67 holidays + 37 rolls + 1 first session + 1 missing bar
    for name, spec in h.EXCLUSIONS.items():
        assert spec["count"] > 0 and spec["rule"], name


def test_the_holiday_rule_covers_every_verified_no_spxw_session() -> None:
    """The derived rule must reproduce the seven independently verified days."""

    rule = h.EXCLUSIONS["equity_holiday_no_spxw"]["rule"]
    assert "13:00" in rule
    assert len(family.NO_OPTION_SESSIONS) == 7


def test_the_stopping_rule_is_declared_before_the_outcome() -> None:
    assert "programme closes" in h.STOPPING_RULE
    assert "second null repair" in h.STOPPING_RULE


def test_the_declaration_computes_no_economics() -> None:
    assert h.declaration()["computes_no_economics"] is True
    source = Path(h.__file__).read_text(encoding="utf-8")
    for forbidden in ("read_parquet", "load_sessions", "close_at", "pnl", "net_points"):
        assert forbidden not in source, f"declaration must not touch data: {forbidden}"


# --- the fixture bug the ten-year corpus exposed ------------------------------


def test_a_short_session_cannot_poison_the_shared_term_fixture() -> None:
    """Regression for a NaN that silently ate 81% of the index.

    `float(np.abs(np.diff(close)).mean() or 1.0)` returns NaN for a session with
    fewer than two bars, because NaN is truthy. That NaN then propagated through
    the carried `level` and turned every LATER session NaN, so the shared-term
    null ran on 457 of 2,435 sessions and looked reassuringly safe. The
    247-session corpus contained no such session; the ten-year corpus has two.
    """

    import numpy as np
    from v5.research.direction import fixtures, loader

    def bars(session, n, instrument=1):
        close = np.linspace(5000.0, 5000.0 + n, n)
        return loader.SessionBars(
            session=session,
            instrument_id=instrument,
            minute_et=tuple(f"{9 + (30 + i)//60:02d}:{(30 + i) % 60:02d}" for i in range(n)),
            open=close.copy(), high=close + 1, low=close - 1, close=close,
            volume=np.ones(n),
        )

    # A one-bar session in the middle must not affect anything after it.
    corpus = (bars("2020-01-02", 390), bars("2020-01-03", 1), bars("2020-01-06", 390))
    built = fixtures.shared_term_sessions(corpus, seed=0)

    assert len(built) == 3
    for session in built:
        assert np.isfinite(session.close).all(), f"{session.session} went non-finite"
        assert np.isfinite(session.open).all()
