"""The alpha ledger is the only thing standing between a loop and a false positive."""
from __future__ import annotations

import json

import pytest

from v5.research.autoresearch import budget as b


def _ledger(tmp_path, sessions=251, ceiling=0.75, universe="phase1_otm"):
    return b.AlphaLedger(
        tmp_path / "alpha.json", option_sessions=sessions,
        universe=universe, ceiling=ceiling,
    )


def _record(ledger, i, outcome="FAIL"):
    return ledger.record(
        experiment_id=f"exp-{i:04d}",
        declaration_sha256=f"{i:064x}",
        declared_on="2026-08-13",
        outcome=outcome,
    )


def test_the_bar_rises_with_every_experiment(tmp_path) -> None:
    led = _ledger(tmp_path)
    first = led.required_accuracy()
    for i in range(20):
        _record(led, i)
    assert led.required_accuracy() > first
    # ...but slowly: Bonferroni grows like sqrt(log k), which is the whole
    # reason an autonomous loop is viable at all.
    assert led.required_accuracy() - first < 0.05


def test_attempt_k_is_judged_at_the_k_experiment_bar(tmp_path) -> None:
    """The failure this exists to prevent: pricing 4,000 attempts as one."""

    led = _ledger(tmp_path)
    bars = []
    for i in range(5):
        bars.append(led.required_accuracy())
        _record(led, i)
    assert bars == sorted(bars)
    assert len(set(bars)) == 5, "each attempt must face its own bar"
    # The recorded bar is the one that was in force when the entry was written.
    assert [round(e.bar_at_time_of_run, 8) for e in led] == [round(x, 8) for x in bars]


def test_a_refused_experiment_still_spends_alpha(tmp_path) -> None:
    """Deciding to look is what costs, not the result of looking."""

    led = _ledger(tmp_path)
    before = led.required_accuracy()
    _record(led, 0, outcome="REFUSED")
    assert led.required_accuracy() > before
    assert led.experiments_run == 1


def test_the_ledger_is_append_only_and_tamper_evident(tmp_path) -> None:
    path = tmp_path / "alpha.json"
    led = b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")
    for i in range(4):
        _record(led, i)

    payload = json.loads(path.read_text())
    # Remove an experiment to make the bar look lower than it was bought at.
    payload["entries"].pop(1)
    path.write_text(json.dumps(payload))
    with pytest.raises(b.BudgetError, match="chain broken"):
        b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")


def test_an_edited_entry_is_refused(tmp_path) -> None:
    path = tmp_path / "alpha.json"
    led = b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")
    _record(led, 0)
    payload = json.loads(path.read_text())
    payload["entries"][0]["outcome"] = "PASS"
    path.write_text(json.dumps(payload))
    with pytest.raises(b.BudgetError, match="edited after the fact"):
        b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")


def test_the_same_experiment_cannot_be_counted_twice(tmp_path) -> None:
    led = _ledger(tmp_path)
    _record(led, 0)
    with pytest.raises(b.BudgetError, match="already recorded"):
        _record(led, 0)


def test_the_ledger_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "alpha.json"
    led = b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")
    for i in range(6):
        _record(led, i)
    head, bar = led.head, led.required_accuracy()

    reopened = b.AlphaLedger(path, option_sessions=251, universe="phase1_otm")
    assert reopened.experiments_run == 6
    assert reopened.head == head
    assert reopened.required_accuracy() == pytest.approx(bar)


def test_the_near_atm_bar_is_the_pooled_four_year_figure(tmp_path) -> None:
    """The constant was wrong twice before it was right, so it is pinned here.

    Frozen 2026-08-13 at 50.60% from one favourable year of quotes; downgraded to
    UNCERTIFIED hours later when a second schema disagreed by 7.7 points;
    re-derived on 905 sessions spanning 2022-2026 at 54.24% pooled. The pooled
    figure is WORSE than ES futures at 51.40%, which reverses the claim the
    single-year number supported.
    """

    led = _ledger(tmp_path, universe="near_atm")
    assert led.breakeven_accuracy() == pytest.approx(0.5424, abs=1e-3)
    assert led.breakeven_accuracy() > 0.5140, (
        "on the pooled sample near-ATM options do NOT beat ES futures; a test "
        "that lets this silently invert is how the claim survived a day"
    )


def test_the_contract_universe_decides_whether_the_loop_can_run(tmp_path) -> None:
    """The 2026-08-13 finding: the universe matters more than the corpus size.

    Only phase1_otm can be exercised here. near_atm was downgraded to UNCERTIFIED
    the same day, when a second schema disagreed with it by 7.7 points -- and the
    test above asserts that a loop pointed at it refuses rather than guessing.
    """

    otm = _ledger(tmp_path / "otm", sessions=251, universe="phase1_otm")
    assert otm.breakeven_accuracy() == pytest.approx(0.5799, abs=1e-3)
    assert otm.required_true_accuracy() == pytest.approx(0.6833, abs=1e-3)
    # On the band a $3-8 price filter selects, 251 sessions leave room for only a
    # few hundred attempts before the bar passes the plausibility ceiling.
    assert otm.experiments_remaining() < 1_000


def test_an_undeclared_universe_is_refused(tmp_path) -> None:
    """A loop must say which contracts it is judged against; there is no default."""

    with pytest.raises(b.BudgetError, match="unknown contract universe"):
        _ledger(tmp_path, universe="whatever_looks_best")


def test_more_sessions_still_lower_the_bar_within_a_universe(tmp_path) -> None:
    small = _ledger(tmp_path / "small", sessions=251)
    large = _ledger(tmp_path / "large", sessions=2520)
    assert large.required_true_accuracy() < small.required_true_accuracy()
    assert large.experiments_remaining() > small.experiments_remaining()


def test_the_loop_stops_rather_than_running_until_something_passes(tmp_path) -> None:
    """A ceiling below the starting bar must halt immediately, not eventually."""

    led = _ledger(tmp_path, sessions=251, ceiling=0.60)
    assert led.exhausted is True
    assert led.experiments_remaining() == 0


def test_experiments_remaining_is_consistent_with_the_bar(tmp_path) -> None:
    led = _ledger(tmp_path, sessions=251, universe="phase1_otm", ceiling=0.70)
    n = led.experiments_remaining()
    assert led.required_true_accuracy(experiments=n) < led.ceiling
    assert led.required_true_accuracy(experiments=n + 1) >= led.ceiling


@pytest.mark.parametrize("bad", [0, -1])
def test_a_ledger_needs_a_real_corpus(tmp_path, bad) -> None:
    with pytest.raises(b.BudgetError):
        b.AlphaLedger(tmp_path / "x.json", option_sessions=bad, universe="phase1_otm")


def test_an_unknown_outcome_is_refused(tmp_path) -> None:
    led = _ledger(tmp_path)
    with pytest.raises(b.BudgetError, match="unknown outcome"):
        led.record(
            experiment_id="x",
            declaration_sha256="0" * 64,
            declared_on="2026-08-13",
            outcome="MAYBE",
        )
