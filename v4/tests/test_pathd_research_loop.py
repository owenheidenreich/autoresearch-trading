"""Tests for the Path-D autoresearch loop.

The loop's value is entirely in its refusals: it must block closed mechanisms,
refuse renamed duplicates, and stop at its budget. Each test below asserts a
refusal that a slot-machine loop would not make.
"""
from __future__ import annotations

import pytest

from v4.research.pathd_model_gate import exit_first_step_rate, label_balance
from v4.research.pathd_research_loop import (
    ExperimentResult,
    Hypothesis,
    LoopError,
    WaveSpec,
    blocked_by_prior_art,
    prior_art_check,
    run_wave,
)


def _tests_pass():
    return [
        exit_first_step_rate([0] * 5 + list(range(1, 96))),
        label_balance([1.0] * 45 + [-1.0] * 55),
    ]


def _result(**overrides) -> ExperimentResult:
    kwargs = dict(
        pooled_policy=10.0,
        pooled_comparator=50.0,           # losing by default -> NO_EDGE
        fold_deltas={str(i): -1.0 for i in range(5)},
        bootstrap_lcb=-5.0,
        negative_controls_accepted={"sign_reversed": False},
        rejection_tests=_tests_pass(),
        maxt_survived=True,
        concentrated=False,
    )
    kwargs.update(overrides)
    return ExperimentResult(**kwargs)


def _winner() -> ExperimentResult:
    return _result(
        pooled_policy=100.0,
        pooled_comparator=50.0,
        fold_deltas={str(i): 1.0 for i in range(5)},
        bootstrap_lcb=10.0,
    )


# --- executable prior art, against the real canonical documents -------------


def test_prior_art_blocks_a_do_not_retest_mechanism() -> None:
    assert blocked_by_prior_art(prior_art_check("score coverage threshold"))


def test_prior_art_blocks_a_rejected_protocol_outside_section_four() -> None:
    """The gap that let Path-D re-run Protocols 029/030/031.

    'loss-only damage-control exit' is a rejected protocol in the farm lineage
    but appears nowhere in the distillation's section 4. A section-4-only check
    waves it through; this must not.
    """

    hits = prior_art_check("loss-only damage-control exit")
    assert hits, "expected a hit in the protocol decoder"
    assert not any(h.in_do_not_retest for h in hits), "precondition: not in section 4"
    assert blocked_by_prior_art(hits), "a rejecting verdict must still block"


def test_prior_art_blocks_the_completed_wave1_recovery_fix() -> None:
    """P065 was allowed before Wave 1; its completed NO_EDGE rerun is now closed."""

    assert blocked_by_prior_art(prior_art_check("recovery penalty"))


def test_prior_art_is_quiet_on_a_novel_mechanism() -> None:
    assert prior_art_check("a mechanism nobody has ever proposed xyzzy") == []


def test_prior_art_requires_a_mechanism() -> None:
    with pytest.raises(LoopError):
        prior_art_check("   ")


# --- wave validation --------------------------------------------------------


def test_wave_rejects_more_hypotheses_than_budget() -> None:
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis(f"h{i}", f"novel mechanism xyzzy {i}") for i in range(3)],
        budget=2,
    )
    with pytest.raises(LoopError):
        spec.validate()


def test_wave_rejects_duplicate_ids() -> None:
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis("h", "alpha xyzzy"), Hypothesis("h", "beta xyzzy")],
        budget=5,
    )
    with pytest.raises(LoopError):
        spec.validate()


# --- the loop ---------------------------------------------------------------


def test_blocked_hypothesis_does_not_consume_budget(tmp_path) -> None:
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[
            Hypothesis("h1", "score coverage threshold"),   # closed -> blocked
            Hypothesis("h2", "novel mechanism xyzzy alpha"),
        ],
        budget=5,
    )
    report = run_wave(spec, lambda h: _result(), registry_path=tmp_path / "r.jsonl")
    statuses = {r["hypothesis_id"]: r["status"] for r in report["results"]}
    assert statuses["h1"] == "BLOCKED_BY_PRIOR_ART"
    assert statuses["h2"] == "RAN"
    assert report["budget_spent"] == 1


def test_semantic_duplicate_is_skipped_even_under_a_new_name(tmp_path) -> None:
    registry = tmp_path / "r.jsonl"
    mech = "novel mechanism xyzzy beta"
    first = WaveSpec("w1", "o", [Hypothesis("h1", mech, {"k": 1})], budget=5)
    run_wave(first, lambda h: _result(), registry_path=registry)

    # Same mechanism and params, different id -> must be refused.
    second = WaveSpec("w2", "o", [Hypothesis("DIFFERENT_NAME", mech, {"k": 1})], budget=5)
    report = run_wave(second, lambda h: _result(), registry_path=registry)
    assert report["results"][0]["status"] == "SKIPPED_SEMANTIC_DUPLICATE"
    assert report["budget_spent"] == 0


def test_changing_params_is_a_new_hypothesis(tmp_path) -> None:
    registry = tmp_path / "r.jsonl"
    mech = "novel mechanism xyzzy gamma"
    run_wave(WaveSpec("w1", "o", [Hypothesis("h1", mech, {"k": 1})], budget=5),
             lambda h: _result(), registry_path=registry)
    report = run_wave(WaveSpec("w2", "o", [Hypothesis("h2", mech, {"k": 2})], budget=5),
                      lambda h: _result(), registry_path=registry)
    assert report["results"][0]["status"] == "RAN"


def test_budget_exhaustion_ends_the_wave_as_no_edge(tmp_path) -> None:
    """The declared hypothesis count IS the family size, so exhaustion is
    running every declared hypothesis without a TIER_A."""

    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis(f"h{i}", f"novel mechanism xyzzy d{i}") for i in range(2)],
        budget=2,
    )
    report = run_wave(spec, lambda h: _result(), registry_path=tmp_path / "r.jsonl")
    assert report["verdict"] == "NO_EDGE"
    assert report["budget_spent"] == 2
    assert report["budget_exhausted"]


def test_wave_stops_on_tier_a(tmp_path) -> None:
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis(f"h{i}", f"novel mechanism xyzzy e{i}") for i in range(3)],
        budget=3,
    )
    report = run_wave(spec, lambda h: _winner(), registry_path=tmp_path / "r.jsonl")
    assert report["verdict"] == "TIER_A"
    assert report["budget_spent"] == 1  # stopped immediately


def test_accepted_negative_control_invalidates_and_stops(tmp_path) -> None:
    bad = _winner()
    bad.negative_controls_accepted = {"sign_reversed": True}
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis(f"h{i}", f"novel mechanism xyzzy f{i}") for i in range(3)],
        budget=3,
    )
    report = run_wave(spec, lambda h: bad, registry_path=tmp_path / "r.jsonl")
    assert report["verdict"] == "INVALID"
    assert report["budget_spent"] == 1


def test_family_size_counts_every_look_at_the_data(tmp_path) -> None:
    spec = WaveSpec(
        wave_id="w",
        objective="o",
        hypotheses=[Hypothesis(f"h{i}", f"novel mechanism xyzzy g{i}") for i in range(3)],
        budget=3,
    )
    report = run_wave(spec, lambda h: _result(), registry_path=tmp_path / "r.jsonl")
    assert report["family_size_for_maxt"] == 3
