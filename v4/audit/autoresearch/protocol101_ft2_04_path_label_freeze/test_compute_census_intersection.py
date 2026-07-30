"""Focused tests for the FT2-04 role-firewall verifier.

Structural only: toy folds, embargo gaps, holdout, and census sets. No market
data or census statistics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ft2_04_intersection",
    Path(__file__).with_name("compute_census_outer_test_intersection.py"),
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _runner_plan(fold_sessions, holdout, role_map=None):
    role_map = role_map or {"train": "train", "validation": "test"}
    return {
        "expanding_folds": {
            "fold_governance": {"fold_role_map": role_map},
            "fold_sessions": fold_sessions,
        },
        "governance": {"protected_holdout_sessions": list(holdout)},
    }


def _corpus(sessions):
    return {"included_sessions": list(sessions)}


def _census(sessions):
    return {"census_sessions": list(sessions)}


# Toy corpus: d2 and d4 are the one-session fold embargoes; d3 and d5 are
# outer-test sessions; d6 is holdout; d1 and d7 are census-eligible.
TOY_FOLDS = {
    "fold_01": {"train": ["d1"], "validation": ["d3"]},
    "fold_02": {"train": ["d1", "d2", "d3"], "validation": ["d5"]},
}
TOY_CORPUS = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
TOY_HOLDOUT = ["d6"]
TOY_CENSUS_GOOD = ["d1", "d7"]


def test_disjoint_and_complete_passes():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(TOY_CENSUS_GOOD)
    )
    assert res["passed"] is True
    assert res["counts"] == {
        "governed_corpus": 7,
        "outer_test_union": 2,
        "protected_holdout": 1,
        "embargo_union": 2,
        "census": 2,
        "outer_test_intersection": 0,
        "holdout_intersection": 0,
        "embargo_intersection": 0,
    }
    assert res["outer_test_intersection_sessions"] == []
    assert res["holdout_intersection_sessions"] == []
    assert res["embargo_intersection_sessions"] == []


def test_outer_test_union_maps_validation_role():
    union, detail = mod.outer_test_union(_runner_plan(TOY_FOLDS, TOY_HOLDOUT))
    assert union == {"d3", "d5"}
    assert detail["test_roles"] == ["validation"]


def test_protected_holdout_extracted():
    assert mod.protected_holdout(_runner_plan(TOY_FOLDS, TOY_HOLDOUT)) == {"d6"}


def test_embargo_sessions_are_derived_from_governed_gaps():
    sessions, per_fold = mod.embargo_sessions(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS)
    )
    assert sessions == {"d2", "d4"}
    assert per_fold == {"fold_01": "d2", "fold_02": "d4"}


def test_outer_test_overlap_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d3", "d7"])
    )
    assert res["disjoint_census_and_outer_test"] is False
    assert res["outer_test_intersection_sessions"] == ["d3"]
    assert res["passed"] is False


def test_holdout_overlap_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d6", "d7"])
    )
    assert res["disjoint_census_and_protected_holdout"] is False
    assert res["holdout_intersection_sessions"] == ["d6"]
    assert res["passed"] is False


def test_embargo_overlap_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT),
        _corpus(TOY_CORPUS),
        _census(["d1", "d2", "d7"]),
    )
    assert res["disjoint_census_and_embargo"] is False
    assert res["embargo_intersection_sessions"] == ["d2"]
    assert res["passed"] is False


def test_incomplete_census_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1"])
    )
    assert res["disjoint_census_and_outer_test"] is True
    assert res["disjoint_census_and_protected_holdout"] is True
    assert res["disjoint_census_and_embargo"] is True
    assert res["census_equals_corpus_minus_outer_test_minus_holdout_minus_embargo"] is False
    assert res["missing_from_census"] == ["d7"]
    assert res["passed"] is False


def test_census_outside_corpus_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d7", "dX"])
    )
    assert res["census_within_corpus"] is False
    assert res["census_outside_corpus"] == ["dX"]
    assert res["passed"] is False


def test_no_test_role_is_ambiguous():
    import pytest

    bad = _runner_plan(TOY_FOLDS, TOY_HOLDOUT, role_map={"train": "train", "validation": "dev"})
    with pytest.raises(ValueError):
        mod.outer_test_union(bad)


def test_missing_holdout_governance_is_caught():
    import pytest

    bad = {"expanding_folds": {"fold_governance": {"fold_role_map": {"validation": "test"}}, "fold_sessions": TOY_FOLDS}, "governance": {}}
    with pytest.raises(ValueError):
        mod.protected_holdout(bad)


def test_ambiguous_embargo_gap_is_caught():
    import pytest

    bad_corpus = _corpus([*TOY_CORPUS, "d2a"])
    with pytest.raises(ValueError):
        mod.embargo_sessions(_runner_plan(TOY_FOLDS, TOY_HOLDOUT), bad_corpus)
