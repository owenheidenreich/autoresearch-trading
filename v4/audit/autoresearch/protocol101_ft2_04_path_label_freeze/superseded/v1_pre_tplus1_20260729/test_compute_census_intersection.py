"""Focused tests for the FT2-04 intersection verifier (protected-holdout repair).

Structural only: toy fold manifests, toy governance holdout, and toy census sets
with hand-made session strings. No real market data, no census statistics.
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


# Toy corpus: 7 sessions. Folds test on d3,d4,d5,d6; d7 is protected holdout;
# d1,d2 are the only census-eligible sessions (train-only, not holdout).
TOY_FOLDS = {
    "fold_01": {"train": ["d1"], "validation": ["d3"]},
    "fold_02": {"train": ["d1", "d3"], "validation": ["d4"]},
    "fold_03": {"train": ["d1", "d3", "d4"], "validation": ["d5"]},
    "fold_04": {"train": ["d1", "d3", "d4", "d5"], "validation": ["d6"]},
}
TOY_CORPUS = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
TOY_HOLDOUT = ["d7"]
# corpus - outer_test(d3..d6) - holdout(d7) = {d1, d2}
TOY_CENSUS_GOOD = ["d1", "d2"]


def test_disjoint_and_complete_passes():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(TOY_CENSUS_GOOD)
    )
    assert res["passed"] is True
    assert res["counts"] == {
        "governed_corpus": 7,
        "outer_test_union": 4,
        "protected_holdout": 1,
        "census": 2,
        "outer_test_intersection": 0,
        "holdout_intersection": 0,
    }
    assert res["outer_test_intersection_sessions"] == []
    assert res["holdout_intersection_sessions"] == []


def test_outer_test_union_maps_validation_role():
    union, detail = mod.outer_test_union(_runner_plan(TOY_FOLDS, TOY_HOLDOUT))
    assert union == {"d3", "d4", "d5", "d6"}
    assert detail["test_roles"] == ["validation"]


def test_protected_holdout_extracted():
    assert mod.protected_holdout(_runner_plan(TOY_FOLDS, TOY_HOLDOUT)) == {"d7"}


def test_outer_test_overlap_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d2", "d4"])
    )
    assert res["disjoint_census_and_outer_test"] is False
    assert res["outer_test_intersection_sessions"] == ["d4"]
    assert res["passed"] is False


def test_holdout_overlap_is_caught():
    # d7 is the protected holdout; leaking it into census (the exact bug this
    # repair fixes) must fail the holdout firewall.
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d2", "d7"])
    )
    assert res["disjoint_census_and_protected_holdout"] is False
    assert res["holdout_intersection_sessions"] == ["d7"]
    assert res["passed"] is False


def test_incomplete_census_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1"])
    )
    assert res["disjoint_census_and_outer_test"] is True
    assert res["disjoint_census_and_protected_holdout"] is True
    assert res["census_equals_corpus_minus_outer_test_minus_holdout"] is False
    assert res["missing_from_census"] == ["d2"]
    assert res["passed"] is False


def test_census_outside_corpus_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS, TOY_HOLDOUT), _corpus(TOY_CORPUS), _census(["d1", "d2", "dX"])
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
