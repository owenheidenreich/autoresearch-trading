"""Focused tests for the FT2-04 intersection verifier.

Structural only: toy fold manifests and toy census sets with hand-made session
strings. No real market data, no census statistics.
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


def _runner_plan(fold_sessions, role_map=None):
    role_map = role_map or {"train": "train", "validation": "test"}
    return {
        "expanding_folds": {
            "fold_governance": {"fold_role_map": role_map},
            "fold_sessions": fold_sessions,
        }
    }


def _corpus(sessions):
    return {"included_sessions": list(sessions)}


def _census(sessions):
    return {"census_sessions": list(sessions)}


# Toy corpus: 6 sessions. Folds test on the last four; first two are train-only.
TOY_FOLDS = {
    "fold_01": {"train": ["d1"], "validation": ["d3"]},
    "fold_02": {"train": ["d1", "d3"], "validation": ["d4"]},
    "fold_03": {"train": ["d1", "d3", "d4"], "validation": ["d5"]},
    "fold_04": {"train": ["d1", "d3", "d4", "d5"], "validation": ["d6"]},
}
TOY_CORPUS = ["d1", "d2", "d3", "d4", "d5", "d6"]
# corpus - outer_test(d3,d4,d5,d6) = {d1, d2}
TOY_CENSUS_GOOD = ["d1", "d2"]


def test_disjoint_and_complete_passes():
    res = mod.verify(_runner_plan(TOY_FOLDS), _corpus(TOY_CORPUS), _census(TOY_CENSUS_GOOD))
    assert res["passed"] is True
    assert res["counts"] == {
        "governed_corpus": 6,
        "outer_test_union": 4,
        "census": 2,
        "intersection": 0,
    }
    assert res["intersection_sessions"] == []


def test_outer_test_union_maps_validation_role():
    union, detail = mod.outer_test_union(_runner_plan(TOY_FOLDS))
    assert union == {"d3", "d4", "d5", "d6"}
    assert detail["test_roles"] == ["validation"]


def test_overlap_is_caught():
    # d4 is an outer-test session; putting it in census must fail disjointness.
    res = mod.verify(
        _runner_plan(TOY_FOLDS), _corpus(TOY_CORPUS), _census(["d1", "d2", "d4"])
    )
    assert res["disjoint_census_and_outer_test"] is False
    assert res["intersection_sessions"] == ["d4"]
    assert res["passed"] is False


def test_incomplete_census_is_caught():
    # Dropping d2 (a legitimate non-test session) breaks completeness.
    res = mod.verify(_runner_plan(TOY_FOLDS), _corpus(TOY_CORPUS), _census(["d1"]))
    assert res["disjoint_census_and_outer_test"] is True
    assert res["census_equals_corpus_minus_outer_test"] is False
    assert res["missing_from_census"] == ["d2"]
    assert res["passed"] is False


def test_census_outside_corpus_is_caught():
    res = mod.verify(
        _runner_plan(TOY_FOLDS), _corpus(TOY_CORPUS), _census(["d1", "d2", "dX"])
    )
    assert res["census_within_corpus"] is False
    assert res["census_outside_corpus"] == ["dX"]
    assert res["passed"] is False


def test_no_test_role_is_ambiguous():
    import pytest

    bad = _runner_plan(TOY_FOLDS, role_map={"train": "train", "validation": "dev"})
    with pytest.raises(ValueError):
        mod.outer_test_union(bad)
