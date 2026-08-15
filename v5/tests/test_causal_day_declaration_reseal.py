from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from v5.research.causal_day_declaration_reseal import (
    DeclarationResealError,
    assert_mechanical_reseal,
    canonical_json,
    load_declaration,
    research_law_sha256,
)
from v5.ops.build_causal_day_dataset import file_sha256


FIT = Path("v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V2.json")
EVALUATION = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V2.json"
)
FIT_V3 = Path("v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V3.json")
EVALUATION_V3 = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V3.json"
)


def _reseal(value: dict) -> None:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _reseal_pair(fit: dict, evaluation: dict) -> None:
    _reseal(fit)
    evaluation["fit_declaration"]["self_hash"] = fit["receipt_sha256"]
    _reseal(evaluation)


def _source() -> tuple[dict, dict]:
    return load_declaration(FIT), load_declaration(EVALUATION)


def test_v2_research_law_has_one_stable_hash() -> None:
    fit, evaluation = _source()
    assert research_law_sha256(fit, evaluation) == (
        "3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627"
    )


def test_v3_is_a_zero_semantic_drift_reseal_with_current_implementations() -> None:
    source_fit, source_evaluation = _source()
    candidate_fit = load_declaration(FIT_V3)
    candidate_evaluation = load_declaration(EVALUATION_V3)
    assert assert_mechanical_reseal(
        source_fit, source_evaluation, candidate_fit, candidate_evaluation
    ) == research_law_sha256(source_fit, source_evaluation)
    for declaration in (candidate_fit, candidate_evaluation):
        for path, digest in declaration["implementation_hashes"].items():
            assert file_sha256(Path(path)) == digest, path


def test_only_mechanical_sealing_fields_may_change() -> None:
    source_fit, source_evaluation = _source()
    candidate_fit = deepcopy(source_fit)
    candidate_evaluation = deepcopy(source_evaluation)
    candidate_fit.update(
        {
            "schema_version": "v5.causal-day-action-value-fit-declaration.v3",
            "supersedes": str(FIT),
            "implementation_hashes": {"future.py": "1" * 64},
            "current_fit_blockers": [],
            "outputs": {"root": "future-fit-v3", "evidence_dir": "future-fit-attempt003"},
        }
    )
    candidate_fit["gate_law"]["reopening_sha256"] = "2" * 64
    candidate_evaluation.update(
        {
            "schema_version": "v5.causal-day-action-value-evaluation-declaration.v3",
            "supersedes": str(EVALUATION),
            "implementation_hashes": {"future.py": "3" * 64},
            "current_fit_blockers": [],
            "future_fit_receipt": "future-fit-attempt003/receipt.json",
            "outputs": {
                "root": "future-economics-v3",
                "evidence_dir": "future-economics-attempt003",
            },
        }
    )
    candidate_evaluation["gate_law"]["reopening_sha256"] = "2" * 64
    candidate_evaluation["fit_declaration"] = {
        "path": "ACTION_VALUE_FIT_DECLARATION_V3.json",
        "sha256": "4" * 64,
        "self_hash": "replaced by dependency-ordered reseal",
    }
    _reseal_pair(candidate_fit, candidate_evaluation)
    assert assert_mechanical_reseal(
        source_fit, source_evaluation, candidate_fit, candidate_evaluation
    ) == research_law_sha256(source_fit, source_evaluation)


@pytest.mark.parametrize(
    ("document", "path", "value"),
    (
        ("fit", ("training", "seed"), 7),
        ("fit", ("label", "horizon_minutes"), 90),
        ("fit", ("selector_attainability", "no_trade_floor"), -1.0),
        ("evaluation", ("inference", "trade_cap"), 2),
        ("evaluation", ("risk", "maximum_loss_usd"), 1_000.0),
        ("evaluation", ("inference_standard", "declared_family_size"), 1),
    ),
)
def test_any_research_law_change_is_refused(
    document: str, path: tuple[str, str], value: object
) -> None:
    source_fit, source_evaluation = _source()
    candidate_fit = deepcopy(source_fit)
    candidate_evaluation = deepcopy(source_evaluation)
    candidate = candidate_fit if document == "fit" else candidate_evaluation
    candidate[path[0]][path[1]] = value
    _reseal_pair(candidate_fit, candidate_evaluation)
    with pytest.raises(DeclarationResealError, match="changes frozen research law"):
        assert_mechanical_reseal(
            source_fit, source_evaluation, candidate_fit, candidate_evaluation
        )


def test_candidate_evaluation_must_bind_candidate_fit() -> None:
    source_fit, source_evaluation = _source()
    candidate_fit = deepcopy(source_fit)
    candidate_evaluation = deepcopy(source_evaluation)
    candidate_evaluation["fit_declaration"]["self_hash"] = "0" * 64
    _reseal(candidate_evaluation)
    with pytest.raises(DeclarationResealError, match="does not bind"):
        assert_mechanical_reseal(
            source_fit, source_evaluation, candidate_fit, candidate_evaluation
        )
