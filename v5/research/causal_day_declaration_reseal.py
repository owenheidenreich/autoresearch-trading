"""Fail-closed semantic equivalence for the action-value declaration reseal.

The V2 declarations accidentally hash the gate that must change to activate
their owner-signed scope.  A later declaration may therefore differ only in
mechanical sealing fields.  This module projects both declarations onto every
research choice that can change the answer and refuses any projected drift.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class DeclarationResealError(RuntimeError):
    """A declaration is corrupt or changes a frozen research law."""


SOURCE_FIT_DECLARATION = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V2.json"
)
SOURCE_EVALUATION_DECLARATION = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V2.json"
)
PINNED_RESEARCH_LAW_SHA256 = (
    "3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627"
)


FIT_RESEARCH_KEYS = (
    "architecture",
    "label",
    "training",
    "null",
    "inference",
    "selector_attainability",
    "evaluation",
    "inputs",
    "forbidden",
)

EVALUATION_RESEARCH_KEYS = (
    "architecture",
    "selector_attainability",
    "inference",
    "outcome_firewall",
    "matched_control",
    "shuffled_control",
    "inference_standard",
    "risk",
    "kill_conditions",
    "additional_completion_requirements",
    "failure_consequence",
    "feature_audit",
    "action_values",
    "forbidden",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def verify_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    expected = value.get("receipt_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DeclarationResealError(f"{name} lacks a SHA-256 self-hash")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if actual != expected:
        raise DeclarationResealError(
            f"{name} self-hash mismatch: {actual} != {expected}"
        )


def load_declaration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DeclarationResealError(f"declaration is not an object: {path}")
    verify_self_hash(value, name=str(path))
    return value


def _project(
    value: Mapping[str, Any], keys: tuple[str, ...], *, name: str
) -> dict[str, Any]:
    missing = [key for key in keys if key not in value]
    if missing:
        raise DeclarationResealError(f"{name} lacks research-law fields: {missing}")
    return {key: deepcopy(value[key]) for key in keys}


def research_law_projection(
    fit: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the complete outcome-affecting law, excluding sealing mechanics.

    Deliberately excluded fields are schema/revision labels, dates, prose
    purpose, supersession pointers, declaration and implementation hashes,
    current blocker text, signed-authority identifiers, future receipt paths,
    and unused output/evidence path versioning.
    """

    return {
        "fit": _project(fit, FIT_RESEARCH_KEYS, name="fit declaration"),
        "evaluation": _project(
            evaluation, EVALUATION_RESEARCH_KEYS, name="evaluation declaration"
        ),
        "gate_contract": {
            "fit_assertion_before_target_or_optimizer": fit.get("gate_law", {}).get(
                "assert_fit_permitted_before_target_load_or_optimizer"
            ),
            "fit_architecture": fit.get("gate_law", {}).get(
                "canonical_architecture_registered"
            ),
            "fit_corpus": fit.get("gate_law", {}).get("corpus"),
            "evaluation_calls_gate": evaluation.get("gate_law", {}).get(
                "fit_and_evaluation_both_call_gate"
            ),
            "evaluation_architecture": evaluation.get("gate_law", {}).get(
                "canonical_architecture_registered"
            ),
            "evaluation_corpus": evaluation.get("gate_law", {}).get("corpus"),
        },
    }


def research_law_sha256(
    fit: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> str:
    return hashlib.sha256(canonical_json(research_law_projection(fit, evaluation))).hexdigest()


def load_pinned_source_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    fit = load_declaration(SOURCE_FIT_DECLARATION)
    evaluation = load_declaration(SOURCE_EVALUATION_DECLARATION)
    actual = research_law_sha256(fit, evaluation)
    if actual != PINNED_RESEARCH_LAW_SHA256:
        raise DeclarationResealError(
            f"V2 research-law projection drift: {actual} != {PINNED_RESEARCH_LAW_SHA256}"
        )
    return fit, evaluation


def _differences(left: Any, right: Any, path: str = "$") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                differences.append(f"{child}: added")
            elif key not in right:
                differences.append(f"{child}: removed")
            else:
                differences.extend(_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}: length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def assert_mechanical_reseal(
    source_fit: Mapping[str, Any],
    source_evaluation: Mapping[str, Any],
    candidate_fit: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
) -> str:
    """Prove candidate declarations change no outcome-affecting choice."""

    for name, value in (
        ("source fit", source_fit),
        ("source evaluation", source_evaluation),
        ("candidate fit", candidate_fit),
        ("candidate evaluation", candidate_evaluation),
    ):
        verify_self_hash(value, name=name)
    for name, fit, evaluation in (
        ("source", source_fit, source_evaluation),
        ("candidate", candidate_fit, candidate_evaluation),
    ):
        linked = evaluation.get("fit_declaration", {}).get("self_hash")
        if linked != fit.get("receipt_sha256"):
            raise DeclarationResealError(
                f"{name} evaluation does not bind its paired fit declaration self-hash"
            )
    source = research_law_projection(source_fit, source_evaluation)
    candidate = research_law_projection(candidate_fit, candidate_evaluation)
    differences = _differences(source, candidate)
    if differences:
        preview = "\n  ".join(differences[:20])
        remainder = len(differences) - min(len(differences), 20)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise DeclarationResealError(
            "mechanical reseal changes frozen research law:\n  " + preview + suffix
        )
    return hashlib.sha256(canonical_json(source)).hexdigest()
