"""Typed hypothesis contract for the lightweight experiment compiler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "autoresearch_v2.hypothesis.v1"
TERMINAL_STATUSES = (
    "INVALID_EXPERIMENT",
    "MECHANICAL_FAILURE",
    "UNDERPOWERED",
    "NO_INCREMENTAL_EDGE",
    "EXIT_ARTIFACT",
    "PROVISIONAL_EDGE",
    "CONFIRMED_EDGE",
)
COMPONENTS = frozenset(
    {"timing", "direction", "contract_choice", "entry_policy", "exit", "target"}
)


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    available_at: str
    live_twin: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Feature":
        return cls(**{key: str(value[key]) for key in ("name", "family", "available_at", "live_twin")})


@dataclass(frozen=True)
class Target:
    kind: str
    fields: tuple[str, ...]
    weights: tuple[float, ...]
    frozen_reference_exit: str
    required_materialized_fields: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Target":
        return cls(
            kind=str(value["kind"]),
            fields=tuple(str(item) for item in value["fields"]),
            weights=tuple(float(item) for item in value["weights"]),
            frozen_reference_exit=str(value["frozen_reference_exit"]),
            required_materialized_fields=tuple(
                str(item) for item in value.get("required_materialized_fields", ())
            ),
        )


@dataclass(frozen=True)
class Threshold:
    kind: str
    value: float
    fit_role: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Threshold":
        return cls(
            kind=str(value["kind"]),
            value=float(value["value"]),
            fit_role=str(value.get("fit_role", "none")),
        )


@dataclass(frozen=True)
class Model:
    family: str
    hyperparameters: Mapping[str, Any]
    seeds: tuple[int, ...]
    max_fits: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Model":
        return cls(
            family=str(value["family"]),
            hyperparameters=dict(value["hyperparameters"]),
            seeds=tuple(int(item) for item in value["seeds"]),
            max_fits=int(value["max_fits"]),
        )


@dataclass(frozen=True)
class HypothesisSpec:
    schema_version: str
    hypothesis_id: str
    claim: str
    component: str
    feature_families_add: tuple[str, ...]
    feature_families_remove: tuple[str, ...]
    features: tuple[Feature, ...]
    target: Target
    arms: tuple[str, ...]
    primary_contrast: str
    threshold: Threshold
    baseline_name: str
    exposure_matching: tuple[str, ...]
    model: Model
    epoch_id: str
    session_manifest: str
    allowed_role: str
    confirmation_rule: str
    practical_effect_dollars_per_session: float
    maximum_mde_dollars_per_session: float
    alpha: float
    power: float
    max_permutations: int
    max_minutes: int
    terminal_statuses: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HypothesisSpec":
        families = value["feature_families"]
        baseline = value["paired_baseline"]
        epoch = value["development_epoch"]
        power = value["power"]
        budget = value["compute_budget"]
        return cls(
            schema_version=str(value["schema_version"]),
            hypothesis_id=str(value["hypothesis_id"]),
            claim=str(value["claim"]),
            component=str(value["component"]),
            feature_families_add=tuple(str(item) for item in families["add"]),
            feature_families_remove=tuple(str(item) for item in families["remove"]),
            features=tuple(Feature.from_mapping(item) for item in value["features"]),
            target=Target.from_mapping(value["target"]),
            arms=tuple(str(item) for item in value["arms"]),
            primary_contrast=str(value["primary_contrast"]),
            threshold=Threshold.from_mapping(value["threshold"]),
            baseline_name=str(baseline["name"]),
            exposure_matching=tuple(str(item) for item in baseline["exposure_matching"]),
            model=Model.from_mapping(value["model"]),
            epoch_id=str(epoch["epoch_id"]),
            session_manifest=str(epoch["session_manifest"]),
            allowed_role=str(epoch["allowed_role"]),
            confirmation_rule=str(epoch["confirmation_rule"]),
            practical_effect_dollars_per_session=float(
                power["practical_effect_dollars_per_session"]
            ),
            maximum_mde_dollars_per_session=float(
                power["maximum_mde_dollars_per_session"]
            ),
            alpha=float(power["alpha"]),
            power=float(power["power"]),
            max_permutations=int(budget["max_permutations"]),
            max_minutes=int(budget["max_minutes"]),
            terminal_statuses=tuple(str(item) for item in value["terminal_statuses"]),
        )
