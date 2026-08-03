"""Compile prose-free hypothesis specs into immutable executable contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .entry_live_feature_catalog import lint_executable_live_twin
from .schema import COMPONENTS, SCHEMA_VERSION, TERMINAL_STATUSES, HypothesisSpec


CAUSAL_AVAILABILITY = frozenset(
    {
        "decision_time",
        "decision_emission",
        "interval_end_plus_frozen_lag",
        "session_open_static",
        "prior_day_eod_static",
    }
)
CAUSAL_THRESHOLDS = frozenset({"fixed", "training_quantile"})
KNOWN_MATERIALIZED_FIELDS = frozenset(
    {
        "policy_net_pnl",
        "policy_mid_pnl",
        "realized_exit_time",
        "source_exit_quote_time",
        "exit_quote_age",
        "exit_reason",
        "executable_exit_bid",
        "policy_deadline",
        "policy_neutral_primary_utility",
    }
)
REQUIRED_EXPOSURES = {
    "exit": frozenset({"entry_identity", "entry_count", "side", "premium", "moneyness"}),
    "timing": frozenset({"signal_identity", "contract_id", "side", "moneyness"}),
    "direction": frozenset({"decision_time", "premium", "absolute_moneyness"}),
    "contract_choice": frozenset({"decision_time", "side", "premium_caliper"}),
    "entry_policy": frozenset({"session", "fixed_time_block", "frozen_exit", "account_rules"}),
    "target": frozenset({"decision_time", "entry_count", "side", "premium", "moneyness"}),
}


class ExperimentCompileError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def semantic_payload(spec: HypothesisSpec) -> dict[str, Any]:
    """Return mechanics only; renaming or rewording cannot evade the registry."""

    payload = asdict(spec)
    payload.pop("hypothesis_id", None)
    payload.pop("claim", None)
    payload["feature_families_add"] = sorted(payload["feature_families_add"])
    payload["feature_families_remove"] = sorted(payload["feature_families_remove"])
    payload["features"] = sorted(payload["features"], key=lambda item: item["name"])
    payload["terminal_statuses"] = list(TERMINAL_STATUSES)
    return payload


def semantic_hash(spec: HypothesisSpec) -> str:
    return hashlib.sha256(canonical_json(semantic_payload(spec)).encode()).hexdigest()


@dataclass(frozen=True)
class CompiledHypothesis:
    spec: HypothesisSpec
    semantic_hash: str
    executable_guards: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "autoresearch_v2.compiled_hypothesis.v1",
            "semantic_hash": self.semantic_hash,
            "executable_guards": list(self.executable_guards),
            "spec": asdict(self.spec),
        }


def _lint(spec: HypothesisSpec, *, require_fit_ready: bool = True) -> list[str]:
    errors: list[str] = []
    if spec.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={spec.schema_version}")
    if spec.component not in COMPONENTS:
        errors.append(f"unknown component={spec.component}")
    names = [feature.name for feature in spec.features]
    if not names or len(names) != len(set(names)):
        errors.append("features must be nonempty and unique")
    for feature in spec.features:
        if feature.available_at not in CAUSAL_AVAILABILITY:
            errors.append(f"future_or_unknown_feature_availability:{feature.name}:{feature.available_at}")
        errors.extend(
            lint_executable_live_twin(feature, require_fit_ready=require_fit_ready)
        )
        if feature.name == "last_causal_open_interest" and feature.available_at != "prior_day_eod_static":
            errors.append("feature_without_live_twin:last_causal_open_interest:intraday")
    if spec.threshold.kind not in CAUSAL_THRESHOLDS:
        errors.append(f"future_session_threshold:{spec.threshold.kind}")
    if spec.threshold.kind == "training_quantile" and spec.threshold.fit_role != "model_fit":
        errors.append("training_quantile_must_be_fit_on_model_fit_only")
    if spec.threshold.kind == "fixed" and spec.threshold.fit_role != "none":
        errors.append("fixed_threshold_cannot_have_fit_role")
    if len(spec.target.fields) == 0 or len(spec.target.fields) != len(spec.target.weights):
        errors.append("target fields and weights must be nonempty and aligned")
    if abs(sum(spec.target.weights)) <= 0.0:
        errors.append("target weights cannot sum to zero")
    missing_materialization = sorted(
        set(spec.target.required_materialized_fields) - KNOWN_MATERIALIZED_FIELDS
    )
    if missing_materialization:
        errors.append("unmaterialized_target_fields:" + ",".join(missing_materialization))
    if len(spec.arms) < 2 or spec.primary_contrast not in spec.arms:
        errors.append("arms must include at least two values and primary_contrast")
    required = REQUIRED_EXPOSURES.get(spec.component, frozenset())
    missing_exposures = sorted(required - set(spec.exposure_matching))
    if missing_exposures:
        errors.append("missing_exposure_matching:" + ",".join(missing_exposures))
    if spec.model.family != "hist_gradient_boosting":
        errors.append("model_family_outside_tiny_budget")
    if not spec.model.seeds or spec.model.max_fits <= 0 or spec.model.max_fits > 10:
        errors.append("model_fit_budget_must_be_between_1_and_10")
    if spec.allowed_role != "development":
        errors.append("only_development_role_is_compilable")
    if spec.confirmation_rule != "fresh_epoch_once_then_roll_to_development":
        errors.append("confirmation_rule_must_be_one_shot_rolling_epoch")
    if not (0.0 < spec.alpha < 0.5 and 0.5 < spec.power < 1.0):
        errors.append("invalid_power_parameters")
    if (
        spec.practical_effect_dollars_per_session <= 0.0
        or spec.maximum_mde_dollars_per_session <= 0.0
    ):
        errors.append("effect_and_mde_bars_must_be_positive")
    if spec.max_permutations < 999 or spec.max_permutations > 100_000:
        errors.append("permutation_budget_outside_999_to_100000")
    if spec.max_minutes <= 0:
        errors.append("compute_minutes_must_be_positive")
    if spec.terminal_statuses != TERMINAL_STATUSES:
        errors.append("terminal_statuses_do_not_match_engine_contract")
    return errors


def compile_hypothesis(
    value: Mapping[str, Any] | HypothesisSpec, *, require_fit_ready: bool = True
) -> CompiledHypothesis:
    try:
        spec = value if isinstance(value, HypothesisSpec) else HypothesisSpec.from_mapping(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentCompileError((f"malformed_hypothesis:{exc}",)) from exc
    errors = _lint(spec, require_fit_ready=require_fit_ready)
    if errors:
        raise ExperimentCompileError(errors)
    return CompiledHypothesis(
        spec=spec,
        semantic_hash=semantic_hash(spec),
        executable_guards=(
            "development_role_only",
            "foundation_hash_before_decode",
            "executable_feature_live_twin_and_availability",
            "future_session_threshold_rejected",
            "mutate_future_invariance",
            "oof_cache_key_excludes_threshold",
            "session_blocked_family_maxT",
            "paired_delta_before_strict_serial",
            "semantic_duplicate_registry",
        ),
    )


def load_and_compile(path: str | Path) -> CompiledHypothesis:
    return compile_hypothesis(json.loads(Path(path).read_text()))
