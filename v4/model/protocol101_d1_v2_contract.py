"""Protocol101 D1 V2 and real-candidate incremental-edge gate law."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence


D1_V2_SCHEMA = "Protocol101D1MatchedRandomControlV2"
INCREMENTAL_EDGE_SCHEMA = "Protocol101CandidateIncrementalEdgeV1"
AMENDMENT_PATH = (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_"
    "AMENDMENT_2026_07_28.md"
)

ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
MONEYNESS_ATM_SLOT = 10
CANDIDATE_MULTIPLICITY_FAMILY_SIZE = 28
CANDIDATE_MULTIPLICITY_ROW_IDS = tuple(
    f"H{hypothesis}/P{policy}"
    for hypothesis in range(4)
    for policy in range(7)
)
CANDIDATE_MULTIPLICITY_FAMILY_SHA256 = hashlib.sha256(
    json.dumps(
        list(CANDIDATE_MULTIPLICITY_ROW_IDS),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()
MATCH_LIMITS = {
    "entry_intent_count_relative_drift": 0.0,
    "executed_trade_count_relative_drift": 0.05,
    "call_put_total_variation": 0.05,
    "moneyness_total_variation": 0.06,
    "executed_premium_mean_relative_drift": 0.10,
    "holding_minutes_mean_relative_drift": 0.10,
    "occupancy_minutes_relative_drift": 0.10,
}


class Protocol101D1V2Error(RuntimeError):
    """Raised when D1 V2 evidence is malformed or incomplete."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_probability(value: Any) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def _relative_drift(left: float, right: float) -> float:
    if math.isclose(left, 0.0, abs_tol=1e-12):
        return 0.0 if math.isclose(right, 0.0, abs_tol=1e-12) else math.inf
    return abs(left - right) / abs(left)


def _total_variation(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0 if left_total == right_total else 1.0
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(
            float(left.get(key, 0.0)) / left_total
            - float(right.get(key, 0.0)) / right_total
        )
        for key in keys
    )


def moneyness_distribution(
    slot_distribution: Mapping[str, int | float],
) -> dict[str, float]:
    result = {"ATM": 0.0, "NEAR": 0.0, "WING": 0.0}
    for slot, count in slot_distribution.items():
        distance = abs(int(slot) - MONEYNESS_ATM_SLOT)
        bucket = "ATM" if distance <= 1 else "NEAR" if distance <= 5 else "WING"
        result[bucket] += float(count)
    return result


def aggregate_exposure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Protocol101D1V2Error("exposure_rows_missing")
    entry_intents = int(sum(int(row["entry_intents"]) for row in rows))
    trades = int(sum(int(row["trades"]) for row in rows))
    call_put: Counter[str] = Counter()
    slots: Counter[str] = Counter()
    for row in rows:
        call_put.update(
            {str(key): int(value) for key, value in row["call_put_distribution"].items()}
        )
        slots.update(
            {str(key): int(value) for key, value in row["slot_distribution"].items()}
        )
    premium_total = sum(
        int(row["trades"]) * float(row["executed_premium_at_risk"]["mean"])
        for row in rows
    )
    occupancy = sum(
        int(row["trades"]) * float(row["executed_holding_minutes"]["mean"])
        for row in rows
    )
    return {
        "entry_intents": entry_intents,
        "trades": trades,
        "call_put_distribution": dict(sorted(call_put.items())),
        "slot_distribution": dict(
            sorted(slots.items(), key=lambda item: int(item[0]))
        ),
        "moneyness_distribution": moneyness_distribution(slots),
        "executed_premium_mean": (
            float(premium_total / trades) if trades else 0.0
        ),
        "holding_minutes_mean": (
            float(occupancy / trades) if trades else 0.0
        ),
        "occupancy_minutes": float(occupancy),
        "fee_adjusted_net_pnl": float(
            sum(float(row["net_pnl"]) for row in rows)
        ),
    }


def exposure_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    exact_opportunity_set_preserved: bool,
    random_selector_receipts_complete: bool,
) -> dict[str, Any]:
    metrics = {
        "entry_intent_count_relative_drift": _relative_drift(
            float(left["entry_intents"]), float(right["entry_intents"])
        ),
        "executed_trade_count_relative_drift": _relative_drift(
            float(left["trades"]), float(right["trades"])
        ),
        "call_put_total_variation": _total_variation(
            left["call_put_distribution"], right["call_put_distribution"]
        ),
        "moneyness_total_variation": _total_variation(
            left["moneyness_distribution"], right["moneyness_distribution"]
        ),
        "executed_premium_mean_relative_drift": _relative_drift(
            float(left["executed_premium_mean"]),
            float(right["executed_premium_mean"]),
        ),
        "holding_minutes_mean_relative_drift": _relative_drift(
            float(left["holding_minutes_mean"]),
            float(right["holding_minutes_mean"]),
        ),
        "occupancy_minutes_relative_drift": _relative_drift(
            float(left["occupancy_minutes"]), float(right["occupancy_minutes"])
        ),
    }
    metric_passes = {
        name: bool(value <= MATCH_LIMITS[name])
        for name, value in metrics.items()
    }
    blockers = [
        name for name, passed in metric_passes.items() if not passed
    ]
    if not exact_opportunity_set_preserved:
        blockers.append("opportunity_set_not_exact")
    if not random_selector_receipts_complete:
        blockers.append("random_selector_receipts_incomplete")
    return {
        "schema_version": "Protocol101StrictExposureMatchV1",
        "pass": not blockers,
        "metrics": metrics,
        "limits": dict(MATCH_LIMITS),
        "metric_passes": metric_passes,
        "exact_opportunity_set_preserved": exact_opportunity_set_preserved,
        "random_selector_receipts_complete": random_selector_receipts_complete,
        "blockers": sorted(blockers),
    }


def evaluate_d1_v2(
    *,
    exposure_matches: Sequence[Mapping[str, Any]],
    effect_fee_adjusted_pnl: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    multiplicity_adjusted_p: float | None,
    absolute_shuffled_pnl: float | None,
    equivalence_bound_dollars: float | None,
) -> dict[str, Any]:
    match_complete = bool(exposure_matches) and all(
        item.get("pass") is True for item in exposure_matches
    )
    evidence_complete = bool(
        all(
            _finite(value)
            for value in (
                effect_fee_adjusted_pnl,
                ci_lower,
                ci_upper,
            )
        )
        and _valid_probability(multiplicity_adjusted_p)
        and float(ci_lower) <= float(ci_upper)
    )
    positive_artifact = bool(
        evidence_complete
        and float(ci_lower) > 0.0
        and float(multiplicity_adjusted_p) <= ALPHA
    )
    if not match_complete:
        status = "INSUFFICIENT_MATCHED_CONTROL"
        passed = False
    elif not evidence_complete:
        status = "INSUFFICIENT_CONTRAST_EVIDENCE"
        passed = False
    elif positive_artifact:
        status = "POSITIVE_SHUFFLED_MODEL_ARTIFACT"
        passed = False
    else:
        status = "PASS_NO_POSITIVE_SHUFFLED_INCREMENT"
        passed = True
    equivalence_diagnostic = {
        "role": "diagnostic_only",
        "absolute_shuffled_pnl": absolute_shuffled_pnl,
        "bound_dollars": equivalence_bound_dollars,
        "ci_within_bound": (
            bool(
                _finite(ci_lower)
                and _finite(ci_upper)
                and _finite(equivalence_bound_dollars)
                and float(ci_lower) >= -float(equivalence_bound_dollars)
                and float(ci_upper) <= float(equivalence_bound_dollars)
            )
            if equivalence_bound_dollars is not None
            else None
        ),
        "changes_gate": False,
    }
    return {
        "schema_version": D1_V2_SCHEMA,
        "pass": passed,
        "status": status,
        "match_complete": match_complete,
        "evidence_complete": evidence_complete,
        "positive_artifact_detected": positive_artifact,
        "effect_fee_adjusted_pnl": effect_fee_adjusted_pnl,
        "confidence_interval_95": [ci_lower, ci_upper],
        "multiplicity_adjusted_p": multiplicity_adjusted_p,
        "alpha": ALPHA,
        "absolute_pnl_is_a_gate": False,
        "exposure_matches": list(exposure_matches),
        "equivalence_diagnostic": equivalence_diagnostic,
    }


def evaluate_candidate_incremental_edge(
    *,
    row_id: str,
    exposure_matches: Sequence[Mapping[str, Any]],
    effect_fee_adjusted_pnl: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    multiplicity_adjusted_p: float | None,
    multiplicity_family_size: int | None,
    multiplicity_family_ids: Sequence[str] | None,
    multiplicity_family_sha256: str | None,
    multiplicity_receipt_complete: bool,
    costs_included: bool,
    evidence_complete: bool,
) -> dict[str, Any]:
    match_complete = bool(exposure_matches) and all(
        item.get("pass") is True for item in exposure_matches
    )
    numeric_complete = bool(
        all(
            _finite(value)
            for value in (
                effect_fee_adjusted_pnl,
                ci_lower,
                ci_upper,
            )
        )
        and _valid_probability(multiplicity_adjusted_p)
        and float(ci_lower) <= float(ci_upper)
    )
    multiplicity_complete = bool(
        multiplicity_receipt_complete
        and multiplicity_family_size == CANDIDATE_MULTIPLICITY_FAMILY_SIZE
        and tuple(multiplicity_family_ids or ())
        == CANDIDATE_MULTIPLICITY_ROW_IDS
        and multiplicity_family_sha256
        == CANDIDATE_MULTIPLICITY_FAMILY_SHA256
    )
    passed = bool(
        evidence_complete
        and numeric_complete
        and multiplicity_complete
        and match_complete
        and costs_included
        and float(ci_lower) > 0.0
        and float(multiplicity_adjusted_p) <= ALPHA
    )
    blockers: list[str] = []
    if not evidence_complete or not numeric_complete:
        blockers.append("incremental_evidence_incomplete")
    if not multiplicity_complete:
        blockers.append("campaign_multiplicity_receipt_incomplete")
    if not match_complete:
        blockers.append("matched_random_exposure_incomplete")
    if not costs_included:
        blockers.append("costs_not_included")
    if numeric_complete and float(ci_lower) <= 0.0:
        blockers.append("confidence_interval_lower_not_strictly_positive")
    if (
        numeric_complete
        and float(multiplicity_adjusted_p) > ALPHA
    ):
        blockers.append("multiplicity_adjusted_p_above_alpha")
    return {
        "schema_version": INCREMENTAL_EDGE_SCHEMA,
        "row_id": row_id,
        "pass": passed,
        "effect_fee_adjusted_pnl": effect_fee_adjusted_pnl,
        "confidence_interval_95": [ci_lower, ci_upper],
        "multiplicity_adjusted_p": multiplicity_adjusted_p,
        "multiplicity_family_size": multiplicity_family_size,
        "multiplicity_family_ids": list(multiplicity_family_ids or ()),
        "multiplicity_family_sha256": multiplicity_family_sha256,
        "required_multiplicity_family_size": (
            CANDIDATE_MULTIPLICITY_FAMILY_SIZE
        ),
        "required_multiplicity_family_sha256": (
            CANDIDATE_MULTIPLICITY_FAMILY_SHA256
        ),
        "multiplicity_receipt_complete": multiplicity_complete,
        "alpha": ALPHA,
        "confidence_interval_lower_rule": "strictly_greater_than_zero",
        "costs_included": costs_included,
        "match_complete": match_complete,
        "evidence_complete": bool(evidence_complete and numeric_complete),
        "exposure_matches": list(exposure_matches),
        "blockers": sorted(blockers),
    }
