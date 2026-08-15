"""Protocol101 live-vs-historical paired replay diffing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from v4.live.protocol101_decision_trace import Protocol101DecisionTrace, normalize_decision_trace


SCHEMA_VERSION = "Protocol101PairedReplayDiffV1"


@dataclass(frozen=True)
class PairedReplayDiffConfig:
    score_abs_tolerance: float = 1e-9
    threshold_adjacent_epsilon: float = 0.02
    candidate_count_tolerance: int = 0
    candidate_identity_overlap_min: float = 0.80
    mode: str = "cross_vendor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DECISION_MISMATCH_CATEGORIES = (
    "missing_live_trace",
    "missing_historical_trace",
    "candidate_universe_drift",
    "feature_drift",
    "feature_contract_mismatch",
    "score_drift",
    "action_drift",
    "lifecycle_drift",
    "account_risk_drift",
)
CROSS_VENDOR_HARD_FAILURE_CATEGORIES = (
    "missing_live_trace",
    "missing_historical_trace",
    "candidate_universe_drift",
    "feature_contract_mismatch",
    "action_drift",
    "lifecycle_drift",
    "account_risk_drift",
)
NON_ACTIONABLE_CANDIDATE_BLOCK_REASONS = frozenset(
    {
        "cooldown",
        "session_trade_cap",
        "daily_loss",
        "max_daily_loss",
    }
)


def diff_decision_traces(
    live: Protocol101DecisionTrace | dict[str, Any] | None,
    historical: Protocol101DecisionTrace | dict[str, Any] | None,
    *,
    config: PairedReplayDiffConfig = PairedReplayDiffConfig(),
) -> dict[str, Any]:
    live_trace = _trace(live, source="live")
    historical_trace = _trace(historical, source="historical")
    categories: list[str] = []

    if live_trace is None:
        categories.append("missing_live_trace")
    if historical_trace is None:
        categories.append("missing_historical_trace")
    if live_trace is None or historical_trace is None:
        return _row_result(live_trace, historical_trace, categories, config)

    candidate_overlap = _candidate_overlap(live_trace, historical_trace)
    candidate_sets_equal = live_trace.candidate_ids == historical_trace.candidate_ids
    if config.mode == "same_input":
        if not candidate_sets_equal or abs(live_trace.candidate_count - historical_trace.candidate_count) > config.candidate_count_tolerance:
            categories.append("candidate_universe_drift")
    elif not candidate_sets_equal:
        if candidate_overlap < config.candidate_identity_overlap_min:
            categories.append("candidate_universe_drift")
        else:
            categories.append("candidate_universe_variation")
    if live_trace.feature_contract_version != historical_trace.feature_contract_version:
        categories.append("feature_contract_mismatch")
    if (
        live_trace.source_quote_ts != historical_trace.source_quote_ts
        or live_trace.source_context_ts != historical_trace.source_context_ts
    ):
        categories.append("source_timestamp_drift")
    if live_trace.feature_hash != historical_trace.feature_hash:
        categories.append("feature_drift")
    if live_trace.score_hash != historical_trace.score_hash and not _score_close(live_trace, historical_trace, config):
        categories.append("score_drift")
    if live_trace.selected_action != historical_trace.selected_action or live_trace.selected_contract_id != historical_trace.selected_contract_id:
        categories.append("action_drift")
    if live_trace.lifecycle_action != historical_trace.lifecycle_action:
        categories.append("lifecycle_drift")
    live_has_account = _has_trace_object(live_trace, "paper_account_state", "account_state")
    historical_has_account = _has_trace_object(historical_trace, "paper_account_state", "account_state")
    if live_has_account != historical_has_account:
        categories.append("account_state_unavailable")
    account_drift = (
        live_has_account
        and historical_has_account
        and live_trace.account_state_hash != historical_trace.account_state_hash
    )
    if (
        account_drift
        or live_trace.risk_gate_hash != historical_trace.risk_gate_hash
        or live_trace.block_reasons != historical_trace.block_reasons
    ):
        categories.append("account_risk_drift")
    if live_trace.execution_hash != historical_trace.execution_hash:
        categories.append("execution_only_drift")
    if (
        config.mode != "same_input"
        and "candidate_universe_drift" in categories
        and _candidate_universe_drift_non_actionable(live_trace, historical_trace)
    ):
        categories = [
            "candidate_universe_drift_non_actionable"
            if category == "candidate_universe_drift"
            else category
            for category in categories
        ]
    return _row_result(live_trace, historical_trace, categories, config)


def build_paired_replay_diff(
    live_rows: Iterable[Protocol101DecisionTrace | dict[str, Any]],
    historical_rows: Iterable[Protocol101DecisionTrace | dict[str, Any]],
    *,
    config: PairedReplayDiffConfig = PairedReplayDiffConfig(),
) -> dict[str, Any]:
    live_by_key = {_key(_trace(row, source="live")): _trace(row, source="live") for row in live_rows}
    historical_by_key = {_key(_trace(row, source="historical")): _trace(row, source="historical") for row in historical_rows}
    keys = sorted(set(live_by_key) | set(historical_by_key))
    rows = [
        diff_decision_traces(live_by_key.get(key), historical_by_key.get(key), config=config)
        for key in keys
    ]
    category_counts: dict[str, int] = {}
    for row in rows:
        for category in row["mismatch_categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    exact_matches = sum(1 for row in rows if row["row_status"] == "exact_match")
    threshold_adjacent = sum(1 for row in rows if row["row_status"] == "threshold_adjacent_review")
    decision_failures = [row for row in rows if row["row_status"] == "decision_mismatch"]
    comparable_action_rows = [
        row for row in rows if row["live_action"] is not None and row["historical_action"] is not None
    ]
    action_matches = sum(
        1 for row in comparable_action_rows if row["live_action"] == row["historical_action"]
    )
    comparable_selected_contract_rows = [
        row
        for row in rows
        if row["live_contract_id"] is not None or row["historical_contract_id"] is not None
    ]
    selected_contract_matches = sum(
        1
        for row in comparable_selected_contract_rows
        if row["live_contract_id"] == row["historical_contract_id"]
    )
    if config.mode == "same_input":
        status = "pass" if exact_matches == len(rows) else "fail"
    else:
        status = "fail" if decision_failures else "pass" if exact_matches == len(rows) else "pass_with_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": config.mode,
        "status": status,
        "config": config.to_dict(),
        "rows": len(rows),
        "exact_matches": exact_matches,
        "threshold_adjacent_reviews": threshold_adjacent,
        "decision_failures": len(decision_failures),
        "action_matches": action_matches,
        "action_mismatches": len(comparable_action_rows) - action_matches,
        "selected_contract_matches": selected_contract_matches,
        "selected_contract_mismatches": (
            len(comparable_selected_contract_rows) - selected_contract_matches
        ),
        "category_counts": category_counts,
        "row_results": rows,
    }


def _row_result(
    live_trace: Protocol101DecisionTrace | None,
    historical_trace: Protocol101DecisionTrace | None,
    categories: list[str],
    config: PairedReplayDiffConfig,
) -> dict[str, Any]:
    categories = sorted(set(categories))
    hard_categories = (
        DECISION_MISMATCH_CATEGORIES
        if config.mode == "same_input"
        else CROSS_VENDOR_HARD_FAILURE_CATEGORIES
    )
    decision_categories = [category for category in categories if category in hard_categories]
    threshold_adjacent = _threshold_adjacent(live_trace, historical_trace, config)
    if not categories:
        status = "exact_match"
    elif categories == ["execution_only_drift"]:
        status = "execution_only_drift"
    elif threshold_adjacent and not any(
        category
        in categories
        for category in (
            "candidate_universe_drift",
            "feature_contract_mismatch",
            "account_risk_drift",
        )
    ):
        status = "threshold_adjacent_review"
    else:
        status = "decision_mismatch" if decision_categories else "review"
    return {
        "key": _key(live_trace or historical_trace),
        "row_status": status,
        "threshold_adjacent": threshold_adjacent,
        "mismatch_categories": categories,
        "live_action": None if live_trace is None else live_trace.selected_action,
        "historical_action": None if historical_trace is None else historical_trace.selected_action,
        "live_contract_id": None if live_trace is None else live_trace.selected_contract_id,
        "historical_contract_id": None if historical_trace is None else historical_trace.selected_contract_id,
        "live_threshold_distance": None if live_trace is None else live_trace.threshold_distance,
        "historical_threshold_distance": None if historical_trace is None else historical_trace.threshold_distance,
        "live_feature_contract_version": None if live_trace is None else live_trace.feature_contract_version,
        "historical_feature_contract_version": None if historical_trace is None else historical_trace.feature_contract_version,
        "live_source_quote_ts": None if live_trace is None else live_trace.source_quote_ts,
        "historical_source_quote_ts": None if historical_trace is None else historical_trace.source_quote_ts,
        "live_source_context_ts": None if live_trace is None else live_trace.source_context_ts,
        "historical_source_context_ts": None if historical_trace is None else historical_trace.source_context_ts,
        "candidate_identity_overlap": None
        if live_trace is None or historical_trace is None
        else _candidate_overlap(live_trace, historical_trace),
        "live_candidate_count": None if live_trace is None else live_trace.candidate_count,
        "historical_candidate_count": None if historical_trace is None else historical_trace.candidate_count,
    }


def _trace(row: Protocol101DecisionTrace | dict[str, Any] | None, *, source: str) -> Protocol101DecisionTrace | None:
    if row is None:
        return None
    if isinstance(row, Protocol101DecisionTrace):
        return row
    return normalize_decision_trace(row, source=source)


def _key(trace: Protocol101DecisionTrace | None) -> str:
    if trace is None:
        return "UNKNOWN|UNKNOWN"
    index = "" if trace.decision_index is None else f"|{trace.decision_index}"
    return f"{trace.session}|{trace.decision_ts}{index}"


def _score_close(
    live: Protocol101DecisionTrace,
    historical: Protocol101DecisionTrace,
    config: PairedReplayDiffConfig,
) -> bool:
    if live.selected_score is None or historical.selected_score is None:
        return False
    return abs(live.selected_score - historical.selected_score) <= config.score_abs_tolerance


def _candidate_overlap(live: Protocol101DecisionTrace, historical: Protocol101DecisionTrace) -> float:
    live_ids = set(live.candidate_ids)
    historical_ids = set(historical.candidate_ids)
    union = live_ids | historical_ids
    return 1.0 if not union else len(live_ids & historical_ids) / len(union)


def _candidate_universe_drift_non_actionable(
    live: Protocol101DecisionTrace,
    historical: Protocol101DecisionTrace,
) -> bool:
    """True when candidate drift cannot change the current decision.

    Cross-vendor quote/tradability differences still matter and remain visible,
    but they are not decision failures when both traces are already under the
    same hard no-entry account/risk block.  Same-input replay remains exact.
    """

    if live.selected_action != historical.selected_action or live.selected_action != "wait":
        return False
    if live.selected_contract_id != historical.selected_contract_id:
        return False
    if live.block_reasons != historical.block_reasons:
        return False
    reasons = set(live.block_reasons)
    return bool(reasons & NON_ACTIONABLE_CANDIDATE_BLOCK_REASONS)


def _has_trace_object(trace: Protocol101DecisionTrace, *names: str) -> bool:
    rows = [trace.payload]
    nested = trace.payload.get("payload") if isinstance(trace.payload, dict) else None
    if isinstance(nested, dict):
        rows.append(nested)
    return any(isinstance(row.get(name), dict) for row in rows for name in names)


def _threshold_adjacent(
    live: Protocol101DecisionTrace | None,
    historical: Protocol101DecisionTrace | None,
    config: PairedReplayDiffConfig,
) -> bool:
    distances = []
    for trace in (live, historical):
        if trace is not None and trace.threshold_distance is not None:
            distances.append(abs(trace.threshold_distance))
    return bool(distances) and min(distances) <= config.threshold_adjacent_epsilon
