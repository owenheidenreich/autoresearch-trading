from __future__ import annotations

from v4.live.protocol101_decision_trace import (
    SCHEMA_VERSION,
    normalize_decision_trace,
    validate_decision_trace,
)
from v4.live.protocol101_paired_replay_diff import (
    PairedReplayDiffConfig,
    build_paired_replay_diff,
    diff_decision_traces,
)


def test_protocol101_decision_trace_normalizes_nested_live_row() -> None:
    trace = normalize_decision_trace(_row(score=0.72, threshold=0.70), source="live")

    assert trace.schema_version == SCHEMA_VERSION
    assert trace.protocol_id == "protocol101"
    assert trace.selected_action == "enter"
    assert trace.selected_contract_id == "SPXW-20260527-06000-C"
    assert trace.candidate_count == 2
    assert trace.feature_contract_version == "protocol101-live-v1"
    assert trace.source_quote_ts == "2026-05-27T13:30:59Z"
    assert trace.threshold_distance == 0.02
    assert validate_decision_trace(trace)["status"] == "pass"


def test_same_input_paired_replay_requires_exact_trace_match() -> None:
    live = _row(score=0.72, threshold=0.70)
    historical = _row(score=0.72, threshold=0.70)

    result = build_paired_replay_diff(
        [live],
        [historical],
        config=PairedReplayDiffConfig(mode="same_input"),
    )

    assert result["status"] == "pass"
    assert result["exact_matches"] == 1


def test_cross_vendor_threshold_adjacent_action_mismatch_is_review_not_failure() -> None:
    live = _row(score=0.701, threshold=0.70, action="enter")
    historical = _row(score=0.699, threshold=0.70, action="wait", contract_id=None)

    result = diff_decision_traces(
        live,
        historical,
        config=PairedReplayDiffConfig(mode="cross_vendor", threshold_adjacent_epsilon=0.005),
    )

    assert result["row_status"] == "threshold_adjacent_review"
    assert "action_drift" in result["mismatch_categories"]


def test_cross_vendor_feature_drift_is_review_when_action_matches() -> None:
    live = _row(score=0.72, threshold=0.70, feature_value=1.0)
    historical = _row(score=0.72, threshold=0.70, feature_value=2.0)

    result = build_paired_replay_diff(
        [live],
        [historical],
        config=PairedReplayDiffConfig(mode="cross_vendor"),
    )

    assert result["status"] == "pass_with_review"
    assert result["category_counts"]["feature_drift"] == 1
    assert result["action_matches"] == 1
    assert result["action_mismatches"] == 0
    assert result["selected_contract_matches"] == 1
    assert result["selected_contract_mismatches"] == 0


def test_cross_vendor_feature_contract_mismatch_fails() -> None:
    live = _row(score=0.72, threshold=0.70, feature_contract_version="protocol101-live-v1")
    historical = _row(score=0.72, threshold=0.70, feature_contract_version="historical-default")

    result = build_paired_replay_diff(
        [live],
        [historical],
        config=PairedReplayDiffConfig(mode="cross_vendor"),
    )

    assert result["status"] == "fail"
    assert result["category_counts"]["feature_contract_mismatch"] == 1


def test_cross_vendor_source_timestamp_drift_is_review_category() -> None:
    live = _row(score=0.90, threshold=0.70, source_quote_ts="2026-05-27T13:30:58Z")
    historical = _row(score=0.90, threshold=0.70, source_quote_ts="2026-05-27T13:30:59Z")

    result = diff_decision_traces(live, historical, config=PairedReplayDiffConfig(mode="cross_vendor"))

    assert result["row_status"] == "review"
    assert "source_timestamp_drift" in result["mismatch_categories"]


def test_cross_vendor_candidate_identity_uses_bounded_overlap() -> None:
    live = _row(score=0.72, threshold=0.70)
    historical = _row(score=0.72, threshold=0.70)
    historical["candidate_set"].append({"contract_id": "SPXW-20260527-06010-C"})

    result = diff_decision_traces(
        live,
        historical,
        config=PairedReplayDiffConfig(mode="cross_vendor", candidate_identity_overlap_min=0.60),
    )

    assert "candidate_universe_variation" in result["mismatch_categories"]
    assert "candidate_universe_drift" not in result["mismatch_categories"]
    assert result["candidate_identity_overlap"] == 2 / 3


def test_cross_vendor_candidate_drift_is_review_when_same_hard_block_prevents_entry() -> None:
    live = _row(score=0.72, threshold=0.70, action="wait", contract_id=None)
    historical = _row(score=0.72, threshold=0.70, action="wait", contract_id=None)
    live["block_reasons"] = ["cooldown"]
    historical["block_reasons"] = ["cooldown"]
    historical["candidate_set"] = [
        {"contract_id": "SPXW-20260527-06010-C"},
        {"contract_id": "SPXW-20260527-06015-C"},
    ]

    result = build_paired_replay_diff(
        [live],
        [historical],
        config=PairedReplayDiffConfig(mode="cross_vendor", candidate_identity_overlap_min=0.80),
    )

    assert result["status"] == "pass_with_review"
    assert result["decision_failures"] == 0
    assert result["category_counts"]["candidate_universe_drift_non_actionable"] == 1


def _row(
    *,
    score: float,
    threshold: float,
    action: str = "enter",
    contract_id: str | None = "SPXW-20260527-06000-C",
    feature_value: float = 1.0,
    feature_contract_version: str = "protocol101-live-v1",
    source_quote_ts: str = "2026-05-27T13:30:59Z",
) -> dict:
    selected_contract = None if contract_id is None else {"contract_id": contract_id, "root": "SPXW"}
    return {
        "protocol_id": "protocol101",
        "session": "2026-05-27",
        "run_id": "unit",
        "timestamp": "2026-05-27T13:31:00Z",
        "feature_contract_version": feature_contract_version,
        "source_quote_ts": source_quote_ts,
        "source_context_ts": "2026-05-27T13:31:00Z",
        "mode": "paper-submit",
        "candidate_set": [
            {"contract_id": "SPXW-20260527-06000-C"},
            {"contract_id": "SPXW-20260527-06005-C"},
        ],
        "model_decision": {
            "score": score,
            "threshold": threshold,
            "features": {"x": feature_value},
        },
        "selected_action": action,
        "selected_contract": selected_contract,
        "risk_gate": {"passed": True},
        "paper_account_state": {"cash_available": 10_000.0, "open_positions": 0},
        "market_snapshot": {
            "underlying": {"spx": 6000.0, "vix": 15.0, "context_age_ms": 100.0},
            "option_nbbo": {"bid": 10.0, "ask": 10.2, "quote_age_ms": 50.0},
        },
    }
