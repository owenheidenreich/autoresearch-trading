from __future__ import annotations

import json

from v4.live.protocol101_decision_trace import stable_hash
from v4.live.protocol101_paired_replay_diff import (
    PairedReplayDiffConfig,
    build_paired_replay_diff,
)
from v4.scripts.run_protocol101_captured_trace_readiness import audit_candidate_set
from v4.scripts.run_protocol101_extract_captured_decision_traces import (
    discover_log_files,
    extract_traces,
)


def test_audit_candidate_set_rejects_legacy_top_token_only_row() -> None:
    passed, errors = audit_candidate_set(
        {
            "event_type": "candidate_set",
            "session": "2026-06-15",
            "timestamp_utc": "2026-06-15T13:31:01Z",
            "top_token_diagnostics": [{"contract_id": "SPXW-20260615-6000-C"}],
        }
    )

    assert not passed
    assert "missing_protocol101_decision_trace_schema" in errors
    assert "missing_candidate_universe" in errors
    assert "missing_features_payload" in errors
    assert "missing_model_scores" in errors


def test_audit_candidate_set_accepts_pre_score_block_with_raw_quote_universe() -> None:
    candidate_universe = [
        {
            "contract_id": "SPXW-20260615-6000-C",
            "bid": 10.0,
            "ask": 10.2,
            "quote_age_ms": 20.0,
        }
    ]
    features = {"blocked_reason": "outside_time_bucket", "candidate_features": []}
    model_scores = {"surface_scores": [], "threshold": 25.0}

    passed, errors = audit_candidate_set(
        {
            "event_type": "candidate_set",
            "session": "2026-06-15",
            "timestamp_utc": "2026-06-15T13:31:01Z",
            "decision_trace_schema_version": "Protocol101DecisionTraceV1",
            "feature_contract_version": "protocol101-live-v1",
            "candidate_count": 0,
            "candidate_gate_diagnostics": {"filter_reason": "outside_time_bucket"},
            "candidate_universe": candidate_universe,
            "candidate_universe_hash": stable_hash(candidate_universe),
            "features": features,
            "feature_hash": stable_hash(features),
            "model_scores": model_scores,
            "score_hash": stable_hash(model_scores),
        }
    )

    assert passed
    assert errors == []


def test_extract_captured_full_trace_log_and_same_input_replay(tmp_path) -> None:
    session = "2026-06-15"
    live_root = tmp_path / "paper_trading"
    session_dir = live_root / session
    session_dir.mkdir(parents=True)
    log_path = session_dir / f"daily_paper_autopilot_{session}.jsonl"
    rows = _full_trace_rows(session=session)
    log_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    paths = discover_log_files(live_root, [session], [])
    traces, summary, failures = extract_traces(paths, mode="floor-minute")

    assert paths == [log_path]
    assert failures == []
    assert summary["candidate_set_rows"] == 1
    assert summary["full_trace_candidate_set_rows"] == 1
    assert len(traces) == 1
    trace = traces[0]
    assert trace["decision_ts"] == "2026-06-15T13:31:00+00:00"
    assert trace["feature_contract_version"] == "protocol101-live-v1"
    assert trace["selected_action"] == "enter"
    assert trace["selected_contract_id"] == "SPXW-20260615-6000-C"
    assert trace["risk_gate"]["passed"] is True
    assert trace["paper_account_state"]["net_liquidation"] == 10000.0

    same_input = build_paired_replay_diff(
        traces,
        traces,
        config=PairedReplayDiffConfig(mode="same_input"),
    )

    assert same_input["status"] == "pass"
    assert same_input["exact_matches"] == 1


def _full_trace_rows(*, session: str) -> list[dict]:
    candidate_universe = [
        {
            "contract_id": "SPXW-20260615-6000-C",
            "right": "C",
            "strike": 6000.0,
            "token_feature_hash": "token-hash-1",
        }
    ]
    features = {
        "token_features": [
            {
                "contract_id": "SPXW-20260615-6000-C",
                "token_feature_hash": "token-hash-1",
                "token_features": [0.1, 0.2, 0.3],
            }
        ]
    }
    model_scores = {"surface_scores": [31.5]}
    candidate_hash = stable_hash(candidate_universe)
    feature_hash = stable_hash(features)
    score_hash = stable_hash(model_scores)
    common = {
        "protocol_id": "protocol101",
        "session": session,
        "run_id": "unit",
        "mode": "paper-submit",
        "timestamp_utc": "2026-06-15T13:31:01Z",
        "decision_timestamp_utc": "2026-06-15T13:31:01Z",
        "candidate_universe_hash": candidate_hash,
    }
    return [
        {
            **common,
            "event_type": "candidate_set",
            "decision_trace_schema_version": "Protocol101DecisionTraceV1",
            "feature_contract_version": "protocol101-live-v1",
            "candidate_count": 1,
            "candidate_universe": candidate_universe,
            "features": features,
            "feature_hash": feature_hash,
            "model_scores": model_scores,
            "score_hash": score_hash,
            "model_threshold": 25.0,
            "source_quote_ts": "2026-06-15T13:31:00Z",
            "source_context_ts": "2026-06-15T13:31:00Z",
        },
        {
            **common,
            "event_type": "model_decision",
            "model_decision": {
                "selected_action": "enter",
                "selected_contract": {"contract_id": "SPXW-20260615-6000-C"},
                "score": 31.5,
                "threshold": 25.0,
            },
        },
        {
            **common,
            "event_type": "risk_gate",
            "risk_gate": {"passed": True, "reasons": []},
        },
        {
            **common,
            "event_type": "paper_account_state",
            "account": {"net_liquidation": 10000.0, "open_positions": 0},
        },
    ]
