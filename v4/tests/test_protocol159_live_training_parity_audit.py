from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS
from v4.scripts.run_protocol159_live_training_parity_audit import build_audit_payload


def test_protocol159_live_training_parity_passes_with_complete_live_context(tmp_path: Path) -> None:
    surface_manifest = tmp_path / "surface.json"
    protocol101_manifest = tmp_path / "protocol101.json"
    surface_manifest.write_text(
        json.dumps({"variant_name": "surface_structure_aplus_side_value_rank", "policy_index": 1})
    )
    protocol101_manifest.write_text(json.dumps({"feature_columns": list(FEATURE_COLUMNS)}))
    rows = [
        {
            "event_type": "market_snapshot",
            "timestamp": "2026-05-19T14:30:00+00:00",
            "session": "2026-05-19",
            "run_id": "test",
            "market_snapshot": {
                "underlying": {"spx": 6000.0, "vix": 18.0},
                "option_nbbo": {"quote_count": 42},
                "context": {
                    "source": "ibkr_live",
                    "spx_market_data_type": "live",
                    "vix_market_data_type": "live",
                },
            },
        },
        {
            "event_type": "candidate_set",
            "timestamp": "2026-05-19T15:30:00+00:00",
            "session": "2026-05-19",
            "run_id": "test",
            "candidate_count": 0,
            "candidate_gate_diagnostics": {"filter_reason": "below_min_edge", "time_bucket": "post_open_morning"},
            "live_index_context": {
                "row_count": 61,
                "minute_row_count": 61,
                "first_timestamp": "2026-05-19T13:30:00+00:00",
                "last_timestamp": "2026-05-19T14:30:00+00:00",
                "span_minutes": 60.0,
            },
        },
    ]

    payload = build_audit_payload(
        rows=rows,
        trade_log=tmp_path / "log.jsonl",
        surface_manifest=surface_manifest,
        protocol101_manifest=protocol101_manifest,
        min_context_minutes=30.0,
        max_open_start_lag_minutes=5.0,
    )

    assert not payload["decision"].startswith("fail_")
    assert all(row["status"] == "pass" for row in payload["checks"])


def test_protocol159_accepts_market_data_type_on_underlying_snapshot(tmp_path: Path) -> None:
    surface_manifest = tmp_path / "surface.json"
    protocol101_manifest = tmp_path / "protocol101.json"
    surface_manifest.write_text(
        json.dumps({"variant_name": "surface_structure_aplus_side_value_rank", "policy_index": 1})
    )
    protocol101_manifest.write_text(json.dumps({"feature_columns": list(FEATURE_COLUMNS)}))
    rows = [
        {
            "event_type": "market_snapshot",
            "timestamp": "2026-05-19T14:30:00+00:00",
            "session": "2026-05-19",
            "run_id": "test",
            "market_snapshot": {
                "underlying": {
                    "spx": 6000.0,
                    "vix": 18.0,
                    "spx_market_data_type": "live",
                    "vix_market_data_type": "live",
                },
                "option_nbbo": {"quote_count": 42},
                "context": {"source": "ibkr_live"},
            },
        },
        {
            "event_type": "candidate_set",
            "timestamp": "2026-05-19T15:30:00+00:00",
            "session": "2026-05-19",
            "run_id": "test",
            "candidate_count": 0,
            "candidate_gate_diagnostics": {"filter_reason": "below_min_edge", "time_bucket": "post_open_morning"},
            "live_index_context": {
                "row_count": 61,
                "minute_row_count": 61,
                "first_timestamp": "2026-05-19T13:30:00+00:00",
                "last_timestamp": "2026-05-19T14:30:00+00:00",
                "span_minutes": 60.0,
            },
        },
    ]

    payload = build_audit_payload(
        rows=rows,
        trade_log=tmp_path / "log.jsonl",
        surface_manifest=surface_manifest,
        protocol101_manifest=protocol101_manifest,
        min_context_minutes=30.0,
        max_open_start_lag_minutes=5.0,
    )

    market_check = next(row for row in payload["checks"] if row["name"] == "latest_market_data_is_live")
    assert market_check["status"] == "pass"
