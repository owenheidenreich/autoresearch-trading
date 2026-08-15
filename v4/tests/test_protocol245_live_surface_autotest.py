from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v4.live.paper_trade_log import load_trade_log, validate_trade_log
from v4.scripts.run_protocol245_premium_blend_live_no_order_surface_check import (
    append_runtime_event,
    candidate_set_summary,
    decision_from_validation,
    expected_contract_count,
    summarize_candidate_breadth,
    validate_live_surface_summary,
)


def test_expected_contract_count_for_atm_50_window() -> None:
    assert expected_contract_count(10) == 42


def test_live_surface_validation_requires_full_ladder_and_no_broker_rows(tmp_path: Path) -> None:
    trade_log = tmp_path / "events.jsonl"
    append_runtime_event(
        trade_log,
        event_type="candidate_set",
        session="2026-05-26",
        run_id="test",
        paper_cash=10_000.0,
        reason="candidate_set",
        market_snapshot={"underlying": {"spx": 7385.0, "vix": 18.0}, "option_nbbo": {}, "context": {}},
        model_decision={"action": "candidate_set"},
    )
    rows = [
        {
            "candidate_set": {
                "requested_contracts": 42,
                "qualified_contracts": 42,
                "raw_quote_count": 40,
                "candidate_count": 40,
                "valid_candidate_count": 38,
                "expected_raw_contracts": 42,
                "call_count": 20,
                "put_count": 20,
                "max_abs_offset": 50.0,
            }
        }
    ]
    breadth = summarize_candidate_breadth(rows, expected_raw=42)
    validation = validate_live_surface_summary(
        rows=rows,
        trade_rows=load_trade_log(trade_log),
        candidate_breadth=breadth,
        min_valid_candidates=30,
        broker_order_endpoint_called=False,
    )

    assert validate_trade_log(load_trade_log(trade_log))["status"] == "pass"
    assert validation["status"] == "pass"
    assert decision_from_validation(validation).startswith("pass_live_surface_autotest")


def test_live_surface_validation_blocks_narrow_recorded_surface() -> None:
    rows = [
        {
            "candidate_set": {
                "requested_contracts": 10,
                "qualified_contracts": 10,
                "raw_quote_count": 10,
                "candidate_count": 10,
                "valid_candidate_count": 10,
                "expected_raw_contracts": 42,
                "call_count": 5,
                "put_count": 5,
            }
        }
    ]
    breadth = summarize_candidate_breadth(rows, expected_raw=42)
    validation = validate_live_surface_summary(
        rows=rows,
        trade_rows=[],
        candidate_breadth=breadth,
        min_valid_candidates=30,
        broker_order_endpoint_called=False,
    )

    assert validation["status"] == "fail"
    assert "full_ladder_not_requested" in validation["errors"]
    assert "insufficient_valid_live_candidates" in validation["errors"]


def test_candidate_set_summary_handles_missing_quote_age_column() -> None:
    candidates = pd.DataFrame(
        [
            {
                "contract_id": "SPXW-20260526-07500.000-C",
                "root": "SPXW",
                "settlement_style": "PM",
                "right": "C",
                "offset": 0.0,
            }
        ]
    )
    summary = candidate_set_summary(
        candidates,
        option_quotes=[{"contract_id": "SPXW-20260526-07500.000-C"}],
        mask=np.asarray([True]),
        chain_meta={"requested_contracts": 1, "qualified_contracts": 1},
        strikes_around_atm=0,
    )

    assert summary["max_option_quote_age_ms"] == 0.0
