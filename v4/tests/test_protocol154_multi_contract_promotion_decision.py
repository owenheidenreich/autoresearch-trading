from __future__ import annotations

from v4.scripts.run_protocol154_multi_contract_promotion_decision import (
    critical_timing_gate,
    decide,
    missing_timing_coverage,
)


def _gate(name: str, status: str) -> dict:
    return {"gate": name, "status": status, "reason": status, "metric": None, "threshold": None}


def test_protocol154_decide_blocks_before_reject_free_promotion() -> None:
    assert decide([_gate("base", "pass"), _gate("timing", "block")]) == (
        "blocked_multi_contract_promotion_missing_timing_evidence"
    )


def test_protocol154_decide_rejects_hard_failures() -> None:
    assert decide([_gate("base", "fail"), _gate("timing", "block")]) == (
        "reject_multi_contract_promotion_hard_gate_failed"
    )


def test_protocol154_decide_promotes_only_when_all_hard_gates_pass() -> None:
    assert decide([_gate("base", "pass"), _gate("timing", "pass"), _gate("warning", "warn")]) == (
        "promote_multi_contract_research_candidate_to_paper_replay_gate"
    )


def test_protocol154_missing_timing_coverage_counts_rows_to_95pct() -> None:
    rows = [{"split": "q1_2026", "delay_seconds": 1, "rows": 359, "coverage": 0.749304}]

    missing = missing_timing_coverage(rows)

    assert missing == [
        {
            "split": "q1_2026",
            "delay_seconds": 1,
            "rows": 359,
            "covered_rows": 269,
            "required_rows": 342,
            "additional_rows_needed": 73,
            "coverage": 0.749304,
            "required_coverage": 0.95,
        }
    ]


def test_protocol154_critical_timing_is_blocked_for_incomplete_positive_coverage() -> None:
    timing = [
        {
            "split": "q1_2026",
            "delay_seconds": 1,
            "rows": 100,
            "coverage": 0.90,
            "challenger_delayed_pnl": 100.0,
            "incremental_delayed_pnl": 10.0,
        },
        {
            "split": "q1_2026",
            "delay_seconds": 5,
            "rows": 100,
            "coverage": 0.90,
            "challenger_delayed_pnl": 100.0,
            "incremental_delayed_pnl": 10.0,
        },
    ]

    gate = critical_timing_gate(timing)

    assert gate["status"] == "block"
    assert "incomplete high-resolution coverage" in gate["reason"]


def test_protocol154_critical_timing_rejects_negative_delay_pnl() -> None:
    timing = [
        {
            "split": "q1_2026",
            "delay_seconds": 1,
            "rows": 100,
            "coverage": 1.0,
            "challenger_delayed_pnl": 100.0,
            "incremental_delayed_pnl": -1.0,
        }
    ]

    gate = critical_timing_gate(timing)

    assert gate["status"] == "fail"
    assert "nonpositive challenger or incremental PnL" in gate["reason"]
