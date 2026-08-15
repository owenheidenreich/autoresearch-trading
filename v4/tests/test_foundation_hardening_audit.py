import math

import pandas as pd

from v4.scripts import run_foundation_hardening_audit as audit


def test_decision_blocks_when_any_foundation_gate_is_blocked() -> None:
    checks = [
        {"name": "paper default", "status": "pass"},
        {"name": "fill model calibration", "status": "blocked"},
    ]

    assert audit.decide(checks) == "foundation_hardening_required_before_model_experiments"


def test_summarize_skips_keeps_missing_quote_premiums_as_none() -> None:
    skips = pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026"],
            "skip_reason": ["missing_contract_quotes", "missing_contract_quotes"],
            "entry_premium": [math.nan, math.nan],
            "account_equity": [math.nan, math.nan],
        }
    )

    rows = audit.summarize_skips(skips)

    assert rows == [
        {
            "reported_split": "q1_2026",
            "skip_reason": "missing_contract_quotes",
            "rows": 2,
            "median_entry_premium": None,
            "median_account_equity": None,
        }
    ]


def test_integrated_label_mismatch_counts_negative_and_positive_losing_trades() -> None:
    trades = pd.DataFrame(
        {
            "a_enter": [-10.0, 50.0, 100.0, -5.0],
            "pnl": [15.0, -20.0, 40.0, -30.0],
        }
    )

    mismatch = audit.integrated_label_mismatch(trades)

    assert mismatch["negative_a_enter_trades_taken"] == 2
    assert mismatch["negative_a_enter_trades_pnl"] == -15.0
    assert mismatch["positive_a_enter_losing_trades"] == 1
    assert mismatch["positive_a_enter_losing_pnl"] == -20.0


def test_foundation_checks_preserve_engineer_response_gates() -> None:
    summaries = {
        "protocol269": {
            "historical_replay_proxy": True,
            "decision": "runtime_protocol265_no_order_parity_passed_historical_proxy",
        },
        "protocol272": {
            "decision": "blocked_insufficient_fill_observations_keep_stress_replay",
            "readiness": {
                "status": "blocked_insufficient_fill_observations",
                "fill_observations": 0,
                "required_fill_observations": 30,
            },
        },
        "protocol273": {
            "decision": "model_selection_overfit_risk_confirmed_reserve_new_untouched_block",
        },
        "protocol276": {
            "decision": "research_only_integrated_entry_lifecycle_does_not_surpass_protocol101",
        },
    }
    attribution = {"trade_rows": 197, "skip_rows": 16_799}
    label_alignment = {
        "integrated_trade_label_mismatch": {
            "negative_a_enter_trades_taken": 165,
        }
    }

    checks = audit.build_foundation_checks(summaries, attribution, label_alignment)
    by_name = {item["name"]: item for item in checks}

    assert by_name["Fill model calibration"]["status"] == "blocked"
    assert by_name["Untouched holdout reservation"]["status"] == "blocked"
    assert by_name["Challenger runtime parity"]["status"] == "partial"
    assert by_name["Unified label/policy alignment"]["status"] == "blocked"
    assert audit.decide(checks) == "foundation_hardening_required_before_model_experiments"


def test_table_overflow_row_matches_column_count() -> None:
    rendered = audit.table([{"a": 1, "b": 2}, {"a": 3, "b": 4}], ["a", "b"], max_rows=1)

    assert rendered.splitlines()[-1] == "| ... | 1 more rows |"


def test_prioritized_checklist_keeps_model_work_after_foundation_gates() -> None:
    checklist = audit.prioritized_checklist()

    assert checklist[0]["item"] == "Stop neural experiments after the completed preregistered run"
    assert checklist[-1]["item"] == "Defer broad historical data purchase"
    assert any("Protocol101 as the strict serial baseline" in item["required_evidence"] for item in checklist)
