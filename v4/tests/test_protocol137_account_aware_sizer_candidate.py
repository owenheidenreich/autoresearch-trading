from __future__ import annotations

from v4.scripts.run_protocol137_account_aware_sizer_candidate import decide


def test_decide_requires_025_gate() -> None:
    rows = [
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.25,
            "passes_cash_level_gate": False,
            "incremental_pnl": 1.0,
        },
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.50,
            "passes_cash_level_gate": True,
            "incremental_pnl": 1.0,
        },
    ]

    assert decide(rows, {"top_day_share_of_positive": 0.0, "top_month_share_of_positive": 0.0}) == "fragile_sizer_candidate_fails_025_gate"


def test_decide_requires_050_positive_incremental() -> None:
    rows = [
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.25,
            "passes_cash_level_gate": True,
            "incremental_pnl": 1.0,
        },
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.50,
            "passes_cash_level_gate": True,
            "incremental_pnl": -1.0,
        },
    ]

    assert decide(rows, {"top_day_share_of_positive": 0.0, "top_month_share_of_positive": 0.0}) == "fragile_sizer_candidate_fails_050_positive_incremental"


def test_decide_passes_clean_candidate() -> None:
    rows = [
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.25,
            "passes_cash_level_gate": True,
            "incremental_pnl": 1.0,
        },
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.50,
            "passes_cash_level_gate": True,
            "incremental_pnl": 1.0,
        },
    ]

    assert decide(rows, {"top_day_share_of_positive": 0.0, "top_month_share_of_positive": 0.0}) == "pass_account_aware_sizer_v1_research_candidate_not_live"

