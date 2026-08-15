import pandas as pd

from v4.scripts import attribute_protocol276_integrated_lifecycle_failure as attr


def test_lifecycle_timing_mode_detects_forced_flat_after_oracle_exit() -> None:
    mode = attr.lifecycle_timing_mode(
        actual_pnl=100.0,
        actual_exit_index=384,
        model_exit_index=384,
        first_oracle_exit_index=100,
        best_pnl=1200.0,
        exit_reason="forced_flat_no_lifecycle_exit_signal",
        giveback=1100.0,
    )

    assert mode == "never_exited_after_oracle_exit"


def test_root_cause_prioritizes_negative_entry_loser() -> None:
    cause = attr.root_cause(
        a_enter=-500.0,
        pnl=-200.0,
        lifecycle_mode="overheld_after_oracle_exit",
        false_hold_rows=10,
        false_exit_rows=0,
    )

    assert cause == "entry_policy_took_negative_advantage_loser"


def test_state_distribution_reports_false_hold_rate_on_actual_paths() -> None:
    frame = pd.DataFrame(
        {
            "reported_split": ["q1_2026"] * 4,
            "candidate_uid": ["a", "a", "b", "b"],
            "oracle_holding_action": ["exit", "exit", "hold", "hold"],
            "model_lifecycle_action": ["hold", "exit", "hold", "exit"],
            "current_pnl": [0.0, 10.0, 20.0, 30.0],
            "a_hold": [-5.0, -1.0, 10.0, 8.0],
            "giveback_from_mfe": [0.0, 0.0, 1.0, 2.0],
            "minutes_since_entry": [1.0, 2.0, 3.0, 4.0],
            "entry_a_enter": [-100.0, -100.0, 50.0, 50.0],
        }
    )

    rows = attr.summarize_state_distribution(frame, "actual", split_column="reported_split")

    assert rows.loc[0, "oracle_hold_fraction"] == 0.5
    assert rows.loc[0, "model_hold_fraction"] == 0.5
    assert rows.loc[0, "false_hold_rate_on_oracle_exit"] == 0.5
    assert rows.loc[0, "false_exit_rate_on_oracle_hold"] == 0.5


def test_decide_keeps_foundation_work_required_after_complete_attribution() -> None:
    trades = pd.DataFrame({"pnl": [1.0]})
    reconstruction_skips = pd.DataFrame()

    assert attr.decide(trades, reconstruction_skips) == "protocol276_failure_attribution_complete_foundation_work_required"
