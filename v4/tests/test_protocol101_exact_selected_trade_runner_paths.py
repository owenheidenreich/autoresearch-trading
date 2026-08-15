from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol101_exact_selected_trade_runner_paths as exact_paths


def test_attach_source_trade_fields_adds_runner_state_and_source_fields() -> None:
    rows = pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "seed": [1],
            "session": ["2026-01-02"],
            "decision_time": ["2026-01-02T15:00:00+00:00"],
            "contract_id": ["SPXW-test-C"],
            "right": ["C"],
            "frozen_pnl": [100.0],
            "oracle_minus_frozen_pnl": [900.0],
            "post_frozen_best_minus_frozen_pnl": [900.0],
            "forced_flat_minus_frozen_pnl": [-500.0],
            "material_continuation_after_frozen_exit": [True],
        }
    )
    trades = pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "seed": [1],
            "session": ["2026-01-02"],
            "decision_ts": [pd.Timestamp("2026-01-02T15:00:00+00:00")],
            "contract_id": ["SPXW-test-C"],
            "right": ["C"],
            "candidate_uid": ["candidate-1"],
            "score": [1.0],
            "threshold": [0.5],
            "offset": [-20.0],
            "entry_bid": [24.8],
            "entry_ask": [25.0],
            "pnl": [100.0],
            "path_mfe": [500.0],
            "path_mae": [-100.0],
            "exit_reason": ["sequence_residual_override"],
        }
    )

    attached = exact_paths.attach_source_trade_fields(rows, trades)

    assert attached.iloc[0]["candidate_uid"] == "candidate-1"
    assert attached.iloc[0]["runner_state"] == "giveback_guard_required"
    assert bool(attached.iloc[0]["giveback_guard_needed"]) is True


def test_runner_state_summary_quantifies_naive_forced_flat_risk() -> None:
    frame = pd.DataFrame(
        {
            "runner_state": ["runner_extension_candidate", "giveback_guard_required"],
            "frozen_pnl": [100.0, 200.0],
            "oracle_minus_frozen_pnl": [500.0, 700.0],
            "post_frozen_best_minus_frozen_pnl": [500.0, 700.0],
            "forced_flat_minus_frozen_pnl": [300.0, -600.0],
            "material_continuation_after_frozen_exit": [True, True],
        }
    )

    summary = exact_paths.summarize_runner_states(frame)

    forced = dict(zip(summary["runner_state"], summary["forced_flat_delta"]))
    assert forced["runner_extension_candidate"] == 300.0
    assert forced["giveback_guard_required"] == -600.0


def test_summary_blocks_challenge_and_records_coverage() -> None:
    trades = pd.DataFrame({"x": [1, 2, 3]})
    paths = pd.DataFrame(
        {
            "frozen_pnl": [100.0, 200.0],
            "post_frozen_best_minus_frozen_pnl": [300.0, 400.0],
            "forced_flat_minus_frozen_pnl": [50.0, -100.0],
            "material_continuation_after_frozen_exit": [True, False],
            "forced_flat_positive": [True, False],
        }
    )

    summary = exact_paths.build_summary(trades, paths, skips=[{"skip_reason": "missing"}])

    assert summary["coverage"]["selected_trades"] == 3
    assert summary["coverage"]["exact_path_rows"] == 2
    assert summary["coverage"]["exact_path_coverage"] == 2 / 3
    assert summary["challenge_allowed"] is False
    assert summary["model_training"] is False
