from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v4.sim.paper_replay import (
    PaperReplayConfig,
    SlippageScenario,
    apply_slippage,
    build_replay_frame,
    live_data_parity_checks,
    load_selected_trades,
    order_state_summary,
    promotion_gate_status,
)


def _selected_path(tmp_path: Path) -> Path:
    rows = [
        {
            "trade_uid": "t1",
            "canonical_entry_uid": "c1",
            "split": "q1_2026",
            "seed": 1,
            "entry_seed": 11,
            "session": "2026-03-06",
            "decision_time": "2026-03-06T15:00:00+00:00",
            "candidate_exit_time": "2026-03-06T15:02:00+00:00",
            "contract_id": "SPXW-20260306-06700.000-C",
            "right": "C",
            "offset": 0.0,
            "candidate_pnl": 100.0,
            "protocol054_pnl": 40.0,
            "delta_vs_protocol054": 60.0,
            "candidate_exit_reason": "sequence_residual_override",
            "candidate_exit_step": 2,
            "protocol054_exit_reason": "model_exit_giveback",
            "protocol054_exit_step": 18,
            "predicted_continuation_value": 12.0,
            "override_threshold": 0.0,
            "predicted_recovery_probability": 0.4,
            "predicted_decay_probability": 0.6,
            "current_pnl_at_exit": 100.0,
            "mfe_to_exit": 120.0,
            "mae_to_exit": -20.0,
            "future_max_delta_at_exit": 0.0,
            "future_min_delta_at_exit": -100.0,
        }
    ]
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(rows))
    return path


def _steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_uid": "t1",
                "step_idx": 0,
                "quote_time": "2026-03-06T15:00:00+00:00",
                "quote_ts": pd.Timestamp("2026-03-06T15:00:00+00:00"),
                "bid": 0.90,
                "ask": 1.00,
                "mid": 0.95,
                "spread": 0.10,
                "spread_frac": 0.10,
                "quote_gap_seconds": 0.0,
                "entry_ask": 1.00,
                "current_pnl": -10.0,
            },
            {
                "trade_uid": "t1",
                "step_idx": 2,
                "quote_time": "2026-03-06T15:02:00+00:00",
                "quote_ts": pd.Timestamp("2026-03-06T15:02:00+00:00"),
                "bid": 2.00,
                "ask": 2.10,
                "mid": 2.05,
                "spread": 0.10,
                "spread_frac": 0.05,
                "quote_gap_seconds": 0.0,
                "entry_ask": 1.00,
                "current_pnl": 100.0,
            },
        ]
    )


def test_build_replay_frame_uses_ask_entry_and_bid_exit(tmp_path: Path) -> None:
    selected = load_selected_trades(_selected_path(tmp_path))
    replay = build_replay_frame(selected, _steps())

    assert len(replay) == 1
    assert replay.iloc[0]["entry_fill_nbbo"] == 1.00
    assert replay.iloc[0]["exit_fill_nbbo"] == 2.00
    assert replay.iloc[0]["paper_pnl_nbbo"] == 100.0
    assert abs(replay.iloc[0]["candidate_pnl_diff_vs_path"]) < 1e-9


def test_slippage_is_adverse_on_entry_and_exit(tmp_path: Path) -> None:
    replay = build_replay_frame(load_selected_trades(_selected_path(tmp_path)), _steps())
    stressed = apply_slippage(replay, SlippageScenario("stress", 0.05, 0.05))

    assert stressed.iloc[0]["entry_fill_price"] == 1.05
    assert stressed.iloc[0]["exit_fill_price"] == 1.95
    assert round(stressed.iloc[0]["paper_pnl"], 6) == 90.0


def test_order_state_summary_reaches_exit_filled(tmp_path: Path) -> None:
    replay = build_replay_frame(load_selected_trades(_selected_path(tmp_path)), _steps())
    stressed = apply_slippage(replay, SlippageScenario("nbbo"))
    summary = order_state_summary(
        stressed,
        config=PaperReplayConfig(protocol_id="test"),
        scenario=SlippageScenario("nbbo"),
    )

    assert summary["all_exit_filled"]
    assert summary["all_one_contract"]
    assert summary["final_state_counts"] == {"exit_filled": 1}


def test_live_parity_blocks_without_shadow_feed(tmp_path: Path) -> None:
    replay = build_replay_frame(load_selected_trades(_selected_path(tmp_path)), _steps())
    checks = live_data_parity_checks(
        replay,
        {"coverage": 1.0, "sign_flip_fraction": 0.0, "abs_diff_p95": 0.0},
        config=PaperReplayConfig(),
    )

    statuses = {check["name"]: check["status"] for check in checks}
    assert statuses["selected_vs_path_pnl"] == "pass"
    assert statuses["live_shadow_feed"] == "blocker"
    assert promotion_gate_status(checks) == "blocked"
