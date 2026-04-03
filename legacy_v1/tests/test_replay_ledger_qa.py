from __future__ import annotations

from training import replay


def _sample_trade() -> dict[str, object]:
    return {
        "num": 1,
        "trade_id": "2026-03-17-T0001",
        "date": "2026-03-17",
        "result": "WIN",
        "entry_time": "10:01",
        "exit_time": "10:03",
        "entry_timestamp_ms": 1773756060000,
        "exit_timestamp_ms": 1773756180000,
        "bars_held": 2,
        "hold_min": 2,
        "reason": "MODEL_EXIT",
        "direction": "BUY_CALL_ATM",
        "strike": 5000.0,
        "entry_option_px": 10.0,
        "exit_option_px": 11.0,
        "actual_px": True,
        "pnl_pct": 9.0,  # 10% move - 1% spread costs (50 bps per side)
        "cum_pnl_pct": 9.0,
        "entry_reason_codes": ["trade_signal", "entry_executed"],
        "exit_reason_codes": ["model_exit"],
        "entry_gate_prob": 0.71,
        "exit_gate_notrade_prob": 0.64,
        "entry_confidence": 0.45,
        "risk_mode": "live_like",
        "min_trade_prob": 0.55,
        "entry_stop_price": 7.5,
        "entry_take_profit_price": 13.0,
        "exit_stop_price": 10.0,
        "exit_take_profit_price": 14.0,
        "risk_updates": 1,
        "spx_entry": 5000.0,
        "spx_exit": 5004.0,
        "spx_move": 4.0,
        "spx_move_pct": 0.08,
        "mfe_points": 6.0,
        "mae_points": 2.0,
    }


def _sample_bars() -> list[dict[str, object]]:
    return [
        {
            "time": "10:01",
            "timestamp_raw": "2026-03-17 10:01",
            "timestamp_ms": 1773756060000,
            "global_idx": 1001,
            "bar_of_day": 31,
            "spx": 5000.0,
            "volume": 10000.0,
            "gate_trade_prob": 0.71,
            "gate_notrade_prob": 0.29,
            "top_direction": "C_ATM",
            "top_dir_prob": 0.61,
            "dir_probs": {"C_ATM": 0.61, "C_OTM5": 0.1, "C_OTM10": 0.1, "P_ATM": 0.1, "P_OTM5": 0.05, "P_OTM10": 0.04},
            "action": "BUY_CALL_ATM",
            "executed_action": "BUY_CALL_ATM",
            "position": "FLAT",
            "policy_gate_reason_codes": ["trade_signal", "entry_executed"],
            "policy_gate_payload": {},
        },
        {
            "time": "10:02",
            "timestamp_raw": "2026-03-17 10:02",
            "timestamp_ms": 1773756120000,
            "global_idx": 1002,
            "bar_of_day": 32,
            "spx": 5002.0,
            "volume": 11000.0,
            "gate_trade_prob": 0.44,
            "gate_notrade_prob": 0.56,
            "top_direction": "C_ATM",
            "top_dir_prob": 0.50,
            "dir_probs": {"C_ATM": 0.50, "C_OTM5": 0.2, "C_OTM10": 0.1, "P_ATM": 0.1, "P_OTM5": 0.05, "P_OTM10": 0.05},
            "action": "EXIT",
            "executed_action": "EXIT",
            "position": "IN_TRADE",
            "policy_gate_reason_codes": ["gate_no_trade_exit", "model_exit_signal"],
            "policy_gate_payload": {},
        },
    ]


def test_replay_ledger_tables_and_qa_pass() -> None:
    trades_df, bars_df, days_df = replay._build_ledger_tables(
        replay_date="2026-03-17",
        replay_run_id="abc123",
        trades=[_sample_trade()],
        bar_log=_sample_bars(),
        session_stats={"total_bars": 2, "avg_gate_prob": 0.575, "max_gate_prob": 0.71},
        model_path="training/best_model.pt",
        model_score=1.23,
        risk_mode="live_like",
        min_trade_prob=0.55,
    )
    assert list(trades_df.columns) == replay.LEDGER_TRADE_COLUMNS
    assert list(bars_df.columns) == replay.LEDGER_BAR_COLUMNS
    assert list(days_df.columns) == replay.LEDGER_DAY_COLUMNS
    qa = replay._run_replay_qa(
        trades_df=trades_df,
        bars_df=bars_df,
        days_df=days_df,
        strategy_params={"option_spread_bps": 50, "stop_loss_pct": 0.30},
    )
    assert qa["passed"] is True
    assert qa["critical_count"] == 0


def test_replay_qa_flags_bad_timestamp_format() -> None:
    trades_df, bars_df, days_df = replay._build_ledger_tables(
        replay_date="2026-03-17",
        replay_run_id="abc123",
        trades=[_sample_trade()],
        bar_log=_sample_bars(),
        session_stats={"total_bars": 2, "avg_gate_prob": 0.575, "max_gate_prob": 0.71},
        model_path="training/best_model.pt",
        model_score=1.23,
        risk_mode="live_like",
        min_trade_prob=0.55,
    )
    trades_df.loc[0, "entry_time"] = "bad"
    bars_df.loc[0, "time"] = "bad"
    qa = replay._run_replay_qa(
        trades_df=trades_df,
        bars_df=bars_df,
        days_df=days_df,
        strategy_params={"option_spread_bps": 50, "stop_loss_pct": 0.30},
    )
    assert qa["passed"] is False
    codes = {a["code"] for a in qa["anomalies"]}
    assert "bad_trade_time_format" in codes
    assert "bad_bar_time_format" in codes


def test_replay_qa_flags_non_monotonic_bar_timestamps() -> None:
    trades_df, bars_df, days_df = replay._build_ledger_tables(
        replay_date="2026-03-17",
        replay_run_id="abc123",
        trades=[_sample_trade()],
        bar_log=_sample_bars(),
        session_stats={"total_bars": 2, "avg_gate_prob": 0.575, "max_gate_prob": 0.71},
        model_path="training/best_model.pt",
        model_score=1.23,
        risk_mode="live_like",
        min_trade_prob=0.55,
    )
    bars_df.loc[1, "timestamp_ms"] = bars_df.loc[0, "timestamp_ms"] - 60_000
    qa = replay._run_replay_qa(
        trades_df=trades_df,
        bars_df=bars_df,
        days_df=days_df,
        strategy_params={"option_spread_bps": 50, "stop_loss_pct": 0.30},
    )
    assert qa["passed"] is False
    codes = {a["code"] for a in qa["anomalies"]}
    assert "bar_timestamp_non_monotonic" in codes
