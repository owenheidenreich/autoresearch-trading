from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_protocol101_loss_reversal_full_serial_replay as replay


def _flat_entries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": ["fold", "fold", "fold"],
            "reported_split": ["q1_2026", "q1_2026", "q1_2026"],
            "split": ["q1_2026", "q1_2026", "q1_2026"],
            "seed": [1, 1, 1],
            "session": ["2026-03-02", "2026-03-02", "2026-03-02"],
            "decision_time": [
                "2026-03-02T15:00:00+00:00",
                "2026-03-02T15:05:00+00:00",
                "2026-03-02T15:25:00+00:00",
            ],
            "decision_dt": pd.to_datetime(
                [
                    "2026-03-02T15:00:00+00:00",
                    "2026-03-02T15:05:00+00:00",
                    "2026-03-02T15:25:00+00:00",
                ],
                utc=True,
            ),
            "candidate_exit_time": [
                "2026-03-02T15:20:00+00:00",
                "2026-03-02T15:10:00+00:00",
                "2026-03-02T15:30:00+00:00",
            ],
            "exit_dt": pd.to_datetime(
                [
                    "2026-03-02T15:20:00+00:00",
                    "2026-03-02T15:10:00+00:00",
                    "2026-03-02T15:30:00+00:00",
                ],
                utc=True,
            ),
            "candidate_uid": ["open-1", "blocked-1", "later-1"],
            "trade_uid": ["t1", "t2", "t3"],
            "contract_id": ["SPXW-C", "SPXW-P", "SPXW-C2"],
            "right": ["C", "P", "C"],
            "offset": [0.0, 0.0, 0.0],
            "score": [1.0, 1.0, 1.0],
            "threshold": [0.0, 0.0, 0.0],
            "candidate_pnl": [-500.0, 800.0, 100.0],
            "entry_ask": [20.0, 20.0, 20.0],
            "entry_spread": [0.2, 0.2, 0.2],
            "edge": [1.0, 1.0, 1.0],
            "time_bucket": ["post_open_morning"] * 3,
        }
    )


def _triggers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "fold": ["fold"],
            "seed": [1],
            "session": ["2026-03-02"],
            "open_trade_candidate_uid": ["open-1"],
            "best_blocked_candidate_uid": ["blocked-1"],
            "best_blocked_contract_id": ["SPXW-P"],
            "best_blocked_right": ["P"],
            "best_blocked_decision_dt": ["2026-03-02T15:05:00+00:00"],
            "current_quote_time": ["2026-03-02T15:05:00+00:00"],
            "live_test_bucket": [replay.PRIORITY_1_BUCKET],
            "current_pnl_at_signal": [-100.0],
            "open_trade_pnl": [-500.0],
            "best_blocked_candidate_pnl": [800.0],
            "best_blocked_minus_open_pnl": [1300.0],
            "current_bid": [19.0],
            "current_ask": [19.4],
            "current_spread": [0.4],
            "quote_lag_seconds": [0.0],
        }
    )


def test_priority1_replay_releases_slot_and_then_takes_later_flat_entry() -> None:
    trades, events = replay.replay_all_variants(_flat_entries(), _triggers(), stress_levels=(0.0,))

    baseline = trades[trades["variant"].eq("baseline")]
    challenger = trades[trades["variant"].eq("priority1_exit_release_slot")]

    assert baseline["pnl"].sum() == -400.0
    assert list(baseline["candidate_uid"]) == ["open-1", "later-1"]
    assert challenger["pnl"].sum() == 800.0
    assert list(challenger["candidate_uid"]) == ["open-1", "blocked-1", "later-1"]
    assert challenger.iloc[0]["trade_action"] == "loss_reversal_exit_release_slot"
    assert events.iloc[0]["event_type"] == "loss_reversal_exit_release_slot"


def test_stress_applies_to_all_round_trips_and_summary_blocks_training() -> None:
    trades, _ = replay.replay_all_variants(_flat_entries(), _triggers(), stress_levels=(0.25,))
    split_summary = replay.summarize_trades(trades)
    variant_summary = replay.summarize_variants(split_summary)
    summary = replay.build_summary(trades, split_summary, variant_summary)

    baseline = variant_summary[variant_summary["variant"].eq("baseline")].iloc[0]
    challenger = variant_summary[variant_summary["variant"].eq("priority1_exit_release_slot")].iloc[0]

    assert baseline["pnl"] == -500.0
    assert challenger["pnl"] == 650.0
    assert challenger["delta_vs_baseline"] == 1150.0
    assert summary["model_training"] is False
    assert summary["challenge_allowed"] is False


def test_run_writes_report_and_doc(tmp_path) -> None:
    flat = tmp_path / "flat.csv"
    triggers = tmp_path / "triggers.csv"
    out = tmp_path / "out"
    doc = tmp_path / "doc.md"
    _flat_entries().drop(columns=["decision_dt", "exit_dt"]).to_csv(flat, index=False)
    _triggers().to_csv(triggers, index=False)

    args = SimpleNamespace(flat_entries=flat, pricing_rows=triggers, output_dir=out, doc=doc, starting_cash=10_000.0)
    summary = replay.run(args)

    assert summary["model_training"] is False
    for path in summary["outputs"].values():
        assert pd.io.common.file_exists(path)
    assert (out / "report.md").read_text() == doc.read_text()
