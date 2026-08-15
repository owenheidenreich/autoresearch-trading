from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_learned_defer_challenger_research_packet as packet


def test_reproduction_gate_fails_on_changed_replay_totals() -> None:
    summary = {
        "stress_results": [
            {
                "slippage_per_side": 0.0,
                "totals": {
                    "challenger_entries": 59,
                    "trades": 739,
                    "delta_vs_protocol101_same_scope": 66_780.0,
                },
            },
            {
                "slippage_per_side": 0.10,
                "totals": {
                    "challenger_entries": 60,
                    "trades": 739,
                    "delta_vs_protocol101_same_scope": 67_580.0,
                },
            },
            {
                "slippage_per_side": 0.25,
                "totals": {
                    "challenger_entries": 60,
                    "trades": 739,
                    "delta_vs_protocol101_same_scope": 68_780.0,
                },
            },
        ]
    }

    result = packet.check_replay_reproduction(summary)

    assert result["status"] == "fail"
    assert result["mismatches"][0]["field"] == "challenger_entries"


def test_freeze_manifest_hashes_required_artifacts(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text('{"config": {"hidden_dim": 8}}')
    estimator_summary = tmp_path / "estimator_summary.json"
    estimator_summary.write_text('{"decision": "ok", "config": {"target": "cost"}}')
    args = SimpleNamespace(model_artifacts=model_dir, slot_estimator_summary=estimator_summary)

    manifest = packet.build_freeze_manifest(
        args,
        {"artifact": artifact},
        {"status": "pass", "missing": []},
        {"stress_results": []},
        {"status": "pass", "mismatches": []},
    )

    assert manifest["files"]["artifact"]["exists"] is True
    assert len(manifest["files"]["artifact"]["sha256"]) == 64
    assert manifest["model_config"] == {"hidden_dim": 8}


def test_blocked_protocol101_attribution_uses_challenger_open_interval() -> None:
    trades = pd.DataFrame(
        {
            "slippage_per_side": [0.0],
            "split": ["q1_2026"],
            "session": ["2026-01-02"],
            "decision_time": ["2026-01-02T15:00:00+00:00"],
            "exit_time": ["2026-01-02T15:10:00+00:00"],
            "decision_dt": [pd.Timestamp("2026-01-02T15:00:00Z")],
            "exit_dt": [pd.Timestamp("2026-01-02T15:10:00Z")],
            "candidate_uid": ["challenger"],
            "source": ["challenger"],
            "pnl": [500.0],
            "trade_key": ["trade-1"],
        }
    )
    baseline = pd.DataFrame(
        {
            "split": ["q1_2026", "q1_2026", "q1_2026"],
            "session": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "decision_dt": [
                pd.Timestamp("2026-01-02T14:59:00Z"),
                pd.Timestamp("2026-01-02T15:00:00Z"),
                pd.Timestamp("2026-01-02T15:09:00Z"),
            ],
            "protocol101_action": ["enter", "enter", "enter"],
            "baseline_trade_pnl": [100.0, 200.0, 300.0],
            "surface_candidate_uid": ["before", "same", "inside"],
            "contract_id": ["a", "b", "c"],
        }
    )

    rows = packet.attribute_blocked_protocol101(trades, baseline, slippage_per_side=0.10)

    assert [row["baseline_candidate_uid"] for row in rows] == ["same", "inside"]
    assert [row["baseline_pnl_stressed"] for row in rows] == [180.0, 280.0]


def test_concentration_summary_warns_and_blocks_top5_trade_share() -> None:
    session_delta = pd.DataFrame(
        {
            "split": ["q4_2025", "q1_2026"],
            "session": ["a", "b"],
            "delta": [90.0, 10.0],
        }
    )
    challenger = pd.DataFrame({"net_contribution": [90.0, 5.0, 5.0]})

    summary = packet.concentration_summary_for_slippage(session_delta, challenger)
    status = packet.decide_concentration_status(
        {
            "stress_results": [
                {
                    "totals": {"delta_vs_protocol101_same_scope": 100.0},
                    "splits": {
                        "q4_2025": {"delta_vs_protocol101_same_scope": 90.0},
                        "q1_2026": {"delta_vs_protocol101_same_scope": 10.0},
                    },
                }
            ]
        },
        {"0.00": summary},
    )

    assert "top_day_positive_share_gt_0_35" in summary["warnings"]
    assert status == "blocked_top5_trade_concentration_gt_0_70"


def test_slot_cost_calibration_status_blocks_undercoverage() -> None:
    calibration = pd.DataFrame(
        {
            "regime": ["split", "split", "split"],
            "bucket": ["q1_2026", "q4_2025", "recent_2026"],
            "charge_coverage": [0.70, 0.95, 0.90],
            "positive_auc": [0.82, 0.91, 0.81],
        }
    )

    assert packet.decide_calibration_status(calibration) == "blocked_undercoverage_or_split_instability"


def test_packet_decision_precedence_blocks_holdout_scoring() -> None:
    assert (
        packet.decide_packet(
            "blocked_unresolved_label_policy_mismatch",
            "pass",
            "pass",
        )
        == packet.FINAL_DECISIONS["label"]
    )
    assert (
        packet.decide_packet(
            "resolved_policy_weighting_mismatch",
            "blocked_top5_trade_concentration_gt_0_70",
            "blocked_undercoverage_or_split_instability",
        )
        == packet.FINAL_DECISIONS["concentration"]
    )
