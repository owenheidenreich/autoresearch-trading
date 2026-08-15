from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_protocol101_strategy_selection_packet as packet


def _open_rows() -> pd.DataFrame:
    rows = []
    specs = [
        ("open-loss-1", "blocked-loss-1", "2026-03-02", "C", "P", -500.0, 700.0, 1200.0, "hard_stop", 8.0),
        ("open-loss-2", "blocked-loss-2", "2026-03-03", "C", "C", -300.0, 800.0, 1100.0, "target", 6.0),
        ("open-loss-3", "blocked-loss-3", "2026-03-04", "P", "P", -100.0, 900.0, 1000.0, "target", 7.0),
        ("open-regime-1", "blocked-regime-1", "2026-03-05", "C", "P", 200.0, 900.0, 700.0, "target", 4.0),
        ("open-regime-2", "blocked-regime-2", "2026-03-06", "P", "C", 500.0, 1300.0, 800.0, "sequence_residual_override", 20.0),
        ("open-same-1", "blocked-same-1", "2026-03-07", "C", "C", 100.0, 400.0, 300.0, "protocol054_fallback", 18.0),
    ]
    for idx, (open_uid, blocked_uid, session, open_right, _blocked_right, open_pnl, blocked_pnl, slot, exit_reason, duration) in enumerate(specs):
        rows.append(
            {
                "reported_split": "q1_2026",
                "fold": "fold",
                "seed": 1,
                "session": session,
                "open_trade_candidate_uid": open_uid,
                "open_trade_contract_id": f"SPXW-{idx}-{open_right}",
                "open_trade_entry_dt": f"{session}T15:00:00+00:00",
                "open_trade_exit_dt": f"{session}T15:10:00+00:00",
                "open_trade_right": open_right,
                "open_trade_exit_reason": exit_reason,
                "best_blocked_candidate_uid": blocked_uid,
                "best_blocked_decision_dt": f"{session}T15:05:00+00:00",
                "best_blocked_candidate_pnl": blocked_pnl,
                "best_blocked_minus_open_pnl": slot,
                "open_trade_pnl": open_pnl,
                "open_time_bucket": "post_open_morning",
                "open_moneyness": "ITM",
                "open_premium_bucket": "gte_3000",
                "open_duration_minutes": duration,
                "open_score_margin": 0.5,
                "open_path_mfe": 1000.0,
                "open_path_mae": -500.0,
                "open_giveback_from_mfe": 200.0,
                "open_large_uncaptured_mfe": False,
            }
        )
    return pd.DataFrame(rows)


def _blocked_rows() -> pd.DataFrame:
    rows = []
    relations = {
        "blocked-loss-1": ("P", "opposite_side"),
        "blocked-loss-2": ("C", "same_side_different_contract"),
        "blocked-loss-3": ("P", "same_side_different_contract"),
        "blocked-regime-1": ("P", "opposite_side"),
        "blocked-regime-2": ("C", "opposite_side"),
        "blocked-same-1": ("C", "same_contract"),
    }
    open_rows = _open_rows()
    for _, open_row in open_rows.iterrows():
        blocked_uid = open_row["best_blocked_candidate_uid"]
        blocked_right, relation = relations[blocked_uid]
        rows.append(
            {
                "reported_split": open_row["reported_split"],
                "fold": open_row["fold"],
                "seed": open_row["seed"],
                "session": open_row["session"],
                "open_trade_candidate_uid": open_row["open_trade_candidate_uid"],
                "blocked_candidate_uid": blocked_uid,
                "blocked_decision_dt": open_row["best_blocked_decision_dt"],
                "blocked_right": blocked_right,
                "blocked_contract_id": f"blocked-{blocked_right}",
                "blocked_relation": relation,
                "blocked_candidate_pnl": open_row["best_blocked_candidate_pnl"],
                "blocked_entry_ask": 30.0,
                "blocked_entry_spread": 0.4,
                "blocked_time_bucket": "post_open_morning",
                "minutes_after_open_entry": 5.0,
                "minutes_before_open_exit": 5.0,
            }
        )
    return pd.DataFrame(rows)


def _thesis_tests() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "thesis": [
                "losing_open_trade_blocks_positive_later_signal",
                "opposite_side_later_signal_blocked",
                "same_side_later_signal_blocked",
                "long_duration_slot_cost",
                "fallback_exit_slot_cost",
            ],
            "mechanism": [""] * 5,
            "open_trades": [3, 3, 3, 2, 1],
            "best_blocked_minus_open_total": [3300.0, 2700.0, 2400.0, 1100.0, 300.0],
            "positive_slot_cost_rate": [1.0] * 5,
            "median_slot_cost": [1100.0, 800.0, 1000.0, 550.0, 300.0],
            "next_action": [""] * 5,
        }
    )


def test_attach_best_blocked_event_adds_relation() -> None:
    evidence = packet.attach_best_blocked_event(_open_rows(), _blocked_rows())

    first = evidence[evidence["open_trade_candidate_uid"].eq("open-loss-1")].iloc[0]
    assert first["best_blocked_relation"] == "opposite_side"
    assert first["best_blocked_right"] == "P"


def test_strategy_selection_recommends_loss_reversal_first() -> None:
    evidence = packet.attach_best_blocked_event(_open_rows(), _blocked_rows())
    selection = packet.build_strategy_selection_matrix(_thesis_tests(), evidence)

    top = selection.sort_values("rank").iloc[0]
    assert top["hypothesis_id"] == "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1"
    assert top["evidence_open_trades"] == 3
    assert top["dedup_positive_slot_cost_total"] == 3300.0


def test_examples_and_summary_block_training() -> None:
    evidence = packet.attach_best_blocked_event(_open_rows(), _blocked_rows())
    selection = packet.build_strategy_selection_matrix(_thesis_tests(), evidence)
    examples = packet.build_hypothesis_evidence_examples(evidence, per_hypothesis=2)
    overlap = packet.build_hypothesis_overlap_matrix(evidence)
    summary = packet.build_summary(selection, examples, overlap, foundational_truth_exists=True)

    assert len(examples) > 0
    assert summary["model_training"] is False
    assert summary["challenge_allowed"] is False
    assert summary["paper_default_baseline"] == "PAPER_DEFAULT_PROTOCOL101"
    assert summary["recommended_first_hypothesis"]["hypothesis_id"] == "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1"
    assert summary["important_overlap"]["loss_reversal_and_regime_flip_open_trades"] == 1


def test_run_writes_declared_outputs_and_doc_mirror(tmp_path) -> None:
    slot_dir = tmp_path / "slot"
    out_dir = tmp_path / "out"
    doc = tmp_path / "doc.md"
    truth = tmp_path / "truth.md"
    slot_dir.mkdir()
    truth.write_text("truth\n")
    _thesis_tests().to_csv(slot_dir / "slot_cost_thesis_tests.csv", index=False)
    _open_rows().to_csv(slot_dir / "enriched_open_trade_slot_summary.csv", index=False)
    _blocked_rows().to_csv(slot_dir / "enriched_blocked_slot_events.csv", index=False)

    args = SimpleNamespace(
        output_dir=out_dir,
        doc=doc,
        slot_cost_dir=slot_dir,
        foundational_truth_doc=truth,
    )
    summary = packet.run(args)

    assert summary["recommended_first_hypothesis"]["hypothesis_id"] == "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1"
    for path in summary["outputs"].values():
        assert pd.io.common.file_exists(path)
    assert (out_dir / "report.md").read_text() == doc.read_text()
    assert pd.read_csv(out_dir / "selected_strategy.csv").iloc[0]["hypothesis_id"] == "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1"
