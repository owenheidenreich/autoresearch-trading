from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from v4.research import pathd_corrected_v32_foundation as foundation
from v4.research import pathd_entry_execution_v32 as execution
from v4.research import pathd_entry_exit as science
from v4.scripts import run_pathd_entry_exit_research as runner


def _sessions(count: int, prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}-{index:03d}" for index in range(count))


def _opportunity(
    *, session: str, decision: int, suffix: str, hgb: bool, neural: bool,
    p5_rank: int | None = 0, money: str = "ATM",
) -> execution.CausalEntryOpportunityV1:
    return execution.seal_opportunity(
        outer_fold=1,
        session=session,
        decision_time_ns=decision,
        source_neutral_contract_id=f"SPXW-{session}-{suffix}",
        right_code="C" if decision % 2 == 0 else "P",
        moneyness_class=money,
        premium_band="small_1_3",
        entry_ask_micros=1_000_000,
        causal_bid_path=(
            (0, 900_000),
            (60, 1_500_000),
            (300, 2_000_000),
            (900, 500_000),
        ),
        hgb_enter=hgb,
        neural_enter=neural,
        p5_rank=p5_rank,
        nearest_atm_rank=0 if money == "ATM" else None,
        signed_score_micros=100 if hgb else -100,
    )


def _fit_rows(sessions: tuple[str, ...]) -> tuple[execution.CausalEntryOpportunityV1, ...]:
    return tuple(
        _opportunity(
            session=session,
            decision=index * 2_000_000_000,
            suffix=f"FIT{index}",
            hgb=index % 2 == 0,
            neural=index % 3 == 0,
        )
        for index, session in enumerate(sessions)
    )


def _outer_rows(sessions: tuple[str, ...]) -> tuple[execution.CausalEntryOpportunityV1, ...]:
    rows = []
    for session_index, session in enumerate(sessions):
        for choice in range(3):
            rows.append(
                _opportunity(
                    session=session,
                    decision=(session_index * 10 + choice) * 2_000_000_000,
                    suffix=f"OUT{session_index}-{choice}",
                    hgb=choice == 0,
                    neural=choice == 1,
                    p5_rank=choice,
                    money=("ATM", "NEAR", "WING")[choice],
                )
            )
    return tuple(rows)


def test_v32_topology_is_science_identical_and_fail_closed() -> None:
    payload, sessions, lineage = foundation.corrected_v32_payload()
    v31 = science.read_json(foundation.V31_ROOT / "preregistration.json")
    assert foundation._science_projection(payload) == foundation._science_projection(v31)
    assert science.stable_hash(sessions) == v31["sessions_hash"]
    assert science.stable_hash(lineage) == v31["feature_lineage_hash"]
    assert payload["supersedes"]["preregistration_sha256"] == (
        foundation.V31_PREREGISTRATION_SHA256
    )
    policy = payload["source_hash_policy"]["executable_generation_enforcement"]
    assert "v4/research/pathd_evidence_gate.py" in policy["source_paths"]
    assert "v4/research/pathd_entry_execution_v32.py" in policy["source_paths"]
    assert "before lock creation" in policy["br_before_state_rule"]
    pause = payload["foundation_correction"]["prefit_pause"]
    assert pause["claude_verification_pending"] is True
    assert pause["model_fit_authorized"] is False
    assert pause["corpus_decode_authorized"] is False
    assert pause["nested_or_outer_evidence_open_authorized"] is False


def test_shared_exit_is_selected_from_reconstructed_p5_journals() -> None:
    sessions = _sessions(25, "FIT")
    rows = _fit_rows(sessions)
    selection = execution.select_shared_control_exit(
        outer_fold=1,
        model_fit_sessions=sessions,
        opportunities=rows,
        legacy_causal_inputs_complete=True,
    )
    assert selection["legacy_completed_trade_count"] == 25
    assert selection["legacy_eligible"] is False
    assert "LEGACY_P5_LIFECYCLE" not in selection["candidate_policy_ids"]
    assert len(selection["terminal_journal_sha256s"]) == 23
    assert selection["selected_policy_id"] in selection["candidate_policy_ids"]

    journal = execution.replay_policy(
        outer_fold=1,
        sessions=sessions,
        opportunities=rows,
        policy_id="P5",
        exit_policy_id=selection["selected_policy_id"],
    )
    forged_trade = dict(journal.trades[0])
    forged_trade["net_pnl_micros"] += 1
    forged = replace(journal, trades=(forged_trade, *journal.trades[1:]))
    with pytest.raises(ValueError, match="journal drift|PnL reconstruction"):
        execution.validate_replay_journal(forged, opportunities=rows)


def test_complete_common_replay_inventory_has_no_nullable_cells() -> None:
    fit_sessions = _sessions(25, "FIT")
    outer_sessions = _sessions(3, "OUT")
    inventory = runner.produce_corrected_v32_outer_replay_inventory(
        outer_fold=1,
        model_fit_sessions=fit_sessions,
        outer_sessions=outer_sessions,
        model_fit_opportunities=_fit_rows(fit_sessions),
        outer_opportunities=_outer_rows(outer_sessions),
        legacy_causal_inputs_complete=False,
    )
    assert inventory["journal_count"] == 4 + 22 + 8
    assert len(inventory["positive_journal_sha256s"]) == 4
    assert len(inventory["negative_control_journal_sha256s"]) == 22
    assert len(inventory["matched_random_journal_sha256s"]) == 8
    assert len(set(inventory["matched_random_journal_sha256s"])) == 8
    assert inventory["hardcoded_pass_flags"] is False
    assert inventory["all_replays_share_exit_policy_id"] == (
        inventory["shared_control_exit"]["selected_policy_id"]
    )


def test_sealed_input_orchestrator_rejects_caller_authored_economics() -> None:
    fit_sessions = _sessions(25, "FIT")
    outer_sessions = _sessions(2, "OUT")
    semantic = {
        "schema_version": "pathd.corrected_v32.outer_replay_input.v1",
        "outer_fold": 1,
        "model_fit_sessions": fit_sessions,
        "outer_sessions": outer_sessions,
        "model_fit_opportunities": tuple(asdict(row) for row in _fit_rows(fit_sessions)),
        "outer_opportunities": tuple(asdict(row) for row in _outer_rows(outer_sessions)),
        "legacy_causal_inputs_complete": False,
        "source_dataset_sha256s": ("a" * 64, "b" * 64),
    }
    payload = {**semantic, "input_sha256": science.stable_hash(semantic)}
    result = execution.produce_outer_replay_inventory_from_sealed_input(payload)
    assert result["journal_count"] == 34
    forged = dict(payload)
    forged["caller_pnl_micros"] = 999_999_999
    with pytest.raises(ValueError, match="input drift"):
        execution.produce_outer_replay_inventory_from_sealed_input(forged)


def test_spx_reference_writer_is_causal_and_lossless() -> None:
    anchor = 2_000_000_000_000
    def row(event: int, available: int, index: int, close: float) -> dict[str, object]:
        semantic = {
            "event_time_ns": event,
            "available_at_ns": available,
            "close_float64_hex": close.hex(),
            "publisher_id": None,
            "instrument_id": None,
            "row_group": 0,
            "row_index": index,
        }
        return {**semantic, "canonical_row_sha256": science.stable_hash(semantic)}

    rows = (
        row(anchor - 60_000_000_000, anchor, 1, 6000.0),
        row(anchor - 960_000_000_000, anchor - 900_000_000_000, 2, 5990.0),
        row(anchor + 1, anchor + 1, 3, 7000.0),
    )
    result = execution.build_spx_reference_transaction(
        anchors=(("2026-08-01", anchor),), rows=rows,
        source_receipt_sha256s=("a" * 64,),
    )
    selection = result["selections"][0]
    assert selection["status"] == "FRESH"
    assert selection["current_locator"]["close_float64_hex"] == (6000.0).hex()
    assert selection["lag15_locator"]["close_float64_hex"] == (5990.0).hex()
    assert selection["current_locator"]["canonical_row_sha256"] == rows[0]["canonical_row_sha256"]
    assert result["source_receipts_root_sha256"] == science.stable_hash(["a" * 64])


def test_direct_legacy_evidence_api_cannot_bypass_br_gate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from v4.research import pathd_evidence_gate as gate

    root = tmp_path / "audit"
    monkeypatch.setattr(gate, "ENTRY_FOLD_ARTIFACT_ROOT", root / "folds")
    monkeypatch.setattr(gate, "_assert_scope_order", lambda _paths: None)
    monkeypatch.setattr(gate, "_assert_no_later_artifacts", lambda _paths: None)
    monkeypatch.setattr(gate, "_assert_unopened", lambda _paths: None)
    paths = gate._scope_paths(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    with pytest.raises(gate.EntryEvidenceGateError, match="cannot open fixed artifact"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )
    assert not paths.lock.exists()
    assert not paths.access.exists()
    assert gate._ACTIVE_CAPABILITIES == {}


def test_build_creates_no_execution_evidence_or_holdout_namespace() -> None:
    assert not (foundation.V32_ROOT / "entry_outer_folds").exists()
    assert not (foundation.V32_ROOT / "execution_inputs").exists()
    assert not science.ENTRY_FOLD_ARTIFACT_ROOT.exists()
    assert not science.PROTECTED_HOLDOUT_ROOT.exists()
    assert not runner.V32_CLAUDE_RELEASE_PATH.exists()
