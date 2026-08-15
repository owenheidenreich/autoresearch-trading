from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.model.protocol101_repair_artifacts import (
    REQUIRED_MANIFEST_HASHES,
    verify_replay_packet,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    RepairedCanonicalDecision,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    Protocol101LegacySyntheticExitArtifactError,
)
from v4.model.protocol101_stage1_reference_multiplicity import (
    D1_BLOCK_SIZE,
    D1_COMPLETE_BLOCKS,
    D1_INCLUDED_ROWS,
    D1_NORMAL_ROWS,
    D1_SEEDS,
    D1_TRAILING_EXCLUDED,
    D6_ROUTE,
    MAXT_ROWS,
    MAXT_SEEDS,
    MaxTConfig,
    Protocol101D1ContractError,
    Protocol101MaxTContractError,
    Protocol101ReferenceIdentityError,
    Protocol101ReferenceNonfiniteError,
    ReferenceOpportunity,
    SessionPnl,
    build_d1_target_overrides,
    build_d6_authority_receipt,
    build_maxT_grid,
    d1_target_arrays,
    evaluate_maxT,
    fixed_heuristic_candidates,
    freeze_maxT_schedule,
    fwer_p_values,
    generate_matched_random_schedule,
    generate_maxT_schedule,
    load_frozen_maxT_schedule,
    matched_random_draw_candidates,
    replay_reference_v5,
    run_fixed_heuristic_reference,
    run_matched_random_reference,
    write_immutable_reference_packet,
)


def _decision(
    row: int,
    *,
    session: str = "2025-01-02",
    gross_labels: tuple[float, ...] = (-20.0, 3.0, 103.0),
    spacing_minutes: int = 1,
) -> RepairedCanonicalDecision:
    decision_time = pd.Timestamp(
        f"{session} 15:00:00",
        tz="UTC",
    ) + pd.Timedelta(minutes=row * spacing_minutes)
    labels = np.asarray(gross_labels, dtype=float)
    count = len(labels)
    asks = np.full(count, 2.0, dtype=float)
    offsets = np.linspace(-5.0, 5.0, count)
    rights = np.asarray(
        (["P", "C", "C"] if count == 3 else ["C"] * count),
        dtype=object,
    )
    source = int(decision_time.value + 60 * 1_000_000_000)
    realized = int(decision_time.value + 2 * 60 * 1_000_000_000)
    base = CanonicalDecision(
        session=session,
        decision_time=decision_time,
        features=np.asarray(
            [
                [float(row), float(candidate), *([1.0] * 12)]
                for candidate in range(count)
            ],
            dtype=float,
        ),
        labels=labels,
        mid_labels=labels + 7.0,
        entry_asks=asks,
        offsets=offsets,
        rights=rights,
        contract_ids=np.asarray(
            [f"{session}-{row:03d}-{index}" for index in range(count)],
            dtype=object,
        ),
        strike_indices=np.arange(count, dtype=int) + 10,
        right_indices=np.asarray(
            [1 if value == "P" else 0 for value in rights],
            dtype=int,
        ),
    )
    return RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=np.full(count, realized, dtype=np.int64),
        source_exit_quote_time_ns=np.full(count, source, dtype=np.int64),
        exit_quote_age_ms=np.full(count, 60_000.0, dtype=float),
        exit_reason_codes=np.full(count, 3, dtype=np.uint8),
        executable_exit_bids=asks + labels / 100.0,
        policy_deadline_ns=np.full(count, realized, dtype=np.int64),
        invalid_reason_codes=np.zeros(count, dtype=np.uint8),
        canonical_strike_slots=np.arange(count, dtype=np.int64) + 10,
        source_quote_time_ns=np.full(
            count,
            int(decision_time.value),
            dtype=np.int64,
        ),
        source_context_time_ns=np.full(
            count,
            int(decision_time.value - 60 * 1_000_000_000),
            dtype=np.int64,
        ),
    )


def _opportunities(
    *,
    policy: int,
    gross_labels: tuple[float, ...] = (-20.0, 3.0, 103.0),
    count: int = 3,
) -> list[ReferenceOpportunity]:
    return [
        ReferenceOpportunity(
            campaign_id="synthetic-reference-v5",
            fold="fold_1",
            split="validation",
            policy_index=policy,
            repaired=_decision(
                index,
                gross_labels=gross_labels,
                spacing_minutes=2,
            ),
            vwap_side="C" if index % 2 == 0 else "P",
            decision_ordinal=index,
        )
        for index in range(count)
    ]


def _maxT_series(
    *,
    sessions_per_fold: int = 6,
) -> dict[str, dict[int, list[SessionPnl]]]:
    result: dict[str, dict[int, list[SessionPnl]]] = {}
    for row_index, row in enumerate(MAXT_ROWS):
        result[row] = {}
        for seed_index, seed in enumerate(MAXT_SEEDS):
            records: list[SessionPnl] = []
            session_index = 0
            for fold_index in range(5):
                for within_fold in range(sessions_per_fold):
                    session = (
                        pd.Timestamp("2025-01-02")
                        + pd.Timedelta(days=session_index)
                    ).strftime("%Y-%m-%d")
                    oscillation = (
                        (
                            within_fold
                            + row_index
                            + 2 * seed_index
                        )
                        % sessions_per_fold
                    ) - (sessions_per_fold - 1) / 2.0
                    records.append(
                        SessionPnl(
                            session=session,
                            fold=f"fold_{fold_index + 1}",
                            pnl=float(
                                oscillation
                                + 0.05 * row_index
                                + 0.1 * seed_index
                            ),
                        )
                    )
                    session_index += 1
            result[row][seed] = records
    return result


def test_matched_random_schedule_is_reproducible_complete_and_v5_only() -> None:
    opportunities = _opportunities(policy=0)
    first = generate_matched_random_schedule(
        opportunities,
        policy_index=0,
        draws=20,
    )
    second = generate_matched_random_schedule(
        list(reversed(opportunities)),
        policy_index=0,
        draws=20,
    )
    assert first == second
    result = run_matched_random_reference(opportunities, first)
    assert result["opportunity_grid_complete"] is True
    assert result["schedule_hash"] == first.schedule_hash
    assert len(result["draw_results"]) == 20
    assert {
        item["metrics"]["simulator_version"]
        for item in result["draw_results"]
    } == {PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION}


def test_random_and_heuristic_routes_reject_v4_poisoning() -> None:
    random_opportunities = _opportunities(policy=0, count=1)
    schedule = generate_matched_random_schedule(
        random_opportunities,
        policy_index=0,
        draws=1,
    )
    random_candidate = matched_random_draw_candidates(
        random_opportunities,
        schedule,
        draw_index=0,
    )[0]
    heuristic_candidate = fixed_heuristic_candidates(
        _opportunities(policy=5, count=1)
    )[0]
    for candidate in (random_candidate, heuristic_candidate):
        poisoned = replace(
            candidate,
            source_simulator_version=(
                PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION
            ),
        )
        with pytest.raises(Protocol101LegacySyntheticExitArtifactError):
            replay_reference_v5([poisoned])


def test_two_clock_same_timestamp_release_and_fee_once() -> None:
    opportunities = _opportunities(
        policy=5,
        gross_labels=(103.0, 103.0, 103.0),
        count=2,
    )
    candidates = fixed_heuristic_candidates(opportunities)
    trades, state = replay_reference_v5(candidates)
    assert len(trades) == 2
    assert state.skipped["overlap"] == 0
    assert trades[0].label_realized_exit_time_ns == (
        trades[1].decision_time_ns
    )
    assert [item.raw_label_pnl_after_campaign_fee for item in trades] == [
        pytest.approx(100.0),
        pytest.approx(100.0),
    ]
    assert state.cash_by_account["validation"] == pytest.approx(10_200.0)


def test_reference_preflight_rejects_duplicates_and_nonfinite_inputs() -> None:
    opportunities = _opportunities(policy=0, count=1)
    with pytest.raises(Protocol101ReferenceIdentityError):
        generate_matched_random_schedule(
            [opportunities[0], opportunities[0]],
            policy_index=0,
        )
    repaired = opportunities[0].repaired
    bad_base = replace(
        repaired.base,
        labels=np.asarray([float("nan")] * len(repaired.base.labels)),
    )
    poisoned = replace(
        opportunities[0],
        repaired=replace(repaired, base=bad_base),
    )
    with pytest.raises(Protocol101ReferenceNonfiniteError):
        generate_matched_random_schedule([poisoned], policy_index=0)


def test_null_canaries_respond_to_positive_no_edge_and_negative_edge() -> None:
    observed: dict[str, float] = {}
    for name, gross in (
        ("positive", (103.0, 103.0, 103.0)),
        ("no_edge", (3.0, 3.0, 3.0)),
        ("negative", (-97.0, -97.0, -97.0)),
    ):
        opportunities = _opportunities(
            policy=0,
            gross_labels=gross,
            count=3,
        )
        schedule = generate_matched_random_schedule(
            opportunities,
            policy_index=0,
            draws=8,
        )
        observed[name] = run_matched_random_reference(
            opportunities,
            schedule,
        )["pooled_net_pnl_distribution"]["mean"]
    assert observed["positive"] > 0.0
    assert observed["no_edge"] == pytest.approx(0.0)
    assert observed["negative"] < 0.0


def test_d5_identities_hashes_order_sensitivity_and_immutable_packet(
    tmp_path: Path,
) -> None:
    opportunities = _opportunities(policy=5)
    first = run_fixed_heuristic_reference(opportunities)
    second = run_fixed_heuristic_reference(list(reversed(opportunities)))
    assert first["ordered_trade_identities"] == (
        second["ordered_trade_identities"]
    )
    assert first["continuous_pooled_metrics"]["candidate_stream_hash"] == (
        second["continuous_pooled_metrics"]["candidate_stream_hash"]
    )
    mutated_repaired = opportunities[0].repaired
    mutated_base = replace(
        mutated_repaired.base,
        contract_ids=np.asarray(
            [
                mutated_repaired.base.contract_ids[0],
                "changed-selected-contract",
                *mutated_repaired.base.contract_ids[2:],
            ],
            dtype=object,
        ),
    )
    mutated = [
        replace(
            opportunities[0],
            repaired=replace(mutated_repaired, base=mutated_base),
        ),
        *opportunities[1:],
    ]
    changed = run_fixed_heuristic_reference(mutated)
    assert first["continuous_pooled_metrics"]["candidate_stream_hash"] != (
        changed["continuous_pooled_metrics"]["candidate_stream_hash"]
    )
    provenance = {
        name: "a" * 64
        for name in REQUIRED_MANIFEST_HASHES
        if name
        not in {
            "simulator_config_hash",
            "candidate_stream_hash",
            "candidate_payload_hash",
            "trade_identity_hash",
            "exit_quote_age_report_hash",
        }
    }
    packet = (
        tmp_path
        / "protocol101_full_trader_stage1_reference_v5_d5_synthetic"
    )
    write_result = write_immutable_reference_packet(
        packet,
        candidates=first["candidates"],
        trades=first["trades"],
        state=first["state"],
        metrics=first,
        provenance_hashes=provenance,
        attempt_id=packet.name,
    )
    assert write_result["write_order"][-1] == "manifest.json"
    assert verify_replay_packet(packet)["manifest_written_last"] is True


def test_d1_exact_geometry_moves_only_paired_target_arrays() -> None:
    decisions = [_decision(index) for index in range(D1_NORMAL_ROWS)]
    included, overrides, receipt = build_d1_target_overrides(
        decisions,
        role="fit",
        seed=D1_SEEDS[0],
    )
    assert len(included) == len(overrides) == D1_INCLUDED_ROWS
    assert D1_INCLUDED_ROWS == D1_BLOCK_SIZE * D1_COMPLETE_BLOCKS
    assert D1_NORMAL_ROWS - D1_INCLUDED_ROWS == D1_TRAILING_EXCLUDED
    assert receipt["source_rows"] != list(range(D1_INCLUDED_ROWS))
    net, mid = d1_target_arrays(included, overrides)
    for index, override in enumerate(overrides):
        source = decisions[override.source_row_index]
        assert np.array_equal(net[index], source.base.labels)
        assert np.array_equal(mid[index], source.base.mid_labels)
        assert np.array_equal(
            included[index].base.features,
            decisions[index].base.features,
        )
        assert included[index].base.contract_ids.tolist() == (
            decisions[index].base.contract_ids.tolist()
        )
        assert np.array_equal(
            included[index].realized_exit_time_ns,
            decisions[index].realized_exit_time_ns,
        )
        assert override.non_candidate is True
        assert override.replay_permitted is False
    validation, validation_overrides, _ = build_d1_target_overrides(
        decisions,
        role="validation",
    )
    assert all(
        item.source_row_index == index
        for index, item in enumerate(validation_overrides)
    )
    assert len(validation) == 330


def test_d1_fails_closed_on_geometry_or_seed_change() -> None:
    decisions = [_decision(index) for index in range(D1_NORMAL_ROWS)]
    with pytest.raises(Protocol101D1ContractError):
        build_d1_target_overrides(
            decisions[:-1],
            role="fit",
            seed=D1_SEEDS[0],
        )
    with pytest.raises(Protocol101D1ContractError):
        build_d1_target_overrides(
            decisions,
            role="fit",
            seed=8599,
        )
    with pytest.raises(Protocol101D1ContractError):
        build_d1_target_overrides(
            decisions,
            role="fit",
            seed=D1_SEEDS[0],
            policy_index=4,
        )


def test_d6_receipt_binds_exact_authority_and_forbids_alpha() -> None:
    receipt = build_d6_authority_receipt(
        Path(__file__).resolve().parents[2]
    )
    assert receipt["route"] == D6_ROUTE
    assert receipt["offline_17_feature_rebuild_allowed"] is True
    assert (
        receipt["repaired_exit_label_audit_metadata_as_alpha_forbidden"]
        is True
    )
    assert receipt["exact_candidate_transfer"] == (
        "deferred_to_mandatory_no_order_shadow"
    )
    assert receipt["sealed_evidence_access"] == "forbidden"
    assert receipt["broker_paper_authority"] == "none"
    assert len(receipt["ordered_17_feature_firewall"]) == 17


def test_maxT_accepts_exact_28_by_3_grid_and_schedule_reproduces(
    tmp_path: Path,
) -> None:
    grid = build_maxT_grid(_maxT_series())
    config = MaxTConfig(replicates=32)
    first = generate_maxT_schedule(grid, config=config)
    second = generate_maxT_schedule(grid, config=config)
    assert np.array_equal(first, second)
    frozen = freeze_maxT_schedule(
        tmp_path / "indices.npy",
        tmp_path / "manifest.json",
        grid,
        config=config,
    )
    loaded = load_frozen_maxT_schedule(
        frozen.schedule_path,
        frozen.manifest_path,
        grid,
        config=config,
    )
    assert np.array_equal(frozen.indices, loaded.indices)
    assert frozen.schedule_sha256 == loaded.schedule_sha256


def test_maxT_grid_fail_closed_cases() -> None:
    missing_row = _maxT_series()
    missing_row.pop(MAXT_ROWS[-1])
    with pytest.raises(Protocol101MaxTContractError):
        build_maxT_grid(missing_row)
    missing_seed = _maxT_series()
    missing_seed[MAXT_ROWS[0]].pop(44)
    with pytest.raises(Protocol101MaxTContractError):
        build_maxT_grid(missing_seed)
    changed_grid = _maxT_series()
    changed_grid[MAXT_ROWS[0]][42][0] = replace(
        changed_grid[MAXT_ROWS[0]][42][0],
        session="2099-01-01",
    )
    with pytest.raises(Protocol101MaxTContractError):
        build_maxT_grid(changed_grid)
    nonfinite = _maxT_series()
    nonfinite[MAXT_ROWS[0]][42][0] = replace(
        nonfinite[MAXT_ROWS[0]][42][0],
        pnl=float("nan"),
    )
    with pytest.raises(Protocol101MaxTContractError):
        build_maxT_grid(nonfinite)


def test_maxT_requires_frozen_untampered_schedule_and_nonzero_variance(
    tmp_path: Path,
) -> None:
    grid = build_maxT_grid(_maxT_series())
    config = MaxTConfig(replicates=32)
    with pytest.raises(Protocol101MaxTContractError):
        load_frozen_maxT_schedule(
            tmp_path / "missing.npy",
            tmp_path / "missing.json",
            grid,
            config=config,
        )
    frozen = freeze_maxT_schedule(
        tmp_path / "indices.npy",
        tmp_path / "manifest.json",
        grid,
        config=config,
    )
    frozen.schedule_path.write_bytes(b"tampered")
    with pytest.raises(Protocol101MaxTContractError):
        evaluate_maxT(grid, frozen)
    constant = _maxT_series()
    constant[MAXT_ROWS[0]][42] = [
        replace(item, pnl=1.0) for item in constant[MAXT_ROWS[0]][42]
    ]
    constant_grid = build_maxT_grid(constant)
    constant_frozen = freeze_maxT_schedule(
        tmp_path / "constant.npy",
        tmp_path / "constant.json",
        constant_grid,
        config=config,
    )
    with pytest.raises(Protocol101MaxTContractError):
        evaluate_maxT(constant_grid, constant_frozen)


def test_maxT_ties_count_greater_than_or_equal_exactly() -> None:
    observed = np.arange(28, dtype=float)
    maxima = np.asarray([0.0, 1.0, 1.0, 100.0], dtype=float)
    counts, p_values = fwer_p_values(observed, maxima)
    assert counts[1] == 3
    assert p_values[1] == pytest.approx(4.0 / 5.0)
    assert counts[2] == 1
    assert p_values[2] == pytest.approx(2.0 / 5.0)


def test_deterministic_20000_replicate_synthetic_integration(
    tmp_path: Path,
) -> None:
    grid = build_maxT_grid(_maxT_series())
    config = MaxTConfig()
    frozen = freeze_maxT_schedule(
        tmp_path / "indices_20000.npy",
        tmp_path / "manifest_20000.json",
        grid,
        config=config,
    )
    first = evaluate_maxT(grid, frozen)
    loaded = load_frozen_maxT_schedule(
        frozen.schedule_path,
        frozen.manifest_path,
        grid,
        config=config,
    )
    second = evaluate_maxT(grid, loaded)
    assert first.exceedance_limit == 999
    assert first.max_null_sha256 == second.max_null_sha256
    assert np.array_equal(first.exceedance_counts, second.exceedance_counts)
    assert np.array_equal(first.p_fwer, second.p_fwer)
