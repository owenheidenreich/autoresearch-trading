from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.model.protocol101_policy_neutral_selector import (
    add_primary_utility,
    block_schedule,
    candidate_order,
    deterministic_group_sample,
    group_balanced_weights,
    midrank_percentiles,
    path_target_record,
    permute_targets_within_decision,
    select_position,
    simultaneous_inference,
)
from v4.scripts.run_protocol101_policy_neutral_selector_stage0 import (
    _fold_sessions,
    decide,
)


def _target_frame() -> pd.DataFrame:
    rows = []
    for decision in (1, 2):
        for candidate in range(3):
            rows.append(
                {
                    "session": "2025-01-02",
                    "decision_time_ns": decision,
                    "contract_id": f"{decision}-{candidate}",
                    "right": "C" if candidate % 2 == 0 else "P",
                    "offset": float(candidate - 1) * 5.0,
                    "strike_index": candidate,
                    "return_1m": float(candidate),
                    "return_2m": float(candidate),
                    "return_5m": float(candidate),
                    "return_10m": float(candidate),
                    "return_15m": float(candidate),
                    "mfe_15m": float(candidate),
                    "mae_15m": float(candidate),
                    "time_to_breakeven_quality": float(candidate),
                    "barrier_f10_a10": float(candidate - 1),
                    "barrier_f25_a20": float(candidate - 1),
                    "barrier_f50_a35": float(candidate - 1),
                }
            )
    return pd.DataFrame(rows)


def test_midrank_percentiles_and_missing_rule() -> None:
    observed = midrank_percentiles([10.0, 20.0, 20.0, np.nan])
    np.testing.assert_allclose(observed, [0.0, 0.75, 0.75, 0.0])
    np.testing.assert_allclose(midrank_percentiles([np.nan, 3.0]), [0.0, 0.5])


def test_primary_utility_is_bounded_and_decision_local() -> None:
    frame = add_primary_utility(_target_frame())
    assert frame["primary_utility"].between(0.0, 1.0).all()
    for _, group in frame.groupby("decision_time_ns"):
        assert group["primary_utility"].is_monotonic_increasing
        assert group["primary_utility"].iloc[0] == 0.0
        assert group["primary_utility"].iloc[-1] == 1.0


def test_path_target_uses_causal_horizons_and_adverse_barrier() -> None:
    minute = 60_000_000_000
    record = path_target_record(
        quote_ns=np.asarray([minute, 2 * minute, 3 * minute], dtype=np.int64),
        bids=np.asarray([1.2, 0.0, 2.0], dtype=float),
        decision_ns=0,
        entry_ask=1.0,
    )
    assert record["return_1m"] == pytest.approx(0.17)
    assert record["barrier_f10_a10"] == 1.0
    assert record["barrier_f25_a20"] == -1.0
    assert record["time_to_breakeven_quality"] == -1.0


def test_frozen_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "offset": [5.0, -5.0, -5.0],
            "strike_index": [2, 1, 1],
            "right": ["P", "P", "C"],
            "contract_id": ["z", "b", "a"],
        }
    )
    assert list(candidate_order(frame)) == [2, 1, 0]
    assert select_position(frame, [1.0, 1.0, 1.0]) == 2


def test_complete_group_sampling_and_weights() -> None:
    rows = []
    for session in ("a", "b"):
        for decision in range(3):
            for candidate in range(decision + 1):
                rows.append(
                    {
                        "session": session,
                        "decision_time_ns": decision,
                        "contract_id": f"{session}-{decision}-{candidate}",
                        "offset": float(candidate),
                    }
                )
    frame = pd.DataFrame(rows)
    sampled = deterministic_group_sample(
        frame,
        maximum_candidates=8,
        fold_id="F1",
    )
    original = frame.groupby(["session", "decision_time_ns"]).size()
    observed = sampled.groupby(["session", "decision_time_ns"]).size()
    assert observed.equals(original.loc[observed.index])

    weights = group_balanced_weights(frame)
    totals = pd.Series(weights).groupby(frame["session"]).sum()
    np.testing.assert_allclose(totals.to_numpy(), [totals.iloc[0]] * 2)


def test_selector_d1_preserves_target_multisets() -> None:
    frame = add_primary_utility(_target_frame())
    shuffled, receipt = permute_targets_within_decision(frame, seed=8600)
    assert receipt["decision_groups"] == 2
    for _, indexes in frame.groupby(
        ["session", "decision_time_ns"], sort=False
    ).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        np.testing.assert_allclose(
            np.sort(shuffled[positions]),
            np.sort(frame["primary_utility"].to_numpy()[positions]),
        )


def test_synchronized_inference_detects_positive_effect() -> None:
    sessions = [f"2025-01-{index:02d}" for index in range(1, 21)]
    folds = ["F1"] * 10 + ["F2"] * 10
    schedule = block_schedule(
        sessions,
        folds,
        replicates=1_000,
        seed=123,
    )
    result = simultaneous_inference(
        {
            "positive": np.linspace(0.01, 0.03, 20),
            "negative": np.linspace(-0.03, -0.01, 20),
        },
        schedule=schedule,
    )
    assert result["rows"]["positive"]["adjusted_ci_95"][0] > 0.0
    assert result["rows"]["negative"]["adjusted_ci_95"][1] < 0.0


def test_stage0_sessions_are_nested_and_exclude_final_oof() -> None:
    training, validation, receipt = _fold_sessions()
    assert len(training) == 15
    assert len(validation) == 5
    assert max(training) < min(validation)
    assert receipt["final_oof_overlap"] == []


def test_stage0_proceeds_only_when_real_beats_controls() -> None:
    machinery = {"all_pass": True}
    results = {
        "validation_sessions": 5,
        "validation_opportunities": 500,
        "selector_effects": {
            "M0": {"minimum_baseline_lift": 0.01},
            "M1": {"minimum_baseline_lift": 0.02},
            "M0_reversed": {"minimum_baseline_lift": -0.01},
            "M1_reversed": {"minimum_baseline_lift": 0.0},
            "M0_shuffle8600": {"minimum_baseline_lift": 0.001},
            "M1_shuffle8600": {"minimum_baseline_lift": -0.001},
            "M0_shuffle8601": {"minimum_baseline_lift": 0.002},
            "M1_shuffle8601": {"minimum_baseline_lift": 0.003},
        },
    }
    assert decide(machinery, results, persist=False)["terminal_decision"] == (
        "proceed_to_full_campaign"
    )
    results["selector_effects"]["M0_shuffle8600"][
        "minimum_baseline_lift"
    ] = 0.03
    assert decide(machinery, results, persist=False)["terminal_decision"] == (
        "stop_no_preliminary_signal"
    )


def test_stage0_contract_defect_does_not_name_a_best_model() -> None:
    machinery = {"all_pass": True}
    results = {
        "validation_sessions": 5,
        "validation_opportunities": 500,
        "selector_effects": {
            "M0": {"minimum_baseline_lift": float("-inf")},
            "M1": {"minimum_baseline_lift": float("-inf")},
            "M0_reversed": {"minimum_baseline_lift": float("-inf")},
        },
    }
    decision = decide(
        machinery,
        results,
        contract_audit={
            "status": "scientific_contract_defect",
            "defect": "undefined baseline",
        },
        persist=False,
    )
    assert decision["terminal_decision"] == "stop_scientific_contract_defect"
    assert decision["performance_interpreted"] is False
    assert decision["best_real_model"] is None
    assert decision["best_control"] is None
