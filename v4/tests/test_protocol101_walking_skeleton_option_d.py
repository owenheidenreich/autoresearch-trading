from __future__ import annotations

import numpy as np
import pandas as pd

from v4.model.protocol101_walking_skeleton import (
    EXPECTED_GATE_HEADS,
    HORIZONS,
    LABEL_PREFIX,
    PATH_HEADS,
    cdf_rank,
    compose_decision,
)
from v4.scripts.run_protocol101_walking_skeleton_option_d_stage1 import (
    INFERENCE_FORBIDDEN_EXACT,
    _expected_gate_bootstrap_seed,
    _label_stripped,
    _phase_qhat_states,
    _session_clustered_mean_lower_correction,
)


def test_cdf_rank_uses_documented_same_band_minimum_fallback() -> None:
    cdfs = {
        "x|medium_3_8|opening_discovery": {
            "distinct_sessions": 7,
            "finite_rows": 3_000,
            "x": [0.0, 1.0],
            "cdf": [0.1, 0.9],
        },
        "x|medium_3_8|__POOLED__": {
            "distinct_sessions": 20,
            "finite_rows": 5_000,
            "x": [0.0, 1.0],
            "cdf": [0.2, 0.8],
        },
    }
    value, trace = cdf_rank(
        cdfs,
        target="x",
        band="medium_3_8",
        phase="opening_discovery",
        value=0.5,
    )
    assert value == 0.5
    assert trace == "x|medium_3_8|__POOLED__"


def test_cdf_rank_never_falls_across_premium_bands() -> None:
    cdfs = {
        "x|large_8_20|__POOLED__": {
            "distinct_sessions": 20,
            "finite_rows": 5_000,
            "x": [0.0, 1.0],
            "cdf": [0.2, 0.8],
        }
    }
    value, trace = cdf_rank(
        cdfs,
        target="x",
        band="medium_3_8",
        phase="opening_discovery",
        value=0.5,
    )
    assert np.isnan(value)
    assert trace == "missing_same_band"


def test_signed_split_conformal_pair_uses_exact_global_qhat() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "decision_time_ns": [
                pd.Timestamp(f"2025-01-0{day} 15:00:00Z").value for day in (1, 2, 3)
            ],
            "market_phase": ["primary_morning"] * 3,
            "h3_censored": [False] * 3,
            "h3_mfe_dollars": [0.0, 2.0, 5.0],
        }
    )
    lower = np.asarray([-1.0, 0.0, 1.0])
    upper = np.asarray([1.0, 3.0, 4.0])
    states, calibrated_lower, calibrated_upper, diagnostics = _phase_qhat_states(
        frame, "h3_mfe_dollars", lower, upper
    )
    # Scores are [-1, -1, 1]; finite-sample 80% uses the capped third order statistic.
    assert states["__GLOBAL__"]["qhat"] == 1.0
    np.testing.assert_allclose(calibrated_lower, lower - 1.0)
    np.testing.assert_allclose(calibrated_upper, upper + 1.0)
    assert diagnostics["raw_monotonicity_violations"] == 0


def test_label_stripped_inference_keeps_causal_ask_but_removes_outcomes() -> None:
    payload = {column: [1.0] for column in INFERENCE_FORBIDDEN_EXACT}
    payload.update(
        {
            "session": ["2025-03-05"],
            "decision_time_ns": [1],
            "contract_id": ["SPXW"],
            "decision_entry_ask": [2.5],
            "upside_h3_q10_mfe_dollars__calibrated": [0.1],
        }
    )
    stripped = _label_stripped(pd.DataFrame(payload))
    assert not (set(stripped.columns) & INFERENCE_FORBIDDEN_EXACT)
    assert "decision_entry_ask" in stripped
    assert "upside_h3_q10_mfe_dollars__calibrated" in stripped


def test_expected_gate_inventory_is_all_four_axes_at_all_horizons() -> None:
    assert len(EXPECTED_GATE_HEADS) == 28
    assert {name.split("_mean_")[0].removeprefix("expected_upside_") for name, _ in EXPECTED_GATE_HEADS} == set(HORIZONS)


def test_expected_gate_lower_correction_is_deterministic_and_session_clustered() -> None:
    sessions = pd.Series([f"2025-01-{day:02d}" for day in range(1, 21) for _ in range(2)])
    observed = np.asarray([1.0, 3.0] * 20)
    predicted = np.zeros(40)
    kwargs = {
        "sessions": sessions,
        "observed": observed,
        "predicted_mean": predicted,
        "valid": np.ones(40, dtype=bool),
        "head_name": EXPECTED_GATE_HEADS[0][0],
        "replicates": 2_000,
    }
    first, state = _session_clustered_mean_lower_correction(**kwargs)
    second, second_state = _session_clustered_mean_lower_correction(**kwargs)
    assert first == second == 2.0
    assert state["bootstrap_seed"] == second_state["bootstrap_seed"]
    assert state["bootstrap_seed"] == _expected_gate_bootstrap_seed(EXPECTED_GATE_HEADS[0][0])
    assert state["distinct_calibration_sessions"] == 20
    assert state["individual_outcome_quantile_used"] is False


def test_positive_gate_uses_expected_lower_while_q10_remains_downstream() -> None:
    decision_ns = pd.Timestamp("2025-03-05 15:00:00Z").value
    row: dict[str, object] = {
        "session": "2025-03-05",
        "decision_time_ns": decision_ns,
        "contract_id": "SPXW_TEST",
        "right": "C",
        "expiry": "2025-03-05",
        "strike_milli_points": 6_000_000,
        "premium_band": "medium_3_8",
        "market_phase": "primary_morning",
        "action_eligible": True,
        "expected_normalized_regret__calibrated": 0.0,
        "q90_regret_upper_bound": 0.0,
    }
    for name, _, _ in PATH_HEADS:
        row[f"{name}__calibrated"] = -1.0
    for name, _ in EXPECTED_GATE_HEADS:
        row[f"{name}__calibrated_lower"] = 1.0
    cdfs = {}
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        for axis in (
            "mfe_dollars",
            "mfe_return",
            "profit_area_dollars",
            "profit_area_return",
        ):
            cdfs[f"{prefix}_{axis}|medium_3_8|primary_morning"] = {
                "distinct_sessions": 20,
                "finite_rows": 2_000,
                "x": [-2.0, 0.0],
                "cdf": [0.0, 1.0],
            }
    result = compose_decision(
        pd.DataFrame([row]),
        cdfs=cdfs,
        model_gap_error=0.0,
        mfe_error_margin_dollars=0.0,
        mfe_error_margin_return=0.0,
        action_conditioned_gate_available=True,
    )
    assert result["proposed_contract_id"] == "SPXW_TEST"
    assert result["selected_gate_mean_expected_mfe_dollars_lower"] == 1.0
    assert result["selected_mean_mfe_dollars"] == -1.0
    assert result["wait_reason"] == "mfe_error_margin_not_strictly_cleared"
