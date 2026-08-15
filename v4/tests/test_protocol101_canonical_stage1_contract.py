from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from v4.model.protocol101_canonical_stage1_contract import (
    CONTEXT_FEATURES,
    D_FEATURES,
    E_FEATURES,
    FEATURE_NAMES,
    HYPOTHESES,
    boundary_stable_mask,
    feature_matrix,
    assert_model_alpha_firewall,
)
from v4.model.protocol101_regimen_repair import Protocol101AlphaFirewallError


SAMPLE = Path(
    "data/processed/spxw_0dte_neural_protocol101_live_v2_microstructure_masked_15mo/"
    "2025-01-02.pkl"
)


def sample_row():
    return pickle.load(SAMPLE.open("rb"))[30]


def test_contract_is_exactly_the_scoped_17_features() -> None:
    assert len(FEATURE_NAMES) == 17
    assert len(set(FEATURE_NAMES)) == 17
    assert FEATURE_NAMES == (*CONTEXT_FEATURES, *D_FEATURES, *E_FEATURES)
    assert set(HYPOTHESES) == {"H0", "H1", "H2", "H3"}
    assert HYPOTHESES["H3"] == FEATURE_NAMES
    assert all(not name.startswith("C.mid.") for name in FEATURE_NAMES)
    assert "E.bs.iv" not in FEATURE_NAMES


def test_feature_builder_has_expected_shape_and_internal_greeks() -> None:
    matrix = feature_matrix(sample_row())
    assert matrix.shape == (21, 2, 17)
    assert np.isfinite(matrix[:, :, : len(CONTEXT_FEATURES)]).all()
    delta_idx = FEATURE_NAMES.index("E.bs.delta")
    gamma_idx = FEATURE_NAMES.index("E.bs.gamma")
    assert np.isfinite(matrix[:, :, delta_idx]).any()
    assert np.isfinite(matrix[:, :, gamma_idx]).any()


def test_near_atm_composites_do_not_populate_wing_slots() -> None:
    row = sample_row()
    offsets = np.asarray(row["strike_offsets"], dtype=float)
    matrix = feature_matrix(row)
    d_indexes = [FEATURE_NAMES.index(name) for name in D_FEATURES]
    assert np.isnan(matrix[np.abs(offsets) > 19][:, :, d_indexes]).all()
    assert np.isfinite(matrix[np.abs(offsets) <= 19][:, :, d_indexes]).any()


def test_boundary_stable_guard_is_stricter_than_candidate_mask() -> None:
    margins = {
        "stable_min_mid": 0.55,
        "stable_max_mid": 34.75,
        "stable_max_spread_abs": 0.45,
        "stable_max_spread_frac": 0.225,
        "stable_max_quote_age_ms": 85000.0,
        "stable_min_bid_size": 2.0,
        "stable_min_ask_size": 2.0,
        "max_affordability_utilization": 0.975,
    }
    row = sample_row()
    stable = boundary_stable_mask(row, margins)
    candidate = np.asarray(row["candidate_mask"], dtype=bool)
    assert stable.shape == candidate.shape
    assert np.all(~stable | candidate)


@pytest.mark.parametrize(
    "injected",
    [
        "label_realized_exit_time_ns",
        "label_source_exit_quote_time_ns",
        "label_exit_quote_age_ms",
        "label_exit_reason_code",
        "label_executable_exit_bid",
        "label_policy_deadline_ns",
        "label_invalid_reason_code",
        "labels_net_pnl",
        "future_path_numeric",
        "*",
    ],
)
def test_exact_alpha_firewall_rejects_injections_ALPHA_EXCLUSION_001(
    injected: str,
) -> None:
    with pytest.raises(Protocol101AlphaFirewallError):
        assert_model_alpha_firewall([*FEATURE_NAMES, injected])


def test_signed_synchronization_authority_D6_AUTHORITY_001() -> None:
    assert HYPOTHESES["H3"] == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 17
    assert all(
        name not in FEATURE_NAMES
        for name in (
            "exact_transfer_feature",
            "label_realized_exit_time_ns",
            "label_source_exit_quote_time_ns",
        )
    )
