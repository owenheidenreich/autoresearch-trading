from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import (
    ACTION_LABEL_NAMES,
    CONTRACT_FEATURE_NAMES,
    _risk_band_contracts_for_side,
    build_contract_action_surface,
    hybrid_live_utility,
    validate_action_surface_bundle,
)
from v3.live_shadow.feature_parity import compare_feature_rows
from v3.logger.schema import BarRecord, ContractRecord, SurfaceSummary
from v3.oracles.opportunity import _ContractPath


def _contract(strike: float, delta: float, *, passed: bool = True) -> ContractRecord:
    return ContractRecord(
        strike=strike,
        right="C",
        mid=5.0,
        delta=delta,
        abs_delta=abs(delta),
        spread_fraction=0.05,
        premium=500.0,
        contract_valid=True,
        premium_cap_ok=True,
        premium_floor_ok=True,
        delta_floor_ok=True,
        spread_cap_ok=True,
        passed=passed,
        gamma_dollar=10.0 + abs(delta),
        theta_to_premium=2.0,
    )


class LiveReadinessRepairTests(unittest.TestCase):
    def test_risk_band_selector_is_deterministic_and_fills_slots(self) -> None:
        contracts = [
            _contract(5000 + i * 5, delta)
            for i, delta in enumerate([0.68, 0.58, 0.49, 0.43, 0.38, 0.32, 0.27, 0.23, 0.20, 0.17, 0.13, 0.10, 0.74])
        ]
        first = _risk_band_contracts_for_side(contracts, 5000.0, top_k=12)
        second = _risk_band_contracts_for_side(list(reversed(contracts)), 5000.0, top_k=12)
        self.assertEqual([(c.strike, role) for c, role in first], [(c.strike, role) for c, role in second])
        self.assertEqual(len(first), 12)
        self.assertEqual([role for _, role in first[:3]], ["high_delta"] * 3)
        self.assertEqual([role for _, role in first[3:7]], ["balanced"] * 4)
        self.assertEqual([role for _, role in first[7:11]], ["convex"] * 4)

    def test_action_surface_schema_fails_on_missing_required_label(self) -> None:
        rows = pd.DataFrame({"day": ["2026-01-02"], "bar_index": [15], "fold_id": [0]})
        bundle = {
            "rows": rows,
            "meta": {
                "n_action_contract_tokens": 2,
                "action_label_names": ["utility_raw", "tradeable_mask"],
                "contract_feature_names": list(CONTRACT_FEATURE_NAMES),
            },
            "sequence_features": np.zeros((1, 1, 1), dtype=np.float32),
            "sequence_mask": np.ones((1, 1), dtype=np.float32),
            "contract_features": np.zeros((1, 2, len(CONTRACT_FEATURE_NAMES)), dtype=np.float32),
            "contract_mask": np.ones((1, 2), dtype=np.float32),
            "contract_strike": np.ones((1, 2), dtype=np.float32),
            "action_labels": {
                "utility_raw": np.zeros((1, 3), dtype=np.float32),
                "tradeable_mask": np.ones((1, 3), dtype=np.float32),
            },
        }
        with self.assertRaisesRegex(RuntimeError, "missing labels"):
            validate_action_surface_bundle(bundle, required_labels={"horizon_pnl"})

    def test_hybrid_utility_penalizes_untradeable_inputs(self) -> None:
        self.assertTrue(np.isnan(hybrid_live_utility(100.0, entry_mid=0.0, spread_fraction=0.1, stopout_risk=0.0, entry_bar=20)))
        good = hybrid_live_utility(100.0, entry_mid=5.0, spread_fraction=0.02, stopout_risk=0.0, entry_bar=20)
        risky = hybrid_live_utility(100.0, entry_mid=5.0, spread_fraction=0.02, stopout_risk=1.0, entry_bar=20)
        self.assertGreater(good, risky)

    def test_action_surface_labels_use_next_bar_fill(self) -> None:
        contract = _contract(5010.0, 0.32)
        surface = SurfaceSummary(
            n_contracts_total=1,
            n_contracts_valid=1,
            n_contracts_passing=1,
            n_blocked_solely_by_premium_cap=0,
            any_contract_passes=True,
            any_passes_without_premium_cap=True,
            passing_abs_delta_q25=0.32,
            passing_abs_delta_q50=0.32,
            passing_abs_delta_q75=0.32,
            max_abs_delta_blocked_solely_by_cap=-1.0,
        )
        bar = BarRecord(
            day="2026-01-02",
            bar_index=15,
            timestamp_ms=0,
            underlying_close=5000.0,
            vwap=5000.0,
            vwap_slope=0.0,
            volume_ratio=1.0,
            first15_high=5010.0,
            first15_low=4990.0,
            first15_range_pct=0.004,
            bars_since_break_above_first15=0,
            bars_since_break_below_first15=0,
            vix=18.0,
            atm_iv=0.20,
            iv_percentile=0.50,
            eligible=True,
            teachers=(),
            contracts=(contract,),
            surface=surface,
            selections=(),
        )
        row_features = np.zeros((1, 22), dtype=np.float32)
        row_features[0, 0] = 1.0
        row_features[0, 1] = contract.strike
        row_features[0, 2] = 0.0
        sidecar = {
            "bar_ptrs": np.array([0] * 15 + [0, 1] + [1] * 20, dtype=np.int32),
            "row_features": row_features,
            "row_labels": np.zeros((1,), dtype=np.float32),
            "row_contract_idx": np.array([7], dtype=np.int32),
        }
        mids = np.full(40, np.nan, dtype=np.float64)
        mids[15] = 2.0
        mids[16:] = np.linspace(5.0, 7.0, 24)
        spread_fracs = np.full(40, 0.02, dtype=np.float64)
        path = _ContractPath(
            contract_idx=7,
            mids=mids,
            spread_fracs=spread_fracs,
            suffix_max=np.maximum.accumulate(np.nan_to_num(mids[::-1], nan=-np.inf))[::-1],
            suffix_min=np.minimum.accumulate(np.nan_to_num(mids[::-1], nan=np.inf))[::-1],
            ives=np.full(40, np.nan, dtype=np.float64),
            deltas=np.full(40, np.nan, dtype=np.float64),
            theta_to_premiums=np.full(40, np.nan, dtype=np.float64),
            gamma_dollars=np.full(40, np.nan, dtype=np.float64),
        )

        _, _, _, labels = build_contract_action_surface(
            bar,
            sidecar=sidecar,
            paths={7: path},
            top_k=12,
            mae_floor_pct=-0.30,
            stopout_horizon_bars=10,
            stopout_target_pct=0.20,
        )

        self.assertEqual(labels["entry_fill_bar"][1], 16.0)
        self.assertEqual(labels["entry_fill_mid"][1], 5.0)
        self.assertNotEqual(labels["entry_fill_mid"][1], mids[15])

    def test_feature_parity_reports_missing_and_mismatched_features(self) -> None:
        result = compare_feature_rows(
            feature_names=["a", "b", "c"],
            historical={"a": 1.0, "b": 2.0, "c": 3.0},
            live_style={"a": 1.0, "b": 2.1},
            atol=0.01,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_features, ["c"])
        self.assertEqual(result.mismatched_features, ["b"])


if __name__ == "__main__":
    unittest.main()
