"""Regression tests for strict contract-level selection targets.

The next hypothesis after exp_172 is that the scorer is trained on the wrong
oracle: strict opportunity is computed from per-contract path metrics, but the
selection loss still ranks all contracts by eventual long-hold PnL.

These tests pin the helper logic that narrows target mass to strict contracts
without changing the logit competition set.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2 import train  # noqa: E402


class TestTrainingEnvVars(unittest.TestCase):
    def test_sel_target_mode_is_captured_for_provenance(self):
        self.assertIn("SEL_TARGET_MODE", train._TRAINING_ENV_VARS)


class TestContractStrictMask(unittest.TestCase):
    def test_compute_contract_strict_mask_marks_only_fast_clean_contracts(self):
        sc = {
            "bar_ptrs": np.array([0, 3], dtype=np.int32),
            "row_raw_returns": np.array([
                [0.05, 0.15],   # strict
                [0.02, 0.10],   # fails r10
                [0.04, 0.20],   # fails mae
            ], dtype=np.float32),
            "row_mfe": np.array([
                [0.05, 0.20],
                [0.01, 0.05],
                [0.05, 0.20],
            ], dtype=np.float32),
            "row_mae": np.array([
                [-0.03, -0.04],
                [-0.02, -0.03],
                [-0.09, -0.10],
            ], dtype=np.float32),
            "row_bars_to_breakeven": np.array([2.0, 3.0, 2.0], dtype=np.float32),
        }
        got = train._compute_contract_strict_mask(sc, local_bar=0, max_contracts=5)
        want = np.array([True, False, False, False, False])
        np.testing.assert_array_equal(got, want)


class TestSelectionTargetPool(unittest.TestCase):
    def test_strict_mode_uses_strict_subset_when_present(self):
        valid = torch.tensor([[True, True, False], [True, True, True]])
        strict = torch.tensor([[False, True, False], [False, False, True]])
        got = train._selection_target_pool(valid, strict, "strict_mask")
        want = torch.tensor([[False, True, False], [False, False, True]])
        self.assertTrue(torch.equal(got, want))

    def test_strict_mode_falls_back_to_valid_when_no_strict_contract_exists(self):
        valid = torch.tensor([[True, True, False], [True, True, True]])
        strict = torch.tensor([[False, False, False], [False, True, False]])
        got = train._selection_target_pool(valid, strict, "strict_mask")
        want = torch.tensor([[True, True, False], [False, True, False]])
        self.assertTrue(torch.equal(got, want))

    def test_default_mode_ignores_strict_mask(self):
        valid = torch.tensor([[True, False, True]])
        strict = torch.tensor([[False, False, True]])
        got = train._selection_target_pool(valid, strict, "default")
        self.assertTrue(torch.equal(got, valid))


if __name__ == "__main__":
    unittest.main()
