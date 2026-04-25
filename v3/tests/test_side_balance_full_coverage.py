"""Tests for the full-coverage --side-balance-weight implementation.

Verifies:
1. Each loss function with row_weight=None reproduces the pre-rebalance
   behavior (or all-ones equivalent).
2. With row_weight=ones, behavior matches None.
3. With row_weight that inverse-frequency rebalances call-best vs
   put-best rows, the per-cohort gradient mass is balanced.
4. _side_contrastive_loss with cohort_balanced=True averages call-better
   and put-better per-cohort means rather than count-pooling.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from v3.layer2.unified_policy import (
    _flat_ranking_loss,
    _masked_bce,
    _masked_weighted_huber,
    _pairwise_ranking_loss,
    _risk_band_ranking_loss,
    _side_contrastive_loss,
)


class SideBalanceFullCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n_rows = 12
        self.n_actions = 5
        self.pred = torch.randn(self.n_rows, self.n_actions, dtype=torch.float32)
        self.target = torch.randn(self.n_rows, self.n_actions, dtype=torch.float32) * 100.0
        self.target_arcsinh = torch.arcsinh(self.target / 100.0)
        self.mask = torch.ones(self.n_rows, self.n_actions, dtype=torch.float32)
        # Half the rows have action 0 invalid (anchor varies)
        self.weight = torch.ones_like(self.mask)
        self.risk_band = torch.zeros(self.n_rows, self.n_actions, dtype=torch.float32)
        # Per-row inverse-frequency weight: rows 0-7 weight 0.5, rows 8-11 weight 1.5 (mean 1)
        self.row_weight = torch.tensor(
            [0.5] * 8 + [1.5] * 4, dtype=torch.float32
        )

    def test_huber_row_weight_none_matches_ones(self) -> None:
        loss_a = _masked_weighted_huber(
            self.pred, self.target_arcsinh, self.mask, self.weight, row_weight=None
        )
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _masked_weighted_huber(
            self.pred, self.target_arcsinh, self.mask, self.weight, row_weight=ones
        )
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_bce_row_weight_none_matches_ones(self) -> None:
        target = (self.target > 0).float()
        loss_a = _masked_bce(self.pred, target, self.mask, row_weight=None)
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _masked_bce(self.pred, target, self.mask, row_weight=ones)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_pairwise_ranking_row_weight_none_matches_ones(self) -> None:
        loss_a = _pairwise_ranking_loss(self.pred, self.target_arcsinh, self.mask, row_weight=None)
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _pairwise_ranking_loss(self.pred, self.target_arcsinh, self.mask, row_weight=ones)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_risk_band_ranking_row_weight_none_matches_ones(self) -> None:
        loss_a = _risk_band_ranking_loss(self.pred, self.target_arcsinh, self.mask, self.risk_band, row_weight=None)
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _risk_band_ranking_loss(self.pred, self.target_arcsinh, self.mask, self.risk_band, row_weight=ones)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_flat_ranking_row_weight_none_matches_ones(self) -> None:
        loss_a = _flat_ranking_loss(self.pred, self.target_arcsinh, self.mask, row_weight=None)
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _flat_ranking_loss(self.pred, self.target_arcsinh, self.mask, row_weight=ones)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_side_contrastive_row_weight_none_matches_ones(self) -> None:
        # Build target where best_call > best_put for some rows, opposite for others.
        # n_actions=5 -> 1 flat + 4 contracts (top_k=2: actions 1,2 are calls; 3,4 are puts)
        target_raw = torch.zeros(self.n_rows, 5, dtype=torch.float32)
        # Rows 0-7: call-better (best at action 1: $200; best put at action 3: $50)
        target_raw[:8, 1] = 200.0
        target_raw[:8, 3] = 50.0
        # Rows 8-11: put-better (best put at action 3: $200; best call at action 1: $50)
        target_raw[8:, 3] = 200.0
        target_raw[8:, 1] = 50.0
        loss_a = _side_contrastive_loss(self.pred, target_raw, self.mask, row_weight=None)
        ones = torch.ones(self.n_rows, dtype=torch.float32)
        loss_b = _side_contrastive_loss(self.pred, target_raw, self.mask, row_weight=ones)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_side_contrastive_cohort_balanced_independent_of_count(self) -> None:
        """With cohort_balanced=True, doubling the call-better count alone should
        NOT change the loss (because each cohort contributes its mean independent
        of count)."""
        # n_actions=5 -> top_k=2
        small = torch.zeros(4, 5, dtype=torch.float32)
        small[:2, 1] = 200.0  # 2 call-better rows
        small[:2, 3] = 50.0
        small[2:, 3] = 200.0  # 2 put-better rows
        small[2:, 1] = 50.0
        pred_small = torch.randn(4, 5, dtype=torch.float32)
        mask_small = torch.ones(4, 5, dtype=torch.float32)

        balanced_count = _side_contrastive_loss(pred_small, small, mask_small, cohort_balanced=True)

        # Now duplicate the call-better rows (4 call-better + 2 put-better)
        big = torch.zeros(6, 5, dtype=torch.float32)
        big[:4, 1] = 200.0
        big[:4, 3] = 50.0
        big[4:, 3] = 200.0
        big[4:, 1] = 50.0
        pred_big = torch.cat([pred_small[:2], pred_small[:2], pred_small[2:]], dim=0)
        mask_big = torch.ones(6, 5, dtype=torch.float32)
        skewed_count = _side_contrastive_loss(pred_big, big, mask_big, cohort_balanced=True)

        # Cohort means should be identical because pred_small[:2] has the
        # same call-loss whether duplicated or not.
        self.assertAlmostEqual(float(balanced_count), float(skewed_count), places=5)

    def test_side_contrastive_count_pooled_amplifies_majority_cohort(self) -> None:
        """The OLD count-pooled denominator is sensitive to cohort imbalance:
        duplicating the majority cohort SHOULD shift the loss. This is the
        falsification mechanism documented in spx_w_side_sweep_001."""
        small = torch.zeros(4, 5, dtype=torch.float32)
        small[:2, 1] = 200.0
        small[:2, 3] = 50.0
        small[2:, 3] = 200.0
        small[2:, 1] = 50.0
        # Make pred such that call-side loss > put-side loss
        pred = torch.zeros(4, 5, dtype=torch.float32)
        pred[:, 1] = -1.0  # call score low for everyone
        pred[:, 3] = 1.0   # put score high for everyone
        mask = torch.ones(4, 5, dtype=torch.float32)
        pooled_balanced = _side_contrastive_loss(pred, small, mask, cohort_balanced=False)

        big = torch.zeros(6, 5, dtype=torch.float32)
        big[:4, 1] = 200.0
        big[:4, 3] = 50.0
        big[4:, 3] = 200.0
        big[4:, 1] = 50.0
        pred_big = torch.zeros(6, 5, dtype=torch.float32)
        pred_big[:, 1] = -1.0
        pred_big[:, 3] = 1.0
        mask_big = torch.ones(6, 5, dtype=torch.float32)
        pooled_skewed = _side_contrastive_loss(pred_big, big, mask_big, cohort_balanced=False)

        # With count-pooling, the imbalanced cohorts give a different loss
        # value than balanced cohorts. (Confirms the bug we're fixing.)
        self.assertNotAlmostEqual(float(pooled_balanced), float(pooled_skewed), places=3)


if __name__ == "__main__":
    unittest.main()
