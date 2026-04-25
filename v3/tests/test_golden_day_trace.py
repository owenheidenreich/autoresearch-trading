from __future__ import annotations

import argparse
import os
import unittest

import numpy as np

from v3.analysis.golden_day_trace import (
    DEFAULT_DAY,
    DEFAULT_ORACLE,
    _compare_day_bundle,
    _load_oracle,
    _rebuild_one_day,
    _run_overfit,
)
from v3.layer2.common import load_export_bundle
from v3.layer2.train_unified_policy import _slice_inputs


DATASET = "v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl"


@unittest.skipUnless(os.path.exists(DATASET) and os.path.exists("v2/data.pt"), "golden-day artifacts not present")
class GoldenDayTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_export_bundle(DATASET)
        cls.rows = cls.bundle["rows"].reset_index(drop=True)
        cls.day_mask = cls.rows["day"].astype(str).eq(DEFAULT_DAY).to_numpy()
        cls.full_idx = np.flatnonzero(cls.day_mask)
        if cls.full_idx.size == 0:
            raise unittest.SkipTest(f"{DEFAULT_DAY} not present in action-surface artifact")

    def test_one_day_rebuild_matches_full_artifact_slice(self) -> None:
        one_day = _rebuild_one_day(DEFAULT_DAY, self.bundle["meta"], 25_000.0)
        compare = _compare_day_bundle(self.bundle, one_day, self.full_idx)
        self.assertTrue(compare["all_match"], compare)

    def test_real_golden_day_next_bar_fill_labels(self) -> None:
        labels = self.bundle["action_labels"]
        bars = self.rows.loc[self.day_mask, "bar_index"].to_numpy(dtype=np.float32)
        entry_fill = labels["entry_fill_bar"][self.day_mask]
        tradeable = labels["tradeable_mask"][self.day_mask] > 0.5
        nonflat_tradeable = tradeable[:, 1:]
        expected = np.repeat((bars[:, None] + 1.0), nonflat_tradeable.shape[1], axis=1)
        self.assertTrue(np.all(entry_fill[:, 1:][nonflat_tradeable] == expected[nonflat_tradeable]))

    def test_real_golden_day_has_complete_risk_band_tokens(self) -> None:
        meta = self.bundle["meta"]
        top_k = int(meta["top_k_contracts_per_side"])
        names = list(meta["contract_feature_names"])
        role_idx = names.index("slot_role")
        # The selected failure bar must expose the full 24-token action space;
        # earlier bars can have fewer listed contracts and are covered by the
        # present mask rather than a forced full-token assertion.
        selected_bar_mask = self.day_mask & self.rows["bar_index"].astype(int).eq(43).to_numpy()
        contract_mask = self.bundle["contract_mask"][selected_bar_mask]
        contracts = self.bundle["contract_features"][selected_bar_mask]
        self.assertEqual(contract_mask.shape[1], top_k * 2)
        self.assertTrue(np.all(contract_mask.sum(axis=1) == top_k * 2))
        for side_start in (0, top_k):
            roles = contracts[:, side_start : side_start + top_k, role_idx]
            self.assertTrue(np.all(np.diff(roles, axis=1) >= 0.0), roles)
            self.assertTrue(np.all(np.isin(roles, [0.0, 1.0, 2.0, 3.0, 4.0])), roles)

    @unittest.skipUnless(os.path.exists(DEFAULT_ORACLE), "golden-day oracle not present")
    def test_golden_day_overfit_canary_passes(self) -> None:
        n_actions = int(self.bundle["action_labels"]["tradeable_mask"].shape[1])
        pnl, exit_bar, _ = _load_oracle(DEFAULT_ORACLE, len(self.rows), n_actions)
        subset = _slice_inputs(
            self.day_mask,
            self.bundle,
            utility_target="hybrid_live",
            simulated_l3_pnl=pnl,
            simulated_l3_exit_bar=exit_bar,
        )
        args = argparse.Namespace(
            overfit_epochs=80,
            overfit_lr=1e-3,
            overfit_min_action_match=0.70,
            overfit_min_strong_put_score_rate=0.70,
            overfit_min_loss_drop=0.25,
            strong_put_edge=250.0,
        )
        summary, _ = _run_overfit(subset, self.bundle["meta"], args=args)
        self.assertTrue(summary["passed"], summary)


if __name__ == "__main__":
    unittest.main()
