"""Same calendar window -> same seed, across screening modes.

Invariant: generate_folds returns a FoldSpec whose window_id is the SHA-1 of
its six boundary dates. window_seed derives the training seed from window_id,
not from fold_idx. Therefore the latest fold of a 5-fold run and the sole fold
of an n_folds=1 run — if they share calendar boundaries — must have the same
seed.
"""
from __future__ import annotations

import unittest

from v2.core.walkforward import (
    generate_folds,
    resolve_fold_indices,
    window_seed,
)


def _fake_dates(n: int) -> list[str]:
    # 400 sequential "dates" — string identity is all that matters.
    return [f"2025-01-{i:04d}" for i in range(1, n + 1)]


class TestFoldSeedStability(unittest.TestCase):

    def test_latest_fold_window_id_is_stable_across_n_folds(self):
        dates = _fake_dates(500)
        full = generate_folds(dates, n_folds=5, test_window=60, val_window=40, shadow_days=20)
        single = generate_folds(dates, n_folds=1, test_window=60, val_window=40, shadow_days=20)
        latest_full = next(f for f in full if f.fold_idx == 4)
        lone = single[0]
        self.assertEqual(latest_full.test_days, lone.test_days)
        self.assertEqual(latest_full.window_id, lone.window_id,
                         "Latest fold's window_id must not depend on n_folds.")
        self.assertEqual(
            window_seed(123, latest_full.window_id),
            window_seed(123, lone.window_id),
            "Same window_id must produce the same seed.",
        )

    def test_resolve_screen_modes(self):
        self.assertEqual(resolve_fold_indices("latest", 5), [4])
        self.assertEqual(resolve_fold_indices("mini", 5), [0, 2, 4])
        self.assertEqual(resolve_fold_indices("full", 5), [0, 1, 2, 3, 4])

    def test_explicit_fold_indices_override(self):
        self.assertEqual(
            resolve_fold_indices("full", 5, explicit=[1, 3]),
            [1, 3],
        )

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            resolve_fold_indices("unknown", 5)

    def test_out_of_range_indices_rejected(self):
        with self.assertRaises(ValueError):
            resolve_fold_indices("full", 5, explicit=[9])


if __name__ == "__main__":
    unittest.main()
