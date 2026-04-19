"""Extractor scope test — proves the extractor reads from bars 0-30 only.

This test does NOT prove that bar-30 features are intrinsically safe from
look-ahead (that is asserted by the feature-audit note in the Fork C plan).
It only proves that the ``extract_tier1_features`` functions do not read
from bars 31..389 of the session.

Procedure
---------
1. Load the real manifest to get its structure and shape.
2. Build a synthetic "blank-tail" manifest: copy ``X_sim`` + ``spot_prices``,
   then overwrite everything after bar 30 (inclusive of bar 31) with NaN.
3. For a fixed labeled session, run the extractor's core routines against
   both the original and the blanked manifest.
4. Assert the base 52/79-length feature row and the 5 aggregates are
   byte-identical.

The test runs against the real ``data.pt`` at its default location. It is
skipped (not failed) if that file is unavailable, so the test is portable to
environments without the large artifact.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from v2.fork_c.extract_tier1_features import (
    AGGREGATE_WINDOW,
    CUTOFF_BAR,
    DEFAULT_DATA,
    build_prior_close_tables,
    build_session_sigmas,
    compute_aggregates,
    session_index,
)


FIXTURE_DATE = "2024-06-21"  # A confirmed label-positive session with full RTH


class TestExtractScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_path = Path(os.environ.get("FORK_C_DATA_PATH", DEFAULT_DATA))
        if not data_path.exists():
            raise unittest.SkipTest(f"manifest not found at {data_path}")
        m = torch.load(data_path, map_location="cpu", weights_only=False)
        cls.X_sim = m["X_sim"].numpy().astype(np.float64).copy()
        cls.dates = list(m["dates"])
        cls.bar_of_day = m["bar_of_day"].numpy().astype(np.int32)
        cls.spot_prices = m["spot_prices"].numpy().astype(np.float64).copy()
        cls.sessions = session_index(cls.dates, cls.bar_of_day)
        if FIXTURE_DATE not in cls.sessions:
            raise unittest.SkipTest(
                f"fixture date {FIXTURE_DATE} missing from manifest"
            )
        cls.sorted_dates = sorted(cls.sessions.keys())

    def _extract_one(self, X_sim: np.ndarray, spot_prices: np.ndarray):
        """Return (base_row, aggregate_vec) for FIXTURE_DATE under the given
        (possibly blanked) arrays."""
        day_start, n = self.sessions[FIXTURE_DATE]
        base_row = X_sim[day_start + CUTOFF_BAR].copy()
        sigmas = build_session_sigmas(self.sorted_dates, self.sessions, spot_prices)
        prior, two_prior = build_prior_close_tables(
            self.sorted_dates, self.sessions, spot_prices
        )
        sigma_scalar = float(sigmas[self.sorted_dates.index(FIXTURE_DATE)])
        agg = compute_aggregates(
            day_start,
            spot_prices,
            prior[FIXTURE_DATE],
            two_prior[FIXTURE_DATE],
            sigma_scalar,
        )
        return base_row, agg

    def test_bar_of_day_cutoff_is_30(self) -> None:
        day_start, _ = self.sessions[FIXTURE_DATE]
        self.assertEqual(int(self.bar_of_day[day_start + CUTOFF_BAR]), CUTOFF_BAR)

    def test_blanking_tail_of_fixture_day_does_not_change_extracted_row(self) -> None:
        """Blank X_sim[day_start+31:day_start+390] on the fixture day only.

        The extracted base feature row and the 5 aggregates must be
        byte-identical to the unblanked version — otherwise the extractor
        is reading from bars beyond the 10:00 ET cutoff for this session.
        """
        base_orig, agg_orig = self._extract_one(self.X_sim, self.spot_prices)

        X_blanked = self.X_sim.copy()
        spot_blanked = self.spot_prices.copy()
        day_start, n = self.sessions[FIXTURE_DATE]
        tail_start = day_start + AGGREGATE_WINDOW  # bar 31 in session coords
        tail_end = day_start + n
        X_blanked[tail_start:tail_end, :] = np.nan
        spot_blanked[tail_start:tail_end] = np.nan

        base_new, agg_new = self._extract_one(X_blanked, spot_blanked)

        np.testing.assert_array_equal(
            base_orig,
            base_new,
            err_msg="base feature row changed when bars 31+ were blanked — extractor scope violation",
        )
        np.testing.assert_array_equal(
            agg_orig,
            agg_new,
            err_msg="aggregates changed when bars 31+ were blanked — aggregate scope violation",
        )

    def test_blanking_tail_of_prior_day_changes_sigma_but_not_cutoff_row(self) -> None:
        """Sanity check: blanking a PRIOR session's tail should change the
        σ-band scalar (which pools prior sessions) even though the cutoff
        row is unchanged. This distinguishes "extractor only reads cutoff
        bar" (what we want) from "extractor reads nothing at all"
        (which would also pass the previous test)."""
        idx = self.sorted_dates.index(FIXTURE_DATE)
        if idx < 10:
            self.skipTest("need at least 10 prior sessions to perturb σ-band")
        prior_date = self.sorted_dates[idx - 1]
        prior_start, prior_n = self.sessions[prior_date]

        base_orig, agg_orig = self._extract_one(self.X_sim, self.spot_prices)

        spot_perturbed = self.spot_prices.copy()
        # Perturb prior session's entire window so its deviation samples change.
        spot_perturbed[prior_start : prior_start + prior_n] *= 1.05

        base_new, agg_new = self._extract_one(self.X_sim, spot_perturbed)

        np.testing.assert_array_equal(
            base_orig,
            base_new,
            err_msg="cutoff X_sim row should not depend on prior-session spot perturbation",
        )
        # Aggregates: the σ-band scalar should shift even though the within-day
        # aggregates (range, end-vs-open) are unchanged because they depend
        # only on today's bars 0-30.
        self.assertNotEqual(
            float(agg_orig[4]),
            float(agg_new[4]),
            msg="σ-band scalar should shift when prior-session spot is perturbed",
        )
        # Within-day aggregates (indices 0, 1) must be unchanged.
        np.testing.assert_array_equal(agg_orig[:2], agg_new[:2])


if __name__ == "__main__":
    unittest.main()
