"""Stale-candidate cleanup: a remote with no candidate must clear the local slot.

If the local v2/models/model_candidate.pt survives across a sync cycle when
the remote has no candidate, `model_manage.keep()` could promote it.
candidate_staging.clear_stale_candidate() is the single point of truth for
that cleanup; all three deploy.sh sync paths route through it.

This test pins:
  1. clear_stale_candidate removes an existing file and reports truthy.
  2. clear_stale_candidate is a no-op when the file is absent.
  3. The deploy.sh source calls candidate_staging from all three sync paths
     (cmd_download, _run_sync initial baseline, _run_sync improvement).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2.ops.candidate_staging import clear_stale_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestClearStaleCandidate(unittest.TestCase):

    def test_removes_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "model_candidate.pt"
            candidate.write_bytes(b"stale-bytes")
            self.assertTrue(candidate.exists())
            removed = clear_stale_candidate(candidate, reason="unit test")
            self.assertTrue(removed)
            self.assertFalse(candidate.exists())

    def test_noop_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "missing.pt"
            self.assertFalse(candidate.exists())
            removed = clear_stale_candidate(candidate, reason="unit test")
            self.assertFalse(removed)

    def test_default_path_preserves_invariant_name(self):
        """Defend the canonical path from silent rename."""
        from v2.ops.candidate_staging import DEFAULT_CANDIDATE
        self.assertEqual(str(DEFAULT_CANDIDATE), "v2/models/model_candidate.pt")


class TestDeployShInvokesCleanup(unittest.TestCase):
    """All three sync paths in deploy.sh must route through candidate_staging."""

    def setUp(self):
        self.src = (PROJECT_ROOT / "v2" / "ops" / "deploy.sh").read_text()

    def test_cmd_download_clears_stale_candidate(self):
        self.assertIn(
            'cmd_download: remote has no model_candidate.pt',
            self.src,
            "cmd_download must invoke candidate_staging when remote has no candidate",
        )

    def test_run_sync_baseline_clears_stale_candidate(self):
        self.assertIn(
            '_run_sync baseline: remote has no model_candidate.pt',
            self.src,
            "_run_sync baseline branch must invoke candidate_staging",
        )

    def test_run_sync_improvement_clears_stale_candidate(self):
        self.assertIn(
            '_run_sync improvement: remote has no model_candidate.pt',
            self.src,
            "_run_sync improvement branch must invoke candidate_staging",
        )

    def test_all_three_paths_use_the_module(self):
        # Count occurrences so a future refactor that removes one doesn't slip by.
        n = self.src.count("python3 -m v2.ops.candidate_staging")
        self.assertGreaterEqual(
            n, 3,
            f"expected at least 3 invocations of v2.ops.candidate_staging in deploy.sh, found {n}",
        )


if __name__ == "__main__":
    unittest.main()
