"""Smoke test for the offline replay harness."""
from __future__ import annotations

import json
import os
import unittest

from v3.live_shadow.snapshot import read_decision_snapshots


class OfflineReplayJsonlTests(unittest.TestCase):
    """Verify that the offline replay JSONL is round-trip readable and
    every decision has the structure live-shadow gate checks expect.
    """

    JSONL = "v3/artifacts/live_shadow_offline_replay/seed42.jsonl"

    def setUp(self) -> None:
        if not os.path.exists(self.JSONL):
            self.skipTest(f"{self.JSONL} not produced (run offline_replay first)")

    def test_jsonl_is_round_trip_readable(self) -> None:
        snapshots = read_decision_snapshots(self.JSONL)
        self.assertGreater(len(snapshots), 0)
        for s in snapshots:
            self.assertIn("session_id", s)
            self.assertIn("day", s)
            self.assertIn("completed_bar_index", s)
            self.assertIn("candidates", s)
            self.assertIn("selected_action_id", s)
            self.assertIn("scores", s)

    def test_every_decision_has_resolved_contract(self) -> None:
        snapshots = read_decision_snapshots(self.JSONL)
        unresolved = []
        for s in snapshots:
            if s["selected_action_id"] > 0 and s["selected_contract"] is None:
                unresolved.append((s["day"], s["completed_bar_index"]))
        self.assertEqual(unresolved, [], f"unresolved decisions: {unresolved[:5]}")

    def test_candidate_count_per_decision(self) -> None:
        snapshots = read_decision_snapshots(self.JSONL)
        # All chosen decisions should have at least one tradeable candidate.
        for s in snapshots:
            if s["selected_action_id"] > 0:
                self.assertGreater(len(s["candidates"]), 0,
                                   f"day {s['day']} bar {s['completed_bar_index']} has 0 candidates")


if __name__ == "__main__":
    unittest.main()
