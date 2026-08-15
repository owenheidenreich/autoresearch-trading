from __future__ import annotations

import pandas as pd

from v5.ops.verify_causal_day_replays import structurally_selected_sessions


def test_replay_examples_are_selected_from_coverage_not_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "session": [f"2025-08-{value:02d}" for value in range(1, 8)],
            "included_for_episode_build": [True] * 7,
            "first_decision_eligible_contracts": [2, 1, 3, 4, 5, 0, 6],
            "future_profit_that_selector_must_ignore": [999, -999, 5, -5, 10, 0, -10],
        }
    )
    assert structurally_selected_sessions(frame) == [
        "2025-08-01",
        "2025-08-04",
        "2025-08-07",
        "2025-08-06",
    ]
