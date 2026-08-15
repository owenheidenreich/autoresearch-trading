from __future__ import annotations

import pandas as pd

from v5.ops.attribute_defined_risk_iron_fly_legs import summarize_components


def test_component_summary_counts_chronological_fold_signs() -> None:
    frame = pd.DataFrame(
        {
            "traded": [True] * 5,
            "call_touch_net_usd": [1.0, 1.0, -1.0, 1.0, 1.0],
            "put_touch_net_usd": [-1.0] * 5,
            "call_mid_gross_usd": [2.0] * 5,
            "put_mid_gross_usd": [0.0] * 5,
        }
    )

    got = summarize_components(frame)

    assert got["call_touch_net_usd"]["positive_chronological_folds"] == 4
    assert got["put_touch_net_usd"]["positive_chronological_folds"] == 0
    assert got["call_mid_gross_usd"]["mean_per_session_usd"] == 2.0
