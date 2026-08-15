from __future__ import annotations

import pandas as pd

from v5.ops import analyze_causal_day_atlas as atlas


def test_candidate_aggregation_distinguishes_opportunity_from_oracle_net() -> None:
    frame = pd.DataFrame(
        {
            "session": ["s", "s"],
            "entry_minute": ["10:00", "10:00"],
            "is_call": [True, False],
            "entry_ask_usd": [500.0, 700.0],
            **{
                f"clock_exit_status_{h}m": ["executable_at_target", "executable_at_target"]
                for h in atlas.HORIZONS
            },
            **{f"net_bid_{h}m_usd": [-100.0, 300.0] for h in atlas.HORIZONS},
            **{f"option_mfe_{h}m_usd": [50.0, 500.0] for h in atlas.HORIZONS},
            **{
                f"reached_{name}_{h}m": [False, depth <= 20]
                for h in atlas.HORIZONS
                for depth, name in ((0, "cross"), (10, "10_itm"), (20, "20_itm"), (30, "30_itm"))
            },
        }
    )
    got = atlas.aggregate_candidates(frame).iloc[0]
    assert got["mean_clock_net_bid_60m_usd"] == 100.0
    assert got["oracle_best_contract_net_bid_60m_usd"] == 300.0
    assert got["any_contract_reached_20_itm_60m"]
    assert not got["any_contract_reached_30_itm_60m"]


def test_bootstrap_difference_uses_sessions_as_the_unit() -> None:
    got = atlas.bootstrap_difference(pd.Series([1.0, -1.0, 0.0]).to_numpy())
    assert got["n_sessions"] == 3
    assert got["mean"] == 0.0
    assert got["ci_low"] <= 0.0 <= got["ci_high"]
