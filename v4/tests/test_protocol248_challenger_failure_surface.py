from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol248_challenger_failure_surface as p248


def test_moneyness_matches_spxw_long_option_convention() -> None:
    assert p248.moneyness("C", -5) == "ITM"
    assert p248.moneyness("C", 5) == "OTM"
    assert p248.moneyness("P", 5) == "ITM"
    assert p248.moneyness("P", -5) == "OTM"
    assert p248.moneyness("C", 0) == "ATM"


def test_scope_guard_rejects_mixed_metric_story() -> None:
    frame = pd.DataFrame(
        {
            "policy": ["challenger", "protocol101"],
            "seed": [1, 1],
        }
    )
    guards = p248.scope_guards(frame)
    assert guards["metric_scope"] == "single_seed_strict_one_account_serial_replay"
    assert guards["seeds"] == [1]
    assert "five_seed_total" in guards["forbidden_scopes_excluded"]
    assert "directional_subset_as_headline_equity" in guards["forbidden_scopes_excluded"]


def test_premium_repair_uses_live_ask_when_entry_premium_missing(tmp_path) -> None:
    path = tmp_path / "enriched.csv"
    pd.DataFrame(
        {
            "seed": [1],
            "policy": ["protocol101"],
            "reported_split": ["q3_2025"],
            "decision_time": ["2025-07-01T14:00:00+00:00"],
            "exit_time": ["2025-07-01T14:10:00+00:00"],
            "decision_ts": ["2025-07-01T14:00:00+00:00"],
            "exit_ts": ["2025-07-01T14:10:00+00:00"],
            "session": ["2025-07-01"],
            "contract_id": ["SPXW-20250701-06165.000-C"],
            "right": ["C"],
            "offset": [-10.0],
            "pnl": [100.0],
            "entry_ask": [float("nan")],
            "entry_ask_live": [12.5],
            "entry_premium": [float("nan")],
        }
    ).to_csv(path, index=False)

    loaded = p248.load_scoped_trades(path, seed=1)

    assert loaded["entry_ask"].iloc[0] == 12.5
    assert loaded["entry_premium"].iloc[0] == 1250.0


def test_drawdown_reports_peak_relative_and_start_relative_risk() -> None:
    frame = pd.DataFrame(
        {
            "policy": ["challenger", "challenger", "challenger"],
            "session": ["2025-07-01", "2025-07-01", "2025-07-02"],
            "decision_ts": pd.to_datetime(
                [
                    "2025-07-01T14:00:00+00:00",
                    "2025-07-01T14:10:00+00:00",
                    "2025-07-02T14:00:00+00:00",
                ],
                utc=True,
            ),
            "pnl": [1000.0, -500.0, -700.0],
        }
    )

    row = p248.summarize_drawdowns(frame, starting_equity=10_000.0)[0]

    assert row["max_drawdown"] == -1200.0
    assert round(row["max_drawdown_pct"], 6) == round(-1200.0 / 11000.0, 6)
    assert row["max_drawdown_pct_of_start"] == -0.12
