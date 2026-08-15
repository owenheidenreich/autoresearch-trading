from __future__ import annotations

import pandas as pd

from v5.ops.audit_spxw_owned_quote_scope import (
    quoted_expirations,
    vertical_entry_feasibility,
)


def test_quoted_expirations_distinguishes_definitions_from_quotes(tmp_path) -> None:
    quote_path = tmp_path / "2026-01-02.cbbo-1m.parquet"
    definition_path = tmp_path / "2026-01-02.definition.parquet"
    pd.DataFrame({"instrument_id": [1, 1, 2]}).to_parquet(quote_path, index=False)
    pd.DataFrame(
        {
            "instrument_id": [1, 2, 3],
            "expiration": ["2026-01-02", "2026-01-02", "2026-01-09"],
        }
    ).to_parquet(definition_path, index=False)

    got = quoted_expirations(quote_path, definition_path)

    assert got["quoted_instruments"] == 2
    assert got["quoted_expirations"] == ["2026-01-02"]
    assert got["same_day_only"] is True
    assert got["future_expiration_instruments"] == 0


def test_vertical_feasibility_is_entry_only_and_counts_fold_coverage() -> None:
    selected = pd.DataFrame(
        {
            "session": ["s1", "s2", "s3"],
            "entry_minute": ["10:00", "10:00", "10:00"],
            "contract_id": ["long1", "long2", "long3"],
            "fold": [1, 2, 3],
        }
    )
    ladder = pd.DataFrame(
        [
            {"session": "s1", "minute": "10:00", "contract_id": "long1", "strike": 100.0, "right": "C", "bid": 2.0, "ask": 2.1},
            {"session": "s1", "minute": "10:00", "contract_id": "short1", "strike": 105.0, "right": "C", "bid": 1.0, "ask": 1.1},
            {"session": "s2", "minute": "10:00", "contract_id": "long2", "strike": 100.0, "right": "P", "bid": 2.0, "ask": 2.1},
            {"session": "s2", "minute": "10:00", "contract_id": "short2", "strike": 95.0, "right": "P", "bid": 1.0, "ask": 1.1},
            {"session": "s3", "minute": "10:00", "contract_id": "long3", "strike": 100.0, "right": "C", "bid": 2.0, "ask": 2.1},
        ]
    )

    got = vertical_entry_feasibility(selected, ladder, widths=[5])[0]

    assert got["exact_live_pairs"] == 2
    assert got["risk_eligible_pairs"] == 2
    assert got["risk_eligible_folds"] == ["1", "2"]
    assert got["four_of_five_chronology_attainable"] is False
