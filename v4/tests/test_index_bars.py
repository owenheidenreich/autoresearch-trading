"""Tests for SPX/VIX index-bar normalization."""
from __future__ import annotations

import pandas as pd

from v4.ingest.index_bars import normalize_index_bars


def test_normalize_index_bars_accepts_value_series() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-01-02 14:30:00", "2026-01-02 14:31:00"],
            "value": [6500.25, 6501.50],
        }
    )

    out = normalize_index_bars(raw, symbol="SPX")

    assert list(out.columns) == [
        "event_time",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert out["symbol"].tolist() == ["SPX", "SPX"]
    assert out["close"].tolist() == [6500.25, 6501.50]
    assert out["open"].tolist() == [6500.25, 6501.50]
    assert str(out["event_time"].dt.tz) == "UTC"
