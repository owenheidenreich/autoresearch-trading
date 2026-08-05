from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from v4.scripts.capture_databento_live_opra_training_twin import (
    ALLOWED_SCHEMAS,
    ONE_MINUTE_NS,
    _interval_end_ns,
    _stable_hash,
    select_session_symbols,
)


def test_select_session_symbols_is_exact_unique_and_spxw_only(tmp_path: Path) -> None:
    path = tmp_path / "definition.parquet"
    pd.DataFrame(
        {
            "raw_symbol": [
                "SPXW  260803C06000000",
                "SPXW  260803P06000000",
                "SPXW  260803C06000000",
                "SPXW  260805C06000000",
                "SPX   260803C06000000",
            ],
            "expiration": [
                "2026-08-03",
                "2026-08-03",
                "2026-08-03",
                "2026-08-05",
                "2026-08-03",
            ],
            "asset": ["SPXW", "SPXW", "SPXW", "SPXW", "SPX"],
        }
    ).to_parquet(path, index=False)
    assert select_session_symbols(path, date(2026, 8, 3)) == (
        "SPXW  260803C06000000",
        "SPXW  260803P06000000",
    )


def test_select_session_symbols_rejects_empty_scope(tmp_path: Path) -> None:
    path = tmp_path / "definition.parquet"
    pd.DataFrame(
        {
            "raw_symbol": ["SPXW  260805C06000000"],
            "expiration": ["2026-08-05"],
            "asset": ["SPXW"],
        }
    ).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="no SPXW definitions"):
        select_session_symbols(path, date(2026, 8, 3))


def test_summary_hash_excludes_only_self_hash() -> None:
    payload = {"schema_version": "x", "value": 3, "summary_sha256": "old"}
    first = _stable_hash(payload)
    payload["summary_sha256"] = "different"
    assert _stable_hash(payload) == first
    payload["value"] = 4
    assert _stable_hash(payload) != first


def test_feature_surface_schemas_are_explicitly_bounded() -> None:
    assert set(ALLOWED_SCHEMAS) == {
        "cbbo-1s",
        "cbbo-1m",
        "cmbp-1",
        "tcbbo",
        "trades",
        "ohlcv-1m",
        "statistics",
        "status",
    }


def test_interval_end_clock_is_family_specific() -> None:
    cbbo = SimpleNamespace(ts_recv=123, ts_event=99)
    ohlcv = SimpleNamespace(ts_recv=None, ts_event=1_000)
    assert _interval_end_ns(cbbo, 192) == 123
    assert _interval_end_ns(cbbo, 193) == 123
    assert _interval_end_ns(ohlcv, 33) == 1_000 + ONE_MINUTE_NS
    assert _interval_end_ns(cbbo, 0) is None
