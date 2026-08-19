"""The corpus build must label settlement per session and never pool the eras."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v5.ops import build_lifecycle_corpus as builder
from v5.ops.build_lifecycle_corpus import (
    SOURCE_OFFICIAL,
    SOURCE_PARITY,
    CorpusBuildError,
    EraSource,
    eligible_from_coverage,
    locate,
)

OWNED = "2025-08-01"
BACKFILL = "2022-06-01"


def _sources(tmp_path: Path) -> tuple[EraSource, ...]:
    owned = tmp_path / "owned"
    backfill = tmp_path / "backfill"
    owned.mkdir(parents=True, exist_ok=True)
    backfill.mkdir(parents=True, exist_ok=True)
    (owned / f"databento_spxw_0dte_{OWNED}.parquet").write_bytes(b"x")
    (backfill / f"databento_spxw_0dte_{BACKFILL}.parquet").write_bytes(b"x")
    return (
        EraSource("owned", owned, SOURCE_OFFICIAL),
        EraSource("backfill", backfill, SOURCE_PARITY),
    )


def test_a_session_resolves_to_its_own_era_and_settlement_law(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    owned_source, _ = locate(sources, OWNED)
    backfill_source, _ = locate(sources, BACKFILL)
    assert owned_source.settlement_source == SOURCE_OFFICIAL
    assert backfill_source.settlement_source == SOURCE_PARITY


def test_a_session_in_no_era_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CorpusBuildError, match="no quote file in any declared era"):
        locate(_sources(tmp_path), "1999-01-04")


def test_an_owned_session_without_a_validated_settlement_is_refused(
    tmp_path: Path,
) -> None:
    """The owned era must never silently fall back to a derived settlement."""

    payload = builder.run(
        sources=_sources(tmp_path),
        es_root=tmp_path / "es",
        sessions=[OWNED],
        out_dir=tmp_path / "out",
        receipt_path=tmp_path / "receipt.json",
        settlement_receipt=None,
    )
    assert payload["gate"] == "SESSIONS_FAILED"
    reason = payload["sessions"][0]["reason"]
    assert "validated receipt does not carry one" in reason or "missing ES candles" in reason


def test_eligibility_is_read_from_coverage_and_unioned(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    pd.DataFrame(
        {"session": ["2022-06-01", "2022-11-25"], "included_for_episode_build": [True, False]}
    ).to_csv(first, index=False)
    pd.DataFrame(
        {"session": ["2025-08-01"], "included_for_episode_build": [True]}
    ).to_csv(second, index=False)

    assert eligible_from_coverage([first, second]) == ["2022-06-01", "2025-08-01"]


def test_a_coverage_csv_without_the_eligibility_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"session": ["2022-06-01"]}).to_csv(path, index=False)
    with pytest.raises(CorpusBuildError, match="lacks eligibility column"):
        eligible_from_coverage([path])


def test_coverage_certifying_nothing_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "none.csv"
    pd.DataFrame(
        {"session": ["2022-06-01"], "included_for_episode_build": [False]}
    ).to_csv(path, index=False)
    with pytest.raises(CorpusBuildError, match="no eligible session"):
        eligible_from_coverage([path])


def test_an_empty_session_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CorpusBuildError, match="no eligible sessions"):
        builder.run(
            sources=_sources(tmp_path),
            es_root=tmp_path / "es",
            sessions=[],
            out_dir=tmp_path / "out",
            receipt_path=tmp_path / "receipt.json",
        )


def test_completed_sessions_are_skipped_so_the_build_resumes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for table in builder.TABLES:
        (out / table).mkdir(parents=True)
        (out / table / f"{BACKFILL}.parquet").write_bytes(b"done")

    payload = builder.run(
        sources=_sources(tmp_path),
        es_root=tmp_path / "es",
        sessions=[BACKFILL],
        out_dir=out,
        receipt_path=tmp_path / "receipt.json",
    )
    assert payload["sessions"][0]["classification"] == "ALREADY_PRESENT"
    assert payload["gate"] == "PASS"


def test_the_receipt_reports_base_rate_per_era_rather_than_pooled(
    tmp_path: Path,
) -> None:
    """The eras differ measurably; a pooled rate would hide that."""

    payload = builder.run(
        sources=_sources(tmp_path),
        es_root=tmp_path / "es",
        sessions=[BACKFILL],
        out_dir=tmp_path / "out",
        receipt_path=tmp_path / "receipt.json",
    )
    written = json.loads((tmp_path / "receipt.json").read_text())
    assert set(written["summary"]["label_base_rate_by_era"]) == {"owned", "backfill"}
    assert written["receipt_sha256"] == payload["receipt_sha256"]


def test_optional_provenance_columns_are_detected_from_the_schema(tmp_path: Path) -> None:
    """Regression: read_parquet(columns=[]) reports every column as absent.

    That mistake shipped once and produced a carried-close footnote of zero
    against 214 genuinely carried sessions -- the diagnostic reported clean
    because it could not see the column it was built to read.
    """

    import pyarrow.parquet as pq

    path = tmp_path / "databento_spxw_0dte_2022-06-22.parquet"
    pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2022-06-22 16:00", tz="America/New_York")],
            "underlying_price_source": ["carried_parity"],
            "underlying_carry_minutes": [1],
        }
    ).to_parquet(path, index=False)

    assert set(pd.read_parquet(path, columns=[]).columns) == set()  # the trap
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    assert {"underlying_price_source", "underlying_carry_minutes"} <= schema


def test_the_tape_source_is_read_from_the_candle_file_not_assumed(tmp_path: Path) -> None:
    """Owner ruling 2026-08-19: the corpus must say what its tape is made of.

    The pinned builder writes the values into `es_*` column names and may not be
    edited to rename them, so the stamp beside them is the only thing that stops
    a later session reading `es_close` and concluding the policy takes a futures
    input it was ruled out of.
    """

    es = tmp_path / "2024-03-15.es_c_0.ohlcv-1m.parquet"
    frame = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                          "volume": [0.0]})
    frame.to_parquet(es)
    assert builder.tape_source(es) == "es_futures"

    frame["tape_source"] = "spx_parity_spot"
    frame.to_parquet(es)
    assert builder.tape_source(es) == "spx_parity_spot"

    frame = pd.concat([frame, frame.assign(tape_source="es_futures")], ignore_index=True)
    frame.to_parquet(es)
    with pytest.raises(builder.CorpusBuildError, match="declares 2 tape sources"):
        builder.tape_source(es)
