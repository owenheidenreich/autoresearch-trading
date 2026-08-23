from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.ops.measure_unconditional_spx_moves import (
    TerrainRunError,
    canonical_json,
    run,
)
from v5.research.unconditional_spx_moves import (
    EXPECTED_MINUTES,
    TapeCorpus,
    UnconditionalMoveError,
    _race_summary,
    analyze_corpus,
    first_crossing,
    load_tape_corpus,
    minutes_to_close,
    path_excursions,
    validate_tables,
    valid_configurations,
    wilson_interval,
)


def valid_tape_frame(session: str) -> pd.DataFrame:
    index = pd.date_range(
        f"{session} 09:30",
        periods=390,
        freq="min",
        tz="America/New_York",
        name="ts_event",
    ).tz_convert("UTC")
    close = 6_000.0 + np.arange(390, dtype=float) / 10.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.zeros(390),
            "bar_observation_minute": EXPECTED_MINUTES,
            "tape_source": "spx_parity_spot",
        },
        index=index,
    )


def tape_path(root: Path, session: str) -> Path:
    return root / f"{session}.es_c_0.ohlcv-1m.parquet"


def test_path_excursions_include_the_zero_entry_and_reverse_by_side() -> None:
    call_path = np.array([[2.0, 4.0, 1.0], [-1.0, -3.0, -2.0], [0.0, 0.0, 0.0]])
    call_favourable, call_adverse = path_excursions(call_path)
    put_favourable, put_adverse = path_excursions(-call_path)

    assert np.array_equal(call_favourable, np.array([4.0, 0.0, 0.0]))
    assert np.array_equal(call_adverse, np.array([0.0, 3.0, 0.0]))
    assert np.array_equal(call_favourable, put_adverse)
    assert np.array_equal(call_adverse, put_favourable)


def test_first_observed_crossing_and_race_partition_keep_neither() -> None:
    path = np.array(
        [
            [1.0, 5.0, -5.0],
            [-1.0, -5.0, 6.0],
            [0.0, 0.0, 0.0],
            [5.0, -5.0, 0.0],
        ]
    )
    gain = first_crossing(path, 5.0, above=True)
    loss = first_crossing(path, -5.0, above=False)
    summary = _race_summary(gain, loss, 3, np.ones(4, dtype=bool))

    assert gain.tolist() == [2, 3, 4, 1]
    assert loss.tolist() == [3, 2, 4, 2]
    assert summary["gain_first_count"] == 2
    assert summary["loss_first_count"] == 1
    assert summary["neither_count"] == 1
    assert summary["same_snapshot_tie_count"] == 0
    assert sum(summary[f"{name}_count"] for name in (
        "gain_first", "loss_first", "same_snapshot_tie", "neither"
    )) == 4


def test_wilson_boundaries_are_not_false_certainty() -> None:
    zero = wilson_interval(0, 100)
    one = wilson_interval(100, 100)
    assert zero[0] == 0.0 < zero[1]
    assert one[0] < one[1] == 1.0


def test_observation_clock_and_late_grid_are_exact() -> None:
    assert minutes_to_close("09:31") == 389
    assert minutes_to_close("10:30") == 330
    cells = {(start, horizon) for start, _, horizon in valid_configurations()}
    assert ("15:55", 5) in cells
    assert ("15:55", 10) not in cells


def test_loader_validates_clock_source_and_excludes_bad_session_before_read(tmp_path: Path) -> None:
    valid_tape_frame("2024-01-02").to_parquet(tape_path(tmp_path, "2024-01-02"))
    # This file is intentionally not Parquet.  A known defect must be excluded
    # before its prices are opened, while its bytes remain bound in the manifest.
    tape_path(tmp_path, "2023-06-26").write_bytes(b"known-defect-placeholder")

    corpus = load_tape_corpus(tmp_path, strict_population=False)

    assert corpus.sessions == ("2024-01-02",)
    assert len(corpus.input_manifest) == 2
    defect = corpus.input_manifest.loc[
        corpus.input_manifest["session"].eq("2023-06-26")
    ].iloc[0]
    assert defect["status"] == "EXCLUDED_KNOWN_DEFECT"


def test_loader_refuses_clock_invention_and_confirmation_session(tmp_path: Path) -> None:
    bad = valid_tape_frame("2024-01-02")
    bad["bar_observation_minute"] = tuple(
        ["09:30", *EXPECTED_MINUTES[:-1]]
    )
    bad.to_parquet(tape_path(tmp_path, "2024-01-02"))
    with pytest.raises(UnconditionalMoveError, match="observation clock"):
        load_tape_corpus(tmp_path, strict_population=False)

    reserved_root = tmp_path / "reserved"
    reserved_root.mkdir()
    valid_tape_frame("2026-08-06").to_parquet(tape_path(reserved_root, "2026-08-06"))
    with pytest.raises(UnconditionalMoveError, match="confirmation-reserved"):
        load_tape_corpus(reserved_root, strict_population=False)


def synthetic_corpus() -> TapeCorpus:
    dates = (
        date(2022, 6, 1),
        date(2022, 7, 1),
        date(2023, 2, 1),
        date(2024, 2, 1),
        date(2025, 2, 1),
        date(2025, 8, 1),
        date(2026, 1, 2),
        date(2026, 6, 1),
        date(2026, 7, 1),
    )
    base = np.arange(390, dtype=float)
    prices = np.vstack(
        [6_000.0 + index * 10.0 + base * ((-1.0) ** index) for index in range(len(dates))]
    )
    sessions = tuple(value.isoformat() for value in dates)
    manifest = pd.DataFrame({"session": sessions, "status": "ANALYZED"})
    return TapeCorpus(
        root=Path("/synthetic"),
        sessions=sessions,
        dates=dates,
        prices=prices,
        minutes=EXPECTED_MINUTES,
        input_manifest=manifest,
        defect_disposition=(),
    )


def test_full_grid_keeps_era_year_common_calendar_and_symmetry() -> None:
    corpus = synthetic_corpus()
    tables = analyze_corpus(corpus)
    quality_control = validate_tables(corpus, tables)
    assert quality_control["status"] == "PASS"
    scopes = set(tables.excursion["scope"])
    assert {
        "pooled",
        "era_backfill_2022-06-01_to_2025-07-31",
        "era_owned_2025-08-01_to_2026-07-30",
        "year_2022",
        "year_2026",
        "year_2022_june_july",
        "year_2026_june_july",
    } <= scopes
    selected = tables.excursion[
        (tables.excursion["scope"] == "pooled")
        & (tables.excursion["start_time_et"] == "10:30")
        & (tables.excursion["horizon_minutes"] == 20)
        & (tables.excursion["threshold_points"] == 10.0)
    ]
    call_favourable = selected[
        (selected["direction"] == "call") & (selected["metric"] == "favourable")
    ].iloc[0]
    put_adverse = selected[
        (selected["direction"] == "put") & (selected["metric"] == "adverse")
    ].iloc[0]
    assert call_favourable["successes"] == put_adverse["successes"]
    assert {
        ("year_2022", "year_2026"),
        ("year_2022_june_july", "year_2026_june_july"),
    } <= set(zip(tables.comparisons["from_scope"], tables.comparisons["to_scope"]))


def test_failure_attempt_is_preserved_and_self_hashed(tmp_path: Path) -> None:
    output = tmp_path / "attempt001"
    repo_root = Path(__file__).resolve().parents[2]
    receipt = run(tmp_path / "missing-tape", output, repo_root=repo_root)

    assert receipt["status"] == "FAIL"
    stored = json.loads((output / "failure_receipt.json").read_text())
    claimed = stored.pop("receipt_sha256")
    assert claimed == hashlib.sha256(canonical_json(stored)).hexdigest()
    assert (output / "measure_unconditional_spx_moves.py").is_file()
    assert (output / "unconditional_spx_moves.py").is_file()
    assert "Traceback" in (output / "run.log").read_text()
    with pytest.raises(TerrainRunError, match="refusing to overwrite"):
        run(tmp_path / "missing-tape", output, repo_root=repo_root)
