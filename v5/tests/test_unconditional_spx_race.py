from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.ops.measure_unconditional_spx_race import (
    RaceRunError,
    canonical_json,
    run,
)
from v5.research import unconditional_spx_race as race
from v5.research import unconditional_spx_iv as iv


def _valid_tape_frame(session: str) -> pd.DataFrame:
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
            "bar_observation_minute": race.EXPECTED_MINUTES,
            "tape_source": race.TAPE_SOURCE,
        },
        index=index,
    )


def _tape_path(root: Path, session: str) -> Path:
    return root / f"{session}.es_c_0.ohlcv-1m.parquet"


def _iv_frame(session: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contracts = (
        (95.0, True, 0.10),
        (97.5, True, 0.10),
        (100.0, True, 0.20),
        (100.0, False, 0.40),
        (102.5, True, 0.0100002),  # solver-bound artefact: excluded
        (102.5, False, 0.60),
        (105.0, True, 0.30),
        (105.0, False, 0.50),
    )
    for start in iv.START_TIMES:
        event_time = pd.Timestamp(f"{session} {start}", tz="America/New_York").tz_convert("UTC")
        for strike, is_call, self_iv in contracts:
            signed_money = 102.5 - strike if is_call else strike - 102.5
            rows.append(
                {
                    "session": session,
                    "event_time": event_time,
                    "expiry": session,
                    "contract_id": f"{session}-{start}-{strike}-{is_call}",
                    "strike": strike,
                    "underlying_price": 102.5,
                    "minute": start,
                    "moneyness_itm_points": signed_money,
                    "is_call": is_call,
                    "minutes_to_expiry": 60.0,
                    "self_iv": self_iv,
                    "tape_source": iv.TAPE_SOURCE,
                }
            )
    return pd.DataFrame(rows, columns=iv.LADDER_COLUMNS)


def _race_counts(path: np.ndarray, *, favourable: float, adverse: float) -> dict[str, int]:
    horizon = path.shape[1]
    gain = race.first_crossing(path, favourable, above=True)
    loss = race.first_crossing(path, -adverse, above=False)
    row = race._race_row(
        gain,
        loss,
        mask=np.ones(path.shape[0], dtype=bool),
        horizon=horizon,
    )
    return {
        "favourable": int(row["favourable_first_count"]),
        "adverse": int(row["adverse_first_count"]),
        "neither": int(row["neither_count"]),
    }


def _small_corpus() -> race.TapeCorpus:
    dates = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2025, 8, 1),
        date(2025, 8, 4),
    )
    # Entry is column zero.  The next three columns are the exact t+1..t+3
    # path consumed by the narrowed synthetic analysis below.
    moves = np.asarray(
        [
            [0.0, 5.0, 10.0, -6.0],  # favourable first, time 1, overshoot 5, pre-M adverse 0
            [0.0, -5.0, 5.0, 20.0],  # exact stop touch first, eventual M at minute 2
            [0.0, -4.0, 5.0, 7.0],   # stop survives, M at minute 2, overshoot 2
            [0.0, 0.0, 0.0, 0.0],    # neither barrier
        ]
    )
    prices = np.empty((len(dates), len(race.EXPECTED_MINUTES)), dtype=float)
    for index, row in enumerate(moves):
        base = 6_000.0 + 100.0 * index
        prices[index] = base
        prices[index, : len(row)] = base + row
    sessions = tuple(value.isoformat() for value in dates)
    return race.TapeCorpus(
        root=Path("/synthetic"),
        sessions=sessions,
        dates=dates,
        prices=prices,
        minutes=race.EXPECTED_MINUTES,
        input_manifest=pd.DataFrame({"session": sessions, "status": "ANALYZED"}),
        defect_disposition=(),
    )


def test_clock_includes_exact_1600_endpoint_and_omits_past_close() -> None:
    cells = {(start, horizon) for start, _, horizon in race.valid_configurations()}

    assert ("15:55", 5) in cells
    assert ("15:55", 10) not in cells
    assert ("15:50", 10) in cells
    assert ("15:50", 15) not in cells


def test_three_scalar_race_states_partition_exactly_and_tie_fails_closed() -> None:
    path = np.asarray(
        [
            [5.0, -5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    gain = race.first_crossing(path, 5.0, above=True)
    adverse = race.first_crossing(path, -5.0, above=False)
    row = race._race_row(
        gain,
        adverse,
        mask=np.ones(3, dtype=bool),
        horizon=3,
    )

    assert row["favourable_first_count"] == 1
    assert row["adverse_first_count"] == 1
    assert row["neither_count"] == 1
    assert (
        row["favourable_first_count"]
        + row["adverse_first_count"]
        + row["neither_count"]
        == row["sessions"]
    )

    with pytest.raises(race.UnconditionalRaceError, match="scalar barriers tied"):
        race._race_row(
            np.asarray([1]),
            np.asarray([1]),
            mask=np.asarray([True]),
            horizon=3,
        )


def test_call_put_race_identity_swaps_favourable_and_adverse_thresholds() -> None:
    call_path = np.asarray(
        [
            [6.0, -3.0, 0.0],
            [-3.0, 6.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -3.0, -6.0],
            [2.0, 7.0, -8.0],
        ]
    )
    favourable, adverse = 5.0, 2.0
    call = _race_counts(call_path, favourable=favourable, adverse=adverse)
    # Put signed motion is -call motion.  Its favourable threshold is J and its
    # adverse threshold is M, so the resolved state names reverse.
    put = _race_counts(-call_path, favourable=adverse, adverse=favourable)

    assert call["favourable"] == put["adverse"]
    assert call["adverse"] == put["favourable"]
    assert call["neither"] == put["neither"]


def test_race_states_obey_threshold_and_horizon_monotonicities() -> None:
    full_path = np.asarray(
        [
            [1.0, 3.0, 6.0, 8.0],
            [-1.0, -3.0, -6.0, -8.0],
            [0.0, 0.0, 0.0, 0.0],
            [-3.0, 1.0, 6.0, -8.0],
            [3.0, 1.0, -6.0, 8.0],
            [2.0, 5.0, 10.0, 12.0],
        ]
    )

    by_adverse = [
        _race_counts(full_path, favourable=5.0, adverse=threshold)
        for threshold in (2.0, 5.0, 10.0)
    ]
    assert [row["favourable"] for row in by_adverse] == sorted(
        row["favourable"] for row in by_adverse
    )
    assert [row["adverse"] for row in by_adverse] == sorted(
        (row["adverse"] for row in by_adverse), reverse=True
    )
    assert [row["neither"] for row in by_adverse] == sorted(
        row["neither"] for row in by_adverse
    )

    by_favourable = [
        _race_counts(full_path, favourable=threshold, adverse=5.0)
        for threshold in (2.0, 5.0, 10.0)
    ]
    assert [row["favourable"] for row in by_favourable] == sorted(
        (row["favourable"] for row in by_favourable), reverse=True
    )
    assert [row["adverse"] for row in by_favourable] == sorted(
        row["adverse"] for row in by_favourable
    )
    assert [row["neither"] for row in by_favourable] == sorted(
        row["neither"] for row in by_favourable
    )

    by_horizon = [
        _race_counts(full_path[:, :horizon], favourable=5.0, adverse=5.0)
        for horizon in (1, 2, 4)
    ]
    assert [row["favourable"] for row in by_horizon] == sorted(
        row["favourable"] for row in by_horizon
    )
    assert [row["adverse"] for row in by_horizon] == sorted(
        row["adverse"] for row in by_horizon
    )
    assert [row["neither"] for row in by_horizon] == sorted(
        (row["neither"] for row in by_horizon), reverse=True
    )


def test_pre_m_adverse_uses_strict_stop_touch_and_first_minute_zero() -> None:
    path = np.asarray(
        [
            [5.0, -10.0, 0.0],
            [-5.0, 5.0, 6.0],
            [-4.0, 5.0, 10.0],
            [-7.0, -8.0, 5.0],
            [0.0, -1.0, 1.0],
        ]
    )
    horizon = path.shape[1]
    gain = race.first_crossing(path, 5.0, above=True)
    adverse = race.first_crossing(path, -5.0, above=False)
    pre_m = race.pre_favourable_adverse(path, gain)
    hit = gain <= horizon
    survives = hit & (pre_m < 5.0)
    row = race._race_row(
        gain,
        adverse,
        mask=np.ones(len(path), dtype=bool),
        horizon=horizon,
    )

    assert pre_m[0] == 0.0
    assert pre_m[1] == 5.0
    assert pre_m[2] == 4.0
    assert pre_m[3] == 8.0
    assert np.isnan(pre_m[4])
    assert not survives[1]  # touching -J is killed; survival is strictly inside J
    assert int(survives.sum()) == row["favourable_first_count"]


def test_sparse_extreme_quantile_keeps_unsupported_tail_bound_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(race, "DISTRIBUTION_QUANTILES", (0.99,))
    rows = race._distribution_rows(
        np.arange(100, dtype=float),
        descriptor={"scope": "pooled"},
        metric="synthetic_tail",
        conditioning="synthetic",
        unit="points",
        total_sessions=100,
    )
    quantile = next(row for row in rows if row["statistic"] == "q99")

    assert quantile["estimate"] == pytest.approx(98.01)
    assert quantile["ci_95_low"] is not None
    assert quantile["ci_95_high"] is None
    assert quantile["status"] == "MEASURED_WITH_OPEN_CONFIDENCE_BOUND"
    assert quantile["interval_method"] == "exact_binomial_order_statistic"


def test_iv_primary_side_balances_medians_and_closest_sensitivity_pairs_strikes() -> None:
    rows = iv._session_rows(_iv_frame("2024-01-02"), "2024-01-02", date(2024, 1, 2))
    values = pd.DataFrame(rows)
    at_open = values[values["start_time_et"].eq("09:31")]

    primary = at_open[at_open["region"].eq("atm_10pt_primary")].set_index("estimand")
    assert primary.loc["call_median", "iv"] == pytest.approx(0.15)
    assert primary.loc["put_median", "iv"] == pytest.approx(0.50)
    assert primary.loc["side_balanced", "iv"] == pytest.approx(0.325)
    assert primary.loc["side_balanced", "solver_bound_excluded_count"] == 1
    assert primary.loc["side_balanced", "selected_call_nodes"] == 4
    assert primary.loc["side_balanced", "selected_put_nodes"] == 3

    closest = at_open[
        at_open["region"].eq("closest_atm_pair_sensitivity")
    ].set_index("estimand")
    assert closest.loc["call_median", "iv"] == pytest.approx(0.25)
    assert closest.loc["put_median", "iv"] == pytest.approx(0.45)
    assert closest.loc["side_balanced", "iv"] == pytest.approx(0.35)
    assert closest.loc["side_balanced", "paired_strike_count"] == 2
    assert closest.loc["side_balanced", "iv_40_point_otm_support"].startswith(
        "UNSUPPORTED"
    )


def test_iv_summaries_keep_one_vote_per_session_and_validate() -> None:
    specifications = (
        ("2025-07-29", date(2025, 7, 29), 0.000),
        ("2025-07-30", date(2025, 7, 30), 0.001),
        ("2025-08-01", date(2025, 8, 1), 0.002),
        ("2025-08-04", date(2025, 8, 4), 0.003),
    )
    session_rows: list[dict[str, object]] = []
    for session, session_date, shift in specifications:
        rows = iv._session_rows(_iv_frame(session), session, session_date)
        for row in rows:
            if row["iv"] is not None:
                row["iv"] = float(row["iv"]) + shift
        session_rows.extend(rows)
    census = iv.IVCensus(
        root=Path("/synthetic"),
        sessions=tuple(item[0] for item in specifications),
        dates=tuple(item[1] for item in specifications),
        session_values=pd.DataFrame(session_rows),
        input_manifest=pd.DataFrame({"session": [item[0] for item in specifications]}),
        strict_population=False,
    )

    levels, changes, session_values = iv.summarize_iv(census)
    quality = iv.validate_iv_tables(census, levels, changes)

    assert quality["status"] == "PASS"
    assert len(session_values) == 4 * 16 * 2 * 3
    pooled = levels[
        levels["scope"].eq("pooled")
        & levels["start_time_et"].eq("09:31")
        & levels["region"].eq("atm_10pt_primary")
        & levels["estimand"].eq("side_balanced")
        & levels["statistic"].eq("q50")
    ].iloc[0]
    assert pooled["usable_sessions"] == 4
    assert pooled["estimate"] == pytest.approx(0.3265)


def test_iv_loader_excludes_known_defect_before_opening_rows(tmp_path: Path) -> None:
    _iv_frame("2024-01-02").to_parquet(tmp_path / "2024-01-02.parquet")
    (tmp_path / "2023-06-26.parquet").write_bytes(b"known-defect-placeholder")

    census = iv.load_iv_census(
        tmp_path,
        ("2024-01-02",),
        (date(2024, 1, 2),),
        strict_population=False,
    )

    assert census.sessions == ("2024-01-02",)
    defect = census.input_manifest.loc[
        census.input_manifest["session"].eq("2023-06-26")
    ].iloc[0]
    assert defect["status"] == "EXCLUDED_KNOWN_DEFECT"


def test_iv_loader_refuses_reserved_file_before_opening_rows(tmp_path: Path) -> None:
    (tmp_path / "2026-08-06.parquet").write_bytes(b"reserved-placeholder")

    with pytest.raises(iv.UnconditionalIVError, match="confirmation-reserved"):
        iv.load_iv_census(tmp_path, (), (), strict_population=False)


def test_analyze_paths_conditions_overshoot_time_and_stop_on_correct_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(race, "START_TIMES", ("09:31",))
    monkeypatch.setattr(race, "HORIZONS_MINUTES", (3,))
    monkeypatch.setattr(race, "FAVOURABLE_THRESHOLDS_POINTS", (5.0,))
    monkeypatch.setattr(race, "ADVERSE_THRESHOLDS_POINTS", (5.0,))
    monkeypatch.setattr(race, "OVERSHOOT_THRESHOLDS_POINTS", (0.0, 2.0, 5.0, 10.0))
    monkeypatch.setattr(race, "DISTRIBUTION_QUANTILES", (0.5,))

    race_surface, overshoot, time_to_event, stop, quantiles = race.analyze_paths(
        _small_corpus()
    )
    quality_control = race.validate_path_tables(
        _small_corpus(), race_surface, overshoot, time_to_event, stop, quantiles
    )
    assert quality_control["status"] == "PASS"

    pooled_race = race_surface[
        race_surface["scope"].eq("pooled") & race_surface["direction"].eq("call")
    ].iloc[0]
    assert pooled_race["favourable_first_count"] == 2
    assert pooled_race["adverse_first_count"] == 1
    assert pooled_race["neither_count"] == 1

    pooled_overshoot = overshoot[
        overshoot["scope"].eq("pooled") & overshoot["direction"].eq("call")
    ].set_index("overshoot_threshold_points")
    assert pooled_overshoot.loc[0.0, "conditional_sessions"] == 2
    assert pooled_overshoot.loc[0.0, "at_or_above_count"] == 2
    assert pooled_overshoot.loc[2.0, "at_or_above_count"] == 2
    assert pooled_overshoot.loc[5.0, "at_or_above_count"] == 1
    assert pooled_overshoot.loc[10.0, "at_or_above_count"] == 0

    pooled_time = time_to_event[
        time_to_event["scope"].eq("pooled") & time_to_event["direction"].eq("call")
    ].set_index("elapsed_minutes")
    assert pooled_time.loc[1, "conditional_sessions"] == 3
    assert pooled_time.loc[1, "reached_by_count"] == 1
    assert pooled_time.loc[2, "reached_by_count"] == 3
    assert pooled_time.loc[3, "reached_by_count"] == 3

    pooled_stop = stop[
        stop["scope"].eq("pooled") & stop["direction"].eq("call")
    ].iloc[0]
    assert pooled_stop["favourable_reached_count"] == 3
    assert pooled_stop["survives_to_favourable_count"] == 2
    assert pooled_stop["killed_before_favourable_count"] == 1
    assert (
        pooled_stop["survives_to_favourable_count"]
        == pooled_race["favourable_first_count"]
    )

    pooled_quantiles = quantiles[
        quantiles["scope"].eq("pooled") & quantiles["direction"].eq("call")
    ]
    overshoot_median = pooled_quantiles[
        pooled_quantiles["metric"].eq("overshoot_after_favourable_first")
        & pooled_quantiles["statistic"].eq("q50")
    ].iloc[0]
    adverse_median = pooled_quantiles[
        pooled_quantiles["metric"].eq("adverse_excursion_before_favourable")
        & pooled_quantiles["statistic"].eq("q50")
    ].iloc[0]
    assert overshoot_median["conditional_sessions"] == 2
    assert overshoot_median["estimate"] == pytest.approx(3.5)
    assert adverse_median["conditional_sessions"] == 3
    assert adverse_median["estimate"] == 4.0


def test_validator_rejects_semantic_grid_interval_and_stop_population_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(race, "START_TIMES", ("09:31",))
    monkeypatch.setattr(race, "HORIZONS_MINUTES", (3,))
    monkeypatch.setattr(race, "FAVOURABLE_THRESHOLDS_POINTS", (2.0, 5.0))
    monkeypatch.setattr(race, "ADVERSE_THRESHOLDS_POINTS", (2.0, 5.0))
    monkeypatch.setattr(race, "OVERSHOOT_THRESHOLDS_POINTS", (0.0, 2.0))
    monkeypatch.setattr(race, "DISTRIBUTION_QUANTILES", (0.5,))

    corpus = _small_corpus()
    original = list(race.analyze_paths(corpus))
    assert race.validate_path_tables(corpus, *original)["status"] == "PASS"

    wrong_conditioning = [frame.copy() for frame in original]
    wrong_conditioning[2]["conditioning"] = "favourable_first"
    with pytest.raises(race.UnconditionalRaceError, match="conditioning is not literally"):
        race.validate_path_tables(corpus, *wrong_conditioning)

    elapsed_zero = [frame.copy() for frame in original]
    elapsed_zero[2].loc[elapsed_zero[2]["elapsed_minutes"].eq(1), "elapsed_minutes"] = 0
    with pytest.raises(race.UnconditionalRaceError, match="outside 1..horizon"):
        race.validate_path_tables(corpus, *elapsed_zero)

    wrong_threshold = [frame.copy() for frame in original]
    wrong_threshold[0].loc[wrong_threshold[0].index[0], "favourable_threshold_points"] = 99.0
    with pytest.raises(race.UnconditionalRaceError, match="outside the declared grid"):
        race.validate_path_tables(corpus, *wrong_threshold)

    vacuous_intervals = [frame.copy() for frame in original]
    for prefix in ("favourable_first", "adverse_first", "neither"):
        vacuous_intervals[0][f"{prefix}_ci_95_low"] = 0.0
        vacuous_intervals[0][f"{prefix}_ci_95_high"] = 1.0
    with pytest.raises(race.UnconditionalRaceError, match="Wilson interval"):
        race.validate_path_tables(corpus, *vacuous_intervals)

    extra_column = [frame.copy() for frame in original]
    extra_column[0]["undeclared"] = 1
    with pytest.raises(race.UnconditionalRaceError, match="schema drifted"):
        race.validate_path_tables(corpus, *extra_column)

    # Keep every mutated probability internally exact while changing one J's
    # eventual-hit population.  The cross-table event binding must still fail.
    wrong_stop_population = [frame.copy() for frame in original]
    stop = wrong_stop_population[3]
    target = (
        stop["scope"].eq("pooled")
        & stop["direction"].eq("call")
        & stop["favourable_threshold_points"].eq(5.0)
        & stop["adverse_threshold_points"].eq(5.0)
    )
    row_index = stop.index[target][0]
    altered_hits = int(stop.at[row_index, "total_sessions"])
    survives = int(stop.at[row_index, "survives_to_favourable_count"])
    stop.at[row_index, "conditional_sessions"] = altered_hits
    for key, value in race._probability_fields(
        "favourable_reached", altered_hits, altered_hits
    ).items():
        stop.at[row_index, key] = value
    for prefix, count in (
        ("survives_to_favourable", survives),
        ("killed_before_favourable", altered_hits - survives),
    ):
        for key, value in race._probability_fields(prefix, count, altered_hits).items():
            stop.at[row_index, key] = value
    with pytest.raises(race.UnconditionalRaceError, match="differs from event hits"):
        race.validate_path_tables(corpus, *wrong_stop_population)


def test_loader_excludes_known_defect_before_opening_its_prices(tmp_path: Path) -> None:
    _valid_tape_frame("2024-01-02").to_parquet(_tape_path(tmp_path, "2024-01-02"))
    # Invalid bytes prove the known defect is classified and hashed without a
    # Parquet read of its prices.
    _tape_path(tmp_path, "2023-06-26").write_bytes(b"known-defect-placeholder")

    corpus = race.load_tape_corpus(tmp_path, strict_population=False)

    assert corpus.sessions == ("2024-01-02",)
    defect = corpus.input_manifest.loc[
        corpus.input_manifest["session"].eq("2023-06-26")
    ].iloc[0]
    assert defect["status"] == "EXCLUDED_KNOWN_DEFECT"


def test_loader_refuses_reserved_session_before_opening_prices(tmp_path: Path) -> None:
    _tape_path(tmp_path, "2026-08-06").write_bytes(b"reserved-placeholder")

    with pytest.raises(race.UnconditionalRaceError, match="confirmation-reserved"):
        race.load_tape_corpus(tmp_path, strict_population=False)


def test_wrapper_preserves_self_hashed_failure_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attempt001"
    repo_root = Path(__file__).resolve().parents[2]
    receipt = run(
        tmp_path / "missing-tape",
        tmp_path / "missing-ladder",
        output,
        repo_root=repo_root,
    )

    assert receipt["status"] == "FAIL"
    stored = json.loads((output / "failure_receipt.json").read_text())
    claimed = stored.pop("receipt_sha256")
    assert claimed == hashlib.sha256(canonical_json(stored)).hexdigest()
    assert (output / "measure_unconditional_spx_race.py").is_file()
    assert (output / "unconditional_spx_race.py").is_file()
    assert "Traceback" in (output / "run.log").read_text()
    with pytest.raises(RaceRunError, match="refusing to overwrite"):
        run(
            tmp_path / "missing-tape",
            tmp_path / "missing-ladder",
            output,
            repo_root=repo_root,
        )
