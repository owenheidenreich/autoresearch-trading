"""Tests for Protocol101 owned raw acceptance verifier."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    AcceptanceThresholds,
    classify_session_status,
    context_causality_quality,
    context_reconstruction_quality,
    databento_symbol_from_contract_id,
    entry_ladder_sweep_quality,
    expected_decision_bounds,
    expected_decision_minutes,
    fold_placement_predicate,
    index_context_gap_quality,
    processed_quality,
    raw_entry_quote_at,
    role_policy_allows,
    sessions_between,
    verify_session,
    _raw_path_label,
)


def test_expected_decision_minutes_handles_full_and_early_close_days() -> None:
    assert expected_decision_minutes("2024-10-01") == 360
    assert expected_decision_minutes("2024-11-29") == 209
    assert expected_decision_minutes("2024-12-24") == 224


def test_expected_decision_bounds_match_calendar() -> None:
    first, last = expected_decision_bounds("2024-10-01")

    assert first.isoformat() == "2024-10-01T13:31:00+00:00"
    assert last.isoformat() == "2024-10-01T19:30:00+00:00"


def test_sessions_between_excludes_market_holidays_but_keeps_early_closes() -> None:
    assert sessions_between("2024-11-27", "2024-11-29") == [
        "2024-11-27",
        "2024-11-29",
    ]


def test_context_causality_quality_requires_one_minute_lag_and_no_open_backfill(tmp_path: Path) -> None:
    rows = []
    for minute in range(3):
        decision = f"2024-10-01T13:{31 + minute:02d}:00+00:00"
        source = f"2024-10-01T13:{30 + minute:02d}:00+00:00"
        rows.append(
            {
                "decision_time": decision,
                "source_context_time": source,
                "context_start_timestamp": "2024-10-01T13:30:00+00:00",
                "context_last_timestamp": source,
                "context_minute_rows": minute + 1,
                "context_ready": False,
            }
        )
    path = tmp_path / "2024-10-01.pkl"
    with path.open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_causality_quality("2024-10-01", tmp_path)

    assert quality["first_decision_matches_calendar"] is True
    assert quality["context_lag_exact_one_minute_share"] == 1.0
    assert quality["future_context_row_count"] == 0
    assert quality["opening_no_leading_backfill"] is True


def test_fold_placement_requires_role_processed_rows_and_acceptance() -> None:
    era_manifest = {
        "sessions": [
            {"session": "2024-10-01", "era": "pre_program_oct2024_jun2025"},
            {"session": "2026-07-01", "era": "confirmation_jun_jul2026"},
        ]
    }
    role_policy = {
        "policy": {
            "pre_program_oct2024_jun2025": {"permitted_roles": ["diagnostics_only", "train", "test"]},
            "confirmation_jun_jul2026": {"permitted_roles": ["confirmation_one_shot", "report_only"]},
        }
    }
    registry = {
        "sessions": [
            {
                "session": "2024-10-01",
                "status": "pass",
                "verifier_version": 35,
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2024-10-02",
                "status": "pass",
                "verifier_version": 35,
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2026-07-01",
                "status": "pass",
                "verifier_version": 35,
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
        ]
    }

    assert role_policy_allows(role_policy, "pre_program_oct2024_jun2025", "diagnostics_only")
    assert fold_placement_predicate(
        session="2024-10-01",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )["placeable"] is True
    missing_era = fold_placement_predicate(
        session="2024-10-02",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert missing_era["placeable"] is False
    assert missing_era["checks"]["era_permits_role"] is False
    confirmation_as_train = fold_placement_predicate(
        session="2026-07-01",
        role="train",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert confirmation_as_train["placeable"] is False
    assert confirmation_as_train["checks"]["era_permits_role"] is False


def test_fold_placement_rejects_old_verifier_records() -> None:
    era_manifest = {"sessions": [{"session": "2024-10-01", "era": "pre_program_oct2024_jun2025"}]}
    role_policy = {"policy": {"pre_program_oct2024_jun2025": {"permitted_roles": ["diagnostics_only"]}}}
    registry = {
        "sessions": [
            {
                "session": "2024-10-01",
                "status": "pass",
                "verifier_version": 2,
                "processed": {"processed_exists": True, "neural_rows": 360},
            }
        ]
    }

    result = fold_placement_predicate(
        session="2024-10-01",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )

    assert result["placeable"] is False
    assert result["checks"]["verifier_version_v3_or_newer"] is False


def test_processed_quality_handles_missing_file(tmp_path: Path) -> None:
    quality = processed_quality("2024-10-01", tmp_path)

    assert quality["processed_exists"] is False
    assert quality["neural_rows"] == 0
    assert quality["expected_decision_minutes"] == 360
    assert quality["ladder_shape_ok_share"] == 0.0
    assert quality["tradable_minute_share"] == 0.0


def test_processed_quality_exposes_all_zero_labels_as_suspicious(tmp_path: Path) -> None:
    row = {
        "decision_time": "2024-10-01T13:31:00+00:00",
        "source_context_time": "2024-10-01T13:30:00+00:00",
        "context_start_timestamp": "2024-10-01T13:30:00+00:00",
        "context_last_timestamp": "2024-10-01T13:30:00+00:00",
        "context_minute_rows": 1,
        "context_ready": False,
        "option_ladder": np.ones((21, 2, 15), dtype=float),
        "candidate_mask": np.ones((21, 2), dtype=bool),
        "strike_offsets": np.arange(-50, 55, 5),
        "labels_net_pnl": np.zeros((21, 2, 3), dtype=float),
        "feature_contract_version": "protocol101-live-v1",
    }
    with (tmp_path / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump([row], handle)

    quality = processed_quality("2024-10-01", tmp_path)

    assert quality["label_finite_share"] == 1.0
    assert quality["label_nonzero_share"] == 0.0
    assert quality["label_positive_share"] == 0.0
    assert quality["label_negative_share"] == 0.0


def test_processed_quality_label_metrics_are_candidate_mask_aware(tmp_path: Path) -> None:
    empty_row = {
        "decision_time": "2024-10-01T13:31:00+00:00",
        "source_context_time": "2024-10-01T13:30:00+00:00",
        "context_start_timestamp": "2024-10-01T13:30:00+00:00",
        "context_last_timestamp": "2024-10-01T13:30:00+00:00",
        "context_minute_rows": 1,
        "context_ready": False,
        "option_ladder": np.full((21, 2, 15), np.nan, dtype=float),
        "candidate_mask": np.zeros((21, 2), dtype=bool),
        "strike_offsets": np.arange(-50, 55, 5),
        "labels_net_pnl": np.full((21, 2, 3), np.nan, dtype=float),
        "feature_contract_version": "protocol101-live-v1",
        "decision_grid": "calendar_v2",
    }
    candidate_row = {
        **empty_row,
        "decision_time": "2024-10-01T13:32:00+00:00",
        "source_context_time": "2024-10-01T13:31:00+00:00",
        "option_ladder": np.ones((21, 2, 15), dtype=float),
        "candidate_mask": np.zeros((21, 2), dtype=bool),
        "labels_net_pnl": np.full((21, 2, 3), np.nan, dtype=float),
    }
    candidate_row["candidate_mask"][10, 0] = True
    candidate_row["labels_net_pnl"][10, 0, :] = [10.0, -5.0, 2.0]
    with (tmp_path / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump([empty_row, candidate_row], handle)

    quality = processed_quality("2024-10-01", tmp_path)

    assert quality["label_total_cell_count"] == 252
    assert quality["label_candidate_cell_count"] == 3
    assert quality["label_finite_share"] == 1.0
    assert quality["label_nonzero_share"] == 1.0
    assert quality["decision_grid"] == "calendar_v2"


def test_verify_session_rejects_zero_labels_and_empty_candidate_population(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    row = {
        "decision_time": "2024-10-01T13:31:00+00:00",
        "source_context_time": "2024-10-01T13:30:00+00:00",
        "context_start_timestamp": "2024-10-01T13:30:00+00:00",
        "context_last_timestamp": "2024-10-01T13:30:00+00:00",
        "context_minute_rows": 1,
        "context_ready": False,
        "option_ladder": np.ones((21, 2, 15), dtype=float),
        "candidate_mask": np.zeros((21, 2), dtype=bool),
        "strike_offsets": np.arange(-50, 55, 5),
        "labels_net_pnl": np.zeros((21, 2, 3), dtype=float),
        "feature_contract_version": "protocol101-live-v1",
    }
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump([row], handle)

    record = verify_session(
        session="2024-10-01",
        raw_root=tmp_path / "raw",
        spx_dir=tmp_path / "spx",
        vix_dir=tmp_path / "vix",
        processed_dir=processed_dir,
        thresholds=AcceptanceThresholds(min_label_spot_check_count=1),
    )

    assert record["status"] == "fail"
    assert record["checks"]["mean_tradable_candidates"] is False
    assert record["checks"]["near_atm_tradable_share"] is False
    assert record["checks"]["label_nonzero_share"] is False


def test_verify_session_marks_early_close_sessions_report_only(tmp_path: Path) -> None:
    record = verify_session(
        session="2024-12-24",
        raw_root=tmp_path / "raw",
        spx_dir=tmp_path / "spx",
        vix_dir=tmp_path / "vix",
        processed_dir=tmp_path / "processed",
        thresholds=AcceptanceThresholds(min_label_spot_check_count=1),
    )

    assert record["status"] == "report_only"
    assert record["early_close_session"] is True
    assert record["report_only_reason"] == "early_close_not_close_aware"
    assert record["checks"]["processed_rows_exist"] is False


def test_context_causality_quality_rejects_zero_lag_context(tmp_path: Path) -> None:
    rows = [
        {
            "decision_time": "2024-10-01T13:31:00+00:00",
            "source_context_time": "2024-10-01T13:31:00+00:00",
            "context_start_timestamp": "2024-10-01T13:31:00+00:00",
            "context_last_timestamp": "2024-10-01T13:31:00+00:00",
            "context_minute_rows": 1,
            "context_ready": False,
        }
    ]
    with (tmp_path / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_causality_quality("2024-10-01", tmp_path)

    assert quality["context_lag_exact_one_minute_share"] == 0.0
    assert quality["future_context_row_count"] > 0
    assert quality["opening_no_leading_backfill"] is False


def test_databento_symbol_from_contract_id_matches_raw_symbol_format() -> None:
    assert databento_symbol_from_contract_id("SPXW-20241001-05700.000-C") == "SPXW  241001C05700000"


def test_raw_entry_quote_at_uses_latest_non_stale_quote() -> None:
    frame = pd.DataFrame(
        {
            "quote_time": pd.to_datetime(
                [
                    "2024-10-01T13:30:00Z",
                    "2024-10-01T13:31:00Z",
                    "2024-10-01T13:32:00Z",
                ],
                utc=True,
            ),
            "bid": [1.0, 1.1, 9.9],
            "ask": [1.2, 1.3, 9.9],
            "mid": [1.1, 1.2, 9.9],
        }
    )

    quote = raw_entry_quote_at(
        frame,
        decision_time=pd.Timestamp("2024-10-01T13:31:30Z"),
        max_quote_age_seconds=90.0,
    )
    stale = raw_entry_quote_at(
        frame.iloc[:1],
        decision_time=pd.Timestamp("2024-10-01T13:32:00Z"),
        max_quote_age_seconds=90.0,
    )

    assert quote["status"] == "ok"
    assert quote["ask"] == 1.3
    assert quote["quote_time"] == "2024-10-01T13:31:00+00:00"
    assert stale["status"] == "stale"


def test_raw_path_label_covers_stop_target_time_forced_and_missing_branches() -> None:
    def quotes(times: list[str], bids: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quote_time": pd.to_datetime(times, utc=True),
                "bid": bids,
            }
        )

    decision = pd.Timestamp("2024-10-01T14:00:00Z")

    stop_pnl, stop_reason = _raw_path_label(
        quotes(["2024-10-01T14:05:00Z"], [4.9]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=1,
    )
    target_pnl, target_reason = _raw_path_label(
        quotes(["2024-10-01T14:05:00Z"], [20.1]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=1,
    )
    time_pnl, time_reason = _raw_path_label(
        quotes(["2024-10-01T14:25:00Z"], [11.0]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=1,
    )
    missing_pnl, missing_reason = _raw_path_label(
        quotes(["2024-10-01T13:59:00Z"], [11.0]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=1,
    )
    forced_pnl, forced_reason = _raw_path_label(
        quotes(["2024-10-01T19:55:00Z"], [11.0]),
        decision_time=pd.Timestamp("2024-10-01T19:40:00Z"),
        entry_ask=10.0,
        policy_idx=1,
    )
    boundary_pnl, boundary_reason = _raw_path_label(
        quotes(["2024-10-01T19:55:00Z"], [11.0]),
        decision_time=pd.Timestamp("2024-10-01T19:30:00Z"),
        entry_ask=10.0,
        policy_idx=1,
    )

    assert stop_reason == "stop_hit"
    assert np.isclose(stop_pnl, -510.0)
    assert target_reason == "target_hit"
    assert np.isclose(target_pnl, 1010.0)
    assert time_reason == "time_exit"
    assert np.isclose(time_pnl, 100.0)
    assert missing_reason == "missing_future_path"
    assert np.isnan(missing_pnl)
    assert forced_reason == "forced_flat_capped"
    assert np.isclose(forced_pnl, 100.0)
    assert boundary_reason == "time_exit"
    assert np.isclose(boundary_pnl, 100.0)


def test_entry_ladder_sweep_checks_all_masked_candidates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "databento" / "opra_spxw_cbbo_1m"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir()
    symbol = "SPXW  241001C05750000"
    frame = pd.DataFrame(
        {
            "bid_px_00": [1.1],
            "ask_px_00": [1.3],
            "bid_sz_00": [10],
            "ask_sz_00": [11],
            "symbol": [symbol],
        },
        index=pd.DatetimeIndex(["2024-10-01T13:31:00Z"], name="ts_recv"),
    )
    frame.to_parquet(raw_dir / "2024-10-01.cbbo-1m.parquet")
    ladder = np.full((21, 2, 15), np.nan, dtype=float)
    ladder[10, 0, 0] = 1.1
    ladder[10, 0, 1] = 1.3
    ladder[10, 0, 2] = 1.2
    mask = np.zeros((21, 2), dtype=bool)
    mask[10, 0] = True
    contract_ids = np.full((21, 2), None, dtype=object)
    contract_ids[10, 0] = "SPXW-20241001-05750.000-C"
    rows = [
        {
            "decision_time": "2024-10-01T13:31:00+00:00",
            "candidate_mask": mask,
            "contract_ids": contract_ids,
            "option_ladder": ladder,
            "contract_quote_metadata": {
                "SPXW-20241001-05750.000-C": {
                    "source_quote_time": "2024-10-01T13:31:00+00:00",
                    "bid": 1.1,
                    "ask": 1.3,
                    "mid": 1.2,
                }
            },
        }
    ]
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)

    quality = entry_ladder_sweep_quality(
        "2024-10-01",
        raw_root=tmp_path / "raw",
        processed_dir=processed_dir,
    )

    assert quality["masked_candidate_count"] == 1
    assert quality["entry_quote_match_share"] == 1.0
    assert quality["ladder_quote_match_share"] == 1.0


def test_entry_ladder_sweep_detects_ladder_quote_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "databento" / "opra_spxw_cbbo_1m"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "bid_px_00": [1.1],
            "ask_px_00": [1.3],
            "symbol": ["SPXW  241001C05750000"],
        },
        index=pd.DatetimeIndex(["2024-10-01T13:31:00Z"], name="ts_recv"),
    )
    frame.to_parquet(raw_dir / "2024-10-01.cbbo-1m.parquet")
    ladder = np.full((21, 2, 15), np.nan, dtype=float)
    ladder[10, 0, 0] = 1.1
    ladder[10, 0, 1] = 9.9
    ladder[10, 0, 2] = 1.2
    mask = np.zeros((21, 2), dtype=bool)
    mask[10, 0] = True
    contract_ids = np.full((21, 2), None, dtype=object)
    contract_ids[10, 0] = "SPXW-20241001-05750.000-C"
    rows = [
        {
            "decision_time": "2024-10-01T13:31:00+00:00",
            "candidate_mask": mask,
            "contract_ids": contract_ids,
            "option_ladder": ladder,
            "contract_quote_metadata": {
                "SPXW-20241001-05750.000-C": {
                    "source_quote_time": "2024-10-01T13:31:00+00:00",
                    "bid": 1.1,
                    "ask": 1.3,
                    "mid": 1.2,
                }
            },
        }
    ]
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)

    quality = entry_ladder_sweep_quality(
        "2024-10-01",
        raw_root=tmp_path / "raw",
        processed_dir=processed_dir,
    )

    assert quality["entry_quote_match_share"] == 1.0
    assert quality["ladder_quote_mismatch_count"] == 1


def test_context_reconstruction_recomputes_market_features_and_atm(tmp_path: Path) -> None:
    spx_dir = tmp_path / "spx"
    vix_dir = tmp_path / "vix"
    processed_dir = tmp_path / "processed"
    spx_dir.mkdir()
    vix_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-10-01T13:30:00Z"], utc=True),
            "symbol": ["SPX"],
            "close": [5752.33],
            "volume": [0],
        }
    ).to_parquet(spx_dir / "2024-10-01.parquet")
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-10-01T13:30:00Z"], utc=True),
            "symbol": ["VIX"],
            "close": [16.99],
            "volume": [0],
        }
    ).to_parquet(vix_dir / "2024-10-01.parquet")
    window = np.full((30, 7), np.nan, dtype=float)
    window[-1] = np.asarray([5752.33, 16.99, 5752.33, 0.0, 0.0, 0.0, 0.0])
    rows = [
        {
            "decision_time": "2024-10-01T13:31:00+00:00",
            "atm_strike": 5750,
            "market_window": window,
        }
    ]
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_reconstruction_quality(
        "2024-10-01",
        spx_dir=spx_dir,
        vix_dir=vix_dir,
        processed_dir=processed_dir,
    )

    assert quality["feature_match_share"] == 1.0
    assert quality["atm_strike_match_share"] == 1.0
    assert quality["window_sample_match_share"] == 1.0
    assert quality["vix_close_finite_share"] == 1.0


def test_context_reconstruction_detects_feature_and_atm_mismatch(tmp_path: Path) -> None:
    spx_dir = tmp_path / "spx"
    vix_dir = tmp_path / "vix"
    processed_dir = tmp_path / "processed"
    spx_dir.mkdir()
    vix_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-10-01T13:30:00Z"], utc=True),
            "symbol": ["SPX"],
            "close": [5752.33],
            "volume": [0],
        }
    ).to_parquet(spx_dir / "2024-10-01.parquet")
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-10-01T13:30:00Z"], utc=True),
            "symbol": ["VIX"],
            "close": [16.99],
            "volume": [0],
        }
    ).to_parquet(vix_dir / "2024-10-01.parquet")
    rows = [
        {
            "decision_time": "2024-10-01T13:31:00+00:00",
            "atm_strike": 5700,
            "market_window": np.zeros((30, 7), dtype=float),
        }
    ]
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_reconstruction_quality(
        "2024-10-01",
        spx_dir=spx_dir,
        vix_dir=vix_dir,
        processed_dir=processed_dir,
    )

    assert quality["feature_mismatch_count"] == 1
    assert quality["atm_strike_mismatch_count"] == 1
    assert quality["window_sample_mismatch_count"] == 1


def test_threshold_defaults_are_data_plane_only() -> None:
    thresholds = AcceptanceThresholds()

    assert thresholds.min_ladder_shape_ok_share == 1.00
    assert thresholds.min_tradable_minute_share == 0.50
    assert thresholds.min_mean_tradable_candidates == 10.0
    assert thresholds.min_label_finite_share == 0.95
    assert thresholds.min_label_nonzero_share == 0.95
    assert thresholds.min_label_positive_share == 0.15
    assert thresholds.min_label_negative_share == 0.40
    assert thresholds.min_entry_quote_match_share == 1.0
    assert thresholds.min_entry_quote_sweep_match_share == 1.0
    assert thresholds.min_context_reconstruction_match_share == 1.0
    assert thresholds.max_missing_processed_rows == 0


def _write_spx_file(spx_dir: Path, minutes: list[str]) -> None:
    spx_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(minutes, utc=True),
            "symbol": ["SPX"] * len(minutes),
            "close": [5000.0 + i for i in range(len(minutes))],
            "volume": [0] * len(minutes),
        }
    ).to_parquet(spx_dir / "2024-10-01.parquet")


def _write_lag_rows(processed_dir: Path, rows_spec: list[tuple[str, str]]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"decision_time": decision, "source_context_time": source}
        for decision, source in rows_spec
    ]
    with (processed_dir / "2024-10-01.pkl").open("wb") as handle:
        pickle.dump(rows, handle)


def test_index_context_gap_attributes_stale_rows_to_vendor_gap(tmp_path: Path) -> None:
    spx_dir = tmp_path / "spx"
    processed_dir = tmp_path / "processed"
    # Vendor file covers the full expected source grid EXCEPT 13:32 UTC.
    first, last = expected_decision_bounds("2024-10-01")
    grid = pd.date_range(
        start=pd.Timestamp(first) - pd.Timedelta(minutes=1),
        end=pd.Timestamp(last) - pd.Timedelta(minutes=1),
        freq="min",
        tz="UTC",
    )
    minutes = [ts.isoformat() for ts in grid if ts != pd.Timestamp("2024-10-01T13:32:00+00:00")]
    _write_spx_file(spx_dir, minutes)
    # Decision 13:33 could not see a 13:32 bar and fell back to 13:31 (stale, past).
    _write_lag_rows(
        processed_dir,
        [
            ("2024-10-01T13:31:00+00:00", "2024-10-01T13:30:00+00:00"),
            ("2024-10-01T13:33:00+00:00", "2024-10-01T13:31:00+00:00"),
        ],
    )

    gap = index_context_gap_quality("2024-10-01", spx_dir=spx_dir, processed_dir=processed_dir)

    assert gap["vendor_missing_index_minutes"] == ["2024-10-01T13:32:00+00:00"]
    assert gap["imperfect_lag_row_count"] == 1
    assert gap["attributed"] is True


def test_index_context_gap_rejects_unattributed_and_oversized_gaps(tmp_path: Path) -> None:
    spx_dir = tmp_path / "spx"
    processed_dir = tmp_path / "processed"
    first, last = expected_decision_bounds("2024-10-01")
    grid = pd.date_range(
        start=pd.Timestamp(first) - pd.Timedelta(minutes=1),
        end=pd.Timestamp(last) - pd.Timedelta(minutes=1),
        freq="min",
        tz="UTC",
    )
    # Case 1: vendor complete, but a row has a stale lag anyway (builder bug).
    _write_spx_file(spx_dir, [ts.isoformat() for ts in grid])
    _write_lag_rows(
        processed_dir,
        [("2024-10-01T13:33:00+00:00", "2024-10-01T13:31:00+00:00")],
    )
    unattributed = index_context_gap_quality("2024-10-01", spx_dir=spx_dir, processed_dir=processed_dir)
    assert unattributed["attributed"] is False

    # Case 2: vendor gap too large (six missing minutes) even though rows map to it.
    missing = set(list(grid[10:16]))
    _write_spx_file(spx_dir, [ts.isoformat() for ts in grid if ts not in missing])
    _write_lag_rows(
        processed_dir,
        [
            (
                (ts + pd.Timedelta(minutes=1)).isoformat(),
                (ts - pd.Timedelta(minutes=1)).isoformat(),
            )
            for ts in sorted(missing)
        ],
    )
    oversized = index_context_gap_quality("2024-10-01", spx_dir=spx_dir, processed_dir=processed_dir)
    assert oversized["vendor_missing_index_minute_count"] == 6
    assert oversized["attributed"] is False

    # Case 3: future-context row can never be attributed.
    _write_spx_file(
        spx_dir,
        [ts.isoformat() for ts in grid if ts != pd.Timestamp("2024-10-01T13:32:00+00:00")],
    )
    _write_lag_rows(
        processed_dir,
        [("2024-10-01T13:33:00+00:00", "2024-10-01T13:34:00+00:00")],
    )
    future = index_context_gap_quality("2024-10-01", spx_dir=spx_dir, processed_dir=processed_dir)
    assert future["attributed"] is False


def test_classify_session_status_missing_index_context_paths() -> None:
    base_checks = {"context_lag_exact_one_minute": True, "label_nonzero_share": True, "mean_tradable_candidates": True}
    attributed = {"attributed": True}
    unattributed = {"attributed": False}

    lag_only = dict(base_checks, context_lag_exact_one_minute=False)
    assert classify_session_status(
        checks=lag_only, early_close_session=False, index_context_gap=attributed
    ) == ("report_only", "missing_index_context")
    assert classify_session_status(
        checks=lag_only, early_close_session=False, index_context_gap=unattributed
    ) == ("fail", "")

    lag_plus_liquidity = dict(lag_only, mean_tradable_candidates=False)
    assert classify_session_status(
        checks=lag_plus_liquidity, early_close_session=False, index_context_gap=attributed
    ) == ("report_only", "missing_index_context")

    lag_plus_label_failure = dict(lag_only, label_nonzero_share=False)
    assert classify_session_status(
        checks=lag_plus_label_failure, early_close_session=False, index_context_gap=attributed
    ) == ("fail", "")

    liquidity_only = dict(base_checks, mean_tradable_candidates=False)
    assert classify_session_status(
        checks=liquidity_only, early_close_session=False, index_context_gap=unattributed
    ) == ("report_only", "low_tradable_liquidity")

    all_pass = dict(base_checks)
    assert classify_session_status(
        checks=all_pass, early_close_session=False, index_context_gap=unattributed
    ) == ("pass", "")
    assert classify_session_status(
        checks=all_pass, early_close_session=True, index_context_gap=unattributed
    ) == ("report_only", "early_close_not_close_aware")


def test_raw_path_label_treats_absent_bid_as_zero_exit() -> None:
    def quotes(times: list[str], bids: list[float | None]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quote_time": pd.to_datetime(times, utc=True),
                "bid": bids,
            }
        )

    decision = pd.Timestamp("2024-10-01T14:00:00Z")

    # Absent bid mid-path triggers the stop at an executable 0.00 for a
    # stopped policy (policy 1, stop 50%).
    stopped_pnl, stopped_reason = _raw_path_label(
        quotes(["2024-10-01T14:05:00Z", "2024-10-01T14:10:00Z"], [None, 8.0]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=1,
    )
    # Absent bid at the final row exits at 0.00 (full premium loss) for a
    # premium-is-the-stop policy (policy 6, stop 100%).
    worthless_pnl, worthless_reason = _raw_path_label(
        quotes(["2024-10-01T14:05:00Z"], [None]),
        decision_time=decision,
        entry_ask=10.0,
        policy_idx=6,
    )

    assert stopped_reason == "stop_hit"
    assert np.isclose(stopped_pnl, -1000.0)
    assert worthless_reason == "stop_hit"
    assert np.isclose(worthless_pnl, -1000.0)
