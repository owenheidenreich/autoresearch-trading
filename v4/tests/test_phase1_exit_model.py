from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.research.phase1_exit_model import (
    EXIT_FEATURE_NAMES,
    ExitModelContractError,
    OOFEntryReceiptV1,
    build_sensitivity_labels_from_features,
    build_trajectory_tables,
    catastrophic_floor_triggered,
    expanding_outer_folds,
    fit_hgb_baseline,
    load_completed_spx_context,
    stable_hash,
    validate_oof_entry_receipt,
)


def receipt() -> OOFEntryReceiptV1:
    fill = pd.Timestamp("2025-08-01 19:53:40", tz="UTC").value
    return OOFEntryReceiptV1.seal(
        session="2025-08-01",
        trajectory_id="trajectory-1",
        raw_symbol="SPXW  250801C06275000",
        instrument_id=123,
        expiry="2025-08-01",
        strike=6275.0,
        right="C",
        decision_time_ns=fill - 1_000_000_000,
        arrival_time_ns=fill,
        fill_time_ns=fill,
        fill_price=4.0,
        quantity=1,
        entry_fee_dollars=1.50,
        realized_session_pnl_dollars=-25.0,
        remaining_risk_budget_dollars=475.0,
        outer_fold=2,
        entry_artifact_role="OUTER_FOLD_OOF",
        entry_artifact_sha256="a" * 64,
        fill_law_sha256="b" * 64,
    )


def cbbo() -> pd.DataFrame:
    times = pd.date_range("2025-08-01 19:53:40", "2025-08-01 19:55:00", freq="1s", tz="UTC")
    bid = 4.0 + np.sin(np.arange(len(times)) / 10.0) * 0.2
    return pd.DataFrame(
        {
            "ts_recv": times,
            "ts_event": times,
            "instrument_id": 123,
            "symbol": "SPXW  250801C06275000",
            "bid_px_00": bid,
            "ask_px_00": bid + 0.10,
            "bid_sz_00": 10,
            "ask_sz_00": 12,
        }
    )


def spx() -> pd.DataFrame:
    events = pd.date_range("2025-08-01 19:50:00", periods=4, freq="1min", tz="UTC")
    close = pd.Series([6273.0, 6274.0, 6275.0, 6276.0])
    frame = pd.DataFrame({"event_time": events, "close": close})
    frame["available_at"] = frame["event_time"] + pd.Timedelta(seconds=60)
    frame["log_return_1m"] = np.log(close / close.shift(1))
    frame["log_return_5m"] = np.nan
    frame["log_return_15m"] = np.nan
    frame["vwap_gap_bps"] = 0.0
    return frame


def official_spx_history(context_source: str) -> pd.DataFrame:
    events = pd.date_range("2025-08-01 13:30:00", periods=20, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "event_time": events,
            "symbol": "SPX",
            "open": 6275.0,
            "high": 6276.0,
            "low": 6274.0,
            "close": np.linspace(6275.0, 6280.0, len(events)),
            "volume": 0,
            "context_source": context_source,
            "is_derived": False,
            "is_proxy": False,
            "is_official_index_data": True,
        }
    )


def test_oof_receipt_rejects_full_fit_and_tamper() -> None:
    valid = receipt()
    validate_oof_entry_receipt(valid)
    full_fit = asdict(valid)
    full_fit.pop("receipt_sha256")
    full_fit["entry_artifact_role"] = "FULL_FIT"
    with pytest.raises(ExitModelContractError, match="non-OOF"):
        OOFEntryReceiptV1.seal(**full_fit)
    with pytest.raises(ExitModelContractError, match="hash drift"):
        validate_oof_entry_receipt(replace(valid, fill_price=5.0))


def test_features_are_causal_and_labels_are_separate() -> None:
    features, labels = build_trajectory_tables(receipt(), cbbo(), spx())
    assert tuple(features.loc[:, EXIT_FEATURE_NAMES].columns) == EXIT_FEATURE_NAMES
    assert "a_ref_dollars" not in features
    assert "a_ref_dollars" in labels
    assert len(features) == len(labels) == 81
    assert features[["session", "trajectory_id", "decision_time_ns"]].equals(
        labels[["session", "trajectory_id", "decision_time_ns"]]
    )

    mutated = cbbo()
    mutated.loc[mutated.index[-1], ["bid_px_00", "ask_px_00"]] = [0.0, 0.0]
    changed_features, changed_labels = build_trajectory_tables(receipt(), mutated, spx())
    pd.testing.assert_frame_equal(features.iloc[:40], changed_features.iloc[:40], check_exact=True)
    assert not labels["a_ref_dollars"].equals(changed_labels["a_ref_dollars"])


@pytest.mark.parametrize(
    "context_source",
    (
        "thetadata_index_history_ohlc",
        "/Users/example/.autoresearch-trading/pathd/vendor/thetadata/index/spx_1m/2025-08-01.parquet",
    ),
)
def test_completed_spx_accepts_authenticated_official_sources(
    tmp_path: Path, context_source: str
) -> None:
    path = tmp_path / "official_spx.parquet"
    official_spx_history(context_source).to_parquet(path, index=False)
    loaded = load_completed_spx_context(path, emission_lag_ms=2_336)
    assert loaded["context_source"].eq("thetadata_index_history_ohlc").all()
    assert loaded["available_at"].equals(
        loaded["event_time"] + pd.Timedelta(seconds=62, milliseconds=336)
    )


def test_completed_spx_rejects_unapproved_source_path(tmp_path: Path) -> None:
    path = tmp_path / "unapproved_spx.parquet"
    official_spx_history("/vendor/other/index/spx_1m/2025-08-01.parquet").to_parquet(
        path, index=False
    )
    with pytest.raises(ExitModelContractError, match="not official ThetaData"):
        load_completed_spx_context(path, emission_lag_ms=2_336)


def test_fill_law_sensitivities_and_floor() -> None:
    baseline = build_trajectory_tables(receipt(), cbbo(), spx(), latency_seconds=1)[1]
    delayed = build_trajectory_tables(receipt(), cbbo(), spx(), latency_seconds=5)[1]
    stress = build_trajectory_tables(receipt(), cbbo(), spx(), fee_per_side_dollars=2.0)[1]
    assert not baseline["exit_until_filled_value_dollars"].equals(
        delayed["exit_until_filled_value_dollars"]
    )
    assert np.allclose(
        baseline["exit_until_filled_value_dollars"] - stress["exit_until_filled_value_dollars"],
        0.5,
    )
    assert catastrophic_floor_triggered(entry_price=4.0, current_bid=1.9)
    assert not catastrophic_floor_triggered(entry_price=4.0, current_bid=2.1)


@pytest.mark.parametrize(
    ("fee", "latency", "expected_hash"),
    (
        (1.5, 0, "173fb9ee39dec3bf2a7c83f67bcbdddbd9e57d9f0f7e49d2ee12867b6f290bfd"),
        (1.5, 1, "8284ed3582281f45c1da7539e35c68b615d5a88ca9bf05dc01a961c54d1eaf49"),
        (1.5, 2, "929e5a8de3a6f48b8c425bfcb6fb70728228804e2a5a41f110c4e4e62ced68df"),
        (1.5, 5, "4ac1ef7c44caf86bfe9137b9214255812ce34f5ecc46c8d3b0d322e24e9e2cf0"),
        (2.0, 0, "4b8cfdb7120b953a58b6d0d2e1bb6854857453cb1cd0c8a4b3c0b8a5dea28878"),
        (2.0, 1, "3cc081ad4a1d7d6c92f087b67178d7fb709644170e1f9d1377abb4b8f2356241"),
        (2.0, 2, "7752deba7b83540c2310b5322d1162a44ffb1b8accea3ffc81dfa527139347f5"),
        (2.0, 5, "5719c5de0a66a8d70d49db2e92ac9ba5bf7c584399b6fe26555ddee10d664bf9"),
    ),
)
def test_sensitivity_repricing_preserves_frozen_label_hashes(
    fee: float, latency: int, expected_hash: str
) -> None:
    features, _ = build_trajectory_tables(receipt(), cbbo(), spx())
    labels = build_sensitivity_labels_from_features(
        receipt(),
        features,
        fee_per_side_dollars=fee,
        latency_seconds=latency,
    )
    assert stable_hash(labels.to_dict(orient="records")) == expected_hash


def test_five_expanding_folds_have_one_session_embargo() -> None:
    sessions = pd.bdate_range("2025-01-02", periods=215).strftime("%Y-%m-%d").tolist()
    folds = expanding_outer_folds(sessions)
    assert len(folds) == 5
    for fold in folds:
        assert len(fold["embargo"]) == 1
        assert set(fold["train"]).isdisjoint(fold["test"])
        assert max(fold["train"]) < fold["embargo"][0] < min(fold["test"])


def test_hgb_baseline_emits_frozen_action_geometry() -> None:
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(
        rng.normal(size=(160, len(EXIT_FEATURE_NAMES))), columns=EXIT_FEATURE_NAMES
    )
    target = pd.Series(matrix.iloc[:, 0] * 2.0 - matrix.iloc[:, 1])
    model = fit_hgb_baseline(matrix.iloc[:120], target.iloc[:120], matrix.iloc[120:], target.iloc[120:])
    prediction = model.predict(matrix.iloc[120:])
    assert set(prediction["action"]) <= {"HOLD", "EXIT"}
    assert (prediction["q10"] <= prediction["q50"]).all()
    assert (prediction["q50"] <= prediction["q90"]).all()
    np.testing.assert_allclose(
        prediction["utility_hold"],
        prediction["mean_lcb90"] + 0.25 * np.minimum(prediction["q10"], 0.0),
    )
