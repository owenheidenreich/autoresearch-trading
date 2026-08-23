"""Structural tests for the outcome-blind signed-flow characterization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.ops.analyze_cmbp_signed_flow import (
    BOOTSTRAP_REPS,
    EXPECTED_MANIFEST_SHA256,
    READ_COLUMNS,
    FlowCharacterizationError,
    SessionBootstrap,
    _self_hash,
    _subset_metrics,
    cli,
    signed_events,
    verify_manifest,
)


SLICE = Path("data/raw/audit/protocol101_highres_opra/cmbp-1")
MANIFEST = Path("v5/work/lifecycle-training/CMBP_SEMANTIC_GATE_MANIFEST_2026_08_23.json")
needs_slice = pytest.mark.skipif(not SLICE.is_dir(), reason="owned cmbp-1 slice not present")


def _frame(rows: list[tuple]) -> pd.DataFrame:
    columns = [
        "ts_recv", "ts_event", "instrument_id", "action", "price", "size",
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "symbol", "publisher_id",
    ]
    return pd.DataFrame(rows, columns=columns)


def _row(
    recv: str,
    event: str,
    instrument: int,
    action: str,
    price: float,
    *,
    symbol: str,
    bid: float = 1.0,
    ask: float = 1.2,
    size: int = 1,
    bid_size: int = 3,
    ask_size: int = 4,
    publisher: int = 30,
) -> tuple:
    return (
        pd.Timestamp(recv, tz="UTC"), pd.Timestamp(event, tz="UTC"), instrument,
        action, price, size, bid, ask, bid_size, ask_size, symbol, publisher,
    )


def test_call_buy_and_put_sell_are_bullish_under_the_pinned_prior_row_law() -> None:
    f = _frame([
        _row("2024-10-01 13:30:00", "2024-10-01 13:30:00", 1, "A", np.nan,
             symbol="SPXW  241001C05700000"),
        _row("2024-10-01 13:30:01", "2024-10-01 13:30:01", 1, "T", 1.2,
             symbol="SPXW  241001C05700000"),
        _row("2024-10-01 13:30:00", "2024-10-01 13:30:00", 2, "A", np.nan,
             symbol="SPXW  241001P05700000"),
        _row("2024-10-01 13:30:01", "2024-10-01 13:30:01", 2, "T", 1.0,
             symbol="SPXW  241001P05700000"),
    ])
    events, diagnostic = signed_events(f, "2024-10-01")
    assert diagnostic["pinned_signed"] == 2
    assert events["execution_sign"].tolist() == [1, -1]
    assert events["directional_sign"].tolist() == [1, 1]
    assert events["strict_earlier_receive"].all()


def test_a_tied_clock_prior_row_is_preserved_but_explicitly_not_strict_earlier() -> None:
    timestamp = "2024-10-01 13:30:01"
    f = _frame([
        _row(timestamp, timestamp, 1, "A", np.nan, symbol="SPXW  241001C05700000"),
        _row(timestamp, timestamp, 1, "T", 1.2, symbol="SPXW  241001C05700000",
             publisher=22),
    ])
    events, diagnostic = signed_events(f, "2024-10-01")
    assert len(events) == 1
    assert events["tied_receive"].iloc[0]
    assert events["tied_event"].iloc[0]
    assert not events["strict_earlier_receive"].iloc[0]
    assert diagnostic["tied_receive_cross_publisher"] == 1


def test_inside_prints_are_not_silently_added_to_flow() -> None:
    f = _frame([
        _row("2024-10-01 13:30:00", "2024-10-01 13:30:00", 1, "A", np.nan,
             symbol="SPXW  241001C05700000"),
        _row("2024-10-01 13:30:01", "2024-10-01 13:30:01", 1, "T", 1.1,
             symbol="SPXW  241001C05700000"),
    ])
    events, diagnostic = signed_events(f, "2024-10-01")
    assert events.empty
    assert diagnostic["all_trades"] == 1


def test_session_bootstrap_is_deterministic_and_never_uses_trade_count_as_n() -> None:
    first = SessionBootstrap(4, seed=7, reps=200)
    second = SessionBootstrap(4, seed=7, reps=200)
    got = first.summarize([0.0, 0.0, 1.0, 1.0], unit="fraction", estimand="mean")
    again = second.summarize([0.0, 0.0, 1.0, 1.0], unit="fraction", estimand="mean")
    assert got == again
    assert got["estimate"] == 0.5 and got["n_sessions"] == 4
    assert "whole-session" in got["uncertainty"]


def test_clustering_metrics_separate_exact_clock_from_positive_gap_persistence() -> None:
    times = pd.to_datetime([
        "2024-10-01 13:30:00+00:00", "2024-10-01 13:30:00+00:00",
        "2024-10-01 13:30:01+00:00", "2024-10-01 13:30:02+00:00",
    ])
    e = pd.DataFrame({
        "ts_recv": times,
        "instrument_id": [1, 1, 1, 2],
        "directional_sign": [1, 1, 1, 1],
        "execution_sign": [1, 1, 1, 1],
        "size": [1, 1, 1, 1],
        "premium_notional": [100.0] * 4,
        "prior_relative_spread_bps": [10.0] * 4,
        "prior_total_depth": [4.0] * 4,
        "same_side_depth": [2.0] * 4,
        "directional_prior_depth_imbalance": [0.0] * 4,
    })
    got = _subset_metrics(e, 570, 571)
    assert got["zero_interarrival_share"] == pytest.approx(1 / 3)
    assert got["same_direction_positive_gap_probability"] == 1.0
    assert got["same_direction_cross_instrument_probability"] == 1.0


def test_failure_path_writes_once_and_then_refuses_overwrite(tmp_path: Path) -> None:
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    argv = [
        "--manifest", str(tmp_path / "missing.json"),
        "--out", str(success),
        "--failure-out", str(failure),
        "--repo-root", str(tmp_path),
    ]
    assert cli(argv) == 5
    payload = json.loads(failure.read_text())
    assert payload["verdict"] == "FAIL_WRAPPER_OR_INPUT_INVARIANT"
    assert payload["reads_no_outcome"] is True and payload["fits_run"] == 0
    assert payload["receipt_sha256"] == _self_hash(payload, "receipt_sha256")
    assert cli(argv) == 6


def test_market_column_allowlist_carries_no_outcome_field() -> None:
    joined = " ".join(READ_COLUMNS).lower()
    assert not any(word in joined for word in ("label", "pnl", "return", "path"))


@needs_slice
def test_the_frozen_manifest_and_first_session_pin_the_clock_order_surprise() -> None:
    manifest, inputs = verify_manifest(MANIFEST, repo_root=Path("."))
    assert manifest["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert len(inputs) == 64 and sum(item.rows for item in inputs) == 173_470_783

    frame = pd.read_parquet(inputs[0].parquet, columns=list(READ_COLUMNS)).reset_index()
    events, diagnostic = signed_events(frame, inputs[0].session)
    assert len(events) == 23_198
    assert diagnostic["tied_receive_signed"] == 4_863
    assert diagnostic["tied_receive_and_event_signed"] == 4_863


def test_production_bootstrap_budget_stays_declared() -> None:
    assert BOOTSTRAP_REPS == 10_000

