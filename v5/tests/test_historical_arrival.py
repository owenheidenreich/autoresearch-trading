"""Adversarial tests for injecting arrival latency into historical rows.

The defect these guard against is measured and specific: the owned corpus
records `receive_time == event_time` on all 47,707,186 of its rows, while the
live CBBO-1m stream measured 226.932 ms at the median and 319.521 ms at p99.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from v5.research import training_twin as tt


OPRA_1M = "DATABENTO_OPRA_CBBO_1M"
THETA = "THETADATA_OFFICIAL_SPX_1M"


def _receipt(**overrides):
    fields = dict(
        source_family=OPRA_1M,
        p50_ms=226.932,
        p99_ms=319.521,
        max_ms=319.543,
        session_count=1,
        measured_on="2026-08-03",
        valid_until="2026-09-03",
        evidence_path=(
            "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/"
            "attempt002/capture_summary.json"
        ),
    )
    fields.update(overrides)
    return tt.make_latency_receipt(**fields)


def _corpus_shaped_frame(rows: int = 3) -> pd.DataFrame:
    """A frame shaped like the real corpus: arrival equals the interval close."""

    minutes = pd.date_range("2026-03-16T14:31:00Z", periods=rows, freq="1min")
    interval_end_ns = [value.value + tt.MINUTE_NS for value in minutes]
    return pd.DataFrame(
        {
            "event_time": minutes,
            "received_at_ns": interval_end_ns,
            "bid_px_00": [1.0] * rows,
            "ask_px_00": [1.2] * rows,
        }
    )


def test_the_corpus_shape_is_rejected_outright() -> None:
    with pytest.raises(tt.TrainingTwinError, match="zero arrival lag"):
        tt.assert_no_zero_lag(_corpus_shaped_frame())


def test_injected_arrival_passes_the_same_check() -> None:
    repaired = tt.simulate_historical_arrival(
        _corpus_shaped_frame(),
        receipt=_receipt(),
        source_family=OPRA_1M,
        as_of="2026-08-05",
    )
    tt.assert_no_zero_lag(repaired)
    lag_ns = repaired["received_at_ns"] - (
        repaired["event_time"].map(lambda value: value.value) + tt.MINUTE_NS
    )
    assert set(lag_ns.unique()) == {319_521_000}
    assert (repaired["arrival_source"] == "SIMULATED_FROM_LATENCY_RECEIPT").all()


def test_a_thetadata_receipt_cannot_license_an_opra_stream() -> None:
    """The 2,336 ms allowance came from the wrong feed; that must not transfer."""

    with pytest.raises(tt.TrainingTwinError, match="does not license"):
        tt.simulate_historical_arrival(
            _corpus_shaped_frame(),
            receipt=_receipt(source_family=THETA, p50_ms=2000.0, p99_ms=2335.23, max_ms=2336.0),
            source_family=OPRA_1M,
            as_of="2026-08-05",
        )


def test_an_expired_receipt_is_refused() -> None:
    with pytest.raises(tt.TrainingTwinError, match="expired"):
        tt.simulate_historical_arrival(
            _corpus_shaped_frame(),
            receipt=_receipt(valid_until="2026-08-04"),
            source_family=OPRA_1M,
            as_of="2026-08-05",
        )


def test_a_tampered_receipt_is_refused() -> None:
    tampered = replace(_receipt(), p99_ms=5.0)
    with pytest.raises(tt.TrainingTwinError, match="self-hash mismatch"):
        tt.simulate_historical_arrival(
            _corpus_shaped_frame(),
            receipt=tampered,
            source_family=OPRA_1M,
            as_of="2026-08-05",
        )


def test_unordered_percentiles_are_refused() -> None:
    with pytest.raises(tt.TrainingTwinError, match="not ordered"):
        _receipt(p50_ms=400.0, p99_ms=100.0, max_ms=500.0).assert_usable(
            source_family=OPRA_1M, as_of="2026-08-05"
        )


def test_injected_lag_actually_changes_which_row_a_selector_picks() -> None:
    """A quarter-second matters: the boundary row stops being consumable."""

    boundary_ns = pd.Timestamp("2026-03-16T14:32:00Z").value
    clock = tt.make_clock(feature_boundary_ns=boundary_ns, emission_lag_ms=100, order_latency_ms=0)
    frame = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-03-16T14:31:00Z")],
            "received_at_ns": [boundary_ns],
            "close": [5000.0],
        }
    )
    # With the corpus's fictional zero lag the bar is available at the boundary.
    _, selected = tt.select_official_spx(frame, clock=clock)
    assert selected.received_at_ns == boundary_ns

    # With a real 319.521 ms arrival it has not landed by a 100 ms emission clock.
    repaired = tt.simulate_historical_arrival(
        frame, receipt=_receipt(), source_family=OPRA_1M, as_of="2026-08-05"
    )
    with pytest.raises(tt.TrainingTwinError, match="expected one exact completed SPX bar"):
        tt.select_official_spx(repaired, clock=clock)


def test_arrival_before_its_own_interval_is_refused() -> None:
    frame = _corpus_shaped_frame(1)
    frame["received_at_ns"] = frame["received_at_ns"] - 1
    with pytest.raises(tt.TrainingTwinError, match="precedes the interval"):
        tt.assert_no_zero_lag(frame)


def test_a_zero_lag_receipt_cannot_be_used_to_reproduce_the_defect() -> None:
    with pytest.raises(tt.TrainingTwinError, match="must be positive"):
        tt.simulate_historical_arrival(
            _corpus_shaped_frame(),
            receipt=_receipt(p50_ms=0.0, p99_ms=0.0, max_ms=0.0),
            source_family=OPRA_1M,
            as_of="2026-08-05",
            percentile="p99_ms",
        )
