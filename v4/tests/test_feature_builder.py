"""Tests for feature row builder + forward-fill age tracking."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from v4.feature_eng import FeatureRow, ForwardFiller, feature_rows_to_table


def _good_kwargs(**overrides) -> dict:
    base = {
        "decision_time": datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc),
        "contract_id": "SPXW-20230102-04000.000-C",
        "feature_name": "spread_bps",
        "feature_value": 12.5,
        "is_live_reproducible": True,
        "is_label": False,
        "feature_age_seconds": 0.0,
        "feature_source": "computed_from_normalized_v4",
        "feature_refresh_interval": 60.0,
        "feature_is_estimated": False,
        "feature_is_revised": False,
        "ingest_run_id": "run-1",
    }
    base.update(overrides)
    return base


def test_feature_row_constructs_when_valid() -> None:
    FeatureRow(**_good_kwargs())


def test_feature_row_rejects_non_live_reproducible() -> None:
    with pytest.raises(ValueError, match="is_live_reproducible=True"):
        FeatureRow(**_good_kwargs(is_live_reproducible=False))


def test_feature_row_rejects_label_in_feature_layer() -> None:
    with pytest.raises(ValueError, match="rejects is_label=True"):
        FeatureRow(**_good_kwargs(is_label=True))


def test_feature_row_rejects_negative_age() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FeatureRow(**_good_kwargs(feature_age_seconds=-1.0))


def test_feature_rows_to_table_round_trip() -> None:
    row = FeatureRow(**_good_kwargs())
    table = feature_rows_to_table([row])
    assert table.num_rows == 1
    assert table["feature_name"][0].as_py() == "spread_bps"
    assert table["feature_age_seconds"][0].as_py() == 0.0


def test_feature_rows_to_table_empty() -> None:
    """Empty input produces an empty but well-formed table."""
    table = feature_rows_to_table([])
    assert table.num_rows == 0


# ---------- forward-fill age tracking ----------

def test_forward_filler_basic() -> None:
    ff = ForwardFiller(refresh_interval=600.0, source="optionsdepth_pro_max")
    t0 = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    ff.observe(timestamp=t0, value=1234.5)

    t1 = t0 + timedelta(seconds=120)
    value, age = ff.value_at(t1)
    assert value == 1234.5
    assert age == 120.0


def test_forward_filler_raises_before_first_observe() -> None:
    ff = ForwardFiller(refresh_interval=600.0, source="x")
    with pytest.raises(RuntimeError, match="no observation"):
        ff.value_at(datetime(2023, 1, 2, tzinfo=timezone.utc))


def test_forward_filler_rejects_backward_time() -> None:
    ff = ForwardFiller(refresh_interval=600.0, source="x")
    t0 = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    ff.observe(timestamp=t0, value=1.0)
    with pytest.raises(ValueError, match="before last_observed_at"):
        ff.value_at(t0 - timedelta(seconds=1))


def test_forward_filler_row_at_carries_age() -> None:
    """Critical leak-prevention test: a 9-minute-stale OptionsDepth value
    shows feature_age_seconds=540, not 0."""
    ff = ForwardFiller(
        refresh_interval=600.0,
        source="optionsdepth_pro_max_10min",
        is_estimated=False,
    )
    t_observe = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    ff.observe(timestamp=t_observe, value=42.0)

    t_decide = t_observe + timedelta(seconds=540)  # 9 minutes later
    row = ff.row_at(
        t_decide,
        contract_id="SPXW-20230102-04000.000-C",
        feature_name="net_dealer_gex",
        ingest_run_id="run-1",
    )
    assert row.feature_value == 42.0
    assert row.feature_age_seconds == 540.0
    assert row.feature_refresh_interval == 600.0
    assert row.feature_source == "optionsdepth_pro_max_10min"


def test_forward_filler_row_at_passes_into_table() -> None:
    """Round trip: ForwardFiller → FeatureRow → pyarrow Table."""
    ff = ForwardFiller(refresh_interval=600.0, source="x")
    t0 = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    ff.observe(timestamp=t0, value=42.0)
    row = ff.row_at(
        t0 + timedelta(seconds=60),
        contract_id="SPXW-20230102-04000.000-C",
        feature_name="net_dealer_gex",
        ingest_run_id="run-1",
    )
    table = feature_rows_to_table([row])
    assert table["feature_age_seconds"][0].as_py() == 60.0
