from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

from v4.research import pathd_entry_exit as frozen_prereg
from v4.research.pathd_feature_live_twin import (
    AVAILABLE_AT_EVENT_PLUS_60_SECONDS,
    BAR_OPEN_TIMESTAMP,
    DATABENTO_OPRA_CBBO_1M,
    DATABENTO_OPRA_CBBO_1M_COMPLETED,
    DATABENTO_OPRA_CBBO_1S_TO_1M,
    DATABENTO_OPRA_CMBP_1_TO_1M,
    DATABENTO_OPRA_OHLCV_1M_COMPLETED,
    DATABENTO_OPRA_STATISTICS_EOD,
    DATABENTO_OPRA_TRADES_1M_COMPLETED,
    DROP,
    DROP_UNTIL_EXACT_ADAPTER_RECEIPT,
    ENTRY17_FEATURE_NAMES,
    ENTRY17_LIVE_TWIN_INVENTORY,
    EXIT47_CORRECTED_FEATURE_NAMES,
    EXIT47_CORRECTED_LIVE_TWIN_INVENTORY,
    EXIT48_INTERMEDIATE_FEATURE_NAMES,
    EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY,
    EXIT49_FEATURE_NAMES,
    EXIT49_LIVE_TWIN_INVENTORY,
    IntradayLiveTwinBinding,
    LIVE_DERIVABLE_PENDING_ADAPTER,
    LIVE_NATIVE_CAUSAL_STATE,
    NO_INTRADAY_LIVE_TWIN,
    PRIOR_DAY_EOD_STATIC,
    SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF,
    SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
    live_twin_record,
    validate_intraday_live_twin_binding,
    validate_inventory_contracts,
)


def _volume_binding(**changes: object) -> IntradayLiveTwinBinding:
    binding = IntradayLiveTwinBinding(
        namespace="exit",
        feature_name="last_causal_minute_volume",
        live_source=DATABENTO_OPRA_OHLCV_1M_COMPLETED,
        adapter="v4.future.pathd_exit_live::completed_option_minute_volume",
        event_time_semantics=BAR_OPEN_TIMESTAMP,
        available_at_rule=AVAILABLE_AT_EVENT_PLUS_60_SECONDS,
        selection_rule=SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF,
        selected_available_at_ns=61_000_000_000,
        decision_time_ns=61_000_000_000,
        completed_only=True,
        latest_exact_contract=True,
        max_feature_age_seconds=90,
        cross_session_carry=False,
        missing_exact_interval_is_zero_after_cutoff=True,
        adapter_implementation_sha256="a" * 64,
        adapter_receipt_sha256="b" * 64,
    )
    return replace(binding, **changes)


def test_exact_signed17_original_exit49_intermediate_exit48_and_corrected_exit47_coverage() -> None:
    validate_inventory_contracts()

    assert len(ENTRY17_FEATURE_NAMES) == len(ENTRY17_LIVE_TWIN_INVENTORY) == 17
    assert tuple(row.feature_name for row in ENTRY17_LIVE_TWIN_INVENTORY) == ENTRY17_FEATURE_NAMES

    superseded_path = frozen_prereg.SUPERSEDED_AUDIT_ROOT / "preregistration.json"
    raw = superseded_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "c811acf57f8d57c9067e9534314957da64b3a1f3860b877c3cdc2a842fcdf60a"
    )
    superseded = json.loads(raw)
    superseded_lineage_path = (
        frozen_prereg.SUPERSEDED_AUDIT_ROOT / "feature_lineage.json"
    )
    lineage_raw = superseded_lineage_path.read_bytes()
    assert hashlib.sha256(lineage_raw).hexdigest() == (
        "1534bd8784715e257e6a89701b05e36512f2dfaf66cc50cfdb2fa4d3506f970a"
    )
    superseded_lineage = json.loads(lineage_raw)
    assert tuple(
        superseded_lineage["entry_feature_names_in_exact_order"]
    ) == ENTRY17_FEATURE_NAMES
    original_exit = tuple(
        superseded["exit"]["feature_contract"]["feature_names_in_exact_order"]
    )
    assert original_exit == EXIT49_FEATURE_NAMES
    assert len(EXIT49_FEATURE_NAMES) == len(EXIT49_LIVE_TWIN_INVENTORY) == 49
    assert tuple(row.feature_name for row in EXIT49_LIVE_TWIN_INVENTORY) == EXIT49_FEATURE_NAMES
    assert tuple(
        frozen_prereg.exit_feature_spec()["feature_names_in_exact_order"]
    ) == EXIT47_CORRECTED_FEATURE_NAMES
    assert len(EXIT48_INTERMEDIATE_FEATURE_NAMES) == 48
    assert tuple(
        row.feature_name for row in EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY
    ) == EXIT48_INTERMEDIATE_FEATURE_NAMES
    assert len(EXIT47_CORRECTED_FEATURE_NAMES) == 47
    assert tuple(
        row.feature_name for row in EXIT47_CORRECTED_LIVE_TWIN_INVENTORY
    ) == EXIT47_CORRECTED_FEATURE_NAMES


def test_every_signed17_feature_is_live_derivable_without_changing_signed17() -> None:
    assert Counter(row.family for row in ENTRY17_LIVE_TWIN_INVENTORY) == {
        "official_spx_completed_minute_derivation": 12,
        "opra_cbbo_completed_minute_ladder_derivation": 3,
        "self_computed_greeks": 2,
    }
    assert all(
        row.live_twin_class == LIVE_DERIVABLE_PENDING_ADAPTER
        and row.adapter_receipt_required_before_fit
        and row.permitted_intraday_sources
        for row in ENTRY17_LIVE_TWIN_INVENTORY
    )


def test_entry_option_features_bind_historical_cbbo_1m() -> None:
    option_rows = [
        row
        for row in ENTRY17_LIVE_TWIN_INVENTORY
        if row.feature_name.startswith("D.")
    ]

    assert len(option_rows) == 3
    assert all(row.historical_source == DATABENTO_OPRA_CBBO_1M for row in option_rows)
    assert all(
        row.permitted_intraday_sources
        == (
            DATABENTO_OPRA_CBBO_1M_COMPLETED,
            DATABENTO_OPRA_CBBO_1S_TO_1M,
            DATABENTO_OPRA_CMBP_1_TO_1M,
        )
        and "completed interval end" in row.clock_semantics
        and row.max_feature_age_seconds == 90
        for row in option_rows
    )


def test_exit49_has_an_explicit_classification_for_every_feature() -> None:
    assert Counter(row.family for row in EXIT49_LIVE_TWIN_INVENTORY) == {
        "opra_cbbo_current_and_rolling": 22,
        "opra_completed_minute_trade_volume": 1,
        "opra_statistics_open_interest": 1,
        "official_spx_completed_minute_derivation": 5,
        "self_computed_greeks_current_and_changes": 7,
        "position_clock_and_account_state": 13,
    }
    assert all(row.clock_semantics for row in EXIT49_LIVE_TWIN_INVENTORY)

    internal = [
        row
        for row in EXIT49_LIVE_TWIN_INVENTORY
        if row.family == "position_clock_and_account_state"
    ]
    assert all(row.live_twin_class == LIVE_NATIVE_CAUSAL_STATE for row in internal)
    assert all(
        row.selection_rule
        and row.carry_policy
        and row.cross_session_carry is False
        for row in (*ENTRY17_LIVE_TWIN_INVENTORY, *EXIT49_LIVE_TWIN_INVENTORY)
    )


def test_corrected_exit47_drops_open_interest_and_unproved_minute_volume() -> None:
    assert len(EXIT47_CORRECTED_FEATURE_NAMES) == 47
    assert tuple(row.feature_name for row in EXIT47_CORRECTED_LIVE_TWIN_INVENTORY) == (
        EXIT47_CORRECTED_FEATURE_NAMES
    )
    assert set(EXIT49_FEATURE_NAMES) - set(EXIT47_CORRECTED_FEATURE_NAMES) == {
        "last_causal_minute_volume",
        "last_causal_open_interest",
    }

    open_interest = live_twin_record("exit", "last_causal_open_interest")
    assert open_interest.historical_source == DATABENTO_OPRA_STATISTICS_EOD
    assert open_interest.permitted_intraday_sources == ()
    assert open_interest.live_twin_class == NO_INTRADAY_LIVE_TWIN
    assert open_interest.prior_day_static_class == PRIOR_DAY_EOD_STATIC
    assert open_interest.recommended_action == DROP
    minute_volume = live_twin_record("exit", "last_causal_minute_volume")
    assert minute_volume.recommended_action == DROP_UNTIL_EXACT_ADAPTER_RECEIPT
    assert minute_volume.feature_name not in EXIT47_CORRECTED_FEATURE_NAMES


def test_open_interest_is_classified_prior_day_static_but_absent_from_exit47() -> None:
    open_interest = live_twin_record("exit", "last_causal_open_interest")

    assert open_interest.prior_day_static_class == PRIOR_DAY_EOD_STATIC
    assert open_interest.selection_rule == (
        "prior_session_eod_only_if_used_as_one_static_session_value"
    )
    assert open_interest.max_feature_age_seconds is None
    assert open_interest.feature_name not in EXIT47_CORRECTED_FEATURE_NAMES


def test_feature_without_live_twin_fixture_rejects_intraday_open_interest_even_with_adapter(
) -> None:
    fake = IntradayLiveTwinBinding(
        namespace="exit",
        feature_name="last_causal_open_interest",
        live_source=DATABENTO_OPRA_STATISTICS_EOD,
        adapter="fake.but.nonempty::intraday_open_interest",
        event_time_semantics=BAR_OPEN_TIMESTAMP,
        available_at_rule=AVAILABLE_AT_EVENT_PLUS_60_SECONDS,
        selection_rule=SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
        selected_available_at_ns=61_000_000_000,
        decision_time_ns=61_000_000_000,
        completed_only=True,
        latest_exact_contract=True,
        max_feature_age_seconds=90,
        cross_session_carry=False,
        missing_exact_interval_is_zero_after_cutoff=False,
        adapter_implementation_sha256="a" * 64,
        adapter_receipt_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="no intraday live twin"):
        validate_intraday_live_twin_binding(fake)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"adapter": ""}, "concrete live adapter"),
        ({"adapter_implementation_sha256": ""}, "implementation SHA-256"),
        ({"adapter_receipt_sha256": "c" * 63}, "receipt SHA-256"),
        ({"live_source": DATABENTO_OPRA_STATISTICS_EOD}, "live source"),
        ({"event_time_semantics": "BAR_CLOSE_TIMESTAMP"}, "bar-open"),
        ({"available_at_rule": "event_time"}, "bar close"),
        ({"selection_rule": "latest_event_time"}, "selection rule"),
        (
            {
                "selected_available_at_ns": 61_000_000_001,
                "decision_time_ns": 61_000_000_000,
            },
            "unavailable at decision",
        ),
        (
            {
                "selected_available_at_ns": 0,
                "decision_time_ns": 91_000_000_000,
                "max_feature_age_seconds": 90,
            },
            "age bound",
        ),
        ({"completed_only": False}, "completed observations"),
        ({"latest_exact_contract": False}, "exact-contract"),
        ({"max_feature_age_seconds": 91}, "90-second"),
        ({"max_feature_age_seconds": -1}, "nonnegative"),
        ({"cross_session_carry": True}, "Cross-session|cross-session"),
        (
            {"missing_exact_interval_is_zero_after_cutoff": False},
            "missing exact interval as zero",
        ),
    ],
)
def test_minute_volume_fails_without_exact_completed_bar_semantics(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_intraday_live_twin_binding(_volume_binding(**change))


@pytest.mark.parametrize(
    "source",
    [DATABENTO_OPRA_OHLCV_1M_COMPLETED, DATABENTO_OPRA_TRADES_1M_COMPLETED],
)
@pytest.mark.parametrize("age_bound", [0, 90])
def test_minute_volume_is_conditionally_live_derivable_with_exact_receipt(
    source: str, age_bound: int
) -> None:
    record = validate_intraday_live_twin_binding(
        _volume_binding(live_source=source, max_feature_age_seconds=age_bound)
    )

    assert record.feature_name == "last_causal_minute_volume"
    assert record.live_twin_class == LIVE_DERIVABLE_PENDING_ADAPTER
    assert record.adapter_receipt_required_before_fit is True
    assert record.recommended_action == DROP_UNTIL_EXACT_ADAPTER_RECEIPT
    assert record.feature_name not in EXIT47_CORRECTED_FEATURE_NAMES


def test_minute_volume_inventory_forbids_prior_bar_carry() -> None:
    record = live_twin_record("exit", "last_causal_minute_volume")
    assert record.selection_rule == SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF
    assert "never carry" in record.carry_policy
