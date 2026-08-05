from __future__ import annotations

from types import SimpleNamespace

from v4.research.autoresearch_v2.entry_live_feature_catalog import (
    BARRED,
    CONTRACTS_BY_ID,
    ENTRY_FEATURE_FAMILY_CATALOG,
    FIT_READY,
    GUARD_ONLY,
    NEEDS_HISTORICAL_SUBSTRATE,
    PENDING_SPARSE_ZERO_ADAPTER,
    lint_executable_live_twin,
)


def _feature(name: str, family: str, available_at: str, live_twin: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        family=family,
        available_at=available_at,
        live_twin=live_twin,
    )


def test_catalog_has_one_contract_id_and_explicit_terminal_classification() -> None:
    assert len(CONTRACTS_BY_ID) == len(ENTRY_FEATURE_FAMILY_CATALOG)
    statuses = {row.fit_status for row in ENTRY_FEATURE_FAMILY_CATALOG}
    assert statuses.issuperset({NEEDS_HISTORICAL_SUBSTRATE, GUARD_ONLY, BARRED})
    assert FIT_READY not in statuses


def test_contract_clock_catalog_snapshot_defers_to_signed_phase0_ledger() -> None:
    ready = [row.contract_id for row in ENTRY_FEATURE_FAMILY_CATALOG if row.fit_status == FIT_READY]
    assert ready == []
    assert lint_executable_live_twin(
        _feature("is_call", "contract_clock", "decision_time", "entry.contract_clock.v1"),
        require_fit_ready=False,
    ) == ()


def test_minute_volume_requires_sparse_zero_semantics_not_carry() -> None:
    row = CONTRACTS_BY_ID["entry.opra_ohlcv1m_sparse.v1"]
    assert row.fit_status == PENDING_SPARSE_ZERO_ADAPTER
    assert "zero volume" in row.carry_or_missing_semantics
    assert "never carry" in row.carry_or_missing_semantics


def test_trades_and_cmbp_are_barred_until_historical_substrate_exists() -> None:
    for contract_id in (
        "entry.opra_tcbbo_trade_flow.v1",
        "entry.opra_cmbp1_event_flow.v1",
    ):
        row = CONTRACTS_BY_ID[contract_id]
        assert row.live_observed is True
        assert row.historical_substrate_owned is False
        assert row.fit_status == NEEDS_HISTORICAL_SUBSTRATE


def test_guard_and_barred_fields_cannot_be_smuggled_as_alpha() -> None:
    guard_errors = lint_executable_live_twin(
        _feature(
            "subscription_ready",
            "feed_health",
            "decision_emission",
            "entry.feed_health_guard.v1",
        )
    )
    barred_errors = lint_executable_live_twin(
        _feature(
            "intraday_open_interest",
            "barred",
            "never",
            "entry.barred_no_live_twin.v1",
        )
    )
    assert any("feature_not_alpha_eligible" in error for error in guard_errors)
    assert any("feature_not_alpha_eligible" in error for error in barred_errors)
