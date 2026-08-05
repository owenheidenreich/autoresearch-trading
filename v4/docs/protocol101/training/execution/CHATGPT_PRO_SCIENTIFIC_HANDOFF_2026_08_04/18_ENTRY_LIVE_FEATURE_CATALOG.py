"""Executable live-first catalog for autoresearch_v2 entry features.

The catalog deliberately distinguishes an implementable idea from a feature
that is fit-ready.  A prose claim such as ``"Protocol101 live option ladder"``
is not a live twin.  Fit-ready means that the exact contract ID below is bound
to a shared historical/live adapter and its required parity receipts exist.

This module does not authorize a model fit.  Most useful feature families are
intentionally pending while the live training twin is completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


CATALOG_SCHEMA_VERSION: Final = "autoresearch_v2.entry-live-feature-catalog.v1"

FIT_READY: Final = "FIT_READY"
PENDING_SAME_SESSION_REPLAY: Final = "PENDING_SAME_SESSION_REPLAY"
PENDING_SHARED_ADAPTER: Final = "PENDING_SHARED_ADAPTER"
PENDING_SPARSE_ZERO_ADAPTER: Final = "PENDING_SPARSE_ZERO_ADAPTER"
PENDING_LIVE_RECEIPT_PROOF: Final = "PENDING_LIVE_RECEIPT_PROOF"
PENDING_CAUSAL_PARENT: Final = "PENDING_CAUSAL_PARENT"
PENDING_COVERAGE_AND_LIVE: Final = "PENDING_COVERAGE_AND_LIVE"
PENDING_PREOPEN_REPLAY: Final = "PENDING_PREOPEN_REPLAY"
PENDING_SERIAL_ADAPTER: Final = "PENDING_SERIAL_ADAPTER"
NEEDS_HISTORICAL_SUBSTRATE: Final = "NEEDS_HISTORICAL_SUBSTRATE"
GUARD_ONLY: Final = "GUARD_ONLY"
BARRED: Final = "BARRED"


@dataclass(frozen=True)
class EntryFeatureFamilyContract:
    contract_id: str
    feature_names: tuple[str, ...]
    declared_families: tuple[str, ...]
    historical_source: str
    historical_substrate_owned: bool
    live_source: str
    live_observed: bool
    available_at: str
    event_time_semantics: str
    carry_or_missing_semantics: str
    cold_start_semantics: str
    required_parity_receipts: tuple[str, ...]
    fit_status: str
    use: str


def _contract(
    contract_id: str,
    feature_names: tuple[str, ...],
    declared_families: tuple[str, ...],
    historical_source: str,
    historical_substrate_owned: bool,
    live_source: str,
    live_observed: bool,
    available_at: str,
    event_time_semantics: str,
    carry_or_missing_semantics: str,
    cold_start_semantics: str,
    required_parity_receipts: tuple[str, ...],
    fit_status: str,
    use: str = "candidate_alpha",
) -> EntryFeatureFamilyContract:
    return EntryFeatureFamilyContract(
        contract_id=contract_id,
        feature_names=feature_names,
        declared_families=declared_families,
        historical_source=historical_source,
        historical_substrate_owned=historical_substrate_owned,
        live_source=live_source,
        live_observed=live_observed,
        available_at=available_at,
        event_time_semantics=event_time_semantics,
        carry_or_missing_semantics=carry_or_missing_semantics,
        cold_start_semantics=cold_start_semantics,
        required_parity_receipts=required_parity_receipts,
        fit_status=fit_status,
        use=use,
    )


ENTRY_FEATURE_FAMILY_CATALOG: Final = (
    _contract(
        "entry.contract_clock.v1",
        (
            "is_call",
            "strike",
            "seconds_to_expiry",
            "seconds_from_open",
            "seconds_to_close",
            "minute_of_session",
            "day_of_week",
            "is_early_close",
        ),
        ("contract_clock", "signed17", "causal_clock"),
        "OPRA definition history plus one exchange-calendar implementation",
        True,
        "live OPRA current-session definitions plus the same exchange calendar",
        True,
        "decision_time",
        "definition received before decision; wall-clock fields evaluated at decision",
        "no carry across sessions; identity is raw OSI symbol, never daily instrument_id",
        "no decision until the current-session definition replay is complete",
        (
            "definition implementation hash",
            "historical/live OSI geometry invariance",
            "exchange-calendar hash",
        ),
        PENDING_SHARED_ADAPTER,
    ),
    _contract(
        "entry.opra_cbbo1m_native.v1",
        (
            "option_bid",
            "option_ask",
            "option_mid",
            "option_spread",
            "option_spread_over_mid",
            "option_bid_size",
            "option_ask_size",
            "size_imbalance",
            "microprice",
            "last_interval_trade_price",
            "last_interval_trade_size",
        ),
        ("opra_cbbo1m_native", "live_safe_microstructure"),
        "owned Path-D OPRA CBBO-1m parquet",
        True,
        "Databento Live OPRA native CBBO-1m",
        True,
        "interval_end_plus_frozen_lag",
        "ts_recv is the completed minute boundary; consume only after local receipt by t+L",
        "no synthetic quote update; absence is missing and fails the exact-contract freshness gate",
        "the first interval after subscription is warm-up-only",
        (
            "same-session live-versus-Historical-API value identity",
            "multi-session local receipt-latency distribution",
            "sparse-minute and freshness receipt",
        ),
        PENDING_SAME_SESSION_REPLAY,
    ),
    _contract(
        "entry.opra_cbbo1m_cross_section.v1",
        (
            "straddle_mid_spot_bps",
            "put_call_mid_ratio",
            "side_smile_slope",
            "D.near_atm.straddle_mid_spot_bps",
            "D.near_atm.put_call_mid_ratio",
            "D.near_atm.side_smile_slope_bps_per_5pt",
            "chain_spread_median",
            "chain_depth_imbalance",
            "smile_curvature",
            "put_call_depth_ratio",
            "candidate_liquidity_rank",
        ),
        ("opra_cbbo1m_cross_section", "signed17", "canonical_signed17"),
        "owned Path-D OPRA CBBO-1m plus definitions",
        True,
        "same live CBBO-1m boundary over the current definition universe",
        True,
        "interval_end_plus_frozen_lag",
        "one atomic ladder snapshot at t after every selected row is received by t+L",
        "no future strike backfill; missing legs produce explicit missingness or block the row",
        "warm-up until definitions and a minimum complete near-ATM ladder are present",
        (
            "entry.opra_cbbo1m_native.v1 receipts",
            "historical/live candidate-universe identity",
            "atomic ladder completeness receipt",
        ),
        PENDING_SHARED_ADAPTER,
    ),
    _contract(
        "entry.opra_cbbo1s_rolling.v1",
        (
            "option_log_mid_return_1s",
            "option_log_mid_return_5s",
            "option_log_mid_return_15s",
            "option_log_mid_return_30s",
            "option_log_mid_return_60s",
            "option_spread_mean_5s",
            "option_spread_mean_15s",
            "option_spread_std_15s",
            "option_imbalance_mean_5s",
            "option_imbalance_mean_15s",
            "option_quote_update_count_60s",
            "option_quote_age_ms",
        ),
        ("opra_cbbo1s_rolling", "live_safe_microstructure"),
        "owned Path-D OPRA CBBO-1s parquet",
        True,
        "Databento Live OPRA native CBBO-1s",
        True,
        "decision_emission",
        "closed one-second intervals selected by receipt time at or before emission",
        "time-grid semantics must distinguish no update from a forward-filled quote",
        "at least 60 closed seconds are required for 60-second features",
        (
            "same-session live-versus-Historical-API value identity",
            "shared rolling-window implementation hash",
            "no-update and reconnect mutation tests",
        ),
        PENDING_SAME_SESSION_REPLAY,
    ),
    _contract(
        "entry.opra_ohlcv1m_sparse.v1",
        (
            "last_causal_minute_volume",
            "option_trade_volume_1m",
            "option_trade_range_1m",
            "option_trade_return_1m",
        ),
        ("opra_completed_minute_trade_volume", "live_safe_microstructure"),
        "owned Path-D OPRA OHLCV-1m parquet",
        True,
        "Databento Live OPRA OHLCV-1m or trades aggregated by the same adapter",
        True,
        "interval_end_plus_frozen_lag",
        "ts_event is bar-open and available_at is ts_event+60s plus observed receipt lag",
        "a healthy subscribed minute with no bar means zero volume; never carry the prior bar",
        "first interval is warm-up-only; zero can be finalized only after the frozen lag",
        (
            "multi-session OHLCV receipt-latency distribution",
            "sparse zero-fill invariance",
            "native OHLCV versus live-trade aggregation identity",
        ),
        PENDING_SPARSE_ZERO_ADAPTER,
    ),
    _contract(
        "entry.opra_tcbbo_trade_flow.v1",
        (
            "trade_count_1m",
            "trade_size_sum_1m",
            "trade_size_mean_1m",
            "trade_vwap_1m",
            "trade_at_bid_fraction_1m",
            "trade_at_ask_fraction_1m",
            "quote_rule_signed_volume_1m",
        ),
        ("opra_trade_flow",),
        "not owned: Path-D has OHLCV but no OPRA trades/TCBBO tick history",
        False,
        "Databento Live OPRA trades plus TCBBO",
        True,
        "decision_emission",
        "tick receipt at or before emission; TCBBO supplies the BBO paired to each trade",
        "no cross-minute carry; empty healthy minute is a zero-count minute",
        "warm-up through the first complete minute",
        (
            "historical trades/TCBBO acquisition",
            "trade/TCBBO one-to-one identity",
            "quote-rule classification audit because native OPRA side is N",
        ),
        NEEDS_HISTORICAL_SUBSTRATE,
    ),
    _contract(
        "entry.opra_cmbp1_event_flow.v1",
        (
            "quote_event_count_1m",
            "bid_update_count_1m",
            "ask_update_count_1m",
            "quote_size_churn_1m",
            "time_weighted_size_imbalance_1m",
            "time_weighted_spread_1m",
        ),
        ("opra_event_flow",),
        "not owned: Path-D has CBBO-1s/1m but no CMBP-1 event history",
        False,
        "Databento Live OPRA CMBP-1",
        True,
        "decision_emission",
        "every received consolidated level-one event through emission",
        "no event synthesis; windows require an explicit healthy subscription interval",
        "warm-up through the longest rolling window and after every reconnect",
        (
            "historical CMBP-1 acquisition",
            "bounded-ladder ingestion latency receipt",
            "event replay identity",
        ),
        NEEDS_HISTORICAL_SUBSTRATE,
    ),
    _contract(
        "entry.opra_implied_spot.v1",
        (
            "opra_implied_spot",
            "opra_implied_spot_dispersion_bps",
            "opra_spot_return_1m_bps",
            "opra_spot_return_5m_bps",
            "opra_spot_return_15m_bps",
            "opra_spot_vwap_gap_bps",
            "opra_spot_session_range_bps",
            "opra_spot_omar_clipped",
        ),
        ("opra_implied_spot",),
        "owned Path-D OPRA CBBO-1m put/call ladder",
        True,
        "same live OPRA CBBO-1m ladder; robust put-call-parity estimator",
        True,
        "interval_end_plus_frozen_lag",
        "derive after the exact atomic ladder closes; no separate vendor clock",
        "no cross-session carry; block when paired-strike coverage or dispersion fails",
        "minimum paired-strike count and dispersion guard must pass",
        (
            "shared parity-estimator implementation hash",
            "historical/live candidate-pair identity",
            "comparison to completed official SPX for measurement only",
        ),
        PENDING_SHARED_ADAPTER,
    ),
    _contract(
        "entry.opra_implied_volatility.v1",
        (
            "opra_atm_iv",
            "opra_atm_iv_change_5m",
            "opra_put_skew",
            "opra_call_skew",
            "opra_smile_curvature",
            "opra_straddle_bps",
        ),
        ("opra_implied_volatility",),
        "owned Path-D OPRA CBBO-1m plus definitions",
        True,
        "same live OPRA CBBO-1m ladder",
        True,
        "interval_end_plus_frozen_lag",
        "pure calculation after causal option prices, implied spot, strike, and expiry pass",
        "inherits ladder missingness; no independent carry",
        "requires the implied-spot and ladder guards",
        (
            "entry.opra_implied_spot.v1 receipts",
            "shared solver/constants hash",
            "mutation and numerical-stability tests",
        ),
        PENDING_CAUSAL_PARENT,
    ),
    _contract(
        "entry.thetadata_completed_spx_vix.v1",
        (
            "spx_vwap_gap_points",
            "spx_vwap_gap_bps",
            "spx_vwap_gap_over_session_range",
            "session_range_bps",
            "momentum_5m_bps",
            "momentum_15m_bps",
            "momentum_5m_over_session_range",
            "momentum_15m_over_session_range",
            "omar_clipped",
            "omar_clipped_neg3_pos3",
            "vwap_side_align",
            "vwap_side_alignment_flag",
            "omar_side_align",
            "omar_side_alignment_flag",
            "momentum15_side_align",
            "momentum15_side_alignment_flag",
            "vix_close",
            "vix_return_5m_bps",
        ),
        ("signed17", "canonical_signed17", "official_spx_completed_minute_derivation"),
        "owned ThetaData official SPX/VIX 1m history",
        True,
        "ThetaData live/snapshot completed SPX/VIX minute adapter",
        False,
        "interval_end_plus_frozen_lag",
        "historical event_time is bar-open; only [t-60s,t) may feed a decision emitted after t",
        "same-session completed bars only; never pair the option quote at t with the bar [t,t+60)",
        "warm-up through VWAP and momentum history; fail closed on late receipt",
        (
            "authorized ThetaData live receipt capture",
            "history/snapshot value identity",
            "multi-session receipt-latency distribution",
        ),
        PENDING_LIVE_RECEIPT_PROOF,
    ),
    _contract(
        "entry.self_computed_greeks.v1",
        ("bs_delta", "bs_gamma", "E.bs.delta", "E.bs.gamma", "bs_theta", "bs_vega", "self_iv"),
        ("signed17", "canonical_signed17", "self_computed_greeks"),
        "pure calculation over historical causal option price, spot, contract, rate, and time",
        True,
        "the identical calculation over causal live inputs",
        True,
        "decision_emission",
        "calculate only after every parent input is available by emission",
        "no independent carry; inherits the oldest parent age",
        "all parents and the numerical-domain guard must pass",
        (
            "causal spot parent receipt",
            "shared solver/constants hash",
            "historical/live golden-vector identity",
        ),
        PENDING_CAUSAL_PARENT,
    ),
    _contract(
        "entry.es_vx_completed_futures.v1",
        ("es_return_1m_bps", "es_return_5m_bps", "es_vwap_gap_bps", "vx_close", "vx_return_5m_bps"),
        ("futures_context",),
        "owned ES OHLCV-1m; owned VX history is incomplete",
        False,
        "Databento GLBX ES and XCBF VX live subscriptions",
        False,
        "interval_end_plus_frozen_lag",
        "closed futures minute at or before the shared decision emission",
        "no cross-session carry; explicit futures session/calendar mapping",
        "continuous-symbol mapping and current-session warm-up must complete",
        (
            "full historical coverage",
            "live entitlement and receipt capture",
            "continuous-contract roll identity",
        ),
        PENDING_COVERAGE_AND_LIVE,
    ),
    _contract(
        "entry.session_static_prior_day.v1",
        ("prior_day_open_interest", "prior_day_option_volume", "prior_day_close", "prior_day_realized_range"),
        ("prior_day_static",),
        "owned prior-session OPRA statistics/OHLCV and context",
        True,
        "pre-open/start-of-day statistics or locally materialized prior-session state",
        False,
        "session_open_static",
        "one value frozen before the first eligible decision for the whole session",
        "never refresh or carry an intraday pseudo-update; explicit missing value if unavailable",
        "session cannot use the field until the pre-open replay/freeze receipt is durable",
        (
            "pre-open replay capture",
            "prior-session date binding",
            "session-static immutability test",
        ),
        PENDING_PREOPEN_REPLAY,
    ),
    _contract(
        "entry.causal_account_state.v1",
        (
            "entries_so_far",
            "realized_session_pnl_dollars",
            "seconds_since_last_exit",
            "loss_streak_so_far",
            "position_occupancy",
            "remaining_entry_budget_dollars",
        ),
        ("causal_account_state",),
        "strict serial simulator ledger through the decision",
        True,
        "the same hash-chained live/paper ledger through the decision",
        False,
        "decision_time",
        "ledger tip only; never use eventual session totals",
        "same-session state only and reset at the session boundary",
        "genesis ledger receipt required before the first decision",
        (
            "historical/live ledger transition identity",
            "serial replay parity",
            "mutate-future invariance",
        ),
        PENDING_SERIAL_ADAPTER,
    ),
    _contract(
        "entry.feed_health_guard.v1",
        (
            "subscription_ready",
            "definition_ready",
            "source_late",
            "quote_stale",
            "ladder_incomplete",
            "reconnect_warmup",
            "is_trading",
            "is_quoting",
        ),
        ("feed_health",),
        "historical data-quality and availability receipts",
        True,
        "live subscription/status/latency state",
        True,
        "decision_emission",
        "observed health state at emission",
        "no alpha carry; a failed guard blocks the decision",
        "all readiness guards start false",
        ("status replay receipt", "reconnect and late-source fault injection"),
        GUARD_ONLY,
        use="guard_only_not_model_input",
    ),
    _contract(
        "entry.barred_no_live_twin.v1",
        (
            "last_causal_open_interest",
            "intraday_open_interest",
            "future_mfe",
            "future_mae",
            "future_pnl",
            "oracle_action",
            "session_final_rank",
            "session_final_quantile",
            "daily_instrument_id_identity",
            "opra_full_depth_queue_position",
        ),
        ("barred", "microstructure"),
        "historical-only, future, unstable-identity, or unavailable depth field",
        False,
        "none with matching decision-time semantics",
        False,
        "never",
        "not causal or not available",
        "forbidden",
        "forbidden",
        (),
        BARRED,
        use="forbidden_model_input",
    ),
)


CONTRACTS_BY_ID: Final = {row.contract_id: row for row in ENTRY_FEATURE_FAMILY_CATALOG}


def validate_catalog() -> None:
    if len(CONTRACTS_BY_ID) != len(ENTRY_FEATURE_FAMILY_CATALOG):
        raise RuntimeError("duplicate entry live-feature contract ID")
    for row in ENTRY_FEATURE_FAMILY_CATALOG:
        if not row.contract_id or not row.feature_names or not row.required_parity_receipts and row.fit_status not in {BARRED}:
            raise RuntimeError(f"incomplete live-feature contract: {row.contract_id}")
        if len(set(row.feature_names)) != len(row.feature_names):
            raise RuntimeError(f"duplicate feature within contract: {row.contract_id}")
        if row.fit_status == FIT_READY and not (
            row.historical_substrate_owned and row.live_observed
        ):
            raise RuntimeError(f"fit-ready contract lacks both substrates: {row.contract_id}")


def lint_executable_live_twin(
    feature: Any, *, require_fit_ready: bool = True
) -> tuple[str, ...]:
    """Validate one typed feature against a catalog contract, fail closed."""

    contract_id = str(getattr(feature, "live_twin", ""))
    name = str(getattr(feature, "name", ""))
    row = CONTRACTS_BY_ID.get(contract_id)
    if row is None:
        return (f"feature_without_executable_live_twin:{name}:{contract_id}",)
    errors: list[str] = []
    if name not in row.feature_names:
        errors.append(f"feature_not_bound_to_live_twin_contract:{name}:{contract_id}")
    family = str(getattr(feature, "family", ""))
    if family not in row.declared_families:
        errors.append(f"feature_family_mismatch:{name}:{family}:{contract_id}")
    available_at = str(getattr(feature, "available_at", ""))
    if available_at != row.available_at:
        errors.append(
            f"feature_clock_mismatch:{name}:{available_at}:{row.available_at}:{contract_id}"
        )
    if require_fit_ready and row.fit_status != FIT_READY:
        errors.append(f"live_twin_not_fit_ready:{name}:{contract_id}:{row.fit_status}")
    if row.use != "candidate_alpha":
        errors.append(f"feature_not_alpha_eligible:{name}:{contract_id}:{row.use}")
    return tuple(errors)


validate_catalog()
