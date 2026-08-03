"""Semantic historical/live-twin inventory for Path-D entry and exit alpha.

This module records the field-parity finding that a nonempty adapter name is not
proof of a real intraday twin.  In particular, OPRA statistics/open interest is
daily/EOD and cannot be made intraday by carrying it for 90 seconds.

The original frozen exit-49 inventory remains visible for audit.  The first
correction removed ``last_causal_open_interest`` but left minute volume pending
an adapter proof.  The final corrected exit-47 contract removes both fields:
open interest has no intraday twin, and minute volume is only conditionally
derivable until a shared historical/live adapter proves exact sparse-minute and
carry semantics.  This inventory is bound into the corrected preregistration,
but does not itself authorize a fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES


INVENTORY_SCHEMA_VERSION: Final = "pathd.feature-live-twin-inventory.v1"

LIVE_DERIVABLE_PENDING_ADAPTER: Final = "LIVE_DERIVABLE_PENDING_ADAPTER"
LIVE_NATIVE_CAUSAL_STATE: Final = "LIVE_NATIVE_CAUSAL_STATE"
NO_INTRADAY_LIVE_TWIN: Final = "NO_INTRADAY_LIVE_TWIN"
PRIOR_DAY_EOD_STATIC: Final = "PRIOR_DAY_EOD_STATIC"

KEEP_AFTER_EXACT_ADAPTER_RECEIPT: Final = "KEEP_AFTER_EXACT_ADAPTER_RECEIPT"
KEEP_CAUSAL_INTERNAL_STATE: Final = "KEEP_CAUSAL_INTERNAL_STATE"
DROP: Final = "DROP"
DROP_UNTIL_EXACT_ADAPTER_RECEIPT: Final = "DROP_UNTIL_EXACT_ADAPTER_RECEIPT"

THETADATA_SPX_1M_COMPLETED: Final = "THETADATA_SPX_1M_COMPLETED"
DATABENTO_OPRA_CBBO_1M: Final = "DATABENTO_OPRA_CBBO_1M"
DATABENTO_OPRA_CBBO_1M_COMPLETED: Final = "DATABENTO_OPRA_CBBO_1M_COMPLETED"
DATABENTO_OPRA_CBBO_1S_TO_1M: Final = (
    "DATABENTO_OPRA_CBBO_1S_CONSOLIDATED_TO_EXACT_CBBO_1M"
)
DATABENTO_OPRA_CMBP_1_TO_1M: Final = (
    "DATABENTO_OPRA_CMBP_1_CONSOLIDATED_TO_EXACT_CBBO_1M"
)
DATABENTO_OPRA_CBBO_1S: Final = "DATABENTO_OPRA_CBBO_1S"
DATABENTO_OPRA_CMBP_1_TO_1S: Final = "DATABENTO_OPRA_CMBP_1_TO_EXACT_CBBO_1S"
SELF_COMPUTED_CAUSAL: Final = "SELF_COMPUTED_FROM_CAUSAL_INPUTS"
INTERNAL_CAUSAL_STATE: Final = "INTERNAL_CAUSAL_STATE"
DATABENTO_OPRA_OHLCV_1M_COMPLETED: Final = "DATABENTO_OPRA_OHLCV_1M_COMPLETED"
DATABENTO_OPRA_TRADES_1M_COMPLETED: Final = (
    "DATABENTO_OPRA_TRADES_AGGREGATED_TO_1M_COMPLETED"
)
DATABENTO_OPRA_STATISTICS_EOD: Final = "DATABENTO_OPRA_STATISTICS_EOD"

BAR_OPEN_TIMESTAMP: Final = "BAR_OPEN_TIMESTAMP"
AVAILABLE_AT_EVENT_PLUS_60_SECONDS: Final = "event_time+60_seconds"
SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION: Final = (
    "latest_available_at_less_than_or_equal_to_decision_time"
)
SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF: Final = (
    "exact_completed_interval_or_zero_after_healthy_frozen_cutoff"
)


@dataclass(frozen=True)
class FeatureLiveTwinRecord:
    """One explicit field-level parity classification."""

    namespace: str
    feature_name: str
    family: str
    historical_source: str
    permitted_intraday_sources: tuple[str, ...]
    clock_semantics: str
    selection_rule: str
    carry_policy: str
    max_feature_age_seconds: int | None
    cross_session_carry: bool
    live_twin_class: str
    prior_day_static_class: str | None
    adapter_receipt_required_before_fit: bool
    recommended_action: str


@dataclass(frozen=True)
class IntradayLiveTwinBinding:
    """Concrete adapter receipt fields checked before an intraday feature may fit.

    ``max_feature_age_seconds`` is the maximum age enforced by the adapter, not
    the age of one sample.  Runtime observations must also satisfy that bound.
    """

    namespace: str
    feature_name: str
    live_source: str
    adapter: str
    event_time_semantics: str
    available_at_rule: str
    selection_rule: str
    selected_available_at_ns: int
    decision_time_ns: int
    completed_only: bool
    latest_exact_contract: bool
    max_feature_age_seconds: int
    cross_session_carry: bool
    missing_exact_interval_is_zero_after_cutoff: bool
    adapter_implementation_sha256: str
    adapter_receipt_sha256: str


ENTRY17_FEATURE_NAMES: Final = (
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
    "D.near_atm.straddle_mid_spot_bps",
    "D.near_atm.put_call_mid_ratio",
    "D.near_atm.side_smile_slope_bps_per_5pt",
    "E.bs.delta",
    "E.bs.gamma",
)

_ENTRY_OFFICIAL_SPX = (
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
)
_ENTRY_OPTION_CBBO = (
    "D.near_atm.straddle_mid_spot_bps",
    "D.near_atm.put_call_mid_ratio",
    "D.near_atm.side_smile_slope_bps_per_5pt",
)
_ENTRY_SELF_GREEKS = ("E.bs.delta", "E.bs.gamma")


EXIT49_FEATURE_NAMES: Final = (
    "option_bid",
    "option_ask",
    "option_mid",
    "option_spread",
    "option_spread_over_mid",
    "option_bid_size",
    "option_ask_size",
    "option_size_imbalance",
    "option_quote_age_ms",
    "option_log_mid_return_1s",
    "option_log_mid_return_5s",
    "option_log_mid_return_15s",
    "option_log_mid_return_30s",
    "option_log_mid_return_60s",
    "option_spread_mean_5s",
    "option_spread_mean_15s",
    "option_spread_mean_60s",
    "option_spread_std_15s",
    "option_spread_std_60s",
    "option_imbalance_mean_5s",
    "option_imbalance_mean_15s",
    "option_imbalance_mean_60s",
    "last_causal_minute_volume",
    "last_causal_open_interest",
    "official_spx_close",
    "official_spx_log_return_1m",
    "official_spx_log_return_5m",
    "official_spx_log_return_15m",
    "official_spx_vwap_gap_bps",
    "self_computed_iv",
    "self_computed_delta",
    "self_computed_gamma",
    "self_computed_iv_change_5s",
    "self_computed_iv_change_30s",
    "self_computed_delta_change_30s",
    "self_computed_gamma_change_30s",
    "held_right_is_call",
    "held_strike_offset_points",
    "entry_fill_option_price",
    "current_net_pnl_dollars",
    "current_return_on_entry_premium",
    "mfe_dollars_to_now",
    "mae_dollars_to_now",
    "giveback_dollars_to_now",
    "seconds_held",
    "seconds_to_15:55",
    "position_occupancy",
    "remaining_d48_budget_dollars",
    "realized_session_pnl_dollars",
)

_EXIT_OPTION_CBBO = EXIT49_FEATURE_NAMES[:22]
_EXIT_MINUTE_VOLUME = ("last_causal_minute_volume",)
_EXIT_OPEN_INTEREST = ("last_causal_open_interest",)
_EXIT_OFFICIAL_SPX = EXIT49_FEATURE_NAMES[24:29]
_EXIT_SELF_GREEKS = EXIT49_FEATURE_NAMES[29:36]
_EXIT_CAUSAL_STATE = EXIT49_FEATURE_NAMES[36:]

EXIT48_INTERMEDIATE_FEATURE_NAMES: Final = tuple(
    name for name in EXIT49_FEATURE_NAMES if name != "last_causal_open_interest"
)

# The intermediate exit-48 list is retained as explicit history.  It is not the
# current fit contract because no concrete minute-volume adapter receipt exists.
EXIT47_CORRECTED_FEATURE_NAMES: Final = tuple(
    name
    for name in EXIT49_FEATURE_NAMES
    if name not in {"last_causal_minute_volume", "last_causal_open_interest"}
)


def _records(
    *,
    namespace: str,
    names: tuple[str, ...],
    family: str,
    historical_source: str,
    permitted_intraday_sources: tuple[str, ...],
    clock_semantics: str,
    selection_rule: str,
    carry_policy: str,
    max_feature_age_seconds: int | None,
    cross_session_carry: bool = False,
    live_twin_class: str = LIVE_DERIVABLE_PENDING_ADAPTER,
    prior_day_static_class: str | None = None,
    adapter_receipt_required_before_fit: bool = True,
    recommended_action: str = KEEP_AFTER_EXACT_ADAPTER_RECEIPT,
) -> tuple[FeatureLiveTwinRecord, ...]:
    return tuple(
        FeatureLiveTwinRecord(
            namespace=namespace,
            feature_name=name,
            family=family,
            historical_source=historical_source,
            permitted_intraday_sources=permitted_intraday_sources,
            clock_semantics=clock_semantics,
            selection_rule=selection_rule,
            carry_policy=carry_policy,
            max_feature_age_seconds=max_feature_age_seconds,
            cross_session_carry=cross_session_carry,
            live_twin_class=live_twin_class,
            prior_day_static_class=prior_day_static_class,
            adapter_receipt_required_before_fit=adapter_receipt_required_before_fit,
            recommended_action=recommended_action,
        )
        for name in names
    )


def _ordered(
    names: tuple[str, ...], records: tuple[FeatureLiveTwinRecord, ...]
) -> tuple[FeatureLiveTwinRecord, ...]:
    by_name = {record.feature_name: record for record in records}
    if len(by_name) != len(records) or set(by_name) != set(names):
        raise RuntimeError("live-twin inventory is duplicated or incomplete")
    return tuple(by_name[name] for name in names)


ENTRY17_LIVE_TWIN_INVENTORY: Final = _ordered(
    ENTRY17_FEATURE_NAMES,
    _records(
        namespace="entry",
        names=_ENTRY_OFFICIAL_SPX,
        family="official_spx_completed_minute_derivation",
        historical_source="THETADATA_OFFICIAL_SPX_1M_HISTORY",
        permitted_intraday_sources=(THETADATA_SPX_1M_COMPLETED,),
        clock_semantics=(
            "ThetaData event_time is bar-open; available_at=event_time+60 seconds; "
            "latest completed same-session minute at or before decision"
        ),
        selection_rule=SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
        carry_policy="same-session completed minute only; age<=90 seconds",
        max_feature_age_seconds=90,
    )
    + _records(
        namespace="entry",
        names=_ENTRY_OPTION_CBBO,
        family="opra_cbbo_completed_minute_ladder_derivation",
        historical_source=DATABENTO_OPRA_CBBO_1M,
        permitted_intraday_sources=(
            DATABENTO_OPRA_CBBO_1M_COMPLETED,
            DATABENTO_OPRA_CBBO_1S_TO_1M,
            DATABENTO_OPRA_CMBP_1_TO_1M,
        ),
        clock_semantics=(
            "historical CBBO-1m timestamp marks the completed interval end; live must "
            "use direct completed CBBO-1m or reproduce the exact one-minute "
            "consolidation, contract identity, and ladder-boundary sampling"
        ),
        selection_rule=SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
        carry_policy="same-session latest completed exact-contract minute; no cross-session carry",
        max_feature_age_seconds=90,
    )
    + _records(
        namespace="entry",
        names=_ENTRY_SELF_GREEKS,
        family="self_computed_greeks",
        historical_source="SELF_COMPUTED_FROM_CAUSAL_OPTION_MID_AND_OFFICIAL_SPX",
        permitted_intraday_sources=(SELF_COMPUTED_CAUSAL,),
        clock_semantics=(
            "same pure calculation and constants over causal option mid, completed "
            "official SPX, contract geometry, and decision time"
        ),
        selection_rule="derive_only_after_all_inputs_pass_their_causal_selection_rules",
        carry_policy="no independent carry; inherits causal input clocks",
        max_feature_age_seconds=None,
    ),
)


EXIT49_LIVE_TWIN_INVENTORY: Final = _ordered(
    EXIT49_FEATURE_NAMES,
    _records(
        namespace="exit",
        names=_EXIT_OPTION_CBBO,
        family="opra_cbbo_current_and_rolling",
        historical_source="DATABENTO_OPRA_CBBO_1S",
        permitted_intraday_sources=(
            DATABENTO_OPRA_CBBO_1S,
            DATABENTO_OPRA_CMBP_1_TO_1S,
        ),
        clock_semantics=(
            "closed 1-second intervals; receipt timestamp <= decision; exact-contract "
            "quote, size, freshness, and rolling-window semantics"
        ),
        selection_rule=SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
        carry_policy="same-session current quote age<=2 seconds; closed rolling history<=60 seconds",
        max_feature_age_seconds=2,
    )
    + _records(
        namespace="exit",
        names=_EXIT_MINUTE_VOLUME,
        family="opra_completed_minute_trade_volume",
        historical_source="DATABENTO_OPRA_OHLCV_1M",
        permitted_intraday_sources=(
            DATABENTO_OPRA_OHLCV_1M_COMPLETED,
            DATABENTO_OPRA_TRADES_1M_COMPLETED,
        ),
        clock_semantics=(
            "ts_event is bar-open; available only after ts_event+60 seconds and the "
            "frozen receipt cutoff; use the exact completed contract-minute bar or "
            "synthesize zero after a healthy cutoff when no bar prints"
        ),
        selection_rule=SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF,
        carry_policy="exact contract-minute only; never carry a prior minute's volume",
        max_feature_age_seconds=90,
        recommended_action=DROP_UNTIL_EXACT_ADAPTER_RECEIPT,
    )
    + _records(
        namespace="exit",
        names=_EXIT_OPEN_INTEREST,
        family="opra_statistics_open_interest",
        historical_source=DATABENTO_OPRA_STATISTICS_EOD,
        permitted_intraday_sources=(),
        clock_semantics=(
            "daily/EOD statistics only; no realtime update and no intraday carry"
        ),
        selection_rule="prior_session_eod_only_if_used_as_one_static_session_value",
        carry_policy="intraday carry forbidden; prior-day static is a separate admissible class",
        max_feature_age_seconds=None,
        live_twin_class=NO_INTRADAY_LIVE_TWIN,
        prior_day_static_class=PRIOR_DAY_EOD_STATIC,
        adapter_receipt_required_before_fit=False,
        recommended_action=DROP,
    )
    + _records(
        namespace="exit",
        names=_EXIT_OFFICIAL_SPX,
        family="official_spx_completed_minute_derivation",
        historical_source="THETADATA_OFFICIAL_SPX_1M_HISTORY",
        permitted_intraday_sources=(THETADATA_SPX_1M_COMPLETED,),
        clock_semantics=(
            "ThetaData event_time is bar-open; available_at=event_time+60 seconds; "
            "latest completed same-session minute, age<=90 seconds"
        ),
        selection_rule=SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION,
        carry_policy="same-session completed minute only; age<=90 seconds",
        max_feature_age_seconds=90,
    )
    + _records(
        namespace="exit",
        names=_EXIT_SELF_GREEKS,
        family="self_computed_greeks_current_and_changes",
        historical_source="SELF_COMPUTED_FROM_CAUSAL_OPTION_MID_AND_OFFICIAL_SPX",
        permitted_intraday_sources=(SELF_COMPUTED_CAUSAL,),
        clock_semantics=(
            "same pure calculation and constants over causal option mid, completed "
            "official SPX, contract geometry, time, and closed causal history"
        ),
        selection_rule="derive_only_after_all_inputs_pass_their_causal_selection_rules",
        carry_policy="no independent carry; current and change features inherit causal input clocks",
        max_feature_age_seconds=None,
    )
    + _records(
        namespace="exit",
        names=_EXIT_CAUSAL_STATE,
        family="position_clock_and_account_state",
        historical_source=INTERNAL_CAUSAL_STATE,
        permitted_intraday_sources=(INTERNAL_CAUSAL_STATE,),
        clock_semantics=(
            "execution ledger, exact held identity, timer, and position/account path "
            "through the current decision only; no future path or label fields"
        ),
        selection_rule="derive_from_the_current_hash_chained_ledger_tip_only",
        carry_policy="same-session causal state only; reset or terminalize at session boundary",
        max_feature_age_seconds=None,
        live_twin_class=LIVE_NATIVE_CAUSAL_STATE,
        recommended_action=KEEP_CAUSAL_INTERNAL_STATE,
    ),
)

EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY: Final = tuple(
    record
    for record in EXIT49_LIVE_TWIN_INVENTORY
    if record.feature_name != "last_causal_open_interest"
)
EXIT47_CORRECTED_LIVE_TWIN_INVENTORY: Final = tuple(
    record
    for record in EXIT49_LIVE_TWIN_INVENTORY
    if record.feature_name
    not in {"last_causal_minute_volume", "last_causal_open_interest"}
)


def live_twin_record(namespace: str, feature_name: str) -> FeatureLiveTwinRecord:
    """Return one exact inventory row, rejecting unregistered aliases."""

    if namespace == "entry":
        inventory = ENTRY17_LIVE_TWIN_INVENTORY
    elif namespace == "exit":
        inventory = EXIT49_LIVE_TWIN_INVENTORY
    else:
        raise ValueError("namespace must be entry or exit")
    matches = [record for record in inventory if record.feature_name == feature_name]
    if len(matches) != 1:
        raise ValueError("feature is absent or duplicated in live-twin inventory")
    return matches[0]


def validate_intraday_live_twin_binding(
    binding: IntradayLiveTwinBinding,
) -> FeatureLiveTwinRecord:
    """Fail closed unless an adapter receipt proves a semantic intraday twin.

    A fabricated nonempty adapter string cannot override a field classified as
    daily/EOD.  Completed option minute volume receives additional clock,
    identity, age, and session checks because these are the conditions under
    which its historical value is live-derivable.
    """

    if type(binding) is not IntradayLiveTwinBinding:
        raise TypeError("intraday live-twin binding must use the exact receipt type")
    record = live_twin_record(binding.namespace, binding.feature_name)
    if record.live_twin_class == NO_INTRADAY_LIVE_TWIN:
        raise ValueError(
            f"{binding.feature_name} has no intraday live twin; "
            "adapter names cannot override EOD semantics"
        )
    if not isinstance(binding.adapter, str) or not binding.adapter.strip():
        raise ValueError("concrete live adapter is required")
    for label, digest in (
        ("adapter implementation", binding.adapter_implementation_sha256),
        ("adapter receipt", binding.adapter_receipt_sha256),
    ):
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} SHA-256 is required")
    if binding.live_source not in record.permitted_intraday_sources:
        raise ValueError("live source is not semantically permitted for feature")
    if type(binding.completed_only) is not bool or not binding.completed_only:
        raise ValueError("intraday features must consume completed observations only")
    if type(binding.cross_session_carry) is not bool or binding.cross_session_carry:
        raise ValueError("cross-session feature carry is forbidden")
    expected_selection = (
        SELECT_EXACT_COMPLETED_INTERVAL_OR_ZERO_AFTER_CUTOFF
        if binding.feature_name == "last_causal_minute_volume"
        else SELECT_LATEST_AVAILABLE_AT_OR_BEFORE_DECISION
    )
    if binding.selection_rule != expected_selection:
        raise ValueError("adapter selection rule does not match the feature contract")
    if (
        type(binding.selected_available_at_ns) is not int
        or type(binding.decision_time_ns) is not int
        or binding.selected_available_at_ns < 0
        or binding.decision_time_ns < 0
        or binding.selected_available_at_ns > binding.decision_time_ns
    ):
        raise ValueError("selected observation is unavailable at decision time")
    if (
        isinstance(binding.max_feature_age_seconds, bool)
        or not isinstance(binding.max_feature_age_seconds, int)
        or binding.max_feature_age_seconds < 0
    ):
        raise ValueError("feature age bound must be a nonnegative integer")
    if (
        binding.decision_time_ns - binding.selected_available_at_ns
        > binding.max_feature_age_seconds * 1_000_000_000
    ):
        raise ValueError("selected observation exceeds the adapter age bound")

    if binding.feature_name == "last_causal_minute_volume":
        if binding.event_time_semantics != BAR_OPEN_TIMESTAMP:
            raise ValueError("minute volume ts_event must be treated as bar-open")
        if binding.available_at_rule != AVAILABLE_AT_EVENT_PLUS_60_SECONDS:
            raise ValueError("minute volume is unavailable before bar close")
        if not binding.latest_exact_contract:
            raise ValueError("minute volume must use the latest exact-contract bar")
        if not binding.missing_exact_interval_is_zero_after_cutoff:
            raise ValueError(
                "sparse minute volume must finalize a missing exact interval as zero"
            )
        if binding.max_feature_age_seconds > 90:
            raise ValueError("minute-volume carry exceeds the 90-second bound")
    return record


def validate_inventory_contracts() -> None:
    """Validate exact coverage and the one-feature correction delta."""

    if tuple(FEATURE_NAMES) != ENTRY17_FEATURE_NAMES:
        raise RuntimeError("signed-17 upstream authority drift")
    if len(ENTRY17_FEATURE_NAMES) != 17 or len(ENTRY17_LIVE_TWIN_INVENTORY) != 17:
        raise RuntimeError("signed-17 inventory count drift")
    if len(EXIT49_FEATURE_NAMES) != 49 or len(EXIT49_LIVE_TWIN_INVENTORY) != 49:
        raise RuntimeError("exit-49 inventory count drift")
    if len(EXIT48_INTERMEDIATE_FEATURE_NAMES) != 48:
        raise RuntimeError("intermediate exit-48 count drift")
    if len(EXIT47_CORRECTED_FEATURE_NAMES) != 47:
        raise RuntimeError("corrected exit-47 count drift")
    if (
        tuple(record.feature_name for record in ENTRY17_LIVE_TWIN_INVENTORY)
        != ENTRY17_FEATURE_NAMES
    ):
        raise RuntimeError("signed-17 inventory order drift")
    entry_option_rows = tuple(
        record
        for record in ENTRY17_LIVE_TWIN_INVENTORY
        if record.feature_name in _ENTRY_OPTION_CBBO
    )
    if (
        len(entry_option_rows) != 3
        or any(
            record.historical_source != DATABENTO_OPRA_CBBO_1M
            or record.permitted_intraday_sources
            != (
                DATABENTO_OPRA_CBBO_1M_COMPLETED,
                DATABENTO_OPRA_CBBO_1S_TO_1M,
                DATABENTO_OPRA_CMBP_1_TO_1M,
            )
            for record in entry_option_rows
        )
    ):
        raise RuntimeError("signed-17 option ladder CBBO-1m lineage drift")
    if tuple(record.feature_name for record in EXIT49_LIVE_TWIN_INVENTORY) != EXIT49_FEATURE_NAMES:
        raise RuntimeError("exit-49 inventory order drift")
    if (
        tuple(record.feature_name for record in EXIT48_INTERMEDIATE_LIVE_TWIN_INVENTORY)
        != EXIT48_INTERMEDIATE_FEATURE_NAMES
    ):
        raise RuntimeError("intermediate exit-48 inventory order drift")
    if (
        tuple(record.feature_name for record in EXIT47_CORRECTED_LIVE_TWIN_INVENTORY)
        != EXIT47_CORRECTED_FEATURE_NAMES
    ):
        raise RuntimeError("corrected exit-47 inventory order drift")
    removed = set(EXIT49_FEATURE_NAMES) - set(EXIT47_CORRECTED_FEATURE_NAMES)
    if removed != {"last_causal_minute_volume", "last_causal_open_interest"}:
        raise RuntimeError("corrected exit inventory must drop OI and unproved minute volume")
    open_interest = live_twin_record("exit", "last_causal_open_interest")
    if (
        open_interest.permitted_intraday_sources
        or open_interest.live_twin_class != NO_INTRADAY_LIVE_TWIN
        or open_interest.prior_day_static_class != PRIOR_DAY_EOD_STATIC
        or open_interest.recommended_action != DROP
    ):
        raise RuntimeError("open-interest live-twin classification drift")
    minute_volume = live_twin_record("exit", "last_causal_minute_volume")
    if (
        minute_volume.live_twin_class != LIVE_DERIVABLE_PENDING_ADAPTER
        or minute_volume.recommended_action != DROP_UNTIL_EXACT_ADAPTER_RECEIPT
        or not minute_volume.adapter_receipt_required_before_fit
        or minute_volume.feature_name in EXIT47_CORRECTED_FEATURE_NAMES
    ):
        raise RuntimeError("minute-volume unresolved adapter disposition drift")
    for record in (*ENTRY17_LIVE_TWIN_INVENTORY, *EXIT49_LIVE_TWIN_INVENTORY):
        if (
            not record.selection_rule
            or not record.carry_policy
            or type(record.cross_session_carry) is not bool
            or record.cross_session_carry
            or (
                record.max_feature_age_seconds is not None
                and (
                    type(record.max_feature_age_seconds) is not int
                    or record.max_feature_age_seconds < 0
                )
            )
        ):
            raise RuntimeError("feature live-twin carry inventory drift")


validate_inventory_contracts()
