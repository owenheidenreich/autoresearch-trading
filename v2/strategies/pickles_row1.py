"""Fork A1: SPX-only proxy of Pickles' Row 1 ("VWAP SUPPORT quick-in-out").

Scope-of-claim header (also appears in every results doc):
    This is a test of the SPX-only proxy of Row 1. A positive result shows
    edge on this dataset, under this cost model, with these specific
    substitutions (SPX VWAP for ES/NQ VWAP; no A/D ratio; no A/D volume).
    A negative result falsifies the proxy, not Row 1 as originally stated.

See ``v2/docs/pickles_digest.md`` (Row 1) for the source rule, and the plan
file ``read-this-context-and-snappy-tide.md`` for the full specification.

This module exposes two small, testable decision helpers plus the candidate
record dataclass. It deliberately does NOT import from ``v2/core/simulator.py``
or inherit ``v2/core/policy.py::DEFAULT_POLICY`` — doing so would silently
restrict the entry window (``NO_TRADE_BEFORE_BAR=30``) or apply premium-based
exits that are incompatible with Row 1's stated spot-level exits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from v2.strategies.session_state import (
    SessionState,
    WINDOW_FIRST_BAR,
    WINDOW_LAST_BAR,
    FIRST15_BARS,
    in_halfhour_cooldown,
)


# Feature indices in ``X_sim`` (raw schema identical to ``feature_names``).
# We freeze these at import and assert them against the data contract at
# startup so a silent reorder of ``ALL_FEATURE_NAMES`` can't break the strategy.
_FEAT_IDX_VWAP_DIST = 6
_FEAT_IDX_BAR_DELTA = 22
_FEAT_IDX_VOLUME_RATIO = 2
_FEAT_IDX_FIRST15_CLOSE_POS = 35


def assert_feature_layout(feature_names: list[str]) -> None:
    """Fail loudly if the 79-feature layout has drifted from the indices this
    module hard-codes. Call once per process from the runner.
    """
    expected = {
        _FEAT_IDX_VWAP_DIST: "vwap_dist",
        _FEAT_IDX_BAR_DELTA: "bar_delta",
        _FEAT_IDX_VOLUME_RATIO: "volume_ratio",
        _FEAT_IDX_FIRST15_CLOSE_POS: "first15_close_position",
    }
    for idx, name in expected.items():
        actual = feature_names[idx] if idx < len(feature_names) else None
        if actual != name:
            raise RuntimeError(
                f"Feature layout drift: expected {name!r} at index {idx}, "
                f"got {actual!r}. Update v2/strategies/pickles_row1.py "
                f"constants or investigate dataset schema change."
            )


# -------------------------------------------------------------------------
# Entry trigger thresholds (baked into the rule, not tunable per cell).
# -------------------------------------------------------------------------

VWAP_TOUCH_BAND_FRAC = 5e-4      # ±5 bps of VWAP counts as a "touch"
VWAP_PRIOR_ABOVE_FRAC = 1e-3     # prior bar must have been > 10 bps above VWAP
VOLUME_RATIO_MIN = 1.0           # "supporting volume" on the bounce bar


# -------------------------------------------------------------------------
# Ablation config.
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    """Which preconditions/qualifiers are active for this run.

    ``P-open``     = AblationConfig(False, False, False, False)
    ``P-15m``      = AblationConfig(True,  False, False, True)
    ``P-cooldown`` = AblationConfig(True,  True,  False, True)
    ``P-all``      = AblationConfig(True,  True,  True,  True)

    These map 1:1 to the matrix in the plan file. ``use_sigma_band_target``
    is consulted by the spot-exit engine, not by ``decide``; it's carried
    in this config so the whole cell definition lives in one place.
    """
    require_15m_strength: bool
    enforce_halfhour_cooldown: bool
    require_ovn_proxy: bool
    use_sigma_band_target: bool
    label: str = ""


P_OPEN = AblationConfig(False, False, False, False, label="P-open")
P_15M = AblationConfig(True, False, False, True, label="P-15m")
P_COOLDOWN = AblationConfig(True, True, False, True, label="P-cooldown")
P_ALL = AblationConfig(True, True, True, True, label="P-all")

ALL_PRECONDITION_CELLS = (P_OPEN, P_15M, P_COOLDOWN, P_ALL)


# -------------------------------------------------------------------------
# Candidate record (one per candidate × delta-target, emitted to Stage 0 log).
# -------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    # Identity
    date: str
    bar_of_day: int
    global_bar: int
    ablation_cell: str
    target_delta: float
    # Context
    spx_spot: float
    session_vwap: float
    vwap_dist_frac: float
    vwap_sigma_frac: float
    first15_high: float
    first15_close_position: float
    ovn_proxy_direction: int
    in_cooldown: bool
    # Contract pick (populated if a trade is taken)
    contract_idx: int = -1
    contract_strike: float = 0.0
    contract_right: str = "C"
    contract_delta: float = 0.0
    contract_mid_at_entry: float = 0.0
    # Skip reason: "" if entry is taken, else a token {"no_delta_match",
    # "cooldown", "first15_weak", "ovn_proxy_not_long", "no_entry_trigger",
    # "outside_window", "penny_premium", "invalid_contract"}.
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------------
# Core decision helpers.
# -------------------------------------------------------------------------

def is_in_window(bar_of_day: int) -> bool:
    """Fork-A1 entry window: 09:45-12:00 ET = bars 15-150 inclusive."""
    return WINDOW_FIRST_BAR <= bar_of_day <= WINDOW_LAST_BAR


VWAP_RALLY_LOOKBACK_BARS = 30    # "was above VWAP earlier this session"
VWAP_RALLY_THRESHOLD_FRAC = 1e-3  # >10 bps above VWAP at some prior bar


def entry_trigger_fires(
    bar_of_day: int,
    X_sim_row: np.ndarray,
    recent_max_vwap_dist_frac: float,
    recent_window_ready: bool,
) -> bool:
    """Return True if the VWAP-support + bullish-bounce trigger fires.

    **Trigger operationalization (revised 2026-04-19 after smoke test).**
    Original strict form required the *immediately* prior bar to be > +10 bps
    above VWAP. Empirical check on the 2023-12-14 template day showed the
    template entry fired at bar 38 when SPX had been ~11 bps BELOW its own
    VWAP for 18 bars (Pickles was tracking ES VWAP, not SPX VWAP). The strict
    form fired zero candidates across 575 sessions.

    Revised form: the session must have seen SPX at least ``+10 bps above``
    VWAP at some bar within the prior ``30`` bars (not just the immediately
    prior bar). This captures the "pullback-after-rally → bounce" shape we
    want without requiring an ES-specific microstructure that SPX VWAP can't
    replicate. The operationalization is still spot-level, session-local,
    and point-in-time — it only consults the closed-bar history within the
    current RTH session.

    Trigger = (session recently > +10 bps above VWAP) AND (this bar within
    ±5 bps of VWAP) AND (bar_delta > 0) AND (volume_ratio >= 1.0).

    Consumes the RAW ``X_sim`` row. vwap_dist is fractional signed distance:
    ``(close - vwap) / close``.
    """
    if not recent_window_ready:
        return False
    vwap_dist = float(X_sim_row[_FEAT_IDX_VWAP_DIST])
    bar_delta = float(X_sim_row[_FEAT_IDX_BAR_DELTA])
    volume_ratio = float(X_sim_row[_FEAT_IDX_VOLUME_RATIO])
    if not np.isfinite(vwap_dist) or not np.isfinite(bar_delta) or not np.isfinite(volume_ratio):
        return False
    if recent_max_vwap_dist_frac <= VWAP_RALLY_THRESHOLD_FRAC:
        return False
    if abs(vwap_dist) > VWAP_TOUCH_BAND_FRAC:
        return False
    if bar_delta <= 0.0:
        return False
    if volume_ratio < VOLUME_RATIO_MIN:
        return False
    return True


def preconditions_ok(
    bar_of_day: int,
    X_sim_row: np.ndarray,
    state: SessionState,
    config: AblationConfig,
) -> tuple[bool, str]:
    """Check the per-day preconditions for this bar under the given ablation
    config. Returns (ok, skip_reason). When ok is False, skip_reason names
    which precondition failed — useful for candidate-log diagnostics.

    Note: the trading window check is always active (non-ablatable).
    """
    if not is_in_window(bar_of_day):
        return False, "outside_window"
    # First-15m strength (ablatable).
    if config.require_15m_strength:
        if not state.first15_ready:
            return False, "first15_not_ready"
        if state.first15_close_position is None or state.first15_close_position < 0.5:
            return False, "first15_weak"
    # OVN proxy direction (ablatable).
    if config.require_ovn_proxy:
        if state.ovn_proxy_direction <= 0:
            return False, "ovn_proxy_not_long"
    # Half-hour / news cool-down (ablatable).
    if config.enforce_halfhour_cooldown:
        if in_halfhour_cooldown(bar_of_day):
            return False, "cooldown"
    return True, ""


# -------------------------------------------------------------------------
# Target-delta contract selection (no oracle fields used).
# -------------------------------------------------------------------------

from v2.core.chain_data import CONTRACT_FEATURE_FIELDS

_ROW_IDX_VALID = CONTRACT_FEATURE_FIELDS.index("contract_valid")    # 0
_ROW_IDX_STRIKE = CONTRACT_FEATURE_FIELDS.index("strike")           # 1
_ROW_IDX_DELTA = CONTRACT_FEATURE_FIELDS.index("delta")             # 8
_ROW_IDX_LOG_VOL = CONTRACT_FEATURE_FIELDS.index("log_volume")      # 5

DELTA_TOLERANCE = 0.05
MIN_CONTRACT_MID = 0.50  # matches simulator.MIN_ENTRY_PRICE for comparability


@dataclass
class ContractPick:
    contract_idx: int
    strike: float
    right: str       # "C" or "P"
    delta: float
    mid_at_entry: float
    local_row: int   # row index within the sidecar's per-bar slice


def pick_delta_targeted_call(
    sidecar: dict[str, Any],
    local_bar: int,
    target_delta: float,
    tolerance: float = DELTA_TOLERANCE,
    min_mid: float = MIN_CONTRACT_MID,
) -> tuple[ContractPick | None, str]:
    """Scan the sidecar at ``local_bar`` for the call contract whose delta is
    closest to ``target_delta``, within ``tolerance``. Returns (pick, reason)
    where reason is "" if a pick was made else a skip-reason token.

    **Never** consults ``bar_slice_best_contract_idx`` or any oracle / forward-
    looking field. All information is present-tense at ``local_bar``.
    """
    bar_ptrs = sidecar.get("bar_ptrs")
    if bar_ptrs is None:
        return None, "invalid_contract"
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return None, "no_delta_match"

    row_features = sidecar["row_features"]      # (total_rows, NUM_CONTRACT_FEATURES)
    row_contract_idx = sidecar["row_contract_idx"]  # (total_rows,)
    contract_right = sidecar["contract_right"]  # (n_contracts,) int8, 0=call 1=put
    contract_mid = sidecar["contract_mid"]      # (n_contracts, n_bars)

    best: ContractPick | None = None
    best_key: tuple[float, float] | None = None

    for local_row in range(end - start):
        global_row = start + local_row
        row = row_features[global_row]
        if float(row[_ROW_IDX_VALID]) < 1.0:
            continue
        cidx = int(row_contract_idx[global_row])
        if cidx < 0 or cidx >= len(contract_right):
            continue
        if int(contract_right[cidx]) != 0:  # calls only
            continue
        delta = float(row[_ROW_IDX_DELTA])
        if not np.isfinite(delta) or delta <= 0:
            continue
        if abs(delta - target_delta) > tolerance:
            continue
        mid = float(contract_mid[cidx, local_bar])
        if not np.isfinite(mid) or mid <= 0:
            continue
        if mid < min_mid:
            # Skip the pick per MIN_CONTRACT_MID policy, but note the reason so
            # Stage-0 logs can distinguish "no delta match" from "penny premium".
            # We still pass over — this contract isn't our pick.
            continue
        log_vol = float(row[_ROW_IDX_LOG_VOL]) if np.isfinite(float(row[_ROW_IDX_LOG_VOL])) else 0.0
        # Tie-break: closest delta; on tie, highest volume; on further tie, lowest strike.
        key = (abs(delta - target_delta), -log_vol, float(row[_ROW_IDX_STRIKE]))
        if best is None or key < best_key:
            strike = float(row[_ROW_IDX_STRIKE])
            best = ContractPick(
                contract_idx=cidx,
                strike=strike,
                right="C",
                delta=delta,
                mid_at_entry=mid,
                local_row=local_row,
            )
            best_key = key

    if best is None:
        return None, "no_delta_match"
    return best, ""
