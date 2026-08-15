"""Unified serial SPXW 0DTE trading game primitives.

This module is the shared research/runtime contract for the next v4 policy
lineage. It keeps the trading game explicit: one account, one position, one
contract, ask-entry, bid-exit, SPXW PM-settled 0DTE candidates, and no
Protocol101-only pre-entry gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v4.live.protocol166_parity_contract import validate_candidate


CONTRACT_MULTIPLIER = 100.0
SESSION_OPEN_MINUTE = 9 * 60 + 30
NO_NEW_ENTRIES_AFTER_MINUTE = 15 * 60 + 30


@dataclass(frozen=True)
class UnifiedSerialGameConfig:
    role_label: str = "UNIFIED_SERIAL_SPXW_0DTE_GAME_V1"
    symbol_root: str = "SPXW"
    settlement_style: str = "PM"
    strike_spacing: float = 5.0
    strike_window: float = 50.0
    starting_cash: float = 10_000.0
    max_contracts: int = 1
    max_concurrent_positions: int = 1
    entry_price: str = "ask"
    exit_price: str = "bid"
    fees_included: bool = False
    max_option_quote_age_ms: int = 1_500
    max_context_age_ms: int = 5_000
    no_new_entries_after_minute_et: int = NO_NEW_ENTRIES_AFTER_MINUTE
    forced_flat_minute_et: int = 15 * 60 + 55
    disallowed_pre_entry_gates: tuple[str, ...] = (
        "protocol101_min_edge_gate",
        "protocol101_allowed_time_bucket_gate",
        "protocol101_selected_candidate_dependency",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_unified_game_config() -> UnifiedSerialGameConfig:
    return UnifiedSerialGameConfig()


def validate_unified_candidate(
    candidate: dict[str, Any],
    account_state: dict[str, Any],
    *,
    config: UnifiedSerialGameConfig = UnifiedSerialGameConfig(),
) -> dict[str, Any]:
    """Validate one flat-state candidate under the unified game."""

    result = validate_candidate(candidate, account_state)
    errors = list(result.get("errors", []))
    quote_age = _finite(candidate.get("quote_age_ms"), _finite(candidate.get("entry_quote_age_ms"), math.nan))
    context_age = _finite(candidate.get("context_age_ms"), 0.0)
    decision_minute = _minute_et(candidate.get("decision_time"))
    if math.isfinite(quote_age) and quote_age > config.max_option_quote_age_ms:
        errors.append("stale_option_quote")
    elif not math.isfinite(quote_age):
        errors.append("missing_quote_age")
    if math.isfinite(context_age) and context_age > config.max_context_age_ms:
        errors.append("stale_context")
    elif not math.isfinite(context_age):
        errors.append("missing_context_age")
    if decision_minute is not None and decision_minute >= config.no_new_entries_after_minute_et:
        errors.append("after_no_new_entries_cutoff")
    return {"status": "pass" if not errors else "fail", "errors": sorted(set(errors))}


def action_mask_for_candidates(
    candidates: pd.DataFrame,
    account_state: dict[str, Any],
    *,
    config: UnifiedSerialGameConfig = UnifiedSerialGameConfig(),
) -> np.ndarray:
    """Return a live-compatible candidate mask for one flat decision event."""

    mask = []
    for _, row in candidates.iterrows():
        candidate = candidate_dict_from_row(row)
        mask.append(validate_unified_candidate(candidate, account_state, config=config)["status"] == "pass")
    return np.asarray(mask, dtype=bool)


def candidate_dict_from_row(row: Any) -> dict[str, Any]:
    bid = _finite(_get(row, "entry_bid"))
    ask = _finite(_get(row, "entry_ask"))
    mid = _finite(_get(row, "entry_mid"), (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    spread = _finite(_get(row, "entry_spread"), ask - bid if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    return {
        "decision_time": _get(row, "decision_time"),
        "contract_id": str(_get(row, "contract_id") or ""),
        "root": str(_get(row, "root") or ""),
        "settlement_style": str(_get(row, "settlement_style") or ""),
        "right": str(_get(row, "right") or ""),
        "offset": _finite(_get(row, "offset")),
        "entry_bid": bid,
        "entry_ask": ask,
        "entry_mid": mid,
        "entry_spread": spread,
        "entry_bid_size": _finite(_get(row, "entry_bid_size")),
        "entry_ask_size": _finite(_get(row, "entry_ask_size")),
        "entry_premium": _finite(_get(row, "entry_premium"), ask * CONTRACT_MULTIPLIER if math.isfinite(ask) else math.nan),
        "entry_delta": _finite(_get(row, "entry_delta")),
        "entry_gamma": _finite(_get(row, "entry_gamma")),
        "entry_theta": _finite(_get(row, "entry_theta")),
        "entry_iv": _finite(_get(row, "entry_iv")),
        "quote_age_ms": _finite(_get(row, "quote_age_ms"), _quote_age_ms(row)),
        "context_age_ms": _finite(_get(row, "context_age_ms"), 0.0),
    }


def build_flat_action_advantage_labels(
    frame: pd.DataFrame,
    *,
    config: UnifiedSerialGameConfig = UnifiedSerialGameConfig(),
    slippage_per_side: float = 0.0,
) -> pd.DataFrame:
    """Compute serial flat-state Q(wait), Q(enter), and A(enter) labels.

    The input is a full-action candidate frame with one row per candidate and
    `candidate_exit_time` / `candidate_pnl` labels. The DP state is flat only:
    entering a candidate blocks later entries until its exit time.
    """

    required = {"session", "decision_time", "candidate_exit_time", "candidate_pnl", "entry_premium"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required action-advantage columns: {missing}")
    rows: list[pd.DataFrame] = []
    source = frame.copy()
    source["decision_dt"] = pd.to_datetime(source.get("decision_dt", source["decision_time"]), utc=True, errors="coerce")
    source["candidate_exit_dt"] = pd.to_datetime(source.get("candidate_exit_dt", source["candidate_exit_time"]), utc=True, errors="coerce")
    source["candidate_pnl"] = pd.to_numeric(source["candidate_pnl"], errors="coerce")
    source["entry_premium"] = pd.to_numeric(source["entry_premium"], errors="coerce")
    source = source[source["decision_dt"].notna() & source["candidate_exit_dt"].notna() & source["candidate_pnl"].notna()].copy()
    for (split, session), session_frame in source.groupby(["split", "session"], sort=True):
        rows.append(_label_one_session(session_frame, split=str(split), session=str(session), config=config, slippage_per_side=slippage_per_side))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    return out.sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)


def _label_one_session(
    session_frame: pd.DataFrame,
    *,
    split: str,
    session: str,
    config: UnifiedSerialGameConfig,
    slippage_per_side: float,
) -> pd.DataFrame:
    decisions = sorted(pd.to_datetime(session_frame["decision_dt"], utc=True).dropna().unique())
    if not decisions:
        return pd.DataFrame()
    decision_index = {pd.Timestamp(ts): idx for idx, ts in enumerate(decisions)}
    value = np.zeros(len(decisions) + 1, dtype=float)
    best_uid_at_step: list[str] = ["wait"] * len(decisions)
    best_enter_value = np.full(len(decisions), -math.inf, dtype=float)
    q_wait = np.zeros(len(decisions), dtype=float)
    groups = {pd.Timestamp(ts): group.copy() for ts, group in session_frame.groupby("decision_dt", sort=False)}
    for idx in range(len(decisions) - 1, -1, -1):
        ts = pd.Timestamp(decisions[idx])
        wait_value = value[idx + 1]
        q_wait[idx] = wait_value
        best_value = wait_value
        best_uid = "wait"
        group = groups.get(ts, pd.DataFrame())
        for row in group.itertuples(index=False):
            premium = _finite(getattr(row, "entry_premium", math.nan))
            if premium <= 0.0 or premium > config.starting_cash:
                continue
            exit_dt = pd.Timestamp(getattr(row, "candidate_exit_dt"))
            next_idx = _first_decision_after(decisions, exit_dt, current_idx=idx)
            pnl = _finite(getattr(row, "candidate_pnl", 0.0)) - (2.0 * float(slippage_per_side) * CONTRACT_MULTIPLIER)
            enter_value = pnl + value[next_idx]
            uid = str(getattr(row, "candidate_uid", getattr(row, "contract_id", "")))
            if enter_value > best_enter_value[idx]:
                best_enter_value[idx] = enter_value
            if enter_value > best_value:
                best_value = enter_value
                best_uid = uid
        value[idx] = best_value
        best_uid_at_step[idx] = best_uid
    out = session_frame.copy()
    out["q_wait"] = out["decision_dt"].map(lambda ts: float(q_wait[decision_index[pd.Timestamp(ts)]]))
    out["session_oracle_value"] = out["decision_dt"].map(lambda ts: float(value[decision_index[pd.Timestamp(ts)]]))
    out["best_enter_value_at_decision"] = out["decision_dt"].map(lambda ts: float(best_enter_value[decision_index[pd.Timestamp(ts)]]))
    out["oracle_action_uid"] = out["decision_dt"].map(lambda ts: best_uid_at_step[decision_index[pd.Timestamp(ts)]])
    q_enter_values = []
    a_enter_values = []
    next_indices = []
    for row in out.itertuples(index=False):
        idx = decision_index[pd.Timestamp(row.decision_dt)]
        exit_dt = pd.Timestamp(row.candidate_exit_dt)
        next_idx = _first_decision_after(decisions, exit_dt, current_idx=idx)
        pnl = _finite(getattr(row, "candidate_pnl", 0.0)) - (2.0 * float(slippage_per_side) * CONTRACT_MULTIPLIER)
        q_enter = pnl + value[next_idx]
        q_enter_values.append(float(q_enter))
        a_enter_values.append(float(q_enter - q_wait[idx]))
        next_indices.append(int(next_idx))
    out["q_enter"] = q_enter_values
    out["a_enter"] = a_enter_values
    out["next_flat_decision_index"] = next_indices
    out["oracle_action"] = np.where(out["candidate_uid"].astype(str).eq(out["oracle_action_uid"].astype(str)), "enter", "wait")
    out["q_exit_now_entry"] = 0.0
    out["q_hold_entry"] = pd.to_numeric(out["candidate_pnl"], errors="coerce").fillna(0.0)
    out["a_hold_entry"] = out["q_hold_entry"] - out["q_exit_now_entry"]
    out["a_switch_entry"] = out["q_wait"] - out["q_hold_entry"]
    out["label_source"] = "serial_flat_dp_action_advantage_v1"
    out["split"] = split
    out["session"] = session
    return out


def hold_exit_advantage_path(path_bid: Iterable[float], entry_ask: float) -> pd.DataFrame:
    """Return causal path labels for hold-vs-exit-now diagnostics/tests."""

    bid = np.asarray(list(path_bid), dtype=float)
    pnl = (bid - float(entry_ask)) * CONTRACT_MULTIPLIER
    future_best = np.maximum.accumulate(pnl[::-1])[::-1]
    future_worst = np.minimum.accumulate(pnl[::-1])[::-1]
    q_exit = pnl
    q_hold = np.maximum(future_best, q_exit)
    return pd.DataFrame(
        {
            "step": np.arange(len(pnl), dtype=int),
            "q_exit": q_exit,
            "q_hold": q_hold,
            "a_hold": q_hold - q_exit,
            "future_best_pnl": future_best,
            "future_worst_pnl": future_worst,
            "oracle_holding_action": np.where((q_hold - q_exit) > 0.0, "hold", "exit"),
        }
    )


def hold_exit_opportunity_advantage_path(
    path_bid: Iterable[float],
    entry_ask: float,
    future_flat_values: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return hold/exit labels that include future flat-slot opportunity value.

    `q_exit` means sell now at bid and then use the freed slot according to the
    flat-state oracle. `q_hold` means continue holding at least one more path
    step, then choose the best later executable exit plus future flat value.
    """

    bid = np.asarray(list(path_bid), dtype=float)
    if future_flat_values is None:
        flat = np.zeros(len(bid), dtype=float)
    else:
        flat = np.asarray(list(future_flat_values), dtype=float)
    if len(flat) != len(bid):
        raise ValueError("future_flat_values length must match path_bid length")
    pnl = (bid - float(entry_ask)) * CONTRACT_MULTIPLIER
    q_exit = pnl + flat
    q_hold = np.empty(len(q_exit), dtype=float)
    if len(q_exit) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "q_exit",
                "q_hold",
                "a_hold",
                "a_switch",
                "current_pnl",
                "future_best_hold_value",
                "oracle_holding_action",
            ]
        )
    future_best = np.maximum.accumulate(q_exit[::-1])[::-1]
    for idx in range(len(q_exit)):
        q_hold[idx] = future_best[idx + 1] if idx + 1 < len(q_exit) else q_exit[idx]
    a_hold = q_hold - q_exit
    return pd.DataFrame(
        {
            "step": np.arange(len(pnl), dtype=int),
            "q_exit": q_exit,
            "q_hold": q_hold,
            "a_hold": a_hold,
            "a_switch": q_exit - q_hold,
            "current_pnl": pnl,
            "future_best_hold_value": q_hold,
            "oracle_holding_action": np.where(a_hold > 0.0, "hold", "exit"),
        }
    )


def _first_decision_after(decisions: list[pd.Timestamp], exit_dt: pd.Timestamp, *, current_idx: int) -> int:
    lo = current_idx + 1
    hi = len(decisions)
    while lo < hi:
        mid = (lo + hi) // 2
        if decisions[mid] <= exit_dt:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _quote_age_ms(row: Any) -> float:
    decision = pd.to_datetime(_get(row, "decision_time"), utc=True, errors="coerce")
    quote = pd.to_datetime(_get(row, "entry_quote_time"), utc=True, errors="coerce")
    if pd.isna(decision) or pd.isna(quote):
        return math.nan
    return max(0.0, float((decision - quote).total_seconds() * 1000.0))


def _minute_et(value: Any) -> int | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    local = pd.Timestamp(ts).tz_convert("America/New_York")
    return int(local.hour * 60 + local.minute)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, key):
        return getattr(row, key)
    try:
        return row[key]
    except Exception:
        return None
