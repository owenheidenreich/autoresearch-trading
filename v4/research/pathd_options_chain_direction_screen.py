"""Frozen H1-only raw options-chain direction screen.

This module contains no estimator, fit, null, option payoff, or paper action. It
measures whether a five-minute change in symmetric SPXW 0DTE risk reversal
predicts the next five-minute signed SPX move under strict serial occupancy.

Frozen by PATHD_OPTIONS_CHAIN_H1_PREREGISTRATION_2026_08_04.md.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, time
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.research.pathd_opra_parity_features import (
    DIVIDEND_YIELD,
    RISK_FREE_RATE,
    ParityPair,
    parity_snapshot,
    solve_implied_volatility,
    _time_to_expiry_years,
)
from v4.research.pathd_phase1_entry import (
    DEVELOPMENT_SESSION_COUNT,
    ENTRY_EMISSION_LAG_MS,
    _load_parquet,
    _official_spx,
    _parse_symbol,
    development_sessions,
    entry_expanding_folds,
)
from v4.research.pathd_research_loop import prior_art_check
from v4.research.pathd_spx_directional_skill_screen import (
    _availability_series,
    _close_available_at,
    _sha256_path,
)
from v4.research.phase1_exit_model import stable_hash


SCHEMA_VERSION = "pathd.options-chain-h1-direction-screen.v1"
HYPOTHESIS_ID = "H1_SYMMETRIC_RISK_REVERSAL_CHANGE_5M"
MECHANISM = (
    "five-minute symmetric SPXW 0DTE risk-reversal change predicts "
    "next-five-minute signed SPX points"
)
EXACT_PRIOR_ART_TERMS = (
    "risk reversal",
    "skew change",
    "option-surface direction",
    "implied volatility skew",
    "put skew",
    "smile slope",
)
BROAD_PRIOR_ART_TERMS = (
    "skew",
    "smile",
    "volatility surface",
    "put-call",
    "put call",
)

NY = ZoneInfo("America/New_York")
MINUTE_NS = 60_000_000_000
HORIZON_MINUTES = 5
HORIZON_NS = HORIZON_MINUTES * MINUTE_NS
WING_OFFSET_POINTS = 20.0
MAX_BRACKET_GAP_POINTS = 5.0
QUOTE_STALENESS_NS = 90_000_000_000
FIRST_BOUNDARY = time(10, 0)
LAST_BOUNDARY = time(15, 50)
PRIOR_BOUNDARY = time(9, 55)
MIN_BOUNDARIES_PER_SESSION = 30
MAX_DROPPED_SESSIONS = 10
MIN_EVALUATION_SESSIONS_PER_FOLD = 30

CORPUS_ROOTS = (
    Path("/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31"),
    Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO_ROOT / (
    "v4/docs/protocol101/training/research/"
    "PATHD_OPTIONS_CHAIN_H1_PREREGISTRATION_2026_08_04.md"
)
OUTPUT_ROOT = REPO_ROOT / (
    "v4/audit/autoresearch/pathd_options_chain_h1_direction_screen_2026_08_04"
)


class OptionsChainScreenError(RuntimeError):
    """The frozen experiment contract is invalid or cannot be reconstructed."""


@dataclass(frozen=True)
class InterpolatedWingIV:
    right: str
    target_strike: float
    lower_strike: float
    upper_strike: float
    lower_symbol: str
    upper_symbol: str
    lower_iv: float
    upper_iv: float
    upper_weight: float
    interpolated_iv: float


@dataclass(frozen=True)
class RiskReversalSnapshot:
    decision_time: pd.Timestamp
    implied_spot: float
    risk_reversal: float
    put_wing: InterpolatedWingIV
    call_wing: InterpolatedWingIV
    parity_pair_ids: tuple[str, ...]


def _corpus_root() -> Path:
    for candidate in CORPUS_ROOTS:
        if (candidate / "raw/databento/opra_spxw_cbbo_1m").is_dir():
            return candidate
    raise OptionsChainScreenError(
        f"no Path-D corpus found at any of {[str(path) for path in CORPUS_ROOTS]}"
    )


def _boundary_grid_ns(session: str, start: time, end: time) -> np.ndarray:
    day = datetime.fromisoformat(session).date()
    first = int(pd.Timestamp(datetime.combine(day, start, tzinfo=NY)).tz_convert("UTC").value)
    last = int(pd.Timestamp(datetime.combine(day, end, tzinfo=NY)).tz_convert("UTC").value)
    if first > last or (last - first) % MINUTE_NS:
        raise OptionsChainScreenError(f"invalid minute boundary grid for {session}")
    return np.arange(first, last + MINUTE_NS, MINUTE_NS, dtype=np.int64)


def _solve_pair_iv(
    pair: ParityPair,
    *,
    right: str,
    implied_spot: float,
    years: float,
) -> float:
    price = pair.put_mid if right == "P" else pair.call_mid
    return solve_implied_volatility(
        price=float(price),
        spot=float(implied_spot),
        strike=float(pair.strike),
        years=float(years),
        right=right,
        rate=RISK_FREE_RATE,
        dividend=DIVIDEND_YIELD,
    )


def _interpolated_wing(
    pairs: Sequence[ParityPair],
    *,
    right: str,
    implied_spot: float,
    years: float,
    target_strike: float,
) -> InterpolatedWingIV:
    if right not in {"C", "P"}:
        raise ValueError("wing right must be C or P")
    ordered = tuple(sorted(pairs, key=lambda pair: (pair.strike, pair.call_symbol, pair.put_symbol)))
    if not ordered:
        raise ValueError("no parity pairs for wing interpolation")
    strikes = np.asarray([pair.strike for pair in ordered], dtype=float)
    exact = np.flatnonzero(np.isclose(strikes, target_strike, rtol=0.0, atol=1e-10))
    if len(exact):
        lower_index = upper_index = int(exact[0])
    else:
        lower = np.flatnonzero(strikes < target_strike)
        upper = np.flatnonzero(strikes > target_strike)
        if not len(lower) or not len(upper):
            raise ValueError("wing target requires extrapolation")
        lower_index = int(lower[-1])
        upper_index = int(upper[0])
    lower_pair = ordered[lower_index]
    upper_pair = ordered[upper_index]
    gap = float(upper_pair.strike - lower_pair.strike)
    if lower_index != upper_index and (
        gap <= 0.0 or gap > MAX_BRACKET_GAP_POINTS + 1e-10
    ):
        raise ValueError("wing interpolation bracket is wider than five points")
    lower_iv = _solve_pair_iv(
        lower_pair,
        right=right,
        implied_spot=implied_spot,
        years=years,
    )
    upper_iv = (
        lower_iv
        if lower_index == upper_index
        else _solve_pair_iv(
            upper_pair,
            right=right,
            implied_spot=implied_spot,
            years=years,
        )
    )
    upper_weight = (
        0.0
        if lower_index == upper_index
        else float((target_strike - lower_pair.strike) / gap)
    )
    if not 0.0 <= upper_weight <= 1.0:
        raise ValueError("wing interpolation weight escaped [0,1]")
    interpolated = float(lower_iv + upper_weight * (upper_iv - lower_iv))
    return InterpolatedWingIV(
        right=right,
        target_strike=float(target_strike),
        lower_strike=float(lower_pair.strike),
        upper_strike=float(upper_pair.strike),
        lower_symbol=(lower_pair.put_symbol if right == "P" else lower_pair.call_symbol),
        upper_symbol=(upper_pair.put_symbol if right == "P" else upper_pair.call_symbol),
        lower_iv=float(lower_iv),
        upper_iv=float(upper_iv),
        upper_weight=upper_weight,
        interpolated_iv=interpolated,
    )


def risk_reversal_snapshot(
    rows: pd.DataFrame, *, decision_time: pd.Timestamp
) -> RiskReversalSnapshot:
    """One symmetric +/-20 point RR snapshot using shared parity and IV laws."""

    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    parity = parity_snapshot(rows, decision_time=decision)
    years = _time_to_expiry_years(decision)
    put = _interpolated_wing(
        parity.selected_pairs,
        right="P",
        implied_spot=parity.implied_spot,
        years=years,
        target_strike=parity.implied_spot - WING_OFFSET_POINTS,
    )
    call = _interpolated_wing(
        parity.selected_pairs,
        right="C",
        implied_spot=parity.implied_spot,
        years=years,
        target_strike=parity.implied_spot + WING_OFFSET_POINTS,
    )
    return RiskReversalSnapshot(
        decision_time=decision,
        implied_spot=float(parity.implied_spot),
        risk_reversal=float(put.interpolated_iv - call.interpolated_iv),
        put_wing=put,
        call_wing=call,
        parity_pair_ids=parity.candidate_pair_ids,
    )


def historical_risk_reversal_snapshot(
    rows: pd.DataFrame, *, decision_time: pd.Timestamp
) -> RiskReversalSnapshot:
    return risk_reversal_snapshot(rows, decision_time=decision_time)


def live_risk_reversal_snapshot(
    rows: pd.DataFrame, *, decision_time: pd.Timestamp
) -> RiskReversalSnapshot:
    return risk_reversal_snapshot(rows, decision_time=decision_time)


def _chain_frame(path: Path, session: str) -> pd.DataFrame:
    frame = _load_parquet(
        path,
        (
            "ts_event",
            "ts_recv",
            "instrument_id",
            "symbol",
            "bid_px_00",
            "ask_px_00",
        ),
    )
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    if frame.duplicated(["ts_recv", "symbol"]).any():
        raise OptionsChainScreenError(f"duplicate CBBO boundary/symbol identity: {session}")
    parsed = frame["symbol"].astype(str).map(lambda symbol: _parse_symbol(symbol, session))
    frame["right"] = [item[0] for item in parsed]
    frame["strike"] = [item[1] for item in parsed]
    return frame.sort_values(["ts_recv", "symbol"], kind="mergesort").reset_index(drop=True)


def _valid_snapshot_rows(
    rows: pd.DataFrame, *, boundary_ns: int
) -> tuple[pd.DataFrame, np.ndarray]:
    event = pd.to_datetime(rows["ts_event"], utc=True)
    recv = pd.to_datetime(rows["ts_recv"], utc=True)
    event_ns = (
        event.astype("datetime64[ns, UTC]").astype("int64").to_numpy(dtype=np.int64)
    )
    recv_ns = (
        recv.astype("datetime64[ns, UTC]").astype("int64").to_numpy(dtype=np.int64)
    )
    if not bool(np.all(recv_ns == int(boundary_ns))):
        raise OptionsChainScreenError("snapshot rows do not share the requested ts_recv")
    age_ns = recv_ns - event_ns
    bid = pd.to_numeric(rows["bid_px_00"], errors="coerce").to_numpy(float)
    ask = pd.to_numeric(rows["ask_px_00"], errors="coerce").to_numpy(float)
    valid = (
        event.notna().to_numpy()
        & (age_ns >= 0)
        & (age_ns <= QUOTE_STALENESS_NS)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid >= 0.0)
        & (ask >= bid)
    )
    filtered = rows.loc[valid, ["strike", "right", "symbol"]].copy()
    filtered["bid"] = bid[valid]
    filtered["ask"] = ask[valid]
    filtered = filtered.rename(columns={"symbol": "raw_symbol"})
    return filtered, age_ns[valid]


def _risk_reversal_series(
    chain: pd.DataFrame, session: str
) -> tuple[dict[int, RiskReversalSnapshot], dict[int, str], dict[str, Any]]:
    candidates = _boundary_grid_ns(session, FIRST_BOUNDARY, LAST_BOUNDARY)
    required = set(map(int, candidates)) | set(map(int, candidates - HORIZON_NS))
    snapshots: dict[int, RiskReversalSnapshot] = {}
    failures: dict[int, str] = {}
    ages: list[np.ndarray] = []
    observed: set[int] = set()
    for boundary, rows in chain.groupby("ts_recv", sort=True):
        boundary_ns = int(pd.Timestamp(boundary).value)
        if boundary_ns not in required:
            continue
        observed.add(boundary_ns)
        try:
            valid_rows, valid_ages = _valid_snapshot_rows(rows, boundary_ns=boundary_ns)
            if len(valid_ages):
                ages.append(valid_ages)
            decision = pd.Timestamp(boundary_ns, unit="ns", tz="UTC") + pd.Timedelta(
                milliseconds=ENTRY_EMISSION_LAG_MS
            )
            snapshots[boundary_ns] = historical_risk_reversal_snapshot(
                valid_rows, decision_time=decision
            )
        except (ValueError, OptionsChainScreenError) as error:
            failures[boundary_ns] = str(error)
    for boundary_ns in sorted(required - observed):
        failures[boundary_ns] = "missing exact CBBO-1m boundary"
    all_ages = np.concatenate(ages) / 1e9 if ages else np.asarray([], dtype=float)
    diagnostics = {
        "required_boundaries": len(required),
        "snapshots_succeeded": len(snapshots),
        "failure_counts": dict(Counter(failures.values())),
        "valid_quote_age_seconds": {
            "count": int(len(all_ages)),
            "p50": float(np.quantile(all_ages, 0.50)) if len(all_ages) else None,
            "p90": float(np.quantile(all_ages, 0.90)) if len(all_ages) else None,
            "p99": float(np.quantile(all_ages, 0.99)) if len(all_ages) else None,
            "max": float(np.max(all_ages)) if len(all_ages) else None,
        },
    }
    return snapshots, failures, diagnostics


def _snapshot_fields(snapshot: RiskReversalSnapshot, prior: RiskReversalSnapshot) -> dict[str, Any]:
    current_ids = (
        snapshot.put_wing.lower_symbol,
        snapshot.put_wing.upper_symbol,
        snapshot.call_wing.lower_symbol,
        snapshot.call_wing.upper_symbol,
    )
    prior_ids = (
        prior.put_wing.lower_symbol,
        prior.put_wing.upper_symbol,
        prior.call_wing.lower_symbol,
        prior.call_wing.upper_symbol,
    )
    return {
        "risk_reversal": snapshot.risk_reversal,
        "risk_reversal_prior_5m": prior.risk_reversal,
        "implied_spot": snapshot.implied_spot,
        "put_target_strike": snapshot.put_wing.target_strike,
        "put_lower_strike": snapshot.put_wing.lower_strike,
        "put_upper_strike": snapshot.put_wing.upper_strike,
        "put_lower_symbol": snapshot.put_wing.lower_symbol,
        "put_upper_symbol": snapshot.put_wing.upper_symbol,
        "put_upper_weight": snapshot.put_wing.upper_weight,
        "put_iv": snapshot.put_wing.interpolated_iv,
        "call_target_strike": snapshot.call_wing.target_strike,
        "call_lower_strike": snapshot.call_wing.lower_strike,
        "call_upper_strike": snapshot.call_wing.upper_strike,
        "call_lower_symbol": snapshot.call_wing.lower_symbol,
        "call_upper_symbol": snapshot.call_wing.upper_symbol,
        "call_upper_weight": snapshot.call_wing.upper_weight,
        "call_iv": snapshot.call_wing.interpolated_iv,
        "parity_pair_count": len(snapshot.parity_pair_ids),
        "wing_pair_turnover_5m": current_ids != prior_ids,
    }


def _decision_frame(
    *,
    spx: pd.DataFrame,
    session: str,
    snapshots: Mapping[int, RiskReversalSnapshot],
    failures: Mapping[int, str],
) -> pd.DataFrame:
    boundaries = _boundary_grid_ns(session, FIRST_BOUNDARY, LAST_BOUNDARY)
    decisions = boundaries + ENTRY_EMISSION_LAG_MS * 1_000_000
    available, closes = _availability_series(spx)
    spot_prior = _close_available_at(available, closes, decisions - HORIZON_NS)
    spot_start = _close_available_at(available, closes, decisions)
    spot_end = _close_available_at(available, closes, decisions + HORIZON_NS)
    rows: list[dict[str, Any]] = []
    for boundary_ns, decision_ns, previous_spot, start_spot, end_spot in zip(
        boundaries, decisions, spot_prior, spot_start, spot_end, strict=True
    ):
        boundary = int(boundary_ns)
        prior_boundary = boundary - HORIZON_NS
        current = snapshots.get(boundary)
        prior = snapshots.get(prior_boundary)
        row: dict[str, Any] = {
            "session": session,
            "feature_boundary_ns": boundary,
            "decision_time_ns": int(decision_ns),
            "label_end_ns": int(decision_ns + HORIZON_NS),
            "h1_score": np.nan,
            "h1_direction": 0,
            "momentum_direction": 0,
            "spx_prior": float(previous_spot),
            "spx_start": float(start_spot),
            "spx_end": float(end_spot),
            "future_spx_points": (
                float(end_spot - start_spot)
                if np.isfinite(start_spot) and np.isfinite(end_spot)
                else np.nan
            ),
            "contemporaneous_spx_points": (
                float(start_spot - previous_spot)
                if np.isfinite(previous_spot) and np.isfinite(start_spot)
                else np.nan
            ),
            "common_valid": False,
            "invalid_reason": "",
        }
        if current is None or prior is None:
            current_reason = failures.get(boundary, "")
            prior_reason = failures.get(prior_boundary, "")
            row["invalid_reason"] = (
                f"current:{current_reason or 'missing'}|prior:{prior_reason or 'missing'}"
            )
            rows.append(row)
            continue
        score = float(-(current.risk_reversal - prior.risk_reversal))
        row.update(_snapshot_fields(current, prior))
        row["h1_score"] = score
        row["h1_direction"] = int(np.sign(score))
        if not np.isfinite([previous_spot, start_spot, end_spot]).all():
            row["invalid_reason"] = "missing causal SPX prior/start/end"
            rows.append(row)
            continue
        row["momentum_direction"] = int(np.sign(start_spot - previous_spot))
        row["common_valid"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def _serial_replay(decisions: pd.DataFrame, *, action_column: str, strategy: str) -> pd.DataFrame:
    eligible = decisions.loc[decisions["common_valid"].astype(bool)].sort_values(
        "decision_time_ns", kind="mergesort"
    )
    free_at_ns = np.iinfo(np.int64).min
    trades: list[dict[str, Any]] = []
    for row in eligible.itertuples(index=False):
        direction = int(getattr(row, action_column))
        if direction == 0:
            continue
        decision_ns = int(row.decision_time_ns)
        if decision_ns < free_at_ns:
            continue
        future_move = float(row.future_spx_points)
        contemporaneous = float(row.contemporaneous_spx_points)
        free_at_ns = decision_ns + HORIZON_NS
        trades.append(
            {
                "session": str(row.session),
                "strategy": strategy,
                "feature_boundary_ns": int(row.feature_boundary_ns),
                "decision_time_ns": decision_ns,
                "exit_time_ns": free_at_ns,
                "direction": direction,
                "future_spx_points": future_move,
                "signed_spx_points": float(direction * future_move),
                "contemporaneous_signed_spx_points": float(direction * contemporaneous),
                "absolute_future_spx_points": float(abs(future_move)),
            }
        )
    return pd.DataFrame(trades)


def _strategy_counts(decisions: pd.DataFrame, trades: pd.DataFrame, action_column: str) -> dict[str, int]:
    eligible = decisions.loc[decisions["common_valid"].astype(bool)]
    actions = pd.to_numeric(eligible[action_column], errors="raise").astype(int)
    nonzero = int(actions.ne(0).sum())
    executed = int(len(trades))
    return {
        "call_signals": int(actions.eq(1).sum()),
        "put_signals": int(actions.eq(-1).sum()),
        "wait_signals": int(actions.eq(0).sum()),
        "executed_intervals": executed,
        "skipped_overlap_signals": nonzero - executed,
    }


def _future_mutation_check(
    chain: pd.DataFrame,
    session: str,
    baseline: Mapping[int, RiskReversalSnapshot],
) -> dict[str, Any]:
    candidates = _boundary_grid_ns(session, FIRST_BOUNDARY, LAST_BOUNDARY)
    cutoff = int(candidates[len(candidates) // 2])
    mutated = chain.copy()
    recv_ns = (
        pd.to_datetime(mutated["ts_recv"], utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy(dtype=np.int64)
    )
    future = recv_ns > cutoff
    for name in ("bid_px_00", "ask_px_00"):
        values = pd.to_numeric(mutated[name], errors="coerce").to_numpy(float).copy()
        change = future & np.isfinite(values)
        values[change] *= 1.125
        mutated[name] = values
    after, _, _ = _risk_reversal_series(mutated, session)
    prefix = sorted(key for key in baseline if key <= cutoff and key in after)
    if not prefix:
        raise OptionsChainScreenError("future-mutation check has no valid prefix")
    for key in prefix:
        if asdict(baseline[key]) != asdict(after[key]):
            raise OptionsChainScreenError("future option mutation changed an earlier H1 snapshot")
    return {
        "session": session,
        "cutoff_boundary_ns": cutoff,
        "mutated_source_rows": int(future.sum()),
        "prefix_snapshots_checked": len(prefix),
        "passed": True,
    }


def _session_evidence(
    corpus_root: Path, session: str, *, run_mutation_check: bool
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    raw = corpus_root / "raw"
    chain_path = raw / "databento/opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet"
    spx_path = raw / "index/spx_1m" / f"{session}.official_spx.parquet"
    chain = _chain_frame(chain_path, session)
    spx = _official_spx(spx_path, session)
    snapshots, failures, diagnostics = _risk_reversal_series(chain, session)
    decisions = _decision_frame(
        spx=spx,
        session=session,
        snapshots=snapshots,
        failures=failures,
    )
    h1_trades = _serial_replay(decisions, action_column="h1_direction", strategy="H1")
    momentum_trades = _serial_replay(
        decisions, action_column="momentum_direction", strategy="MOMENTUM"
    )
    trades = pd.concat([h1_trades, momentum_trades], ignore_index=True)
    eligible = int(decisions["common_valid"].astype(bool).sum())
    h1_counts = _strategy_counts(decisions, h1_trades, "h1_direction")
    momentum_counts = _strategy_counts(decisions, momentum_trades, "momentum_direction")
    dropped_reason = ""
    if eligible < MIN_BOUNDARIES_PER_SESSION:
        dropped_reason = (
            f"common_valid_boundaries:{eligible}<minimum:{MIN_BOUNDARIES_PER_SESSION}"
        )
    elif h1_counts["executed_intervals"] == 0 or momentum_counts["executed_intervals"] == 0:
        dropped_reason = "zero executed intervals for H1 or momentum"
    summary = {
        "session": session,
        "raw_chain_rows": int(len(chain)),
        "distinct_contracts": int(chain["symbol"].astype(str).nunique()),
        "distinct_cbbo_boundaries": int(chain["ts_recv"].nunique()),
        "common_valid_boundaries": eligible,
        "invalid_boundaries": int(len(decisions) - eligible),
        "invalid_reason_counts": dict(
            Counter(decisions.loc[~decisions["common_valid"], "invalid_reason"].astype(str))
        ),
        "wing_pair_turnover_rate": (
            float(
                decisions.loc[decisions["common_valid"], "wing_pair_turnover_5m"]
                .astype(bool)
                .mean()
            )
            if eligible
            else None
        ),
        "h1_session_points": (
            float(h1_trades["signed_spx_points"].sum()) if len(h1_trades) else None
        ),
        "h1_interval_points": (
            float(h1_trades["signed_spx_points"].mean()) if len(h1_trades) else None
        ),
        "momentum_session_points": (
            float(momentum_trades["signed_spx_points"].sum())
            if len(momentum_trades)
            else None
        ),
        "momentum_interval_points": (
            float(momentum_trades["signed_spx_points"].mean())
            if len(momentum_trades)
            else None
        ),
        **{f"h1_{key}": value for key, value in h1_counts.items()},
        **{f"momentum_{key}": value for key, value in momentum_counts.items()},
        "snapshot_diagnostics": diagnostics,
        "source_hashes": {
            "cbbo_1m": _sha256_path(chain_path),
            "official_spx": _sha256_path(spx_path),
        },
        "dropped_reason": dropped_reason,
    }
    mutation = (
        _future_mutation_check(chain, session, snapshots) if run_mutation_check else None
    )
    return summary, decisions, trades, mutation


def _fold_ledger(
    sessions: Sequence[str], summaries: pd.DataFrame, trades: pd.DataFrame
) -> list[dict[str, Any]]:
    by_session = summaries.set_index("session")
    rows: list[dict[str, Any]] = []
    for fold in entry_expanding_folds(sessions):
        test_sessions = [session for session in fold["test"] if session in by_session.index]
        if len(test_sessions) < MIN_EVALUATION_SESSIONS_PER_FOLD:
            raise OptionsChainScreenError(
                f"fold {fold['fold']} retained {len(test_sessions)} evaluation sessions"
            )
        test = by_session.loc[test_sessions]
        fold_trades = trades[trades["session"].isin(test_sessions)]
        h1 = fold_trades[fold_trades["strategy"].eq("H1")]
        momentum = fold_trades[fold_trades["strategy"].eq("MOMENTUM")]
        if h1.empty or momentum.empty:
            raise OptionsChainScreenError(f"fold {fold['fold']} has no H1 or momentum trades")
        h1_session = float(test["h1_session_points"].mean())
        h1_interval = float(h1["signed_spx_points"].mean())
        mom_session = float(test["momentum_session_points"].mean())
        mom_interval = float(momentum["signed_spx_points"].mean())
        conditions = {
            "h1_session_positive": h1_session > 0.0,
            "h1_interval_positive": h1_interval > 0.0,
            "h1_session_beats_momentum": h1_session > mom_session,
            "h1_interval_beats_momentum": h1_interval > mom_interval,
        }
        rows.append(
            {
                "fold": int(fold["fold"]),
                "evaluation_sessions": len(test_sessions),
                "h1_executed_intervals": int(len(h1)),
                "momentum_executed_intervals": int(len(momentum)),
                "h1_mean_points_per_session": h1_session,
                "h1_mean_points_per_interval": h1_interval,
                "momentum_mean_points_per_session": mom_session,
                "momentum_mean_points_per_interval": mom_interval,
                "incremental_points_per_session": h1_session - mom_session,
                "incremental_points_per_interval": h1_interval - mom_interval,
                **conditions,
                "passed": all(conditions.values()),
            }
        )
    return rows


def _assert_sign_reversal_identity(trades: pd.DataFrame) -> dict[str, Any]:
    h1 = trades[trades["strategy"].eq("H1")]
    if h1.empty:
        raise OptionsChainScreenError("sign-reversal identity has no H1 trades")
    forward = h1["signed_spx_points"].to_numpy(float)
    reversed_points = -h1["direction"].to_numpy(int) * h1["future_spx_points"].to_numpy(float)
    if not np.array_equal(reversed_points, -forward):
        raise OptionsChainScreenError("exact sign-reversal identity failed")
    return {"trades_checked": int(len(h1)), "passed": True}


def _prior_art_receipt() -> dict[str, Any]:
    exact = prior_art_check(MECHANISM, extra_terms=EXACT_PRIOR_ART_TERMS)
    broad = prior_art_check("symmetric 0DTE risk reversal change", extra_terms=BROAD_PRIOR_ART_TERMS)
    blocking = [hit for hit in (*exact, *broad) if hit.blocking]
    if blocking:
        raise OptionsChainScreenError(
            "H1 blocked by prior art: "
            + "; ".join(f"{hit.source}:{hit.line_number}" for hit in blocking)
        )
    return {
        "mechanism": MECHANISM,
        "exact_terms": list(EXACT_PRIOR_ART_TERMS),
        "broad_terms": list(BROAD_PRIOR_ART_TERMS),
        "exact_hits": [asdict(hit) for hit in exact],
        "broad_hits": [asdict(hit) for hit in broad],
        "blocking": False,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _render_result(
    verdict: str,
    folds: Sequence[Mapping[str, Any]],
    *,
    sessions_used: int,
    sessions_dropped: int,
) -> str:
    lines = [
        "# Options-chain H1 raw direction screen — results",
        "",
        f"verdict: `{verdict}`",
        f"sessions used: {sessions_used}; dropped: {sessions_dropped}",
        "family: K=1; horizon: 5m; strict serial occupancy; comparator: 5m SPX momentum",
        "",
        "| Fold | Sessions | H1 pts/session | H1 pts/interval | Momentum pts/session | Momentum pts/interval | Pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['fold']} | {row['evaluation_sessions']} | "
            f"{row['h1_mean_points_per_session']:+.6f} | "
            f"{row['h1_mean_points_per_interval']:+.6f} | "
            f"{row['momentum_mean_points_per_session']:+.6f} | "
            f"{row['momentum_mean_points_per_interval']:+.6f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "`DIRECTION_SCREEN_PASS` permits only a separately frozen option-wrapper screen.",
        "`NO_EDGE` terminates this H1 family without changing sign, threshold, wing, or horizon.",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise OptionsChainScreenError(f"output already exists; refusing rerun: {OUTPUT_ROOT}")
    if not PREREGISTRATION.is_file():
        raise OptionsChainScreenError(f"missing preregistration: {PREREGISTRATION}")
    if RISK_FREE_RATE != 0.04 or DIVIDEND_YIELD != 0.0:
        raise OptionsChainScreenError("shared parity/IV constants changed from the frozen law")
    prior_art = _prior_art_receipt()
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    if len(sessions) != DEVELOPMENT_SESSION_COUNT:
        raise OptionsChainScreenError(
            f"expected {DEVELOPMENT_SESSION_COUNT} development sessions, got {len(sessions)}"
        )
    print(f"corpus: {corpus_root}")
    print(f"development sessions: {len(sessions)}")
    session_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    mutation: dict[str, Any] | None = None
    for position, session in enumerate(sessions, start=1):
        summary, decisions, trades, checked = _session_evidence(
            corpus_root,
            session,
            run_mutation_check=mutation is None,
        )
        if checked is not None:
            mutation = checked
        session_rows.append(summary)
        decision_frames.append(decisions)
        trade_frames.append(trades)
        if position % 10 == 0 or position == len(sessions):
            dropped_so_far = sum(bool(row["dropped_reason"]) for row in session_rows)
            print(f"  {position}/{len(sessions)} sessions ({dropped_so_far} dropped)")
    session_frame = pd.DataFrame(session_rows)
    dropped = session_frame[session_frame["dropped_reason"].astype(str).ne("")].copy()
    if len(dropped) > MAX_DROPPED_SESSIONS:
        raise OptionsChainScreenError(
            f"{len(dropped)} sessions dropped; cap is {MAX_DROPPED_SESSIONS}"
        )
    used = session_frame[session_frame["dropped_reason"].astype(str).eq("")].copy()
    if used.empty:
        raise OptionsChainScreenError("no usable development sessions")
    decisions = pd.concat(decision_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    decisions = decisions[decisions["session"].isin(used["session"])].reset_index(drop=True)
    trades = trades[trades["session"].isin(used["session"])].reset_index(drop=True)
    sign_identity = _assert_sign_reversal_identity(trades)
    folds = _fold_ledger(sessions, used, trades)
    verdict = "DIRECTION_SCREEN_PASS" if all(row["passed"] for row in folds) else "NO_EDGE"
    inventory = {
        "sessions_declared": len(sessions),
        "sessions_used": int(len(used)),
        "sessions_dropped": int(len(dropped)),
        "raw_rows_min": int(session_frame["raw_chain_rows"].min()),
        "raw_rows_median": float(session_frame["raw_chain_rows"].median()),
        "raw_rows_mean": float(session_frame["raw_chain_rows"].mean()),
        "raw_rows_max": int(session_frame["raw_chain_rows"].max()),
        "distinct_contracts_min": int(session_frame["distinct_contracts"].min()),
        "distinct_contracts_median": float(session_frame["distinct_contracts"].median()),
        "distinct_contracts_mean": float(session_frame["distinct_contracts"].mean()),
        "distinct_contracts_max": int(session_frame["distinct_contracts"].max()),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "hypothesis_id": HYPOTHESIS_ID,
        "mechanism": MECHANISM,
        "family_size": 1,
        "horizon_minutes": HORIZON_MINUTES,
        "wing_offset_points": WING_OFFSET_POINTS,
        "max_bracket_gap_points": MAX_BRACKET_GAP_POINTS,
        "risk_free_rate": RISK_FREE_RATE,
        "dividend_yield": DIVIDEND_YIELD,
        "quote_staleness_seconds": QUOTE_STALENESS_NS / 1e9,
        "decision_emission_lag_ms": ENTRY_EMISSION_LAG_MS,
        "boundary_grid_et": [FIRST_BOUNDARY.isoformat(), LAST_BOUNDARY.isoformat()],
        "serial_occupancy": True,
        "primary_metric": "mean_signed_spx_points_per_session",
        "secondary_metric": "mean_signed_spx_points_per_executed_interval",
        "comparator": "five_minute_spx_momentum",
        "gate": "four strict inequalities in all five folds; no pooled rescue",
        "prior_art": prior_art,
        "preregistration_path": str(PREREGISTRATION.relative_to(REPO_ROOT)),
        "preregistration_sha256": _sha256_path(PREREGISTRATION),
        "inventory": inventory,
        "dropped_sessions": dropped[["session", "dropped_reason"]].to_dict("records"),
        "fold_results": folds,
        "future_mutation_check": mutation,
        "sign_reversal_identity": sign_identity,
        "source_hashes": {
            row["session"]: row["source_hashes"] for row in session_rows
        },
        "code_hashes": {
            "screen": _sha256_path(Path(__file__)),
            "shared_parity_iv": _sha256_path(
                REPO_ROOT / "v4/research/pathd_opra_parity_features.py"
            ),
            "spx_clock": _sha256_path(
                REPO_ROOT / "v4/research/pathd_spx_directional_skill_screen.py"
            ),
            "prior_art": _sha256_path(REPO_ROOT / "v4/research/pathd_research_loop.py"),
        },
        "protected_holdout_opened": False,
        "model_trained": False,
        "null_run": False,
        "option_wrapper_run": False,
        "paper_order_submitted": False,
        "paper_default_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    fold_frame = pd.DataFrame(folds)
    fold_path = OUTPUT_ROOT / "fold_results.csv"
    session_path = OUTPUT_ROOT / "session_results.csv"
    decision_path = OUTPUT_ROOT / "decisions.parquet"
    trade_path = OUTPUT_ROOT / "trades.csv"
    result_path = OUTPUT_ROOT / "results.md"
    fold_frame.to_csv(fold_path, index=False)
    session_frame.to_csv(session_path, index=False)
    decisions.to_parquet(decision_path, index=False, compression="zstd")
    trades.to_csv(trade_path, index=False)
    result_path.write_text(
        _render_result(
            verdict,
            folds,
            sessions_used=len(used),
            sessions_dropped=len(dropped),
        ),
        encoding="utf-8",
    )
    payload["artifact_hashes"] = {
        path.name: _sha256_path(path)
        for path in (fold_path, session_path, decision_path, trade_path, result_path)
    }
    payload = _clean(payload)
    payload["receipt_sha256"] = stable_hash(payload)
    receipt_path = OUTPUT_ROOT / "receipt.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"verdict: {verdict}")
    print(f"receipt: {receipt_path}")
    return payload


if __name__ == "__main__":
    main()
