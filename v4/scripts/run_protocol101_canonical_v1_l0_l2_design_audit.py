"""Protocol101 canonical minute-game v1 L0/L2 design audit.

Offline-only diagnostic. Reads the certified paired recorder-day decision
traces, derives preregistered canonical minute-game features from
``payload.candidate_filter_trace`` slots, runs L0 drift/bias checks and L2
source-discriminator controls, then emits a routing packet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import stdev
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4.scripts.run_protocol101_feature_recovery_group1_parity_audit import (
    load_trace_rows,
    local_minute,
    parse_ts,
    trace_paths,
)


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_canonical_v1_l0_l2_design_audit_attempt001"
TRACE_PREFIX = "protocol101_live_v2_candidate_universe_parity_source_aligned"
SESSIONS = ("2026-06-30", "2026-07-01", "2026-07-02")
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
SCHEMA_VERSION = "Protocol101CanonicalV1L0L2DesignAuditAttempt001"
NY_OPEN_MINUTE = 9 * 60 + 30
EXCLUDED_PRE_WINDOW_MINUTE = "09:31"
OPTION_TICK = 0.05
BS_RISK_FREE_RATE = 0.05
BS_DIVIDEND_YIELD = 0.0
REPAIR_BUDGET = 3
BOOTSTRAP_REPEATS = 300
RANDOM_STATE = 101
NY = ZoneInfo("America/New_York")


CONTINUOUS_FEATURES: list[dict[str, Any]] = [
    {"name": "A.context.spx_for_ladder", "family": "A", "applicability": "all_slots"},
    {"name": "A.context.atm_strike", "family": "A", "applicability": "all_slots"},
    {"name": "B.day.session_15m_bucket", "family": "B", "applicability": "all_slots"},
    {"name": "B.ladder.strike", "family": "B", "applicability": "all_slots"},
    {"name": "B.ladder.offset", "family": "B", "applicability": "all_slots"},
    {"name": "B.ladder.abs_offset", "family": "B", "applicability": "all_slots"},
    {"name": "B.ladder.moneyness_bps", "family": "B", "applicability": "all_slots"},
    {"name": "C.mid.mid_tick_q", "family": "C", "applicability": "mid_positive"},
    {"name": "C.mid.mid_spot_bps", "family": "C", "applicability": "mid_positive"},
    {"name": "C.mid.logret_1m", "family": "C", "applicability": "has_1m_lag"},
    {"name": "C.mid.logret_5m", "family": "C", "applicability": "has_5m_lag"},
    {"name": "C.mid.logret_15m", "family": "C", "applicability": "has_15m_lag"},
    {"name": "C.mid.path_range_15m", "family": "C", "applicability": "has_15m_window"},
    {"name": "C.mid.realized_vol_15m", "family": "C", "applicability": "has_15m_return_window"},
    {"name": "D.near_atm.straddle_mid_spot_bps", "family": "D", "applicability": "abs_offset_lte_25"},
    {"name": "D.near_atm.put_call_mid_ratio", "family": "D", "applicability": "abs_offset_lte_25"},
    {"name": "D.near_atm.side_smile_slope_bps_per_5pt", "family": "D", "applicability": "abs_offset_lte_25"},
    {"name": "E.bs.iv", "family": "E", "applicability": "mid_gte_2_ticks"},
    {"name": "E.bs.delta", "family": "E", "applicability": "mid_gte_2_ticks"},
    {"name": "E.bs.gamma", "family": "E", "applicability": "mid_gte_2_ticks"},
]

FLAG_FEATURES: list[dict[str, Any]] = [
    {"name": "B.ladder.right_is_call", "family": "B", "applicability": "all_slots"},
    {"name": "B.ladder.right_is_put", "family": "B", "applicability": "all_slots"},
]

FEATURE_NAMES = [item["name"] for item in CONTINUOUS_FEATURES + FLAG_FEATURES]
CONTINUOUS_NAMES = [item["name"] for item in CONTINUOUS_FEATURES]
FLAG_NAMES = [item["name"] for item in FLAG_FEATURES]
FEATURE_FAMILY = {item["name"]: item["family"] for item in CONTINUOUS_FEATURES + FLAG_FEATURES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-prefix", default=TRACE_PREFIX)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def quantize_tick(value: Any, tick: float = OPTION_TICK) -> float | None:
    number = finite(value)
    if number is None or number <= 0:
        return None
    return round(number / tick) * tick


def minute_of_day(minute_et: str) -> int:
    hour, minute = minute_et.split(":")
    return int(hour) * 60 + int(minute)


def minutes_since_open(minute_et: str) -> int:
    return minute_of_day(minute_et) - NY_OPEN_MINUTE


def safe_log_return(current: Any, prior: Any) -> float | None:
    current_f = finite(current)
    prior_f = finite(prior)
    if current_f is None or prior_f is None or current_f <= 0 or prior_f <= 0:
        return None
    return math.log(current_f / prior_f)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(spot: float, strike: float, tte: float, vol: float, right: str) -> float:
    if spot <= 0 or strike <= 0 or tte <= 0 or vol <= 0:
        return float("nan")
    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + (BS_RISK_FREE_RATE - BS_DIVIDEND_YIELD + 0.5 * vol * vol) * tte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_r = math.exp(-BS_RISK_FREE_RATE * tte)
    df_q = math.exp(-BS_DIVIDEND_YIELD * tte)
    if right == "C":
        return spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2)
    return strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)


def bs_greeks(spot: float, strike: float, tte: float, vol: float, right: str) -> tuple[float, float]:
    sqrt_t = math.sqrt(max(tte, 1e-12))
    d1 = (math.log(spot / strike) + (BS_RISK_FREE_RATE - BS_DIVIDEND_YIELD + 0.5 * vol * vol) * tte) / (vol * sqrt_t)
    df_q = math.exp(-BS_DIVIDEND_YIELD * tte)
    if right == "C":
        delta = df_q * normal_cdf(d1)
    else:
        delta = df_q * (normal_cdf(d1) - 1.0)
    gamma = df_q * normal_pdf(d1) / (spot * vol * sqrt_t)
    return delta, gamma


def implied_vol(mid: float, spot: float, strike: float, tte: float, right: str) -> float | None:
    if mid < 2.0 * OPTION_TICK or spot <= 0 or strike <= 0 or tte <= 0 or right not in {"C", "P"}:
        return None
    lo = 1e-4
    hi = 5.0
    intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if mid < max(0.0, intrinsic - 1.0):
        return None
    price_hi = bs_price(spot, strike, tte, hi, right)
    if not math.isfinite(price_hi) or price_hi < mid:
        return None
    for _ in range(45):
        mid_vol = (lo + hi) / 2.0
        price = bs_price(spot, strike, tte, mid_vol, right)
        if not math.isfinite(price):
            return None
        if price < mid:
            lo = mid_vol
        else:
            hi = mid_vol
    return (lo + hi) / 2.0


def tte_years(decision_ts: datetime) -> float:
    local_date = decision_ts.astimezone(NY).date()
    expiry = datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        16,
        0,
        0,
        tzinfo=NY,
    ).astimezone(UTC)
    seconds = max((expiry - decision_ts.astimezone(UTC)).total_seconds(), 60.0)
    return seconds / (365.0 * 24.0 * 60.0 * 60.0)


def canonical_feature_definition() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101CanonicalFeatureDefinitionV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "base_contract": CONTRACT,
        "base_transform": TRANSFORM,
        "source_trace_rule": "features derive from payload.candidate_filter_trace; each row must expose 42 slots",
        "row_rule": {
            "exclude_from_l0_l2_scoring": EXCLUDED_PRE_WINDOW_MINUTE,
            "allow_as_t_minus_1_endpoint_for_09_32_returns": True,
        },
        "families": {
            "A": "inherited index/context canaries",
            "B": "day structure and static ladder geometry canaries",
            "C": "per-slot option mid features",
            "D": "near-ATM composites for abs(offset) <= 25",
            "E": "internal Black-Scholes IV/delta/gamma from quantized mid + SPX + TTE",
        },
        "continuous_features": CONTINUOUS_FEATURES,
        "flag_or_bucket_features": FLAG_FEATURES,
        "excluded_features": [
            "vix_change_5m",
            "vix_change_15m",
            "raw_bid_alpha",
            "raw_ask_alpha",
            "raw_spread_alpha",
            "raw_size_alpha",
            "quote_age_ms_alpha",
            "volume_or_open_interest_alpha",
            "vendor_greeks",
            "breakeven_distance",
            "distance_to_guard_boundary",
            "sub_minute_fields",
        ],
        "mid_quantization": {"tick_size": OPTION_TICK, "policy": "round_to_nearest_tick"},
        "bs_parameters": {"r": BS_RISK_FREE_RATE, "q": BS_DIVIDEND_YIELD, "expiry": "16:00 ET", "minimum_mid_ticks": 2},
        "feature_drift_standardization_policy": {
            "name": "historical_reference_stddev_with_floor_v1",
            "reference_values": "historical stream canonical feature values over the paired recorder-day audit window",
            "scale": "per-feature sample standard deviation of finite historical reference values",
            "scale_floor": 1.0,
            "standardized_abs_drift": "abs(ibkr_value - historical_value) / max(reference_scale, scale_floor)",
        },
        "l0_thresholds": {
            "continuous_coverage_min": 0.80,
            "continuous_standardized_abs_drift_p95_max": 0.10,
            "flag_exact_match_rate_min": 0.90,
            "reject_standardized_abs_drift_p95_gt": 0.50,
            "bias_abs_rho_max": 0.05,
            "bias_abs_ci_upper_max": 0.15,
        },
        "l2_thresholds": {
            "pass_auc_max": 0.55,
            "repair_auc_max": 0.65,
            "null_control_auc_min": 0.45,
            "null_control_auc_max": 0.55,
            "positive_control_auc_min": 0.80,
        },
    }


def preregistration(definition_path: Path, definition_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101CanonicalV1L0L2PreregistrationV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "base_contract": CONTRACT,
        "base_model_facing_transform": TRANSFORM,
        "sessions": list(SESSIONS),
        "trace_prefix": TRACE_PREFIX,
        "canonical_feature_definition_path": str(definition_path),
        "canonical_feature_definition_sha256": definition_hash,
        "comparisons_started_after_preregistration": True,
        "bias_label": {
            "name": "15m_forward_conservative_ibkr_plane_return",
            "entry": "IBKR ask at decision minute t",
            "exit": "IBKR bid at t+15m",
            "missing_policy": "exclude from bias-correlation tests only; do not backfill or shorten horizon",
        },
        "cv": {"source_discriminator": "leave_one_day_out"},
        "controls": {
            "null": "odd/even minute classifier inside historical plane only, required AUC in [0.45, 0.55]",
            "positive": "source classifier using raw bid/ask/spread/quote_age_ms, required AUC >= 0.80",
            "price_only": "source classifier using raw bid/ask/spread; diagnostic only",
        },
        "repair_iterations_used": 0,
        "repair_budget": REPAIR_BUDGET,
        "evidence_grade": "design_burned_days_only",
        "forbidden_actions": {
            "trading_model_training": False,
            "uplift_cv": False,
            "threshold_tuning": False,
            "broker_api_calls": False,
            "paid_downloads": False,
            "paper_submit": False,
            "promotion_default_runtime_launchd_edits": False,
        },
    }


def write_preregistration(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    definition_path = out_dir / "canonical_feature_definition.json"
    definition = canonical_feature_definition()
    definition["attempt_id"] = out_dir.name
    write_json(definition_path, definition)
    definition_hash = sha256_path(definition_path)
    prereg = preregistration(definition_path, definition_hash)
    prereg["attempt_id"] = out_dir.name
    prereg_path = out_dir / "preregistration.json"
    write_json(prereg_path, prereg)
    return prereg


def slot_rows_for_trace(session: str, source: str, rows_by_ts: dict[datetime, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    slot_count_failures: list[str] = []
    for ts, row in sorted(rows_by_ts.items()):
        minute_et = local_minute(ts)
        trace = row.get("candidate_filter_trace")
        if not isinstance(trace, list) or len(trace) != 42:
            slot_count_failures.append(ts.isoformat())
            continue
        for slot in trace:
            strike = finite(slot.get("strike"))
            right = str(slot.get("right") or "")
            mid_q = quantize_tick(slot.get("mid"))
            spot = finite(slot.get("spx_for_ladder"))
            offset = finite(slot.get("offset"))
            bid = finite(slot.get("bid"))
            ask = finite(slot.get("ask"))
            raw_spread = None
            if bid is not None and ask is not None:
                raw_spread = ask - bid
            rec: dict[str, Any] = {
                "source": source,
                "session_date": session,
                "decision_ts_utc": ts.isoformat(),
                "decision_minute_et": minute_et,
                "score_row": minute_et != EXCLUDED_PRE_WINDOW_MINUTE,
                "minute_since_open": minutes_since_open(minute_et),
                "strike": strike,
                "strike_key": f"{strike:.3f}" if strike is not None else "",
                "right": right,
                "contract_id": slot.get("contract_id"),
                "raw_bid": bid,
                "raw_ask": ask,
                "raw_spread": raw_spread,
                "raw_quote_age_ms": finite(slot.get("quote_age_ms")),
                "raw_bid_size": finite(slot.get("bid_size")),
                "raw_ask_size": finite(slot.get("ask_size")),
                "A.context.spx_for_ladder": spot,
                "A.context.atm_strike": finite(slot.get("atm_strike")),
                "B.day.session_15m_bucket": math.floor(minutes_since_open(minute_et) / 15.0),
                "B.ladder.strike": strike,
                "B.ladder.offset": offset,
                "B.ladder.abs_offset": abs(offset) if offset is not None else None,
                "B.ladder.right_is_call": 1 if right == "C" else 0 if right == "P" else None,
                "B.ladder.right_is_put": 1 if right == "P" else 0 if right == "C" else None,
                "C.mid.mid_tick_q": mid_q,
            }
            if strike is not None and spot not in (None, 0):
                rec["B.ladder.moneyness_bps"] = ((strike / float(spot)) - 1.0) * 10_000.0
            else:
                rec["B.ladder.moneyness_bps"] = None
            if mid_q is not None and spot not in (None, 0):
                rec["C.mid.mid_spot_bps"] = mid_q / float(spot) * 10_000.0
            else:
                rec["C.mid.mid_spot_bps"] = None
            records.append(rec)
    out = pd.DataFrame.from_records(records)
    if not slot_count_failures:
        out.attrs["slot_count_failures"] = []
    else:
        out.attrs["slot_count_failures"] = slot_count_failures
    return out


def add_time_series_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["source", "session_date", "strike_key", "right", "decision_ts_utc"]).copy()
    grouped = frame.groupby(["source", "session_date", "strike_key", "right"], sort=False)
    for lag in (1, 5, 15):
        prior = grouped["C.mid.mid_tick_q"].shift(lag)
        frame[f"C.mid.logret_{lag}m"] = [
            safe_log_return(current, previous) for current, previous in zip(frame["C.mid.mid_tick_q"], prior, strict=False)
        ]
    rolling_mid = grouped["C.mid.mid_tick_q"].rolling(window=15, min_periods=15)
    frame["C.mid.path_range_15m"] = (
        np.log(rolling_mid.max().reset_index(level=[0, 1, 2, 3], drop=True) / rolling_mid.min().reset_index(level=[0, 1, 2, 3], drop=True))
        .replace([np.inf, -np.inf], np.nan)
    )
    rolling_ret = grouped["C.mid.logret_1m"].rolling(window=15, min_periods=15)
    frame["C.mid.realized_vol_15m"] = rolling_ret.std().reset_index(level=[0, 1, 2, 3], drop=True) * math.sqrt(15.0)
    return frame


def add_near_atm_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in (
        "D.near_atm.straddle_mid_spot_bps",
        "D.near_atm.put_call_mid_ratio",
        "D.near_atm.side_smile_slope_bps_per_5pt",
    ):
        frame[name] = np.nan
    for _, group in frame.groupby(["source", "session_date", "decision_ts_utc"], sort=False):
        idxs = group.index
        by_strike: dict[str, dict[str, pd.Series]] = defaultdict(dict)
        for idx, row in group.iterrows():
            by_strike[str(row["strike_key"])][str(row["right"])] = row
        slopes: dict[tuple[str, str], float] = {}
        for right in ("C", "P"):
            side = group[(group["right"] == right) & (group["B.ladder.offset"].abs() <= 25)].sort_values("B.ladder.offset")
            offsets = side["B.ladder.offset"].to_numpy(dtype=float)
            mids_bps = side["C.mid.mid_spot_bps"].to_numpy(dtype=float)
            side_indexes = list(side.index)
            for pos, idx in enumerate(side_indexes):
                slope = np.nan
                if len(side_indexes) >= 2:
                    if 0 < pos < len(side_indexes) - 1:
                        dx = offsets[pos + 1] - offsets[pos - 1]
                        dy = mids_bps[pos + 1] - mids_bps[pos - 1]
                    elif pos == 0:
                        dx = offsets[1] - offsets[0]
                        dy = mids_bps[1] - mids_bps[0]
                    else:
                        dx = offsets[-1] - offsets[-2]
                        dy = mids_bps[-1] - mids_bps[-2]
                    if math.isfinite(dx) and dx != 0 and math.isfinite(dy):
                        slope = dy / dx * 5.0
                slopes[(str(frame.at[idx, "strike_key"]), right)] = slope
        for idx in idxs:
            offset = finite(frame.at[idx, "B.ladder.offset"])
            if offset is None or abs(offset) > 25:
                continue
            strike_key = str(frame.at[idx, "strike_key"])
            call = by_strike.get(strike_key, {}).get("C")
            put = by_strike.get(strike_key, {}).get("P")
            spot = finite(frame.at[idx, "A.context.spx_for_ladder"])
            if call is not None and put is not None and spot not in (None, 0):
                call_mid = finite(call.get("C.mid.mid_tick_q"))
                put_mid = finite(put.get("C.mid.mid_tick_q"))
                if call_mid is not None and put_mid is not None:
                    frame.at[idx, "D.near_atm.straddle_mid_spot_bps"] = (call_mid + put_mid) / float(spot) * 10_000.0
                    if call_mid > 0:
                        frame.at[idx, "D.near_atm.put_call_mid_ratio"] = put_mid / call_mid
            side = str(frame.at[idx, "right"])
            frame.at[idx, "D.near_atm.side_smile_slope_bps_per_5pt"] = slopes.get((strike_key, side), np.nan)
    return frame


def add_bs_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    iv_values: list[float | None] = []
    delta_values: list[float | None] = []
    gamma_values: list[float | None] = []
    for _, row in frame.iterrows():
        mid = finite(row.get("C.mid.mid_tick_q"))
        spot = finite(row.get("A.context.spx_for_ladder"))
        strike = finite(row.get("strike"))
        right = str(row.get("right"))
        ts = parse_ts(row.get("decision_ts_utc"))
        iv = None
        delta = None
        gamma = None
        if mid is not None and spot is not None and strike is not None and ts is not None:
            tte = tte_years(ts)
            iv = implied_vol(mid, spot, strike, tte, right)
            if iv is not None:
                delta, gamma = bs_greeks(spot, strike, tte, iv, right)
        iv_values.append(iv)
        delta_values.append(delta)
        gamma_values.append(gamma)
    frame["E.bs.iv"] = iv_values
    frame["E.bs.delta"] = delta_values
    frame["E.bs.gamma"] = gamma_values
    return frame


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["source", "session_date", "strike_key", "right", "decision_ts_utc"]).copy()
    live = frame["source"] == "ibkr"
    grouped = frame[live].groupby(["session_date", "strike_key", "right"], sort=False)
    exit_bid = grouped["raw_bid"].shift(-15)
    frame["ibkr_exit_bid_t_plus_15m"] = np.nan
    frame.loc[live, "ibkr_exit_bid_t_plus_15m"] = exit_bid
    frame["y_15m_conservative_return"] = np.nan
    mask = live & frame["raw_ask"].notna() & frame["ibkr_exit_bid_t_plus_15m"].notna() & (frame["raw_ask"] > 0)
    frame.loc[mask, "y_15m_conservative_return"] = (
        (frame.loc[mask, "ibkr_exit_bid_t_plus_15m"] - frame.loc[mask, "raw_ask"]) / frame.loc[mask, "raw_ask"]
    )
    context = frame[live].drop_duplicates(["session_date", "decision_ts_utc"]).sort_values(["session_date", "decision_ts_utc"])
    context["spx_5m_prior"] = context.groupby("session_date")["A.context.spx_for_ladder"].shift(5)
    context["realized_5m_spx_vol"] = [
        abs(safe_log_return(now, prior)) if safe_log_return(now, prior) is not None else np.nan
        for now, prior in zip(context["A.context.spx_for_ladder"], context["spx_5m_prior"], strict=False)
    ]
    vol_map = {
        (row.session_date, row.decision_ts_utc): row.realized_5m_spx_vol
        for row in context.itertuples(index=False)
    }
    frame["realized_5m_spx_vol"] = [vol_map.get((row.session_date, row.decision_ts_utc), np.nan) for row in frame.itertuples(index=False)]
    return frame


def add_applicability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    abs_offset = frame["B.ladder.offset"].abs()
    for name in FEATURE_NAMES:
        app = pd.Series(True, index=frame.index)
        if name.startswith("D."):
            app = abs_offset <= 25
        elif name.startswith("E."):
            app = frame["C.mid.mid_tick_q"] >= 2.0 * OPTION_TICK
        elif name == "C.mid.logret_1m":
            app = frame.groupby(["source", "session_date", "strike_key", "right"]).cumcount() >= 1
        elif name == "C.mid.logret_5m":
            app = frame.groupby(["source", "session_date", "strike_key", "right"]).cumcount() >= 5
        elif name == "C.mid.logret_15m":
            app = frame.groupby(["source", "session_date", "strike_key", "right"]).cumcount() >= 15
        elif name in {"C.mid.path_range_15m", "C.mid.realized_vol_15m"}:
            app = frame.groupby(["source", "session_date", "strike_key", "right"]).cumcount() >= 14
        elif name in {"C.mid.mid_tick_q", "C.mid.mid_spot_bps"}:
            app = frame["C.mid.mid_tick_q"] > 0
        frame[f"{name}.__applicable"] = app.fillna(False).astype(bool)
    return frame


def build_slot_frame(trace_prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    readiness: dict[str, Any] = {"blockers": [], "sessions": {}}
    for session in SESSIONS:
        live_path, historical_path = trace_paths(trace_prefix, session)
        readiness["sessions"][session] = {
            "ibkr_trace": str(live_path),
            "historical_trace": str(historical_path),
            "ibkr_exists": live_path.exists(),
            "historical_exists": historical_path.exists(),
        }
        if not live_path.exists() or not historical_path.exists():
            readiness["blockers"].append(f"missing_trace_{session}")
            continue
        live_rows = load_trace_rows(live_path)
        historical_rows = load_trace_rows(historical_path)
        for source, rows in (("ibkr", live_rows), ("historical", historical_rows)):
            frame = slot_rows_for_trace(session, source, rows)
            if frame.attrs.get("slot_count_failures"):
                readiness["blockers"].append(f"{session}_{source}_non_42_slot_rows")
            readiness["sessions"][session][f"{source}_rows"] = len(rows)
            readiness["sessions"][session][f"{source}_slot_rows"] = int(len(frame))
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), readiness
    frame = pd.concat(frames, ignore_index=True)
    frame = add_time_series_features(frame)
    frame = add_near_atm_features(frame)
    frame = add_bs_features(frame)
    frame = add_labels(frame)
    frame = add_applicability(frame)
    return frame, readiness


def paired_frame(slot_frame: pd.DataFrame) -> pd.DataFrame:
    score = slot_frame[slot_frame["score_row"]].copy()
    key_cols = ["session_date", "decision_minute_et", "decision_ts_utc", "strike_key", "right"]
    hist = score[score["source"] == "historical"]
    live = score[score["source"] == "ibkr"]
    keep_cols = key_cols + ["contract_id", "strike", "raw_bid", "raw_ask", "raw_spread", "raw_quote_age_ms", "y_15m_conservative_return", "realized_5m_spx_vol"]
    keep_cols += FEATURE_NAMES + [f"{name}.__applicable" for name in FEATURE_NAMES]
    paired = hist[keep_cols].merge(
        live[keep_cols],
        on=key_cols,
        how="inner",
        suffixes=("_historical", "_ibkr"),
        validate="one_to_one",
    )
    paired["pair_key"] = (
        paired["session_date"].astype(str)
        + "|"
        + paired["decision_minute_et"].astype(str)
        + "|"
        + paired["strike_key"].astype(str)
        + "|"
        + paired["right"].astype(str)
    )
    return paired


def percentile(values: pd.Series | np.ndarray, q: float) -> float | None:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, q))


def corr_value(x: np.ndarray, y: np.ndarray) -> float | None:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if int(finite_mask.sum()) < 20:
        return None
    x2 = x[finite_mask]
    y2 = y[finite_mask]
    if np.nanstd(x2) == 0:
        return 0.0
    if np.nanstd(y2) == 0:
        return None
    return float(pearsonr(x2, y2).statistic)


def block_bootstrap_abs_ci(
    data: pd.DataFrame,
    feature_col: str,
    target_col: str,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    usable = data[["session_date", "decision_minute_et", feature_col, target_col]].dropna()
    if len(usable) < 30:
        return {"n": int(len(usable)), "rho": None, "ci_low": None, "ci_high": None, "abs_ci_upper": None}
    usable = usable.copy()
    usable["block"] = usable["session_date"].astype(str) + "_" + (
        usable["decision_minute_et"].map(minutes_since_open) // 30
    ).astype(str)
    rho = corr_value(usable[feature_col].to_numpy(dtype=float), usable[target_col].to_numpy(dtype=float))
    if rho is None:
        return {"n": int(len(usable)), "rho": None, "ci_low": None, "ci_high": None, "abs_ci_upper": None}
    rng = np.random.default_rng(RANDOM_STATE)
    boot: list[float] = []
    blocks_by_day = {
        day: [block for block, _ in day_frame.groupby("block")]
        for day, day_frame in usable.groupby("session_date")
    }
    rows_by_block = {block: block_frame for block, block_frame in usable.groupby("block")}
    for _ in range(repeats):
        parts: list[pd.DataFrame] = []
        for _, blocks in blocks_by_day.items():
            if not blocks:
                continue
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            parts.extend(rows_by_block[str(block)] for block in chosen)
        if not parts:
            continue
        sample = pd.concat(parts, ignore_index=True)
        value = corr_value(sample[feature_col].to_numpy(dtype=float), sample[target_col].to_numpy(dtype=float))
        if value is not None:
            boot.append(value)
    if len(boot) < 20:
        return {"n": int(len(usable)), "rho": rho, "ci_low": None, "ci_high": None, "abs_ci_upper": None}
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return {
        "n": int(len(usable)),
        "rho": rho,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "abs_ci_upper": max(abs(ci_low), abs(ci_high)),
    }


def l0_audit(paired: pd.DataFrame, out_dir: Path, definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    thresholds = definition["l0_thresholds"]
    distribution_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    bias_results: dict[str, Any] = {
        "schema_version": "Protocol101CanonicalV1BiasTestsV1",
        "bias_label_definition": "15-minute forward conservative IBKR-plane return: enter IBKR ask at t, exit IBKR bid at t+15m",
        "bootstrap": {"method": "30-minute block bootstrap within day", "repeats": BOOTSTRAP_REPEATS, "inference": "design-grade"},
        "tests": {},
    }
    per_day: dict[str, Any] = {}
    for session, day in paired.groupby("session_date"):
        per_day[session] = {
            "paired_slot_rows": int(len(day)),
            "paired_context_minutes": int(day["decision_minute_et"].nunique()),
            "first_decision_et": str(day["decision_minute_et"].min()),
            "last_decision_et": str(day["decision_minute_et"].max()),
            "features": {},
        }

    for name in CONTINUOUS_NAMES:
        hist_col = f"{name}_historical"
        live_col = f"{name}_ibkr"
        app = paired[f"{name}.__applicable_historical"].astype(bool) | paired[f"{name}.__applicable_ibkr"].astype(bool)
        both = app & paired[hist_col].notna() & paired[live_col].notna()
        ref = paired.loc[both, hist_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        scale = max(stdev(ref) if len(ref) > 1 else 0.0, float(definition["feature_drift_standardization_policy"]["scale_floor"]))
        signed = (paired.loc[both, live_col].astype(float) - paired.loc[both, hist_col].astype(float)) / scale
        raw_abs = (paired.loc[both, live_col].astype(float) - paired.loc[both, hist_col].astype(float)).abs()
        std_abs = signed.abs()
        coverage = float(both.sum() / max(app.sum(), 1))
        feature_bias: dict[str, Any] = {}
        bias_pass = True
        bias_data = paired.loc[both, ["session_date", "decision_minute_et", "y_15m_conservative_return_ibkr", "realized_5m_spx_vol_ibkr"]].copy()
        bias_data["signed_standardized_drift"] = signed.to_numpy(dtype=float)
        bias_data["abs_y"] = bias_data["y_15m_conservative_return_ibkr"].abs()
        for target_name, target_col in (
            ("y", "y_15m_conservative_return_ibkr"),
            ("abs_y", "abs_y"),
            ("realized_5m_spx_vol", "realized_5m_spx_vol_ibkr"),
        ):
            result = block_bootstrap_abs_ci(bias_data, "signed_standardized_drift", target_col)
            feature_bias[target_name] = result
            rho = result.get("rho")
            ci_upper = result.get("abs_ci_upper")
            if rho is None or ci_upper is None:
                bias_pass = False
            elif abs(float(rho)) > float(thresholds["bias_abs_rho_max"]) or float(ci_upper) > float(thresholds["bias_abs_ci_upper_max"]):
                bias_pass = False
        bias_results["tests"][name] = feature_bias
        p95 = percentile(std_abs, 95)
        status = "admit"
        if coverage < float(thresholds["continuous_coverage_min"]):
            status = "insufficient"
        elif p95 is None:
            status = "insufficient"
        elif p95 > float(thresholds["reject_standardized_abs_drift_p95_gt"]) or not bias_pass:
            status = "reject" if p95 > float(thresholds["reject_standardized_abs_drift_p95_gt"]) else "repair"
        elif p95 > float(thresholds["continuous_standardized_abs_drift_p95_max"]):
            status = "repair"
        field_rows.append(
            {
                "feature": name,
                "family": FEATURE_FAMILY[name],
                "feature_type": "continuous",
                "applicable_rows": int(app.sum()),
                "paired_finite_rows": int(both.sum()),
                "coverage": coverage,
                "standardization_scale": scale,
                "raw_abs_median": percentile(raw_abs, 50),
                "raw_abs_p95": percentile(raw_abs, 95),
                "raw_abs_max": percentile(raw_abs, 100),
                "standardized_abs_median": percentile(std_abs, 50),
                "standardized_abs_p95": p95,
                "standardized_abs_max": percentile(std_abs, 100),
                "bias_pass": bias_pass,
                "verdict": status,
            }
        )
        subset = paired.loc[both, ["session_date", "decision_minute_et", "strike_key", "right", "pair_key"]].copy()
        subset["feature"] = name
        subset["family"] = FEATURE_FAMILY[name]
        subset["historical_value"] = paired.loc[both, hist_col].to_numpy(dtype=float)
        subset["ibkr_value"] = paired.loc[both, live_col].to_numpy(dtype=float)
        subset["raw_abs_drift"] = raw_abs.to_numpy(dtype=float)
        subset["signed_standardized_drift"] = signed.to_numpy(dtype=float)
        subset["standardized_abs_drift"] = std_abs.to_numpy(dtype=float)
        subset["y_15m_conservative_return"] = paired.loc[both, "y_15m_conservative_return_ibkr"].to_numpy(dtype=float)
        subset["realized_5m_spx_vol"] = paired.loc[both, "realized_5m_spx_vol_ibkr"].to_numpy(dtype=float)
        distribution_rows.extend(subset.to_dict(orient="records"))
        for session, day in paired.loc[app].groupby("session_date"):
            day_both = both.loc[day.index]
            day_std = ((day.loc[day_both, live_col].astype(float) - day.loc[day_both, hist_col].astype(float)).abs() / scale)
            per_day[session]["features"][name] = {
                "applicable_rows": int(len(day)),
                "paired_finite_rows": int(day_both.sum()),
                "coverage": float(day_both.sum() / max(len(day), 1)),
                "standardized_abs_p95": percentile(day_std, 95),
            }

    for name in FLAG_NAMES:
        hist_col = f"{name}_historical"
        live_col = f"{name}_ibkr"
        app = paired[f"{name}.__applicable_historical"].astype(bool) | paired[f"{name}.__applicable_ibkr"].astype(bool)
        both = app & paired[hist_col].notna() & paired[live_col].notna()
        exact = both & (paired[hist_col].astype(float) == paired[live_col].astype(float))
        match_rate = float(exact.sum() / max(both.sum(), 1))
        verdict = "admit" if match_rate >= float(thresholds["flag_exact_match_rate_min"]) else "repair"
        field_rows.append(
            {
                "feature": name,
                "family": FEATURE_FAMILY[name],
                "feature_type": "flag_or_bucket",
                "applicable_rows": int(app.sum()),
                "paired_finite_rows": int(both.sum()),
                "coverage": float(both.sum() / max(app.sum(), 1)),
                "standardization_scale": None,
                "raw_abs_median": None,
                "raw_abs_p95": None,
                "raw_abs_max": None,
                "standardized_abs_median": None,
                "standardized_abs_p95": None,
                "standardized_abs_max": None,
                "exact_match_rate": match_rate,
                "bias_pass": None,
                "verdict": verdict,
            }
        )
        for session, day in paired.loc[app].groupby("session_date"):
            day_both = both.loc[day.index]
            day_exact = exact.loc[day.index]
            per_day[session]["features"][name] = {
                "applicable_rows": int(len(day)),
                "paired_finite_rows": int(day_both.sum()),
                "coverage": float(day_both.sum() / max(len(day), 1)),
                "exact_match_rate": float(day_exact.sum() / max(day_both.sum(), 1)),
            }

    field_path = out_dir / "field_divergence.csv"
    with field_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in field_rows for key in row}))
        writer.writeheader()
        writer.writerows(field_rows)
    write_json(out_dir / "bias_tests.json", bias_results)
    dist = pd.DataFrame(distribution_rows)
    dist.to_parquet(out_dir / "divergence_distributions.parquet", index=False)
    write_json(out_dir / "per_day_summary.json", {"schema_version": "Protocol101CanonicalV1PerDaySummaryV1", "days": per_day})
    return field_rows, bias_results, dist, per_day


def clean_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def auc_cv(frame: pd.DataFrame, feature_cols: list[str], label_col: str, *, hgb: bool) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    predictions: list[float] = []
    labels: list[int] = []
    if not feature_cols:
        return {"auc": None, "folds": [], "model": "hgb" if hgb else "logistic", "reason": "no_features"}
    for day in sorted(frame["session_date"].unique()):
        train = frame[frame["session_date"] != day]
        test = frame[frame["session_date"] == day]
        y_train = train[label_col].astype(int).to_numpy()
        y_test = test[label_col].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            fold_rows.append({"heldout_session": day, "auc": None, "reason": "single_class"})
            continue
        x_train = clean_matrix(train, feature_cols)
        x_test = clean_matrix(test, feature_cols)
        if hgb:
            model = HistGradientBoostingClassifier(max_iter=160, max_depth=3, learning_rate=0.05, l2_regularization=0.01, random_state=RANDOM_STATE)
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=RANDOM_STATE),
            )
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        fold_rows.append({"heldout_session": day, "auc": auc, "n_test": int(len(y_test))})
        predictions.extend(float(item) for item in proba)
        labels.extend(int(item) for item in y_test)
    pooled = float(roc_auc_score(labels, predictions)) if len(set(labels)) == 2 else None
    return {"auc": pooled, "folds": fold_rows, "model": "hgb" if hgb else "logistic"}


def permutation_importance_cv(frame: pd.DataFrame, feature_cols: list[str], label_col: str) -> list[dict[str, Any]]:
    totals: dict[str, list[float]] = {name: [] for name in feature_cols}
    for day in sorted(frame["session_date"].unique()):
        train = frame[frame["session_date"] != day]
        test = frame[frame["session_date"] == day]
        y_train = train[label_col].astype(int).to_numpy()
        y_test = test[label_col].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        if len(test) > 5000:
            test = test.sample(5000, random_state=RANDOM_STATE)
            y_test = test[label_col].astype(int).to_numpy()
        model = HistGradientBoostingClassifier(max_iter=160, max_depth=3, learning_rate=0.05, l2_regularization=0.01, random_state=RANDOM_STATE)
        model.fit(clean_matrix(train, feature_cols), y_train)
        result = permutation_importance(
            model,
            clean_matrix(test, feature_cols),
            y_test,
            scoring="roc_auc",
            n_repeats=3,
            random_state=RANDOM_STATE,
        )
        for name, value in zip(feature_cols, result.importances_mean, strict=False):
            totals[name].append(float(value))
    return [
        {"feature": name, "family": FEATURE_FAMILY.get(name, "control"), "importance_mean_auc_drop": float(np.nanmean(values))}
        for name, values in totals.items()
        if values
    ]


def l2_audit(slot_frame: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    score = slot_frame[slot_frame["score_row"]].copy()
    score["source_label"] = (score["source"] == "ibkr").astype(int)
    score["odd_even_label"] = (score["minute_since_open"].astype(int) % 2).astype(int)
    feature_cols = FEATURE_NAMES
    slot_hgb = auc_cv(score, feature_cols, "source_label", hgb=True)
    slot_logit = auc_cv(score, feature_cols, "source_label", hgb=False)
    context = (
        score.groupby(["source", "session_date", "decision_ts_utc", "decision_minute_et"], as_index=False)[feature_cols]
        .mean(numeric_only=True)
    )
    context["source_label"] = (context["source"] == "ibkr").astype(int)
    context_hgb = auc_cv(context, feature_cols, "source_label", hgb=True)
    context_logit = auc_cv(context, feature_cols, "source_label", hgb=False)
    historical = score[score["source"] == "historical"].copy()
    null_hgb = auc_cv(historical, feature_cols, "odd_even_label", hgb=True)
    control_cols = ["raw_bid", "raw_ask", "raw_spread", "raw_quote_age_ms"]
    price_cols = ["raw_bid", "raw_ask", "raw_spread"]
    positive_hgb = auc_cv(score, control_cols, "source_label", hgb=True)
    positive_logit = auc_cv(score, control_cols, "source_label", hgb=False)
    price_hgb = auc_cv(score, price_cols, "source_label", hgb=True)
    family_ablation: dict[str, Any] = {}
    for family in ("A", "B", "C", "D", "E"):
        cols = [name for name in feature_cols if FEATURE_FAMILY[name] == family]
        family_ablation[family] = auc_cv(score, cols, "source_label", hgb=True)
    importances = permutation_importance_cv(score, feature_cols, "source_label")
    with (out_dir / "permutation_importances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "family", "importance_mean_auc_drop"])
        writer.writeheader()
        writer.writerows(sorted(importances, key=lambda item: item["importance_mean_auc_drop"], reverse=True))
    write_json(out_dir / "family_ablation.json", {"schema_version": "Protocol101CanonicalV1FamilyAblationV1", "families": family_ablation})
    source = {
        "schema_version": "Protocol101CanonicalV1SourceDiscriminatorV1",
        "cv": "leave_one_day_out",
        "models": {
            "slot_hgb": slot_hgb,
            "slot_logistic": slot_logit,
            "context_hgb": context_hgb,
            "context_logistic": context_logit,
            "null_control_hgb_historical_odd_even": null_hgb,
            "positive_control_hgb_raw_bid_ask_spread_quote_age": positive_hgb,
            "positive_control_logistic_raw_bid_ask_spread_quote_age": positive_logit,
            "price_only_diagnostic_hgb_raw_bid_ask_spread": price_hgb,
        },
        "permutation_importances": importances,
        "family_ablation": family_ablation,
    }
    write_json(out_dir / "source_discriminator.json", source)
    return source


def feature_verdicts(field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for row in field_rows:
        verdict = str(row.get("verdict"))
        verdicts.append(
            {
                "feature": row["feature"],
                "family": row["family"],
                "feature_type": row["feature_type"],
                "verdict": verdict,
                "admitted": verdict == "admit",
                "repair_needed": verdict == "repair",
                "rejected": verdict == "reject",
                "insufficient": verdict == "insufficient",
                "coverage": row.get("coverage"),
                "standardized_abs_p95": row.get("standardized_abs_p95"),
                "exact_match_rate": row.get("exact_match_rate"),
                "bias_pass": row.get("bias_pass"),
            }
        )
    return verdicts


def infer_leak_families(source: dict[str, Any], verdicts: list[dict[str, Any]]) -> list[str]:
    leak_families: set[str] = set()
    for row in verdicts:
        if row["repair_needed"] or row["rejected"]:
            leak_families.add(str(row["family"]))
    for family, result in source.get("family_ablation", {}).items():
        auc = (result or {}).get("auc")
        if auc is not None and float(auc) > 0.55:
            leak_families.add(str(family))
    return sorted(leak_families)


def route(verdicts: list[dict[str, Any]], source: dict[str, Any], field_rows: list[dict[str, Any]], per_day: dict[str, Any]) -> dict[str, Any]:
    thresholds = canonical_feature_definition()["l2_thresholds"]
    models = source["models"]
    slot_auc = models["slot_hgb"].get("auc")
    context_auc = models["context_hgb"].get("auc")
    null_auc = models["null_control_hgb_historical_odd_even"].get("auc")
    positive_auc = models["positive_control_hgb_raw_bid_ask_spread_quote_age"].get("auc")
    price_auc = models["price_only_diagnostic_hgb_raw_bid_ask_spread"].get("auc")
    controls_ok = (
        null_auc is not None
        and float(thresholds["null_control_auc_min"]) <= float(null_auc) <= float(thresholds["null_control_auc_max"])
        and positive_auc is not None
        and float(positive_auc) >= float(thresholds["positive_control_auc_min"])
    )
    admitted = [row["feature"] for row in verdicts if row["admitted"]]
    repair = [row["feature"] for row in verdicts if row["repair_needed"]]
    rejected = [row["feature"] for row in verdicts if row["rejected"]]
    insufficient = [row["feature"] for row in verdicts if row["insufficient"]]
    leak_families = infer_leak_families(source, verdicts)
    decision = "canonical_v1_design_pass"
    if not controls_ok:
        decision = "canonical_v1_insufficient_artifacts"
    elif rejected or (slot_auc is not None and float(slot_auc) > float(thresholds["repair_auc_max"])) or (
        context_auc is not None and float(context_auc) > float(thresholds["repair_auc_max"])
    ):
        decision = "canonical_v1_rejected_bias_irreducible"
    elif repair or insufficient or (slot_auc is not None and float(slot_auc) > float(thresholds["pass_auc_max"])) or (
        context_auc is not None and float(context_auc) > float(thresholds["pass_auc_max"])
    ):
        decision = "canonical_v1_repair_iteration_needed"
    new_alpha = any(FEATURE_FAMILY[name] in {"C", "D", "E"} for name in admitted)
    return {
        "schema_version": "Protocol101CanonicalV1RoutingDecisionV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "routing_decision": decision,
        "new_alpha_families_admitted": bool(new_alpha),
        "admitted_features": admitted,
        "repair_features": repair,
        "rejected_features": rejected,
        "insufficient_features": insufficient,
        "l0_pooled_metrics": field_rows,
        "l0_per_day_metrics": per_day,
        "l2_slot_auc": slot_auc,
        "l2_context_auc": context_auc,
        "l2_null_control_auc": null_auc,
        "l2_positive_control_auc": positive_auc,
        "l2_price_only_auc": price_auc,
        "leak_families": leak_families,
        "repair_iterations_used": 0,
        "repair_budget": REPAIR_BUDGET,
        "evidence_grade": "design_burned_days_only",
        "control_status": {
            "null_control_pass": null_auc is not None
            and float(thresholds["null_control_auc_min"]) <= float(null_auc) <= float(thresholds["null_control_auc_max"]),
            "positive_control_pass": positive_auc is not None and float(positive_auc) >= float(thresholds["positive_control_auc_min"]),
            "price_only_is_diagnostic_only": True,
        },
        "claims_not_made": {
            "training_ready": True,
            "paper_ready": True,
            "promotion_ready": True,
            "real_money_ready": True,
        },
    }


def write_verdicts(out_dir: Path, verdict_rows: list[dict[str, Any]]) -> None:
    with (out_dir / "per_feature_verdicts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in verdict_rows for key in row}))
        writer.writeheader()
        writer.writerows(verdict_rows)


def write_report(out_dir: Path, routing: dict[str, Any], readiness: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Canonical Minute-Game v1 L0/L2 Design Audit",
        "",
        f"- Status: `{routing['routing_decision']}`",
        f"- Contract: `{CONTRACT}`",
        f"- Transform: `{TRANSFORM}`",
        "- Highest allowed claim: `canonical transform v1 L0/L2 design audit complete`",
        "- Evidence grade: `design_burned_days_only`",
        f"- New alpha families admitted: `{routing['new_alpha_families_admitted']}`",
        "",
        "## Inputs",
        "",
    ]
    for session, info in readiness.get("sessions", {}).items():
        lines.append(
            f"- `{session}`: historical_rows=`{info.get('historical_rows')}`, ibkr_rows=`{info.get('ibkr_rows')}`, "
            f"historical_slot_rows=`{info.get('historical_slot_rows')}`, ibkr_slot_rows=`{info.get('ibkr_slot_rows')}`"
        )
    lines.extend(
        [
            "",
            "## L0 Summary",
            "",
            f"- Admitted features: `{len(routing['admitted_features'])}`",
            f"- Repair features: `{len(routing['repair_features'])}`",
            f"- Rejected features: `{len(routing['rejected_features'])}`",
            f"- Insufficient features: `{len(routing['insufficient_features'])}`",
            "",
            "## L2 Source Discriminator",
            "",
            f"- Slot HGB AUC: `{routing['l2_slot_auc']}`",
            f"- Context HGB AUC: `{routing['l2_context_auc']}`",
            f"- Null control AUC: `{routing['l2_null_control_auc']}`",
            f"- Positive control AUC: `{routing['l2_positive_control_auc']}`",
            f"- Price-only diagnostic AUC: `{routing['l2_price_only_auc']}`",
            f"- Leak families: `{routing['leak_families']}`",
            "",
            "## Routing",
            "",
            f"- Routing decision: `{routing['routing_decision']}`",
            f"- Repair iterations used: `{routing['repair_iterations_used']}` / `{routing['repair_budget']}`",
            "",
            "Critical rule: `canonical_v1_design_pass` with `new_alpha_families_admitted=false` is not training-ready; it means only Family A/B survived and the audit merely rediscovered masked-v2.",
            "",
            "## Artifacts",
            "",
            "- `canonical_feature_definition.json`",
            "- `preregistration.json`",
            "- `field_divergence.csv`",
            "- `bias_tests.json`",
            "- `divergence_distributions.parquet`",
            "- `per_day_summary.json`",
            "- `source_discriminator.json`",
            "- `permutation_importances.csv`",
            "- `family_ablation.json`",
            "- `per_feature_verdicts.csv`",
            "- `routing_decision.json`",
            "- `progress.json`",
            "",
            "## Side Effects",
            "",
            "- trading_model_training: `false`",
            "- uplift_cv: `false`",
            "- threshold_tuning: `false`",
            "- broker_endpoint_called: `false`",
            "- paid_data_download: `false`",
            "- paper_submit: `false`",
            "- promotion/default/runtime/launchd edits: `false`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite audit artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = write_preregistration(out_dir)
    progress = {
        "schema_version": "Protocol101CanonicalV1ProgressV1",
        "attempt_id": out_dir.name,
        "status": "preregistered",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "model_training_executed": False,
        "uplift_cv_executed": False,
        "threshold_tuning_executed": False,
        "broker_endpoint_called": False,
        "paid_data_download": False,
        "paper_submit_allowed": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
    }
    write_json(out_dir / "progress.json", progress)
    definition = json.loads(Path(prereg["canonical_feature_definition_path"]).read_text())
    slot_frame, readiness = build_slot_frame(args.trace_prefix)
    write_json(out_dir / "input_readiness.json", {"schema_version": "Protocol101CanonicalV1InputReadinessV1", **readiness})
    if readiness.get("blockers") or slot_frame.empty:
        routing = {
            "schema_version": "Protocol101CanonicalV1RoutingDecisionV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_insufficient_artifacts",
            "new_alpha_families_admitted": False,
            "admitted_features": [],
            "repair_features": [],
            "rejected_features": [],
            "insufficient_features": FEATURE_NAMES,
            "l0_pooled_metrics": [],
            "l0_per_day_metrics": {},
            "l2_slot_auc": None,
            "l2_context_auc": None,
            "l2_null_control_auc": None,
            "l2_positive_control_auc": None,
            "l2_price_only_auc": None,
            "leak_families": [],
            "repair_iterations_used": 0,
            "repair_budget": REPAIR_BUDGET,
            "evidence_grade": "design_burned_days_only",
            "blockers": readiness.get("blockers", []),
        }
        write_json(out_dir / "routing_decision.json", routing)
        write_report(out_dir, routing, readiness)
        progress["status"] = "complete_insufficient_artifacts"
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    paired = paired_frame(slot_frame)
    readiness["paired_slot_rows_excluding_09_31"] = int(len(paired))
    field_rows, _, _, per_day = l0_audit(paired, out_dir, definition)
    source = l2_audit(slot_frame, out_dir)
    verdict_rows = feature_verdicts(field_rows)
    write_verdicts(out_dir, verdict_rows)
    routing = route(verdict_rows, source, field_rows, per_day)
    routing["attempt_id"] = out_dir.name
    routing["preregistration_sha256"] = sha256_path(out_dir / "preregistration.json")
    routing["canonical_feature_definition_sha256"] = sha256_path(out_dir / "canonical_feature_definition.json")
    routing["paired_slot_rows_excluding_09_31"] = int(len(paired))
    routing["input_readiness_blockers"] = readiness.get("blockers", [])
    write_json(out_dir / "routing_decision.json", routing)
    write_report(out_dir, routing, readiness)
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    progress["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(out_dir / "progress.json", progress)


if __name__ == "__main__":
    main()
