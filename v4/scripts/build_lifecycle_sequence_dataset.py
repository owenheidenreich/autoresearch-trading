"""Build a trade-level lifecycle sequence dataset for Protocol 054.

Protocol 060 is data infrastructure, not a new trading rule. It converts the
frozen Protocol 054 selected trades into:

* one trade-level attribution row per selected trade
* one step-level row per post-entry minute in the same contract path

The step table keeps causal state features separate from future path labels so
the next lifecycle model can learn hold/exit behavior without reconstructing
paths ad hoc or leaking future information into inputs.

No paid data is downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.dataset.spxw_0dte_neural import LabelPolicy
from v4.greeks.repair import compute_repaired_greeks
from v4.model.environment_diagnostics import time_bucket


LOOP_ID = "v4_aplus_hypothesis_060_lifecycle_sequence_dataset"
_NY = ZoneInfo("America/New_York")
_CONTRACT_MULTIPLIER = 100.0
_POLICY = LabelPolicy(0.50, 1.00, 25)

_NORMALIZED_COLUMNS = [
    "quote_time",
    "event_time",
    "contract_id",
    "raw_symbol",
    "strike",
    "right",
    "settlement_time_utc",
    "bid",
    "ask",
    "mid",
    "bid_size",
    "ask_size",
    "quote_gap_seconds",
    "option_ohlcv_volume",
    "stat_open_interest",
    "underlying_price",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
]

CAUSAL_STEP_FEATURE_COLUMNS = [
    "minutes_since_entry",
    "minutes_to_deadline",
    "minutes_to_forced_flat",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "bid_size",
    "ask_size",
    "quote_gap_seconds",
    "option_ohlcv_volume",
    "stat_open_interest",
    "underlying_price",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "giveback_fraction",
    "time_since_mfe_minutes",
    "pnl_velocity_1",
    "pnl_velocity_3",
    "pnl_velocity_5",
    "realized_pnl_vol_5",
    "realized_pnl_vol_10",
    "bid_over_entry_ask",
    "mid_over_entry_ask",
    "theta_over_mid",
    "gamma_theta_ratio",
    "time_theta_burden",
    "entry_edge",
    "entry_offset",
    "entry_is_call",
    "entry_is_put",
]

FUTURE_LABEL_COLUMNS = [
    "future_max_pnl",
    "future_min_pnl",
    "future_final_pnl",
    "future_max_delta",
    "future_min_delta",
    "future_final_delta",
    "future_recovery_100",
    "future_recovery_200",
    "future_decay_100",
    "future_decay_200",
    "future_reaches_baseline_pnl",
    "exit_now_regret_to_baseline",
    "exit_now_saves_vs_baseline",
    "exit_now_regret_to_future_best",
    "baseline_remaining_pnl_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-trades",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/"
            "v4_aplus_hypothesis_054_protocol052_lifecycle_10seed_validation/"
            "selected_trades_with_lifecycle_exits.json"
        ),
    )
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized_official_context"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"v4/audit/autoresearch/{LOOP_ID}"),
    )
    parser.add_argument("--max-trades", type=int, default=0)
    parser.add_argument("--forced-flat-time", default="15:55")
    return parser.parse_args()


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _finite_float(value: object, default: float | None = np.nan) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return default
    return float(numerator / denominator)


def _uid(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def _deadline(decision_time: pd.Timestamp, forced_flat_time: str) -> pd.Timestamp:
    max_hold = decision_time + pd.Timedelta(minutes=_POLICY.max_hold_minutes)
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    local_day = decision_time.tz_convert(_NY).date()
    forced = pd.Timestamp(local_day).replace(hour=hour, minute=minute, tzinfo=_NY).tz_convert("UTC")
    return min(max_hold, forced)


def _minutes_to_forced_flat(timestamp: pd.Timestamp, forced_flat_time: str) -> float:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    local = timestamp.tz_convert(_NY)
    forced = pd.Timestamp(local.date()).replace(hour=hour, minute=minute, tzinfo=_NY)
    return max(0.0, (forced - local).total_seconds() / 60.0)


def _normalized_session_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def _load_selected(path: Path, max_trades: int) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no selected trades found in {path}")
    if max_trades > 0:
        frame = frame.head(max_trades).copy()
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame["session"] = frame["session"].astype(str)
    frame["canonical_entry_uid"] = [
        _uid(row.session, row.decision_time, row.contract_id) for row in frame.itertuples(index=False)
    ]
    frame["trade_uid"] = [
        _uid(row.fold, row.split, row.seed, row.session, row.decision_time, row.contract_id, row.source_row)
        for row in frame.itertuples(index=False)
    ]
    for column in [
        "baseline_pnl",
        "dynamic_pnl",
        "hold_minutes",
        "edge",
        "offset",
        "predicted_headroom",
        "mfe",
        "mae",
        "giveback",
        "giveback_fraction",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_session(path: Path, contracts: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=_NORMALIZED_COLUMNS)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame = frame[frame["contract_id"].isin(contracts)].copy()
    if frame.empty:
        return frame
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["settlement_time_utc"] = pd.to_datetime(frame["settlement_time_utc"], utc=True)
    numeric_columns = [
        column
        for column in _NORMALIZED_COLUMNS
        if column not in {"quote_time", "event_time", "settlement_time_utc", "contract_id", "raw_symbol", "right"}
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = _fill_missing_greeks(frame)
    return frame.sort_values(["contract_id", "quote_time"]).reset_index(drop=True)


def _time_to_settlement_years(row: pd.Series) -> float | None:
    settlement = row.get("settlement_time_utc")
    quote_time = row.get("quote_time")
    if pd.isna(settlement) or pd.isna(quote_time):
        return None
    seconds = (pd.Timestamp(settlement) - pd.Timestamp(quote_time)).total_seconds()
    if seconds <= 0:
        return None
    return seconds / (365.0 * 24.0 * 60.0 * 60.0)


def _compute_greek_row(row: pd.Series) -> tuple[float, float, float, float, float]:
    mid = _finite_float(row.get("mid"))
    ask = _finite_float(row.get("ask"))
    bid = _finite_float(row.get("bid"))
    underlying = _finite_float(row.get("underlying_price"))
    strike = _finite_float(row.get("strike"))
    t_years = _time_to_settlement_years(row)
    if (
        underlying is None
        or strike is None
        or t_years is None
        or not np.isfinite(underlying)
        or not np.isfinite(strike)
        or underlying <= 0.0
        or strike <= 0.0
    ):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    is_call = str(row.get("right")) == "C"
    estimate = compute_repaired_greeks(
        S=underlying,
        K=strike,
        T=t_years,
        is_call=is_call,
        mid=mid,
        ask=ask,
        bid=bid,
        r=0.05,
        q=0.0,
    )
    if estimate is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return estimate.iv, estimate.delta, estimate.gamma, estimate.theta_per_day, estimate.vega


def _fill_missing_greeks(frame: pd.DataFrame) -> pd.DataFrame:
    greek_columns = ["iv", "delta", "gamma", "theta", "vega"]
    if frame.empty:
        return frame
    missing = frame[greek_columns].isna().any(axis=1)
    if not missing.any():
        return frame
    out = frame.copy()
    computed = out.loc[missing].apply(_compute_greek_row, axis=1, result_type="expand")
    computed.columns = greek_columns
    for column in greek_columns:
        out.loc[missing, column] = out.loc[missing, column].where(out.loc[missing, column].notna(), computed[column])
    return out


def _row_at_or_before(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = frame[frame["quote_time"] <= timestamp]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _first_exit_index(path: pd.DataFrame, entry_ask: float) -> tuple[int, str]:
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    target_pnl = _POLICY.take_profit_pct * entry_ask * _CONTRACT_MULTIPLIER
    for idx, pnl in enumerate(path["path_pnl"].to_numpy(dtype=float)):
        if pnl <= stop_pnl:
            return idx, "hard_stop"
        if pnl >= target_pnl:
            return idx, "target"
    return len(path) - 1, "time_flat"


def _dynamic_exit_index(path: pd.DataFrame, decision_time: pd.Timestamp, hold_minutes: float | None) -> int:
    if hold_minutes is None or not np.isfinite(hold_minutes):
        return len(path) - 1
    requested = decision_time + pd.Timedelta(minutes=float(hold_minutes))
    eligible = np.where((path["quote_time"] <= requested).to_numpy())[0]
    if len(eligible):
        return int(eligible[-1])
    return 0


def _velocity(values: np.ndarray, idx: int, lookback: int) -> float:
    lookback = min(lookback, idx)
    if lookback <= 0:
        return 0.0
    return float((values[idx] - values[idx - lookback]) / lookback)


def _vol(values: np.ndarray, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback + 1)
    window = values[start : idx + 1]
    if len(window) < 2:
        return 0.0
    return float(np.std(window, ddof=0))


def _build_step_rows(
    trade: pd.Series,
    path: pd.DataFrame,
    *,
    entry_row: pd.Series,
    entry_ask: float,
    deadline: pd.Timestamp,
    baseline_idx: int,
    dynamic_idx: int,
    forced_flat_time: str,
) -> list[dict]:
    pnls = path["path_pnl"].to_numpy(dtype=float)
    baseline_pnl = _finite_float(trade.get("baseline_pnl"), 0.0) or 0.0
    dynamic_pnl = _finite_float(trade.get("dynamic_pnl"), 0.0) or 0.0
    step_rows: list[dict] = []
    entry_mid = _finite_float(entry_row.get("mid"))
    entry_delta = _finite_float(entry_row.get("delta"))
    entry_gamma = _finite_float(entry_row.get("gamma"))
    entry_theta = _finite_float(entry_row.get("theta"))
    entry_iv = _finite_float(entry_row.get("iv"))
    for idx, row in path.reset_index(drop=True).iterrows():
        timestamp = pd.Timestamp(row["quote_time"])
        pnl = float(pnls[idx])
        so_far = pnls[: idx + 1]
        future = pnls[idx:]
        mfe_to_now = float(np.max(so_far))
        mae_to_now = float(np.min(so_far))
        mfe_idx = int(np.argmax(so_far))
        time_since_mfe = idx - mfe_idx
        giveback = max(0.0, mfe_to_now - pnl)
        mid = _finite_float(row.get("mid"))
        bid = _finite_float(row.get("bid"))
        ask = _finite_float(row.get("ask"))
        spread = (ask - bid) if ask is not None and bid is not None and np.isfinite(ask) and np.isfinite(bid) else np.nan
        spread_frac = _safe_ratio(spread, mid, np.nan)
        theta = _finite_float(row.get("theta"))
        gamma = _finite_float(row.get("gamma"))
        theta_over_mid = _safe_ratio(abs(theta), abs(mid), 0.0)
        gamma_theta = _safe_ratio(abs(gamma), abs(theta), 0.0)
        minutes_to_deadline = max(0.0, (deadline - timestamp).total_seconds() / 60.0)
        future_max = float(np.max(future))
        future_min = float(np.min(future))
        future_final = float(future[-1])
        step_rows.append(
            {
                "trade_uid": trade["trade_uid"],
                "canonical_entry_uid": trade["canonical_entry_uid"],
                "source_row": int(trade["source_row"]),
                "fold": trade.get("fold"),
                "split": trade.get("split"),
                "seed": int(trade.get("seed")),
                "session": trade.get("session"),
                "decision_time": pd.Timestamp(trade["decision_ts"]).isoformat(),
                "quote_time": timestamp.isoformat(),
                "local_time": timestamp.tz_convert(_NY).strftime("%H:%M"),
                "time_bucket": time_bucket(timestamp.to_pydatetime()),
                "contract_id": trade.get("contract_id"),
                "right": trade.get("right"),
                "step_idx": int(idx),
                "path_points": int(len(path)),
                "is_protocol054_exit_step": bool(idx == dynamic_idx),
                "is_after_protocol054_exit": bool(idx > dynamic_idx),
                "is_baseline_exit_step": bool(idx == baseline_idx),
                "is_after_baseline_exit": bool(idx > baseline_idx),
                "protocol054_exit_reason": trade.get("exit_reason"),
                "baseline_exit_reason": path.attrs.get("baseline_exit_reason"),
                "minutes_since_entry": max(1.0, (timestamp - trade["decision_ts"]).total_seconds() / 60.0),
                "minutes_to_deadline": minutes_to_deadline,
                "minutes_to_forced_flat": _minutes_to_forced_flat(timestamp, forced_flat_time),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_frac": spread_frac,
                "bid_size": _finite_float(row.get("bid_size"), 0.0),
                "ask_size": _finite_float(row.get("ask_size"), 0.0),
                "quote_gap_seconds": _finite_float(row.get("quote_gap_seconds"), np.nan),
                "option_ohlcv_volume": _finite_float(row.get("option_ohlcv_volume"), np.nan),
                "stat_open_interest": _finite_float(row.get("stat_open_interest"), np.nan),
                "underlying_price": _finite_float(row.get("underlying_price"), np.nan),
                "iv": _finite_float(row.get("iv"), np.nan),
                "delta": _finite_float(row.get("delta"), np.nan),
                "gamma": gamma,
                "theta": theta,
                "vega": _finite_float(row.get("vega"), np.nan),
                "current_pnl": pnl,
                "mfe_to_now": mfe_to_now,
                "mae_to_now": mae_to_now,
                "giveback_from_mfe": giveback,
                "giveback_fraction": _safe_ratio(giveback, mfe_to_now, 0.0) if mfe_to_now > 0 else 0.0,
                "time_since_mfe_minutes": float(time_since_mfe),
                "pnl_velocity_1": _velocity(pnls, idx, 1),
                "pnl_velocity_3": _velocity(pnls, idx, 3),
                "pnl_velocity_5": _velocity(pnls, idx, 5),
                "realized_pnl_vol_5": _vol(pnls, idx, 5),
                "realized_pnl_vol_10": _vol(pnls, idx, 10),
                "bid_over_entry_ask": _safe_ratio(bid, entry_ask, np.nan),
                "mid_over_entry_ask": _safe_ratio(mid, entry_ask, np.nan),
                "theta_over_mid": theta_over_mid,
                "gamma_theta_ratio": gamma_theta,
                "time_theta_burden": theta_over_mid * minutes_to_deadline,
                "entry_edge": _finite_float(trade.get("edge"), np.nan),
                "entry_offset": _finite_float(trade.get("offset"), np.nan),
                "entry_is_call": float(str(trade.get("right")) == "C"),
                "entry_is_put": float(str(trade.get("right")) == "P"),
                "entry_ask": entry_ask,
                "entry_mid": entry_mid,
                "entry_delta": entry_delta,
                "entry_gamma": entry_gamma,
                "entry_theta": entry_theta,
                "entry_iv": entry_iv,
                "protocol054_recorded_pnl": dynamic_pnl,
                "baseline_recorded_pnl": baseline_pnl,
                "future_max_pnl": future_max,
                "future_min_pnl": future_min,
                "future_final_pnl": future_final,
                "future_max_delta": future_max - pnl,
                "future_min_delta": future_min - pnl,
                "future_final_delta": future_final - pnl,
                "future_recovery_100": bool(future_max - pnl >= 100.0),
                "future_recovery_200": bool(future_max - pnl >= 200.0),
                "future_decay_100": bool(pnl - future_min >= 100.0),
                "future_decay_200": bool(pnl - future_min >= 200.0),
                "future_reaches_baseline_pnl": bool(future_max >= baseline_pnl),
                "exit_now_regret_to_baseline": baseline_pnl - pnl,
                "exit_now_saves_vs_baseline": pnl - baseline_pnl,
                "exit_now_regret_to_future_best": future_max - pnl,
                "baseline_remaining_pnl_delta": baseline_pnl - pnl,
            }
        )
    return step_rows


def _build_for_trade(
    trade: pd.Series,
    contract_rows: pd.DataFrame,
    *,
    forced_flat_time: str,
) -> tuple[dict, list[dict]]:
    decision_time = _utc_timestamp(trade["decision_time"])
    deadline = _deadline(decision_time, forced_flat_time)
    entry = _row_at_or_before(contract_rows, decision_time)
    base_trade = {
        "trade_uid": trade["trade_uid"],
        "canonical_entry_uid": trade["canonical_entry_uid"],
        "source_row": int(trade["source_row"]),
        "fold": trade.get("fold"),
        "split": trade.get("split"),
        "seed": int(trade.get("seed")),
        "session": trade.get("session"),
        "decision_time": decision_time.isoformat(),
        "local_time": decision_time.tz_convert(_NY).strftime("%H:%M"),
        "time_bucket": time_bucket(decision_time.to_pydatetime()),
        "contract_id": trade.get("contract_id"),
        "right": trade.get("right"),
        "offset": _finite_float(trade.get("offset"), np.nan),
        "edge": _finite_float(trade.get("edge"), np.nan),
        "baseline_pnl": _finite_float(trade.get("baseline_pnl"), 0.0),
        "protocol054_pnl": _finite_float(trade.get("dynamic_pnl"), 0.0),
        "protocol054_exit_reason": trade.get("exit_reason"),
        "protocol054_hold_minutes": _finite_float(trade.get("hold_minutes"), np.nan),
        "protocol054_predicted_headroom": _finite_float(trade.get("predicted_headroom"), np.nan),
        "deadline": deadline.isoformat(),
    }
    if entry is None:
        return {**base_trade, "path_status": "missing_entry_quote"}, []
    entry_ask = _finite_float(entry.get("ask"))
    if entry_ask is None or not np.isfinite(entry_ask) or entry_ask <= 0:
        return {**base_trade, "path_status": "invalid_entry_ask", "entry_ask": entry_ask}, []
    path = contract_rows[
        (contract_rows["quote_time"] > decision_time)
        & (contract_rows["quote_time"] <= deadline)
        & contract_rows["bid"].notna()
    ].copy()
    if path.empty:
        return {**base_trade, "path_status": "missing_future_path", "entry_ask": entry_ask}, []
    path["path_pnl"] = (path["bid"].astype(float) - entry_ask) * _CONTRACT_MULTIPLIER
    path = path.reset_index(drop=True)
    baseline_idx, baseline_reason = _first_exit_index(path, entry_ask)
    path.attrs["baseline_exit_reason"] = baseline_reason
    dynamic_idx = _dynamic_exit_index(path, decision_time, _finite_float(trade.get("hold_minutes")))
    pnls = path["path_pnl"].to_numpy(dtype=float)
    baseline_path_pnl = float(pnls[baseline_idx])
    protocol054_path_pnl = float(pnls[dynamic_idx])
    protocol054_recorded_pnl = _finite_float(trade.get("dynamic_pnl"), protocol054_path_pnl) or protocol054_path_pnl
    baseline_recorded_pnl = _finite_float(trade.get("baseline_pnl"), baseline_path_pnl) or baseline_path_pnl
    post_dynamic = pnls[dynamic_idx + 1 :]
    post_dynamic_max = protocol054_path_pnl if len(post_dynamic) == 0 else float(np.max(post_dynamic))
    post_dynamic_min = protocol054_path_pnl if len(post_dynamic) == 0 else float(np.min(post_dynamic))
    post_dynamic_final = protocol054_path_pnl if len(post_dynamic) == 0 else float(post_dynamic[-1])
    lifecycle_delta = protocol054_recorded_pnl - baseline_recorded_pnl
    attribution_bucket = "neutral"
    if baseline_recorded_pnl < 0.0 and lifecycle_delta > 0.0:
        attribution_bucket = "saved_or_reduced_loss"
    elif baseline_recorded_pnl > 0.0 and lifecycle_delta > 0.0:
        attribution_bucket = "improved_winner"
    elif baseline_recorded_pnl > 0.0 and lifecycle_delta < 0.0:
        attribution_bucket = "clipped_winner"
    elif baseline_recorded_pnl < 0.0 and lifecycle_delta < 0.0:
        attribution_bucket = "worsened_loser"
    recovered_to_baseline = bool(baseline_recorded_pnl > protocol054_recorded_pnl and post_dynamic_max >= baseline_recorded_pnl)
    trade_row = {
        **base_trade,
        "path_status": "ok",
        "entry_quote_time": pd.Timestamp(entry["quote_time"]).isoformat(),
        "entry_bid": _finite_float(entry.get("bid"), np.nan),
        "entry_ask": entry_ask,
        "entry_mid": _finite_float(entry.get("mid"), np.nan),
        "entry_spread": _finite_float(entry.get("ask"), np.nan) - _finite_float(entry.get("bid"), np.nan),
        "entry_spread_frac": _safe_ratio(
            _finite_float(entry.get("ask"), np.nan) - _finite_float(entry.get("bid"), np.nan),
            _finite_float(entry.get("mid"), np.nan),
            np.nan,
        ),
        "entry_bid_size": _finite_float(entry.get("bid_size"), np.nan),
        "entry_ask_size": _finite_float(entry.get("ask_size"), np.nan),
        "entry_underlying_price": _finite_float(entry.get("underlying_price"), np.nan),
        "entry_iv": _finite_float(entry.get("iv"), np.nan),
        "entry_delta": _finite_float(entry.get("delta"), np.nan),
        "entry_gamma": _finite_float(entry.get("gamma"), np.nan),
        "entry_theta": _finite_float(entry.get("theta"), np.nan),
        "path_points": int(len(path)),
        "path_first_time": pd.Timestamp(path.iloc[0]["quote_time"]).isoformat(),
        "path_last_time": pd.Timestamp(path.iloc[-1]["quote_time"]).isoformat(),
        "path_max_pnl": float(np.max(pnls)),
        "path_min_pnl": float(np.min(pnls)),
        "path_final_pnl": float(pnls[-1]),
        "path_mfe_step": int(np.argmax(pnls)),
        "path_mae_step": int(np.argmin(pnls)),
        "baseline_exit_step": int(baseline_idx),
        "baseline_exit_time": pd.Timestamp(path.iloc[baseline_idx]["quote_time"]).isoformat(),
        "baseline_exit_reason": baseline_reason,
        "baseline_path_pnl": baseline_path_pnl,
        "baseline_path_vs_recorded": baseline_path_pnl - baseline_recorded_pnl,
        "protocol054_exit_step": int(dynamic_idx),
        "protocol054_exit_time": pd.Timestamp(path.iloc[dynamic_idx]["quote_time"]).isoformat(),
        "protocol054_path_pnl": protocol054_path_pnl,
        "protocol054_path_vs_recorded": protocol054_path_pnl - protocol054_recorded_pnl,
        "lifecycle_delta": lifecycle_delta,
        "attribution_bucket": attribution_bucket,
        "recovered_to_baseline_after_protocol054_exit": recovered_to_baseline,
        "recovered_positive_after_protocol054_exit": bool(protocol054_recorded_pnl < 0.0 and post_dynamic_max > 0.0),
        "kept_falling_after_protocol054_exit": bool(post_dynamic_min < protocol054_recorded_pnl),
        "post_protocol054_max_pnl": post_dynamic_max,
        "post_protocol054_min_pnl": post_dynamic_min,
        "post_protocol054_final_pnl": post_dynamic_final,
        "post_protocol054_max_delta": post_dynamic_max - protocol054_recorded_pnl,
        "post_protocol054_min_delta": post_dynamic_min - protocol054_recorded_pnl,
        "post_protocol054_final_delta": post_dynamic_final - protocol054_recorded_pnl,
    }
    step_rows = _build_step_rows(
        trade,
        path,
        entry_row=entry,
        entry_ask=entry_ask,
        deadline=deadline,
        baseline_idx=baseline_idx,
        dynamic_idx=dynamic_idx,
        forced_flat_time=forced_flat_time,
    )
    return trade_row, step_rows


def _build_dataset(selected: pd.DataFrame, *, normalized_dir: Path, forced_flat_time: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_rows: list[dict] = []
    step_rows: list[dict] = []
    for session, session_group in selected.groupby("session", sort=True):
        session_path = _normalized_session_path(normalized_dir, str(session))
        if session_path is None:
            for _, trade in session_group.iterrows():
                trade_rows.append(
                    {
                        "trade_uid": trade["trade_uid"],
                        "canonical_entry_uid": trade["canonical_entry_uid"],
                        "source_row": int(trade["source_row"]),
                        "fold": trade.get("fold"),
                        "split": trade.get("split"),
                        "seed": int(trade.get("seed")),
                        "session": trade.get("session"),
                        "decision_time": _utc_timestamp(trade["decision_time"]).isoformat(),
                        "contract_id": trade.get("contract_id"),
                        "path_status": "missing_normalized_session",
                    }
                )
            continue
        contracts = set(session_group["contract_id"].astype(str))
        normalized = _load_session(session_path, contracts)
        by_contract = {contract_id: group for contract_id, group in normalized.groupby("contract_id", sort=False)}
        for _, trade in session_group.iterrows():
            contract_rows = by_contract.get(str(trade["contract_id"]))
            if contract_rows is None or contract_rows.empty:
                trade_rows.append(
                    {
                        "trade_uid": trade["trade_uid"],
                        "canonical_entry_uid": trade["canonical_entry_uid"],
                        "source_row": int(trade["source_row"]),
                        "fold": trade.get("fold"),
                        "split": trade.get("split"),
                        "seed": int(trade.get("seed")),
                        "session": trade.get("session"),
                        "decision_time": _utc_timestamp(trade["decision_time"]).isoformat(),
                        "contract_id": trade.get("contract_id"),
                        "path_status": "missing_contract_path",
                    }
                )
                continue
            trade_row, rows = _build_for_trade(trade, contract_rows, forced_flat_time=forced_flat_time)
            trade_rows.append(trade_row)
            step_rows.extend(rows)
    return pd.DataFrame(trade_rows), pd.DataFrame(step_rows)


def _group_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    if frame.empty:
        return []
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = {column: key for column, key in zip(columns, keys)}
        out.update(
            {
                "trades": int(len(group)),
                "baseline_pnl": float(pd.to_numeric(group.get("baseline_pnl"), errors="coerce").fillna(0.0).sum()),
                "protocol054_pnl": float(pd.to_numeric(group.get("protocol054_pnl"), errors="coerce").fillna(0.0).sum()),
                "lifecycle_delta": float(pd.to_numeric(group.get("lifecycle_delta"), errors="coerce").fillna(0.0).sum()),
                "avg_steps": float(pd.to_numeric(group.get("path_points"), errors="coerce").fillna(0.0).mean()),
                "recovered_to_baseline_fraction": float(
                    group.get("recovered_to_baseline_after_protocol054_exit", pd.Series(False, index=group.index)).fillna(False).mean()
                ),
            }
        )
        rows.append(out)
    return sorted(rows, key=lambda row: tuple(str(row.get(column, "")) for column in columns))


def _safe_abs_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().abs()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 060 Lifecycle Sequence Dataset",
        "",
        "No paid data was downloaded. This is a dataset build, not a model or trading-rule change.",
        "",
        "## Purpose",
        "",
        "Build a trade-level and step-level lifecycle attribution dataset from frozen Protocol 054 selected trades.",
        "The step table separates causal post-entry state from future path labels for a later sequence model.",
        "",
        "## Coverage",
        "",
        f"- Selected trades: `{payload['selected_trades']}`",
        f"- Trade rows written: `{payload['trade_rows']}`",
        f"- Step rows written: `{payload['step_rows']}`",
        f"- Unique canonical entries: `{payload['unique_canonical_entries']}`",
        f"- Path status: `{payload['path_status_counts']}`",
        f"- Greek step coverage: `{payload['feature_coverage'].get('gamma', 0.0):.3f}` gamma / `{payload['feature_coverage'].get('theta', 0.0):.3f}` theta",
        "",
        "## Attribution By Split",
        "",
        "| Split | Trades | Baseline PnL | Protocol 054 PnL | Delta | Avg Steps | Recovered To Baseline |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_split"]:
        lines.append(
            f"| {row['split']} | {row['trades']} | {row['baseline_pnl']:.0f} | "
            f"{row['protocol054_pnl']:.0f} | {row['lifecycle_delta']:.0f} | "
            f"{row['avg_steps']:.1f} | {row['recovered_to_baseline_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Trades: `{payload['outputs']['trades']}`",
        f"- Steps: `{payload['outputs']['steps']}`",
        f"- Schema: `{payload['outputs']['schema']}`",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
    ]
    path.write_text("\n".join(lines))


def _schema_payload(args: argparse.Namespace) -> dict:
    return {
        "loop_id": LOOP_ID,
        "paid_data_downloaded": False,
        "source_selected_trades": str(args.selected_trades),
        "normalized_dir": str(args.normalized_dir),
        "policy": {
            "name": _POLICY.name,
            "stop_loss_pct": _POLICY.stop_loss_pct,
            "take_profit_pct": _POLICY.take_profit_pct,
            "max_hold_minutes": _POLICY.max_hold_minutes,
            "forced_flat_time": args.forced_flat_time,
        },
        "causal_step_feature_columns": CAUSAL_STEP_FEATURE_COLUMNS,
        "future_label_columns": FUTURE_LABEL_COLUMNS,
        "identity_columns": [
            "trade_uid",
            "canonical_entry_uid",
            "fold",
            "split",
            "seed",
            "session",
            "decision_time",
            "contract_id",
            "right",
        ],
        "notes": [
            "Features are computed from information available at or before each step quote_time.",
            "Future label columns are for supervised training targets and audits only.",
            "Rows after Protocol 054's exit are retained and flagged so recovery/decay after the frozen exit can be learned or audited.",
            "Missing IV/Greeks in normalized quote paths are computed with Black-Scholes from mid, official-context underlying price, strike, right, and settlement time.",
            "Theta is stored per calendar day; vega is stored per 1.0 volatility change.",
        ],
    }


def main() -> int:
    args = parse_args()
    selected = _load_selected(args.selected_trades, args.max_trades)
    trade_frame, step_frame = _build_dataset(selected, normalized_dir=args.normalized_dir, forced_flat_time=args.forced_flat_time)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_dir / "protocol054_lifecycle_trades.parquet"
    steps_path = args.out_dir / "protocol054_lifecycle_steps.parquet"
    schema_path = args.out_dir / "schema.json"
    report_json_path = args.out_dir / "report.json"
    report_md_path = args.out_dir / "report.md"
    trade_frame.to_parquet(trades_path, index=False)
    step_frame.to_parquet(steps_path, index=False)
    schema_path.write_text(json.dumps(_schema_payload(args), indent=2, allow_nan=False) + "\n")
    ok_trades = trade_frame[trade_frame["path_status"] == "ok"].copy()
    feature_coverage = {}
    for column in ["iv", "delta", "gamma", "theta", "vega", "underlying_price", "current_pnl"]:
        if column in step_frame.columns and len(step_frame):
            feature_coverage[column] = float(step_frame[column].notna().mean())
    payload = {
        "loop_id": LOOP_ID,
        "paid_data_downloaded": False,
        "source_selected_trades": str(args.selected_trades),
        "normalized_dir": str(args.normalized_dir),
        "selected_trades": int(len(selected)),
        "trade_rows": int(len(trade_frame)),
        "step_rows": int(len(step_frame)),
        "unique_canonical_entries": int(trade_frame["canonical_entry_uid"].nunique()) if "canonical_entry_uid" in trade_frame else 0,
        "path_status_counts": trade_frame["path_status"].value_counts(dropna=False).to_dict() if "path_status" in trade_frame else {},
        "feature_coverage": feature_coverage,
        "by_split": _group_summary(ok_trades, ["split"]),
        "by_attribution_bucket": _group_summary(ok_trades, ["split", "attribution_bucket"]),
        "path_reconciliation": {
            "baseline_path_vs_recorded_abs_p50": _safe_abs_quantile(ok_trades.get("baseline_path_vs_recorded", pd.Series(dtype=float)), 0.50),
            "baseline_path_vs_recorded_abs_p95": _safe_abs_quantile(ok_trades.get("baseline_path_vs_recorded", pd.Series(dtype=float)), 0.95),
            "protocol054_path_vs_recorded_abs_p50": _safe_abs_quantile(ok_trades.get("protocol054_path_vs_recorded", pd.Series(dtype=float)), 0.50),
            "protocol054_path_vs_recorded_abs_p95": _safe_abs_quantile(ok_trades.get("protocol054_path_vs_recorded", pd.Series(dtype=float)), 0.95),
        },
        "outputs": {
            "trades": str(trades_path),
            "steps": str(steps_path),
            "schema": str(schema_path),
            "report_md": str(report_md_path),
        },
        "decision": (
            "Keep Protocol 060 as the lifecycle-sequence data foundation. "
            "Do not treat this as model promotion; use it to train and audit the next sequence-model protocol."
        ),
    }
    report_json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(report_md_path, payload)
    print(json.dumps({key: payload[key] for key in ["selected_trades", "trade_rows", "step_rows", "path_status_counts"]}, indent=2))
    print(report_md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
