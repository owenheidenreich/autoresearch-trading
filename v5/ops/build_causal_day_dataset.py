"""Build causal day episodes, ladder state, and OTM-to-ITM labels.

No policy is fitted here.  Each output separates information available at a
decision boundary from forward outcomes used only for a model-free atlas or a
future, separately authorized fit.

Outputs
-------
``candles.parquet``
    Raw ES bars and the minute boundary at which each bar becomes usable.
``minutes.parquet``
    One causal state row per decision/management minute.
``ladder.parquet``
    Every live two-sided contract inside the signed near-ATM state band.
``candidates.parquet``
    Every affordable OTM entry action plus fixed 60/90/120-minute outcomes.
``atlas.parquet``
    Underlying forward structure for every possible entry minute, with the two
    declared key-time bands marked rather than selected.

The files intentionally do not materialize every candidate-minute forward
path.  The original quote files remain the path source; when an authorized
entry policy eventually exists, only its out-of-fold entries should be expanded
for an exit fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import (
    AFTERNOON_START,
    CONTRACT_MULTIPLIER,
    ENTRY_MINUTES,
    FIRST_DECISION_MINUTE,
    KEY_MINUTES,
    LAST_ENTRY_MINUTE,
    LAST_QUOTE_MINUTE,
    MAX_ENTRY_ASK_USD,
    NEAR_ATM_POINTS,
    QUOTE_AGE_CAP_MS,
    QUOTE_MINUTES,
    causal_es_bar_minute,
    eligible_entry,
    live_two_sided,
    session_from_path,
    signed_moneyness,
)
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.research import greeks as gk


ET = "America/New_York"
HORIZONS = (60, 90, 120)
ITM_DEPTHS = (0, 10, 20, 30)
FIRST_TOUCH_GAINS = (0.30, 0.50, 1.00)
FIRST_TOUCH_LOSS = -0.30
LOOKBACKS = (1, 3, 5, 15, 30, 60, 120)
STATE_NODES = np.arange(-25.0, 25.1, 5.0)
NODE_TOLERANCE = 2.5 + 1e-9
MAX_CANDLE_SEQUENCE = 120

KEY_BANDS = (("09:50", "10:10", "magic_time"), ("13:20", "13:40", "algo"))

QUOTE_COLUMNS = (
    "event_time",
    "expiry",
    "contract_id",
    "raw_symbol",
    "strike",
    "right",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "quote_age_ms",
    "volume",
    "open_interest",
    "underlying_price",
)


class DatasetError(RuntimeError):
    """A session cannot satisfy the declared causal dataset contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minute_number(minute: str) -> int:
    hour, value = (int(part) for part in minute.split(":"))
    return 60 * hour + value


def minute_label(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def regime_for_minute(minute: str) -> str:
    """The total flat-account router declared in GOAL.md section 4.0."""

    if "09:30" <= minute < AFTERNOON_START:
        return "morning"
    if AFTERNOON_START <= minute <= LAST_QUOTE_MINUTE:
        return "afternoon"
    raise DatasetError(f"minute outside declared regular-session router: {minute}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)


def prepare_quotes(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    """Normalize one session without looking beyond a row's own timestamp."""

    missing = sorted(set(QUOTE_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetError(f"{session}: quote columns missing: {missing}")
    got = frame.loc[:, QUOTE_COLUMNS].copy()
    stamped = pd.to_datetime(got["event_time"], utc=True).dt.tz_convert(ET)
    got["minute"] = stamped.dt.strftime("%H:%M")
    got = got[got["minute"].isin(QUOTE_MINUTES)].copy()
    _numeric(
        got,
        (
            "strike",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "mid",
            "quote_age_ms",
            "volume",
            "open_interest",
            "underlying_price",
        ),
    )
    got["contract_id"] = got["contract_id"].astype(str)
    got["right"] = got["right"].astype(str)
    got = got.sort_values(["minute", "contract_id", "event_time"]).drop_duplicates(
        ["minute", "contract_id"], keep="last"
    )
    expiry = pd.to_datetime(got["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    if not expiry.eq(session).all():
        raise DatasetError(f"{session}: non-same-day contract in 0DTE quote file")
    return got.reset_index(drop=True)


def prepare_es(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    """Raw ES bars with their exact causal availability boundary."""

    needed = {"open", "high", "low", "close", "volume"}
    if not needed <= set(frame.columns):
        raise DatasetError(f"{session}: ES columns missing: {sorted(needed - set(frame.columns))}")
    got = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    stamped = pd.to_datetime(got.index, utc=True).tz_convert(ET)
    got["bar_minute"] = stamped.strftime("%H:%M")
    got = got[got["bar_minute"].between("09:30", "15:59")].reset_index(drop=True)
    _numeric(got, ("open", "high", "low", "close", "volume"))
    if tuple(got["bar_minute"]) != tuple(minute_label(i) for i in range(570, 960)):
        raise DatasetError(f"{session}: ES regular-session clock is not the complete 09:30-15:59 grid")
    got.insert(0, "session", session)
    got["knowable_at"] = [minute_label(minute_number(value) + 1) for value in got["bar_minute"]]
    return got


def candle_states(candles: pd.DataFrame) -> pd.DataFrame:
    """Causal chart state; every row uses bars ending at its boundary or earlier."""

    values = candles.reset_index(drop=True)
    close = values["close"].to_numpy(float)
    high = values["high"].to_numpy(float)
    low = values["low"].to_numpy(float)
    open_ = values["open"].to_numpy(float)
    volume = values["volume"].to_numpy(float)
    one_minute_returns = np.r_[np.nan, close[1:] / close[:-1] - 1.0]
    first_open = float(open_[0])
    rows: list[dict[str, Any]] = []

    for i, decision in enumerate(values["knowable_at"]):
        if decision < FIRST_DECISION_MINUTE or decision > LAST_QUOTE_MINUTE:
            continue
        body_top = max(open_[i], close[i])
        body_bottom = min(open_[i], close[i])
        session_high = float(np.max(high[: i + 1]))
        session_low = float(np.min(low[: i + 1]))
        session_range = session_high - session_low
        row: dict[str, Any] = {
            "session": values.iloc[i]["session"],
            "minute": decision,
            "regime": regime_for_minute(decision),
            "entry_clock_open": FIRST_DECISION_MINUTE <= decision <= LAST_ENTRY_MINUTE,
            "history_minutes": i + 1,
            "latest_es_bar_minute": values.iloc[i]["bar_minute"],
            "es_open": open_[i],
            "es_high": high[i],
            "es_low": low[i],
            "es_close": close[i],
            "es_volume": volume[i],
            "session_open_es": first_open,
            "session_high_es": session_high,
            "session_low_es": session_low,
            "move_from_open_points": close[i] - first_open,
            "move_from_open_rel": close[i] / first_open - 1.0,
            "session_range_points": session_range,
            "range_position": (close[i] - session_low) / session_range if session_range > 0 else 0.5,
            "body_points": close[i] - open_[i],
            "upper_wick_points": high[i] - body_top,
            "lower_wick_points": body_bottom - low[i],
            "log_volume": np.log1p(max(volume[i], 0.0)),
            "volume_vs_expanding_median": (
                volume[i] / np.median(volume[: i + 1])
                if np.median(volume[: i + 1]) > 0
                else np.nan
            ),
            "cumulative_volume": float(np.sum(volume[: i + 1])),
            "minutes_from_open": minute_number(decision) - minute_number("09:30"),
            "minutes_to_close": minute_number("16:00") - minute_number(decision),
            "minutes_to_magic_time": minute_number("10:00") - minute_number(decision),
            "minutes_to_algo": minute_number("13:30") - minute_number(decision),
        }
        for lookback in LOOKBACKS:
            start = max(0, i - lookback)
            observed = i - start
            row[f"history_{lookback}m"] = observed
            row[f"move_{lookback}m_rel"] = close[i] / close[start] - 1.0 if observed else 0.0
            local_returns = one_minute_returns[start + 1 : i + 1]
            local_returns = local_returns[np.isfinite(local_returns)]
            row[f"realised_vol_{lookback}m"] = (
                float(np.std(local_returns, ddof=0)) if local_returns.size else 0.0
            )
            row[f"range_{lookback}m_points"] = float(
                np.max(high[start : i + 1]) - np.min(low[start : i + 1])
            )
            row[f"volume_{lookback}m"] = float(np.sum(volume[start : i + 1]))
        rows.append(row)
    return pd.DataFrame(rows)


def _minutes_to_close(minutes: pd.Series) -> np.ndarray:
    return np.asarray(
        [minute_number(LAST_QUOTE_MINUTE) - minute_number(value) for value in minutes],
        dtype=float,
    )


def _ladder_state(
    quotes: pd.DataFrame, session: str, *, whole_live_chain: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enrich either the whole live chain or the declared near-ATM state band."""

    live = live_two_sided(quotes)
    spot = quotes["underlying_price"].to_numpy(float)
    strike = quotes["strike"].to_numpy(float)
    money = signed_moneyness(spot, strike, quotes["right"].to_numpy())
    allowed_state = live & np.isfinite(money)
    if not whole_live_chain:
        allowed_state &= np.abs(money) <= NEAR_ATM_POINTS
    state = quotes[allowed_state].copy()
    state["moneyness_itm_points"] = money[state.index]
    state["spread"] = state["ask"] - state["bid"]
    state["spread_usd"] = state["spread"] * CONTRACT_MULTIPLIER
    state["is_call"] = state["right"].eq("C")
    state["minutes_to_expiry"] = _minutes_to_close(state["minute"])
    nearest = np.round(state["moneyness_itm_points"] / 5.0) * 5.0
    state["tensor_node_itm_points"] = nearest
    state["tensor_node_distance"] = np.abs(state["moneyness_itm_points"] - nearest)
    state["tensor_node_valid"] = (
        state["tensor_node_distance"].le(NODE_TOLERANCE)
        & state["tensor_node_itm_points"].isin(STATE_NODES)
    )

    solved = gk.greeks_batch(
        state["mid"].to_numpy(float),
        state["underlying_price"].to_numpy(float),
        state["strike"].to_numpy(float),
        state["minutes_to_expiry"].to_numpy(float),
        state["is_call"].to_numpy(bool),
    )
    state["self_iv"] = solved["iv"]
    state["self_delta"] = solved["delta"]
    state["self_gamma"] = solved["gamma"]
    state["self_gamma_dollars"] = (
        solved["gamma"] * state["underlying_price"].to_numpy(float) ** 2 / 100.0
    )
    state["self_theta_per_minute"] = solved["theta_per_minute"]
    state["self_vega"] = solved["vega"]
    state["entry_eligible"] = eligible_entry(state).to_numpy(bool)
    state.insert(0, "session", session)

    candidates = state[
        state["entry_eligible"]
        & state["minute"].between(FIRST_DECISION_MINUTE, LAST_ENTRY_MINUTE)
    ].copy()
    candidates["entry_ask_usd"] = candidates["ask"] * CONTRACT_MULTIPLIER
    candidates["entry_bid_usd"] = candidates["bid"] * CONTRACT_MULTIPLIER
    candidates["entry_mid_usd"] = candidates["mid"] * CONTRACT_MULTIPLIER
    candidates["entry_minute"] = candidates["minute"]
    candidates["entry_regime"] = [regime_for_minute(value) for value in candidates["minute"]]
    candidates["trade_id"] = (
        candidates["session"].astype(str)
        + "|"
        + candidates["entry_minute"].astype(str)
        + "|"
        + candidates["contract_id"].astype(str)
    )
    return state.reset_index(drop=True), candidates.reset_index(drop=True)


def ladder_state(quotes: pd.DataFrame, session: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Declared near-ATM research table and its affordable OTM action subset.

    This bounded table keeps the model-free label/atlas artifact compact.  It is
    not the policy observation.  The simulator and future policy tensorization
    must use :func:`full_ladder_state` so that deep strikes remain visible as
    contemporaneous context even though the declared entry action stays inside
    the near-ATM OTM band.
    """

    return _ladder_state(quotes, session, whole_live_chain=False)


def full_ladder_state(
    quotes: pd.DataFrame, session: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every live two-sided contract plus the declared affordable action subset."""

    return _ladder_state(quotes, session, whole_live_chain=True)


def _surface_slope(frame: pd.DataFrame) -> float:
    clean = frame[["moneyness_itm_points", "self_iv"]].dropna()
    if len(clean) < 2 or clean["moneyness_itm_points"].nunique() < 2:
        return float("nan")
    return float(np.polyfit(clean["moneyness_itm_points"], clean["self_iv"], 1)[0])


def surface_summaries(ladder: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session, minute), group in ladder.groupby(["session", "minute"], sort=False):
        calls = group[group["is_call"]]
        puts = group[~group["is_call"]]
        size_denom = float(group["bid_size"].sum() + group["ask_size"].sum())
        rows.append(
            {
                "session": session,
                "minute": minute,
                "live_ladder_contracts": len(group),
                "live_calls": len(calls),
                "live_puts": len(puts),
                "eligible_entry_contracts": int(group["entry_eligible"].sum()),
                "median_spread_usd": float(group["spread_usd"].median()),
                "median_mid_usd": float(group["mid"].median() * CONTRACT_MULTIPLIER),
                "displayed_size_imbalance": (
                    float((group["bid_size"].sum() - group["ask_size"].sum()) / size_denom)
                    if size_denom > 0
                    else 0.0
                ),
                "chain_volume_observed": float(group["volume"].fillna(0.0).sum()),
                "chain_volume_coverage": float(group["volume"].notna().mean()),
                "call_iv_median": float(calls["self_iv"].median()),
                "put_iv_median": float(puts["self_iv"].median()),
                "call_iv_slope": _surface_slope(calls),
                "put_iv_slope": _surface_slope(puts),
            }
        )
    result = pd.DataFrame(rows)
    result["call_minus_put_iv"] = result["call_iv_median"] - result["put_iv_median"]
    return result


def _wide_quote_matrix(
    quotes: pd.DataFrame, contracts: list[str], column: str, valid: pd.Series | None = None
) -> np.ndarray:
    values = quotes[column].where(valid) if valid is not None else quotes[column]
    wide = quotes.assign(_value=values).pivot_table(
        index="minute", columns="contract_id", values="_value", aggfunc="last"
    )
    return wide.reindex(index=QUOTE_MINUTES, columns=contracts).to_numpy(float)


def _first_hit(path: np.ndarray, threshold: float, valid: np.ndarray) -> np.ndarray:
    hit = (path >= threshold) & valid & np.isfinite(path)
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1) + 1
    return np.where(any_hit, first.astype(float), np.nan)


def _nan_extreme(path: np.ndarray, valid: np.ndarray, kind: str) -> np.ndarray:
    usable = valid & np.isfinite(path)
    has_value = usable.any(axis=1)
    if kind == "max":
        value = np.max(np.where(usable, path, -np.inf), axis=1)
        return np.where(has_value, value, np.nan)
    if kind == "min":
        value = np.min(np.where(usable, path, np.inf), axis=1)
        return np.where(has_value, value, np.nan)
    raise ValueError(kind)


def _first_touch_order(
    returns: np.ndarray,
    valid: np.ndarray,
    *,
    gain: float,
    loss: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label gain-before-loss without inventing a result across missing quotes.

    A label is one when the first observed threshold crossing is the gain, zero
    when loss arrives first or the complete horizon touches neither, and NaN
    only when the path becomes unobservable before either event. A same-minute
    double touch is unresolvable at minute cadence and is deliberately unknown.
    """

    labels = np.full(len(returns), np.nan, dtype=float)
    gain_minutes = np.full(len(returns), np.nan, dtype=float)
    loss_minutes = np.full(len(returns), np.nan, dtype=float)
    for row, (path, usable) in enumerate(zip(returns, valid, strict=True)):
        outcome_known = False
        for minute, (value, is_valid) in enumerate(zip(path, usable, strict=True), 1):
            if not is_valid or not np.isfinite(value):
                break
            gain_hit, loss_hit = value >= gain, value <= loss
            if gain_hit:
                gain_minutes[row] = float(minute)
            if loss_hit:
                loss_minutes[row] = float(minute)
            if gain_hit and loss_hit:
                outcome_known = True
                break
            if gain_hit:
                labels[row] = 1.0
                outcome_known = True
                break
            if loss_hit:
                labels[row] = 0.0
                outcome_known = True
                break
        else:
            labels[row] = 0.0
            outcome_known = True
        if outcome_known and np.isfinite(gain_minutes[row]) and np.isfinite(loss_minutes[row]):
            labels[row] = np.nan
    return labels, gain_minutes, loss_minutes


def attach_candidate_outcomes(
    candidates: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    settlement_spx: float | None = None,
) -> pd.DataFrame:
    """Fixed forward labels; no row is removed for its later quote path."""

    result = candidates.copy()
    if result.empty:
        return result
    if settlement_spx is not None and (
        not np.isfinite(settlement_spx) or settlement_spx <= 0.0
    ):
        raise DatasetError("validated settlement SPX must be finite and positive")
    contracts = sorted(quotes["contract_id"].unique())
    contract_to_column = {contract: i for i, contract in enumerate(contracts)}
    minute_to_row = {minute: i for i, minute in enumerate(QUOTE_MINUTES)}
    columns = result["contract_id"].map(contract_to_column).to_numpy(int)
    entries = result["entry_minute"].map(minute_to_row).to_numpy(int)

    exit_valid = live_two_sided(quotes) & quotes["bid_size"].ge(1.0)
    bid = _wide_quote_matrix(quotes, contracts, "bid", exit_valid)
    mid = _wide_quote_matrix(quotes, contracts, "mid", exit_valid)
    spot = (
        quotes.groupby("minute", sort=False)["underlying_price"]
        .median()
        .reindex(QUOTE_MINUTES)
        .to_numpy(float)
    )
    if not np.isfinite(spot).all():
        raise DatasetError(f"{result.iloc[0]['session']}: missing underlying snapshot in full clock")

    # First executable bid on or after each minute for every contract. This is
    # an execution rule, not a best-price oracle.
    n_minutes, n_contracts = bid.shape
    sentinel = n_minutes
    next_live = np.full((n_minutes + 1, n_contracts), sentinel, dtype=np.int16)
    for i in range(n_minutes - 1, -1, -1):
        next_live[i] = np.where(np.isfinite(bid[i]), i, next_live[i + 1])

    steps = np.arange(1, max(HORIZONS) + 1, dtype=int)
    raw_path_index = entries[:, None] + steps[None, :]
    path_valid = raw_path_index < n_minutes
    path_index = np.minimum(raw_path_index, n_minutes - 1)
    spot_path = spot[path_index]
    strike = result["strike"].to_numpy(float)[:, None]
    is_call = result["is_call"].to_numpy(bool)[:, None]
    entry_spot = result["underlying_price"].to_numpy(float)[:, None]
    itm_path = np.where(is_call, spot_path - strike, strike - spot_path)
    directional_path = np.where(is_call, spot_path - entry_spot, entry_spot - spot_path)
    bid_path = bid[path_index, columns[:, None]]
    option_net_path = (
        (bid_path * CONTRACT_MULTIPLIER)
        - result["entry_ask_usd"].to_numpy(float)[:, None]
        - FEES_PER_ROUND_TRIP_USD
    )

    # The first-touch member is defined on quote mids, while its execution
    # accounting remains separately ask-in/bid-out. It never removes a
    # candidate for a missing future row: that target becomes unknown instead.
    entry_mid = result["entry_mid_usd"].to_numpy(float) / CONTRACT_MULTIPLIER
    option_return_path = mid[raw_path_index.clip(max=n_minutes - 1), columns[:, None]]
    option_return_path = option_return_path / entry_mid[:, None] - 1.0
    first_touch_valid = path_valid & np.isfinite(option_return_path)
    for gain in FIRST_TOUCH_GAINS:
        label, gain_minute, loss_minute = _first_touch_order(
            option_return_path[:, :60],
            first_touch_valid[:, :60],
            gain=gain,
            loss=FIRST_TOUCH_LOSS,
        )
        gain_name = f"{int(gain * 100)}pct"
        result[f"first_touch_{gain_name}_before_loss_30pct_60m"] = label
        result[f"first_touch_{gain_name}_minute_60m"] = gain_minute
        result[f"first_touch_loss_30pct_minute_{gain_name}_60m"] = loss_minute

    for horizon in HORIZONS:
        inside = path_valid & (steps[None, :] <= horizon)
        result[f"maximum_itm_depth_{horizon}m"] = _nan_extreme(itm_path, inside, "max")
        result[f"underlying_mfe_{horizon}m_points"] = _nan_extreme(
            directional_path, inside, "max"
        )
        result[f"underlying_mae_{horizon}m_points"] = _nan_extreme(
            directional_path, inside, "min"
        )
        result[f"option_mfe_{horizon}m_usd"] = _nan_extreme(option_net_path, inside, "max")
        result[f"option_mae_{horizon}m_usd"] = _nan_extreme(option_net_path, inside, "min")
        for depth in ITM_DEPTHS:
            name = "cross" if depth == 0 else f"{depth}_itm"
            first = _first_hit(itm_path[:, :horizon], float(depth), inside[:, :horizon])
            result[f"time_to_{name}_{horizon}m"] = first
            result[f"reached_{name}_{horizon}m"] = np.isfinite(first)

        raw_target = entries + horizon
        target = np.minimum(raw_target, n_minutes - 1)
        chosen = next_live[target, columns].astype(int)
        executable = chosen < n_minutes
        cash_settled = ~executable & (settlement_spx is not None)
        safe = np.minimum(chosen, n_minutes - 1)
        exit_bid = np.where(executable, bid[safe, columns], np.nan)
        exit_mid = np.where(executable, mid[safe, columns], np.nan)
        terminal_intrinsic = np.where(
            result["is_call"].to_numpy(bool),
            np.maximum(0.0, float(settlement_spx or 0.0) - result["strike"].to_numpy(float)),
            np.maximum(0.0, result["strike"].to_numpy(float) - float(settlement_spx or 0.0)),
        )
        exit_value = np.where(cash_settled, terminal_intrinsic, exit_bid)
        exit_mid_value = np.where(cash_settled, terminal_intrinsic, exit_mid)
        actual_row = np.where(cash_settled, n_minutes - 1, chosen)
        delay = np.where(executable | cash_settled, actual_row - target, np.nan)
        truncated = raw_target >= n_minutes
        result[f"clock_target_minute_{horizon}m"] = [minute_label(571 + i) for i in target]
        result[f"clock_truncated_by_close_{horizon}m"] = truncated
        result[f"clock_exit_minute_{horizon}m"] = np.where(
            executable,
            np.asarray([minute_label(571 + i) for i in safe], dtype=object),
            np.where(cash_settled, LAST_QUOTE_MINUTE, None),
        )
        result[f"clock_exit_delay_{horizon}m"] = delay
        result[f"clock_exit_bid_{horizon}m"] = exit_bid
        result[f"clock_exit_mid_{horizon}m"] = exit_mid
        result[f"clock_exit_value_{horizon}m"] = exit_value
        result[f"clock_exit_mid_value_{horizon}m"] = exit_mid_value
        result[f"clock_terminal_intrinsic_{horizon}m"] = np.where(
            cash_settled, terminal_intrinsic, np.nan
        )
        result[f"clock_settlement_spx_{horizon}m"] = np.where(
            cash_settled, float(settlement_spx or 0.0), np.nan
        )
        result[f"clock_exit_type_{horizon}m"] = np.where(
            executable,
            "executable_bid",
            np.where(cash_settled, "validated_cash_settlement", "blocked"),
        )
        result[f"clock_exit_status_{horizon}m"] = np.where(
            cash_settled,
            "validated_cash_settlement",
            np.where(
                ~executable,
            "blocked_no_executable_bid",
                np.where(delay > 0, "delayed_first_later_bid", "executable_at_target"),
            ),
        )
        result[f"net_bid_{horizon}m_usd"] = (
            exit_value * CONTRACT_MULTIPLIER
            - result["entry_ask_usd"].to_numpy(float)
            - FEES_PER_ROUND_TRIP_USD
        )
        result[f"net_mid_{horizon}m_usd"] = (
            exit_mid_value * CONTRACT_MULTIPLIER
            - result["entry_mid_usd"].to_numpy(float)
            - FEES_PER_ROUND_TRIP_USD
        )
        zero_recovery_value = np.where(cash_settled, 0.0, exit_value)
        zero_recovery_mid_value = np.where(cash_settled, 0.0, exit_mid_value)
        result[f"net_bid_zero_recovery_{horizon}m_usd"] = (
            zero_recovery_value * CONTRACT_MULTIPLIER
            - result["entry_ask_usd"].to_numpy(float)
            - FEES_PER_ROUND_TRIP_USD
        )
        result[f"net_mid_zero_recovery_{horizon}m_usd"] = (
            zero_recovery_mid_value * CONTRACT_MULTIPLIER
            - result["entry_mid_usd"].to_numpy(float)
            - FEES_PER_ROUND_TRIP_USD
        )
        result[f"realised_hold_{horizon}m"] = np.where(
            executable | cash_settled, actual_row - entries, np.nan
        )
    return result


def underlying_atlas(quotes: pd.DataFrame, session: str) -> pd.DataFrame:
    """Outcome-balanced time surface for every declared entry minute."""

    spot = (
        quotes.groupby("minute", sort=False)["underlying_price"]
        .median()
        .reindex(QUOTE_MINUTES)
        .to_numpy(float)
    )
    minute_to_row = {minute: i for i, minute in enumerate(QUOTE_MINUTES)}
    rows: list[dict[str, Any]] = []
    for minute in ENTRY_MINUTES:
        entry = minute_to_row[minute]
        start_spot = spot[entry]
        row: dict[str, Any] = {
            "session": session,
            "minute": minute,
            "regime": regime_for_minute(minute),
            "is_named_minute": minute in ("10:00", "13:30"),
            "key_band": next(
                (name for first, last, name in KEY_BANDS if first <= minute <= last), None
            ),
            "underlying_price": start_spot,
        }
        for horizon in HORIZONS:
            stop = min(entry + horizon, len(spot) - 1)
            future = spot[entry + 1 : stop + 1]
            up = future - start_spot
            down = start_spot - future
            row[f"up_excursion_{horizon}m_points"] = float(np.max(up)) if len(up) else np.nan
            row[f"down_excursion_{horizon}m_points"] = float(np.max(down)) if len(down) else np.nan
            row[f"final_move_{horizon}m_points"] = float(spot[stop] - start_spot)
            row[f"truncated_by_close_{horizon}m"] = entry + horizon >= len(spot)
            for depth in (10, 20, 30):
                up_hit = np.flatnonzero(up >= depth)
                down_hit = np.flatnonzero(down >= depth)
                row[f"time_to_up_{depth}_{horizon}m"] = (
                    float(up_hit[0] + 1) if up_hit.size else np.nan
                )
                row[f"time_to_down_{depth}_{horizon}m"] = (
                    float(down_hit[0] + 1) if down_hit.size else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_session(
    quote_path: Path,
    es_path: Path,
    *,
    settlement_spx: float | None = None,
) -> dict[str, pd.DataFrame]:
    session = session_from_path(quote_path)
    quotes = prepare_quotes(pd.read_parquet(quote_path, columns=list(QUOTE_COLUMNS)), session)
    candles = prepare_es(pd.read_parquet(es_path), session)
    minute_state = candle_states(candles)
    ladder, candidates = ladder_state(quotes, session)
    surface = surface_summaries(ladder)
    minute_state = minute_state.merge(surface, on=["session", "minute"], how="left")
    spot = quotes.groupby("minute")["underlying_price"].median().rename("spx_snapshot")
    minute_state = minute_state.merge(spot, left_on="minute", right_index=True, how="left")
    candidates = attach_candidate_outcomes(
        candidates, quotes, settlement_spx=settlement_spx
    )
    atlas = underlying_atlas(quotes, session)
    return {
        "candles": candles,
        "minutes": minute_state,
        "ladder": ladder,
        "candidates": candidates,
        "atlas": atlas,
    }


def _prevalence(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = frame[column]
    return {
        "n": int(values.notna().sum()),
        "positive": int(values.fillna(False).sum()),
        "share": float(values.fillna(False).mean()),
    }


def load_validated_settlements(receipt_path: Path) -> dict[str, float]:
    """Verify the immutable audit receipt before accepting terminal values."""

    receipt = json.loads(receipt_path.read_text())
    expected = receipt.get("receipt_sha256")
    semantic = dict(receipt)
    semantic.pop("receipt_sha256", None)
    actual = hashlib.sha256(canonical_json(semantic)).hexdigest()
    if expected != actual:
        raise DatasetError("terminal settlement receipt hash mismatch")
    if receipt.get("validation", {}).get("status") != "VALIDATED_FOR_TERMINAL_ACCOUNTING":
        raise DatasetError("terminal settlement receipt is not validated")
    artifact = receipt.get("artifacts", {}).get("settlements", {})
    path = Path(str(artifact.get("path", "")))
    if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
        raise DatasetError("terminal settlement artifact is missing or hash-mismatched")
    frame = pd.read_csv(path)
    needed = {"session", "settlement_spx", "exact_aligned_identity"}
    if not needed <= set(frame.columns):
        raise DatasetError("terminal settlement artifact schema drift")
    if not frame["exact_aligned_identity"].astype(bool).all():
        raise DatasetError("terminal settlement artifact contains an unvalidated session")
    if frame["session"].astype(str).duplicated().any():
        raise DatasetError("terminal settlement artifact contains duplicate sessions")
    values = pd.to_numeric(frame["settlement_spx"], errors="coerce")
    if not np.isfinite(values).all() or not values.gt(0.0).all():
        raise DatasetError("terminal settlement artifact contains an invalid SPX value")
    return dict(zip(frame["session"].astype(str), values.astype(float), strict=True))


def run(
    quote_root: Path,
    es_root: Path,
    coverage_csv: Path,
    declaration_path: Path,
    out_dir: Path,
    evidence_dir: Path,
    settlement_receipt: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if out_dir.exists():
        raise DatasetError(f"refusing to overwrite derived dataset: {out_dir}")
    if evidence_dir.exists():
        raise DatasetError(f"refusing to overwrite evidence: {evidence_dir}")
    coverage = pd.read_csv(coverage_csv)
    sessions = coverage.loc[coverage["included_for_episode_build"], "session"].astype(str).tolist()
    if limit is not None:
        sessions = sessions[:limit]
    settlements = (
        load_validated_settlements(settlement_receipt)
        if settlement_receipt is not None
        else {}
    )
    if settlements and set(sessions) - set(settlements):
        missing = sorted(set(sessions) - set(settlements))
        raise DatasetError(f"validated settlement missing sessions: {missing}")
    parts: dict[str, list[pd.DataFrame]] = {
        "candles": [],
        "minutes": [],
        "ladder": [],
        "candidates": [],
        "atlas": [],
    }
    for i, session in enumerate(sessions, 1):
        quote_path = quote_root / f"databento_spxw_0dte_{session}.parquet"
        es_path = es_root / f"{session}.es_c_0.ohlcv-1m.parquet"
        built = build_session(
            quote_path,
            es_path,
            settlement_spx=settlements.get(session),
        )
        for name, frame in built.items():
            parts[name].append(frame)
        if i % 10 == 0 or i == len(sessions):
            print(
                f"sessions {i}/{len(sessions)}; candidates "
                f"{sum(len(x) for x in parts['candidates']):,}",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Path] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, blocks in parts.items():
        frame = pd.concat(blocks, ignore_index=True)
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        outputs[name] = path
        frames[name] = frame

    candidates = frames["candidates"]
    receipt: dict[str, Any] = {
        "schema_version": (
            "v5.causal-day-dataset.v2" if settlements else "v5.causal-day-dataset.v1"
        ),
        "created_on": "2026-08-14",
        "purpose": "model-free causal episodes, ladder surface, fixed OTM-to-ITM labels and time atlas",
        "declaration_path": str(declaration_path),
        "declaration_sha256": file_sha256(declaration_path),
        "coverage_csv": str(coverage_csv),
        "sessions": {"n": len(sessions), "first": min(sessions), "last": max(sessions)},
        "rows": {name: len(frame) for name, frame in frames.items()},
        "earliest": {
            "minute_state": str(frames["minutes"]["minute"].min()),
            "candidate": str(candidates["entry_minute"].min()),
            "latest_candidate": str(candidates["entry_minute"].max()),
        },
        "entry_population": {
            "unique_trade_ids": int(candidates["trade_id"].nunique()),
            "sessions": int(candidates["session"].nunique()),
            "mean_entry_ask_usd": float(candidates["entry_ask_usd"].mean()),
            "maximum_entry_ask_usd": float(candidates["entry_ask_usd"].max()),
            "moneyness_min": float(candidates["moneyness_itm_points"].min()),
            "moneyness_max": float(candidates["moneyness_itm_points"].max()),
        },
        "target_prevalence": {
            f"reached_{name}_{horizon}m": _prevalence(
                candidates, f"reached_{name}_{horizon}m"
            )
            for horizon in HORIZONS
            for depth, name in ((0, "cross"), (10, "10_itm"), (20, "20_itm"), (30, "30_itm"))
        },
        "clock_exit_status": {
            f"{horizon}m": {
                str(key): int(value)
                for key, value in candidates[f"clock_exit_status_{horizon}m"].value_counts().items()
            }
            for horizon in HORIZONS
        },
        "key_time_rows": {
            "all_atlas_rows": len(frames["atlas"]),
            "diagnostic_band_rows": int(frames["atlas"]["key_band"].notna().sum()),
            "named_minute_rows": int(frames["atlas"]["is_named_minute"].sum()),
        },
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in outputs.items()
        },
        "terminal_settlement": {
            "status": (
                "VALIDATED_FOR_TERMINAL_ACCOUNTING" if settlements else "NOT_VALIDATED"
            ),
            "receipt": str(settlement_receipt) if settlement_receipt is not None else None,
            "effect": (
                "terminal no-bid rows use separately labelled intrinsic cash accounting; "
                "clock_exit_bid remains missing and zero-recovery sensitivity is retained"
                if settlements
                else "candidate clock rows without an executable later bid stay blocked; none are dropped"
            ),
        },
        "model_fit": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    (evidence_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(evidence_dir / "receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    parser.add_argument(
        "--es",
        type=Path,
        default=Path(
            "/Users/och/.autoresearch-trading/es_1m_2016-08-01_2026-07-31/"
            "raw/databento/glbx_es_ohlcv_1m"
        ),
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/"
            "session_coverage.csv"
        ),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path("v5/work/entry-exit-attribution/DECLARATION.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--settlement-receipt", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(
        args.quotes,
        args.es,
        args.coverage_csv,
        args.declaration,
        args.out_dir,
        args.evidence_dir,
        args.settlement_receipt,
        args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
