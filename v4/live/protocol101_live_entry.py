"""Live Protocol101 entry-row construction and paper-order intent helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, NeuralDatasetConfig, OPTION_FEATURE_NAMES
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
    LIVE_FEATURE_CONTRACTS,
    candidate_filter_diagnostics,
    candidate_is_tradable_values,
    candidate_ladder_slots,
    feature_contract_metadata,
    feature_contract_requires_model_scoring_greeks,
    feature_contract_version,
    missing_candidate_slot_diagnostics,
    option_feature_values,
    quote_source_metadata,
    round_to_strike_step,
    strike_ladder_context,
)
from v4.live.ibkr_paper_guard import PaperOrderIntent
from v4.live.paper_trade_log import stable_json_hash
from v4.model.hypothesis_protocol import SurfaceDecision, SurfaceVariant, surface_decision_from_row, time_bucket


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
OPTION_INDEX = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
MARKET_WIDTH = len(MARKET_FEATURE_NAMES)
STRUCTURE_WIDTH = 28
LIVE_DATASET_CONFIG = NeuralDatasetConfig()


@dataclass
class LiveIndexState:
    """Causal in-memory index context for the current paper session."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    # A full SPX session can exceed 50k updates. Retaining less silently drops
    # the opening minute and corrupts OMAR/market-structure features late in a
    # session and when an immutable full-day capture is replayed after close.
    max_rows: int = 250_000
    prior_session_close: float | None = None

    def set_previous_session_close(self, value: float) -> None:
        close = float(value)
        if math.isfinite(close) and close > 0:
            self.prior_session_close = close

    def add(self, *, timestamp: datetime | pd.Timestamp, spx: float, vix: float) -> None:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self.rows.append({"timestamp": ts, "spx": float(spx), "vix": float(vix)})
        self.rows = self.rows[-max(int(self.max_rows), 1) :]

    def session_context_summary(self, decision_time: datetime | pd.Timestamp) -> dict[str, Any]:
        ts = _utc_timestamp(decision_time)
        context_ts = self._completed_context_time(ts)
        frame = self._session_frame(context_ts)
        minute_frame = self._session_minute_frame(context_ts)
        expected_first = self._session_open_minute(ts)
        if frame.empty:
            return {
                "row_count": 0,
                "minute_row_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "span_minutes": 0.0,
                "expected_first_timestamp": expected_first.isoformat(),
                "opening_context_ready": False,
                "missing_opening_minutes": None,
            }
        span_frame = minute_frame if not minute_frame.empty else frame
        first = pd.Timestamp(span_frame["timestamp"].iloc[0])
        last = pd.Timestamp(span_frame["timestamp"].iloc[-1])
        opening_ready = bool(first <= expected_first)
        missing_opening_minutes = (
            max(int((first - expected_first).total_seconds() // 60), 0)
            if not opening_ready
            else 0
        )
        return {
            "row_count": int(len(frame)),
            "minute_row_count": int(len(minute_frame)),
            "first_timestamp": first.isoformat(),
            "last_timestamp": last.isoformat(),
            "span_minutes": float((last - first).total_seconds() / 60.0),
            "expected_first_timestamp": expected_first.isoformat(),
            "opening_context_ready": opening_ready,
            "missing_opening_minutes": int(missing_opening_minutes),
        }

    def opening_context_ready(self, decision_time: datetime | pd.Timestamp) -> bool:
        summary = self.session_context_summary(decision_time)
        return bool(summary.get("opening_context_ready"))

    def frame(self, decision_time: datetime | pd.Timestamp) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["timestamp", "spx", "vix"])
        ts = _utc_timestamp(decision_time)
        frame = pd.DataFrame(self.rows).sort_values("timestamp")
        return frame[frame["timestamp"] <= ts].reset_index(drop=True)

    def market_window(self, decision_time: datetime | pd.Timestamp, *, minutes: int = 30) -> np.ndarray:
        ts = _utc_timestamp(decision_time)
        context_ts = self._completed_context_time(ts)
        session_minutes = self._session_minute_frame(context_ts)
        if session_minutes.empty:
            return np.zeros((minutes, MARKET_WIDTH), dtype=np.float32)
        end = context_ts.floor("min")
        minute_index = pd.date_range(end - pd.Timedelta(minutes=minutes - 1), end, freq="min", tz="UTC")
        working = session_minutes.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        regular = working.reindex(minute_index, method="ffill")
        regular[["spx", "vix"]] = regular[["spx", "vix"]].ffill()
        rows = []
        session_frame = session_minutes
        for minute_ts, row in regular.iterrows():
            spx_value = float(row["spx"])
            vix_value = float(row["vix"])
            if not math.isfinite(spx_value) or not math.isfinite(vix_value):
                rows.append(np.full(MARKET_WIDTH, np.nan, dtype=np.float32))
                continue
            rows.append(_market_features_from_live(session_frame, minute_ts, spx_value, vix_value))
        return np.vstack(rows).astype(np.float32)

    def structure_features(self, decision_time: datetime | pd.Timestamp) -> np.ndarray:
        ts = _utc_timestamp(decision_time)
        context_ts = self._completed_context_time(ts)
        frame = self._session_minute_frame(context_ts)
        if frame.empty:
            return np.zeros(STRUCTURE_WIDTH, dtype=np.float32)
        if not self.opening_context_ready(ts):
            return np.zeros(STRUCTURE_WIDTH, dtype=np.float32)
        close = frame["spx"].astype(float).to_numpy()
        high_arr = frame.get("spx_high", frame["spx"]).astype(float).to_numpy()
        low_arr = frame.get("spx_low", frame["spx"]).astype(float).to_numpy()
        vix = frame["vix"].astype(float).to_numpy()
        current = float(close[-1])
        current_vix = float(vix[-1]) if len(vix) else 0.0
        count = np.arange(1, len(close) + 1, dtype=float)
        vwap = np.cumsum(close) / count
        std = np.asarray([np.nanstd(close[: idx + 1]) for idx in range(len(close))], dtype=float)
        std = np.maximum(std, 1e-6)
        sigma = float((current - vwap[-1]) / std[-1]) if len(vwap) else 0.0
        omar_high = float(high_arr[0])
        omar_low = float(low_arr[0])
        omar_range = max(omar_high - omar_low, 0.01)
        omar_mid = (omar_high + omar_low) / 2.0
        nearest_omar = min(abs(current - omar_high), abs(current - omar_low), abs(current - omar_mid))
        minute = min(_minute_of_session(ts), len(close) - 1)
        first15_available = len(close) >= 15
        if first15_available:
            first15_high = float(np.nanmax(high_arr[:15]))
            first15_low = float(np.nanmin(low_arr[:15]))
            first15_close = float(close[min(14, len(close) - 1)])
            first15_range = max(first15_high - first15_low, 0.01)
            first15_close_pos = (first15_close - first15_low) / first15_range
            first15_accept = np.clip((current - ((first15_high + first15_low) / 2.0)) / (first15_range / 2.0), -1.0, 1.0)
            inside_first15 = float(first15_low <= current <= first15_high)
            first15_range_pct = first15_range / max(abs(current), 1.0)
        else:
            first15_close_pos = 0.5
            first15_accept = 0.0
            inside_first15 = 0.0
            first15_range_pct = 0.0
        if minute > 0:
            lo = max(minute - 10, 0)
            last10_high = float(np.nanmax(high_arr[lo:minute]))
            last10_low = float(np.nanmin(low_arr[lo:minute]))
            last10_range = last10_high - last10_low
            last10_range_over_omar = last10_range / omar_range
            last10_break_state = 1.0 if current > last10_high else -1.0 if current < last10_low else 0.0
        else:
            last10_range_over_omar = 0.0
            last10_break_state = 0.0
        atr15_pct = float(
            np.nanmean(high_arr[-15:] - low_arr[-15:]) / max(abs(current), 1.0)
        ) if len(close) else 0.0
        slope = 0.0
        if len(vwap) > 5:
            slope = float((vwap[-1] - vwap[-6]) / max(abs(current), 1.0))
        prev_close = self.previous_session_close(ts)
        open_price = float(close[0]) if len(close) else current
        opening_gap_pct = (open_price - prev_close) / prev_close if math.isfinite(prev_close) and prev_close > 0 else 0.0
        bucket = time_bucket(ts.to_pydatetime())
        buckets = [
            float(bucket == "first_30"),
            float(bucket == "post_open_morning"),
            float(bucket == "midday"),
            float(bucket == "late_afternoon"),
        ]
        out = np.asarray(
            [
                current,
                current_vix,
                sigma,
                abs(sigma),
                float(vwap[-1]),
                (current - float(vwap[-1])) / max(abs(current), 1.0),
                slope,
                omar_high,
                omar_low,
                omar_mid,
                omar_range,
                omar_range / max(abs(current), 1.0),
                (current - omar_mid) / omar_range,
                nearest_omar / omar_range,
                float(first15_available),
                first15_range_pct,
                first15_close_pos,
                float(first15_accept),
                inside_first15,
                opening_gap_pct,
                last10_range_over_omar,
                last10_break_state,
                atr15_pct,
                minute / 390.0,
                *buckets,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0)

    def previous_session_close(self, decision_time: datetime | pd.Timestamp) -> float:
        ts = _utc_timestamp(decision_time)
        if not self.rows:
            return math.nan
        local_day = ts.tz_convert(NY).date().isoformat()
        frame = pd.DataFrame(self.rows).sort_values("timestamp")
        local = frame["timestamp"].dt.tz_convert(NY)
        prior = frame[(local.dt.date.astype(str) < local_day) & (frame["timestamp"] <= ts)]
        if not prior.empty:
            return float(prior.iloc[-1]["spx"])
        if self.prior_session_close is not None and math.isfinite(float(self.prior_session_close)):
            return float(self.prior_session_close)
        return math.nan

    def _session_frame(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["timestamp", "spx", "vix"])
        local_day = decision_time.tz_convert(NY).date().isoformat()
        frame = pd.DataFrame(self.rows).sort_values("timestamp")
        local = frame["timestamp"].dt.tz_convert(NY)
        same_day = local.dt.date.astype(str) == local_day
        open_minute = self._session_open_minute(decision_time)
        return frame[
            same_day
            & (frame["timestamp"] >= open_minute)
            & (frame["timestamp"] <= decision_time)
        ].reset_index(drop=True)

    def _session_minute_frame(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        """Return one row per minute, matching the historical training cadence."""

        raw = self._session_frame(decision_time)
        if raw.empty:
            return raw
        local_day = decision_time.tz_convert(NY).date().isoformat()
        first_minute = pd.Timestamp(f"{local_day} 09:30", tz=NY).tz_convert(UTC)
        last_raw_minute = pd.Timestamp(raw["timestamp"].iloc[-1]).floor("min")
        end_minute = min(decision_time.floor("min"), last_raw_minute)
        if end_minute < first_minute:
            end_minute = first_minute
        minute_index = pd.date_range(first_minute, end_minute, freq="min", tz="UTC")
        working = raw.copy()
        working["_minute"] = working["timestamp"].dt.floor("min")
        working = (
            working.sort_values("timestamp")
            .groupby("_minute", as_index=True)
            .agg(
                spx_open=("spx", "first"),
                spx_high=("spx", "max"),
                spx_low=("spx", "min"),
                spx=("spx", "last"),
                vix=("vix", "last"),
            )
            .sort_index()
        )
        regular = working.reindex(minute_index, method="ffill")
        regular[["spx", "vix"]] = regular[["spx", "vix"]].ffill()
        for col in ("spx_open", "spx_high", "spx_low"):
            regular[col] = regular[col].where(regular[col].notna(), regular["spx"])
        regular = regular[["spx", "vix", "spx_open", "spx_high", "spx_low"]].dropna(subset=["spx", "vix"])
        regular.index.name = "timestamp"
        regular = regular.reset_index()
        return regular[["timestamp", "spx", "vix", "spx_open", "spx_high", "spx_low"]]

    def _session_open_minute(self, decision_time: pd.Timestamp) -> pd.Timestamp:
        local_day = decision_time.tz_convert(NY).date().isoformat()
        return pd.Timestamp(f"{local_day} 09:30", tz=NY).tz_convert(UTC)

    @staticmethod
    def _completed_context_time(decision_time: pd.Timestamp) -> pd.Timestamp:
        return decision_time.floor("min") - pd.Timedelta(microseconds=1)


class LiveMarketStructureAdapter:
    def __init__(self, state: LiveIndexState) -> None:
        self.state = state
        self.cache_id = "ibkr_live_in_memory"

    def features_for(self, decision_time: datetime) -> np.ndarray:
        return self.state.structure_features(decision_time)


def build_live_surface_row(
    *,
    decision_time: datetime | pd.Timestamp,
    spx: float,
    vix: float,
    option_quotes: Iterable[dict[str, Any]],
    index_state: LiveIndexState,
    policy_count: int = 3,
    feature_contract_name: str | None = FEATURE_CONTRACT_VERSION,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Create a normalized-like decision row from live option quotes."""

    ts = _utc_timestamp(decision_time)
    contract_version = feature_contract_version(feature_contract_name)
    index_state.add(timestamp=ts, spx=float(spx), vix=float(vix))
    config = LIVE_DATASET_CONFIG
    raw_quotes = list(option_quotes)
    quotes = [quote for quote in raw_quotes if _valid_quote(quote)]
    rights = ("C", "P")
    atm_strike = _round_to_5(float(spx), step=config.strike_step)
    ladder_context = strike_ladder_context(
        spx_for_ladder=float(spx),
        strike_step=config.strike_step,
    )
    strike_offsets = np.arange(
        -config.ladder_dollars,
        config.ladder_dollars + config.strike_step,
        config.strike_step,
        dtype=np.float32,
    )
    slots = candidate_ladder_slots(
        spx_for_ladder=float(spx),
        ladder_dollars=config.ladder_dollars,
        strike_step=config.strike_step,
        rights=rights,
    )
    strikes = [float(atm_strike + int(offset)) for offset in strike_offsets]
    option_ladder = np.zeros((len(strikes), len(rights), len(OPTION_FEATURE_NAMES)), dtype=np.float32)
    candidate_mask = np.zeros((len(strikes), len(rights)), dtype=bool)
    contract_ids = np.empty((len(strikes), len(rights)), dtype=object)
    lookup: dict[str, dict[str, Any]] = {}
    candidate_filter_trace: list[dict[str, Any]] = []
    raw_by_key = {
        (float(quote["strike"]), str(quote["right"]).upper()): quote
        for quote in raw_quotes
        if math.isfinite(_float(quote.get("strike"), math.nan))
        and str(quote.get("right") or "").upper() in {"C", "P"}
    }
    by_key = {(float(quote["strike"]), str(quote["right"]).upper()): quote for quote in quotes}
    for slot in slots:
        strike_idx = int(slot["strike_idx"])
        right_idx = int(slot["right_idx"])
        strike = float(slot["strike"])
        right = str(slot["right"])
        quote = by_key.get((strike, right))
        raw_quote = raw_by_key.get((strike, right))
        quote_for_identity = quote or raw_quote or {}
        contract_id = str(quote_for_identity.get("contract_id") or _contract_id(strike=strike, expiry=str(quote_for_identity.get("expiry") or ts.strftime("%Y%m%d")), right=right))
        contract_ids[strike_idx, right_idx] = contract_id
        if quote is None:
            if raw_quote is not None:
                source_metadata = quote_source_metadata(
                    raw_quote,
                    decision_time=ts,
                    feature_contract_name=contract_version,
                )
                prefilter = _quote_prefilter_diagnostics(raw_quote)
                pre_filter_candidate = True
                filter_reasons = list(prefilter.get("reasons") or [])
                freshness_pass = prefilter.get("freshness_pass")
                bid = _float_or_none(raw_quote.get("bid"))
                ask = _float_or_none(raw_quote.get("ask"))
                mid = _float_or_none(raw_quote.get("mid"))
                bid_size = _float_or_none(raw_quote.get("bid_size"))
                ask_size = _float_or_none(raw_quote.get("ask_size"))
                quote_age_source = source_metadata.get("quote_age_source")
            else:
                source_metadata = {
                    "feature_contract_version": contract_version,
                    "source_quote_time": None,
                    "source_quote_ts": None,
                    "source_context_time": None,
                    "source_context_ts": None,
                    "quote_age_ms": None,
                    "quote_age_source": "no_ibkr_quote_at_decision",
                    "raw_quote_timestamp_utc": None,
                    "received_timestamp_utc": None,
                    "decision_timestamp_utc": ts.isoformat(),
                }
                prefilter = missing_candidate_slot_diagnostics("no_ibkr_quote_at_decision")
                pre_filter_candidate = False
                filter_reasons = list(prefilter["reasons"])
                freshness_pass = None
                bid = ask = mid = bid_size = ask_size = None
                quote_age_source = "no_ibkr_quote_at_decision"
            candidate_filter_trace.append(
                {
                    **source_metadata,
                    **slot,
                    "contract_id": contract_id,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "pre_filter_candidate": pre_filter_candidate,
                    "post_filter_candidate": False,
                    "candidate_filter": prefilter,
                    "filter_reasons": filter_reasons,
                    "tradability_pass": False,
                    "freshness_pass": freshness_pass,
                    "quote_age_source": quote_age_source,
                }
            )
            continue
        row = _option_feature_row(
            quote=quote,
            decision_time=ts,
            spx=float(spx),
            strike=strike,
            right=right,
            atm_strike=atm_strike,
        )
        feature_values = {name: float(row[idx]) for idx, name in enumerate(OPTION_FEATURE_NAMES)}
        option_ladder[strike_idx, right_idx] = row
        post_filter_candidate = _live_candidate_is_tradable(
            row,
            feature_contract_name=contract_version,
        )
        candidate_mask[strike_idx, right_idx] = post_filter_candidate
        filter_values = {
            **feature_values,
            "quote_age_ms": quote.get("quote_age_ms"),
        }
        filter_diagnostics = candidate_filter_diagnostics(
            filter_values,
            LIVE_FEATURE_CONTRACTS.get(contract_version, DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT),
            require_greeks=feature_contract_requires_model_scoring_greeks(contract_version),
            enforce_freshness=True,
        )
        source_metadata = quote_source_metadata(
            quote,
            decision_time=ts,
            feature_contract_name=contract_version,
        )
        lookup[contract_id] = {
            **quote,
            **source_metadata,
            "contract_id": contract_id,
            "strike": strike,
            "right": right,
            "offset": float(strike - atm_strike),
            "strike_idx": int(strike_idx),
            "right_idx": int(right_idx),
            **ladder_context,
            "trading_class": str(quote.get("trading_class") or "SPXW"),
            "symbol": str(quote.get("symbol") or "SPX"),
            "expiry": str(quote.get("expiry") or ts.strftime("%Y%m%d")),
            "ask": float(quote["ask"]),
            "bid": float(quote["bid"]),
            "mid": feature_values["mid"],
            "spread": feature_values["spread"],
            "spread_frac": feature_values["spread_frac"],
            "option_ohlcv_volume": feature_values["option_ohlcv_volume"],
            "stat_open_interest": feature_values["stat_open_interest"],
            "feature_contract_version": contract_version,
            "pre_filter_candidate": True,
            "post_filter_candidate": bool(post_filter_candidate),
            "candidate_filter": filter_diagnostics,
            "filter_reasons": list(filter_diagnostics.get("reasons") or []),
            "tradability_pass": filter_diagnostics.get("tradability_pass"),
            "freshness_pass": filter_diagnostics.get("freshness_pass"),
            "model_scoring_greek_pass": filter_diagnostics.get("greek_pass"),
        }
        candidate_filter_trace.append(
            {
                key: value
                for key, value in lookup[contract_id].items()
                if key != "raw_vendor_fields"
            }
        )
    labels = np.zeros((len(strikes), len(rights), int(policy_count)), dtype=np.float32)
    source_quote_times = [
        pd.Timestamp(item["source_quote_time"])
        for item in lookup.values()
        if item.get("source_quote_time")
    ]
    source_quote_time = max(source_quote_times).to_pydatetime() if source_quote_times else ts.to_pydatetime()
    context_summary = index_state.session_context_summary(ts)
    source_context_time = context_summary.get("last_timestamp") or ts.isoformat()
    for item in lookup.values():
        item["source_context_time"] = source_context_time
        item["source_context_ts"] = source_context_time
    for item in candidate_filter_trace:
        item["source_context_time"] = source_context_time
        item["source_context_ts"] = source_context_time
    ladder_context["source_context_time"] = source_context_time
    ladder_context["source_context_ts"] = source_context_time
    candidate_filter_trace.sort(
        key=lambda item: (
            int(item.get("strike_idx") or 0),
            int(item.get("right_idx") or 0),
            str(item.get("contract_id") or ""),
        )
    )
    max_quote_age_ms = max(
        [float(item["quote_age_ms"]) for item in lookup.values() if item.get("quote_age_ms") is not None],
        default=math.nan,
    )
    context_ready = bool(
        float(context_summary.get("span_minutes") or 0.0) >= max(float(config.market_window_minutes) - 1.0, 0.0)
        and int(context_summary.get("minute_row_count") or 0) >= int(math.ceil(float(config.market_window_minutes)))
        and bool(context_summary.get("opening_context_ready"))
    )
    row = {
        "decision_time": ts.to_pydatetime(),
        "source_quote_time": source_quote_time,
        "source_context_time": source_context_time,
        "max_quote_age_ms": max_quote_age_ms,
        "feature_contract_version": contract_version,
        "feature_contract": feature_contract_metadata(contract_version),
        "position_state": "flat",
        "atm_strike": atm_strike,
        "strike_offsets": strike_offsets,
        "rights": rights,
        "feature_names": tuple(OPTION_FEATURE_NAMES),
        "option_ladder": option_ladder,
        "candidate_mask": candidate_mask,
        "contract_ids": contract_ids,
        "label_names": tuple(f"live_policy_{idx}" for idx in range(int(policy_count))),
        "labels_net_pnl": labels,
        "labels_mid_pnl": labels.copy(),
        "market_feature_names": tuple(MARKET_FEATURE_NAMES),
        "market_window": index_state.market_window(ts),
        "contract_quote_metadata": lookup,
        "candidate_filter_trace": candidate_filter_trace,
        "ladder_context": dict(ladder_context),
        "context_ready": context_ready,
        "context_required_minutes": float(config.market_window_minutes),
        "context_minute_rows": int(context_summary.get("minute_row_count") or 0),
        "context_span_minutes": float(context_summary.get("span_minutes") or 0.0),
        "context_start_timestamp": context_summary.get("first_timestamp"),
        "context_last_timestamp": context_summary.get("last_timestamp"),
    }
    return row, lookup


def scored_token_universe_payload(
    decision: SurfaceDecision,
    surface_scores: np.ndarray,
    lookup: dict[str, dict[str, Any]],
    *,
    include_features: bool = True,
) -> list[dict[str, Any]]:
    """Return every valid token the surface model saw, including scores/hashes."""

    scores = np.asarray(surface_scores, dtype=float)
    flat_score = float(scores[0]) if len(scores) else math.nan
    rows: list[dict[str, Any]] = []
    for token_idx, valid in enumerate(np.asarray(decision.token_mask, dtype=bool)):
        if not bool(valid):
            continue
        contract_id = str(decision.contract_ids[token_idx])
        quote = dict(lookup.get(contract_id) or {})
        action_score = float(scores[token_idx + 1]) if token_idx + 1 < len(scores) else math.nan
        token_features = [float(value) for value in np.asarray(decision.token_features[token_idx], dtype=float)]
        token_by_name = {
            name: token_features[idx]
            for idx, name in enumerate(OPTION_FEATURE_NAMES)
            if idx < len(token_features)
        }
        row = {
            "contract_id": contract_id,
            "token_idx": int(token_idx),
            "right": str(decision.rights[token_idx]),
            "offset_points": float(decision.offsets[token_idx]),
            "surface_action_score": action_score if math.isfinite(action_score) else None,
            "surface_flat_score": flat_score if math.isfinite(flat_score) else None,
            "edge": (action_score - flat_score) if math.isfinite(action_score) and math.isfinite(flat_score) else None,
            "token_feature_hash": stable_json_hash(token_features),
            "bid": quote.get("bid", token_by_name.get("bid")),
            "ask": quote.get("ask", token_by_name.get("ask")),
            "mid": quote.get("mid", token_by_name.get("mid")),
            "spread": quote.get("spread", token_by_name.get("spread")),
            "spread_frac": quote.get("spread_frac", token_by_name.get("spread_frac")),
            "bid_size": quote.get("bid_size", token_by_name.get("bid_size")),
            "ask_size": quote.get("ask_size", token_by_name.get("ask_size")),
            "iv": quote.get("iv", token_by_name.get("iv")),
            "delta": quote.get("delta", token_by_name.get("delta")),
            "gamma": quote.get("gamma", token_by_name.get("gamma")),
            "theta": quote.get("theta", token_by_name.get("theta")),
            "feature_contract_version": quote.get("feature_contract_version") or FEATURE_CONTRACT_VERSION,
            "source_quote_time": quote.get("source_quote_time") or quote.get("source_quote_ts"),
            "source_context_time": quote.get("source_context_time") or quote.get("source_context_ts"),
            "raw_quote_timestamp_utc": quote.get("raw_quote_timestamp_utc") or quote.get("quote_timestamp"),
            "quote_age_ms": quote.get("quote_age_ms"),
            "quote_age_source": quote.get("quote_age_source"),
        }
        if include_features:
            row["token_features"] = token_features
        rows.append(row)
    rows.sort(key=lambda item: (float(item["edge"]) if item.get("edge") is not None else -math.inf), reverse=True)
    return rows


def scalar_feature_payload(decision: SurfaceDecision) -> dict[str, Any]:
    scalar = [float(value) for value in np.asarray(decision.scalar_features, dtype=float)]
    market_last = [float(value) for value in np.asarray(decision.market_last, dtype=float)]
    return {
        "scalar_features": scalar,
        "scalar_feature_hash": stable_json_hash(scalar),
        "market_last": market_last,
        "market_last_hash": stable_json_hash(market_last),
    }


def live_surface_decision(
    *,
    session: str,
    row: dict[str, Any],
    variant: SurfaceVariant,
    policy_index: int,
    index_state: LiveIndexState,
) -> SurfaceDecision:
    return surface_decision_from_row(
        session=session,
        row=row,
        policy_index=int(policy_index),
        variant=variant,
        market_cache=LiveMarketStructureAdapter(index_state),  # type: ignore[arg-type]
    )


def order_intent_from_prediction(
    prediction: dict[str, Any],
    contract_lookup: dict[str, dict[str, Any]],
    *,
    quantity: int = 1,
) -> PaperOrderIntent | None:
    selected = prediction.get("selected") or {}
    contract_id = str(selected.get("contract_id") or "")
    if prediction.get("action") != "enter" or not contract_id:
        return None
    contract = contract_lookup.get(contract_id)
    if not contract:
        return None
    return PaperOrderIntent(
        action="BUY",
        symbol=str(contract.get("symbol") or "SPX"),
        expiry=str(contract["expiry"]),
        strike=float(contract["strike"]),
        right=str(contract["right"]).upper(),
        quantity=int(quantity),
        limit_price=float(contract["ask"]),
        trading_class=str(contract.get("trading_class") or "SPXW"),
        exchange=str(contract.get("exchange") or "SMART"),
        currency=str(contract.get("currency") or "USD"),
    )


def selected_contract_payload(intent: PaperOrderIntent | None, contract_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if intent is None:
        return {}
    for contract_id, contract in contract_lookup.items():
        if (
            str(contract.get("expiry")) == intent.expiry
            and float(contract.get("strike")) == float(intent.strike)
            and str(contract.get("right")).upper() == intent.right
        ):
            payload = {
                "contract_id": contract_id,
                "symbol": intent.symbol,
                "root": "SPXW",
                "trading_class": intent.trading_class,
                "settlement": str(contract.get("settlement") or "PM"),
                "expiry": intent.expiry,
                "strike": intent.strike,
                "right": intent.right,
                "exchange": intent.exchange,
                "currency": intent.currency,
            }
            for key in (
                "bid",
                "ask",
                "mid",
                "spread",
                "spread_frac",
                "bid_size",
                "ask_size",
                "quote_age_ms",
                "quote_age_source",
                "raw_quote_timestamp_utc",
                "quote_timestamp",
                "quote_timestamp_ms",
                "received_timestamp_utc",
                "received_timestamp",
                "received_timestamp_ms",
                "decision_timestamp_utc",
                "decision_timestamp",
                "decision_timestamp_ms",
            ):
                if key in contract:
                    payload[key] = contract[key]
            if "raw_quote_timestamp_utc" not in payload and "quote_timestamp" in payload:
                payload["raw_quote_timestamp_utc"] = payload["quote_timestamp"]
            if "received_timestamp_utc" not in payload and "received_timestamp" in payload:
                payload["received_timestamp_utc"] = payload["received_timestamp"]
            if "decision_timestamp_utc" not in payload and "decision_timestamp" in payload:
                payload["decision_timestamp_utc"] = payload["decision_timestamp"]
            return payload
    return {
        "symbol": intent.symbol,
        "root": "SPXW",
        "trading_class": intent.trading_class,
        "settlement": "PM",
        "expiry": intent.expiry,
        "strike": intent.strike,
        "right": intent.right,
        "exchange": intent.exchange,
        "currency": intent.currency,
    }


def _option_feature_row(
    *,
    quote: dict[str, Any],
    decision_time: datetime | pd.Timestamp,
    spx: float,
    strike: float,
    right: str,
    atm_strike: int,
) -> np.ndarray:
    values = option_feature_values(
        quote,
        decision_time=_utc_timestamp(decision_time),
        spx=float(spx),
        strike=float(strike),
        right=right,
        atm_strike=atm_strike,
        feature_names=OPTION_FEATURE_NAMES,
        live_contract=True,
        risk_free_rate=LIVE_DATASET_CONFIG.risk_free_rate,
        dividend_yield=LIVE_DATASET_CONFIG.dividend_yield,
    )
    return np.asarray([values[name] for name in OPTION_FEATURE_NAMES], dtype=np.float32)


def _market_features_from_live(frame: pd.DataFrame, decision_time: pd.Timestamp, spx_close: float, vix_close: float) -> np.ndarray:
    hist = frame[frame["timestamp"] <= decision_time]
    if hist.empty:
        return np.asarray([spx_close, vix_close, spx_close, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close = hist["spx"].astype(float).to_numpy()
    vwap = float(np.nanmean(close))
    session_open = float(close[0])
    session_high = float(np.nanmax(close))
    session_low = float(np.nanmin(close))
    session_range = session_high - session_low
    omar = (spx_close - session_open) / session_range if session_range > 0 else 0.0
    momentum_5 = spx_close - float(close[-6]) if len(close) >= 6 else 0.0
    momentum_15 = spx_close - float(close[-16]) if len(close) >= 16 else 0.0
    return np.asarray([spx_close, vix_close, vwap, omar, session_range, momentum_5, momentum_15], dtype=np.float32)


def _valid_quote(quote: dict[str, Any]) -> bool:
    quote_age_ms = _float(quote.get("quote_age_ms"), math.inf)
    return (
        str(quote.get("trading_class") or "SPXW") == "SPXW"
        and str(quote.get("right") or "").upper() in {"C", "P"}
        and math.isfinite(_float(quote.get("strike"), math.nan))
        and _float(quote.get("bid"), -1.0) >= 0.0
        and _float(quote.get("ask"), 0.0) > 0.0
        and _float(quote.get("ask"), 0.0) >= _float(quote.get("bid"), 0.0)
        and quote_age_ms <= LIVE_DATASET_CONFIG.max_quote_age_seconds * 1000.0
    )


def _quote_prefilter_diagnostics(quote: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    trading_class = str(quote.get("trading_class") or "SPXW")
    right = str(quote.get("right") or "").upper()
    strike = _float(quote.get("strike"), math.nan)
    bid = _float(quote.get("bid"), math.nan)
    ask = _float(quote.get("ask"), math.nan)
    mid = _float(quote.get("mid"), math.nan)
    quote_age_ms = _float(quote.get("quote_age_ms"), math.inf)
    max_quote_age_ms = LIVE_DATASET_CONFIG.max_quote_age_seconds * 1000.0
    freshness_pass = quote_age_ms <= max_quote_age_ms
    if trading_class != "SPXW":
        reasons.append("invalid_trading_class")
    if right not in {"C", "P"}:
        reasons.append("invalid_right")
    if not math.isfinite(strike):
        reasons.append("missing_strike")
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0.0 or ask <= 0.0 or ask < bid:
        reasons.append("invalid_bid_ask")
    if not freshness_pass:
        reasons.append("stale_quote")
    return {
        "passed": not reasons,
        "tradability_pass": not any(reason != "stale_quote" for reason in reasons),
        "freshness_pass": freshness_pass,
        "reasons": sorted(set(reasons)),
        "thresholds": {
            "max_quote_age_ms": max_quote_age_ms,
        },
        "observed": {
            "strike": _float_or_none(strike),
            "bid": _float_or_none(bid),
            "ask": _float_or_none(ask),
            "mid": _float_or_none(mid),
            "quote_age_ms": _float_or_none(quote_age_ms),
        },
    }


def _live_candidate_is_tradable(
    features: np.ndarray,
    *,
    feature_contract_name: str | None = FEATURE_CONTRACT_VERSION,
) -> bool:
    values = {name: float(features[idx]) for idx, name in enumerate(OPTION_FEATURE_NAMES)}
    contract_version = feature_contract_version(feature_contract_name)
    contract = LIVE_FEATURE_CONTRACTS.get(contract_version, DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT)
    return candidate_is_tradable_values(
        values,
        contract,
        require_greeks=feature_contract_requires_model_scoring_greeks(contract_version),
    )


def _contract_id(*, strike: float, expiry: str, right: str) -> str:
    return f"SPXW-{expiry}-{float(strike):09.3f}-{right}"


def _round_to_5(value: float, *, step: int = 5) -> int:
    return round_to_strike_step(value, step)


def _utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _minute_of_session(value: pd.Timestamp) -> int:
    local = value.tz_convert(NY)
    return max(0, (local.hour * 60 + local.minute) - (9 * 60 + 30))


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
