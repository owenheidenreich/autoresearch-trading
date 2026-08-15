"""Causal Protocol101 replay over immutable IBKR recorder checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.live.ibkr_market_capture import apply_option_capture_event, clean_json, iter_capture_rows, stable_hash
from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol066_inference import load_protocol066_artifact, predict_protocol066_sequence, prediction_for_step
from v4.live.protocol101_decision_trace import normalize_decision_trace
from v4.live.protocol101_synchronization import CanonicalMarketMinuteV2
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_frame_from_surface,
    protocol101_candidate_gate_diagnostics,
)
from v4.live.protocol101_feature_contract import (
    DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
    FEATURE_CONTRACT_VERSION,
    quote_source_metadata,
    round_to_strike_step,
)
from v4.live.protocol101_live_entry import (
    LiveIndexState,
    build_live_surface_row,
    live_surface_decision,
    scalar_feature_payload,
    scored_token_universe_payload,
)
from v4.model.hypothesis_protocol import registered_aplus_surface_variants
from v4.scripts.run_protocol081_live_shadow_router import _live_feature_row


NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ReplayArtifacts:
    surface_manifest: Path
    protocol101_manifest: Path
    protocol101_summary: Path
    lifecycle_manifest: Path | None = None


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def variant_for(name: str) -> Any:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise ValueError(f"no registered surface variant named {name!r}")


def canonical_quote(raw: dict[str, Any], *, spx: float, decision_time: pd.Timestamp, received_time: str | None) -> dict[str, Any]:
    greeks = raw.get("model_greeks") if isinstance(raw.get("model_greeks"), dict) else {}
    local = decision_time.tz_convert(NY)
    settlement = local.replace(hour=16, minute=0, second=0, microsecond=0).tz_convert("UTC")
    received = raw.get("last_received_timestamp_utc") or received_time
    source = raw.get("source_timestamp_utc") or received
    quote = {
        **raw,
        "contract_id": raw.get("contract_id"),
        "symbol": "SPX",
        "trading_class": "SPXW",
        "settlement": "PM",
        "expiry": str(raw.get("expiry") or local.strftime("%Y%m%d"))[:8],
        "bid": raw.get("bid"),
        "ask": raw.get("ask"),
        "mid": raw.get("mid"),
        "bid_size": raw.get("bid_size"),
        "ask_size": raw.get("ask_size"),
        "underlying_price": spx,
        "iv": greeks.get("implied_vol"),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "prefer_repaired_greeks": True,
        "settlement_time_utc": settlement.isoformat(),
        "raw_quote_timestamp_utc": source,
        "received_timestamp_utc": received,
        "decision_timestamp_utc": decision_time.isoformat(),
        "market_data_type": raw.get("market_data_type"),
        "market_data_type_name": raw.get("market_data_type_name"),
        "raw_vendor_fields": raw,
    }
    return {
        **quote,
        **quote_source_metadata(
            quote,
            decision_time=decision_time,
            feature_contract_name=FEATURE_CONTRACT_VERSION,
        ),
    }


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _index_value(payload: dict[str, Any]) -> float | None:
    """Return a tradable index observation, excluding IBKR's stale close fallback."""

    last = _positive_number(payload.get("last"))
    if last is not None:
        return last
    bid = _positive_number(payload.get("bid"))
    ask = _positive_number(payload.get("ask"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    price = _positive_number(payload.get("price"))
    close = _positive_number(payload.get("close"))
    if price is None:
        return None
    # IBKR's marketPrice() falls back to the prior close before the first live
    # index print. Treating that fallback as a 09:30 observation inflates OMAR.
    if close is not None and math.isclose(price, close, rel_tol=0.0, abs_tol=1e-9):
        return None
    return price


def build_replay_inputs(
    events_path: Path,
    *,
    session: str,
    decision_start_et: str = "09:31",
    decision_end_et: str = "15:30",
) -> tuple[list[dict[str, Any]], LiveIndexState]:
    index_state = LiveIndexState()
    latest_spx: float | None = None
    latest_vix: float | None = None
    latest_spx_payload: dict[str, Any] = {}
    latest_vix_payload: dict[str, Any] = {}
    option_state: dict[str, dict[str, Any]] = {}
    causal_checkpoints: dict[str, dict[str, Any]] = {}
    fallback_checkpoints: dict[str, dict[str, Any]] = {}
    boundaries = list(
        pd.date_range(
            pd.Timestamp(f"{session} {decision_start_et}", tz=NY).tz_convert("UTC"),
            pd.Timestamp(f"{session} {decision_end_et}", tz=NY).tz_convert("UTC"),
            freq="min",
        )
    )
    boundary_index = 0

    def capture_boundary(boundary: pd.Timestamp) -> None:
        if latest_spx is None or latest_vix is None or not option_state:
            return
        atm = round_to_strike_step(
            float(latest_spx),
            DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT.strike_step,
        )
        contract_window = (
            float(DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT.ladder_dollars)
            + float(DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT.strike_step) * 2.0
        )
        expiry = session.replace("-", "")
        contracts = [
            dict(item)
            for item in option_state.values()
            if str(item.get("expiry") or "")[:8] == expiry
            and str(item.get("right") or "").upper() in {"C", "P"}
            and math.isfinite(float(item.get("strike") or math.nan))
            and abs(float(item["strike"]) - float(atm)) <= contract_window
        ]
        contracts.sort(key=lambda item: (float(item.get("strike") or 0.0), str(item.get("right") or "")))
        if not contracts:
            return
        local = boundary.tz_convert(NY)
        completed = local - pd.Timedelta(minutes=1)
        key = boundary.isoformat()
        causal_checkpoints[key] = {
            "completed_minute_et": completed.isoformat(),
            "decision_time_et": local.isoformat(),
            "spx": {**latest_spx_payload, "symbol": "SPX", "price": latest_spx},
            "vix": {**latest_vix_payload, "symbol": "VIX", "price": latest_vix},
            "atm_strike": atm,
            "contracts": contracts,
            "contract_count": len(contracts),
            "checkpoint_contract_filter_atm_strike": atm,
            "checkpoint_contract_filter_window_points": contract_window,
            "replay_checkpoint_source": "event_boundary_at_or_before_decision",
            "_received_timestamp_utc": boundary.isoformat(),
        }

    for _, row in iter_capture_rows(events_path):
        if row is None:
            continue
        event_type = row.get("event_type")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        received = parse_timestamp(row.get("received_timestamp_utc"))
        while boundary_index < len(boundaries) and received is not None and received > boundaries[boundary_index]:
            capture_boundary(boundaries[boundary_index])
            boundary_index += 1
        if event_type == "index_update" and received is not None:
            if payload.get("symbol") == "SPX":
                prior_close = _positive_number(payload.get("close"))
                if prior_close is not None:
                    index_state.set_previous_session_close(prior_close)
            value = _index_value(payload)
            if payload.get("symbol") == "SPX" and value is not None:
                latest_spx = value
                latest_spx_payload = dict(payload)
            elif payload.get("symbol") == "VIX" and value is not None:
                latest_vix = value
                latest_vix_payload = dict(payload)
            if latest_spx is not None and latest_vix is not None:
                index_state.add(timestamp=received, spx=latest_spx, vix=latest_vix)
        elif event_type in {"option_update", "option_delta"}:
            apply_option_capture_event(option_state, row)
        elif event_type == "ladder_checkpoint":
            copied = dict(payload)
            copied["_received_timestamp_utc"] = row.get("received_timestamp_utc")
            decision = parse_timestamp(copied.get("decision_time_et"))
            if decision is not None:
                fallback_checkpoints[decision.isoformat()] = copied
    existing = {str(item.get("completed_minute_et")) for item in fallback_checkpoints.values()}
    overlay = events_path.parent / "derived_checkpoint_overlays.jsonl"
    if overlay.exists():
        for line in overlay.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            minute = str(payload.get("completed_minute_et") or "")
            replacement = bool(payload.get("replacement_for_raw_checkpoint"))
            if not minute or (minute in existing and not replacement):
                continue
            copied = dict(payload)
            copied["_received_timestamp_utc"] = row.get("received_timestamp_utc")
            decision = parse_timestamp(copied.get("decision_time_et"))
            if decision is not None:
                fallback_checkpoints[decision.isoformat()] = copied
            existing.add(minute)
    checkpoints = [
        causal_checkpoints.get(boundary.isoformat()) or fallback_checkpoints.get(boundary.isoformat())
        for boundary in boundaries
    ]
    checkpoints = [item for item in checkpoints if item is not None]
    checkpoints.sort(key=lambda item: str(item.get("completed_minute_et") or ""))
    return checkpoints, _compact_index_state(index_state)


def lifecycle_canonical_rows(
    replay_inputs: tuple[list[dict[str, Any]], LiveIndexState],
    *,
    session: str,
) -> list[dict[str, Any]]:
    """Build causal quote rows through the lifecycle forced-flat window."""

    checkpoints, raw_index_state = replay_inputs
    causal_state = LiveIndexState(prior_session_close=raw_index_state.prior_session_close)
    raw_rows = sorted(raw_index_state.rows, key=lambda item: item["timestamp"])
    raw_cursor = 0
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        decision_time = parse_timestamp(checkpoint.get("decision_time_et"))
        if decision_time is None:
            continue
        while raw_cursor < len(raw_rows) and pd.Timestamp(raw_rows[raw_cursor]["timestamp"]) <= decision_time:
            item = raw_rows[raw_cursor]
            causal_state.add(timestamp=item["timestamp"], spx=float(item["spx"]), vix=float(item["vix"]))
            raw_cursor += 1
        spx_payload = checkpoint.get("spx") if isinstance(checkpoint.get("spx"), dict) else {}
        vix_payload = checkpoint.get("vix") if isinstance(checkpoint.get("vix"), dict) else {}
        completed_context = causal_state.frame(decision_time.floor("min") - pd.Timedelta(microseconds=1))
        spx = float(completed_context.iloc[-1]["spx"]) if not completed_context.empty else _index_value(spx_payload)
        vix = float(completed_context.iloc[-1]["vix"]) if not completed_context.empty else _index_value(vix_payload)
        if spx is None or vix is None:
            continue
        rows.append(
            {
                "session": session,
                "decision_time": decision_time.isoformat(),
                "completed_minute_et": checkpoint.get("completed_minute_et"),
                "spx": spx,
                "vix": vix,
                "atm_strike": checkpoint.get("atm_strike"),
                "raw_contract_count": len(checkpoint.get("contracts", [])),
                "raw_vendor_checkpoint_json": json.dumps(clean_json(checkpoint), sort_keys=True),
            }
        )
    return rows


def _compact_index_state(state: LiveIndexState) -> LiveIndexState:
    """Preserve minute OHLC extrema without replaying every index tick."""

    if not state.rows:
        return state
    frame = pd.DataFrame(state.rows).sort_values("timestamp")
    frame["minute"] = frame["timestamp"].dt.floor("min")
    grouped = frame.groupby("minute", sort=True).agg(
        spx_open=("spx", "first"),
        spx_high=("spx", "max"),
        spx_low=("spx", "min"),
        spx_close=("spx", "last"),
        vix_close=("vix", "last"),
    )
    compact = LiveIndexState(
        max_rows=max(int(state.max_rows), len(grouped) * 4 + 1),
        prior_session_close=state.prior_session_close,
    )
    offsets = (0, 15, 30, 45)
    for minute, row in grouped.iterrows():
        for offset, spx in zip(
            offsets,
            (row["spx_open"], row["spx_high"], row["spx_low"], row["spx_close"]),
        ):
            compact.add(
                timestamp=pd.Timestamp(minute) + pd.Timedelta(seconds=offset),
                spx=float(spx),
                vix=float(row["vix_close"]),
            )
    return compact


def replay_capture(
    events_path: Path,
    artifacts: ReplayArtifacts,
    *,
    session: str,
    run_id: str,
    decision_start_et: str = "09:31",
    decision_end_et: str = "15:30",
    replay_inputs: tuple[list[dict[str, Any]], LiveIndexState] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints, raw_index_state = replay_inputs or build_replay_inputs(
        events_path,
        session=session,
        decision_start_et=decision_start_et,
        decision_end_et=decision_end_et,
    )
    surface = load_surface_edge_artifact(artifacts.surface_manifest)
    protocol101 = load_protocol101_entry_artifact(artifacts.protocol101_manifest, artifacts.protocol101_summary)
    variant = variant_for(surface.variant_name)
    history = Protocol101HistoryState()
    causal_state = LiveIndexState(prior_session_close=raw_index_state.prior_session_close)
    raw_rows = sorted(raw_index_state.rows, key=lambda item: item["timestamp"])
    raw_cursor = 0
    traces: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []

    for decision_index, checkpoint in enumerate(checkpoints):
        decision_time = parse_timestamp(checkpoint.get("decision_time_et"))
        if decision_time is None:
            continue
        local = decision_time.tz_convert(NY)
        if local.time() < datetime.strptime(decision_start_et, "%H:%M").time() or local.time() > datetime.strptime(decision_end_et, "%H:%M").time():
            continue
        while raw_cursor < len(raw_rows) and pd.Timestamp(raw_rows[raw_cursor]["timestamp"]) <= decision_time:
            item = raw_rows[raw_cursor]
            causal_state.add(timestamp=item["timestamp"], spx=float(item["spx"]), vix=float(item["vix"]))
            raw_cursor += 1
        spx_payload = checkpoint.get("spx") if isinstance(checkpoint.get("spx"), dict) else {}
        vix_payload = checkpoint.get("vix") if isinstance(checkpoint.get("vix"), dict) else {}
        completed_context = causal_state.frame(decision_time.floor("min") - pd.Timedelta(microseconds=1))
        spx = float(completed_context.iloc[-1]["spx"]) if not completed_context.empty else _index_value(spx_payload)
        vix = float(completed_context.iloc[-1]["vix"]) if not completed_context.empty else _index_value(vix_payload)
        if spx is None or vix is None:
            continue
        if not causal_state.rows:
            causal_state.add(timestamp=decision_time - pd.Timedelta(seconds=1), spx=spx, vix=vix)
        received = checkpoint.get("_received_timestamp_utc")
        quotes = [canonical_quote(item, spx=spx, decision_time=decision_time, received_time=received) for item in checkpoint.get("contracts", []) if isinstance(item, dict)]
        row, lookup = build_live_surface_row(
            decision_time=decision_time,
            spx=spx,
            vix=vix,
            option_quotes=quotes,
            index_state=causal_state,
        )
        decision = live_surface_decision(session=session, row=row, variant=variant, policy_index=surface.policy_index, index_state=causal_state)
        surface_scores = score_surface_decisions(surface, [decision])[0]
        diagnostics = protocol101_candidate_gate_diagnostics(decision, surface_scores, min_edge=25.0, max_rows=1000)
        candidates = protocol101_candidate_frame_from_surface(decision, surface_scores, history, min_edge=25.0)
        prediction = predict_protocol101_entry(protocol101, candidates)
        if not candidates.empty:
            history.update(candidates, decision_time)
        universe = scored_token_universe_payload(decision, surface_scores, lookup, include_features=True)
        scalar = scalar_feature_payload(decision)
        features = {
            **scalar,
            "token_features": [
                {
                    "contract_id": item.get("contract_id"),
                    "token_idx": item.get("token_idx"),
                    "token_feature_hash": item.get("token_feature_hash"),
                    "token_features": item.get("token_features"),
                }
                for item in universe
            ],
        }
        scores = {
            "surface_scores": [float(value) for value in np.asarray(surface_scores, dtype=float)],
            "protocol101_raw_logits": prediction.get("raw_logits", []),
            "protocol101_wait_logit": prediction.get("wait_logit"),
            "protocol101_candidate_logits": prediction.get("candidate_logits", []),
        }
        selected = prediction.get("selected") or {}
        context = causal_state.session_context_summary(decision_time)
        block_reasons = [] if prediction.get("action") == "enter" else [str(prediction.get("reason") or "wait")]
        source_quote_times = [parse_timestamp(item.get("source_quote_time")) for item in universe]
        source_quote_times = [item for item in source_quote_times if item is not None]
        filter_reason = diagnostics.get("filter_reason")
        trace_input = {
            "protocol_id": "protocol101",
            "source": "ibkr_captured_offline_replay",
            "session": session,
            "run_id": run_id,
            "mode": "recorder-offline-replay",
            "decision_ts": decision_time.isoformat(),
            "decision_index": decision_index,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "source_quote_ts": max(source_quote_times).isoformat() if source_quote_times else received,
            "source_context_ts": context.get("last_timestamp"),
            "candidate_count": len(universe),
            "candidate_universe": universe,
            "candidate_universe_hash": stable_hash(universe),
            "features": features,
            "feature_hash": stable_hash(features),
            "model_scores": scores,
            "score_hash": stable_hash(scores),
            "selected_action": prediction.get("selected_action") or "wait",
            "selected_contract_id": selected.get("contract_id"),
            "selected_score": prediction.get("margin"),
            "decision_threshold": protocol101.threshold,
            "block_reasons": block_reasons,
            "risk_gate": {
                "passed": True,
                "reason": filter_reason,
                "reasons": []
                if filter_reason in {"candidates_available", "below_min_edge", "no_valid_surface_scores", None}
                else [filter_reason],
            },
            "account_state": {"mode": "offline_flat_one_account", "open_positions": 0},
            "quote_freshness_ms": row.get("max_quote_age_ms"),
            "candidate_gate_diagnostics": clean_json(diagnostics),
            "opening_context_ready": context.get("opening_context_ready"),
            "missing_opening_minutes": context.get("missing_opening_minutes"),
            "broker_order_endpoint_called": False,
            "raw_checkpoint_hash": stable_hash(checkpoint),
        }
        trace = normalize_decision_trace(trace_input, source="ibkr_captured_offline_replay").to_dict()
        traces.append(clean_json(trace))
        canonical_contract = CanonicalMarketMinuteV2(
            session=session,
            completed_minute_et=str(checkpoint.get("completed_minute_et") or ""),
            decision_time_et=local.isoformat(),
            source="ibkr_captured_offline_replay",
            source_quote_time_utc=max(source_quote_times).isoformat() if source_quote_times else received,
            source_context_time_utc=context.get("last_timestamp"),
            raw_input_hash=stable_hash(checkpoint),
            canonical_feature_hash=str(trace.get("feature_hash") or ""),
            candidate_universe_hash=str(trace.get("candidate_universe_hash") or ""),
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            opening_context_ready=bool(context.get("opening_context_ready")),
            missing_opening_minutes=int(context.get("missing_opening_minutes") or 0),
        )
        canonical.append({
            **canonical_contract.to_dict(),
            "session": session,
            "decision_time": decision_time.isoformat(),
            "completed_minute_et": checkpoint.get("completed_minute_et"),
            "spx": spx,
            "vix": vix,
            "atm_strike": checkpoint.get("atm_strike"),
            "raw_contract_count": len(checkpoint.get("contracts", [])),
            "tradable_token_count": len(universe),
            "candidate_count": len(candidates),
            "selected_action": trace.get("selected_action"),
            "selected_contract_id": trace.get("selected_contract_id"),
            "selected_score": trace.get("selected_score"),
            "decision_threshold": trace.get("decision_threshold"),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "candidate_universe_hash": trace.get("candidate_universe_hash"),
            "feature_hash": trace.get("feature_hash"),
            "score_hash": trace.get("score_hash"),
            "opening_context_ready": context.get("opening_context_ready"),
            "missing_opening_minutes": context.get("missing_opening_minutes"),
            "raw_vendor_checkpoint_json": json.dumps(clean_json(checkpoint), sort_keys=True),
        })
    return canonical, traces


def trace_identity(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_ts": trace.get("decision_ts"),
        "decision_index": trace.get("decision_index"),
        "candidate_universe_hash": trace.get("candidate_universe_hash"),
        "feature_hash": trace.get("feature_hash"),
        "score_hash": trace.get("score_hash"),
        "selected_action": trace.get("selected_action"),
        "selected_contract_id": trace.get("selected_contract_id"),
        "block_reasons": trace.get("block_reasons"),
    }


def deterministic_trace_hash(traces: list[dict[str, Any]]) -> str:
    return stable_hash([trace_identity(trace) for trace in traces])


def replay_lifecycle(
    canonical_rows: list[dict[str, Any]],
    entry_traces: list[dict[str, Any]],
    lifecycle_manifest: Path | None,
) -> list[dict[str, Any]]:
    """Run the frozen current live lifecycle semantics over simulated ask fills."""

    if lifecycle_manifest is None:
        return []
    artifact = load_protocol066_artifact(lifecycle_manifest)
    trace_by_time = {str(row.get("decision_ts")): row for row in entry_traces}
    active: dict[str, Any] | None = None
    lifecycle_rows: list[dict[str, Any]] = []

    def open_trade_from_entry(
        entry_trace: dict[str, Any] | None,
        contracts: dict[str, dict[str, Any]],
        decision_time: pd.Timestamp,
    ) -> dict[str, Any] | None:
        if not entry_trace or entry_trace.get("selected_action") != "enter":
            return None
        contract_id = str(entry_trace.get("selected_contract_id") or "")
        quote = contracts.get(contract_id) or {}
        try:
            entry_ask = float(quote.get("ask"))
        except (TypeError, ValueError):
            return None
        session_forced_flat = (
            decision_time.tz_convert(NY)
            .replace(hour=15, minute=55, second=0, microsecond=0)
            .tz_convert("UTC")
        )
        trade_deadline = min(decision_time + pd.Timedelta(minutes=25), session_forced_flat)
        return {
            "contract_id": contract_id,
            "entry_time": decision_time,
            "deadline": trade_deadline,
            "session_forced_flat": session_forced_flat,
            "entry_ask": entry_ask,
            "entry_edge": float(entry_trace.get("selected_score") or 0.0),
            "mfe": (float(quote.get("bid") or entry_ask) - entry_ask) * 100.0,
            "mae": (float(quote.get("bid") or entry_ask) - entry_ask) * 100.0,
            "mfe_step": 0,
            "feature_rows": [],
            "pnl_path": [],
            "last_decision_time": None,
            "last_quote": dict(quote),
            "last_quote_time": decision_time,
        }

    for canonical in canonical_rows:
        decision_time = parse_timestamp(canonical.get("decision_time"))
        if decision_time is None:
            continue
        checkpoint = json.loads(str(canonical.get("raw_vendor_checkpoint_json") or "{}"))
        contracts = {
            str(item.get("contract_id")): item
            for item in checkpoint.get("contracts", [])
            if isinstance(item, dict) and item.get("contract_id")
        }
        entry_trace = trace_by_time.get(decision_time.isoformat())
        if active is None:
            active = open_trade_from_entry(entry_trace, contracts, decision_time)
            continue
        quote = contracts.get(str(active["contract_id"]))
        if not quote:
            if decision_time >= pd.Timestamp(active["deadline"]) and active.get("last_quote"):
                last_quote = dict(active["last_quote"])
                last_bid = float(last_quote.get("bid") or active["entry_ask"])
                pnl = (last_bid - float(active["entry_ask"])) * 100.0
                lifecycle_rows.append(
                    {
                        "decision_ts": decision_time.isoformat(),
                        "contract_id": active["contract_id"],
                        "action": "forced_flat",
                        "reason": "mandatory_time_flat_last_captured_quote",
                        "feature_hash": stable_hash({}),
                        "sequence_feature_hash": stable_hash(
                            clean_json(
                                [
                                    row.to_dict() if isinstance(row, pd.Series) else row
                                    for row in active["feature_rows"]
                                ]
                            )
                        ),
                        "sequence_length": len(active["feature_rows"]),
                        "entry_ask": active["entry_ask"],
                        "current_bid": last_bid,
                        "current_pnl": pnl,
                        "quote_missing": True,
                        "terminal_quote_stale": True,
                        "last_quote_time": pd.Timestamp(active["last_quote_time"]).isoformat(),
                        "broker_order_endpoint_called": False,
                    }
                )
                active = open_trade_from_entry(entry_trace, contracts, decision_time)
                continue
            lifecycle_rows.append({
                "decision_ts": decision_time.isoformat(),
                "contract_id": active["contract_id"],
                "action": "hold",
                "reason": "captured_contract_quote_missing",
                "feature_hash": stable_hash({}),
                "quote_missing": True,
            })
            continue
        try:
            bid = float(quote["bid"])
            ask = float(quote["ask"])
        except (KeyError, TypeError, ValueError):
            continue
        mid = float(quote.get("mid") or (bid + ask) / 2.0)
        active["last_quote"] = dict(quote)
        active["last_quote_time"] = decision_time
        spx = float(canonical["spx"])
        contract = SimpleNamespace(
            strike=float(quote.get("strike") or 0.0),
            right=str(quote.get("right") or ""),
            lastTradeDateOrContractMonth=str(quote.get("expiry") or ""),
        )
        live_quote = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": ask - bid,
            "spread_frac": (ask - bid) / mid if mid > 0 else 0.0,
            "bid_size": float(quote.get("bid_size") or 0.0),
            "ask_size": float(quote.get("ask_size") or 0.0),
        }
        feature_row = _live_feature_row(
            contract=contract,
            quote=live_quote,
            spx_price=spx,
            now=decision_time.to_pydatetime(),
            atm_strike=int(canonical.get("atm_strike") or round(spx / 5.0) * 5),
            feature_columns=artifact.feature_columns,
        )
        if feature_row is None:
            continue
        minutes_since_entry = max((decision_time - pd.Timestamp(active["entry_time"])).total_seconds() / 60.0, 0.0)
        pnl = (bid - float(active["entry_ask"])) * 100.0
        active["mfe"] = max(float(active["mfe"]), pnl)
        active["mae"] = min(float(active["mae"]), pnl)
        feature_row["minutes_since_entry"] = minutes_since_entry
        feature_row["current_pnl"] = pnl
        feature_row["mfe_to_now"] = active["mfe"]
        feature_row["mae_to_now"] = active["mae"]
        feature_row["giveback_from_mfe"] = max(0.0, float(active["mfe"]) - pnl)
        feature_row["giveback_fraction"] = max(0.0, (float(active["mfe"]) - pnl) / max(abs(float(active["mfe"])), 1.0))
        feature_row["bid_over_entry_ask"] = bid / max(float(active["entry_ask"]), 1e-6)
        feature_row["mid_over_entry_ask"] = mid / max(float(active["entry_ask"]), 1e-6)
        feature_row["entry_edge"] = active["entry_edge"]
        forced_flat = pd.Timestamp(active["session_forced_flat"])
        deadline = pd.Timestamp(active["deadline"])
        minutes_to_forced_flat = (forced_flat - decision_time).total_seconds() / 60.0
        minutes_to_deadline = (deadline - decision_time).total_seconds() / 60.0
        feature_row["minutes_to_forced_flat"] = max(0.0, minutes_to_forced_flat)
        feature_row["minutes_to_deadline"] = max(0.0, minutes_to_deadline)
        feature_row["is_baseline_exit_step"] = bool(minutes_to_deadline <= 0.0)
        feature_row["baseline_exit_reason"] = "time_flat" if minutes_to_deadline <= 0.0 else ""
        prior_time = active.get("last_decision_time")
        feature_row["quote_gap_seconds"] = (
            max((decision_time - pd.Timestamp(prior_time)).total_seconds(), 0.0)
            if prior_time is not None
            else 0.0
        )
        active["last_decision_time"] = decision_time
        active["pnl_path"].append(float(pnl))
        pnl_path = list(active["pnl_path"])
        if pnl >= float(active["mfe"]):
            active["mfe_step"] = len(pnl_path) - 1
        feature_row["time_since_mfe_minutes"] = float(len(pnl_path) - 1 - int(active["mfe_step"]))
        for horizon in (1, 3, 5):
            earlier = max(0, len(pnl_path) - 1 - horizon)
            feature_row[f"pnl_velocity_{horizon}"] = (pnl_path[-1] - pnl_path[earlier]) / max(
                len(pnl_path) - 1 - earlier,
                1,
            )
        for horizon in (5, 10):
            window = pnl_path[-horizon:]
            feature_row[f"realized_pnl_vol_{horizon}"] = float(np.std(window)) if len(window) > 1 else 0.0
        feature_row["step_idx"] = len(active["feature_rows"])
        active["feature_rows"].append(feature_row.copy())
        frame = pd.DataFrame(active["feature_rows"])
        value, recovery, decay = predict_protocol066_sequence(artifact, frame)
        step_index = len(frame) - 1
        prediction = prediction_for_step(
            step_index=step_index,
            value=value,
            recovery=recovery,
            decay=decay,
            override_threshold=artifact.selected_override_threshold,
            step_row=frame.iloc[step_index],
        )
        payload = {
            "decision_ts": decision_time.isoformat(),
            "contract_id": active["contract_id"],
            "action": prediction.action,
            "reason": prediction.reason,
            "predicted_continuation_value": prediction.predicted_continuation_value,
            "predicted_recovery_probability": prediction.predicted_recovery_probability,
            "predicted_decay_probability": prediction.predicted_decay_probability,
            "override_threshold": prediction.override_threshold,
            "step_index": step_index,
            "sequence_length": len(frame),
            "feature_hash": stable_hash(clean_json(frame.iloc[step_index].to_dict())),
            "sequence_feature_hash": stable_hash(clean_json(frame.to_dict(orient="records"))),
            "features": clean_json(frame.iloc[step_index].to_dict()),
            "entry_ask": active["entry_ask"],
            "current_bid": bid,
            "current_pnl": pnl,
            "broker_order_endpoint_called": False,
        }
        lifecycle_rows.append(payload)
        if prediction.action in {"exit", "stop", "forced_flat"}:
            active = open_trade_from_entry(entry_trace, contracts, decision_time)
    return lifecycle_rows
