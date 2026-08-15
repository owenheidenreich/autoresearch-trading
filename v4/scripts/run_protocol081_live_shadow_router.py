"""Protocol 088: no-order shadow router for Protocol 051 + 054 + 081.

This is a promotion-readiness harness, not a trading bot. It never submits,
stages, or constructs broker orders. Its job is to prove that the persisted
modular stack can be loaded, routed, logged as shadow JSONL, and validated
before any paper-trading endpoint is allowed.

Two modes are intentionally separate:

* offline-smoke: replay existing causal lifecycle rows through the persisted
  Protocol 081 artifact and emit no-order JSONL with router metadata.
* ibkr-live-probe: connect to IBKR/TWS market data only, check whether a live
  capture is currently possible, and write a blocked report if the market/feed
  is unavailable. It does not emit fake passing rows.
* ibkr-live-capture: subscribe to a small SPXW 0DTE market-data-only universe,
  emit no-order shadow observations if fresh live NBBO is available, and run
  the same parity checks. Delayed/unsubscribed market data remains a blocker.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from v4.dataset.spxw_0dte_neural import OPTION_FEATURE_NAMES
from v4.greeks.repair import compute_repaired_greeks
from v4.live.protocol066_inference import (
    load_protocol066_artifact,
    prediction_for_step,
    predict_protocol066_sequence,
)
from v4.live.shadow_parity import ShadowParityConfig, summarize_shadow_parity
from v4.scripts.run_protocol066_offline_shadow_rehearsal import (
    _load_vix_lookup,
    _prepare_trade_steps,
    _read_sequence,
    _select_trade_uids,
    _timestamp_ms,
    _vix_at,
)


_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")
_CONTRACT_MULTIPLIER = 100.0
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_088_protocol081_live_shadow_router")
DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_077_q4start_lifecycle_sequence_dataset")
DEFAULT_PROTOCOL081_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/"
    "model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1/manifest.json"
)
DEFAULT_STACK_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/"
    "model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/manifest.json"
)


@dataclass(frozen=True)
class RouterArtifacts:
    stack_manifest: Path
    protocol081_manifest: Path
    stack_payload: dict[str, Any]

    @property
    def stack_id(self) -> str:
        return f"{self.stack_payload.get('entry_protocol', 'Protocol 051')} + {self.stack_payload.get('fallback_protocol', 'Protocol 054')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("offline-smoke", "ibkr-live-probe", "ibkr-live-capture"),
        default="offline-smoke",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol-id", default="protocol081")
    parser.add_argument("--protocol-label", default="Protocol 081")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--protocol081-manifest", type=Path, default=DEFAULT_PROTOCOL081_MANIFEST)
    parser.add_argument("--stack-manifest", type=Path, default=DEFAULT_STACK_MANIFEST)
    parser.add_argument("--vix-dir", type=Path, default=Path("data/raw/index/vix_1m"))
    parser.add_argument("--split", default="q1_2026")
    parser.add_argument("--session-start", default="2026-03-01")
    parser.add_argument("--max-trades", type=int, default=50)
    parser.add_argument("--max-shadow-rows", type=int, default=1500)
    parser.add_argument("--max-quote-age-ms", type=int, default=1500)
    parser.add_argument("--max-context-age-ms", type=int, default=5000)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=7497)
    parser.add_argument("--ibkr-auto-ports", default="4002,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=81)
    parser.add_argument("--live-capture-seconds", type=float, default=45.0)
    parser.add_argument("--live-sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--live-strikes-around-atm", type=int, default=2)
    parser.add_argument("--min-live-shadow-rows", type=int, default=1)
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    return parser.parse_args()


def _load_router_artifacts(stack_manifest: Path, protocol081_manifest: Path) -> RouterArtifacts:
    if not stack_manifest.exists():
        raise SystemExit(f"missing Protocol 051/054 stack manifest: {stack_manifest}")
    if not protocol081_manifest.exists():
        raise SystemExit(f"missing Protocol 081 manifest: {protocol081_manifest}")
    stack_payload = json.loads(stack_manifest.read_text())
    required = {"entry_model", "entry_standardizer", "protocol054_risk_model", "protocol054_risk_scaler"}
    files = stack_payload.get("files", {})
    missing = sorted(name for name in required if not Path(files.get(name, "")).exists())
    if missing:
        raise SystemExit(f"stack manifest has missing files: {missing}")
    return RouterArtifacts(
        stack_manifest=stack_manifest,
        protocol081_manifest=protocol081_manifest,
        stack_payload=stack_payload,
    )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _feature_dict(row: pd.Series, feature_columns: list[str]) -> dict[str, float]:
    return {column: _finite_float(row.get(column)) for column in feature_columns}


def _shadow_observation(
    *,
    protocol_id: str,
    router: RouterArtifacts,
    artifact_ref: str,
    row: pd.Series,
    prediction,
    vix_value: float,
    vix_ts: pd.Timestamp,
    vix_source: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    quote_ts = pd.Timestamp(row["quote_ts"])
    quote_ms = _timestamp_ms(quote_ts)
    return {
        "protocol_id": protocol_id,
        "timestamp_ms": quote_ms,
        "decision_time": quote_ts.isoformat(),
        "entry_decision_time": str(row["decision_time"]),
        "session": str(row["session"]),
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": str(row["contract_id"]),
        "trade_uid": str(row["trade_uid"]),
        "sequence_step_index": int(row["step_idx"]),
        "router": {
            "mode": "no_order_shadow",
            "entry_protocol": "Protocol 051",
            "fallback_protocol": "Protocol 054",
            "override_protocol": "Protocol 081",
            "stack_manifest": str(router.stack_manifest),
            "protocol081_manifest": str(router.protocol081_manifest),
            "order_endpoint_enabled": False,
            "order_intent_policy": "always_null",
        },
        "nbbo": {
            "bid": _finite_float(row["bid"]),
            "ask": _finite_float(row["ask"]),
            "timestamp_ms": quote_ms,
        },
        "context": {
            "spx": _finite_float(row["underlying_price"]),
            "vix": float(vix_value),
            "source": f"offline_router_rehearsal:{vix_source}",
            "timestamp_ms": _timestamp_ms(vix_ts),
        },
        "features": _feature_dict(row, feature_columns),
        "model": {
            "artifact": artifact_ref,
            "fold": str(getattr(prediction, "fold", "")),
            "predicted_continuation_value": prediction.predicted_continuation_value,
            "predicted_recovery_probability": prediction.predicted_recovery_probability,
            "predicted_decay_probability": prediction.predicted_decay_probability,
            "override_threshold": prediction.override_threshold if math.isfinite(prediction.override_threshold) else 1e18,
        },
        "decision": {
            "action": prediction.action,
            "reason": prediction.reason,
        },
        "order_intent": None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(_json_sanitize(row), sort_keys=True, allow_nan=False) for row in rows)
        + ("\n" if rows else "")
    )


def _run_offline_smoke(args: argparse.Namespace) -> dict[str, Any]:
    router = _load_router_artifacts(args.stack_manifest, args.protocol081_manifest)
    artifact = load_protocol066_artifact(args.protocol081_manifest)
    trades, steps = _read_sequence(args.sequence_dir)
    trade_uids = _select_trade_uids(
        trades,
        split=args.split,
        session_start=args.session_start,
        max_trades=args.max_trades,
    )
    if not trade_uids:
        raise SystemExit("no trades selected for Protocol 081 router smoke")
    selected_steps = steps[steps["trade_uid"].isin(trade_uids)].copy()
    selected_steps = selected_steps.sort_values(["session", "trade_uid", "step_idx"])
    vix_lookup = _load_vix_lookup(args.vix_dir, set(selected_steps["session"].unique()))

    shadow_rows: list[dict[str, Any]] = []
    vix_hits = 0
    vix_misses = 0
    feature_fill_counts: dict[str, int] = {}
    for _, group in selected_steps.groupby("trade_uid", sort=False):
        prepared, fill_counts = _prepare_trade_steps(group, artifact)
        for column, count in fill_counts.items():
            feature_fill_counts[column] = feature_fill_counts.get(column, 0) + int(count)
        value, recovery, decay = predict_protocol066_sequence(artifact, prepared)
        for local_idx, row in prepared.iterrows():
            if len(shadow_rows) >= args.max_shadow_rows:
                break
            vix_value, vix_ts, vix_source = _vix_at(vix_lookup, str(row["session"]), pd.Timestamp(row["quote_ts"]))
            if vix_value is None or vix_ts is None or vix_source is None:
                vix_misses += 1
                continue
            vix_hits += 1
            prediction = prediction_for_step(
                step_index=int(local_idx),
                value=value,
                recovery=recovery,
                decay=decay,
                override_threshold=artifact.selected_override_threshold,
                step_row=row,
            )
            shadow_rows.append(
                _shadow_observation(
                    protocol_id=args.protocol_id,
                    router=router,
                    artifact_ref=artifact.artifact_ref,
                    row=row,
                    prediction=prediction,
                    vix_value=float(vix_value),
                    vix_ts=vix_ts,
                    vix_source=str(vix_source),
                    feature_columns=artifact.feature_columns,
                )
            )
        if len(shadow_rows) >= args.max_shadow_rows:
            break

    config = ShadowParityConfig(
        protocol_id=args.protocol_id,
        max_quote_age_ms=args.max_quote_age_ms,
        max_context_age_ms=args.max_context_age_ms,
    )
    parity = summarize_shadow_parity(shadow_rows, config=config)
    shadow_log = args.out_dir / "offline_router_shadow_observations.jsonl"
    _write_jsonl(shadow_log, shadow_rows)
    return {
        "mode": "offline-smoke",
        "decision": "pass" if parity["status"] == "pass" else "blocked",
        "stack_manifest": str(args.stack_manifest),
        "protocol081_manifest": str(args.protocol081_manifest),
        "shadow_log": str(shadow_log),
        "trades_rehearsed": len(trade_uids),
        "shadow_parity": parity,
        "feature_fill_counts": feature_fill_counts,
        "vix_hits": vix_hits,
        "vix_misses": vix_misses,
        "vix_coverage": float(vix_hits / max(1, vix_hits + vix_misses)),
        "no_order_guarantee": {
            "broker_order_endpoint_called": False,
            "order_intent_non_null_rows": 0,
        },
    }


def _is_regular_market_hours(now: datetime) -> bool:
    local = now.astimezone(_NY)
    if local.weekday() >= 5:
        return False
    start = local.replace(hour=9, minute=30, second=0, microsecond=0)
    end = local.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= local <= end


class IbkrErrorLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def handler(self, req_id: int, error_code: int, error_string: str, contract: Any = None, *args: Any) -> None:
        self.events.append(
            {
                "req_id": int(req_id),
                "error_code": int(error_code),
                "error_string": str(error_string),
                "contract": str(contract) if contract is not None else None,
            }
        )

    @property
    def subscription_errors(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.events
            if int(event.get("error_code", 0)) in {354, 10167, 10168}
            or "not subscribed" in str(event.get("error_string", "")).lower()
            or "delayed market data is available" in str(event.get("error_string", "")).lower()
        ]


def _candidate_ibkr_ports(args: argparse.Namespace) -> list[int]:
    ports: list[int] = []
    for value in [str(args.ibkr_port), *str(args.ibkr_auto_ports).split(",")]:
        try:
            port = int(str(value).strip())
        except ValueError:
            continue
        if port > 0 and port not in ports:
            ports.append(port)
    return ports


def _connect_ibkr(args: argparse.Namespace, ib_cls: Any) -> tuple[Any | None, int | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for port in _candidate_ibkr_ports(args):
        ib = ib_cls()
        try:
            ib.connect(args.ibkr_host, port, clientId=args.ibkr_client_id, timeout=8)
            attempts.append({"host": args.ibkr_host, "port": port, "status": "connected"})
            return ib, port, attempts
        except Exception as exc:
            attempts.append({"host": args.ibkr_host, "port": port, "status": "failed", "error": str(exc)})
            if ib.isConnected():
                ib.disconnect()
    return None, None, attempts


def _ticker_market_data_type(ticker: Any) -> int | None:
    value = getattr(ticker, "marketDataType", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _market_data_type_name(value: int | None) -> str:
    return {
        1: "live",
        2: "frozen",
        3: "delayed",
        4: "delayed_frozen",
    }.get(value, "unknown")


def _ticker_price(ticker: Any) -> float | None:
    if ticker is None:
        return None
    for name in ("last", "close"):
        value = _finite_or_none(getattr(ticker, name, None))
        if value is not None and value > 0:
            return value
    bid = _finite_or_none(getattr(ticker, "bid", None))
    ask = _finite_or_none(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    try:
        value = _finite_or_none(ticker.marketPrice())
    except Exception:
        value = None
    return value if value is not None and value > 0 else None


def _option_quote(ticker: Any) -> dict[str, float] | None:
    bid = _finite_or_none(getattr(ticker, "bid", None))
    ask = _finite_or_none(getattr(ticker, "ask", None))
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0 or ask < bid:
        return None
    bid_size = _finite_float(getattr(ticker, "bidSize", 0.0), default=0.0)
    ask_size = _finite_float(getattr(ticker, "askSize", 0.0), default=0.0)
    mid = (bid + ask) / 2.0
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": ask - bid,
        "spread_frac": (ask - bid) / mid if mid > 0 else 0.0,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }


def _wait_for_price(ib: Any, ticker: Any, *, seconds: float = 6.0) -> float | None:
    deadline = time.monotonic() + seconds
    price = _ticker_price(ticker)
    while price is None and time.monotonic() < deadline:
        ib.sleep(0.5)
        price = _ticker_price(ticker)
    return price


def _request_index_ticker(ib: Any, index_cls: Any, symbol: str) -> tuple[Any | None, Any | None]:
    contract = index_cls(symbol, "CBOE", "USD")
    qualified = ib.qualifyContracts(contract)
    contract = qualified[0] if qualified else contract
    ticker = ib.reqMktData(contract, "", False, False)
    return contract, ticker


def _today_spxw_expiry(now: datetime) -> str:
    return now.astimezone(_NY).strftime("%Y%m%d")


def _round_to_5(value: float) -> int:
    return int(round(value / 5.0) * 5)


def _discover_spxw_0dte_contracts(
    *,
    ib: Any,
    option_cls: Any,
    spx_contract: Any,
    spx_price: float,
    now: datetime,
    strikes_around_atm: int,
) -> tuple[list[Any], dict[str, Any]]:
    expiry = _today_spxw_expiry(now)
    chains = ib.reqSecDefOptParams("SPX", "", "IND", int(spx_contract.conId))
    candidates = [
        chain
        for chain in chains
        if getattr(chain, "tradingClass", "") == "SPXW" and expiry in set(getattr(chain, "expirations", ()))
    ]
    if not candidates:
        return [], {
            "expiry": expiry,
            "chains_returned": len(chains),
            "blocked_reason": "no_spxw_0dte_option_chain",
        }
    preferred = sorted(
        candidates,
        key=lambda chain: (
            0 if getattr(chain, "exchange", "") == "SMART" else 1 if getattr(chain, "exchange", "") == "CBOE" else 2,
            getattr(chain, "exchange", ""),
        ),
    )[0]
    atm = _round_to_5(spx_price)
    min_strike = atm - strikes_around_atm * 5
    max_strike = atm + strikes_around_atm * 5
    strikes = sorted(
        float(strike)
        for strike in getattr(preferred, "strikes", ())
        if min_strike <= float(strike) <= max_strike and _round_to_5(float(strike)) == int(float(strike))
    )
    contracts = [
        option_cls("SPX", expiry, strike, right, "SMART", currency="USD", tradingClass="SPXW")
        for strike in strikes
        for right in ("C", "P")
    ]
    qualified = ib.qualifyContracts(*contracts) if contracts else []
    return list(qualified), {
        "expiry": expiry,
        "chain_exchange": getattr(preferred, "exchange", None),
        "atm_strike": atm,
        "strike_count": len(strikes),
        "qualified_contracts": len(qualified),
        "requested_contracts": len(contracts),
    }


def _contract_id(contract: Any) -> str:
    strike = _finite_float(getattr(contract, "strike", 0.0), default=0.0)
    right = str(getattr(contract, "right", ""))
    expiry = str(getattr(contract, "lastTradeDateOrContractMonth", ""))
    if len(expiry) == 8:
        expiry = f"{expiry[:4]}{expiry[4:6]}{expiry[6:]}"
    return f"SPXW-{expiry}-{strike:09.3f}-{right}"


def _settlement_time_utc(now: datetime) -> datetime:
    local = now.astimezone(_NY)
    return local.replace(hour=16, minute=0, second=0, microsecond=0).astimezone(_UTC)


def _forced_flat_time_utc(now: datetime) -> datetime:
    local = now.astimezone(_NY)
    return local.replace(hour=15, minute=55, second=0, microsecond=0).astimezone(_UTC)


def _compute_live_greeks(
    *,
    spx: float,
    strike: float,
    right: str,
    mid: float,
    now: datetime,
    ask: float | None = None,
    bid: float | None = None,
) -> dict[str, float] | None:
    settlement = _settlement_time_utc(now)
    seconds = (settlement - now.astimezone(_UTC)).total_seconds()
    if seconds <= 0.0 or spx <= 0.0 or strike <= 0.0:
        return None
    t_years = seconds / (365.0 * 24.0 * 60.0 * 60.0)
    is_call = right == "C"
    estimate = compute_repaired_greeks(
        S=spx,
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
        return None
    return {
        "iv": estimate.iv,
        "delta": estimate.delta,
        "gamma": estimate.gamma,
        "theta": estimate.theta_per_day,
        "vega": estimate.vega,
    }


def _live_feature_row(
    *,
    contract: Any,
    quote: dict[str, float],
    spx_price: float,
    now: datetime,
    atm_strike: int,
    feature_columns: list[str],
) -> pd.Series | None:
    strike = _finite_float(getattr(contract, "strike", 0.0), default=0.0)
    right = str(getattr(contract, "right", ""))
    greek_row = _compute_live_greeks(
        spx=spx_price,
        strike=strike,
        right=right,
        mid=quote["mid"],
        ask=quote.get("ask"),
        bid=quote.get("bid"),
        now=now,
    )
    if greek_row is None:
        return None
    forced_flat = _forced_flat_time_utc(now)
    minutes_to_forced = max(0.0, (forced_flat - now.astimezone(_UTC)).total_seconds() / 60.0)
    minutes_to_deadline = min(25.0, minutes_to_forced)
    entry_ask = quote["ask"]
    current_pnl = (quote["bid"] - entry_ask) * _CONTRACT_MULTIPLIER
    mfe = current_pnl
    mae = current_pnl
    breakeven = (strike + entry_ask) if right == "C" else (strike - entry_ask)
    breakeven_distance = (breakeven - spx_price) if right == "C" else (spx_price - breakeven)
    option_ladder_features = {
        "bid": quote["bid"],
        "ask": quote["ask"],
        "mid": quote["mid"],
        "spread": quote["spread"],
        "spread_frac": quote["spread_frac"],
        "bid_size": quote["bid_size"],
        "ask_size": quote["ask_size"],
        "option_ohlcv_volume": 0.0,
        "stat_open_interest": 0.0,
        "underlying_price": spx_price,
        **greek_row,
        "distance_points": strike - atm_strike,
        "breakeven_distance": breakeven_distance,
    }
    row = {
        "quote_ts": pd.Timestamp(now.astimezone(_UTC)),
        "decision_time": pd.Timestamp(now.astimezone(_UTC)),
        "session": now.astimezone(_NY).date().isoformat(),
        "contract_id": _contract_id(contract),
        "trade_uid": f"live_shadow_{_contract_id(contract)}_{int(now.timestamp())}",
        "step_idx": 0,
        "is_baseline_exit_step": False,
        "baseline_exit_reason": "",
        "minutes_since_entry": 0.0,
        "minutes_to_deadline": minutes_to_deadline,
        "minutes_to_forced_flat": minutes_to_forced,
        "quote_gap_seconds": 0.0,
        "current_pnl": current_pnl,
        "mfe_to_now": mfe,
        "mae_to_now": mae,
        "giveback_from_mfe": max(0.0, mfe - current_pnl),
        "giveback_fraction": 0.0,
        "time_since_mfe_minutes": 0.0,
        "pnl_velocity_1": 0.0,
        "pnl_velocity_3": 0.0,
        "pnl_velocity_5": 0.0,
        "realized_pnl_vol_5": 0.0,
        "realized_pnl_vol_10": 0.0,
        "bid_over_entry_ask": quote["bid"] / max(entry_ask, 1e-6),
        "mid_over_entry_ask": quote["mid"] / max(entry_ask, 1e-6),
        "theta_over_mid": abs(greek_row["theta"]) / max(abs(quote["mid"]), 1e-6),
        "gamma_theta_ratio": abs(greek_row["gamma"]) / max(abs(greek_row["theta"]), 1e-6),
        "time_theta_burden": (abs(greek_row["theta"]) * (minutes_to_deadline / 390.0)) / max(abs(quote["mid"]), 1e-6),
        "entry_edge": 0.0,
        "entry_offset": strike - atm_strike,
        "entry_is_call": 1.0 if right == "C" else 0.0,
        "entry_is_put": 1.0 if right == "P" else 0.0,
        **option_ladder_features,
    }
    for column in feature_columns:
        row[column] = _finite_float(row.get(column), default=0.0)
    return pd.Series(row)


def _live_shadow_observation(
    *,
    protocol_id: str,
    router: RouterArtifacts,
    artifact_ref: str,
    row: pd.Series,
    prediction,
    spx_price: float,
    vix_value: float,
    feature_columns: list[str],
    market_data_type: str,
) -> dict[str, Any]:
    out = _shadow_observation(
        protocol_id=protocol_id,
        router=router,
        artifact_ref=artifact_ref,
        row=row,
        prediction=prediction,
        vix_value=vix_value,
        vix_ts=pd.Timestamp(row["quote_ts"]),
        vix_source=f"ibkr_{market_data_type}",
        feature_columns=feature_columns,
    )
    out["router"]["mode"] = "no_order_live_shadow"
    out["router"]["live_feed"] = "IBKR"
    out["router"]["market_data_type"] = market_data_type
    out["context"]["spx"] = float(spx_price)
    out["context"]["source"] = f"ibkr_{market_data_type}:SPX/VIX"
    return out


def _run_ibkr_live_probe(args: argparse.Namespace) -> dict[str, Any]:
    router = _load_router_artifacts(args.stack_manifest, args.protocol081_manifest)
    now = datetime.now(tz=_NY)
    live_log = args.out_dir / "live_router_shadow_observations.jsonl"
    _write_jsonl(live_log, [])
    base = {
        "mode": "ibkr-live-probe",
        "decision": "blocked",
        "stack_manifest": str(router.stack_manifest),
        "protocol081_manifest": str(router.protocol081_manifest),
        "shadow_log": str(live_log),
        "captured_rows": 0,
        "no_order_guarantee": {
            "broker_order_endpoint_called": False,
            "order_intent_non_null_rows": 0,
        },
        "market_time": now.isoformat(),
        "regular_market_hours": _is_regular_market_hours(now),
    }
    if not _is_regular_market_hours(now):
        return {
            **base,
            "blocked_reason": "outside_regular_market_hours",
            "detail": "Live SPXW no-order capture must run during 09:30-16:00 New York time.",
        }
    try:
        from ib_insync import IB, Index  # type: ignore
    except ImportError:
        return {**base, "blocked_reason": "missing_ib_insync"}

    ib = None
    error_log = IbkrErrorLog()
    try:
        ib, port, attempts = _connect_ibkr(args, IB)
        if ib is None:
            return {
                **base,
                "blocked_reason": "ibkr_connection_failed",
                "detail": "No configured IBKR API port accepted a connection.",
                "ibkr_connected": False,
                "connection_attempts": attempts,
            }
        ib.errorEvent += error_log.handler
        ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)
        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        spx_price = _wait_for_price(ib, spx_ticker, seconds=6.0)
        market_data_type = _market_data_type_name(_ticker_market_data_type(spx_ticker))
        if spx_price is None:
            return {
                **base,
                "blocked_reason": "ibkr_market_data_subscription_missing",
                "detail": (
                    "Connected to IBKR, but SPX market data did not produce a live positive price. "
                    "IBKR reported delayed/unsubscribed data if error 354 is present."
                ),
                "ibkr_connected": True,
                "ibkr_port": port,
                "connection_attempts": attempts,
                "ibkr_errors": error_log.events[-10:],
                "subscription_errors": error_log.subscription_errors[-10:],
                "spx_probe": {
                    "contract": str(spx_contract),
                    "market_price": None,
                    "market_data_type": market_data_type,
                },
            }
        return {
            **base,
            "blocked_reason": "option_chain_capture_not_started",
            "detail": (
                "Connected to IBKR market data and probed SPX, but this harness does not emit fake option rows. "
                "Run ibkr-live-capture to subscribe to a small SPXW no-order universe and emit live shadow rows."
            ),
            "ibkr_connected": True,
            "ibkr_port": port,
            "connection_attempts": attempts,
            "ibkr_errors": error_log.events[-10:],
            "subscription_errors": error_log.subscription_errors[-10:],
            "spx_probe": {
                "has_ticker": spx_ticker is not None,
                "market_price": spx_price,
                "market_data_type": market_data_type,
            },
        }
    except Exception as exc:
        return {
            **base,
            "blocked_reason": "ibkr_connection_or_market_data_failed",
            "detail": str(exc),
            "ibkr_connected": False,
            "ibkr_errors": error_log.events[-10:],
        }
    finally:
        if ib is not None and ib.isConnected():
            ib.disconnect()


def _run_ibkr_live_capture(args: argparse.Namespace) -> dict[str, Any]:
    router = _load_router_artifacts(args.stack_manifest, args.protocol081_manifest)
    artifact = load_protocol066_artifact(args.protocol081_manifest)
    now = datetime.now(tz=_NY)
    live_log = args.out_dir / "live_router_shadow_observations.jsonl"
    _write_jsonl(live_log, [])
    base = {
        "mode": "ibkr-live-capture",
        "decision": "blocked",
        "stack_manifest": str(router.stack_manifest),
        "protocol081_manifest": str(router.protocol081_manifest),
        "shadow_log": str(live_log),
        "captured_rows": 0,
        "no_order_guarantee": {
            "broker_order_endpoint_called": False,
            "order_intent_non_null_rows": 0,
        },
        "market_time": now.isoformat(),
        "regular_market_hours": _is_regular_market_hours(now),
    }
    if not _is_regular_market_hours(now):
        return {
            **base,
            "blocked_reason": "outside_regular_market_hours",
            "detail": "Live SPXW no-order capture must run during 09:30-16:00 New York time.",
        }
    try:
        from ib_insync import IB, Index, Option  # type: ignore
    except ImportError:
        return {**base, "blocked_reason": "missing_ib_insync"}

    ib = None
    error_log = IbkrErrorLog()
    subscribed: list[Any] = []
    try:
        ib, port, attempts = _connect_ibkr(args, IB)
        if ib is None:
            return {
                **base,
                "blocked_reason": "ibkr_connection_failed",
                "detail": "No configured IBKR API port accepted a connection.",
                "ibkr_connected": False,
                "connection_attempts": attempts,
            }
        ib.errorEvent += error_log.handler
        ib.reqMarketDataType(3 if args.allow_delayed_market_data else 1)

        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
        subscribed.extend([spx_contract, vix_contract])
        spx_price = _wait_for_price(ib, spx_ticker, seconds=8.0)
        vix_value = _wait_for_price(ib, vix_ticker, seconds=6.0)
        spx_market_data_type = _market_data_type_name(_ticker_market_data_type(spx_ticker))
        vix_market_data_type = _market_data_type_name(_ticker_market_data_type(vix_ticker))
        market_data_type = (
            spx_market_data_type
            if spx_market_data_type == vix_market_data_type
            else f"spx={spx_market_data_type},vix={vix_market_data_type}"
        )
        if spx_price is None or vix_value is None:
            return {
                **base,
                "blocked_reason": "ibkr_market_data_subscription_missing",
                "detail": (
                    "Connected to IBKR, but SPX/VIX did not both produce positive live prices. "
                    "This blocks live shadow parity; delayed data is not promotion-grade."
                ),
                "ibkr_connected": True,
                "ibkr_port": port,
                "connection_attempts": attempts,
                "ibkr_errors": error_log.events[-20:],
                "subscription_errors": error_log.subscription_errors[-20:],
                "feed_probe": {
                    "spx_price": spx_price,
                    "vix_value": vix_value,
                    "spx_market_data_type": spx_market_data_type,
                    "vix_market_data_type": vix_market_data_type,
                },
            }

        contracts, chain_meta = _discover_spxw_0dte_contracts(
            ib=ib,
            option_cls=Option,
            spx_contract=spx_contract,
            spx_price=spx_price,
            now=now,
            strikes_around_atm=args.live_strikes_around_atm,
        )
        if not contracts:
            return {
                **base,
                "blocked_reason": chain_meta.get("blocked_reason", "no_spxw_0dte_contracts"),
                "detail": "IBKR connected and index quotes were available, but no PM-settled SPXW 0DTE contracts were qualified.",
                "ibkr_connected": True,
                "ibkr_port": port,
                "connection_attempts": attempts,
                "ibkr_errors": error_log.events[-20:],
                "subscription_errors": error_log.subscription_errors[-20:],
                "chain_meta": chain_meta,
                "feed_probe": {
                    "spx_price": spx_price,
                    "vix_value": vix_value,
                    "market_data_type": market_data_type,
                },
            }

        option_tickers: list[tuple[Any, Any]] = []
        for contract in contracts:
            ticker = ib.reqMktData(contract, "", False, False)
            option_tickers.append((contract, ticker))
            subscribed.append(contract)

        shadow_rows: list[dict[str, Any]] = []
        greek_failures = 0
        quote_misses = 0
        observed_market_data_types: dict[str, int] = {}
        deadline = time.monotonic() + max(args.live_capture_seconds, args.live_sample_interval_seconds)
        while time.monotonic() < deadline and len(shadow_rows) < args.max_shadow_rows:
            ib.sleep(max(0.5, args.live_sample_interval_seconds))
            capture_now = datetime.now(tz=_UTC)
            current_spx = _ticker_price(spx_ticker)
            current_vix = _ticker_price(vix_ticker)
            if current_spx is None or current_vix is None:
                quote_misses += len(option_tickers)
                continue
            atm_strike = _round_to_5(current_spx)
            for contract, ticker in option_tickers:
                type_name = _market_data_type_name(_ticker_market_data_type(ticker))
                observed_market_data_types[type_name] = observed_market_data_types.get(type_name, 0) + 1
                quote = _option_quote(ticker)
                if quote is None:
                    quote_misses += 1
                    continue
                row = _live_feature_row(
                    contract=contract,
                    quote=quote,
                    spx_price=current_spx,
                    now=capture_now,
                    atm_strike=atm_strike,
                    feature_columns=artifact.feature_columns,
                )
                if row is None:
                    greek_failures += 1
                    continue
                prepared = pd.DataFrame([row])
                value, recovery, decay = predict_protocol066_sequence(artifact, prepared)
                prediction = prediction_for_step(
                    step_index=0,
                    value=value,
                    recovery=recovery,
                    decay=decay,
                    override_threshold=artifact.selected_override_threshold,
                    step_row=row,
                )
                shadow_rows.append(
                    _live_shadow_observation(
                        protocol_id=args.protocol_id,
                        router=router,
                        artifact_ref=artifact.artifact_ref,
                        row=row,
                        prediction=prediction,
                        spx_price=current_spx,
                        vix_value=current_vix,
                        feature_columns=artifact.feature_columns,
                        market_data_type=type_name if type_name != "unknown" else market_data_type,
                    )
                )

        _write_jsonl(live_log, shadow_rows)
        config = ShadowParityConfig(
            protocol_id=args.protocol_id,
            max_quote_age_ms=args.max_quote_age_ms,
            max_context_age_ms=args.max_context_age_ms,
        )
        parity = summarize_shadow_parity(shadow_rows, config=config)
        subscription_errors = error_log.subscription_errors
        delayed_types = {name: count for name, count in observed_market_data_types.items() if "delayed" in name}
        live_ok = not delayed_types and not subscription_errors and parity["status"] in {"pass", "warn"}
        row_count_ok = len(shadow_rows) >= args.min_live_shadow_rows
        decision = "pass" if live_ok and row_count_ok else "blocked"
        blocked_reason = None
        if not row_count_ok:
            blocked_reason = "no_live_option_nbbo_rows"
        elif delayed_types:
            blocked_reason = "delayed_market_data_not_promotion_grade"
        elif subscription_errors:
            blocked_reason = "ibkr_market_data_subscription_missing"
        elif parity["status"] not in {"pass", "warn"}:
            blocked_reason = "shadow_parity_failed"
        return {
            **base,
            "decision": decision,
            "blocked_reason": blocked_reason,
            "detail": (
                "Captured fresh live SPXW no-order shadow rows and passed parity."
                if decision == "pass"
                else "Live shadow capture did not clear the gate; see blocker, IBKR errors, and parity rows."
            ),
            "ibkr_connected": True,
            "ibkr_port": port,
            "connection_attempts": attempts,
            "captured_rows": len(shadow_rows),
            "shadow_parity": parity,
            "ibkr_errors": error_log.events[-30:],
            "subscription_errors": subscription_errors[-30:],
            "chain_meta": chain_meta,
            "feed_probe": {
                "spx_price": spx_price,
                "vix_value": vix_value,
                "spx_market_data_type": spx_market_data_type,
                "vix_market_data_type": vix_market_data_type,
                "observed_option_market_data_types": observed_market_data_types,
            },
            "capture_diagnostics": {
                "requested_contracts": len(contracts),
                "subscribed_market_data_lines": len(subscribed),
                "quote_misses": quote_misses,
                "greek_compute_failures": greek_failures,
                "delayed_option_type_counts": delayed_types,
            },
        }
    except Exception as exc:
        return {
            **base,
            "blocked_reason": "ibkr_live_capture_failed",
            "detail": str(exc),
            "ibkr_connected": bool(ib is not None and ib.isConnected()),
            "ibkr_errors": error_log.events[-30:],
        }
    finally:
        if ib is not None and ib.isConnected():
            for contract in subscribed:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
            ib.disconnect()


def _write_report(path: Path, payload: dict[str, Any], *, protocol_label: str) -> None:
    parity = payload.get("shadow_parity", {})
    lines = [
        f"# {protocol_label} No-Order Live Shadow Router",
        "",
        "No broker order endpoint is called by this harness.",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Shadow log: `{payload.get('shadow_log')}`",
        f"- Stack manifest: `{payload.get('stack_manifest')}`",
        f"- Protocol 081 manifest: `{payload.get('protocol081_manifest')}`",
        "",
    ]
    if payload.get("blocked_reason"):
        lines += [
            "## Blocker",
            "",
            f"- Blocked reason: `{payload.get('blocked_reason')}`",
            f"- Detail: `{payload.get('detail')}`",
            "",
        ]
    if payload.get("feed_probe"):
        lines += [
            "## Feed Probe",
            "",
            "```json",
            json.dumps(_json_sanitize(payload["feed_probe"]), indent=2, sort_keys=True),
            "```",
            "",
        ]
    if parity:
        lines += [
            "## Shadow Parity",
            "",
            f"- Status: `{parity['status']}`",
            f"- Rows: `{parity['rows']}`",
            f"- Passed rows: `{parity['passed_rows']}`",
            f"- Failed rows: `{parity['failed_rows']}`",
            f"- Action counts: `{parity['action_counts']}`",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for check in parity["checks"]:
            lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    else:
        lines += [
            "## Live Probe",
            "",
            f"- Blocked reason: `{payload.get('blocked_reason')}`",
            f"- Detail: `{payload.get('detail')}`",
            f"- Regular market hours: `{payload.get('regular_market_hours')}`",
            f"- Captured rows: `{payload.get('captured_rows')}`",
        ]
    lines += [
        "",
        "## Order Safety",
        "",
        "```json",
        json.dumps(_json_sanitize(payload["no_order_guarantee"]), indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "offline-smoke":
        payload = _run_offline_smoke(args)
    elif args.mode == "ibkr-live-probe":
        payload = _run_ibkr_live_probe(args)
    else:
        payload = _run_ibkr_live_capture(args)
    payload["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.out_dir / f"{args.mode}_summary.json").write_text(
        json.dumps(_json_sanitize(payload), indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    _write_report(args.out_dir / f"{args.mode}_report.md", payload, protocol_label=args.protocol_label)
    print(args.out_dir / f"{args.mode}_report.md")
    print(json.dumps(_json_sanitize({
        "mode": payload["mode"],
        "decision": payload["decision"],
        "rows": payload.get("shadow_parity", {}).get("rows", payload.get("captured_rows", 0)),
        "blocked_reason": payload.get("blocked_reason"),
    }), sort_keys=True))
    return 0 if payload["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
