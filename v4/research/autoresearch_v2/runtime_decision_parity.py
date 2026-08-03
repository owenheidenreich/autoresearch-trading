"""Offline decision-parity gate for the frozen autoresearch-v2 entry model.

The module is intentionally incapable of opening a network connection or
submitting an order.  It replays development-only recorded Databento OPRA
CBBO-1m files and recorded ThetaData index bars through a source-neutral
live-twin builder, then compares that path with the exact feature path used to
fit the frozen model.

The live clock is strict: a ThetaData minute whose ``event_time`` labels the
bar open becomes available at ``event_time + 60 seconds``.  Databento CBBO-1m
``ts_recv`` labels the completed interval boundary and is eligible at that
boundary.  Both sources are same-session only and have a 90-second age cap.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, time
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn import __version__ as sklearn_version

from v4.greeks.repair import compute_repaired_greeks

from .cache import stable_hash
from .corrected_v3_foundation import CORRECTED_ROOT
from .dataset import (
    ROOT,
    SIGNED17,
    _flatten_session,
    rolling_folds,
    sha256_path,
    verify_foundation,
)
from .timing_policy_experiment import _block, _derived


SCHEMA_VERSION = "autoresearch_v2.runtime_decision_parity.v1"
POLICY_ID = "signed18_model_side_nearest"
EXPECTED_MODEL_SHA256 = (
    "c5d0115ba187ae8f443ac57ea813fa74cc689c03bbf546cb72d8262bc65f463e"
)
MODEL_RELATIVE_PATH = Path(
    "v4/audit/autoresearch/"
    "autoresearch_v2_entry_model_confirmation_2026_08_02_attempt001/"
    "model/entry_model.pkl"
)
FREEZE_RELATIVE_PATH = MODEL_RELATIVE_PATH.parent.parent / "pre_holdout_model_freeze.json"
ACCESS_RECEIPT_RELATIVE_PATH = MODEL_RELATIVE_PATH.parent.parent / "holdout_access_receipt.json"
CONFIRMATION_RESULT_RELATIVE_PATH = MODEL_RELATIVE_PATH.parent.parent / "confirmation_result.json"
FOUNDATION_RELATIVE_PATH = Path(
    "v4/research/autoresearch_v2/foundations/"
    "development_2025-08-01_2026-06-09_corrected_v3_two_clock.json"
)
WINNER_HYPOTHESIS_RELATIVE_PATH = Path(
    "v4/research/autoresearch_v2/"
    "hypotheses_corrected_v3_signal_policy_2026_08_02/"
    "entry_signal_policy_signed18_model_side_nearest_short25_v1.json"
)

FEATURE_ATOL = 1e-12
FEATURE_RTOL = 0.0
SOURCE_MAX_AGE_NS = 90_000_000_000
MINUTE_NS = 60_000_000_000
STRIKE_STEP = 5
LADDER_DOLLARS = 50
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

EXPECTED_FEATURES = (
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped",
    "vwap_side_align",
    "omar_side_align",
    "momentum15_side_align",
    "straddle_mid_spot_bps",
    "put_call_mid_ratio",
    "side_smile_slope",
    "bs_delta",
    "bs_gamma",
    "is_call",
)

if tuple(SIGNED17) != EXPECTED_FEATURES:
    raise RuntimeError("frozen signed-18 feature tuple drifted")


@dataclass(frozen=True)
class Quote:
    symbol: str
    instrument_id: int
    available_at_ns: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def mid(self) -> float:
        return float((self.bid + self.ask) / 2.0)


@dataclass(frozen=True)
class ContractDefinition:
    symbol: str
    instrument_id: int
    strike: float
    right: str
    settlement_time_ns: int

    @property
    def contract_id(self) -> str:
        expiry = pd.Timestamp(self.settlement_time_ns, unit="ns", tz="UTC").tz_convert(NY)
        return f"SPXW-{expiry:%Y%m%d}-{self.strike:09.3f}-{self.right}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = stable_hash(result)
    return result


def _verify_self_hash(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    semantic = dict(payload)
    observed = semantic.pop(field, None)
    if observed != stable_hash(semantic):
        raise RuntimeError(f"self-hash drift:{path}:{field}")
    return payload


def assert_frozen_binding() -> dict[str, Any]:
    """Assert immutable bytes before any pickle or development row is decoded."""

    model_path = ROOT / MODEL_RELATIVE_PATH
    observed = sha256_path(model_path)
    if observed != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"FROZEN MODEL DRIFT: expected {EXPECTED_MODEL_SHA256}, observed {observed}"
        )
    freeze = _verify_self_hash(ROOT / FREEZE_RELATIVE_PATH, "freeze_sha256")
    if (
        freeze.get("status") != "MODEL_FROZEN_BEFORE_HOLDOUT_OPEN"
        or freeze.get("winner") != POLICY_ID
        or freeze.get("model_path") != str(MODEL_RELATIVE_PATH)
        or freeze.get("model_sha256") != EXPECTED_MODEL_SHA256
        or tuple(freeze.get("features", ())) != EXPECTED_FEATURES
        or freeze.get("sklearn_version") != sklearn_version
    ):
        raise RuntimeError("frozen model metadata drifted")
    access = _verify_self_hash(ROOT / ACCESS_RECEIPT_RELATIVE_PATH, "receipt_sha256")
    confirmation = json.loads((ROOT / CONFIRMATION_RESULT_RELATIVE_PATH).read_text())
    if (
        access.get("status") != "ACCESS_COMPLETE"
        or access.get("holdout_open_count") != 1
        or access.get("model_sha256") != EXPECTED_MODEL_SHA256
        or confirmation.get("status") != "CONFIRMED_EDGE"
    ):
        raise RuntimeError("confirmation/access binding drifted")
    return {
        "model_path": str(MODEL_RELATIVE_PATH),
        "model_sha256": observed,
        "freeze_path": str(FREEZE_RELATIVE_PATH),
        "freeze_sha256": freeze["freeze_sha256"],
        "winner": freeze["winner"],
        "features": list(EXPECTED_FEATURES),
        "training_rows": int(freeze["training_rows"]),
        "training_sessions": int(freeze["training_sessions"]),
        "sklearn_version": sklearn_version,
        "protected_holdout_status": "SPENT_AND_SEALED_NOT_OPENED_BY_THIS_GATE",
        "prior_holdout_open_count": 1,
    }


def live_decision_feature_contract() -> dict[str, Any]:
    context = {
        "source_plane": "THETADATA_LIVE_OFFICIAL_SPX_1M",
        "historical_twin": "vendor/thetadata/index/spx_1m/{session}.parquet",
        "causal_availability_clock": (
            "bar-open event_time + 60 seconds; latest completed same-session bar "
            "with available_at<=decision_time and age<=90 seconds"
        ),
        "carry": "same session only; no forward observation and no cross-session carry",
    }
    ladder = {
        "source_plane": "DATABENTO_LIVE_OPRA_CBBO_1M_COMPLETED",
        "historical_twin": "raw/databento/opra_spxw_cbbo_1m/{session}.cbbo-1m.parquet",
        "causal_availability_clock": (
            "CBBO-1m ts_recv is the completed interval boundary; latest exact-contract "
            "quote with ts_recv<=decision_time and age<=90 seconds"
        ),
        "carry": "same session and exact contract only; no cross-session carry",
    }
    greek = {
        "source_plane": "SELF_COMPUTED_FROM_DATABENTO_CBBO_AND_THETADATA_SPX",
        "historical_twin": "v4.greeks.repair.compute_repaired_greeks",
        "causal_availability_clock": "after both causal quote and completed SPX inputs pass",
        "carry": "none; recompute from current causal inputs with r=0.05 and q=0.0",
    }
    structural = {
        "source_plane": "DATABENTO_LIVE_OPRA_DEFINITION",
        "historical_twin": "raw/databento/opra_spxw_definition/{session}.definition.parquet",
        "causal_availability_clock": "definition known before the decision; SPXW 0DTE PM only",
        "carry": "contract-scoped static metadata for the same session",
    }
    formulas = {
        "spx_vwap_gap_points": "spx_close - cumulative_spx_vwap",
        "spx_vwap_gap_bps": "gap / spx_close * 10000",
        "spx_vwap_gap_over_session_range": "gap / session_range, else 0",
        "session_range_bps": "session_range / spx_close * 10000",
        "momentum_5m_bps": "(close - close[t-5]) / spx_close * 10000",
        "momentum_15m_bps": "(close - close[t-15]) / spx_close * 10000",
        "momentum_5m_over_session_range": "momentum_5m / session_range, else 0",
        "momentum_15m_over_session_range": "momentum_15m / session_range, else 0",
        "omar_clipped": "clip((close-first_close)/session_range, -3, 3)",
        "vwap_side_align": "1 iff gap has the candidate side sign",
        "omar_side_align": "1 iff OMAR has the candidate side sign",
        "momentum15_side_align": "1 iff 15m momentum has the candidate side sign",
        "straddle_mid_spot_bps": "(ATM call mid + ATM put mid) / SPX * 10000",
        "put_call_mid_ratio": "ATM put mid / ATM call mid",
        "side_smile_slope": "(call mid at +5 - call mid at -5) / SPX * 10000",
        "bs_delta": "repaired Black-Scholes delta from causal BBO mid/ask/bid",
        "bs_gamma": "repaired Black-Scholes gamma from causal BBO mid/ask/bid",
        "is_call": "1 for OPRA call definition, else 0",
    }
    rows = []
    for index, name in enumerate(EXPECTED_FEATURES, start=1):
        source = context if index <= 12 else ladder if index <= 15 else greek if index <= 17 else structural
        rows.append({"index": index, "feature": name, "formula": formulas[name], **source})
    return _self_hashed(
        {
            "schema_version": "autoresearch_v2.live_decision_feature_contract.v1",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "policy_id": POLICY_ID,
            "ordered_feature_count": len(rows),
            "features": rows,
            "float_comparison": {"atol": FEATURE_ATOL, "rtol": FEATURE_RTOL},
            "vix_context_disposition": (
                "ThetaData VIX may coexist on the context plane but is not consumed by any "
                "of the frozen 18 columns; changing VIX cannot change model input or score."
            ),
            "ibkr_decision_features": False,
            "live_network_required": False,
        },
        "contract_sha256",
    )


def _paths(session: str, training_path: Path) -> dict[str, Path]:
    raw = CORRECTED_ROOT / "raw/databento"
    vendor = CORRECTED_ROOT / "vendor/thetadata/index"
    return {
        "training": training_path,
        "cbbo": raw / "opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet",
        "definition": raw / "opra_spxw_definition" / f"{session}.definition.parquet",
        "spx": vendor / "spx_1m" / f"{session}.parquet",
        "vix": vendor / "vix_1m" / f"{session}.parquet",
    }


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"recorded source absent or unsafe:{path}")
    return pq.read_table(path).to_pandas()


def _definition_map(frame: pd.DataFrame, session: str) -> dict[tuple[int, str], ContractDefinition]:
    required = {"symbol", "instrument_id", "instrument_class", "strike_price", "expiration", "asset"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Databento definition schema drift")
    expiry = pd.to_datetime(frame["expiration"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    subset = frame[(expiry == session) & (frame["asset"].astype(str) == "SPXW")].copy()
    subset = subset[subset["instrument_class"].astype(str).isin(("C", "P"))]
    subset = subset.sort_values(["symbol", "ts_event"], na_position="first").groupby("symbol").tail(1)
    local_date = pd.Timestamp(session).date()
    settlement = datetime.combine(local_date, time(16, 0), tzinfo=NY).astimezone(UTC)
    settlement_ns = int(pd.Timestamp(settlement).value)
    result: dict[tuple[int, str], ContractDefinition] = {}
    for row in subset.to_dict("records"):
        strike = float(row["strike_price"])
        right = str(row["instrument_class"])
        if not math.isfinite(strike) or not np.isclose(strike % STRIKE_STEP, 0.0):
            continue
        key = (int(round(strike)), right)
        item = ContractDefinition(
            symbol=str(row["symbol"]),
            instrument_id=int(row["instrument_id"]),
            strike=strike,
            right=right,
            settlement_time_ns=settlement_ns,
        )
        if key in result and result[key] != item:
            raise RuntimeError(f"duplicate 0DTE definition:{session}:{key}")
        result[key] = item
    if not result:
        raise RuntimeError(f"no Databento SPXW 0DTE definitions:{session}")
    return result


def _quote_paths(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, tuple[Quote, ...]]]:
    if frame.index.name == "ts_recv":
        frame = frame.reset_index()
    required = {
        "ts_recv", "symbol", "instrument_id", "bid_px_00", "ask_px_00",
        "bid_sz_00", "ask_sz_00",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Databento CBBO-1m schema drift")
    frame = frame.sort_values(["symbol", "ts_recv", "instrument_id"])
    result = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        quotes = tuple(
            Quote(
                symbol=str(symbol),
                instrument_id=int(row.instrument_id),
                available_at_ns=int(pd.Timestamp(row.ts_recv).value),
                bid=float(row.bid_px_00),
                ask=float(row.ask_px_00),
                bid_size=int(row.bid_sz_00),
                ask_size=int(row.ask_sz_00),
            )
            for row in group.itertuples(index=False)
        )
        clocks = np.asarray([item.available_at_ns for item in quotes], dtype=np.int64)
        if len(clocks) != len(np.unique(clocks)) or np.any(np.diff(clocks) <= 0):
            raise RuntimeError(f"duplicate or unordered CBBO minute:{symbol}")
        result[str(symbol)] = (clocks, quotes)
    return result


def _latest_quote(
    definition: ContractDefinition,
    paths: Mapping[str, tuple[np.ndarray, tuple[Quote, ...]]],
    decision_time_ns: int,
) -> Quote | None:
    item = paths.get(definition.symbol)
    if item is None:
        return None
    clocks, quotes = item
    index = int(np.searchsorted(clocks, decision_time_ns, side="right") - 1)
    if index < 0:
        return None
    quote = quotes[index]
    age = decision_time_ns - quote.available_at_ns
    if age < 0 or age > SOURCE_MAX_AGE_NS:
        return None
    if quote.instrument_id != definition.instrument_id:
        raise RuntimeError(f"definition/CBBO instrument mismatch:{definition.symbol}")
    return quote


def _validate_index(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    required = {
        "event_time", "symbol", "close", "volume", "context_source",
        "is_derived", "is_proxy", "is_official_index_data",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"ThetaData {symbol} schema drift")
    out = frame.copy()
    if not (
        (out["symbol"].astype(str) == symbol).all()
        and (out["context_source"].astype(str) == "thetadata_index_history_ohlc").all()
        and (~out["is_derived"].astype(bool)).all()
        and (~out["is_proxy"].astype(bool)).all()
        and out["is_official_index_data"].astype(bool).all()
    ):
        raise RuntimeError(f"ThetaData {symbol} official-row invariants drift")
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    return out.sort_values("event_time").reset_index(drop=True)


def _context_at(
    spx: pd.DataFrame,
    decision_time_ns: int,
    *,
    completion_lag_seconds: int,
) -> dict[str, Any] | None:
    # ThetaData parquet is timestamp[us]; normalizing explicitly to datetime64[ns]
    # avoids accidentally treating its integer representation as nanoseconds.
    event_ns = (
        spx["event_time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    available = event_ns + int(completion_lag_seconds * 1_000_000_000)
    index = int(np.searchsorted(available, decision_time_ns, side="right") - 1)
    if index < 0:
        return None
    age = decision_time_ns - int(available[index])
    if age < 0 or age > SOURCE_MAX_AGE_NS or index + 1 < 30:
        return None
    closes = spx["close"].astype(float).to_numpy()[: index + 1]
    volumes = spx["volume"].astype(np.int64).to_numpy()[: index + 1]
    if not np.isfinite(closes).all() or np.any(closes <= 0.0) or np.any(volumes < 0):
        raise RuntimeError("ThetaData SPX close/volume invalid")
    total_volume = int(volumes.sum(dtype=np.int64))
    vwap = (
        float(np.dot(closes, volumes.astype(float)) / total_volume)
        if total_volume > 0
        else float(np.mean(closes))
    )
    close = float(closes[-1])
    session_range = float(np.max(closes) - np.min(closes))
    omar = 0.0 if session_range == 0.0 else (close - float(closes[0])) / session_range
    momentum5 = close - float(closes[-6]) if len(closes) >= 6 else 0.0
    momentum15 = close - float(closes[-16]) if len(closes) >= 16 else 0.0
    return {
        "spx": close,
        "vwap": vwap,
        "omar": omar,
        "session_range": session_range,
        "momentum5": momentum5,
        "momentum15": momentum15,
        "event_time_ns": int(event_ns[index]),
        "available_at_ns": int(available[index]),
        "age_ns": int(age),
    }


def _tradable(quote: Quote) -> bool:
    values = (quote.bid, quote.ask, quote.mid)
    if not np.isfinite(values).all():
        return False
    spread = quote.ask - quote.bid
    return bool(
        quote.bid >= 0.0
        and quote.ask > 0.0
        and quote.ask >= quote.bid
        and 0.50 <= quote.mid <= 35.0
        and spread <= 0.50
        and spread / quote.mid <= 0.25
        and quote.bid_size >= 1
        and quote.ask_size >= 1
    )


def _round_atm(value: float) -> int:
    return int(round(value / STRIKE_STEP) * STRIKE_STEP)


def _feature_rows_at(
    *,
    session: str,
    decision_time_ns: int,
    context: Mapping[str, Any],
    definitions: Mapping[tuple[int, str], ContractDefinition],
    quote_paths: Mapping[str, tuple[np.ndarray, tuple[Quote, ...]]],
) -> list[dict[str, Any]]:
    spx = float(context["spx"])
    atm = _round_atm(spx)
    offsets = tuple(range(-LADDER_DOLLARS, LADDER_DOLLARS + STRIKE_STEP, STRIKE_STEP))
    ladder: dict[tuple[int, str], tuple[ContractDefinition, Quote, Any]] = {}
    for offset in offsets:
        for right in ("C", "P"):
            definition = definitions.get((atm + offset, right))
            if definition is None:
                continue
            quote = _latest_quote(definition, quote_paths, decision_time_ns)
            if quote is None or not _tradable(quote):
                continue
            seconds = (definition.settlement_time_ns - decision_time_ns) / 1_000_000_000
            greek = compute_repaired_greeks(
                S=spx,
                K=definition.strike,
                T=seconds / (365.0 * 24.0 * 60.0 * 60.0),
                is_call=right == "C",
                mid=quote.mid,
                ask=quote.ask,
                bid=quote.bid,
                r=0.05,
                q=0.0,
            )
            # The corrected corpus' model-scoring mask rejects candidates whose
            # causal BBO cannot produce finite repaired Greeks, even though the
            # older metadata label says historical-default.
            if greek is None:
                continue
            ladder[(offset, right)] = (definition, quote, greek)
    atm_call = ladder.get((0, "C"))
    atm_put = ladder.get((0, "P"))
    straddle = (
        float((atm_call[1].mid + atm_put[1].mid) / spx * 1e4)
        if atm_call is not None and atm_put is not None
        else float("nan")
    )
    put_call = (
        float(atm_put[1].mid / atm_call[1].mid)
        if atm_call is not None and atm_put is not None and atm_call[1].mid > 0.0
        else float("nan")
    )
    plus = ladder.get((5, "C"))
    minus = ladder.get((-5, "C"))
    smile = (
        float((plus[1].mid - minus[1].mid) / spx * 1e4)
        if plus is not None and minus is not None
        else float("nan")
    )
    gap = spx - float(context["vwap"])
    session_range = float(context["session_range"])
    omar = float(context["omar"])
    momentum5 = float(context["momentum5"])
    momentum15 = float(context["momentum15"])
    rows = []
    for (offset, right), (definition, quote, greek) in sorted(ladder.items()):
        if not 3.0 <= quote.mid <= 8.0:
            continue
        direction = 1.0 if right == "C" else -1.0
        values = {
            "spx_vwap_gap_points": gap,
            "spx_vwap_gap_bps": gap / spx * 1e4,
            "spx_vwap_gap_over_session_range": gap / session_range if session_range else 0.0,
            "session_range_bps": session_range / spx * 1e4,
            "momentum_5m_bps": momentum5 / spx * 1e4,
            "momentum_15m_bps": momentum15 / spx * 1e4,
            "momentum_5m_over_session_range": momentum5 / session_range if session_range else 0.0,
            "momentum_15m_over_session_range": momentum15 / session_range if session_range else 0.0,
            "omar_clipped": float(np.clip(omar, -3.0, 3.0)),
            "vwap_side_align": float(gap * direction > 0.0),
            "omar_side_align": float(omar * direction > 0.0),
            "momentum15_side_align": float(momentum15 * direction > 0.0),
            "straddle_mid_spot_bps": straddle,
            "put_call_mid_ratio": put_call,
            "side_smile_slope": smile,
            "bs_delta": float(greek.delta) if greek is not None else float("nan"),
            "bs_gamma": float(greek.gamma) if greek is not None else float("nan"),
            "is_call": float(right == "C"),
        }
        rows.append(
            {
                "candidate_uid": f"{session}|{decision_time_ns}|{definition.contract_id}",
                "session": session,
                "decision_time_ns": decision_time_ns,
                "source_quote_time_ns": quote.available_at_ns,
                "source_context_time_ns": int(context["available_at_ns"]),
                "contract_id": definition.contract_id,
                "right": right,
                "abs_moneyness": abs(offset),
                "entry_bid": quote.bid,
                "entry_ask": quote.ask,
                "entry_mid": quote.mid,
                "opt_spread": quote.ask - quote.bid,
                **values,
            }
        )
    return rows


def build_live_twin_frame(
    *,
    session: str,
    decision_times_ns: Sequence[int],
    cbbo: pd.DataFrame,
    definitions: pd.DataFrame,
    spx: pd.DataFrame,
    vix: pd.DataFrame,
    completion_lag_seconds: int = 60,
) -> pd.DataFrame:
    """Build exact frozen-model rows from recorded live-path source schemas."""

    spx = _validate_index(spx, symbol="SPX")
    # VIX is schema-validated but intentionally never consumed by the 18 columns.
    _validate_index(vix, symbol="VIX")
    definitions_by_key = _definition_map(definitions, session)
    quotes_by_symbol = _quote_paths(cbbo)
    rows: list[dict[str, Any]] = []
    for decision_time_ns in sorted(set(int(value) for value in decision_times_ns)):
        context = _context_at(
            spx, decision_time_ns, completion_lag_seconds=completion_lag_seconds
        )
        if context is None:
            continue
        rows.extend(
            _feature_rows_at(
                session=session,
                decision_time_ns=decision_time_ns,
                context=context,
                definitions=definitions_by_key,
                quote_paths=quotes_by_symbol,
            )
        )
    columns = [
        "candidate_uid", "session", "decision_time_ns", "source_quote_time_ns",
        "source_context_time_ns", "contract_id", "right", "abs_moneyness",
        "entry_bid", "entry_ask", "entry_mid", "opt_spread", *EXPECTED_FEATURES,
    ]
    return pd.DataFrame(rows, columns=columns)


def _score_bits(value: float) -> str | None:
    if not math.isfinite(float(value)):
        return None
    bits = np.asarray([value], dtype=np.float64).view(np.uint64)[0]
    return f"{int(bits):016x}"


def _predict(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    result = _derived(frame)
    if result.empty:
        result["_score"] = pd.Series(dtype=float)
        return result
    result["_score"] = model.predict(result.loc[:, EXPECTED_FEATURES].to_numpy(float))
    return result


def decision_trace(
    scored: pd.DataFrame,
    *,
    session: str,
    decision_times_ns: Sequence[int],
) -> pd.DataFrame:
    """Apply the exact first-positive-per-block, model-side, nearest rule."""

    groups = {
        int(decision): group.sort_values(
            ["_score", "candidate_uid"], ascending=[False, True]
        )
        for decision, group in scored.groupby("decision_time_ns", sort=True)
    }
    consumed: set[int] = set()
    rows = []
    for decision in sorted(set(int(value) for value in decision_times_ns)):
        probe = pd.DataFrame({"decision_time_ns": [decision]})
        probe = _derived(probe)
        block = int(_block(probe)[0])
        if block < 0:
            continue
        group = groups.get(decision)
        top_score = float("nan")
        top_side = None
        top_uid = None
        action = "WAIT"
        selected_contract = None
        reason = "NO_ELIGIBLE_CANDIDATE"
        if group is not None and not group.empty:
            top = group.iloc[0]
            top_score = float(top["_score"])
            top_side = str(top["right"])
            top_uid = str(top["candidate_uid"])
            if block in consumed:
                reason = "BLOCK_ALREADY_CONSUMED"
            elif top_score > 0.0:
                action = "ENTER"
                reason = "FIRST_POSITIVE_SCORE_IN_BLOCK"
                pool = group[group["right"].astype(str) == top_side].sort_values(
                    ["abs_moneyness", "candidate_uid"]
                )
                selected_contract = str(pool.iloc[0]["contract_id"])
                consumed.add(block)
            else:
                reason = "TOP_SCORE_NOT_POSITIVE"
        rows.append(
            {
                "session": session,
                "decision_time_ns": decision,
                "block": block,
                "score": top_score,
                "score_bits": _score_bits(top_score),
                "action": action,
                "side": top_side if action == "ENTER" else None,
                "selected_contract": selected_contract,
                "top_candidate_uid": top_uid,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _candidate_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["candidate_uid"].astype(str)


def compare_features(training: pd.DataFrame, live: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left = training.set_index("candidate_uid")
    right = live.set_index("candidate_uid")
    if not left.index.is_unique or not right.index.is_unique:
        raise RuntimeError("duplicate candidate identity in parity comparison")
    union = sorted(set(left.index) | set(right.index))
    matched = sorted(set(left.index) & set(right.index))
    rows = []
    total_matches = 0
    total_cells = len(union) * len(EXPECTED_FEATURES)
    for name in EXPECTED_FEATURES:
        exact = within = nonfinite_mismatch = 0
        max_abs = 0.0
        for key in matched:
            a = float(left.at[key, name])
            b = float(right.at[key, name])
            if math.isnan(a) and math.isnan(b):
                exact += 1
                within += 1
                continue
            if not math.isfinite(a) or not math.isfinite(b):
                nonfinite_mismatch += 1
                continue
            exact += int(_score_bits(a) == _score_bits(b))
            delta = abs(a - b)
            max_abs = max(max_abs, delta)
            within += int(np.isclose(a, b, atol=FEATURE_ATOL, rtol=FEATURE_RTOL))
        total_matches += within
        rows.append(
            {
                "feature": name,
                "training_candidate_count": len(left),
                "live_candidate_count": len(right),
                "matched_candidate_count": len(matched),
                "missing_candidate_count": len(union) - len(matched),
                "exact_bit_match_count": exact,
                "within_tolerance_count": within,
                "within_tolerance_rate_on_matched": within / len(matched) if matched else 0.0,
                "nonfinite_mismatch_count": nonfinite_mismatch,
                "max_abs_divergence": max_abs,
                "atol": FEATURE_ATOL,
                "rtol": FEATURE_RTOL,
            }
        )
    return rows, {
        "training_candidate_count": len(left),
        "live_candidate_count": len(right),
        "candidate_union_count": len(union),
        "matched_candidate_count": len(matched),
        "candidate_set_identical": set(left.index) == set(right.index),
        "feature_cell_count": total_cells,
        "feature_cell_match_count": total_matches,
        "feature_within_tolerance_rate": total_matches / total_cells if total_cells else 0.0,
    }


def compare_scores(training: pd.DataFrame, live: pd.DataFrame) -> dict[str, Any]:
    left_frame = training.set_index("candidate_uid")
    right_frame = live.set_index("candidate_uid")
    if not left_frame.index.is_unique or not right_frame.index.is_unique:
        raise RuntimeError("duplicate candidate identity in score comparison")
    left = left_frame["_score"]
    right = right_frame["_score"]
    union = sorted(set(left.index) | set(right.index))
    matched = sorted(set(left.index) & set(right.index))
    exact = sum(_score_bits(float(left[key])) == _score_bits(float(right[key])) for key in matched)
    max_abs = max((abs(float(left[key]) - float(right[key])) for key in matched), default=0.0)
    return {
        "candidate_union_count": len(union),
        "matched_candidate_count": len(matched),
        "bit_identical_score_count": exact,
        "bit_identical_score_rate": exact / len(union) if union else 0.0,
        "max_abs_score_divergence": max_abs,
    }


def compare_decisions(training: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["session", "decision_time_ns", "block"]
    merged = training.merge(live, on=keys, how="outer", suffixes=("_training", "_live"), indicator=True)
    merged["score_match"] = merged.apply(
        lambda row: row["_merge"] == "both"
        and row.get("score_bits_training") == row.get("score_bits_live"), axis=1
    )
    for field in ("action", "side", "selected_contract"):
        left = merged[f"{field}_training"].fillna("<NONE>")
        right = merged[f"{field}_live"].fillna("<NONE>")
        merged[f"{field}_match"] = (merged["_merge"] == "both") & (left == right)
    match_fields = ["score_match", "action_match", "side_match", "selected_contract_match"]
    merged["decision_match"] = merged[match_fields].all(axis=1)
    count = len(merged)
    return merged, {
        "decision_count": count,
        "score_match_rate": float(merged["score_match"].mean()) if count else 0.0,
        "enter_wait_match_rate": float(merged["action_match"].mean()) if count else 0.0,
        "side_match_rate": float(merged["side_match"].mean()) if count else 0.0,
        "selected_contract_match_rate": float(merged["selected_contract_match"].mean()) if count else 0.0,
        "complete_decision_match_rate": float(merged["decision_match"].mean()) if count else 0.0,
        "complete_decision_match_count": int(merged["decision_match"].sum()),
    }


def block_policy_outcomes(trace: pd.DataFrame) -> pd.DataFrame:
    """Reduce the decision trace to the one frozen outcome per session/block."""

    rows = []
    for (session, block), group in trace.groupby(["session", "block"], sort=True):
        enters = group[group["action"] == "ENTER"]
        if len(enters) > 1:
            raise RuntimeError(f"more than one entry in fixed block:{session}:{block}")
        if enters.empty:
            rows.append(
                {
                    "session": str(session),
                    "block": int(block),
                    "action": "WAIT",
                    "signal_time_ns": None,
                    "score_bits": None,
                    "score": None,
                    "side": None,
                    "selected_contract": None,
                }
            )
        else:
            row = enters.iloc[0]
            rows.append(
                {
                    "session": str(session),
                    "block": int(block),
                    "action": "ENTER",
                    "signal_time_ns": int(row["decision_time_ns"]),
                    "score_bits": row["score_bits"],
                    "score": float(row["score"]),
                    "side": str(row["side"]),
                    "selected_contract": str(row["selected_contract"]),
                }
            )
    return pd.DataFrame(rows)


def compare_block_policies(
    training: pd.DataFrame, live: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["session", "block"]
    merged = training.merge(
        live, on=keys, how="outer", suffixes=("_training", "_live"), indicator=True
    )
    for field in (
        "action", "signal_time_ns", "score_bits", "side", "selected_contract"
    ):
        left = merged[f"{field}_training"].fillna("<NONE>").astype(str)
        right = merged[f"{field}_live"].fillna("<NONE>").astype(str)
        merged[f"{field}_match"] = (merged["_merge"] == "both") & (left == right)
    fields = [
        "action_match", "signal_time_ns_match", "score_bits_match", "side_match",
        "selected_contract_match",
    ]
    merged["block_policy_match"] = merged[fields].all(axis=1)
    count = len(merged)
    both_enter = (
        (merged["action_training"] == "ENTER") & (merged["action_live"] == "ENTER")
    )
    return merged, {
        "block_count": count,
        "action_match_rate": float(merged["action_match"].mean()) if count else 0.0,
        "signal_time_match_rate": float(merged["signal_time_ns_match"].mean()) if count else 0.0,
        "bit_identical_signal_score_rate": float(merged["score_bits_match"].mean()) if count else 0.0,
        "side_match_rate": float(merged["side_match"].mean()) if count else 0.0,
        "selected_contract_match_rate": float(merged["selected_contract_match"].mean()) if count else 0.0,
        "complete_block_policy_match_rate": float(merged["block_policy_match"].mean()) if count else 0.0,
        "complete_block_policy_match_count": int(merged["block_policy_match"].sum()),
        "both_paths_enter_count": int(both_enter.sum()),
        "same_signal_time_side_contract_count": int(
            (
                both_enter
                & merged["signal_time_ns_match"]
                & merged["side_match"]
                & merged["selected_contract_match"]
            ).sum()
        ),
    }


def _selected_sessions(foundation: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    session_paths = {
        str(item["session"]): ROOT / str(item["path"]) for item in foundation["sessions"]
    }
    folds = rolling_folds(session_paths)
    selected = []
    for fold in folds:
        tests = tuple(fold["outer_test"])
        for position, session in (("first", tests[0]), ("last", tests[-1])):
            selected.append({"fold": int(fold["fold"]), "position": position, "session": session})
    if len(selected) != 10 or len({item["session"] for item in selected}) != 10:
        raise RuntimeError("deterministic parity session selection drifted")
    return selected, session_paths


def audit_fitted_training_clock(
    foundation: Mapping[str, Any], *, expected_training_rows: int
) -> dict[str, Any]:
    """Reproduce the final-fit row count and test its declared context clock.

    This reads development foundation rows only.  It does not load labels or
    rows from the spent protected set.
    """

    hypothesis_path = ROOT / WINNER_HYPOTHESIS_RELATIVE_PATH
    hypothesis = json.loads(hypothesis_path.read_text())
    declared = {str(row["name"]): str(row["available_at"]) for row in hypothesis["features"]}
    if tuple(declared) != EXPECTED_FEATURES or set(declared.values()) != {
        "completed_minute_plus_60s"
    }:
        raise RuntimeError("winner hypothesis availability declaration drifted")
    totals = {
        "decoded_decision_rows": 0,
        "fit_candidate_rows": 0,
        "fit_candidate_rows_with_context_unavailable_at_decision": 0,
        "fit_candidate_rows_with_context_available_at_decision": 0,
        "same_timestamp_context_decision_rows": 0,
    }
    per_session = []
    lead_values = []
    for item in foundation["sessions"]:
        session = str(item["session"])
        processed = pd.read_pickle(ROOT / str(item["path"]))
        session_fit = session_violations = decision_rows = same_time = 0
        for entry in processed:
            if not entry.get("context_ready"):
                continue
            option_names = {name: index for index, name in enumerate(entry["feature_names"])}
            market_names = {
                name: index for index, name in enumerate(entry["market_feature_names"])
            }
            required_option = {"bid", "ask", "mid"}
            required_market = {
                "spx_close", "spx_vwap", "omar", "session_range",
                "momentum_5m", "momentum_15m",
            }
            if not required_option.issubset(option_names) or not required_market.issubset(market_names):
                raise RuntimeError(f"training clock audit feature axis drift:{session}")
            market = np.asarray(entry["market_window"], dtype=float)[-1]
            market_values = np.asarray(
                [market[market_names[name]] for name in sorted(required_market)], dtype=float
            )
            if not np.isfinite(market_values).all() or float(
                market[market_names["spx_close"]]
            ) <= 0.0:
                continue
            decision_ns = int(pd.Timestamp(entry["decision_time"]).value)
            source_context_ns = int(pd.Timestamp(entry["source_context_time"]).value)
            declared_available_ns = source_context_ns + MINUTE_NS
            lead_ns = declared_available_ns - decision_ns
            lead_values.append(lead_ns)
            decision_rows += 1
            same_time += int(source_context_ns == decision_ns)
            ladder = np.asarray(entry["option_ladder"], dtype=float)
            mask = np.asarray(entry["candidate_mask"], dtype=bool)
            labels = np.asarray(entry["labels_net_pnl"], dtype=float)[:, :, 1]
            bid = ladder[:, :, option_names["bid"]]
            ask = ladder[:, :, option_names["ask"]]
            mid = ladder[:, :, option_names["mid"]]
            fitted = (
                mask
                & np.isfinite(bid)
                & np.isfinite(ask)
                & np.isfinite(mid)
                & (ask > 0.0)
                & (mid >= 3.0)
                & (mid <= 8.0)
                & np.isfinite(labels)
            )
            count = int(fitted.sum())
            session_fit += count
            session_violations += count if declared_available_ns > decision_ns else 0
        totals["decoded_decision_rows"] += decision_rows
        totals["fit_candidate_rows"] += session_fit
        totals["fit_candidate_rows_with_context_unavailable_at_decision"] += session_violations
        totals["fit_candidate_rows_with_context_available_at_decision"] += session_fit - session_violations
        totals["same_timestamp_context_decision_rows"] += same_time
        per_session.append(
            {
                "session": session,
                "fit_candidate_rows": session_fit,
                "unavailable_context_rows": session_violations,
                "all_fit_rows_violate_declared_clock": session_fit > 0 and session_fit == session_violations,
            }
        )
    if totals["fit_candidate_rows"] != int(expected_training_rows):
        raise RuntimeError(
            "training clock audit did not reproduce frozen fit row count:"
            f"{totals['fit_candidate_rows']}!={expected_training_rows}"
        )
    if not lead_values:
        raise RuntimeError("training clock audit found no decoded decisions")
    affected_sessions = sum(item["all_fit_rows_violate_declared_clock"] for item in per_session)
    return _self_hashed(
        {
            "schema_version": "autoresearch_v2.fitted_training_clock_audit.v1",
            "foundation_sha256": foundation["foundation_sha256"],
            "hypothesis_path": str(WINNER_HYPOTHESIS_RELATIVE_PATH),
            "hypothesis_sha256": sha256_path(hypothesis_path),
            "declared_feature_availability": "completed_minute_plus_60s",
            "executable_training_selection": "source_context_time<=decision_time without adding the declared 60-second completion lag",
            "session_count": len(per_session),
            "affected_session_count": int(affected_sessions),
            "minimum_unavailability_lead_seconds": min(lead_values) / 1e9,
            "maximum_unavailability_lead_seconds": max(lead_values) / 1e9,
            **totals,
            "unavailable_context_fit_row_rate": (
                totals["fit_candidate_rows_with_context_unavailable_at_decision"]
                / totals["fit_candidate_rows"]
            ),
            "per_session": per_session,
            "holdout_open_count": 0,
        },
        "audit_sha256",
    )


def _source_receipt(session: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "session": session,
        "role": "development_non_holdout",
        "paths": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in paths.items()
        },
    }


def _residual_risks() -> list[dict[str, Any]]:
    return [
        {"check": "feed_timing", "future_shadow_proof": "record event, receive, adapter-available, and decision clocks; fail on late/out-of-order data"},
        {"check": "cross_feed_watermark", "future_shadow_proof": "prove ThetaData SPX and Databento OPRA observations were both complete before one decision watermark"},
        {"check": "quote_age", "future_shadow_proof": "enforce exact-contract same-session age<=90s and log stale/no-quote abstentions"},
        {"check": "cbbo_consolidation", "future_shadow_proof": "compare direct live CBBO-1m with historical CBBO-1m, or prove CBBO-1s/CMBP-1 aggregation boundary and last-update rule"},
        {"check": "duplicates_corrections_reconnect", "future_shadow_proof": "exercise duplicate, corrected, missing, reconnect, and sequence-gap messages without changing a sealed decision"},
        {"check": "definition_identity", "future_shadow_proof": "bind live instrument_id to SPXW 0DTE PM right/strike/expiry before ladder construction"},
        {"check": "candidate_filter_and_ties", "future_shadow_proof": "log every guard input and deterministic score/side/nearest-ATM tie break"},
        {"check": "runtime_numeric_environment", "future_shadow_proof": "pin Python/numpy/sklearn/model hashes and compare live-shadow feature/score bytes"},
        {"check": "scheduler_and_early_close", "future_shadow_proof": "prove ET block boundaries, holidays, early close, clock sync, and decision cadence"},
        {"check": "execution_parity_out_of_scope", "future_shadow_proof": "separate owner-authorized gate for IBKR discovery, orders, fills, cancels, and flattening"},
    ]


def run(*, output: Path) -> dict[str, Any]:
    binding = assert_frozen_binding()  # first and before every pickle decode
    if output.exists():
        raise FileExistsError(f"parity output already exists:{output}")
    foundation = verify_foundation(ROOT / FOUNDATION_RELATIVE_PATH)
    if foundation.get("role") != "development" or foundation.get("holdout_access_count") != 0:
        raise RuntimeError("parity replay foundation is not development-only")
    selected, training_paths = _selected_sessions(foundation)
    contract = live_decision_feature_contract()
    training_clock_audit = audit_fitted_training_clock(
        foundation, expected_training_rows=int(binding["training_rows"])
    )

    with (ROOT / MODEL_RELATIVE_PATH).open("rb") as handle:
        model = pickle.load(handle)

    feature_rows: list[dict[str, Any]] = []
    decision_comparisons = []
    block_comparisons = []
    session_results = []
    receipts = []
    diagnostic_summaries = []
    for selection in selected:
        session = str(selection["session"])
        paths = _paths(session, training_paths[session])
        if any("holdout" in str(path).lower() for path in paths.values()):
            raise RuntimeError("parity replay attempted a holdout-labelled path")
        receipts.append(_source_receipt(session, paths))
        training = _flatten_session(session, paths["training"])
        decision_times = sorted(training["decision_time_ns"].astype(np.int64).unique().tolist())
        cbbo = _read_parquet(paths["cbbo"])
        definitions = _read_parquet(paths["definition"])
        spx = _read_parquet(paths["spx"])
        vix = _read_parquet(paths["vix"])
        live = build_live_twin_frame(
            session=session,
            decision_times_ns=decision_times,
            cbbo=cbbo,
            definitions=definitions,
            spx=spx,
            vix=vix,
            completion_lag_seconds=60,
        )
        diagnostic = build_live_twin_frame(
            session=session,
            decision_times_ns=decision_times,
            cbbo=cbbo,
            definitions=definitions,
            spx=spx,
            vix=vix,
            completion_lag_seconds=0,
        )
        training_scored = _predict(model, training)
        live_scored = _predict(model, live)
        diagnostic_scored = _predict(model, diagnostic)
        per_feature, feature_summary = compare_features(training_scored, live_scored)
        score_summary = compare_scores(training_scored, live_scored)
        training_trace = decision_trace(
            training_scored, session=session, decision_times_ns=decision_times
        )
        live_trace = decision_trace(live_scored, session=session, decision_times_ns=decision_times)
        compared, decision_summary = compare_decisions(training_trace, live_trace)
        compared["fold"] = int(selection["fold"])
        decision_comparisons.append(compared)
        training_blocks = block_policy_outcomes(training_trace)
        live_blocks = block_policy_outcomes(live_trace)
        compared_blocks, block_summary = compare_block_policies(
            training_blocks, live_blocks
        )
        compared_blocks["fold"] = int(selection["fold"])
        block_comparisons.append(compared_blocks)
        for row in per_feature:
            feature_rows.append({"session": session, "fold": int(selection["fold"]), **row})
        diag_features, diag_feature_summary = compare_features(training_scored, diagnostic_scored)
        diag_scores = compare_scores(training_scored, diagnostic_scored)
        diag_trace = decision_trace(diagnostic_scored, session=session, decision_times_ns=decision_times)
        _, diag_decisions = compare_decisions(training_trace, diag_trace)
        _, diag_blocks = compare_block_policies(
            training_blocks, block_policy_outcomes(diag_trace)
        )
        diagnostic_summaries.append(
            {
                "session": session,
                "feature": diag_feature_summary,
                "score": diag_scores,
                "decision": diag_decisions,
                "block_policy": diag_blocks,
                "max_feature_divergence": max(row["max_abs_divergence"] for row in diag_features),
            }
        )
        session_results.append(
            {
                **selection,
                "feature": feature_summary,
                "score": score_summary,
                "decision": decision_summary,
                "block_policy": block_summary,
            }
        )

    feature_frame = pd.DataFrame(feature_rows)
    decisions = pd.concat(decision_comparisons, ignore_index=True)
    blocks = pd.concat(block_comparisons, ignore_index=True)
    feature_aggregate = []
    for name, group in feature_frame.groupby("feature", sort=False):
        feature_aggregate.append(
            {
                "feature": name,
                "max_abs_divergence": float(group["max_abs_divergence"].max()),
                "matched_candidate_count": int(group["matched_candidate_count"].sum()),
                "missing_candidate_count": int(group["missing_candidate_count"].sum()),
                "exact_bit_match_count": int(group["exact_bit_match_count"].sum()),
                "within_tolerance_count": int(group["within_tolerance_count"].sum()),
                "within_tolerance_rate_on_matched": float(
                    group["within_tolerance_count"].sum() / group["matched_candidate_count"].sum()
                ),
                "nonfinite_mismatch_count": int(group["nonfinite_mismatch_count"].sum()),
            }
        )
    candidate_union = sum(item["feature"]["candidate_union_count"] for item in session_results)
    feature_matches = sum(item["feature"]["feature_cell_match_count"] for item in session_results)
    feature_cells = sum(item["feature"]["feature_cell_count"] for item in session_results)
    score_matches = sum(item["score"]["bit_identical_score_count"] for item in session_results)
    decision_count = len(decisions)
    summary = {
        "candidate_set_identical": all(item["feature"]["candidate_set_identical"] for item in session_results),
        "feature_cell_match_count": feature_matches,
        "feature_cell_count": feature_cells,
        "feature_within_tolerance_rate": feature_matches / feature_cells if feature_cells else 0.0,
        "bit_identical_score_count": score_matches,
        "score_candidate_union_count": candidate_union,
        "bit_identical_score_rate": score_matches / candidate_union if candidate_union else 0.0,
        "decision_count": decision_count,
        "enter_wait_match_rate": float(decisions["action_match"].mean()),
        "side_match_rate": float(decisions["side_match"].mean()),
        "selected_contract_match_rate": float(decisions["selected_contract_match"].mean()),
        "complete_decision_match_rate": float(decisions["decision_match"].mean()),
        "complete_decision_match_count": int(decisions["decision_match"].sum()),
        "block_count": len(blocks),
        "block_enter_wait_match_rate": float(blocks["action_match"].mean()),
        "block_signal_time_match_rate": float(blocks["signal_time_ns_match"].mean()),
        "block_signal_score_match_rate": float(blocks["score_bits_match"].mean()),
        "block_side_match_rate": float(blocks["side_match"].mean()),
        "block_selected_contract_match_rate": float(blocks["selected_contract_match"].mean()),
        "complete_block_policy_match_rate": float(blocks["block_policy_match"].mean()),
        "complete_block_policy_match_count": int(blocks["block_policy_match"].sum()),
    }
    certified = bool(
        summary["candidate_set_identical"]
        and summary["feature_within_tolerance_rate"] == 1.0
        and summary["bit_identical_score_rate"] == 1.0
        and summary["complete_decision_match_rate"] == 1.0
        and summary["complete_block_policy_match_rate"] == 1.0
    )
    diagnostic_exact = all(
        item["feature"]["candidate_set_identical"]
        and item["feature"]["feature_within_tolerance_rate"] == 1.0
        and item["score"]["bit_identical_score_rate"] == 1.0
        and item["decision"]["complete_decision_match_rate"] == 1.0
        and item["block_policy"]["complete_block_policy_match_rate"] == 1.0
        for item in diagnostic_summaries
    )
    every_fitted_row_violates = bool(
        training_clock_audit["fit_candidate_rows"] == binding["training_rows"]
        and training_clock_audit[
            "fit_candidate_rows_with_context_unavailable_at_decision"
        ]
        == binding["training_rows"]
        and training_clock_audit["unavailable_context_fit_row_rate"] == 1.0
    )
    deferred_emission_assessment = {
        "candidate": "emit snapshot t at wall clock t+60s using OPRA quote t and ThetaData bar-open t close",
        "byte_parity_if_used": diagnostic_exact,
        "opra_quote_age_at_earliest_emission_seconds": 60,
        "spx_bar_age_at_earliest_emission_seconds": 0,
        "source_availability_causal_at_emission": True,
        "market_interval_coherent": False,
        "reason_market_interval_incoherent": (
            "OPRA CBBO-1m at t represents the completed interval ending at t, while "
            "ThetaData bar-open t represents the following minute and its close is "
            "available at t+60s"
        ),
        "confirmed_entry_quote_executable_at_emission": False,
        "same_frozen_trading_game": False,
        "verdict": "REJECTED_CROSSED_TIME_REPLAY_NOT_A_LIVE_DECISION_TWIN",
    }
    root_cause = {
        "classification": (
            "NONE_PARITY_CERTIFIED"
            if certified
            else "FROZEN_MODEL_TRAINED_ON_CONTEXT_UNAVAILABLE_AT_DECLARED_DECISION_CLOCK"
            if diagnostic_exact and every_fitted_row_violates
            else "TRAINING_CONTEXT_CLOCK_ONE_MINUTE_AHEAD_OF_LIVE_AVAILABILITY"
            if diagnostic_exact
            else "UNRESOLVED_MULTIPLE_PATH_DIVERGENCE"
        ),
        "training_behavior": (
            "the frozen training rows consume ThetaData SPX event_time at the same "
            "decision timestamp"
        ),
        "causal_live_behavior": (
            "ThetaData event_time is a bar-open label, so the same close becomes "
            "available only at event_time+60 seconds"
        ),
        "isolating_evidence": (
            "All ten sessions reproduce identical candidate sets, every feature cell, "
            "every model score, every decision, and every fixed-block outcome when the "
            "raw-source adapter is run with the training path's non-causal zero-second "
            "availability offset. Only the causal +60-second clock is changed in the "
            "certifying path."
        ),
        "downstream_effect": (
            "the prior completed SPX minute changes context, ATM geometry, Greek "
            "repair eligibility, model scores, signal time, side, and selected contract"
        ),
        "immutable_model_disposition": (
            "INVALID_EXPERIMENT_AND_NOT_RUNTIME_DECISION_PARITY_ELIGIBLE; every fitted "
            "row violates the winner hypothesis's declared completed-minute+60s clock; "
            "an adapter cannot repair this without changing the trading game, and this "
            "gate forbids refit or retuning"
            if not certified
            else "ELIGIBLE_FOR_SEPARATE_LIVE_SHADOW_GATE"
        ),
    }
    result = _self_hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "OFFLINE_DECISION_PARITY_CERTIFIED" if certified else "OFFLINE_DECISION_PARITY_FAILED",
            "claim": (
                "offline decision-parity certified for the frozen model on the "
                "Databento-live-OPRA feature path; live-shadow + execution parity remain separate gates."
                if certified
                else "offline decision-parity is not certified; the frozen training and causal live-twin paths diverge."
            ),
            "frozen_binding": binding,
            "foundation_path": str(FOUNDATION_RELATIVE_PATH),
            "foundation_sha256": foundation["foundation_sha256"],
            "session_selection_rule": "first and last outer-test session from each of five fixed rolling development folds",
            "session_count": len(selected),
            "sessions": session_results,
            "float_comparison": {"atol": FEATURE_ATOL, "rtol": FEATURE_RTOL},
            "aggregate": summary,
            "per_feature": feature_aggregate,
            "root_cause": root_cause,
            "fitted_training_clock_audit": training_clock_audit,
            "deferred_emission_assessment": deferred_emission_assessment,
            "diagnostic_training_clock_only_noncertifying": {
                "purpose": "isolates causal +60s ThetaData availability-clock effects; never a live contract",
                "all_ten_sessions_exact": diagnostic_exact,
                "sessions": diagnostic_summaries,
            },
            "residual_live_risk_checklist": _residual_risks(),
            "hard_stop_attestation": {
                "model_refit_or_retune": False,
                "protected_holdout_reopened": False,
                "holdout_open_count_this_gate": 0,
                "live_market_data_contacted": False,
                "broker_contacted": False,
                "paper_or_order_action": False,
                "paper_default_or_promotion_changed": False,
                "execution_or_fill_parity_claimed": False,
            },
        },
        "result_sha256",
    )

    output.mkdir(parents=True)
    _write_json(output / "live_decision_feature_contract.json", contract)
    _write_json(output / "fitted_training_clock_audit.json", training_clock_audit)
    _write_json(
        output / "session_selection.json",
        _self_hashed(
            {
                "schema_version": "autoresearch_v2.runtime_parity_session_selection.v1",
                "foundation_sha256": foundation["foundation_sha256"],
                "rule": "first_and_last_outer_test_session_per_fixed_fold",
                "sessions": selected,
                "protected_holdout_open_count": 0,
            },
            "selection_sha256",
        ),
    )
    _write_json(
        output / "source_receipts.json",
        _self_hashed(
            {
                "schema_version": "autoresearch_v2.runtime_parity_source_receipts.v1",
                "receipts": receipts,
                "network_access": False,
                "holdout_open_count": 0,
            },
            "receipts_sha256",
        ),
    )
    feature_frame.to_csv(output / "per_feature_divergence_by_session.csv", index=False)
    pd.DataFrame(feature_aggregate).to_csv(output / "per_feature_max_divergence.csv", index=False)
    decisions.to_csv(output / "per_decision_comparison.csv", index=False)
    blocks.to_csv(output / "per_block_policy_comparison.csv", index=False)
    _write_json(output / "parity_result.json", result)
    (output / "report.md").write_text(_report(result, contract))
    return result


def _report(result: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    agg = result["aggregate"]
    rows = [
        "# Frozen entry-model runtime decision-parity gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Model SHA-256: `{result['frozen_binding']['model_sha256']}`",
        f"Policy: `{POLICY_ID}`",
        f"Sessions: `{result['session_count']}` development-only sessions; holdout opens in this gate: `0`.",
        "",
        "## Offline parity result",
        "",
        f"- Candidate set identical: `{agg['candidate_set_identical']}`",
        f"- Feature cells within atol={FEATURE_ATOL}, rtol={FEATURE_RTOL}: `{agg['feature_within_tolerance_rate']:.9%}`",
        f"- Bit-identical candidate score rate: `{agg['bit_identical_score_rate']:.9%}`",
        f"- ENTER/WAIT match rate: `{agg['enter_wait_match_rate']:.9%}`",
        f"- Side match rate: `{agg['side_match_rate']:.9%}`",
        f"- Selected-contract match rate: `{agg['selected_contract_match_rate']:.9%}`",
        f"- Complete score/action/side/contract decision match: `{agg['complete_decision_match_rate']:.9%}`",
        f"- Complete fixed-block policy-outcome match: `{agg['complete_block_policy_match_rate']:.9%}`",
        f"- Fixed-block signal-time match: `{agg['block_signal_time_match_rate']:.9%}`",
        f"- Fixed-block selected-contract match: `{agg['block_selected_contract_match_rate']:.9%}`",
        "",
        "## Per-feature maximum divergence",
        "",
        "| Feature | Max absolute divergence | Within-tolerance rate | Missing candidates |",
        "|---|---:|---:|---:|",
    ]
    for item in result["per_feature"]:
        rows.append(
            f"| `{item['feature']}` | {item['max_abs_divergence']:.12g} | "
            f"{item['within_tolerance_rate_on_matched']:.9%} | {item['missing_candidate_count']} |"
        )
    rows.extend(
        [
            "",
            "## Root cause and disposition",
            "",
            f"Classification: **{result['root_cause']['classification']}**.",
            "",
            result["root_cause"]["training_behavior"] + ". "
            + result["root_cause"]["causal_live_behavior"] + ".",
            "",
            result["root_cause"]["isolating_evidence"],
            "",
            "Downstream effect: " + result["root_cause"]["downstream_effect"] + ".",
            "",
            (
                "Full fitted-row audit: "
                f"`{result['fitted_training_clock_audit']['fit_candidate_rows_with_context_unavailable_at_decision']}`/"
                f"`{result['fitted_training_clock_audit']['fit_candidate_rows']}` fitted rows across "
                f"`{result['fitted_training_clock_audit']['affected_session_count']}`/"
                f"`{result['fitted_training_clock_audit']['session_count']}` sessions used context "
                "that was unavailable under the preregistered completed-minute+60s clock."
            ),
            "",
            (
                "Deferred-emission escape hatch: `"
                + result["deferred_emission_assessment"]["verdict"]
                + "`. Waiting 60 seconds makes the bytes available but pairs an OPRA interval "
                "ending at t with the following SPX minute and cannot execute the confirmed entry quote."
            ),
            "",
            "Frozen-model disposition: `"
            + result["root_cause"]["immutable_model_disposition"]
            + "`.",
            "",
            (
                "The requested 100% target was met."
                if result["status"] == "OFFLINE_DECISION_PARITY_CERTIFIED"
                else "The requested 100% target was not met, so the certification sentence is prohibited for this frozen artifact."
            ),
            "",
            "## Live decision-feature contract",
            "",
            f"Contract SHA-256: `{contract['contract_sha256']}`. The JSON artifact contains all 18 ordered fields, formulas, historical/live twins, clocks, and carry rules.",
            "",
            "## Residual live-only risks",
            "",
        ]
    )
    rows.extend(f"- [ ] `{item['check']}` — {item['future_shadow_proof']}" for item in result["residual_live_risk_checklist"])
    rows.extend(
        [
            "",
            "## Scope attestation",
            "",
            "No live market-data connection, broker connection, paper action, order, model refit/retune, protected-holdout reopen, paper-default change, promotion change, or execution/fill-parity claim occurred.",
            "",
            "STOP_FOR_CLAUDE_VERIFICATION",
            "",
        ]
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(output=args.output.resolve())
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))
    return 0 if result["status"] == "OFFLINE_DECISION_PARITY_CERTIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
