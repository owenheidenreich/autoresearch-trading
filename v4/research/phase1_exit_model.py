"""Development-only one-second exit dataset and HGB baseline.

This is a source-neutral implementation of the Phase-1 interface.  It reads
one session and one exact OPRA contract at a time, emits causal feature rows,
and writes future-derived targets separately.  It has no broker, paper-order,
registry, paid-download, or protected-holdout capability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.greeks.repair import compute_repaired_greeks
from v4.research.pathd_feature_live_twin import EXIT47_CORRECTED_FEATURE_NAMES


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
HEX64 = re.compile(r"[0-9a-f]{64}")
RECEIPT_SCHEMA = "pathd.oof-entry-receipt.v1"
STATE_SCHEMA = "pathd.exit-state-row.v1"
LABEL_SCHEMA = "pathd.exit-label-row.v1"
MODEL_SCHEMA = "pathd.phase1-exit-hgb.v1"
DEFAULT_FEE_PER_SIDE = 1.50
STRESS_FEE_PER_SIDE = 2.00
CONTRACT_MULTIPLIER = 100
HEADLINE_LATENCY_SECONDS = 1
LATENCY_SENSITIVITIES = (0, 1, 2, 5)
OPRA_COMPLETED_SECOND_EMISSION_LAG_MS = 320
MAX_FIT_ROWS = 500_000
MAX_CALIBRATION_ROWS = 150_000

PNL_VELOCITY_FEATURE_NAMES = (
    "pnl_velocity_1s_dollars",
    "pnl_velocity_5s_dollars",
    "pnl_velocity_15s_dollars",
    "pnl_velocity_30s_dollars",
    "pnl_velocity_60s_dollars",
)
EXIT_FEATURE_NAMES = tuple(EXIT47_CORRECTED_FEATURE_NAMES) + PNL_VELOCITY_FEATURE_NAMES
FORBIDDEN_FEATURE_FRAGMENTS = (
    "future",
    "oracle",
    "label",
    "best_price",
    "unrealizedpnl",
    "ibkr_bid",
    "ibkr_ask",
    "vendor_delta",
    "vendor_gamma",
    "vendor_iv",
)


class ExitModelContractError(RuntimeError):
    """The causal exit-model contract was violated."""


def _canonical(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


@dataclass(frozen=True)
class OOFEntryReceiptV1:
    schema_version: str
    session: str
    trajectory_id: str
    raw_symbol: str
    instrument_id: int
    expiry: str
    strike: float
    right: str
    decision_time_ns: int
    arrival_time_ns: int
    fill_time_ns: int
    fill_price: float
    quantity: int
    entry_fee_dollars: float
    realized_session_pnl_dollars: float
    remaining_risk_budget_dollars: float
    outer_fold: int
    entry_artifact_role: str
    entry_artifact_sha256: str
    fill_law_sha256: str
    receipt_sha256: str

    @classmethod
    def seal(cls, **values: Any) -> "OOFEntryReceiptV1":
        semantic = dict(values)
        semantic.pop("receipt_sha256", None)
        semantic.setdefault("schema_version", RECEIPT_SCHEMA)
        receipt = cls(**semantic, receipt_sha256=stable_hash(semantic))
        validate_oof_entry_receipt(receipt)
        return receipt


def validate_oof_entry_receipt(receipt: OOFEntryReceiptV1) -> None:
    if type(receipt) is not OOFEntryReceiptV1 or receipt.schema_version != RECEIPT_SCHEMA:
        raise ExitModelContractError("exact OOFEntryReceiptV1 required")
    semantic = asdict(receipt)
    digest = semantic.pop("receipt_sha256")
    if digest != stable_hash(semantic):
        raise ExitModelContractError("OOF entry receipt hash drift")
    if receipt.entry_artifact_role != "OUTER_FOLD_OOF":
        raise ExitModelContractError("full-fit or non-OOF entry artifact may not create trajectories")
    if receipt.session != receipt.expiry:
        raise ExitModelContractError("Phase-1 entry is not an SPXW 0DTE identity")
    if receipt.right not in {"C", "P"} or not receipt.raw_symbol.startswith("SPXW  "):
        raise ExitModelContractError("entry identity is not an exact SPXW option")
    if receipt.instrument_id <= 0 or receipt.strike <= 0 or receipt.quantity <= 0:
        raise ExitModelContractError("invalid entry identity, strike, or quantity")
    if not (0 < receipt.fill_price and 0 <= receipt.entry_fee_dollars):
        raise ExitModelContractError("invalid entry fill accounting")
    if not receipt.decision_time_ns <= receipt.arrival_time_ns <= receipt.fill_time_ns:
        raise ExitModelContractError("entry receipt timestamps are noncausal")
    if receipt.outer_fold not in range(5):
        raise ExitModelContractError("outer fold must be 0..4")
    for name in ("entry_artifact_sha256", "fill_law_sha256"):
        if not _is_hex64(getattr(receipt, name)):
            raise ExitModelContractError(f"invalid {name}")


@dataclass(frozen=True)
class ExitStateRowV1:
    schema_version: str
    session: str
    trajectory_id: str
    raw_symbol: str
    instrument_id: int
    decision_time_ns: int
    decision_available_time_ns: int
    outer_fold: int
    entry_receipt_sha256: str
    feature_order_sha256: str
    features: tuple[float, ...]
    row_sha256: str


@dataclass(frozen=True)
class ExitLabelRowV1:
    schema_version: str
    session: str
    trajectory_id: str
    decision_time_ns: int
    fill_law_sha256: str
    latency_seconds: int
    fee_per_side_dollars: float
    a_ref_dollars: float
    hold_to_1555_value_dollars: float
    exit_until_filled_value_dollars: float
    downside_300_dollars: float
    recovery_300_dollars: float
    giveback_300_dollars: float
    remaining_tail_300_dollars: float
    label_sha256: str


def _seal_state(
    receipt: OOFEntryReceiptV1,
    timestamp_ns: int,
    decision_available_time_ns: int,
    values: Sequence[float],
) -> ExitStateRowV1:
    if len(values) != len(EXIT_FEATURE_NAMES):
        raise ExitModelContractError("exit feature width drift")
    semantic = {
        "schema_version": STATE_SCHEMA,
        "session": receipt.session,
        "trajectory_id": receipt.trajectory_id,
        "raw_symbol": receipt.raw_symbol,
        "instrument_id": receipt.instrument_id,
        "decision_time_ns": int(timestamp_ns),
        "decision_available_time_ns": int(decision_available_time_ns),
        "outer_fold": receipt.outer_fold,
        "entry_receipt_sha256": receipt.receipt_sha256,
        "feature_order_sha256": stable_hash(EXIT_FEATURE_NAMES),
        "features": tuple(float(value) for value in values),
    }
    return ExitStateRowV1(**semantic, row_sha256=stable_hash(semantic))


def _seal_label(**values: Any) -> ExitLabelRowV1:
    semantic = {"schema_version": LABEL_SCHEMA, **values}
    return ExitLabelRowV1(**semantic, label_sha256=stable_hash(semantic))


def validate_feature_registry() -> None:
    if len(EXIT_FEATURE_NAMES) != len(set(EXIT_FEATURE_NAMES)):
        raise ExitModelContractError("duplicate exit feature")
    for name in EXIT_FEATURE_NAMES:
        lowered = name.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise ExitModelContractError(f"forbidden exit feature: {name}")


def load_exact_cbbo_path(
    parquet_path: Path,
    receipt: OOFEntryReceiptV1,
    *,
    terminal_time_ns: int | None = None,
) -> pd.DataFrame:
    """Use Arrow filters so only one contract/session path reaches memory."""

    validate_oof_entry_receipt(receipt)
    parquet_path = parquet_path.resolve(strict=True)
    expected_name = f"{receipt.session}.cbbo-1s.parquet"
    if parquet_path.name != expected_name:
        raise ExitModelContractError(f"session partition mismatch: {parquet_path.name}")
    finish = terminal_time_ns or session_terminal_ns(receipt.session)
    columns = [
        "ts_recv",
        "ts_event",
        "instrument_id",
        "symbol",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    ]
    dataset = pads.dataset(parquet_path, format="parquet")
    expression = (
        (pads.field("instrument_id") == receipt.instrument_id)
        & (pads.field("symbol") == receipt.raw_symbol)
        & (pads.field("ts_recv") >= pd.Timestamp(receipt.fill_time_ns, unit="ns", tz="UTC"))
        & (pads.field("ts_recv") <= pd.Timestamp(finish, unit="ns", tz="UTC"))
    )
    frame = dataset.to_table(columns=columns, filter=expression).to_pandas(
        ignore_metadata=True
    )
    if frame.empty:
        raise ExitModelContractError("exact-contract CBBO-1s join produced no rows")
    if not frame["instrument_id"].eq(receipt.instrument_id).all() or not frame["symbol"].eq(receipt.raw_symbol).all():
        raise ExitModelContractError("exact-contract predicate escaped")
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    frame = frame.sort_values("ts_recv", kind="mergesort").reset_index(drop=True)
    if frame["ts_recv"].duplicated().any():
        raise ExitModelContractError("duplicate completed-second exact-contract row")
    if not frame["ts_recv"].is_monotonic_increasing:
        raise ExitModelContractError("CBBO path is not chronological")
    if int(frame["ts_recv"].iloc[-1].value) < finish - 5_000_000_000:
        raise ExitModelContractError("exact-contract path is incomplete before forced flat")
    return frame


def session_terminal_ns(session: str) -> int:
    day = datetime.fromisoformat(session).date()
    terminal = datetime.combine(day, time(15, 55), tzinfo=NY).astimezone(UTC)
    return int(terminal.timestamp() * 1_000_000_000)


def load_completed_spx_context(path: Path, *, emission_lag_ms: int) -> pd.DataFrame:
    if not 0 <= emission_lag_ms <= 10_000:
        raise ExitModelContractError("ThetaData emission lag must be frozen in 0..10000 ms")
    frame = pd.read_parquet(path)
    required = {
        "event_time", "symbol", "open", "high", "low", "close", "volume",
        "context_source", "is_derived", "is_proxy", "is_official_index_data",
    }
    if required - set(frame):
        raise ExitModelContractError("official SPX context schema drift")
    source = frame["context_source"].astype(str)
    official_source = source.eq("thetadata_index_history_ohlc") | source.str.contains(
        "/vendor/thetadata/index/spx_1m/", regex=False
    )
    if not (
        frame["symbol"].astype(str).eq("SPX").all()
        and official_source.all()
        and (~frame["is_derived"].astype(bool)).all()
        and (~frame["is_proxy"].astype(bool)).all()
        and frame["is_official_index_data"].astype(bool).all()
    ):
        raise ExitModelContractError("SPX context is not official ThetaData history")
    # The owned corpus stores the immutable vendor path in context_source;
    # normalize only after authenticating it against the same entry-plane law.
    frame["context_source"] = "thetadata_index_history_ohlc"
    frame = frame.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["available_at"] = frame["event_time"] + pd.Timedelta(seconds=60, milliseconds=emission_lag_ms)
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weighted = (close * volume).cumsum()
    denominator = volume.cumsum()
    fallback = close.expanding(min_periods=1).mean()
    frame["causal_vwap"] = weighted.div(denominator.where(denominator > 0)).fillna(fallback)
    for minutes in (1, 5, 15):
        frame[f"log_return_{minutes}m"] = np.log(close / close.shift(minutes))
    frame["vwap_gap_bps"] = (close - frame["causal_vwap"]) / frame["causal_vwap"] * 10_000.0
    return frame


def _asof_spx(option_times: pd.Series, spx: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({
        "decision_time": pd.to_datetime(option_times, utc=True).astype("datetime64[ns, UTC]")
    }).sort_values("decision_time")
    right = spx[[
        "available_at", "close", "log_return_1m", "log_return_5m", "log_return_15m", "vwap_gap_bps"
    ]].copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    right = right.sort_values("available_at")
    joined = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=90),
        allow_exact_matches=True,
    )
    if (joined["available_at"] > joined["decision_time"]).fillna(False).any():
        raise ExitModelContractError("future SPX minute entered exit features")
    return joined


def _rolling(series: pd.Series, seconds: int, statistic: str) -> pd.Series:
    window = series.rolling(seconds, min_periods=1)
    return getattr(window, statistic)()


def _tick(price: float) -> float:
    return 0.05 if price < 3.0 else 0.10


def _actionable(frame: pd.DataFrame, max_quote_age_ms: int) -> pd.Series:
    return (
        frame["bid"].gt(0)
        & frame["ask"].gt(frame["bid"])
        & frame["quote_state_age_ms"].le(max_quote_age_ms)
    )


def build_sensitivity_labels_from_features(
    receipt: OOFEntryReceiptV1,
    features: pd.DataFrame,
    *,
    fee_per_side_dollars: float = DEFAULT_FEE_PER_SIDE,
    latency_seconds: int = HEADLINE_LATENCY_SECONDS,
    max_quote_age_ms: int = 2_000,
) -> pd.DataFrame:
    """Reprice labels from one already-built causal feature trajectory."""

    validate_oof_entry_receipt(receipt)
    if latency_seconds not in LATENCY_SENSITIVITIES:
        raise ExitModelContractError("latency is outside the frozen sensitivity ladder")
    if fee_per_side_dollars not in {DEFAULT_FEE_PER_SIDE, STRESS_FEE_PER_SIDE}:
        raise ExitModelContractError("fee path must be $3 or $4 round trip")
    required = {
        "session", "trajectory_id", "decision_time_ns", "option_bid",
        "option_ask", "option_quote_age_ms",
    }
    if required - set(features):
        raise ExitModelContractError("sensitivity feature trajectory schema drift")
    if (
        not features["session"].astype(str).eq(receipt.session).all()
        or not features["trajectory_id"].astype(str).eq(receipt.trajectory_id).all()
    ):
        raise ExitModelContractError("sensitivity feature trajectory identity drift")
    decision_ns = pd.to_numeric(features["decision_time_ns"], errors="raise").to_numpy(
        dtype="int64"
    )
    bids = pd.to_numeric(features["option_bid"], errors="coerce").to_numpy(float)
    asks = pd.to_numeric(features["option_ask"], errors="coerce").to_numpy(float)
    quote_age = pd.to_numeric(
        features["option_quote_age_ms"], errors="coerce"
    ).to_numpy(float)
    actionable = (bids > 0.0) & (asks > bids) & (quote_age <= max_quote_age_ms)
    n = len(features)
    exit_value = np.full(n, np.nan, dtype=float)
    for index in range(n - 1, -1, -1):
        arrival = index + latency_seconds
        limit = bids[index] - _tick(bids[index]) if np.isfinite(bids[index]) else np.nan
        if (
            actionable[index]
            and arrival < n
            and actionable[arrival]
            and np.isfinite(limit)
            and bids[arrival] >= limit
        ):
            exit_value[index] = (
                (limit - receipt.fill_price) * CONTRACT_MULTIPLIER * receipt.quantity
                - receipt.entry_fee_dollars
                - fee_per_side_dollars
            )
        elif index + 1 < n:
            exit_value[index] = exit_value[index + 1]
        else:
            exit_value[index] = (
                (0.0 - receipt.fill_price) * CONTRACT_MULTIPLIER * receipt.quantity
                - receipt.entry_fee_dollars
                - fee_per_side_dollars
            )
    terminal_decision = max(0, n - 1 - latency_seconds)
    terminal_limit = (
        bids[terminal_decision] - _tick(bids[terminal_decision])
        if actionable[terminal_decision] and np.isfinite(bids[terminal_decision])
        else np.nan
    )
    terminal_bid = (
        terminal_limit
        if (
            np.isfinite(terminal_limit)
            and actionable[-1]
            and bids[-1] >= terminal_limit
        )
        else 0.0
    )
    hold_value = (
        (terminal_bid - receipt.fill_price) * CONTRACT_MULTIPLIER * receipt.quantity
        - receipt.entry_fee_dollars
        - fee_per_side_dollars
    )
    current = (
        (bids - receipt.fill_price) * CONTRACT_MULTIPLIER * receipt.quantity
        - receipt.entry_fee_dollars
        - fee_per_side_dollars
    )
    label_rows: list[ExitLabelRowV1] = []
    for index, timestamp in enumerate(decision_ns):
        future = current[index : min(n, index + 301)]
        peak = np.nanmax(future)
        trough = np.nanmin(future)
        label_rows.append(
            _seal_label(
                session=receipt.session,
                trajectory_id=receipt.trajectory_id,
                decision_time_ns=int(timestamp),
                fill_law_sha256=receipt.fill_law_sha256,
                latency_seconds=latency_seconds,
                fee_per_side_dollars=fee_per_side_dollars,
                a_ref_dollars=float(hold_value - exit_value[index]),
                hold_to_1555_value_dollars=float(hold_value),
                exit_until_filled_value_dollars=float(exit_value[index]),
                downside_300_dollars=float(trough - current[index]),
                recovery_300_dollars=float(peak - current[index]),
                giveback_300_dollars=float(peak - future[-1]),
                remaining_tail_300_dollars=float(hold_value - future[-1]),
            )
        )
    labels = pd.DataFrame([asdict(row) for row in label_rows])
    identity = ["session", "trajectory_id", "decision_time_ns"]
    if features[identity].duplicated().any() or labels[identity].duplicated().any():
        raise ExitModelContractError("duplicate trajectory row identity")
    if not features[identity].reset_index(drop=True).equals(labels[identity]):
        raise ExitModelContractError("feature/label identity alignment drift")
    return labels


def build_trajectory_tables(
    receipt: OOFEntryReceiptV1,
    cbbo: pd.DataFrame,
    spx: pd.DataFrame,
    *,
    fee_per_side_dollars: float = DEFAULT_FEE_PER_SIDE,
    latency_seconds: int = HEADLINE_LATENCY_SECONDS,
    opra_emission_lag_ms: int = OPRA_COMPLETED_SECOND_EMISSION_LAG_MS,
    max_quote_age_ms: int = 2_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return causal features and future labels as separate, hash-bound tables."""

    validate_feature_registry()
    validate_oof_entry_receipt(receipt)
    if latency_seconds not in LATENCY_SENSITIVITIES:
        raise ExitModelContractError("latency is outside the frozen sensitivity ladder")
    if opra_emission_lag_ms != OPRA_COMPLETED_SECOND_EMISSION_LAG_MS:
        raise ExitModelContractError("OPRA completed-second emission lag differs from frozen 320 ms")
    if fee_per_side_dollars not in {DEFAULT_FEE_PER_SIDE, STRESS_FEE_PER_SIDE}:
        raise ExitModelContractError("fee path must be $3 or $4 round trip")
    if cbbo.empty:
        raise ExitModelContractError("empty trajectory path")
    frame = cbbo.copy()
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    start = pd.Timestamp(receipt.fill_time_ns, unit="ns", tz="UTC").ceil("s")
    terminal = pd.Timestamp(session_terminal_ns(receipt.session), unit="ns", tz="UTC")
    grid = pd.date_range(start, terminal, freq="1s")
    if not len(grid):
        raise ExitModelContractError("entry fill occurs after the forced-flat boundary")
    frame["source_quote_time"] = frame["ts_recv"]
    frame = frame.set_index("ts_recv").reindex(grid)
    quote_columns = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    frame[quote_columns] = frame[quote_columns].ffill()
    frame["source_quote_time"] = frame["source_quote_time"].ffill()
    frame["instrument_id"] = receipt.instrument_id
    frame["symbol"] = receipt.raw_symbol
    frame = frame.rename_axis("decision_time").reset_index()
    frame["bid"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    frame["bid_size"] = pd.to_numeric(frame["bid_sz_00"], errors="coerce")
    frame["ask_size"] = pd.to_numeric(frame["ask_sz_00"], errors="coerce")
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["quote_state_age_ms"] = (
        frame["decision_time"] - frame["source_quote_time"]
    ).dt.total_seconds() * 1000.0
    frame["imbalance"] = (frame["bid_size"] - frame["ask_size"]) / (
        frame["bid_size"] + frame["ask_size"]
    ).replace(0, np.nan)
    log_mid = np.log(frame["mid"].where(frame["mid"] > 0))
    for seconds in (1, 5, 15, 30, 60):
        frame[f"log_mid_return_{seconds}s"] = log_mid - log_mid.shift(seconds)
    for seconds in (5, 15, 60):
        frame[f"spread_mean_{seconds}s"] = _rolling(frame["spread"], seconds, "mean")
        frame[f"imbalance_mean_{seconds}s"] = _rolling(frame["imbalance"], seconds, "mean")
    for seconds in (15, 60):
        frame[f"spread_std_{seconds}s"] = _rolling(frame["spread"], seconds, "std")

    context = _asof_spx(frame["decision_time"], spx)
    frame["spx_close"] = context["close"].to_numpy()
    frame["spx_log_return_1m"] = context["log_return_1m"].to_numpy()
    frame["spx_log_return_5m"] = context["log_return_5m"].to_numpy()
    frame["spx_log_return_15m"] = context["log_return_15m"].to_numpy()
    frame["spx_vwap_gap_bps"] = context["vwap_gap_bps"].to_numpy()

    terminal_ns = session_terminal_ns(receipt.session)
    greek_rows: list[tuple[float, float, float]] = []
    expiry_dt = datetime.fromisoformat(receipt.expiry).replace(
        hour=16, minute=0, tzinfo=NY
    ).astimezone(UTC)
    for row in frame.itertuples(index=False):
        current = row.decision_time.to_pydatetime()
        years = max((expiry_dt - current).total_seconds(), 1.0) / (365.0 * 24 * 3600)
        estimate = compute_repaired_greeks(
            S=row.spx_close,
            K=receipt.strike,
            T=years,
            is_call=receipt.right == "C",
            mid=row.mid,
            ask=row.ask,
            bid=row.bid,
        )
        greek_rows.append(
            (np.nan, np.nan, np.nan)
            if estimate is None
            else (estimate.iv, estimate.delta, estimate.gamma)
        )
    frame[["self_iv", "self_delta", "self_gamma"]] = np.asarray(greek_rows)
    frame["iv_change_5s"] = frame["self_iv"] - frame["self_iv"].shift(5)
    frame["iv_change_30s"] = frame["self_iv"] - frame["self_iv"].shift(30)
    frame["delta_change_30s"] = frame["self_delta"] - frame["self_delta"].shift(30)
    frame["gamma_change_30s"] = frame["self_gamma"] - frame["self_gamma"].shift(30)

    gross = (frame["bid"] - receipt.fill_price) * CONTRACT_MULTIPLIER * receipt.quantity
    frame["net_pnl"] = gross - receipt.entry_fee_dollars - fee_per_side_dollars
    frame["mfe"] = frame["net_pnl"].cummax()
    frame["mae"] = frame["net_pnl"].cummin()
    frame["giveback"] = frame["mfe"] - frame["net_pnl"]
    for seconds in (1, 5, 15, 30, 60):
        frame[f"pnl_velocity_{seconds}s"] = (
            frame["net_pnl"] - frame["net_pnl"].shift(seconds)
        ) / float(seconds)
    decision_ns = frame["decision_time"].astype("int64")
    decision_available_ns = decision_ns + int(opra_emission_lag_ms) * 1_000_000
    seconds_held = (decision_ns - receipt.fill_time_ns) / 1e9
    seconds_to_flat = (terminal_ns - decision_ns) / 1e9
    strike_offset = receipt.strike - frame["spx_close"]

    feature_columns: dict[str, pd.Series | float] = {
        "option_bid": frame["bid"],
        "option_ask": frame["ask"],
        "option_mid": frame["mid"],
        "option_spread": frame["spread"],
        "option_spread_over_mid": frame["spread"] / frame["mid"].replace(0, np.nan),
        "option_bid_size": frame["bid_size"],
        "option_ask_size": frame["ask_size"],
        "option_size_imbalance": frame["imbalance"],
        "option_quote_age_ms": frame["quote_state_age_ms"],
        **{f"option_log_mid_return_{s}s": frame[f"log_mid_return_{s}s"] for s in (1, 5, 15, 30, 60)},
        **{f"option_spread_mean_{s}s": frame[f"spread_mean_{s}s"] for s in (5, 15, 60)},
        **{f"option_spread_std_{s}s": frame[f"spread_std_{s}s"] for s in (15, 60)},
        **{f"option_imbalance_mean_{s}s": frame[f"imbalance_mean_{s}s"] for s in (5, 15, 60)},
        "official_spx_close": frame["spx_close"],
        "official_spx_log_return_1m": frame["spx_log_return_1m"],
        "official_spx_log_return_5m": frame["spx_log_return_5m"],
        "official_spx_log_return_15m": frame["spx_log_return_15m"],
        "official_spx_vwap_gap_bps": frame["spx_vwap_gap_bps"],
        "self_computed_iv": frame["self_iv"],
        "self_computed_delta": frame["self_delta"],
        "self_computed_gamma": frame["self_gamma"],
        "self_computed_iv_change_5s": frame["iv_change_5s"],
        "self_computed_iv_change_30s": frame["iv_change_30s"],
        "self_computed_delta_change_30s": frame["delta_change_30s"],
        "self_computed_gamma_change_30s": frame["gamma_change_30s"],
        "held_right_is_call": float(receipt.right == "C"),
        "held_strike_offset_points": strike_offset,
        "entry_fill_option_price": receipt.fill_price,
        "current_net_pnl_dollars": frame["net_pnl"],
        "current_return_on_entry_premium": frame["net_pnl"] / (receipt.fill_price * CONTRACT_MULTIPLIER * receipt.quantity),
        "mfe_dollars_to_now": frame["mfe"],
        "mae_dollars_to_now": frame["mae"],
        "giveback_dollars_to_now": frame["giveback"],
        "seconds_held": seconds_held,
        "seconds_to_15:55": seconds_to_flat,
        "position_occupancy": 1.0,
        "remaining_d48_budget_dollars": receipt.remaining_risk_budget_dollars,
        "realized_session_pnl_dollars": receipt.realized_session_pnl_dollars,
        **{f"pnl_velocity_{s}s_dollars": frame[f"pnl_velocity_{s}s"] for s in (1, 5, 15, 30, 60)},
    }
    if tuple(feature_columns) != EXIT_FEATURE_NAMES:
        raise ExitModelContractError("feature construction order differs from registry")

    feature_matrix = pd.DataFrame(feature_columns, index=frame.index).astype("float64")
    state_rows = [
        _seal_state(receipt, int(timestamp), int(available), row)
        for timestamp, available, row in zip(
            decision_ns, decision_available_ns, feature_matrix.to_numpy(), strict=True
        )
    ]
    features = pd.DataFrame(
        [
            {
                "schema_version": item.schema_version,
                "session": item.session,
                "trajectory_id": item.trajectory_id,
                "raw_symbol": item.raw_symbol,
                "instrument_id": item.instrument_id,
                "decision_time_ns": item.decision_time_ns,
                "decision_available_time_ns": item.decision_available_time_ns,
                "outer_fold": item.outer_fold,
                "entry_receipt_sha256": item.entry_receipt_sha256,
                "feature_order_sha256": item.feature_order_sha256,
                "row_sha256": item.row_sha256,
                **dict(zip(EXIT_FEATURE_NAMES, item.features, strict=True)),
            }
            for item in state_rows
        ]
    )

    labels = build_sensitivity_labels_from_features(
        receipt,
        features,
        fee_per_side_dollars=fee_per_side_dollars,
        latency_seconds=latency_seconds,
        max_quote_age_ms=max_quote_age_ms,
    )
    return features, labels


def write_trajectory_partition(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    scratch_root: Path,
    session: str,
    trajectory_id: str,
) -> dict[str, str]:
    if not scratch_root.is_dir():
        raise ExitModelContractError(f"scratch root is not provisioned: {scratch_root}")
    feature_path = scratch_root / "exit_features" / f"session={session}" / f"{trajectory_id}.parquet"
    label_path = scratch_root / "exit_labels" / f"session={session}" / f"{trajectory_id}.parquet"
    if feature_path.exists() or label_path.exists():
        raise ExitModelContractError("trajectory partition already exists")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False, compression="zstd")
    labels.to_parquet(label_path, index=False, compression="zstd")
    return {
        "features": str(feature_path),
        "features_sha256": _sha256_path(feature_path),
        "labels": str(label_path),
        "labels_sha256": _sha256_path(label_path),
    }


def sensitivity_label_path(
    scratch_root: Path,
    *,
    session: str,
    trajectory_id: str,
    fee_per_side_dollars: float,
    latency_seconds: int,
) -> Path:
    roundtrip = 2.0 * float(fee_per_side_dollars)
    return (
        Path(scratch_root)
        / "exit_sensitivity_labels"
        / f"roundtrip_fee_dollars={roundtrip:.2f}"
        / f"latency_seconds={int(latency_seconds)}"
        / f"session={session}"
        / f"{trajectory_id}.parquet"
    )


def write_sensitivity_label_partition(
    labels: pd.DataFrame,
    *,
    scratch_root: Path,
    session: str,
    trajectory_id: str,
    fee_per_side_dollars: float,
    latency_seconds: int,
) -> dict[str, str]:
    path = sensitivity_label_path(
        scratch_root,
        session=session,
        trajectory_id=trajectory_id,
        fee_per_side_dollars=fee_per_side_dollars,
        latency_seconds=latency_seconds,
    )
    if path.exists():
        raise ExitModelContractError(f"sensitivity label partition already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False, compression="zstd")
    return {"labels": str(path), "labels_sha256": _sha256_path(path)}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanding_outer_folds(
    sessions: Sequence[str], *, initial_fit_sessions: int = 70, test_sessions: int = 28
) -> tuple[dict[str, tuple[str, ...]], ...]:
    ordered = tuple(sorted(set(sessions)))
    required = initial_fit_sessions + 5 * (1 + test_sessions)
    if len(ordered) < required:
        raise ExitModelContractError(
            f"five-fold design needs at least {required} chronological sessions, got {len(ordered)}"
        )
    folds = []
    cursor = initial_fit_sessions
    for fold in range(5):
        train = ordered[:cursor]
        embargo = ordered[cursor : cursor + 1]
        test = ordered[cursor + 1 : cursor + 1 + test_sessions]
        if set(train) & set(test) or set(embargo) & (set(train) | set(test)):
            raise ExitModelContractError("fold overlap or embargo failure")
        folds.append({"train": train, "embargo": embargo, "test": test})
        cursor += 1 + test_sessions
    return tuple(folds)


@dataclass
class ExitHGBEnsembleV1:
    schema_version: str
    feature_names: tuple[str, ...]
    seeds: tuple[int, ...]
    mean_models: tuple[HistGradientBoostingRegressor, ...]
    q10_models: tuple[HistGradientBoostingRegressor, ...]
    q50_models: tuple[HistGradientBoostingRegressor, ...]
    q90_models: tuple[HistGradientBoostingRegressor, ...]
    calibration_offsets: dict[str, float]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(frame.columns) != self.feature_names:
            raise ExitModelContractError("HGB inference feature order drift")
        matrix = frame.to_numpy(dtype=np.float64)
        means = np.vstack([model.predict(matrix) for model in self.mean_models])
        result = {
            "mean": np.median(means, axis=0),
            "mean_lcb90": np.median(means, axis=0) + self.calibration_offsets["mean_lcb90"],
        }
        raw_quantiles: dict[str, np.ndarray] = {}
        for name, models in (
            ("q10", self.q10_models), ("q50", self.q50_models), ("q90", self.q90_models)
        ):
            raw = np.median(np.vstack([model.predict(matrix) for model in models]), axis=0)
            raw_quantiles[name] = raw + self.calibration_offsets[name]
        result["q10"] = raw_quantiles["q10"]
        result["q50"] = result["q10"] + np.logaddexp(
            0.0, raw_quantiles["q50"] - result["q10"]
        )
        result["q90"] = result["q50"] + np.logaddexp(
            0.0, raw_quantiles["q90"] - result["q50"]
        )
        predictions = pd.DataFrame(result, index=frame.index)
        predictions["utility_hold"] = predictions["mean_lcb90"] + 0.25 * np.minimum(
            predictions["q10"], 0.0
        )
        predictions["action"] = np.where(predictions["utility_hold"] > 0.0, "HOLD", "EXIT")
        return predictions


def _model(seed: int, *, loss: str, quantile: float | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=3,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def fit_hgb_baseline(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    calibration_features: pd.DataFrame,
    calibration_labels: pd.Series,
    *,
    sample_weight: np.ndarray | None = None,
    seeds: Sequence[int] = (301, 302, 303),
) -> ExitHGBEnsembleV1:
    validate_feature_registry()
    if tuple(train_features.columns) != EXIT_FEATURE_NAMES or tuple(calibration_features.columns) != EXIT_FEATURE_NAMES:
        raise ExitModelContractError("HGB fit feature registry drift")
    if len(train_features) != len(train_labels) or len(calibration_features) != len(calibration_labels):
        raise ExitModelContractError("HGB feature/target row mismatch")
    if not len(train_features) or not len(calibration_features):
        raise ExitModelContractError("HGB fit and calibration sets must be nonempty")
    x_train = train_features.to_numpy(dtype=np.float64)
    y_train = train_labels.to_numpy(dtype=np.float64)
    x_cal = calibration_features.to_numpy(dtype=np.float64)
    y_cal = calibration_labels.to_numpy(dtype=np.float64)
    bundles: dict[str, list[HistGradientBoostingRegressor]] = {
        "mean": [], "q10": [], "q50": [], "q90": []
    }
    for seed in seeds:
        for name, loss, quantile in (
            ("mean", "squared_error", None),
            ("q10", "quantile", 0.10),
            ("q50", "quantile", 0.50),
            ("q90", "quantile", 0.90),
        ):
            fitted = _model(int(seed), loss=loss, quantile=quantile)
            fitted.fit(x_train, y_train, sample_weight=sample_weight)
            bundles[name].append(fitted)
    raw = {
        name: np.median(np.vstack([model.predict(x_cal) for model in models]), axis=0)
        for name, models in bundles.items()
    }
    offsets = {
        "mean_lcb90": float(np.quantile(y_cal - raw["mean"], 0.10)),
        "q10": float(np.quantile(y_cal - raw["q10"], 0.10)),
        "q50": float(np.quantile(y_cal - raw["q50"], 0.50)),
        "q90": float(np.quantile(y_cal - raw["q90"], 0.90)),
    }
    return ExitHGBEnsembleV1(
        schema_version=MODEL_SCHEMA,
        feature_names=EXIT_FEATURE_NAMES,
        seeds=tuple(int(seed) for seed in seeds),
        mean_models=tuple(bundles["mean"]),
        q10_models=tuple(bundles["q10"]),
        q50_models=tuple(bundles["q50"]),
        q90_models=tuple(bundles["q90"]),
        calibration_offsets=offsets,
    )


def _deterministic_sha_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Take a deterministic identity-ranked sample without reading future labels."""

    if maximum_rows <= 0:
        raise ExitModelContractError("sample cap must be positive")
    if len(frame) <= maximum_rows:
        return frame.sort_values("decision_time_ns", kind="mergesort").reset_index(drop=True)
    required = {"session", "trajectory_id", "decision_time_ns"}
    if required - set(frame):
        raise ExitModelContractError("trajectory sample identity columns missing")
    keys = (
        frame["session"].astype(str)
        + "|"
        + frame["trajectory_id"].astype(str)
        + "|"
        + frame["decision_time_ns"].astype(str)
    )
    ranks = np.fromiter(
        (int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) for value in keys),
        dtype=np.uint64,
        count=len(frame),
    )
    selected = np.argpartition(ranks, maximum_rows - 1)[:maximum_rows]
    return frame.iloc[selected].sort_values(
        ["session", "trajectory_id", "decision_time_ns"], kind="mergesort"
    ).reset_index(drop=True)


def _partition_pairs(
    scratch_root: Path, sessions: Iterable[str]
) -> dict[str, list[tuple[Path, Path]]]:
    pairs: dict[str, list[tuple[Path, Path]]] = {}
    for session in sorted(set(map(str, sessions))):
        feature_dir = Path(scratch_root) / "exit_features" / f"session={session}"
        label_dir = Path(scratch_root) / "exit_labels" / f"session={session}"
        session_pairs = []
        for feature_path in sorted(feature_dir.glob("*.parquet")):
            label_path = label_dir / feature_path.name
            if not label_path.is_file():
                raise ExitModelContractError(f"missing trajectory label partition: {label_path}")
            session_pairs.append((feature_path, label_path))
        if session_pairs:
            pairs[session] = session_pairs
    return pairs


def load_balanced_trajectory_sample(
    scratch_root: Path,
    sessions: Iterable[str],
    *,
    maximum_rows: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load a session/trajectory-balanced deterministic fit sample."""

    pairs = _partition_pairs(scratch_root, sessions)
    if not pairs:
        raise ExitModelContractError("no exit trajectory partitions for requested sessions")
    per_session = max(1, int(math.ceil(maximum_rows / len(pairs))))
    pieces: list[pd.DataFrame] = []
    for session, session_pairs in pairs.items():
        per_trajectory = max(1, int(math.ceil(per_session / len(session_pairs))))
        for feature_path, label_path in session_pairs:
            features = pd.read_parquet(feature_path)
            labels = pd.read_parquet(label_path)
            identity = ["session", "trajectory_id", "decision_time_ns"]
            if not features[identity].equals(labels[identity]):
                raise ExitModelContractError(f"trajectory feature/label drift: {feature_path}")
            joined = features.merge(
                labels[identity + ["a_ref_dollars"]],
                on=identity,
                validate="one_to_one",
            )
            pieces.append(_deterministic_sha_sample(joined, per_trajectory))
    sample = pd.concat(pieces, ignore_index=True)
    sample = _deterministic_sha_sample(sample, maximum_rows)
    finite = np.isfinite(sample["a_ref_dollars"].to_numpy(float))
    sample = sample.loc[finite].reset_index(drop=True)
    if sample.empty:
        raise ExitModelContractError("exit fit sample has no finite A_ref labels")
    trajectory_count = sample.groupby(["session", "trajectory_id"])["decision_time_ns"].transform(
        "size"
    ).to_numpy(float)
    trajectories_per_session = sample.groupby("session")["trajectory_id"].transform(
        "nunique"
    ).to_numpy(float)
    weights = 1.0 / (trajectory_count * trajectories_per_session)
    weights *= len(weights) / float(weights.sum())
    return (
        sample.loc[:, EXIT_FEATURE_NAMES].copy(),
        sample["a_ref_dollars"].copy(),
        weights,
    )


def _save_exit_artifact(
    artifact: ExitHGBEnsembleV1,
    *,
    directory: Path,
    role: str,
    fold: int | None,
    fit_sessions: Sequence[str],
    calibration_sessions: Sequence[str],
    test_sessions: Sequence[str],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    joblib.dump(artifact, model_path)
    manifest = {
        "schema_version": MODEL_SCHEMA,
        "role": role,
        "fold": fold,
        "feature_names": list(EXIT_FEATURE_NAMES),
        "feature_order_sha256": stable_hash(EXIT_FEATURE_NAMES),
        "decision_clock": {
            "opra_completed_second_emission_lag_ms": OPRA_COMPLETED_SECOND_EMISSION_LAG_MS,
            "headline_order_latency_seconds": HEADLINE_LATENCY_SECONDS,
            "latency_sensitivities_seconds": list(LATENCY_SENSITIVITIES),
        },
        "utility": "mean_lcb90 + 0.25 * min(q10, 0); HOLD iff > 0",
        "seeds": list(artifact.seeds),
        "calibration_offsets": artifact.calibration_offsets,
        "fit_sessions": list(fit_sessions),
        "calibration_sessions": list(calibration_sessions),
        "test_sessions": list(test_sessions),
        "fit_row_cap": MAX_FIT_ROWS,
        "calibration_row_cap": MAX_CALIBRATION_ROWS,
        "model_path": str(model_path),
        "model_sha256": _sha256_path(model_path),
        "protected_holdout_opened": False,
        "paper_or_broker_authorized": False,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_path": str(manifest_path)}


def _fit_exit_scope(
    scratch_root: Path,
    train_sessions: Sequence[str],
    *,
    seeds: Sequence[int],
) -> tuple[ExitHGBEnsembleV1, tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(set(map(str, train_sessions))))
    if len(ordered) < 5:
        raise ExitModelContractError("exit fit scope has fewer than five sessions")
    calibration_count = max(1, int(math.ceil(0.20 * len(ordered))))
    calibration_sessions = ordered[-calibration_count:]
    fit_sessions = ordered[:-calibration_count]
    train_x, train_y, train_weight = load_balanced_trajectory_sample(
        scratch_root, fit_sessions, maximum_rows=MAX_FIT_ROWS
    )
    calibration_x, calibration_y, _ = load_balanced_trajectory_sample(
        scratch_root, calibration_sessions, maximum_rows=MAX_CALIBRATION_ROWS
    )
    artifact = fit_hgb_baseline(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        sample_weight=train_weight,
        seeds=seeds,
    )
    return artifact, fit_sessions, calibration_sessions


def _score_exit_test_partitions(
    artifact: ExitHGBEnsembleV1,
    *,
    scratch_root: Path,
    sessions: Sequence[str],
    output_root: Path,
) -> dict[str, Any]:
    pairs = _partition_pairs(scratch_root, sessions)
    target_values: list[np.ndarray] = []
    prediction_values: list[np.ndarray] = []
    action_counts = {"HOLD": 0, "EXIT": 0}
    rows = 0
    for session, session_pairs in pairs.items():
        session_out = output_root / f"session={session}"
        session_out.mkdir(parents=True, exist_ok=True)
        for feature_path, label_path in session_pairs:
            features = pd.read_parquet(feature_path)
            labels = pd.read_parquet(label_path)
            prediction = artifact.predict(features.loc[:, EXIT_FEATURE_NAMES])
            identity = features[["session", "trajectory_id", "decision_time_ns"]]
            output = pd.concat([identity.reset_index(drop=True), prediction.reset_index(drop=True)], axis=1)
            output_path = session_out / feature_path.name
            output.to_parquet(output_path, index=False, compression="zstd")
            target = labels["a_ref_dollars"].to_numpy(float)
            finite = np.isfinite(target) & np.isfinite(prediction["mean"].to_numpy(float))
            target_values.append(target[finite])
            prediction_values.append(prediction.loc[finite, "mean"].to_numpy(float))
            counts = prediction["action"].value_counts()
            for action in action_counts:
                action_counts[action] += int(counts.get(action, 0))
            rows += len(prediction)
    if not rows:
        return {"rows": 0, "target_skill_correlation": None, "mse": None, "actions": action_counts}
    target = np.concatenate(target_values) if target_values else np.array([], dtype=float)
    pred = np.concatenate(prediction_values) if prediction_values else np.array([], dtype=float)
    correlation = (
        float(np.corrcoef(target, pred)[0, 1])
        if len(target) >= 2 and np.std(target) > 0 and np.std(pred) > 0
        else None
    )
    return {
        "rows": rows,
        "target_skill_correlation": correlation,
        "mse": float(np.mean((target - pred) ** 2)) if len(target) else None,
        "actions": action_counts,
    }


def train_exit_campaign(
    *,
    scratch_root: Path,
    entry_campaign_path: Path,
    artifact_root: Path,
    seeds: Sequence[int] = (301, 302, 303),
) -> dict[str, Any]:
    """Fit five OOF exit folds and one full-development shadow artifact."""

    entry_campaign = json.loads(Path(entry_campaign_path).read_text())
    semantic = dict(entry_campaign)
    observed = semantic.pop("campaign_sha256", None)
    if observed != stable_hash(semantic):
        raise ExitModelContractError("entry campaign receipt drift")
    if entry_campaign.get("protected_holdout_opened") is not False:
        raise ExitModelContractError("entry campaign is not development-only")
    root = Path(artifact_root) / "exit_v1"
    root.mkdir(parents=True, exist_ok=False)
    manifests: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    all_sessions: set[str] = set()
    for fold_key, manifest_path_value in sorted(
        entry_campaign["fold_manifests"].items(), key=lambda item: int(item[0])
    ):
        entry_manifest = json.loads(Path(manifest_path_value).read_text())
        entry_semantic = dict(entry_manifest)
        entry_hash = entry_semantic.pop("manifest_sha256", None)
        if entry_hash != stable_hash(entry_semantic):
            raise ExitModelContractError("entry fold manifest drift")
        fold_spec = entry_manifest["fold_spec"]
        train_sessions = tuple(map(str, fold_spec["train"]))
        test_sessions = tuple(map(str, fold_spec["test"]))
        all_sessions.update(train_sessions)
        all_sessions.update(test_sessions)
        artifact, fit_sessions, calibration_sessions = _fit_exit_scope(
            scratch_root, train_sessions, seeds=seeds
        )
        fold = int(fold_key)
        manifest = _save_exit_artifact(
            artifact,
            directory=root / f"fold={fold}",
            role="OUTER_FOLD_OOF",
            fold=fold,
            fit_sessions=fit_sessions,
            calibration_sessions=calibration_sessions,
            test_sessions=test_sessions,
        )
        manifests[str(fold)] = manifest["manifest_path"]
        metrics[str(fold)] = _score_exit_test_partitions(
            artifact,
            scratch_root=scratch_root,
            sessions=test_sessions,
            output_root=root / "oof_predictions" / f"fold={fold}",
        )
    ordered = tuple(sorted(all_sessions))
    full, full_fit, full_calibration = _fit_exit_scope(
        scratch_root, ordered, seeds=seeds
    )
    full_manifest = _save_exit_artifact(
        full,
        directory=root / "full_development",
        role="FULL_DEVELOPMENT_SHADOW_ONLY",
        fold=None,
        fit_sessions=full_fit,
        calibration_sessions=full_calibration,
        test_sessions=(),
    )
    correlations = [
        value["target_skill_correlation"]
        for value in metrics.values()
        if value["target_skill_correlation"] is not None
    ]
    payload = {
        "schema_version": "pathd.phase1-exit-campaign.v1",
        "status": "TRAINED_DEVELOPMENT_ONLY",
        "fold_manifests": manifests,
        "fold_metrics": metrics,
        "positive_target_skill_folds": int(sum(value > 0.0 for value in correlations)),
        "full_development_manifest": full_manifest["manifest_path"],
        "protected_holdout_opened": False,
        "paper_or_broker_authorized": False,
    }
    payload["campaign_sha256"] = stable_hash(payload)
    campaign_path = root / "campaign.json"
    campaign_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "campaign_path": str(campaign_path)}


def load_exit_artifact(manifest_path: Path) -> ExitHGBEnsembleV1:
    manifest = json.loads(Path(manifest_path).read_text())
    semantic = dict(manifest)
    expected = semantic.pop("manifest_sha256", None)
    if expected != stable_hash(semantic):
        raise ExitModelContractError("exit artifact manifest drift")
    model_path = Path(manifest["model_path"])
    if _sha256_path(model_path) != manifest["model_sha256"]:
        raise ExitModelContractError("exit artifact bytes drift")
    artifact = joblib.load(model_path)
    if type(artifact) is not ExitHGBEnsembleV1:
        raise ExitModelContractError("exit artifact type drift")
    return artifact


def catastrophic_floor_triggered(
    *, entry_price: float, current_bid: float, entry_fee_dollars: float = DEFAULT_FEE_PER_SIDE,
    exit_fee_dollars: float = DEFAULT_FEE_PER_SIDE,
) -> bool:
    """Deterministic net -50% premium backstop; never a learned feature/label."""

    premium = entry_price * CONTRACT_MULTIPLIER
    net_pnl = (current_bid - entry_price) * CONTRACT_MULTIPLIER - entry_fee_dollars - exit_fee_dollars
    return net_pnl <= -0.50 * premium
