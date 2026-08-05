"""Causal Path-D Phase-1 entry dataset and transparent HGB baseline.

This module is deliberately development-only.  It rebuilds the repaired
signed-18 entry rows from native OPRA CBBO-1m, exact-contract CBBO-1s, and
official completed ThetaData SPX bars.  The quarantined processed corpus is
never accepted as an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES, feature_matrix
from v4.research.pathd_entry_features import official_spx_market_window_from_rows
from v4.research.phase1_exit_model import OOFEntryReceiptV1, stable_hash


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MINUTE_NS = 60_000_000_000
SECOND_NS = 1_000_000_000
ENTRY_EMISSION_LAG_MS = 2_336
ENTRY_ORDER_LATENCY_MS = 1_000
ENTRY_QUOTE_MAX_AGE_MS = 2_000
ENTRY_FEE_PER_SIDE_DOLLARS = 1.50
ENTRY_FEATURE_NAMES = tuple(FEATURE_NAMES) + ("is_call",)
ENTRY_ROW_SCHEMA = "pathd.causal-entry-row.v2"
ENTRY_LABEL_SCHEMA = "pathd.causal-entry-label.v2"
ENTRY_ARTIFACT_SCHEMA = "pathd.entry-hgb-artifact.v2"
ENTRY_CAMPAIGN_SCHEMA = "pathd.entry-campaign.v2"
REFERENCE_POLICY_ID = "ask_to_bid_stop50_target100_hold25m"
DEVELOPMENT_SESSION_COUNT = 215
FILL_LAW = {
    "schema_version": "pathd.phase1-entry-fill-law.v2",
    "decision_emission_lag_ms": ENTRY_EMISSION_LAG_MS,
    "order_arrival_latency_ms": ENTRY_ORDER_LATENCY_MS,
    "entry_quote_max_age_ms": ENTRY_QUOTE_MAX_AGE_MS,
    "limit": "completed_cbbo_1m_ask_plus_one_valid_tick",
    "arrival": "latest_exact_contract_cbbo_1s_at_or_before_arrival",
    "fill": "arrival_ask_at_or_below_limit_fills_at_limit",
    "fee_per_filled_side_dollars": ENTRY_FEE_PER_SIDE_DOLLARS,
    "reference_exit": REFERENCE_POLICY_ID,
}
FILL_LAW_SHA256 = stable_hash(FILL_LAW)
_OSI_RE = re.compile(
    r"^SPXW  (?P<expiry>[0-9]{6})(?P<right>[CP])(?P<strike>[0-9]{8})$"
)


class EntryCampaignError(RuntimeError):
    """Raised when the repaired entry contract is violated."""


def development_sessions(corpus_root: Path) -> tuple[str, ...]:
    """Return the pre-firewall prefix without decoding any firewall file."""

    directory = Path(corpus_root) / "raw/databento/opra_spxw_cbbo_1m"
    sessions = tuple(
        sorted(
            path.name.removesuffix(".cbbo-1m.parquet")
            for path in directory.glob("*.cbbo-1m.parquet")
        )
    )
    if len(sessions) != 251:
        raise EntryCampaignError(
            f"expected the frozen 251-session corpus inventory, observed {len(sessions)}"
        )
    return sessions[:DEVELOPMENT_SESSION_COUNT]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tick(price: float) -> float:
    return 0.05 if price < 3.0 else 0.10


def _session_terminal_ns(session: str) -> int:
    day = datetime.fromisoformat(session).date()
    return int(
        datetime.combine(day, time(15, 55), tzinfo=NY)
        .astimezone(UTC)
        .timestamp()
        * 1e9
    )


def _parse_symbol(symbol: str, session: str) -> tuple[str, float]:
    match = _OSI_RE.fullmatch(symbol)
    if match is None or match.group("expiry") != pd.Timestamp(session).strftime("%y%m%d"):
        raise EntryCampaignError(f"invalid same-session SPXW OSI symbol: {symbol}")
    return match.group("right"), int(match.group("strike")) / 1000.0


@dataclass(frozen=True)
class CausalEntryRowV2:
    schema_version: str
    candidate_uid: str
    session: str
    feature_boundary_ns: int
    decision_emission_ns: int
    entry_arrival_ns: int
    raw_symbol: str
    instrument_id: int
    right: str
    strike: float
    atm_strike: float
    strike_offset: float
    entry_feature_bid: float
    entry_feature_ask: float
    entry_feature_mid: float
    feature_order_sha256: str
    source_receipt_sha256: str
    row_sha256: str


@dataclass(frozen=True)
class CausalEntryLabelV2:
    schema_version: str
    candidate_uid: str
    session: str
    filled: bool
    fill_time_ns: int
    fill_price: float
    entry_fee_dollars: float
    exit_time_ns: int
    exit_bid: float
    exit_reason: str
    net_pnl_dollars: float
    fill_law_sha256: str
    label_sha256: str


@dataclass(frozen=True)
class NonnegativeAffineCalibratorV1:
    intercept: float
    slope: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(values, dtype=float)


@dataclass
class EntryHGBArtifactV2:
    feature_names: tuple[str, ...]
    model: HistGradientBoostingRegressor
    calibrator: NonnegativeAffineCalibratorV1
    fold: int | None
    role: str
    artifact_sha256: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if tuple(frame.columns) != self.feature_names:
            raise EntryCampaignError("entry inference feature order drift")
        raw = self.model.predict(frame.to_numpy(dtype=np.float64))
        return self.calibrator.predict(raw)


def _seal_row(semantic: dict[str, Any], feature_values: np.ndarray) -> dict[str, Any]:
    row_hash = stable_hash({**semantic, "features": feature_values.tolist()})
    return {
        **semantic,
        "row_sha256": row_hash,
        **dict(zip(ENTRY_FEATURE_NAMES, feature_values, strict=True)),
    }


def _seal_label(semantic: dict[str, Any]) -> dict[str, Any]:
    return {**semantic, "label_sha256": stable_hash(semantic)}


def _load_parquet(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise EntryCampaignError(f"missing source partition: {path}")
    available = set(pads.dataset(path, format="parquet").schema.names)
    missing = set(columns) - available
    if missing:
        raise EntryCampaignError(f"source schema missing {sorted(missing)}: {path}")
    frame = pd.read_parquet(path, columns=list(columns))
    for name in columns:
        if name not in frame.columns and frame.index.name == name:
            frame = frame.reset_index()
    return frame


def _official_spx(path: Path, session: str) -> pd.DataFrame:
    frame = _load_parquet(
        path,
        (
            "event_time",
            "symbol",
            "close",
            "volume",
            "context_source",
            "is_derived",
            "is_proxy",
            "is_official_index_data",
        ),
    )
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
        raise EntryCampaignError("official SPX provenance drift")
    # The owned corpus stores the immutable vendor path in context_source;
    # normalize only the in-memory adapter value after authenticating it.
    frame["context_source"] = "thetadata_index_history_ohlc"
    event = pd.to_datetime(frame["event_time"], utc=True)
    local = event.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    frame = frame.loc[local.eq(session)].copy()
    if frame.empty or event.duplicated().any():
        raise EntryCampaignError("official SPX session is empty or duplicated")
    return frame.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def _cbbo_1m(path: Path, session: str) -> pd.DataFrame:
    frame = _load_parquet(
        path,
        (
            "ts_recv",
            "instrument_id",
            "symbol",
            "bid_px_00",
            "ask_px_00",
            "bid_sz_00",
            "ask_sz_00",
        ),
    )
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    if frame["ts_recv"].duplicated().all():
        raise EntryCampaignError("CBBO-1m source has no contract dimension")
    parsed = frame["symbol"].astype(str).map(lambda value: _parse_symbol(value, session))
    frame["right"] = [value[0] for value in parsed]
    frame["strike"] = [value[1] for value in parsed]
    return frame.sort_values(["ts_recv", "symbol"], kind="mergesort").reset_index(drop=True)


def _cbbo_1s_for_symbols(path: Path, symbols: Sequence[str]) -> pd.DataFrame:
    dataset = pads.dataset(path, format="parquet")
    expression = pads.field("symbol").isin(list(symbols))
    columns = [
        "ts_recv",
        "instrument_id",
        "symbol",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    ]
    frame = dataset.to_table(columns=columns, filter=expression).to_pandas(
        ignore_metadata=True
    )
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    if frame.duplicated(["ts_recv", "symbol"]).any():
        raise EntryCampaignError("duplicate exact-contract CBBO-1s identity")
    return frame.sort_values(["symbol", "ts_recv"], kind="mergesort").reset_index(drop=True)


def _label_reference_path(
    path: pd.DataFrame,
    *,
    boundary_ns: int,
    arrival_ns: int,
    completed_ask: float,
    session_terminal_ns: int,
) -> dict[str, Any] | None:
    times = path["ts_recv"].astype("int64").to_numpy()
    before = np.flatnonzero((times >= boundary_ns) & (times <= arrival_ns))
    if not len(before):
        return None
    arrival_index = int(before[-1])
    quote_age_ms = (arrival_ns - int(times[arrival_index])) / 1e6
    if quote_age_ms < 0 or quote_age_ms > ENTRY_QUOTE_MAX_AGE_MS:
        return None
    arrival_ask = float(path["ask_px_00"].iloc[arrival_index])
    arrival_bid = float(path["bid_px_00"].iloc[arrival_index])
    limit = round(completed_ask + _tick(completed_ask), 2)
    if (
        not np.isfinite([arrival_bid, arrival_ask, limit]).all()
        or arrival_bid <= 0.0
        or arrival_ask <= arrival_bid
        or arrival_ask > limit + 1e-12
    ):
        return None
    deadline_ns = min(arrival_ns + 25 * MINUTE_NS, session_terminal_ns)
    through = int(np.searchsorted(times, deadline_ns, side="right"))
    future = path.iloc[arrival_index:through]
    if future.empty:
        return None
    bids = pd.to_numeric(future["bid_px_00"], errors="coerce").to_numpy(float)
    future_times = future["ts_recv"].astype("int64").to_numpy()
    valid = np.isfinite(bids) & (bids >= 0.0)
    if not valid.any():
        return None
    exit_at = None
    exit_reason = "MAX_HOLD_25M"
    for index in np.flatnonzero(valid):
        if bids[index] <= 0.50 * limit:
            exit_at = int(index)
            exit_reason = "STOP_50"
            break
        if bids[index] >= 2.00 * limit:
            exit_at = int(index)
            exit_reason = "TARGET_100"
            break
    if exit_at is None:
        exit_at = int(np.flatnonzero(valid)[-1])
    exit_bid = float(bids[exit_at])
    pnl = (exit_bid - limit) * 100.0 - 2.0 * ENTRY_FEE_PER_SIDE_DOLLARS
    return {
        "fill_time_ns": int(arrival_ns),
        "fill_price": limit,
        "entry_fee_dollars": ENTRY_FEE_PER_SIDE_DOLLARS,
        "exit_time_ns": int(future_times[exit_at]),
        "exit_bid": exit_bid,
        "exit_reason": exit_reason,
        "net_pnl_dollars": float(pnl),
    }


def build_causal_entry_session(
    corpus_root: Path,
    session: str,
    *,
    emission_lag_ms: int = ENTRY_EMISSION_LAG_MS,
    order_latency_ms: int = ENTRY_ORDER_LATENCY_MS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialize one repaired session from immutable raw sources."""

    if emission_lag_ms != ENTRY_EMISSION_LAG_MS or order_latency_ms != ENTRY_ORDER_LATENCY_MS:
        raise EntryCampaignError("entry clock parameters differ from the frozen law")
    if session not in development_sessions(corpus_root):
        raise EntryCampaignError(
            f"session is outside the 215-session pre-firewall development prefix: {session}"
        )
    raw = Path(corpus_root) / "raw"
    one_m_path = raw / "databento/opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet"
    one_s_path = raw / "databento/opra_spxw_cbbo_1s" / f"{session}.cbbo-1s.parquet"
    spx_path = raw / "index/spx_1m" / f"{session}.official_spx.parquet"
    spx = _official_spx(spx_path, session)
    one_m = _cbbo_1m(one_m_path, session)

    # Predicate-push the high-resolution load to the union of contracts that
    # can enter a +/-50 point ladder anywhere in this session.  Loading every
    # quoted SPXW contract would defeat the 16 GB memory contract.
    spx_close = pd.to_numeric(spx["close"], errors="coerce")
    if not np.isfinite(spx_close).any():
        raise EntryCampaignError("official SPX has no finite session close")
    strike_floor = math.floor((float(np.nanmin(spx_close)) - 55.0) / 5.0) * 5.0
    strike_ceiling = math.ceil((float(np.nanmax(spx_close)) + 55.0) / 5.0) * 5.0
    one_m = one_m[one_m["strike"].between(strike_floor, strike_ceiling)].copy()

    local = one_m["ts_recv"].dt.tz_convert(NY)
    minute_of_session = (local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 30)
    one_m = one_m.loc[(minute_of_session >= 31) & (minute_of_session <= 350)].copy()
    if one_m.empty:
        raise EntryCampaignError("no CBBO-1m rows in the six frozen entry blocks")
    symbols = tuple(sorted(one_m["symbol"].astype(str).unique()))
    one_s = _cbbo_1s_for_symbols(one_s_path, symbols)
    paths = {symbol: group.reset_index(drop=True) for symbol, group in one_s.groupby("symbol")}

    state_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    source_hashes = {
        "cbbo_1m": _sha256_path(one_m_path),
        "cbbo_1s": _sha256_path(one_s_path),
        "official_spx": _sha256_path(spx_path),
    }
    terminal_ns = _session_terminal_ns(session)
    for boundary, minute_frame in one_m.groupby("ts_recv", sort=True):
        boundary_ns = int(pd.Timestamp(boundary).value)
        emission_ns = boundary_ns + emission_lag_ms * 1_000_000
        arrival_ns = emission_ns + order_latency_ms * 1_000_000
        try:
            market_window, market_names, spx_available_ns = official_spx_market_window_from_rows(
                spx,
                session=session,
                decision_time_ns=boundary_ns,
                history_minutes=30,
            )
        except ValueError:
            continue
        market = market_window[-1]
        spx_close = float(market[market_names.index("spx_close")])
        if not np.isfinite(spx_close):
            continue
        atm = float(round(spx_close / 5.0) * 5.0)
        minute_frame = minute_frame.copy()
        minute_frame["offset"] = minute_frame["strike"] - atm
        minute_frame = minute_frame[minute_frame["offset"].between(-50.0, 50.0)]
        if minute_frame.empty:
            continue
        offsets = np.arange(-50.0, 50.1, 5.0)
        rights = ("C", "P")
        ladder = np.full((len(offsets), 2, 1), np.nan, dtype=float)
        contract_ids = np.full((len(offsets), 2), "", dtype=object)
        row_lookup: dict[tuple[float, str], pd.Series] = {}
        for _, quote in minute_frame.sort_values("symbol").iterrows():
            key = (float(quote["offset"]), str(quote["right"]))
            if key in row_lookup:
                raise EntryCampaignError(f"duplicate CBBO-1m contract at boundary: {session}:{boundary}")
            if key[0] not in set(offsets):
                continue
            row_lookup[key] = quote
            strike_index = int(np.flatnonzero(offsets == key[0])[0])
            right_index = rights.index(key[1])
            bid = float(quote["bid_px_00"])
            ask = float(quote["ask_px_00"])
            ladder[strike_index, right_index, 0] = (bid + ask) / 2.0
            contract_ids[strike_index, right_index] = str(quote["symbol"])
        features = feature_matrix(
            {
                "decision_time": pd.Timestamp(boundary),
                "atm_strike": atm,
                "strike_offsets": offsets,
                "rights": rights,
                "option_ladder": ladder,
                "feature_names": ("mid",),
                "market_window": market_window,
                "market_feature_names": market_names,
            }
        )
        for (offset, right), quote in row_lookup.items():
            symbol = str(quote["symbol"])
            bid = float(quote["bid_px_00"])
            ask = float(quote["ask_px_00"])
            mid = (bid + ask) / 2.0
            if (
                not np.isfinite([bid, ask, mid]).all()
                or bid <= 0.0
                or ask <= bid
                or not 3.0 <= mid <= 8.0
            ):
                continue
            strike_index = int(np.flatnonzero(offsets == offset)[0])
            right_index = rights.index(right)
            vector = np.concatenate(
                [features[strike_index, right_index], [float(right == "C")]]
            ).astype(np.float64)
            if not np.isfinite(vector).all():
                continue
            path = paths.get(symbol)
            if path is None:
                continue
            label = _label_reference_path(
                path,
                boundary_ns=boundary_ns,
                arrival_ns=arrival_ns,
                completed_ask=ask,
                session_terminal_ns=terminal_ns,
            )
            if label is None:
                continue
            candidate_uid = f"{session}|{boundary_ns}|{symbol}"
            source_receipt = {
                "session": session,
                "feature_boundary_ns": boundary_ns,
                "decision_emission_ns": emission_ns,
                "entry_arrival_ns": arrival_ns,
                "official_spx_event_time_ns": spx_available_ns - MINUTE_NS,
                "official_spx_available_at_ns": spx_available_ns,
                "raw_symbol": symbol,
                "instrument_id": int(quote["instrument_id"]),
                "source_hashes": source_hashes,
                "fill_law_sha256": FILL_LAW_SHA256,
            }
            semantic = {
                "schema_version": ENTRY_ROW_SCHEMA,
                "candidate_uid": candidate_uid,
                "session": session,
                "feature_boundary_ns": boundary_ns,
                "decision_emission_ns": emission_ns,
                "entry_arrival_ns": arrival_ns,
                "raw_symbol": symbol,
                "instrument_id": int(quote["instrument_id"]),
                "right": right,
                "strike": float(quote["strike"]),
                "atm_strike": atm,
                "strike_offset": float(offset),
                "entry_feature_bid": bid,
                "entry_feature_ask": ask,
                "entry_feature_mid": mid,
                "feature_order_sha256": stable_hash(ENTRY_FEATURE_NAMES),
                "source_receipt_sha256": stable_hash(source_receipt),
            }
            state_rows.append(_seal_row(semantic, vector))
            label_rows.append(
                _seal_label(
                    {
                        "schema_version": ENTRY_LABEL_SCHEMA,
                        "candidate_uid": candidate_uid,
                        "session": session,
                        "filled": True,
                        **label,
                        "fill_law_sha256": FILL_LAW_SHA256,
                    }
                )
            )
    states = pd.DataFrame(state_rows)
    labels = pd.DataFrame(label_rows)
    if states.empty or labels.empty:
        raise EntryCampaignError(f"causal entry materialization produced no rows: {session}")
    identity = ["candidate_uid", "session"]
    if states[identity].duplicated().any() or labels[identity].duplicated().any():
        raise EntryCampaignError("duplicate entry row identity")
    if not states[identity].equals(labels[identity]):
        raise EntryCampaignError("entry feature/label identity mismatch")
    receipt = {
        "schema_version": "pathd.causal-entry-session-receipt.v2",
        "session": session,
        "rows": len(states),
        "feature_order": list(ENTRY_FEATURE_NAMES),
        "feature_order_sha256": stable_hash(ENTRY_FEATURE_NAMES),
        "fill_law": FILL_LAW,
        "fill_law_sha256": FILL_LAW_SHA256,
        "source_hashes": source_hashes,
        "protected_holdout_opened": False,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return states, labels, receipt


def write_causal_entry_session(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    receipt: Mapping[str, Any],
    *,
    canonical_root: Path,
) -> dict[str, Any]:
    session = str(receipt["session"])
    directory = Path(canonical_root) / "entry_v2" / f"session={session}"
    directory.mkdir(parents=True, exist_ok=False)
    feature_path = directory / "features.parquet"
    label_path = directory / "labels.parquet"
    receipt_path = directory / "receipt.json"
    states.to_parquet(feature_path, index=False, compression="zstd")
    labels.to_parquet(label_path, index=False, compression="zstd")
    receipt_path.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n")
    return {
        "session": session,
        "features": str(feature_path),
        "features_sha256": _sha256_path(feature_path),
        "labels": str(label_path),
        "labels_sha256": _sha256_path(label_path),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_path(receipt_path),
    }


def load_entry_dataset(canonical_root: Path) -> pd.DataFrame:
    directories = sorted((Path(canonical_root) / "entry_v2").glob("session=*"))
    if not directories:
        raise EntryCampaignError("no causal entry_v2 partitions")
    frames = []
    for directory in directories:
        receipt = json.loads((directory / "receipt.json").read_text())
        semantic = dict(receipt)
        expected = semantic.pop("receipt_sha256", None)
        if expected != stable_hash(semantic) or receipt.get("protected_holdout_opened") is not False:
            raise EntryCampaignError(f"entry session receipt drift: {directory.name}")
        features = pd.read_parquet(directory / "features.parquet")
        labels = pd.read_parquet(directory / "labels.parquet")
        merged = features.merge(labels, on=["candidate_uid", "session"], validate="one_to_one")
        frames.append(merged)
    frame = pd.concat(frames, ignore_index=True)
    if frame["candidate_uid"].duplicated().any():
        raise EntryCampaignError("duplicate entry candidate across partitions")
    return frame.sort_values(["session", "feature_boundary_ns", "candidate_uid"]).reset_index(drop=True)


def entry_expanding_folds(
    sessions: Sequence[str], *, initial_train_sessions: int = 44, n_folds: int = 5
) -> tuple[dict[str, tuple[str, ...]], ...]:
    ordered = tuple(sorted(set(map(str, sessions))))
    if len(ordered) < initial_train_sessions + 5 * 10 + n_folds:
        raise EntryCampaignError("not enough sessions for five entry folds and embargoes")
    test_total = len(ordered) - initial_train_sessions - n_folds
    sizes = [len(chunk) for chunk in np.array_split(np.arange(test_total), n_folds)]
    cursor = initial_train_sessions
    folds = []
    for index, size in enumerate(sizes):
        train = ordered[:cursor]
        embargo = ordered[cursor : cursor + 1]
        test = ordered[cursor + 1 : cursor + 1 + size]
        folds.append({"fold": index, "train": train, "embargo": embargo, "test": test})
        cursor += 1 + size
    tests = [session for fold in folds for session in fold["test"]]
    if len(tests) != len(set(tests)):
        raise EntryCampaignError("entry outer-test session overlap")
    return tuple(folds)


def _session_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("session")["session"].transform("size").to_numpy(float)
    return 1.0 / counts


def _fit_affine(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> NonnegativeAffineCalibratorV1:
    x = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean_x = float(np.average(x, weights=w))
    mean_y = float(np.average(y, weights=w))
    variance = float(np.average((x - mean_x) ** 2, weights=w))
    covariance = float(np.average((x - mean_x) * (y - mean_y), weights=w))
    slope = max(0.0, covariance / variance) if variance > 0.0 else 0.0
    return NonnegativeAffineCalibratorV1(intercept=mean_y - slope * mean_x, slope=slope)


def _fit_entry_artifact(
    train: pd.DataFrame,
    *,
    fold: int | None,
    role: str,
) -> EntryHGBArtifactV2:
    sessions = tuple(sorted(train["session"].astype(str).unique()))
    calibration_count = max(1, int(math.ceil(0.20 * len(sessions))))
    calibration_sessions = sessions[-calibration_count:]
    fit_sessions = sessions[:-calibration_count]
    if not fit_sessions:
        raise EntryCampaignError("entry fit has no pre-calibration sessions")
    fit = train[train["session"].isin(fit_sessions)]
    calibration = train[train["session"].isin(calibration_sessions)]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=3,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=211,
    )
    model.fit(
        fit.loc[:, ENTRY_FEATURE_NAMES].to_numpy(np.float64),
        fit["net_pnl_dollars"].to_numpy(float),
        sample_weight=_session_balanced_weights(fit),
    )
    raw = model.predict(calibration.loc[:, ENTRY_FEATURE_NAMES].to_numpy(np.float64))
    calibrator = _fit_affine(
        raw,
        calibration["net_pnl_dollars"].to_numpy(float),
        _session_balanced_weights(calibration),
    )
    semantic = {
        "schema_version": ENTRY_ARTIFACT_SCHEMA,
        "feature_names": list(ENTRY_FEATURE_NAMES),
        "feature_order_sha256": stable_hash(ENTRY_FEATURE_NAMES),
        "fill_law_sha256": FILL_LAW_SHA256,
        "fold": fold,
        "role": role,
        "fit_sessions": list(fit_sessions),
        "calibration_sessions": list(calibration_sessions),
        "model_parameters": model.get_params(),
        "calibrator": asdict(calibrator),
    }
    return EntryHGBArtifactV2(
        feature_names=ENTRY_FEATURE_NAMES,
        model=model,
        calibrator=calibrator,
        fold=fold,
        role=role,
        artifact_sha256=stable_hash(semantic),
    )


def _write_entry_artifact(
    artifact: EntryHGBArtifactV2,
    *,
    directory: Path,
    fold_spec: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    joblib.dump(artifact, model_path)
    manifest = {
        "schema_version": ENTRY_ARTIFACT_SCHEMA,
        "role": artifact.role,
        "fold": artifact.fold,
        "feature_names": list(artifact.feature_names),
        "feature_order_sha256": stable_hash(artifact.feature_names),
        "fill_law": FILL_LAW,
        "fill_law_sha256": FILL_LAW_SHA256,
        "clock": {
            "emission_lag_ms": ENTRY_EMISSION_LAG_MS,
            "order_latency_ms": ENTRY_ORDER_LATENCY_MS,
        },
        "identity_transform": True,
        "calibrator": asdict(artifact.calibrator),
        "composer": "ENTER_IFF_CALIBRATED_EXPECTED_NET_PNL_GT_ZERO",
        "fold_spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in fold_spec.items()
        },
        "model_path": str(model_path),
        "model_sha256": _sha256_path(model_path),
        "artifact_semantic_sha256": artifact.artifact_sha256,
        "protected_holdout_opened": False,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_path": str(manifest_path)}


def _fixed_block(boundary_ns: pd.Series) -> np.ndarray:
    local = pd.to_datetime(boundary_ns, unit="ns", utc=True).dt.tz_convert(NY)
    minute = local.dt.hour * 60 + local.dt.minute - (9 * 60 + 30)
    return np.select(
        [
            (minute >= 31) & (minute < 90),
            (minute >= 90) & (minute < 150),
            (minute >= 150) & (minute < 210),
            (minute >= 210) & (minute < 270),
            (minute >= 270) & (minute < 330),
            (minute >= 330) & (minute <= 350),
        ],
        [0, 1, 2, 3, 4, 5],
        default=-1,
    )


def select_oof_entry_policy(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["fixed_block"] = _fixed_block(frame["feature_boundary_ns"])
    frame = frame[(frame["fixed_block"] >= 0) & (frame["calibrated_prediction"] > 0.0)]
    if frame.empty:
        return frame
    best = (
        frame.sort_values(
            ["session", "feature_boundary_ns", "calibrated_prediction", "candidate_uid"],
            ascending=[True, True, False, True],
        )
        .groupby(["session", "feature_boundary_ns"], sort=True)
        .head(1)
    )
    choices = []
    groups = {
        key: group
        for key, group in frame.groupby(["session", "feature_boundary_ns"], sort=False)
    }
    for signal in best.to_dict("records"):
        group = groups[(signal["session"], signal["feature_boundary_ns"])]
        side = str(signal["right"])
        candidate = (
            group[group["right"].astype(str).eq(side)]
            .assign(abs_moneyness=lambda value: value["strike_offset"].abs())
            .sort_values(["abs_moneyness", "candidate_uid"])
            .iloc[0]
            .to_dict()
        )
        candidate["signal_prediction"] = float(signal["calibrated_prediction"])
        choices.append(candidate)
    selected = pd.DataFrame(choices)
    return (
        selected.sort_values(["session", "fixed_block", "feature_boundary_ns"])
        .groupby(["session", "fixed_block"], sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def select_deterministic_control_policy(scored: pd.DataFrame) -> pd.DataFrame:
    """Select one causal trend-following ATM control per fixed time block.

    The control does not inspect the fitted score or any label.  It chooses the
    first actionable boundary in each block, uses the sign of the completed
    15-minute SPX momentum to choose call versus put, and then picks the exact
    fillable contract nearest ATM with a stable identity tie-break.
    """
    frame = scored.copy()
    frame["fixed_block"] = _fixed_block(frame["feature_boundary_ns"])
    frame = frame[frame["fixed_block"] >= 0]
    if frame.empty:
        return frame
    choices: list[dict[str, Any]] = []
    for (_, _), block in frame.groupby(["session", "fixed_block"], sort=True):
        boundary = int(block["feature_boundary_ns"].min())
        candidates = block[block["feature_boundary_ns"].eq(boundary)].copy()
        momentum = float(candidates["momentum_15m_bps"].iloc[0])
        desired_right = "C" if momentum >= 0.0 else "P"
        same_side = candidates[candidates["right"].astype(str).eq(desired_right)]
        if same_side.empty:
            same_side = candidates
        choice = (
            same_side.assign(abs_moneyness=lambda value: value["strike_offset"].abs())
            .sort_values(["abs_moneyness", "candidate_uid"])
            .iloc[0]
            .to_dict()
        )
        choice["signal_prediction"] = float("nan")
        choices.append(choice)
    return pd.DataFrame(choices).sort_values(
        ["session", "fixed_block", "feature_boundary_ns"]
    ).reset_index(drop=True)


def _entry_receipts(
    selected: pd.DataFrame,
    artifact_hashes: Mapping[int, str],
    *,
    policy_role: str,
) -> list[OOFEntryReceiptV1]:
    receipts: list[OOFEntryReceiptV1] = []
    for session, session_frame in selected.groupby("session", sort=True):
        realized = 0.0
        occupied_until = -1
        for row in session_frame.sort_values("feature_boundary_ns").to_dict("records"):
            if int(row["fill_time_ns"]) < occupied_until:
                continue
            fold = int(row["outer_fold"])
            trajectory_id = stable_hash(
                {
                    "session": session,
                    "candidate_uid": row["candidate_uid"],
                    "fold": fold,
                    "role": policy_role,
                }
            )[:24]
            receipts.append(
                OOFEntryReceiptV1.seal(
                    session=str(session),
                    trajectory_id=trajectory_id,
                    raw_symbol=str(row["raw_symbol"]),
                    instrument_id=int(row["instrument_id"]),
                    expiry=str(session),
                    strike=float(row["strike"]),
                    right=str(row["right"]),
                    decision_time_ns=int(row["feature_boundary_ns"]),
                    arrival_time_ns=int(row["entry_arrival_ns"]),
                    fill_time_ns=int(row["fill_time_ns"]),
                    fill_price=float(row["fill_price"]),
                    quantity=1,
                    entry_fee_dollars=float(row["entry_fee_dollars"]),
                    realized_session_pnl_dollars=float(realized),
                    remaining_risk_budget_dollars=max(0.0, 500.0 + realized),
                    outer_fold=fold,
                    entry_artifact_role="OUTER_FOLD_OOF",
                    entry_artifact_sha256=str(artifact_hashes[fold]),
                    fill_law_sha256=FILL_LAW_SHA256,
                )
            )
            realized += float(row["net_pnl_dollars"])
            occupied_until = int(row["exit_time_ns"])
    return receipts


def train_entry_campaign(
    dataset: pd.DataFrame,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    required = {
        "candidate_uid",
        "session",
        "net_pnl_dollars",
        "fill_time_ns",
        "exit_time_ns",
        *ENTRY_FEATURE_NAMES,
    }
    if required - set(dataset):
        raise EntryCampaignError(f"entry campaign missing columns: {sorted(required-set(dataset))}")
    if dataset["candidate_uid"].duplicated().any():
        raise EntryCampaignError("entry campaign candidate identities are not unique")
    sessions = sorted(dataset["session"].astype(str).unique())
    folds = entry_expanding_folds(sessions)
    root = Path(artifact_root) / "entry_v2"
    root.mkdir(parents=True, exist_ok=False)
    oof_rows = []
    manifests: dict[int, dict[str, Any]] = {}
    for fold in folds:
        train = dataset[dataset["session"].isin(fold["train"])]
        test = dataset[dataset["session"].isin(fold["test"])].copy()
        artifact = _fit_entry_artifact(
            train, fold=int(fold["fold"]), role="OUTER_FOLD_OOF"
        )
        manifest = _write_entry_artifact(
            artifact,
            directory=root / f"fold={fold['fold']}",
            fold_spec=fold,
        )
        manifests[int(fold["fold"])] = manifest
        test["calibrated_prediction"] = artifact.predict(test.loc[:, ENTRY_FEATURE_NAMES])
        test["outer_fold"] = int(fold["fold"])
        oof_rows.append(test)
    oof = pd.concat(oof_rows, ignore_index=True)
    selected = select_oof_entry_policy(oof)
    control = select_deterministic_control_policy(oof)
    learned_hashes = {
        fold: str(manifest["manifest_sha256"]) for fold, manifest in manifests.items()
    }
    control_specs = {
        int(fold["fold"]): stable_hash(
            {
                "schema_version": "pathd.deterministic-entry-control.v1",
                "policy": "first_boundary_trend_side_nearest_atm",
                "fold": int(fold["fold"]),
                "train": list(fold["train"]),
                "embargo": list(fold["embargo"]),
                "test": list(fold["test"]),
                "feature": "completed_spx_momentum_15m_bps",
                "fill_law_sha256": FILL_LAW_SHA256,
            }
        )
        for fold in folds
    }
    learned_receipts = _entry_receipts(
        selected, learned_hashes, policy_role="LEARNED_OOF"
    )
    control_receipts = _entry_receipts(
        control, control_specs, policy_role="DETERMINISTIC_CONTROL_OOF"
    )
    initial_history_receipts: list[tuple[str, OOFEntryReceiptV1]] = []
    initial_history_sessions = tuple(folds[0]["train"])
    initial_history_root = root / "initial_history_oof"
    initial_history_root.mkdir()
    for test_index in range(21, len(initial_history_sessions)):
        fit_scope = initial_history_sessions[: test_index - 1]
        embargo_session = initial_history_sessions[test_index - 1]
        test_session = initial_history_sessions[test_index]
        artifact = _fit_entry_artifact(
            dataset[dataset["session"].isin(fit_scope)],
            fold=0,
            role="OUTER_FOLD_OOF",
        )
        manifest = _write_entry_artifact(
            artifact,
            directory=initial_history_root / f"test_session={test_session}",
            fold_spec={
                "train": tuple(fit_scope),
                "embargo": (embargo_session,),
                "test": (test_session,),
                "purpose": "INITIAL_HISTORY_EXIT_TRAINING_ONLY",
            },
        )
        scored = dataset[dataset["session"].eq(test_session)].copy()
        scored["calibrated_prediction"] = artifact.predict(
            scored.loc[:, ENTRY_FEATURE_NAMES]
        )
        scored["outer_fold"] = 0
        learned_initial = select_oof_entry_policy(scored)
        control_initial = select_deterministic_control_policy(scored)
        artifact_hash = {0: str(manifest["manifest_sha256"])}
        initial_history_receipts.extend(
            ("INITIAL_HISTORY_TRAIN_OOF_LEARNED", receipt)
            for receipt in _entry_receipts(
                learned_initial,
                artifact_hash,
                policy_role="INITIAL_HISTORY_TRAIN_OOF_LEARNED",
            )
        )
        initial_history_receipts.extend(
            ("INITIAL_HISTORY_TRAIN_OOF_CONTROL", receipt)
            for receipt in _entry_receipts(
                control_initial,
                artifact_hash,
                policy_role="INITIAL_HISTORY_TRAIN_OOF_CONTROL",
            )
        )
    receipts = learned_receipts + control_receipts + [
        receipt for _, receipt in initial_history_receipts
    ]
    receipt_root = root / "oof_receipts"
    receipt_root.mkdir()
    trajectory_index: list[dict[str, Any]] = []
    for policy_role, policy_receipts in (
        ("LEARNED_OOF", learned_receipts),
        ("DETERMINISTIC_CONTROL_OOF", control_receipts),
        (
            "INITIAL_HISTORY_TRAIN_OOF_LEARNED",
            [
                receipt
                for role, receipt in initial_history_receipts
                if role == "INITIAL_HISTORY_TRAIN_OOF_LEARNED"
            ],
        ),
        (
            "INITIAL_HISTORY_TRAIN_OOF_CONTROL",
            [
                receipt
                for role, receipt in initial_history_receipts
                if role == "INITIAL_HISTORY_TRAIN_OOF_CONTROL"
            ],
        ),
    ):
        policy_root = receipt_root / policy_role.lower()
        policy_root.mkdir()
        for receipt in policy_receipts:
            receipt_path = policy_root / f"{receipt.trajectory_id}.json"
            receipt_path.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n")
            trajectory_index.append(
                {
                    "trajectory_id": receipt.trajectory_id,
                    "session": receipt.session,
                    "outer_fold": receipt.outer_fold,
                    "entry_policy_role": policy_role,
                    "evaluation_eligible": policy_role
                    in {"LEARNED_OOF", "DETERMINISTIC_CONTROL_OOF"},
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt.receipt_sha256,
                }
            )
    oof_path = root / "oof_scores.parquet"
    selected_path = root / "oof_selected_entries.parquet"
    control_path = root / "oof_control_entries.parquet"
    trajectory_index_path = root / "trajectory_index.parquet"
    evaluation_trajectory_index_path = root / "evaluation_trajectory_index.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    selected.to_parquet(selected_path, index=False, compression="zstd")
    control.to_parquet(control_path, index=False, compression="zstd")
    trajectory_index_frame = pd.DataFrame(trajectory_index)
    trajectory_index_frame.to_parquet(
        trajectory_index_path, index=False, compression="zstd"
    )
    trajectory_index_frame[trajectory_index_frame["evaluation_eligible"]].to_parquet(
        evaluation_trajectory_index_path, index=False, compression="zstd"
    )
    full = _fit_entry_artifact(dataset, fold=None, role="FULL_DEVELOPMENT_SHADOW_ONLY")
    full_manifest = _write_entry_artifact(
        full,
        directory=root / "full_development",
        fold_spec={"train": tuple(sessions), "embargo": tuple(), "test": tuple()},
    )
    session_pnl = selected.groupby("session")["net_pnl_dollars"].sum()
    payload = {
        "schema_version": ENTRY_CAMPAIGN_SCHEMA,
        "status": "TRAINED_DEVELOPMENT_ONLY",
        "sessions": len(sessions),
        "candidate_rows": len(dataset),
        "oof_rows": len(oof),
        "oof_selected_entries": len(selected),
        "oof_control_entries": len(control),
        "oof_learned_receipts": len(learned_receipts),
        "oof_control_receipts": len(control_receipts),
        "initial_history_training_receipts": len(initial_history_receipts),
        "oof_receipts": len(receipts),
        "oof_net_pnl_dollars": float(selected["net_pnl_dollars"].sum()) if len(selected) else 0.0,
        "positive_oof_sessions": int((session_pnl > 0.0).sum()),
        "fold_manifests": {str(key): value["manifest_path"] for key, value in manifests.items()},
        "full_development_manifest": full_manifest["manifest_path"],
        "full_development_may_generate_exit_training": False,
        "oof_scores": str(oof_path),
        "oof_selected_entries_path": str(selected_path),
        "oof_control_entries_path": str(control_path),
        "trajectory_index_path": str(trajectory_index_path),
        "evaluation_trajectory_index_path": str(evaluation_trajectory_index_path),
        "oof_receipt_root": str(receipt_root),
        "protected_holdout_opened": False,
        "paper_or_broker_authorized": False,
    }
    payload["campaign_sha256"] = stable_hash(payload)
    campaign_path = root / "campaign.json"
    campaign_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "campaign_path": str(campaign_path)}


def load_entry_artifact(manifest_path: Path) -> EntryHGBArtifactV2:
    manifest = json.loads(Path(manifest_path).read_text())
    semantic = dict(manifest)
    expected = semantic.pop("manifest_sha256", None)
    if expected != stable_hash(semantic):
        raise EntryCampaignError("entry manifest hash drift")
    model_path = Path(manifest["model_path"])
    if _sha256_path(model_path) != manifest["model_sha256"]:
        raise EntryCampaignError("entry model bytes drift")
    artifact = joblib.load(model_path)
    if type(artifact) is not EntryHGBArtifactV2:
        raise EntryCampaignError("entry artifact type drift")
    if artifact.artifact_sha256 != manifest["artifact_semantic_sha256"]:
        raise EntryCampaignError("entry artifact semantic drift")
    return artifact


__all__ = [
    "CausalEntryRowV2",
    "CausalEntryLabelV2",
    "EntryHGBArtifactV2",
    "EntryCampaignError",
    "ENTRY_FEATURE_NAMES",
    "ENTRY_EMISSION_LAG_MS",
    "ENTRY_ORDER_LATENCY_MS",
    "FILL_LAW_SHA256",
    "DEVELOPMENT_SESSION_COUNT",
    "development_sessions",
    "build_causal_entry_session",
    "write_causal_entry_session",
    "load_entry_dataset",
    "entry_expanding_folds",
    "select_oof_entry_policy",
    "select_deterministic_control_policy",
    "train_entry_campaign",
    "load_entry_artifact",
]
