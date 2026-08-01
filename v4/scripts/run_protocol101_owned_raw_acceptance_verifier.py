"""Verify owned raw data before Protocol101 fair-contract fold placement.

This data-plane-only verifier turns "raw files exist" into an acceptance
registry. A future fold scaffold must join all three predicates before placing a
session: era role permits the requested role, canonical processed rows exist,
and this acceptance registry marks the session pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
)


SCHEMA_VERSION = "Protocol101OwnedRawAcceptanceRegistryV3_5"
PREDICATE_SCHEMA_VERSION = "Protocol101FoldPlacementPredicateV1"
VERIFIER_VERSION = 35
MIN_FOLD_PLACEMENT_VERIFIER_VERSION = 35
# A session may be classified report_only/missing_index_context only when the
# official SPX vendor file itself lacks at most this many expected source
# minutes and every imperfect-lag row is attributable to those exact minutes.
MAX_ATTRIBUTABLE_MISSING_INDEX_MINUTES = 5
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_owned_raw_acceptance")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_VIX_DIR = Path("data/vendor/thetadata/index/vix_1m")
DEFAULT_PROCESSED_DIR = Path("data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_2024_10_live_v1")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_protocol101_owned_raw_acceptance_2024_10_live_v1")
DEFAULT_ERA_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
DEFAULT_ROLE_POLICY = Path("v4/audit/autoresearch/protocol101_era_role_policy/summary.json")
NY = ZoneInfo("America/New_York")

EARLY_CLOSE_TIMES_ET = {
    # Cboe U.S. Options RTH early closes only. Keep in sync with the
    # neural dataset builder; do not import bond-market early closes here.
    "2024-11-29": "13:00",
    "2024-12-24": "13:15",
    "2025-07-03": "13:00",
    "2025-11-28": "13:00",
    "2025-12-24": "13:15",
}
MARKET_HOLIDAYS = {
    "2024-11-28",
    "2024-12-25",
    "2025-01-01",
    "2025-01-09",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
}

PRODUCTS = {
    "definition": (
        ("databento", "opra_spxw_definition", "definition"),
        "definition",
    ),
    "cbbo_1m": (
        ("databento", "opra_spxw_cbbo_1m", "cbbo-1m"),
        "cbbo-1m",
    ),
    "ohlcv_1m": (
        ("databento", "opra_spxw_ohlcv_1m", "ohlcv-1m"),
        "ohlcv-1m",
    ),
    "statistics": (
        ("databento", "opra_spxw_statistics", "statistics"),
        "statistics",
    ),
}
CONTRACT_ID_RE = re.compile(
    r"^(?P<root>[A-Z]+)-(?P<expiry>\d{8})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>[CP])$"
)
PINNED_MARKET_FEATURE_NAMES = (
    "spx_close",
    "vix_close",
    "spx_vwap",
    "omar",
    "session_range",
    "momentum_5m",
    "momentum_15m",
)
OPTION_LADDER_QUOTE_INDEX = {"bid": 0, "ask": 1, "mid": 2}
PINNED_FEE_MODEL = {"fee_model": "gross_no_fees", "fee_per_contract": 0.0}
PINNED_MAX_QUOTE_AGE_SECONDS = 90.0
PINNED_CONTRACT_MULTIPLIER = 100
PINNED_FORCED_FLAT_BEFORE_ET = "15:55"
# Trade-shape menu v2, owner-approved 2026-07-07; must mirror
# NeuralDatasetConfig.label_policies exactly (independent duplicate by design;
# any drift fails the raw-label recompute loudly).
PINNED_LABEL_POLICIES = (
    {"policy_idx": 0, "stop_loss_pct": 0.35, "take_profit_pct": 0.60, "max_hold_minutes": 10},
    {"policy_idx": 1, "stop_loss_pct": 0.50, "take_profit_pct": 1.00, "max_hold_minutes": 25},
    {"policy_idx": 2, "stop_loss_pct": 0.65, "take_profit_pct": 1.50, "max_hold_minutes": 45},
    {"policy_idx": 3, "stop_loss_pct": 0.50, "take_profit_pct": 2.00, "max_hold_minutes": 90},
    {"policy_idx": 4, "stop_loss_pct": 1.00, "take_profit_pct": 3.00, "max_hold_minutes": 120},
    {"policy_idx": 5, "stop_loss_pct": 1.00, "take_profit_pct": 9.99, "max_hold_minutes": 384},
    {"policy_idx": 6, "stop_loss_pct": 1.00, "take_profit_pct": 99.0, "max_hold_minutes": 384},
)
# Pinned 2026-07-07 after encoding archaeology: Databento CBBO wrote absent
# bids as 0.00 through 2025-02-19 and as null from 2025-02-20 onward. Both
# encodings mean "no one will pay anything right now"; the exit-path label
# convention treats them identically as an executable 0.00 (worst-case honest
# for a long option). Entry tradability and features keep NaN semantics.
PINNED_NO_BID_CONVENTION = {
    "exit_path_absent_bid": "executable_zero",
    "vendor_encoding_boundary": "2025-02-20 null replaces 0.00 in cbbo bid_px_00",
    "scope": "labels only; entry filters and features unchanged",
}
CBBO_STAMPING_ASSUMPTION = {
    "schema": "Databento OPRA cbbo-1m",
    "timestamp_policy": "minute_end_interval_stamp",
    "causal_interpretation": (
        "A record at HH:MM:00 is treated as the last BBO for the completed "
        "[HH:MM-1, HH:MM) interval and is available to a decision at HH:MM:00."
    ),
}


@dataclass(frozen=True)
class AcceptanceThresholds:
    min_ladder_shape_ok_share: float = 1.00
    min_tradable_minute_share: float = 0.50
    min_mean_tradable_candidates: float = 10.0
    min_near_atm_tradable_share: float = 0.50
    # Calibrated 2026-07-06 on the Oct 2024 - Jun 2025 v3.2 rerun after
    # candidate-mask-aware label metrics: finite min=0.9999, nonzero min=0.9711,
    # positive min=0.2752, negative min=0.5462 across 186 sessions. These floors
    # clear the observed corpus with margin while still catching all-zero,
    # placeholder, or polarity-collapsed labels.
    min_label_finite_share: float = 0.95
    min_label_nonzero_share: float = 0.95
    # Recalibrated 2026-07-07 on the menu-v2 v3.5 corpus (314 sessions):
    # per-session positive share spans 0.188 (2025-04-07 crash day, honest)
    # to ~0.5, p5=0.244. 0.15 clears the observed honest tail with margin;
    # the placeholder pathology this floor exists to catch sits at 0.00.
    min_label_positive_share: float = 0.15
    min_label_negative_share: float = 0.40
    min_label_spot_check_count: int = 5
    min_vix_close_finite_share: float = 0.95
    min_context_reconstruction_match_share: float = 1.00
    min_context_window_sample_match_share: float = 1.00
    min_entry_quote_match_share: float = 1.00
    min_entry_quote_sweep_match_share: float = 1.00
    min_ladder_quote_sweep_match_share: float = 1.00
    max_missing_processed_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--official-spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--official-vix-dir", type=Path, default=DEFAULT_VIX_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--era-manifest", type=Path, default=DEFAULT_ERA_MANIFEST)
    parser.add_argument("--role-policy", type=Path, default=DEFAULT_ROLE_POLICY)
    parser.add_argument("--role", default="diagnostics_only")
    parser.add_argument("--min-ladder-shape-ok-share", type=float, default=1.00)
    parser.add_argument("--min-tradable-minute-share", type=float, default=0.50)
    parser.add_argument("--min-mean-tradable-candidates", type=float, default=10.0)
    parser.add_argument("--min-near-atm-tradable-share", type=float, default=0.50)
    parser.add_argument("--min-label-finite-share", type=float, default=0.95)
    parser.add_argument("--min-label-nonzero-share", type=float, default=0.95)
    parser.add_argument("--min-label-positive-share", type=float, default=0.15)
    parser.add_argument("--min-label-negative-share", type=float, default=0.40)
    parser.add_argument("--min-label-spot-check-count", type=int, default=5)
    parser.add_argument("--min-vix-close-finite-share", type=float, default=0.95)
    parser.add_argument("--min-context-reconstruction-match-share", type=float, default=1.00)
    parser.add_argument("--min-context-window-sample-match-share", type=float, default=1.00)
    parser.add_argument("--min-entry-quote-match-share", type=float, default=1.00)
    parser.add_argument("--min-entry-quote-sweep-match-share", type=float, default=1.00)
    parser.add_argument("--min-ladder-quote-sweep-match-share", type=float, default=1.00)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


REGISTRY_HASH_FIELDS = (
    "schema_version",
    "verifier_version",
    "minimum_fold_placement_verifier_version",
    "status",
    "batch_id",
    "evidence_grade",
    "start_date",
    "end_date",
    "session_count",
    "pass_count",
    "report_only_count",
    "fail_count",
    "thresholds",
    "thresholds_are_defaults",
    "fee_model",
    "max_quote_age_seconds",
    "no_bid_convention",
    "label_policies",
    "forced_flat_before_et",
    "cbbo_stamping_assumption",
    "governance_checks",
    "batch_checks",
    "batch_label_outcome_reasons",
    "labels_used_for_strategy_selection",
    "pnl_used_for_strategy_selection",
    "strategy_metrics_used",
    "era_manifest_hash",
    "role_policy_hash",
    "sessions",
    "placement_predicates",
)


def registry_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the self-hashed, non-self-referential registry payload."""
    return {field: payload.get(field) for field in REGISTRY_HASH_FIELDS}


def compute_registry_hash(payload: dict[str, Any]) -> str:
    return stable_hash(registry_hash_payload(payload))


def sessions_between(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    sessions: list[str] = []
    while cursor <= final:
        session = cursor.isoformat()
        if cursor.weekday() < 5 and session not in MARKET_HOLIDAYS:
            sessions.append(session)
        cursor += timedelta(days=1)
    return sessions


def product_paths(raw_root: Path, session: str) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for product, ((_, directory, _), suffix) in PRODUCTS.items():
        root = raw_root / "databento" / directory
        out[product] = {
            "parquet": root / f"{session}.{suffix}.parquet",
            "dbn": root / f"{session}.{suffix}.dbn.zst",
        }
    return out


def parquet_row_count(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def dbn_row_count(path: Path) -> int | None:
    try:
        import databento as db

        return int(len(db.DBNStore.from_file(str(path)).to_df()))
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_file_path(directory: Path, session: str, symbol: str) -> Path:
    """Return the canonical index file, including v2 official-index fallback names."""
    simple = directory / f"{session}.parquet"
    if simple.exists():
        return simple
    official = directory / f"{session}.official_{symbol.lower()}.parquet"
    if official.exists():
        return official
    return simple


def expected_decision_minutes(session: str) -> int:
    close_text = EARLY_CLOSE_TIMES_ET.get(session, "16:00")
    hour, minute = [int(part) for part in close_text.split(":", 1)]
    session_date = date.fromisoformat(session)
    first = datetime.combine(session_date, time(9, 31), tzinfo=NY)
    if close_text == "16:00":
        last = datetime.combine(session_date, time(15, 30), tzinfo=NY)
    else:
        last = datetime.combine(session_date, time(hour, minute), tzinfo=NY) - timedelta(minutes=1)
    return max(int((last - first).total_seconds() // 60) + 1, 0)


def expected_decision_minutes_for_contract(session: str, feature_contract_version: str) -> int:
    minutes = expected_decision_minutes(session)
    if feature_contract_version == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED:
        # v2 uses completed index context through t-1m and refuses to fabricate
        # a 09:31 ET row from unavailable pre-open context.
        return max(minutes - 1, 0)
    return minutes


def expected_decision_bounds(session: str) -> tuple[datetime, datetime]:
    close_text = EARLY_CLOSE_TIMES_ET.get(session, "16:00")
    hour, minute = [int(part) for part in close_text.split(":", 1)]
    session_date = date.fromisoformat(session)
    first = datetime.combine(session_date, time(9, 31), tzinfo=NY)
    if close_text == "16:00":
        last = datetime.combine(session_date, time(15, 30), tzinfo=NY)
    else:
        last = datetime.combine(session_date, time(hour, minute), tzinfo=NY) - timedelta(minutes=1)
    return first.astimezone(ZoneInfo("UTC")), last.astimezone(ZoneInfo("UTC"))


def expected_decision_bounds_for_contract(session: str, feature_contract_version: str) -> tuple[datetime, datetime]:
    first, last = expected_decision_bounds(session)
    if feature_contract_version == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED:
        return first + timedelta(minutes=1), last
    return first, last


def to_utc_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def load_pickle_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def finite_share(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def thresholds_are_defaults(thresholds: AcceptanceThresholds) -> bool:
    return asdict(thresholds) == asdict(AcceptanceThresholds())


def processed_quality(session: str, processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / f"{session}.pkl"
    rows = load_pickle_rows(path)
    if not rows:
        return {
            "processed_path": str(path),
            "processed_exists": path.exists(),
            "processed_bytes": path.stat().st_size if path.exists() else 0,
            "processed_sha256": file_sha256(path) if path.exists() else "",
            "neural_rows": 0,
            "expected_decision_minutes": expected_decision_minutes(session),
            "ladder_shape_ok_share": 0.0,
            "tradable_minute_share": 0.0,
            "mean_tradable_candidates": 0.0,
            "min_tradable_candidates": 0,
            "near_atm_tradable_share": 0.0,
            "label_total_cell_count": 0,
            "label_candidate_cell_count": 0,
            "label_finite_share": 0.0,
            "label_nonzero_share": 0.0,
            "label_positive_share": 0.0,
            "label_negative_share": 0.0,
            "first_decision_time": None,
            "last_decision_time": None,
            "feature_contract_version": "",
            "decision_grid": "",
        }
    ladder_shape_ok = 0
    tradable_minutes = 0
    tradable_candidate_counts: list[int] = []
    near_atm_cells = 0
    near_atm_tradable = 0
    label_values: list[np.ndarray] = []
    label_total_cell_count = 0
    label_candidate_cell_count = 0
    feature_contract_versions: set[str] = set()
    decision_grid_versions: set[str] = set()
    for row in rows:
        ladder = np.asarray(row.get("option_ladder"))
        if ladder.shape[:2] == (21, 2):
            ladder_shape_ok += 1
        mask = np.asarray(row.get("candidate_mask"), dtype=bool)
        count = int(mask.sum()) if mask.size else 0
        tradable_candidate_counts.append(count)
        if mask.size and bool(mask.any()):
            tradable_minutes += 1
        offsets = np.asarray(row.get("strike_offsets"))
        if offsets.size and mask.ndim == 2 and mask.shape[0] == offsets.size:
            near_mask = np.abs(offsets.astype(float)) <= 20.0
            near_atm_cells += int(mask[near_mask, :].size)
            near_atm_tradable += int(mask[near_mask, :].sum())
        labels = np.asarray(row.get("labels_net_pnl"), dtype=float)
        if labels.size:
            label_total_cell_count += int(labels.size)
            if mask.ndim == 2 and labels.ndim == 3 and mask.shape == labels.shape[:2]:
                masked_labels = labels[mask, :]
            else:
                masked_labels = labels.reshape(-1)
            label_candidate_cell_count += int(masked_labels.size)
            if masked_labels.size:
                label_values.append(masked_labels.reshape(-1))
        version = row.get("feature_contract_version") or row.get("feature_contract")
        if version:
            feature_contract_versions.add(str(version))
        decision_grid = row.get("decision_grid")
        if decision_grid:
            decision_grid_versions.add(str(decision_grid))
    labels_flat = np.concatenate(label_values) if label_values else np.asarray([], dtype=float)
    finite_labels = labels_flat[np.isfinite(labels_flat)]
    nonzero_labels = finite_labels[np.abs(finite_labels) > 1e-9]
    feature_contract_version = ",".join(sorted(feature_contract_versions))
    expected = expected_decision_minutes_for_contract(session, feature_contract_version)
    return {
        "processed_path": str(path),
        "processed_exists": path.exists(),
        "processed_bytes": path.stat().st_size if path.exists() else 0,
        "processed_sha256": file_sha256(path) if path.exists() else "",
        "neural_rows": int(len(rows)),
        "expected_decision_minutes": expected,
        "ladder_shape_ok_share": float(ladder_shape_ok / len(rows)),
        "tradable_minute_share": float(tradable_minutes / len(rows)),
        "mean_tradable_candidates": float(np.mean(tradable_candidate_counts)) if tradable_candidate_counts else 0.0,
        "min_tradable_candidates": int(min(tradable_candidate_counts)) if tradable_candidate_counts else 0,
        "near_atm_tradable_share": float(near_atm_tradable / near_atm_cells) if near_atm_cells else 0.0,
        "label_total_cell_count": int(label_total_cell_count),
        "label_candidate_cell_count": int(label_candidate_cell_count),
        "label_finite_share": finite_share(labels_flat),
        "label_nonzero_share": float(len(nonzero_labels) / len(finite_labels)) if len(finite_labels) else 0.0,
        "label_positive_share": float((finite_labels > 0).mean()) if len(finite_labels) else 0.0,
        "label_negative_share": float((finite_labels < 0).mean()) if len(finite_labels) else 0.0,
        "first_decision_time": rows[0].get("decision_time").isoformat() if hasattr(rows[0].get("decision_time"), "isoformat") else str(rows[0].get("decision_time")),
        "last_decision_time": rows[-1].get("decision_time").isoformat() if hasattr(rows[-1].get("decision_time"), "isoformat") else str(rows[-1].get("decision_time")),
        "feature_contract_version": feature_contract_version,
        "decision_grid": ",".join(sorted(decision_grid_versions)),
    }


def index_quality(session: str, spx_dir: Path, vix_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for symbol, directory in (("spx", spx_dir), ("vix", vix_dir)):
        path = index_file_path(directory, session, symbol)
        count = parquet_row_count(path) if path.exists() else None
        out[f"{symbol}_path"] = str(path)
        out[f"{symbol}_exists"] = path.exists()
        out[f"{symbol}_bytes"] = path.stat().st_size if path.exists() else 0
        out[f"{symbol}_sha256"] = file_sha256(path) if path.exists() else ""
        out[f"{symbol}_rows"] = count or 0
        try:
            frame = pd.read_parquet(path)
            times = pd.to_datetime(frame["event_time"], utc=True)
            out[f"{symbol}_first_event_time"] = times.min().isoformat()
            out[f"{symbol}_last_event_time"] = times.max().isoformat()
            source_columns = {}
            for column in ("context_source", "is_derived", "is_proxy", "is_official_index_data"):
                if column in frame.columns:
                    source_columns[column] = sorted(str(value) for value in frame[column].dropna().unique())
            out[f"{symbol}_source_columns"] = source_columns
        except Exception:
            out[f"{symbol}_first_event_time"] = None
            out[f"{symbol}_last_event_time"] = None
            out[f"{symbol}_source_columns"] = {}
    return out


def raw_quality(session: str, raw_root: Path) -> dict[str, Any]:
    paths = product_paths(raw_root, session)
    products: dict[str, Any] = {}
    for product, files in paths.items():
        parquet_path = files["parquet"]
        dbn_path = files["dbn"]
        parquet_rows = parquet_row_count(parquet_path) if parquet_path.exists() else None
        dbn_rows = dbn_row_count(dbn_path) if dbn_path.exists() else None
        products[product] = {
            "parquet_path": str(parquet_path),
            "dbn_path": str(dbn_path),
            "parquet_exists": parquet_path.exists(),
            "dbn_exists": dbn_path.exists(),
            "parquet_bytes": parquet_path.stat().st_size if parquet_path.exists() else 0,
            "dbn_bytes": dbn_path.stat().st_size if dbn_path.exists() else 0,
            "parquet_rows": parquet_rows,
            "dbn_rows": dbn_rows,
            "row_count_match": parquet_rows is not None and dbn_rows is not None and parquet_rows == dbn_rows,
            "parquet_sha256": file_sha256(parquet_path) if parquet_path.exists() else "",
            "dbn_sha256": file_sha256(dbn_path) if dbn_path.exists() else "",
        }
    return products


def is_early_close_session(session: str) -> bool:
    return session in EARLY_CLOSE_TIMES_ET


def databento_symbol_from_contract_id(contract_id: str) -> str | None:
    match = CONTRACT_ID_RE.match(str(contract_id))
    if not match:
        return None
    expiry = match.group("expiry")
    strike = float(match.group("strike"))
    strike_int = int(round(strike * 1000.0))
    return f"{match.group('root')}  {expiry[2:]}{match.group('right')}{strike_int:08d}"


def raw_cbbo_frame(raw_root: Path, session: str) -> pd.DataFrame:
    path = raw_root / "databento" / "opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "ts_recv" in frame.columns:
        quote_time = frame["ts_recv"]
    elif frame.index.name == "ts_recv":
        quote_time = frame.index
    else:
        quote_time = frame.index
    frame = frame.copy()
    frame["quote_time"] = pd.to_datetime(quote_time, utc=True)
    frame["bid"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    return frame[["quote_time", "symbol", "bid", "ask", "mid"]].sort_values(
        ["symbol", "quote_time"]
    )


def raw_entry_quote_at(
    contract_quotes: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    max_quote_age_seconds: float,
) -> dict[str, Any]:
    eligible = contract_quotes[contract_quotes["quote_time"] <= decision_time].sort_values("quote_time")
    if eligible.empty:
        return {"status": "missing_before_decision"}
    row = eligible.iloc[-1]
    quote_time = pd.Timestamp(row["quote_time"])
    age_seconds = float((decision_time - quote_time).total_seconds())
    if age_seconds > float(max_quote_age_seconds):
        return {
            "status": "stale",
            "quote_time": quote_time.isoformat(),
            "quote_age_seconds": age_seconds,
        }
    bid = float(row["bid"]) if pd.notna(row.get("bid")) else np.nan
    ask = float(row["ask"]) if pd.notna(row.get("ask")) else np.nan
    mid = float(row["mid"]) if pd.notna(row.get("mid")) else np.nan
    return {
        "status": "ok",
        "quote_time": quote_time.isoformat(),
        "quote_age_seconds": age_seconds,
        "bid": bid,
        "ask": ask,
        "mid": mid,
    }


def _float_match(observed: Any, expected: Any, *, atol: float = 1e-9) -> bool:
    try:
        observed_float = float(observed)
        expected_float = float(expected)
    except (TypeError, ValueError):
        return observed is None and expected is None
    if not np.isfinite(observed_float) and not np.isfinite(expected_float):
        return True
    return bool(abs(observed_float - expected_float) <= atol)


def _time_match(observed: Any, expected: Any) -> bool:
    observed_dt = to_utc_datetime(observed)
    expected_dt = to_utc_datetime(expected)
    return observed_dt is not None and expected_dt is not None and observed_dt == expected_dt


def entry_ladder_sweep_quality(
    session: str,
    *,
    raw_root: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    if not rows:
        return {
            "masked_candidate_count": 0,
            "entry_quote_match_count": 0,
            "entry_quote_mismatch_count": 0,
            "entry_quote_missing_count": 0,
            "entry_quote_stale_count": 0,
            "entry_quote_match_share": 0.0,
            "ladder_quote_match_count": 0,
            "ladder_quote_mismatch_count": 0,
            "ladder_quote_match_share": 0.0,
            "examples": [],
        }
    cbbo = raw_cbbo_frame(raw_root, session)
    if cbbo.empty:
        return {
            "masked_candidate_count": 0,
            "entry_quote_match_count": 0,
            "entry_quote_mismatch_count": 0,
            "entry_quote_missing_count": 0,
            "entry_quote_stale_count": 0,
            "entry_quote_match_share": 0.0,
            "ladder_quote_match_count": 0,
            "ladder_quote_mismatch_count": 0,
            "ladder_quote_match_share": 0.0,
            "examples": [{"reason": "missing_raw_cbbo"}],
        }
    by_symbol = {
        str(symbol): frame.sort_values("quote_time")
        for symbol, frame in cbbo.groupby("symbol", sort=False)
    }
    masked_candidate_count = 0
    entry_quote_match_count = 0
    entry_quote_mismatch_count = 0
    entry_quote_missing_count = 0
    entry_quote_stale_count = 0
    ladder_quote_match_count = 0
    ladder_quote_mismatch_count = 0
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        decision_time = to_utc_datetime(row.get("decision_time"))
        if decision_time is None:
            continue
        mask = np.asarray(row.get("candidate_mask"), dtype=bool)
        contract_ids = np.asarray(row.get("contract_ids"), dtype=object)
        ladder = np.asarray(row.get("option_ladder"), dtype=float)
        metadata = row.get("contract_quote_metadata") if isinstance(row.get("contract_quote_metadata"), dict) else {}
        if mask.ndim != 2 or contract_ids.shape != mask.shape or ladder.shape[:2] != mask.shape:
            if len(examples) < 10:
                examples.append({"row_idx": row_idx, "reason": "bad_candidate_shapes"})
            continue
        for strike_idx, right_idx in np.argwhere(mask):
            masked_candidate_count += 1
            contract_id = str(contract_ids[strike_idx, right_idx])
            symbol = databento_symbol_from_contract_id(contract_id)
            contract_quotes = by_symbol.get(str(symbol))
            if contract_quotes is None or contract_quotes.empty:
                entry_quote_missing_count += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "row_idx": row_idx,
                            "contract_id": contract_id,
                            "reason": "missing_raw_path",
                            "raw_symbol": symbol,
                        }
                    )
                continue
            raw_quote = raw_entry_quote_at(
                contract_quotes,
                decision_time=pd.Timestamp(decision_time),
                max_quote_age_seconds=PINNED_MAX_QUOTE_AGE_SECONDS,
            )
            if raw_quote.get("status") != "ok":
                if raw_quote.get("status") == "stale":
                    entry_quote_stale_count += 1
                else:
                    entry_quote_missing_count += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "row_idx": row_idx,
                            "contract_id": contract_id,
                            "reason": f"raw_quote_{raw_quote.get('status')}",
                            "raw_symbol": symbol,
                            "raw_quote_time": raw_quote.get("quote_time"),
                        }
                    )
                continue
            item = metadata.get(contract_id) or {}
            metadata_matches = (
                _float_match(item.get("bid"), raw_quote.get("bid"))
                and _float_match(item.get("ask"), raw_quote.get("ask"))
                and _float_match(item.get("mid"), raw_quote.get("mid"))
                and _time_match(item.get("source_quote_time"), raw_quote.get("quote_time"))
            )
            if metadata_matches:
                entry_quote_match_count += 1
            else:
                entry_quote_mismatch_count += 1
            ladder_matches = True
            for field, feature_idx in OPTION_LADDER_QUOTE_INDEX.items():
                ladder_matches = ladder_matches and _float_match(
                    ladder[strike_idx, right_idx, feature_idx],
                    raw_quote.get(field),
                )
            if ladder_matches:
                ladder_quote_match_count += 1
            else:
                ladder_quote_mismatch_count += 1
            if (not metadata_matches or not ladder_matches) and len(examples) < 10:
                examples.append(
                    {
                        "row_idx": row_idx,
                        "decision_time": pd.Timestamp(decision_time).isoformat(),
                        "contract_id": contract_id,
                        "raw_symbol": symbol,
                        "reason": "metadata_or_ladder_mismatch",
                        "metadata_matches": metadata_matches,
                        "ladder_matches": ladder_matches,
                        "metadata": {
                            "source_quote_time": item.get("source_quote_time"),
                            "bid": item.get("bid"),
                            "ask": item.get("ask"),
                            "mid": item.get("mid"),
                        },
                        "ladder": {
                            field: float(ladder[strike_idx, right_idx, idx])
                            if np.isfinite(ladder[strike_idx, right_idx, idx])
                            else None
                            for field, idx in OPTION_LADDER_QUOTE_INDEX.items()
                        },
                        "raw": {
                            "quote_time": raw_quote.get("quote_time"),
                            "bid": raw_quote.get("bid"),
                            "ask": raw_quote.get("ask"),
                            "mid": raw_quote.get("mid"),
                        },
                    }
                )
    return {
        "masked_candidate_count": int(masked_candidate_count),
        "entry_quote_match_count": int(entry_quote_match_count),
        "entry_quote_mismatch_count": int(entry_quote_mismatch_count),
        "entry_quote_missing_count": int(entry_quote_missing_count),
        "entry_quote_stale_count": int(entry_quote_stale_count),
        "entry_quote_match_share": float(entry_quote_match_count / masked_candidate_count)
        if masked_candidate_count
        else 0.0,
        "ladder_quote_match_count": int(ladder_quote_match_count),
        "ladder_quote_mismatch_count": int(ladder_quote_mismatch_count),
        "ladder_quote_match_share": float(ladder_quote_match_count / masked_candidate_count)
        if masked_candidate_count
        else 0.0,
        "examples": examples,
    }


def _normalize_vendor_index_frame(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["event_time", "symbol", "close", "volume"])
    frame = pd.read_parquet(path)
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    frame = frame[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["event_time", "symbol", "close", "volume"])
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    return frame.sort_values("event_time").reset_index(drop=True)


def _independent_index_close_at(frame: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    if frame.empty:
        return np.nan
    eligible = frame[frame["event_time"] <= cutoff]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1]["close"])


def _independent_market_features(
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> np.ndarray:
    if tuple(MARKET_FEATURE_NAMES) != PINNED_MARKET_FEATURE_NAMES:
        raise ValueError(
            f"market feature names changed: {tuple(MARKET_FEATURE_NAMES)!r} != {PINNED_MARKET_FEATURE_NAMES!r}"
        )
    spx_hist = spx_bars[spx_bars["event_time"] <= cutoff]
    if spx_hist.empty:
        return np.full(len(MARKET_FEATURE_NAMES), np.nan, dtype=float)
    spx_close = float(spx_hist.iloc[-1]["close"])
    vix_close = _independent_index_close_at(vix_bars, cutoff)
    volume = pd.to_numeric(spx_hist["volume"], errors="coerce").fillna(0.0).astype(float)
    closes = pd.to_numeric(spx_hist["close"], errors="coerce").astype(float)
    if float(volume.sum()) > 0:
        vwap = float((closes * volume).sum() / volume.sum())
    else:
        vwap = float(closes.mean())
    session_open = float(closes.iloc[0])
    session_high = float(closes.max())
    session_low = float(closes.min())
    session_range = session_high - session_low
    omar = (spx_close - session_open) / session_range if session_range > 0 else 0.0
    momentum_5 = spx_close - float(closes.iloc[-6]) if len(closes) >= 6 else 0.0
    momentum_15 = spx_close - float(closes.iloc[-16]) if len(closes) >= 16 else 0.0
    return np.asarray(
        [spx_close, vix_close, vwap, omar, session_range, momentum_5, momentum_15],
        dtype=float,
    )


def _independent_market_window(
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    minutes: int = 30,
    session_only: bool = True,
) -> np.ndarray:
    if session_only:
        local_day = cutoff.tz_convert(NY).date()
        if not spx_bars.empty:
            spx_local = pd.to_datetime(spx_bars["event_time"], utc=True).dt.tz_convert(NY)
            spx_bars = spx_bars[spx_local.dt.date == local_day]
        if not vix_bars.empty:
            vix_local = pd.to_datetime(vix_bars["event_time"], utc=True).dt.tz_convert(NY)
            vix_bars = vix_bars[vix_local.dt.date == local_day]
    start = cutoff - pd.Timedelta(minutes=int(minutes) - 1)
    grid = pd.date_range(start=start, end=cutoff, freq="min", tz="UTC")
    return np.vstack([_independent_market_features(spx_bars, vix_bars, ts) for ts in grid])


def _round_to_five(value: float) -> int:
    return int(round(float(value) / 5.0) * 5)


def context_reconstruction_quality(
    session: str,
    *,
    spx_dir: Path,
    vix_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    if not rows:
        return {
            "row_count": 0,
            "feature_match_count": 0,
            "feature_mismatch_count": 0,
            "feature_match_share": 0.0,
            "atm_strike_match_count": 0,
            "atm_strike_mismatch_count": 0,
            "atm_strike_match_share": 0.0,
            "window_sample_count": 0,
            "window_sample_match_count": 0,
            "window_sample_mismatch_count": 0,
            "window_sample_match_share": 0.0,
            "vix_close_finite_share": 0.0,
            "examples": [],
        }
    spx = _normalize_vendor_index_frame(index_file_path(spx_dir, session, "spx"), "SPX")
    vix = _normalize_vendor_index_frame(index_file_path(vix_dir, session, "vix"), "VIX")
    comparable = 0
    feature_matches = 0
    feature_mismatches = 0
    atm_matches = 0
    atm_mismatches = 0
    vix_finite = 0
    examples: list[dict[str, Any]] = []
    sample_indices = sorted(
        set(
            [
                0,
                min(5, len(rows) - 1),
                min(29, len(rows) - 1),
                len(rows) // 2,
                len(rows) - 1,
            ]
        )
    )
    window_sample_count = 0
    window_sample_matches = 0
    window_sample_mismatches = 0
    for row_idx, row in enumerate(rows):
        decision_time = to_utc_datetime(row.get("decision_time"))
        if decision_time is None:
            continue
        cutoff = pd.Timestamp(decision_time) - pd.Timedelta(minutes=1)
        expected_features = _independent_market_features(spx, vix, cutoff)
        row_window = np.asarray(row.get("market_window"), dtype=float)
        if row_window.ndim != 2 or row_window.shape[1] != len(MARKET_FEATURE_NAMES):
            feature_mismatches += 1
            atm_mismatches += 1
            examples.append({"row_idx": row_idx, "reason": "bad_market_window_shape"})
            continue
        comparable += 1
        if np.isfinite(expected_features[1]):
            vix_finite += 1
        observed_features = row_window[-1]
        feature_match = bool(
            np.allclose(observed_features, expected_features, rtol=0.0, atol=1e-6, equal_nan=True)
        )
        feature_matches += int(feature_match)
        feature_mismatches += int(not feature_match)
        expected_atm = _round_to_five(expected_features[0]) if np.isfinite(expected_features[0]) else None
        observed_atm = row.get("atm_strike")
        try:
            observed_atm_int = int(observed_atm)
        except (TypeError, ValueError):
            observed_atm_int = None
        atm_match = expected_atm is not None and observed_atm_int == int(expected_atm)
        atm_matches += int(atm_match)
        atm_mismatches += int(not atm_match)
        if (not feature_match or not atm_match) and len(examples) < 10:
            examples.append(
                {
                    "row_idx": row_idx,
                    "decision_time": pd.Timestamp(decision_time).isoformat(),
                    "expected_context_time": cutoff.isoformat(),
                    "feature_match": feature_match,
                    "atm_match": atm_match,
                    "observed_features": [
                        None if not np.isfinite(value) else float(value) for value in observed_features
                    ],
                    "expected_features": [
                        None if not np.isfinite(value) else float(value) for value in expected_features
                    ],
                    "observed_atm_strike": observed_atm_int,
                    "expected_atm_strike": expected_atm,
                }
            )
        if row_idx in sample_indices:
            window_sample_count += 1
            expected_window = _independent_market_window(spx, vix, cutoff, minutes=row_window.shape[0], session_only=True)
            window_match = bool(
                row_window.shape == expected_window.shape
                and np.allclose(row_window, expected_window, rtol=0.0, atol=1e-6, equal_nan=True)
            )
            window_sample_matches += int(window_match)
            window_sample_mismatches += int(not window_match)
    return {
        "row_count": int(comparable),
        "feature_match_count": int(feature_matches),
        "feature_mismatch_count": int(feature_mismatches),
        "feature_match_share": float(feature_matches / comparable) if comparable else 0.0,
        "atm_strike_match_count": int(atm_matches),
        "atm_strike_mismatch_count": int(atm_mismatches),
        "atm_strike_match_share": float(atm_matches / comparable) if comparable else 0.0,
        "window_sample_count": int(window_sample_count),
        "window_sample_match_count": int(window_sample_matches),
        "window_sample_mismatch_count": int(window_sample_mismatches),
        "window_sample_match_share": float(window_sample_matches / window_sample_count) if window_sample_count else 0.0,
        "vix_close_finite_share": float(vix_finite / comparable) if comparable else 0.0,
        "examples": examples,
    }


def _raw_path_label(
    contract_quotes: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    entry_ask: float,
    policy_idx: int,
) -> tuple[float, str]:
    policy = PINNED_LABEL_POLICIES[int(policy_idx)]
    max_hold = decision_time + pd.Timedelta(minutes=int(policy["max_hold_minutes"]))
    hour, minute = [int(part) for part in PINNED_FORCED_FLAT_BEFORE_ET.split(":", 1)]
    forced_local = pd.Timestamp.combine(
        decision_time.tz_convert(NY).date(),
        time(hour, minute),
    ).tz_localize(NY)
    deadline = min(max_hold, forced_local.tz_convert("UTC"))
    future = contract_quotes[
        (contract_quotes["quote_time"] > decision_time)
        & (contract_quotes["quote_time"] <= deadline)
    ].sort_values("quote_time")
    if future.empty:
        return np.nan, "missing_future_path"
    stop_bid = entry_ask * (1.0 - float(policy["stop_loss_pct"]))
    target_bid = entry_ask * (1.0 + float(policy["take_profit_pct"]))
    exit_row = future.iloc[-1]
    reason = "time_exit"
    for _, row in future.iterrows():
        # PINNED_NO_BID_CONVENTION: absent bid is an executable 0.00 on exit.
        bid = 0.0 if pd.isna(row.get("bid")) else float(row.get("bid"))
        if bid <= stop_bid:
            exit_row = row
            reason = "stop_hit"
            break
        if bid >= target_bid:
            exit_row = row
            reason = "target_hit"
            break
    if reason == "time_exit" and deadline < decision_time + pd.Timedelta(minutes=int(policy["max_hold_minutes"])):
        reason = "forced_flat_capped"
    exit_bid = 0.0 if pd.isna(exit_row["bid"]) else float(exit_row["bid"])
    return (exit_bid - entry_ask) * PINNED_CONTRACT_MULTIPLIER, reason


def _forced_flat_reachable(decision_time: datetime, policy_idx: int) -> bool:
    max_hold_minutes_by_policy = {
        int(policy["policy_idx"]): int(policy["max_hold_minutes"])
        for policy in PINNED_LABEL_POLICIES
    }
    hold = max_hold_minutes_by_policy.get(int(policy_idx))
    if hold is None:
        return False
    local = pd.Timestamp(decision_time).tz_convert(NY)
    hour, minute = [int(part) for part in PINNED_FORCED_FLAT_BEFORE_ET.split(":", 1)]
    forced = pd.Timestamp.combine(local.date(), time(hour, minute)).tz_localize(NY)
    return local + pd.Timedelta(minutes=hold) > forced


def _sample_label_tuples(rows: list[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    pools: dict[tuple[int, str], list[dict[str, Any]]] = {}
    by_right: dict[str, list[dict[str, Any]]] = {"C": [], "P": []}
    all_candidates: list[dict[str, Any]] = []
    forced_flat_candidates: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        decision_time = to_utc_datetime(row.get("decision_time"))
        if decision_time is None:
            continue
        mask = np.asarray(row.get("candidate_mask"), dtype=bool)
        labels = np.asarray(row.get("labels_net_pnl"), dtype=float)
        contract_ids = np.asarray(row.get("contract_ids"), dtype=object)
        strike_offsets = np.asarray(row.get("strike_offsets"), dtype=float)
        if mask.ndim != 2 or labels.ndim != 3 or contract_ids.shape != mask.shape:
            continue
        metadata = row.get("contract_quote_metadata") if isinstance(row.get("contract_quote_metadata"), dict) else {}
        for strike_idx in range(mask.shape[0]):
            offset = float(strike_offsets[strike_idx]) if strike_idx < len(strike_offsets) else np.nan
            offset_bucket = "near" if abs(offset) <= 20.0 else "far"
            for right_idx in range(mask.shape[1]):
                if not bool(mask[strike_idx, right_idx]):
                    continue
                contract_id = str(contract_ids[strike_idx, right_idx])
                if not contract_id or contract_id == "None":
                    continue
                entry_ask = (metadata.get(contract_id) or {}).get("ask")
                if entry_ask is None or not np.isfinite(float(entry_ask)):
                    continue
                entry_bid = (metadata.get(contract_id) or {}).get("bid")
                entry_mid = (metadata.get(contract_id) or {}).get("mid")
                source_quote_time = (metadata.get(contract_id) or {}).get("source_quote_time")
                for policy_idx in range(labels.shape[2]):
                    expected = float(labels[strike_idx, right_idx, policy_idx])
                    if not np.isfinite(expected):
                        continue
                    item = {
                        "row_idx": row_idx,
                        "decision_time": decision_time.isoformat(),
                        "contract_id": contract_id,
                        "strike_idx": strike_idx,
                        "right_idx": right_idx,
                        "right": "C" if int(right_idx) == 0 else "P",
                        "policy_idx": policy_idx,
                        "offset_bucket": offset_bucket,
                        "source_quote_time": source_quote_time,
                        "entry_bid": float(entry_bid) if entry_bid is not None and np.isfinite(float(entry_bid)) else None,
                        "entry_ask": float(entry_ask),
                        "entry_mid": float(entry_mid) if entry_mid is not None and np.isfinite(float(entry_mid)) else None,
                        "expected_label": expected,
                        "forced_flat_reachable": _forced_flat_reachable(decision_time, policy_idx),
                    }
                    pools.setdefault((policy_idx, offset_bucket), []).append(item)
                    by_right[item["right"]].append(item)
                    all_candidates.append(item)
                    if item["forced_flat_reachable"]:
                        forced_flat_candidates.append(item)
    desired = [
        (int(policy["policy_idx"]), bucket)
        for bucket in ("near", "far")
        for policy in PINNED_LABEL_POLICIES
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    budget = max(max_samples, 2 * len(PINNED_LABEL_POLICIES) * 2 + 8)

    def add(item: dict[str, Any]) -> None:
        identity = (item["row_idx"], item["strike_idx"], item["right_idx"], item["policy_idx"])
        if identity not in seen and len(selected) < budget:
            selected.append(item)
            seen.add(identity)

    for key in desired:
        candidates = pools.get(key) or []
        if not candidates:
            continue
        for quantile in (0.25, 0.75):
            idx = int(round((len(candidates) - 1) * quantile))
            add(candidates[idx])
    for right in ("C", "P"):
        candidates = by_right.get(right) or []
        if candidates:
            add(candidates[len(candidates) // 2])
    if all_candidates:
        sorted_by_label = sorted(all_candidates, key=lambda item: float(item["expected_label"]))
        add(sorted_by_label[0])
        add(sorted_by_label[-1])
    if forced_flat_candidates:
        add(forced_flat_candidates[len(forced_flat_candidates) // 2])
    for candidates in pools.values():
        for item in candidates:
            add(item)
            if len(selected) >= budget:
                return selected
    return selected


def label_spot_check_quality(
    session: str,
    *,
    raw_root: Path,
    processed_dir: Path,
    min_samples: int,
) -> dict[str, Any]:
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    if not rows:
        return {
            "sample_count": 0,
            "match_count": 0,
            "mismatch_count": 0,
            "missing_raw_path_count": 0,
            "entry_quote_match_count": 0,
            "entry_quote_mismatch_count": 0,
            "entry_quote_missing_count": 0,
            "entry_quote_match_share": 0.0,
            "match_share": 0.0,
            "fee_model": "unknown",
            "outcome_reasons": {},
            "sampled_policy_offset_cells": [],
            "sampled_rights": [],
            "forced_flat_reachable_sample_count": 0,
            "samples": [],
        }
    cbbo = raw_cbbo_frame(raw_root, session)
    samples = _sample_label_tuples(rows, max_samples=max(min_samples, 5))
    if cbbo.empty:
        return {
            "sample_count": len(samples),
            "match_count": 0,
            "mismatch_count": 0,
            "missing_raw_path_count": len(samples),
            "entry_quote_match_count": 0,
            "entry_quote_mismatch_count": 0,
            "entry_quote_missing_count": len(samples),
            "entry_quote_match_share": 0.0,
            "match_share": 0.0,
            "fee_model": "unknown",
            "outcome_reasons": {},
            "sampled_policy_offset_cells": sorted(
                {f"{item.get('policy_idx')}:{item.get('offset_bucket')}" for item in samples}
            ),
            "sampled_rights": sorted({str(item.get("right")) for item in samples if item.get("right")}),
            "forced_flat_reachable_sample_count": sum(1 for item in samples if item.get("forced_flat_reachable")),
            "samples": samples,
        }
    by_symbol = {
        str(symbol): frame.sort_values("quote_time")
        for symbol, frame in cbbo.groupby("symbol", sort=False)
    }
    checked: list[dict[str, Any]] = []
    match_count = 0
    mismatch_count = 0
    missing_raw_path_count = 0
    entry_quote_match_count = 0
    entry_quote_mismatch_count = 0
    entry_quote_missing_count = 0
    outcome_reasons: dict[str, int] = {}
    for sample in samples:
        symbol = databento_symbol_from_contract_id(str(sample["contract_id"]))
        contract_quotes = by_symbol.get(str(symbol))
        if contract_quotes is None or contract_quotes.empty:
            missing_raw_path_count += 1
            entry_quote_missing_count += 1
            checked.append({**sample, "raw_symbol": symbol, "status": "missing_raw_path"})
            continue
        raw_entry_quote = raw_entry_quote_at(
            contract_quotes,
            decision_time=pd.Timestamp(sample["decision_time"]),
            max_quote_age_seconds=PINNED_MAX_QUOTE_AGE_SECONDS,
        )
        if raw_entry_quote.get("status") != "ok":
            entry_quote_missing_count += 1
            checked.append(
                {
                    **sample,
                    "raw_symbol": symbol,
                    "status": "missing_entry_quote",
                    "raw_entry_quote_status": raw_entry_quote.get("status"),
                    "raw_entry_quote_time": raw_entry_quote.get("quote_time"),
                    "raw_quote_age_seconds": raw_entry_quote.get("quote_age_seconds"),
                }
            )
            continue
        quote_checks = []
        for field in ("bid", "ask", "mid"):
            observed = sample.get(f"entry_{field}")
            expected = raw_entry_quote.get(field)
            if observed is None and (expected is None or not np.isfinite(float(expected))):
                quote_checks.append(True)
            elif observed is None or expected is None:
                quote_checks.append(False)
            else:
                quote_checks.append(abs(float(observed) - float(expected)) <= 1e-9)
        observed_source_quote = to_utc_datetime(sample.get("source_quote_time"))
        expected_source_quote = to_utc_datetime(raw_entry_quote.get("quote_time"))
        source_quote_matches = observed_source_quote == expected_source_quote
        entry_quote_matches = bool(all(quote_checks) and source_quote_matches)
        entry_quote_match_count += int(entry_quote_matches)
        entry_quote_mismatch_count += int(not entry_quote_matches)
        observed, outcome_reason = _raw_path_label(
            contract_quotes,
            decision_time=pd.Timestamp(sample["decision_time"]),
            entry_ask=float(raw_entry_quote["ask"]),
            policy_idx=int(sample["policy_idx"]),
        )
        expected = float(sample["expected_label"])
        matched = bool(np.isfinite(observed) and abs(float(observed) - expected) <= 1e-9)
        match_count += int(matched)
        mismatch_count += int(not matched)
        outcome_reasons[outcome_reason] = outcome_reasons.get(outcome_reason, 0) + 1
        checked.append(
            {
                **sample,
                "raw_symbol": symbol,
                "entry_quote_status": "match" if entry_quote_matches else "mismatch",
                "raw_entry_quote_time": raw_entry_quote.get("quote_time"),
                "raw_quote_age_seconds": raw_entry_quote.get("quote_age_seconds"),
                "raw_entry_bid": raw_entry_quote.get("bid"),
                "raw_entry_ask": raw_entry_quote.get("ask"),
                "raw_entry_mid": raw_entry_quote.get("mid"),
                "observed_label": float(observed) if np.isfinite(observed) else None,
                "outcome_reason": outcome_reason,
                "status": "match" if matched else "mismatch",
            }
        )
    sample_count = len(checked)
    sampled_policy_offset_cells = sorted(
        {f"{item.get('policy_idx')}:{item.get('offset_bucket')}" for item in checked}
    )
    sampled_rights = sorted({str(item.get("right")) for item in checked if item.get("right")})
    forced_flat_reachable_sample_count = sum(1 for item in checked if item.get("forced_flat_reachable"))
    fee_model = (
        "gross_no_fees"
        if sample_count
        and match_count == sample_count
        and missing_raw_path_count == 0
        and entry_quote_mismatch_count == 0
        and entry_quote_missing_count == 0
        else "unknown"
    )
    return {
        "sample_count": sample_count,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "missing_raw_path_count": missing_raw_path_count,
        "entry_quote_match_count": entry_quote_match_count,
        "entry_quote_mismatch_count": entry_quote_mismatch_count,
        "entry_quote_missing_count": entry_quote_missing_count,
        "entry_quote_match_share": float(entry_quote_match_count / sample_count) if sample_count else 0.0,
        "match_share": float(match_count / sample_count) if sample_count else 0.0,
        "fee_model": fee_model,
        "outcome_reasons": outcome_reasons,
        "sampled_policy_offset_cells": sampled_policy_offset_cells,
        "sampled_rights": sampled_rights,
        "forced_flat_reachable_sample_count": int(forced_flat_reachable_sample_count),
        "samples": checked,
    }


def context_causality_quality(
    session: str,
    processed_dir: Path,
    *,
    feature_contract_version: str = FEATURE_CONTRACT_VERSION,
) -> dict[str, Any]:
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    expected_first, expected_last = expected_decision_bounds_for_contract(
        session,
        feature_contract_version,
    )
    if not rows:
        return {
            "first_decision_time": None,
            "last_decision_time": None,
            "expected_first_decision_time": expected_first.isoformat(),
            "expected_last_decision_time": expected_last.isoformat(),
            "first_decision_matches_calendar": False,
            "last_decision_matches_calendar": False,
            "one_minute_decision_steps": False,
            "context_lag_exact_one_minute_share": 0.0,
            "future_context_row_count": 0,
            "opening_no_leading_backfill": False,
        }
    decision_times = [to_utc_datetime(row.get("decision_time")) for row in rows]
    source_context_times = [to_utc_datetime(row.get("source_context_time")) for row in rows]
    context_last_times = [to_utc_datetime(row.get("context_last_timestamp")) for row in rows]
    valid_decision_times = [item for item in decision_times if item is not None]
    first_decision = valid_decision_times[0] if valid_decision_times else None
    last_decision = valid_decision_times[-1] if valid_decision_times else None
    one_minute_steps = all(
        (right - left) == timedelta(minutes=1)
        for left, right in zip(valid_decision_times, valid_decision_times[1:])
    )
    exact_lag_count = 0
    comparable_lag_count = 0
    future_context_row_count = 0
    for decision_time, source_context_time, context_last_time in zip(
        decision_times, source_context_times, context_last_times
    ):
        if decision_time is None:
            continue
        expected_source_time = decision_time - timedelta(minutes=1)
        if source_context_time is not None:
            comparable_lag_count += 1
            if source_context_time == expected_source_time:
                exact_lag_count += 1
            if source_context_time > expected_source_time:
                future_context_row_count += 1
        if context_last_time is not None and context_last_time > expected_source_time:
            future_context_row_count += 1
    first_row = rows[0]
    opening_context_start = to_utc_datetime(first_row.get("context_start_timestamp"))
    opening_source_context = to_utc_datetime(first_row.get("source_context_time"))
    opening_no_leading_backfill = bool(
        first_decision == expected_first
        and opening_source_context == expected_first - timedelta(minutes=1)
        and opening_context_start == opening_source_context
        and int(first_row.get("context_minute_rows") or 0) <= 1
        and not bool(first_row.get("context_ready"))
    )
    return {
        "first_decision_time": first_decision.isoformat() if first_decision else None,
        "last_decision_time": last_decision.isoformat() if last_decision else None,
        "expected_first_decision_time": expected_first.isoformat(),
        "expected_last_decision_time": expected_last.isoformat(),
        "first_decision_matches_calendar": first_decision == expected_first,
        "last_decision_matches_calendar": last_decision == expected_last,
        "one_minute_decision_steps": one_minute_steps,
        "context_lag_exact_one_minute_share": float(
            exact_lag_count / comparable_lag_count if comparable_lag_count else 0.0
        ),
        "future_context_row_count": int(future_context_row_count),
        "opening_no_leading_backfill": opening_no_leading_backfill,
    }


def index_context_gap_quality(
    session: str,
    *,
    spx_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Attribute imperfect context-lag rows to enumerated vendor SPX gaps.

    A session may qualify for report_only/missing_index_context only when the
    official SPX vendor file itself is missing a small number of expected
    source minutes, every row whose context is not exactly one minute old maps
    to one of those exact missing minutes, and the stale context is always in
    the past (never the future).
    """
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    expected_first, expected_last = expected_decision_bounds(session)
    expected_sources = pd.date_range(
        start=pd.Timestamp(expected_first) - pd.Timedelta(minutes=1),
        end=pd.Timestamp(expected_last) - pd.Timedelta(minutes=1),
        freq="min",
        tz="UTC",
    )
    spx = _normalize_vendor_index_frame(index_file_path(spx_dir, session, "spx"), "SPX")
    vendor_minutes = (
        set(pd.to_datetime(spx["event_time"], utc=True)) if not spx.empty else set()
    )
    missing = [ts for ts in expected_sources if ts not in vendor_minutes]
    missing_set = set(missing)
    imperfect_rows: list[dict[str, Any]] = []
    all_rows_attributed = True
    for row in rows:
        decision_time = to_utc_datetime(row.get("decision_time"))
        source_context_time = to_utc_datetime(row.get("source_context_time"))
        if decision_time is None or source_context_time is None:
            continue
        expected_source = pd.Timestamp(decision_time) - pd.Timedelta(minutes=1)
        source_ts = pd.Timestamp(source_context_time)
        if source_ts == expected_source:
            continue
        row_attributed = expected_source in missing_set and source_ts < expected_source
        imperfect_rows.append(
            {
                "decision_time": pd.Timestamp(decision_time).isoformat(),
                "source_context_time": source_ts.isoformat(),
                "expected_source_time": expected_source.isoformat(),
                "attributed_to_vendor_gap": bool(row_attributed),
            }
        )
        all_rows_attributed = all_rows_attributed and row_attributed
    attributed = bool(
        rows
        and missing
        and imperfect_rows
        and all_rows_attributed
        and len(missing) <= MAX_ATTRIBUTABLE_MISSING_INDEX_MINUTES
    )
    return {
        "vendor_missing_index_minutes": [ts.isoformat() for ts in missing],
        "vendor_missing_index_minute_count": len(missing),
        "max_attributable_missing_index_minutes": MAX_ATTRIBUTABLE_MISSING_INDEX_MINUTES,
        "imperfect_lag_row_count": len(imperfect_rows),
        "imperfect_lag_rows": imperfect_rows[:20],
        "attributed": attributed,
    }


def classify_session_status(
    *,
    checks: dict[str, bool],
    early_close_session: bool,
    index_context_gap: dict[str, Any],
) -> tuple[str, str]:
    """Return (status, report_only_reason) for one session record.

    Precedence: early close > low-liquidity-only > vendor-attributed index
    context gap. Any failed check outside those envelopes means fail.
    """
    liquidity_check_names = {
        "tradable_minute_share",
        "mean_tradable_candidates",
        "near_atm_tradable_share",
    }
    context_gap_check_names = {"context_lag_exact_one_minute"}
    failed_checks = {name for name, passed in checks.items() if not passed}
    if early_close_session:
        return "report_only", "early_close_not_close_aware"
    if failed_checks and failed_checks.issubset(liquidity_check_names):
        return "report_only", "low_tradable_liquidity"
    if (
        failed_checks
        and "context_lag_exact_one_minute" in failed_checks
        and failed_checks.issubset(liquidity_check_names | context_gap_check_names)
        and bool(index_context_gap.get("attributed"))
    ):
        return "report_only", "missing_index_context"
    if not failed_checks:
        return "pass", ""
    return "fail", ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def role_policy_allows(role_policy: dict[str, Any], era: str, role: str) -> bool:
    policy = role_policy.get("policy") or {}
    item = policy.get(era) or {}
    return str(role) in set(item.get("permitted_roles") or [])


def acceptance_passes(record: dict[str, Any]) -> bool:
    return record.get("status") == "pass"


def verifier_version_passes(record: dict[str, Any]) -> bool:
    try:
        return int(record.get("verifier_version") or 0) >= MIN_FOLD_PLACEMENT_VERIFIER_VERSION
    except (TypeError, ValueError):
        return False


def canonical_processed_rows_exist(record: dict[str, Any]) -> bool:
    return bool(record.get("processed", {}).get("processed_exists")) and int(record.get("processed", {}).get("neural_rows") or 0) > 0


def fold_placement_predicate(
    *,
    session: str,
    role: str,
    era_manifest: dict[str, Any],
    role_policy: dict[str, Any],
    acceptance_registry: dict[str, Any],
) -> dict[str, Any]:
    session_records = {
        str(item.get("session")): item for item in era_manifest.get("sessions", []) if item.get("session")
    }
    acceptance_records = {
        str(item.get("session")): item for item in acceptance_registry.get("sessions", []) if item.get("session")
    }
    session_record = session_records.get(session, {})
    acceptance_record = acceptance_records.get(session, {})
    era = str(session_record.get("era") or "")
    checks = {
        "era_permits_role": role_policy_allows(role_policy, era, role),
        "canonical_processed_rows_exist": canonical_processed_rows_exist(acceptance_record),
        "acceptance_status_pass": acceptance_passes(acceptance_record),
        "verifier_version_v3_or_newer": verifier_version_passes(acceptance_record),
        "not_early_close_fold_session": not bool(acceptance_record.get("early_close_session")),
    }
    return {
        "schema_version": PREDICATE_SCHEMA_VERSION,
        "session": session,
        "role": role,
        "era": era,
        "placeable": all(checks.values()),
        "checks": checks,
    }


def verify_session(
    *,
    session: str,
    raw_root: Path,
    spx_dir: Path,
    vix_dir: Path,
    processed_dir: Path,
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    raw = raw_quality(session, raw_root)
    index = index_quality(session, spx_dir, vix_dir)
    processed = processed_quality(session, processed_dir)
    processed_feature_contract = str(processed.get("feature_contract_version") or "")
    context = context_causality_quality(
        session,
        processed_dir,
        feature_contract_version=processed_feature_contract,
    )
    context_reconstruction = context_reconstruction_quality(
        session,
        spx_dir=spx_dir,
        vix_dir=vix_dir,
        processed_dir=processed_dir,
    )
    entry_ladder_sweep = entry_ladder_sweep_quality(
        session,
        raw_root=raw_root,
        processed_dir=processed_dir,
    )
    label_spot = label_spot_check_quality(
        session,
        raw_root=raw_root,
        processed_dir=processed_dir,
        min_samples=thresholds.min_label_spot_check_count,
    )
    early_close_session = is_early_close_session(session)
    required_policy_offset_cells = {
        f"{int(policy['policy_idx'])}:{bucket}"
        for policy in PINNED_LABEL_POLICIES
        for bucket in ("near", "far")
    }
    sampled_policy_offset_cells = set(label_spot.get("sampled_policy_offset_cells") or [])
    sampled_rights = set(label_spot.get("sampled_rights") or [])
    checks = {
        "raw_product_files_present": all(
            item["parquet_exists"] and item["dbn_exists"] and item["parquet_bytes"] > 0 and item["dbn_bytes"] > 0
            for item in raw.values()
        ),
        "raw_dbn_parquet_row_counts_match": all(item["row_count_match"] for item in raw.values()),
        "index_products_present": bool(index.get("spx_exists")) and bool(index.get("vix_exists")),
        "processed_rows_exist": bool(processed.get("processed_exists")) and int(processed.get("neural_rows") or 0) > 0,
        "expected_decision_row_count": abs(
            int(processed.get("neural_rows") or 0) - int(processed.get("expected_decision_minutes") or 0)
        )
        <= thresholds.max_missing_processed_rows,
        "ladder_shape_ok": float(processed.get("ladder_shape_ok_share") or 0.0) >= thresholds.min_ladder_shape_ok_share,
        "tradable_minute_share": float(processed.get("tradable_minute_share") or 0.0) >= thresholds.min_tradable_minute_share,
        "mean_tradable_candidates": float(processed.get("mean_tradable_candidates") or 0.0) >= thresholds.min_mean_tradable_candidates,
        "near_atm_tradable_share": float(processed.get("near_atm_tradable_share") or 0.0) >= thresholds.min_near_atm_tradable_share,
        "label_finite_share": float(processed.get("label_finite_share") or 0.0) >= thresholds.min_label_finite_share,
        "label_nonzero_share": float(processed.get("label_nonzero_share") or 0.0) >= thresholds.min_label_nonzero_share,
        "label_positive_values_present": float(processed.get("label_positive_share") or 0.0) >= thresholds.min_label_positive_share,
        "label_negative_values_present": float(processed.get("label_negative_share") or 0.0) >= thresholds.min_label_negative_share,
        "label_spot_recompute": int(label_spot.get("sample_count") or 0) >= thresholds.min_label_spot_check_count
        and int(label_spot.get("mismatch_count") or 0) == 0
        and int(label_spot.get("missing_raw_path_count") or 0) == 0,
        "entry_quote_recompute": int(label_spot.get("sample_count") or 0) >= thresholds.min_label_spot_check_count
        and int(label_spot.get("entry_quote_mismatch_count") or 0) == 0
        and int(label_spot.get("entry_quote_missing_count") or 0) == 0
        and float(label_spot.get("entry_quote_match_share") or 0.0) >= thresholds.min_entry_quote_match_share,
        "label_spot_policy_offset_coverage": required_policy_offset_cells.issubset(sampled_policy_offset_cells),
        "label_spot_call_put_coverage": {"C", "P"}.issubset(sampled_rights),
        "label_spot_forced_flat_coverage": early_close_session
        or int(label_spot.get("forced_flat_reachable_sample_count") or 0) >= 1,
        "entry_quote_full_sweep": int(entry_ladder_sweep.get("masked_candidate_count") or 0) > 0
        and int(entry_ladder_sweep.get("entry_quote_mismatch_count") or 0) == 0
        and int(entry_ladder_sweep.get("entry_quote_missing_count") or 0) == 0
        and int(entry_ladder_sweep.get("entry_quote_stale_count") or 0) == 0
        and float(entry_ladder_sweep.get("entry_quote_match_share") or 0.0)
        >= thresholds.min_entry_quote_sweep_match_share,
        "ladder_quote_full_sweep": int(entry_ladder_sweep.get("masked_candidate_count") or 0) > 0
        and int(entry_ladder_sweep.get("ladder_quote_mismatch_count") or 0) == 0
        and float(entry_ladder_sweep.get("ladder_quote_match_share") or 0.0)
        >= thresholds.min_ladder_quote_sweep_match_share,
        "feature_contract_version_present": processed_feature_contract
        in {FEATURE_CONTRACT_VERSION, FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED},
        "decision_timestamps_match_calendar": bool(context.get("first_decision_matches_calendar"))
        and bool(context.get("last_decision_matches_calendar"))
        and bool(context.get("one_minute_decision_steps")),
        "context_lag_exact_one_minute": float(context.get("context_lag_exact_one_minute_share") or 0.0) == 1.0,
        "no_future_context": int(context.get("future_context_row_count") or 0) == 0,
        "no_leading_backfill_at_open": bool(context.get("opening_no_leading_backfill")),
        "context_features_reconstruct_from_index": float(context_reconstruction.get("feature_match_share") or 0.0)
        >= thresholds.min_context_reconstruction_match_share
        and int(context_reconstruction.get("feature_mismatch_count") or 0) == 0
        and int(context_reconstruction.get("atm_strike_mismatch_count") or 0) == 0,
        "market_window_samples_reconstruct_from_index": float(context_reconstruction.get("window_sample_match_share") or 0.0)
        >= thresholds.min_context_window_sample_match_share
        and int(context_reconstruction.get("window_sample_mismatch_count") or 0) == 0,
        "vix_close_coverage": float(context_reconstruction.get("vix_close_finite_share") or 0.0)
        >= thresholds.min_vix_close_finite_share,
    }
    index_context_gap = index_context_gap_quality(
        session,
        spx_dir=spx_dir,
        processed_dir=processed_dir,
    )
    status, report_only_reason = classify_session_status(
        checks=checks,
        early_close_session=early_close_session,
        index_context_gap=index_context_gap,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "session": session,
        "status": status,
        "early_close_session": early_close_session,
        "report_only_reason": report_only_reason,
        "evidence_grade": "data_plane_only",
        "labels_used_for_strategy_selection": False,
        "pnl_used_for_strategy_selection": False,
        "strategy_metrics_used": False,
        "checks": checks,
        "raw": raw,
        "index": index,
        "processed": processed,
        "context_causality": context,
        "context_reconstruction": context_reconstruction,
        "index_context_gap": index_context_gap,
        "entry_ladder_sweep": entry_ladder_sweep,
        "label_spot_check": label_spot,
        "registry_record_hash": stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "verifier_version": VERIFIER_VERSION,
                "session": session,
                "early_close_session": early_close_session,
                "report_only_reason": report_only_reason,
                "checks": checks,
                "raw": raw,
                "index": index,
                "processed": processed,
                "context_causality": context,
                "context_reconstruction": context_reconstruction,
                "index_context_gap": index_context_gap,
                "entry_ladder_sweep": entry_ladder_sweep,
                "label_spot_check": label_spot,
            }
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Owned Raw Acceptance Registry",
        "",
        f"- Status: `{payload['status']}`",
        f"- Verifier version: `{payload['verifier_version']}`",
        f"- Batch id: `{payload['batch_id']}`",
        f"- Session count: `{payload['session_count']}`",
        f"- Pass count: `{payload['pass_count']}`",
        f"- Report-only count: `{payload['report_only_count']}`",
        f"- Fail count: `{payload['fail_count']}`",
        f"- Evidence grade: `{payload['evidence_grade']}`",
        f"- Thresholds are defaults: `{str(payload.get('thresholds_are_defaults')).lower()}`",
        f"- Era manifest status: `{payload.get('governance_checks', {}).get('era_manifest_status')}`",
        f"- Role policy status: `{payload.get('governance_checks', {}).get('role_policy_status')}`",
        "",
        "## Sessions",
        "",
    ]
    for record in payload["sessions"]:
        failed = [name for name, passed in record["checks"].items() if not passed]
        lines.append(
            f"- `{record['session']}`: status=`{record['status']}`, rows=`{record['processed']['neural_rows']}`, "
            f"expected=`{record['processed']['expected_decision_minutes']}`, ladder_shape=`{record['processed']['ladder_shape_ok_share']:.3f}`, "
            f"tradable_minutes=`{record['processed']['tradable_minute_share']:.3f}`, mean_candidates=`{record['processed']['mean_tradable_candidates']:.2f}`, "
            f"near_atm=`{record['processed']['near_atm_tradable_share']:.3f}`, labels_finite=`{record['processed']['label_finite_share']:.3f}`, "
            f"labels_nonzero=`{record['processed']['label_nonzero_share']:.3f}`, grid=`{record['processed'].get('decision_grid', '')}`, "
            f"spot_checks=`{record['label_spot_check']['match_count']}/{record['label_spot_check']['sample_count']}`, "
            f"entry_quotes=`{record['label_spot_check']['entry_quote_match_count']}/{record['label_spot_check']['sample_count']}`, "
            f"sweep=`{record['entry_ladder_sweep']['entry_quote_match_count']}/{record['entry_ladder_sweep']['masked_candidate_count']}`, "
            f"context_lag=`{record['context_causality']['context_lag_exact_one_minute_share']:.3f}`, failed=`{failed}`"
        )
    lines.extend(["", "## Batch Checks", ""])
    for name, passed in (payload.get("batch_checks") or {}).items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")
    lines.extend(["", "## Guardrails", ""])
    lines.append("- No broker endpoints, paid downloads, model training, threshold tuning, promotion, or real-money paths are used.")
    lines.append("- Registry records are data-plane-only and are not strategy-performance claims.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "session",
        "status",
        "neural_rows",
        "expected_decision_minutes",
        "early_close_session",
        "report_only_reason",
        "ladder_shape_ok_share",
        "tradable_minute_share",
        "mean_tradable_candidates",
        "min_tradable_candidates",
        "near_atm_tradable_share",
        "label_total_cell_count",
        "label_candidate_cell_count",
        "label_finite_share",
        "label_nonzero_share",
        "label_positive_share",
        "label_negative_share",
        "label_spot_sample_count",
        "label_spot_match_count",
        "label_spot_mismatch_count",
        "entry_quote_match_count",
        "entry_quote_mismatch_count",
        "entry_quote_missing_count",
        "entry_quote_match_share",
        "entry_quote_sweep_match_count",
        "entry_quote_sweep_mismatch_count",
        "entry_quote_sweep_missing_count",
        "entry_quote_sweep_stale_count",
        "entry_quote_sweep_match_share",
        "ladder_quote_sweep_match_count",
        "ladder_quote_sweep_mismatch_count",
        "ladder_quote_sweep_match_share",
        "label_fee_model",
        "feature_contract_version",
        "decision_grid",
        "context_lag_exact_one_minute_share",
        "future_context_row_count",
        "opening_no_leading_backfill",
        "context_reconstruction_match_share",
        "context_reconstruction_mismatch_count",
        "context_window_sample_match_share",
        "vix_close_finite_share",
        "failed_checks",
        "record_hash",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "session": record["session"],
                    "status": record["status"],
                    "neural_rows": record["processed"]["neural_rows"],
                    "expected_decision_minutes": record["processed"]["expected_decision_minutes"],
                    "early_close_session": record["early_close_session"],
                    "report_only_reason": record.get("report_only_reason", ""),
                    "ladder_shape_ok_share": record["processed"]["ladder_shape_ok_share"],
                    "tradable_minute_share": record["processed"]["tradable_minute_share"],
                    "mean_tradable_candidates": record["processed"]["mean_tradable_candidates"],
                    "min_tradable_candidates": record["processed"]["min_tradable_candidates"],
                    "near_atm_tradable_share": record["processed"]["near_atm_tradable_share"],
                    "label_total_cell_count": record["processed"].get("label_total_cell_count", 0),
                    "label_candidate_cell_count": record["processed"].get("label_candidate_cell_count", 0),
                    "label_finite_share": record["processed"]["label_finite_share"],
                    "label_nonzero_share": record["processed"]["label_nonzero_share"],
                    "label_positive_share": record["processed"]["label_positive_share"],
                    "label_negative_share": record["processed"]["label_negative_share"],
                    "label_spot_sample_count": record["label_spot_check"]["sample_count"],
                    "label_spot_match_count": record["label_spot_check"]["match_count"],
                    "label_spot_mismatch_count": record["label_spot_check"]["mismatch_count"],
                    "entry_quote_match_count": record["label_spot_check"]["entry_quote_match_count"],
                    "entry_quote_mismatch_count": record["label_spot_check"]["entry_quote_mismatch_count"],
                    "entry_quote_missing_count": record["label_spot_check"]["entry_quote_missing_count"],
                    "entry_quote_match_share": record["label_spot_check"]["entry_quote_match_share"],
                    "entry_quote_sweep_match_count": record["entry_ladder_sweep"]["entry_quote_match_count"],
                    "entry_quote_sweep_mismatch_count": record["entry_ladder_sweep"]["entry_quote_mismatch_count"],
                    "entry_quote_sweep_missing_count": record["entry_ladder_sweep"]["entry_quote_missing_count"],
                    "entry_quote_sweep_stale_count": record["entry_ladder_sweep"]["entry_quote_stale_count"],
                    "entry_quote_sweep_match_share": record["entry_ladder_sweep"]["entry_quote_match_share"],
                    "ladder_quote_sweep_match_count": record["entry_ladder_sweep"]["ladder_quote_match_count"],
                    "ladder_quote_sweep_mismatch_count": record["entry_ladder_sweep"]["ladder_quote_mismatch_count"],
                    "ladder_quote_sweep_match_share": record["entry_ladder_sweep"]["ladder_quote_match_share"],
                    "label_fee_model": record["label_spot_check"]["fee_model"],
                    "feature_contract_version": record["processed"]["feature_contract_version"],
                    "decision_grid": record["processed"].get("decision_grid", ""),
                    "context_lag_exact_one_minute_share": record["context_causality"]["context_lag_exact_one_minute_share"],
                    "future_context_row_count": record["context_causality"]["future_context_row_count"],
                    "opening_no_leading_backfill": record["context_causality"]["opening_no_leading_backfill"],
                    "context_reconstruction_match_share": record["context_reconstruction"]["feature_match_share"],
                    "context_reconstruction_mismatch_count": record["context_reconstruction"]["feature_mismatch_count"],
                    "context_window_sample_match_share": record["context_reconstruction"]["window_sample_match_share"],
                    "vix_close_finite_share": record["context_reconstruction"]["vix_close_finite_share"],
                    "failed_checks": ";".join(name for name, passed in record["checks"].items() if not passed),
                    "record_hash": record["registry_record_hash"],
                }
            )


def main() -> int:
    args = parse_args()
    thresholds = AcceptanceThresholds(
        min_ladder_shape_ok_share=float(args.min_ladder_shape_ok_share),
        min_tradable_minute_share=float(args.min_tradable_minute_share),
        min_mean_tradable_candidates=float(args.min_mean_tradable_candidates),
        min_near_atm_tradable_share=float(args.min_near_atm_tradable_share),
        min_label_finite_share=float(args.min_label_finite_share),
        min_label_nonzero_share=float(args.min_label_nonzero_share),
        min_label_positive_share=float(args.min_label_positive_share),
        min_label_negative_share=float(args.min_label_negative_share),
        min_label_spot_check_count=int(args.min_label_spot_check_count),
        min_vix_close_finite_share=float(args.min_vix_close_finite_share),
        min_context_reconstruction_match_share=float(args.min_context_reconstruction_match_share),
        min_context_window_sample_match_share=float(args.min_context_window_sample_match_share),
        min_entry_quote_match_share=float(args.min_entry_quote_match_share),
        min_entry_quote_sweep_match_share=float(args.min_entry_quote_sweep_match_share),
        min_ladder_quote_sweep_match_share=float(args.min_ladder_quote_sweep_match_share),
    )
    sessions = sessions_between(args.start_date, args.end_date)
    records = [
        verify_session(
            session=session,
            raw_root=args.raw_root,
            spx_dir=args.official_spx_dir,
            vix_dir=args.official_vix_dir,
            processed_dir=args.processed_dir,
            thresholds=thresholds,
        )
        for session in sessions
    ]
    era_manifest = load_json(args.era_manifest)
    role_policy = load_json(args.role_policy)
    governance_checks = {
        "era_manifest_status": str(era_manifest.get("status") or "missing"),
        "role_policy_status": str(role_policy.get("status") or "missing"),
        "era_manifest_pass": era_manifest.get("status") == "pass",
        "role_policy_pass": role_policy.get("status") == "pass",
    }
    registry_for_predicate = {"sessions": records}
    placement = [
        fold_placement_predicate(
            session=record["session"],
            role=str(args.role),
            era_manifest=era_manifest,
            role_policy=role_policy,
            acceptance_registry=registry_for_predicate,
        )
        for record in records
    ]
    label_outcomes: dict[str, int] = {}
    for record in records:
        if record.get("status") == "report_only":
            continue
        for reason, count in (record.get("label_spot_check", {}).get("outcome_reasons") or {}).items():
            label_outcomes[str(reason)] = label_outcomes.get(str(reason), 0) + int(count)
    expected_monthly_outcomes = {"stop_hit", "target_hit", "time_exit"}
    batch_checks = {
        "governance_artifacts_pass": bool(governance_checks["era_manifest_pass"] and governance_checks["role_policy_pass"]),
        "monthly_label_outcome_coverage": len(records) < 5
        or expected_monthly_outcomes.issubset(set(label_outcomes)),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "minimum_fold_placement_verifier_version": MIN_FOLD_PLACEMENT_VERIFIER_VERSION,
        "status": "pass"
        if all(record["status"] != "fail" for record in records) and all(batch_checks.values())
        else "fail",
        "batch_id": f"owned_raw_acceptance_{args.start_date}_to_{args.end_date}",
        "evidence_grade": "data_plane_only",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "session_count": len(records),
        "pass_count": sum(1 for record in records if record["status"] == "pass"),
        "report_only_count": sum(1 for record in records if record["status"] == "report_only"),
        "fail_count": sum(1 for record in records if record["status"] == "fail"),
        "thresholds": {
            "min_ladder_shape_ok_share": thresholds.min_ladder_shape_ok_share,
            "min_tradable_minute_share": thresholds.min_tradable_minute_share,
            "min_mean_tradable_candidates": thresholds.min_mean_tradable_candidates,
            "min_near_atm_tradable_share": thresholds.min_near_atm_tradable_share,
            "min_label_finite_share": thresholds.min_label_finite_share,
            "min_label_nonzero_share": thresholds.min_label_nonzero_share,
            "min_label_positive_share": thresholds.min_label_positive_share,
            "min_label_negative_share": thresholds.min_label_negative_share,
            "min_label_spot_check_count": thresholds.min_label_spot_check_count,
            "min_vix_close_finite_share": thresholds.min_vix_close_finite_share,
            "min_context_reconstruction_match_share": thresholds.min_context_reconstruction_match_share,
            "min_context_window_sample_match_share": thresholds.min_context_window_sample_match_share,
            "min_entry_quote_match_share": thresholds.min_entry_quote_match_share,
            "min_entry_quote_sweep_match_share": thresholds.min_entry_quote_sweep_match_share,
            "min_ladder_quote_sweep_match_share": thresholds.min_ladder_quote_sweep_match_share,
            "max_missing_processed_rows": thresholds.max_missing_processed_rows,
        },
        "thresholds_are_defaults": thresholds_are_defaults(thresholds),
        "fee_model": PINNED_FEE_MODEL,
        "max_quote_age_seconds": PINNED_MAX_QUOTE_AGE_SECONDS,
        "no_bid_convention": PINNED_NO_BID_CONVENTION,
        "label_policies": PINNED_LABEL_POLICIES,
        "forced_flat_before_et": PINNED_FORCED_FLAT_BEFORE_ET,
        "cbbo_stamping_assumption": CBBO_STAMPING_ASSUMPTION,
        "governance_checks": governance_checks,
        "batch_checks": batch_checks,
        "batch_label_outcome_reasons": label_outcomes,
        "labels_used_for_strategy_selection": False,
        "pnl_used_for_strategy_selection": False,
        "strategy_metrics_used": False,
        "sessions": records,
        "placement_predicates": placement,
    }
    payload["era_manifest_hash"] = stable_hash(era_manifest)
    payload["role_policy_hash"] = stable_hash(role_policy)
    payload["registry_hash"] = compute_registry_hash(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    registry_path = args.out_dir / "acceptance_registry.json"
    csv_path = args.out_dir / "acceptance_registry.csv"
    placement_path = args.out_dir / "fold_placement_predicates.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    placement_path.write_text(json.dumps(placement, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, records)
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "pass_count": payload["pass_count"],
                "report_only_count": payload["report_only_count"],
                "fail_count": payload["fail_count"],
                "registry_hash": payload["registry_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
