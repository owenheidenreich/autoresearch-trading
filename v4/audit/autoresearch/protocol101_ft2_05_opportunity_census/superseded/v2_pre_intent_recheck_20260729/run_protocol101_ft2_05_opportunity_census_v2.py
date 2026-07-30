"""Run the bounded Protocol101 FT2-05 label-side opportunity census.

This node computes future-path labels and transparent hindsight ceilings.  It
does not fit a model, select features, tune guardrails, or access any session
outside the frozen FT2-04 census manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from v4.model.protocol101_divergence_noise import moneyness_band
from v4.model.protocol101_governed_loader import (
    load_governed_loader_artifacts,
    validate_governance_artifacts,
    validate_session_for_role,
)
from v4.model.protocol101_regimen_repair import ExitReason, InvalidReason
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    DEFAULT_ACCEPTANCE_REGISTRY,
    DEFAULT_ERA_MANIFEST,
    DEFAULT_PROTECTED_HOLDOUT,
    DEFAULT_ROLE_POLICY,
)


AUTHORITY_PATH = Path(
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_SHA256 = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
GRAPH_PATH = Path(
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
FREEZE_DIR = Path(
    "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze"
)
DEFAULT_MANIFEST_PATH = Path(
    "v4/audit/autoresearch/"
    "protocol101_live_v2_microstructure_masked_15mo_training_preflight/"
    "canonical_processed_session_manifest.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
)
NODE_ID = "FT2-05-OPPORTUNITY-CENSUS"
SCHEMA_VERSION = "Protocol101FT205OpportunityCensusV2"
MAX_QUOTE_AGE_MS = 90_000.0
STARTING_CASH = 10_000.0
MULTIPLIER = 100.0
ROUND_TRIP_FEE = 3.0
PREMIUM_BUDGET_FRACTION = 0.05
SOFT_CLOSE_MIN_PREMIUM = 1.00
FORCED_FLAT_ET = "15:55"
NO_NEW_ENTRIES_AFTER_ET = "15:30"
HORIZONS: tuple[int | str, ...] = (
    3,
    5,
    10,
    20,
    45,
    90,
    "remaining_session",
)
GUARDRAIL_LEVELS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2_026_072_905
ONE_MINUTE_NS = 60 * 1_000_000_000
POLICY_INDEX_BY_HORIZON = {
    3: 0,
    5: 1,
    10: 2,
    20: 3,
    45: 4,
    90: 5,
    "remaining_session": 6,
}


class CensusContractError(RuntimeError):
    """Fail-closed error for the bounded census node."""


@dataclass(frozen=True)
class SessionSource:
    session: str
    processed_path: Path
    normalized_path: Path
    spx_path: Path
    processed_sha256: str
    normalized_sha256: str
    spx_sha256: str


@dataclass(frozen=True)
class QuotePath:
    quote_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    quote_age_ms: np.ndarray


@dataclass(frozen=True)
class ReplaySelection:
    row_index: int
    decision_time_ns: int
    exit_time_ns: int
    source_exit_time_ns: int
    deadline_ns: int
    exit_bid: float
    pnl: float
    exit_reason: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke-session",
        help="Engineering smoke only; must be one frozen census session.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise CensusContractError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n"
    )


def _finite_or_none(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _finite_or_none(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _manifest_by_session(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    included = list(manifest.get("included_sessions") or [])
    out: dict[str, dict[str, Any]] = {}
    for item in included:
        session = str(item.get("session") or "")
        if not session or session in out:
            raise CensusContractError(
                f"duplicate or empty canonical manifest session: {session!r}"
            )
        out[session] = dict(item)
    return out


def verify_authority_and_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    if file_sha256(AUTHORITY_PATH) != AUTHORITY_SHA256:
        raise CensusContractError("consolidated authority SHA-256 mismatch")
    receipt = read_json(FREEZE_DIR / "receipt.json")
    if receipt.get("product_contract_hash") != AUTHORITY_SHA256:
        raise CensusContractError("FT2-04 receipt product contract hash mismatch")
    expected = receipt.get("deliverable_hashes") or {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("label_spec.json", "oracle_rules.json", "census_sessions.json"):
        path = FREEZE_DIR / name
        actual = file_sha256(path)
        if actual != str(expected.get(name) or ""):
            raise CensusContractError(f"FT2-04 frozen input hash mismatch: {name}")
        payloads[name] = read_json(path)
    graph = read_json(GRAPH_PATH)
    nodes = [
        item
        for item in graph.get("nodes") or []
        if str(item.get("id")) == NODE_ID
    ]
    if len(nodes) != 1 or nodes[0].get("model_training") is not False:
        raise CensusContractError("Graph V2 FT2-05 node is missing or malformed")
    legal_edges = {
        (str(item.get("outcome")), str(item.get("to")))
        for item in graph.get("edges") or []
        if str(item.get("from")) == NODE_ID
    }
    required_edges = {
        ("feasible", "FT2-08-DATA-TENSOR-LABEL-CONTRACT"),
        ("genuine_impossibility", "STOP-OWNER-DECISION"),
    }
    if legal_edges != required_edges:
        raise CensusContractError(
            f"unexpected FT2-05 legal outcomes: {sorted(legal_edges)}"
        )
    return (
        receipt,
        payloads["label_spec.json"],
        payloads["oracle_rules.json"],
        payloads["census_sessions.json"],
    )


def resolve_session_sources(
    census_manifest: dict[str, Any],
    *,
    smoke_session: str | None = None,
) -> tuple[list[SessionSource], dict[str, Any]]:
    frozen_sessions = [str(value) for value in census_manifest["census_sessions"]]
    if len(frozen_sessions) != 45 or len(set(frozen_sessions)) != 45:
        raise CensusContractError("frozen census manifest is not exactly 45 unique sessions")
    requested = frozen_sessions
    if smoke_session:
        if smoke_session not in frozen_sessions:
            raise CensusContractError("smoke session is outside the frozen census set")
        requested = [smoke_session]

    manifest = read_json(DEFAULT_MANIFEST_PATH)
    by_session = _manifest_by_session(manifest)
    artifacts = load_governed_loader_artifacts(
        acceptance_registry_path=DEFAULT_ACCEPTANCE_REGISTRY,
        era_manifest_path=DEFAULT_ERA_MANIFEST,
        role_policy_path=DEFAULT_ROLE_POLICY,
        protected_holdout_path=DEFAULT_PROTECTED_HOLDOUT,
    )
    governance_blockers = validate_governance_artifacts(artifacts)
    if governance_blockers:
        raise CensusContractError(
            f"governed loader artifact blockers: {governance_blockers}"
        )
    acceptance_rows = {
        str(item.get("session")): item
        for item in artifacts.acceptance_registry.get("sessions") or []
    }
    validations: list[dict[str, Any]] = []
    sources: list[SessionSource] = []
    for session in requested:
        manifest_row = by_session.get(session)
        if manifest_row is None:
            raise CensusContractError(f"session absent from canonical manifest: {session}")
        processed_path = Path(str(manifest_row["processed_file"]))
        validation = validate_session_for_role(
            session=session,
            role="train",
            processed_path=processed_path,
            artifacts=artifacts,
            governance_blockers=governance_blockers,
        )
        validations.append(validation)
        if not validation["placeable"]:
            raise CensusContractError(
                f"governed loader rejected census session {session}: "
                f"{validation['blockers']}"
            )
        normalized_path = Path(str(manifest_row["normalized_base_file"]))
        acceptance = acceptance_rows.get(session) or {}
        spx_path = Path(str((acceptance.get("index") or {}).get("spx_path") or ""))
        if not normalized_path.exists() or not spx_path.exists():
            raise CensusContractError(
                f"required governed source is missing for {session}"
            )
        sources.append(
            SessionSource(
                session=session,
                processed_path=processed_path,
                normalized_path=normalized_path,
                spx_path=spx_path,
                processed_sha256=file_sha256(processed_path),
                normalized_sha256=file_sha256(normalized_path),
                spx_sha256=file_sha256(spx_path),
            )
        )
    resolved_sessions = [item.session for item in sources]
    if resolved_sessions != requested:
        raise CensusContractError("resolved session order differs from frozen manifest")
    governance = {
        "schema_version": "Protocol101FT205GovernanceReceiptV2",
        "frozen_session_count": len(frozen_sessions),
        "requested_session_count": len(requested),
        "smoke_session": smoke_session,
        "requested_sessions": requested,
        "outer_test_intersection": sorted(
            set(requested)
            & {
                str(value)
                for values in census_manifest["outer_test_sessions_by_fold"].values()
                for value in values
            }
        ),
        "protected_holdout_intersection": sorted(
            set(requested)
            & set(map(str, census_manifest["protected_holdout_sessions"]))
        ),
        "embargo_intersection": sorted(
            set(requested) & set(map(str, census_manifest["embargo_sessions"]))
        ),
        "validations": validations,
        "sources": [
            {
                **asdict(item),
                "processed_path": str(item.processed_path),
                "normalized_path": str(item.normalized_path),
                "spx_path": str(item.spx_path),
            }
            for item in sources
        ],
    }
    if (
        governance["outer_test_intersection"]
        or governance["protected_holdout_intersection"]
        or governance["embargo_intersection"]
    ):
        raise CensusContractError("census session firewall intersection is nonempty")
    governance["governance_hash"] = stable_hash(governance)
    return sources, governance


def market_phase(timestamp: pd.Timestamp) -> str:
    local = timestamp.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if minute < 9 * 60 + 45:
        return "opening_discovery"
    if minute < 11 * 60 + 15:
        return "primary_morning"
    if minute < 11 * 60 + 45:
        return "europe_close_transition"
    if minute < 13 * 60 + 30:
        return "lunch"
    if minute < 15 * 60:
        return "afternoon"
    if minute <= 15 * 60 + 30:
        return "power_hour_entry"
    return "manage_exit_only"


def premium_band(ask: float) -> str:
    if ask <= 1.0:
        return "cheap_le_1"
    if ask <= 3.0:
        return "small_1_3"
    if ask <= 8.0:
        return "medium_3_8"
    if ask <= 20.0:
        return "large_8_20"
    return "very_large_20p"


def _forced_flat_ns(session: str) -> int:
    timestamp = pd.Timestamp(f"{session} {FORCED_FLAT_ET}", tz="America/New_York")
    return int(timestamp.tz_convert("UTC").value)


def _d49_should_soft_close(
    *,
    session_start_equity: float,
    realized_session_pnl: float,
) -> bool:
    daily_budget = PREMIUM_BUDGET_FRACTION * session_start_equity
    realized_loss = max(0.0, -realized_session_pnl)
    remaining_budget = daily_budget - realized_loss
    minimum_lane_cash = SOFT_CLOSE_MIN_PREMIUM * MULTIPLIER + ROUND_TRIP_FEE
    return remaining_budget + 1e-9 < minimum_lane_cash


def _longest_true_run(values: np.ndarray) -> tuple[int, int]:
    longest = 0
    runs = 0
    active = 0
    for value in values:
        if bool(value):
            if active == 0:
                runs += 1
            active += 1
            longest = max(longest, active)
        else:
            active = 0
    return longest, runs


def _quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else math.nan


def _path_window(
    path: QuotePath,
    start_exclusive_ns: int,
    deadline_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = int(np.searchsorted(path.quote_ns, start_exclusive_ns, side="right"))
    right = int(np.searchsorted(path.quote_ns, deadline_ns, side="right"))
    quote_ns = path.quote_ns[left:right]
    bid = path.bid[left:right].astype(float, copy=True)
    valid = (
        np.isfinite(bid)
        & (bid > 0.0)
        & np.isfinite(path.quote_age_ms[left:right])
        & (path.quote_age_ms[left:right] <= MAX_QUOTE_AGE_MS)
    )
    bid[~valid] = np.nan
    return quote_ns, bid, valid


def _grid_values(
    quote_ns: np.ndarray,
    values: np.ndarray,
    start_exclusive_ns: int,
    deadline_ns: int,
    *,
    missing_value: float = math.nan,
) -> tuple[np.ndarray, np.ndarray]:
    grid_ns = np.arange(
        start_exclusive_ns + ONE_MINUTE_NS,
        deadline_ns + ONE_MINUTE_NS,
        ONE_MINUTE_NS,
        dtype=np.int64,
    )
    by_time = {
        int(timestamp): float(value)
        for timestamp, value in zip(quote_ns, values, strict=True)
        if math.isfinite(float(value))
    }
    grid = np.asarray([by_time.get(int(value), missing_value) for value in grid_ns])
    return grid_ns, grid


def _quote_at_minute(
    path: QuotePath,
    timestamp_ns: int,
) -> tuple[float, float, bool, bool]:
    left = int(np.searchsorted(path.quote_ns, timestamp_ns, side="left"))
    right = int(np.searchsorted(path.quote_ns, timestamp_ns, side="right"))
    if right - left > 1:
        raise CensusContractError(
            f"duplicate contract quote identity at {timestamp_ns}"
        )
    if left >= len(path.quote_ns) or int(path.quote_ns[left]) != timestamp_ns:
        return math.nan, math.nan, False, False
    age = float(path.quote_age_ms[left])
    age_valid = math.isfinite(age) and age <= MAX_QUOTE_AGE_MS
    bid = float(path.bid[left])
    ask = float(path.ask[left])
    bid_valid = age_valid and math.isfinite(bid) and bid > 0.0
    ask_valid = age_valid and math.isfinite(ask) and ask > 0.0
    return (
        bid if bid_valid else math.nan,
        ask if ask_valid else math.nan,
        bid_valid,
        ask_valid,
    )


def compute_horizon_labels(
    *,
    path: QuotePath,
    entry_fill_ns: int,
    session: str,
    entry_ask: float,
    horizon: int | str,
) -> dict[str, Any]:
    forced_flat_ns = _forced_flat_ns(session)
    nominal_deadline = (
        forced_flat_ns
        if horizon == "remaining_session"
        else entry_fill_ns + int(horizon) * ONE_MINUTE_NS
    )
    deadline_ns = min(nominal_deadline, forced_flat_ns)
    quote_ns, executable_bid, executable = _path_window(
        path, entry_fill_ns, deadline_ns
    )
    grid_ns, bid_grid = _grid_values(
        quote_ns,
        executable_bid,
        entry_fill_ns,
        deadline_ns,
    )
    _, executable_grid_float = _grid_values(
        quote_ns,
        executable.astype(float),
        entry_fill_ns,
        deadline_ns,
        missing_value=0.0,
    )
    executable_grid = executable_grid_float.astype(bool)
    conservative_bid_grid = np.where(executable_grid, bid_grid, 0.0)
    pnl = (conservative_bid_grid - entry_ask) * MULTIPLIER - ROUND_TRIP_FEE
    returns = pnl / (entry_ask * MULTIPLIER)
    available = np.ones(len(grid_ns), dtype=bool)
    positive = pnl > 0.0
    underwater = pnl < 0.0
    valid_pnl = pnl
    valid_returns = returns
    available_ns = grid_ns

    out: dict[str, Any] = {
        "available_minutes": int(available.sum()),
        "executable_bid_minutes": int(executable_grid.sum()),
        "no_bid_minutes": int((~executable_grid).sum()),
        "nominal_minutes": int(len(grid_ns)),
        "censored": bool(nominal_deadline > forced_flat_ns),
        "deadline_ns": int(deadline_ns),
        "entry_fill_time_ns": int(entry_fill_ns),
    }
    if not available.any():
        return out

    excluded_pnl = pnl[executable_grid]
    excluded_returns = returns[executable_grid]
    excluded_ns = grid_ns[executable_grid]
    out["mnar_excluded_available_minutes"] = int(executable_grid.sum())
    if executable_grid.any():
        excluded_positive = excluded_pnl > 0.0
        out.update(
            {
                "mnar_excluded_uwi_dollars": float(
                    np.sum(-excluded_pnl[excluded_pnl < 0.0])
                ),
                "mnar_excluded_uwi_return": float(
                    np.sum(-excluded_returns[excluded_returns < 0.0])
                ),
                "mnar_excluded_fraction_positive": float(
                    np.mean(excluded_positive)
                ),
                "mnar_excluded_mfe_dollars": float(np.max(excluded_pnl)),
                "mnar_excluded_mfe_return": float(np.max(excluded_returns)),
            }
        )
        if excluded_positive.any():
            excluded_first = int(np.flatnonzero(excluded_positive)[0])
            out["mnar_excluded_ttfp_minutes"] = float(
                (int(excluded_ns[excluded_first]) - entry_fill_ns) / ONE_MINUTE_NS
            )
        else:
            out["mnar_excluded_ttfp_minutes"] = float(
                (int(excluded_ns[-1]) - entry_fill_ns) / ONE_MINUTE_NS
            )
        if horizon in {3, 5, 10}:
            out.update(
                {
                    "mnar_excluded_early_dd_dollars": float(
                        np.min(excluded_pnl)
                    ),
                    "mnar_excluded_early_dd_return": float(
                        np.min(excluded_returns)
                    ),
                }
            )

    first_positive_index = int(np.flatnonzero(positive)[0]) if positive.any() else -1
    if first_positive_index >= 0:
        first_ns = int(grid_ns[first_positive_index])
        pre = np.arange(len(grid_ns)) < first_positive_index
        pre_available = pre & available
        ppae_pnl = float(np.min(pnl[pre_available])) if pre_available.any() else 0.0
        ppae_ret = (
            float(np.min(returns[pre_available])) if pre_available.any() else 0.0
        )
        out.update(
            {
                "ttfp_minutes": (first_ns - entry_fill_ns) / ONE_MINUTE_NS,
                "ttfp_decision_relative_minutes": (
                    (first_ns - entry_fill_ns) / ONE_MINUTE_NS + 1.0
                ),
                "ttfp_censored": False,
                "first_profit_time_ns": first_ns,
                "first_profit_pnl_dollars": float(pnl[first_positive_index]),
                "first_profit_return": float(returns[first_positive_index]),
                "ppae_dollars": ppae_pnl,
                "ppae_return": ppae_ret,
                "immediate_profit": bool(not pre_available.any()),
            }
        )
    else:
        last_ns = int(available_ns[-1])
        out.update(
            {
                "ttfp_minutes": (last_ns - entry_fill_ns) / ONE_MINUTE_NS,
                "ttfp_decision_relative_minutes": (
                    (last_ns - entry_fill_ns) / ONE_MINUTE_NS + 1.0
                ),
                "ttfp_censored": True,
                "first_profit_time_ns": None,
                "first_profit_pnl_dollars": None,
                "first_profit_return": None,
                "ppae_dollars": float(np.min(valid_pnl)),
                "ppae_return": float(np.min(valid_returns)),
                "immediate_profit": False,
            }
        )

    longest_underwater, _ = _longest_true_run(underwater)
    longest_profitable, profitable_runs = _longest_true_run(positive)
    fraction_positive = float(positive.sum() / available.sum())

    bid_by_ns = {
        int(timestamp): float(value)
        for timestamp, value in zip(grid_ns, conservative_bid_grid, strict=True)
    }
    shifted_fractions: list[float] = []
    for shift in (-ONE_MINUTE_NS, ONE_MINUTE_NS):
        shifted = np.asarray(
            [bid_by_ns.get(int(timestamp + shift), math.nan) for timestamp in grid_ns]
        )
        shifted_pnl = (shifted - entry_ask) * MULTIPLIER - ROUND_TRIP_FEE
        shifted_valid = np.isfinite(shifted_pnl)
        if shifted_valid.any():
            shifted_fractions.append(
                float((shifted_pnl[shifted_valid] > 0.0).mean())
            )
    jitter = (
        max(abs(value - fraction_positive) for value in shifted_fractions)
        if shifted_fractions
        else math.nan
    )

    executable_indices = np.flatnonzero(executable_grid)
    argmax_valid = (
        int(executable_indices[np.argmax(pnl[executable_grid])])
        if executable_indices.size
        else -1
    )
    argmin_valid = int(np.argmin(valid_pnl))
    out.update(
        {
            "uwi_dollars": float(np.sum(-pnl[underwater])),
            "uwi_return": float(np.sum(-returns[underwater])),
            "underwater_minutes": int(underwater.sum()),
            "longest_underwater_minutes": int(longest_underwater),
            "fraction_positive": fraction_positive,
            "longest_profitable_minutes": int(longest_profitable),
            "profitable_window_count": int(profitable_runs),
            "jitter": float(jitter),
            "mfe_dollars": float(np.max(valid_pnl)),
            "mfe_return": float(np.max(valid_returns)),
            "argmax_time_ns": int(available_ns[int(np.argmax(valid_pnl))]),
            "profit_area_dollars": float(np.sum(valid_pnl[valid_pnl > 0.0])),
            "profit_area_return": float(np.sum(valid_returns[valid_returns > 0.0])),
        }
    )
    if argmax_valid >= 0:
        out.update(
            {
                "best_exit_bid": float(conservative_bid_grid[argmax_valid]),
                "best_exit_time_ns": int(grid_ns[argmax_valid]),
            }
        )
    for q in (0.75, 0.90, 0.95, 1.00):
        suffix = str(int(q * 100))
        out[f"upside_q{suffix}_dollars"] = _quantile(valid_pnl, q)
        out[f"upside_q{suffix}_return"] = _quantile(valid_returns, q)
    if horizon in {3, 5, 10}:
        out.update(
            {
                "early_dd_dollars": float(valid_pnl[argmin_valid]),
                "early_dd_return": float(valid_returns[argmin_valid]),
                "early_dd_argmin_time_ns": int(available_ns[argmin_valid]),
            }
        )
        for q in (0.00, 0.05, 0.10, 0.25):
            suffix = str(int(q * 100)).zfill(2)
            out[f"early_dd_q{suffix}_dollars"] = _quantile(valid_pnl, q)
            out[f"early_dd_q{suffix}_return"] = _quantile(valid_returns, q)
    return out


def _flatten_horizon(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def _build_quote_paths(normalized: pd.DataFrame) -> dict[str, QuotePath]:
    normalized = normalized.sort_values(["contract_id", "event_time"])
    out: dict[str, QuotePath] = {}
    for contract_id, group in normalized.groupby("contract_id", sort=False):
        timestamps = pd.to_datetime(group["event_time"], utc=True)
        out[str(contract_id)] = QuotePath(
            quote_ns=timestamps.array.as_unit("ns").asi8.copy(),
            bid=pd.to_numeric(group["bid"], errors="coerce").to_numpy(dtype=float),
            ask=pd.to_numeric(group["ask"], errors="coerce").to_numpy(dtype=float),
            quote_age_ms=pd.to_numeric(
                group["quote_age_ms"], errors="coerce"
            ).to_numpy(dtype=float),
        )
    return out


def _session_vol_proxy(session: str, spx_path: Path) -> tuple[float, str]:
    frame = pd.read_parquet(spx_path)
    time_column = "event_time" if "event_time" in frame.columns else frame.index.name
    if time_column is None:
        raise CensusContractError(f"SPX context has no timestamp: {session}")
    if time_column in frame.columns:
        timestamps = pd.to_datetime(frame[time_column], utc=True)
    else:
        timestamps = pd.to_datetime(frame.index, utc=True)
    local = timestamps.dt.tz_convert("America/New_York")
    regular = frame[(local.dt.time >= pd.Timestamp("09:30").time()) & (local.dt.time <= pd.Timestamp("16:00").time())]
    close_name = "close" if "close" in regular.columns else "spx_close"
    close = pd.to_numeric(regular[close_name], errors="coerce").dropna()
    if close.empty:
        raise CensusContractError(f"SPX context has no regular-session close: {session}")
    value = float((close.max() - close.min()) / max(abs(close.iloc[0]), 1.0) * 10_000.0)
    return value, "session_spx_realized_range_bps"


def build_session_labels(source: SessionSource) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.monotonic()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    with source.processed_path.open("rb") as handle:
        rows = pickle.load(handle)
    normalized = pd.read_parquet(
        source.normalized_path,
        columns=[
            "event_time",
            "contract_id",
            "bid",
            "ask",
            "quote_age_ms",
        ],
    )
    normalized["event_time"] = pd.to_datetime(normalized["event_time"], utc=True)
    quote_paths = _build_quote_paths(normalized)
    vol_value, vol_name = _session_vol_proxy(source.session, source.spx_path)
    output: list[dict[str, Any]] = []
    missing_paths = 0
    missing_entry_fills = 0
    malformed_rows = 0

    for row in rows:
        decision = pd.Timestamp(row["decision_time"])
        if decision.tzinfo is None:
            decision = decision.tz_localize("UTC")
        else:
            decision = decision.tz_convert("UTC")
        decision_ns = int(decision.value)
        decision_local = decision.tz_convert("America/New_York")
        if decision_local.strftime("%H:%M") >= NO_NEW_ENTRIES_AFTER_ET:
            continue
        entry_fill_ns = decision_ns + ONE_MINUTE_NS
        mask = np.asarray(row.get("candidate_mask"), dtype=bool)
        ids = np.asarray(row.get("contract_ids"), dtype=object)
        offsets = np.asarray(row.get("strike_offsets"), dtype=float)
        rights = tuple(str(value) for value in (row.get("rights") or ("C", "P")))
        metadata = row.get("contract_quote_metadata") or {}
        if mask.shape != ids.shape or mask.shape != (len(offsets), len(rights)):
            malformed_rows += 1
            continue
        market = np.asarray(row.get("market_window"), dtype=float)
        market_names = tuple(row.get("market_feature_names") or ())
        if market.ndim != 2 or not len(market):
            malformed_rows += 1
            continue
        market_last = market[-1]
        market_map = {
            str(name): float(market_last[index])
            for index, name in enumerate(market_names)
            if index < len(market_last) and math.isfinite(float(market_last[index]))
        }
        spx_close = market_map.get("spx_close", math.nan)
        spx_vwap = market_map.get("spx_vwap", math.nan)

        for strike_idx, right_idx in np.argwhere(mask):
            contract_id = str(ids[strike_idx, right_idx] or "")
            quote_meta = metadata.get(contract_id) or {}
            decision_ask = _finite_or_none(quote_meta.get("ask"))
            decision_bid = _finite_or_none(quote_meta.get("bid"))
            if (
                not contract_id
                or not isinstance(decision_ask, (float, int))
                or float(decision_ask) <= 0.0
            ):
                continue
            path = quote_paths.get(contract_id)
            if path is None:
                missing_paths += 1
                continue
            fill_bid, fill_ask, fill_bid_valid, fill_ask_valid = _quote_at_minute(
                path, entry_fill_ns
            )
            if not fill_ask_valid:
                missing_entry_fills += 1
                continue
            ask = float(fill_ask)
            bid = float(fill_bid) if fill_bid_valid else 0.0
            offset = float(offsets[int(strike_idx)])
            required_cash = ask * MULTIPLIER + ROUND_TRIP_FEE
            record: dict[str, Any] = {
                "session": source.session,
                "month": source.session[:7],
                "decision_time_ns": decision_ns,
                "decision_time_utc": decision.isoformat(),
                "decision_minute_et": decision_local.strftime("%H:%M"),
                "entry_fill_time_ns": entry_fill_ns,
                "entry_fill_time_utc": pd.Timestamp(
                    entry_fill_ns, tz="UTC"
                ).isoformat(),
                "market_phase": market_phase(decision),
                "contract_id": contract_id,
                "right": rights[int(right_idx)],
                "strike_idx": int(strike_idx),
                "right_idx": int(right_idx),
                "offset": offset,
                "abs_offset": abs(offset),
                "moneyness_band": moneyness_band(abs(offset)),
                "decision_entry_bid": (
                    float(decision_bid)
                    if isinstance(decision_bid, (float, int))
                    else math.nan
                ),
                "decision_entry_ask": float(decision_ask),
                "entry_bid": bid,
                "entry_ask": ask,
                "entry_bid_executable": bool(fill_bid_valid),
                "entry_spread": ask - bid if fill_bid_valid else math.nan,
                "premium_at_risk": ask * MULTIPLIER,
                "premium_plus_fee": required_cash,
                "premium_band": premium_band(ask),
                "friction_dollars": (
                    (ask - bid) * MULTIPLIER + ROUND_TRIP_FEE
                    if fill_bid_valid
                    else ask * MULTIPLIER + ROUND_TRIP_FEE
                ),
                "friction_fraction_of_premium": (
                    (
                        (ask - bid) * MULTIPLIER + ROUND_TRIP_FEE
                        if fill_bid_valid
                        else ask * MULTIPLIER + ROUND_TRIP_FEE
                    )
                    / (ask * MULTIPLIER)
                ),
                "d48_reference_equity": STARTING_CASH,
                "d48_reference_eligible": bool(
                    required_cash
                    <= PREMIUM_BUDGET_FRACTION * STARTING_CASH + 1e-9
                ),
                "spx_close": spx_close,
                "spx_vwap": spx_vwap,
                "vwap_side": (
                    "C"
                    if math.isfinite(spx_close)
                    and math.isfinite(spx_vwap)
                    and spx_close >= spx_vwap
                    else "P"
                ),
                "vol_proxy_name": vol_name,
                "vol_proxy_value": vol_value,
            }
            for horizon in HORIZONS:
                key = "session" if horizon == "remaining_session" else f"h{horizon}"
                record.update(
                    _flatten_horizon(
                        key,
                        compute_horizon_labels(
                            path=path,
                            entry_fill_ns=entry_fill_ns,
                            session=source.session,
                            entry_ask=ask,
                            horizon=horizon,
                        ),
                    )
                )
            session_available = int(record.get("session_available_minutes") or 0)
            if session_available:
                first_time = record.get("session_first_profit_time_ns")
                if first_time is not None:
                    record.update(
                        {
                            "first_profit_exit_time_ns": int(first_time),
                            "first_profit_source_time_ns": int(first_time),
                            "first_profit_exit_bid": (
                                ask
                                + (
                                    float(record["session_first_profit_pnl_dollars"])
                                    + ROUND_TRIP_FEE
                                )
                                / MULTIPLIER
                            ),
                            "first_profit_exit_pnl": float(
                                record["session_first_profit_pnl_dollars"]
                            ),
                        }
                    )
                else:
                    session_deadline = int(record["session_deadline_ns"])
                    flat_bid, _, flat_bid_valid, _ = _quote_at_minute(
                        path, session_deadline
                    )
                    exit_bid = float(flat_bid) if flat_bid_valid else 0.0
                    record.update(
                        {
                            "first_profit_exit_time_ns": session_deadline,
                            "first_profit_source_time_ns": session_deadline,
                            "first_profit_exit_bid": exit_bid,
                            "first_profit_exit_pnl": (
                                (exit_bid - ask) * MULTIPLIER - ROUND_TRIP_FEE
                            ),
                            "first_profit_forced_flat_no_bid": bool(
                                not flat_bid_valid
                            ),
                        }
                    )
                session_deadline = int(record["session_deadline_ns"])
                flat_bid, _, flat_bid_valid, _ = _quote_at_minute(
                    path, session_deadline
                )
                exact_flat_bid = float(flat_bid) if flat_bid_valid else 0.0
                record.update(
                    {
                        "hold_flat_exit_time_ns": session_deadline,
                        "hold_flat_source_time_ns": session_deadline,
                        "hold_flat_exit_bid": exact_flat_bid,
                        "hold_flat_exit_pnl": (
                            (exact_flat_bid - ask) * MULTIPLIER - ROUND_TRIP_FEE
                        ),
                        "hold_flat_no_bid": bool(not flat_bid_valid),
                    }
                )
            output.append(record)

    frame = pd.DataFrame(output)
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "session": source.session,
        "decision_rows": len(rows),
        "label_rows": int(len(frame)),
        "reference_d48_rows": int(
            frame.get("d48_reference_eligible", pd.Series(dtype=bool)).sum()
        ),
        "missing_contract_paths": int(missing_paths),
        "missing_entry_fill_asks": int(missing_entry_fills),
        "malformed_processed_rows": int(malformed_rows),
        "wall_seconds": float(time.monotonic() - started),
        "user_cpu_seconds": float(usage_end.ru_utime - usage_start.ru_utime),
        "system_cpu_seconds": float(usage_end.ru_stime - usage_start.ru_stime),
        "vol_proxy_name": vol_name,
        "vol_proxy_value": vol_value,
    }
    return frame, summary


def _horizon_key(horizon: int | str) -> str:
    return "session" if horizon == "remaining_session" else f"h{horizon}"


def assign_vol_regimes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    sessions = (
        frame[["session", "vol_proxy_value"]]
        .drop_duplicates("session")
        .sort_values("session")
    )
    values = pd.to_numeric(sessions["vol_proxy_value"], errors="coerce")
    if values.nunique(dropna=True) < 3:
        out = frame.copy()
        out["vol_regime"] = "single_session_smoke"
        value = float(values.dropna().iloc[0]) if values.notna().any() else math.nan
        return out, [value, value]
    q1, q2 = [float(value) for value in values.quantile([1 / 3, 2 / 3])]
    out = frame.copy()
    out["vol_regime"] = pd.cut(
        pd.to_numeric(out["vol_proxy_value"], errors="coerce"),
        bins=[-math.inf, q1, q2, math.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return out, [q1, q2]


def _summary_rows(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[tuple[str, str, str, str]],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for groups, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        group_values = groups if isinstance(groups, tuple) else (groups,)
        group_payload = dict(zip(group_columns, group_values, strict=True))
        for column, family, horizon, unit in value_columns:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **group_payload,
                    "family": family,
                    "horizon": horizon,
                    "unit": unit,
                    "metric": column,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "minimum": float(values.min()),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def family_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    values: list[tuple[str, str, str, str]] = []
    for horizon in HORIZONS:
        key = _horizon_key(horizon)
        horizon_text = str(horizon)
        for column, family, unit in (
            (f"{key}_ttfp_minutes", "time_to_first_profit", "minutes"),
            (f"{key}_ppae_dollars", "pre_profit_adverse_excursion", "dollars"),
            (f"{key}_ppae_return", "pre_profit_adverse_excursion", "return"),
            (f"{key}_uwi_dollars", "underwater_burden", "dollar_minutes"),
            (f"{key}_uwi_return", "underwater_burden", "return_minutes"),
            (f"{key}_fraction_positive", "profitable_window_stability", "share"),
            (f"{key}_jitter", "profitable_window_stability", "share"),
            (f"{key}_mfe_dollars", "upside", "dollars"),
            (f"{key}_mfe_return", "upside", "return"),
            (f"{key}_profit_area_dollars", "upside", "dollar_minutes"),
            (f"{key}_profit_area_return", "upside", "return_minutes"),
        ):
            values.append((column, family, horizon_text, unit))
        if horizon in {3, 5, 10}:
            values.extend(
                [
                    (
                        f"{key}_early_dd_dollars",
                        "early_drawdown",
                        horizon_text,
                        "dollars",
                    ),
                    (
                        f"{key}_early_dd_return",
                        "early_drawdown",
                        horizon_text,
                        "return",
                    ),
                ]
            )
    eligible = frame[frame["d48_reference_eligible"]].copy()
    stratifications = (
        ("market_phase",),
        ("premium_band",),
        ("moneyness_band",),
        ("month",),
        ("vol_regime",),
        ("month", "vol_regime"),
    )
    tables = [
        _summary_rows(
            eligible,
            value_columns=values,
            group_columns=columns,
        ).assign(stratification=" x ".join(columns))
        for columns in stratifications
    ]
    return pd.concat(tables, ignore_index=True, sort=False)


def mnar_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["d48_reference_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    pairs = (
        ("ttfp_minutes", "time_to_first_profit"),
        ("uwi_dollars", "underwater_burden_dollars"),
        ("uwi_return", "underwater_burden_return"),
        ("fraction_positive", "profitable_window_stability"),
        ("mfe_dollars", "upside_dollars"),
        ("mfe_return", "upside_return"),
    )
    for horizon in HORIZONS:
        key = _horizon_key(horizon)
        horizon_pairs = list(pairs)
        if horizon in {3, 5, 10}:
            horizon_pairs.extend(
                [
                    ("early_dd_dollars", "early_drawdown_dollars"),
                    ("early_dd_return", "early_drawdown_return"),
                ]
            )
        for suffix, family in horizon_pairs:
            primary_column = f"{key}_{suffix}"
            excluded_column = f"{key}_mnar_excluded_{suffix}"
            if primary_column not in eligible or excluded_column not in eligible:
                continue
            pair = eligible[[primary_column, excluded_column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if pair.empty:
                continue
            delta = pair[primary_column] - pair[excluded_column]
            rows.append(
                {
                    "horizon": str(horizon),
                    "family": family,
                    "paired_rows": int(len(pair)),
                    "primary_no_bid_as_loss_mean": float(
                        pair[primary_column].mean()
                    ),
                    "sensitivity_no_bid_excluded_mean": float(
                        pair[excluded_column].mean()
                    ),
                    "mean_primary_minus_excluded": float(delta.mean()),
                    "median_primary_minus_excluded": float(delta.median()),
                    "p05_primary_minus_excluded": float(delta.quantile(0.05)),
                    "p95_primary_minus_excluded": float(delta.quantile(0.95)),
                    "spearman_rank_correlation": float(
                        pair[primary_column].corr(
                            pair[excluded_column], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


QUALITY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("h10_early_dd_return", "early_drawdown", "higher"),
    ("session_ttfp_minutes", "time_to_first_profit", "lower"),
    ("session_ppae_return", "pre_profit_adverse_excursion", "higher"),
    ("session_uwi_return", "underwater_burden", "lower"),
    ("session_fraction_positive", "profitable_window_stability", "higher"),
)


def guardrail_masks_and_thresholds(
    frame: pd.DataFrame,
    level: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    masks = {
        family: np.ones(len(frame), dtype=bool)
        for _, family, _ in QUALITY_METRICS
    }
    thresholds: dict[str, dict[str, float]] = {}
    for phase, phase_indices in frame.groupby("market_phase", sort=True).groups.items():
        phase_positions = frame.index.get_indexer(phase_indices)
        for column, family, direction in QUALITY_METRICS:
            values = pd.to_numeric(frame.loc[phase_indices, column], errors="coerce")
            q = level if direction == "higher" else 1.0 - level
            threshold = float(values.quantile(q)) if values.notna().any() else math.nan
            thresholds.setdefault(str(phase), {})[family] = threshold
            observed = pd.to_numeric(frame.loc[phase_indices, column], errors="coerce").to_numpy()
            passed = np.isfinite(observed)
            if math.isfinite(threshold):
                passed &= observed >= threshold if direction == "higher" else observed <= threshold
            else:
                passed[:] = False
            masks[family][phase_positions] = passed
    return masks, thresholds


def add_quality_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ranks: list[pd.Series] = []
    for column, family, direction in QUALITY_METRICS:
        ascending = direction == "higher"
        rank = out.groupby("market_phase", sort=False)[column].rank(
            pct=True,
            ascending=ascending,
            na_option="bottom",
        )
        out[f"quality_rank_{family}"] = rank
        ranks.append(rank)
    out["quality_score"] = pd.concat(ranks, axis=1).mean(axis=1)
    out["conservative_upside_return"] = pd.to_numeric(
        out["session_upside_q75_return"], errors="coerce"
    )
    return out


def _regime_subsets(
    frame: pd.DataFrame,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", frame
    for dimension in ("month", "vol_regime"):
        for value, group in frame.groupby(dimension, sort=True):
            yield dimension, str(value), group


def _pareto_frontier_slice(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[
        frame["d48_reference_eligible"]
        & np.isfinite(frame["quality_score"])
        & np.isfinite(frame["session_mfe_return"])
    ].copy()
    eligible = eligible.sort_values(
        ["quality_score", "session_mfe_return"],
        ascending=[False, False],
    )
    frontier: list[pd.Series] = []
    best_upside = -math.inf
    for _, row in eligible.iterrows():
        upside = float(row["session_mfe_return"])
        if upside > best_upside + 1e-15:
            frontier.append(row)
            best_upside = upside
    columns = [
        "session",
        "decision_time_utc",
        "contract_id",
        "right",
        "offset",
        "market_phase",
        "premium_band",
        "moneyness_band",
        "quality_score",
        "session_mfe_return",
        "session_mfe_dollars",
        "h10_early_dd_return",
        "session_ttfp_minutes",
        "session_uwi_return",
        "session_fraction_positive",
    ]
    if not frontier:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(frontier)[columns].reset_index(drop=True)


def pareto_frontier_table(frame: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for dimension, value, subset in _regime_subsets(frame):
        table = _pareto_frontier_slice(subset)
        if len(table):
            table.insert(0, "regime_value", value)
            table.insert(0, "regime_dimension", dimension)
            tables.append(table)
    if not tables:
        return pd.DataFrame(
            columns=["regime_dimension", "regime_value"]
        )
    return pd.concat(tables, ignore_index=True, sort=False)


def guardrail_curve_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = frame[frame["d48_reference_eligible"]].copy().reset_index(drop=True)
    total_minutes = eligible[["session", "decision_time_ns"]].drop_duplicates()
    curve_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    threshold_payload: dict[str, Any] = {}
    big_winner = pd.to_numeric(eligible["session_mfe_return"], errors="coerce") >= 0.40

    for level in GUARDRAIL_LEVELS:
        masks, thresholds = guardrail_masks_and_thresholds(eligible, level)
        threshold_payload[f"{level:.2f}"] = thresholds
        joint = np.logical_and.reduce(list(masks.values()))
        positive_upside = (
            pd.to_numeric(
                eligible["conservative_upside_return"], errors="coerce"
            ).to_numpy()
            > 0.0
        )
        joint_with_upside = joint & positive_upside
        for dimension, value, subset in _regime_subsets(eligible):
            subset_mask = eligible.index.isin(subset.index)
            subset_minutes = subset[
                ["session", "decision_time_ns"]
            ].drop_duplicates()
            subset_sessions = sorted(subset["session"].unique())
            for label, mask in (
                ("quality_only", joint),
                (
                    "quality_plus_positive_conservative_upside",
                    joint_with_upside,
                ),
            ):
                combined = mask & subset_mask
                qualified = eligible.loc[
                    combined, ["session", "decision_time_ns"]
                ].drop_duplicates()
                per_session = qualified.groupby("session").size()
                curve_rows.append(
                    {
                        "regime_dimension": dimension,
                        "regime_value": value,
                        "guardrail_exclusion_level": level,
                        "screen": label,
                        "eligible_contract_rows": int(combined.sum()),
                        "eligible_minutes": int(len(qualified)),
                        "total_minutes": int(len(subset_minutes)),
                        "qualifying_minute_share": (
                            float(len(qualified) / len(subset_minutes))
                            if len(subset_minutes)
                            else math.nan
                        ),
                        "mean_qualifying_minutes_per_session": float(
                            per_session.reindex(
                                subset_sessions, fill_value=0
                            ).mean()
                        ),
                        "median_qualifying_minutes_per_session": float(
                            per_session.reindex(
                                subset_sessions, fill_value=0
                            ).median()
                        ),
                    }
                )
            subset_big_winner = big_winner & subset_mask
            for family, mask in [*masks.items(), ("joint", joint)]:
                denominator = int(subset_big_winner.sum())
                excluded = int((subset_big_winner & ~mask).sum())
                excluded_rows.append(
                    {
                        "regime_dimension": dimension,
                        "regime_value": value,
                        "guardrail_exclusion_level": level,
                        "family": family,
                        "oracle_big_win_paths": denominator,
                        "excluded_big_win_paths": excluded,
                        "excluded_winners_rate": (
                            float(excluded / denominator)
                            if denominator
                            else math.nan
                        ),
                    }
                )
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(excluded_rows),
        threshold_payload,
    )


def _exit_columns(variant: str) -> tuple[str, str, str, str, str]:
    if variant.startswith("best_"):
        suffix = variant.removeprefix("best_")
        key = "session" if suffix == "session" else f"h{suffix}"
        return (
            f"{key}_best_exit_time_ns",
            f"{key}_best_exit_time_ns",
            f"{key}_deadline_ns",
            f"{key}_best_exit_bid",
            f"{key}_mfe_dollars",
        )
    if variant == "hold_to_forced_flat":
        return (
            "hold_flat_exit_time_ns",
            "hold_flat_source_time_ns",
            "session_deadline_ns",
            "hold_flat_exit_bid",
            "hold_flat_exit_pnl",
        )
    if variant == "first_real_profit_exit":
        return (
            "first_profit_exit_time_ns",
            "first_profit_source_time_ns",
            "session_deadline_ns",
            "first_profit_exit_bid",
            "first_profit_exit_pnl",
        )
    raise CensusContractError(f"unknown oracle variant: {variant}")


def _oracle_variants() -> list[str]:
    return [
        *[
            f"best_{'session' if horizon == 'remaining_session' else horizon}"
            for horizon in HORIZONS
        ],
        "hold_to_forced_flat",
        "first_real_profit_exit",
    ]


def _replay_selection_from_row(
    row_index: int,
    row: pd.Series,
    variant: str,
) -> ReplaySelection | None:
    exit_column, source_column, deadline_column, bid_column, pnl_column = _exit_columns(
        variant
    )
    values = [
        row.get(exit_column),
        row.get(source_column),
        row.get(deadline_column),
        row.get(bid_column),
        row.get(pnl_column),
    ]
    if any(value is None or pd.isna(value) for value in values):
        return None
    decision_ns = int(row["decision_time_ns"])
    exit_ns = int(row[exit_column])
    source_ns = int(row[source_column])
    deadline_ns = int(row[deadline_column])
    exit_bid = float(row[bid_column])
    pnl = float(row[pnl_column])
    if not decision_ns < source_ns <= exit_ns <= deadline_ns:
        return None
    reason = (
        int(ExitReason.FORCED_FLAT)
        if variant in {"hold_to_forced_flat", "first_real_profit_exit"}
        and exit_ns == _forced_flat_ns(str(row["session"]))
        else int(ExitReason.TAKE_PROFIT)
    )
    return ReplaySelection(
        row_index=row_index,
        decision_time_ns=decision_ns,
        exit_time_ns=exit_ns,
        source_exit_time_ns=source_ns,
        deadline_ns=deadline_ns,
        exit_bid=exit_bid,
        pnl=pnl,
        exit_reason=reason,
    )


def _build_serial_candidate(
    row: pd.Series,
    selection: ReplaySelection,
    *,
    variant: str,
    strategy: str,
) -> SerialCandidateV5:
    horizon = variant.removeprefix("best_") if variant.startswith("best_") else "session"
    policy_index = POLICY_INDEX_BY_HORIZON.get(
        "remaining_session" if horizon == "session" else int(horizon)
        if str(horizon).isdigit()
        else "remaining_session",
        6,
    )
    return SerialCandidateV5(
        split="census",
        fold="ft2_05",
        session=str(row["session"]),
        decision_time_ns=int(row["decision_time_ns"]),
        contract_id=str(row["contract_id"]),
        right=str(row["right"]),
        canonical_strike_slot=int(row["strike_idx"]),
        policy_index=int(policy_index),
        entry_ask=float(row["entry_ask"]),
        score=float(selection.pnl),
        raw_label_pnl_after_campaign_fee=float(selection.pnl),
        label_mid_pnl_before_campaign_fee=float(selection.pnl + ROUND_TRIP_FEE),
        label_realized_exit_time_ns=int(selection.exit_time_ns),
        label_source_exit_quote_time_ns=int(selection.source_exit_time_ns),
        label_exit_quote_age_ms=float(
            (selection.exit_time_ns - selection.source_exit_time_ns) / 1_000_000.0
        ),
        label_exit_reason_code=int(selection.exit_reason),
        label_executable_exit_bid=float(selection.exit_bid),
        label_policy_deadline_ns=int(selection.deadline_ns),
        label_invalid_reason_code=int(InvalidReason.NONE),
        feature_hash="",
        source_quote_time_ns=int(row["decision_time_ns"]),
        source_context_time_ns=int(row["decision_time_ns"]) - ONE_MINUTE_NS,
        strategy=strategy,
        metadata={
            "oracle_variant": variant,
            "d48_applied": True,
            "d49_applied": True,
            "source_row_index": int(selection.row_index),
        },
    )


def replay_variant(
    frame: pd.DataFrame,
    *,
    variant: str,
    selector: str,
    eligible_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selector not in {"oracle", "p5", "quality_curve"}:
        raise CensusContractError(f"unknown replay selector: {selector}")
    work = frame.reset_index(drop=True)
    if eligible_mask is None:
        eligible_mask = np.ones(len(work), dtype=bool)
    if len(eligible_mask) != len(work):
        raise CensusContractError("replay eligibility mask length mismatch")

    cash = STARTING_CASH
    candidates: list[SerialCandidateV5] = []
    custom_rows: list[dict[str, Any]] = []
    soft_closed_sessions: dict[str, int] = {}
    for session, session_frame in work.groupby("session", sort=True):
        session_start = cash
        realized_session = 0.0
        pending: tuple[int, float] | None = None
        soft_closed = False
        for decision_ns, minute in session_frame.groupby(
            "decision_time_ns", sort=True
        ):
            decision_ns = int(decision_ns)
            if pending is not None and int(pending[0]) <= decision_ns:
                cash += float(pending[1])
                realized_session += float(pending[1])
                pending = None
            if pending is not None:
                continue
            if soft_closed:
                continue
            daily_budget = PREMIUM_BUDGET_FRACTION * session_start
            realized_loss = max(0.0, -realized_session)
            if realized_session <= -daily_budget + 1e-9:
                continue
            remaining_budget = daily_budget - realized_loss
            if _d49_should_soft_close(
                session_start_equity=session_start,
                realized_session_pnl=realized_session,
            ):
                soft_closed = True
                soft_closed_sessions[str(session)] = decision_ns
                continue
            positions = minute.index.to_numpy(dtype=int)
            candidate_minute = minute.loc[eligible_mask[positions]].copy()
            if candidate_minute.empty:
                continue
            required = pd.to_numeric(
                candidate_minute["premium_plus_fee"], errors="coerce"
            )
            candidate_minute = candidate_minute[
                (required <= daily_budget + 1e-9)
                & (realized_loss + required <= daily_budget + 1e-9)
                & (required <= cash + 1e-9)
            ]
            if candidate_minute.empty:
                continue

            selections: list[tuple[int, pd.Series, ReplaySelection]] = []
            for row_index, row in candidate_minute.iterrows():
                selection = _replay_selection_from_row(row_index, row, variant)
                if selection is not None:
                    selections.append((row_index, row, selection))
            if not selections:
                continue

            if selector == "p5":
                side = str(candidate_minute["vwap_side"].iloc[0])
                selections = [
                    item for item in selections if str(item[1]["right"]) == side
                ]
                if not selections:
                    continue
                chosen = min(
                    selections,
                    key=lambda item: (
                        abs(float(item[1]["offset"])),
                        float(item[1]["offset"]),
                        0 if str(item[1]["right"]) == "C" else 1,
                        str(item[1]["contract_id"]),
                    ),
                )
            elif selector == "quality_curve":
                selections = [
                    item
                    for item in selections
                    if math.isfinite(
                        float(item[1].get("conservative_upside_return", math.nan))
                    )
                    and float(item[1]["conservative_upside_return"]) > 0.0
                ]
                if not selections:
                    continue
                chosen = max(
                    selections,
                    key=lambda item: (
                        float(item[1]["conservative_upside_return"]),
                        float(item[2].pnl),
                        -abs(float(item[1]["offset"])),
                        str(item[1]["contract_id"]),
                    ),
                )
            else:
                chosen = max(
                    selections,
                    key=lambda item: (
                        float(item[2].pnl),
                        -abs(float(item[1]["offset"])),
                        str(item[1]["contract_id"]),
                    ),
                )
                if float(chosen[2].pnl) <= 0.0:
                    continue

            row_index, row, selection = chosen
            candidate = _build_serial_candidate(
                row,
                selection,
                variant=variant,
                strategy=f"ft2_05_{selector}_{variant}",
            )
            candidates.append(candidate)
            pending = (selection.exit_time_ns, selection.pnl)
            custom_rows.append(
                {
                    "selector": selector,
                    "variant": variant,
                    "session": session,
                    "month": str(row["month"]),
                    "vol_regime": str(row["vol_regime"]),
                    "decision_time_ns": decision_ns,
                    "contract_id": str(row["contract_id"]),
                    "right": str(row["right"]),
                    "offset": float(row["offset"]),
                    "premium_band": str(row["premium_band"]),
                    "moneyness_band": str(row["moneyness_band"]),
                    "entry_ask": float(row["entry_ask"]),
                    "exit_bid": float(selection.exit_bid),
                    "exit_time_ns": int(selection.exit_time_ns),
                    "pnl": float(selection.pnl),
                    "return_on_premium": float(
                        selection.pnl / (float(row["entry_ask"]) * MULTIPLIER)
                    ),
                    "session_start_equity": float(session_start),
                    "realized_session_pnl_before": float(realized_session),
                    "d48_cap_dollars": float(daily_budget),
                    "d49_realized_loss": float(realized_loss),
                    "d49_remaining_budget": float(remaining_budget),
                    "d49_soft_close_floor_premium": SOFT_CLOSE_MIN_PREMIUM,
                }
            )
        if pending is not None:
            cash += float(pending[1])
            realized_session += float(pending[1])

    config = SerialSimulatorV5Config(
        starting_cash=STARTING_CASH,
        contract_multiplier=MULTIPLIER,
        campaign_round_trip_fee_dollars=ROUND_TRIP_FEE,
        affordability_reserve_per_trade=ROUND_TRIP_FEE,
        max_trades_per_session=0,
        max_daily_loss_fraction_of_session_start_equity=PREMIUM_BUDGET_FRACTION,
        split_order=("census",),
        lexical_split_order_declared=False,
    )
    trades, state = simulate_serial_candidates_v5(candidates, config=config)
    custom = pd.DataFrame(custom_rows)
    expected_pnl = float(custom["pnl"].sum()) if len(custom) else 0.0
    observed_pnl = float(
        sum(item.raw_label_pnl_after_campaign_fee for item in trades)
    )
    if len(trades) != len(custom) or not math.isclose(
        expected_pnl, observed_pnl, rel_tol=0.0, abs_tol=1e-7
    ):
        raise CensusContractError(
            f"custom D48/D49 selection diverged from simulator v5: "
            f"{selector}/{variant}"
        )
    session_pnl = (
        custom.groupby(
            ["selector", "variant", "session", "month", "vol_regime"],
            as_index=False,
        )
        .agg(
            pnl=("pnl", "sum"),
            trade_count=("pnl", "size"),
            mean_return=("return_on_premium", "mean"),
        )
        if len(custom)
        else pd.DataFrame(
            columns=[
                "selector",
                "variant",
                "session",
                "month",
                "vol_regime",
                "pnl",
                "trade_count",
                "mean_return",
            ]
        )
    )
    all_sessions = frame[
        ["session", "month", "vol_regime"]
    ].drop_duplicates()
    session_pnl = all_sessions.merge(
        session_pnl,
        on=["session", "month", "vol_regime"],
        how="left",
    )
    session_pnl["selector"] = selector
    session_pnl["variant"] = variant
    session_pnl["pnl"] = session_pnl["pnl"].fillna(0.0)
    session_pnl["trade_count"] = session_pnl["trade_count"].fillna(0).astype(int)
    summary = {
        "selector": selector,
        "variant": variant,
        "trade_count": int(len(custom)),
        "pooled_pnl": observed_pnl,
        "ending_cash": float(state.cash_by_account.get("census", STARTING_CASH)),
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "candidate_stream_hash": state.candidate_stream_hash,
        "trade_identity_hash": state.trade_identity_hash,
        "d48_applied_before_simulator": True,
        "d49_applied_before_simulator": True,
        "d49_soft_close_floor_premium": SOFT_CLOSE_MIN_PREMIUM,
        "soft_closed_session_count": len(soft_closed_sessions),
        "soft_closed_sessions": soft_closed_sessions,
        "simulator_skips": dict(state.skipped),
    }
    return session_pnl, {"summary": summary, "trades": custom}


def outcome_bucket(return_value: float) -> str:
    if return_value >= 0.40:
        return "big_win"
    if return_value >= -0.05:
        return "scratch_or_small_win"
    if return_value > -0.30:
        return "small_loss"
    return "big_loss"


def _session_block_ci(
    session_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(session_values, dtype=float)
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    totals = np.empty(replicates, dtype=float)
    for index in range(replicates):
        totals[index] = float(rng.choice(values, size=len(values), replace=True).sum())
    return float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))


def oracle_and_p5_tables(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    session_tables: list[pd.DataFrame] = []
    trade_tables: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    seed_offset = 0
    for variant in _oracle_variants():
        results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
        for selector in ("oracle", "p5"):
            session_table, payload = replay_variant(
                frame,
                variant=variant,
                selector=selector,
            )
            results[selector] = (session_table, payload)
            session_tables.append(session_table)
            trades = payload["trades"].copy()
            if len(trades):
                trades["outcome_bucket"] = trades["return_on_premium"].map(
                    outcome_bucket
                )
                trade_tables.append(trades)
            session_values = session_table["pnl"].to_numpy(dtype=float)
            ci_low, ci_high = _session_block_ci(
                session_values,
                replicates=bootstrap_replicates,
                seed=BOOTSTRAP_SEED + seed_offset,
            )
            seed_offset += 1
            bucket_counts = (
                trades["outcome_bucket"].value_counts().to_dict()
                if len(trades)
                else {}
            )
            trade_count = int(len(trades))
            summaries.append(
                {
                    **payload["summary"],
                    "session_block_ci_95_low": ci_low,
                    "session_block_ci_95_high": ci_high,
                    **{
                        f"bucket_{name}_count": int(bucket_counts.get(name, 0))
                        for name in (
                            "big_win",
                            "scratch_or_small_win",
                            "small_loss",
                            "big_loss",
                        )
                    },
                    **{
                        f"bucket_{name}_share": (
                            float(bucket_counts.get(name, 0) / trade_count)
                            if trade_count
                            else math.nan
                        )
                        for name in (
                            "big_win",
                            "scratch_or_small_win",
                            "small_loss",
                            "big_loss",
                        )
                    },
                }
            )
            audit[f"{selector}:{variant}"] = payload["summary"]
        oracle_pnl = float(results["oracle"][1]["summary"]["pooled_pnl"])
        p5_pnl = float(results["p5"][1]["summary"]["pooled_pnl"])
        for row in summaries:
            if row["variant"] == variant and row["selector"] == "p5":
                row["p5_share_of_oracle_ceiling"] = (
                    p5_pnl / oracle_pnl if oracle_pnl > 0.0 else math.nan
                )
    summary = pd.DataFrame(summaries)
    sessions = pd.concat(session_tables, ignore_index=True)
    trades = (
        pd.concat(trade_tables, ignore_index=True)
        if trade_tables
        else pd.DataFrame()
    )
    return summary, sessions, trades, audit


def guardrail_trade_rate_table(
    frame: pd.DataFrame,
    threshold_payload: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = frame.reset_index(drop=True)
    for level in GUARDRAIL_LEVELS:
        masks, _ = guardrail_masks_and_thresholds(eligible, level)
        joint = np.logical_and.reduce(list(masks.values()))
        session_table, payload = replay_variant(
            eligible,
            variant="best_session",
            selector="quality_curve",
            eligible_mask=joint,
        )
        for dimension, value, subset in _regime_subsets(session_table):
            trade_count = int(subset["trade_count"].sum())
            sessions = int(subset["session"].nunique())
            rows.append(
                {
                    "regime_dimension": dimension,
                    "regime_value": value,
                    "guardrail_exclusion_level": level,
                    "trade_count": trade_count,
                    "sessions": sessions,
                    "mean_trades_per_session": float(
                        trade_count / max(sessions, 1)
                    ),
                    "pooled_pnl_oracle_assisted": float(
                        subset["pnl"].sum()
                    ),
                    "threshold_hash": stable_hash(
                        threshold_payload[f"{level:.2f}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def friction_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["d48_reference_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    for columns in (
        ("premium_band",),
        ("premium_band", "month"),
        ("premium_band", "vol_regime"),
        ("premium_band", "moneyness_band"),
    ):
        for groups, group in eligible.groupby(list(columns), dropna=False, sort=True):
            values = groups if isinstance(groups, tuple) else (groups,)
            friction = pd.to_numeric(
                group["friction_fraction_of_premium"], errors="coerce"
            ).dropna()
            spread = pd.to_numeric(group["entry_spread"], errors="coerce").dropna()
            if friction.empty:
                continue
            rows.append(
                {
                    **dict(zip(columns, values, strict=True)),
                    "stratification": " x ".join(columns),
                    "contract_minutes": int(len(group)),
                    "friction_mean": float(friction.mean()),
                    "friction_median": float(friction.median()),
                    "friction_p90": float(friction.quantile(0.90)),
                    "friction_p95": float(friction.quantile(0.95)),
                    "entry_spread_median": (
                        float(spread.median()) if len(spread) else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def regime_headline_table(
    frame: pd.DataFrame,
    oracle_sessions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = frame[frame["d48_reference_eligible"]]
    for dimension in ("month", "vol_regime"):
        for value, group in eligible.groupby(dimension, sort=True):
            rows.append(
                {
                    "headline_type": "opportunity_surface",
                    "regime_dimension": dimension,
                    "regime_value": str(value),
                    "session_count": int(group["session"].nunique()),
                    "reference_d48_contract_minutes": int(len(group)),
                    "reference_d48_distinct_minutes": int(
                        len(group[["session", "decision_time_ns"]].drop_duplicates())
                    ),
                    "big_win_path_count": int(
                        (
                            pd.to_numeric(
                                group["session_mfe_return"], errors="coerce"
                            )
                            >= 0.40
                        ).sum()
                    ),
                    "median_friction_fraction": float(
                        pd.to_numeric(
                            group["friction_fraction_of_premium"], errors="coerce"
                        ).median()
                    ),
                }
            )
        for (selector, variant, value), group in oracle_sessions.groupby(
            ["selector", "variant", dimension], sort=True
        ):
            rows.append(
                {
                    "headline_type": "serial_replay",
                    "regime_dimension": dimension,
                    "regime_value": str(value),
                    "selector": str(selector),
                    "variant": str(variant),
                    "session_count": int(group["session"].nunique()),
                    "pnl": float(group["pnl"].sum()),
                    "trade_count": int(group["trade_count"].sum()),
                    "mean_session_pnl": float(group["pnl"].mean()),
                }
            )
    return pd.DataFrame(rows)


def power_tables(
    oracle_sessions: pd.DataFrame,
    oracle_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variance_rows: list[dict[str, Any]] = []
    mde_rows: list[dict[str, Any]] = []
    z_value = float(norm.ppf(0.975) + norm.ppf(0.80))
    for dimension, value, session_subset in _regime_subsets(oracle_sessions):
        merged = session_subset.pivot_table(
            index=["session"],
            columns=["selector", "variant"],
            values="pnl",
            aggfunc="sum",
            fill_value=0.0,
        )
        variants = sorted(set(session_subset["variant"].astype(str)))
        for variant in variants:
            oracle_key = ("oracle", variant)
            p5_key = ("p5", variant)
            if oracle_key not in merged or p5_key not in merged:
                continue
            difference = (
                merged[oracle_key].to_numpy(dtype=float)
                - merged[p5_key].to_numpy(dtype=float)
            )
            session_std = (
                float(np.std(difference, ddof=1))
                if len(difference) > 1
                else 0.0
            )
            variance_rows.append(
                {
                    "regime_dimension": dimension,
                    "regime_value": value,
                    "variant": variant,
                    "session_count": int(len(difference)),
                    "mean_session_increment": float(np.mean(difference)),
                    "session_increment_variance": (
                        float(np.var(difference, ddof=1))
                        if len(difference) > 1
                        else 0.0
                    ),
                    "session_increment_std": session_std,
                }
            )
            for session_count in (20, 30, 45, 60, 100, 150, 225, 301):
                mde_rows.append(
                    {
                        "regime_dimension": dimension,
                        "regime_value": value,
                        "variant": variant,
                        "basis": "session_cluster",
                        "sample_count": session_count,
                        "mde_total_pnl_95pct_80pct_power": (
                            z_value * session_std * math.sqrt(session_count)
                        ),
                        "mde_mean_session_pnl_95pct_80pct_power": (
                            z_value * session_std / math.sqrt(session_count)
                        ),
                    }
                )

    if len(oracle_trades):
        for dimension, value, trade_subset in _regime_subsets(oracle_trades):
            for (selector, variant), group in trade_subset.groupby(
                ["selector", "variant"], sort=True
            ):
                values = pd.to_numeric(
                    group["pnl"], errors="coerce"
                ).dropna().to_numpy()
                trade_std = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
                sessions = max(int(group["session"].nunique()), 1)
                trades_per_session = float(len(group) / sessions)
                session_totals = group.groupby("session")["pnl"].sum()
                session_var = (
                    float(session_totals.var(ddof=1))
                    if len(session_totals) > 1
                    else 0.0
                )
                independent_var = max(
                    trades_per_session * trade_std**2, 1e-12
                )
                design_effect = max(session_var / independent_var, 1.0)
                for trade_count in (50, 100, 200, 400, 800, 1200):
                    mde_rows.append(
                        {
                            "regime_dimension": dimension,
                            "regime_value": value,
                            "variant": str(variant),
                            "selector": str(selector),
                            "basis": "trade_count_cluster_adjusted",
                            "sample_count": trade_count,
                            "estimated_design_effect": design_effect,
                            "mde_total_pnl_95pct_80pct_power": (
                                z_value
                                * trade_std
                                * math.sqrt(trade_count * design_effect)
                            ),
                            "mde_mean_trade_pnl_95pct_80pct_power": (
                                z_value
                                * trade_std
                                * math.sqrt(design_effect / trade_count)
                            ),
                        }
                    )
    return pd.DataFrame(variance_rows), pd.DataFrame(mde_rows)


def _top_breakdown(
    frame: pd.DataFrame,
    column: str,
    *,
    count: int = 4,
) -> list[dict[str, Any]]:
    values = frame[column].value_counts(dropna=False).head(count)
    total = max(int(values.sum()), 1)
    return [
        {
            column: str(index),
            "count": int(value),
            "share": float(value / total),
        }
        for index, value in values.items()
    ]


def determine_outcome(
    frame: pd.DataFrame,
    oracle_summary: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    reference = frame[frame["d48_reference_eligible"]]
    if reference.empty:
        reasons.append("no D48-eligible contract-minute labels")
        return "genuine_impossibility", reasons
    label_coverage = float(
        (
            pd.to_numeric(reference["session_available_minutes"], errors="coerce")
            > 0
        ).mean()
    )
    if label_coverage < 0.80:
        reasons.append(f"usable remaining-session label coverage only {label_coverage:.3f}")
        return "genuine_impossibility", reasons
    oracle = oracle_summary[oracle_summary["selector"] == "oracle"]
    positive_trades = int(oracle["trade_count"].sum())
    if positive_trades == 0 or float(oracle["pooled_pnl"].max()) <= 0.0:
        reasons.append("no executable positive post-fee oracle opportunity")
        return "genuine_impossibility", reasons
    if (
        pd.to_numeric(oracle["session_block_ci_95_high"], errors="coerce")
        <= 0.0
    ).all():
        reasons.append("every oracle ceiling is indistinguishable from fee drag")
        return "genuine_impossibility", reasons
    reasons.append(
        "executable post-fee opportunities exist; FT2-05 cannot green-light training"
    )
    return "feasible", reasons


def write_table(out_dir: Path, name: str, frame: pd.DataFrame) -> Path:
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def render_report(
    *,
    outcome: str,
    reasons: list[str],
    frame: pd.DataFrame,
    guardrail_curve: pd.DataFrame,
    trade_rate_curve: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    friction: pd.DataFrame,
    mde: pd.DataFrame,
    mnar_sensitivity: pd.DataFrame,
    v1_v2_impact: dict[str, Any],
    compute_record: dict[str, Any],
) -> str:
    eligible = frame[frame["d48_reference_eligible"]]
    sessions = int(frame["session"].nunique())
    eligible_minutes = int(
        len(eligible[["session", "decision_time_ns"]].drop_duplicates())
    )
    opportunities_per_day = eligible_minutes / max(sessions, 1)
    loose = guardrail_curve[
        (guardrail_curve["regime_dimension"] == "overall")
        & (guardrail_curve["guardrail_exclusion_level"] == 0.25)
        & (
            guardrail_curve["screen"]
            == "quality_plus_positive_conservative_upside"
        )
    ]
    loose_per_day = (
        float(loose["mean_qualifying_minutes_per_session"].iloc[0])
        if len(loose)
        else math.nan
    )
    oracle_best = oracle_summary[
        (oracle_summary["selector"] == "oracle")
        & (oracle_summary["variant"] == "best_session")
    ]
    p5_best = oracle_summary[
        (oracle_summary["selector"] == "p5")
        & (oracle_summary["variant"] == "best_session")
    ]
    oracle_pnl = float(oracle_best["pooled_pnl"].iloc[0]) if len(oracle_best) else 0.0
    p5_pnl = float(p5_best["pooled_pnl"].iloc[0]) if len(p5_best) else 0.0
    median_friction = (
        float(
            friction[
                friction["stratification"] == "premium_band"
            ]["friction_median"].median()
        )
        if len(friction)
        else math.nan
    )
    mde_45 = mde[
        (mde["regime_dimension"] == "overall")
        & (mde["basis"] == "session_cluster")
        & (mde["sample_count"] == 45)
        & (mde["variant"] == "best_session")
    ]
    mde_value = (
        float(mde_45["mde_total_pnl_95pct_80pct_power"].iloc[0])
        if len(mde_45)
        else math.nan
    )
    phase = _top_breakdown(eligible, "market_phase")
    money = _top_breakdown(eligible, "moneyness_band")
    premium = _top_breakdown(eligible, "premium_band")
    reason_text = "; ".join(reasons)
    no_bid_minutes = int(
        pd.to_numeric(
            eligible["session_no_bid_minutes"], errors="coerce"
        ).fillna(0).sum()
    )
    path_minutes = int(
        pd.to_numeric(
            eligible["session_nominal_minutes"], errors="coerce"
        ).fillna(0).sum()
    )
    no_bid_rate = no_bid_minutes / max(path_minutes, 1)
    minimum_mnar_rank = float(
        pd.to_numeric(
            mnar_sensitivity["spearman_rank_correlation"], errors="coerce"
        ).min()
    )
    scalar_impact = v1_v2_impact["scalar_changes"]
    oracle_impact = {
        (str(item["selector"]), str(item["variant"])): item
        for item in v1_v2_impact["oracle_changes"]
    }
    best_impact = oracle_impact[("oracle", "best_session")]

    return f"""# Protocol101 FT2-05 Opportunity Census

## Owner Memo

**Outcome: `{outcome}`.** {reason_text}

This is a label-side map of the governed game, not evidence that a profitable
model exists. Across `{sessions}` development-only sessions, the fixed $10,000
D48 reference mask left `{eligible_minutes:,}` distinct executable minutes
with at least one contract, or about `{opportunities_per_day:,.1f}` minutes per
day. A deliberately loose 25%-tail quality screen plus positive conservative
upside left about `{loose_per_day:,.1f}` qualifying minutes per day. This is a
curve diagnostic, not a selected guardrail.

The hindsight `best_session` selector made `${oracle_pnl:,.2f}` under one-account
simulator-v5 semantics, D48, D49, fees, and the daily stop. P5 using the same
hindsight exit rule made `${p5_pnl:,.2f}`. These are ceilings: they prove paths
exist, not that present-time features can identify them. The median entry
friction across premium bands was `{median_friction:.1%}` of premium.

The primary remaining-session paths contained `{no_bid_minutes:,}` no-bid
minutes out of `{path_minutes:,}` marks (`{no_bid_rate:.2%}`). Those minutes are
valued at full loss rather than erased. Across the preregistered MNAR
no-bid-excluded sensitivity, the minimum path-metric rank correlation was
`{minimum_mnar_rank:.4f}`. This is label-surface stability, not candidate-policy
stability; FT2-10/11 must still verify the latter.

Compared with preserved FT2-05 v1, v2 removes
`{int(-scalar_impact['label_rows_all_governed_candidates']['v2_minus_v1']):,}`
candidate rows and one embargo session. The transparent best-session oracle
changed by `${float(best_impact['pooled_pnl_delta']):,.2f}` and
`{int(best_impact['trade_count_delta'])}` trades. This impact combines the
next-minute fill law, no-bid law, exact-boundary flat, embargo exclusion, and
D49 soft-close; it must not be attributed to any one change in isolation.

The 45-session cluster calculation estimates that a best-session improvement
would need to be roughly `${mde_value:,.2f}` in total PnL to achieve conventional
95%/80% detection under the observed session variance. FT2-11 must refine this
before paid training.

Where D48-reference opportunities live:

- Market phase: `{json.dumps(phase, sort_keys=True)}`.
- Moneyness: `{json.dumps(money, sort_keys=True)}`.
- Premium band: `{json.dumps(premium, sort_keys=True)}`.

**Regime warning:** this census sees only January through early March 2025.
April 2025 volatility is in fold-1 outer test, and May-June 2025 is protected
holdout. No conclusion here may be generalized to those regimes. Every headline
table is therefore also stratified by month and the governed SPX
regular-session realized-range proxy.

The only allowed interpretation of `feasible` is: the census characterizes the
opportunity surface of the governed game on the 45-session window, and FT2-08
design work may start. It does not authorize design acceptance, training,
selection, holdout access, shadow operation, or paper trading.

## Frozen Scope

- Node: `{NODE_ID}`.
- Product-contract SHA-256: `{AUTHORITY_SHA256}`.
- Sessions: exactly 45 from the repaired FT2-04 manifest.
- Horizons: `{HORIZONS}`.
- Entry/exit accounting: BUY decision at `t` fills at the `t+1` executable ask;
  EXIT decision at `v` fills at the `v+1` executable bid; $3 round trip.
- No-bid accounting: full-loss path state; exact 15:55 forced-flat boundary.
- D49 soft-close: sub-$1 contracts cannot keep the lane open once remaining
  daily budget cannot fund a $1 contract plus fees.
- Hard safety: D48 premium cap, D49 remaining loss budget, one open position,
  15:30 entry cutoff, 15:55 forced flat.
- Volatility proxy: session SPX realized range in basis points.
- No model was fitted or scored.

## Threshold-Independent Surface

The packet contains the complete family distributions, Pareto frontier,
guardrail curves, excluded-winner curves, and oracle-assisted trade-rate curves.
Guardrail levels are percentile screens across `0%, 10%, 20%, 25%, 30%, 40%,
50%`; none is selected here.

## Oracle And P5 Ceilings

All nine transparent variants are reported: seven best-achievable-bid horizons,
hold-to-forced-flat, and first-real-profit. P5 controls direction and nearest
eligible-under-cap strike only; each comparison uses the same exit oracle to
isolate its entry choice.

## Friction, Power, And Regimes

Premium-band friction includes spread crossing plus the $3 fee. MDE tables use
session clusters and separately show cluster-adjusted trade-count projections.
Regime headline tables split every replay by month and SPX range tercile.

## Compute

- Label rows: `{len(frame):,}`.
- D48-reference rows: `{len(eligible):,}`.
- Wall time: `{compute_record['wall_seconds'] / 3600.0:.3f}` hours.
- CPU time: `{compute_record['core_hours']:.3f}` core-hours.
- Resume checkpoints: one Parquet file and summary per session.

## Route

Terminal outcome is `{outcome}`. The node stops here. FT2-08 was not started.
"""


def _checkpoint_paths(out_dir: Path, session: str) -> tuple[Path, Path]:
    root = out_dir / "checkpoint"
    return (
        root / "session_labels" / f"{session}.parquet",
        root / "session_summaries" / f"{session}.json",
    )


def build_or_resume_labels(
    sources: Sequence[SessionSource],
    *,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    state_path = out_dir / "checkpoint" / "state.json"
    completed: list[str] = []
    for index, source in enumerate(sources, start=1):
        label_path, summary_path = _checkpoint_paths(out_dir, source.session)
        if label_path.exists() and summary_path.exists():
            frame = pd.read_parquet(label_path)
            summary = read_json(summary_path)
            if set(frame["session"].astype(str).unique()) != {source.session}:
                raise CensusContractError(
                    f"checkpoint session identity mismatch: {source.session}"
                )
        else:
            frame, summary = build_session_labels(source)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(label_path, index=False, compression="zstd")
            summary = {
                **summary,
                "label_path": str(label_path),
                "label_sha256": file_sha256(label_path),
            }
            write_json(summary_path, summary)
        if file_sha256(label_path) != str(summary["label_sha256"]):
            raise CensusContractError(
                f"checkpoint label hash mismatch: {source.session}"
            )
        frames.append(frame)
        summaries.append(summary)
        completed.append(source.session)
        write_json(
            state_path,
            {
                "schema_version": "Protocol101FT205CheckpointStateV2",
                "node": NODE_ID,
                "status": "label_build_in_progress",
                "completed_sessions": completed,
                "completed_count": len(completed),
                "required_count": len(sources),
                "last_completed_session": source.session,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "progress_fraction": index / max(len(sources), 1),
            },
        )
    return pd.concat(frames, ignore_index=True, sort=False), summaries


def _file_inventory(out_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(out_dir)): file_sha256(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }


def write_v1_v2_impact(
    *,
    out_dir: Path,
    result_v2: dict[str, Any],
    oracle_summary_v2: pd.DataFrame,
) -> dict[str, Any]:
    v1_dir = out_dir / "superseded" / "v1_pre_tplus1_20260729"
    v1_result_path = v1_dir / "census_results.json"
    v1_oracle_path = v1_dir / "oracle_ceiling_summary.csv"
    if not v1_result_path.is_file() or not v1_oracle_path.is_file():
        raise CensusContractError(
            "preserved FT2-05 v1 packet is required for the v1-v2 impact note"
        )
    result_v1 = read_json(v1_result_path)
    oracle_v1 = pd.read_csv(v1_oracle_path)
    keys = (
        "session_count",
        "label_rows_all_governed_candidates",
        "label_rows_d48_reference",
        "distinct_d48_reference_minutes",
        "remaining_session_label_coverage",
    )
    scalar_changes: dict[str, Any] = {}
    for key in keys:
        before = result_v1.get(key)
        after = result_v2.get(key)
        scalar_changes[key] = {
            "v1": _finite_or_none(before),
            "v2": _finite_or_none(after),
            "v2_minus_v1": (
                float(after) - float(before)
                if isinstance(before, (int, float))
                and isinstance(after, (int, float))
                else None
            ),
        }

    join_keys = ["selector", "variant"]
    columns = [*join_keys, "trade_count", "pooled_pnl"]
    merged = oracle_v1[columns].merge(
        oracle_summary_v2[columns],
        on=join_keys,
        how="outer",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    merged["trade_count_delta"] = (
        pd.to_numeric(merged["trade_count_v2"], errors="coerce")
        - pd.to_numeric(merged["trade_count_v1"], errors="coerce")
    )
    merged["pooled_pnl_delta"] = (
        pd.to_numeric(merged["pooled_pnl_v2"], errors="coerce")
        - pd.to_numeric(merged["pooled_pnl_v1"], errors="coerce")
    )
    payload = {
        "schema_version": "Protocol101FT205V1V2ImpactV1",
        "v1_receipt_sha256": file_sha256(v1_dir / "receipt.json"),
        "v2_product_contract_hash": AUTHORITY_SHA256,
        "semantic_changes": [
            "same-minute entry/exit prices replaced by next-completed-minute fills",
            "no-bid minutes changed from exclusion/earlier-bid substitution to full-loss states",
            "exact 15:55 forced-flat valuation replaces latest-earlier executable bid",
            "the five fold embargo sessions are excluded (46 sessions becomes 45)",
            "D49 permanently soft-closes new entries when remaining budget cannot fund a $1 premium plus fees",
        ],
        "scalar_changes": scalar_changes,
        "oracle_changes": json_safe_records(merged),
        "interpretation": (
            "v1 and v2 economics are not directly interchangeable because execution, "
            "missingness, role membership, and serial budget semantics all changed"
        ),
    }
    write_json(out_dir / "v1_v2_impact.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    overall_started = time.monotonic()
    overall_usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt04, label_spec, oracle_rules, census_manifest = (
        verify_authority_and_inputs()
    )
    if args.force and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    write_json(
        progress_path,
        {
            "node": NODE_ID,
            "status": "authority_verified",
            "product_contract_hash": AUTHORITY_SHA256,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    sources, governance = resolve_session_sources(
        census_manifest,
        smoke_session=args.smoke_session,
    )
    write_json(args.out_dir / "governance_receipt.json", governance)
    write_json(
        progress_path,
        {
            "node": NODE_ID,
            "status": "building_path_labels",
            "session_count": len(sources),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    prior_compute_path = args.out_dir / "label_build_compute.json"
    prior_compute = (
        read_json(prior_compute_path)
        if prior_compute_path.is_file() and not args.force
        else None
    )
    label_started = time.monotonic()
    label_usage = resource.getrusage(resource.RUSAGE_SELF)
    frame, session_summaries = build_or_resume_labels(
        sources,
        out_dir=args.out_dir,
    )
    label_usage_end = resource.getrusage(resource.RUSAGE_SELF)
    label_compute = {
        "wall_seconds": float(time.monotonic() - label_started),
        "user_cpu_seconds": float(label_usage_end.ru_utime - label_usage.ru_utime),
        "system_cpu_seconds": float(
            label_usage_end.ru_stime - label_usage.ru_stime
        ),
    }
    label_compute["core_hours"] = (
        label_compute["user_cpu_seconds"] + label_compute["system_cpu_seconds"]
    ) / 3600.0
    if (
        prior_compute
        and len(prior_compute.get("per_session", [])) == len(sources)
        and all(
            key in prior_compute
            for key in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "core_hours",
            )
        )
    ):
        resumed_load = label_compute
        label_compute = {
            key: prior_compute[key]
            for key in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "core_hours",
            )
        }
        label_compute["checkpoint_resume_used"] = True
        label_compute["latest_checkpoint_load"] = resumed_load

    frame, vol_cutpoints = assign_vol_regimes(frame)
    frame = add_quality_scores(frame)
    if args.smoke_session:
        smoke_path = args.out_dir / "smoke_labels.parquet"
        frame.to_parquet(smoke_path, index=False, compression="zstd")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "node": NODE_ID,
            "status": "engineering_smoke_complete_nonterminal",
            "session": args.smoke_session,
            "row_count": int(len(frame)),
            "d48_reference_rows": int(frame["d48_reference_eligible"].sum()),
            "label_compute": label_compute,
        }
        write_json(args.out_dir / "smoke_summary.json", payload)
        return payload

    if [item.session for item in sources] != list(
        map(str, census_manifest["census_sessions"])
    ):
        raise CensusContractError("full run did not load the exact frozen session set")
    if int(frame["session"].nunique()) != 45:
        raise CensusContractError("full census label frame does not contain 45 sessions")

    write_json(
        progress_path,
        {
            "node": NODE_ID,
            "status": "aggregating_census",
            "label_rows": int(len(frame)),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    distributions = family_distribution_table(frame)
    mnar_sensitivity = mnar_sensitivity_table(frame)
    pareto = pareto_frontier_table(frame)
    guardrail_curve, excluded_winners, guardrail_thresholds = (
        guardrail_curve_tables(frame)
    )
    oracle_summary, oracle_sessions, oracle_trades, replay_audit = (
        oracle_and_p5_tables(
            frame,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    )
    trade_rate_curve = guardrail_trade_rate_table(
        frame,
        guardrail_thresholds,
    )
    friction = friction_table(frame)
    variance, mde = power_tables(oracle_sessions, oracle_trades)
    regimes = regime_headline_table(frame, oracle_sessions)
    outcome, outcome_reasons = determine_outcome(frame, oracle_summary)

    table_map = {
        "family_distributions": distributions,
        "mnar_sensitivity": mnar_sensitivity,
        "pareto_frontier": pareto,
        "guardrail_curves": guardrail_curve,
        "excluded_winners": excluded_winners,
        "guardrail_trade_rates": trade_rate_curve,
        "oracle_ceiling_summary": oracle_summary,
        "oracle_session_results": oracle_sessions,
        "oracle_trade_results": oracle_trades,
        "friction_by_premium_band": friction,
        "session_variance_components": variance,
        "minimum_detectable_improvement": mde,
        "regime_headlines": regimes,
        "label_session_inventory": pd.DataFrame(session_summaries),
    }
    table_paths = {
        name: str(write_table(args.out_dir, name, table))
        for name, table in table_map.items()
    }
    write_json(
        args.out_dir / "guardrail_threshold_curves.json",
        guardrail_thresholds,
    )
    write_json(args.out_dir / "oracle_replay_audit.json", replay_audit)

    overall_usage_end = resource.getrusage(resource.RUSAGE_SELF)
    compute_record = {
        **label_compute,
        "overall_wall_seconds": float(time.monotonic() - overall_started),
        "overall_user_cpu_seconds": float(
            overall_usage_end.ru_utime - overall_usage.ru_utime
        ),
        "overall_system_cpu_seconds": float(
            overall_usage_end.ru_stime - overall_usage.ru_stime
        ),
        "process_max_rss_platform_units": int(overall_usage_end.ru_maxrss),
        "host_cpu_count": int(os.cpu_count() or 1),
        "recorder_hours_observed": False,
        "execution_window_note": (
            "Full run started outside the 05:30-13:15 local recorder window."
        ),
        "per_session": session_summaries,
    }
    write_json(args.out_dir / "label_build_compute.json", compute_record)

    eligible = frame[frame["d48_reference_eligible"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "node": NODE_ID,
        "outcome": outcome,
        "highest_allowed_claim": (
            "the census characterizes the opportunity surface of the governed "
            "game on the 45-session window; "
            + (
                "FT2-08 design is unblocked"
                if outcome == "feasible"
                else "owner decision required"
            )
        ),
        "outcome_reasons": outcome_reasons,
        "product_contract_hash": AUTHORITY_SHA256,
        "session_count": 45,
        "session_window": {
            "first": str(frame["session"].min()),
            "last": str(frame["session"].max()),
        },
        "label_rows_all_governed_candidates": int(len(frame)),
        "label_rows_d48_reference": int(len(eligible)),
        "distinct_d48_reference_minutes": int(
            len(eligible[["session", "decision_time_ns"]].drop_duplicates())
        ),
        "remaining_session_label_coverage": float(
            (
                pd.to_numeric(
                    eligible["session_available_minutes"], errors="coerce"
                )
                > 0
            ).mean()
        ),
        "remaining_session_no_bid_minutes": int(
            pd.to_numeric(
                eligible["session_no_bid_minutes"], errors="coerce"
            ).fillna(0).sum()
        ),
        "remaining_session_total_path_minutes": int(
            pd.to_numeric(
                eligible["session_nominal_minutes"], errors="coerce"
            ).fillna(0).sum()
        ),
        "vol_proxy": {
            "name": "session_spx_realized_range_bps",
            "tercile_cutpoints": vol_cutpoints,
        },
        "regime_caveat": (
            "January through early March 2025 only; April 2025 is outer-test "
            "and May-June 2025 is protected holdout."
        ),
        "guardrail_levels_are_diagnostic_not_selected": list(GUARDRAIL_LEVELS),
        "tables": {
            name: {
                "path": path,
                "row_count": int(len(table_map[name])),
                "records": json_safe_records(table_map[name]),
            }
            for name, path in table_paths.items()
        },
        "input_contracts": {
            "label_spec_sha256": file_sha256(FREEZE_DIR / "label_spec.json"),
            "oracle_rules_sha256": file_sha256(FREEZE_DIR / "oracle_rules.json"),
            "census_sessions_sha256": file_sha256(
                FREEZE_DIR / "census_sessions.json"
            ),
            "label_spec_schema": label_spec.get("schema_version"),
            "oracle_rule_schema": oracle_rules.get("schema_version"),
            "ft2_04_receipt_schema": receipt04.get("schema_version"),
        },
        "side_effects": {
            "model_training": False,
            "model_fitting": False,
            "threshold_tuning": False,
            "gpu_or_paid_compute": False,
            "broker_or_api": False,
            "recorder_day_access": False,
            "paid_download": False,
            "runtime_change": False,
            "promotion_or_paper": False,
            "outer_test_access": False,
            "protected_holdout_access": False,
            "ft2_08_started": False,
        },
    }
    impact = write_v1_v2_impact(
        out_dir=args.out_dir,
        result_v2=result,
        oracle_summary_v2=oracle_summary,
    )
    result["v1_v2_impact"] = {
        "path": str(args.out_dir / "v1_v2_impact.json"),
        "semantic_changes": impact["semantic_changes"],
    }
    write_json(args.out_dir / "census_results.json", result)
    (args.out_dir / "report.md").write_text(
        render_report(
            outcome=outcome,
            reasons=outcome_reasons,
            frame=frame,
            guardrail_curve=guardrail_curve,
            trade_rate_curve=trade_rate_curve,
            oracle_summary=oracle_summary,
            friction=friction,
            mde=mde,
            mnar_sensitivity=mnar_sensitivity,
            v1_v2_impact=impact,
            compute_record=compute_record,
        )
    )
    write_json(
        args.out_dir / "checkpoint" / "state.json",
        {
            "schema_version": "Protocol101FT205CheckpointStateV2",
            "node": NODE_ID,
            "status": "complete",
            "completed_sessions": [item.session for item in sources],
            "completed_count": 45,
            "required_count": 45,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    write_json(
        progress_path,
        {
            "node": NODE_ID,
            "status": "complete",
            "outcome": outcome,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    deliverable_hashes = _file_inventory(args.out_dir)
    receipt = {
        "schema_version": "Protocol101FT205NodeReceiptV2",
        "node": NODE_ID,
        "outcome": outcome,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repair_of": {
            "schema_version": "Protocol101FT205NodeReceiptV1",
            "receipt_sha256": (
                "9085a09cbc1583973fdb372c78d78568e3fb1baa0fa1d9b42c23b05543ee24de"
            ),
            "preserved_path": (
                "superseded/v1_pre_tplus1_20260729/receipt.json"
            ),
        },
        "product_contract_hash": AUTHORITY_SHA256,
        "graph_sha256": file_sha256(GRAPH_PATH),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "input_hashes": {
            "label_spec.json": file_sha256(FREEZE_DIR / "label_spec.json"),
            "oracle_rules.json": file_sha256(FREEZE_DIR / "oracle_rules.json"),
            "census_sessions.json": file_sha256(
                FREEZE_DIR / "census_sessions.json"
            ),
        },
        "session_count": 45,
        "session_manifest_hash": stable_hash([item.session for item in sources]),
        "governance_hash": governance["governance_hash"],
        "deliverable_hashes": deliverable_hashes,
        "side_effects": result["side_effects"],
        "next": (
            "FT2-08-DATA-TENSOR-LABEL-CONTRACT"
            if outcome == "feasible"
            else "STOP-OWNER-DECISION"
        ),
    }
    write_json(args.out_dir / "receipt.json", receipt)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        if args.out_dir.exists():
            write_json(
                args.out_dir / "failure.json",
                {
                    "node": NODE_ID,
                    "outcome": "scientific_contract_defect",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                },
            )
        raise
    print(
        json.dumps(
            {
                "node": NODE_ID,
                "outcome": result.get("outcome") or result.get("status"),
                "session_count": result.get("session_count", 1),
                "label_rows": result.get("label_rows_all_governed_candidates")
                or result.get("row_count"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
