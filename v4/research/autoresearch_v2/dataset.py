"""Holdout-incapable loader for the verified Protocol101 two-clock corpus."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from v4.model.protocol101_regimen_repair import TWO_CLOCK_PROCESSED_ROW_SCHEMA


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SESSION_MANIFEST = ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
    "session_assignments.json"
)
TWO_CLOCK_ROOT = ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_entry_campaign_execution_attempt001/"
    "two_clock_processed"
)
RECEIPT_ROOT = TWO_CLOCK_ROOT.parent / "two_clock_materialization_receipts"

POLICIES = (
    "ask_to_bid_stop35_target60_hold10m",
    "ask_to_bid_stop50_target100_hold25m",
    "ask_to_bid_stop65_target150_hold45m",
    "ask_to_bid_stop50_target200_hold90m",
    "ask_to_bid_stop100_target300_hold120m",
    "ask_to_bid_stop100_target999_hold384m",
    "ask_to_bid_stop100_target9900_hold384m",
)
POLICY_INDEX = {name: index for index, name in enumerate(POLICIES)}

# Historical name retained for continuity; the authoritative tuple contains 18 columns.
SIGNED17 = (
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

FEATURE_INVENTORY = {
    name: {
        "family": "signed17",
        "available_at": "completed_minute_plus_60s",
        "live_twin": (
            "Protocol101 completed-minute SPX/VIX context"
            if not name.startswith("bs_") and name != "is_call"
            else "Protocol101 live option ladder"
        ),
    }
    for name in SIGNED17
}


class FoundationMismatch(RuntimeError):
    pass


class HoldoutAccessRefused(RuntimeError):
    pass


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _assignments(path: str | Path = DEFAULT_SESSION_MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def protected_sessions(path: str | Path = DEFAULT_SESSION_MANIFEST) -> frozenset[str]:
    assignments = _assignments(path)
    return frozenset(
        set(assignments["protected_holdout_30"])
        | set(assignments.get("burned_smoke_dates", ()))
    )


def development_sessions(path: str | Path = DEFAULT_SESSION_MANIFEST) -> tuple[str, ...]:
    assignments = _assignments(path)
    forbidden = protected_sessions(path) | frozenset(
        assignments.get("opra_degraded_diagnostic_only", ())
    )
    available = {item.stem for item in TWO_CLOCK_ROOT.glob("*.pkl")}
    return tuple(
        session
        for session in assignments["pre_holdout_session_indices_1_215"]
        if session not in forbidden and session in available
    )


def freeze_foundation(
    output_path: str | Path,
    *,
    session_manifest: str | Path = DEFAULT_SESSION_MANIFEST,
) -> dict[str, Any]:
    """Freeze the current development-only byte foundation once.

    This function has no code path that accepts a holdout session. Existing
    manifests are never overwritten.
    """

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"foundation already exists: {output}")
    sessions = development_sessions(session_manifest)
    if len(sessions) < 100:
        raise FoundationMismatch("fewer than 100 two-clock development sessions")
    forbidden = protected_sessions(session_manifest)
    rows = []
    for session in sessions:
        if session in forbidden:
            raise HoldoutAccessRefused(f"HOLDOUT SEALED: {session}")
        source = TWO_CLOCK_ROOT / f"{session}.pkl"
        receipt_path = RECEIPT_ROOT / f"{session}.json"
        receipt = json.loads(receipt_path.read_text())
        observed = sha256_path(source)
        if receipt.get("output_sha256") != observed:
            raise FoundationMismatch(f"receipt mismatch before freeze: {session}")
        rows.append(
            {
                "session": session,
                "path": str(source.relative_to(ROOT)),
                "sha256": observed,
                "receipt_path": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": sha256_path(receipt_path),
            }
        )
    payload = {
        "schema_version": "autoresearch_v2.development_foundation.v1",
        "role": "development",
        "holdout_access_count": 0,
        "session_manifest": str(Path(session_manifest).resolve().relative_to(ROOT)),
        "session_manifest_sha256": sha256_path(session_manifest),
        "session_count": len(rows),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "sessions": rows,
    }
    payload["foundation_sha256"] = stable_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_foundation(path: str | Path) -> dict[str, Any]:
    """Hash every frozen input before any pickle is decoded."""

    payload = json.loads(Path(path).read_text())
    semantic = dict(payload)
    expected = semantic.pop("foundation_sha256", None)
    if expected != stable_hash(semantic):
        raise FoundationMismatch("foundation manifest hash mismatch")
    if payload.get("role") != "development" or payload.get("holdout_access_count") != 0:
        raise HoldoutAccessRefused("foundation is not development-only")
    session_manifest = ROOT / payload["session_manifest"]
    if sha256_path(session_manifest) != payload["session_manifest_sha256"]:
        raise FoundationMismatch("session assignment bytes drifted")
    forbidden = protected_sessions(session_manifest)
    if payload.get("session_count") != len(payload.get("sessions", ())):
        raise FoundationMismatch("foundation session count drifted")
    for row in payload["sessions"]:
        session = str(row["session"])
        if session in forbidden:
            raise HoldoutAccessRefused(f"HOLDOUT SEALED: {session}")
        source = ROOT / row["path"]
        receipt = ROOT / row["receipt_path"]
        if sha256_path(receipt) != row["receipt_sha256"]:
            raise FoundationMismatch(f"receipt bytes drifted: {session}")
        if sha256_path(source) != row["sha256"]:
            raise FoundationMismatch(f"two-clock bytes drifted: {session}")
    return payload


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _flatten_session(session: str, path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    processed = pd.read_pickle(path)
    for minute_index, entry in enumerate(processed):
        if not entry.get("context_ready"):
            continue
        if entry.get("processed_row_schema_version") != TWO_CLOCK_PROCESSED_ROW_SCHEMA:
            raise FoundationMismatch(f"missing two-clock row schema: {session}")
        if tuple(entry["label_names"]) != POLICIES:
            raise FoundationMismatch(f"policy axis drifted: {session}")
        option_names = {name: index for index, name in enumerate(entry["feature_names"])}
        market_names = {
            name: index for index, name in enumerate(entry["market_feature_names"])
        }
        required_options = {
            "bid",
            "ask",
            "mid",
            "spread",
            "bid_size",
            "ask_size",
            "iv",
            "delta",
            "gamma",
            "theta",
        }
        required_market = {
            "spx_close",
            "spx_vwap",
            "omar",
            "session_range",
            "momentum_5m",
            "momentum_15m",
        }
        if not required_options.issubset(option_names) or not required_market.issubset(market_names):
            raise FoundationMismatch(f"feature axis drifted: {session}")
        ladder = np.asarray(entry["option_ladder"], dtype=float)
        mask = np.asarray(entry["candidate_mask"], dtype=bool)
        market = np.asarray(entry["market_window"], dtype=float)[-1]
        offsets = np.asarray(entry["strike_offsets"], dtype=int)
        contract_ids = np.asarray(entry["contract_ids"], dtype=object)
        labels_net = np.asarray(entry["labels_net_pnl"], dtype=float)
        labels_mid = np.asarray(entry["labels_mid_pnl"], dtype=float)
        exit_ns = np.asarray(entry["label_realized_exit_time_ns"], dtype=np.int64)
        source_exit_ns = np.asarray(entry["label_source_exit_quote_time_ns"], dtype=np.int64)
        quote_age = np.asarray(entry["label_exit_quote_age_ms"], dtype=float)
        reasons = np.asarray(entry["label_exit_reason_code"], dtype=np.uint8)
        exit_bid = np.asarray(entry["label_executable_exit_bid"], dtype=float)
        deadline_ns = np.asarray(entry["label_policy_deadline_ns"], dtype=np.int64)
        invalid = np.asarray(entry["label_invalid_reason_code"], dtype=np.uint8)
        spx = float(market[market_names["spx_close"]])
        vwap = float(market[market_names["spx_vwap"]])
        omar = float(market[market_names["omar"]])
        session_range = float(market[market_names["session_range"]])
        momentum5 = float(market[market_names["momentum_5m"]])
        momentum15 = float(market[market_names["momentum_15m"]])
        if not np.isfinite([spx, vwap, omar, session_range, momentum5, momentum15]).all() or spx <= 0:
            continue
        gap = spx - vwap
        atm = int(np.argmin(np.abs(offsets)))
        atm_call = ladder[atm, 0, option_names["mid"]]
        atm_put = ladder[atm, 1, option_names["mid"]]
        straddle = (
            float((atm_call + atm_put) / spx * 1e4)
            if np.isfinite([atm_call, atm_put]).all()
            else float("nan")
        )
        put_call = (
            float(atm_put / atm_call)
            if np.isfinite([atm_call, atm_put]).all() and atm_call > 0
            else float("nan")
        )
        plus = np.flatnonzero(offsets == 5)
        minus = np.flatnonzero(offsets == -5)
        smile = float("nan")
        if len(plus) and len(minus):
            high = ladder[int(plus[0]), 0, option_names["mid"]]
            low = ladder[int(minus[0]), 0, option_names["mid"]]
            if np.isfinite([high, low]).all():
                smile = float((high - low) / spx * 1e4)
        decision_ns = int(pd.Timestamp(entry["decision_time"]).value)
        source_quote_ns = int(pd.Timestamp(entry["source_quote_time"]).value)
        source_context_ns = int(pd.Timestamp(entry["source_context_time"]).value)
        for slot in range(ladder.shape[0]):
            for right_index, right in enumerate(("C", "P")):
                if not mask[slot, right_index]:
                    continue
                bid = float(ladder[slot, right_index, option_names["bid"]])
                ask = float(ladder[slot, right_index, option_names["ask"]])
                mid = float(ladder[slot, right_index, option_names["mid"]])
                if not np.isfinite([bid, ask, mid]).all() or ask <= 0 or not 3.0 <= mid <= 8.0:
                    continue
                direction = 1.0 if right == "C" else -1.0
                bid_size = float(ladder[slot, right_index, option_names["bid_size"]])
                ask_size = float(ladder[slot, right_index, option_names["ask_size"]])
                depth = bid_size + ask_size
                row: dict[str, Any] = {
                    "candidate_uid": f"{session}|{decision_ns}|{contract_ids[slot, right_index]}",
                    "session": session,
                    "minute_index": int(minute_index),
                    "decision_time_ns": decision_ns,
                    "source_quote_time_ns": source_quote_ns,
                    "source_context_time_ns": source_context_ns,
                    "contract_id": str(contract_ids[slot, right_index]),
                    "right": right,
                    "right_index": right_index,
                    "canonical_strike_slot": int(offsets[slot]),
                    "offset": int(offsets[slot]),
                    "abs_moneyness": abs(int(offsets[slot])),
                    "entry_bid": bid,
                    "entry_ask": ask,
                    "entry_mid": mid,
                    "premium_dollars": ask * 100.0,
                    "nearest_atm_rank": abs(int(offsets[slot])),
                    "spx_vwap_gap_points": gap,
                    "spx_vwap_gap_bps": gap / spx * 1e4,
                    "spx_vwap_gap_over_session_range": gap / session_range if session_range else 0.0,
                    "session_range_bps": session_range / spx * 1e4,
                    "momentum_5m_bps": momentum5 / spx * 1e4,
                    "momentum_15m_bps": momentum15 / spx * 1e4,
                    "momentum_5m_over_session_range": momentum5 / session_range if session_range else 0.0,
                    "momentum_15m_over_session_range": momentum15 / session_range if session_range else 0.0,
                    "omar_clipped": float(np.clip(omar, -3.0, 3.0)),
                    "vwap_side_align": float(gap * direction > 0),
                    "omar_side_align": float(omar * direction > 0),
                    "momentum15_side_align": float(momentum15 * direction > 0),
                    "straddle_mid_spot_bps": straddle,
                    "put_call_mid_ratio": put_call,
                    "side_smile_slope": smile,
                    "bs_delta": float(ladder[slot, right_index, option_names["delta"]]),
                    "bs_gamma": float(ladder[slot, right_index, option_names["gamma"]]),
                    "is_call": float(right == "C"),
                    "size_imbalance": (bid_size - ask_size) / depth if _finite(depth) and depth > 0 else 0.0,
                    "depth_total": depth if _finite(depth) else 0.0,
                    "opt_spread": float(ladder[slot, right_index, option_names["spread"]]),
                    "iv": float(ladder[slot, right_index, option_names["iv"]]),
                    "theta": float(ladder[slot, right_index, option_names["theta"]]),
                }
                for policy_index, policy in enumerate(POLICIES):
                    suffix = f"p{policy_index}"
                    row[f"pnl_{suffix}"] = float(labels_net[slot, right_index, policy_index])
                    row[f"mid_pnl_{suffix}"] = float(labels_mid[slot, right_index, policy_index])
                    row[f"exit_ns_{suffix}"] = int(exit_ns[slot, right_index, policy_index])
                    row[f"source_exit_ns_{suffix}"] = int(source_exit_ns[slot, right_index, policy_index])
                    row[f"quote_age_ms_{suffix}"] = float(quote_age[slot, right_index, policy_index])
                    row[f"reason_{suffix}"] = int(reasons[slot, right_index, policy_index])
                    row[f"exit_bid_{suffix}"] = float(exit_bid[slot, right_index, policy_index])
                    row[f"deadline_ns_{suffix}"] = int(deadline_ns[slot, right_index, policy_index])
                    row[f"invalid_{suffix}"] = int(invalid[slot, right_index, policy_index])
                rows.append(row)
    return pd.DataFrame(rows)


def load_development_frame(foundation_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    foundation = verify_foundation(foundation_path)
    frames = [
        _flatten_session(str(row["session"]), ROOT / row["path"])
        for row in foundation["sessions"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["session"]) != {str(row["session"]) for row in foundation["sessions"]}:
        raise FoundationMismatch("decoded sessions differ from frozen foundation")
    return frame, foundation


def rolling_folds(
    sessions: Iterable[str], *, n_folds: int = 5, minimum_train_sessions: int = 44
) -> tuple[dict[str, Any], ...]:
    ordered = tuple(sorted(set(str(item) for item in sessions)))
    if len(ordered) < minimum_train_sessions + n_folds * 10:
        raise ValueError("not enough sessions for five expanding folds")
    test_chunks = np.array_split(np.asarray(ordered[minimum_train_sessions:]), n_folds)
    folds = []
    start = minimum_train_sessions
    for index, chunk in enumerate(test_chunks, start=1):
        test = tuple(str(item) for item in chunk.tolist())
        train = ordered[:start]
        folds.append({"fold": index, "model_fit": train, "outer_test": test})
        start += len(test)
    return tuple(folds)


def target_column(policy: str) -> str:
    return f"pnl_p{POLICY_INDEX[policy]}"


def label_columns(policy: str) -> dict[str, str]:
    suffix = f"p{POLICY_INDEX[policy]}"
    return {
        "pnl": f"pnl_{suffix}",
        "mid_pnl": f"mid_pnl_{suffix}",
        "exit_ns": f"exit_ns_{suffix}",
        "source_exit_ns": f"source_exit_ns_{suffix}",
        "quote_age_ms": f"quote_age_ms_{suffix}",
        "reason": f"reason_{suffix}",
        "exit_bid": f"exit_bid_{suffix}",
        "deadline_ns": f"deadline_ns_{suffix}",
        "invalid": f"invalid_{suffix}",
    }
