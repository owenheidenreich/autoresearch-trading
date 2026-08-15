"""Frozen Protocol 101 live-entry inference helpers.

This module wires the upstream Protocol 051/A+ surface `edge` scorer into the
Protocol 101 event-policy feature schema. It is intentionally broker-agnostic:
callers provide a causal `SurfaceDecision` built from either historical replay
or a no-order live market-data snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from v4.dataset.spxw_0dte_neural import OPTION_FEATURE_NAMES
from v4.model.hypothesis_protocol import SurfaceDecision, time_bucket
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol097_sequential_event_policy import EventSetPolicy, MAX_CANDIDATES
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS


_NY = ZoneInfo("America/New_York")
_OPTION_INDEX = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
_DEFAULT_ALLOWED_BUCKETS = ("post_open_morning", "late_afternoon")


@dataclass(frozen=True)
class Protocol101EntryArtifact:
    manifest_path: Path
    model_path: Path
    scaler_path: Path
    model: EventSetPolicy
    scaler: FeatureScaler
    threshold: float
    feature_columns: tuple[str, ...]
    fold: str
    seed: int


@dataclass
class Protocol101HistoryState:
    """Causal event-history state matching Protocol 101 short-history features."""

    summaries: list[dict[str, float]] = field(default_factory=list)
    previous_time: pd.Timestamp | None = None
    events_seen: int = 0

    def features_for(self, decision_time: pd.Timestamp) -> dict[str, float]:
        previous = self.summaries[-1] if self.summaries else _empty_summary()
        rolling = self.summaries[-3:] if self.summaries else [_empty_summary()]
        minutes_since_prev = 999.0
        if self.previous_time is not None:
            minutes_since_prev = float((decision_time - self.previous_time).total_seconds() / 60.0)
        return {
            "hist_events_seen": float(min(self.events_seen, 50)),
            "hist_minutes_since_prev_event": float(minutes_since_prev),
            "hist_prev_candidate_count": previous["candidate_count"],
            "hist_prev_max_edge": previous["max_edge"],
            "hist_prev_mean_edge": previous["mean_edge"],
            "hist_prev_max_gamma": previous["max_gamma"],
            "hist_prev_mean_theta_burden": previous["mean_theta_burden"],
            "hist_prev_min_spread_over_mid": previous["min_spread_over_mid"],
            "hist_prev_call_count": previous["call_count"],
            "hist_prev_put_count": previous["put_count"],
            "hist_prev_call_minus_put_edge": previous["call_minus_put_edge"],
            "hist_roll3_candidate_count_mean": _mean(rolling, "candidate_count"),
            "hist_roll3_max_edge": max(item["max_edge"] for item in rolling),
            "hist_roll3_mean_edge": _mean(rolling, "mean_edge"),
            "hist_roll3_max_gamma": max(item["max_gamma"] for item in rolling),
            "hist_roll3_mean_theta_burden": _mean(rolling, "mean_theta_burden"),
            "hist_roll3_min_spread_over_mid": min(item["min_spread_over_mid"] for item in rolling),
            "hist_roll3_call_minus_put_edge": _mean(rolling, "call_minus_put_edge"),
        }

    def update(self, candidates: pd.DataFrame, decision_time: pd.Timestamp) -> None:
        if candidates.empty:
            return
        self.summaries.append(_candidate_summary(candidates))
        self.previous_time = decision_time
        self.events_seen += 1


def load_protocol101_entry_artifact(manifest_path: Path, summary_path: Path) -> Protocol101EntryArtifact:
    """Load the frozen Protocol 101 event-policy model, scaler, and threshold."""

    manifest = json.loads(manifest_path.read_text())
    feature_columns = tuple(manifest["feature_columns"])
    model_path = manifest_path.parent / "model.pt"
    scaler_path = manifest_path.parent / "scaler.json"
    hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 96))
    model = EventSetPolicy(input_dim=len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler_payload = json.loads(scaler_path.read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    fold = str(manifest["fold"])
    seed = int(manifest["seed"])
    threshold = _threshold_from_summary(summary_path, fold=fold, seed=seed)
    return Protocol101EntryArtifact(
        manifest_path=manifest_path,
        model_path=model_path,
        scaler_path=scaler_path,
        model=model,
        scaler=scaler,
        threshold=float(threshold),
        feature_columns=feature_columns,
        fold=fold,
        seed=seed,
    )


def protocol101_candidate_frame_from_surface(
    decision: SurfaceDecision,
    surface_scores: np.ndarray,
    history: Protocol101HistoryState,
    *,
    min_edge: float = 25.0,
    max_candidates: int = MAX_CANDIDATES,
    allowed_buckets: Sequence[str] = _DEFAULT_ALLOWED_BUCKETS,
) -> pd.DataFrame:
    """Build Protocol 101 candidate features from one surface-scored decision."""

    decision_time = pd.Timestamp(decision.decision_time)
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    else:
        decision_time = decision_time.tz_convert("UTC")
    if time_bucket(decision.decision_time) not in set(allowed_buckets):
        return pd.DataFrame(columns=list(FEATURE_COLUMNS) + ["contract_id", "token_idx", "score"])

    scores = np.asarray(surface_scores, dtype=float).copy()
    if len(scores) != len(decision.token_mask) + 1:
        raise ValueError("surface_scores must contain flat score plus one score per token")
    mask = np.concatenate([[True], np.asarray(decision.token_mask, dtype=bool)])
    scores[~mask] = -np.inf
    flat_score = float(scores[0])
    history_features = history.features_for(decision_time)
    rows: list[dict[str, Any]] = []
    for token_idx, action_score in enumerate(scores[1:]):
        if not np.isfinite(action_score):
            continue
        edge = float(action_score - flat_score)
        if edge < float(min_edge):
            continue
        features = _entry_feature_row(decision, token_idx, edge=edge, decision_time=decision_time)
        features.update(history_features)
        rows.append(
            {
                **{column: float(features.get(column, 0.0)) for column in FEATURE_COLUMNS},
                "contract_id": str(decision.contract_ids[token_idx]),
                "token_idx": int(token_idx),
                "surface_action_score": float(action_score),
                "surface_flat_score": flat_score,
                "score": edge,
                "right": str(decision.rights[token_idx]),
                "offset_points": float(decision.offsets[token_idx]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS) + ["contract_id", "token_idx", "score"])
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["score", "contract_id"], ascending=[False, True]).head(int(max_candidates)).reset_index(drop=True)
    return frame


def protocol101_candidate_gate_diagnostics(
    decision: SurfaceDecision,
    surface_scores: np.ndarray,
    *,
    min_edge: float = 25.0,
    allowed_buckets: Sequence[str] = _DEFAULT_ALLOWED_BUCKETS,
    max_rows: int = 5,
) -> dict[str, Any]:
    """Explain why the live entry gate did or did not expose candidates.

    This mirrors the pre-model filters in `protocol101_candidate_frame_from_surface`
    for logging only. It must not change entry, risk, or order behavior.
    """

    bucket = time_bucket(decision.decision_time)
    allowed = bucket in set(allowed_buckets)
    token_mask = np.asarray(decision.token_mask, dtype=bool)
    payload: dict[str, Any] = {
        "time_bucket": bucket,
        "allowed_time_bucket": bool(allowed),
        "allowed_buckets": list(allowed_buckets),
        "min_edge": float(min_edge),
        "token_count": int(len(token_mask)),
        "eligible_token_count": int(token_mask.sum()),
    }
    scores = np.asarray(surface_scores, dtype=float).copy()
    if len(scores) != len(token_mask) + 1:
        payload.update(
            {
                "filter_reason": "surface_score_shape_mismatch",
                "surface_score_count": int(len(scores)),
                "expected_surface_score_count": int(len(token_mask) + 1),
                "valid_score_count": 0,
                "above_min_edge_count": 0,
                "top_surface_tokens": [],
                "top_rejected_contracts": [],
            }
        )
        return payload

    flat_score = float(scores[0])
    rows: list[dict[str, Any]] = []
    for token_idx, action_score in enumerate(scores[1:]):
        if not token_mask[token_idx] or not np.isfinite(action_score):
            continue
        edge = float(action_score - flat_score)
        rows.append(
            {
                "contract_id": str(decision.contract_ids[token_idx]),
                "right": str(decision.rights[token_idx]),
                "offset_points": _finite_or_none(float(decision.offsets[token_idx])),
                "edge": _finite_or_none(edge),
                "surface_action_score": _finite_or_none(float(action_score)),
                "surface_flat_score": _finite_or_none(flat_score),
                "bid": _option_or_none(decision, token_idx, "bid"),
                "ask": _option_or_none(decision, token_idx, "ask"),
                "spread_frac": _option_or_none(decision, token_idx, "spread_frac"),
                "delta": _option_or_none(decision, token_idx, "delta"),
                "gamma": _option_or_none(decision, token_idx, "gamma"),
                "theta": _option_or_none(decision, token_idx, "theta"),
                "iv": _option_or_none(decision, token_idx, "iv"),
            }
        )
    rows.sort(key=lambda row: (float(row["edge"]) if row["edge"] is not None else -math.inf), reverse=True)
    above = [row for row in rows if row["edge"] is not None and float(row["edge"]) >= float(min_edge)]
    call_edges = [float(row["edge"]) for row in rows if row["edge"] is not None and row.get("right") == "C"]
    put_edges = [float(row["edge"]) for row in rows if row["edge"] is not None and row.get("right") == "P"]
    if not allowed:
        filter_reason = "outside_time_bucket"
        top_rejected = rows[: int(max_rows)]
    elif not rows:
        filter_reason = "no_valid_surface_scores"
        top_rejected = []
    elif not above:
        filter_reason = "below_min_edge"
        top_rejected = rows[: int(max_rows)]
    else:
        filter_reason = "candidates_available"
        top_rejected = [row for row in rows if row not in above][: int(max_rows)]
    payload.update(
        {
            "filter_reason": filter_reason,
            "flat_score": _finite_or_none(flat_score),
            "valid_score_count": int(len(rows)),
            "above_min_edge_count": int(len(above)),
            "max_edge": _finite_or_none(max((float(row["edge"]) for row in rows if row["edge"] is not None), default=-math.inf)),
            "best_call_edge": _finite_or_none(max(call_edges, default=-math.inf)),
            "best_put_edge": _finite_or_none(max(put_edges, default=-math.inf)),
            "top_surface_tokens": rows[: int(max_rows)],
            "top_rejected_contracts": top_rejected,
        }
    )
    return payload


def predict_protocol101_entry(
    artifact: Protocol101EntryArtifact,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Run the frozen Protocol 101 event policy on one decision's candidates."""

    if candidates.empty:
        return {
            "action": "no_entry",
            "selected_action": "wait",
            "margin": None,
            "selected_margin": None,
            "threshold": artifact.threshold,
            "selected": None,
            "selected_contract": {},
            "reason": "no_candidates",
            "no_entry_reason": "no_candidates",
            "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
            "raw_logits": [],
            "wait_logit": None,
            "candidate_logits": [],
        }
    missing = [column for column in artifact.feature_columns if column not in candidates.columns]
    if missing:
        raise ValueError(f"Protocol 101 candidates missing feature columns: {missing}")
    raw = candidates[list(artifact.feature_columns)].to_numpy(dtype=np.float32)
    n = min(len(raw), MAX_CANDIDATES)
    x = np.zeros((1, MAX_CANDIDATES, len(artifact.feature_columns)), dtype=np.float32)
    mask = np.zeros((1, MAX_CANDIDATES), dtype=bool)
    x[0, :n, :] = artifact.scaler.transform(raw[:n])
    mask[0, :n] = True
    with torch.no_grad():
        logits = artifact.model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    wait = float(logits[0])
    candidate_logits = logits[1 : n + 1]
    local_idx = int(np.argmax(candidate_logits))
    margin = float(candidate_logits[local_idx] - wait)
    raw_logits = [float(value) for value in logits[: n + 1]]
    candidate_logit_list = [float(value) for value in candidate_logits]
    action_mask = {"wait": True, "candidate_count": int(n), "enter": bool(n > 0)}
    if margin < artifact.threshold:
        return {
            "action": "no_entry",
            "selected_action": "wait",
            "margin": margin,
            "selected_margin": margin,
            "threshold": artifact.threshold,
            "selected": None,
            "selected_contract": {},
            "reason": "below_protocol101_threshold",
            "no_entry_reason": "below_protocol101_threshold",
            "action_mask": action_mask,
            "raw_logits": raw_logits,
            "wait_logit": wait,
            "candidate_logits": candidate_logit_list,
        }
    selected = candidates.iloc[local_idx].to_dict()
    return {
        "action": "enter",
        "selected_action": "enter",
        "margin": margin,
        "selected_margin": margin,
        "threshold": artifact.threshold,
        "selected": selected,
        "selected_contract": selected,
        "reason": "protocol101_margin_above_threshold",
        "no_entry_reason": None,
        "action_mask": action_mask,
        "raw_logits": raw_logits,
        "wait_logit": wait,
        "candidate_logits": candidate_logit_list,
    }


def _entry_feature_row(decision: SurfaceDecision, token_idx: int, *, edge: float, decision_time: pd.Timestamp) -> dict[str, float]:
    base = np.asarray(decision.token_features[token_idx], dtype=float)
    right = str(decision.rights[token_idx])
    offset = float(decision.offsets[token_idx])
    local = decision_time.tz_convert(_NY)
    minutes = local.hour * 60.0 + local.minute
    open_minutes = 9 * 60 + 30
    no_new_entries_after = 15 * 60 + 30
    session_minutes = float(no_new_entries_after - open_minutes)
    elapsed = min(max(minutes - open_minutes, 0.0), session_minutes)
    progress = elapsed / session_minutes if session_minutes > 0 else 0.0
    radians = 2.0 * math.pi * progress

    bid = _option(base, "bid")
    ask = _option(base, "ask")
    mid = _option(base, "mid")
    spread = _option(base, "spread")
    spread_frac = _option(base, "spread_frac")
    bid_size = _option(base, "bid_size")
    ask_size = _option(base, "ask_size")
    underlying = float(decision.market_last[0]) if len(decision.market_last) else 0.0
    iv = _option(base, "iv")
    delta = _option(base, "delta")
    gamma = _option(base, "gamma")
    theta = _option(base, "theta")
    abs_theta = abs(theta)
    safe_mid = max(abs(mid), 0.01)
    safe_ask = max(abs(ask), 0.01)
    size_sum = bid_size + ask_size
    is_call = 1.0 if right == "C" else 0.0
    is_put = 1.0 if right == "P" else 0.0
    return {
        "entry_minutes_since_open": elapsed,
        "entry_minutes_to_forced_flat": max(no_new_entries_after - minutes, 0.0),
        "entry_progress": progress,
        "entry_progress_sin": math.sin(radians),
        "entry_progress_cos": math.cos(radians),
        "entry_is_first_30m": float(minutes < 10 * 60),
        "entry_is_post_open_morning": float(10 * 60 <= minutes < 11 * 60 + 30),
        "entry_is_midday": float(11 * 60 + 30 <= minutes < 13 * 60 + 30),
        "entry_is_late_afternoon": float(minutes >= 13 * 60 + 30),
        "right_is_call": is_call,
        "right_is_put": is_put,
        "offset": offset,
        "abs_offset": abs(offset),
        "edge": float(edge),
        "entry_bid": bid,
        "entry_ask": ask,
        "entry_mid": mid,
        "entry_spread": spread,
        "entry_spread_frac": spread_frac,
        "entry_bid_size": bid_size,
        "entry_ask_size": ask_size,
        "entry_underlying_price": underlying,
        "entry_iv": iv,
        "entry_delta": delta,
        "entry_abs_delta": abs(delta),
        "entry_gamma": gamma,
        "entry_theta": theta,
        "entry_abs_theta": abs_theta,
        "entry_gamma_theta_ratio": gamma / max(abs_theta, 1e-6),
        "entry_theta_over_mid": abs_theta / safe_mid,
        "entry_theta_burden": abs_theta * max(no_new_entries_after - minutes, 0.0) / safe_mid,
        "entry_gamma_per_premium": gamma / safe_ask,
        "entry_premium_over_underlying": ask / max(abs(underlying), 1.0),
        "entry_spread_over_mid": spread / safe_mid,
        "entry_size_imbalance": (bid_size - ask_size) / size_sum if size_sum else 0.0,
        "entry_call_delta_signed": delta * is_call,
        "entry_put_delta_signed": delta * is_put,
    }


def _candidate_summary(candidates: pd.DataFrame) -> dict[str, float]:
    calls = candidates[candidates["right"] == "C"]
    puts = candidates[candidates["right"] == "P"]
    max_call_edge = float(calls["edge"].max()) if not calls.empty else 0.0
    max_put_edge = float(puts["edge"].max()) if not puts.empty else 0.0
    return {
        "candidate_count": float(len(candidates)),
        "max_edge": float(candidates["edge"].max()),
        "mean_edge": float(candidates["edge"].mean()),
        "max_gamma": float(candidates["entry_gamma"].max()),
        "mean_theta_burden": float(candidates["entry_theta_burden"].mean()),
        "min_spread_over_mid": float(candidates["entry_spread_over_mid"].min()),
        "call_count": float(len(calls)),
        "put_count": float(len(puts)),
        "call_minus_put_edge": float(max_call_edge - max_put_edge),
    }


def _empty_summary() -> dict[str, float]:
    return {
        "candidate_count": 0.0,
        "max_edge": 0.0,
        "mean_edge": 0.0,
        "max_gamma": 0.0,
        "mean_theta_burden": 0.0,
        "min_spread_over_mid": 0.0,
        "call_count": 0.0,
        "put_count": 0.0,
        "call_minus_put_edge": 0.0,
    }


def _mean(rows: Sequence[dict[str, float]], column: str) -> float:
    return float(sum(row[column] for row in rows) / max(len(rows), 1))


def _option(base: np.ndarray, name: str) -> float:
    idx = _OPTION_INDEX[name]
    value = float(base[idx]) if idx < len(base) else 0.0
    return value if math.isfinite(value) else 0.0


def _option_or_none(decision: SurfaceDecision, token_idx: int, name: str) -> float | None:
    idx = _OPTION_INDEX.get(name)
    if idx is None:
        return None
    try:
        value = float(np.asarray(decision.token_features[token_idx], dtype=float)[idx])
    except (IndexError, TypeError, ValueError):
        return None
    return _finite_or_none(value)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _threshold_from_summary(summary_path: Path, *, fold: str, seed: int) -> float:
    summary = json.loads(summary_path.read_text())
    for row in summary.get("fold_results", []):
        if str(row.get("fold")) == fold and int(row.get("seed", -1)) == int(seed):
            return float(row["threshold"])
    raise ValueError(f"missing Protocol 101 threshold for fold={fold} seed={seed}")
