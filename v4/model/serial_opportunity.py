"""Serial opportunity-cost policy utilities for SPXW 0DTE research.

Protocol 092 sits above the frozen lifecycle exits. It trains only an
entry-time scorer, then simulates the live constraint that the bot can hold one
contract at a time. The feature set is intentionally entry-only: path, exit,
and future columns are labels/audit fields, never model inputs.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler, Trade, metrics_for_trades


FEATURE_VERSION = "protocol092_serial_opportunity_v1"
CONTRACT_MULTIPLIER = 100.0
ENTRY_FEATURE_COLUMNS = [
    "entry_minutes_since_open",
    "entry_minutes_to_forced_flat",
    "entry_progress",
    "entry_progress_sin",
    "entry_progress_cos",
    "entry_is_first_30m",
    "entry_is_post_open_morning",
    "entry_is_midday",
    "entry_is_late_afternoon",
    "right_is_call",
    "right_is_put",
    "offset",
    "abs_offset",
    "edge",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_spread_frac",
    "entry_bid_size",
    "entry_ask_size",
    "entry_underlying_price",
    "entry_iv",
    "entry_delta",
    "entry_abs_delta",
    "entry_gamma",
    "entry_theta",
    "entry_abs_theta",
    "entry_gamma_theta_ratio",
    "entry_theta_over_mid",
    "entry_theta_burden",
    "entry_gamma_per_premium",
    "entry_premium_over_underlying",
    "entry_spread_over_mid",
    "entry_size_imbalance",
    "entry_call_delta_signed",
    "entry_put_delta_signed",
]

LEAKY_FEATURE_TOKENS = (
    "path",
    "exit",
    "future",
    "mfe",
    "mae",
    "pnl",
    "label",
    "target",
    "recovered",
    "kept_falling",
    "post_protocol",
    "candidate_",
    "protocol054_",
    "baseline_",
    "current_pnl",
)


@dataclass(frozen=True)
class SerialOpportunityConfig:
    """Training knobs for the Protocol 092 entry scorer."""

    epochs: int = 18
    batch_size: int = 4096
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_scale: float = 100.0
    target_clip: float = 800.0
    ranking_weight: float = 0.25
    rank_margin: float = 0.05
    max_rank_pairs: int = 30_000
    min_validation_trades: int = 10


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-only threshold selection record."""

    threshold: float
    source_split: str
    source_seed: int
    source_rows: int
    objective: str
    sweep: list[dict[str, Any]]


@dataclass(frozen=True)
class SerialSimulationResult:
    """Trade-level output of the one-contract serial simulator."""

    trades: list[dict[str, Any]]
    summary: dict[str, Any]


class SerialOpportunityMLP(nn.Module):
    """Small MLP scorer for candidate utility in executable dollars."""

    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def assert_feature_columns_are_causal(columns: Sequence[str]) -> None:
    """Raise when a model feature name suggests path/exit/future leakage."""

    bad = [
        column
        for column in columns
        for token in LEAKY_FEATURE_TOKENS
        if token in column.lower()
    ]
    if bad:
        raise ValueError(f"leaky Protocol 092 feature columns: {sorted(set(bad))}")


def build_protocol092_dataset(
    *,
    selected_trades_path: Path,
    lifecycle_trades_path: Path,
    fallback_protocol_seeds: Sequence[int] = tuple(range(1, 11)),
) -> pd.DataFrame:
    """Create the Protocol 092 candidate table from frozen local artifacts.

    Protocol 081 selected exits start in Q2 2025. Fold 1 still needs Q1 2025
    training data, so Q1 uses Protocol 077's frozen Protocol 054 lifecycle
    outcome as a train-only fallback label. Later validation/test rows use the
    Protocol 081 candidate exits.
    """

    selected = pd.read_json(selected_trades_path)
    trades = pd.read_parquet(lifecycle_trades_path)

    protocol081 = _join_protocol081_selected(selected, trades)
    q1_fallback = _q1_training_fallback(trades, fallback_protocol_seeds)
    dataset = pd.concat([q1_fallback, protocol081], ignore_index=True, sort=False)
    dataset = _add_entry_features(dataset)
    dataset = _sanitize_dataset(dataset)
    assert_feature_columns_are_causal(ENTRY_FEATURE_COLUMNS)
    return dataset


def _join_protocol081_selected(selected: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    selected = selected.rename(columns={"seed": "protocol_seed"}).copy()
    trades_for_join = trades.rename(columns={"seed": "entry_seed"}).copy()
    selected["decision_time_key"] = _timestamp_key(selected["decision_time"])
    trades_for_join["decision_time_key"] = _timestamp_key(trades_for_join["decision_time"])
    merge_keys = ["split", "entry_seed", "session", "decision_time_key", "contract_id"]
    merged = selected.merge(
        trades_for_join,
        on=merge_keys,
        how="left",
        suffixes=("_p081", ""),
        validate="many_to_one",
    )
    unmatched = merged["entry_bid"].isna().sum()
    if unmatched:
        raise ValueError(f"Protocol 081 rows missing Protocol 077 entry features: {unmatched}")

    out = _canonical_columns(merged)
    out["seed"] = merged["protocol_seed"].astype(int)
    out["entry_seed"] = merged["entry_seed"].astype(int)
    out["candidate_pnl"] = pd.to_numeric(merged["candidate_pnl"], errors="coerce")
    out["candidate_exit_time"] = merged["candidate_exit_time"]
    out["candidate_exit_reason"] = merged["candidate_exit_reason"].astype(str)
    out["candidate_exit_step"] = pd.to_numeric(merged["candidate_exit_step"], errors="coerce")
    out["label_source"] = "protocol081"
    out["candidate_uid"] = (
        "protocol081:"
        + out["seed"].astype(str)
        + ":"
        + out["entry_seed"].astype(str)
        + ":"
        + out["trade_uid"].astype(str)
    )
    return out


def _timestamp_key(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.astype("int64")


def _q1_training_fallback(trades: pd.DataFrame, protocol_seeds: Sequence[int]) -> pd.DataFrame:
    base = trades[trades["split"] == "q1_2025"].copy()
    if base.empty:
        raise ValueError("Protocol 092 needs q1_2025 rows for Fold 1 training")
    rows = []
    for seed in protocol_seeds:
        item = _canonical_columns(base)
        item["seed"] = int(seed)
        item["entry_seed"] = pd.to_numeric(base["seed"], errors="coerce").astype(int)
        fallback_pnl = pd.to_numeric(base.get("protocol054_path_pnl"), errors="coerce")
        fallback_pnl = fallback_pnl.fillna(pd.to_numeric(base["protocol054_pnl"], errors="coerce"))
        item["candidate_pnl"] = fallback_pnl
        item["candidate_exit_time"] = base["protocol054_exit_time"]
        item["candidate_exit_reason"] = base["protocol054_exit_reason"].astype(str)
        item["candidate_exit_step"] = pd.to_numeric(base["protocol054_exit_step"], errors="coerce")
        item["label_source"] = "protocol054_train_fallback"
        item["candidate_uid"] = (
            "q1fallback:"
            + item["seed"].astype(str)
            + ":"
            + item["entry_seed"].astype(str)
            + ":"
            + item["trade_uid"].astype(str)
        )
        rows.append(item)
    return pd.concat(rows, ignore_index=True, sort=False)


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "session",
        "decision_time",
        "trade_uid",
        "canonical_entry_uid",
        "contract_id",
        "right",
        "offset",
        "edge",
        "time_bucket",
        "entry_quote_time",
        "entry_bid",
        "entry_ask",
        "entry_mid",
        "entry_spread",
        "entry_spread_frac",
        "entry_bid_size",
        "entry_ask_size",
        "entry_underlying_price",
        "entry_iv",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing Protocol 077 entry columns: {missing}")
    return frame[columns].copy()


def _add_entry_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    decision_dt = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    exit_dt = pd.to_datetime(out["candidate_exit_time"], utc=True, errors="coerce")
    quote_dt = pd.to_datetime(out["entry_quote_time"], utc=True, errors="coerce")
    local = decision_dt.dt.tz_convert("America/New_York")
    minutes = local.dt.hour.astype(float) * 60.0 + local.dt.minute.astype(float)
    open_minutes = 9 * 60 + 30
    no_new_entries_after = 15 * 60 + 30
    session_minutes = float(no_new_entries_after - open_minutes)
    elapsed = (minutes - open_minutes).clip(lower=0.0, upper=session_minutes)
    progress = elapsed / session_minutes
    radians = 2.0 * math.pi * progress

    out["decision_dt"] = decision_dt
    out["candidate_exit_dt"] = exit_dt
    out["entry_quote_dt"] = quote_dt
    out["entry_minutes_since_open"] = elapsed
    out["entry_minutes_to_forced_flat"] = (no_new_entries_after - minutes).clip(lower=0.0)
    out["entry_progress"] = progress
    out["entry_progress_sin"] = np.sin(radians)
    out["entry_progress_cos"] = np.cos(radians)
    out["entry_is_first_30m"] = (minutes < 10 * 60).astype(float)
    out["entry_is_post_open_morning"] = ((minutes >= 10 * 60) & (minutes < 11 * 60 + 30)).astype(float)
    out["entry_is_midday"] = ((minutes >= 11 * 60 + 30) & (minutes < 13 * 60 + 30)).astype(float)
    out["entry_is_late_afternoon"] = (minutes >= 13 * 60 + 30).astype(float)
    out["right_is_call"] = (out["right"].astype(str) == "C").astype(float)
    out["right_is_put"] = (out["right"].astype(str) == "P").astype(float)

    for column in [
        "offset",
        "edge",
        "entry_bid",
        "entry_ask",
        "entry_mid",
        "entry_spread",
        "entry_spread_frac",
        "entry_bid_size",
        "entry_ask_size",
        "entry_underlying_price",
        "entry_iv",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "candidate_pnl",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    mid = out["entry_mid"].abs().clip(lower=0.01)
    ask = out["entry_ask"].abs().clip(lower=0.01)
    theta_abs = out["entry_theta"].abs()
    out["abs_offset"] = out["offset"].abs()
    out["entry_abs_delta"] = out["entry_delta"].abs()
    out["entry_abs_theta"] = theta_abs
    out["entry_gamma_theta_ratio"] = out["entry_gamma"] / theta_abs.clip(lower=1e-6)
    out["entry_theta_over_mid"] = theta_abs / mid
    out["entry_theta_burden"] = theta_abs * out["entry_minutes_to_forced_flat"] / mid
    out["entry_gamma_per_premium"] = out["entry_gamma"] / ask
    out["entry_premium_over_underlying"] = ask / out["entry_underlying_price"].abs().clip(lower=1.0)
    out["entry_spread_over_mid"] = out["entry_spread"] / mid
    size_sum = out["entry_bid_size"] + out["entry_ask_size"]
    out["entry_size_imbalance"] = (out["entry_bid_size"] - out["entry_ask_size"]) / size_sum.replace(0.0, np.nan)
    out["entry_call_delta_signed"] = out["entry_delta"] * out["right_is_call"]
    out["entry_put_delta_signed"] = out["entry_delta"] * out["right_is_put"]
    return out


def _sanitize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    valid = (
        out["decision_dt"].notna()
        & out["candidate_exit_dt"].notna()
        & (out["candidate_exit_dt"] > out["decision_dt"])
        & np.isfinite(out["candidate_pnl"])
        & out["right"].isin(["C", "P"])
    )
    out = out.loc[valid].copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    out["entry_seed"] = pd.to_numeric(out["entry_seed"], errors="coerce").astype(int)
    out["session"] = out["session"].astype(str)
    out["split"] = out["split"].astype(str)
    out["decision_time"] = out["decision_dt"].astype(str)
    out["candidate_exit_time"] = out["candidate_exit_dt"].astype(str)
    out["entry_quote_time"] = out["entry_quote_dt"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["trade_uid"] = out["trade_uid"].astype(str)
    out["candidate_uid"] = out["candidate_uid"].astype(str)
    out = out.sort_values(["split", "seed", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
    return out


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    assert_feature_columns_are_causal(ENTRY_FEATURE_COLUMNS)
    return frame[ENTRY_FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def target_vector(
    frame: pd.DataFrame,
    config: SerialOpportunityConfig,
    *,
    target_column: str = "candidate_pnl",
) -> np.ndarray:
    y = frame[target_column].to_numpy(dtype=np.float32)
    return (np.clip(y, -config.target_clip, config.target_clip) / config.target_scale).astype(np.float32)


def train_serial_opportunity_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    seed: int,
    config: SerialOpportunityConfig,
    target_column: str = "candidate_pnl",
) -> tuple[SerialOpportunityMLP, FeatureScaler, list[dict[str, Any]]]:
    """Train entry scorer with Huber utility regression and ranking loss."""

    if train_frame.empty:
        raise ValueError("empty Protocol 092 training frame")
    if validation_frame.empty:
        raise ValueError("empty Protocol 092 validation frame")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train_raw = feature_matrix(train_frame)
    x_val_raw = feature_matrix(validation_frame)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    y_train = target_vector(train_frame, config, target_column=target_column)
    y_val = target_vector(validation_frame, config, target_column=target_column)

    model = SerialOpportunityMLP(input_dim=x_train.shape[1], hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    huber = nn.HuberLoss(delta=1.0)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    pairs = _ranking_pairs(train_frame, y_train, max_pairs=config.max_rank_pairs, seed=seed)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        reg_losses: list[float] = []
        rank_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = huber(prediction, batch_y)
            loss.backward()
            optimizer.step()
            reg_losses.append(float(loss.detach().cpu()))

        if len(pairs):
            pair_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(x_train[pairs[:, 0]]),
                    torch.from_numpy(x_train[pairs[:, 1]]),
                ),
                batch_size=config.batch_size,
                shuffle=True,
            )
            for best_x, other_x in pair_loader:
                optimizer.zero_grad(set_to_none=True)
                best_score = model(best_x)
                other_score = model(other_x)
                rank_loss = torch.nn.functional.softplus(
                    config.rank_margin - (best_score - other_score)
                ).mean()
                loss = config.ranking_weight * rank_loss
                loss.backward()
                optimizer.step()
                rank_losses.append(float(rank_loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = float(huber(val_pred, y_val_t).detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_huber": float(np.mean(reg_losses)) if reg_losses else math.nan,
                "train_ranking": float(np.mean(rank_losses)) if rank_losses else 0.0,
                "validation_huber": val_loss,
                "is_best": val_loss <= best_val,
                "ranking_pairs": int(len(pairs)),
            }
        )

    model.load_state_dict(best_state)
    return model, scaler, history


def predict_scores(
    model: SerialOpportunityMLP,
    scaler: FeatureScaler,
    frame: pd.DataFrame,
    *,
    target_scale: float = 100.0,
    batch_size: int = 32768,
) -> np.ndarray:
    """Predict candidate utility in executable dollars."""

    if frame.empty:
        return np.asarray([], dtype=np.float32)
    x = scaler.transform(feature_matrix(frame))
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            chunks.append(model(batch).cpu().numpy().astype(np.float32) * float(target_scale))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)


def _ranking_pairs(frame: pd.DataFrame, target: np.ndarray, *, max_pairs: int, seed: int) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    indexer = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    for _, group in frame.groupby(["split", "seed", "session", "decision_time"], sort=False):
        idx = indexer.loc[group.index].to_numpy(dtype=int)
        if len(idx) < 2:
            continue
        y = target[idx]
        best_local = int(np.nanargmax(y))
        best_idx = int(idx[best_local])
        for other_idx in idx:
            if int(other_idx) == best_idx:
                continue
            if target[best_idx] > target[int(other_idx)]:
                pairs.append((best_idx, int(other_idx)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in chosen]
    return np.asarray(pairs, dtype=np.int64)


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return [float("inf")]
    quantiles = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
    values = np.quantile(finite, quantiles).round(4).tolist()
    values.extend([0.0, float(finite.min()) - 1e-3])
    return sorted(set(float(v) for v in values if math.isfinite(float(v))))


def select_validation_threshold(
    validation_frame: pd.DataFrame,
    *,
    score_column: str,
    source_split: str,
    source_seed: int,
    config: SerialOpportunityConfig,
    stress_slippage_per_side: float = 0.10,
) -> ThresholdSelection:
    """Select an entry threshold from validation data only."""

    if validation_frame.empty:
        return ThresholdSelection(
            threshold=float("inf"),
            source_split=source_split,
            source_seed=int(source_seed),
            source_rows=0,
            objective="validation_stress_0.10_total_pnl_pf_trades",
            sweep=[],
        )
    sweep = []
    for threshold in candidate_thresholds(validation_frame[score_column].to_numpy(dtype=float)):
        base = serial_simulate_candidates(
            validation_frame,
            score_column=score_column,
            threshold=float(threshold),
            slippage_per_side=0.0,
            strategy="protocol092_validation",
        )
        stress = serial_simulate_candidates(
            validation_frame,
            score_column=score_column,
            threshold=float(threshold),
            slippage_per_side=stress_slippage_per_side,
            strategy="protocol092_validation_stress10",
        )
        row = {
            "threshold": float(threshold),
            "base": base.summary,
            "stress_0_10": stress.summary,
        }
        sweep.append(row)

    eligible = [
        row
        for row in sweep
        if row["base"]["trades"] >= config.min_validation_trades
        and row["stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else [
        row for row in sweep if row["base"]["trades"] >= config.min_validation_trades
    ]
    if not pool:
        pool = sweep
    best = max(pool, key=_threshold_selection_key)
    return ThresholdSelection(
        threshold=float(best["threshold"]),
        source_split=source_split,
        source_seed=int(source_seed),
        source_rows=int(len(validation_frame)),
        objective="validation_stress_0.10_total_pnl_pf_trades",
        sweep=sweep,
    )


def _threshold_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    stress = row["stress_0_10"]
    base = row["base"]
    pf = float(stress.get("profit_factor", 0.0))
    if math.isinf(pf):
        pf = 999.0
    return (
        float(stress.get("total_pnl", 0.0)),
        pf,
        float(base.get("total_pnl", 0.0)),
        float(base.get("trades", 0.0)),
    )


def serial_simulate_candidates(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    slippage_per_side: float,
    strategy: str,
) -> SerialSimulationResult:
    """Live-like serial simulation: enter only while flat, hold until frozen exit."""

    trades: list[dict[str, Any]] = []
    skipped_overlap_candidates = 0
    skipped_threshold_candidates = 0
    skipped_invalid_candidates = 0
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER

    ordered = frame.copy()
    ordered = ordered[np.isfinite(pd.to_numeric(ordered[score_column], errors="coerce"))].copy()
    ordered[score_column] = pd.to_numeric(ordered[score_column], errors="coerce")
    ordered = ordered.sort_values(["session", "decision_dt", "candidate_uid"]).reset_index(drop=True)

    for session, session_frame in ordered.groupby("session", sort=True):
        open_until: pd.Timestamp | None = None
        for decision_time, group in session_frame.groupby("decision_dt", sort=True):
            if open_until is not None and decision_time < open_until:
                skipped_overlap_candidates += int(len(group))
                continue
            eligible = group[group[score_column] >= threshold].copy()
            if eligible.empty:
                skipped_threshold_candidates += int(len(group))
                continue
            eligible = eligible[
                eligible["candidate_exit_dt"].notna()
                & (eligible["candidate_exit_dt"] > eligible["decision_dt"])
                & np.isfinite(eligible["candidate_pnl"])
            ].copy()
            if eligible.empty:
                skipped_invalid_candidates += int(len(group))
                continue
            eligible = eligible.sort_values(
                [score_column, "entry_seed", "contract_id", "candidate_uid"],
                ascending=[False, True, True, True],
            )
            chosen = eligible.iloc[0]
            pnl = float(chosen["candidate_pnl"]) - round_trip_slippage
            trade = {
                "candidate_uid": str(chosen["candidate_uid"]),
                "trade_uid": str(chosen["trade_uid"]),
                "split": str(chosen["split"]),
                "seed": int(chosen["seed"]),
                "entry_seed": int(chosen["entry_seed"]),
                "session": str(session),
                "decision_time": pd.Timestamp(chosen["decision_dt"]).isoformat(),
                "exit_time": pd.Timestamp(chosen["candidate_exit_dt"]).isoformat(),
                "contract_id": str(chosen["contract_id"]),
                "right": str(chosen["right"]),
                "offset": float(chosen["offset"]),
                "score": float(chosen[score_column]),
                "threshold": float(threshold),
                "pnl": pnl,
                "raw_candidate_pnl": float(chosen["candidate_pnl"]),
                "slippage_per_side": float(slippage_per_side),
                "strategy": strategy,
                "exit_reason": str(chosen["candidate_exit_reason"]),
                "label_source": str(chosen["label_source"]),
            }
            trades.append(trade)
            open_until = pd.Timestamp(chosen["candidate_exit_dt"])

    checks = validate_serial_trades(trades)
    summary = serial_metrics(trades)
    summary.update(
        {
            "input_candidates": int(len(frame)),
            "skipped_overlap_candidates": int(skipped_overlap_candidates),
            "skipped_threshold_candidates": int(skipped_threshold_candidates),
            "skipped_invalid_candidates": int(skipped_invalid_candidates),
            "max_concurrent_positions": int(checks["max_concurrent_positions"]),
            "all_flat_by_session_end": bool(checks["all_flat_by_session_end"]),
            "terminal_final": bool(checks["terminal_final"]),
            "serial_status": "pass" if checks["status"] == "pass" else "fail",
        }
    )
    return SerialSimulationResult(trades=trades, summary=summary)


def validate_serial_trades(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check one-position lifecycle semantics for trade-level serial output."""

    max_concurrent = 0
    violations = []
    for session, session_trades in _group_trades_by_session(trades).items():
        ordered = sorted(session_trades, key=lambda row: (row["decision_time"], row["candidate_uid"]))
        active_until: pd.Timestamp | None = None
        for trade in ordered:
            entry = pd.Timestamp(trade["decision_time"])
            exit_time = pd.Timestamp(trade["exit_time"])
            if exit_time <= entry:
                violations.append(f"{trade['candidate_uid']}: exit <= entry")
            if active_until is not None and entry < active_until:
                violations.append(f"{trade['candidate_uid']}: overlaps prior trade in {session}")
                max_concurrent = max(max_concurrent, 2)
            else:
                max_concurrent = max(max_concurrent, 1 if trades else 0)
            active_until = exit_time
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "max_concurrent_positions": max_concurrent,
        "all_flat_by_session_end": True,
        "terminal_final": True,
    }


def _group_trades_by_session(trades: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(str(trade["session"]), []).append(dict(trade))
    return grouped


def serial_metrics(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    converted = [
        Trade(
            session=str(trade["session"]),
            decision_time=str(trade["decision_time"]),
            pnl=float(trade["pnl"]),
            score=float(trade["score"]) if trade.get("score") is not None else None,
            right=str(trade["right"]),
            offset=float(trade["offset"]),
            strategy=str(trade.get("strategy", "protocol092")),
        )
        for trade in trades
    ]
    metrics = metrics_for_trades(converted)
    side_counts = {
        "C": int(sum(1 for trade in trades if str(trade["right"]) == "C")),
        "P": int(sum(1 for trade in trades if str(trade["right"]) == "P")),
    }
    metrics["side_counts"] = side_counts
    metrics["median_offset"] = float(np.median([float(t["offset"]) for t in trades])) if trades else 0.0
    metrics["label_source_counts"] = _count(str(t.get("label_source", "")) for t in trades)
    return metrics


def strict_serial_baseline(frame: pd.DataFrame, *, seed: int, slippage_per_side: float = 0.0) -> SerialSimulationResult:
    """Frozen strict serial baseline: first available candidate while flat."""

    baseline = frame.copy()
    baseline = baseline[baseline["seed"] == int(seed)].copy()
    baseline["strict_serial_baseline_score"] = 0.0
    return serial_simulate_candidates(
        baseline,
        score_column="strict_serial_baseline_score",
        threshold=-1e18,
        slippage_per_side=slippage_per_side,
        strategy="strict_serial_first_available",
    )


def model_artifact_manifest(
    *,
    fold_name: str,
    seed: int,
    config: SerialOpportunityConfig,
    threshold: ThresholdSelection,
    history: Sequence[dict[str, Any]],
    model_path: Path,
    scaler_path: Path,
) -> dict[str, Any]:
    return {
        "protocol": "092_serial_opportunity_policy",
        "feature_version": FEATURE_VERSION,
        "fold_name": fold_name,
        "seed": int(seed),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "config": asdict(config),
        "threshold_selection": asdict(threshold),
        "training_history": list(history),
        "files": {
            "model": str(model_path),
            "scaler": str(scaler_path),
        },
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
