"""Small supervised SPXW 0DTE pilot model.

This is intentionally a first falsification harness, not a promotion candidate.
It trains a candidate-level network to estimate executable ask-entry/bid-exit
PnL and evaluates a one-contract-at-a-time policy against simple baselines.
"""
from __future__ import annotations

import json
import math
import pickle
import random
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_VERSION = "candidate_v3_envpriors"
CLASSIFIER_TARGET_MODES = {
    "profit_classifier",
    "decision_profit_presence_classifier",
    "decision_top_profit_classifier",
    "protocol101_teacher_classifier",
    "protocol101_teacher_profitable_classifier",
}
LISTWISE_TARGET_MODES = {"decision_top_profit_listwise"}
TEACHER_TARGET_MODES = {
    "protocol101_teacher_classifier",
    "protocol101_teacher_profitable_classifier",
    "protocol101_teacher_edge_regression",
}
FEATURE_TRANSFORM_NONE = "none"
FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE = (
    "mask_vendor_sensitive_option_microstructure"
)
FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE = (
    "bucket_vendor_sensitive_option_microstructure"
)
FEATURE_TRANSFORM_CHOICES = {
    FEATURE_TRANSFORM_NONE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
}
FEATURE_NOISE_AUGMENTATION_NONE = "none"
FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1 = (
    "vendor_microstructure_jitter_v1"
)
FEATURE_NOISE_AUGMENTATION_CHOICES = {
    FEATURE_NOISE_AUGMENTATION_NONE,
    FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
}
SELECTION_MODE_TOP_SCORE = "top_score"
SELECTION_MODE_STABLE_ABS_OFFSET_10 = "stable_abs_offset_10"
SELECTION_MODE_STABLE_ABS_OFFSET_15 = "stable_abs_offset_15"
SELECTION_MODE_STABLE_ABS_OFFSET_20 = "stable_abs_offset_20"
SELECTION_MODE_CHOICES = {
    SELECTION_MODE_TOP_SCORE,
    SELECTION_MODE_STABLE_ABS_OFFSET_10,
    SELECTION_MODE_STABLE_ABS_OFFSET_15,
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
}
VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS: tuple[tuple[str, dict[str, float]], ...] = (
    ("baseline", {}),
    ("iv_up_002", {"iv_delta": 0.002}),
    ("iv_down_002", {"iv_delta": -0.002}),
    ("spread_widen_005", {"spread_delta": 0.05, "spread_frac_delta": 0.005}),
    ("spread_tighten_005", {"spread_delta": -0.05, "spread_frac_delta": -0.005}),
    ("bid_size_half_ask_size_double", {"bid_size_scale": 0.5, "ask_size_scale": 2.0}),
    ("bid_size_double_ask_size_half", {"bid_size_scale": 2.0, "ask_size_scale": 0.5}),
)
_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE_INDICES = np.asarray(
    [
        3,  # spread
        4,  # spread_frac
        5,  # bid_size
        6,  # ask_size
        7,  # option_ohlcv_volume
        8,  # stat_open_interest
        9,  # iv
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the first supervised pilot."""

    policy_index: int = 1
    policy_name: str = "ask_to_bid_stop50_target100_hold25m"
    cooldown_minutes: int = 25
    max_train_examples: int = 350_000
    epochs: int = 8
    batch_size: int = 8192
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_mode: str = "regression"
    target_scale: float = 100.0
    target_clip: float = 600.0
    positive_label_threshold: float = 20.0
    relative_target_weight: float = 0.5
    entry_filter: str = "none"
    min_score_margin: float = 0.0
    max_score_ceiling: float = 0.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    sample_weight_mode: str = "none"
    feature_transform: str = FEATURE_TRANSFORM_NONE
    feature_noise_augmentation: str = FEATURE_NOISE_AUGMENTATION_NONE
    selection_mode: str = SELECTION_MODE_TOP_SCORE
    min_validation_trades: int = 20
    random_seeds: int = 20
    seed: int = 42


@dataclass
class FeatureScaler:
    """Median-impute, standardize, and sanitize numeric feature vectors."""

    fill: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        x = np.asarray(features, dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.nan)
        fill = np.nanmedian(x, axis=0).astype(np.float32)
        fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(x), x, fill)
        mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(fill=fill, mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.fill)
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "fill": self.fill.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass
class DecisionCandidates:
    """Candidate rows for one decision minute."""

    session: str
    decision_time: datetime
    features: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    rights: np.ndarray
    market_last: np.ndarray
    contract_ids: np.ndarray | None = None
    teacher_labels: np.ndarray | None = None
    entry_asks: np.ndarray | None = None


@dataclass(frozen=True)
class Trade:
    """One simulated long-option trade outcome."""

    session: str
    decision_time: str
    pnl: float
    score: float | None
    right: str
    offset: float
    strategy: str


class CandidateMLP(nn.Module):
    """Tiny MLP for executable PnL regression."""

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


def split_name(session: str) -> str:
    """Chronological pilot split: Jan train, Feb validation, Mar test."""
    month = int(session[5:7])
    if month == 1:
        return "train"
    if month == 2:
        return "validation"
    if month == 3:
        return "test"
    raise ValueError(f"session outside pilot window: {session}")


def session_from_path(path: Path) -> str:
    return path.name.removesuffix(".pkl")


def candidate_feature_vector(row: dict, strike_idx: int, right_idx: int) -> np.ndarray:
    """Build one causal candidate feature vector from a decision row."""
    option_features = np.asarray(row["option_ladder"][strike_idx, right_idx], dtype=np.float32)
    market_window = np.asarray(row["market_window"], dtype=np.float32)
    market_last = market_window[-1]
    finite_market = np.isfinite(market_window)
    market_count = finite_market.sum(axis=0)
    market_sum = np.where(finite_market, market_window, 0.0).sum(axis=0, dtype=np.float64)
    market_mean = np.divide(
        market_sum,
        market_count,
        out=np.zeros_like(market_sum, dtype=np.float64),
        where=market_count > 0,
    ).astype(np.float32)
    centered = np.where(finite_market, market_window - market_mean, 0.0)
    market_var = np.divide(
        (centered * centered).sum(axis=0, dtype=np.float64),
        market_count,
        out=np.zeros_like(market_sum, dtype=np.float64),
        where=market_count > 0,
    )
    market_std = np.sqrt(market_var).astype(np.float32)
    market_delta = market_last - market_window[0]
    offset = float(row["strike_offsets"][strike_idx])
    right = str(row["rights"][right_idx])
    side = np.array([1.0 if right == "C" else 0.0, 1.0 if right == "P" else 0.0], dtype=np.float32)
    shape = np.array([offset / 50.0, abs(offset) / 50.0], dtype=np.float32)
    environment = environment_prior_features(market_last, right)
    time_features = decision_time_features(row["decision_time"])
    return np.concatenate(
        [
            option_features,
            market_last,
            market_mean,
            market_std,
            market_delta,
            side,
            shape,
            environment,
            time_features,
        ]
    ).astype(np.float32)


def _finite_flag(value: float, predicate) -> float:
    if not np.isfinite(value):
        return 0.0
    return 1.0 if predicate(float(value)) else 0.0


def environment_prior_features(market_last: np.ndarray, right: str) -> np.ndarray:
    """Causal v3/v4-prior features expressed without hard-coded rules."""
    spx_close = float(market_last[0])
    spx_vwap = float(market_last[2])
    omar = float(market_last[3])
    session_range = float(market_last[4])
    momentum_5m = float(market_last[5])
    momentum_15m = float(market_last[6])
    is_call = right == "C"
    is_put = right == "P"

    above_vwap = _finite_flag(spx_close - spx_vwap, lambda x: x > 0.0)
    below_vwap = _finite_flag(spx_close - spx_vwap, lambda x: x < 0.0)
    omar_pos = _finite_flag(omar, lambda x: x > 0.0)
    omar_neg = _finite_flag(omar, lambda x: x < 0.0)
    mom5_pos = _finite_flag(momentum_5m, lambda x: x > 0.0)
    mom5_neg = _finite_flag(momentum_5m, lambda x: x < 0.0)
    mom15_pos = _finite_flag(momentum_15m, lambda x: x > 0.0)
    mom15_neg = _finite_flag(momentum_15m, lambda x: x < 0.0)

    if np.isfinite(spx_close) and np.isfinite(spx_vwap):
        vwap_gap_over_range = (spx_close - spx_vwap) / max(abs(session_range), 1.0)
    else:
        vwap_gap_over_range = 0.0
    range_pct = session_range / max(abs(spx_close), 1.0) if np.isfinite(session_range) else 0.0

    vwap_trend_aligned = float((is_call and above_vwap) or (is_put and below_vwap))
    vwap_mean_reversion_side = float((is_call and below_vwap) or (is_put and above_vwap))
    omar_aligned = float((is_call and omar_pos) or (is_put and omar_neg))
    omar_counter = float((is_call and omar_neg) or (is_put and omar_pos))
    momentum15_aligned = float((is_call and mom15_pos) or (is_put and mom15_neg))
    momentum15_counter = float((is_call and mom15_neg) or (is_put and mom15_pos))

    return np.asarray(
        [
            vwap_gap_over_range,
            range_pct,
            above_vwap,
            below_vwap,
            omar_pos,
            omar_neg,
            mom5_pos,
            mom5_neg,
            mom15_pos,
            mom15_neg,
            vwap_trend_aligned,
            vwap_mean_reversion_side,
            omar_aligned,
            omar_counter,
            momentum15_aligned,
            momentum15_counter,
        ],
        dtype=np.float32,
    )


def decision_time_features(decision_time: datetime) -> np.ndarray:
    """Causal time-of-day features for the regular-session decision minute."""
    local = decision_time.astimezone(ZoneInfo("America/New_York"))
    minutes = local.hour * 60 + local.minute
    open_minutes = 9 * 60 + 30
    no_new_entries_after = 15 * 60 + 30
    session_minutes = max(no_new_entries_after - open_minutes, 1)
    elapsed = min(max(minutes - open_minutes, 0), session_minutes)
    progress = elapsed / session_minutes
    radians = 2.0 * math.pi * progress
    first_30 = 1.0 if minutes < 10 * 60 else 0.0
    post_open_morning = 1.0 if 10 * 60 <= minutes < 11 * 60 + 30 else 0.0
    midday = 1.0 if 11 * 60 + 30 <= minutes < 13 * 60 + 30 else 0.0
    late_afternoon = 1.0 if minutes >= 13 * 60 + 30 else 0.0
    return np.asarray(
        [
            progress,
            1.0 - progress,
            math.sin(radians),
            math.cos(radians),
            first_30,
            post_open_morning,
            midday,
            late_afternoon,
        ],
        dtype=np.float32,
    )


def apply_candidate_feature_transform(
    features: np.ndarray,
    feature_transform: str = FEATURE_TRANSFORM_NONE,
) -> np.ndarray:
    """Return model-facing candidate features for a named live-causal transform."""
    mode = str(feature_transform or FEATURE_TRANSFORM_NONE)
    if mode not in FEATURE_TRANSFORM_CHOICES:
        raise ValueError(f"unknown feature_transform: {mode}")
    x = np.asarray(features, dtype=np.float32).copy()
    if mode == FEATURE_TRANSFORM_NONE:
        return x
    if mode == FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE:
        if x.ndim == 1:
            x[_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE_INDICES] = 0.0
        else:
            x[..., _VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE_INDICES] = 0.0
        return x
    if mode == FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE:
        # Keep option-quality signal, but make vendor/timestamp micro-differences less decisive.
        x[..., 3] = np.ceil(np.maximum(x[..., 3], 0.0) / 0.25) * 0.25
        x[..., 4] = np.ceil(np.maximum(x[..., 4], 0.0) / 0.01) * 0.01
        x[..., 5] = np.round(np.log1p(np.maximum(x[..., 5], 0.0)) * 2.0) / 2.0
        x[..., 6] = np.round(np.log1p(np.maximum(x[..., 6], 0.0)) * 2.0) / 2.0
        x[..., 7] = 0.0
        x[..., 8] = 0.0
        x[..., 9] = np.round(x[..., 9] / 0.01) * 0.01
        return x
    return x


def transform_decision_features(
    decisions: Sequence[DecisionCandidates],
    feature_transform: str = FEATURE_TRANSFORM_NONE,
) -> list[DecisionCandidates]:
    """Copy decision candidates with model-facing feature transforms applied."""
    mode = str(feature_transform or FEATURE_TRANSFORM_NONE)
    if mode == FEATURE_TRANSFORM_NONE:
        return list(decisions)
    return [
        replace(
            decision,
            features=apply_candidate_feature_transform(decision.features, mode),
        )
        for decision in decisions
    ]


def apply_vendor_microstructure_jitter(
    features: np.ndarray,
    *,
    iv_delta: float = 0.0,
    spread_delta: float = 0.0,
    spread_frac_delta: float = 0.0,
    bid_size_scale: float = 1.0,
    ask_size_scale: float = 1.0,
) -> np.ndarray:
    """Return features with deterministic live-plausible option quote jitter."""
    x = np.asarray(features, dtype=np.float32).copy()
    if x.shape[-1] <= 9:
        return x
    x[..., 3] = np.maximum(x[..., 3] + float(spread_delta), 0.0)
    x[..., 4] = np.maximum(x[..., 4] + float(spread_frac_delta), 0.0)
    x[..., 5] = np.maximum(x[..., 5] * float(bid_size_scale), 0.0)
    x[..., 6] = np.maximum(x[..., 6] * float(ask_size_scale), 0.0)
    x[..., 9] = np.maximum(x[..., 9] + float(iv_delta), 0.0)
    return x.astype(np.float32)


def jitter_decision_features(
    decisions: Sequence[DecisionCandidates],
    **scenario: float,
) -> list[DecisionCandidates]:
    """Copy decision candidates with one deterministic vendor-jitter scenario."""
    return [
        replace(
            decision,
            features=apply_vendor_microstructure_jitter(decision.features, **scenario),
        )
        for decision in decisions
    ]


def augment_decision_features_with_noise(
    decisions: Sequence[DecisionCandidates],
    feature_noise_augmentation: str = FEATURE_NOISE_AUGMENTATION_NONE,
) -> list[DecisionCandidates]:
    """Copy fit-only decisions with deterministic vendor-sensitive feature noise.

    This is intended only for model fitting. Calibration, validation,
    diagnostic, and live/IBKR confirmation rows should remain unaugmented.
    Labels and candidate identities are unchanged.
    """
    mode = str(feature_noise_augmentation or FEATURE_NOISE_AUGMENTATION_NONE)
    if mode not in FEATURE_NOISE_AUGMENTATION_CHOICES:
        raise ValueError(f"unknown feature_noise_augmentation: {mode}")
    if mode == FEATURE_NOISE_AUGMENTATION_NONE:
        return list(decisions)
    augmented: list[DecisionCandidates] = []
    for decision in decisions:
        for _name, scenario in VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS:
            augmented.append(
                replace(
                    decision,
                    features=apply_vendor_microstructure_jitter(
                        decision.features,
                        **scenario,
                    ),
                )
            )
    return augmented


def decision_candidates_from_row(
    *,
    session: str,
    row: dict,
    policy_index: int,
) -> DecisionCandidates | None:
    """Extract valid candidate examples for one decision row."""
    mask = np.asarray(row["candidate_mask"], dtype=bool)
    labels = np.asarray(row["labels_net_pnl"], dtype=np.float32)[:, :, policy_index]
    valid = mask & np.isfinite(labels)
    if not valid.any():
        return None

    features: list[np.ndarray] = []
    ys: list[float] = []
    offsets: list[float] = []
    rights: list[str] = []
    contract_ids: list[str] = []
    entry_asks: list[float] = []
    contract_id_matrix = np.asarray(row.get("contract_ids"), dtype=object)
    quote_metadata = row.get("contract_quote_metadata") or {}
    teacher_matrix = row.get("teacher_candidate_labels")
    teacher_values: list[float] | None = [] if teacher_matrix is not None else None
    if teacher_matrix is not None:
        teacher_matrix = np.asarray(teacher_matrix, dtype=np.float32)
    for strike_idx, right_idx in zip(*np.where(valid)):
        option_features = np.asarray(
            row["option_ladder"][int(strike_idx), int(right_idx)],
            dtype=np.float32,
        )
        features.append(candidate_feature_vector(row, int(strike_idx), int(right_idx)))
        ys.append(float(labels[strike_idx, right_idx]))
        offsets.append(float(row["strike_offsets"][strike_idx]))
        rights.append(str(row["rights"][right_idx]))
        contract_id = str(contract_id_matrix[int(strike_idx), int(right_idx)])
        contract_ids.append(contract_id)
        quote = quote_metadata.get(contract_id) or {}
        try:
            ask = float(quote.get("ask"))
        except (TypeError, ValueError):
            ask = float("nan")
        if not np.isfinite(ask) and len(option_features) > 1:
            ask = float(option_features[1])
        entry_asks.append(ask)
        if teacher_values is not None:
            teacher_values.append(float(teacher_matrix[int(strike_idx), int(right_idx)]))

    market_window = np.asarray(row["market_window"], dtype=np.float32)
    return DecisionCandidates(
        session=session,
        decision_time=row["decision_time"],
        features=np.vstack(features).astype(np.float32),
        labels=np.asarray(ys, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.float32),
        rights=np.asarray(rights, dtype=object),
        market_last=market_window[-1].astype(np.float32),
        contract_ids=np.asarray(contract_ids, dtype=object),
        teacher_labels=(
            np.asarray(teacher_values, dtype=np.float32)
            if teacher_values is not None
            else None
        ),
        entry_asks=np.asarray(entry_asks, dtype=np.float32),
    )


def load_decisions(paths: Sequence[Path], *, policy_index: int) -> list[DecisionCandidates]:
    decisions: list[DecisionCandidates] = []
    for path in sorted(paths):
        session = session_from_path(path)
        with path.open("rb") as f:
            rows = pickle.load(f)
        for row in rows:
            decision = decision_candidates_from_row(
                session=session, row=row, policy_index=policy_index
            )
            if decision is not None:
                decisions.append(decision)
    return decisions


def collect_examples(
    decisions: Sequence[DecisionCandidates],
    *,
    max_examples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack([d.features for d in decisions]).astype(np.float32)
    labels = np.concatenate([d.labels for d in decisions]).astype(np.float32)
    if max_examples is not None and len(labels) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(labels), size=max_examples, replace=False)
        features = features[idx]
        labels = labels[idx]
    return features, labels


def _target(labels: np.ndarray, config: PilotConfig) -> np.ndarray:
    if config.target_mode == "profit_classifier":
        return (labels > config.positive_label_threshold).astype(np.float32)
    clipped = np.clip(labels, -config.target_clip, config.target_clip)
    return (clipped / config.target_scale).astype(np.float32)


def collect_training_examples(
    decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
    max_examples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect features and transformed training targets for the selected objective."""
    if config.target_mode in TEACHER_TARGET_MODES:
        features = np.vstack([d.features for d in decisions]).astype(np.float32)
        targets_by_decision: list[np.ndarray] = []
        for decision in decisions:
            labels = np.asarray(decision.labels, dtype=np.float32)
            teacher = (
                np.asarray(decision.teacher_labels, dtype=np.float32)
                if decision.teacher_labels is not None
                else np.zeros(len(labels), dtype=np.float32)
            )
            targets = (teacher > 0.0).astype(np.float32)
            if config.target_mode == "protocol101_teacher_profitable_classifier":
                targets = (
                    (targets > 0.0)
                    & np.isfinite(labels)
                    & (labels > float(config.positive_label_threshold))
                ).astype(np.float32)
            elif config.target_mode == "protocol101_teacher_edge_regression":
                edge = np.where(
                    np.isfinite(labels),
                    labels - float(config.positive_label_threshold),
                    -float(config.target_clip),
                )
                targets = (
                    np.clip(edge, -config.target_clip, config.target_clip)
                    / float(config.target_scale)
                    * targets
                ).astype(np.float32)
            targets_by_decision.append(targets)
        targets = np.concatenate(targets_by_decision).astype(np.float32)
        if max_examples is not None and len(targets) > max_examples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(targets), size=max_examples, replace=False)
            features = features[idx]
            targets = targets[idx]
        return features, targets

    if config.target_mode in {
        "decision_profit_presence_classifier",
        "decision_best_profit_regression",
    }:
        features = np.vstack([d.features for d in decisions]).astype(np.float32)
        targets_by_decision: list[np.ndarray] = []
        for decision in decisions:
            labels = np.asarray(decision.labels, dtype=np.float32)
            if len(labels) == 0 or not np.isfinite(labels).any():
                targets_by_decision.append(np.zeros(len(labels), dtype=np.float32))
                continue
            max_label = float(np.nanmax(labels))
            if config.target_mode == "decision_profit_presence_classifier":
                target_value = 1.0 if max_label > float(config.positive_label_threshold) else 0.0
            else:
                target_value = (
                    np.clip(max_label, -config.target_clip, config.target_clip)
                    / config.target_scale
                )
            targets_by_decision.append(
                np.full(len(labels), float(target_value), dtype=np.float32)
            )
        targets = np.concatenate(targets_by_decision).astype(np.float32)
        if max_examples is not None and len(targets) > max_examples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(targets), size=max_examples, replace=False)
            features = features[idx]
            targets = targets[idx]
        return features, targets

    if config.target_mode in {"decision_top_profit_classifier", "decision_top_profit_regression"}:
        features = np.vstack([d.features for d in decisions]).astype(np.float32)
        targets_by_decision: list[np.ndarray] = []
        for decision in decisions:
            labels = np.asarray(decision.labels, dtype=np.float32)
            if len(labels) == 0 or not np.isfinite(labels).any():
                targets_by_decision.append(np.zeros(len(labels), dtype=np.float32))
                continue
            max_label = float(np.nanmax(labels))
            if max_label <= float(config.positive_label_threshold):
                targets_by_decision.append(np.zeros(len(labels), dtype=np.float32))
            else:
                if config.target_mode == "decision_top_profit_classifier":
                    targets_by_decision.append(
                        np.asarray(labels == max_label, dtype=np.float32)
                    )
                else:
                    target = np.zeros(len(labels), dtype=np.float32)
                    target[np.asarray(labels == max_label, dtype=bool)] = (
                        np.clip(max_label, -config.target_clip, config.target_clip)
                        / config.target_scale
                    )
                    targets_by_decision.append(target)
        targets = np.concatenate(targets_by_decision).astype(np.float32)
        if max_examples is not None and len(targets) > max_examples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(targets), size=max_examples, replace=False)
            features = features[idx]
            targets = targets[idx]
        return features, targets

    if config.target_mode not in {"decision_relative_regression", "blended_relative_regression"}:
        features, labels = collect_examples(decisions, max_examples=max_examples, seed=seed)
        return features, _target(labels, config)

    features = np.vstack([d.features for d in decisions]).astype(np.float32)
    relative_targets = np.concatenate(
        [
            np.clip(d.labels - float(np.nanmedian(d.labels)), -config.target_clip, config.target_clip)
            / config.target_scale
            for d in decisions
        ]
    ).astype(np.float32)
    if config.target_mode == "blended_relative_regression":
        raw_labels = np.concatenate([d.labels for d in decisions]).astype(np.float32)
        absolute_targets = _target(raw_labels, config)
        weight = min(max(float(config.relative_target_weight), 0.0), 1.0)
        targets = ((1.0 - weight) * absolute_targets + weight * relative_targets).astype(np.float32)
    else:
        targets = relative_targets
    if max_examples is not None and len(targets) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(targets), size=max_examples, replace=False)
        features = features[idx]
        targets = targets[idx]
    return features, targets


def _listwise_training_decisions(
    decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
) -> list[tuple[DecisionCandidates, int]]:
    """Return decisions whose best candidate is profitable enough for listwise ranking."""
    usable: list[tuple[DecisionCandidates, int]] = []
    for decision in decisions:
        labels = np.asarray(decision.labels, dtype=np.float32)
        if len(labels) == 0 or not np.isfinite(labels).any():
            continue
        best_idx = int(np.nanargmax(labels))
        if float(labels[best_idx]) <= float(config.positive_label_threshold):
            continue
        usable.append((decision, best_idx))
    return usable


def _listwise_validation_loss(
    model: CandidateMLP,
    scaler: FeatureScaler,
    decisions: Sequence[tuple[DecisionCandidates, int]],
) -> float:
    """Mean cross-entropy loss across variable-width decision candidate sets."""
    if not decisions:
        return math.inf
    model.eval()
    losses: list[float] = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for decision, best_idx in decisions:
            x = scaler.transform(decision.features)
            logits = model(torch.from_numpy(x)).reshape(1, -1)
            target = torch.tensor([best_idx], dtype=torch.long)
            losses.append(float(loss_fn(logits, target).detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def train_model(
    train_decisions: Sequence[DecisionCandidates],
    validation_decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
) -> tuple[CandidateMLP, FeatureScaler, list[dict]]:
    """Fit the small MLP and return training history."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.target_mode in LISTWISE_TARGET_MODES:
        x_train_raw, _labels = collect_examples(
            train_decisions,
            max_examples=config.max_train_examples,
            seed=config.seed,
        )
        scaler = FeatureScaler.fit(x_train_raw)
        input_dim = x_train_raw.shape[1]
        del x_train_raw, _labels

        train_rank_decisions = _listwise_training_decisions(
            train_decisions,
            config=config,
        )
        validation_rank_decisions = _listwise_training_decisions(
            validation_decisions,
            config=config,
        )
        model = CandidateMLP(input_dim=input_dim, hidden_dim=config.hidden_dim)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        history: list[dict] = []
        rng = random.Random(config.seed)

        for epoch in range(1, config.epochs + 1):
            model.train()
            shuffled = list(train_rank_decisions)
            rng.shuffle(shuffled)
            losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            batch_decisions = max(1, min(int(config.batch_size // 64), 256))
            for decision, best_idx in shuffled:
                x = scaler.transform(decision.features)
                logits = model(torch.from_numpy(x)).reshape(1, -1)
                target = torch.tensor([best_idx], dtype=torch.long)
                loss = loss_fn(logits, target) / float(batch_decisions)
                loss.backward()
                losses.append(float(loss.detach().cpu()) * float(batch_decisions))
                pending += 1
                if pending >= batch_decisions:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
            if pending:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            val_loss = _listwise_validation_loss(model, scaler, validation_rank_decisions)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "epoch": epoch,
                    "target_mode": config.target_mode,
                    "train_cross_entropy": float(np.mean(losses)) if losses else math.nan,
                    "validation_cross_entropy": val_loss,
                    "train_loss": float(np.mean(losses)) if losses else math.nan,
                    "validation_loss": val_loss,
                    "train_rank_decisions": len(train_rank_decisions),
                    "validation_rank_decisions": len(validation_rank_decisions),
                    "is_best": val_loss <= best_val_loss,
                }
            )
        model.load_state_dict(best_state)
        return model, scaler, history

    x_train_raw, y_train = collect_training_examples(
        train_decisions,
        config=config,
        max_examples=config.max_train_examples,
        seed=config.seed,
    )
    x_val_raw, y_val = collect_training_examples(
        validation_decisions,
        config=config,
        max_examples=120_000,
        seed=config.seed + 1,
    )
    scaler = FeatureScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    model = CandidateMLP(input_dim=x_train.shape[1], hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if config.target_mode in CLASSIFIER_TARGET_MODES:
        pos_weight = None
        if config.sample_weight_mode == "balanced_classifier":
            positives = int(np.sum(y_train > 0.5))
            negatives = int(len(y_train) - positives)
            if positives > 0 and negatives > 0:
                pos_weight = torch.tensor(
                    [float(negatives) / float(positives)],
                    dtype=torch.float32,
                )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "bce"
    else:
        loss_fn = nn.HuberLoss(delta=1.0)
        loss_name = "huber"
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    history: list[dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).detach().cpu())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "target_mode": config.target_mode,
                f"train_{loss_name}": float(np.mean(losses)) if losses else math.nan,
                f"validation_{loss_name}": val_loss,
                "train_loss": float(np.mean(losses)) if losses else math.nan,
                "validation_loss": val_loss,
                "is_best": val_loss <= best_val_loss,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def predict_decisions(
    model: CandidateMLP,
    scaler: FeatureScaler,
    decisions: Sequence[DecisionCandidates],
    *,
    target_scale: float,
    prediction_transform: str = "identity",
    batch_size: int = 32768,
) -> list[np.ndarray]:
    """Predict executable PnL dollars for each decision candidate."""
    if not decisions:
        return []
    lengths = [len(d.labels) for d in decisions]
    x_raw = np.vstack([d.features for d in decisions]).astype(np.float32)
    x = scaler.transform(x_raw)
    preds: list[np.ndarray] = []
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            out = model(batch)
            if prediction_transform == "sigmoid":
                out = torch.sigmoid(out)
                scale = 1.0
            elif prediction_transform == "identity_unit":
                scale = 1.0
            else:
                scale = target_scale
            out_chunks.append(out.cpu().numpy().astype(np.float32) * scale)
    flat = np.concatenate(out_chunks)
    cursor = 0
    for length in lengths:
        preds.append(flat[cursor : cursor + length])
        cursor += length
    return preds


def simulate_model_policy(
    decisions: Sequence[DecisionCandidates],
    predictions: Sequence[np.ndarray],
    *,
    threshold: float,
    cooldown_minutes: int,
    strategy: str,
    entry_filter: str = "none",
    min_score_margin: float = 0.0,
    max_score_ceiling: float = 0.0,
    max_trades_per_session: int = 0,
    max_daily_loss: float = 0.0,
    selection_mode: str = SELECTION_MODE_TOP_SCORE,
    starting_cash: float = 10_000.0,
    contract_multiplier: float = 100.0,
    enforce_affordability: bool = True,
    cash_pnl_adjustment: float = 0.0,
) -> list[Trade]:
    trades: list[Trade] = []
    cash = float(starting_cash)
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    realized_pnl_by_session: dict[str, float] = {}
    for decision, pred in zip(decisions, predictions):
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        if max_trades_per_session > 0 and trades_by_session.get(decision.session, 0) >= max_trades_per_session:
            continue
        if max_daily_loss > 0.0 and realized_pnl_by_session.get(decision.session, 0.0) <= -float(max_daily_loss):
            continue
        if len(pred) == 0:
            continue
        affordable_mask = None
        if enforce_affordability and decision.entry_asks is not None:
            asks = np.asarray(decision.entry_asks, dtype=np.float32)
            affordable_mask = np.asarray(
                np.isfinite(asks)
                & (asks > 0.0)
                & ((asks * float(contract_multiplier)) <= cash + 1e-9),
                dtype=bool,
            )
        top = top_prediction(
            decision,
            pred,
            entry_filter=entry_filter,
            extra_allowed_mask=affordable_mask,
            selection_mode=selection_mode,
        )
        if top is None:
            continue
        idx, score, margin = top
        if max_score_ceiling > 0.0 and score >= float(max_score_ceiling):
            continue
        if score < threshold:
            continue
        if min_score_margin > 0.0 and margin < float(min_score_margin):
            continue
        pnl = float(decision.labels[idx])
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=score,
                right=str(decision.rights[idx]),
                offset=float(decision.offsets[idx]),
                strategy=strategy,
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        realized_pnl_by_session[decision.session] = realized_pnl_by_session.get(decision.session, 0.0) + pnl
        cash += pnl + float(cash_pnl_adjustment)
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=cooldown_minutes
        )
    return trades


def top_prediction(
    decision: DecisionCandidates,
    pred: np.ndarray,
    *,
    entry_filter: str = "none",
    extra_allowed_mask: np.ndarray | None = None,
    selection_mode: str = SELECTION_MODE_TOP_SCORE,
) -> tuple[int, float, float] | None:
    """Return top eligible candidate index, score, and top-vs-runner-up margin."""
    mode = str(selection_mode or SELECTION_MODE_TOP_SCORE)
    if mode not in SELECTION_MODE_CHOICES:
        raise ValueError(f"unknown selection_mode: {mode}")
    scores = np.asarray(pred, dtype=np.float32)
    if len(scores) == 0:
        return None
    allowed = entry_filter_mask(decision, entry_filter)
    if len(allowed) != len(scores):
        return None
    if extra_allowed_mask is not None:
        extra_allowed = np.asarray(extra_allowed_mask, dtype=bool)
        if len(extra_allowed) != len(scores):
            return None
        allowed = allowed & extra_allowed
    allowed = np.asarray(allowed & np.isfinite(scores), dtype=bool)
    if not allowed.any():
        return None
    eligible_idx = np.flatnonzero(allowed)
    eligible_scores = scores[eligible_idx]
    if mode == SELECTION_MODE_TOP_SCORE:
        order = np.argsort(eligible_scores)
        top_pos = int(order[-1])
        top_idx = int(eligible_idx[top_pos])
    else:
        target = {
            SELECTION_MODE_STABLE_ABS_OFFSET_10: 10.0,
            SELECTION_MODE_STABLE_ABS_OFFSET_15: 15.0,
            SELECTION_MODE_STABLE_ABS_OFFSET_20: 20.0,
        }[mode]
        offsets = np.asarray(decision.offsets, dtype=np.float32)
        rights = np.asarray(decision.rights, dtype=object)
        top_idx = min(
            (int(idx) for idx in eligible_idx),
            key=lambda idx: (
                abs(abs(float(offsets[idx])) - target),
                abs(float(offsets[idx])),
                float(offsets[idx]),
                str(rights[idx]),
                idx,
            ),
        )
    top_score = float(scores[top_idx])
    score_order = np.argsort(eligible_scores)
    if len(score_order) == 1:
        margin = float("inf")
    else:
        if mode == SELECTION_MODE_TOP_SCORE:
            second_score = float(eligible_scores[int(score_order[-2])])
        else:
            other_scores = [
                float(scores[int(idx)])
                for idx in eligible_idx
                if int(idx) != int(top_idx)
            ]
            second_score = max(other_scores) if other_scores else float("-inf")
        margin = top_score - second_score
    return top_idx, top_score, float(margin)


def _put_near_after_0940_vwap_m2_10_base_mask(decision: DecisionCandidates) -> np.ndarray:
    local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
    minutes = local.hour * 60 + local.minute
    if minutes < 9 * 60 + 40:
        return np.zeros(len(decision.labels), dtype=bool)
    spx_close = float(decision.market_last[0])
    spx_vwap = float(decision.market_last[2])
    if not np.isfinite(spx_close) or not np.isfinite(spx_vwap):
        return np.zeros(len(decision.labels), dtype=bool)
    vwap_gap = spx_close - spx_vwap
    if vwap_gap < -2.0 or vwap_gap >= 10.0:
        return np.zeros(len(decision.labels), dtype=bool)
    offsets = np.asarray(decision.offsets, dtype=np.float32)
    rights = np.asarray(decision.rights, dtype=object)
    return np.asarray(
        (rights == "P")
        & np.isfinite(offsets)
        & (np.abs(offsets) >= 10.0)
        & (np.abs(offsets) <= 20.0),
        dtype=bool,
    )


def _near_after_0940_vwap_m2_10_base_mask(decision: DecisionCandidates) -> np.ndarray:
    local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
    minutes = local.hour * 60 + local.minute
    if minutes < 9 * 60 + 40:
        return np.zeros(len(decision.labels), dtype=bool)
    spx_close = float(decision.market_last[0])
    spx_vwap = float(decision.market_last[2])
    if not np.isfinite(spx_close) or not np.isfinite(spx_vwap):
        return np.zeros(len(decision.labels), dtype=bool)
    vwap_gap = spx_close - spx_vwap
    if vwap_gap < -2.0 or vwap_gap >= 10.0:
        return np.zeros(len(decision.labels), dtype=bool)
    offsets = np.asarray(decision.offsets, dtype=np.float32)
    return np.asarray(
        np.isfinite(offsets)
        & (np.abs(offsets) >= 10.0)
        & (np.abs(offsets) <= 20.0),
        dtype=bool,
    )


def _market_last_value(decision: DecisionCandidates, index: int) -> float:
    try:
        value = float(decision.market_last[index])
    except (IndexError, TypeError, ValueError):
        return float("nan")
    return value


def entry_filter_mask(decision: DecisionCandidates, entry_filter: str) -> np.ndarray:
    """Return candidate eligibility for live-reproducible two-stage filters."""
    if entry_filter in {"", "none"}:
        return np.ones(len(decision.labels), dtype=bool)
    if entry_filter == "vwap_aligned":
        spx_close = float(decision.market_last[0])
        spx_vwap = float(decision.market_last[2])
        if not np.isfinite(spx_close) or not np.isfinite(spx_vwap) or spx_close == spx_vwap:
            return np.zeros(len(decision.labels), dtype=bool)
        if spx_close > spx_vwap:
            return np.asarray(decision.rights == "C", dtype=bool)
        return np.asarray(decision.rights == "P", dtype=bool)
    if entry_filter == "premium_floor_3":
        features = np.asarray(decision.features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] <= 1:
            return np.zeros(len(decision.labels), dtype=bool)
        asks = features[:, 1]
        return np.asarray(np.isfinite(asks) & (asks >= 3.0), dtype=bool)
    if entry_filter == "near_10_20_offset":
        offsets = np.asarray(decision.offsets, dtype=np.float32)
        return np.asarray(
            np.isfinite(offsets) & (np.abs(offsets) >= 10.0) & (np.abs(offsets) <= 20.0),
            dtype=bool,
        )
    if entry_filter == "put_only":
        return np.asarray(decision.rights == "P", dtype=bool)
    if entry_filter == "put_near_10_20_offset":
        offsets = np.asarray(decision.offsets, dtype=np.float32)
        rights = np.asarray(decision.rights, dtype=object)
        return np.asarray(
            (rights == "P")
            & np.isfinite(offsets)
            & (np.abs(offsets) >= 10.0)
            & (np.abs(offsets) <= 20.0),
            dtype=bool,
        )
    if entry_filter == "put_near_after_0940":
        local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if minutes < 9 * 60 + 40:
            return np.zeros(len(decision.labels), dtype=bool)
        offsets = np.asarray(decision.offsets, dtype=np.float32)
        rights = np.asarray(decision.rights, dtype=object)
        return np.asarray(
            (rights == "P")
            & np.isfinite(offsets)
            & (np.abs(offsets) >= 10.0)
            & (np.abs(offsets) <= 20.0),
            dtype=bool,
        )
    if entry_filter == "put_near_after_0940_vwap_m2_10":
        return _put_near_after_0940_vwap_m2_10_base_mask(decision)
    if entry_filter == "put_near_after_0940_vwap_m2_10_omar_neg":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        omar = float(decision.market_last[3])
        return base if np.isfinite(omar) and omar < 0.0 else np.zeros(len(decision.labels), dtype=bool)
    if entry_filter == "put_near_after_0940_vwap_m2_10_range_20_45":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        session_range = float(decision.market_last[4])
        return (
            base
            if np.isfinite(session_range) and 20.0 <= session_range < 45.0
            else np.zeros(len(decision.labels), dtype=bool)
        )
    if entry_filter == "put_near_after_0940_vwap_m2_10_near_vwap":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        spx_close = float(decision.market_last[0])
        spx_vwap = float(decision.market_last[2])
        if not np.isfinite(spx_close) or not np.isfinite(spx_vwap):
            return np.zeros(len(decision.labels), dtype=bool)
        return base if -2.0 <= spx_close - spx_vwap <= 2.0 else np.zeros(len(decision.labels), dtype=bool)
    if entry_filter == "put_near_after_0940_vwap_m2_10_premium_gte_7_5":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        if decision.entry_asks is None:
            return np.zeros(len(decision.labels), dtype=bool)
        asks = np.asarray(decision.entry_asks, dtype=np.float32)
        return np.asarray(base & np.isfinite(asks) & (asks >= 7.5), dtype=bool)
    if entry_filter == "put_near_after_0940_vwap_m2_10_mom15_nonpos":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        momentum15 = _market_last_value(decision, 6)
        return base if np.isfinite(momentum15) and momentum15 <= 0.0 else np.zeros(len(decision.labels), dtype=bool)
    if entry_filter == "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        omar = _market_last_value(decision, 3)
        momentum15 = _market_last_value(decision, 6)
        return (
            base
            if np.isfinite(omar)
            and omar > 0.0
            and np.isfinite(momentum15)
            and momentum15 <= 0.0
            else np.zeros(len(decision.labels), dtype=bool)
        )
    if entry_filter == "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos":
        base = _put_near_after_0940_vwap_m2_10_base_mask(decision)
        momentum15 = _market_last_value(decision, 6)
        if decision.entry_asks is None or not np.isfinite(momentum15) or momentum15 > 0.0:
            return np.zeros(len(decision.labels), dtype=bool)
        asks = np.asarray(decision.entry_asks, dtype=np.float32)
        return np.asarray(base & np.isfinite(asks) & (asks >= 7.5), dtype=bool)
    if entry_filter in {
        "near_after_0940_vwap_m2_10_mom15_side",
        "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
    }:
        base = _near_after_0940_vwap_m2_10_base_mask(decision)
        momentum15 = _market_last_value(decision, 6)
        if not np.isfinite(momentum15):
            return np.zeros(len(decision.labels), dtype=bool)
        rights = np.asarray(decision.rights, dtype=object)
        side_mask = np.asarray(rights == ("C" if momentum15 > 0.0 else "P"), dtype=bool)
        out = np.asarray(base & side_mask, dtype=bool)
        if entry_filter == "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5":
            if decision.entry_asks is None:
                return np.zeros(len(decision.labels), dtype=bool)
            asks = np.asarray(decision.entry_asks, dtype=np.float32)
            out = np.asarray(out & np.isfinite(asks) & (asks >= 7.5), dtype=bool)
        return out
    if entry_filter == "morning_1000_1129":
        local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if 10 * 60 <= minutes < 11 * 60 + 30:
            return np.ones(len(decision.labels), dtype=bool)
        return np.zeros(len(decision.labels), dtype=bool)
    if entry_filter == "morning_near_10_20_offset":
        local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if not (10 * 60 <= minutes < 11 * 60 + 30):
            return np.zeros(len(decision.labels), dtype=bool)
        offsets = np.asarray(decision.offsets, dtype=np.float32)
        return np.asarray(
            np.isfinite(offsets) & (np.abs(offsets) >= 10.0) & (np.abs(offsets) <= 20.0),
            dtype=bool,
        )
    if entry_filter == "above_vwap_omar_pos_after_open":
        local = decision.decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if minutes < 10 * 60:
            return np.zeros(len(decision.labels), dtype=bool)
        spx_close = float(decision.market_last[0])
        spx_vwap = float(decision.market_last[2])
        omar = float(decision.market_last[3])
        if (
            not np.isfinite(spx_close)
            or not np.isfinite(spx_vwap)
            or not np.isfinite(omar)
            or spx_close <= spx_vwap
            or omar <= 0.0
        ):
            return np.zeros(len(decision.labels), dtype=bool)
        return np.ones(len(decision.labels), dtype=bool)
    raise ValueError(f"unknown entry_filter: {entry_filter}")


def _choose_nearest(decision: DecisionCandidates, right: str) -> int | None:
    idx = np.where(decision.rights == right)[0]
    if len(idx) == 0:
        return None
    local = idx[np.argmin(np.abs(decision.offsets[idx]))]
    return int(local)


def _choose_vwap_omar(decision: DecisionCandidates) -> int | None:
    spx_close = float(decision.market_last[0])
    spx_vwap = float(decision.market_last[2])
    omar = float(decision.market_last[3])
    if not all(np.isfinite([spx_close, spx_vwap, omar])):
        return None
    if spx_close > spx_vwap and omar > 0:
        return _choose_nearest(decision, "C")
    if spx_close < spx_vwap and omar < 0:
        return _choose_nearest(decision, "P")
    return None


def simulate_baseline(
    decisions: Sequence[DecisionCandidates],
    *,
    kind: str,
    cooldown_minutes: int,
    seed: int = 42,
) -> list[Trade]:
    rng = np.random.default_rng(seed)
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    for decision in decisions:
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        if kind == "random_valid":
            idx = int(rng.integers(0, len(decision.labels)))
        elif kind == "atm_call":
            chosen = _choose_nearest(decision, "C")
            if chosen is None:
                continue
            idx = chosen
        elif kind == "atm_put":
            chosen = _choose_nearest(decision, "P")
            if chosen is None:
                continue
            idx = chosen
        elif kind == "vwap_omar":
            chosen = _choose_vwap_omar(decision)
            if chosen is None:
                continue
            idx = chosen
        else:
            raise ValueError(f"unknown baseline kind: {kind}")
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=float(decision.labels[idx]),
                score=None,
                right=str(decision.rights[idx]),
                offset=float(decision.offsets[idx]),
                strategy=kind,
            )
        )
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=cooldown_minutes
        )
    return trades


def metrics_for_trades(trades: Sequence[Trade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sessions_traded": 0,
            "positive_day_fraction": 0.0,
        }
    ordered = sorted(trades, key=lambda t: (t.decision_time, t.strategy))
    pnl = np.asarray([t.pnl for t in ordered], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = equity - peak
    by_day: dict[str, float] = {}
    for trade in ordered:
        by_day[trade.session] = by_day.get(trade.session, 0.0) + trade.pnl
    daily = np.asarray(list(by_day.values()), dtype=float)
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(ordered)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sessions_traded": int(len(by_day)),
        "mean_daily_pnl": float(daily.mean()) if len(daily) else 0.0,
        "median_daily_pnl": float(np.median(daily)) if len(daily) else 0.0,
        "positive_day_fraction": float((daily > 0).mean()) if len(daily) else 0.0,
    }


def choose_threshold(
    decisions: Sequence[DecisionCandidates],
    predictions: Sequence[np.ndarray],
    *,
    config: PilotConfig,
) -> tuple[float, list[dict]]:
    top_scores = np.asarray(
        [
            float(top[1])
            for decision, pred in zip(decisions, predictions)
            if (
                top := top_prediction(
                    decision,
                    pred,
                    entry_filter=config.entry_filter,
                    selection_mode=config.selection_mode,
                )
            )
            is not None
            and (
                float(config.max_score_ceiling) <= 0.0
                or float(top[1]) < float(config.max_score_ceiling)
            )
        ],
        dtype=float,
    )
    if len(top_scores) == 0:
        return float("inf"), []
    quantiles = np.linspace(0.0, 0.98, 50)
    thresholds = sorted(set(np.quantile(top_scores, quantiles).round(4).tolist() + [0.0]))
    sweep: list[dict] = []
    for threshold in thresholds:
        trades = simulate_model_policy(
            decisions,
            predictions,
            threshold=float(threshold),
            cooldown_minutes=config.cooldown_minutes,
            strategy="neural_threshold",
            entry_filter=config.entry_filter,
            min_score_margin=config.min_score_margin,
            max_score_ceiling=config.max_score_ceiling,
            max_trades_per_session=config.max_trades_per_session,
            max_daily_loss=config.max_daily_loss,
            selection_mode=config.selection_mode,
        )
        metric = metrics_for_trades(trades)
        sweep.append({"threshold": float(threshold), **metric})

    eligible = [x for x in sweep if x["trades"] >= config.min_validation_trades]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda x: (
            x["total_pnl"],
            x["profit_factor"] if np.isfinite(x["profit_factor"]) else 999.0,
            x["trades"],
        ),
    )
    return float(best["threshold"]), sweep


def summarize_random_baseline(
    decisions: Sequence[DecisionCandidates],
    *,
    config: PilotConfig,
) -> dict:
    metrics = []
    for seed in range(config.seed, config.seed + config.random_seeds):
        trades = simulate_baseline(
            decisions,
            kind="random_valid",
            cooldown_minutes=config.cooldown_minutes,
            seed=seed,
        )
        metrics.append(metrics_for_trades(trades))
    total = np.asarray([m["total_pnl"] for m in metrics], dtype=float)
    trades_n = np.asarray([m["trades"] for m in metrics], dtype=float)
    return {
        "runs": config.random_seeds,
        "trades_mean": float(trades_n.mean()) if len(trades_n) else 0.0,
        "total_pnl_mean": float(total.mean()) if len(total) else 0.0,
        "total_pnl_std": float(total.std()) if len(total) else 0.0,
        "best_total_pnl": float(total.max()) if len(total) else 0.0,
        "worst_total_pnl": float(total.min()) if len(total) else 0.0,
    }


def model_state_dict(
    *,
    model: CandidateMLP,
    scaler: FeatureScaler,
    config: PilotConfig,
    history: Sequence[dict],
    input_dim: int,
) -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "input_dim": input_dim,
        "config": asdict(config),
        "scaler": scaler.to_dict(),
        "history": list(history),
        "model_state": model.state_dict(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
