"""Generalization Protocol 002 research harness primitives.

This module keeps the next v4 research pass deliberately bounded:

* Q1 v4 CBBO-derived labels remain executable truth.
* Q4 2025 remains frozen audit data.
* v2/v3 market-structure caches may add causal context, but never labels.
* Surface variants choose among flat plus SPXW call/put contract tokens.

The code is research infrastructure, not a live trading engine.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.dataset.spxw_0dte_neural import OPTION_FEATURE_NAMES
from v4.ingest.index_bars import load_spx_1m, load_vix_1m
from v4.model import market_structure
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import FeatureScaler, PilotConfig, Trade
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration


_NY = ZoneInfo("America/New_York")
_OPTION_INDEX = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
_BASE_MARKET_FEATURES = 28
_STRUCTURE_FEATURE_NAMES = (
    "true_spx_close",
    "true_vix_close",
    "sigma_pos",
    "abs_sigma_pos",
    "spx_vwap_est",
    "vwap_dist_pct",
    "vwap_slope_5m_pct",
    "omar_high",
    "omar_low",
    "omar_mid",
    "omar_range",
    "omar_range_pct",
    "omar_mid_pos_units",
    "omar_retest_dist_norm",
    "first15_available",
    "first15_range_pct",
    "first15_close_position",
    "first15_acceptance",
    "inside_first15",
    "opening_gap_pct",
    "last10_range_over_omar",
    "last10_break_state",
    "atr15_pct",
    "minute_fraction",
    "bucket_first_30",
    "bucket_post_open_morning",
    "bucket_midday",
    "bucket_late_afternoon",
)
_PRESSURE_FEATURE_NAMES = (
    "log_mid",
    "log_open_interest",
    "log_option_volume",
    "gamma_abs",
    "theta_abs",
    "gamma_theta_ratio",
    "theta_over_mid",
    "spread_dollars_over_gamma",
    "premium_decay_burden",
    "liquidity_score",
)
_APLUS_PATTERN_NAMES = (
    "sigma_trend_continuation",
    "vwap_reclaim",
    "vwap_hold_continuation",
    "vwap_pullback_resume",
    "last10_breakout",
    "compression_breakout",
    "momentum_ignition",
    "pullback_resume",
    "omar_mid_reclaim",
    "omar_retest_bounce",
    "first15_acceptance_break",
    "first15_inside_reversal",
)
_APLUS_PATTERN_FEATURE_NAMES = (
    *[f"pattern_{name}" for name in _APLUS_PATTERN_NAMES],
    "pattern_count_norm",
    "pattern_side_gap_atr",
    "pattern_side_sigma",
    "pattern_side_move1_atr",
    "pattern_side_move5_atr",
    "pattern_side_move15_atr",
    "pattern_compression_ratio",
)
_APLUS_VALUE_FEATURE_NAMES = (
    "abs_delta",
    "delta_atr_capture",
    "convexity_per_premium",
    "theta_burden_hold",
    "spread_tax",
    "breakeven_atr",
    "gamma_theta_ratio_scaled",
    "contract_value_score",
    "worth_spread_flag",
    "obvious_overpay_flag",
)
_APLUS_INTERACTION_FEATURE_NAMES = (
    "pattern_value_score",
    "pattern_gamma_theta",
    "pattern_spread_quality",
    "move1_gamma_theta",
    "move5_gamma_theta",
    "move1_minus_breakeven_atr",
    "move5_minus_breakeven_atr",
    "gap_minus_breakeven_atr",
    "sigma_delta_atr_capture",
    "sigma_convexity_per_premium",
    "sigma_value_score",
    "pattern_overpay_flag",
)
_APLUS_RELATIVE_QUALITY_FEATURE_NAMES = (
    "side_ask_rank",
    "side_spread_frac_rank",
    "side_theta_burden_rank",
    "side_value_rank",
    "ask_vs_side_median",
    "spread_vs_side_median",
    "theta_burden_vs_side_median",
    "gamma_theta_vs_side_median",
    "value_minus_side_median",
    "ask_vs_neighbor_median",
    "spread_vs_neighbor_median",
    "theta_burden_vs_neighbor_median",
    "gamma_theta_vs_neighbor_median",
    "value_minus_neighbor_median",
    "local_premium_curvature",
    "local_gamma_curvature",
    "relative_overpay_score",
)
_POLICY_HOLD_MINUTES = {0: 10.0, 1: 25.0, 2: 45.0}


@dataclass(frozen=True)
class GeneralizationWindow:
    """Stable date window identity for protocol reports and seeds."""

    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    selection_start: str
    selection_end: str
    audit_start: str
    audit_end: str

    @property
    def window_id(self) -> str:
        payload = "|".join(asdict(self).values())
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def window_seed(base_seed: int, window_id: str) -> int:
    """Derive a stable seed from a window id, not from an ordinal."""

    return (int(window_id[:8], 16) + base_seed) & 0x7FFFFFFF


@dataclass(frozen=True)
class SurfaceVariant:
    """One pre-registered Protocol 002 model/feature variant."""

    name: str
    action_space: str
    market_mode: str
    token_mode: str
    loss_mode: str

    @property
    def variant_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def registered_surface_variants() -> tuple[SurfaceVariant, ...]:
    """Fixed variant surface for the hypothesis screen.

    The variants intentionally isolate the highest-value hypotheses:
    ATM vs full surface, current vs v2/v3 market structure, option pressure
    features, and side/flat-aware loss.
    """

    return (
        SurfaceVariant(
            name="atm_current_huber",
            action_space="atm",
            market_mode="current",
            token_mode="base",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="atm_structure_pressure_huber",
            action_space="atm",
            market_mode="structure",
            token_mode="pressure",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_current_huber",
            action_space="surface",
            market_mode="current",
            token_mode="base",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_structure_base_huber",
            action_space="surface",
            market_mode="structure",
            token_mode="base",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_structure_pressure_huber",
            action_space="surface",
            market_mode="structure",
            token_mode="pressure",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_structure_pressure_sidecontrast",
            action_space="surface",
            market_mode="structure",
            token_mode="pressure",
            loss_mode="decision_side",
        ),
    )


def registered_aplus_surface_variants() -> tuple[SurfaceVariant, ...]:
    """Fixed Protocol 003 variants centered on pattern + contract value.

    These variants intentionally keep the same action surface and trial grid as
    Protocol 002. The new variable is whether A+ token features and the
    auxiliary pattern/value losses improve survival on frozen March/Q4 audits.
    """

    return (
        SurfaceVariant(
            name="surface_structure_base_huber",
            action_space="surface",
            market_mode="structure",
            token_mode="base",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_structure_pressure_sidecontrast",
            action_space="surface",
            market_mode="structure",
            token_mode="pressure",
            loss_mode="decision_side",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_huber",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="huber",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_teacher_margin",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_teacher_margin",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_side_quality_margin",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_side_quality_margin",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_side_value_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_side_value_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_side_value_rank",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_side_value_rank",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_balanced_value_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_balanced_value_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_interactions_side_value_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus_interactions",
            loss_mode="aplus_side_value_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_interactions_balanced_value_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus_interactions",
            loss_mode="aplus_balanced_value_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_relative_quality_side_value_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus_relative_quality",
            loss_mode="aplus_side_value_multitask",
        ),
        SurfaceVariant(
            name="surface_structure_aplus_soft_quality_confidence",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_soft_quality_confidence",
        ),
    )


@dataclass
class SurfaceDecision:
    """One minute-level decision with flat plus contract-token actions."""

    session: str
    decision_time: datetime
    scalar_features: np.ndarray
    token_features: np.ndarray
    token_mask: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    rights: np.ndarray
    contract_ids: np.ndarray
    market_last: np.ndarray
    pattern_targets: np.ndarray | None = None
    value_targets: np.ndarray | None = None
    quality_targets: np.ndarray | None = None


@dataclass
class SurfaceStandardizer:
    """Independent robust scalers for scalar and valid token features."""

    scalar: FeatureScaler
    token: FeatureScaler

    @classmethod
    def fit(cls, decisions: Sequence[SurfaceDecision]) -> "SurfaceStandardizer":
        scalar = np.vstack([d.scalar_features for d in decisions]).astype(np.float32)
        token_rows = []
        for decision in decisions:
            if decision.token_mask.any():
                token_rows.append(decision.token_features[decision.token_mask])
        if token_rows:
            token = np.vstack(token_rows).astype(np.float32)
        else:
            token = np.zeros((1, decisions[0].token_features.shape[-1]), dtype=np.float32)
        return cls(scalar=FeatureScaler.fit(scalar), token=FeatureScaler.fit(token))

    def transform(self, decisions: Sequence[SurfaceDecision]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scalar = self.scalar.transform(
            np.vstack([d.scalar_features for d in decisions]).astype(np.float32)
        )
        token_raw = np.stack([d.token_features for d in decisions]).astype(np.float32)
        flat = token_raw.reshape(-1, token_raw.shape[-1])
        token = self.token.transform(flat).reshape(token_raw.shape)
        mask = np.stack([d.token_mask for d in decisions]).astype(bool)
        token *= mask[..., None].astype(np.float32)
        return scalar, token, mask

    def to_dict(self) -> dict:
        return {"scalar": self.scalar.to_dict(), "token": self.token.to_dict()}


class MarketStructureCache:
    """Causal SPX/VIX market-structure lookup.

    The default keeps the historical v2 cache path used by prior protocols.
    Promotion-grade reruns can set ``source="index_bars"`` with SPX/VIX
    one-minute files to avoid relying on the v2 cache for index context.
    """

    def __init__(
        self,
        *,
        spx_path: Path | None = None,
        spy_path: Path | None = None,
        vix_path: Path | None = None,
        source: str = "v2_cache",
        index_spx_dir: Path | None = None,
        index_vix_dir: Path | None = None,
        es_vwap_dir: Path | None = None,
        decision_context_lag_minutes: int = 0,
    ) -> None:
        self.source = source
        self.decision_context_lag_minutes = int(decision_context_lag_minutes)
        data_dir = Path.home() / ".cache/autoresearch-trading/data"
        self.spx_path = spx_path or data_dir / "spx_1min.pkl"
        self.spy_path = spy_path or data_dir / "spy_1min.pkl"
        self.vix_path = vix_path or data_dir / "vix_1min.pkl"
        self.index_spx_dir = index_spx_dir
        self.index_vix_dir = index_vix_dir
        self.es_vwap_dir = es_vwap_dir
        if source == "v2_cache":
            self.spx_by_day = market_structure.build_spx_bars(self.spx_path)
            self.spy_vwap_by_day = market_structure.build_spy_vwap(self.spy_path)
            self.vix_by_day = self._load_vix(self.vix_path)
            self.cache_id = f"v2_cache:{self.spx_path}:{self.spy_path}:{self.vix_path}"
        elif source == "index_bars":
            if index_spx_dir is None or index_vix_dir is None:
                raise ValueError("index_bars market structure requires index_spx_dir and index_vix_dir")
            self.spx_by_day, self.spy_vwap_by_day = self._load_spx_index_dir(index_spx_dir)
            self.vix_by_day = self._load_vix_index_dir(index_vix_dir)
            self.cache_id = f"index_bars:{index_spx_dir.resolve()}:{index_vix_dir.resolve()}"
        else:
            raise ValueError(f"unknown market structure source: {source}")
        if es_vwap_dir is not None:
            self.spy_vwap_by_day = self._load_es_vwap_dir(es_vwap_dir)
            self.cache_id = f"{self.cache_id}:es_vwap:{es_vwap_dir.resolve()}"
        self.omar_by_day = market_structure.build_omar_map(self.spx_by_day)
        self.prev_close_by_day = self._prev_closes()

    @staticmethod
    def _load_vix(path: Path) -> dict[str, np.ndarray]:
        frame = pickle.load(open(path, "rb"))
        out = {}
        for day, group in frame.groupby("date"):
            out[str(day)] = group.reset_index(drop=True)["vix_close"].to_numpy(dtype=float)
        return out

    @staticmethod
    def _index_files(directory: Path) -> list[Path]:
        suffixes = {".parquet", ".csv", ".jsonl", ".txt"}
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)

    @staticmethod
    def _rth_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
        local = out["event_time"].dt.tz_convert(_NY)
        minutes = local.dt.hour * 60 + local.dt.minute
        out = out[(minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)].copy()
        out["session"] = local.dt.date.astype(str)
        return out.sort_values("event_time").reset_index(drop=True)

    @staticmethod
    def _regularize_rth_group(group: pd.DataFrame) -> pd.DataFrame:
        """Put sparse official index prints on a minute-offset grid.

        MarketStructureCache features index arrays by minute since 09:30 ET.
        Vendor index bars can miss minutes, so arrays must be reindexed before
        minute-offset lookup. Missing values are forward-filled causally; leading
        values remain NaN and are later converted to neutral feature values.
        """
        if group.empty:
            return group
        day = str(group["session"].iloc[0])
        local_start = pd.Timestamp(f"{day} 09:30", tz=_NY)
        local_end = pd.Timestamp(f"{day} 16:00", tz=_NY)
        minute_index = pd.date_range(
            local_start.tz_convert("UTC"),
            local_end.tz_convert("UTC"),
            freq="min",
        )
        working = group.drop_duplicates("event_time").set_index("event_time").sort_index()
        regular = working.reindex(minute_index)
        regular["session"] = day
        for col in ("open", "high", "low", "close", "volume"):
            if col in regular:
                regular[col] = pd.to_numeric(regular[col], errors="coerce")
        for col in ("open", "high", "low", "close"):
            if col in regular:
                regular[col] = regular[col].ffill()
        if "volume" in regular:
            regular["volume"] = regular["volume"].fillna(0.0)
        regular = regular.reset_index(names="event_time")
        return regular

    @classmethod
    def _load_spx_index_dir(cls, directory: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        frames = [load_spx_1m(path) for path in cls._index_files(directory)]
        if not frames:
            raise FileNotFoundError(f"no SPX index-bar files found under {directory}")
        frame = cls._rth_frame(pd.concat(frames, ignore_index=True))
        spx_by_day: dict[str, dict[str, np.ndarray]] = {}
        vwap_by_day: dict[str, dict[str, np.ndarray]] = {}
        for day, group in frame.groupby("session"):
            group = cls._regularize_rth_group(group).reset_index(drop=True)
            close = group["close"].to_numpy(dtype=float)
            high = group.get("high", group["close"]).to_numpy(dtype=float)
            low = group.get("low", group["close"]).to_numpy(dtype=float)
            volume = group.get("volume", pd.Series(np.zeros(len(group)))).to_numpy(dtype=float)
            if np.nansum(volume) > 0:
                cum_v = np.maximum(np.nancumsum(volume), 1.0)
                vwap = np.nancumsum(close * volume) / cum_v
                sq_dev = (close - vwap) ** 2
                std = np.sqrt(np.maximum(np.nancumsum(volume * sq_dev) / cum_v, 1e-12))
            else:
                count = np.arange(1, len(close) + 1, dtype=float)
                vwap = np.nancumsum(close) / count
                std = np.asarray(
                    [np.nanstd(close[: idx + 1]) for idx in range(len(close))],
                    dtype=float,
                )
                std = np.maximum(std, 1e-6)
            spx_by_day[str(day)] = {"high": high, "low": low, "close": close}
            vwap_by_day[str(day)] = {"vwap": vwap, "close": close, "std": std}
        return spx_by_day, vwap_by_day

    @classmethod
    def _load_vix_index_dir(cls, directory: Path) -> dict[str, np.ndarray]:
        frames = [load_vix_1m(path) for path in cls._index_files(directory)]
        if not frames:
            raise FileNotFoundError(f"no VIX index-bar files found under {directory}")
        frame = cls._rth_frame(pd.concat(frames, ignore_index=True))
        return {
            str(day): cls._regularize_rth_group(group).reset_index(drop=True)["close"].to_numpy(dtype=float)
            for day, group in frame.groupby("session")
        }

    @classmethod
    def _load_es_vwap_dir(cls, directory: Path) -> dict[str, dict[str, np.ndarray]]:
        frames = []
        for path in cls._index_files(directory):
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix.lower() in {".csv", ".txt"}:
                frame = pd.read_csv(path)
            elif path.suffix.lower() == ".jsonl":
                frame = pd.read_json(path, lines=True)
            else:
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"no ES VWAP files found under {directory}")
        frame = cls._rth_frame(pd.concat(frames, ignore_index=True))
        out: dict[str, dict[str, np.ndarray]] = {}
        for day, group in frame.groupby("session"):
            group = group.reset_index(drop=True)
            close = pd.to_numeric(group["close"], errors="coerce").to_numpy(dtype=float)
            high = pd.to_numeric(group.get("high", group["close"]), errors="coerce").to_numpy(dtype=float)
            low = pd.to_numeric(group.get("low", group["close"]), errors="coerce").to_numpy(dtype=float)
            volume = pd.to_numeric(group.get("volume", pd.Series(np.zeros(len(group)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            typical = (high + low + close) / 3.0
            cum_v = np.maximum(np.nancumsum(volume), 1.0)
            vwap = np.nancumsum(typical * volume) / cum_v
            zero_volume = cum_v <= 1.0
            if np.any(zero_volume):
                count = np.arange(1, len(close) + 1, dtype=float)
                fallback = np.nancumsum(close) / count
                vwap = np.where(zero_volume, fallback, vwap)
            sq_dev = (close - vwap) ** 2
            std = np.sqrt(np.maximum(np.nancumsum(volume * sq_dev) / cum_v, 1e-12))
            out[str(day)] = {"vwap": vwap, "close": close, "std": np.maximum(std, 1e-6)}
        return out

    def _prev_closes(self) -> dict[str, float]:
        days = sorted(self.spx_by_day)
        out = {}
        prev_close = math.nan
        for day in days:
            out[day] = prev_close
            closes = self.spx_by_day[day]["close"]
            if len(closes):
                prev_close = float(closes[-1])
        return out

    @staticmethod
    def minute_of_session(decision_time: datetime) -> int:
        local = decision_time.astimezone(_NY)
        return (local.hour * 60 + local.minute) - (9 * 60 + 30)

    @staticmethod
    def session_string(decision_time: datetime) -> str:
        return decision_time.astimezone(_NY).date().isoformat()

    def features_for(self, decision_time: datetime) -> np.ndarray:
        semantic_decision_time = decision_time
        if self.decision_context_lag_minutes:
            decision_time = decision_time - timedelta(minutes=self.decision_context_lag_minutes)
        day = self.session_string(decision_time)
        minute = self.minute_of_session(decision_time)
        spx_day = self.spx_by_day.get(day)
        spy_day = self.spy_vwap_by_day.get(day)
        vix_day = self.vix_by_day.get(day)
        omar = self.omar_by_day.get(day)
        if spx_day is None or spy_day is None or omar is None or minute < 0:
            return np.zeros(len(_STRUCTURE_FEATURE_NAMES), dtype=np.float32)

        close_arr = spx_day["close"]
        high_arr = spx_day["high"]
        low_arr = spx_day["low"]
        minute = min(minute, len(close_arr) - 1)
        if minute < 0:
            return np.zeros(len(_STRUCTURE_FEATURE_NAMES), dtype=np.float32)

        close = float(close_arr[minute])
        vix = float(vix_day[min(minute, len(vix_day) - 1)]) if vix_day is not None and len(vix_day) else 0.0
        sigma = market_structure.sigma_pos(spy_day, minute, close)
        sigma_value = 0.0 if sigma is None else float(sigma)

        spy_close = float(spy_day["close"][minute]) if minute < len(spy_day["close"]) else 0.0
        spy_vwap = float(spy_day["vwap"][minute]) if minute < len(spy_day["vwap"]) else 0.0
        ratio = close / spy_close if spy_close > 0 else 1.0
        spx_vwap = spy_vwap * ratio
        vwap_dist_pct = (close - spx_vwap) / max(abs(close), 1.0)
        if minute >= 5 and minute < len(spy_day["vwap"]):
            prior_spy_close = float(spy_day["close"][minute - 5])
            prior_ratio = float(close_arr[minute - 5]) / prior_spy_close if prior_spy_close > 0 else ratio
            prior_vwap = float(spy_day["vwap"][minute - 5]) * prior_ratio
            vwap_slope = (spx_vwap - prior_vwap) / max(abs(close), 1.0)
        else:
            vwap_slope = 0.0

        omar_range = max(float(omar["range"]), 0.01)
        omar_mid = float(omar["mid"])
        nearest_omar = min(
            abs(close - float(omar["high"])),
            abs(close - float(omar["low"])),
            abs(close - omar_mid),
        )

        first15_minute = 14 if self.decision_context_lag_minutes else 15
        first15_available = minute >= first15_minute and len(high_arr) >= 15
        if first15_available:
            first15_high = float(np.nanmax(high_arr[:15]))
            first15_low = float(np.nanmin(low_arr[:15]))
            first15_close = float(close_arr[14])
            first15_range = max(first15_high - first15_low, 0.01)
            first15_close_pos = (first15_close - first15_low) / first15_range
            first15_accept = np.clip((close - ((first15_high + first15_low) / 2.0)) / (first15_range / 2.0), -1.0, 1.0)
            inside_first15 = float(first15_low <= close <= first15_high)
            first15_range_pct = first15_range / max(abs(close), 1.0)
        else:
            first15_range_pct = 0.0
            first15_close_pos = 0.5
            first15_accept = 0.0
            inside_first15 = 0.0

        prev_close = self.prev_close_by_day.get(day, math.nan)
        open_price = float(close_arr[0]) if len(close_arr) else close
        opening_gap_pct = (open_price - prev_close) / prev_close if np.isfinite(prev_close) and prev_close > 0 else 0.0

        last10 = market_structure.last10(spx_day, minute)
        if last10 is None:
            last10_range_over_omar = 0.0
        else:
            last10_range_over_omar = float(last10["range"]) / omar_range
        last10_break = market_structure.last10_break_state(spx_day, minute, close)
        last10_break_state = 0.0 if last10_break is None else float(last10_break)

        atr_start = max(0, minute - 14)
        atr_range = np.nanmean(high_arr[atr_start : minute + 1] - low_arr[atr_start : minute + 1])
        atr15_pct = float(atr_range / max(abs(close), 1.0)) if np.isfinite(atr_range) else 0.0
        # Lag market observations, not decision-clock semantics.  Otherwise the
        # historical path reports the prior bucket at 10:00, 11:30, and 13:30
        # while live replay correctly uses the bucket of the actual decision.
        bucket = time_bucket(semantic_decision_time)
        buckets = [
            float(bucket == "first_30"),
            float(bucket == "post_open_morning"),
            float(bucket == "midday"),
            float(bucket == "late_afternoon"),
        ]
        out = [
            close,
            vix,
            sigma_value,
            abs(sigma_value),
            spx_vwap,
            vwap_dist_pct,
            vwap_slope,
            float(omar["high"]),
            float(omar["low"]),
            omar_mid,
            omar_range,
            omar_range / max(abs(close), 1.0),
            (close - omar_mid) / omar_range,
            nearest_omar / omar_range,
            float(first15_available),
            first15_range_pct,
            first15_close_pos,
            first15_accept,
            inside_first15,
            opening_gap_pct,
            last10_range_over_omar,
            last10_break_state,
            atr15_pct,
            minute / 390.0,
            *buckets,
        ]
        return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=8.0, neginf=-8.0)


def _safe_log1p(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(np.log1p(value))


def _finite(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def _side_value(value: float, side: str) -> float:
    direction = 1.0 if side == "C" else -1.0
    return direction * value if np.isfinite(value) else 0.0


def _base_scalar_features(row: dict) -> np.ndarray:
    market_window = np.asarray(row["market_window"], dtype=np.float32)
    market_last = market_window[-1]
    market_mean = np.nanmean(market_window, axis=0)
    market_std = np.nanstd(market_window, axis=0)
    market_delta = market_last - market_window[0]
    return np.concatenate([market_last, market_mean, market_std, market_delta]).astype(np.float32)


def _aplus_pattern_values(
    market_window: np.ndarray,
    structure: np.ndarray,
    side: str,
) -> dict[str, float | bool]:
    """Side-specific, causal timing primitives used by the A+ neural variants."""
    close = np.asarray(market_window[:, 0], dtype=float)
    vwap = np.asarray(market_window[:, 2], dtype=float)
    if len(close) < 2:
        return {name: False for name in _APLUS_PATTERN_NAMES}

    current = float(close[-1])
    prev = float(close[-2])
    gap = close - vwap
    current_gap = float(gap[-1])
    prev_gap = float(gap[-2]) if len(gap) >= 2 else 0.0
    side_gap = _side_value(current_gap, side)
    side_prev_gap = _side_value(prev_gap, side)
    prior5_gap = gap[-6:-1] if len(gap) >= 6 else gap[:-1]
    prior5_close = close[-6:-1] if len(close) >= 6 else close[:-1]

    move1 = current - prev
    move3 = current - float(close[-4]) if len(close) >= 4 else move1
    move5 = current - float(close[-6]) if len(close) >= 6 else move3
    move15 = current - float(close[-16]) if len(close) >= 16 else move5
    side_move1 = _side_value(move1, side)
    side_move3 = _side_value(move3, side)
    side_move5 = _side_value(move5, side)
    side_move15 = _side_value(move15, side)

    last5_range = float(np.nanmax(close[-5:]) - np.nanmin(close[-5:])) if len(close) >= 5 else 0.0
    last15_range = float(np.nanmax(close[-15:]) - np.nanmin(close[-15:])) if len(close) >= 15 else max(last5_range, 0.01)
    compression_ratio = last5_range / max(last15_range, 0.01)

    sigma_pos = _finite(structure[2], 0.0) if len(structure) > 2 else 0.0
    omar_mid = _finite(structure[9], current) if len(structure) > 9 else current
    omar_retest_dist = _finite(structure[13], 9.0) if len(structure) > 13 else 9.0
    first15_acceptance = _finite(structure[17], 0.0) if len(structure) > 17 else 0.0
    inside_first15 = _finite(structure[18], 0.0) if len(structure) > 18 else 0.0
    last10_break_state = _finite(structure[21], 0.0) if len(structure) > 21 else 0.0
    atr15_points = max((_finite(structure[22], 0.0) if len(structure) > 22 else 0.0) * max(abs(current), 1.0), 0.5)

    side_sigma = _side_value(sigma_pos, side)
    side_last10_break = _side_value(last10_break_state, side)
    side_first15 = _side_value(first15_acceptance, side)
    side_omar_now = _side_value(current - omar_mid, side)
    side_omar_prev = _side_value(prev - omar_mid, side)
    prior_side_gaps = np.asarray([_side_value(x, side) for x in prior5_gap], dtype=float)
    prior_counter_or_near_vwap = bool(len(prior_side_gaps) and np.nanmin(prior_side_gaps) <= 0.75)
    prior_aligned_vwap = bool(len(prior_side_gaps) and np.nanmin(prior_side_gaps) > 0.75)
    finite_prior5 = prior5_close[np.isfinite(prior5_close)]
    if len(finite_prior5):
        pullback_reference = float(np.nanmax(finite_prior5) if side == "P" else np.nanmin(finite_prior5))
        prior_counter_move = _side_value(current - pullback_reference, side) > 0
    else:
        prior_counter_move = False
    prior15_compressed = compression_ratio <= 0.45 or last5_range <= 0.85 * atr15_points

    out: dict[str, float | bool] = {
        "side_gap_atr": side_gap / atr15_points,
        "side_sigma": side_sigma,
        "side_move1_atr": side_move1 / atr15_points,
        "side_move5_atr": side_move5 / atr15_points,
        "side_move15_atr": side_move15 / atr15_points,
        "compression_ratio": compression_ratio,
        "sigma_trend_continuation": bool(side_sigma > 0.25 and side_move5 > 1.0 and side_move15 > 2.0),
        "vwap_reclaim": bool(side_gap > 0.75 and side_prev_gap <= 0.0 and side_move1 > 0.5),
        "vwap_hold_continuation": bool(side_sigma > 0.25 and prior_aligned_vwap and side_move5 > 0.5),
        "vwap_pullback_resume": bool(side_sigma > 0.25 and side_gap > 0.75 and prior_counter_or_near_vwap and side_move1 > 0.5),
        "last10_breakout": bool(side_last10_break > 0.5 and side_move1 > 0.5),
        "compression_breakout": bool(side_last10_break > 0.5 and prior15_compressed and side_move1 > 0.5),
        "momentum_ignition": bool(side_move1 > 0.75 and side_move3 > 1.5 and side_move5 > 2.0),
        "pullback_resume": bool(side_sigma > 0.25 and prior_counter_move and side_move1 > 0.5),
        "omar_mid_reclaim": bool(side_omar_now > 0.0 and side_omar_prev <= 0.0 and side_move1 > 0.5),
        "omar_retest_bounce": bool(side_sigma > 0.25 and omar_retest_dist <= 0.35 and side_move1 > 0.5),
        "first15_acceptance_break": bool(side_first15 > 0.65 and side_move1 > 0.5),
        "first15_inside_reversal": bool(inside_first15 >= 0.5 and side_first15 > 0.15 and side_move1 > 0.5),
    }
    return out


def _aplus_value_values(
    base: np.ndarray,
    *,
    policy_index: int,
    structure: np.ndarray,
) -> dict[str, float]:
    ask = max(_finite(base[_OPTION_INDEX["ask"]], 0.0), 0.01)
    spread = max(_finite(base[_OPTION_INDEX["spread"]], 0.0), 0.0)
    abs_delta = abs(_finite(base[_OPTION_INDEX["delta"]], 0.0))
    gamma_abs = abs(_finite(base[_OPTION_INDEX["gamma"]], 0.0))
    theta_abs = abs(_finite(base[_OPTION_INDEX["theta"]], 0.0))
    breakeven = _finite(base[_OPTION_INDEX["breakeven_distance"]], 99.0)
    spx = max(abs(_finite(structure[0], 0.0) if len(structure) else 0.0), 1.0)
    atr_points = max(abs(_finite(structure[22], 0.0) if len(structure) > 22 else 0.0) * spx, 0.5)
    hold_minutes = _POLICY_HOLD_MINUTES.get(int(policy_index), 25.0)

    spread_tax = spread / ask
    theta_burden_hold = (theta_abs * (hold_minutes / 390.0)) / ask
    convexity_per_premium = (0.5 * gamma_abs * (atr_points**2)) / ask
    delta_atr_capture = (abs_delta * atr_points) / ask
    breakeven_atr = breakeven / atr_points
    gamma_theta_ratio = gamma_abs / (theta_abs / 390.0 + 1e-6)
    cost = 1.4 * min(spread_tax, 1.0) + 1.2 * min(theta_burden_hold, 1.0) + 0.25 * min(max(breakeven_atr, 0.0), 6.0)
    benefit = 0.65 * min(delta_atr_capture, 4.0) + 0.85 * min(convexity_per_premium, 4.0) + 0.15 * min(gamma_theta_ratio, 20.0) / 20.0
    score = benefit - cost
    obvious_overpay = float(spread_tax > 0.14 or theta_burden_hold > 0.20 or breakeven_atr > 2.50)
    worth_spread = float(
        abs_delta >= 0.15
        and spread_tax <= 0.14
        and theta_burden_hold <= 0.20
        and breakeven_atr <= 2.50
        and score >= -1.25
    )
    return {
        "abs_delta": abs_delta,
        "delta_atr_capture": delta_atr_capture,
        "convexity_per_premium": convexity_per_premium,
        "theta_burden_hold": theta_burden_hold,
        "spread_tax": spread_tax,
        "breakeven_atr": breakeven_atr,
        "gamma_theta_ratio_scaled": min(gamma_theta_ratio, 20.0) / 20.0,
        "contract_value_score": score,
        "worth_spread_flag": worth_spread,
        "obvious_overpay_flag": obvious_overpay,
    }


def _quality_target_from_value_score(score: float) -> float:
    """Soft contract-quality target; never a hard trade/no-trade label."""
    clean = np.clip(_finite(score, 0.0), -4.0, 4.0)
    return float(1.0 / (1.0 + np.exp(-1.4 * (clean + 0.35))))


def _rank_fraction(values: np.ndarray, idx: int) -> float:
    clean = values[np.isfinite(values)]
    current = float(values[idx]) if 0 <= idx < len(values) and np.isfinite(values[idx]) else math.nan
    if len(clean) <= 1 or not np.isfinite(current):
        return 0.5
    return float(np.mean(clean <= current))


def _relative_quality_features(
    *,
    option: np.ndarray,
    right_idx: int,
    strike_idx: int,
    policy_index: int,
    structure: np.ndarray,
) -> np.ndarray:
    side = np.asarray(option[:, right_idx, :], dtype=np.float32)
    ask = np.asarray(side[:, _OPTION_INDEX["ask"]], dtype=float)
    mid = np.asarray(side[:, _OPTION_INDEX["mid"]], dtype=float)
    spread_frac = np.asarray(side[:, _OPTION_INDEX["spread_frac"]], dtype=float)
    gamma_abs = np.abs(np.asarray(side[:, _OPTION_INDEX["gamma"]], dtype=float))
    theta_abs = np.abs(np.asarray(side[:, _OPTION_INDEX["theta"]], dtype=float))
    value_rows = [
        _aplus_value_values(side[i], policy_index=policy_index, structure=structure)
        for i in range(side.shape[0])
    ]
    theta_burden = np.asarray([row["theta_burden_hold"] for row in value_rows], dtype=float)
    gamma_theta = np.asarray([row["gamma_theta_ratio_scaled"] for row in value_rows], dtype=float)
    value_score = np.asarray([row["contract_value_score"] for row in value_rows], dtype=float)
    valid = np.isfinite(ask) & (ask > 0.0) & np.isfinite(mid) & (mid > 0.0)
    if not np.any(valid):
        return np.zeros(len(_APLUS_RELATIVE_QUALITY_FEATURE_NAMES), dtype=np.float32)

    def side_median(values: np.ndarray, default: float = 1.0) -> float:
        clean = values[valid & np.isfinite(values)]
        return float(np.median(clean)) if len(clean) else default

    current_ask = max(_finite(ask[strike_idx], 0.0), 0.01)
    current_spread = max(_finite(spread_frac[strike_idx], 0.0), 0.0)
    current_theta_burden = max(_finite(theta_burden[strike_idx], 0.0), 0.0)
    current_gamma_theta = max(_finite(gamma_theta[strike_idx], 0.0), 0.0)
    current_value = _finite(value_score[strike_idx], 0.0)
    median_ask = max(side_median(ask, current_ask), 0.01)
    median_spread = max(side_median(spread_frac, current_spread), 1e-6)
    median_theta_burden = max(side_median(theta_burden, current_theta_burden), 1e-6)
    median_gamma_theta = max(side_median(gamma_theta, current_gamma_theta), 1e-6)
    median_value = side_median(value_score, current_value)

    neighbor_idx = [idx for idx in (strike_idx - 1, strike_idx + 1) if 0 <= idx < len(ask) and valid[idx]]

    def neighbor_median(values: np.ndarray, default: float) -> float:
        clean = np.asarray([values[idx] for idx in neighbor_idx if np.isfinite(values[idx])], dtype=float)
        return float(np.median(clean)) if len(clean) else default

    neighbor_ask = max(neighbor_median(ask, median_ask), 0.01)
    neighbor_spread = max(neighbor_median(spread_frac, median_spread), 1e-6)
    neighbor_theta_burden = max(neighbor_median(theta_burden, median_theta_burden), 1e-6)
    neighbor_gamma_theta = max(neighbor_median(gamma_theta, median_gamma_theta), 1e-6)
    neighbor_value = neighbor_median(value_score, median_value)

    if strike_idx > 0 and strike_idx + 1 < len(mid) and np.isfinite(mid[strike_idx - 1]) and np.isfinite(mid[strike_idx + 1]):
        local_premium_curvature = (float(mid[strike_idx - 1]) - 2.0 * float(mid[strike_idx]) + float(mid[strike_idx + 1])) / max(abs(float(mid[strike_idx])), 0.01)
    else:
        local_premium_curvature = 0.0
    if strike_idx > 0 and strike_idx + 1 < len(gamma_abs) and np.isfinite(gamma_abs[strike_idx - 1]) and np.isfinite(gamma_abs[strike_idx + 1]):
        local_gamma_curvature = (float(gamma_abs[strike_idx - 1]) - 2.0 * float(gamma_abs[strike_idx]) + float(gamma_abs[strike_idx + 1])) / max(abs(float(gamma_abs[strike_idx])), 1e-6)
    else:
        local_gamma_curvature = 0.0

    # High means the contract looks expensive relative to its same-side ladder.
    relative_overpay = (
        0.45 * (current_ask / median_ask - 1.0)
        + 0.70 * (current_spread / median_spread - 1.0)
        + 0.85 * (current_theta_burden / median_theta_burden - 1.0)
        - 0.65 * (current_value - median_value)
        - 0.35 * (current_gamma_theta / median_gamma_theta - 1.0)
    )
    out = np.asarray(
        [
            _rank_fraction(ask, strike_idx),
            _rank_fraction(spread_frac, strike_idx),
            _rank_fraction(theta_burden, strike_idx),
            _rank_fraction(value_score, strike_idx),
            current_ask / median_ask,
            current_spread / median_spread,
            current_theta_burden / median_theta_burden,
            current_gamma_theta / median_gamma_theta,
            current_value - median_value,
            current_ask / neighbor_ask,
            current_spread / neighbor_spread,
            current_theta_burden / neighbor_theta_burden,
            current_gamma_theta / neighbor_gamma_theta,
            current_value - neighbor_value,
            local_premium_curvature,
            local_gamma_curvature,
            relative_overpay,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.clip(out, -9.0, 9.0), nan=0.0, posinf=9.0, neginf=-9.0)


def _relative_quality_matrix(
    *,
    option: np.ndarray,
    policy_index: int,
    structure: np.ndarray,
) -> np.ndarray:
    out = np.zeros(
        (option.shape[0], option.shape[1], len(_APLUS_RELATIVE_QUALITY_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for right_idx in range(option.shape[1]):
        side = np.asarray(option[:, right_idx, :], dtype=np.float32)
        ask = np.asarray(side[:, _OPTION_INDEX["ask"]], dtype=float)
        mid = np.asarray(side[:, _OPTION_INDEX["mid"]], dtype=float)
        spread_frac = np.asarray(side[:, _OPTION_INDEX["spread_frac"]], dtype=float)
        gamma_abs = np.abs(np.asarray(side[:, _OPTION_INDEX["gamma"]], dtype=float))
        value_rows = [
            _aplus_value_values(side[i], policy_index=policy_index, structure=structure)
            for i in range(side.shape[0])
        ]
        theta_burden = np.asarray([row["theta_burden_hold"] for row in value_rows], dtype=float)
        gamma_theta = np.asarray([row["gamma_theta_ratio_scaled"] for row in value_rows], dtype=float)
        value_score = np.asarray([row["contract_value_score"] for row in value_rows], dtype=float)
        valid = np.isfinite(ask) & (ask > 0.0) & np.isfinite(mid) & (mid > 0.0)
        if not np.any(valid):
            continue

        def side_median(values: np.ndarray, default: float = 1.0) -> float:
            clean = values[valid & np.isfinite(values)]
            return float(np.median(clean)) if len(clean) else default

        median_ask = max(side_median(ask), 0.01)
        median_spread = max(side_median(spread_frac), 1e-6)
        median_theta_burden = max(side_median(theta_burden), 1e-6)
        median_gamma_theta = max(side_median(gamma_theta), 1e-6)
        median_value = side_median(value_score, 0.0)

        for strike_idx in range(option.shape[0]):
            current_ask = max(_finite(ask[strike_idx], 0.0), 0.01)
            current_spread = max(_finite(spread_frac[strike_idx], 0.0), 0.0)
            current_theta_burden = max(_finite(theta_burden[strike_idx], 0.0), 0.0)
            current_gamma_theta = max(_finite(gamma_theta[strike_idx], 0.0), 0.0)
            current_value = _finite(value_score[strike_idx], 0.0)
            neighbor_idx = [idx for idx in (strike_idx - 1, strike_idx + 1) if 0 <= idx < len(ask) and valid[idx]]

            def neighbor_median(values: np.ndarray, default: float) -> float:
                clean = np.asarray([values[idx] for idx in neighbor_idx if np.isfinite(values[idx])], dtype=float)
                return float(np.median(clean)) if len(clean) else default

            neighbor_ask = max(neighbor_median(ask, median_ask), 0.01)
            neighbor_spread = max(neighbor_median(spread_frac, median_spread), 1e-6)
            neighbor_theta_burden = max(neighbor_median(theta_burden, median_theta_burden), 1e-6)
            neighbor_gamma_theta = max(neighbor_median(gamma_theta, median_gamma_theta), 1e-6)
            neighbor_value = neighbor_median(value_score, median_value)

            if strike_idx > 0 and strike_idx + 1 < len(mid) and np.isfinite(mid[strike_idx - 1]) and np.isfinite(mid[strike_idx + 1]):
                premium_curvature = (float(mid[strike_idx - 1]) - 2.0 * float(mid[strike_idx]) + float(mid[strike_idx + 1])) / max(abs(float(mid[strike_idx])), 0.01)
            else:
                premium_curvature = 0.0
            if strike_idx > 0 and strike_idx + 1 < len(gamma_abs) and np.isfinite(gamma_abs[strike_idx - 1]) and np.isfinite(gamma_abs[strike_idx + 1]):
                gamma_curvature = (float(gamma_abs[strike_idx - 1]) - 2.0 * float(gamma_abs[strike_idx]) + float(gamma_abs[strike_idx + 1])) / max(abs(float(gamma_abs[strike_idx])), 1e-6)
            else:
                gamma_curvature = 0.0

            relative_overpay = (
                0.45 * (current_ask / median_ask - 1.0)
                + 0.70 * (current_spread / median_spread - 1.0)
                + 0.85 * (current_theta_burden / median_theta_burden - 1.0)
                - 0.65 * (current_value - median_value)
                - 0.35 * (current_gamma_theta / median_gamma_theta - 1.0)
            )
            row = np.asarray(
                [
                    _rank_fraction(ask, strike_idx),
                    _rank_fraction(spread_frac, strike_idx),
                    _rank_fraction(theta_burden, strike_idx),
                    _rank_fraction(value_score, strike_idx),
                    current_ask / median_ask,
                    current_spread / median_spread,
                    current_theta_burden / median_theta_burden,
                    current_gamma_theta / median_gamma_theta,
                    current_value - median_value,
                    current_ask / neighbor_ask,
                    current_spread / neighbor_spread,
                    current_theta_burden / neighbor_theta_burden,
                    current_gamma_theta / neighbor_gamma_theta,
                    current_value - neighbor_value,
                    premium_curvature,
                    gamma_curvature,
                    relative_overpay,
                ],
                dtype=np.float32,
            )
            out[strike_idx, right_idx] = np.nan_to_num(
                np.clip(row, -9.0, 9.0),
                nan=0.0,
                posinf=9.0,
                neginf=-9.0,
            )
    return out


def _aplus_token_features_and_targets(
    row: dict,
    *,
    policy_index: int,
    structure: np.ndarray,
    include_interactions: bool = False,
    include_relative_quality: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    option = np.asarray(row["option_ladder"], dtype=np.float32)
    offsets = np.asarray(row["strike_offsets"], dtype=np.float32)
    rights = tuple(row["rights"])
    market_window = np.asarray(row["market_window"], dtype=np.float32)
    relative_quality = (
        _relative_quality_matrix(option=option, policy_index=policy_index, structure=structure)
        if include_relative_quality
        else None
    )
    rows = []
    pattern_targets = []
    value_targets = []
    quality_targets = []
    for strike_idx, offset in enumerate(offsets):
        for right_idx, right in enumerate(rights):
            base = option[strike_idx, right_idx]
            side = np.asarray([1.0 if right == "C" else 0.0, 1.0 if right == "P" else 0.0], dtype=np.float32)
            shape = np.asarray([offset / 50.0, abs(offset) / 50.0], dtype=np.float32)
            pressure = _pressure_features(base)
            patterns = _aplus_pattern_values(market_window, structure, str(right))
            pattern_flags = np.asarray([float(bool(patterns.get(name, False))) for name in _APLUS_PATTERN_NAMES], dtype=np.float32)
            pattern_count = float(pattern_flags.sum())
            pattern_extra = np.asarray(
                [
                    pattern_count / max(len(_APLUS_PATTERN_NAMES), 1),
                    np.clip(_finite(patterns.get("side_gap_atr", 0.0)), -6.0, 6.0),
                    np.clip(_finite(patterns.get("side_sigma", 0.0)), -6.0, 6.0),
                    np.clip(_finite(patterns.get("side_move1_atr", 0.0)), -6.0, 6.0),
                    np.clip(_finite(patterns.get("side_move5_atr", 0.0)), -6.0, 6.0),
                    np.clip(_finite(patterns.get("side_move15_atr", 0.0)), -6.0, 6.0),
                    np.clip(_finite(patterns.get("compression_ratio", 0.0)), 0.0, 6.0),
                ],
                dtype=np.float32,
            )
            values = _aplus_value_values(base, policy_index=policy_index, structure=structure)
            value_vec = np.asarray([np.clip(_finite(values[name]), -9.0, 9.0) for name in _APLUS_VALUE_FEATURE_NAMES], dtype=np.float32)
            interaction_vec = np.zeros(0, dtype=np.float32)
            if include_interactions:
                pattern_count_norm = float(pattern_extra[0])
                side_gap_atr = float(pattern_extra[1])
                side_sigma = float(pattern_extra[2])
                side_move1_atr = float(pattern_extra[3])
                side_move5_atr = float(pattern_extra[4])
                value_score = float(values["contract_value_score"])
                spread_tax = float(values["spread_tax"])
                breakeven_atr = float(values["breakeven_atr"])
                gamma_theta = float(values["gamma_theta_ratio_scaled"])
                delta_capture = float(values["delta_atr_capture"])
                convexity = float(values["convexity_per_premium"])
                overpay = float(values["obvious_overpay_flag"])
                positive_move1 = max(side_move1_atr, 0.0)
                positive_move5 = max(side_move5_atr, 0.0)
                positive_gap = max(side_gap_atr, 0.0)
                positive_sigma = max(side_sigma, 0.0)
                interaction_vec = np.asarray(
                    [
                        pattern_count_norm * value_score,
                        pattern_count_norm * gamma_theta,
                        pattern_count_norm * (1.0 - min(max(spread_tax, 0.0), 1.0)),
                        positive_move1 * gamma_theta,
                        positive_move5 * gamma_theta,
                        positive_move1 - breakeven_atr,
                        positive_move5 - breakeven_atr,
                        positive_gap - breakeven_atr,
                        positive_sigma * delta_capture,
                        positive_sigma * convexity,
                        positive_sigma * value_score,
                        pattern_count_norm * overpay,
                    ],
                    dtype=np.float32,
                )
                interaction_vec = np.clip(interaction_vec, -9.0, 9.0)
            relative_quality_vec = np.zeros(0, dtype=np.float32)
            if include_relative_quality:
                relative_quality_vec = relative_quality[strike_idx, right_idx]
            pattern_present = float(pattern_count > 0.0)
            value_present = float(values["worth_spread_flag"] > 0.0)
            quality_target = _quality_target_from_value_score(float(values["contract_value_score"]))
            rows.append(
                np.concatenate(
                    [
                        base,
                        side,
                        shape,
                        pressure,
                        pattern_flags,
                        pattern_extra,
                        value_vec,
                        interaction_vec,
                        relative_quality_vec,
                    ]
                ).astype(np.float32)
            )
            pattern_targets.append(pattern_present)
            value_targets.append(value_present)
            quality_targets.append(quality_target)
    return (
        np.vstack(rows),
        np.asarray(pattern_targets, dtype=np.float32),
        np.asarray(value_targets, dtype=np.float32),
        np.asarray(quality_targets, dtype=np.float32),
    )


def _pressure_features(base: np.ndarray) -> np.ndarray:
    mid = float(base[_OPTION_INDEX["mid"]])
    spread = float(base[_OPTION_INDEX["spread"]])
    oi = float(base[_OPTION_INDEX["stat_open_interest"]])
    volume = float(base[_OPTION_INDEX["option_ohlcv_volume"]])
    gamma = float(base[_OPTION_INDEX["gamma"]])
    theta = float(base[_OPTION_INDEX["theta"]])
    bid_size = float(base[_OPTION_INDEX["bid_size"]])
    ask_size = float(base[_OPTION_INDEX["ask_size"]])
    gamma_abs = abs(gamma) if np.isfinite(gamma) else 0.0
    theta_abs = abs(theta) if np.isfinite(theta) else 0.0
    return np.asarray(
        [
            _safe_log1p(mid),
            _safe_log1p(oi),
            _safe_log1p(volume),
            gamma_abs,
            theta_abs,
            gamma_abs / max(theta_abs, 1e-6),
            theta_abs / max(mid, 1e-6),
            spread / max(gamma_abs, 1e-6),
            theta_abs * max(mid, 0.0),
            (bid_size + ask_size) / max(spread, 0.01),
        ],
        dtype=np.float32,
    )


def _token_features_and_aux(
    row: dict,
    *,
    token_mode: str,
    policy_index: int,
    structure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if token_mode in {"aplus", "aplus_interactions", "aplus_relative_quality"}:
        return _aplus_token_features_and_targets(
            row,
            policy_index=policy_index,
            structure=structure,
            include_interactions=token_mode in {"aplus_interactions", "aplus_relative_quality"},
            include_relative_quality=token_mode == "aplus_relative_quality",
        )
    option = np.asarray(row["option_ladder"], dtype=np.float32)
    offsets = np.asarray(row["strike_offsets"], dtype=np.float32)
    rights = tuple(row["rights"])
    rows = []
    for strike_idx, offset in enumerate(offsets):
        for right_idx, right in enumerate(rights):
            base = option[strike_idx, right_idx]
            side = np.asarray([1.0 if right == "C" else 0.0, 1.0 if right == "P" else 0.0], dtype=np.float32)
            shape = np.asarray([offset / 50.0, abs(offset) / 50.0], dtype=np.float32)
            features = [base, side, shape]
            if token_mode == "pressure":
                features.append(_pressure_features(base))
            elif token_mode != "base":
                raise ValueError(f"unknown token_mode={token_mode}")
            rows.append(np.concatenate(features).astype(np.float32))
    count = len(rows)
    zeros = np.zeros(count, dtype=np.float32)
    return np.vstack(rows), zeros, zeros, zeros


def _action_mask(row: dict, *, action_space: str, policy_index: int) -> np.ndarray:
    mask = np.asarray(row["candidate_mask"], dtype=bool).reshape(-1)
    if action_space == "surface":
        return mask
    if action_space != "atm":
        raise ValueError(f"unknown action_space={action_space}")
    labels = np.asarray(row["labels_net_pnl"], dtype=np.float32)[:, :, policy_index]
    rights = tuple(row["rights"])
    offsets = np.asarray(row["strike_offsets"], dtype=np.float32)
    keep = np.zeros_like(mask, dtype=bool)
    for right in ("C", "P"):
        if right not in rights:
            continue
        right_idx = rights.index(right)
        valid = np.asarray(row["candidate_mask"], dtype=bool)[:, right_idx] & np.isfinite(labels[:, right_idx])
        if not valid.any():
            continue
        strike_candidates = np.where(valid)[0]
        strike_idx = int(strike_candidates[np.argmin(np.abs(offsets[strike_candidates]))])
        keep[strike_idx * len(rights) + right_idx] = True
    return mask & keep


def surface_decision_from_row(
    *,
    session: str,
    row: dict,
    policy_index: int,
    variant: SurfaceVariant,
    market_cache: MarketStructureCache,
) -> SurfaceDecision:
    scalar_parts = [_base_scalar_features(row)]
    structure = np.zeros(len(_STRUCTURE_FEATURE_NAMES), dtype=np.float32)
    if variant.market_mode == "structure":
        structure = market_cache.features_for(row["decision_time"])
        scalar_parts.append(structure)
    elif variant.market_mode != "current":
        raise ValueError(f"unknown market_mode={variant.market_mode}")
    scalar = np.concatenate(scalar_parts).astype(np.float32)
    if variant.token_mode in {"aplus", "aplus_interactions", "aplus_relative_quality"} and variant.market_mode != "structure":
        structure = market_cache.features_for(row["decision_time"])
    tokens, pattern_targets, value_targets, quality_targets = _token_features_and_aux(
        row,
        token_mode=variant.token_mode,
        policy_index=policy_index,
        structure=structure,
    )
    token_mask = _action_mask(row, action_space=variant.action_space, policy_index=policy_index)
    labels = np.asarray(row["labels_net_pnl"], dtype=np.float32)[:, :, policy_index].reshape(-1)
    labels = np.where(np.isfinite(labels), labels, np.nan).astype(np.float32)
    offsets_grid = np.repeat(np.asarray(row["strike_offsets"], dtype=np.float32), len(row["rights"]))
    rights_grid = np.tile(np.asarray(row["rights"], dtype=object), len(row["strike_offsets"]))
    contract_ids_grid = np.asarray(row["contract_ids"], dtype=object).reshape(-1)
    return SurfaceDecision(
        session=session,
        decision_time=row["decision_time"],
        scalar_features=np.nan_to_num(scalar, nan=0.0, posinf=8.0, neginf=-8.0),
        token_features=np.nan_to_num(tokens, nan=0.0, posinf=8.0, neginf=-8.0),
        token_mask=token_mask & np.isfinite(labels),
        labels=labels,
        offsets=offsets_grid.astype(np.float32),
        rights=rights_grid,
        contract_ids=contract_ids_grid,
        market_last=np.nan_to_num(
            np.asarray(row["market_window"], dtype=np.float32)[-1],
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        ),
        pattern_targets=pattern_targets,
        value_targets=value_targets,
        quality_targets=quality_targets,
    )


def load_surface_decisions(
    paths: Sequence[Path],
    *,
    policy_index: int,
    variant: SurfaceVariant,
    market_cache: MarketStructureCache,
) -> list[SurfaceDecision]:
    decisions = []
    for path in sorted(paths):
        session = path.name.removesuffix(".pkl")
        with path.open("rb") as f:
            rows = pickle.load(f)
        for row in rows:
            decisions.append(
                surface_decision_from_row(
                    session=session,
                    row=row,
                    policy_index=policy_index,
                    variant=variant,
                    market_cache=market_cache,
                )
            )
    return decisions


class SurfaceActionModel(nn.Module):
    """Flat plus contract-token action scorer."""

    def __init__(
        self,
        *,
        scalar_dim: int,
        token_dim: int,
        hidden_dim: int = 128,
        token_hidden_dim: int = 64,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, token_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(token_hidden_dim),
            nn.Linear(token_hidden_dim, token_hidden_dim),
            nn.GELU(),
        )
        self.action_trunk = nn.Sequential(
            nn.Linear(hidden_dim + token_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.flat_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.utility = nn.Linear(hidden_dim // 2, 1)
        self.pattern_logit = nn.Linear(hidden_dim // 2, 1)
        self.value_logit = nn.Linear(hidden_dim // 2, 1)

    def forward_with_aux(
        self,
        scalar: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.scalar_proj(scalar)
        token_h = self.token_proj(tokens)
        expanded = state.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        action_h = self.action_trunk(torch.cat([expanded, token_h], dim=-1))
        token_score = self.utility(action_h).squeeze(-1)
        pattern_logit = self.pattern_logit(action_h).squeeze(-1)
        value_logit = self.value_logit(action_h).squeeze(-1)
        flat_score = self.utility(self.flat_trunk(state)).squeeze(-1)
        return torch.cat([flat_score.unsqueeze(1), token_score], dim=1), pattern_logit, value_logit

    def forward(self, scalar: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        score, _, _ = self.forward_with_aux(scalar, tokens)
        return score


def _surface_targets(
    decisions: Sequence[SurfaceDecision],
    *,
    target_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.stack([d.labels for d in decisions]).astype(np.float32)
    token_mask = np.stack([d.token_mask for d in decisions]).astype(bool)
    targets = np.concatenate([np.zeros((len(decisions), 1), dtype=np.float32), labels], axis=1)
    action_mask = np.concatenate([np.ones((len(decisions), 1), dtype=bool), token_mask], axis=1)
    targets = np.clip(targets, -600.0, 600.0) / target_scale
    is_call = np.stack([d.rights == "C" for d in decisions]).astype(bool)
    is_put = np.stack([d.rights == "P" for d in decisions]).astype(bool)
    pattern_targets = np.stack(
        [
            getattr(d, "pattern_targets", None)
            if getattr(d, "pattern_targets", None) is not None
            else np.zeros_like(d.labels)
            for d in decisions
        ]
    ).astype(np.float32)
    value_targets = np.stack(
        [
            getattr(d, "value_targets", None)
            if getattr(d, "value_targets", None) is not None
            else np.zeros_like(d.labels)
            for d in decisions
        ]
    ).astype(np.float32)
    quality_targets = np.stack(
        [
            getattr(d, "quality_targets", None)
            if getattr(d, "quality_targets", None) is not None
            else np.zeros_like(d.labels)
            for d in decisions
        ]
    ).astype(np.float32)
    return targets.astype(np.float32), action_mask, is_call, is_put, pattern_targets, value_targets, quality_targets


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=1.0)


def _decision_side_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    is_call: torch.Tensor,
    is_put: torch.Tensor,
    *,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_pred = pred.masked_fill(~action_mask, -1_000_000.0)
    masked_target = target.masked_fill(~action_mask, -1_000_000.0)
    best_action = torch.argmax(masked_target, dim=1)
    decision_ce = F.cross_entropy(masked_pred / 0.75, best_action)

    no_trade_score = pred[:, 0]
    best_trade_score = masked_pred[:, 1:].max(dim=1).values
    best_action_score = pred.gather(1, best_action.unsqueeze(1)).squeeze(1)
    no_trade_is_best = best_action == 0
    trade_is_best = best_action != 0
    false_trade = F.relu(best_trade_score - no_trade_score + 25.0 / target_scale)
    missed_trade = F.relu(no_trade_score - best_action_score + 10.0 / target_scale)
    false_trade = false_trade[no_trade_is_best].mean() if no_trade_is_best.any() else pred.sum() * 0.0
    missed_trade = missed_trade[trade_is_best].mean() if trade_is_best.any() else pred.sum() * 0.0

    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    valid_call = is_call & action_mask[:, 1:]
    valid_put = is_put & action_mask[:, 1:]
    call_pred = token_pred.masked_fill(~valid_call, -1_000_000.0).max(dim=1).values
    put_pred = token_pred.masked_fill(~valid_put, -1_000_000.0).max(dim=1).values
    call_target = token_target.masked_fill(~valid_call, -1_000_000.0).max(dim=1).values
    put_target = token_target.masked_fill(~valid_put, -1_000_000.0).max(dim=1).values
    side_valid = valid_call.any(dim=1) & valid_put.any(dim=1)
    side_delta = call_target - put_target
    side_known = side_valid & (torch.abs(side_delta) >= 25.0 / target_scale)
    if side_known.any():
        sign = torch.sign(side_delta[side_known])
        side_rank = F.softplus(-(call_pred[side_known] - put_pred[side_known]) * sign).mean()
    else:
        side_rank = pred.sum() * 0.0

    total = (
        0.20 * decision_ce
        + 1.25 * false_trade
        + 0.35 * missed_trade
        + 0.35 * side_rank
    )
    return total, {
        "decision_ce": decision_ce,
        "false_trade_margin": false_trade,
        "missed_trade_margin": missed_trade,
        "side_rank": side_rank,
    }


def _aplus_teacher_margin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    target_scale: float,
    pattern_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    flat_score = pred[:, 0].unsqueeze(1)
    token_mask = action_mask[:, 1:]
    teacher_context = (pattern_target > 0.5) & (value_target > 0.5) & token_mask
    positive_teacher = teacher_context & (token_target >= 25.0 / target_scale)
    losing_token = token_mask & (token_target <= 0.0)

    if positive_teacher.any():
        positive_margin = F.relu(flat_score + 25.0 / target_scale - token_pred)[positive_teacher].mean()
    else:
        positive_margin = pred.sum() * 0.0
    if losing_token.any():
        losing_margin = F.relu(token_pred - flat_score + 5.0 / target_scale)[losing_token].mean()
    else:
        losing_margin = pred.sum() * 0.0
    total = positive_margin + 0.20 * losing_margin
    return total, {
        "teacher_positive_margin": positive_margin,
        "teacher_losing_margin": losing_margin,
    }


def _contract_rank_margin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    is_call: torch.Tensor,
    is_put: torch.Tensor,
    *,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank same-minute contracts by realized executable value.

    This is deliberately side-local: it teaches contract choice once a call or
    put context exists, while leaving the existing decision-side loss to handle
    flat-vs-trade and call-vs-put.
    """

    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    token_mask = action_mask[:, 1:]
    losses = []
    margin = 15.0 / target_scale
    min_winner = 50.0 / target_scale
    min_gap = 50.0 / target_scale
    for side_mask in (is_call, is_put):
        valid = token_mask & side_mask
        winner_valid = valid & (token_target >= min_winner)
        if not winner_valid.any():
            continue
        winner_target = token_target.unsqueeze(2)
        loser_target = token_target.unsqueeze(1)
        winner_pred = token_pred.unsqueeze(2)
        loser_pred = token_pred.unsqueeze(1)
        pair_mask = (
            winner_valid.unsqueeze(2)
            & valid.unsqueeze(1)
            & ((winner_target - loser_target) >= min_gap)
        )
        if pair_mask.any():
            losses.append(F.relu(loser_pred - winner_pred + margin)[pair_mask].mean())
    if losses:
        rank_margin = torch.stack(losses).mean()
    else:
        rank_margin = pred.sum() * 0.0
    return rank_margin, {"contract_rank_margin": rank_margin}


def _aplus_side_quality_margin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    is_put: torch.Tensor,
    *,
    target_scale: float,
    pattern_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A stricter in-network A+ contract-quality objective.

    This keeps profitable pattern/value tokens active, while making the model
    more skeptical of marginal or losing put tokens when the contract-value
    target says the spread/theta/breakeven economics are not good enough.
    """

    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    flat_score = pred[:, 0].unsqueeze(1)
    token_mask = action_mask[:, 1:]
    pattern_context = (pattern_target > 0.5) & token_mask
    value_good = (value_target > 0.5) & token_mask
    value_bad = ~value_good & token_mask
    positive_teacher = pattern_context & value_good & (token_target >= 50.0 / target_scale)
    losing_token = token_mask & (token_target <= 0.0)
    marginal_value_bad = value_bad & (token_target < 25.0 / target_scale)
    put_value_bad = is_put & pattern_context & value_bad
    put_losing_pattern = is_put & pattern_context & losing_token

    if positive_teacher.any():
        positive_margin = F.relu(flat_score + 35.0 / target_scale - token_pred)[positive_teacher].mean()
    else:
        positive_margin = pred.sum() * 0.0
    if losing_token.any():
        losing_margin = F.relu(token_pred - flat_score + 8.0 / target_scale)[losing_token].mean()
    else:
        losing_margin = pred.sum() * 0.0
    if marginal_value_bad.any():
        value_bad_margin = F.relu(token_pred - flat_score + 12.0 / target_scale)[marginal_value_bad].mean()
    else:
        value_bad_margin = pred.sum() * 0.0
    put_gate = put_value_bad | put_losing_pattern
    if put_gate.any():
        put_quality_margin = F.relu(token_pred - flat_score + 28.0 / target_scale)[put_gate].mean()
    else:
        put_quality_margin = pred.sum() * 0.0

    total = (
        positive_margin
        + 0.18 * losing_margin
        + 0.35 * value_bad_margin
        + 0.70 * put_quality_margin
    )
    return total, {
        "teacher_positive_margin": positive_margin,
        "teacher_losing_margin": losing_margin + value_bad_margin + put_quality_margin,
    }


def _aplus_soft_quality_confidence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    target_scale: float,
    pattern_target: torch.Tensor,
    quality_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft quality shaping without a hard quality veto.

    Low-quality losing tokens are nudged down, but profitable tokens always keep
    a positive margin opportunity. This is intentionally weaker than the prior
    proposal-level gates because March showed static contract economics can
    reject convex winners.
    """

    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    flat_score = pred[:, 0].unsqueeze(1)
    token_mask = action_mask[:, 1:]
    quality = torch.clamp(quality_target, 0.0, 1.0)
    pattern = torch.clamp(pattern_target, 0.0, 1.0)
    positive = token_mask & (token_target >= 25.0 / target_scale)
    losing = token_mask & (token_target <= 0.0)

    # Positive winners still get a margin even when quality is mediocre; the
    # margin merely grows when economics also look good.
    if positive.any():
        positive_margin = F.relu(
            flat_score + (12.0 + 20.0 * quality) / target_scale - token_pred
        )[positive].mean()
    else:
        positive_margin = pred.sum() * 0.0

    # Bad-quality losers are discouraged, but this is a soft calibration term,
    # not a command to suppress every expensive-looking contract.
    if losing.any():
        bad_quality = 1.0 - quality
        losing_raw = F.relu(
            token_pred - flat_score + (4.0 + 12.0 * bad_quality) / target_scale
        )
        losing_weight = 0.10 + 0.30 * bad_quality + 0.10 * pattern
        losing_margin = (losing_raw[losing] * losing_weight[losing]).mean()
    else:
        losing_margin = pred.sum() * 0.0

    total = 0.55 * positive_margin + 0.35 * losing_margin
    return total, {
        "quality_positive_margin": positive_margin,
        "quality_losing_margin": losing_margin,
    }


def _aplus_put_pattern_recall_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    is_put: torch.Tensor,
    *,
    target_scale: float,
    pattern_target: torch.Tensor,
    quality_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Softly preserve profitable put-pattern availability.

    Protocol 043 found that official context was suppressing too many
    post-open morning put winners with pattern confirmation. This objective is
    intentionally not a hard put filter: profitable put-pattern tokens get a
    recall margin, while losing puts are penalized more when contract quality
    is poor.
    """

    token_pred = pred[:, 1:]
    token_target = target[:, 1:]
    flat_score = pred[:, 0].unsqueeze(1)
    token_mask = action_mask[:, 1:]
    valid_put = is_put & token_mask
    quality = torch.clamp(quality_target, 0.0, 1.0)
    pattern = torch.clamp(pattern_target, 0.0, 1.0)

    positive_put_pattern = valid_put & (pattern > 0.5) & (token_target >= 25.0 / target_scale)
    losing_put = valid_put & (token_target <= 0.0)

    if positive_put_pattern.any():
        recall_margin = F.relu(
            flat_score + (14.0 + 18.0 * quality) / target_scale - token_pred
        )[positive_put_pattern].mean()
    else:
        recall_margin = pred.sum() * 0.0

    if losing_put.any():
        bad_quality = 1.0 - quality
        losing_raw = F.relu(
            token_pred - flat_score + (4.0 + 10.0 * bad_quality) / target_scale
        )
        losing_weight = 0.08 + 0.18 * bad_quality + 0.05 * (1.0 - pattern)
        losing_margin = (losing_raw[losing_put] * losing_weight[losing_put]).mean()
    else:
        losing_margin = pred.sum() * 0.0

    total = 0.70 * recall_margin + 0.28 * losing_margin
    return total, {
        "quality_positive_margin": recall_margin,
        "quality_losing_margin": losing_margin,
    }


def surface_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    is_call: torch.Tensor,
    is_put: torch.Tensor,
    *,
    target_scale: float,
    loss_mode: str,
    pattern_logit: torch.Tensor | None = None,
    value_logit: torch.Tensor | None = None,
    pattern_target: torch.Tensor | None = None,
    value_target: torch.Tensor | None = None,
    quality_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    huber = _masked_huber(pred, target, action_mask)
    zero = huber * 0.0
    parts = {
        "total": huber,
        "huber": huber,
        "decision_ce": zero,
        "false_trade_margin": zero,
        "missed_trade_margin": zero,
        "side_rank": zero,
        "pattern_bce": zero,
        "value_bce": zero,
        "quality_bce": zero,
        "teacher_positive_margin": zero,
        "teacher_losing_margin": zero,
        "quality_positive_margin": zero,
        "quality_losing_margin": zero,
        "contract_rank_margin": zero,
    }
    if loss_mode == "huber":
        return huber, parts
    if loss_mode in {
        "decision_side",
        "aplus_multitask",
        "aplus_teacher_margin",
        "aplus_side_quality_margin",
        "aplus_side_value_multitask",
        "aplus_side_value_rank",
        "aplus_balanced_value_multitask",
        "aplus_soft_quality_confidence",
        "aplus_put_pattern_recall",
    }:
        extra, extra_parts = _decision_side_loss(
            pred,
            target,
            action_mask,
            is_call,
            is_put,
            target_scale=target_scale,
        )
        total = huber + extra
        parts.update(extra_parts)
        parts["total"] = total
        if loss_mode == "aplus_multitask":
            if (
                pattern_logit is None
                or value_logit is None
                or pattern_target is None
                or value_target is None
            ):
                raise ValueError("aplus_multitask requires pattern/value logits and targets")
            token_mask = action_mask[:, 1:]
            pattern_loss = F.binary_cross_entropy_with_logits(
                pattern_logit,
                pattern_target,
                reduction="none",
            )
            value_loss = F.binary_cross_entropy_with_logits(
                value_logit,
                value_target,
                reduction="none",
            )
            denom = token_mask.float().sum().clamp(min=1.0)
            pattern_bce = (pattern_loss * token_mask.float()).sum() / denom
            value_bce = (value_loss * token_mask.float()).sum() / denom
            total = total + 0.12 * pattern_bce + 0.18 * value_bce
            parts["pattern_bce"] = pattern_bce
            parts["value_bce"] = value_bce
            parts["total"] = total
        if loss_mode == "aplus_soft_quality_confidence":
            if (
                pattern_logit is None
                or value_logit is None
                or pattern_target is None
                or quality_target is None
            ):
                raise ValueError("aplus_soft_quality_confidence requires pattern/value logits and pattern/quality targets")
            token_mask = action_mask[:, 1:]
            pattern_loss = F.binary_cross_entropy_with_logits(
                pattern_logit,
                pattern_target,
                reduction="none",
            )
            quality_loss = F.binary_cross_entropy_with_logits(
                value_logit,
                torch.clamp(quality_target, 0.0, 1.0),
                reduction="none",
            )
            denom = token_mask.float().sum().clamp(min=1.0)
            pattern_bce = (pattern_loss * token_mask.float()).sum() / denom
            quality_bce = (quality_loss * token_mask.float()).sum() / denom
            soft_extra, soft_parts = _aplus_soft_quality_confidence_loss(
                pred,
                target,
                action_mask,
                target_scale=target_scale,
                pattern_target=pattern_target,
                quality_target=quality_target,
            )
            total = total + 0.05 * pattern_bce + 0.16 * quality_bce + soft_extra
            parts["pattern_bce"] = pattern_bce
            parts["quality_bce"] = quality_bce
            parts.update(soft_parts)
            parts["total"] = total
        if loss_mode == "aplus_put_pattern_recall":
            if (
                pattern_logit is None
                or value_logit is None
                or pattern_target is None
                or quality_target is None
            ):
                raise ValueError("aplus_put_pattern_recall requires pattern/value logits and pattern/quality targets")
            token_mask = action_mask[:, 1:]
            pattern_loss = F.binary_cross_entropy_with_logits(
                pattern_logit,
                pattern_target,
                reduction="none",
            )
            quality_loss = F.binary_cross_entropy_with_logits(
                value_logit,
                torch.clamp(quality_target, 0.0, 1.0),
                reduction="none",
            )
            pattern_weight = token_mask.float() * (1.0 + 0.25 * is_put.float())
            quality_weight = token_mask.float() * (1.0 + 0.15 * is_put.float())
            pattern_bce = (pattern_loss * pattern_weight).sum() / pattern_weight.sum().clamp(min=1.0)
            quality_bce = (quality_loss * quality_weight).sum() / quality_weight.sum().clamp(min=1.0)
            recall_extra, recall_parts = _aplus_put_pattern_recall_loss(
                pred,
                target,
                action_mask,
                is_put,
                target_scale=target_scale,
                pattern_target=pattern_target,
                quality_target=quality_target,
            )
            total = total + 0.07 * pattern_bce + 0.10 * quality_bce + recall_extra
            parts["pattern_bce"] = pattern_bce
            parts["quality_bce"] = quality_bce
            parts.update(recall_parts)
            parts["total"] = total
        if loss_mode in {"aplus_side_value_multitask", "aplus_side_value_rank"}:
            if (
                pattern_logit is None
                or value_logit is None
                or pattern_target is None
                or value_target is None
            ):
                raise ValueError("aplus_side_value_multitask requires pattern/value logits and targets")
            token_mask = action_mask[:, 1:]
            pattern_loss = F.binary_cross_entropy_with_logits(
                pattern_logit,
                pattern_target,
                reduction="none",
            )
            value_loss = F.binary_cross_entropy_with_logits(
                value_logit,
                value_target,
                reduction="none",
            )
            value_weight = token_mask.float() * (
                1.0
                + 0.40 * is_put.float()
                + 0.35 * (value_target <= 0.5).float()
                + 0.45 * (is_put & (value_target <= 0.5)).float()
            )
            pattern_weight = token_mask.float() * (1.0 + 0.15 * is_put.float())
            pattern_bce = (pattern_loss * pattern_weight).sum() / pattern_weight.sum().clamp(min=1.0)
            value_bce = (value_loss * value_weight).sum() / value_weight.sum().clamp(min=1.0)
            total = total + 0.08 * pattern_bce + 0.24 * value_bce
            parts["pattern_bce"] = pattern_bce
            parts["value_bce"] = value_bce
            if pattern_target is None or value_target is None:
                raise ValueError("aplus_side_value_multitask requires pattern/value targets")
            teacher_extra, teacher_parts = _aplus_teacher_margin_loss(
                pred,
                target,
                action_mask,
                target_scale=target_scale,
                pattern_target=pattern_target,
                value_target=value_target,
            )
            total = total + 0.90 * teacher_extra
            parts.update(teacher_parts)
            if loss_mode == "aplus_side_value_rank":
                rank_extra, rank_parts = _contract_rank_margin_loss(
                    pred,
                    target,
                    action_mask,
                    is_call,
                    is_put,
                    target_scale=target_scale,
                )
                total = total + 0.35 * rank_extra
                parts.update(rank_parts)
            parts["total"] = total
        if loss_mode == "aplus_balanced_value_multitask":
            if (
                pattern_logit is None
                or value_logit is None
                or pattern_target is None
                or value_target is None
            ):
                raise ValueError("aplus_balanced_value_multitask requires pattern/value logits and targets")
            token_mask = action_mask[:, 1:]
            pattern_loss = F.binary_cross_entropy_with_logits(
                pattern_logit,
                pattern_target,
                reduction="none",
            )
            value_loss = F.binary_cross_entropy_with_logits(
                value_logit,
                value_target,
                reduction="none",
            )
            value_weight = token_mask.float() * (
                1.0
                + 0.18 * is_put.float()
                + 0.18 * (value_target <= 0.5).float()
                + 0.18 * (is_put & (value_target <= 0.5)).float()
            )
            pattern_weight = token_mask.float() * (1.0 + 0.08 * is_put.float())
            pattern_bce = (pattern_loss * pattern_weight).sum() / pattern_weight.sum().clamp(min=1.0)
            value_bce = (value_loss * value_weight).sum() / value_weight.sum().clamp(min=1.0)
            total = total + 0.06 * pattern_bce + 0.12 * value_bce
            parts["pattern_bce"] = pattern_bce
            parts["value_bce"] = value_bce
            teacher_extra, teacher_parts = _aplus_teacher_margin_loss(
                pred,
                target,
                action_mask,
                target_scale=target_scale,
                pattern_target=pattern_target,
                value_target=value_target,
            )
            total = total + 1.10 * teacher_extra
            parts.update(teacher_parts)
            parts["total"] = total
        if loss_mode == "aplus_teacher_margin":
            if pattern_target is None or value_target is None:
                raise ValueError("aplus_teacher_margin requires pattern/value targets")
            teacher_extra, teacher_parts = _aplus_teacher_margin_loss(
                pred,
                target,
                action_mask,
                target_scale=target_scale,
                pattern_target=pattern_target,
                value_target=value_target,
            )
            total = total + 1.25 * teacher_extra
            parts.update(teacher_parts)
            parts["total"] = total
        if loss_mode == "aplus_side_quality_margin":
            if pattern_target is None or value_target is None:
                raise ValueError("aplus_side_quality_margin requires pattern/value targets")
            side_quality_extra, side_quality_parts = _aplus_side_quality_margin_loss(
                pred,
                target,
                action_mask,
                is_put,
                target_scale=target_scale,
                pattern_target=pattern_target,
                value_target=value_target,
            )
            total = total + 1.10 * side_quality_extra
            parts.update(side_quality_parts)
            parts["total"] = total
        return total, parts
    raise ValueError(f"unknown surface loss_mode={loss_mode}")


def train_surface_model(
    train_decisions: Sequence[SurfaceDecision],
    validation_decisions: Sequence[SurfaceDecision],
    *,
    config: PilotConfig,
    variant: SurfaceVariant,
) -> tuple[SurfaceActionModel, SurfaceStandardizer, list[dict]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    standardizer = SurfaceStandardizer.fit(train_decisions)
    x_scalar, x_token, token_mask = standardizer.transform(train_decisions)
    v_scalar, v_token, _ = standardizer.transform(validation_decisions)
    (
        y_train,
        action_mask,
        is_call,
        is_put,
        pattern_target,
        value_target,
        quality_target,
    ) = _surface_targets(train_decisions, target_scale=config.target_scale)
    (
        y_val,
        val_action_mask,
        val_is_call,
        val_is_put,
        val_pattern_target,
        val_value_target,
        val_quality_target,
    ) = _surface_targets(validation_decisions, target_scale=config.target_scale)

    model = SurfaceActionModel(
        scalar_dim=x_scalar.shape[1],
        token_dim=x_token.shape[-1],
        hidden_dim=max(config.hidden_dim, 128),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_scalar),
            torch.from_numpy(x_token),
            torch.from_numpy(y_train),
            torch.from_numpy(action_mask),
            torch.from_numpy(is_call),
            torch.from_numpy(is_put),
            torch.from_numpy(pattern_target),
            torch.from_numpy(value_target),
            torch.from_numpy(quality_target),
            torch.from_numpy(token_mask),
        ),
        batch_size=min(config.batch_size, len(x_scalar)),
        shuffle=True,
    )
    val_tensors = (
        torch.from_numpy(v_scalar),
        torch.from_numpy(v_token),
        torch.from_numpy(y_val),
        torch.from_numpy(val_action_mask),
        torch.from_numpy(val_is_call),
        torch.from_numpy(val_is_put),
        torch.from_numpy(val_pattern_target),
        torch.from_numpy(val_value_target),
        torch.from_numpy(val_quality_target),
    )
    aux_loss_modes = {
        "aplus_multitask",
        "aplus_side_value_multitask",
        "aplus_side_value_rank",
        "aplus_balanced_value_multitask",
        "aplus_soft_quality_confidence",
        "aplus_put_pattern_recall",
    }
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_parts = {
            name: []
            for name in (
                "total",
                "huber",
                "decision_ce",
                "false_trade_margin",
                "missed_trade_margin",
                "side_rank",
                "pattern_bce",
                "value_bce",
                "quality_bce",
                "teacher_positive_margin",
                "teacher_losing_margin",
                "quality_positive_margin",
                "quality_losing_margin",
                "contract_rank_margin",
            )
        }
        for scalar_b, token_b, y_b, mask_b, call_b, put_b, pattern_b, value_b, quality_b, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            if variant.loss_mode in aux_loss_modes:
                pred, pattern_logit, value_logit = model.forward_with_aux(scalar_b, token_b)
            else:
                pred = model(scalar_b, token_b)
                pattern_logit = value_logit = None
            loss, parts = surface_loss(
                pred,
                y_b,
                mask_b,
                call_b,
                put_b,
                target_scale=config.target_scale,
                loss_mode=variant.loss_mode,
                pattern_logit=pattern_logit,
                value_logit=value_logit,
                pattern_target=pattern_b,
                value_target=value_b,
                quality_target=quality_b,
            )
            loss.backward()
            optimizer.step()
            for name, value in parts.items():
                train_parts[name].append(float(value.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if variant.loss_mode in aux_loss_modes:
                val_pred, val_pattern_logit, val_value_logit = model.forward_with_aux(val_tensors[0], val_tensors[1])
            else:
                val_pred = model(val_tensors[0], val_tensors[1])
                val_pattern_logit = val_value_logit = None
            _, val_parts_t = surface_loss(
                val_pred,
                val_tensors[2],
                val_tensors[3],
                val_tensors[4],
                val_tensors[5],
                target_scale=config.target_scale,
                loss_mode=variant.loss_mode,
                pattern_logit=val_pattern_logit,
                value_logit=val_value_logit,
                pattern_target=val_tensors[6],
                value_target=val_tensors[7],
                quality_target=val_tensors[8],
            )
        val_parts = {name: float(value.detach().cpu()) for name, value in val_parts_t.items()}
        is_best = val_parts["total"] < best_val
        if is_best:
            best_val = val_parts["total"]
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "loss_mode": variant.loss_mode,
                "train_total": float(np.mean(train_parts["total"])),
                "train_huber": float(np.mean(train_parts["huber"])),
                "train_side_rank": float(np.mean(train_parts["side_rank"])),
                "train_pattern_bce": float(np.mean(train_parts["pattern_bce"])),
                "train_value_bce": float(np.mean(train_parts["value_bce"])),
                "train_quality_bce": float(np.mean(train_parts["quality_bce"])),
                "train_teacher_positive_margin": float(np.mean(train_parts["teacher_positive_margin"])),
                "train_teacher_losing_margin": float(np.mean(train_parts["teacher_losing_margin"])),
                "train_quality_positive_margin": float(np.mean(train_parts["quality_positive_margin"])),
                "train_quality_losing_margin": float(np.mean(train_parts["quality_losing_margin"])),
                "train_contract_rank_margin": float(np.mean(train_parts["contract_rank_margin"])),
                "validation_total": val_parts["total"],
                "validation_huber": val_parts["huber"],
                "validation_side_rank": val_parts["side_rank"],
                "validation_pattern_bce": val_parts["pattern_bce"],
                "validation_value_bce": val_parts["value_bce"],
                "validation_quality_bce": val_parts["quality_bce"],
                "validation_teacher_positive_margin": val_parts["teacher_positive_margin"],
                "validation_teacher_losing_margin": val_parts["teacher_losing_margin"],
                "validation_quality_positive_margin": val_parts["quality_positive_margin"],
                "validation_quality_losing_margin": val_parts["quality_losing_margin"],
                "validation_contract_rank_margin": val_parts["contract_rank_margin"],
                "is_best": is_best,
            }
        )
    model.load_state_dict(best_state)
    return model, standardizer, history


def predict_surface_actions(
    model: SurfaceActionModel,
    standardizer: SurfaceStandardizer,
    decisions: Sequence[SurfaceDecision],
    *,
    target_scale: float,
    batch_size: int = 4096,
) -> np.ndarray:
    if not decisions:
        return np.empty((0, 1), dtype=np.float32)
    scalar, tokens, _ = standardizer.transform(decisions)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(scalar), batch_size):
            pred = model(torch.from_numpy(scalar[start : start + batch_size]), torch.from_numpy(tokens[start : start + batch_size]))
            out.append(pred.cpu().numpy().astype(np.float32) * target_scale)
    return np.vstack(out)


@dataclass(frozen=True)
class ProtocolTrial:
    """One fixed executable-policy trial used across all variants."""

    name: str
    allowed_buckets: tuple[str, ...]
    min_edge_vs_no_trade: float
    max_trades_per_day: int
    daily_loss_stop: float | None

    @property
    def config_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def registered_protocol_trials() -> tuple[ProtocolTrial, ...]:
    return (
        ProtocolTrial("all_times_edge0_max4", ("first_30", "post_open_morning", "midday", "late_afternoon"), 0.0, 4, None),
        ProtocolTrial("skip_first30_edge0_max4_stop1000", ("post_open_morning", "midday", "late_afternoon"), 0.0, 4, -1000.0),
        ProtocolTrial("post_open_late_edge0_max4_stop1000", ("post_open_morning", "late_afternoon"), 0.0, 4, -1000.0),
        ProtocolTrial("late_afternoon_edge0_max2", ("late_afternoon",), 0.0, 2, None),
        ProtocolTrial("post_open_late_edge25_max4", ("post_open_morning", "late_afternoon"), 25.0, 4, None),
        ProtocolTrial("post_open_late_edge25_max2", ("post_open_morning", "late_afternoon"), 25.0, 2, None),
        ProtocolTrial("late_afternoon_edge25_max2", ("late_afternoon",), 25.0, 2, None),
    )


def simulate_surface_policy(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    trades = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=edge,
                right=str(decision.rights[token_idx]),
                offset=float(decision.offsets[token_idx]),
                strategy=strategy,
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return trades


def simulate_random_surface_policy(
    decisions: Sequence[SurfaceDecision],
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
    seed: int,
    target_trade_count: int | None = None,
) -> list[Trade]:
    """Random valid-action baseline under the same time/risk trial.

    The random policy uses the same allowed buckets, cooldown, max trades per
    day, and daily stop as the neural policy. If `target_trade_count` is
    supplied, the baseline randomly thins eligible minutes toward the same
    approximate activity level instead of trading every possible cooldown slot.
    """

    rng = np.random.default_rng(seed)
    allowed = set(trial.allowed_buckets)
    eligible = [
        d
        for d in decisions
        if time_bucket(d.decision_time) in allowed and d.token_mask.any()
    ]
    if not eligible:
        return []
    accept_prob = 1.0
    if target_trade_count is not None:
        accept_prob = min(1.0, max(0.01, float(target_trade_count) / max(len(eligible), 1)))

    trades = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    for decision in eligible:
        if rng.random() > accept_prob:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        valid = np.where(decision.token_mask & np.isfinite(decision.labels))[0]
        if len(valid) == 0:
            continue
        token_idx = int(rng.choice(valid))
        pnl = float(decision.labels[token_idx])
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=None,
                right=str(decision.rights[token_idx]),
                offset=float(decision.offsets[token_idx]),
                strategy=strategy,
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
        if target_trade_count is not None and len(trades) >= target_trade_count:
            break
    return trades


def stress_trades(trades: Sequence[Trade], *, extra_cost_per_trade: float) -> list[Trade]:
    """Subtract an extra round-trip friction cost from every trade."""

    if extra_cost_per_trade <= 0:
        return list(trades)
    return [
        Trade(
            session=trade.session,
            decision_time=trade.decision_time,
            pnl=float(trade.pnl) - extra_cost_per_trade,
            score=trade.score,
            right=trade.right,
            offset=trade.offset,
            strategy=f"{trade.strategy}:extra_cost_{extra_cost_per_trade:.0f}",
        )
        for trade in trades
    ]


def summarize_random_baseline(
    decisions: Sequence[SurfaceDecision],
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    seed: int,
    target_trade_count: int,
    runs: int = 20,
) -> dict:
    """Median random-baseline metrics across repeated random seeds."""

    if target_trade_count <= 0:
        empty = metrics_with_concentration([])
        return {
            "runs": runs,
            "target_trade_count": int(target_trade_count),
            "trades_median": float(empty["trades"]),
            "total_pnl_median": float(empty["total_pnl"]),
            "profit_factor_median": float(empty["profit_factor"]),
            "max_drawdown_median": float(empty["max_drawdown"]),
            "positive_day_fraction_median": float(empty["positive_day_fraction"]),
            "top_day_profit_share_median": float(empty["top_day_profit_share"]),
        }

    metrics = []
    for i in range(runs):
        trades = simulate_random_surface_policy(
            decisions,
            trial=trial,
            cooldown_minutes=cooldown_minutes,
            strategy="matched_random_surface",
            seed=seed + i * 10_007,
            target_trade_count=target_trade_count,
        )
        metrics.append(metrics_with_concentration(trades))

    def median(name: str) -> float:
        return float(np.median([m[name] for m in metrics]))

    return {
        "runs": runs,
        "target_trade_count": int(target_trade_count),
        "trades_median": median("trades"),
        "total_pnl_median": median("total_pnl"),
        "profit_factor_median": median("profit_factor"),
        "max_drawdown_median": median("max_drawdown"),
        "positive_day_fraction_median": median("positive_day_fraction"),
        "top_day_profit_share_median": median("top_day_profit_share"),
    }


def selection_reward(metrics: dict) -> float:
    trades = float(metrics["trades"])
    if trades < 12:
        return -1_000_000.0 + trades
    profit_factor = float(metrics["profit_factor"])
    if not np.isfinite(profit_factor):
        profit_factor = 5.0
    return (
        float(metrics["total_pnl"])
        + 1_000.0 * (min(profit_factor, 5.0) - 1.0)
        + 1_500.0 * (float(metrics["positive_day_fraction"]) - 0.50)
        + 0.15 * float(metrics["max_drawdown"])
        - 1_500.0 * max(0.0, float(metrics.get("top_day_profit_share", 1.0)) - 0.45)
    )


def aggregate_protocol_rows(rows: Sequence[dict]) -> dict:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"]["name"], int(row["policy_index"]), row["trial"]["name"]), []).append(row)
    by_combo = {}
    for (variant_name, policy_index, trial_name), group in sorted(groups.items()):
        selection = [r["metrics_by_split"]["selection"] for r in group]
        march = [r["metrics_by_split"]["march"] for r in group]
        q4 = [r["metrics_by_split"]["q4"] for r in group]
        rewards = np.asarray([selection_reward(m) for m in selection], dtype=float)

        def med(name: str, metrics: list[dict]) -> float:
            return float(np.median([m[name] for m in metrics]))

        summary = {
            "variant": variant_name,
            "policy_index": policy_index,
            "policy_name": group[0]["policy_name"],
            "trial": trial_name,
            "runs": len(group),
            "selection_reward_median": float(np.median(rewards)),
            "selection_pnl_median": med("total_pnl", selection),
            "selection_pf_median": med("profit_factor", selection),
            "selection_trades_median": med("trades", selection),
            "selection_positive_days_median": med("positive_day_fraction", selection),
            "selection_top_day_share_median": med("top_day_profit_share", selection),
            "march_pnl_median": med("total_pnl", march),
            "march_pf_median": med("profit_factor", march),
            "march_dd_median": med("max_drawdown", march),
            "march_trades_median": med("trades", march),
            "march_positive_seed_fraction": float(np.mean([m["total_pnl"] > 0 for m in march])),
            "march_positive_days_median": med("positive_day_fraction", march),
            "march_top_day_share_median": med("top_day_profit_share", march),
            "q4_pnl_median": med("total_pnl", q4),
            "q4_pf_median": med("profit_factor", q4),
            "q4_dd_median": med("max_drawdown", q4),
            "q4_trades_median": med("trades", q4),
            "q4_positive_seed_fraction": float(np.mean([m["total_pnl"] > 0 for m in q4])),
            "q4_positive_days_median": med("positive_day_fraction", q4),
            "q4_top_day_share_median": med("top_day_profit_share", q4),
        }
        summary["selection_floor"] = bool(
            summary["selection_trades_median"] >= 12
            and summary["selection_pnl_median"] > 0
            and summary["selection_pf_median"] >= 1.05
            and summary["selection_top_day_share_median"] <= 0.75
        )
        summary["march_survives"] = bool(
            summary["march_trades_median"] >= 20
            and summary["march_pnl_median"] > 0
            and summary["march_pf_median"] >= 1.05
            and summary["march_positive_seed_fraction"] >= 2 / 3
        )
        summary["q4_survives"] = bool(
            summary["q4_trades_median"] >= 20
            and summary["q4_pnl_median"] > 0
            and summary["q4_pf_median"] >= 1.05
            and summary["q4_positive_seed_fraction"] >= 2 / 3
        )
        summary["passes_protocol_002_gate"] = bool(
            summary["selection_floor"]
            and summary["march_survives"]
            and summary["q4_survives"]
            and summary["march_top_day_share_median"] <= 0.50
            and summary["q4_top_day_share_median"] <= 0.50
        )
        by_combo[f"{variant_name}|policy{policy_index}|{trial_name}"] = summary
    ranked_selection = sorted(by_combo.values(), key=lambda r: r["selection_reward_median"], reverse=True)
    ranked_generalization = sorted(
        by_combo.values(),
        key=lambda r: (
            r["passes_protocol_002_gate"],
            r["selection_floor"],
            r["selection_trades_median"] > 0,
            r["q4_pnl_median"] + r["march_pnl_median"],
            r["q4_pf_median"],
            r["march_pf_median"],
            -r["q4_top_day_share_median"],
        ),
        reverse=True,
    )
    return {
        "by_combo": by_combo,
        "ranked_selection": ranked_selection,
        "ranked_generalization": ranked_generalization,
        "selection_champion": ranked_selection[0] if ranked_selection else None,
        "generalization_champion": ranked_generalization[0] if ranked_generalization else None,
        "protocol_pass_count": int(sum(1 for row in by_combo.values() if row["passes_protocol_002_gate"])),
    }


def bootstrap_trade_pnl(trades: Sequence[Trade], *, seed: int = 20260430, n: int = 500) -> dict:
    if not trades:
        return {"n": n, "pnl_p05": 0.0, "pnl_p50": 0.0, "pnl_p95": 0.0, "positive_fraction": 0.0}
    rng = np.random.default_rng(seed)
    pnl = np.asarray([t.pnl for t in trades], dtype=float)
    samples = []
    for _ in range(n):
        samples.append(float(rng.choice(pnl, size=len(pnl), replace=True).sum()))
    arr = np.asarray(samples, dtype=float)
    return {
        "n": n,
        "pnl_p05": float(np.quantile(arr, 0.05)),
        "pnl_p50": float(np.quantile(arr, 0.50)),
        "pnl_p95": float(np.quantile(arr, 0.95)),
        "positive_fraction": float((arr > 0).mean()),
    }


def structural_feature_names(market_mode: str) -> list[str]:
    names = [f"base_market_{i}" for i in range(_BASE_MARKET_FEATURES)]
    if market_mode == "structure":
        names.extend(_STRUCTURE_FEATURE_NAMES)
    return names


def token_feature_names(token_mode: str) -> list[str]:
    names = list(OPTION_FEATURE_NAMES) + ["is_call", "is_put", "offset_norm", "abs_offset_norm"]
    if token_mode == "pressure":
        names.extend(_PRESSURE_FEATURE_NAMES)
    elif token_mode in {"aplus", "aplus_interactions", "aplus_relative_quality"}:
        names.extend(_PRESSURE_FEATURE_NAMES)
        names.extend(_APLUS_PATTERN_FEATURE_NAMES)
        names.extend(_APLUS_VALUE_FEATURE_NAMES)
        if token_mode in {"aplus_interactions", "aplus_relative_quality"}:
            names.extend(_APLUS_INTERACTION_FEATURE_NAMES)
        if token_mode == "aplus_relative_quality":
            names.extend(_APLUS_RELATIVE_QUALITY_FEATURE_NAMES)
    elif token_mode != "base":
        raise ValueError(f"unknown token_mode={token_mode}")
    return names
