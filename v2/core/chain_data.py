"""Shared exact-chain data contracts and helpers for the v4 harness.

The exact-chain harness represents the tradable universe as the real SPXW 0DTE
contracts visible on each bar. ``v2/data.pt`` stores the market-context
manifest, while day sidecars store contract matrices plus per-bar executable
snapshots and labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch


CHAIN_SCHEMA_VERSION = "v4_exact_chain_v2_paths"
SIDECAR_DIR_DEFAULT = os.path.join("v2", "data_sidecars")
QUALITY_VALID = 2
QUALITY_PARTIAL = 1
QUALITY_CORRUPT = 0

CONTRACT_FEATURE_FIELDS = [
    "contract_valid",       # 0
    "strike",               # 1
    "right_is_put",         # 2
    "mid",                  # 3
    "spread_fraction",      # 4
    "log_volume",           # 5
    "log_transactions",     # 6
    "iv",                   # 7
    "delta",                # 8
    "gamma",                # 9
    "theta",                # 10
    "moneyness_pct",        # 11
    "distance_points",      # 12
    "minutes_to_close_frac",# 13
    "quality_flag",         # 14
    "vega",                 # 15  IV sensitivity
    "charm",                # 16  dDelta/dTime — 0DTE dealer hedging signal
    "mid_chg_5",            # 17  contract price momentum (5-bar mid change %)
    "mid_chg_10",           # 18  contract price momentum (10-bar mid change %)
    "theta_to_premium",     # 19  |theta_per_bar| / mid — decay rate relative to price
    "breakeven_bars_est",   # 20  mid / |theta_per_bar| clamped [1,200] — time budget
    "gamma_dollar",         # 21  gamma * spot^2 * 0.01 — dollar gamma per 1% move
]
NUM_CONTRACT_FEATURES = len(CONTRACT_FEATURE_FIELDS)


@dataclass(frozen=True)
class ContractId:
    """Exact option identity."""

    expiry: str
    strike: float
    right: str


@dataclass(frozen=True)
class HarnessEvalCase:
    """One tagged harness regression case."""

    case_id: str
    date: str
    bar_of_day: int
    tags: tuple[str, ...]
    notes: str = ""


def right_to_int(right: str) -> int:
    return 1 if right == "P" else 0


def right_from_int(v: int | float) -> str:
    return "P" if int(v) == 1 else "C"


def sidecar_path(sidecar_dir: str, day: str) -> str:
    return os.path.join(sidecar_dir, f"{day}.pt")


def ensure_sidecar_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_wide_bar(bar_contracts: dict[tuple[float, str], dict[str, float]]) -> dict[float, dict[str, float]]:
    """Convert contract-keyed bar data into the wide-grid shape used by feature code."""

    wide: dict[float, dict[str, float]] = {}
    for (strike, right), fields in bar_contracts.items():
        row = wide.setdefault(float(strike), {})
        side = "call" if right == "C" else "put"
        for src, dst in (
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("transactions", "transactions"),
        ):
            row[f"{side}_{dst}"] = float(fields.get(src, 0.0) or 0.0)
    return wide


def spread_fraction_proxy(mid: float, high: float, low: float, tick_floor: float = 0.05) -> float:
    """Conservative spread proxy from OHLC when historical bid/ask is unavailable."""

    if not np.isfinite(mid) or mid <= 0:
        return 1.0
    intrabar = max(0.0, float(high) - float(low))
    floor = tick_floor / mid
    return max(floor, intrabar / max(mid, 1e-6) * 0.35, 0.02)


def quality_from_bar(close: float, volume: float, transactions: float) -> int:
    """Classify the observed contract row."""

    if not np.isfinite(close) or close <= 0:
        return QUALITY_CORRUPT
    if (volume or 0) > 0 or (transactions or 0) > 0:
        return QUALITY_VALID
    return QUALITY_PARTIAL


def manifest_sidecar_digest(sidecar_paths: list[str]) -> str:
    """Hash the set of sidecar files into one short digest."""

    hasher = hashlib.sha256()
    for path in sorted(sidecar_paths):
        hasher.update(os.path.basename(path).encode("utf-8"))
        hasher.update(file_sha256(path).encode("utf-8"))
    return hasher.hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def load_sidecar_cached(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def contract_snapshot(sidecar: dict[str, Any], local_bar: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, labels, contract_indices) for one bar snapshot."""

    ptrs = sidecar["bar_ptrs"]
    start = int(ptrs[local_bar])
    end = int(ptrs[local_bar + 1])
    feats = sidecar["row_features"][start:end]
    labels = sidecar["row_labels"][start:end]
    contract_idx = sidecar["row_contract_idx"][start:end]
    return feats, labels, contract_idx


def padded_snapshot(
    sidecar: dict[str, Any],
    local_bar: int,
    max_contracts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad one bar snapshot to the dataset-wide max contract count."""

    feats, labels, contract_idx = contract_snapshot(sidecar, local_bar)
    n = feats.shape[0]
    feat_pad = np.zeros((max_contracts, NUM_CONTRACT_FEATURES), dtype=np.float32)
    label_pad = np.full((max_contracts,), np.nan, dtype=np.float32)
    idx_pad = np.full((max_contracts,), -1, dtype=np.int32)
    if n > 0:
        use = min(n, max_contracts)
        feat_pad[:use] = feats[:use]
        label_pad[:use] = labels[:use]
        idx_pad[:use] = contract_idx[:use]
    return feat_pad, label_pad, idx_pad


def extract_contract_series(sidecar: dict[str, Any], contract_idx: int) -> dict[str, np.ndarray]:
    """Return the full intraday series for one contract."""

    return {
        "mid": sidecar["contract_mid"][contract_idx],
        "bid": sidecar["contract_bid"][contract_idx],
        "ask": sidecar["contract_ask"][contract_idx],
        "quality": sidecar["contract_quality"][contract_idx],
    }


def describe_contract(sidecar: dict[str, Any], contract_idx: int) -> ContractId:
    return ContractId(
        expiry=sidecar["expiry"],
        strike=float(sidecar["contract_strike"][contract_idx]),
        right=right_from_int(sidecar["contract_right"][contract_idx]),
    )


def build_contract_row(
    *,
    strike: float,
    right: str,
    mid: float,
    spread_frac: float,
    volume: float,
    transactions: float,
    iv: float,
    delta: float,
    gamma: float,
    theta: float,
    spot: float,
    minutes_to_close: int,
    quality: int,
    is_executable: bool,
    vega: float = 0.0,
    charm: float = 0.0,
    mid_chg_5: float = 0.0,
    mid_chg_10: float = 0.0,
) -> np.ndarray:
    """Build one numeric row for the scorer."""

    row = np.zeros(NUM_CONTRACT_FEATURES, dtype=np.float32)
    row[0] = 1.0 if is_executable else 0.0
    row[1] = float(strike)
    row[2] = float(right_to_int(right))
    row[3] = float(mid)
    row[4] = float(spread_frac)
    row[5] = math.log1p(max(0.0, float(volume)))
    row[6] = math.log1p(max(0.0, float(transactions)))
    row[7] = float(iv) if np.isfinite(iv) else 0.0
    row[8] = float(delta) if np.isfinite(delta) else 0.0
    row[9] = float(gamma) if np.isfinite(gamma) else 0.0
    row[10] = float(theta) if np.isfinite(theta) else 0.0
    if np.isfinite(spot) and spot > 0:
        row[11] = float((strike - spot) / spot * 100.0)
        row[12] = float(strike - spot)
    row[13] = float(max(1, minutes_to_close)) / 390.0
    row[14] = float(quality)
    row[15] = float(vega) if np.isfinite(vega) else 0.0
    row[16] = float(charm) if np.isfinite(charm) else 0.0
    row[17] = float(mid_chg_5) if np.isfinite(mid_chg_5) else 0.0
    row[18] = float(mid_chg_10) if np.isfinite(mid_chg_10) else 0.0
    # Derived economic features
    theta_val = row[10]  # theta_per_bar (already per-bar from bs_greeks_vec)
    mid_val = row[3]
    gamma_val = row[9]
    abs_theta = abs(theta_val) if np.isfinite(theta_val) else 0.0
    # [19] theta_to_premium: how fast this contract decays relative to price
    row[19] = abs_theta / max(mid_val, 1e-6) if mid_val > 0 and abs_theta > 0 else 0.0
    # [20] breakeven_bars_est: how many bars before theta eats the premium
    row[20] = min(200.0, max(1.0, mid_val / max(abs_theta, 1e-8))) if mid_val > 0 and abs_theta > 1e-8 else 200.0
    # [21] gamma_dollar: dollar gamma per 1% move in underlying
    spot_val = float(spot)
    row[21] = gamma_val * spot_val * spot_val * 0.01 if np.isfinite(gamma_val) and spot_val > 0 else 0.0
    return row


def dump_suite(path: str, suite: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(suite, f, indent=2, sort_keys=True)


def load_suite(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
