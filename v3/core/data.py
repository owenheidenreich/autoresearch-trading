"""Shared exact-chain data loading for v3."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from v2.core.chain_data import extract_contract_series, load_sidecar_cached, padded_snapshot
from v3.core.market_state import DEFAULT_MARKET_STATE_PATH, load_market_state_cache
from v3.core.schema import FIVE_MINUTE_BUCKET, LOOKBACK_1M, LOOKBACK_5M, SESSION_STATE_FEATURE_NAMES


@dataclass
class EpisodeMarket:
    """One day episode backed by exact-chain sidecars."""

    date: str
    global_indices: np.ndarray
    context_source: np.ndarray
    spot_prices: np.ndarray
    bar_of_day: np.ndarray
    sidecar: dict[str, Any]
    max_contracts: int
    context_5m_source: np.ndarray
    session_state_source: np.ndarray
    lookback_1m: int = LOOKBACK_1M
    lookback_5m: int = LOOKBACK_5M

    def __post_init__(self) -> None:
        self.global_indices = np.asarray(self.global_indices, dtype=np.int32)
        self.spot_prices = np.asarray(self.spot_prices, dtype=np.float32)
        self.bar_of_day = np.asarray(self.bar_of_day, dtype=np.int32)
        self.context_5m_source = np.asarray(self.context_5m_source, dtype=np.float32)
        self.session_state_source = np.asarray(self.session_state_source, dtype=np.float32)
        if len(self.global_indices) != len(self.spot_prices):
            raise ValueError("global_indices and spot_prices must align")
        if len(self.session_state_source) != len(self.global_indices):
            raise ValueError("session_state_source must align with day bars")

    @property
    def n_bars(self) -> int:
        return int(len(self.global_indices))

    def context_1m_window(self, local_bar: int) -> np.ndarray:
        """Return a fixed-length 1-minute lookback window with zero-padding."""

        g = int(self.global_indices[local_bar])
        start = max(0, g - self.lookback_1m)
        window = self.context_source[start:g]
        out = np.zeros((self.lookback_1m, self.context_source.shape[1]), dtype=np.float32)
        if len(window):
            out[-len(window) :] = window
        return out

    def context_5m_window(self, local_bar: int) -> np.ndarray:
        """Return the causal 5-minute bucket context up to the current bar."""

        n_feat = self.context_5m_source.shape[1] if self.context_5m_source.ndim == 2 and self.context_5m_source.size else 0
        out = np.zeros((self.lookback_5m, n_feat), dtype=np.float32)
        if n_feat == 0:
            return out
        completed = min(len(self.context_5m_source), (int(local_bar) + 1) // FIVE_MINUTE_BUCKET)
        if completed <= 0:
            return out
        window = self.context_5m_source[:completed]
        use = min(len(window), self.lookback_5m)
        out[-use:] = window[-use:]
        return out

    def session_state(self, local_bar: int) -> np.ndarray:
        return self.session_state_source[int(local_bar)].astype(np.float32, copy=False)

    def snapshot(self, local_bar: int) -> tuple[np.ndarray, np.ndarray]:
        feats, _, contract_indices = padded_snapshot(self.sidecar, int(local_bar), self.max_contracts)
        valid = feats[:, 0] > 0.5
        return feats.astype(np.float32), contract_indices.astype(np.int32), valid.astype(bool)

    def contract_series(self, contract_idx: int) -> dict[str, np.ndarray]:
        return extract_contract_series(self.sidecar, int(contract_idx))

    def contract_features_for_position(self, contract_idx: int, local_bar: int) -> np.ndarray:
        feats, idxs, _ = self.snapshot(local_bar)
        mask = idxs == int(contract_idx)
        if mask.any():
            return feats[mask][0]
        return np.zeros((feats.shape[-1],), dtype=np.float32)


@dataclass
class RLDataBundle:
    """Manifest plus sidecar index for v3 episodes."""

    data_path: str
    raw: dict[str, Any]
    market_state: dict[str, Any]
    max_contracts: int
    sidecar_dir: str
    day_to_indices: dict[str, np.ndarray]

    @property
    def dataset_fingerprint(self) -> str:
        return str(self.raw.get("metadata", {}).get("fingerprint", "unknown"))

    @property
    def context_source(self) -> np.ndarray:
        return self.raw["X"].numpy()

    @property
    def spot_prices(self) -> np.ndarray:
        return self.raw["spot_prices"].numpy()

    @property
    def bar_of_day(self) -> np.ndarray:
        return self.raw["bar_of_day"].numpy()

    def days_for_mask(self, mask_key: str) -> list[str]:
        mask = self.raw[mask_key].numpy().astype(bool)
        dates = self.raw["dates"]
        days = sorted({dates[i] for i, flag in enumerate(mask) if flag})
        return days

    def episode_for_day(self, day: str) -> EpisodeMarket:
        idxs = self.day_to_indices[day]
        sidecar = load_sidecar_cached(f"{self.sidecar_dir}/{day}.pt")
        day_state = self.market_state["days"].get(day)
        if day_state is None:
            raise KeyError(f"market state missing day: {day}")
        return EpisodeMarket(
            date=day,
            global_indices=idxs,
            context_source=self.context_source,
            spot_prices=self.spot_prices[idxs],
            bar_of_day=self.bar_of_day[idxs],
            sidecar=sidecar,
            max_contracts=self.max_contracts,
            context_5m_source=np.asarray(day_state["context_5m"], dtype=np.float32),
            session_state_source=np.asarray(day_state["session_state"], dtype=np.float32),
        )


@lru_cache(maxsize=4)
def load_data_bundle(
    data_path: str = "v2/data.pt",
    market_state_path: str = DEFAULT_MARKET_STATE_PATH,
) -> RLDataBundle:
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    market_state = load_market_state_cache(data_path=data_path, cache_path=market_state_path)
    dates = raw["dates"]
    day_to_indices: dict[str, list[int]] = {}
    for i, day in enumerate(dates):
        day_to_indices.setdefault(day, []).append(i)
    return RLDataBundle(
        data_path=data_path,
        raw=raw,
        market_state=market_state,
        max_contracts=int(raw["metadata"]["max_contracts_per_bar"]),
        sidecar_dir=str(raw["metadata"]["chain_sidecar_dir"]),
        day_to_indices={k: np.asarray(v, dtype=np.int32) for k, v in day_to_indices.items()},
    )
