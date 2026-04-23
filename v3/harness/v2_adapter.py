"""v2 → v3 adapter.

Reads v2/data.pt (unnormalized X_sim features, spot prices, dates, bar_of_day)
and the per-day sidecars in v2/data_sidecars/, and emits v3 inputs:
- `BarContext` for each bar (for teacher evaluation)
- `list[ChainRowInput]` for each bar (for the guardrail-and-select path)
- extra per-bar context (vix, atm_iv, iv_percentile) passed to the builder

Raw `first15_high` / `first15_low` are computed from the SPX 1-min cache
because they are not stored as features — the v2 pipeline stores derived
ratios (`first15_range_pct`, `first15_close_position`) but not the absolute
levels that the failed-break teacher needs for in-range detection.

Feature-index constants are pinned at module load time by looking up names
in the dataset's `feature_names`. If v2's pipeline reorders features, this
module notices at load time instead of silently pulling wrong columns.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from v2.core.chain_data import (
    SIDECAR_DIR_DEFAULT,
    contract_snapshot,
    load_sidecar_cached,
)
from v2.core import market_structure
from v3.logger.builder import ChainRowInput
from v3.teachers.base import BarContext


DATA_PT_DEFAULT = os.path.join("v2", "data.pt")
SPX_1MIN_DEFAULT = os.path.expanduser(
    "~/.cache/autoresearch-trading/data/spx_1min.pkl"
)
SPY_1MIN_DEFAULT = os.path.expanduser(
    "~/.cache/autoresearch-trading/data/spy_1min.pkl"
)


# Contract-feature column indices in sidecar row_features.
# Matches v2.core.chain_data.CONTRACT_FEATURE_FIELDS as of schema v5.
CIDX_VALID = 0
CIDX_STRIKE = 1
CIDX_RIGHT_IS_PUT = 2
CIDX_MID = 3
CIDX_SPREAD_FRAC = 4
CIDX_DELTA = 8
CIDX_THETA_TO_PREMIUM = 19
CIDX_GAMMA_DOLLAR = 21


@dataclass
class V2Dataset:
    """Loaded v2 dataset artifacts. One instance per process."""

    X_sim: np.ndarray              # (n_bars, n_features) unnormalized features
    spot_prices: np.ndarray         # (n_bars,) underlying close
    dates: list[str]                # per-bar date strings
    bar_of_day: np.ndarray          # (n_bars,) minute of session [0, 389]
    feature_names: list[str]
    first15_by_day: dict[str, tuple[float, float]]
    sidecar_dir: str

    # Per-feature-name index cache, resolved at load time.
    idx: dict[str, int]
    # Per-day bar range cache (populated on demand).
    _day_ranges: dict[str, tuple[int, int]]
    # SPY-derived per-minute VWAP / std arrays keyed by date string. Loaded
    # once at .load() time so every BarContext can carry sigma_pos without a
    # separate adapter computing it inline (research-vs-production drift).
    spy_vwap: dict[str, dict[str, np.ndarray]] = None  # type: ignore[assignment]

    @classmethod
    def load(
        cls,
        data_pt_path: str = DATA_PT_DEFAULT,
        spx_1min_path: str = SPX_1MIN_DEFAULT,
        sidecar_dir: str = SIDECAR_DIR_DEFAULT,
        spy_1min_path: str = SPY_1MIN_DEFAULT,
    ) -> "V2Dataset":
        d = torch.load(data_pt_path, map_location="cpu", weights_only=False)
        X_sim = (
            d["X_sim"].numpy() if hasattr(d["X_sim"], "numpy") else np.asarray(d["X_sim"])
        )
        spot = np.asarray(d["spot_prices"])
        dates = [str(x) for x in d["dates"]]
        bod = (
            d["bar_of_day"].numpy()
            if hasattr(d["bar_of_day"], "numpy")
            else np.asarray(d["bar_of_day"])
        )
        names = list(d["feature_names"])

        required = {
            "vwap_dist",
            "vwap_slope",
            "volume_ratio",
            "first15_range_pct",
            "bars_since_break_above_first15",
            "bars_since_break_below_first15",
            "vix_regime",
            "atm_iv",
            "iv_percentile",
        }
        # Optional features from W2a — present on FEATURE_CONTRACT_VERSION >= v2.2.
        # v3's BarContext does not consume these directly (sigma_pos still comes
        # from the shared helper so old and new data.pt both work). We track
        # them in `idx` opportunistically so downstream research scripts can
        # read them from the flat array without another feature_names lookup.
        optional = {
            "sigma_pos",
            "omar_retest_dist_norm",
            "omar_range_pct",
            "last10_range_over_omar",
            "inside_first15",
            "late_window_40_120_flag",
            "omar_mid_pos_units",
            "last10_break_state",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(
                f"data.pt missing required features: {sorted(missing)}. "
                "Regenerate with the v3 compute_features.py."
            )
        idx = {name: names.index(name) for name in required}
        for name in optional:
            if name in names:
                idx[name] = names.index(name)

        spx_df = pickle.load(open(spx_1min_path, "rb"))
        first15: dict[str, tuple[float, float]] = {}
        for day, group in spx_df.groupby("date"):
            first15_rows = group.head(15)
            first15[str(day)] = (
                float(first15_rows["spx_high"].max()),
                float(first15_rows["spx_low"].min()),
            )

        if not os.path.exists(spy_1min_path):
            raise RuntimeError(
                f"SPY 1-min cache not found at {spy_1min_path}. "
                "sigma_pos requires SPY-derived VWAP; cannot load v3 dataset without it."
            )
        spy_vwap = market_structure.build_spy_vwap(spy_1min_path)

        return cls(
            X_sim=X_sim,
            spot_prices=spot,
            dates=dates,
            bar_of_day=bod,
            feature_names=names,
            first15_by_day=first15,
            sidecar_dir=sidecar_dir,
            idx=idx,
            _day_ranges={},
            spy_vwap=spy_vwap,
        )

    def day_bar_range(self, day: str) -> tuple[int, int]:
        cached = self._day_ranges.get(day)
        if cached is not None:
            return cached
        start = self.dates.index(day)
        end = start
        while end < len(self.dates) and self.dates[end] == day:
            end += 1
        self._day_ranges[day] = (start, end)
        return start, end


def bar_context_from_dataset(dataset: V2Dataset, abs_idx: int) -> BarContext:
    """Build a `BarContext` for one absolute bar index.

    Recovers raw VWAP from `vwap_dist = (close - vwap) / close` stored as a
    feature. Raw `first15_high/low` come from the per-day SPX-1min lookup.
    `sigma_pos` comes from the shared `v2.core.market_structure` helper using
    the SPY cache loaded at dataset.load() time; falls back to None when the
    minute is out of range or SPY data is missing for the day.
    """
    row = dataset.X_sim[abs_idx]
    day = dataset.dates[abs_idx]
    close = float(dataset.spot_prices[abs_idx])
    vwap_dist = float(row[dataset.idx["vwap_dist"]])
    vwap = close * (1.0 - vwap_dist) if close > 0 else close
    f15_high, f15_low = dataset.first15_by_day.get(day, (0.0, 0.0))
    minute = int(dataset.bar_of_day[abs_idx])
    spy_day = dataset.spy_vwap.get(day) if dataset.spy_vwap is not None else None
    sigma = market_structure.sigma_pos(spy_day, minute, close) if spy_day is not None else None
    return BarContext(
        minute_of_session=minute,
        close=close,
        vwap=vwap,
        vwap_slope=float(row[dataset.idx["vwap_slope"]]),
        volume_ratio=float(row[dataset.idx["volume_ratio"]]),
        first15_high=f15_high,
        first15_low=f15_low,
        first15_range_pct=float(row[dataset.idx["first15_range_pct"]]),
        bars_since_break_above_first15=int(row[dataset.idx["bars_since_break_above_first15"]]),
        bars_since_break_below_first15=int(row[dataset.idx["bars_since_break_below_first15"]]),
        sigma_pos=sigma,
    )


def extra_context_from_dataset(dataset: V2Dataset, abs_idx: int) -> dict[str, float]:
    row = dataset.X_sim[abs_idx]
    return {
        "vix": float(row[dataset.idx["vix_regime"]]),
        "atm_iv": float(row[dataset.idx["atm_iv"]]),
        "iv_percentile": float(row[dataset.idx["iv_percentile"]]),
    }


def chain_from_sidecar(sidecar: dict, local_bar: int) -> list[ChainRowInput]:
    """Build a list of `ChainRowInput` from one bar's sidecar snapshot.

    Invalid contracts (contract_valid < 0.5) are dropped — the plan keeps
    those bars in the log via the upstream `eligible` flag, but we don't
    carry dead contract rows through into the guardrail filter.
    """
    feats, _, _ = contract_snapshot(sidecar, local_bar)
    rows: list[ChainRowInput] = []
    if feats.shape[0] == 0:
        return rows
    for i in range(feats.shape[0]):
        r = feats[i]
        if r[CIDX_VALID] < 0.5:
            continue
        mid = float(r[CIDX_MID])
        if not (mid > 0):
            continue
        rows.append(
            ChainRowInput(
                strike=float(r[CIDX_STRIKE]),
                right="P" if r[CIDX_RIGHT_IS_PUT] > 0.5 else "C",
                mid=mid,
                delta=float(r[CIDX_DELTA]),
                spread_fraction=float(r[CIDX_SPREAD_FRAC]),
                contract_valid=True,
                gamma_dollar=float(r[CIDX_GAMMA_DOLLAR]),
                theta_to_premium=float(r[CIDX_THETA_TO_PREMIUM]),
            )
        )
    return rows


@lru_cache(maxsize=16)
def _sidecar_path(sidecar_dir: str, day: str) -> str:
    return os.path.join(sidecar_dir, f"{day}.pt")


def load_day_sidecar(dataset: V2Dataset, day: str) -> Optional[dict]:
    path = _sidecar_path(dataset.sidecar_dir, day)
    if not os.path.exists(path):
        return None
    return load_sidecar_cached(path)


def has_chain_snapshot(sidecar: dict, local_bar: int) -> bool:
    """Return True iff the sidecar has any contract rows for this bar.

    Distinguishes "bar is outside v2 chain coverage" (no rows at all) from
    "bar has contracts but all are invalid" (rows with contract_valid=False).
    Only the first case should be excluded from the eligible-bar log — the
    plan's eligibility rule says "valid ... chain snapshot," i.e. the chain
    existed at this bar, not that any contract passed any filter.
    """
    ptrs = sidecar["bar_ptrs"]
    if local_bar + 1 >= len(ptrs):
        return False
    return int(ptrs[local_bar + 1]) > int(ptrs[local_bar])
