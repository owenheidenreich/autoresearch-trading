"""HVN / LVN investigation — do oracle bars cluster near high-volume nodes
(stall zones) or travel through low-volume nodes (air pockets)?

POC/VAH/VAL is the 70%-value summary of the histogram. HVN/LVN is the richer
structural decomposition:

- HVN (High Volume Node) = a local peak in the volume-by-price histogram.
  Price historically stalls/rotates here because it's an acceptance zone.
- LVN (Low Volume Node) = a local trough between two HVNs. Price often
  accelerates through these because there's no prior acceptance.

Hypotheses:
- Entered_right (ORC continuation): price crossing an LVN → momentum continues
  → "inside LVN" rate higher than control.
- Abstention (late-session setup): price stalled at an HVN → breakout setup
  → "near HVN" rate higher than control.

If either hypothesis holds materially, HVN/LVN is a filter worth adding.
If neither does, POC/VAH/VAL already captured the signal.

Computed from SPY volume aggregated at (day, checkpoint) into buckets, with
simple 3-bucket smoothing before peak detection. Prominence filter: a bucket
must carry >= 2% of session-to-date total volume to qualify as an HVN.
"""
from __future__ import annotations

import glob
import json
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from v3.harness.v2_adapter import V2Dataset


def _latest_attribution_file() -> str:
    files = sorted(glob.glob("v3/reference/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError
    return files[-1]


def _load_per_session(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_histogram(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    bucket_width: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (sorted_bucket_indices, volumes_per_bucket, total_vol).

    Builds a contiguous-range histogram (fills empty buckets with zero) so
    peak/trough detection works over a dense array.
    """
    if len(closes) == 0 or volumes.sum() == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0.0
    typical = (highs + lows + closes) / 3.0
    buckets: dict[int, float] = defaultdict(float)
    for i in range(len(closes)):
        if volumes[i] <= 0 or not np.isfinite(typical[i]):
            continue
        bucket_idx = int(round(typical[i] / bucket_width))
        buckets[bucket_idx] += float(volumes[i])
    if not buckets:
        return np.array([], dtype=int), np.array([], dtype=float), 0.0
    lo = min(buckets.keys())
    hi = max(buckets.keys())
    idxs = np.arange(lo, hi + 1, dtype=int)
    vols = np.array([buckets.get(int(i), 0.0) for i in idxs], dtype=float)
    return idxs, vols, float(vols.sum())


def _smooth(x: np.ndarray, w: int = 3) -> np.ndarray:
    if len(x) < w:
        return x.copy()
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def _find_hvns_lvns(
    idxs: np.ndarray,
    vols: np.ndarray,
    total_vol: float,
    hvn_prominence: float = 0.02,
    lvn_ratio: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Return (hvn_bucket_indices, lvn_bucket_indices).

    HVN: local max of smoothed histogram AND vol >= hvn_prominence * total.
    LVN: local min BETWEEN two HVNs AND vol <= lvn_ratio * min(adjacent_HVN_vols).
    """
    if len(vols) < 3 or total_vol <= 0:
        return [], []
    smoothed = _smooth(vols, w=3)
    hvn_positions: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
            and smoothed[i] >= hvn_prominence * total_vol
        ):
            hvn_positions.append(i)
    # Also consider the edges as "HVNs" if they dominate (optional; skip for
    # now — edges without neighbors on one side shouldn't be structural levels).

    # LVNs: between consecutive HVNs, find the local min
    lvn_positions: list[int] = []
    for j in range(len(hvn_positions) - 1):
        a, b = hvn_positions[j], hvn_positions[j + 1]
        if b - a < 2:
            continue
        seg = smoothed[a + 1: b]
        min_pos_local = int(np.argmin(seg))
        min_pos = a + 1 + min_pos_local
        adj_min = min(smoothed[a], smoothed[b])
        if smoothed[min_pos] <= lvn_ratio * adj_min:
            lvn_positions.append(min_pos)

    hvn_buckets = [int(idxs[p]) for p in hvn_positions]
    lvn_buckets = [int(idxs[p]) for p in lvn_positions]
    return hvn_buckets, lvn_buckets


def _build_developing_nodes_lookup(
    spy_path: str,
    bucket_pct: float = 0.0002,
    checkpoints: tuple[int, ...] = (30, 60, 90, 120),
    hvn_prominence: float = 0.02,
    lvn_ratio: float = 0.5,
) -> dict[tuple[str, int], dict[str, object]]:
    """Per (day, checkpoint_minute): list of HVN/LVN prices (SPY space) + ref close."""
    import pandas as pd

    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)

    out: dict[tuple[str, int], dict[str, object]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        mean_price = float(g["close"].mean())
        bucket_width = max(mean_price * bucket_pct, 0.01)
        for cp in checkpoints:
            end = min(cp, len(g))
            if end < 10:
                continue
            sub = g.iloc[:end]
            idxs, vols, total = _build_histogram(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float),
                sub["close"].to_numpy(dtype=float),
                sub["volume"].to_numpy(dtype=float),
                bucket_width,
            )
            hvn_buckets, lvn_buckets = _find_hvns_lvns(
                idxs, vols, total, hvn_prominence, lvn_ratio
            )
            hvn_prices = [b * bucket_width for b in hvn_buckets]
            lvn_prices = [b * bucket_width for b in lvn_buckets]
            out[(day, cp)] = {
                "hvns": hvn_prices,
                "lvns": lvn_prices,
                "spy_close_at_cp": float(sub.iloc[-1]["close"]),
                "bucket_width": bucket_width,
                "n_hvn": len(hvn_prices),
                "n_lvn": len(lvn_prices),
            }
    return out


def _omar_range_for_day(spx_1min_path: str) -> dict[str, float]:
    import pandas as pd

    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, float] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[day] = max(h - l, 0.01)
    return out


def _nearest(
    spx_close: float, levels_spy: list[float], spy_ref: float, omar_range: float
) -> float | None:
    """Min |dist to any level| in OMAR units. None if empty."""
    if not levels_spy or omar_range <= 0 or spy_ref <= 0:
        return None
    scale = spx_close / spy_ref
    diffs = [abs(spx_close - lv * scale) / omar_range for lv in levels_spy]
    return float(min(diffs))


def _is_inside_lvn(
    spx_close: float,
    hvns_spy: list[float],
    lvns_spy: list[float],
    spy_ref: float,
) -> bool:
    """True if the nearest HVN-or-LVN is an LVN AND price sits between two HVNs."""
    if not lvns_spy or not hvns_spy or spy_ref <= 0:
        return False
    scale = spx_close / spy_ref
    hvns_spx = sorted(lv * scale for lv in hvns_spy)
    lvns_spx = sorted(lv * scale for lv in lvns_spy)
    # bracketing HVNs
    below = [h for h in hvns_spx if h <= spx_close]
    above = [h for h in hvns_spx if h >= spx_close]
    if not below or not above:
        return False
    h_lo, h_hi = below[-1], above[0]
    # Is there an LVN between h_lo and h_hi?
    inner_lvns = [lv for lv in lvns_spx if h_lo <= lv <= h_hi]
    if not inner_lvns:
        return False
    # Distance to nearest HVN vs nearest inner LVN
    dist_hvn = min(abs(spx_close - h_lo), abs(spx_close - h_hi))
    dist_lvn = min(abs(spx_close - lv) for lv in inner_lvns)
    return dist_lvn < dist_hvn


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"

    print("Computing intraday HVN/LVN nodes at checkpoints 30/60/90/120...")
    dev_nodes = _build_developing_nodes_lookup(
        str(spy_path), checkpoints=(30, 60, 90, 120)
    )
    print(f"  {len(dev_nodes)} (day, checkpoint) records")

    # Node density sanity check
    hvn_counts = [int(v["n_hvn"]) for v in dev_nodes.values()]
    lvn_counts = [int(v["n_lvn"]) for v in dev_nodes.values()]
    print(f"  HVN count per checkpoint: median={np.median(hvn_counts):.1f} "
          f"p25={np.percentile(hvn_counts, 25):.1f} p75={np.percentile(hvn_counts, 75):.1f}")
    print(f"  LVN count per checkpoint: median={np.median(lvn_counts):.1f} "
          f"p25={np.percentile(lvn_counts, 25):.1f} p75={np.percentile(lvn_counts, 75):.1f}")

    omar_map = _omar_range_for_day(str(spx_path))
    sorted_days = sorted(set(d for (d, _) in dev_nodes.keys()))

    sessions = _load_per_session(_latest_attribution_file())
    cohorts: dict[str, list[dict]] = {
        "entered_right": [s for s in sessions if s.get("outcome") == "entered_right"],
        "abstention": [s for s in sessions if s.get("outcome") == "abstention"],
        "side_error": [s for s in sessions if s.get("outcome") == "side_error"],
    }
    rng = random.Random(11)
    control = [{"day": d, "oracle_bar": rng.randint(40, 120)} for d in sorted_days]
    cohorts["control"] = control

    print()
    print("=== Oracle-bar proximity to HVN / LVN (developing VP checkpoint ≤ bar) ===")
    print(
        f"{'cohort':<18}{'n':>6}{'med_dist_HVN':>14}{'pct_near_HVN':>14}"
        f"{'med_dist_LVN':>14}{'pct_near_LVN':>14}{'pct_inside_LVN':>16}"
    )
    print("-" * 96)

    summary: dict[str, dict] = {}
    for cohort_name, rows in cohorts.items():
        d_hvns: list[float] = []
        d_lvns: list[float] = []
        inside_lvn_flags: list[int] = []
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            cp = None
            for c_try in (120, 90, 60, 30):
                if c_try < minute and (day, c_try) in dev_nodes:
                    cp = c_try
                    break
            if cp is None:
                continue
            nodes = dev_nodes[(day, cp)]
            omar = omar_map.get(day)
            if omar is None:
                continue
            try:
                day_start, day_end = ds.day_bar_range(day)
            except ValueError:
                continue
            abs_idx = day_start + minute
            if abs_idx >= day_end:
                continue
            spx_close = float(ds.spot_prices[abs_idx])
            d_h = _nearest(spx_close, nodes["hvns"], nodes["spy_close_at_cp"], omar)
            d_l = _nearest(spx_close, nodes["lvns"], nodes["spy_close_at_cp"], omar)
            inside = _is_inside_lvn(
                spx_close, nodes["hvns"], nodes["lvns"], nodes["spy_close_at_cp"]
            )
            if d_h is not None:
                d_hvns.append(d_h)
            if d_l is not None:
                d_lvns.append(d_l)
            inside_lvn_flags.append(1 if inside else 0)

        if not d_hvns:
            print(f"{cohort_name:<18}{0:>6}")
            continue
        dh = np.asarray(d_hvns)
        dl = np.asarray(d_lvns) if d_lvns else np.array([])
        flags = np.asarray(inside_lvn_flags) if inside_lvn_flags else np.array([])
        med_hvn = float(np.median(dh))
        pct_near_hvn = 100 * float(np.mean(dh <= 0.5))
        med_lvn = float(np.median(dl)) if dl.size else float("nan")
        pct_near_lvn = 100 * float(np.mean(dl <= 0.5)) if dl.size else 0.0
        pct_inside = 100 * float(np.mean(flags)) if flags.size else 0.0
        summary[cohort_name] = {
            "n": len(dh),
            "med_hvn": med_hvn,
            "pct_near_hvn": pct_near_hvn,
            "med_lvn": med_lvn,
            "pct_near_lvn": pct_near_lvn,
            "pct_inside_lvn": pct_inside,
        }
        print(
            f"{cohort_name:<18}{len(dh):>6}"
            f"{med_hvn:>+14.3f}{pct_near_hvn:>13.1f}%"
            f"{med_lvn:>+14.3f}{pct_near_lvn:>13.1f}%"
            f"{pct_inside:>15.1f}%"
        )

    print()
    print("=== Enrichment ratios (cohort / control) ===")
    ctrl = summary.get("control")
    if ctrl:
        for cohort_name in ("abstention", "entered_right", "side_error"):
            s = summary.get(cohort_name)
            if not s:
                continue
            r_hvn = s["pct_near_hvn"] / max(ctrl["pct_near_hvn"], 1e-9)
            r_lvn = s["pct_near_lvn"] / max(ctrl["pct_near_lvn"], 1e-9)
            r_in = s["pct_inside_lvn"] / max(ctrl["pct_inside_lvn"], 1e-9)
            print(
                f"  {cohort_name:<18}: near_HVN={r_hvn:.2f}×  "
                f"near_LVN={r_lvn:.2f}×  inside_LVN={r_in:.2f}×"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
