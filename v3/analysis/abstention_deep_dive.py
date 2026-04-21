"""Deep-dive: what distinguishes the 467 abstention oracle bars from the
117 entered-right oracle bars?

Both cohorts are "session's best trade" bars by construction. The difference
is that teachers fired on entered_right and not on abstention. So any feature
that systematically differs between the two cohorts is a candidate signal for
a new late-session playbook.

Approach:
- For each cohort, extract bar-level state at the oracle bar:
  - direct features from data.pt (vwap_dist, vwap_slope, volume_ratio,
    first15_range_pct, bars_since_break_*, atm_iv, vix_regime)
  - derived features computed from the raw SPX minute close series:
    * pre_range_10: max(close) - min(close) over last 10 bars / close
    * momentum_5min: (close[T] - close[T-5]) / close[T-5]
    * momentum_20min: (close[T] - close[T-20]) / close[T-20]
    * vwap_cross_3bar: did sign(close - vwap) change in last 3 bars
    * squeeze_ratio: pre_range_10 / first15_range
- Characterize oracle_direction and time-of-session
- Compare distributions (median, iqr) across cohorts
- Flag features that differ by >2× median ratio or by >1 iqr

Output: structured comparison table + text notes on candidate signals.
"""
from __future__ import annotations

import glob
import json
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


def _feature_at(ds: V2Dataset, day: str, minute: int, name: str) -> float | None:
    if name not in ds.idx:
        return None
    day_start, day_end = ds.day_bar_range(day)
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    return float(ds.X_sim[abs_idx, ds.idx[name]])


def _derived_features(ds: V2Dataset, day: str, minute: int) -> dict[str, float | None]:
    """Compute features not stored in data.pt, from the raw close/vwap series."""
    day_start, day_end = ds.day_bar_range(day)
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return {}
    closes = np.asarray(ds.spot_prices[day_start:day_end], dtype=float)
    if minute >= len(closes):
        return {}
    close_T = float(closes[minute])

    def _pre_range(n: int) -> float | None:
        lo_i = max(0, minute - n)
        window = closes[lo_i : minute + 1]
        if len(window) < 2:
            return None
        return float(window.max() - window.min())

    def _momentum(n: int) -> float | None:
        if minute - n < 0:
            return None
        prior = float(closes[minute - n])
        if prior <= 0:
            return None
        return (close_T - prior) / prior

    pre_range_10 = _pre_range(10)
    first15_h, first15_l = ds.first15_by_day.get(day, (0.0, 0.0))
    first15_range = first15_h - first15_l if first15_h > 0 and first15_l > 0 else 0.0

    squeeze_ratio: float | None = None
    if pre_range_10 is not None and first15_range > 0:
        squeeze_ratio = pre_range_10 / first15_range

    # VWAP cross in last 3 bars — needs raw vwap; reconstruct from vwap_dist
    vwap_dist_name = "vwap_dist"
    vwap_cross_3bar: float | None = None
    if vwap_dist_name in ds.idx:
        dist_idx = ds.idx[vwap_dist_name]
        recent = [
            float(ds.X_sim[day_start + b, dist_idx])
            for b in range(max(0, minute - 3), minute + 1)
        ]
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in recent]
        vwap_cross_3bar = 1.0 if len(set(signs) - {0}) > 1 else 0.0

    return {
        "pre_range_10_pct": pre_range_10 / close_T if pre_range_10 else None,
        "momentum_5min_pct": _momentum(5),
        "momentum_20min_pct": _momentum(20),
        "squeeze_ratio": squeeze_ratio,
        "vwap_cross_3bar": vwap_cross_3bar,
    }


# Columns we'll compare across cohorts
FEATURES = [
    ("vwap_dist", "raw"),
    ("vwap_slope", "raw"),
    ("volume_ratio", "raw"),
    ("first15_range_pct", "raw"),
    ("bars_since_break_above_first15", "raw"),
    ("bars_since_break_below_first15", "raw"),
    ("atm_iv", "raw"),
    ("vix_regime", "raw"),
    ("pre_range_10_pct", "derived"),
    ("momentum_5min_pct", "derived"),
    ("momentum_20min_pct", "derived"),
    ("squeeze_ratio", "derived"),
    ("vwap_cross_3bar", "derived"),
]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(arr.mean()),
    }


def main() -> int:
    path = _latest_attribution_file()
    sessions = _load_per_session(path)
    abstention = [s for s in sessions if s.get("outcome") == "abstention"]
    entered = [s for s in sessions if s.get("outcome") == "entered_right"]
    print(f"abstention: {len(abstention)}  entered_right: {len(entered)}")

    print("Loading dataset...")
    ds = V2Dataset.load()

    def _collect(cohort: list[dict]) -> dict[str, list[float]]:
        buf: dict[str, list[float]] = defaultdict(list)
        buf["oracle_minute"] = []
        buf["oracle_direction_call"] = []
        buf["inside_first15"] = []
        for s in cohort:
            day = s["day"]
            minute = s["oracle_bar"]
            buf["oracle_minute"].append(float(minute))
            buf["oracle_direction_call"].append(
                1.0 if s.get("oracle_direction") == "call" else 0.0
            )
            day_start, day_end = ds.day_bar_range(day)
            abs_idx = day_start + minute
            if abs_idx < day_end:
                close = float(ds.spot_prices[abs_idx])
                h, l = ds.first15_by_day.get(day, (0.0, 0.0))
                if h > 0 and l > 0:
                    buf["inside_first15"].append(1.0 if l <= close <= h else 0.0)
                    buf["above_first15"].append(1.0 if close > h else 0.0)
                    buf["below_first15"].append(1.0 if close < l else 0.0)
            for name, kind in FEATURES:
                if kind == "raw":
                    v = _feature_at(ds, day, minute, name)
                else:
                    d = _derived_features(ds, day, minute)
                    v = d.get(name)
                if v is not None and np.isfinite(v):
                    buf[name].append(float(v))
        return dict(buf)

    abs_feats = _collect(abstention)
    ent_feats = _collect(entered)

    print()
    print(
        f"{'feature':<34}{'abstention':>24}{'entered_right':>24}{'delta(med)':>12}"
    )
    print("-" * 94)
    # Show metadata first
    for name in [
        "oracle_minute",
        "oracle_direction_call",
        "inside_first15",
        "above_first15",
        "below_first15",
    ]:
        a = _summary(abs_feats.get(name, []))
        e = _summary(ent_feats.get(name, []))

        def _fmt(s: dict) -> str:
            if s.get("n", 0) == 0:
                return "n=0"
            return f"med={s['median']:.3f} p25/75={s['p25']:.2f}/{s['p75']:.2f}"

        delta = (
            (a.get("median") - e.get("median"))
            if a.get("n") and e.get("n")
            else None
        )
        d_str = f"{delta:+.3f}" if delta is not None else "---"
        print(f"{name:<34}{_fmt(a):>24}{_fmt(e):>24}{d_str:>12}")

    print()
    print(
        f"{'feature':<34}{'abstention (n,med)':>24}{'entered_right (n,med)':>24}"
        f"{'ratio(abs/ent)':>18}"
    )
    print("-" * 100)
    for name, _ in FEATURES:
        a = _summary(abs_feats.get(name, []))
        e = _summary(ent_feats.get(name, []))
        am = a.get("median")
        em = e.get("median")
        ratio = None
        if am is not None and em is not None and em != 0:
            ratio = am / em
        a_str = (
            f"n={a.get('n')} med={a['median']:.4f}" if a.get("n") else "n=0"
        )
        e_str = (
            f"n={e.get('n')} med={e['median']:.4f}" if e.get("n") else "n=0"
        )
        r_str = f"{ratio:+.3f}" if ratio is not None else "---"
        print(f"{name:<34}{a_str:>24}{e_str:>24}{r_str:>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
