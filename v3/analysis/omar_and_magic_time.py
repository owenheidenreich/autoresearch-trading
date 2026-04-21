"""Empirical investigation: OMAR, MAGIC TIME, and daily-scale calibration.

Answers three questions:

1. Is OMAR (first-minute range) a useful scale metric for "tight
   consolidation"? Compare against first15_range and ATR-14 as alternatives.
   A good scale metric should:
   - Vary meaningfully across days (big range on high-vol days, small on calm)
   - Correlate with intraday movement magnitude (days with big OMAR should
     tend to have bigger intraday moves overall)
   - Produce a STABLE "squeeze_ratio" (pre-entry 10-bar range / scale) across
     VIX regimes — if scaling works, a "tight squeeze" should mean the same
     relative thing regardless of VIX level.

2. Does MAGIC TIME (9:55-10:10 ET = minutes 25-40) concentrate entered_right
   bars? Does abstention concentrate OUTSIDE this window?

3. Given the best scale metric, what squeeze_ratio threshold captures
   abstention oracle bars without firing on every random bar?
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from v3.harness.v2_adapter import V2Dataset


MAGIC_TIME_LO = 25  # 09:55 ET
MAGIC_TIME_HI = 40  # 10:10 ET


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


def _build_omar_and_scales(spx_1min_path: str) -> dict[str, dict[str, float]]:
    """For each day, compute OMAR, first15_range, ATR-proxy.

    OMAR = high - low of the first 1-minute bar (9:30-9:31 ET).
    first15_range = high[0..14] - low[0..14].
    atr15 = mean true range over first 15 minutes (proxy for expected
        per-minute movement).

    Returns dict keyed by date string.
    """
    df = pickle.load(open(spx_1min_path, "rb"))
    scales: dict[str, dict[str, float]] = {}
    for day, group in df.groupby("date"):
        g = group.reset_index(drop=True)
        if len(g) < 15:
            continue
        omar_range = float(g.iloc[0]["spx_high"] - g.iloc[0]["spx_low"])
        first15 = g.head(15)
        f15_range = float(first15["spx_high"].max() - first15["spx_low"].min())
        # True range for each of first 15 bars
        tr = []
        for i in range(1, 15):
            h, l = float(g.iloc[i]["spx_high"]), float(g.iloc[i]["spx_low"])
            pc = float(g.iloc[i - 1]["spx_close"])
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr15 = float(np.mean(tr)) if tr else float(g.iloc[0]["spx_high"] - g.iloc[0]["spx_low"])
        close_open = float(g.iloc[0]["spx_close"])
        scales[str(day)] = {
            "omar_range": omar_range,
            "first15_range": f15_range,
            "atr15": atr15,
            "open_close": close_open,
            "omar_pct": omar_range / close_open if close_open > 0 else 0.0,
            "first15_pct": f15_range / close_open if close_open > 0 else 0.0,
            "atr15_pct": atr15 / close_open if close_open > 0 else 0.0,
        }
    return scales


def _correlation_with_vix(scales: dict[str, dict[str, float]], ds: V2Dataset) -> dict[str, float]:
    """Correlate each scale with same-day VIX to see which tracks daily vol best."""
    vix_values: dict[str, float] = {}
    for day in scales.keys():
        try:
            day_start, _ = ds.day_bar_range(day)
        except ValueError:
            continue
        vix = float(ds.X_sim[day_start, ds.idx["vix_regime"]])
        vix_values[day] = vix

    common = sorted(set(scales.keys()) & set(vix_values.keys()))
    if not common:
        return {}
    vix_arr = np.asarray([vix_values[d] for d in common])
    out = {}
    for col in ("omar_pct", "first15_pct", "atr15_pct"):
        v = np.asarray([scales[d][col] for d in common])
        mask = np.isfinite(vix_arr) & np.isfinite(v)
        if mask.sum() < 10:
            out[col] = float("nan")
            continue
        out[col] = float(np.corrcoef(vix_arr[mask], v[mask])[0, 1])
    return out


def _squeeze_by_scale(
    sessions: list[dict],
    ds: V2Dataset,
    scales: dict[str, dict[str, float]],
) -> dict[str, list[float]]:
    """For each oracle bar, compute pre_range_10 / each scale candidate."""
    out: dict[str, list[float]] = {"omar": [], "first15": [], "atr15": []}
    for s in sessions:
        day = s["day"]
        minute = s["oracle_bar"]
        sc = scales.get(day)
        if sc is None:
            continue
        try:
            day_start, day_end = ds.day_bar_range(day)
        except ValueError:
            continue
        abs_idx = day_start + minute
        if abs_idx >= day_end:
            continue
        closes = np.asarray(ds.spot_prices[day_start:day_end], dtype=float)
        lo_i = max(0, minute - 10)
        win = closes[lo_i : minute + 1]
        if len(win) < 2:
            continue
        pre_range = float(win.max() - win.min())
        if sc["omar_range"] > 0:
            out["omar"].append(pre_range / sc["omar_range"])
        if sc["first15_range"] > 0:
            out["first15"].append(pre_range / sc["first15_range"])
        if sc["atr15"] > 0:
            out["atr15"].append(pre_range / sc["atr15"])
    return out


def main() -> int:
    print("Loading dataset + SPX 1-min cache...")
    ds = V2Dataset.load()
    scales = _build_omar_and_scales(
        str(Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl")
    )
    print(f"  built scales for {len(scales)} days")

    # --- Q1: OMAR vs first15 vs ATR — which tracks daily vol best? ---
    print()
    print("=== Q1: Which scale correlates with VIX-regime (daily vol proxy)? ===")
    corrs = _correlation_with_vix(scales, ds)
    for col, r in corrs.items():
        print(f"  corr({col}, vix_regime) = {r:+.3f}")

    # Distribution of each scale (raw dollars)
    print()
    print("=== Scale distributions (raw dollar range across days) ===")
    for col in ("omar_range", "first15_range", "atr15"):
        vals = np.asarray([scales[d][col] for d in scales])
        print(
            f"  {col:<16} median={np.median(vals):7.2f}  "
            f"p10={np.percentile(vals, 10):7.2f}  p90={np.percentile(vals, 90):7.2f}  "
            f"ratio_p90/p10={np.percentile(vals, 90)/max(np.percentile(vals, 10), 0.01):6.1f}x"
        )

    # --- Q2: MAGIC TIME concentration ---
    path = _latest_attribution_file()
    sessions = _load_per_session(path)
    abstention = [s for s in sessions if s.get("outcome") == "abstention"]
    entered_right = [s for s in sessions if s.get("outcome") == "entered_right"]
    side_error = [s for s in sessions if s.get("outcome") == "side_error"]

    def _pct_in_magic(rows: list[dict]) -> tuple[int, int, float]:
        n_in = sum(1 for r in rows if MAGIC_TIME_LO <= r["oracle_bar"] <= MAGIC_TIME_HI)
        return n_in, len(rows), 100 * n_in / max(len(rows), 1)

    print()
    print(f"=== Q2: MAGIC TIME window [{MAGIC_TIME_LO}, {MAGIC_TIME_HI}] = 09:55-10:10 ET ===")
    for name, rows in (
        ("entered_right", entered_right),
        ("abstention", abstention),
        ("side_error", side_error),
    ):
        n_in, n_total, pct = _pct_in_magic(rows)
        print(f"  {name:<16} {n_in}/{n_total} oracle bars in MAGIC TIME ({pct:.1f}%)")
    print()
    # Also report full distribution of oracle-bar minute for each cohort
    print("Oracle-bar minute distribution (p10, median, p90):")
    for name, rows in (
        ("entered_right", entered_right),
        ("abstention", abstention),
        ("side_error", side_error),
    ):
        minutes = np.asarray([r["oracle_bar"] for r in rows])
        print(
            f"  {name:<16} p10={np.percentile(minutes, 10):5.0f}  "
            f"median={np.percentile(minutes, 50):5.0f}  p90={np.percentile(minutes, 90):5.0f}"
        )

    # --- Q3: Squeeze ratio by scale choice, for abstention vs entered_right ---
    print()
    print("=== Q3: Pre-entry 10-bar range, scaled by each candidate ===")
    print("Lower ratio = tighter squeeze going into the oracle bar.")
    print()
    print(f"{'cohort':<20}{'scale':<12}{'n':>6}{'p10':>10}{'p25':>10}{'median':>10}{'p75':>10}{'p90':>10}")

    for cohort_name, rows in (("abstention", abstention), ("entered_right", entered_right)):
        squeeze = _squeeze_by_scale(rows, ds, scales)
        for scale_name, vals in squeeze.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            print(
                f"{cohort_name:<20}{scale_name:<12}{len(arr):>6}"
                f"{np.percentile(arr, 10):>10.2f}"
                f"{np.percentile(arr, 25):>10.2f}"
                f"{np.percentile(arr, 50):>10.2f}"
                f"{np.percentile(arr, 75):>10.2f}"
                f"{np.percentile(arr, 90):>10.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
