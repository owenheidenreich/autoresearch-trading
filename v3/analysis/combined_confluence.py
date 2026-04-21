"""Combined-confluence test: are the three late-session teacher filters
(intraday VP, VWAP direction, OMAR retest) independent or redundant?

If independent: stacking them produces compound enrichment meaningfully
greater than the strongest single filter (1.32× from intraday VP).
If redundant: compound enrichment is ~= max(individual), meaning we could
drop one or two filters without losing signal.

Measure per cohort (abstention, entered_right, control):

- Share satisfying filter A: intraday VP proximity (|d| ≤ 0.5× OMAR from POC/VAH/VAL)
- Share satisfying filter B: VWAP direction tilt
  (for call candidates: sigma_pos ≤ +0.5; for put candidates: sigma_pos ≥ -0.5)
- Share satisfying filter C: OMAR retest (|d to OMAR H/L/M| ≤ 0.5× OMAR)
- Share satisfying A ∩ B
- Share satisfying A ∩ C
- Share satisfying B ∩ C
- Share satisfying A ∩ B ∩ C
- Enrichment ratios: each stacked filter vs control baseline

Also: 2×2 contingency check on whether A|B and A|C are statistically
associated (Pearson's χ² on the control cohort).
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


def _compute_vp(
    highs, lows, closes, volumes, bucket_width: float
):
    from collections import defaultdict as dd
    typical = (highs + lows + closes) / 3.0
    buckets: dict[int, float] = dd(float)
    for i in range(len(closes)):
        if volumes[i] <= 0 or not np.isfinite(typical[i]):
            continue
        b = int(round(typical[i] / bucket_width))
        buckets[b] += float(volumes[i])
    if not buckets:
        return None, None, None
    sorted_idx = sorted(buckets.keys())
    poc_idx = max(buckets.items(), key=lambda kv: kv[1])[0]
    poc_pos = sorted_idx.index(poc_idx)
    target = sum(buckets.values()) * 0.70
    inc = buckets[poc_idx]
    lo_pos = poc_pos
    hi_pos = poc_pos
    while inc < target:
        lv = buckets[sorted_idx[lo_pos - 1]] if lo_pos > 0 else -1
        hv = buckets[sorted_idx[hi_pos + 1]] if hi_pos < len(sorted_idx) - 1 else -1
        if lv < 0 and hv < 0:
            break
        if lv >= hv and lo_pos > 0:
            lo_pos -= 1
            inc += buckets[sorted_idx[lo_pos]]
        elif hi_pos < len(sorted_idx) - 1:
            hi_pos += 1
            inc += buckets[sorted_idx[hi_pos]]
        else:
            break
    poc = poc_idx * bucket_width
    val = sorted_idx[lo_pos] * bucket_width
    vah = sorted_idx[hi_pos] * bucket_width
    return poc, vah, val


def _build_developing_vp_lookup(
    spy_path: str, checkpoints=(30, 60, 90, 120)
) -> dict[tuple[str, int], dict[str, float]]:
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        bucket_width = max(float(g["close"].mean()) * 0.0002, 0.01)
        for cp in checkpoints:
            end = min(cp, len(g))
            if end < 10:
                continue
            sub = g.iloc[:end]
            poc, vah, val = _compute_vp(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float),
                sub["close"].to_numpy(dtype=float),
                sub["volume"].to_numpy(dtype=float),
                bucket_width,
            )
            if poc is None:
                continue
            out[(day, cp)] = {
                "poc": poc,
                "vah": vah,
                "val": val,
                "spy_close_at_cp": float(sub.iloc[-1]["close"]),
            }
    return out


def _build_omar_map(spx_1min_path: str) -> dict[str, dict[str, float]]:
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[day] = {
            "high": h,
            "low": l,
            "mid": (h + l) / 2.0,
            "range": max(h - l, 0.01),
        }
    return out


def _build_spy_vwap(spy_path: str) -> dict[str, dict[str, np.ndarray]]:
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        vwap = g["vwap"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        sq_dev = (close - vwap) ** 2
        cum_v = np.maximum(np.cumsum(volume), 1.0)
        cum_sq = np.cumsum(volume * sq_dev)
        std_run = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        out[day] = {"vwap": vwap, "close": close, "std": std_run}
    return out


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    omar_map = _build_omar_map(str(spx_path))
    dev_vp = _build_developing_vp_lookup(str(spy_path))
    spy_vwap = _build_spy_vwap(str(spy_path))

    sessions = _load_per_session(_latest_attribution_file())
    cohorts = {
        "entered_right": [s for s in sessions if s.get("outcome") == "entered_right"],
        "abstention": [s for s in sessions if s.get("outcome") == "abstention"],
        "side_error": [s for s in sessions if s.get("outcome") == "side_error"],
    }
    rng = random.Random(101)
    control = [
        {"day": d, "oracle_bar": rng.randint(40, 120), "oracle_direction": "call"}
        for d in sorted(set(ds.dates))
    ]
    cohorts["control"] = control

    # For each bar: compute filter A, B, C satisfaction
    def evaluate_bar(s: dict) -> dict[str, bool] | None:
        day = s["day"]
        minute = s["oracle_bar"]
        direction = s.get("oracle_direction", "call")  # for control, direction doesn't matter for filters A/C; B uses direction
        try:
            day_start, day_end = ds.day_bar_range(day)
        except ValueError:
            return None
        abs_idx = day_start + minute
        if abs_idx >= day_end:
            return None
        spx_close = float(ds.spot_prices[abs_idx])

        # Filter A: intraday VP proximity
        cp = None
        for c_try in (120, 90, 60, 30):
            if c_try < minute and (day, c_try) in dev_vp:
                cp = c_try
                break
        A = False
        if cp is not None:
            vp = dev_vp[(day, cp)]
            spy_ref = vp["spy_close_at_cp"]
            omar = omar_map.get(day)
            if omar and spy_ref > 0:
                r = omar["range"]
                ratio = spx_close / spy_ref
                for k in ("poc", "vah", "val"):
                    lv_spx = vp[k] * ratio
                    if abs(spx_close - lv_spx) / r <= 0.5:
                        A = True
                        break

        # Filter B: VWAP direction tilt (direction-dependent)
        B = False
        spy_day = spy_vwap.get(day)
        if spy_day and minute < len(spy_day["vwap"]):
            spy_vwap_v = float(spy_day["vwap"][minute])
            spy_close_v = float(spy_day["close"][minute])
            spy_std_v = float(spy_day["std"][minute])
            if all(v > 0 for v in (spy_vwap_v, spy_close_v, spy_std_v, spx_close)):
                ratio = spx_close / spy_close_v
                spx_vwap = spy_vwap_v * ratio
                spx_std = spy_std_v * ratio
                sigma_pos = (spx_close - spx_vwap) / spx_std
                if direction == "call":
                    B = sigma_pos <= 0.5  # "at or below VWAP + 0.5σ"
                else:
                    B = sigma_pos >= -0.5

        # Filter C: OMAR retest proximity
        C = False
        omar = omar_map.get(day)
        if omar:
            r = omar["range"]
            for k in ("high", "low", "mid"):
                if abs(spx_close - omar[k]) / r <= 0.5:
                    C = True
                    break
        return {"A": A, "B": B, "C": C}

    # Aggregate per cohort
    print()
    print("=== Filter-satisfaction rates and stacked combinations ===")
    print()
    header = (
        f"{'cohort':<16}{'n':>6}"
        f"{'A (VP)':>10}{'B (VWAP)':>11}{'C (OMAR)':>11}"
        f"{'A∩B':>8}{'A∩C':>8}{'B∩C':>8}{'A∩B∩C':>10}"
    )
    print(header)
    print("-" * len(header))

    summary: dict[str, dict[str, float]] = {}
    for cohort_name, rows in cohorts.items():
        a = b = c = 0
        ab = ac = bc = abc = 0
        n = 0
        for s in rows:
            res = evaluate_bar(s)
            if res is None:
                continue
            n += 1
            if res["A"]:
                a += 1
            if res["B"]:
                b += 1
            if res["C"]:
                c += 1
            if res["A"] and res["B"]:
                ab += 1
            if res["A"] and res["C"]:
                ac += 1
            if res["B"] and res["C"]:
                bc += 1
            if res["A"] and res["B"] and res["C"]:
                abc += 1
        summary[cohort_name] = {
            "n": n,
            "A": 100 * a / n if n else 0,
            "B": 100 * b / n if n else 0,
            "C": 100 * c / n if n else 0,
            "AB": 100 * ab / n if n else 0,
            "AC": 100 * ac / n if n else 0,
            "BC": 100 * bc / n if n else 0,
            "ABC": 100 * abc / n if n else 0,
        }
        s_row = summary[cohort_name]
        print(
            f"{cohort_name:<16}{n:>6}"
            f"{s_row['A']:>9.1f}%{s_row['B']:>10.1f}%{s_row['C']:>10.1f}%"
            f"{s_row['AB']:>7.1f}%{s_row['AC']:>7.1f}%{s_row['BC']:>7.1f}%{s_row['ABC']:>9.1f}%"
        )

    # Independence test: if A, B, C were independent,
    # P(A ∩ B) = P(A) × P(B) on the control cohort
    print()
    print("=== Independence check (control cohort) ===")
    ctrl = summary.get("control")
    if ctrl:
        for (f1, f2, comb) in [("A", "B", "AB"), ("A", "C", "AC"), ("B", "C", "BC")]:
            expected = ctrl[f1] * ctrl[f2] / 100
            observed = ctrl[comb]
            ratio = observed / expected if expected > 0 else float("nan")
            print(
                f"  {f1}∩{f2}: observed={observed:.1f}%  expected-if-independent={expected:.1f}%  "
                f"ratio={ratio:.2f}× ({'independent' if abs(ratio-1) < 0.1 else 'NOT independent'})"
            )
        # ABC joint independence
        exp_abc = ctrl["A"] * ctrl["B"] * ctrl["C"] / 10000
        obs_abc = ctrl["ABC"]
        print(
            f"  A∩B∩C: observed={obs_abc:.1f}%  expected-if-independent={exp_abc:.2f}%  "
            f"ratio={obs_abc / max(exp_abc, 0.01):.2f}×"
        )

    # Compute enrichment vs control for each filter combination
    print()
    print("=== Enrichment vs control (cohort_rate / control_rate) ===")
    if ctrl:
        print(
            f"{'cohort':<16}"
            f"{'A':>8}{'B':>8}{'C':>8}"
            f"{'AB':>8}{'AC':>8}{'BC':>8}{'ABC':>10}"
        )
        for cohort_name in ("abstention", "entered_right", "side_error"):
            s = summary.get(cohort_name, {})
            if not s:
                continue
            print(
                f"{cohort_name:<16}"
                f"{s.get('A', 0) / max(ctrl['A'], 0.01):>7.2f}×"
                f"{s.get('B', 0) / max(ctrl['B'], 0.01):>7.2f}×"
                f"{s.get('C', 0) / max(ctrl['C'], 0.01):>7.2f}×"
                f"{s.get('AB', 0) / max(ctrl['AB'], 0.01):>7.2f}×"
                f"{s.get('AC', 0) / max(ctrl['AC'], 0.01):>7.2f}×"
                f"{s.get('BC', 0) / max(ctrl['BC'], 0.01):>7.2f}×"
                f"{s.get('ABC', 0) / max(ctrl['ABC'], 0.01):>9.2f}×"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
