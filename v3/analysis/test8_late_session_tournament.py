"""W4 first-pass tournament: 2 detectors x 4 families late-session sweep.

Question
    For the late-session regime (minute in [40, 120], inside first15, squeeze
    last10_range <= 1.0 * OMAR), which detector + family combination produces
    a directional edge usable as a teacher?

Detectors
    A (narrow / legacy filtered): the 5-gate universe used by
        late_session_fakeout_split.py — adds sigma_pos B-direction gate
        (sigma <= +0.5 for calls, sigma >= -0.5 for puts) and OMAR retest
        (close within 0.5 * OMAR.range of high/low/mid) on top of the base
        squeeze + inside_first15 + late-session-window gates.
    B (wide / research): the base 3-gate universe only. sigma_pos / OMAR
        distance / VP context are router-input candidates, not detector
        gates. Per the plan, this is where the 22/22 fakeout_split evidence
        cannot be cited — measurement on Detector B is fresh.

Families
    1. immediate breakout: enter at trigger bar in break direction.
    2. confirm_1bar continuation: enter at trigger+1 if still outside the
       pre-break last10 range, in break direction.
    3. failed_break_reversal: scan trigger+1..trigger+K; enter at first bar
       whose close is back inside last10 range, in the OPPOSITE direction.
    4. two_stage_router: at trigger+K, decide CLEAN vs FAKEOUT.
       CLEAN (no re-entry into last10 range within K bars): enter at
       trigger+K in the break direction.
       FAKEOUT (re-entry observed): enter at the first re-entry bar in
       the OPPOSITE direction.

Both router branches are aggregated into a single 'router_pooled' family
for the headline cell; per-branch (router_clean, router_fakeout) breakdown
is also reported so the user can inspect whether one branch is carrying.

Forward-move horizon: 20 bars (mfe20 / final20).

Qualifying bar (per the plan; walk-forward in test9 is the next gate)
    triggers/day in [0.3, 3.0]
    direction-match rate at oracle coincidences >= 60%
    mfe20 median edge over matched control >= +3 bps

If no cell clears, the late-session regime is Layer-2-only (W2 features)
and no W4 teacher gets written.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from v2.core import market_structure
from v3.harness.v2_adapter import V2Dataset
from v3.reporter.trigger_quality import (
    latest_attribution_file,
    oracle_index_from_attribution,
)


HORIZON = 20
ROUTER_K = 3
CONTROL_SAMPLES = 5000
RNG_SEED = 101


# --- Forward move (per-day arrays variant; keeps this script self-contained) ---

def _forward_move_perday(close_arr: np.ndarray, minute: int, direction: str, horizon: int = HORIZON) -> tuple[float, float]:
    n = len(close_arr)
    end = min(minute + horizon + 1, n)
    if end <= minute + 1:
        return 0.0, 0.0
    entry = float(close_arr[minute])
    if entry <= 0:
        return 0.0, 0.0
    window = close_arr[minute + 1: end]
    if direction == "call":
        mfe = float(np.max((window - entry) / entry) * 10000.0)
        final = float((window[-1] - entry) / entry * 10000.0)
    else:
        mfe = float(np.max((entry - window) / entry) * 10000.0)
        final = float((entry - window[-1]) / entry * 10000.0)
    return mfe, final


# --- Detector predicates ----------------------------------------------------

def _detector_a(close: float, l10: dict, omar: dict, sigma: Optional[float], f15_high: float, f15_low: float) -> Optional[str]:
    if not (f15_low <= close <= f15_high and f15_high > 0 and f15_low > 0):
        return None
    if l10 is None or l10["range"] > omar["range"]:
        return None
    breaks_high = close > l10["high"]
    breaks_low = close < l10["low"]
    if not (breaks_high or breaks_low):
        return None
    direction = "call" if breaks_high else "put"
    if sigma is None:
        return None
    if direction == "call" and sigma > 0.5:
        return None
    if direction == "put" and sigma < -0.5:
        return None
    r = omar["range"]
    if not any(abs(close - omar[k]) / r <= 0.5 for k in ("high", "low", "mid")):
        return None
    return direction


def _detector_b(close: float, l10: dict, omar: dict, f15_high: float, f15_low: float) -> Optional[str]:
    if not (f15_low <= close <= f15_high and f15_high > 0 and f15_low > 0):
        return None
    if l10 is None or l10["range"] > omar["range"]:
        return None
    breaks_high = close > l10["high"]
    breaks_low = close < l10["low"]
    if not (breaks_high or breaks_low):
        return None
    return "call" if breaks_high else "put"


# --- Family entry resolvers (return list of (entry_minute, direction, branch_tag)) ---

def _family_immediate(trigger_minute: int, direction: str, *_args) -> list[tuple[int, str, str]]:
    return [(trigger_minute, direction, "")]


def _family_confirm_1bar(trigger_minute: int, direction: str, close_arr: np.ndarray, l10: dict, n_bars: int) -> list[tuple[int, str, str]]:
    next_min = trigger_minute + 1
    if next_min >= n_bars:
        return []
    next_close = float(close_arr[next_min])
    if direction == "call" and next_close > l10["high"]:
        return [(next_min, direction, "")]
    if direction == "put" and next_close < l10["low"]:
        return [(next_min, direction, "")]
    return []


def _family_failed_break_reversal(trigger_minute: int, direction: str, close_arr: np.ndarray, l10: dict, n_bars: int, K: int = ROUTER_K) -> list[tuple[int, str, str]]:
    reverse = "put" if direction == "call" else "call"
    for probe in range(trigger_minute + 1, min(trigger_minute + 1 + K, n_bars)):
        c = float(close_arr[probe])
        if l10["low"] <= c <= l10["high"]:
            return [(probe, reverse, "")]
    return []


def _family_router(trigger_minute: int, direction: str, close_arr: np.ndarray, l10: dict, n_bars: int, K: int = ROUTER_K) -> list[tuple[int, str, str]]:
    reverse = "put" if direction == "call" else "call"
    for probe in range(trigger_minute + 1, min(trigger_minute + 1 + K, n_bars)):
        c = float(close_arr[probe])
        if l10["low"] <= c <= l10["high"]:
            return [(probe, reverse, "fakeout")]
    clean_entry = trigger_minute + K
    if clean_entry >= n_bars:
        return []
    return [(clean_entry, direction, "clean")]


FAMILIES = [
    ("immediate", _family_immediate),
    ("confirm_1bar", _family_confirm_1bar),
    ("failed_break_reversal", _family_failed_break_reversal),
    ("router", _family_router),
]


# --- Matched control for forward-move comparison ---------------------------

def _matched_control_stats(ds: V2Dataset, spx_bars: dict, n_samples: int = CONTROL_SAMPLES, rng_seed: int = RNG_SEED) -> dict[str, float]:
    """Random late-session inside-first15 bars with random direction.

    Returns mfe20 median/mean and final20 median/mean — the baseline that
    each tournament cell needs to beat by +3 bps on mfe20 median to clear
    the qualifying bar.
    """
    rng = np.random.default_rng(rng_seed)
    all_days = list(spx_bars.keys())
    mfe_list: list[float] = []
    final_list: list[float] = []
    attempts = 0
    while len(mfe_list) < n_samples and attempts < n_samples * 10:
        attempts += 1
        day = str(rng.choice(all_days))
        arrs = spx_bars.get(day)
        if arrs is None:
            continue
        n_bars = len(arrs["close"])
        if n_bars < 50:
            continue
        minute = int(rng.integers(40, min(121, n_bars)))
        close = float(arrs["close"][minute])
        if close <= 0:
            continue
        f15_high, f15_low = ds.first15_by_day.get(day, (0.0, 0.0))
        if not (f15_low <= close <= f15_high and f15_high > 0 and f15_low > 0):
            continue
        direction = "call" if rng.random() < 0.5 else "put"
        mfe, final = _forward_move_perday(arrs["close"], minute, direction)
        mfe_list.append(mfe)
        final_list.append(final)
    mfe_arr = np.asarray(mfe_list, dtype=float)
    final_arr = np.asarray(final_list, dtype=float)
    return {
        "n": len(mfe_list),
        "mfe20_median": float(np.median(mfe_arr)),
        "mfe20_mean": float(mfe_arr.mean()),
        "final20_median": float(np.median(final_arr)),
        "final20_mean": float(final_arr.mean()),
    }


# --- Per-cell aggregation ---------------------------------------------------

@dataclass
class CellRecord:
    detector: str
    family: str
    entries: list[dict] = field(default_factory=list)
    branch_entries: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))


def _scan(ds: V2Dataset, spx_bars: dict, omar_cache: dict, oracle_index: dict) -> dict[tuple[str, str], CellRecord]:
    cells: dict[tuple[str, str], CellRecord] = {
        (det, fam): CellRecord(detector=det, family=fam)
        for det in ("A_legacy", "B_wide")
        for fam, _ in FAMILIES
    }

    all_days = sorted(set(ds.dates))
    progress_step = max(1, len(all_days) // 10)
    for i, day in enumerate(all_days):
        if i % progress_step == 0:
            print(f"  scanning day {i+1}/{len(all_days)} ({day})")
        arrs = spx_bars.get(day)
        omar = omar_cache.get(day)
        spy_day = ds.spy_vwap.get(day) if ds.spy_vwap is not None else None
        f15 = ds.first15_by_day.get(day)
        if arrs is None or omar is None or omar["range"] <= 0 or f15 is None or spy_day is None:
            continue
        f15_high, f15_low = f15
        n_bars = len(arrs["close"])
        for minute in range(40, min(121, n_bars - 1)):
            close = float(arrs["close"][minute])
            if close <= 0:
                continue
            l10 = market_structure.last10(arrs, minute)
            sigma = market_structure.sigma_pos(spy_day, minute, close)

            # Detector B is a strict superset of Detector A's geometry filters;
            # both detectors compute the same break direction. Apply each.
            for det_name, det_fn, det_args in (
                ("A_legacy", _detector_a, (close, l10, omar, sigma, f15_high, f15_low)),
                ("B_wide",   _detector_b, (close, l10, omar, f15_high, f15_low)),
            ):
                direction = det_fn(*det_args)
                if direction is None:
                    continue
                for fam_name, fam_fn in FAMILIES:
                    entries = fam_fn(minute, direction, arrs["close"], l10, n_bars)
                    for entry_min, entry_dir, branch in entries:
                        if entry_min >= n_bars:
                            continue
                        mfe, final = _forward_move_perday(arrs["close"], entry_min, entry_dir)
                        oracle = oracle_index.get((day, entry_min))
                        rec = {
                            "day": day,
                            "trigger_minute": minute,
                            "entry_minute": entry_min,
                            "direction": entry_dir,
                            "mfe20": mfe,
                            "final20": final,
                            "oracle_match": oracle is not None,
                            "direction_match": (
                                oracle is not None and oracle.get("direction") == entry_dir
                            ),
                            "oracle_outcome": oracle.get("outcome") if oracle is not None else None,
                            "branch": branch,
                        }
                        cell = cells[(det_name, fam_name)]
                        cell.entries.append(rec)
                        if branch:
                            cell.branch_entries[branch].append(rec)
    return cells


# --- Reporting --------------------------------------------------------------

def _summarize_entries(name: str, entries: list[dict], n_days: int, control: dict) -> dict:
    n_trig = len(entries)
    if n_trig == 0:
        return {
            "name": name, "n": 0, "per_day": 0.0,
            "n_oracle": 0, "n_dir_match": 0, "dir_match_rate": None,
            "mfe20_median": 0.0, "mfe20_mean": 0.0,
            "final20_median": 0.0, "final20_mean": 0.0,
            "mfe20_edge_vs_control": 0.0,
        }
    mfe = np.asarray([r["mfe20"] for r in entries], dtype=float)
    final = np.asarray([r["final20"] for r in entries], dtype=float)
    n_oracle = sum(1 for r in entries if r["oracle_match"])
    n_match = sum(1 for r in entries if r["direction_match"])
    dir_match_rate = (n_match / n_oracle) if n_oracle > 0 else None
    mfe20_med = float(np.median(mfe))
    return {
        "name": name,
        "n": n_trig,
        "per_day": n_trig / max(n_days, 1),
        "n_oracle": n_oracle,
        "n_dir_match": n_match,
        "dir_match_rate": dir_match_rate,
        "mfe20_median": mfe20_med,
        "mfe20_mean": float(mfe.mean()),
        "final20_median": float(np.median(final)),
        "final20_mean": float(final.mean()),
        "mfe20_edge_vs_control": mfe20_med - control["mfe20_median"],
    }


def _row_str(s: dict) -> str:
    dr = f"{100*s['dir_match_rate']:5.1f}%" if s["dir_match_rate"] is not None else "  n/a"
    return (
        f"  {s['name']:<28} n={s['n']:>5} ({s['per_day']:>4.2f}/day)  "
        f"oracle={s['n_oracle']:>3}  dir_match={s['n_dir_match']}/{s['n_oracle']} ({dr})  "
        f"mfe20 med={s['mfe20_median']:+6.2f}bps (edge={s['mfe20_edge_vs_control']:+5.2f}bps)  "
        f"final20 med={s['final20_median']:+6.2f}bps  mean={s['final20_mean']:+6.2f}bps"
    )


def _qualify(s: dict) -> str:
    """Returns 'PASS' / 'FAIL: ...' against the plan's qualifying bar."""
    fails: list[str] = []
    if not (0.3 <= s["per_day"] <= 3.0):
        fails.append(f"density {s['per_day']:.2f}/day not in [0.3, 3.0]")
    if s["dir_match_rate"] is None or s["dir_match_rate"] < 0.60:
        dm = "n/a" if s["dir_match_rate"] is None else f"{100*s['dir_match_rate']:.1f}%"
        fails.append(f"dir_match {dm} below 60%")
    if s["mfe20_edge_vs_control"] < 3.0:
        fails.append(f"mfe20 edge {s['mfe20_edge_vs_control']:+.2f}bps below +3.0 bps")
    if not fails:
        return "PASS"
    return "FAIL: " + "; ".join(fails)


def main() -> int:
    print(f"Loading oracle index from {latest_attribution_file()}")
    oracle = oracle_index_from_attribution(latest_attribution_file())
    print(f"  oracle entries indexed: {len(oracle)}")

    print("Loading v2 dataset + SPX/OMAR caches...")
    ds = V2Dataset.load()
    spx_bars = market_structure.build_spx_bars()
    omar_cache = market_structure.build_omar_map(spx_bars)
    n_days = len(set(ds.dates))

    print("Computing matched-control forward moves (random late-session inside-first15 bars + random direction)...")
    control = _matched_control_stats(ds, spx_bars)
    print(f"  control n={control['n']}  mfe20 median={control['mfe20_median']:+.2f}bps  "
          f"final20 median={control['final20_median']:+.2f}bps")
    print()
    print("Scanning 986-day cache for tournament cells...")
    cells = _scan(ds, spx_bars, omar_cache, oracle)

    print()
    print("=" * 130)
    print(f"Tournament results: 2 detectors x 4 families. Horizon={HORIZON}, K={ROUTER_K}.")
    print(f"Qualifying bar: density in [0.3, 3.0]/day; dir_match >= 60%; mfe20 edge vs control >= +3.00 bps.")
    print(f"Control mfe20 median = {control['mfe20_median']:+.2f} bps.")
    print("=" * 130)

    qualifiers: list[tuple[str, str, dict]] = []
    for det_name in ("A_legacy", "B_wide"):
        print()
        print(f"--- Detector {det_name} ---")
        for fam_name, _ in FAMILIES:
            cell = cells[(det_name, fam_name)]
            s = _summarize_entries(fam_name, cell.entries, n_days, control)
            verdict = _qualify(s)
            print(_row_str(s) + f"  [{verdict}]")
            if verdict == "PASS":
                qualifiers.append((det_name, fam_name, s))
            # Router branch breakdown
            if fam_name == "router" and cell.entries:
                for branch_name in ("clean", "fakeout"):
                    branch_rows = cell.branch_entries.get(branch_name, [])
                    if not branch_rows:
                        continue
                    sb = _summarize_entries(f"  router_{branch_name}", branch_rows, n_days, control)
                    print(_row_str(sb))

    print()
    print("=" * 130)
    if qualifiers:
        print(f"QUALIFIERS ({len(qualifiers)} cell(s) cleared the first-pass bar):")
        for det, fam, s in qualifiers:
            print(f"  {det} / {fam}: {_row_str(s).strip()}")
        print()
        print("Next gate: test9 walk-forward over 5 chronological folds. Promotion to teacher")
        print("only after positive net_gap in all 5 folds for the promoted detector/family combination.")
    else:
        print("NO CELL CLEARED. Per the plan stopping rule, accept the late-session regime is")
        print("Layer-2-only for now: capture via soft features in W2 (post-G1) and do NOT keep")
        print("inventing teachers in the same cycle. Next steps: write the W4 verdict doc and stop.")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
