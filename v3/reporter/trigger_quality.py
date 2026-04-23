"""Trigger-quality reporter.

Given a trigger predicate (a function that, for each (day, minute), returns
'call' / 'put' / None) and an oracle labeling, report how the trigger would
behave at runtime:

- triggers/day (density)
- total triggers
- oracle-bar coincidence rate (triggers ∩ oracle bars / total triggers)
- direction match rate at oracle coincidences
- forward-move distribution (mfe20, final20)

This is the ONLY report that can project a teacher's live impact. It is the
correct frame to evaluate a candidate teacher AGAINST — `localization.py`
measures `P(feature | oracle)` which differs from `P(oracle | trigger ∧ feature)`
whenever the trigger changes the base rate. The combined_confluence.py "39.6%
abstention capture" claim was a localization measurement misread as a
trigger-capture rate; test2_nr10_teacher_dryrun.py later showed the actual
trigger-level capture was 4.5%. This reporter exists so that next time, the
two quantities are computed by separately-named functions and cannot be
confused.

CLI usage: see `main()` — reproduces the test2 NR10 trigger evaluation
labeled as a `trigger_quality` measurement, plus an ORC + sigma_pos baseline.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from v2.core import market_structure
from v3.harness.v2_adapter import V2Dataset


TriggerFn = Callable[[V2Dataset, str, int], Optional[str]]
"""(dataset, day, minute) -> 'call' / 'put' / None."""

SliceFn = Callable[[V2Dataset, str, int], bool]


@dataclass
class TriggerQualityMeasurement:
    trigger_name: str
    universe_name: str
    n_days: int
    n_triggers: int
    triggers_per_day: float
    n_oracle_coincidences: int
    oracle_coincidence_rate: float
    n_direction_matches: int
    direction_match_rate: Optional[float]
    mfe20_median: float
    mfe20_mean: float
    final20_median: float
    final20_mean: float


# --- Oracle index from attribution JSON -------------------------------------

def latest_attribution_file(reference_dir: str = "v3/reference") -> str:
    files = sorted(glob.glob(f"{reference_dir}/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError(f"no attribution_full_*.txt in {reference_dir}")
    return files[-1]


def oracle_index_from_attribution(path: str) -> dict[tuple[str, int], dict]:
    """Map (day, minute) -> {direction, outcome}."""
    out: dict[tuple[str, int], dict] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        day = row.get("day")
        bar = row.get("oracle_bar")
        direction = row.get("oracle_direction")
        outcome = row.get("outcome")
        if day is None or bar is None or direction not in ("call", "put"):
            continue
        out[(str(day), int(bar))] = {"direction": direction, "outcome": outcome}
    return out


# --- Forward-move helpers ---------------------------------------------------

def _forward_move(
    spx_close: np.ndarray, abs_idx: int, day_end: int, direction: str, horizon: int = 20
) -> tuple[float, float]:
    end = min(abs_idx + horizon + 1, day_end)
    if end <= abs_idx + 1:
        return 0.0, 0.0
    entry = float(spx_close[abs_idx])
    if entry <= 0:
        return 0.0, 0.0
    window = spx_close[abs_idx + 1 : end]
    if direction == "call":
        mfe = float(np.max((window - entry) / entry) * 10000.0)
        final = float((window[-1] - entry) / entry * 10000.0)
    else:
        mfe = float(np.max((entry - window) / entry) * 10000.0)
        final = float((entry - window[-1]) / entry * 10000.0)
    return mfe, final


# --- Measurement core -------------------------------------------------------

def measure_trigger_quality(
    ds: V2Dataset,
    trigger_name: str,
    trigger_fn: TriggerFn,
    oracle_index: dict[tuple[str, int], dict],
    *,
    universe_slice: Optional[SliceFn] = None,
    universe_name: str = "all_bars",
    minute_window: tuple[int, int] = (0, 389),
    horizon: int = 20,
) -> TriggerQualityMeasurement:
    """Scan every (day, minute) in the universe; record triggers + forward moves."""
    all_days = sorted(set(ds.dates))
    n_days = len(all_days)

    n_triggers = 0
    n_oracle = 0
    n_match = 0
    mfe_list: list[float] = []
    final_list: list[float] = []

    lo_min, hi_min = minute_window
    for day in all_days:
        try:
            day_start, day_end = ds.day_bar_range(day)
        except (ValueError, IndexError):
            continue
        n_bars = day_end - day_start
        for minute in range(max(lo_min, 0), min(hi_min + 1, n_bars)):
            if universe_slice is not None and not universe_slice(ds, day, minute):
                continue
            direction = trigger_fn(ds, day, minute)
            if direction not in ("call", "put"):
                continue
            n_triggers += 1
            abs_idx = day_start + minute
            mfe, final = _forward_move(ds.spot_prices, abs_idx, day_end, direction, horizon=horizon)
            mfe_list.append(mfe)
            final_list.append(final)
            oracle = oracle_index.get((day, minute))
            if oracle is not None:
                n_oracle += 1
                if oracle["direction"] == direction:
                    n_match += 1

    triggers_per_day = n_triggers / max(n_days, 1)
    oracle_coincidence_rate = (n_oracle / n_triggers) if n_triggers > 0 else 0.0
    direction_match_rate: Optional[float] = (n_match / n_oracle) if n_oracle > 0 else None
    mfe_arr = np.asarray(mfe_list, dtype=float) if mfe_list else np.zeros(0, dtype=float)
    final_arr = np.asarray(final_list, dtype=float) if final_list else np.zeros(0, dtype=float)
    return TriggerQualityMeasurement(
        trigger_name=trigger_name,
        universe_name=universe_name,
        n_days=n_days,
        n_triggers=n_triggers,
        triggers_per_day=triggers_per_day,
        n_oracle_coincidences=n_oracle,
        oracle_coincidence_rate=oracle_coincidence_rate,
        n_direction_matches=n_match,
        direction_match_rate=direction_match_rate,
        mfe20_median=float(np.median(mfe_arr)) if mfe_arr.size else 0.0,
        mfe20_mean=float(mfe_arr.mean()) if mfe_arr.size else 0.0,
        final20_median=float(np.median(final_arr)) if final_arr.size else 0.0,
        final20_mean=float(final_arr.mean()) if final_arr.size else 0.0,
    )


# --- Built-in trigger predicates --------------------------------------------

def _bar_close(ds: V2Dataset, day: str, minute: int) -> Optional[float]:
    try:
        day_start, day_end = ds.day_bar_range(day)
    except (ValueError, IndexError):
        return None
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    c = float(ds.spot_prices[abs_idx])
    return c if c > 0 else None


def make_trigger_orc_with_sigma() -> TriggerFn:
    """Reproduce the post-A1 ORC teacher purely as a trigger predicate.

    Mirrors the BarContext-based ORC + sigma_pos gate but operates directly
    on the V2Dataset arrays (so trigger_quality can scan the universe
    without going through run_full_attribution).
    """
    def _fn(ds: V2Dataset, day: str, minute: int) -> Optional[str]:
        if minute < 15 or minute > 90:
            return None
        try:
            day_start, day_end = ds.day_bar_range(day)
        except (ValueError, IndexError):
            return None
        abs_idx = day_start + minute
        if abs_idx >= day_end:
            return None
        row = ds.X_sim[abs_idx]
        close = float(ds.spot_prices[abs_idx])
        if close <= 0:
            return None
        vwap_dist = float(row[ds.idx["vwap_dist"]])
        vwap = close * (1.0 - vwap_dist) if close > 0 else close
        vwap_slope = float(row[ds.idx["vwap_slope"]])
        f15_high, f15_low = ds.first15_by_day.get(day, (0.0, 0.0))
        breaks_high = close > f15_high and f15_high > 0
        breaks_low = close < f15_low and f15_low > 0
        vwap_trend_up = close > vwap and vwap_slope > 0.0
        vwap_trend_dn = close < vwap and vwap_slope < 0.0
        spy_day = ds.spy_vwap.get(day) if ds.spy_vwap is not None else None
        sigma = market_structure.sigma_pos(spy_day, minute, close) if spy_day is not None else None
        if breaks_high and vwap_trend_up:
            if sigma is not None and sigma > 0.0:
                return None
            return "call"
        if breaks_low and vwap_trend_dn:
            if sigma is not None and sigma < 0.0:
                return None
            return "put"
        return None
    return _fn


def make_trigger_nr10_full_gate(omar_cache: dict[str, dict[str, float]]) -> TriggerFn:
    """Replicate the falsified test2 NR10 5-gate trigger as a TriggerFn so we
    can re-report its trigger_quality numbers under the new label.

    Gates:
      [40, 120] minute window
      close inside first15 range
      last10 SPX range ≤ 1.0 × OMAR range
      close breaks last10 high (call) or low (put)
      sigma_pos ≤ +0.5 for calls, ≥ −0.5 for puts
      OMAR retest: close within 0.5 × OMAR range of high/low/mid
    """
    spx_bars = market_structure.build_spx_bars()

    def _fn(ds: V2Dataset, day: str, minute: int) -> Optional[str]:
        if minute < 40 or minute > 120:
            return None
        arrs = spx_bars.get(day)
        if arrs is None:
            return None
        n_bars = len(arrs["close"])
        if minute >= n_bars:
            return None
        close = float(arrs["close"][minute])
        if close <= 0:
            return None
        f15_high, f15_low = ds.first15_by_day.get(day, (0.0, 0.0))
        if not (f15_high > 0 and f15_low > 0 and f15_low <= close <= f15_high):
            return None
        omar = omar_cache.get(day)
        if omar is None or omar["range"] <= 0:
            return None
        l10 = market_structure.last10(arrs, minute)
        if l10 is None or l10["range"] > omar["range"]:
            return None
        breaks_high = close > l10["high"]
        breaks_low = close < l10["low"]
        if not (breaks_high or breaks_low):
            return None
        direction = "call" if breaks_high else "put"
        spy_day = ds.spy_vwap.get(day) if ds.spy_vwap is not None else None
        sigma = market_structure.sigma_pos(spy_day, minute, close) if spy_day is not None else None
        if sigma is None:
            return None
        if direction == "call" and sigma > 0.5:
            return None
        if direction == "put" and sigma < -0.5:
            return None
        # OMAR retest
        r = omar["range"]
        retest = any(abs(close - omar[k]) / r <= 0.5 for k in ("high", "low", "mid"))
        if not retest:
            return None
        return direction

    return _fn


# --- Default slices (mirror localization.py for stratified eval) ------------

def slice_late_session(ds: V2Dataset, day: str, minute: int) -> bool:
    return 40 <= minute <= 120


def slice_inside_first15(ds: V2Dataset, day: str, minute: int) -> bool:
    high, low = ds.first15_by_day.get(day, (0.0, 0.0))
    if high <= 0 or low <= 0:
        return False
    close = _bar_close(ds, day, minute)
    if close is None:
        return False
    return low <= close <= high


def make_slice_near_omar(
    omar_cache: dict[str, dict[str, float]], threshold: float = 0.5
) -> SliceFn:
    def _fn(ds: V2Dataset, day: str, minute: int) -> bool:
        omar = omar_cache.get(day)
        if omar is None or omar["range"] <= 0:
            return False
        close = _bar_close(ds, day, minute)
        if close is None:
            return False
        r = omar["range"]
        return any(abs(close - omar[k]) / r <= threshold for k in ("high", "low", "mid"))
    return _fn


# --- Formatting -------------------------------------------------------------

def format_measurement(m: TriggerQualityMeasurement) -> str:
    head = (
        f"{m.trigger_name:<28} universe={m.universe_name:<20} "
        f"days={m.n_days:>4}  triggers={m.n_triggers:>5} ({m.triggers_per_day:>4.2f}/day)"
    )
    oracle = (
        f"  oracle coincidences: {m.n_oracle_coincidences:>4}  "
        f"({100*m.oracle_coincidence_rate:5.2f}% of triggers)  "
        f"direction match: {m.n_direction_matches}/{m.n_oracle_coincidences} "
        f"({(100*m.direction_match_rate):5.1f}% if defined)"
        if m.direction_match_rate is not None
        else f"  oracle coincidences: {m.n_oracle_coincidences:>4}  direction match: n/a"
    )
    fwd = (
        f"  mfe20 median={m.mfe20_median:+6.2f} bps  mean={m.mfe20_mean:+6.2f} bps  "
        f"final20 median={m.final20_median:+6.2f} bps  mean={m.final20_mean:+6.2f} bps"
    )
    return f"{head}\n{oracle}\n{fwd}"


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Trigger-quality reporter")
    p.add_argument("--attribution", default=None,
                   help="Path to attribution_full_*.txt (defaults to latest in v3/reference/)")
    p.add_argument("--horizon", type=int, default=20)
    args = p.parse_args(argv)

    attribution_path = args.attribution or latest_attribution_file()
    print(f"Loading oracle index from {attribution_path}")
    oracle = oracle_index_from_attribution(attribution_path)
    print(f"  oracle entries indexed: {len(oracle)}")

    print("Loading v2 dataset + OMAR cache...")
    ds = V2Dataset.load()
    spx_bars = market_structure.build_spx_bars()
    omar_cache = market_structure.build_omar_map(spx_bars)

    print()
    print("=" * 100)
    print("Trigger: ORC + sigma_pos gate (post-A1 baseline, restated as a trigger predicate)")
    print("=" * 100)
    m = measure_trigger_quality(
        ds,
        trigger_name="orc_plus_sigma",
        trigger_fn=make_trigger_orc_with_sigma(),
        oracle_index=oracle,
        minute_window=(15, 90),
        horizon=args.horizon,
    )
    print(format_measurement(m))

    print()
    print("=" * 100)
    print("Trigger: NR10 full 5-gate (the falsified late-session teacher).")
    print("  Reproduces test2_nr10_teacher_dryrun.py results under the explicit")
    print("  trigger_quality label so the measurement type is unambiguous.")
    print("=" * 100)
    m = measure_trigger_quality(
        ds,
        trigger_name="nr10_full_gate",
        trigger_fn=make_trigger_nr10_full_gate(omar_cache),
        oracle_index=oracle,
        universe_slice=slice_late_session,
        universe_name="late_session",
        minute_window=(40, 120),
        horizon=args.horizon,
    )
    print(format_measurement(m))

    print()
    print("=" * 100)
    print("Trigger: NR10 full 5-gate, abstention coincidences only (W4 input)")
    print("=" * 100)
    abstention_oracle = {
        k: v for k, v in oracle.items() if v.get("outcome") == "abstention"
    }
    m = measure_trigger_quality(
        ds,
        trigger_name="nr10_full_gate (vs abstention only)",
        trigger_fn=make_trigger_nr10_full_gate(omar_cache),
        oracle_index=abstention_oracle,
        universe_slice=slice_late_session,
        universe_name="late_session",
        minute_window=(40, 120),
        horizon=args.horizon,
    )
    print(format_measurement(m))

    return 0


if __name__ == "__main__":
    sys.exit(main())
