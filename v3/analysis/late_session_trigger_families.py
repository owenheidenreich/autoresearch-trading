"""Point-in-time comparison of late-session trigger families.

This script starts from the localized late-session regime implied by the
reference research:
    - minute in [40, 120]
    - inside first15 range
    - squeeze <= 1.0 * OMAR
    - VWAP direction gate
    - OMAR retest gate

Then it compares entry families that are actually observable point in time:

1. `immediate_breakout`
   Enter on the breakout bar in the breakout direction.

2. `confirm_1bar`
   Enter one bar later, same direction, only if the next close remains
   outside the pre-break range.

3. `reentry_reversal_3bar`
   Enter on the first close back inside the pre-break range within the next
   3 bars, in the opposite direction.

The goal is not to claim a final teacher. It is to test whether the late-
session regime has more than one tradable routing path.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from v3.harness.v2_adapter import V2Dataset


def _build_spx_bars(spx_1min_path: str) -> dict[str, dict[str, np.ndarray]]:
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        out[day] = {
            "high": g["spx_high"].to_numpy(dtype=float),
            "low": g["spx_low"].to_numpy(dtype=float),
            "close": g["spx_close"].to_numpy(dtype=float),
        }
    return out


def _build_omar_map(spx_bars: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for day, arrs in spx_bars.items():
        h = float(arrs["high"][0])
        l = float(arrs["low"][0])
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
        vwap = g["vwap"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        sq_dev = (close - vwap) ** 2
        cum_v = np.maximum(np.cumsum(volume), 1.0)
        cum_sq = np.cumsum(volume * sq_dev)
        std_run = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        out[day] = {"vwap": vwap, "close": close, "std": std_run}
    return out


def _sigma_pos(
    spy_day: dict[str, np.ndarray], minute: int, spx_close: float
) -> float | None:
    if minute < 0 or minute >= len(spy_day["vwap"]):
        return None
    spy_vwap = float(spy_day["vwap"][minute])
    spy_close = float(spy_day["close"][minute])
    spy_std = float(spy_day["std"][minute])
    if not all(v > 0 for v in (spy_vwap, spy_close, spy_std, spx_close)):
        return None
    ratio = spx_close / spy_close
    return (spx_close - spy_vwap * ratio) / (spy_std * ratio)


def _retest_pass(spx_close: float, omar: dict[str, float], threshold: float = 0.5) -> bool:
    r = omar["range"]
    if r <= 0:
        return False
    return any(abs(spx_close - omar[k]) / r <= threshold for k in ("high", "low", "mid"))


def _forward_move(
    spx_close_array: np.ndarray, minute: int, direction: str, horizon: int = 20
) -> tuple[float, float]:
    end = min(minute + horizon + 1, len(spx_close_array))
    if end <= minute + 1:
        return 0.0, 0.0
    entry = float(spx_close_array[minute])
    if entry <= 0:
        return 0.0, 0.0
    window = spx_close_array[minute + 1 : end]
    if direction == "call":
        mfe = float(np.max((window - entry) / entry) * 10000.0)
        final = float((window[-1] - entry) / entry * 10000.0)
    else:
        mfe = float(np.max((entry - window) / entry) * 10000.0)
        final = float((entry - window[-1]) / entry * 10000.0)
    return mfe, final


def _summarize(name: str, rows: list[dict], n_days: int) -> None:
    if not rows:
        print(f"{name}: n=0")
        return
    mfe = np.asarray([r["mfe20"] for r in rows], dtype=float)
    final = np.asarray([r["final20"] for r in rows], dtype=float)
    print(f"{name}: n={len(rows)}  density/day={len(rows)/max(n_days,1):.2f}")
    print(
        f"  mfe20 median={np.median(mfe):+6.2f} bps  "
        f"mean={mfe.mean():+6.2f} bps"
    )
    print(
        f"  final20 median={np.median(final):+6.2f} bps  "
        f"mean={final.mean():+6.2f} bps  "
        f"pct>0={100*np.mean(final > 0):.1f}%"
    )


def main() -> int:
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    spx_bars = _build_spx_bars(str(spx_path))
    omar_map = _build_omar_map(spx_bars)
    spy_vwap = _build_spy_vwap(str(spy_path))

    immediate_rows: list[dict] = []
    confirm_rows: list[dict] = []
    reversal_rows: list[dict] = []

    dataset_days = sorted(set(ds.dates))
    for day in dataset_days:
        arrs = spx_bars.get(day)
        if arrs is None:
            continue
        f15 = ds.first15_by_day.get(day)
        omar = omar_map.get(day)
        spy_day = spy_vwap.get(day)
        if not f15 or omar is None or spy_day is None:
            continue
        f15_high, f15_low = f15
        n_bars = len(arrs["close"])
        for minute in range(40, min(121, n_bars - 4)):
            spx_close = float(arrs["close"][minute])
            if spx_close <= 0:
                continue
            if not (f15_low <= spx_close <= f15_high):
                continue

            lo = max(minute - 10, 0)
            last10_high = float(arrs["high"][lo:minute].max()) if minute > lo else spx_close
            last10_low = float(arrs["low"][lo:minute].min()) if minute > lo else spx_close
            last10_range = last10_high - last10_low
            if last10_range > omar["range"]:
                continue

            breaks_high = spx_close > last10_high
            breaks_low = spx_close < last10_low
            if not (breaks_high or breaks_low):
                continue
            direction = "call" if breaks_high else "put"

            sigma = _sigma_pos(spy_day, minute, spx_close)
            if sigma is None:
                continue
            if direction == "call" and sigma > 0.5:
                continue
            if direction == "put" and sigma < -0.5:
                continue
            if not _retest_pass(spx_close, omar, threshold=0.5):
                continue

            # Family 1: immediate breakout-follow
            mfe20, final20 = _forward_move(arrs["close"], minute, direction, horizon=20)
            immediate_rows.append(
                {"day": day, "minute": minute, "direction": direction, "mfe20": mfe20, "final20": final20}
            )

            # Family 2: one-bar confirmation
            confirm_bar = minute + 1
            if confirm_bar < n_bars:
                confirm_close = float(arrs["close"][confirm_bar])
                confirm_ok = (
                    (direction == "call" and confirm_close > last10_high)
                    or (direction == "put" and confirm_close < last10_low)
                )
                if confirm_ok:
                    mfe20, final20 = _forward_move(arrs["close"], confirm_bar, direction, horizon=20)
                    confirm_rows.append(
                        {
                            "day": day,
                            "minute": confirm_bar,
                            "direction": direction,
                            "mfe20": mfe20,
                            "final20": final20,
                        }
                    )

            # Family 3: failed-break reversal on first re-entry inside range within 3 bars
            reverse_direction = "put" if direction == "call" else "call"
            reentry_bar = None
            for probe in range(minute + 1, min(minute + 4, n_bars)):
                probe_close = float(arrs["close"][probe])
                if last10_low <= probe_close <= last10_high:
                    reentry_bar = probe
                    break
            if reentry_bar is not None:
                mfe20, final20 = _forward_move(arrs["close"], reentry_bar, reverse_direction, horizon=20)
                reversal_rows.append(
                    {
                        "day": day,
                        "minute": reentry_bar,
                        "direction": reverse_direction,
                        "mfe20": mfe20,
                        "final20": final20,
                    }
                )

    n_days = len(dataset_days)
    print(f"late-session localized regime days: {n_days}")
    print()
    _summarize("immediate_breakout", immediate_rows, n_days)
    print()
    _summarize("confirm_1bar", confirm_rows, n_days)
    print()
    _summarize("reentry_reversal_3bar", reversal_rows, n_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
