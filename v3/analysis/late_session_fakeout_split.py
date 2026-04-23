"""Late-session fakeout split for the filtered NR10-style regime.

This script does not try to prove a shippable teacher. It asks a narrower
question:

Given the late-session regime defined by the filtered NR10-style setup
    - minute in [40, 120]
    - close inside first15 range
    - last10 range <= 1.0 * OMAR
    - directional VWAP gate
    - OMAR retest gate

what happens if we separate:
    1. breakouts that quickly fake out back inside the pre-break range
    2. breakouts that do not fake out

This is useful because the earlier dry-run showed that the filtered trigger
hits some oracle bars but almost always in the wrong direction. If fakeouts
and clean breaks behave differently, then the real edge may be a two-stage
decision process rather than a one-bar breakout trigger.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from v3.harness.v2_adapter import V2Dataset


def _latest_attribution_file() -> str:
    files = sorted(glob.glob("v3/reference/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError("No attribution file found in v3/reference/")
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


def _summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: n=0")
        return
    asis_mfe = np.asarray([r["mfe20_asis"] for r in rows], dtype=float)
    asis_final = np.asarray([r["final20_asis"] for r in rows], dtype=float)
    rev_mfe = np.asarray([r["mfe20_rev"] for r in rows], dtype=float)
    rev_final = np.asarray([r["final20_rev"] for r in rows], dtype=float)
    print(f"{name}: n={len(rows)}")
    print(
        f"  as-is   mfe20 median={np.median(asis_mfe):+6.2f} bps  "
        f"final20 median={np.median(asis_final):+6.2f} bps  "
        f"final20 mean={asis_final.mean():+6.2f} bps"
    )
    print(
        f"  reverse mfe20 median={np.median(rev_mfe):+6.2f} bps  "
        f"final20 median={np.median(rev_final):+6.2f} bps  "
        f"final20 mean={rev_final.mean():+6.2f} bps"
    )
    print(
        f"  reverse beats as-is on final20: {100.0 * np.mean(rev_final > asis_final):.1f}%"
    )


def main() -> int:
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    spx_bars = _build_spx_bars(str(spx_path))
    omar_map = _build_omar_map(spx_bars)
    spy_vwap = _build_spy_vwap(str(spy_path))

    sessions = _load_per_session(_latest_attribution_file())
    oracle_index = {(s["day"], s["oracle_bar"]): s for s in sessions}

    records: list[dict] = []
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
        for minute in range(40, min(121, n_bars)):
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

            fwd10 = arrs["close"][minute + 1 : min(minute + 11, n_bars)]
            fake_out = bool(
                len(fwd10) > 0
                and np.any((fwd10 >= last10_low) & (fwd10 <= last10_high))
            )

            reverse_direction = "put" if direction == "call" else "call"
            mfe20_asis, final20_asis = _forward_move(arrs["close"], minute, direction, horizon=20)
            mfe20_rev, final20_rev = _forward_move(
                arrs["close"], minute, reverse_direction, horizon=20
            )

            oracle = oracle_index.get((day, minute))
            records.append(
                {
                    "day": day,
                    "minute": minute,
                    "direction": direction,
                    "fake_out": fake_out,
                    "mfe20_asis": mfe20_asis,
                    "final20_asis": final20_asis,
                    "mfe20_rev": mfe20_rev,
                    "final20_rev": final20_rev,
                    "oracle": oracle,
                }
            )

    print(f"filtered late-session triggers: {len(records)}")
    print()
    _summarize("all", records)
    print()
    _summarize("fakeout<=10bar", [r for r in records if r["fake_out"]])
    print()
    _summarize("no_fakeout<=10bar", [r for r in records if not r["fake_out"]])
    print()

    oracle_hits = [r for r in records if r["oracle"]]
    print(f"oracle coincidences: {len(oracle_hits)}")
    if oracle_hits:
        asis_match = sum(
            1 for r in oracle_hits if r["oracle"].get("oracle_direction") == r["direction"]
        )
        rev_match = len(oracle_hits) - asis_match
        print(f"  as-is direction matches: {asis_match}/{len(oracle_hits)}")
        print(f"  reverse direction matches: {rev_match}/{len(oracle_hits)}")

    abstention_hits = [
        r for r in oracle_hits if r["oracle"] and r["oracle"].get("outcome") == "abstention"
    ]
    if abstention_hits:
        asis_match = sum(
            1 for r in abstention_hits if r["oracle"].get("oracle_direction") == r["direction"]
        )
        rev_match = len(abstention_hits) - asis_match
        print(f"abstention oracle coincidences: {len(abstention_hits)}")
        print(f"  as-is direction matches: {asis_match}/{len(abstention_hits)}")
        print(f"  reverse direction matches: {rev_match}/{len(abstention_hits)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
