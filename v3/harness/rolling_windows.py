"""Phase R1 — Rolling-window harness for methodology overhaul.

Generates disjoint 60-day non-overlapping OOS windows from the 986
cached days, with expanding training pools. Used by Phase R3-R6 to
re-evaluate V0 vs V1 across a meaningful sample size (780 OOS days
vs the previous 20).

Unlike v2/core/walkforward.py (which works backward from end to build
5 expanding folds), this harness works forward from a min_train_days
offset and slides N × oos_days windows forward until dataset end.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence


DEFAULT_MIN_TRAIN_DAYS = 180
DEFAULT_OOS_DAYS = 60
DEFAULT_VAL_DAYS = 40


@dataclass
class RollingWindow:
    """One rolling-window specification."""
    window_idx: int
    window_id: str
    train_days: list[str]    # expanding: all days before oos window
    val_days: list[str]      # last N train_days for calibration
    oos_days: list[str]      # 60 consecutive days

    @property
    def train_n(self) -> int: return len(self.train_days)

    @property
    def val_n(self) -> int: return len(self.val_days)

    @property
    def oos_n(self) -> int: return len(self.oos_days)


def _compute_window_id(
    train_start: str, train_end: str,
    val_start: str, val_end: str,
    oos_start: str, oos_end: str,
) -> str:
    """Stable SHA-1 of the six boundary dates. Same windows -> same seed."""
    key = f"{train_start}|{train_end}|{val_start}|{val_end}|{oos_start}|{oos_end}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def generate_rolling_windows(
    unique_days: Sequence[str],
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    oos_days: int = DEFAULT_OOS_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> list[RollingWindow]:
    """Generate disjoint forward-sliding OOS windows.

    Window k:
      train = unique_days[:min_train_days + k*oos_days]
      val   = last val_days of train
      oos   = unique_days[min_train_days + k*oos_days : min_train_days + (k+1)*oos_days]

    Stops when the OOS window would extend beyond dataset end.
    Assumes `unique_days` is already sorted chronologically.
    """
    all_days = list(unique_days)
    n = len(all_days)
    if n < min_train_days + oos_days:
        raise ValueError(
            f"Need at least {min_train_days + oos_days} days; got {n}"
        )
    if min_train_days < val_days + 50:
        raise ValueError(
            f"min_train_days={min_train_days} too small for val_days={val_days}"
        )

    windows: list[RollingWindow] = []
    window_idx = 0
    train_end_idx = min_train_days

    while train_end_idx + oos_days <= n:
        oos_slice = all_days[train_end_idx : train_end_idx + oos_days]
        train_slice = all_days[:train_end_idx]
        val_slice = train_slice[-val_days:]

        window_id = _compute_window_id(
            train_start=train_slice[0], train_end=train_slice[-1],
            val_start=val_slice[0], val_end=val_slice[-1],
            oos_start=oos_slice[0], oos_end=oos_slice[-1],
        )
        windows.append(RollingWindow(
            window_idx=window_idx,
            window_id=window_id,
            train_days=list(train_slice),
            val_days=list(val_slice),
            oos_days=list(oos_slice),
        ))
        window_idx += 1
        train_end_idx += oos_days

    return windows


def verify_windows(windows: Sequence[RollingWindow]) -> None:
    """Assert disjoint OOS sets, monotonic dates, no overlap."""
    assert len(windows) > 0, "no windows generated"

    # monotonic OOS start dates
    oos_starts = [w.oos_days[0] for w in windows]
    assert oos_starts == sorted(oos_starts), "OOS windows are not chronologically ordered"

    # disjoint OOS day sets
    oos_sets = [set(w.oos_days) for w in windows]
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            overlap = oos_sets[i] & oos_sets[j]
            assert not overlap, f"windows {i},{j} overlap on: {sorted(overlap)[:5]}"

    # within each window, train and oos are disjoint
    for w in windows:
        intersect = set(w.train_days) & set(w.oos_days)
        assert not intersect, f"window {w.window_idx} has train∩oos overlap"

    # within each window, val ⊆ train
    for w in windows:
        assert set(w.val_days) <= set(w.train_days), (
            f"window {w.window_idx} val is not a subset of train"
        )

    # each window's train OOS = train + oos last day > oos first day ordering
    for w in windows:
        assert w.train_days[-1] < w.oos_days[0], (
            f"window {w.window_idx} train extends into oos"
        )


def print_window_summary(windows: Sequence[RollingWindow]) -> None:
    """Print per-window day counts and date ranges."""
    print(f"Generated {len(windows)} rolling windows:")
    print(f"{'idx':<5}{'train_n':>9}{'val_n':>8}{'oos_n':>7}  "
          f"{'train_range':<28}{'oos_range':<28}{'window_id':<14}")
    for w in windows:
        tr = f"{w.train_days[0]}..{w.train_days[-1]}"
        oo = f"{w.oos_days[0]}..{w.oos_days[-1]}"
        print(f"{w.window_idx:<5}{w.train_n:>9}{w.val_n:>8}{w.oos_n:>7}  "
              f"{tr:<28}{oo:<28}{w.window_id:<14}")
    total_oos = sum(w.oos_n for w in windows)
    print(f"\nTotal OOS coverage: {total_oos} days across {len(windows)} windows")


if __name__ == "__main__":
    # Smoke test: generate from v2/data.pt dates
    import torch
    d = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    dates = d["dates"]
    uniq = sorted(set(dates))
    print(f"Loaded {len(uniq)} unique days: {uniq[0]} .. {uniq[-1]}\n")

    windows = generate_rolling_windows(uniq)
    verify_windows(windows)
    print_window_summary(windows)
