"""Wrapper around v3.layer2.build_simulated_l3_oracle that extends oracle
predictions onto trailing forward-walk days.

The standard build only predicts within rolling-window OOS days (W0-W12,
ending 2026-02-24). Days after that get NaN predictions, which is why
the forward-walk test couldn't measure with-oracle performance.

This wrapper monkey-patches `_day_to_window` to also map trailing days to
the LAST window's index — meaning W12's trained classifier predicts on
the forward-walk rows (2026-02-25 → 2026-04-24). This is exactly what we'd
do at deployment time: use the latest trained model for new bars.

Usage:
    .venv/bin/python -m scripts.build_l3_oracle_with_trailing --seed 42 \
        --output v3/artifacts/simulated_l3_oracle_..._fresh.npz
"""
from __future__ import annotations

import sys

from v3.layer2 import build_simulated_l3_oracle as bso


_orig_day_to_window = bso._day_to_window


def _day_to_window_with_trailing(windows):
    """Map every day → last window's index if the day is past last OOS end."""
    base = _orig_day_to_window(windows)
    if not windows:
        return base
    last_window_idx = windows[-1].window_idx
    last_oos_end = windows[-1].oos_days[-1]
    # We need access to the unique_days list passed to generate_rolling_windows,
    # but it's not stored on RollingWindow. The call site has unique_days
    # available; we extend at call site instead. This wrapper alone is
    # insufficient — patch the call site instead.
    return base


# Patch the call site instead: wrap `main` so that AFTER day_window_map is
# built, we also assign trailing days to the last window.

_orig_main = bso.main


def main_with_trailing() -> None:
    # Inject behavior: monkey-patch generate_rolling_windows return path so
    # the post-W12 days get mapped. We do this by wrapping _day_to_window
    # and patching it dynamically based on the unique_days available in
    # the calling frame. Simpler: edit the function to accept unique_days.
    pass  # placeholder


# Cleaner approach: patch _day_to_window to take unique_days from a
# module-level slot.

_unique_days_slot: list[str] = []


def _patched_day_to_window(windows):
    base = {d: w.window_idx for w in windows for d in w.oos_days}
    if windows and _unique_days_slot:
        last_window_idx = windows[-1].window_idx
        last_oos_end = windows[-1].oos_days[-1]
        for d in _unique_days_slot:
            if d > last_oos_end and d not in base:
                base[d] = last_window_idx
    return base


# To capture unique_days, we wrap generate_rolling_windows.
_orig_generate = bso.generate_rolling_windows


def _patched_generate(unique_days, **kw):
    _unique_days_slot.clear()
    _unique_days_slot.extend(list(unique_days))
    return _orig_generate(unique_days, **kw)


def install_patches() -> None:
    bso._day_to_window = _patched_day_to_window
    bso.generate_rolling_windows = _patched_generate
    print(f"[patch] _day_to_window patched to map trailing days to last window", flush=True)


if __name__ == "__main__":
    install_patches()
    sys.exit(_orig_main())
