"""Fork C Phase 1 feature extraction at bar_of_day == 30 cutoff (10:00 ET).

For each row in ``tier1_labels.csv`` this:
- Extracts ``X_sim`` at the session's ``bar_of_day == 30`` row (the point-in-time
  10:00 ET feature vector) — shape (n_base_features,).
- Computes 5 aggregates from bars 0-30 only:
    am_session_range_bps, am_session_end_vs_open_bps, ovn_proxy_direction,
    prior_session_end_ret, vwap_sigma_session_frac.
- Joins with the label row and emits ``tier1_dataset.csv``.

Design notes
------------
- ``n_base_features`` is read from ``X_sim.shape[1]`` and stamped into the
  emitted metadata. We do NOT hardcode 52 or 79 — the extractor works with
  whatever canonical manifest is provided.
- ``--data`` defaults to the main worktree's ``data.pt`` because the fork_c
  worktree deliberately does not track the 265 MB manifest. The loaded file's
  SHA256 is recorded in the output metadata for reproducibility.
- The σ-band scalar reuses Fork A1's ``compute_daily_vwap_sigma`` with unit
  volumes — same definition as ``v2/strategies/fork_a1_stage1.py:254-256``.
  Using unit volumes matches how Fork A1 computed it for continuity; switching
  to real volumes would be a new feature and out of scope for Phase 1.

Usage::

    python3 -m v2.fork_c.extract_tier1_features \\
        --data /Users/gduby/Documents/autoresearch-trading/v2/data.pt \\
        --labels v2/fork_c/tier1_labels.csv \\
        --out v2/fork_c/tier1_dataset.csv \\
        --out-meta v2/fork_c/tier1_dataset_meta.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from v2.strategies.session_state import compute_daily_vwap_sigma


CUTOFF_BAR = 30             # bar_of_day at 10:00 ET
AGGREGATE_WINDOW = 31       # bars 0..30 inclusive
AGGREGATE_NAMES = [
    "am_session_range_bps",
    "am_session_end_vs_open_bps",
    "ovn_proxy_direction",
    "prior_session_end_ret",
    "vwap_sigma_session_frac",
]
DEFAULT_DATA = "/Users/gduby/Documents/autoresearch-trading/v2/data.pt"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def session_index(
    dates: list[str], bar_of_day: np.ndarray
) -> dict[str, tuple[int, int]]:
    """Map ``date_str -> (day_start_bar, session_length)``.

    Treats ``bar_of_day == 0`` as the session-start marker and verifies that
    each session's bars form a contiguous run. Raises on any irregularity —
    the manifest is supposed to be strictly regular RTH sessions.
    """
    out: dict[str, tuple[int, int]] = {}
    n = len(dates)
    i = 0
    while i < n:
        if int(bar_of_day[i]) != 0:
            raise ValueError(
                f"Expected bar_of_day == 0 at session start; got "
                f"{int(bar_of_day[i])} at index {i} (date={dates[i]})"
            )
        j = i + 1
        while j < n and int(bar_of_day[j]) != 0:
            j += 1
        d = dates[i]
        if d in out:
            raise ValueError(f"Duplicate session for date {d}")
        out[d] = (i, j - i)
        i = j
    return out


def compute_aggregates(
    day_start: int,
    spot_prices: np.ndarray,
    prior_last_spot: Optional[float],
    two_prior_last_spot: Optional[float],
    vwap_sigma_scalar: float,
) -> np.ndarray:
    """Return the 5-element aggregate vector for one session.

    Every input must be derivable from bars <= day_start+30 (plus prior
    sessions' closing spots — all strictly point-in-time).
    """
    window = spot_prices[day_start : day_start + AGGREGATE_WINDOW]
    if window.shape[0] != AGGREGATE_WINDOW:
        raise ValueError(
            f"Short session at day_start={day_start}: only {window.shape[0]} bars"
        )
    open_spot = float(window[0])
    spot_cut = float(window[-1])
    if open_spot <= 0:
        raise ValueError(f"Nonpositive open spot at day_start={day_start}")
    max_spot = float(np.max(window))
    min_spot = float(np.min(window))

    am_range_bps = (max_spot - min_spot) / open_spot * 1e4
    am_end_vs_open_bps = (spot_cut - open_spot) / open_spot * 1e4

    if prior_last_spot is None or prior_last_spot <= 0:
        ovn_dir = 0.0
    else:
        ovn_dir = float(np.sign(open_spot - prior_last_spot))

    if (
        prior_last_spot is None
        or two_prior_last_spot is None
        or two_prior_last_spot <= 0
    ):
        prior_end_ret = 0.0
    else:
        prior_end_ret = (prior_last_spot - two_prior_last_spot) / two_prior_last_spot

    sigma = float(vwap_sigma_scalar) if np.isfinite(vwap_sigma_scalar) else 0.0

    return np.array(
        [am_range_bps, am_end_vs_open_bps, ovn_dir, prior_end_ret, sigma],
        dtype=np.float64,
    )


def build_session_sigmas(
    sorted_dates: list[str],
    sessions: dict[str, tuple[int, int]],
    spot_prices: np.ndarray,
) -> np.ndarray:
    """Fork A1's ``compute_daily_vwap_sigma`` with unit volumes per session.

    Using ``np.ones_like(spot)`` mirrors
    ``v2/strategies/fork_a1_stage1.py:254-256`` — VWAP reduces to a cumulative
    mean. This keeps the σ-band definition identical to Fork A1 so the
    Phase-1 feature is continuous with prior work.
    """
    spot_by_day: list[np.ndarray] = []
    for d in sorted_dates:
        day_start, n = sessions[d]
        spot_by_day.append(spot_prices[day_start : day_start + n])
    vol_by_day = [np.ones_like(s) for s in spot_by_day]
    return compute_daily_vwap_sigma(spot_by_day, vol_by_day)


def build_prior_close_tables(
    sorted_dates: list[str],
    sessions: dict[str, tuple[int, int]],
    spot_prices: np.ndarray,
) -> tuple[dict[str, Optional[float]], dict[str, Optional[float]]]:
    """Return ``(prior_close_by_date, two_prior_close_by_date)``.

    The first session has ``None`` priors; the second has ``None`` for
    two-prior. Strictly point-in-time — the session's own close is not read
    until after its entry is recorded."""
    prior: dict[str, Optional[float]] = {}
    two_prior: dict[str, Optional[float]] = {}
    prev: Optional[float] = None
    two_prev: Optional[float] = None
    for d in sorted_dates:
        prior[d] = prev
        two_prior[d] = two_prev
        day_start, n = sessions[d]
        two_prev = prev
        prev = float(spot_prices[day_start + n - 1])
    return prior, two_prior


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--labels", default="v2/fork_c/tier1_labels.csv")
    ap.add_argument("--label-rules", default="v2/fork_c/label_rules.md")
    ap.add_argument("--out", default="v2/fork_c/tier1_dataset.csv")
    ap.add_argument("--out-meta", default="v2/fork_c/tier1_dataset_meta.json")
    args = ap.parse_args()

    data_path = Path(args.data)
    labels_path = Path(args.labels)
    rules_path = Path(args.label_rules)
    out_path = Path(args.out)
    meta_path = Path(args.out_meta)

    if not data_path.exists():
        print(f"ERROR: manifest not found: {data_path}", file=sys.stderr)
        return 2

    print(f"Loading manifest: {data_path}")
    manifest = torch.load(data_path, map_location="cpu", weights_only=False)
    X_sim = manifest["X_sim"].numpy().astype(np.float64)
    dates: list[str] = list(manifest["dates"])
    bar_of_day = manifest["bar_of_day"].numpy().astype(np.int32)
    spot_prices = manifest["spot_prices"].numpy().astype(np.float64)
    feature_names: list[str] = list(manifest["feature_names"])

    n_bars, n_base = int(X_sim.shape[0]), int(X_sim.shape[1])
    print(f"  N_bars = {n_bars}, n_base_features = {n_base}")
    if n_base != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != X_sim.shape[1] {n_base}"
        )
    if len(dates) != n_bars:
        raise ValueError(f"dates length {len(dates)} != N_bars {n_bars}")
    if bar_of_day.shape[0] != n_bars:
        raise ValueError(
            f"bar_of_day length {bar_of_day.shape[0]} != N_bars {n_bars}"
        )

    sessions = session_index(dates, bar_of_day)
    sorted_dates = sorted(sessions.keys())
    print(f"  Indexed {len(sessions)} distinct sessions.")

    sigmas = build_session_sigmas(sorted_dates, sessions, spot_prices)
    sigma_by_date = {d: float(sigmas[i]) for i, d in enumerate(sorted_dates)}

    prior_close_by_date, two_prior_close_by_date = build_prior_close_tables(
        sorted_dates, sessions, spot_prices
    )

    with open(labels_path, newline="", encoding="utf-8") as f:
        label_rows = list(csv.DictReader(f))
    print(f"Loaded {len(label_rows)} canonical label rows from {labels_path}")

    out_rows: list[dict] = []
    skipped: list[dict] = []
    nonfinite_per_date: list[tuple[str, int]] = []

    for r in label_rows:
        d = r["date"]
        if d not in sessions:
            skipped.append({"date": d, "reason": "date_not_in_manifest"})
            continue
        day_start, n = sessions[d]
        if n < AGGREGATE_WINDOW:
            skipped.append({"date": d, "reason": f"short_session_n={n}"})
            continue
        cutoff_idx = day_start + CUTOFF_BAR
        bd = int(bar_of_day[cutoff_idx])
        if bd != CUTOFF_BAR:
            raise ValueError(
                f"{d}: expected bar_of_day == {CUTOFF_BAR} at offset "
                f"{CUTOFF_BAR}, got {bd}"
            )
        base_row = X_sim[cutoff_idx].copy()
        nonfinite = int(np.sum(~np.isfinite(base_row)))
        if nonfinite:
            nonfinite_per_date.append((d, nonfinite))
            base_row = np.where(np.isfinite(base_row), base_row, 0.0)

        agg = compute_aggregates(
            day_start,
            spot_prices,
            prior_close_by_date[d],
            two_prior_close_by_date[d],
            sigma_by_date[d],
        )

        out = {
            "date": d,
            "label": int(r["label"]),
            "confidence": r["confidence"],
            "evidence_type": r.get("evidence_type", "") or "",
            "side": r.get("side", "") or "",
            "hedge_only": r.get("hedge_only", "") or "",
            "first_qualifying_time_et": r.get("first_qualifying_time_et", "") or "",
        }
        for i, name in enumerate(feature_names):
            out[f"f_{name}"] = float(base_row[i])
        for i, name in enumerate(AGGREGATE_NAMES):
            out[f"agg_{name}"] = float(agg[i])
        out_rows.append(out)

    if not out_rows:
        print("ERROR: extractor produced zero rows.", file=sys.stderr)
        return 3

    label_cols = [
        "date",
        "label",
        "confidence",
        "evidence_type",
        "side",
        "hedge_only",
        "first_qualifying_time_et",
    ]
    feature_cols = [f"f_{n}" for n in feature_names]
    agg_cols = [f"agg_{n}" for n in AGGREGATE_NAMES]
    fieldnames = label_cols + feature_cols + agg_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    data_sha = sha256_of_file(data_path)
    labels_sha = sha256_of_file(labels_path)
    rules_sha = sha256_of_file(rules_path) if rules_path.exists() else None

    skipped_by_reason = Counter(s["reason"] for s in skipped)
    label_dist = Counter(r["label"] for r in out_rows)
    conf_dist = Counter(r["confidence"] for r in out_rows)

    meta = {
        "cutoff_bar": CUTOFF_BAR,
        "aggregate_window_bars": AGGREGATE_WINDOW,
        "n_base_features": n_base,
        "n_aggregate_features": len(AGGREGATE_NAMES),
        "n_total_features": n_base + len(AGGREGATE_NAMES),
        "feature_names_base": feature_names,
        "aggregate_names": AGGREGATE_NAMES,
        "n_label_rows": len(label_rows),
        "n_extracted": len(out_rows),
        "n_skipped": len(skipped),
        "skipped_by_reason": dict(skipped_by_reason),
        "skipped_rows": skipped,
        "nonfinite_feature_cells": [
            {"date": d, "count": c} for d, c in nonfinite_per_date
        ],
        "label_distribution": {str(k): v for k, v in label_dist.items()},
        "confidence_distribution": dict(conf_dist),
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "label_rules_path": str(rules_path),
        "label_rules_sha256": rules_sha,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("-" * 60)
    print(f"Extracted {len(out_rows)} rows → {out_path}")
    print(f"  feature columns: {n_base} base + {len(AGGREGATE_NAMES)} aggregate")
    print(f"  label dist:      {dict(label_dist)}")
    print(f"  confidence dist: {dict(conf_dist)}")
    if skipped:
        print(f"  SKIPPED {len(skipped)} rows:")
        for reason, count in skipped_by_reason.items():
            print(f"    {reason}: {count}")
        for s in skipped:
            print(f"    - {s['date']}: {s['reason']}")
    if nonfinite_per_date:
        print(
            f"  nonfinite cells replaced with 0 on "
            f"{len(nonfinite_per_date)} dates"
        )
    print(f"Metadata → {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
