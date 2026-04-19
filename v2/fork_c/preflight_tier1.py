"""Fork C Phase 1 preflight audit.

Reads ``tier1_dataset.csv``, applies a chronological 60/20/20 split, and
reports the class-balance and timing numbers that the plan requires before
any modeling can begin.

Halt triggers (plan §"Target-timing preflight"):

1. **Late-positives trigger:** ≥ 60% of positive-labeled days have
   ``first_qualifying_time_et > 10:00 ET``. If this fires, the task is
   largely forecasting later-day behavior rather than detecting imminent
   behavior. Human decides to continue with explicit forecast framing,
   move the cutoff, or redefine the target.

2. **Minority-count trigger:** any split has ``< 5`` samples in either
   class. Human decides to rebalance splits, reduce to train/test only,
   or accept inflated CIs.

The script exits with code 0 if neither trigger fires.
It exits with code 10 if one or both trigger, and in that case
``preflight.json`` records the triggers, and ``train_tier1.py`` refuses
to run until a sibling ``preflight_acknowledged.json`` is created by hand.

Usage::
    python3 -m v2.fork_c.preflight_tier1 \\
        --dataset v2/fork_c/tier1_dataset.csv \\
        --out v2/artifacts/fork_c_tier1/preflight.json
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


LATE_POSITIVE_THRESHOLD = 0.60   # ≥60% late-positives → trigger
MINIMUM_MINORITY_PER_SPLIT = 5   # <5 in either class in any split → trigger

TIME_BUCKETS = [
    ("<=10:00", (0, 600)),        # first_qualifying_time in minutes-of-day <= 600 (10:00)
    ("10:01-11:00", (601, 660)),
    ("11:01-12:00", (661, 720)),
    ("12:01-close", (721, 960)),  # up to ~16:00
]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hhmm(s: str) -> Optional[int]:
    """Return minutes-of-day or None if empty/unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None


def bucket_time(minutes: Optional[int]) -> str:
    if minutes is None:
        return "missing"
    for name, (lo, hi) in TIME_BUCKETS:
        if lo <= minutes <= hi:
            return name
    return "other"


def split_indices(n: int) -> tuple[range, range, range]:
    """Chronological 60/20/20 split boundaries (indices into a date-sorted list)."""
    train_end = n * 60 // 100
    val_end = n * 80 // 100
    return range(0, train_end), range(train_end, val_end), range(val_end, n)


def summarize_split(rows: list[dict], name: str) -> dict:
    labels = [int(r["label"]) for r in rows]
    confs = [r["confidence"] for r in rows]
    evis = [r.get("evidence_type", "") or "missing" for r in rows]
    sides = [r.get("side", "") or "empty" for r in rows]
    label_counts = Counter(labels)
    return {
        "name": name,
        "n": len(rows),
        "date_start": rows[0]["date"] if rows else None,
        "date_end": rows[-1]["date"] if rows else None,
        "n_positive": int(label_counts.get(1, 0)),
        "n_negative": int(label_counts.get(0, 0)),
        "positive_rate": (
            float(label_counts.get(1, 0) / len(rows)) if rows else 0.0
        ),
        "confidence": dict(Counter(confs)),
        "evidence_type": dict(Counter(evis)),
        "side": dict(Counter(sides)),
    }


def positive_timing_distribution(rows: list[dict]) -> dict:
    positives = [r for r in rows if int(r["label"]) == 1]
    minutes = [parse_hhmm(r.get("first_qualifying_time_et", "")) for r in positives]
    buckets = Counter(bucket_time(m) for m in minutes)
    total = len(positives)
    late = sum(1 for m in minutes if m is not None and m > 600)
    late_frac = (late / total) if total else 0.0
    return {
        "n_positives": total,
        "buckets": {k: int(buckets.get(k, 0)) for k in [b[0] for b in TIME_BUCKETS] + ["missing", "other"]},
        "n_late_positives": int(late),
        "late_positive_fraction": late_frac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2/fork_c/tier1_dataset.csv")
    ap.add_argument("--out", default="v2/artifacts/fork_c_tier1/preflight.json")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    with open(dataset_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["date"])
    n = len(rows)
    if n == 0:
        print("ERROR: dataset has 0 rows.", file=sys.stderr)
        return 2

    train_idx, val_idx, test_idx = split_indices(n)
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    split_boundaries = {
        "train_start": train_rows[0]["date"],
        "train_end": train_rows[-1]["date"],
        "val_start": val_rows[0]["date"],
        "val_end": val_rows[-1]["date"],
        "test_start": test_rows[0]["date"],
        "test_end": test_rows[-1]["date"],
    }

    per_split = {
        "train": summarize_split(train_rows, "train"),
        "val": summarize_split(val_rows, "val"),
        "test": summarize_split(test_rows, "test"),
    }

    overall_labels = Counter(int(r["label"]) for r in rows)
    overall = {
        "n": n,
        "n_positive": int(overall_labels.get(1, 0)),
        "n_negative": int(overall_labels.get(0, 0)),
        "base_rate": float(overall_labels.get(1, 0) / n),
    }

    timing = positive_timing_distribution(rows)

    triggers = []
    if timing["late_positive_fraction"] >= LATE_POSITIVE_THRESHOLD:
        triggers.append(
            {
                "name": "late_positives",
                "details": (
                    f"{timing['n_late_positives']}/{timing['n_positives']} "
                    f"positives ({timing['late_positive_fraction']:.1%}) have "
                    f"first_qualifying_time_et > 10:00 ET; threshold is "
                    f"{LATE_POSITIVE_THRESHOLD:.0%}. Task is largely "
                    f"forecasting later-day behavior. Choose: continue with "
                    f"forecast framing, move cutoff later, or redefine target."
                ),
            }
        )
    for name, split in per_split.items():
        min_class = min(split["n_positive"], split["n_negative"])
        if min_class < MINIMUM_MINORITY_PER_SPLIT:
            triggers.append(
                {
                    "name": "minority_count",
                    "details": (
                        f"Split '{name}' has only {min_class} samples in the "
                        f"smaller class "
                        f"(pos={split['n_positive']}, neg={split['n_negative']}); "
                        f"threshold is {MINIMUM_MINORITY_PER_SPLIT}. Choose: "
                        f"rebalance split ratios, reduce to train/test only, "
                        f"or accept inflated uncertainty."
                    ),
                }
            )

    preflight = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_of_file(dataset_path),
        "overall": overall,
        "per_split": per_split,
        "split_boundaries": split_boundaries,
        "positive_timing": timing,
        "triggers": triggers,
        "halted": bool(triggers),
        "thresholds": {
            "late_positive_fraction": LATE_POSITIVE_THRESHOLD,
            "minimum_minority_per_split": MINIMUM_MINORITY_PER_SPLIT,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2, sort_keys=True)

    print("=" * 60)
    print(f"Fork C Phase 1 preflight — {dataset_path}")
    print("=" * 60)
    print(
        f"Overall: n={overall['n']}  pos={overall['n_positive']}  "
        f"neg={overall['n_negative']}  base_rate={overall['base_rate']:.3f}"
    )
    print()
    print("Chronological 60/20/20 split boundaries:")
    for k, v in split_boundaries.items():
        print(f"  {k:12s} {v}")
    print()
    for name in ("train", "val", "test"):
        s = per_split[name]
        print(
            f"  {name:5s}: n={s['n']:3d}  pos={s['n_positive']:3d}  "
            f"neg={s['n_negative']:3d}  pos_rate={s['positive_rate']:.3f}  "
            f"[{s['date_start']} .. {s['date_end']}]"
        )
        print(f"         confidence: {s['confidence']}")
    print()
    print(
        f"Positive timing (first_qualifying_time_et): "
        f"n={timing['n_positives']}  late(>10:00)={timing['n_late_positives']} "
        f"({timing['late_positive_fraction']:.1%})"
    )
    for k, v in timing["buckets"].items():
        print(f"  {k:12s} {v}")
    print()
    if triggers:
        print(f"HALT: {len(triggers)} preflight trigger(s) fired:")
        for t in triggers:
            print(f"  [{t['name']}] {t['details']}")
        print(
            "\nCreate v2/artifacts/fork_c_tier1/preflight_acknowledged.json "
            "to record the human decision before running train_tier1.py."
        )
        print(f"Preflight JSON: {out_path}")
        return 10
    print("PASS: no preflight triggers fired.")
    print(f"Preflight JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
