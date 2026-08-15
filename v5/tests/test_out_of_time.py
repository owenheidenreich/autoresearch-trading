"""The training window must end before the scoring window starts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.train_selective_policy import FEATURES


def _table(sessions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for s in sessions:
        for minute in range(575, 900, 5):
            for k in range(6):
                rows.append({
                    "session": s, "entry_minute": f"{minute // 60:02d}:{minute % 60:02d}",
                    "is_call": k % 2 == 0, "spread_usd": 20.0,
                    **{f: float(rng.normal()) for f in FEATURES if f != "is_call_int"},
                })
    t = pd.DataFrame(rows)
    t["entry_premium"] = 1000.0 + 100 * rng.normal(size=len(t))
    t["delta"] = 0.5
    t["net_label"] = rng.normal(0, 50, len(t))
    t["net_label_fair"] = t["net_label"] + 20.0
    t["profitable"] = (t["net_label"] > 0).astype(int)
    t["profitable_fair"] = (t["net_label_fair"] > 0).astype(int)
    return t


def test_scoring_window_is_strictly_after_the_training_window(tmp_path) -> None:
    early = [f"2024-01-{d:02d}" for d in range(1, 29)]
    late = [f"2025-06-{d:02d}" for d in range(1, 29)]
    # Overlap on purpose: the late sessions also appear in the training table.
    _table(early + late, 1).to_parquet(tmp_path / "train.parquet")
    _table(late, 2).to_parquet(tmp_path / "score.parquet")

    out = tmp_path / "receipt.json"
    got = subprocess.run(
        [sys.executable, "-m", "v5.ops.score_policy_out_of_time",
         "--train-table", str(tmp_path / "train.parquet"),
         "--score-table", str(tmp_path / "score.parquet"), "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert got.returncode == 0, got.stderr
    payload = json.loads(out.read_text())
    assert payload["train_window"][1] < payload["score_window"][0]
    assert payload["train_sessions"] == len(early)
