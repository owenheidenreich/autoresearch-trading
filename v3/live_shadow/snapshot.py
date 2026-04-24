from __future__ import annotations

import json
import os
from typing import Iterable

from v3.live_shadow.schema import DecisionSnapshot


def append_decision_snapshot(path: str, snapshot: DecisionSnapshot) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(snapshot.to_json_dict(), sort_keys=True))
        f.write("\n")


def read_decision_snapshots(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stale_quote_count(snapshots: Iterable[DecisionSnapshot]) -> int:
    total = 0
    for snapshot in snapshots:
        for candidate in snapshot.candidates:
            if candidate.quote is not None and candidate.quote.stale:
                total += 1
    return total

