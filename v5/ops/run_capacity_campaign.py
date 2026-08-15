"""Run the known-answer capacity campaign and write its receipt.

Usage:
    ./.venv/bin/python v5/ops/run_capacity_campaign.py --out <receipt.json>
    ./.venv/bin/python v5/ops/run_capacity_campaign.py --smoke   # timing only

The smoke mode exists to size the compute before the declaration is frozen; it
runs a shrunken law, reports elapsed seconds, and deliberately discards all
metrics so nothing is read before the declaration hash exists.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

from v5.research.capacity_campaign import (
    LAW,
    CampaignLaw,
    run_trial,
    wilson_lower,
)

REPO = Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(law: CampaignLaw, *, quiet: bool = False) -> dict:
    worlds = {
        "null": 0.0,
        "edge_small": law.effect_small_usd_per_sd,
        "edge_medium": law.effect_medium_usd_per_sd,
    }
    cells = []
    for sessions in law.training_sessions:
        for world_name, effect in worlds.items():
            started = time.time()
            trials = []
            for trial in range(law.trials_per_cell):
                seed = hash((sessions, world_name, trial)) % (2**31)
                trials.append(
                    run_trial(
                        training_sessions=sessions,
                        effect_usd_per_sd=effect,
                        trial_seed=seed,
                        law=law,
                    )
                )
            recovered = sum(t.recovered for t in trials)
            clean = sum(t.null_clean for t in trials)
            cell = {
                "training_sessions": sessions,
                "world": world_name,
                "effect_usd_per_sd": effect,
                "trials": law.trials_per_cell,
                "recovery_rate": recovered / law.trials_per_cell,
                "recovery_rate_wilson_lower": wilson_lower(
                    recovered, law.trials_per_cell
                ),
                "null_clean_rate": clean / law.trials_per_cell,
                "null_clean_rate_wilson_lower": wilson_lower(
                    clean, law.trials_per_cell
                ),
                "mean_oof_policy_ev_per_minute": sum(
                    t.oof_policy_ev_per_minute for t in trials
                )
                / law.trials_per_cell,
                "mean_oracle_ev_per_minute": sum(
                    t.oracle_ev_per_minute for t in trials
                )
                / law.trials_per_cell,
                "mean_entry_rate": sum(t.entry_rate for t in trials)
                / law.trials_per_cell,
                "mean_train_apparent_edge_usd": sum(
                    t.train_apparent_edge_usd for t in trials
                )
                / law.trials_per_cell,
                "elapsed_seconds": round(time.time() - started, 1),
            }
            cells.append(cell)
            if not quiet:
                print(
                    f"n={sessions} {world_name}: recovery {cell['recovery_rate']:.2f}"
                    f" clean {cell['null_clean_rate']:.2f}"
                    f" ev {cell['mean_oof_policy_ev_per_minute']:.2f}"
                    f"/{cell['mean_oracle_ev_per_minute']:.2f}"
                    f" ({cell['elapsed_seconds']}s)",
                    flush=True,
                )

    smallest_supported = None
    for sessions in law.training_sessions:
        small = next(
            c
            for c in cells
            if c["training_sessions"] == sessions and c["world"] == "edge_small"
        )
        null = next(
            c
            for c in cells
            if c["training_sessions"] == sessions and c["world"] == "null"
        )
        if (
            small["recovery_rate"] >= law.recovery_rate_required
            and null["null_clean_rate"] >= law.null_clean_rate_required
        ):
            smallest_supported = sessions
            break

    return {
        "schema_version": "v5.capacity-known-answer.v1",
        "created_on": time.strftime("%Y-%m-%d"),
        "law": dataclasses.asdict(law),
        "law_sha256": law.sha256(),
        "implementation_sha256": _file_sha256(
            REPO / "v5" / "research" / "capacity_campaign.py"
        ),
        "cells": cells,
        "verdict": {
            "smallest_supported_training_sessions_small_edge": smallest_supported,
            "decision_rule": (
                "smallest n with edge_small recovery_rate >= "
                f"{law.recovery_rate_required} and null clean_rate >= "
                f"{law.null_clean_rate_required}"
            ),
        },
        "integrity": {
            "real_targets_opened": False,
            "real_economics_opened": False,
            "vendor_contacted": False,
            "money_spent": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()

    if arguments.smoke:
        law = dataclasses.replace(
            LAW,
            training_sessions=(60,),
            trials_per_cell=2,
            max_epochs=10,
            plateau_patience=3,
            score_sessions=30,
        )
        started = time.time()
        run(law, quiet=True)
        print(
            f"smoke elapsed {time.time() - started:.1f}s for 6 shrunken trials;"
            " metrics discarded"
        )
        return

    receipt = run(LAW)
    body = json.dumps(receipt, indent=2, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = arguments.out or (
        REPO
        / "v4"
        / "audit"
        / "autoresearch"
        / "capacity_known_answer_2026_08_15"
        / "receipt_v2.json"
    )
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"receipt written to {out}")


if __name__ == "__main__":
    main()
