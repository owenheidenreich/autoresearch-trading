"""Run the known-answer development harness and write its receipt.

Usage:
    ./.venv/bin/python -m v5.ops.run_capacity_campaign \
        --declaration v5/work/entry-exit-attribution/<DECLARATION>.json
    ./.venv/bin/python -m v5.ops.run_capacity_campaign --smoke   # timing only

The runner refuses to execute unless the declaration's law hash and
implementation hashes match the code that is about to run, and it binds the
declaration hash, per-trial results, seeds and software environment into the
receipt. Smoke mode sizes compute before a declaration is frozen; it runs a
shrunken law and discards all metrics.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy
import torch

from v5.research.capacity_campaign import (
    LAW,
    CampaignLaw,
    run_trial,
    stable_seed,
    wilson_lower,
    wilson_upper,
)

REPO = Path(__file__).resolve().parents[2]
IMPLEMENTATION = REPO / "v5" / "research" / "capacity_campaign.py"
RUNNER = Path(__file__).resolve()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "seed_derivation": "sha256, process-stable; PYTHONHASHSEED irrelevant",
    }


def verify_declaration(path: Path, law: CampaignLaw) -> dict:
    declaration = json.loads(path.read_text())
    problems = []
    if declaration["law_sha256"] != law.sha256():
        problems.append("declared law hash does not match the built CampaignLaw")
    if declaration["implementation_sha256"] != _file_sha256(IMPLEMENTATION):
        problems.append("declared implementation hash does not match capacity_campaign.py")
    if declaration.get("runner_sha256") != _file_sha256(RUNNER):
        problems.append("declared runner hash does not match run_capacity_campaign.py")
    if problems:
        raise SystemExit("declaration mismatch: " + "; ".join(problems))
    return declaration


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
                seed = stable_seed("capacity-v3", sessions, world_name, trial)
                trials.append(
                    run_trial(
                        training_sessions=sessions,
                        effect_usd_per_sd=effect,
                        trial_seed=seed,
                        law=law,
                    )
                )
            recovered = sum(t.recovered for t in trials)
            abstained = sum(t.null_abstained for t in trials)
            cell = {
                "training_sessions": sessions,
                "world": world_name,
                "effect_usd_per_sd": effect,
                "trials": law.trials_per_cell,
                "recovery_rate": recovered / law.trials_per_cell,
                "recovery_rate_wilson_lower": wilson_lower(
                    recovered, law.trials_per_cell
                ),
                "null_abstention_rate": abstained / law.trials_per_cell,
                "null_abstention_wilson_lower": wilson_lower(
                    abstained, law.trials_per_cell
                ),
                "entry_rate_wilson_upper": wilson_upper(
                    law.trials_per_cell - abstained, law.trials_per_cell
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
                "trial_records": [dataclasses.asdict(t) for t in trials],
            }
            cells.append(cell)
            if not quiet:
                print(
                    f"n={sessions} {world_name}: recovery {cell['recovery_rate']:.2f}"
                    f" abstain {cell['null_abstention_rate']:.2f}"
                    f" ev {cell['mean_oof_policy_ev_per_minute']:.2f}"
                    f"/{cell['mean_oracle_ev_per_minute']:.2f}"
                    f" ({cell['elapsed_seconds']}s)",
                    flush=True,
                )

    return {
        "schema_version": "v5.capacity-known-answer.v3",
        "classification": (
            "DEVELOPMENT DIAGNOSTIC of the entry training law on one synthetic "
            "task. Not a power measurement, not a full-pipeline rehearsal; "
            "cannot close the lifecycle member or authorize a fit or purchase."
        ),
        "created_on": time.strftime("%Y-%m-%d"),
        "law": dataclasses.asdict(law),
        "law_sha256": law.sha256(),
        "implementation_sha256": _file_sha256(IMPLEMENTATION),
        "runner_sha256": _file_sha256(RUNNER),
        "environment": _environment(),
        "cells": cells,
        "integrity": {
            "real_targets_opened": False,
            "real_economics_opened": False,
            "vendor_contacted": False,
            "money_spent": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, default=None)
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

    if arguments.declaration is None:
        raise SystemExit("a real run requires --declaration")
    declaration = verify_declaration(arguments.declaration, LAW)

    receipt = run(LAW)
    receipt["declaration_path"] = str(
        arguments.declaration.resolve().relative_to(REPO)
    )
    receipt["declaration_sha256"] = declaration["declaration_sha256"]
    body = json.dumps(receipt, indent=2, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = arguments.out or (
        REPO
        / "v4"
        / "audit"
        / "autoresearch"
        / "capacity_known_answer_2026_08_15"
        / "receipt_v3.json"
    )
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"receipt written to {out}")


if __name__ == "__main__":
    main()
