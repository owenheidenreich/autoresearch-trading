"""Operator-facing status summary for the v3 pure-RL surface."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch

from v3.core.market_state import DEFAULT_MARKET_STATE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3"
RESULTS_PATH = V3_ROOT / "results.tsv"
DEPLOY_STATE_PATH = PROJECT_ROOT / ".deploy-state"


def load_dataset_summary(data_path: str) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    meta = data.get("metadata", {})
    return {
        "path": data_path,
        "fingerprint": meta.get("fingerprint", "unknown"),
        "version": meta.get("version", "unknown"),
        "schema": meta.get("chain_schema_version", "unknown"),
        "days": len(sorted(set(data.get("dates", [])))),
        "bars": int(data["X"].shape[0]),
        "features": int(data["X"].shape[1]),
        "max_contracts": int(meta.get("max_contracts_per_bar", -1)),
    }


def load_results_summary() -> dict:
    if not RESULTS_PATH.exists():
        return {"latest": None, "best": None, "rows": 0}
    with RESULTS_PATH.open(newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    entries = []
    for row in rows[1:]:
        if len(row) < 8:
            continue
        entries.append(
            {
                "experiment_id": row[0],
                "created_at": row[1],
                "mask": row[2],
                "score": float(row[3]),
                "return": float(row[4]),
                "drawdown": float(row[5]),
                "sortino": float(row[6]),
                "trades": int(row[7]),
                "status": row[8] if len(row) > 8 else "",
            }
        )
    latest = entries[-1] if entries else None
    best = max(entries, key=lambda x: x["score"]) if entries else None
    return {"latest": latest, "best": best, "rows": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the current v3 status")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--market-state", default=DEFAULT_MARKET_STATE_PATH)
    args = parser.parse_args()

    dataset = load_dataset_summary(args.data)
    results = load_results_summary()

    print("ART2 v3 Status")
    print("Phase: PURE-RL DYNAMIC RESET")
    print("Mission: learn trade, contract, size, and active management from replayed outcomes")
    print()
    print("Dataset")
    print(f"  Path: {dataset['path']}")
    print(f"  Fingerprint: {dataset['fingerprint']}")
    print(f"  Version: {dataset['version']} ({dataset['schema']})")
    print(f"  Days/Bars: {dataset['days']} / {dataset['bars']}")
    print(f"  Features / max contracts: {dataset['features']} / {dataset['max_contracts']}")
    print()
    print("Operational Readiness")
    print(f"  Program: {V3_ROOT / 'program.md'}")
    print(f"  Pre-run gate: python3 -m v3.ops.pre_run_gate --data {args.data} --market-state {args.market_state}")
    print(
        "  Market state: "
        + (f"ready ({args.market_state})" if Path(args.market_state).exists() else f"missing ({args.market_state})")
    )
    print(
        f"  Build market state: python3 -m v3.build_market_state --data {args.data} --output {args.market_state}"
    )
    print(f"  sshpass: {'present' if shutil.which('sshpass') else 'missing'}")
    print(
        "  GPU state: "
        + ("booted (.deploy-state present)" if DEPLOY_STATE_PATH.exists() else "not booted (run ./v3/ops/deploy.sh boot)")
    )
    print(
        "  Local run: python3 -m v3.ops.run_experiment "
        f"--id v3_exp_001 --data {args.data} --market-state {args.market_state} "
        "--device cpu --updates 10 --rollout-days 8"
    )
    print(
        "  GPU wrapper: ./v3/ops/deploy.sh run_one v3_exp_001 "
        "--updates 10 --rollout-days 8"
    )
    print()
    print("Results")
    print(f"  Logged runs: {results['rows']}")
    if results["latest"]:
        latest = results["latest"]
        print(
            f"  Latest: {latest['experiment_id']} score={latest['score']:.4f} "
            f"return={latest['return']:.4f} dd={latest['drawdown']:.4f} status={latest['status']}"
        )
    if results["best"]:
        best = results["best"]
        print(
            f"  Best: {best['experiment_id']} score={best['score']:.4f} "
            f"return={best['return']:.4f} dd={best['drawdown']:.4f}"
        )


if __name__ == "__main__":
    main()
