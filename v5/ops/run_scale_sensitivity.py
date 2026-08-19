"""Run the job-46 scale-sensitivity study and write a receipt.

Read-only: no fit, no purchase, no vendor contact, no reserved sessions, and no
read of the partially-acquired backfill. The declaration is self-hashed and
verified before any outcome is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.scale_sensitivity import (
    SESSIONS_PER_YEAR,
    SignedLaw,
    build_population,
    simulate_scale,
    stable_seed,
)

PATHS_PARQUET = "/Volumes/AR_TRADING_DATA/derived/quoted_exit_paths.parquet"
DECLARATION_SCHEMA = "v5.scale-sensitivity-declaration.v1"


def _verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("declaration_sha256")
    unsigned = {k: v for k, v in value.items() if k != "declaration_sha256"}
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"declaration self-hash mismatch: {path}")
    if value.get("schema_version") != DECLARATION_SCHEMA:
        raise RuntimeError("declaration schema mismatch")
    for relative, digest in value["implementation_hashes"].items():
        if file_sha256(Path(relative)) != digest:
            raise RuntimeError(f"implementation hash mismatch: {relative}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    declaration = _verified(args.declaration)
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite receipt: {args.out}")

    law = SignedLaw()
    if law.sha256() != declaration["law_sha256"]:
        raise RuntimeError("declared law differs from the code about to run")

    population = build_population(PATHS_PARQUET, law)
    accounts = declaration["family"]["accounts_usd"]
    win_rates = declaration["family"]["win_rates"]
    paths = int(declaration["family"]["paths"])

    rows: list[dict[str, Any]] = []
    for account in accounts:
        for win_rate in win_rates:
            rng = np.random.default_rng(
                stable_seed("scale-sensitivity-v1", account, win_rate)
            )
            result = simulate_scale(
                population,
                account_usd=float(account),
                win_rate=float(win_rate),
                law=law,
                paths=paths,
                sessions=SESSIONS_PER_YEAR,
                rng=rng,
            )
            row = asdict(result)
            rows.append(row)
            print(
                f"${account:>7,.0f} p={win_rate:.2f} "
                f"ruin {result.ruin_share:6.3f} [{result.ruin_ci95[0]:.3f},{result.ruin_ci95[1]:.3f}] "
                f"trades/sess {result.trades_per_session_mean:5.3f} "
                f"net/trade {result.net_per_trade_mean:8.2f} "
                f"breaker {result.breaker_sessions_share:6.4f} "
                f"blocked {result.affordability_blocked_share:6.4f} "
                f"maxprem ${result.max_premium_bought_usd:,.0f}",
                flush=True,
            )

    receipt: dict[str, Any] = {
        "schema_version": "v5.scale-sensitivity-receipt.v1",
        "declaration_sha256": declaration["declaration_sha256"],
        "law": asdict(law),
        "law_sha256": law.sha256(),
        "population": {
            "source": PATHS_PARQUET,
            "eligible_trades": len(population),
            "winners": len(population.winners),
            "losers": len(population.losers),
            "measured_win_rate": len(population.winners) / len(population),
            "mean_ticket_usd": float(population.ticket_usd.mean()),
            "max_ticket_usd": float(population.ticket_usd.max()),
        },
        "assumes_hypothetical_edge": True,
        "not_evidence_of_edge": (
            "Win rates are swept parameters. Every measurement on this "
            "population at random entry is negative. No row supports the claim "
            "that an edge exists."
        ),
        "integrity": {
            "model_fitted": False,
            "money_spent": False,
            "vendor_contacted": False,
            "backfill_read": False,
            "reserved_sessions_used": False,
        },
        "results": rows,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
