"""Verify whole-day/whole-chain policy observations on owned sessions.

This is model-free.  It proves that the interface handed to a future policy is
not the compact near-ATM atlas table: every live two-sided strike remains
visible, entry actions are masked separately, and afternoon observations still
contain the opening candle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import live_two_sided
from v5.ops.build_causal_day_dataset import (
    canonical_json,
    file_sha256,
    full_ladder_state,
    prepare_quotes,
)
from v5.research.causal_day_tensorizer import (
    AccountObservation,
    CANDLE_FEATURES,
    LADDER_FEATURES,
    tensorize_observation,
)


MINUTES = ("09:35", "10:00", "13:30", "15:00")


def run(
    quote_root: Path,
    coverage_csv: Path,
    candles_path: Path,
    declaration_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {out_dir}")
    coverage = pd.read_csv(coverage_csv)
    sessions = coverage.loc[
        coverage["included_for_episode_build"].astype(bool), "session"
    ].astype(str).tolist()
    chosen = [sessions[0], sessions[len(sessions) // 2], sessions[-1]]
    candles = pd.read_parquet(candles_path)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for session in chosen:
        quote_path = quote_root / f"databento_spxw_0dte_{session}.parquet"
        raw = pd.read_parquet(quote_path)
        quotes = prepare_quotes(raw, session)
        whole_chain, _ = full_ladder_state(quotes, session)
        for minute in MINUTES:
            snapshot = whole_chain[whole_chain["minute"].eq(minute)].copy()
            raw_snapshot = quotes[quotes["minute"].eq(minute)]
            expected_live = int(live_two_sided(raw_snapshot).sum())
            if len(snapshot) != expected_live:
                raise AssertionError(
                    f"{session} {minute}: policy chain {len(snapshot)} != live source {expected_live}"
                )
            role = "morning_entry" if minute < "12:46" else "afternoon_entry"
            observation = tensorize_observation(
                candles,
                whole_chain,
                session=session,
                minute=minute,
                role=role,
                account=AccountObservation(10_000.0, 0.0, 0, 1, False),
            )
            batch = observation.batch
            if not np.isfinite(batch.candles.numpy()).all() or not np.isfinite(
                batch.ladder.numpy()
            ).all():
                raise AssertionError(f"{session} {minute}: non-finite policy tensor")
            outside_action_band = int(snapshot["moneyness_itm_points"].abs().gt(25.0).sum())
            if outside_action_band <= 0:
                raise AssertionError(f"{session} {minute}: no deep-strike context survived")
            action_count = int(batch.entry_action_mask.sum().item())
            expected_actions = int(snapshot["entry_eligible"].sum())
            if action_count != expected_actions:
                raise AssertionError(
                    f"{session} {minute}: action mask {action_count} != eligible {expected_actions}"
                )
            rows.append(
                {
                    "session": session,
                    "minute": minute,
                    "role": role,
                    "completed_candles": int(batch.candle_mask.sum().item()),
                    "first_candle": observation.candle_minutes[0],
                    "last_candle": observation.candle_minutes[-1],
                    "whole_live_contracts": len(snapshot),
                    "calls": int(snapshot["right"].eq("C").sum()),
                    "puts": int(snapshot["right"].eq("P").sum()),
                    "contracts_outside_action_band": outside_action_band,
                    "entry_actions": action_count,
                    "volume_observed_share": float(snapshot["volume"].notna().mean()),
                    "iv_solved_share": float(snapshot["self_iv"].notna().mean()),
                }
            )
        sources.append(
            {
                "path": str(quote_path),
                "sha256": file_sha256(quote_path),
                "size_bytes": quote_path.stat().st_size,
            }
        )

    cells = pd.DataFrame(rows)
    expected_prefixes = {"09:35": 5, "10:00": 30, "13:30": 240, "15:00": 330}
    for minute, expected in expected_prefixes.items():
        if not cells.loc[cells["minute"].eq(minute), "completed_candles"].eq(expected).all():
            raise AssertionError(f"{minute}: completed prefix does not equal {expected}")
    if not cells["first_candle"].eq("09:30").all():
        raise AssertionError("an observation dropped the opening candle")

    out_dir.mkdir(parents=True, exist_ok=False)
    cells_path = out_dir / "observation_cells.csv"
    cells.to_csv(cells_path, index=False)
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-observation-contract.v1",
        "created_on": "2026-08-14",
        "purpose": "prove whole completed day prefix, whole live chain context and separate action mask",
        "sessions": chosen,
        "minutes": list(MINUTES),
        "cells": len(cells),
        "dimensions": {
            "candle_features": len(CANDLE_FEATURES),
            "ladder_features": len(LADDER_FEATURES),
            "maximum_completed_candles": 390,
        },
        "assertions": {
            "policy_chain_equals_source_live_two_sided_count": True,
            "opening_candle_retained_at_every_checked_minute": True,
            "completed_prefixes": expected_prefixes,
            "deep_strikes_visible_but_not_actions": True,
            "entry_action_mask_matches_declared_eligibility": True,
            "all_tensors_finite_with_observation_flags": True,
        },
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "implementation_hashes": {
            str(path): file_sha256(path)
            for path in (
                Path("v5/ops/build_causal_day_dataset.py"),
                Path("v5/research/causal_day_tensorizer.py"),
                Path("v5/research/causal_day_architectures.py"),
            )
        },
        "sources": sources,
        "artifacts": {
            "observation_cells": {
                "path": str(cells_path),
                "sha256": file_sha256(cells_path),
                "size_bytes": cells_path.stat().st_size,
            }
        },
        "fit_performed": False,
        "economic_result": None,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quotes",
        type=Path,
        default=Path(
            "/Users/och/.autoresearch-trading/pathd_2025-08-01_2026-07-31/"
            "aligned/normalized"
        ),
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/"
            "session_coverage.csv"
        ),
    )
    parser.add_argument(
        "--candles",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2/candles.parquet"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path("v5/work/entry-exit-attribution/DECLARATION_V2.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.quotes, args.coverage_csv, args.candles, args.declaration, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
