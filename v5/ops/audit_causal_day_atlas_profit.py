"""Bug audit of the surprising positive 10:00 fixed-clock atlas diagnostic.

This fits nothing and selects no contract.  It audits all six predeclared
named-minute x clock cells together, including no-contract days as zero and
blocked exits under the declared zero-recovery sensitivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import HORIZONS, canonical_json, file_sha256


NAMED_MINUTES = ("10:00", "13:30")
FAMILY_SIZE = len(NAMED_MINUTES) * len(HORIZONS)
BOOTSTRAP_REPS = 50_000
BOOTSTRAP_SEED = 0
PREMIUM_BUCKETS = ((0.0, 400.0), (400.0, 700.0), (700.0, 1000.0), (1000.0, 1300.000001))


def corrected_interval(values: np.ndarray) -> dict[str, float | int]:
    clean = np.asarray(values, float)
    if not np.isfinite(clean).all():
        raise ValueError("session economics must be finite after declared sensitivity fill")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clean), size=(BOOTSTRAP_REPS, len(clean)))
    means = clean[draws].mean(axis=1)
    alpha = 0.05 / FAMILY_SIZE
    return {
        "n_sessions": len(clean),
        "mean_usd_per_session": float(clean.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def _summarize(values: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(values.notna().sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_share": float(values.gt(0.0).mean()),
    }


def run(candidates_path: Path, declaration: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {out_dir}")
    candidates = pd.read_parquet(candidates_path)
    sessions = sorted(candidates["session"].astype(str).unique())
    cells: dict[str, Any] = {}

    for minute in NAMED_MINUTES:
        minute_frame = candidates[candidates["entry_minute"].eq(minute)].copy()
        for horizon in HORIZONS:
            net_bid = f"net_bid_{horizon}m_usd"
            net_mid = f"net_mid_{horizon}m_usd"
            status = f"clock_exit_status_{horizon}m"
            blocked = minute_frame[status].eq("blocked_no_executable_bid")
            settled = minute_frame[status].eq("validated_cash_settlement")
            zero_column = f"net_bid_zero_recovery_{horizon}m_usd"
            zero_recovery = (
                minute_frame[zero_column]
                if zero_column in minute_frame
                else minute_frame[net_bid].where(
                    ~blocked, -minute_frame["entry_ask_usd"] - 3.08
                )
            )
            minute_frame["zero_recovery_net"] = zero_recovery

            # A feasible *randomized baseline expectation*: one uniformly
            # selected eligible contract when any exist, otherwise no trade.
            # This is not a learned or deterministic contract-selection policy.
            session_bid = minute_frame.groupby("session")[net_bid].mean().reindex(sessions).fillna(0.0)
            session_zero = (
                minute_frame.groupby("session")["zero_recovery_net"].mean().reindex(sessions).fillna(0.0)
            )
            by_side = {
                side: _summarize(part[net_bid])
                for side, part in minute_frame.groupby(
                    minute_frame["is_call"].map({True: "call", False: "put"})
                )
            }
            by_premium = {}
            for low, high in PREMIUM_BUCKETS:
                part = minute_frame[
                    minute_frame["entry_ask_usd"].ge(low)
                    & minute_frame["entry_ask_usd"].lt(high)
                ]
                by_premium[f"{low:.0f}-{high:.0f}"] = _summarize(part[net_bid])
            first_half_sessions = set(sessions[: len(sessions) // 2])
            by_period = {
                "first_half": _summarize(
                    minute_frame[minute_frame["session"].isin(first_half_sessions)][net_bid]
                ),
                "second_half": _summarize(
                    minute_frame[~minute_frame["session"].isin(first_half_sessions)][net_bid]
                ),
            }
            key = f"{minute}_{horizon}m"
            cells[key] = {
                "contracts": len(minute_frame),
                "sessions_with_contract": int(minute_frame["session"].nunique()),
                "blocked_contracts": int(blocked.sum()),
                "blocked_share": float(blocked.mean()),
                "cash_settled_contracts": int(settled.sum()),
                "cash_settled_share": float(settled.mean()),
                "pooled_bid": _summarize(minute_frame[net_bid]),
                "pooled_mid": _summarize(minute_frame[net_mid]),
                "mean_mid_minus_bid_usd": float((minute_frame[net_mid] - minute_frame[net_bid]).mean()),
                "session_equal_bid_including_no_trade_days": corrected_interval(
                    session_bid.to_numpy(float)
                ),
                "session_equal_zero_recovery": corrected_interval(session_zero.to_numpy(float)),
                "by_side": by_side,
                "by_entry_ask_usd": by_premium,
                "by_chronological_half": by_period,
                "mechanical_checks": {
                    "entry_ask_max_le_1300": bool(minute_frame["entry_ask_usd"].le(1300.0).all()),
                    "all_entry_moneyness_otm_near_atm": bool(
                        minute_frame["moneyness_itm_points"].ge(-25.0).all()
                        and minute_frame["moneyness_itm_points"].lt(0.0).all()
                    ),
                    "duplicate_trade_ids": int(minute_frame["trade_id"].duplicated().sum()),
                    "fees_in_net_identity": bool(
                        np.allclose(
                            minute_frame[net_bid].dropna().to_numpy(float),
                            (
                                minute_frame.loc[
                                    minute_frame[net_bid].notna(),
                                    (
                                        f"clock_exit_value_{horizon}m"
                                        if f"clock_exit_value_{horizon}m" in minute_frame
                                        else f"clock_exit_bid_{horizon}m"
                                    ),
                                ]
                                * 100.0
                                - minute_frame.loc[minute_frame[net_bid].notna(), "entry_ask_usd"]
                                - 3.08
                            ).to_numpy(float),
                        )
                    ),
                    "cash_settlement_never_labeled_as_bid": bool(
                        minute_frame.loc[
                            settled, f"clock_exit_bid_{horizon}m"
                        ].isna().all()
                    ),
                },
            }

    receipt: dict[str, Any] = {
        "schema_version": (
            "v5.causal-day-atlas-profit-audit.v2"
            if "clock_exit_value_60m" in candidates
            else "v5.causal-day-atlas-profit-audit.v1"
        ),
        "created_on": "2026-08-14",
        "purpose": "post-result bug audit of all six predeclared named-minute fixed-clock cells",
        "family_size": FAMILY_SIZE,
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "unit": "session",
            "two_sided_familywise_level": 0.95,
        },
        "declaration_sha256": file_sha256(declaration),
        "cells": cells,
        "conclusion_law": [
            "the session-equal random-contract expectation is a diagnostic baseline, not a trained or deterministic policy",
            "hindsight best-contract economics are excluded from this audit",
            "a cell is not promoted or re-opened because the current do-not-retest and training gates still bind",
            "positive results require separate chronological action selection and the full $10,000 simulator, neither performed here",
        ],
        "model_fit": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(out_dir / "receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v1/candidates.parquet"),
    )
    parser.add_argument(
        "--declaration", type=Path, default=Path("v5/work/entry-exit-attribution/DECLARATION.json")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidates, args.declaration, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
