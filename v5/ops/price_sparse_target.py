"""How many sessions does the owner's 10-30 point target actually cost to prove?

The owner's stated strategy is not the one every previous screen measured. Those
measured an **average** — a small per-trade edge against a friction bar. This one
is a **tail**: buy a near-OTM 0DTE contract, hold it through a large directional
move, and let a few big winners pay for many small losses.

That difference matters for measurement, and not in the direction intuition
suggests. A rare large payoff has enormous per-trade dispersion, so proving that
a policy selects it better than chance takes *more* evidence than proving a
small steady edge, not less.

This module prices that. It reads the banked prevalence of each declared target
from the causal-day dataset receipt, builds the two-point payoff each target
implies, and reports the number of sessions needed to prove a policy sits a given
distance above break-even. It fits nothing, tunes nothing, and proposes no rule.
It answers only "what would this answer cost, in trading days".

The exit assumption is declared, not inherited: a target is scored as though the
position is closed when the underlying first reaches the declared depth. A policy
that exits earlier earns less; one that holds past the depth may earn more or
give it all back. That assumption belongs to any screen that freezes a number
from this, and must be restated rather than silently carried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v5.research import statistics as st

# The affordable near-OTM entry population the simulator actually offers, from
# the settlement-validated dataset receipt. 698,231 candidates over 243 sessions.
DEFAULT_DATASET_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json"
)

# Measured spread cost of leaving at the bid, from the 10:00/60m atlas cell.
# Charged on every exit, winners included.
EXIT_SPREAD_USD = 17.54

# Fee-only round trip measured from a real IBKR paper fill.
FEES_USD = 3.08

SPX_MULTIPLIER = 100.0

# The declared depth targets, in SPX points.
DEPTHS = (10, 20, 30)
HORIZONS = (60, 90, 120)

# A miss does not always expire worthless. The full-loss column is the
# conservative bound; the others show what partial recovery buys.
RECOVERY_FRACTIONS = (0.0, 0.25, 0.50)

# How far above break-even a policy might be proven to sit. These are not
# predictions — they are the axis the purchase decision is priced along.
PRECISION_MULTIPLES = (1.5, 2.0, 3.0)


def payoff_pair(depth: int, entry_ask: float, recovery: float) -> tuple[float, float]:
    """Win and loss magnitudes in dollars for one depth target.

    A win closes at the declared depth and pays intrinsic value less the entry
    ask, the exit spread and fees. A loss gives up the premium less whatever
    fraction is recovered, plus fees.
    """

    win = depth * SPX_MULTIPLIER - entry_ask - EXIT_SPREAD_USD - FEES_USD
    loss = entry_ask * (1.0 - recovery) + FEES_USD
    return win, loss


def price_target(
    *,
    depth: int,
    horizon: int,
    base_rate: float,
    entry_ask: float,
    recovery: float,
    trades_per_session: float,
    z_alpha: float,
    penalty: float,
) -> dict:
    """What one declared target costs to prove, in sessions."""

    win, loss = payoff_pair(depth, entry_ask, recovery)
    breakeven = st.breakeven_accuracy(win=win, loss=loss)
    row: dict = {
        "depth_points": depth,
        "horizon_minutes": horizon,
        "base_rate": round(base_rate, 6),
        "win_usd": round(win, 2),
        "loss_usd": round(loss, 2),
        "breakeven_precision": round(breakeven, 6),
        "lift_needed_over_base_rate": round(breakeven / base_rate, 2),
        "policy_beats_breakeven_by": {},
    }
    for multiple in PRECISION_MULTIPLES:
        target = breakeven * multiple
        if target >= 1.0:
            row["policy_beats_breakeven_by"][f"{multiple}x"] = {
                "target_precision": round(target, 6),
                "sessions_required": None,
                "note": "a precision above 1.0 is not achievable",
            }
            continue
        row["policy_beats_breakeven_by"][f"{multiple}x"] = {
            "target_precision": round(target, 6),
            "sessions_required": st.sessions_for_accuracy_edge(
                target,
                win=win,
                loss=loss,
                z_alpha=z_alpha,
                penalty=penalty,
                trades_per_session=trades_per_session,
            ),
        }
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-receipt", type=Path, default=DEFAULT_DATASET_RECEIPT)
    p.add_argument(
        "--trades-per-session",
        type=float,
        default=2.0,
        help="the owner's stated 'a few trades per day'; the charter caps it at 3",
    )
    p.add_argument(
        "--family-size",
        type=int,
        default=len(DEPTHS) * len(HORIZONS),
        help="declared comparison family; nine depth/horizon cells by default",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source = json.loads(args.dataset_receipt.read_text())
    prevalence = source["target_prevalence"]
    entry_ask = source["entry_population"]["mean_entry_ask_usd"]
    owned_sessions = source["sessions"]["n"]

    z_alpha = st.bonferroni_quantile(args.family_size)

    cells = {}
    for depth in DEPTHS:
        for horizon in HORIZONS:
            key = f"reached_{depth}_itm_{horizon}m"
            if key not in prevalence:
                continue
            base_rate = prevalence[key]["share"]
            cells[f"{depth}pt_{horizon}m"] = {
                str(recovery): price_target(
                    depth=depth,
                    horizon=horizon,
                    base_rate=base_rate,
                    entry_ask=entry_ask,
                    recovery=recovery,
                    trades_per_session=args.trades_per_session,
                    z_alpha=z_alpha,
                    penalty=1.0,
                )
                for recovery in RECOVERY_FRACTIONS
            }

    payload = {
        "schema_version": "v5.sparse-target-price.v1",
        "question": (
            "The owner's target is a sparse large payoff, not a small steady "
            "edge. How many sessions would it take to prove a policy selects it "
            "better than chance, and is that reachable?"
        ),
        "dataset_receipt": str(args.dataset_receipt),
        "owned_sessions": owned_sessions,
        "assumptions": {
            "mean_entry_ask_usd": round(entry_ask, 2),
            "exit_spread_usd": EXIT_SPREAD_USD,
            "fees_usd": FEES_USD,
            "spx_multiplier": SPX_MULTIPLIER,
            "trades_per_session": args.trades_per_session,
            "family_size": args.family_size,
            "z_alpha": round(z_alpha, 6),
            "exit_assumption": (
                "the position closes when the underlying first reaches the "
                "declared depth; declared here, not inherited"
            ),
            "recovery_fractions": list(RECOVERY_FRACTIONS),
        },
        "cells": cells,
        "computes_no_policy": True,
        "fit_performed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\nowned sessions: {owned_sessions}   mean entry ask: ${entry_ask:,.2f}")
    print(
        f"charging ${EXIT_SPREAD_USD:.2f} exit spread + ${FEES_USD:.2f} fees, "
        f"{args.trades_per_session:g} trades/session, family of {args.family_size}\n"
    )
    head = (
        f"{'target':>12} {'base rate':>10} {'break-even':>11} {'lift':>7} "
        f"{'sessions @2x':>13} {'@3x':>10}"
    )
    print(head)
    print("-" * len(head))
    for name, by_recovery in cells.items():
        row = by_recovery["0.0"]
        at2 = row["policy_beats_breakeven_by"]["2.0x"]["sessions_required"]
        at3 = row["policy_beats_breakeven_by"]["3.0x"]["sessions_required"]
        print(
            f"{name:>12} {100 * row['base_rate']:>9.2f}% "
            f"{100 * row['breakeven_precision']:>10.2f}% "
            f"{row['lift_needed_over_base_rate']:>6.1f}x "
            f"{('n/a' if at2 is None else f'{at2:,}'):>13} "
            f"{('n/a' if at3 is None else f'{at3:,}'):>10}"
        )
    print("\n(full-loss column; the receipt also carries 25% and 50% recovery)")
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
