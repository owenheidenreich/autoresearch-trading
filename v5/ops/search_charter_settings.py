"""Which charter settings, if any, let a $10,000 account actually trade this?

The charter risk check found two signed clauses that cannot both hold: a ticket
may be up to **13% of equity**, and a **5% daily breaker** stops the session.
A near-ATM 0DTE contract at 10.5% of a $10,000 account loses about 6.1% of the
account on a typical wrong call, so one losing trade ends the session and the
bot gets roughly one trade a day at any hold length.

Only two things can give: the ticket gets cheaper, or the breaker gets wider.
This module measures the whole grid of both, so the amendment that follows is a
reading of a surface rather than an argument.

**It is a measurement, not a search for a setting that passes.** Every cell is
reported, the pass criteria are the ones declared before the occupancy work, and
the accuracy assumed is break-even — no skill — so a cell that survives here
survives on the instrument's own dispersion rather than on hoped-for edge. A
setting that only passes at an assumed edge is marked as such.

The honest failure mode this is built to expose: a wider breaker always
"improves" occupancy, because the breaker is the thing cutting it. Widening it
must therefore be judged on the survival floor and on the worst session it
permits, never on occupancy alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v5.ops.check_occupancy_risk import (
    BREAKER_TOLERANCE,
    CHARTER_PREMIUM_SHARE,
    DAILY_BREAKER,
    MIN_OCCUPANCY_RETAINED,
    SESSIONS_PER_YEAR,
    SURVIVAL_FLOOR,
    resample,
)

# Daily breaker levels to report. The signed 5% first, then wider settings. The
# ladder stops at 25% because a breaker at half the survival floor has stopped
# being a daily control.
BREAKERS = (0.05, 0.08, 0.10, 0.15, 0.20, 0.25)

# A breaker is only a control if the account can survive a run of sessions that
# each hit it. Declared: at the reported breaker level, a year must still leave
# every simulated path above the survival floor.
ACCOUNT_UNDER_TEST_USD = 10_000.0


def simulate_fractional(
    *,
    trades_per_session: int,
    win_ratios: np.ndarray,
    loss_ratios: np.ndarray,
    ticket_share: float,
    accuracy: float,
    sessions: int,
    paths: int,
    rng: np.random.Generator,
    daily_breaker: float,
) -> dict:
    """A year of sessions where the ticket is a fixed share of session equity.

    The occupancy risk check sizes the ticket in **dollars**, which is literally
    what the charter says — one contract, whatever it costs. That model has a
    property which turns out to dominate its results: as the account falls, the
    same contract becomes a larger share of it, and below the ceiling the bot
    cannot open the position at all, so trading simply stops. Occupancy then
    measures affordability rather than the breaker.

    This is the other standard model, and the one a sizing rule is usually
    written in: the bot targets ``ticket_share`` of session-starting equity and
    buys whichever contract costs about that. It is the **more favourable** of
    the two, and it is reported because a charter amendment should be judged
    against the friendlier reading rather than the harsher one.

    Its assumption is stated rather than hidden: the payoff distribution is held
    at the one measured for this ticket size. In reality a shrinking account
    migrates to cheaper contracts, whose break-even is measurably worse — $200
    of premium needs 55.1% at fifteen minutes against 51.6% for $800 — so a real
    account in drawdown fares worse than this model says.
    """

    equity = np.ones(paths)
    alive = np.ones(paths, dtype=bool)
    breaker_days = np.zeros(paths)
    trades_taken = np.zeros(paths)

    for _ in range(sessions):
        start = equity.copy()
        session_return = np.zeros(paths)
        tripped = np.zeros(paths, dtype=bool)
        correct = rng.random((paths, trades_per_session)) < accuracy
        ratios = np.where(
            correct,
            resample(rng, win_ratios, (paths, trades_per_session)),
            resample(rng, loss_ratios, (paths, trades_per_session)),
        )
        for t in range(trades_per_session):
            active = alive & ~tripped
            if not active.any():
                break
            session_return[active] += ratios[active, t] * ticket_share
            trades_taken[active] += 1
            tripped |= active & (session_return <= -daily_breaker)
        equity = np.where(alive, start * (1.0 + session_return), equity)
        breaker_days += tripped
        alive &= equity > SURVIVAL_FLOOR

    return {
        "daily_breaker": daily_breaker,
        "ticket_share": round(ticket_share, 4),
        "nominal_trades_per_session": trades_per_session,
        "realised_trades_per_session": round(float(trades_taken.mean() / sessions), 2),
        "occupancy_kept": round(
            float(trades_taken.mean() / sessions / trades_per_session), 4
        ),
        "share_of_sessions_hitting_the_breaker": round(
            float(breaker_days.mean() / sessions), 4
        ),
        "share_of_years_breaching_the_survival_floor": round(
            float(1.0 - alive.mean()), 4
        ),
        "median_year_end_equity_multiple": round(float(np.median(equity)), 4),
        "passes_breaker_tolerance": bool(
            breaker_days.mean() / sessions <= BREAKER_TOLERANCE
        ),
        "passes_survival_floor": bool(1.0 - alive.mean() <= 0.0),
        "passes_occupancy_retention": bool(
            trades_taken.mean() / sessions / trades_per_session
            >= MIN_OCCUPANCY_RETAINED
        ),
    }


def grid(source: dict, hold: str, *, paths: int, seed: int, account: float) -> list[dict]:
    """Every ticket size against every breaker level, at one hold."""

    block = source["by_hold"][hold]
    rng = np.random.default_rng(seed)
    out = []
    for name, row in block["by_ticket"].items():
        if not row.get("breakeven_accuracy"):
            out.append(
                {
                    "ticket": name,
                    "mean_premium_usd": row["mean_premium_usd"],
                    "verdict": row.get("verdict", "unwinnable"),
                    "cells": [],
                }
            )
            continue
        premium = row["mean_premium_usd"]
        # Return on the ticket, stratified per trade against that trade's own
        # premium rather than against the bucket mean, so a contract that came
        # in above or below its target is not mis-scaled.
        wins = np.asarray(row["outcome_when_correct"]["strata_ratio"], float)
        losses = np.asarray(row["outcome_when_wrong"]["strata_ratio"], float)
        trades = max(1, int(round(row["trades_per_session"])))
        cells = []
        for breaker in BREAKERS:
            for label, accuracy in (
                ("no_skill", row["breakeven_accuracy"]),
                ("provable_skill", row["provable_accuracy"]),
            ):
                got = simulate_fractional(
                    trades_per_session=trades,
                    win_ratios=wins,
                    loss_ratios=losses,
                    ticket_share=premium / account,
                    accuracy=accuracy,
                    sessions=SESSIONS_PER_YEAR,
                    paths=paths,
                    rng=rng,
                    daily_breaker=breaker,
                )
                got["scenario"] = label
                cells.append(got)
        out.append(
            {
                "ticket": name,
                "mean_premium_usd": premium,
                "ticket_share_of_account": round(premium / account, 4),
                "breakeven_accuracy": row["breakeven_accuracy"],
                "provable_accuracy": row["provable_accuracy"],
                "trades_per_session": row["trades_per_session"],
                "round_trip_share_of_premium": row["round_trip_share_of_premium"],
                "cells": cells,
            }
        )
    return out


def _passes(cell: dict) -> bool:
    return (
        cell["passes_breaker_tolerance"]
        and cell["passes_survival_floor"]
        and cell["passes_occupancy_retention"]
    )


def verdict(cells: list[dict]) -> dict:
    """Both readings, because neither alone is the answer.

    At exactly break-even accuracy any dispersed strategy loses ground
    geometrically, so "passes with no skill" is a demanding test and failing it
    is not by itself a reason to reject a setting — it is a statement of what
    happens if the model turns out to have nothing. What it must not be allowed
    to do is disappear, so both readings are reported and the cost of being
    wrong is carried alongside the setting that works when right.
    """

    by_scenario = {
        name: [c for c in cells if c["scenario"] == name]
        for name in ("no_skill", "provable_skill")
    }
    passing_blind = [c for c in by_scenario["no_skill"] if _passes(c)]
    passing_skilled = [c for c in by_scenario["provable_skill"] if _passes(c)]
    narrowest = min((c["daily_breaker"] for c in passing_skilled), default=None)
    cost_if_wrong = next(
        (
            c["share_of_years_breaching_the_survival_floor"]
            for c in by_scenario["no_skill"]
            if narrowest is not None and c["daily_breaker"] == narrowest
        ),
        None,
    )
    return {
        "passes_with_no_skill": bool(passing_blind),
        "narrowest_breaker_passing_with_no_skill": min(
            (c["daily_breaker"] for c in passing_blind), default=None
        ),
        "narrowest_breaker_passing_if_the_edge_is_real": narrowest,
        "no_skill_ruin_at_that_breaker": cost_if_wrong,
        "no_skill_ruin_at_the_signed_breaker": next(
            (
                c["share_of_years_breaching_the_survival_floor"]
                for c in by_scenario["no_skill"]
                if c["daily_breaker"] == DAILY_BREAKER
            ),
            None,
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ticket-receipt",
        type=Path,
        default=Path("v4/audit/autoresearch/ticket_size_surface_2026_08_13/receipt.json"),
    )
    p.add_argument("--account", type=float, default=ACCOUNT_UNDER_TEST_USD)
    p.add_argument("--paths", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source = json.loads(args.ticket_receipt.read_text())
    results = {}
    for hold in source["by_hold"]:
        rows = grid(source, hold, paths=args.paths, seed=args.seed, account=args.account)
        for row in rows:
            if row["cells"]:
                row["verdict"] = verdict(row["cells"])
        results[hold] = rows

    payload = {
        "schema_version": "v5.charter-settings-grid.v1",
        "question": (
            "The 13% ticket ceiling and the 5% daily breaker cannot both hold at "
            "a $10,000 account. Which combinations of ticket size and breaker "
            "level let the bot trade at all?"
        ),
        "account_usd": args.account,
        "signed_charter": {
            "daily_circuit_breaker": DAILY_BREAKER,
            "survival_floor": SURVIVAL_FLOOR,
            "premium_ceiling_share_of_equity": CHARTER_PREMIUM_SHARE,
            "reference": "v5/governance/CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md",
        },
        "breakers_tested": list(BREAKERS),
        "declared_criteria": {
            "breaker_firing_rate": BREAKER_TOLERANCE,
            "survival_floor_breaches": 0.0,
            "occupancy_retained": MIN_OCCUPANCY_RETAINED,
            "primary_reading": (
                "no skill assumed. A setting that only passes at the provable "
                "accuracy is reported separately and is not a pass."
            ),
        },
        "source_receipt": str(args.ticket_receipt),
        "paths_per_cell": args.paths,
        "seed": args.seed,
        "by_hold": results,
        "known_limitations": [
            "trades are resampled independently within a session, so a day that "
            "trends against every position is under-represented and the real "
            "breaker rate is at least this high",
            "the round trip is carried per band from the owned quote corpus",
            "widening the breaker mechanically improves occupancy, so occupancy "
            "is never sufficient on its own here",
        ],
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, rows in results.items():
        print(f"\n{hold} hold, ${args.account:,.0f} account")
        head = (
            f"  {'ticket':>8} {'share':>7} {'cost/prem':>10} {'break-even':>11} "
            f"{'provable':>9} {'breaker needed':>15} {'ruin if wrong':>14}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in sorted(rows, key=lambda r: r["mean_premium_usd"]):
            if not row["cells"]:
                print(
                    f"  {row['ticket']:>8} {'':>7} {'':>10} {'unwinnable':>11}"
                )
                continue
            v = row["verdict"]
            breaker = v["narrowest_breaker_passing_if_the_edge_is_real"]
            note = f"{100 * breaker:.0f}%" if breaker else "none up to 25%"
            ruin = v["no_skill_ruin_at_that_breaker"]
            ruin_txt = f"{100 * ruin:.0f}%" if ruin is not None else "-"
            print(
                f"  {row['ticket']:>8} {100 * row['ticket_share_of_account']:>6.1f}% "
                f"{100 * row['round_trip_share_of_premium']:>9.1f}% "
                f"{100 * row['breakeven_accuracy']:>10.2f}% "
                f"{100 * row['provable_accuracy']:>8.2f}% {note:>15} {ruin_txt:>14}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
