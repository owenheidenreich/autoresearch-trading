"""Can the charter's risk limits survive the occupancy the plan wants to trade?

Phase 2 of job 24 makes a choice between hold lengths, and the plan states the
binding question plainly: 77 trades a session at 13% of equity per ticket turns
the account over many times a day, and must be checked against the **5% daily
circuit breaker** and the **50% survival floor** before any cell can be
declared. A cell that cannot satisfy them is not a candidate.

This module answers it by simulation on measured outcomes. It reads the
per-trade profit and loss from the occupancy receipt — quantiles of net dollars,
measured, not modelled — and resamples it. Nothing here fits or searches
anything: the two accuracies it reports at are break-even, which assumes no
skill, and the accuracy a screen on this corpus could actually prove, which is
the most optimistic honest bound.

It works in dollars rather than in return-on-premium because the charter buys
**one contract**. The money at stake is whatever that contract costs, so the
account size is what sets the risk per trade and it is the account, not a
position-size fraction, that this module puts on a ladder.

Four things the simulation captures that arithmetic does not:

**The breaker cuts occupancy.** A session that trips the 5% breaker stops
trading, so the trades-per-session the plan is buying is not the number the
account actually gets. This reports the difference.

**Losses arrive in sequence, not in aggregate.** A session can end flat having
been 8% down at midday. Only a path can see that.

**Sizing compounds across sessions.** The survival floor is a property of a
year of sessions, not of one.

**A small account cannot open the position at all.** The charter's 13% ceiling
on one ticket is a floor on the account, given a near-ATM contract costs about
$1,050.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Charter limits, from the signed 2026-08-13 amendment. Declared, never varied:
# the amendment is explicit that the risk limits are not the part worth
# amending to make an occupancy cell fit.
DAILY_BREAKER = 0.05
SURVIVAL_FLOOR = 0.50
CHARTER_PREMIUM_SHARE = 0.13  # superseded 2026-08-16; retained to reproduce prior receipts
# The signed 2026-08-16 amendment states the ceiling in DOLLARS. A share of
# equity drifts as the account compounds; a dollar ceiling cannot.
CHARTER_PREMIUM_CEILING_USD = 2_500.0
SESSIONS_PER_YEAR = 252

# Starting account sizes to report. The bot buys one contract, so the account is
# what sets the risk per trade — not a chosen fraction. $10,000 is the account
# the charter amendment was written against.
ACCOUNTS_USD = (10_000.0, 25_000.0, 50_000.0, 100_000.0, 250_000.0, 500_000.0, 1_000_000.0)

# How often a breaker may fire before it has stopped being a circuit breaker and
# become the strategy. Declared here, in advance, rather than read off a result.
BREAKER_TOLERANCE = 0.05

# How much of the hold's nominal occupancy the account must actually get to
# trade. Without this a tiny account "passes" both charter limits by tripping
# the breaker after one trade every session: it never halves, it never trades,
# and the occupancy the cell was chosen for is not what it delivers. Declared
# in advance for the same reason as the tolerance above.
MIN_OCCUPANCY_RETAINED = 0.50


def resample(rng: np.random.Generator, strata: np.ndarray, size: tuple) -> np.ndarray:
    """Draw from a measured distribution given as equal-probability strata.

    These must be **stratum means**, not quantiles. Uniform draws from a
    quantile grid can never exceed the highest quantile stored, so their mean is
    a trimmed mean — measured here, that understated the mean winning trade by
    6% to 19% while leaving the bounded losing side accurate to under 1%, which
    biased every simulation against the strategy. Stratum means reproduce the
    true mean exactly and keep the extreme tail inside the top slice. See
    ``measure_hold_occupancy.strata_means``.
    """

    return strata[rng.integers(0, len(strata), size=size)]


def simulate(
    *,
    trades_per_session: int,
    win_quantiles_usd: np.ndarray,
    loss_quantiles_usd: np.ndarray,
    mean_premium_usd: float,
    accuracy: float,
    account_usd: float,
    sessions: int,
    paths: int,
    rng: np.random.Generator,
    daily_breaker: float = DAILY_BREAKER,
    premium_ceiling_usd: float = CHARTER_PREMIUM_CEILING_USD,
) -> dict:
    """One year of sessions, ``paths`` times over, in dollars.

    The bot buys one contract, so the money at risk each trade is the contract's
    premium and the account only changes how much of the account that is. Both
    charter limits are measured against session-starting equity.

    ``daily_breaker`` and ``premium_ceiling_usd`` default to the signed charter
    values and are parameters only so that a proposed amendment can be measured
    against the same machinery rather than argued for.

    The ceiling is in **dollars**, not a share of equity. It was a share until
    2026-08-16, which meant it widened as the account compounded — at a $100,000
    account a 13% share is $13,000 and buys deep ITM. The signed amendment
    states the ceiling in dollars precisely so it cannot drift, and this
    simulator now matches it. Affordability is
    ``min(ceiling, session-starting equity)``: the ceiling binds the ticket, and
    a small account additionally cannot buy what it cannot afford.
    """

    equity = np.full(paths, account_usd)
    alive = np.ones(paths, dtype=bool)
    breaker_days = np.zeros(paths)
    trades_taken = np.zeros(paths)
    worst_session = np.zeros(paths)

    for _ in range(sessions):
        start = equity.copy()
        session_pnl = np.zeros(paths)
        tripped = np.zeros(paths, dtype=bool)
        correct = rng.random((paths, trades_per_session)) < accuracy
        wins = resample(rng, win_quantiles_usd, (paths, trades_per_session))
        losses = resample(rng, loss_quantiles_usd, (paths, trades_per_session))
        dollars = np.where(correct, wins, losses)
        # The charter caps one ticket at a fixed dollar amount, and an account
        # smaller than the contract cannot open the position regardless. The
        # binding limit is therefore the lesser of the two, and the ceiling term
        # does not grow with equity.
        affordable = np.minimum(premium_ceiling_usd, start) >= mean_premium_usd
        for t in range(trades_per_session):
            active = alive & ~tripped & affordable
            if not active.any():
                break
            session_pnl[active] += dollars[active, t]
            trades_taken[active] += 1
            tripped |= active & (session_pnl <= -daily_breaker * start)
        worst_session = np.minimum(worst_session, session_pnl / start)
        equity = np.where(alive, start + session_pnl, equity)
        breaker_days += tripped
        alive &= equity > SURVIVAL_FLOOR * account_usd

    return {
        "account_usd": account_usd,
        "daily_breaker": daily_breaker,
        "premium_ceiling_usd": premium_ceiling_usd,
        "ticket_share_of_account": round(mean_premium_usd / account_usd, 4),
        "affordable_under_the_charter_ceiling": bool(
            min(premium_ceiling_usd, account_usd) >= mean_premium_usd
        ),
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
        "median_year_end_equity_multiple": round(
            float(np.median(equity) / account_usd), 4
        ),
        "worst_session_return_p01": round(float(np.quantile(worst_session, 0.01)), 4),
        "passes_breaker_tolerance": bool(
            breaker_days.mean() / sessions <= BREAKER_TOLERANCE
        ),
        "passes_survival_floor": bool(1.0 - alive.mean() <= 0.0),
        "passes_occupancy_retention": bool(
            trades_taken.mean() / sessions / trades_per_session
            >= MIN_OCCUPANCY_RETAINED
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--occupancy-receipt",
        type=Path,
        default=Path("v4/audit/autoresearch/hold_occupancy_2026_08_13/receipt.json"),
    )
    p.add_argument("--paths", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source = json.loads(args.occupancy_receipt.read_text())
    rng = np.random.default_rng(args.seed)

    results = {}
    for hold, row in source["by_hold"].items():
        det = row.get("detectable")
        if not det:
            continue
        trades = max(1, int(round(row["trades_per_session_measured"])))
        premium = row["mean_entry_premium_usd"]
        # Two exit policies, because the charter's 13% ticket is justified by the
        # declared stop and not without it, and two declared accuracies, because
        # a risk limit has to hold both when the strategy has nothing and when it
        # has as much as the screen could prove.
        policies = {
            "hold_to_horizon": ("outcome_when_correct", "outcome_when_wrong"),
            "with_declared_stop": (
                "outcome_when_correct_stopped",
                "outcome_when_wrong_stopped",
            ),
        }
        scenarios = {
            "at_breakeven_accuracy": det["breakeven_accuracy"],
            "at_provable_accuracy": det["accuracy_at_effective_n"],
        }
        cell = {
            "trades_per_session_measured": row["trades_per_session_measured"],
            "mean_entry_premium_usd": premium,
            "declared_stop_level": row.get("declared_stop_level"),
            "declared_stop_fired_share": row.get("declared_stop_fired_share"),
            "accuracies": dict(scenarios),
            "accuracy_assumption": (
                "break-even is the neutral case: no policy exists, so no skill may "
                "be assumed. The provable accuracy is the most a screen on this "
                "corpus could ever certify, so it is the optimistic bound rather "
                "than an expectation. Both are the hold-to-horizon break-even, so "
                "the stopped policy is charged the same accuracy rather than a "
                "flattering one."
            ),
        }
        for policy, (win_key, loss_key) in policies.items():
            wins = np.asarray(row[win_key]["strata_usd"], float)
            losses = np.asarray(row[loss_key]["strata_usd"], float)
            for name, accuracy in scenarios.items():
                ladder = [
                    simulate(
                        trades_per_session=trades,
                        win_quantiles_usd=wins,
                        loss_quantiles_usd=losses,
                        mean_premium_usd=premium,
                        accuracy=accuracy,
                        account_usd=account,
                        sessions=SESSIONS_PER_YEAR,
                        paths=args.paths,
                        rng=rng,
                    )
                    for account in ACCOUNTS_USD
                ]
                cell[f"{policy}__{name}"] = {
                    "accounts": ladder,
                    "smallest_account_passing_all_three": next(
                        (
                            item["account_usd"]
                            for item in ladder
                            if item["passes_breaker_tolerance"]
                            and item["passes_survival_floor"]
                            and item["passes_occupancy_retention"]
                        ),
                        None,
                    ),
                }
        results[hold] = cell

    payload = {
        "schema_version": "v5.occupancy-risk.v1",
        "source_receipt": str(args.occupancy_receipt),
        "charter": {
            "daily_circuit_breaker": DAILY_BREAKER,
            "survival_floor": SURVIVAL_FLOOR,
            "premium_ceiling_share_of_equity": CHARTER_PREMIUM_SHARE,
            "reference": "v5/governance/CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md",
        },
        "declared_tolerances": {
            "breaker_firing_rate": BREAKER_TOLERANCE,
            "breaker_rationale": (
                "A breaker that fires on more than one session in twenty is not a "
                "circuit breaker, it is the exit policy. Declared before the run."
            ),
            "occupancy_retained": MIN_OCCUPANCY_RETAINED,
            "occupancy_rationale": (
                "A cell chosen for its occupancy has to deliver it. Without this "
                "an account small enough to trip the breaker after one trade "
                "passes both charter limits while never trading."
            ),
        },
        "paths_per_cell": args.paths,
        "sessions_per_path": SESSIONS_PER_YEAR,
        "seed": args.seed,
        "by_hold": results,
        "known_limitations": [
            "trades are resampled independently within a session, so a day that "
            "trends against every position is under-represented; the real breaker "
            "rate is therefore at least this high and probably higher",
            "the measured outcome distribution comes from a conditional payoff "
            "study, not from a policy, so it carries no skill and no timing",
            "the round-trip cost is already netted into the measured returns and "
            "is carried from the owned quote corpus",
        ],
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, row in results.items():
        print(
            f"\n{hold}: {row['trades_per_session_measured']:.1f} trades/session, "
            f"mean ticket ${row['mean_entry_premium_usd']:,.0f}"
        )
        for key in (
            "hold_to_horizon__at_breakeven_accuracy",
            "hold_to_horizon__at_provable_accuracy",
            "with_declared_stop__at_breakeven_accuracy",
            "with_declared_stop__at_provable_accuracy",
        ):
            block = row[key]
            policy, name = key.split("__")
            print(
                f"  {policy.replace('_', ' ')}, {name.replace('_', ' ')} "
                f"({100 * row['accuracies'][name]:.2f}%)"
            )
            head = (
                f"    {'account':>10} {'ticket':>7} {'kept':>7} {'realised':>9} "
                f"{'breaker':>9} {'ruin/yr':>9} {'med x':>7}"
            )
            print(head)
            print("    " + "-" * (len(head) - 4))
            for item in block["accounts"]:
                ok = (
                    item["passes_breaker_tolerance"]
                    and item["passes_survival_floor"]
                    and item["passes_occupancy_retention"]
                )
                print(
                    f"    ${item['account_usd']:>9,.0f} "
                    f"{100 * item['ticket_share_of_account']:>6.1f}% "
                    f"{100 * item['occupancy_kept']:>6.1f}% "
                    f"{item['realised_trades_per_session']:>9.1f} "
                    f"{100 * item['share_of_sessions_hitting_the_breaker']:>8.1f}% "
                    f"{100 * item['share_of_years_breaching_the_survival_floor']:>8.1f}% "
                    f"{item['median_year_end_equity_multiple']:>7.2f}"
                    f"{'  ok' if ok else ''}"
                )
            smallest = block["smallest_account_passing_all_three"]
            print(
                "    smallest account passing all three: "
                + (f"${smallest:,.0f}" if smallest else "none on this ladder")
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
