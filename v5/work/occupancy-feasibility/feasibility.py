"""Occupancy-specific detectability surface for a higher-frequency G1 family.

Ledger row 185 requires "a frozen higher-frequency mechanism family *with its
own occupancy-specific MDE receipt*" before such a family may be declared, and
row 186 names "a materially higher-occupancy design" as one of the few
genuinely new alternatives to waiting four years for more sessions.  This
module produces that receipt, and it produces it *before* any family exists, so
the design cannot be chosen to fit a result.

What this computes is **noise**, not edge.  It reads owned ES bars, measures the
dispersion of non-overlapping h-minute moves, counts how many such moves fit in
a session under a serial one-position clock, and asks the only question that
matters before designing anything:

    at this occupancy, is the smallest detectable per-trade effect below the
    0.358-point friction bar?

If the answer is no in every cell, no family on this corpus can both be
profitable and be seen, and the job closes without a family.  No rule, sign,
threshold, or return of any policy is computed anywhere in this file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from v5.research import knobs, statistics
from v5.research.direction import family, loader


# The G1 warm-up slot.  Kept identical so the one-trade-per-session cell of this
# surface reproduces the measurement review's published numbers and can be
# checked against them rather than taken on trust.
FIRST_ENTRY_MINUTE = "09:35"

# Horizons.  15/30/60 are the project's declared scope and are what the
# measurement review costed; 5/10/20 are included because row 186 names "a
# different instrument or horizon with a better signal-to-friction ratio" as
# genuinely new, and the surface is worthless if it cannot see outside the
# range that already failed.
HORIZONS_MINUTES = (5, 10, 15, 20, 30, 60)

# Candidate declared family sizes.  18 is what G1 froze; the smaller sizes are
# the real lever, because the Bonferroni correction that costs G1 most of its
# power scales with the number of members declared.
FAMILY_SIZES = (1, 2, 4, 6, 18)

# Measured 2026-08-09: G1's true detection floor was 8-16 points/session where
# the analytic single-z-test MDE said 2.2-4.0 -- a 2-4x conjunction penalty.
#
# That measured penalty *includes* Bonferroni across G1's eighteen members, and
# this module already applies Bonferroni explicitly through ``family_size``, so
# carrying 2-4x straight over would charge the same correction twice.  The
# eighteen-member Bonferroni factor is (z(1-0.05/18) + z_power) / (z(0.95) +
# z_power) = 3.612 / 2.487 = 1.452, leaving a residual penalty of 2/1.452 to
# 4/1.452 for the parts this module does *not* model: the 4-of-5 fold rule on
# both absolute and paired results, the session-block bootstrap lower bound,
# and the paired-comparator criterion.
#
# This decomposition assumes the penalty is multiplicative and separable, which
# is an approximation and not a measurement.  It is a screening bracket only.
# Row 184 is explicit that "exact per-policy power must be recomputed after
# occupancy is frozen", so any family that reaches declaration must still pass
# its own known-answer campaign; this bracket only decides whether designing
# one is worth the effort.
BONFERRONI_18_FACTOR = (statistics.bonferroni_quantile(18) + statistics.Z_POWER_80) / (
    statistics.Z_95 + statistics.Z_POWER_80
)
CONJUNCTION_PENALTY_BRACKET = (2.0 / BONFERRONI_18_FACTOR, 4.0 / BONFERRONI_18_FACTOR)

POWER = 0.80
Z_POWER = statistics.Z_POWER_80


def _minute_to_index(minute: str) -> int:
    hours, minutes = (int(part) for part in minute.split(":"))
    return hours * 60 + minutes


def eligible_sessions() -> tuple:
    """Owned sessions minus the seven the product cannot trade.

    The exclusion is the owner's 2026-08-09 ruling, reused verbatim rather than
    re-derived, so this surface describes the same index a real screen would
    run on.
    """

    sessions = loader.load_sessions()
    return tuple(s for s in sessions if s.session not in family.NO_OPTION_SESSIONS)


def moves_for_horizon(sessions, horizon: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Non-overlapping close-to-close moves under a serial one-position clock.

    Returns the pooled moves, the per-session trade counts, and the per-session
    consecutive-move pairs used for the independence check.  A trade entered at
    minute ``m`` exits at ``m + horizon``; the next may only be entered then,
    which is what "serial, one position" means and is the same occupancy law the
    frozen G1 game uses.
    """

    pooled: list[float] = []
    counts: list[int] = []
    pairs: list[tuple[float, float]] = []
    for bars in sessions:
        first = _minute_to_index(FIRST_ENTRY_MINUTE)
        last_label = bars.minute_et[-1]
        last = _minute_to_index(last_label)
        session_moves: list[float] = []
        entry = first
        while entry + horizon <= last:
            entry_label = f"{entry // 60:02d}:{entry % 60:02d}"
            exit_minute = entry + horizon
            exit_label = f"{exit_minute // 60:02d}:{exit_minute % 60:02d}"
            entry_price = bars.close_at(entry_label)
            exit_price = bars.close_at(exit_label)
            if entry_price is None or exit_price is None:
                break
            session_moves.append(exit_price - entry_price)
            entry = exit_minute
        pooled.extend(session_moves)
        counts.append(len(session_moves))
        pairs.extend(zip(session_moves[:-1], session_moves[1:]))
    return np.asarray(pooled, float), np.asarray(counts, int), pairs


def lag1_autocorrelation(pairs) -> float:
    """Correlation of consecutive non-overlapping moves inside a session.

    The MDE below treats trades as independent.  A materially positive value
    here would make that optimistic and the whole surface too flattering; the
    measurement review found -0.055 at 15 minutes, so the assumption was
    slightly conservative.  This recomputes it at every horizon rather than
    assuming the 15-minute result carries.
    """

    if len(pairs) < 2:
        return float("nan")
    left = np.asarray([p[0] for p in pairs], float)
    right = np.asarray([p[1] for p in pairs], float)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def sharpe_ladder() -> dict:
    """Detectable Sharpe against session count, under each gate standard.

    The invariant itself lives in :mod:`v5.research.statistics` rather than here,
    because this work packet moves to ``v5/history/jobs/`` when job 15 closes and
    the relationship outlives the job that found it.
    """

    z_single = statistics.Z_95
    z_family18 = statistics.bonferroni_quantile(18)
    strict = CONJUNCTION_PENALTY_BRACKET[1]
    standards = {
        "single_hypothesis": {"z_alpha": z_single, "penalty": 1.0},
        "single_hypothesis_conservative": {"z_alpha": z_single, "penalty": strict},
        "g1_eighteen_member_gate": {"z_alpha": z_family18, "penalty": strict},
    }
    counts = (247, 500, 1260, 2520, 5040, 11824)
    return {
        "standards": {
            name: {
                "z_alpha": round(cfg["z_alpha"], 6),
                "penalty": round(cfg["penalty"], 6),
                "detectable_sharpe_by_sessions": {
                    str(n): round(statistics.detectable_sharpe(n, **cfg), 4)
                    for n in counts
                },
                "sessions_for_sharpe": {
                    str(target): statistics.sessions_for_sharpe(target, **cfg)
                    for target in (0.5, 1.0, 1.5, 2.0)
                },
            }
            for name, cfg in standards.items()
        },
        "note": (
            "Friction, volatility, horizon, occupancy and session length cancel "
            "out of this relationship. Only the session count and the "
            "statistical standard appear."
        ),
    }


def surface() -> dict:
    sessions = eligible_sessions()
    friction = float(knobs.frozen_value("es_round_trip_friction_points"))
    rows = []
    for horizon in HORIZONS_MINUTES:
        moves, counts, pairs = moves_for_horizon(sessions, horizon)
        total_trades = int(counts.sum())
        sd = float(moves.std(ddof=1))
        per_session = float(counts.mean())
        rho = lag1_autocorrelation(pairs)
        for size in FAMILY_SIZES:
            alpha = 0.05 / size
            z_alpha = statistics.normal_quantile(1.0 - alpha)
            mde = (z_alpha + Z_POWER) * sd / math.sqrt(total_trades)
            # A candidate must clear BOTH bars to be worth anything: the
            # economic one (beat friction) and the measurement one (be visible
            # above the gate's floor).  Whichever is larger is the real
            # requirement, and which of the two binds is the finding -- a cell
            # where friction binds is a cell where the screen has stopped
            # measuring its own power and started measuring the market.
            mde_conservative = mde * CONJUNCTION_PENALTY_BRACKET[1]
            required = max(friction, mde_conservative)
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "family_size": size,
                    "trades_per_session": round(per_session, 2),
                    "total_trades": total_trades,
                    "move_sd_points": round(sd, 4),
                    "lag1_autocorrelation": round(rho, 4),
                    "analytic_mde_points_per_trade": round(mde, 4),
                    "analytic_mde_over_friction": round(mde / friction, 3),
                    "conjunction_adjusted_low": round(
                        mde * CONJUNCTION_PENALTY_BRACKET[0] / friction, 3
                    ),
                    "conjunction_adjusted_high": round(mde_conservative / friction, 3),
                    "binding_constraint": (
                        "friction" if friction >= mde_conservative else "measurement"
                    ),
                    "required_points_per_trade": round(required, 4),
                    "required_per_trade_skill_share_of_sd": round(required / sd, 4),
                    "friction_share_of_move_sd": round(friction / sd, 4),
                    "session_friction_points": round(per_session * friction, 3),
                    # What the required per-trade skill implies if sustained.
                    # A design is only interesting if this number is one a real
                    # strategy could plausibly reach; an implausibly high value
                    # means the cell is measurable but not winnable, which is a
                    # different failure from being unmeasurable.
                    "implied_annual_sharpe_at_required_edge": round(
                        (required / sd) * math.sqrt(per_session * 252.0), 2
                    ),
                }
            )
    return {
        "schema_version": "v5.occupancy-feasibility-surface.v1",
        "sessions": len(sessions),
        "first_entry_minute": FIRST_ENTRY_MINUTE,
        "friction_points": friction,
        "power": POWER,
        "familywise_level": 0.95,
        "conjunction_penalty_bracket": list(CONJUNCTION_PENALTY_BRACKET),
        "computes_no_policy_economics": True,
        "detectable_sharpe_invariant": sharpe_ladder(),
        "rows": rows,
    }


def main() -> None:
    result = surface()
    out = Path(__file__).with_name("feasibility_receipt.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    friction = result["friction_points"]
    print(
        f"{result['sessions']} eligible sessions, friction {friction} pts/round trip, "
        f"{int(result['power'] * 100)}% power, 95% familywise\n"
    )
    header = (
        f"{'horizon':>7} {'trd/sess':>9} {'move SD':>8} {'lag1':>7} {'m':>3} "
        f"{'MDE':>7} {'w/conjunction':>14} {'binds':>12} {'need pts':>9} {'need IR':>8}"
        f" {'implied SR':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        print(
            f"{row['horizon_minutes']:>7} {row['trades_per_session']:>9} "
            f"{row['move_sd_points']:>8.3f} {row['lag1_autocorrelation']:>7.3f} "
            f"{row['family_size']:>3} {row['analytic_mde_points_per_trade']:>7.4f} "
            f"{row['conjunction_adjusted_low']:>6.2f}-{row['conjunction_adjusted_high']:<7.2f} "
            f"{row['binding_constraint']:>12} {row['required_points_per_trade']:>9.3f} "
            f"{row['required_per_trade_skill_share_of_sd'] * 100:>7.1f}%"
            f" {row['implied_annual_sharpe_at_required_edge']:>11.1f}"
        )
    print(
        "\n'need IR' is the per-trade edge, as a share of one move's standard "
        "deviation,\nthat a candidate must have to be both profitable and "
        "visible. Lower is easier."
    )

    ladder = result["detectable_sharpe_invariant"]["standards"]
    counts = sorted(
        (int(n) for n in next(iter(ladder.values()))["detectable_sharpe_by_sessions"]),
    )
    print("\nSmallest annualised Sharpe a screen can detect, by session count.")
    print("Instrument, horizon and occupancy cancel out of this entirely.\n")
    name_width = max(len(name) for name in ladder)
    print(f"{'standard':>{name_width}} " + " ".join(f"{n:>8,}" for n in counts))
    print("-" * (name_width + 1 + 9 * len(counts)))
    for name, payload in ladder.items():
        cells = payload["detectable_sharpe_by_sessions"]
        print(
            f"{name:>{name_width}} "
            + " ".join(f"{cells[str(n)]:>8.2f}" for n in counts)
        )
    print(f"\n(owned today: {result['sessions']} sessions)")

    print(f"\nreceipt written to {out}")


if __name__ == "__main__":
    main()
