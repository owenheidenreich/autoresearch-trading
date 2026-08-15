"""What must now be proven before an entry or exit model may be fitted?

Training on the option layer is prohibited until the ES direction screen passes.
That rule is sound in intent — a model fitted on top of a direction signal that
does not exist cannot create information — but the *number* it was calibrated
against has moved a long way.

The 65.73% bar was derived on 2026-08-12 from **251 owned option sessions** and
a payoff pair measured at the **fee-only $3.08** cost. Two things have changed
since, both measured:

* the option corpus is **909 sessions**, acquired free, and a serial clock puts
  five to thirty-six trades in each of them rather than one;
* the payoff pair is measured on the settled exit-price convention and charged
  the **full $25 round trip**, which includes spread crossing.

This module recomputes the requirement from those measurements and puts the two
candidate screens side by side: the ES proxy the gate chain currently uses, and
a direct screen on the option corpus itself. It computes no policy and proposes
no rule; it answers "what would have to be true, and which instrument can see
it first".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from v5.research import statistics as st

# The ES corpus frozen for the second G1 attempt.
ES_SESSIONS = 2_435
ES_FRICTION_POINTS = 0.358

# Non-overlapping move dispersion by horizon, in ES points, measured on the
# owned 247 sessions by the job-15 feasibility surface. The ten-year pooled
# 60-minute figure is 15.971 against 18.0116 here, so the owned year is the more
# volatile: these are scaled by that ratio to approximate the ten-year corpus,
# and the unscaled figures are reported alongside so the adjustment is visible.
ES_MOVE_SD_OWNED_YEAR = {5: 5.2367, 10: 7.3387, 15: 8.8983, 30: 12.4674, 60: 18.0116}
ES_TEN_YEAR_SCALE = 15.971 / 18.0116

# Serial one-position occupancy on ES, from the same surface.
ES_TRADES_PER_SESSION = {5: 75.73, 10: 37.86, 15: 24.91, 30: 11.96, 60: 5.98}

# What the historical bar was, and where it came from.
LEGACY_BAR = {
    "accuracy": 0.6573,
    "breakeven": 0.5799,
    "option_sessions": 251,
    "trades_per_session": 1,
    "cost_charged_usd": 3.08,
    "source": "v5/research/findings/G1_THRESHOLD_FROM_THE_OPTION_LAYER_2026_08_12.md",
}


def es_screen(horizon: int, sessions: int, *, design_effect: float) -> dict:
    """What an ES direction screen can prove at one horizon.

    A correct call earns the move less friction and a wrong one loses the move
    plus friction, with ``E|move| = sd * sqrt(2/pi)`` for a symmetric move. That
    is the same assumption the 2026-08-12 derivation declared, reused rather
    than re-argued.
    """

    sd = ES_MOVE_SD_OWNED_YEAR[horizon] * ES_TEN_YEAR_SCALE
    expected_move = sd * math.sqrt(2.0 / math.pi)
    win = expected_move - ES_FRICTION_POINTS
    loss = expected_move + ES_FRICTION_POINTS
    trades = sessions * ES_TRADES_PER_SESSION[horizon]
    effective = max(1, int(trades / design_effect))
    return {
        "horizon_minutes": horizon,
        "sessions": sessions,
        "move_sd_points": round(sd, 4),
        "move_sd_points_owned_year": ES_MOVE_SD_OWNED_YEAR[horizon],
        "trades_per_session": ES_TRADES_PER_SESSION[horizon],
        "effective_trades": effective,
        "breakeven_accuracy": round(st.breakeven_accuracy(win=win, loss=loss), 6),
        "provable_accuracy": round(
            st.detectable_accuracy(effective, win=win, loss=loss, z_alpha=st.Z_95), 6
        ),
    }


def option_screen(row: dict) -> dict:
    """What a direct screen on the option corpus can prove, from measurement."""

    det = row["detectable"]
    return {
        "sessions": row["sessions"],
        "trades_per_session": row["trades_per_session_measured"],
        "effective_trades": det["effective_trades"],
        "breakeven_accuracy": det["breakeven_accuracy"],
        "provable_accuracy": det["accuracy_at_effective_n"],
        "provable_accuracy_with_conjunction_penalty": det[
            "accuracy_at_effective_n_with_conjunction_penalty"
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--occupancy-receipt",
        type=Path,
        default=Path("v4/audit/autoresearch/hold_occupancy_2026_08_13/receipt.json"),
    )
    p.add_argument("--es-sessions", type=int, default=ES_SESSIONS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source = json.loads(args.occupancy_receipt.read_text())
    rows = {}
    for hold, row in source["by_hold"].items():
        horizon = int(hold.rstrip("m"))
        if horizon not in ES_MOVE_SD_OWNED_YEAR or not row.get("detectable"):
            continue
        option = option_screen(row)
        # The ES side is charged the same intra-session dependence the option
        # side measured, rather than being credited with independence it was
        # never shown to have.
        es = es_screen(
            horizon,
            args.es_sessions,
            design_effect=max(1.0, row["detectable"]["design_effect"]),
        )
        binding = (
            "option corpus"
            if option["provable_accuracy"] > es["provable_accuracy"]
            else "ES corpus"
        )
        rows[hold] = {
            "option_screen": option,
            "es_screen": es,
            "harder_to_prove_on": binding,
            "margin_option": round(
                option["provable_accuracy"] - option["breakeven_accuracy"], 6
            ),
            "margin_es": round(es["provable_accuracy"] - es["breakeven_accuracy"], 6),
        }

    best = min(rows, key=lambda h: rows[h]["option_screen"]["provable_accuracy"])
    payload = {
        "schema_version": "v5.screen-requirement.v1",
        "question": (
            "The training block is calibrated to a 65.73% bar derived from 251 "
            "option sessions at a fee-only cost. What is the bar now, on 909 "
            "sessions at the measured round trip, and which corpus is binding?"
        ),
        "legacy_bar": LEGACY_BAR,
        "es_assumptions": {
            "friction_points": ES_FRICTION_POINTS,
            "move_sd_source": (
                "job-15 feasibility surface on 247 owned sessions, scaled by the "
                "measured ten-year/owned-year ratio at 60 minutes"
            ),
            "ten_year_scale": round(ES_TEN_YEAR_SCALE, 6),
            "magnitude_assumption": (
                "the size of the move is independent of whether the call was "
                "right, so E[signed] = (2p-1) * E|move|. Declared on 2026-08-12 "
                "and reused, not re-argued"
            ),
        },
        "occupancy_receipt": str(args.occupancy_receipt),
        "by_horizon": rows,
        "lowest_option_bar": {
            "horizon": best,
            "provable_accuracy": rows[best]["option_screen"]["provable_accuracy"],
        },
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\nlegacy bar: {100 * LEGACY_BAR['accuracy']:.2f}% "
        f"(251 option sessions, 1 trade each, ${LEGACY_BAR['cost_charged_usd']} charged)\n"
    )
    head = (
        f"{'horizon':>8} {'OPTION break-even':>18} {'provable':>9} "
        f"{'ES break-even':>14} {'provable':>9} {'harder on':>14}"
    )
    print(head)
    print("-" * len(head))
    for hold, row in rows.items():
        o, e = row["option_screen"], row["es_screen"]
        print(
            f"{hold:>8} {100 * o['breakeven_accuracy']:>17.2f}% "
            f"{100 * o['provable_accuracy']:>8.2f}% "
            f"{100 * e['breakeven_accuracy']:>13.2f}% "
            f"{100 * e['provable_accuracy']:>8.2f}% {row['harder_to_prove_on']:>14}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
