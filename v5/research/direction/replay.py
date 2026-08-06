"""Replay one frozen family member into per-session economics.

The replay is intentionally dumb: it fits nothing, tunes nothing, and has no
free parameter. Occupancy and side come from the frozen declaration, the fill
prices come from the loader, and friction comes from the knob registry. Its
only job is to turn "this member, on these bars" into one net number per
declared calendar session.

Two rules carry most of the honesty here:

* **Every declared session produces a row**, including a `0.0` on a day the
  member declined to trade. Handing the gate only the days that traded is the
  precise optimism the gate exists to refuse.
* **A zero score is no position.** `side = sign(score)` with `sign(0) = 0`
  follows from the declared side rule rather than adding a tie-break to it.

Because the member reads a features frame rather than the market, the identical
code path serves real bars and matched surrogates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v5.research import knobs
from v5.research.direction import family, loader


SCORE_FIELD = {
    "M1": "first_five_minute_return",
    "M3": "overnight_gap",
    "JOINT": "overnight_gap",
}


class ReplayError(RuntimeError):
    """A member cannot be replayed as declared."""


def occupancy_mask(features: pd.DataFrame, mechanism: str) -> pd.Series:
    """Which declared sessions the mechanism is willing to trade.

    Standing down is a first-class outcome, not a failure: an unoccupied
    session still contributes its zero to the session index.
    """

    if mechanism == "M1":
        # Volume at or above the expanding median of strictly earlier sessions.
        # 1.0 is the median itself, not a tuned threshold.
        return features["volume_surprise"] >= 1.0
    if mechanism == "M3":
        return features["overnight_gap"].notna()
    if mechanism == "JOINT":
        gap = features["overnight_gap"]
        five = features["first_five_minute_return"]
        return gap.notna() & five.notna() & (np.sign(gap) == np.sign(five)) & (np.sign(gap) != 0)
    raise ReplayError(f"unknown mechanism: {mechanism}")


def replay_member(features: pd.DataFrame, member: family.Member) -> pd.DataFrame:
    """One row per declared calendar session for this member.

    Columns: ``session``, ``fold``, ``net_points``, ``gross_points``,
    ``trades``, ``side``.
    """

    family.assert_member_registered(member.name)
    eligible = loader.eligible_sessions(features, member.mechanism)
    frame = features.loc[eligible].reset_index(drop=True)

    score = frame[SCORE_FIELD[member.mechanism]].to_numpy(float)
    entry = frame["entry_price"].to_numpy(float)
    exit_price = frame[f"exit_price_{member.horizon_minutes}m"].to_numpy(float)

    occupied = occupancy_mask(frame, member.mechanism).to_numpy(bool)
    # A price we could not read is not a trade we may claim.
    tradable = occupied & np.isfinite(score) & np.isfinite(entry) & np.isfinite(exit_price)

    side = np.sign(score, out=np.zeros_like(score), where=np.isfinite(score))
    if member.side_rule == family.DIRECTIONS["against"]:
        side = -side
    elif member.side_rule != family.DIRECTIONS["with"]:
        raise ReplayError(f"unknown side rule: {member.side_rule}")
    side = np.where(tradable, side, 0.0)

    traded = side != 0.0
    gross = np.where(traded, side * (exit_price - entry), 0.0)
    friction = float(knobs.frozen_value(family.FRICTION_KNOB))
    net = gross - np.where(traded, friction, 0.0)

    folds = loader.chronological_folds(
        len(frame), int(knobs.frozen_value(family.FOLD_COUNT_KNOB))
    )
    result = pd.DataFrame(
        {
            "session": frame["session"].to_numpy(),
            "fold": folds,
            "side": side,
            "trades": traded.astype(int),
            "gross_points": gross,
            "net_points": net,
        }
    )
    if len(result) != member.eligible_sessions:
        raise ReplayError(
            f"{member.name} produced {len(result)} sessions but the frozen "
            f"declaration requires {member.eligible_sessions}; a declared "
            "session may never be dropped"
        )
    return result


def constant_side_sessions(
    features: pd.DataFrame, member: family.Member, control: str
) -> pd.DataFrame:
    """A frozen constant-side control on the member's own calendar and clock.

    Same sessions, same entry and exit prices, same friction — only the side is
    replaced. That is what makes the paired comparison a test of *selection*
    rather than of market drift.
    """

    eligible = loader.eligible_sessions(features, member.mechanism)
    frame = features.loc[eligible].reset_index(drop=True)
    entry = frame["entry_price"].to_numpy(float)
    exit_price = frame[f"exit_price_{member.horizon_minutes}m"].to_numpy(float)
    priced = np.isfinite(entry) & np.isfinite(exit_price)

    if control == "always_long":
        side = np.where(priced, 1.0, 0.0)
    elif control == "always_short":
        side = np.where(priced, -1.0, 0.0)
    elif control == "no_trade":
        side = np.zeros(len(frame))
    else:
        raise ReplayError(f"unknown constant-side control: {control}")

    traded = side != 0.0
    gross = np.where(traded, side * (exit_price - entry), 0.0)
    friction = float(knobs.frozen_value(family.FRICTION_KNOB))
    net = gross - np.where(traded, friction, 0.0)
    folds = loader.chronological_folds(
        len(frame), int(knobs.frozen_value(family.FOLD_COUNT_KNOB))
    )
    return pd.DataFrame(
        {
            "session": frame["session"].to_numpy(),
            "fold": folds,
            "trades": traded.astype(int),
            "gross_points": gross,
            "net_points": net,
        }
    )


def causal_comparator_net(
    features: pd.DataFrame, member: family.Member
) -> tuple[np.ndarray, dict[int, str]]:
    """The comparator's per-session net, chosen without seeing the rows it faces.

    For fold ``k`` the comparator is whichever constant side did best on the
    sessions strictly *before* fold ``k`` begins. Fold 0 has no prior sessions,
    so it takes the declared ``no_trade`` default. This is the audit's
    "selection inside training folds" made concrete, and it is why the paired
    bound is not a comparison a policy can win by hindsight.
    """

    controls = {
        name: constant_side_sessions(features, member, name)
        for name in family.COMPARATOR_CONTROLS
    }
    reference = controls[family.COMPARATOR_CONTROLS[0]]
    folds = reference["fold"].to_numpy()
    chosen: dict[int, str] = {}
    net = np.zeros(len(reference))

    for fold in sorted(set(folds.tolist())):
        prior = folds < fold
        if not prior.any():
            pick = family.COMPARATOR_FOLD0
        else:
            pick = max(
                family.COMPARATOR_CONTROLS,
                key=lambda name: float(controls[name]["net_points"].to_numpy()[prior].mean()),
            )
        chosen[int(fold)] = pick
        here = folds == fold
        net[here] = controls[pick]["net_points"].to_numpy()[here]
    return net, chosen
