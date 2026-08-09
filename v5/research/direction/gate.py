"""The frozen G1 pass criteria, applied to one member's session index.

Six criteria, all of which must hold, taken verbatim from
:data:`family.PASS_CRITERIA`.  Nothing here is tunable: the confidence level,
fold count, fold rule, bootstrap seed and friction all come from the knob
registry, so a gate cannot loosen by drift.

The gate is deliberately usable on *any* session index — real or surrogate —
because the known-answer campaign has to push thousands of surrogate indices
through the identical code.  A gate that behaved differently on a null than on
the real thing would measure nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from v5.research import knobs
from v5.research.direction import family, replay
from v5.research.validation.replay_gate import block_bootstrap_lower_bound


class GateError(RuntimeError):
    """A member cannot be judged as the frozen declaration requires."""


@dataclass(frozen=True)
class MemberResult:
    """One member's verdict, with every criterion's answer kept separately."""

    member: str
    passed: bool
    criteria: Mapping[str, bool]
    net_lower_bound: float
    paired_lower_bound: float
    mean_net_points: float
    gross_points_per_trade: float
    trades: int
    sessions: int
    folds_net_positive: int
    folds_paired_positive: int
    comparator_by_fold: Mapping[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member,
            "passed": self.passed,
            "criteria": dict(self.criteria),
            "net_lower_bound": self.net_lower_bound,
            "paired_lower_bound": self.paired_lower_bound,
            "mean_net_points": self.mean_net_points,
            "gross_points_per_trade": self.gross_points_per_trade,
            "trades": self.trades,
            "sessions": self.sessions,
            "folds_net_positive": self.folds_net_positive,
            "folds_paired_positive": self.folds_paired_positive,
            "comparator_by_fold": {int(k): v for k, v in self.comparator_by_fold.items()},
        }


def _folds_positive(values: np.ndarray, folds: np.ndarray) -> int:
    return sum(
        float(values[folds == fold].mean()) > 0.0 for fold in sorted(set(folds.tolist()))
    )


def _fold_rule() -> tuple[int, int]:
    rule = str(knobs.frozen_value(family.FOLD_RULE_KNOB))
    needed, _, total = rule.partition("_of_")
    return int(needed), int(total)


def familywise_confidence(family_size: int = family.FAMILY_SIZE) -> float:
    """Per-member confidence, Bonferroni-adjusted for the declared family size.

    Criterion 5 requires the whole eighteen-member family to be familywise
    controlled. Judging each member at the raw 95% level and then reporting
    "any member passed" makes eighteen attempts at significance while quoting
    the price of one: the 2026-08-09 pilot measured 11.7% familywise false-pass
    on matched surrogates against a declared 5.0% limit.

    The adjustment is applied here, inside the gate, so the null campaign and
    the real replay are judged by identical arithmetic.
    """

    level = float(knobs.frozen_value(family.CONFIDENCE_KNOB))
    alpha = (1.0 - level) / max(int(family_size), 1)
    return 1.0 - alpha


def judge_member(
    features: pd.DataFrame,
    member: family.Member,
    *,
    seed_offset: int = 0,
    family_size: int = family.FAMILY_SIZE,
) -> MemberResult:
    """Apply all six frozen criteria to one member on one session index."""

    sessions = replay.replay_member(features, member)
    comparator_net, comparator_by_fold = replay.causal_comparator_net(features, member)

    net = sessions["net_points"].to_numpy(float)
    folds = sessions["fold"].to_numpy()
    paired = net - comparator_net
    trades = int(sessions["trades"].sum())
    gross_total = float(sessions["gross_points"].sum())

    seed = int(knobs.frozen_value(family.BOOTSTRAP_SEED_KNOB)) + int(seed_offset)
    confidence = familywise_confidence(family_size)
    net_lb = block_bootstrap_lower_bound(net, confidence=confidence, seed=seed)
    paired_lb = block_bootstrap_lower_bound(paired, confidence=confidence, seed=seed)

    needed, total = _fold_rule()
    folds_net = _folds_positive(net, folds)
    folds_paired = _folds_positive(paired, folds)
    friction = float(knobs.frozen_value(family.FRICTION_KNOB))
    per_trade = gross_total / trades if trades else 0.0

    criteria = {
        "absolute_lower_bound_positive": bool(net_lb > 0.0),
        "paired_lower_bound_positive": bool(paired_lb > 0.0),
        "four_of_five_folds_both": bool(folds_net >= needed and folds_paired >= needed),
        "gross_per_trade_clears_friction": bool(trades > 0 and per_trade > friction),
        # Familywise control is a property of the campaign, not of one member,
        # so it is asserted by the caller that ran the campaign rather than
        # silently assumed true here.
        "full_index_evaluated": bool(len(sessions) == member.eligible_sessions),
        "fold_count_as_declared": bool(len(set(folds.tolist())) == total),
    }

    return MemberResult(
        member=member.name,
        passed=all(criteria.values()),
        criteria=criteria,
        net_lower_bound=float(net_lb),
        paired_lower_bound=float(paired_lb),
        mean_net_points=float(net.mean()),
        gross_points_per_trade=float(per_trade),
        trades=trades,
        sessions=len(sessions),
        folds_net_positive=int(folds_net),
        folds_paired_positive=int(folds_paired),
        comparator_by_fold=comparator_by_fold,
    )


def judge_family(
    features: pd.DataFrame, *, seed_offset: int = 0
) -> tuple[MemberResult, ...]:
    """Every frozen member on one session index, in declaration order."""

    return tuple(
        judge_member(
            features, member, seed_offset=seed_offset, family_size=family.FAMILY_SIZE
        )
        for member in family.FAMILY
    )


def any_member_passed(results: tuple[MemberResult, ...]) -> bool:
    """The familywise event the surrogate campaign measures.

    A campaign counts as a false pass when **any** of the eighteen members
    clears the gate, because a screen that reports the best of eighteen is
    making eighteen attempts at significance whether or not it says so.
    """

    return any(result.passed for result in results)
