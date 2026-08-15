"""The outer loop: a search over constraints, refereed by the knob registry.

Two levels. The **inner** loop trains and scores candidates under a fixed set of
constraints. The **outer** loop varies the constraints themselves -- which part
of the strike ladder may be bought, which horizon, how long to hold.

The 2026-08-13 moneyness measurement is why this level exists at all: the
contract universe was worth 3.9 accuracy points, more than nine years of
additional data collection would have bought, and it had been pinned implicitly
by a price filter nobody had written down as a decision.

Two controls keep that from becoming a machine for manufacturing false
positives, which is what an unrefereed constraint search would be:

**The registry is the referee.** Every proposed setting goes through
:func:`v5.research.knobs.assert_search_space`, which refuses an unknown name, a
``FROZEN`` constant, an ``UNCERTIFIED`` one, a knob whose releasing gate has not
passed, and any value outside the declared space. The outer loop therefore
cannot improve a result by quietly relaxing friction, the fill law, or the
emission allowance.

**Constraint settings spend alpha.** Trying a new configuration is a hypothesis
like any other. Every setting the outer loop occupies is recorded in the same
append-only ledger the inner experiments use, so the bar rises whether the
search widened the model space or the constraint space.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from v5.research import knobs
from v5.research.autoresearch import budget, experiment


class OuterLoopError(RuntimeError):
    """A proposed constraint setting may not be occupied."""


#: Knobs released by a signed charter amendment rather than by a gate. The value
#: is the amendment that released it, so a receipt can name which signed document
#: opened each degree of freedom. An amendment can release only a SEARCHABLE
#: knob: it changes what the bot is permitted to try, never what the evidence says.
SIGNED_AMENDMENTS: Mapping[str, str] = {
    "moneyness_band": (
        "v5/governance/CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md"
    ),
}


@dataclass(frozen=True)
class ConstraintSetting:
    """One point in the constraint space, declared before it is occupied."""

    setting_id: str
    params: Mapping[str, Any]
    rationale: str
    declared_on: str = "unknown"

    def __post_init__(self) -> None:
        if not self.setting_id:
            raise OuterLoopError("a constraint setting needs an id")
        if not self.params:
            raise OuterLoopError("a constraint setting must vary something")
        if not self.rationale:
            raise OuterLoopError(
                "a constraint setting must say why it is worth an attempt; an "
                "unexplained setting is indistinguishable from a fishing trip"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "params": {str(k): v for k, v in sorted(self.params.items())},
            "rationale": self.rationale,
            "declared_on": self.declared_on,
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def check(setting: ConstraintSetting, *, released_gates: Iterable[str] = ()) -> None:
    """Ask the registry whether this setting may be occupied at all.

    Raises :class:`v5.research.knobs.KnobError` with the exact reason, which is
    the message worth surfacing: it names the gate that has not passed or the
    evidence that froze the constant.
    """

    knobs.assert_search_space(
        setting.params,
        released_gates=released_gates,
        released_by_amendment=SIGNED_AMENDMENTS,
    )


def occupiable(
    settings: Sequence[ConstraintSetting], *, released_gates: Iterable[str] = ()
) -> tuple[tuple[ConstraintSetting, ...], dict[str, str]]:
    """Split proposed settings into those the registry permits and those it does not.

    Returned rather than raised, so a loop can report the whole refusal surface
    in one pass instead of stopping at the first blocked knob.
    """

    allowed: list[ConstraintSetting] = []
    refused: dict[str, str] = {}
    for setting in settings:
        try:
            check(setting, released_gates=released_gates)
        except knobs.KnobError as exc:
            refused[setting.setting_id] = str(exc)
        else:
            allowed.append(setting)
    return tuple(allowed), refused


@dataclass(frozen=True)
class OuterResult:
    setting_id: str
    setting_sha256: str
    inner_experiments: int
    best_accuracy: float | None
    best_lower_bound: float | None
    any_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "setting_sha256": self.setting_sha256,
            "inner_experiments": self.inner_experiments,
            "best_accuracy": (
                round(self.best_accuracy, 6) if self.best_accuracy is not None else None
            ),
            "best_lower_bound": (
                round(self.best_lower_bound, 6)
                if self.best_lower_bound is not None
                else None
            ),
            "any_passed": self.any_passed,
        }


def run_setting(
    setting: ConstraintSetting,
    *,
    ledger: budget.AlphaLedger,
    inner: Callable[[ConstraintSetting], Iterable[tuple[experiment.Declaration, Sequence[int], Sequence[float]]]],
    released_gates: Iterable[str] = (),
) -> OuterResult:
    """Occupy one constraint setting and run its inner experiments.

    The setting itself is charged to the ledger first, before any inner
    experiment runs. Occupying a configuration is a look at the data whether or
    not any candidate under it turns out to be worth scoring.
    """

    check(setting, released_gates=released_gates)
    if ledger.exhausted:
        raise OuterLoopError(
            "alpha budget exhausted before this setting could be occupied; the "
            f"next attempt would need {100 * ledger.required_true_accuracy():.2f}% "
            f"true accuracy against a ceiling of {100 * ledger.ceiling:.2f}%"
        )

    ledger.record(
        experiment_id=f"setting:{setting.setting_id}",
        declaration_sha256=setting.sha256(),
        declared_on=setting.declared_on,
        outcome="REFUSED",  # a setting is an attempt, never a result
    )

    best_acc: float | None = None
    best_lb: float | None = None
    passed = False
    count = 0
    for declaration, calls, realized in inner(setting):
        if ledger.exhausted:
            break
        score = experiment.run_and_record(
            declaration, calls, realized, ledger=ledger
        )
        count += 1
        if best_acc is None or score.accuracy > best_acc:
            best_acc, best_lb = score.accuracy, score.accuracy_lower_bound
        passed = passed or score.passed
    return OuterResult(
        setting_id=setting.setting_id,
        setting_sha256=setting.sha256(),
        inner_experiments=count,
        best_accuracy=best_acc,
        best_lower_bound=best_lb,
        any_passed=passed,
    )
