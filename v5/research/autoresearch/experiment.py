"""One experiment: declared and hashed before it is scored, then counted.

An experiment produces a per-session directional call. It is scored on accuracy,
and accuracy is translated into option economics through the frozen conditional
outcomes rather than by replaying the chain, because those outcomes are already
measured on 852 real trajectories.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from v5.research import knobs, statistics
from v5.research.autoresearch import budget
from v5.research.validation.replay_gate import block_bootstrap_lower_bound


class ExperimentError(RuntimeError):
    """An experiment cannot be declared or scored as stated."""


CALL_VALUES = (-1, 0, 1)


@dataclass(frozen=True)
class Declaration:
    """What an experiment commits to, before it sees an outcome."""

    experiment_id: str
    hypothesis: str
    features_used: tuple[str, ...]
    horizon_minutes: int
    declared_on: str

    def payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "features_used": list(self.features_used),
            "horizon_minutes": self.horizon_minutes,
            "declared_on": self.declared_on,
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.hypothesis:
            raise ExperimentError("an experiment needs an id and a stated hypothesis")
        if not self.features_used:
            raise ExperimentError(
                "an experiment must name the features it uses, so a later reader "
                "can tell whether it was causal"
            )
        if self.horizon_minutes <= 0:
            raise ExperimentError("horizon must be positive")


@dataclass(frozen=True)
class Score:
    experiment_id: str
    declaration_sha256: str
    called: int
    correct: int
    accuracy: float
    accuracy_lower_bound: float
    bar: float
    passed: bool
    expected_option_pnl_per_trade: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "declaration_sha256": self.declaration_sha256,
            "called": self.called,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "accuracy_lower_bound": round(self.accuracy_lower_bound, 6),
            "bar": round(self.bar, 6),
            "passed": self.passed,
            "expected_option_pnl_per_trade": round(
                self.expected_option_pnl_per_trade, 4
            ),
        }


def score_calls(
    declaration: Declaration,
    calls: Sequence[int],
    realized: Sequence[float],
    *,
    ledger: budget.AlphaLedger,
    seed: int = 0,
) -> Score:
    """Score one experiment against the bar its position in the ledger buys.

    ``calls`` is +1, -1 or 0 per session; ``realized`` is the signed underlying
    move. A zero call stands down and is excluded rather than counted as wrong,
    because abstention is a first-class outcome. A zero *move* under a non-zero
    call is counted as incorrect.
    """

    call_array = np.asarray(calls, int)
    move = np.asarray(realized, float)
    if call_array.shape != move.shape:
        raise ExperimentError("calls and realized moves are not aligned")
    if not np.isin(call_array, CALL_VALUES).all():
        raise ExperimentError("calls must be -1, 0 or +1")
    if not np.isfinite(move).all():
        raise ExperimentError("realized moves contain non-finite values")

    traded = call_array != 0
    if not traded.any():
        raise ExperimentError("experiment stood down on every session")
    hits = (call_array[traded] * move[traded] > 0.0).astype(float)

    bar = ledger.required_accuracy()
    confidence = 1.0 - 0.05 / max(ledger.experiments_run + 1, 1)
    lower = float(
        block_bootstrap_lower_bound(hits, confidence=min(confidence, 0.9999), seed=seed)
    )
    accuracy = float(hits.mean())

    win = knobs.frozen_value("option_correct_call_dollars")
    loss = knobs.frozen_value("option_wrong_call_dollars")
    return Score(
        experiment_id=declaration.experiment_id,
        declaration_sha256=declaration.sha256(),
        called=int(traded.sum()),
        correct=int(hits.sum()),
        accuracy=accuracy,
        accuracy_lower_bound=lower,
        bar=bar,
        passed=bool(lower > bar),
        expected_option_pnl_per_trade=statistics.accuracy_payoff(
            accuracy, win=win, loss=loss
        ),
    )


def run_and_record(
    declaration: Declaration,
    calls: Sequence[int],
    realized: Sequence[float],
    *,
    ledger: budget.AlphaLedger,
    seed: int = 0,
) -> Score:
    """Score an experiment and spend its alpha, in that order and always both.

    If scoring raises, the attempt is still recorded as REFUSED: an experiment
    that was started and abandoned has still consumed a look at the data.
    """

    if ledger.exhausted:
        raise ExperimentError(
            f"alpha budget exhausted: the next experiment would need "
            f"{100 * ledger.required_true_accuracy():.2f}% true accuracy, at or "
            f"above the declared plausibility ceiling of {100 * ledger.ceiling:.2f}%"
        )
    try:
        score = score_calls(declaration, calls, realized, ledger=ledger, seed=seed)
    except ExperimentError:
        ledger.record(
            experiment_id=declaration.experiment_id,
            declaration_sha256=declaration.sha256(),
            declared_on=declaration.declared_on,
            outcome="REFUSED",
        )
        raise
    ledger.record(
        experiment_id=declaration.experiment_id,
        declaration_sha256=declaration.sha256(),
        declared_on=declaration.declared_on,
        outcome="PASS" if score.passed else "FAIL",
        observed_accuracy=score.accuracy,
        accuracy_lower_bound=score.accuracy_lower_bound,
    )
    return score
