"""Candidate entry x exit pairs, declared before they are scored.

The owner's standing claim is that entry and exit are profitable *in
combination* rather than separately, and the 2026-08-13 stop study is consistent
with that: a declared -30% exit halves the loss when the direction call is wrong
(7.55% of account to 4.29%) while changing mean return by an amount the data
cannot distinguish from noise. An exit alone is a risk control. Whether entry
selection plus exit discipline compounds is the open question, and it is the
thing this generator exists to enumerate.

Every pair is a *declaration*: a named entry rule, a named exit rule, and the
features each is allowed to read. Nothing here fits anything. The rules are
closed-form and enumerable so that the multiplicity is exactly countable, which
is what lets the alpha ledger price the search honestly. A learned ranker enters
this same interface later as one more declared candidate, judged at whatever bar
the ledger has reached by then.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from v5.research.autoresearch import experiment


class GeneratorError(RuntimeError):
    """A candidate cannot be declared as stated."""


# --- entry rules -------------------------------------------------------------
# Each maps causal per-session features to a call in {-1, 0, +1}. Zero is a
# genuine abstention and is excluded from scoring rather than counted wrong.
# Every rule reads only fields available at the decision instant.

EntryRule = Callable[[Mapping[str, np.ndarray]], np.ndarray]


def _sign(values: np.ndarray) -> np.ndarray:
    return np.sign(values)


def entry_gap_with(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade with the overnight gap."""

    return _sign(f["overnight_gap"])


def entry_gap_against(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fade the overnight gap."""

    return -_sign(f["overnight_gap"])


def entry_open_range_with(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade with the first five minutes."""

    return _sign(f["first_five_minute_return"])


def entry_open_range_against(f: Mapping[str, np.ndarray]) -> np.ndarray:
    return -_sign(f["first_five_minute_return"])


def entry_confirmed_gap(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade the gap only when the opening range agrees with it.

    Abstains on disagreement, which is the point: standing down is a first-class
    outcome and a conjunction rule buys selectivity at the cost of occupancy.
    """

    gap = _sign(f["overnight_gap"])
    opening = _sign(f["first_five_minute_return"])
    return np.where(gap == opening, gap, 0.0)


def entry_contrarian_gap(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fade the gap only when the opening range disagrees with it."""

    gap = _sign(f["overnight_gap"])
    opening = _sign(f["first_five_minute_return"])
    return np.where(gap != opening, opening, 0.0)


ENTRY_RULES: Mapping[str, tuple[EntryRule, tuple[str, ...]]] = {
    "gap_with": (entry_gap_with, ("overnight_gap",)),
    "gap_against": (entry_gap_against, ("overnight_gap",)),
    "open_range_with": (entry_open_range_with, ("first_five_minute_return",)),
    "open_range_against": (entry_open_range_against, ("first_five_minute_return",)),
    "confirmed_gap": (
        entry_confirmed_gap,
        ("overnight_gap", "first_five_minute_return"),
    ),
    "contrarian_gap": (
        entry_contrarian_gap,
        ("overnight_gap", "first_five_minute_return"),
    ),
}

# --- exit rules --------------------------------------------------------------
# Declared levels, never searched. A searched stop level would be a threshold
# search, which is owner-gated and would also make the multiplicity unbounded.
EXIT_RULES: Mapping[str, float | None] = {
    "hold_to_horizon": None,
    "stop_20": -0.20,
    "stop_30": -0.30,
    "stop_40": -0.40,
}


@dataclass(frozen=True)
class Candidate:
    """One declared entry x exit pair."""

    entry: str
    exit: str
    horizon_minutes: int = 60

    def __post_init__(self) -> None:
        if self.entry not in ENTRY_RULES:
            raise GeneratorError(f"undeclared entry rule: {self.entry}")
        if self.exit not in EXIT_RULES:
            raise GeneratorError(f"undeclared exit rule: {self.exit}")

    @property
    def name(self) -> str:
        return f"{self.entry}+{self.exit}@{self.horizon_minutes}m"

    @property
    def features(self) -> tuple[str, ...]:
        return ENTRY_RULES[self.entry][1]

    @property
    def stop_level(self) -> float | None:
        return EXIT_RULES[self.exit]

    def declaration(self, *, declared_on: str) -> experiment.Declaration:
        return experiment.Declaration(
            experiment_id=self.name,
            hypothesis=(
                f"entry {self.entry} combined with exit {self.exit} over "
                f"{self.horizon_minutes} minutes"
            ),
            features_used=self.features,
            horizon_minutes=self.horizon_minutes,
            declared_on=declared_on,
        )

    def calls(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        rule, needed = ENTRY_RULES[self.entry]
        missing = [name for name in needed if name not in features]
        if missing:
            raise GeneratorError(f"{self.name} needs missing features: {missing}")
        calls = rule(features)
        calls = np.where(np.isfinite(calls), calls, 0.0)
        return calls.astype(int)


def enumerate_candidates(
    *,
    entries: Sequence[str] | None = None,
    exits: Sequence[str] | None = None,
    horizon_minutes: int = 60,
) -> tuple[Candidate, ...]:
    """The full declared cross product, in a stable order.

    Enumerated rather than sampled so the multiplicity is exactly countable
    before the loop starts: the alpha ledger can be asked what the bar will be
    at the end of the sweep before a single experiment runs.
    """

    entries = tuple(entries or ENTRY_RULES)
    exits = tuple(exits or EXIT_RULES)
    return tuple(
        Candidate(entry=e, exit=x, horizon_minutes=horizon_minutes)
        for e, x in product(entries, exits)
    )


def apply_exit(
    stop_level: float | None,
    *,
    path_ratios: np.ndarray,
    final_ratio: np.ndarray,
) -> np.ndarray:
    """Realised outcome ratio per trade under a declared exit.

    ``path_ratios`` is the per-minute return ratio of the *option* after entry,
    shaped (trades, minutes). A stop takes the fill actually available at the
    first minute the level is breached rather than the level itself, so the
    slippage a declared stop cannot assume away is carried through.
    """

    if stop_level is None:
        return final_ratio.astype(float)
    breached = path_ratios <= stop_level
    any_breach = breached.any(axis=1)
    first = np.argmax(breached, axis=1)
    fills = path_ratios[np.arange(len(path_ratios)), first]
    return np.where(any_breach, fills, final_ratio).astype(float)


def inner_loop(
    features: Mapping[str, np.ndarray],
    realized: np.ndarray,
    *,
    candidates: Sequence[Candidate],
    declared_on: str,
) -> Callable[[Any], Iterator[tuple[experiment.Declaration, np.ndarray, np.ndarray]]]:
    """An inner loop the outer constraint search can call.

    Yields one declared experiment per candidate. The exit rule does not change
    which sessions are *called*, only what each trade returns, so directional
    accuracy is scored on the entry and the exit is judged by what it does to
    the account -- which is the split the 2026-08-13 stop study argues for.
    """

    def make(_setting: Any) -> Iterator[
        tuple[experiment.Declaration, np.ndarray, np.ndarray]
    ]:
        for candidate in candidates:
            yield (
                candidate.declaration(declared_on=declared_on),
                candidate.calls(features),
                realized,
            )

    return make
