"""The alpha ledger: an append-only, hash-chained count of every experiment run.

The ledger is the control. Each entry links to its predecessor's hash, so an
experiment cannot be removed after the fact to make a later result look better
than it was -- which is the precise move that would turn this loop back into the
thing it replaces.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from v5.research import knobs, statistics


SCHEMA_VERSION = "v5.autoresearch-alpha-ledger.v1"
GENESIS = "0" * 64

# Measured 2026-08-12: the gap between a declared threshold and the true accuracy
# a bootstrap lower bound needs to clear it, at these sample sizes. Recorded so
# the stop condition is stated in true accuracy rather than in a bound.
LOWER_BOUND_GAP = 0.026


class BudgetError(RuntimeError):
    """The ledger would be made to say something untrue."""


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Entry:
    """One experiment, counted whether it succeeded or not."""

    index: int
    experiment_id: str
    declaration_sha256: str
    declared_on: str
    outcome: str  # "PASS" | "FAIL" | "REFUSED"
    observed_accuracy: float | None
    accuracy_lower_bound: float | None
    bar_at_time_of_run: float
    previous_hash: str
    entry_hash: str

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("entry_hash")
        return data


def _hash_entry(entry_fields: dict[str, Any]) -> str:
    return _sha256(entry_fields)


class AlphaLedger:
    """Append-only record of experiments, and the bar they have bought."""

    #: Re-exported so callers can catch the registry error without a second import.
    knobs = knobs

    #: Payoff pairs by contract universe, measured 2026-08-13 across the strike
    #: ladder. Which one a loop is judged against is a *declaration*, not a
    #: default: the same directional skill breaks even at 50.60% near the money
    #: and 54.50% in the OTM band a $3-8 price filter selects, and the ladder is
    #: not monotonic -- deeper ITM is worse, because the dollar spread grows
    #: faster than the payoff.
    UNIVERSES: Mapping[str, tuple[str, str]] = {
        "near_atm": ("near_atm_correct_call_dollars", "near_atm_wrong_call_dollars"),
        "phase1_otm": ("option_correct_call_dollars", "option_wrong_call_dollars"),
    }

    def __init__(
        self,
        path: Path,
        *,
        option_sessions: int,
        universe: str,
        ceiling: float = 0.75,
    ):
        if option_sessions <= 0:
            raise BudgetError("option_sessions must be positive")
        if not 0.5 < ceiling < 1.0:
            raise BudgetError("ceiling must be a plausible accuracy in (0.5, 1)")
        if universe not in self.UNIVERSES:
            raise BudgetError(
                f"unknown contract universe {universe!r}; declare one of "
                f"{sorted(self.UNIVERSES)}"
            )
        self.path = Path(path)
        self.option_sessions = int(option_sessions)
        self.universe = universe
        self.ceiling = float(ceiling)
        self._entries: list[Entry] = []
        if self.path.exists():
            self._load()

    # --- reading ----------------------------------------------------------
    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise BudgetError(f"unknown ledger schema: {payload.get('schema_version')}")
        previous = GENESIS
        for raw in payload.get("entries", []):
            entry = Entry(**raw)
            if entry.previous_hash != previous:
                raise BudgetError(
                    f"ledger chain broken at entry {entry.index}: an experiment was "
                    "removed or reordered"
                )
            if _hash_entry(entry.payload()) != entry.entry_hash:
                raise BudgetError(f"ledger entry {entry.index} was edited after the fact")
            previous = entry.entry_hash
            self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries)

    @property
    def experiments_run(self) -> int:
        return len(self._entries)

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    # --- the bar ----------------------------------------------------------
    def required_accuracy(self, *, experiments: int | None = None) -> float:
        """The accuracy an experiment must clear, given how many have been run.

        The count includes the experiment about to run: attempt number k is
        judged at the k-experiment bar, never the one-experiment bar.
        """

        k = self.experiments_run + 1 if experiments is None else int(experiments)
        if k < 1:
            raise BudgetError("experiment count must be at least 1")
        z = statistics.normal_quantile(1.0 - 0.05 / k)
        win, loss = self.payoffs()
        return statistics.detectable_accuracy(
            self.option_sessions, win=win, loss=loss, z_alpha=z
        )

    def payoffs(self) -> tuple[float, float]:
        """The declared universe's measured correct/wrong pair."""

        win_knob, loss_knob = self.UNIVERSES[self.universe]
        return (
            float(knobs.frozen_value(win_knob)),
            abs(float(knobs.frozen_value(loss_knob))),
        )

    def breakeven_accuracy(self) -> float:
        win, loss = self.payoffs()
        return statistics.breakeven_accuracy(win=win, loss=loss)

    def required_true_accuracy(self, *, experiments: int | None = None) -> float:
        """What a real strategy must actually have, not what a bound must show."""

        return self.required_accuracy(experiments=experiments) + LOWER_BOUND_GAP

    @property
    def exhausted(self) -> bool:
        """True once the next experiment could not be proven even if perfect."""

        return self.required_true_accuracy() >= self.ceiling

    def experiments_remaining(self) -> int:
        """How many more attempts the corpus can support before the bar passes
        the plausible ceiling. Answers 'how long may this loop run?' up front."""

        if self.exhausted:
            return 0
        lo, hi = self.experiments_run + 1, 10_000_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.required_true_accuracy(experiments=mid) < self.ceiling:
                lo = mid
            else:
                hi = mid - 1
        return lo - self.experiments_run

    # --- writing ----------------------------------------------------------
    def record(
        self,
        *,
        experiment_id: str,
        declaration_sha256: str,
        declared_on: str,
        outcome: str,
        observed_accuracy: float | None = None,
        accuracy_lower_bound: float | None = None,
    ) -> Entry:
        """Append an experiment. Counted whether it passed, failed or was refused.

        A refused experiment still spent an attempt: the decision to look is what
        costs alpha, not the result of looking.
        """

        if outcome not in {"PASS", "FAIL", "REFUSED"}:
            raise BudgetError(f"unknown outcome: {outcome}")
        if any(e.experiment_id == experiment_id for e in self._entries):
            raise BudgetError(f"experiment already recorded: {experiment_id}")

        bar = self.required_accuracy()
        fields = {
            "index": len(self._entries),
            "experiment_id": experiment_id,
            "declaration_sha256": declaration_sha256,
            "declared_on": declared_on,
            "outcome": outcome,
            "observed_accuracy": observed_accuracy,
            "accuracy_lower_bound": accuracy_lower_bound,
            "bar_at_time_of_run": round(bar, 8),
            "previous_hash": self.head,
        }
        entry = Entry(**fields, entry_hash=_hash_entry(fields))
        self._entries.append(entry)
        self._write()
        return entry

    def _write(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "option_sessions": self.option_sessions,
            "contract_universe": self.universe,
            "breakeven_accuracy": round(self.breakeven_accuracy(), 8),
            "plausible_accuracy_ceiling": self.ceiling,
            "experiments_run": self.experiments_run,
            "next_bar": round(self.required_accuracy(), 8),
            "next_true_accuracy_needed": round(self.required_true_accuracy(), 8),
            "exhausted": self.exhausted,
            "head": self.head,
            "entries": [asdict(e) for e in self._entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def summary(self) -> dict[str, Any]:
        passes = [e for e in self._entries if e.outcome == "PASS"]
        return {
            "experiments_run": self.experiments_run,
            "option_sessions": self.option_sessions,
            "contract_universe": self.universe,
            "breakeven_accuracy": round(self.breakeven_accuracy(), 6),
            "next_bar_accuracy": round(self.required_accuracy(), 6),
            "next_true_accuracy_needed": round(self.required_true_accuracy(), 6),
            "plausible_accuracy_ceiling": self.ceiling,
            "exhausted": self.exhausted,
            "experiments_remaining": self.experiments_remaining(),
            "passes": [e.experiment_id for e in passes],
            "head": self.head,
        }
