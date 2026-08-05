"""Autoresearch loop for the Path-D Full Trader.

Runs waves of pre-registered hypotheses: prior-art check -> pre-flight ->
experiment -> gate -> registry -> continue or stop. The loop is deliberately
decoupled from any particular trainer; a runner is any callable taking a
``Hypothesis`` and returning an ``ExperimentResult``.

Three properties make autonomous iteration statistically valid rather than a
slot machine:

1. **Bounded budget.** A wave declares its hypothesis count up front. Exhausting
   it without TIER_A ends the wave as NO_EDGE. Abandoned hypotheses still count,
   because they were still looks at the data.
2. **Semantic dedup.** The registry rejects a mechanism already run, even under a
   new name, so the same idea cannot be retried until it happens to pass.
3. **Executable prior art.** Every hypothesis is checked against the project's
   do-not-retest ledger before it runs. Path-D re-ran Protocols 029/030/031 and
   April's closed coverage-threshold result because that check was manual and
   nobody performed it.

Governance: the protected holdout is SPENT. TIER_A means "worth a forward
live-paper test", never "deployable".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from v4.research.pathd_model_gate import TestResult, acceptance_tier
from v5.research.prior_art import (
    LoopError,
    PriorArtHit,
    blocked_by_prior_art,
    prior_art_check,
)


# --------------------------------------------------------------------------
# Hypotheses and waves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    """One pre-registered experiment.

    ``mechanism`` is the plain-language name searched against prior art, and is
    what the semantic hash is built from. Renaming a hypothesis without changing
    its mechanism will not evade dedup.
    """

    hypothesis_id: str
    mechanism: str
    params: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def semantic_hash(self) -> str:
        payload = json.dumps(
            {"mechanism": self.mechanism.strip().lower(), "params": dict(sorted(self.params.items()))},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class WaveSpec:
    wave_id: str
    objective: str
    hypotheses: Sequence[Hypothesis]
    budget: int

    def validate(self) -> None:
        if self.budget <= 0:
            raise LoopError("wave budget must be positive")
        if len(self.hypotheses) > self.budget:
            raise LoopError(
                f"wave declares {len(self.hypotheses)} hypotheses but a budget of {self.budget}; "
                "the budget is the multiplicity family size and cannot be exceeded"
            )
        ids = [h.hypothesis_id for h in self.hypotheses]
        if len(set(ids)) != len(ids):
            raise LoopError("duplicate hypothesis_id in wave")


@dataclass
class ExperimentResult:
    """What a runner must return for the gate to score a hypothesis."""

    pooled_policy: float
    pooled_comparator: float
    fold_deltas: Mapping[str, float]
    bootstrap_lcb: float
    negative_controls_accepted: Mapping[str, bool]
    rejection_tests: Sequence[TestResult]
    maxt_survived: bool = False
    concentrated: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


Runner = Callable[[Hypothesis], ExperimentResult]


# --------------------------------------------------------------------------
# Append-only registry with semantic dedup
# --------------------------------------------------------------------------


class Registry:
    """Append-only JSONL of every hypothesis ever run, keyed by semantic hash."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def seen(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                entries[record["semantic_hash"]] = record
        return entries

    def record(self, entry: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_wave(
    spec: WaveSpec,
    runner: Runner,
    *,
    registry_path: Path,
    allow_prior_art_override: bool = False,
) -> dict[str, Any]:
    """Execute one bounded wave and return its report.

    Stops early on TIER_A (found what we were looking for) or INVALID (an accepted
    negative control discards the family). Otherwise runs to budget exhaustion and
    returns NO_EDGE, which is a successful outcome.
    """

    spec.validate()
    registry = Registry(registry_path)
    already = registry.seen()

    results: list[dict[str, Any]] = []
    spent = 0
    verdict = "NO_EDGE"

    for hypothesis in spec.hypotheses:
        if spent >= spec.budget:
            break

        semantic = hypothesis.semantic_hash()
        entry: dict[str, Any] = {
            "wave_id": spec.wave_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "mechanism": hypothesis.mechanism,
            "params": dict(hypothesis.params),
            "semantic_hash": semantic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if semantic in already:
            entry["status"] = "SKIPPED_SEMANTIC_DUPLICATE"
            entry["duplicate_of"] = already[semantic].get("hypothesis_id")
            results.append(entry)
            registry.record(entry)
            continue

        hits = prior_art_check(hypothesis.mechanism)
        entry["prior_art_hits"] = len(hits)
        entry["prior_art_blocking"] = [asdict(h) for h in hits if h.blocking]

        if blocked_by_prior_art(hits) and not allow_prior_art_override:
            entry["status"] = "BLOCKED_BY_PRIOR_ART"
            results.append(entry)
            registry.record(entry)
            continue

        # The look at the data is spent whether or not the hypothesis pays off.
        spent += 1
        result = runner(hypothesis)
        scored = acceptance_tier(
            pooled_policy=result.pooled_policy,
            pooled_comparator=result.pooled_comparator,
            fold_deltas=result.fold_deltas,
            bootstrap_lcb=result.bootstrap_lcb,
            negative_controls_accepted=result.negative_controls_accepted,
            rejection_tests=result.rejection_tests,
            maxt_survived=result.maxt_survived,
            concentrated=result.concentrated,
        )
        entry["status"] = "RAN"
        entry["verdict"] = scored
        entry["diagnostics"] = dict(result.diagnostics)
        results.append(entry)
        registry.record(entry)

        tier = scored["tier"]
        if tier == "INVALID":
            verdict = "INVALID"
            break
        if tier == "TIER_A":
            verdict = "TIER_A"
            break
        if tier == "TIER_B" and verdict == "NO_EDGE":
            verdict = "TIER_B"

    return {
        "wave_id": spec.wave_id,
        "objective": spec.objective,
        "verdict": verdict,
        "budget": spec.budget,
        "budget_spent": spent,
        "budget_exhausted": spent >= spec.budget,
        "family_size_for_maxt": spent,
        "results": results,
        "note": (
            "family_size_for_maxt counts every hypothesis that consumed a look at the "
            "data. Apply maxT across that family. TIER_A means 'worth a forward "
            "live-paper test', never 'deployable' -- the protected holdout is SPENT."
        ),
    }
