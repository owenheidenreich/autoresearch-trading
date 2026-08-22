"""The Outcome Run Gate: no outcome-bearing fit without a verified declaration.

Why this exists
---------------
Job 46 skipped its Phase 5 declaration three times -- the void 2026-08-20 entry
fit, and both 2026-08-21 fits -- and nothing stopped it. A cold adversarial
review named the root cause on 2026-08-22, and it is not "someone used a
scratchpad script":

  **Governance here was opt-in.** Phases 4a and 4b were declared correctly only
  because their dedicated `v5/ops` runners chose to do it. `train_entry_phase`
  and `train_exit_head` accept no declaration and no ledger, so any script that
  imports them reaches production training directly. A control that a caller can
  decline to use is documentation, not a control.

So the check moves to the irreversible point: the trainers themselves refuse to
fit corpus-derived outcomes without a one-use `DeclaredFitPermit`.

Three properties are load-bearing, each because of a measured failure.

1. **The permit is keyed to the DATA, not to a flag.** `SessionEpisode` and
   `Trajectory` carry a `provenance` field; `build_episode` stamps `"corpus"` and
   nothing else does. A synthetic test episode needs no permit and a corpus
   episode cannot be talked out of needing one. A boolean argument would have
   reproduced exactly the opt-in hole this closes.

2. **The exposure is journalled BEFORE any outcome opens, not after.** The
   decision to look is what spends alpha, so a run that crashes mid-fit has still
   spent one. An unresolved `STARTED` record therefore BLOCKS the next permit
   until a human classifies it. Silence must not read as "nothing happened" --
   memo §7 records four separate defects where a watcher reported success while
   something was lost.

3. **The permit is single-use and verified at open time.** It re-hashes the
   declaration under the repository's own convention (sha256 over the canonical
   JSON with `declaration_sha256` removed), re-hashes every file the declaration
   pins, and refuses on any mismatch. A declaration that no longer describes the
   code that would run is not a declaration.

What this does NOT do, said plainly: it cannot make a declaration honest. It
enforces that one exists, matches the code, and is paid for. Whether its content
was chosen before the outcomes were known is a governance question that no
runtime check can answer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.autoresearch.budget import AlphaLedger

SCHEMA_VERSION = "v5.outcome-run-gate.exposure-journal.v1"

#: The provenance stamp `build_episode` applies. Anything carrying it is
#: corpus-derived and may not be fitted without a permit.
CORPUS_PROVENANCE = "corpus"
SYNTHETIC_PROVENANCE = "synthetic"

STARTED = "STARTED"
COMPLETED = "COMPLETED"
ABANDONED = "ABANDONED"


class OutcomeGateError(RuntimeError):
    """An outcome-bearing run was attempted without a valid, paid-for permit."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Declaration verification
# --------------------------------------------------------------------------- #


def declaration_digest(declaration: dict[str, Any]) -> str:
    """The repository's self-hash convention, applied to a declaration payload.

    Verified against `PHASE_4B_DECLARATION_V1.json`, which reproduces its own
    `declaration_sha256` exactly under this rule.
    """

    without = {k: v for k, v in declaration.items() if k != "declaration_sha256"}
    return hashlib.sha256(canonical_json(without)).hexdigest()


def verify_declaration(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Load a declaration, prove it is unedited, and prove it still fits the code.

    Raises rather than warns on every failure: a declaration that does not match
    the code that would run is worse than no declaration, because it looks like
    governance while describing something else.
    """

    path = Path(path)
    if not path.is_file():
        raise OutcomeGateError(f"no declaration at {path}")
    declaration = json.loads(path.read_text(encoding="utf-8"))

    claimed = declaration.get("declaration_sha256")
    if not claimed:
        raise OutcomeGateError(f"{path.name} carries no declaration_sha256")
    actual = declaration_digest(declaration)
    if actual != claimed:
        raise OutcomeGateError(
            f"{path.name} was edited after it was sealed: it claims {claimed[:12]}... "
            f"and hashes to {actual[:12]}..."
        )

    pinned = declaration.get("implementation_hashes") or {}
    if not pinned:
        raise OutcomeGateError(
            f"{path.name} pins no implementation_hashes; it cannot prove which code it describes"
        )
    drifted = []
    for relative, expected in sorted(pinned.items()):
        target = repo_root / relative
        if not target.is_file():
            drifted.append(f"{relative} (missing)")
            continue
        found = file_sha256(target)
        if found != expected:
            drifted.append(f"{relative} ({found[:12]}... != {expected[:12]}...)")
    if drifted:
        raise OutcomeGateError(
            f"{path.name} no longer describes the code that would run: " + "; ".join(drifted)
        )
    return declaration


# --------------------------------------------------------------------------- #
# The exposure journal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exposure:
    """One decision to look at outcomes, recorded before the looking starts."""

    index: int
    experiment_id: str
    declaration_sha256: str
    state: str
    started_at: str
    resolved_at: str | None = None
    outcome: str | None = None
    note: str | None = None


class ExposureJournal:
    """Append-only record of every decision to open outcome data.

    Separate from the alpha ledger on purpose. The ledger records *finished*
    experiments and the bar they bought; this records *attempts*, including the
    ones that died halfway. A run that crashes after opening outcomes has spent
    an attempt, and only a person can say what it was.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: list[Exposure] = []
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise OutcomeGateError(f"unknown journal schema: {payload.get('schema_version')}")
            self._entries = [Exposure(**raw) for raw in payload.get("exposures", [])]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def unresolved(self) -> list[Exposure]:
        return [e for e in self._entries if e.state == STARTED]

    def _write(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "exposures": [asdict(e) for e in self._entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def start(self, *, experiment_id: str, declaration_sha256: str) -> Exposure:
        if self.unresolved:
            stuck = ", ".join(e.experiment_id for e in self.unresolved)
            raise OutcomeGateError(
                f"an earlier exposure was never resolved: {stuck}. A run that opened outcome "
                "data and did not finish still spent an attempt. Classify it with "
                "`resolve(...)` -- as COMPLETED with its outcome, or ABANDONED with a reason -- "
                "before opening another."
            )
        if any(e.experiment_id == experiment_id for e in self._entries):
            raise OutcomeGateError(f"exposure already journalled: {experiment_id}")
        exposure = Exposure(
            index=len(self._entries),
            experiment_id=experiment_id,
            declaration_sha256=declaration_sha256,
            state=STARTED,
            started_at=_now(),
        )
        self._entries.append(exposure)
        self._write()
        return exposure

    def resolve(self, experiment_id: str, *, state: str, outcome: str | None = None,
                note: str | None = None) -> Exposure:
        if state not in {COMPLETED, ABANDONED}:
            raise OutcomeGateError(f"an exposure resolves to {COMPLETED} or {ABANDONED}, not {state}")
        for i, entry in enumerate(self._entries):
            if entry.experiment_id == experiment_id:
                if entry.state != STARTED:
                    raise OutcomeGateError(
                        f"{experiment_id} is already {entry.state}; the journal is append-only "
                        "and a resolved exposure is not re-openable"
                    )
                resolved = Exposure(
                    **{**asdict(entry), "state": state, "resolved_at": _now(),
                       "outcome": outcome, "note": note}
                )
                self._entries[i] = resolved
                self._write()
                return resolved
        raise OutcomeGateError(f"no exposure journalled for {experiment_id}")


# --------------------------------------------------------------------------- #
# The permit
# --------------------------------------------------------------------------- #


class DeclaredFitPermit:
    """A one-use authorisation to fit corpus-derived outcomes.

    Open it with `DeclaredFitPermit.open(...)`, which verifies the declaration
    and journals the exposure *before* returning. Pass it to the trainer. Resolve
    it when the run finishes. Nothing about that order is optional: the journal
    entry exists before the fit so that a crash is visible as a spent attempt.
    """

    def __init__(self, *, declaration: dict[str, Any], experiment_id: str,
                 journal: ExposureJournal, ledger: AlphaLedger | None):
        self.declaration = declaration
        self.experiment_id = experiment_id
        self.declaration_sha256 = declaration["declaration_sha256"]
        self._journal = journal
        self._ledger = ledger
        self._resolved = False
        self.fits = 0

    @classmethod
    def open(cls, declaration_path: Path, *, experiment_id: str, journal_path: Path,
             repo_root: Path, ledger: AlphaLedger | None = None) -> "DeclaredFitPermit":
        declaration = verify_declaration(Path(declaration_path), repo_root=Path(repo_root))
        journal = ExposureJournal(Path(journal_path))
        journal.start(experiment_id=experiment_id,
                      declaration_sha256=declaration["declaration_sha256"])
        return cls(declaration=declaration, experiment_id=experiment_id,
                   journal=journal, ledger=ledger)

    @property
    def resolved(self) -> bool:
        return self._resolved

    def spend(self, what: str) -> None:
        """Register one fit against this permit.

        **A permit authorises one EXPERIMENT, not one call.** Nested out-of-fold
        trajectory generation trains an entry model per inner fold, and those are
        four fits inside a single declared experiment -- requiring a fresh permit
        for each would price honest nesting out of existence and teach callers to
        route around the gate. What the permit forbids is reuse *after the
        experiment is closed*, which is the actual hole: a resolved permit
        authorising a second, undeclared look.

        The fit count is journalled, because how many fits an experiment really
        ran was itself invisible before this gate existed.
        """

        if self._resolved:
            raise OutcomeGateError(
                f"permit for {self.experiment_id} is already resolved and cannot authorise "
                f"{what}. Its exposure is closed and charged. A further look is a new "
                "experiment: open a new permit under its own declaration and pay for it."
            )
        self.fits += 1

    def resolve(self, *, outcome: str, note: str | None = None,
                observed_accuracy: float | None = None,
                accuracy_lower_bound: float | None = None) -> None:
        """Close the exposure and, if a ledger was supplied, charge it."""

        detail = f"{self.fits} fit(s)" + (f"; {note}" if note else "")
        self._journal.resolve(self.experiment_id, state=COMPLETED, outcome=outcome, note=detail)
        self._resolved = True
        if self._ledger is not None:
            self._ledger.record(
                experiment_id=self.experiment_id,
                declaration_sha256=self.declaration_sha256,
                declared_on=str(self.declaration.get("declared_on", _now()[:10])),
                outcome=outcome,
                observed_accuracy=observed_accuracy,
                accuracy_lower_bound=accuracy_lower_bound,
            )

    def abandon(self, note: str) -> None:
        self._journal.resolve(
            self.experiment_id, state=ABANDONED, note=f"{self.fits} fit(s); {note}"
        )
        self._resolved = True


def require_permit(provenances: set[str], permit: DeclaredFitPermit | None, *, what: str) -> None:
    """The gate itself. Corpus-derived outcomes need a live permit; nothing else does.

    Called by `train_entry_phase` and `train_exit_head`. Deliberately takes the
    observed provenances rather than a caller-supplied flag, so the requirement
    follows the data.
    """

    if CORPUS_PROVENANCE not in provenances:
        return
    if permit is None:
        raise OutcomeGateError(
            f"{what} was given corpus-derived outcomes and no DeclaredFitPermit. "
            "Phase 5 was skipped three times because this check did not exist: the fits ran "
            "from ad-hoc scripts straight into the trainer, and nothing required a declaration. "
            "Open a permit with DeclaredFitPermit.open(declaration_path, ...), which verifies "
            "the declaration against the code and journals the exposure before you look."
        )
    permit.spend(what)
