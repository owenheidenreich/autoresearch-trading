"""The gate that makes a declaration mandatory rather than customary.

Job 46 skipped its Phase 5 declaration three times and nothing stopped it,
because the checks lived in cooperative runners while the trainers themselves
accepted anything. These tests pin the repair: the requirement follows the
*data*, a crashed run stays visible as a spent attempt, and a declaration that
no longer matches the code is refused rather than warned about.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from v5.ops.build_causal_day_dataset import canonical_json
from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_shared_lifecycle import CompactSharedLifecyclePolicy
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)
from v5.research.lifecycle_trainer import (
    ChronologySplit,
    LifecycleTrainingError,
    SessionEpisode,
    Trajectory,
    TrainingLaw,
    generate_oof_trajectories,
    train_entry_phase,
    train_exit_head,
)
from v5.research.outcome_run_gate import (
    ABANDONED,
    COMPLETED,
    CORPUS_PROVENANCE,
    STARTED,
    DeclaredFitPermit,
    ExposureJournal,
    OutcomeGateError,
    declaration_digest,
    verify_declaration,
)

FAST = TrainingLaw(max_epochs=4, plateau_patience=2)
REPO = Path(__file__).resolve().parents[2]


def _batch(rows: int = 4, nodes: int = 3, *, seed: int = 0) -> CausalPolicyBatch:
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(rows, nodes, dtype=torch.bool)
    mask[:, : max(nodes - 1, 1)] = True
    return CausalPolicyBatch(
        candles=torch.randn(rows, 1, len(CANDLE_FEATURES), generator=g),
        candle_mask=torch.ones(rows, 1, dtype=torch.bool),
        ladder=torch.randn(rows, nodes, len(LADDER_FEATURES), generator=g),
        ladder_mask=torch.ones(rows, nodes, dtype=torch.bool),
        entry_action_mask=mask,
        account=torch.zeros(rows, ACCOUNT_FEATURES),
        position=torch.zeros(rows, POSITION_FEATURES),
        clock=torch.rand(rows, CLOCK_FEATURES, generator=g),
        roles=tuple("morning_entry" for _ in range(rows)),
    )


def _episode(session: str, *, provenance: str | None = None, seed: int = 0) -> SessionEpisode:
    batch = _batch(seed=seed)
    values = torch.full((4, batch.ladder.shape[1]), float("nan"))
    values[batch.entry_action_mask] = torch.randn(
        int(batch.entry_action_mask.sum()), generator=torch.Generator().manual_seed(seed)
    ) * 100.0
    paths = {
        (i, k): np.array([100.0, 140.0 + 10.0 * k, 120.0], dtype=np.float64)
        for i in range(4)
        for k in range(batch.ladder.shape[1])
    }
    kwargs = {} if provenance is None else {"provenance": provenance}
    return SessionEpisode(
        session=session, entry_batch=batch, entry_value_usd=values, sell_paths=paths, **kwargs
    )


def _declaration(tmp_path: Path, *, pins: dict[str, str] | None = None) -> Path:
    """A minimal declaration that self-hashes under the repository's convention."""

    target = REPO / "v5" / "research" / "outcome_run_gate.py"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    payload = {
        "schema_version": "test.declaration.v1",
        "declared_on": "2026-08-22",
        "purpose": "exercise the gate",
        "implementation_hashes": pins
        if pins is not None
        else {"v5/research/outcome_run_gate.py": digest},
    }
    payload["declaration_sha256"] = declaration_digest(payload)
    path = tmp_path / "DECLARATION.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _permit(tmp_path: Path, *, experiment_id: str = "test.experiment.v1") -> DeclaredFitPermit:
    return DeclaredFitPermit.open(
        _declaration(tmp_path),
        experiment_id=experiment_id,
        journal_path=tmp_path / "journal.json",
        repo_root=REPO,
    )


# ----------------------------------------------------------- the gate itself


def test_a_corpus_episode_cannot_be_fitted_without_a_permit() -> None:
    """The exact hole that let Phase 5 run undeclared three times."""

    corpus = [_episode("2024-01-01", provenance=CORPUS_PROVENANCE)]
    with pytest.raises(OutcomeGateError, match="no DeclaredFitPermit"):
        train_entry_phase(corpus, seed=1, law=FAST, model=CompactSharedLifecyclePolicy())


def test_a_synthetic_episode_still_needs_no_permit() -> None:
    """The requirement follows the data. Test fixtures carry no economics."""

    model = train_entry_phase(
        [_episode("2024-01-01")], seed=1, law=FAST, model=CompactSharedLifecyclePolicy()
    )
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_a_permit_admits_the_corpus_fit_and_counts_it(tmp_path: Path) -> None:
    permit = _permit(tmp_path)
    corpus = [_episode("2024-01-01", provenance=CORPUS_PROVENANCE)]
    train_entry_phase(corpus, seed=1, law=FAST, model=CompactSharedLifecyclePolicy(), permit=permit)
    assert permit.fits == 1


def test_the_exit_phase_is_gated_on_the_same_evidence(tmp_path: Path) -> None:
    trajectories = [
        Trajectory(
            session="2024-01-01",
            decision_index=0,
            ladder_column=0,
            sell_path_usd=np.array([10.0, 20.0, 15.0]),
            generator_train_sessions=("2023-12-31",),
            provenance=CORPUS_PROVENANCE,
        )
    ]
    batches = [_batch(rows=3)]
    with pytest.raises(OutcomeGateError, match="no DeclaredFitPermit"):
        train_exit_head(
            CompactSharedLifecyclePolicy(), trajectories, batches, seed=1, law=FAST
        )


def test_a_resolved_permit_cannot_authorise_a_second_look(tmp_path: Path) -> None:
    """A closed, charged exposure is not a licence to look again."""

    permit = _permit(tmp_path)
    corpus = [_episode("2024-01-01", provenance=CORPUS_PROVENANCE)]
    train_entry_phase(corpus, seed=1, law=FAST, model=CompactSharedLifecyclePolicy(), permit=permit)
    permit.resolve(outcome="FAIL", note="did not survive")
    with pytest.raises(OutcomeGateError, match="already resolved"):
        train_entry_phase(
            corpus, seed=1, law=FAST, model=CompactSharedLifecyclePolicy(), permit=permit
        )


def test_nested_folds_are_one_experiment_not_four(tmp_path: Path) -> None:
    """Honest nesting must not be priced out of existence by the gate.

    `generate_oof_trajectories` trains one entry model per inner fold. Those are
    fits inside a single declared experiment, so one permit covers them and the
    count is journalled rather than refused.
    """

    sessions = [f"2024-01-{i + 1:02d}" for i in range(20)]
    episodes = [
        _episode(s, provenance=CORPUS_PROVENANCE, seed=i) for i, s in enumerate(sessions)
    ]
    split = ChronologySplit.build(sessions)
    permit = _permit(tmp_path)
    generate_oof_trajectories(
        episodes,
        split,
        folds=3,
        seed=1,
        model_factory=CompactSharedLifecyclePolicy,
        law=FAST,
        permit=permit,
    )
    assert permit.fits >= 2
    # ...and without one, the same call is refused rather than quietly nesting.
    with pytest.raises(OutcomeGateError, match="no DeclaredFitPermit"):
        generate_oof_trajectories(
            episodes, split, folds=3, seed=1,
            model_factory=CompactSharedLifecyclePolicy, law=FAST,
        )


# -------------------------------------------------- declaration verification


def test_the_real_phase_4b_declaration_reproduces_its_own_digest() -> None:
    """The convention is verified against a real sealed artifact, not asserted."""

    payload = json.loads(
        (REPO / "v5/work/lifecycle-training/PHASE_4B_DECLARATION_V1.json").read_text()
    )
    assert declaration_digest(payload) == payload["declaration_sha256"]


def test_an_edited_declaration_is_refused(tmp_path: Path) -> None:
    path = _declaration(tmp_path)
    payload = json.loads(path.read_text())
    payload["purpose"] = "something else entirely"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(OutcomeGateError, match="edited after it was sealed"):
        verify_declaration(path, repo_root=REPO)


def test_a_declaration_that_no_longer_matches_the_code_is_refused(tmp_path: Path) -> None:
    """A stale pin is worse than none: it looks like governance."""

    path = _declaration(tmp_path, pins={"v5/research/outcome_run_gate.py": "0" * 64})
    with pytest.raises(OutcomeGateError, match="no longer describes the code"):
        verify_declaration(path, repo_root=REPO)


def test_a_declaration_pinning_nothing_is_refused(tmp_path: Path) -> None:
    path = _declaration(tmp_path, pins={})
    with pytest.raises(OutcomeGateError, match="pins no implementation_hashes"):
        verify_declaration(path, repo_root=REPO)


# ------------------------------------------------------------- the journal


def test_an_unresolved_exposure_blocks_the_next_run(tmp_path: Path) -> None:
    """A crash mid-fit still spent an attempt, and silence must not hide it."""

    journal_path = tmp_path / "journal.json"
    DeclaredFitPermit.open(
        _declaration(tmp_path), experiment_id="first.v1",
        journal_path=journal_path, repo_root=REPO,
    )
    # The process dies here. Nothing resolved the exposure.
    assert [e.state for e in ExposureJournal(journal_path)] == [STARTED]
    with pytest.raises(OutcomeGateError, match="never resolved"):
        DeclaredFitPermit.open(
            _declaration(tmp_path), experiment_id="second.v1",
            journal_path=journal_path, repo_root=REPO,
        )


def test_classifying_the_crashed_run_unblocks_the_gate(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.json"
    DeclaredFitPermit.open(
        _declaration(tmp_path), experiment_id="first.v1",
        journal_path=journal_path, repo_root=REPO,
    )
    ExposureJournal(journal_path).resolve(
        "first.v1", state=ABANDONED, note="process died during the prefix build"
    )
    permit = DeclaredFitPermit.open(
        _declaration(tmp_path), experiment_id="second.v1",
        journal_path=journal_path, repo_root=REPO,
    )
    states = {e.experiment_id: e.state for e in ExposureJournal(journal_path)}
    assert states == {"first.v1": ABANDONED, "second.v1": STARTED}
    permit.resolve(outcome="FAIL")
    assert {e.state for e in ExposureJournal(journal_path)} == {ABANDONED, COMPLETED}


def test_the_journal_is_append_only(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.json"
    permit = DeclaredFitPermit.open(
        _declaration(tmp_path), experiment_id="first.v1",
        journal_path=journal_path, repo_root=REPO,
    )
    permit.resolve(outcome="FAIL")
    with pytest.raises(OutcomeGateError, match="already COMPLETED"):
        ExposureJournal(journal_path).resolve("first.v1", state=COMPLETED, outcome="PASS")
    with pytest.raises(OutcomeGateError, match="already journalled"):
        DeclaredFitPermit.open(
            _declaration(tmp_path), experiment_id="first.v1",
            journal_path=journal_path, repo_root=REPO,
        )


def test_the_exposure_is_journalled_before_the_fit_not_after(tmp_path: Path) -> None:
    """The decision to look is what spends alpha, so it is recorded first."""

    journal_path = tmp_path / "journal.json"
    permit = DeclaredFitPermit.open(
        _declaration(tmp_path), experiment_id="first.v1",
        journal_path=journal_path, repo_root=REPO,
    )
    # No fit has run yet, and the record already exists on disk.
    assert permit.fits == 0
    assert json.loads(journal_path.read_text())["exposures"][0]["state"] == STARTED
