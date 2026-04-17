"""Artifact-kind enum: the type-level distinction between evaluation and deploy artifacts.

A fold checkpoint and a final-train checkpoint are not the same thing. This enum
encodes that distinction at the type level so a fold checkpoint can never
accidentally become the production model.

Promotion interlock:
    Only artifacts tagged ArtifactKind.FINAL_TRAIN are accepted by
    v2.ops.model_manage.keep(). A missing or wrong kind is a hard failure.

See /Users/gduby/.claude/plans/delightful-yawning-tiger.md Appendix C.
"""
from __future__ import annotations

from enum import Enum


class ArtifactKind(str, Enum):
    """What an artifact is for. Defines whether it can be promoted."""

    CV_EVAL = "cv_eval"
    """Output of walk-forward CV. Contains per-fold checkpoints and a CVReport.
    NOT promotable to v2/models/model.pt — CV selects configs, not weights."""

    FOLD_CHECKPOINT = "fold_checkpoint"
    """Per-fold model checkpoint from walk-forward CV. Debugging artifact only.
    NOT promotable."""

    FINAL_TRAIN = "final_train"
    """Output of v2.ops.run_final_train. Trained on the chosen pre-shadow span
    using a named internal validation slice. THE ONLY promotable kind."""


def is_promotable(kind: str | ArtifactKind | None) -> bool:
    """True iff the artifact kind allows promotion to v2/models/model.pt."""
    if kind is None:
        return False
    try:
        return ArtifactKind(kind) == ArtifactKind.FINAL_TRAIN
    except ValueError:
        return False
