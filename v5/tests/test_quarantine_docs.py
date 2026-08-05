"""Guards on the quarantine tool itself.

The tool moves files, so its refusals matter more than its successes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from v5.ops import quarantine_docs as qd
from v5.ops.check_project import ROOT


BATCH_ROOT = ROOT / "_cleanup_quarantine/2026-08-05b-docs"


def test_the_applied_batch_left_a_manifest() -> None:
    assert (BATCH_ROOT / "MANIFEST.md").is_file()


def test_nothing_the_batch_moved_is_still_a_live_path_constant() -> None:
    """The regression that broke v4/foundation/project_sections.py.

    A whole-tree sweep must not orphan a file that live code requires by path.
    """

    manifest = (BATCH_ROOT / "MANIFEST.md").read_text()
    moved = {
        line.split("`")[1] for line in manifest.splitlines() if line.startswith("| `")
    }
    code_text = qd._active_code_text()
    assert {path for path in moved if path in code_text} == set()


def test_apply_refuses_a_candidate_that_active_code_still_requires() -> None:
    live = qd.Selection(
        # quarantine_docs.py holds this exact path constant, so moving it would
        # orphan a live reference the way retiring the docs drawer once did.
        path=ROOT / "v5/ops/check_project.py",
        reason="retired tree docs/",
        tracked=False,
        size=1,
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="still named by active code"):
        qd._assert_safe([live])


def test_apply_refuses_protected_evidence() -> None:
    evidence = qd.Selection(
        path=ROOT / "v4/audit/some_run/report.md",
        reason="retired tree docs/",
        tracked=False,
        size=1,
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="protected evidence"):
        qd._assert_safe([evidence])


def test_apply_refuses_to_overwrite_a_populated_batch() -> None:
    with pytest.raises(RuntimeError, match="already populated"):
        qd.apply_batch([], [])


def test_a_second_dry_run_selects_nothing() -> None:
    """The batch is complete: rerunning must not find more to move."""

    selections, broken = qd.derive()
    assert selections == []
    assert broken == []
