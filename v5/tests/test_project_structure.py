from __future__ import annotations

from v5.ops.check_project import (
    RETIRED_TREES,
    ROOT,
    V5,
    _is_planning_name,
    scan,
    scan_repository,
    v5_link_targets,
)


def test_v5_project_structure_is_clean() -> None:
    assert scan() == []


def test_repository_has_no_stray_planning_files() -> None:
    assert scan_repository() == []


def test_planning_names_match_whole_words_only() -> None:
    for stem in (
        "PATHD_MECHANISM_FIRST_MASTER_PLAN_2026_08_04",
        "CODEX_HANDOFF_OPTIONS_CHAIN_DIRECTION_2026_08_04",
        "CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH",
        "PROTOCOL101_STATUS",
        "PLAN",
    ):
        assert _is_planning_name(stem), stem
    for stem in (
        # "PLANE" must not read as "PLAN"; these are real filenames.
        "PROTOCOL101_PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01",
        "DO_NOT_RETEST",
        "GATE_CHAIN_AUDIT_2026_08_05",
        "README",
    ):
        assert not _is_planning_name(stem), stem


def test_evidence_cited_by_v5_is_exempt_from_the_naming_rule() -> None:
    """A planning-shaped filename is allowed only when v5 cites it as evidence."""

    cited = {
        path
        for path in v5_link_targets()
        if path.exists() and _is_planning_name(path.stem)
    }
    assert cited, "expected at least one planning-shaped evidence file cited by v5"
    flagged = "\n".join(scan_repository())
    for path in cited:
        assert path.relative_to(ROOT).as_posix() not in flagged


def test_retired_trees_stay_gone() -> None:
    for tree in RETIRED_TREES:
        assert not (ROOT / tree).exists(), f"{tree}/ was recreated"


def test_root_docs_drawer_holds_only_a_pointer() -> None:
    docs = ROOT / "docs"
    if docs.exists():
        assert {path.name for path in docs.iterdir()} <= {"README.md"}


def test_root_bootstrap_points_to_v5() -> None:
    expected = {
        "README.md": "v5/README.md",
        "STATUS.md": "v5/STATUS.md",
        "AGENTS.md": "v5/AGENTS.md",
        "CLAUDE.md": "v5/AGENTS.md",
    }
    for name, marker in expected.items():
        assert marker in (ROOT / name).read_text()


def test_v5_has_no_generic_docs_directory() -> None:
    assert not (V5 / "docs").exists()


def test_readme_alone_answers_the_fresh_agent_orientation_drill() -> None:
    readme = " ".join((V5 / "README.md").read_text().lower().split())
    required_answers = {
        "goal": ("automated day-trading bot", "spx 0dte", "directional skill"),
        "canonical authority": ("only current-state page", "status.md"),
        "current job": ("job 47", "safe local phase 0", "phase 1 is not authorised"),
        "research blocker": ("effect size", "causal identification", "no directional family"),
        "next permitted action": ("repository and guard infrastructure", "do not open outcomes"),
        "economic distinction": ("0.358 es points", "$17.92", "not a universal spxw"),
        "forward evidence": ("2026-08-06", "confirmation-only"),
        "closed experiments": ("do-not-retest", "unchanged retry"),
        "durable guards": ("outcome run gate", "interior book-liveness", "cmbp touch-semantics"),
        "safety restrictions": ("no training", "broker/vendor contact", "owner authorization"),
    }
    missing = {
        answer: terms
        for answer, terms in required_answers.items()
        if not all(term in readme for term in terms)
    }
    assert missing == {}
    for obsolete in (
        "with 254 owned es sessions",
        "current job: obtain the independent",
        "only rung 1 is currently authorized",
    ):
        assert obsolete not in readme


def test_training_readiness_work_is_registered_and_catalogued() -> None:
    status = (V5 / "STATUS.md").read_text()
    toolbox = (V5 / "TOOLBOX.md").read_text()
    assert "Inventory and promote training-readiness controls" in status
    for marker in (
        "research/training_twin.py",
        "research/feature_admission.py",
        "research/validation/candidate_packet.py",
        "v4/path_d/",
        "v4/research/autoresearch_v2/",
        "v4/scripts/export_protocol101_trade_charts.py",
    ):
        assert marker in toolbox
