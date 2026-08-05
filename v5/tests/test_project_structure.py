from __future__ import annotations

from v5.ops.check_project import ROOT, V5, scan


def test_v5_project_structure_is_clean() -> None:
    assert scan() == []


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
        "measurement blocker": ("254 owned es sessions", "cannot reliably measure"),
        "current job": ("independent", "measurement-capacity review"),
        "next permitted action": ("review the measurement limits", "do not run the g1"),
        "closed experiments": ("five research campaigns", "no edge", "do-not-retest"),
        "cost bar": ("0.358 es points", "$17.92"),
        "safety restrictions": ("no training", "broker/vendor contact", "owner authorization"),
    }
    missing = {
        answer: terms
        for answer, terms in required_answers.items()
        if not all(term in readme for term in terms)
    }
    assert missing == {}
