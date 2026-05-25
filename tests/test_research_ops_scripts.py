import ast
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_OPS = REPO_ROOT / "research_ops"
SCRIPTS = RESEARCH_OPS / "scripts"
PROMPTS = RESEARCH_OPS / "prompts"


def test_research_ops_scripts_do_not_import_trading_or_external_systems():
    forbidden_prefixes = {
        "v4",
        "ib_insync",
        "polygon",
        "databento",
        "theta",
        "torch",
        "pandas",
        "numpy",
    }
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        bad = sorted(name for name in imports if name.split(".")[0] in forbidden_prefixes)
        assert bad == [], f"{path} imports protected modules: {bad}"


def test_assumption_registry_has_expected_header():
    with (RESEARCH_OPS / "ASSUMPTION_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "id",
        "priority",
        "layer",
        "assumption",
        "current_evidence",
        "risk_if_false",
        "falsification_test",
        "confidence_increases_if",
        "confidence_collapses_if",
        "required_artifacts",
        "blocked_actions",
        "status",
        "next_diagnostic",
    ]


def test_assumption_registry_seeds_required_stage3_assumptions():
    with (RESEARCH_OPS / "ASSUMPTION_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    assert {f"A{idx:03d}" for idx in range(1, 13)} <= set(by_id)
    assert by_id["A001"]["assumption"] == "quote age truth"
    assert by_id["A002"]["assumption"] == "ask-entry/bid-exit replay is executable"
    assert by_id["A011"]["priority"] == "P0"
    assert by_id["A014"]["status"] == "blocked"
    assert set(row["priority"] for row in rows) <= {"P0", "P1", "P2"}


def test_current_state_declares_audit_control_fields():
    text = (RESEARCH_OPS / "CURRENT_STATE.yaml").read_text(encoding="utf-8")
    for required in [
        "current_default:",
        "name: \"PAPER_DEFAULT_PROTOCOL101\"",
        "scope: \"guarded IBKR paper runtime only\"",
        "real_money: false",
        "frozen_control:",
        "protocol: \"Protocol101\"",
        "surface_model: \"Protocol051\"",
        "lifecycle_model: \"Protocol066_081\"",
        "blocked_actions:",
        "p0_risks:",
        "source_documents:",
        "next_phase: \"execution-and-parity falsification\"",
        "not_next_phase: \"model capacity expansion\"",
    ]:
        assert required in text


def test_do_not_touch_boundaries_cover_stage4_required_surfaces():
    text = (RESEARCH_OPS / "DO_NOT_TOUCH_WITHOUT_APPROVAL.md").read_text(encoding="utf-8")
    required_terms = [
        "v4/runtime/protocol101_paper_order_enablement.json",
        "v4/ops/launchd/*",
        "v4/ops/ibkr/run_protocol101_paper_session.sh",
        "v4/live/ibkr_paper_executor.py",
        "v4/live/ibkr_paper_guard.py",
        "Protocol101 model, scaler, manifest",
        "Protocol051/054 surface model, scaler, manifest",
        "Protocol066/081 lifecycle model, scaler, manifest",
        "Paid data download scripts",
        "Protected holdout scoring scripts",
        "Threshold selection logic",
        "paper-submit default behavior",
        "placeOrder",
        "safe read-only work",
        "Requires CEO Decision Memo",
        "Requires A Separate Branch",
        "Requires Human Approval",
    ]
    for required in required_terms:
        assert required in text


def test_pull_request_template_ties_prs_to_research_ops_controls():
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for required in [
        "## Purpose",
        "What assumption or RFC does this PR address?",
        "Iteration ID:",
        "Assumption ID:",
        "No trading logic changed unless explicitly approved",
        "No runtime flags changed",
        "No launchd files changed",
        "No broker calls added",
        "No paid data downloads added",
        "No training added",
        "No threshold tuning",
        "No model promotion",
        "No protected holdout scored for exploration",
        "## CEO Decision Required",
    ]:
        assert required in template


def test_stage6_agent_prompts_exist_and_keep_roles_separate():
    expected = {
        "01_cartographer.md": [
            "You are the Codebase Cartographer.",
            "You may not modify files.",
            "research_ops/iterations/<ITER_ID>/01_cartography.md",
            "No broker calls",
            "No paid data",
            "No training",
            "No threshold tuning",
            "No runtime flag mutation",
        ],
        "02_experiment_designer.md": [
            "You are the Experiment Designer.",
            "Use `research_ops/iterations/<ITER_ID>/01_cartography.md`",
            "research_ops/iterations/<ITER_ID>/02_rfc.md",
            "Do not implement code.",
            "Null hypothesis",
            "Pass/fail criteria",
        ],
        "03_implementation_agent.md": [
            "You are the Implementation Agent.",
            "Before coding, list files you will create or modify.",
            "research_ops/iterations/<ITER_ID>/03_implementation_summary.md",
            "No trading behavior changes",
            "No broker calls",
            "No paid downloads",
        ],
        "04_verifier.md": [
            "You are the Verifier / Red Team.",
            "research_ops/iterations/<ITER_ID>/04_verifier_report.md",
            "Leakage",
            "Broker risk",
            "Mismatch between RFC and implementation",
            "Do not implement new features.",
        ],
        "05_dashboard_agent.md": [
            "You are the Dashboard Agent.",
            "research_ops/CEO_DASHBOARD.md",
            "Current operational default",
            "Newly falsified assumptions",
            "No trading code imports.",
            "python research_ops/scripts/update_dashboard.py",
        ],
        "06_decision_memo_writer.md": [
            "You are the Decision Memo Writer.",
            "research_ops/iterations/<ITER_ID>/05_decision_memo.md",
            "Question asked",
            "Actions still blocked",
            "CEO decision required",
        ],
    }
    for filename, required_terms in expected.items():
        text = (PROMPTS / filename).read_text(encoding="utf-8")
        for required in required_terms:
            assert required in text


def test_new_iteration_validate_summarize_and_dashboard_roundtrip(tmp_path):
    sandbox = tmp_path / "repo"
    shutil.copytree(RESEARCH_OPS, sandbox / "research_ops")

    create = subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "new_iteration.py"),
            "--id",
            "ITER-001_quote_age_truth",
            "--assumption",
            "A001",
            "--title",
            "Quote age truth",
            "--created-at",
            "2026-05-24T00:00:00Z",
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    iteration_dir = Path(create.stdout.strip())
    assert iteration_dir.name == "ITER-001_quote_age_truth"

    for filename in [
        "manifest.yaml",
        "00_request.md",
        "01_cartography.md",
        "02_rfc.md",
        "03_implementation_summary.md",
        "04_verifier_report.md",
        "05_decision_memo.md",
    ]:
        assert (iteration_dir / filename).exists()
    assert (iteration_dir / "artifacts" / ".gitkeep").exists()
    manifest_text = (iteration_dir / "manifest.yaml").read_text(encoding="utf-8")
    assert 'iteration_id: "ITER-001_quote_age_truth"' in manifest_text
    assert 'assumption_id: "A001"' in manifest_text

    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "validate_iteration.py"),
            str(iteration_dir),
            "--root",
            str(sandbox),
            "--registry",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest_path = iteration_dir / "manifest.yaml"
    manifest_path.write_text(
        manifest_text.replace('status: "draft"', 'status: "completed"'),
        encoding="utf-8",
    )
    (iteration_dir / "04_verifier_report.md").write_text(
        "# Verifier Report\n\n"
        "## Newly Confirmed Evidence\n\n"
        "- Quote timestamps can be reconstructed for sampled rows.\n\n"
        "## Newly Falsified Assumptions\n\n"
        "- None yet.\n\n"
        "## Verdict\n\n"
        "supported\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "summarize_iteration.py"),
            str(iteration_dir),
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (iteration_dir / "artifacts" / "iteration_summary.md").exists()

    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "update_dashboard.py"),
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    dashboard = (sandbox / "research_ops" / "CEO_DASHBOARD.md").read_text(encoding="utf-8")
    for required in [
        "## 1. Current Operational Default",
        "## 2. Current Safety Posture",
        "## 3. Active Iteration",
        "## 4. Latest Completed Iteration",
        "## 5. Decisions Required",
        "## 6. P0 Assumptions",
        "## 7. Blocked Actions",
        "## 8. Newly Confirmed Evidence",
        "## 9. Newly Falsified Assumptions",
        "## 10. Next Recommended Codex Prompt",
        "ITER-001_quote_age_truth",
        "Quote timestamps can be reconstructed for sampled rows.",
    ]:
        assert required in dashboard


def test_validate_iteration_rejects_forbidden_modified_paths(tmp_path):
    sandbox = tmp_path / "repo"
    shutil.copytree(RESEARCH_OPS, sandbox / "research_ops")
    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "new_iteration.py"),
            "--id",
            "ITER-002_runtime_boundary",
            "--assumption",
            "A002",
            "--title",
            "Runtime boundary",
            "--created-at",
            "2026-05-24T00:00:00Z",
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    iteration_dir = sandbox / "research_ops" / "iterations" / "ITER-002_runtime_boundary"
    (iteration_dir / "03_implementation_summary.md").write_text(
        "# Implementation Summary\n\n"
        "## Files Modified\n\n"
        "- v4/runtime/protocol101_paper_order_enablement.json\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "validate_iteration.py"),
            str(iteration_dir),
            "--root",
            str(sandbox),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "forbidden path listed as modified" in result.stderr
